"""Four-rank Full Diffusion Geo training, validation, and finite acceptance."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset

from method.full_diffusion.data import GarageGeoDataset
from method.full_diffusion.diffusion import VPFullDiffusion
from method.full_diffusion.garage import FrozenGaragePerception
from method.full_diffusion.model import TrajectoryFullDiffuGeoHead
from tools.launch_full_diffusion import AUTHORIZED, guard_context, write_status


STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(path, head, optimizer, scheduler, progress, config, manifest):
    states = [None] * distributed.get_world_size()
    distributed.all_gather_object(states, rng_state())
    if distributed.get_rank() == 0:
        checkpoint = {"model": head.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "progress": progress,
                      "rng_by_rank": states, "config": config, "dataset": manifest,
                      "created_at": datetime.now(timezone.utc).isoformat()}
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    distributed.barrier()


def load_checkpoint(path, head, optimizer, scheduler, config, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != config or checkpoint["dataset"] != manifest:
        raise ValueError("Resume config or dataset index differs from the saved checkpoint")
    if len(checkpoint["rng_by_rank"]) != distributed.get_world_size():
        raise ValueError("Resume requires the original distributed world size")
    head.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng_by_rank"][distributed.get_rank()])
    return checkpoint["progress"]


class RemainingBatches:
    def __init__(self, batch_sampler, offset=0):
        self.batch_sampler = batch_sampler
        self.offset = offset

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.offset)

    def __iter__(self):
        return itertools.islice(self.batch_sampler, self.offset, None)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def autocast(config):
    return torch.autocast("cuda", dtype=torch.bfloat16) if config["precision"] == "bf16" else nullcontext()


def update(batch, perception, policy, optimizer, diffusion, config):
    head = policy.module
    optimizer.zero_grad(set_to_none=True)
    with autocast(config):
        bev, memory, ego = perception(batch)
        clean = head.codec.encode(batch["path"], batch["target_speed"])
        noisy, timestep = diffusion.corrupt(clean)
        predictions = policy(noisy, timestep, (bev, memory, ego, batch["target_point"]))
        loss = head.loss(predictions, clean)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite denoising loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), norm.detach()


@torch.no_grad()
def validate(dataset, perception, head, diffusion, config, device):
    rank, world = distributed.get_rank(), distributed.get_world_size()
    indices = range(rank, len(dataset), world)
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_per_gpu"], num_workers=0)
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    head.eval()
    generator = torch.Generator(device=device).manual_seed(config["seed"] + rank)
    for batch_id, batch in enumerate(loader):
        batch = move_batch(batch, device)
        count = len(batch["path"])
        with autocast(config):
            bev, memory, ego = perception(batch)
            condition = head.prepare_condition(bev, memory, ego, batch["target_point"])
            clean = head.codec.encode(batch["path"], batch["target_speed"])
            timesteps = torch.randint(diffusion.train_steps, (count,), generator=generator, device=device)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy, timestep = diffusion.corrupt(clean, timesteps, noise)
            prediction = head(noisy, timestep, condition)
            loss = head.loss(prediction, clean)
            totals[0] += loss.double() * count
            totals[1] += count
            if batch_id < 2:
                sample = diffusion.sample(lambda noisy, timestep: head(noisy, timestep, condition)[-1],
                                          count, device, generator, config["diffusion"]["inference_steps"])
                path, speed = head.codec.decode(sample)
                if not torch.isfinite(path).all() or not torch.isfinite(speed).all():
                    raise FloatingPointError("Nonfinite full-noise sampled path or speed")
                distances = (path - batch["path"]).norm(dim=-1)
                totals[2] += distances.mean(-1).double().sum()
                totals[3] += distances[:, -1].double().sum()
                totals[4] += (speed - batch["target_speed"]).abs().double().sum()
                totals[5] += count
    distributed.all_reduce(totals)
    head.train()
    return {"denoising_l1": float(totals[0] / totals[1]), "frames": int(totals[1]),
            "sampled_path_ade_m": float(totals[2] / totals[5]),
            "sampled_path_fde_m": float(totals[3] / totals[5]),
            "sampled_speed_mae_mps": float(totals[4] / totals[5]), "sampled_frames": int(totals[5])}


def acceptance(args, config, manifest, perception, head, policy, optimizer, scheduler, diffusion, dataset, device):
    from torch.utils.data import default_collate
    rank, world = distributed.get_rank(), distributed.get_world_size()
    batch = move_batch(default_collate([dataset[(rank * config["batch_per_gpu"] + offset) % len(dataset)]
                                      for offset in range(config["batch_per_gpu"])]), device)
    before = torch.cat([parameter.detach().flatten() for parameter in head.parameters()]).clone()
    losses = []
    started = time.monotonic()
    for step in range(3):
        loss, _ = update(batch, perception, policy, optimizer, diffusion, config)
        scheduler.step((step + 1) / 100)
        losses.append(float(loss))
    save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler,
                    {"epoch": 0, "next_batch": 3, "updates": 3}, config, manifest)
    update(batch, perception, policy, optimizer, diffusion, config)
    uninterrupted = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    load_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, config, manifest)
    update(batch, perception, policy, optimizer, diffusion, config)
    resume_error = max(float((parameter - uninterrupted[name]).abs().max())
                       for name, parameter in head.named_parameters())
    if resume_error > 2e-6:
        raise AssertionError(f"Resumed next update differs from uninterrupted training: {resume_error}")
    after = torch.cat([parameter.detach().flatten() for parameter in head.parameters()])
    delta = float((after - before).abs().max())
    if delta == 0 or any(parameter.grad is not None for parameter in perception.parameters()):
        raise AssertionError("No head update or frozen perception received gradients")
    checksum = after.double().sum()
    checksums = [torch.zeros_like(checksum) for _ in range(world)]
    distributed.all_gather(checksums, checksum)
    if max(float((value - checksums[0]).abs()) for value in checksums) > 1e-6:
        raise AssertionError("Distributed parameter replicas disagree")
    with torch.no_grad(), autocast(config):
        head.eval()
        bev, memory, ego = perception(batch)
        condition = head.prepare_condition(bev, memory, ego, batch["target_point"])
        sample = diffusion.sample(lambda noisy, timestep: head(noisy, timestep, condition)[-1],
                                  len(batch["path"]), device, inference_steps=100)
        if not torch.isfinite(sample).all():
            raise FloatingPointError("100-step generation produced nonfinite values")
    result = {"passed": True, "rank": rank, "world_size": world, "hostname": socket.gethostname(),
              "losses": losses, "updates": 4, "resume_max_parameter_error": resume_error,
              "parameter_max_delta": delta, "replica_checksum": float(checksum),
              "frozen_perception": True, "gaussian_sampling_steps": 100,
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "elapsed_seconds": time.monotonic() - started}
    write_status(args.output / f"acceptance_rank{rank}.json", result)
    print(json.dumps(result), flush=True)
    distributed.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("acceptance", "train"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = next((job for job, host in AUTHORIZED.items() if host == socket.gethostname().split(".")[0]), None)
    if job is None:
        raise RuntimeError("Not on an authorized training host")
    guard_context(job)
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("This training run requires exactly four ranks")
    config = json.loads(args.config.read_text())
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(seconds=180))
    probe = torch.tensor(float(rank), device=device)
    distributed.all_reduce(probe)
    if float(probe) != 6.0:
        raise AssertionError("Four-rank NCCL all-reduce failed")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    index = ROOT / config["index_directory"]
    manifest = json.loads((index / "manifest.json").read_text())
    if manifest["smoke_only"] or not manifest["val_frames"]:
        raise RuntimeError("Production/acceptance requires the complete index and a held-out validation split")
    dataset = GarageGeoDataset(index, backbone_directory=ROOT / config["backbone_directory"])
    validation = GarageGeoDataset(index, split="val", backbone_directory=ROOT / config["backbone_directory"])
    perception = FrozenGaragePerception(ROOT / config["backbone_directory"]).to(device)
    head = TrajectoryFullDiffuGeoHead(**config["model"], normalization=manifest["normalization"]).to(device)
    policy = DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    diffusion = VPFullDiffusion(head.codec, **{key: value for key, value in config["diffusion"].items()
                                            if key != "inference_steps"})
    torch.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "acceptance":
        acceptance(args, config, manifest, perception, head, policy, optimizer, scheduler, diffusion, dataset, device)
        distributed.destroy_process_group()
        return
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress = {"epoch": 0, "next_batch": 0, "updates": 0}
    if args.resume:
        progress = load_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, config, manifest)
    elif (args.output / "latest.pt").exists():
        raise FileExistsError("Existing checkpoint requires explicit --resume")
    sampler = DistributedSampler(dataset, seed=config["seed"], drop_last=True)
    batches = BatchSampler(sampler, config["batch_per_gpu"], drop_last=True)
    updates_per_epoch = len(batches)
    started = time.monotonic()
    log_started = started
    starting_updates = progress["updates"]
    running_loss = torch.zeros((), device=device)
    running_count = 0
    for epoch in range(progress["epoch"], config["epochs"]):
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        offset = progress["next_batch"] if epoch == progress["epoch"] else 0
        workers = config["workers_per_rank"]
        loader = DataLoader(dataset, batch_sampler=RemainingBatches(batches, offset),
                            num_workers=workers, pin_memory=True,
                            **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}))
        for batch_id, batch in enumerate(loader, start=offset):
            batch = move_batch(batch, device)
            loss, norm = update(batch, perception, policy, optimizer, diffusion, config)
            progress = {"epoch": epoch, "next_batch": batch_id + 1, "updates": progress["updates"] + 1}
            scheduler.step(epoch + (batch_id + 1) / updates_per_epoch)
            running_loss += loss
            running_count += 1
            stop = torch.tensor(int(STOP_REQUESTED), device=device)
            distributed.all_reduce(stop, op=distributed.ReduceOp.MAX)
            if progress["updates"] == 1 or progress["updates"] % config["log_every_updates"] == 0:
                distributed.all_reduce(running_loss)
                elapsed = time.monotonic() - started
                record = {"epoch": epoch + 1, "batch": batch_id + 1, "updates_per_epoch": updates_per_epoch,
                          "updates": progress["updates"], "loss": float(running_loss) / (running_count * 4),
                          "lr": optimizer.param_groups[0]["lr"], "gradient_norm_rank0": float(norm),
                          "elapsed_seconds": elapsed,
                          "samples_per_second": (progress["updates"] - starting_updates) * config["batch_per_gpu"] * 4 / elapsed,
                          "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 1024**3,
                          "utc": datetime.now(timezone.utc).isoformat()}
                if rank == 0:
                    with (args.output / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    write_status(args.output / "progress.json", record)
                    print(json.dumps(record), flush=True)
                running_loss.zero_()
                running_count = 0
                log_started = time.monotonic()
            if progress["updates"] == 1 or progress["updates"] % config["checkpoint_every_updates"] == 0 or int(stop):
                save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest)
            if int(stop):
                distributed.destroy_process_group()
                return
        metrics = validate(validation, perception, head, diffusion, config, device)
        progress = {"epoch": epoch + 1, "next_batch": 0, "updates": progress["updates"]}
        save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest)
        if rank == 0:
            os.link(args.output / "latest.pt", args.output / f"epoch_{epoch + 1:02}.pt")
            with (args.output / "validation.jsonl").open("a") as handle:
                handle.write(json.dumps({"epoch": epoch + 1, **metrics}) + "\n")
            print(json.dumps({"epoch": epoch + 1, "validation": metrics}), flush=True)
        distributed.barrier()
    if rank == 0:
        write_status(args.output / "completed.json", {**progress, "epochs": config["epochs"],
                                                     "finished_at": datetime.now(timezone.utc).isoformat()})
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()