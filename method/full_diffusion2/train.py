"""Eight-rank pure-E2E Full Diffusion2 training and finite distributed acceptance."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset, default_collate

from method.full_diffusion.diffusion import VPFullDiffusion
from method.full_diffusion2.data import GarageTemporalDataset
from method.full_diffusion2.model import TemporalFullDiffusionHead
from method.full_diffusion2.perception import FrozenPlannerPerception
from tools.launch_full_diffusion import write_status
from train_full_diffusion import RemainingBatches, autocast, load_checkpoint, move_batch, save_checkpoint


STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def condition_inputs(batch, perception, config):
    bev, memory, ego = perception(batch)
    history = perception.encode_history(batch, config["history_encode_chunk"])
    return (bev, memory, ego, batch["target_point"], history,
            batch["history_transforms"], batch["history_mask"], batch["speed"], batch["command"])


def update(batch, perception, policy, optimizer, diffusion, config):
    head = policy.module
    optimizer.zero_grad(set_to_none=True)
    with autocast(config):
        condition = condition_inputs(batch, perception, config)
        clean = head.codec.encode(batch["path"], batch["target_speed"])
        noisy, timestep = diffusion.corrupt(clean)
        predictions = policy(noisy, timestep, condition)
        loss = head.loss(predictions, clean)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Full Diffusion2 denoising loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return loss.detach(), norm.detach()


@torch.no_grad()
def validate(dataset, perception, head, diffusion, config, device):
    rank, world = distributed.get_rank(), distributed.get_world_size()
    loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)),
                        batch_size=config["batch_per_gpu"], num_workers=0)
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + rank)
    head.eval()
    for batch_id, batch in enumerate(loader):
        batch = move_batch(batch, device)
        count = len(batch["path"])
        with autocast(config):
            condition = head.prepare_condition(*condition_inputs(batch, perception, config))
            clean = head.codec.encode(batch["path"], batch["target_speed"])
            timesteps = torch.randint(diffusion.train_steps, (count,), generator=generator, device=device)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy, timestep = diffusion.corrupt(clean, timesteps, noise)
            totals[0] += head.loss(head(noisy, timestep, condition), clean).double() * count
            totals[1] += count
            if batch_id < 2:
                sample = diffusion.sample(lambda noisy, timestep: head(noisy, timestep, condition)[-1],
                                          count, device, generator, config["diffusion"]["inference_steps"])
                path, speed = head.codec.decode(sample)
                if not torch.isfinite(sample).all():
                    raise FloatingPointError("Nonfinite Gaussian-start validation output")
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
        raise AssertionError(f"Resumed next update differs: {resume_error}")
    missing = [name for name, parameter in head.named_parameters() if parameter.grad is None]
    if missing or any(parameter.grad is not None for parameter in perception.parameters()):
        raise AssertionError(f"Missing planner gradients or unfrozen teacher: {missing}")
    temporal_gradient = float(head.temporal.fusion[0].weight.grad.abs().sum())
    state_gradient = float(head.state_encoder[0].weight.grad.abs().sum())
    gate_gradient = float(head.temporal.gate[0].weight.grad.abs().sum())
    if min(temporal_gradient, state_gradient, gate_gradient) <= 0:
        raise AssertionError("Temporal fusion, gate, or state concatenation received no gradient")
    after = torch.cat([parameter.detach().flatten() for parameter in head.parameters()])
    delta = float((after - before).abs().max())
    if delta == 0:
        raise AssertionError("Planner parameters did not update")
    checksum = after.double().sum()
    checksums = [torch.zeros_like(checksum) for _ in range(world)]
    distributed.all_gather(checksums, checksum)
    if max(float((value - checksums[0]).abs()) for value in checksums) > 1e-6:
        raise AssertionError("Distributed parameter replicas disagree")
    with torch.no_grad(), autocast(config):
        head.eval()
        condition = head.prepare_condition(*condition_inputs(batch, perception, config))
        sample = diffusion.sample(lambda noisy, timestep: head(noisy, timestep, condition)[-1],
                                  len(batch["path"]), device, inference_steps=100)
        if not torch.isfinite(sample).all():
            raise FloatingPointError("100-step Gaussian generation is nonfinite")
    result = {"passed": True, "rank": rank, "world_size": world, "hostname": socket.gethostname(),
              "losses": losses, "updates": 4, "resume_max_parameter_error": resume_error,
              "parameter_max_delta": delta, "replica_checksum": float(checksum),
              "temporal_gradient_l1": temporal_gradient, "state_gradient_l1": state_gradient,
              "gate_gradient_l1": gate_gradient, "frozen_perception": True,
              "gaussian_sampling_steps": 100, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "elapsed_seconds": time.monotonic() - started}
    write_status(args.output / f"acceptance_rank{rank}.json", result)
    print(json.dumps(result), flush=True)
    distributed.barrier()


def check_config(config):
    if config["world_size"] != 8 or config["batch_per_gpu"] * config["world_size"] != 64:
        raise ValueError("The authorized comparison uses eight ranks and global batch64")
    if config["model"]["history_frames"] != 2 or config["history_stride"] != 1:
        raise ValueError("This experiment uses current plus two consecutive historical observations")
    index = ROOT / config["index_directory"]
    manifest = json.loads((index / "manifest.json").read_text())
    if manifest["smoke_only"] or not manifest["val_frames"]:
        raise ValueError("Training requires the full index and held-out validation routes")
    for name in ("train.npy", "val.npy", "routes.json"):
        if not (index / name).is_file():
            raise FileNotFoundError(index / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("acceptance", "train"), default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--workers-per-rank", type=int)
    parser.add_argument("--trial-updates", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workers = config["workers_per_rank"] if args.workers_per_rank is None else args.workers_per_rank
    if workers < 0 or (args.trial_updates is not None and args.trial_updates <= 0):
        parser.error("Workers must be nonnegative and trial updates must be positive")
    manifest = check_config(config)
    if args.check_config:
        print(json.dumps({"method": config["method"], "world_size": config["world_size"],
                          "global_batch": config["batch_per_gpu"] * config["world_size"],
                          "epochs": config["epochs"], "train_frames": manifest["train_frames"],
                          "val_frames": manifest["val_frames"], "history_frames": 2,
                          "state_width": config["model"]["state_width"], "world_model": False,
                          "workers_per_rank": workers, "trial_updates": args.trial_updates,
                          "checkpoint_config_workers_per_rank": config["workers_per_rank"]}))
        return
    from method.full_diffusion2.launch import guard_context
    guard_context()
    if int(os.environ.get("WORLD_SIZE", "0")) != config["world_size"]:
        raise RuntimeError("Full Diffusion2 requires exactly eight distributed ranks")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(seconds=300))
    probe = torch.tensor(float(rank), device=device)
    distributed.all_reduce(probe)
    if float(probe) != 28.0:
        raise AssertionError("Eight-rank NCCL all-reduce failed")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    index = ROOT / config["index_directory"]
    dataset_args = {"backbone_directory": ROOT / config["backbone_directory"],
                    "history_frames": config["model"]["history_frames"], "history_stride": config["history_stride"]}
    dataset = GarageTemporalDataset(index, **dataset_args)
    validation = GarageTemporalDataset(index, split="val", **dataset_args)
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device)
    head = TemporalFullDiffusionHead(**config["model"], normalization=manifest["normalization"]).to(device)
    policy = DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    diffusion = VPFullDiffusion(head.codec, **{key: value for key, value in config["diffusion"].items()
                                            if key != "inference_steps"})
    torch.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_status(args.output / "run_config.json", config)
        write_status(args.output / "dataset.json", manifest)
        source_paths = list(Path(__file__).parent.glob("*.py")) + [args.config]
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in source_paths}
        previous_hashes = args.output / "code_sha256.json"
        initial_hashes = args.output / "code_sha256.initial.json"
        if previous_hashes.exists() and not initial_hashes.exists():
            os.link(previous_hashes, initial_hashes)
        write_status(args.output / "code_sha256.json", hashes)
    if args.phase == "acceptance":
        acceptance(args, config, manifest, perception, head, policy, optimizer, scheduler, diffusion, dataset, device)
        distributed.destroy_process_group()
        return
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    progress = {"epoch": 0, "next_batch": 0, "updates": 0}
    if args.resume:
        progress = load_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, config, manifest)
    elif (args.output / "latest.pt").exists():
        raise FileExistsError("Existing checkpoint requires explicit --resume")
    sampler = DistributedSampler(dataset, seed=config["seed"], drop_last=True)
    batches = BatchSampler(sampler, config["batch_per_gpu"], drop_last=True)
    updates_per_epoch = len(batches)
    started = time.monotonic()
    starting_updates = progress["updates"]
    runtime_session = f"step{starting_updates}_workers{workers}_{time.time_ns()}"
    runtime = {"runtime_session": runtime_session, "workers_per_rank": workers,
               "checkpoint_config_workers_per_rank": config["workers_per_rank"],
               "validation_workers_per_rank": 0, "prefetch_factor": 2,
               "trial_updates": args.trial_updates, "resumed": args.resume,
               "starting_updates": starting_updates,
               "started_at": datetime.now(timezone.utc).isoformat()}
    if rank == 0:
        write_status(args.output / "runtime.json", runtime)
        with (args.output / "runtime_history.jsonl").open("a") as handle:
            handle.write(json.dumps(runtime) + "\n")
    running_loss = torch.zeros((), device=device)
    running_count = 0
    for epoch in range(progress["epoch"], config["epochs"]):
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        offset = progress["next_batch"] if epoch == progress["epoch"] else 0
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
            trial_reached = (args.trial_updates is not None
                             and progress["updates"] - starting_updates >= args.trial_updates)
            stop = torch.tensor(int(STOP_REQUESTED or trial_reached), device=device)
            distributed.all_reduce(stop, op=distributed.ReduceOp.MAX)
            if (progress["updates"] == starting_updates + 1
                    or progress["updates"] % config["log_every_updates"] == 0):
                distributed.all_reduce(running_loss)
                elapsed = time.monotonic() - started
                record = {"epoch": epoch + 1, "batch": batch_id + 1, "updates_per_epoch": updates_per_epoch,
                          "updates": progress["updates"], "loss": float(running_loss) / (running_count * 8),
                          "lr": optimizer.param_groups[0]["lr"], "gradient_norm_rank0": float(norm),
                          "elapsed_seconds": elapsed,
                          "samples_per_second": (progress["updates"] - starting_updates) * 64 / elapsed,
                          "peak_gpu_gib_rank0": torch.cuda.max_memory_allocated() / 1024**3,
                          "workers_per_rank": workers, "runtime_session": runtime_session,
                          "starting_updates": starting_updates,
                          "utc": datetime.now(timezone.utc).isoformat()}
                if rank == 0:
                    with (args.output / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    write_status(args.output / "progress.json", record)
                    print(json.dumps(record), flush=True)
                running_loss.zero_()
                running_count = 0
            if (progress["updates"] == starting_updates + 1
                    or progress["updates"] % config["checkpoint_every_updates"] == 0 or int(stop)):
                save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest)
            if int(stop):
                if rank == 0 and trial_reached:
                    write_status(args.output / "worker_trial_completed.json",
                                 {**runtime, "progress": progress,
                                  "elapsed_seconds": time.monotonic() - started,
                                  "completed_at": datetime.now(timezone.utc).isoformat()})
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