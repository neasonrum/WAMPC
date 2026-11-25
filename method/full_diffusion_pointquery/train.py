"""Eight-A100 pure-E2E point-query training on the frozen dev44 protocol."""
import argparse
from collections import defaultdict
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3_ego12.allocation import job_memory
from . import METHOD, FORMAT
from .data import PointDataset, GlobalBatches, ValidationBatches
from .model import PointQueryHead, load_compatible_ego12
from .utils import (ROOT, atomic_json, atomic_save, inspect_allocation, move, restore_rng,
                    rng_state, seed, sha, verify_data)

STOP = False
FIELDS = ("loss", "spatial_l1", "temporal_delta_l1", "behavior_ce")


def request_stop(*_):
    global STOP
    STOP = True


def build(config, device, initialize=True):
    seed(config["seed"])
    artifact = json.loads((ROOT / config["normalization_file"]).read_text())
    if artifact["label_config"] != config["labels"]:
        raise ValueError("Planner labels and coordinate normalization differ")
    head = PointQueryHead(**config["model"], normalization=artifact["normalization"])
    report = None
    if initialize:
        if sha(config["planner_checkpoint"]) != config["planner_checkpoint_sha256"]:
            raise ValueError("The agreed ego12 epoch10 initialization changed")
        checkpoint = torch.load(config["planner_checkpoint"], map_location="cpu", weights_only=False)
        report = load_compatible_ego12(head, checkpoint)
        del checkpoint
    head = head.to(device)
    diffusion = SpatialTemporalVPFullDiffusion(head.codec, **{
        k: v for k, v in config["diffusion"].items() if k != "inference_steps"})
    return head, diffusion, artifact, report


@torch.no_grad()
def encode_context(batch, perception, config):
    bevs, memories, egos = [], [], []
    for start in range(0, len(batch["rgb"]), config["perception_chunk"]):
        inputs = {key: batch[key][start:start + config["perception_chunk"]]
                  for key in ("rgb", "lidar", "speed", "command", "target_point")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bev, memory, ego = perception(inputs)
        bevs.append(bev)
        memories.append(memory)
        egos.append(ego)
    batch.update(bev=torch.cat(bevs), memory=torch.cat(memories), ego=torch.cat(egos))
    del batch["rgb"], batch["lidar"]
    return batch


def condition_inputs(batch):
    return tuple(batch[key] for key in ("bev", "memory", "ego", "target_point", "history_bev",
                                       "history_transforms", "history_mask", "speed", "command"))


def losses_for(head, output, clean, batch, config):
    return head.loss_components(output, clean, batch["spatial_mask"], batch["temporal_mask"],
                                batch["behavior_targets"], batch["behavior_mask"], config["loss"])


@torch.no_grad()
def validate(dataset, perception, head, diffusion, config, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    loader = DataLoader(dataset, batch_sampler=ValidationBatches(dataset, rank, world, 8),
                        num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(config["seed"] + 700001 + rank))
    generator = torch.Generator(device=device).manual_seed(config["validation"]["fixed_noise_seed"] + rank)
    was_training = head.training
    head.eval()
    routes = {}
    for value in loader:
        batch = encode_context(move(value, device), perception, config)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition = head.prepare_condition(*condition_inputs(batch))
            sample = diffusion.sample(lambda noisy, t: head(noisy, t, condition)["predictions"][-1],
                                      len(batch["path"]), device, generator, 25)
            path, delta = head.codec.decode(sample)
        if not bool(torch.isfinite(path).all() and torch.isfinite(delta).all()):
            raise FloatingPointError("Nonfinite 25-step validation trajectories")
        errors = torch.stack(((path.float() - batch["path"]).norm(dim=-1).mean(-1),
                              (delta.float().cumsum(1) - batch["future_positions"]).norm(dim=-1).mean(-1),
                              (head.codec.target_speed(delta) - head.codec.target_speed(batch["temporal_delta"])).abs()), -1)
        for route, values in zip(batch["route_id"].tolist(), errors.cpu().tolist()):
            item = routes.setdefault(str(route), [0., 0., 0., 0])
            for i, value in enumerate(values):
                item[i] += value
            item[3] += 1
    gathered = [None] * world
    dist.all_gather_object(gathered, routes)
    combined = {}
    for values in gathered:
        if combined.keys() & values.keys():
            raise RuntimeError("Validation route was assigned to multiple ranks")
        combined.update(values)
    if len(combined) != 44 or sum(value[3] for value in combined.values()) != 2223:
        raise RuntimeError("Validation must cover all fixed 44 routes and 2223 anchors")
    macro = np.mean([np.asarray(value[:3]) / value[3] for value in combined.values()], 0)
    report = dict(zip(("route_macro_spatial_ade_m", "route_macro_future_ade_m",
                       "route_macro_control_speed_mae_mps"), macro.tolist()))
    report.update(routes=44, samples=2223, inference_steps=25, per_route=combined, unix=time.time())
    head.train(was_training)
    return report


def save_latest(output, head, optimizer, scheduler, config, metadata, progress, generator, initialization):
    torch.cuda.synchronize()
    dist.barrier()
    local = rng_state(generator)
    states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, states, dst=0)
    if dist.get_rank() == 0:
        state = {"format": FORMAT, "method": METHOD, "model": head.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "progress": dict(progress), "config": config,
                 "dataset": metadata, "rank_states": states, "initialization": initialization}
        atomic_save(output / "latest.pt", state)
        atomic_json(output / "latest.json", {"progress": progress, "optimizer_saved": True,
                    "scheduler_saved": True, "rng_saved_for_ranks": len(states),
                    "sha256": sha(output / "latest.pt"), "bytes": (output / "latest.pt").stat().st_size,
                    "unix": time.time()})
    dist.barrier()


def worker_init(_):
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    proof = inspect_allocation(config, args.job_id)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or rank != 2 * config["jobs"][args.job_id]["node_rank"] + local:
        raise RuntimeError("Rank does not belong to the authorized eight-A100 experiment")
    output = Path(config["output"])
    metadata = verify_data(config)
    dataset, validation = PointDataset(config), PointDataset(config, "val")
    head, diffusion, artifact, initialization = build(config, device, initialize=not args.resume)
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device).eval()
    if any(p.requires_grad for p in perception.parameters()):
        raise ValueError("TF++ must remain frozen")
    settings = config["training"]
    steps = len(dataset) // (settings["batch_per_gpu"] * world)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + rank + 100003)
    seed(config["seed"] + rank + 200003)
    progress = {"epoch": 0, "step_in_epoch": 0, "updates": 0, "best_score": None,
                "best_epoch": None, "validation_pending": False}
    if args.resume:
        state = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        if state["format"] != FORMAT or state["config"] != config or state["dataset"] != metadata:
            raise ValueError("Resume config, architecture, or fixed data identity mismatch")
        head.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        progress, initialization = state["progress"], state["initialization"]
        restore_rng(state["rank_states"][rank], generator)
        del state
    elif (output / "latest.pt").exists():
        raise FileExistsError("Existing training state requires explicit --resume")
    ddp = DDP(head, device_ids=[local], broadcast_buffers=False)
    if rank == 0:
        atomic_json(output / "initialization.json", initialization)
        atomic_json(output / "training_started.json", {"unix": time.time(), "world_size": world,
            "global_batch": world * settings["batch_per_gpu"], "steps_per_epoch": steps,
            "epochs": settings["epochs"], "train_samples": len(dataset), "val_samples": len(validation),
            "dropped_tail_per_epoch": len(dataset) % (world * settings["batch_per_gpu"]), "resume": args.resume})
    if local == 0:
        atomic_json(output / f"allocation_{args.job_id}.json", proof)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start_time, initial_updates = time.monotonic(), progress["updates"]
    torch.cuda.reset_peak_memory_stats()
    try:
        while progress["epoch"] < settings["epochs"]:
            epoch = progress["epoch"]
            if not progress["validation_pending"]:
                batches = GlobalBatches(len(dataset), settings["batch_per_gpu"], rank, world,
                                        config["seed"], epoch, progress["step_in_epoch"])
                workers = config["jobs"][args.job_id]["workers_per_rank"]
                loader = DataLoader(dataset, batch_sampler=batches, num_workers=workers, pin_memory=True,
                    prefetch_factor=settings["prefetch_factor"] if workers else None,
                    persistent_workers=False, timeout=300 if workers else 0, worker_init_fn=worker_init,
                    multiprocessing_context="spawn" if workers else None,
                    generator=torch.Generator().manual_seed(config["seed"] + rank + epoch * 100))
                previous, running = time.monotonic(), defaultdict(float)
                count = 0
                for value in loader:
                    wait_time = time.monotonic() - previous
                    stage_start = time.monotonic()
                    batch = encode_context(move(value, device), perception, config)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        clean = head.codec.encode(batch["path"], batch["temporal_delta"])
                        times = torch.randint(diffusion.train_steps, (len(clean),), device=device, generator=generator)
                        noise = torch.randn(clean.shape, device=device, generator=generator)
                        noisy, normalized_time = diffusion.corrupt(clean, times, noise)
                        result = ddp(noisy, normalized_time, condition_inputs(batch))
                        losses = losses_for(head, result, clean, batch, config)
                    if not bool(torch.isfinite(losses["loss"])):
                        raise FloatingPointError("Nonfinite point-query denoising loss")
                    losses["loss"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    progress["step_in_epoch"] += 1
                    progress["updates"] += 1
                    scheduler.step(epoch + progress["step_in_epoch"] / steps)
                    for key in FIELDS:
                        running[key] += float(losses[key].detach())
                    running["data_wait_seconds"] += wait_time
                    running["compute_seconds"] += time.monotonic() - stage_start
                    count += 1
                    update = progress["updates"]
                    should_stop = STOP or time.time() >= config["train_stop_unix"]
                    if update == initial_updates + 1 or update % 25 == 0:
                        memory = job_memory(args.job_id)
                        should_stop |= memory["noncache_bytes"] > .78 * memory["limit_bytes"]
                        if local == 0:
                            atomic_json(output / f"memory_{args.job_id}.json", {**memory, "updates": update})
                    stop = torch.tensor(int(should_stop), device=device)
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if update == initial_updates + 1 or update % settings["log_every"] == 0:
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - start_time
                        totals = torch.tensor([running[k] / count for k in FIELDS], device=device)
                        dist.all_reduce(totals)
                        totals /= world
                        heartbeat = {**progress, "rank": rank, "job": args.job_id,
                            "gradient_norm": float(norm), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                            "workers": workers, "unix": time.time()}
                        atomic_json(output / f"rank_{rank}.json", heartbeat)
                        if rank == 0:
                            record = {**heartbeat, "epoch_display": epoch + 1, "updates_per_epoch": steps,
                                **dict(zip(FIELDS, totals.cpu().tolist())), "lr": optimizer.param_groups[0]["lr"],
                                "data_wait_seconds_rank0": running["data_wait_seconds"] / count,
                                "compute_seconds_rank0": running["compute_seconds"] / count,
                                "elapsed_seconds": elapsed, "world_size": world,
                                "global_batch": world * settings["batch_per_gpu"],
                                "samples_per_second": (update - initial_updates) * world * settings["batch_per_gpu"] / elapsed}
                            with (output / "metrics.jsonl").open("a") as handle:
                                handle.write(json.dumps(record) + "\n")
                            atomic_json(output / "progress.json", record)
                            print(json.dumps(record), flush=True)
                        running.clear()
                        count = 0
                    if update == initial_updates + 1 or update % settings["checkpoint_every"] == 0 or bool(stop):
                        save_latest(output, head, optimizer, scheduler, config, metadata, progress, generator, initialization)
                    previous = time.monotonic()
                    if bool(stop):
                        if rank == 0:
                            atomic_json(output / "paused.json", {**progress, "unix": time.time(), "reason": "signal/deadline/RAM guard"})
                        return
                del loader
                progress["validation_pending"] = True
                save_latest(output, head, optimizer, scheduler, config, metadata, progress, generator, initialization)
            report = validate(validation, perception, head, diffusion, config, device)
            score = report[config["validation"]["selection_metric"]]
            improved = progress["best_score"] is None or score < progress["best_score"]
            progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=False)
            if improved:
                progress.update(best_score=score, best_epoch=epoch + 1)
            save_latest(output, head, optimizer, scheduler, config, metadata, progress, generator, initialization)
            if rank == 0:
                state = {"format": FORMAT, "method": METHOD, "model": head.state_dict(), "config": config,
                         "dataset": metadata, "progress": dict(progress), "initialization": initialization}
                atomic_save(output / f"epoch_{epoch + 1:02}.pt", state)
                if improved:
                    atomic_save(output / "best.pt", state)
                    atomic_json(output / "best.json", {"epoch": epoch + 1, "score": score,
                        "metric": config["validation"]["selection_metric"], "checkpoint": str(output / "best.pt"),
                        "sha256": sha(output / "best.pt"), "unix": time.time()})
                atomic_json(output / f"validation_epoch_{epoch + 1:02}.json", report)
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, **{k: v for k, v in report.items() if k != "per_route"}}) + "\n")
                print(json.dumps({"epoch": epoch + 1, "validation": {k: v for k, v in report.items() if k != "per_route"}}), flush=True)
            dist.barrier()
        if rank == 0:
            atomic_json(output / "completed.json", {**progress, "unix": time.time()})
    except BaseException as error:
        atomic_json(output / f"rank_{rank}.failure.{time.time_ns()}.json", {"error": repr(error), "progress": progress, "unix": time.time()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

