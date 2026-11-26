"""Resume the unchanged navigation planner on six GPUs with eight RNG streams."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
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
from method.full_diffusionv3_ego12.allocation import job_memory
from . import FORMAT, METHOD
from . import train as original
from .recovery_data import RecoveryDataset, VirtualBatches, ValidationBatches
from .utils import (ROOT, atomic_json, atomic_save, inspect_allocation, move,
                    restore_rng, rng_state, seed, sha, verify_data)


@torch.no_grad()
def validate(dataset, perception, head, diffusion, config, device, virtual_ranks):
    routes = {}
    was_training = head.training
    head.eval()
    for virtual in virtual_ranks:
        loader = DataLoader(dataset, batch_sampler=ValidationBatches(dataset, virtual, 8, 8),
                            num_workers=0, pin_memory=True,
                            generator=torch.Generator().manual_seed(config["seed"] + 700001 + virtual))
        generator = torch.Generator(device=device).manual_seed(config["validation"]["fixed_noise_seed"] + virtual)
        for value in loader:
            batch = original.encode_context(move(value, device), perception, config)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                condition = head.prepare_condition(*original.condition_inputs(batch))
                sample = diffusion.sample(lambda x, t: head(x, t, condition)["predictions"][-1],
                                          len(batch["path"]), device, generator, 25)
                path, delta = head.codec.decode(sample)
            if not bool(torch.isfinite(path).all() and torch.isfinite(delta).all()):
                raise FloatingPointError("Nonfinite fixed-validation trajectory")
            errors = torch.stack(((path.float() - batch["path"]).norm(dim=-1).mean(-1),
                                  (delta.float().cumsum(1) - batch["future_positions"]).norm(dim=-1).mean(-1),
                                  (head.codec.target_speed(delta) - head.codec.target_speed(batch["temporal_delta"])).abs()), -1)
            for route, values in zip(batch["route_id"].tolist(), errors.cpu().tolist()):
                row = routes.setdefault(str(route), [0., 0., 0., 0])
                for i, value in enumerate(values):
                    row[i] += value
                row[3] += 1
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, routes)
    combined = {}
    for values in gathered:
        if combined.keys() & values.keys():
            raise RuntimeError("A validation route has multiple owners")
        combined.update(values)
    if len(combined) != 44 or sum(v[3] for v in combined.values()) != 2223:
        raise RuntimeError("The fixed 44-route / 2223-anchor validation changed")
    macro = np.mean([np.asarray(combined[k][:3]) / combined[k][3] for k in sorted(combined)], 0)
    report = dict(zip(("route_macro_spatial_ade_m", "route_macro_future_ade_m",
                       "route_macro_control_speed_mae_mps"), macro.tolist()))
    report.update(routes=44, samples=2223, inference_steps=25, per_route=combined,
                  virtual_shards=8, physical_world_size=dist.get_world_size(), unix=time.time())
    head.train(was_training)
    return report


def save_latest(output, head, optimizer, scheduler, config, metadata, progress, stream_states, initialization):
    torch.cuda.synchronize()
    dist.barrier()
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(stream_states, gathered, dst=0)
    if dist.get_rank() == 0:
        streams = {}
        for group in gathered:
            if streams.keys() & group.keys():
                raise RuntimeError("Duplicate RNG-stream owner")
            streams.update(group)
        if sorted(streams) != list(range(8)):
            raise RuntimeError("Lost an original RNG stream")
        state = {"format": FORMAT, "method": METHOD, "model": head.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "progress": dict(progress), "config": config, "dataset": metadata,
                 "rank_states": [streams[i] for i in range(8)], "initialization": initialization,
                 "physical_world_size": dist.get_world_size(), "logical_world_size": 8}
        atomic_save(output / "latest.pt", state)
        atomic_json(output / "latest.json", {"progress": progress, "optimizer_saved": True,
                    "scheduler_saved": True, "rng_saved_for_ranks": 8, "physical_ranks": dist.get_world_size(),
                    "sha256": sha(output / "latest.pt"), "bytes": (output / "latest.pt").stat().st_size,
                    "unix": time.time()})
    dist.barrier()


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
    if world != 6 or rank != 2 * config["jobs"][args.job_id]["node_rank"] + local:
        raise RuntimeError("Unexpected physical allocation/rank mapping")
    groups = config["virtual_rank_groups"]
    if sorted(v for group in groups for v in group) != list(range(8)) or len(groups) != world:
        raise RuntimeError("Expected all eight original 32-sample streams exactly once")
    virtual_ranks = groups[rank]
    output = Path(config["output"])
    metadata = verify_data(config)
    dataset, validation = RecoveryDataset(config), RecoveryDataset(config, "val")
    head, diffusion, _, _ = original.build(config, device, initialize=False)
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device).eval()
    if any(p.requires_grad for p in perception.parameters()):
        raise RuntimeError("TF++ must stay frozen")
    settings = config["training"]
    if settings["batch_per_gpu"] != 32 or config["global_batch"] != 256:
        raise RuntimeError("Recovery changed the optimization batch geometry")
    steps = len(dataset) // 256
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    path = output / "latest.pt" if args.resume else Path(config["resume_from"])
    if not args.resume and sha(path) != config["resume_checkpoint_sha256"]:
        raise ValueError("Recovery source checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format"] != FORMAT or state["method"] != METHOD or state["dataset"] != metadata:
        raise ValueError("Resume architecture or frozen data identity changed")
    operational = {"world_size", "jobs", "output", "master_port", "train_stop_unix",
                   "validate_initialization", "virtual_rank_groups", "resume_from",
                   "resume_checkpoint_sha256", "recovery_source_run", "recovery_preflight"}
    differences = {k for k in set(state["config"]) | set(config) if state["config"].get(k) != config.get(k)}
    if (args.resume and differences) or (not args.resume and differences - operational):
        raise ValueError(f"Recovery changed optimization/data/model settings: {sorted(differences)}")
    head.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    progress, initialization = state["progress"], state["initialization"]
    generators = {v: torch.Generator(device=device) for v in virtual_ranks}
    stream_states = {v: state["rank_states"][v] for v in virtual_ranks}
    for virtual in virtual_ranks:
        generators[virtual].set_state(stream_states[virtual]["generator"])
    del state
    ddp = DDP(head, device_ids=[local], broadcast_buffers=False)
    if rank == 0:
        atomic_json(output / "training_started.json", {"unix": time.time(), "world_size": world,
                    "logical_world_size": 8, "global_batch": 256, "steps_per_epoch": steps,
                    "epochs": settings["epochs"], "resume_progress": progress, "resume_from": str(path),
                    "train_samples": len(dataset), "val_samples": len(validation),
                    "virtual_rank_groups": groups, "changed_operational_fields": sorted(differences)})
    if local == 0:
        atomic_json(output / f"allocation_{args.job_id}.json", proof)
    signal.signal(signal.SIGTERM, original.request_stop)
    signal.signal(signal.SIGINT, original.request_stop)
    initial_updates, began = progress["updates"], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        while progress["epoch"] < settings["epochs"]:
            epoch = progress["epoch"]
            if not progress["validation_pending"]:
                workers = config["jobs"][args.job_id]["workers_per_rank"]
                loader = DataLoader(dataset, batch_sampler=VirtualBatches(len(dataset), virtual_ranks,
                    config["seed"], epoch, progress["step_in_epoch"]), num_workers=workers, pin_memory=True,
                    prefetch_factor=settings["prefetch_factor"], persistent_workers=False, timeout=300,
                    worker_init_fn=original.worker_init, multiprocessing_context="spawn",
                    generator=torch.Generator().manual_seed(config["seed"] + rank + epoch * 100))
                previous, running, count = time.monotonic(), defaultdict(float), 0
                for value in loader:
                    wait = time.monotonic() - previous
                    start = time.monotonic()
                    encoded = original.encode_context(move(value, device), perception, config)
                    optimizer.zero_grad(set_to_none=True)
                    for index, virtual in enumerate(virtual_ranks):
                        restore_rng(stream_states[virtual], generators[virtual])
                        batch = {k: v[index * 32:(index + 1) * 32] if isinstance(v, torch.Tensor) else v
                                 for k, v in encoded.items()}
                        sync = nullcontext() if index == len(virtual_ranks) - 1 else ddp.no_sync()
                        with sync:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                clean = head.codec.encode(batch["path"], batch["temporal_delta"])
                                times = torch.randint(diffusion.train_steps, (32,), device=device, generator=generators[virtual])
                                noise = torch.randn(clean.shape, device=device, generator=generators[virtual])
                                noisy, time_value = diffusion.corrupt(clean, times, noise)
                                result = ddp(noisy, time_value, original.condition_inputs(batch))
                                losses = original.losses_for(head, result, clean, batch, config)
                            if not bool(torch.isfinite(losses["loss"])):
                                raise FloatingPointError("Nonfinite recovery loss")
                            # DDP averages six physical ranks: this factor gives
                            # exactly the original average over eight equal batches.
                            (losses["loss"] * (world / 8.)).backward()
                        for key in original.FIELDS:
                            running[key] += float(losses[key].detach()) * (world / 8.)
                        stream_states[virtual] = rng_state(generators[virtual])
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    progress["step_in_epoch"] += 1
                    progress["updates"] += 1
                    scheduler.step(epoch + progress["step_in_epoch"] / steps)
                    running["wait"] += wait
                    running["compute"] += time.monotonic() - start
                    count += 1
                    update = progress["updates"]
                    should_stop = original.STOP or time.time() >= config["train_stop_unix"]
                    if update == initial_updates + 1 or update % 25 == 0:
                        memory = job_memory(args.job_id)
                        should_stop |= memory["noncache_bytes"] > .78 * memory["limit_bytes"]
                        if local == 0:
                            atomic_json(output / f"memory_{args.job_id}.json", {**memory, "updates": update})
                    stop = torch.tensor(int(should_stop), device=device)
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if update == initial_updates + 1 or update % settings["log_every"] == 0:
                        torch.cuda.synchronize()
                        totals = torch.tensor([running[k] / count for k in original.FIELDS], device=device)
                        dist.all_reduce(totals)
                        totals /= world
                        record = {**progress, "rank": rank, "job": args.job_id, "virtual_ranks": virtual_ranks,
                                  **head.navigation_statistics(), "gradient_norm": float(norm),
                                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                                  "workers": workers, "unix": time.time()}
                        atomic_json(output / f"rank_{rank}.json", record)
                        if rank == 0:
                            record.update(epoch_display=epoch + 1, updates_per_epoch=steps,
                                **dict(zip(original.FIELDS, totals.cpu().tolist())), lr=optimizer.param_groups[0]["lr"],
                                data_wait_seconds_rank0=running["wait"] / count,
                                compute_seconds_rank0=running["compute"] / count, elapsed_seconds=time.monotonic() - began,
                                world_size=world, logical_world_size=8, global_batch=256)
                            with (output / "metrics.jsonl").open("a") as handle:
                                handle.write(json.dumps(record) + "\n")
                            atomic_json(output / "progress.json", record)
                            print(json.dumps(record), flush=True)
                        running.clear()
                        count = 0
                    if update == initial_updates + 1 or update % settings["checkpoint_every"] == 0 or bool(stop):
                        save_latest(output, head, optimizer, scheduler, config, metadata, progress, stream_states, initialization)
                    previous = time.monotonic()
                    if bool(stop):
                        if rank == 0:
                            atomic_json(output / "paused.json", {**progress, "unix": time.time(), "reason": "signal/deadline/RAM guard"})
                        return
                del loader
                progress["validation_pending"] = True
                save_latest(output, head, optimizer, scheduler, config, metadata, progress, stream_states, initialization)
            report = validate(validation, perception, head, diffusion, config, device, virtual_ranks)
            score = report[config["validation"]["selection_metric"]]
            improved = progress["best_score"] is None or score < progress["best_score"]
            progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=False)
            if improved:
                progress.update(best_score=score, best_epoch=epoch + 1)
            save_latest(output, head, optimizer, scheduler, config, metadata, progress, stream_states, initialization)
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
