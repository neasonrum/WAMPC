"""Independent V3 planner training; execution requires explicit GPU approval."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset

from method.full_diffusion2.launch import guard_context
from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.data import GarageSpatialTemporalDataset, check_label_settings
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead
from method.full_diffusionv3.checkpoint import load_checkpoint, save_checkpoint, select_complete
from method.full_diffusionv3.runtime import (
    ControlGroup, PhaseTrace, RemainingBatches, coordinated_next, gradient_hook,
    local_directory, make_training_loader,
)
from tools.launch_full_diffusion import write_status
from train_full_diffusion import autocast, move_batch


STOP_REQUESTED = False
COMPONENTS = ("loss", "spatial_l1", "temporal_delta_l1", "behavior_ce")


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def training_metadata(config):
    if config.get("method") != "full_diffusionv3":
        raise ValueError("This entry point only trains full_diffusionv3")
    if (config.get("architecture") != "independent_dual_query64"
            or config.get("controller") != "automot_temporal_pid_v1"
            or config["model"]["slot_width"] != 64):
        raise ValueError("V3 requires independent dual 256D queries, 64D slots, and AutoMoT temporal PID")
    index = ROOT / config["index_directory"]
    manifest = json.loads((index / "manifest.json").read_text())
    if manifest["smoke_only"]:
        raise ValueError("V3 training requires the full retained training index")
    artifact = json.loads((ROOT / config["normalization_file"]).read_text())
    if artifact.get("method") != "full_diffusionv3" or artifact["label_config"] != config["labels"]:
        raise ValueError("V3 normalization identity or label settings differ")
    for name, expected in artifact["source_sha256"].items():
        if hashlib.sha256((index / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"The index changed after V3 normalization: {name}")
    if artifact["train_frames"] != manifest["train_frames"] or min(artifact["valid_slot_counts"]) <= 0:
        raise ValueError("V3 normalization has inconsistent training target counts")
    check_label_settings(config["labels"])
    if not np.isclose(config["model"]["temporal_interval_seconds"], config["labels"]["temporal_interval_seconds"]):
        raise ValueError("Model and supervision temporal intervals differ")
    if config["loss"]["temporal_interval_weights"] != [3, 3, 2, 2, 1, 1]:
        raise ValueError("V3 temporal supervision must match AutoMoT's six weighted offsets")
    if not np.isclose(config["observation_interval_seconds"], config["labels"]["raw_measurement_interval_seconds"] * config["history_stride"]):
        raise ValueError("Online history cadence differs from training")
    sources = {}
    if config.get("source_identity_file"):
        sources = json.loads(Path(config["source_identity_file"]).read_text())
        for name, expected in sources.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen V3 source changed: {name}")
    return {**manifest, "method": "full_diffusionv3", "v3_normalization": artifact,
            "runtime_source_sha256": sources}


def condition_inputs(batch, perception, config):
    bev, memory, ego = perception(batch)
    history = perception.encode_history(batch, chunk=config["history_encode_chunk"])
    return (bev, memory, ego, batch["target_point"], history, batch["history_transforms"],
            batch["history_mask"], batch["speed"], batch["command"])


def compute_losses(head, output, clean, batch, config):
    return head.loss_components(output, clean, batch["spatial_mask"], batch["temporal_mask"],
                                batch["behavior_targets"], batch["behavior_mask"], config["loss"])


def update(batch, perception, policy, optimizer, diffusion, config, control, trace):
    head = policy.module
    optimizer.zero_grad(set_to_none=True)
    with autocast(config):
        with trace.stage("perception", cuda=True):
            inputs = condition_inputs(batch, perception, config)
        with trace.stage("forward", cuda=True):
            clean = head.codec.encode(batch["path"], batch["temporal_delta"])
            noisy, time_step = diffusion.corrupt(clean)
            output = policy(noisy, time_step, inputs)
            losses = compute_losses(head, output, clean, batch, config)
    with trace.stage("loss_control"):
        if control.any(not bool(torch.isfinite(losses["loss"]))):
            raise FloatingPointError("Nonfinite V3 denoising or behavior loss on a rank")
    try:
        with trace.stage("backward", cuda=True):
            losses["loss"].backward()
        with trace.stage("clip", cuda=True):
            norm = nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        with trace.stage("optimizer", cuda=True):
            optimizer.step()
    except BaseException:
        control.failed = True
        raise
    return {name: value.detach() for name, value in losses.items()}, norm.detach()


@torch.no_grad()
def validate(dataset, perception, head, diffusion, config, device, control, trace):
    rank, world = distributed.get_rank(), distributed.get_world_size()
    loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)),
                        batch_size=config["batch_per_gpu"], num_workers=0,
                        generator=torch.Generator().manual_seed(config["seed"] + 700001 + rank))
    names = list(COMPONENTS) + [
        "frames", "behavior_valid", "sampled_frames", "path_distance", "path_valid",
        "path_final_distance", "path_final_valid", "future_distance", "future_valid",
        "future_final_distance", "future_final_valid", "interval_speed_error", "interval_valid",
        "control_speed_error", "control_speed_valid",
    ]
    totals = {name: torch.zeros((), dtype=torch.float64, device=device) for name in names}
    confusion = torch.zeros((3, 4, 4), dtype=torch.float64, device=device)
    interval_totals = torch.zeros((6, 2), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + rank)
    head.eval()
    iterator = iter(loader)
    for batch_id in range(len(loader)):
        with trace.stage("validation_get"):
            batch = next(iterator)
        batch = move_batch(batch, device)
        count = len(batch["path"])
        with autocast(config), trace.stage("validation_forward", cuda=True):
            condition = head.prepare_condition(*condition_inputs(batch, perception, config))
            clean = head.codec.encode(batch["path"], batch["temporal_delta"])
            timesteps = torch.randint(diffusion.train_steps, (count,), generator=generator, device=device)
            noise = torch.randn(clean.shape, generator=generator, device=device)
            noisy, time_step = diffusion.corrupt(clean, timesteps, noise)
            output = head(noisy, time_step, condition)
            losses = compute_losses(head, output, clean, batch, config)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite V3 validation loss")
            valid_behavior = batch["behavior_mask"].sum()
            for name in COMPONENTS:
                totals[name] += losses[name].double() * (valid_behavior if name == "behavior_ce" else count)
            totals["frames"] += count
            totals["behavior_valid"] += valid_behavior
            classes = output["behavior_logits"].argmax(dim=-1)
            for horizon in range(3):
                valid = batch["behavior_mask"][:, horizon]
                pairs = batch["behavior_targets"][valid, horizon] * 4 + classes[valid, horizon]
                confusion[horizon] += torch.bincount(pairs, minlength=16).reshape(4, 4)
            if batch_id >= config["validation_sample_batches_per_rank"]:
                continue
            with trace.stage("validation_sample", cuda=True):
                sample = diffusion.sample(
                    lambda noisy, time_step: head(noisy, time_step, condition)["predictions"][-1],
                    count, device, generator, config["diffusion"]["inference_steps"],
                )
            path, delta = head.codec.decode(sample)
        if not torch.isfinite(path).all() or not torch.isfinite(delta).all():
            raise FloatingPointError("V3 full-noise sampling generated nonfinite coordinates")
        path_distance = (path.float() - batch["path"]).norm(dim=-1)
        path_mask = batch["spatial_mask"]
        positions = delta.float().cumsum(dim=1)
        future_distance = (positions - batch["future_positions"]).norm(dim=-1)
        future_mask = batch["future_position_mask"]
        interval_mask = batch["temporal_mask"]
        interval_speed = delta.float().norm(dim=-1) / head.codec.temporal_interval_seconds
        target_interval_speed = batch["temporal_delta"].float().norm(dim=-1) / head.codec.temporal_interval_seconds
        interval_error = (interval_speed - target_interval_speed).abs()
        totals["sampled_frames"] += count
        totals["path_distance"] += (path_distance * path_mask).sum().double()
        totals["path_valid"] += path_mask.sum()
        totals["path_final_distance"] += (path_distance[:, -1] * path_mask[:, -1]).sum().double()
        totals["path_final_valid"] += path_mask[:, -1].sum()
        totals["future_distance"] += (future_distance * future_mask).sum().double()
        totals["future_valid"] += future_mask.sum()
        totals["future_final_distance"] += (future_distance[:, -1] * future_mask[:, -1]).sum().double()
        totals["future_final_valid"] += future_mask[:, -1].sum()
        totals["interval_speed_error"] += (interval_error * interval_mask).sum().double()
        totals["interval_valid"] += interval_mask.sum()
        totals["control_speed_error"] += (interval_error[:, 1] * interval_mask[:, 1]).sum().double()
        totals["control_speed_valid"] += interval_mask[:, 1].sum()
        interval_totals += torch.stack(((interval_error * interval_mask).sum(dim=0),
                                        interval_mask.sum(dim=0)), dim=-1).double()
    reduced = torch.stack([totals[name] for name in names])
    with trace.stage("validation_reduce"):
        reduced = control.reduce(reduced)
        confusion = control.reduce(confusion)
        interval_totals = control.reduce(interval_totals)
    totals = dict(zip(names, reduced))

    def ratio(numerator, denominator):
        return float(totals[numerator] / totals[denominator]) if int(totals[denominator]) else None

    metrics = {name: ratio(name, "behavior_valid" if name == "behavior_ce" else "frames")
               for name in COMPONENTS}
    metrics.update({
        "frames": int(totals["frames"]), "sampled_frames": int(totals["sampled_frames"]),
        "sampled_path_ade_m": ratio("path_distance", "path_valid"),
        "sampled_path_fde_m": ratio("path_final_distance", "path_final_valid"),
        "sampled_future_ade_m": ratio("future_distance", "future_valid"),
        "sampled_future_fde_m": ratio("future_final_distance", "future_final_valid"),
        "sampled_interval_speed_mae_mps": ratio("interval_speed_error", "interval_valid"),
        "sampled_control_speed_mae_mps": ratio("control_speed_error", "control_speed_valid"),
        "sampled_speed_mae_mps_by_interval": [float(error / count) if int(count) else None
                             for error, count in interval_totals],
        "behavior_confusion_by_horizon": confusion.cpu().tolist(),
        "behavior_accuracy_by_horizon": [float(matrix.diag().sum() / matrix.sum()) if int(matrix.sum()) else None
                                        for matrix in confusion],
    })
    head.train()
    return metrics


def save_sources(output, config_path, resuming):
    paths = list(Path(__file__).parent.rglob("*.py")) + [config_path]
    paths += [ROOT / "method" / "full_diffusion" / name for name in ("data.py", "diffusion.py", "garage.py", "model.py")]
    paths += [ROOT / "method" / "full_diffusion2" / name for name in ("data.py", "model.py", "perception.py", "launch.py", "snapshot.py", "inference/policy.py")]
    paths += [ROOT / "train" / "train_full_diffusion.py", ROOT / "tools" / "launch_full_diffusion.py"]
    tag = f"resume_{time.time_ns()}" if resuming else f"initial_{time.time_ns()}"
    with zipfile.ZipFile(output / f"source_snapshot_{tag}.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
              hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write_status(output / f"code_sha256_{tag}.json", hashes)


def record_validation(output, progress):
    result = progress.get("last_validation")
    if result is None:
        return
    destination = output / f"epoch_{result['epoch']:02}.pt"
    if not destination.exists():
        complete = select_complete(output, full_hash=False)
        os.link(complete["absolute_path"], destination)
    path = output / "validation.jsonl"
    previous = [json.loads(line)["epoch"] for line in path.read_text().splitlines()] if path.exists() else []
    if result["epoch"] not in previous:
        with path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
    print(json.dumps({"epoch": result["epoch"], "validation": result}), flush=True)


def new_head(config, manifest, device, local_rank, trace):
    head = SpatialTemporalFullDiffusionHead(**config["model"],
                                            normalization=manifest["v3_normalization"]["normalization"]).to(device)
    policy = DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False)
    policy.register_comm_hook(trace, gradient_hook)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    diffusion = SpatialTemporalVPFullDiffusion(head.codec, **{
        key: value for key, value in config["diffusion"].items() if key != "inference_steps"
    })
    return head, policy, optimizer, scheduler, diffusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--phase", choices=("acceptance", "train"), default="train")
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    if not args.gpu_authorized:
        parser.error("GPU training is not authorized by default; obtain explicit approval first")
    config = json.loads(args.config.read_text())
    manifest = training_metadata(config)
    guard_context()
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world != config["world_size"] or world * config["batch_per_gpu"] != 64:
        raise RuntimeError("V3 expects eight designated ranks and global batch64")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(seconds=300))
    control = ControlGroup.create(timeout=config.get("runtime", {}).get("control_timeout_seconds", 240))
    trace = PhaseTrace(local_directory(args.output, args.attempt), rank)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset_args = {"backbone_directory": ROOT / config["backbone_directory"],
                    "data_root": manifest["data_root"], "label_config": config["labels"],
                    "history_frames": config["model"]["history_frames"], "history_stride": config["history_stride"]}
    if config.get("sensor_snapshot"):
        dataset_args["sensor_snapshot"] = ROOT / config["sensor_snapshot"]
    dataset = GarageSpatialTemporalDataset(ROOT / config["index_directory"], **dataset_args)
    validation = GarageSpatialTemporalDataset(ROOT / config["index_directory"], split="val", **dataset_args)
    dataset.source_identity = manifest["runtime_source_sha256"]
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device)
    head, policy, optimizer, scheduler, diffusion = new_head(config, manifest, device, local_rank, trace)
    torch.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    sampler = DistributedSampler(dataset, seed=config["seed"], drop_last=True)
    batches = BatchSampler(sampler, config["batch_per_gpu"], drop_last=True)
    remaining = RemainingBatches(batches)
    workers = config["workers_per_rank"]
    loader = make_training_loader(dataset, remaining, workers, config["seed"], rank,
                                  config.get("runtime", {}).get("loader_timeout_seconds", 180))
    progress = {"epoch": 0, "next_batch": 0, "updates": 0}
    if args.resume:
        complete = select_complete(args.output, config, manifest)
        if complete is None:
            raise RuntimeError("Resume requested without a complete checkpoint")
        checkpoint = args.resume_checkpoint or Path(complete["absolute_path"])
        if checkpoint.resolve() != Path(complete["absolute_path"]):
            raise RuntimeError("All nodes must use the coordinator's same last complete generation")
        progress = load_checkpoint(checkpoint, head, optimizer, scheduler, config, manifest, loader)
    elif (args.output / "latest.pt").exists():
        raise FileExistsError("Existing V3 checkpoints require explicit --resume")
    if rank == 0:
        write_status(args.output / "run_config.json", config)
        write_status(args.output / "dataset.json", manifest)
        save_sources(args.output, args.config, args.resume)
        record_validation(args.output, progress)
    control.barrier()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.phase == "acceptance":
        from method.full_diffusionv3.acceptance import run_acceptance
        try:
            run_acceptance(args, config, manifest, dataset, perception, head, policy, optimizer,
                           scheduler, diffusion, device, control, trace)
        except BaseException:
            trace.finish(failed=True)
            raise
        else:
            trace.finish()
            distributed.destroy_process_group()
        finally:
            trace.close()
        return
    updates_per_epoch = len(batches)
    started = time.monotonic()
    initial_updates = progress["updates"]
    running = torch.zeros(len(COMPONENTS), device=device)
    running_count = 0
    try:
        if progress.get("validation_pending", False):
            metrics = validate(validation, perception, head, diffusion, config, device, control, trace)
            progress["validation_pending"] = False
            progress["last_validation"] = {"epoch": progress["epoch"], **metrics}
            save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest, control, trace, loader)
            if rank == 0:
                record_validation(args.output, progress)
        starting_epoch = progress["epoch"]
        for epoch in range(starting_epoch, config["epochs"]):
            dataset.epoch = epoch
            sampler.set_epoch(epoch)
            remaining.offset = progress["next_batch"] if epoch == starting_epoch else 0
            iterator = iter(loader)
            for batch_id in range(remaining.offset, updates_per_epoch):
                trace.progress(epoch, progress["updates"])
                batch = coordinated_next(iterator, control, trace)
                with trace.stage("to_device", cuda=True):
                    batch = move_batch(batch, device)
                losses, norm = update(batch, perception, policy, optimizer, diffusion, config, control, trace)
                progress = {"epoch": epoch, "next_batch": batch_id + 1, "updates": progress["updates"] + 1}
                trace.progress(epoch, progress["updates"])
                scheduler.step(epoch + (batch_id + 1) / updates_per_epoch)
                running += torch.stack([losses[name] for name in COMPONENTS])
                running_count += 1
                if trace.abort_path.exists():
                    control.failed = True
                    raise RuntimeError("The global attempt was aborted; no fresh checkpoint is permitted")
                with trace.stage("stop_control"):
                    stop = control.any(STOP_REQUESTED)
                if progress["updates"] == initial_updates + 1 or progress["updates"] % config["log_every_updates"] == 0:
                    with trace.stage("metrics_control"):
                        averages = control.reduce(running) / (running_count * world)
                        cuda_times = trace.cuda_timings()
                    elapsed = time.monotonic() - started
                    record = {**progress, "epoch_display": epoch + 1, "updates_per_epoch": updates_per_epoch,
                              **{name: float(value) for name, value in zip(COMPONENTS, averages)},
                              "lr": optimizer.param_groups[0]["lr"], "gradient_norm_rank0": float(norm),
                              "workers_per_rank": workers, "elapsed_seconds": elapsed,
                              "samples_per_second": (progress["updates"] - initial_updates) * 64 / elapsed,
                              "cuda_seconds_rank0": cuda_times, "attempt": args.attempt,
                              "utc": datetime.now(timezone.utc).isoformat()}
                    if rank == 0:
                        with (args.output / "metrics.jsonl").open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                        write_status(args.output / "progress.json", record)
                        print(json.dumps(record), flush=True)
                    running.zero_()
                    running_count = 0
                if (progress["updates"] == initial_updates + 1
                        or progress["updates"] % config["checkpoint_every_updates"] == 0 or stop):
                    save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest, control, trace, loader)
                if stop:
                    return
            progress = {"epoch": epoch + 1, "next_batch": 0, "updates": progress["updates"], "validation_pending": True}
            save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest, control, trace, loader)
            metrics = validate(validation, perception, head, diffusion, config, device, control, trace)
            progress["validation_pending"] = False
            progress["last_validation"] = {"epoch": epoch + 1, **metrics}
            save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress, config, manifest, control, trace, loader)
            if rank == 0:
                record_validation(args.output, progress)
            control.barrier()
        if rank == 0:
            write_status(args.output / "completed.json", {**progress, "finished_at": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        control.failed = True
        trace.finish(failed=True)
        write_status(args.output / f"rank{rank}.failure.attempt{args.attempt}.json",
                     {"rank": rank, "progress": progress, "error": repr(error), "fresh_checkpoint_saved": False,
                      "utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        trace.finish(failed=control.failed)
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
        if not control.failed:
            distributed.destroy_process_group()
        trace.close()


if __name__ == "__main__":
    main()