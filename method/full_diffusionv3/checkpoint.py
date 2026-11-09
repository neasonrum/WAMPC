"""Complete checkpoint generations committed through the independent CPU group."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as distributed

from method.full_diffusion2.snapshot import atomic_json, sha_file


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: cpu_copy(item) for name, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def rng_state(loader=None):
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate(),
            "loader": loader.generator.get_state() if loader is not None else None}


def restore_rng(state, loader=None):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if loader is not None and state["loader"] is not None:
        loader.generator.set_state(state["loader"])


def commit(path, checkpoint):
    path = Path(path)
    directory = path.parent / "checkpoints"
    directory.mkdir(exist_ok=True)
    generation = directory / f"step_{checkpoint['progress']['updates']:08}_{time.time_ns()}.pt"
    temporary = generation.with_suffix(".partial")
    torch.save(checkpoint, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    digest = sha_file(temporary)
    temporary.replace(generation)
    record = {"path": str(generation.relative_to(path.parent)), "bytes": generation.stat().st_size,
              "sha256": digest, "progress": checkpoint["progress"],
              "config_identity": identity(checkpoint["config"]),
              "dataset_identity": identity(checkpoint["dataset"]),
              "world_size": len(checkpoint["rng_by_rank"]),
              "created_at": checkpoint["created_at"], "complete": True}
    index_path = directory / "complete.json"
    previous = json.loads(index_path.read_text())["generations"] if index_path.exists() else []
    candidates = previous + [record]
    retained = [item for item in candidates if item in candidates[-2:] or
                (item["progress"]["epoch"] > 0 and item["progress"]["next_batch"] == 0
                 and not item["progress"].get("validation_pending", False))]
    atomic_json(index_path, {"format": 1, "generations": retained, "latest": record["path"]})
    link = path.with_name(path.name + f".{os.getpid()}.tmp")
    link.symlink_to(generation.relative_to(path.parent))
    link.replace(path)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    for item in previous:
        if item not in retained:
            try:
                (path.parent / item["path"]).unlink()
            except OSError:
                pass
    return record


def select_complete(output, config=None, dataset=None, full_hash=True):
    output = Path(output)
    path = output / "checkpoints/complete.json"
    if not path.exists():
        return None
    candidates = json.loads(path.read_text())["generations"]
    for record in reversed(candidates):
        target = output / record["path"]
        target.resolve().relative_to(output.resolve())
        if not record.get("complete") or not target.is_file() or target.stat().st_size != record["bytes"]:
            continue
        if config is not None and record["config_identity"] != identity(config):
            raise ValueError("Complete checkpoint has different training configuration")
        if dataset is not None and record["dataset_identity"] != identity(dataset):
            raise ValueError("Complete checkpoint has different dataset/source identity")
        if full_hash and sha_file(target) != record["sha256"]:
            continue
        return {**record, "absolute_path": str(target.resolve())}
    raise RuntimeError("No verifiable complete V3 checkpoint remains")


def save_checkpoint(path, head, optimizer, scheduler, progress, config, manifest, control, trace, loader=None):
    if control.failed or trace.abort_path.exists():
        raise RuntimeError("Refusing checkpoint save after an attempt/communication failure")
    with trace.stage("checkpoint_sync"):
        if next(head.parameters()).is_cuda:
            torch.cuda.synchronize()
        control.barrier()
    with trace.stage("checkpoint_rng"):
        states = control.gather(rng_state(loader))
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    outcome = None
    if rank == 0:
        try:
            with trace.stage("checkpoint_copy"):
                checkpoint = {"format": 1, "model": cpu_copy(head.state_dict()),
                              "optimizer": cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
                              "progress": dict(progress), "rng_by_rank": states, "config": config,
                              "dataset": manifest, "created_at": datetime.now(timezone.utc).isoformat()}
            with trace.stage("checkpoint_write"):
                outcome = {"ok": True, "record": commit(path, checkpoint)}
        except BaseException as error:
            outcome = {"ok": False, "error": repr(error)}
    with trace.stage("checkpoint_commit"):
        outcomes = control.gather(outcome)
        if not outcomes[0]["ok"]:
            raise RuntimeError(f"Checkpoint commit failed; preserve prior complete generation: {outcomes[0]}")
    return outcomes[0]["record"]


def load_checkpoint(path, head, optimizer, scheduler, config, manifest, loader=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != config or checkpoint["dataset"] != manifest:
        raise ValueError("Resume config or dataset/source identity differs")
    world = distributed.get_world_size() if distributed.is_initialized() else 1
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    if len(checkpoint["rng_by_rank"]) != world:
        raise ValueError("Resume must use the original world size")
    head.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng_by_rank"][rank], loader)
    return dict(checkpoint["progress"])