"""Collective checkpoint writes with fsync and a retained previous generation."""

import os
from pathlib import Path

import torch
import torch.distributed as distributed

from train_full_diffusion import rng_state, load_checkpoint as original_load_checkpoint
from method.full_diffusion2.snapshot import retry_read


def load_checkpoint(*args, **kwargs):
    return retry_read(lambda: original_load_checkpoint(*args, **kwargs), "load-checkpoint")


def save_checkpoint(path, head, optimizer, scheduler, progress, config, manifest):
    from datetime import datetime, timezone
    path = Path(path)
    states = [None] * distributed.get_world_size()
    distributed.all_gather_object(states, rng_state())
    outcome = [None]
    if distributed.get_rank() == 0:
        try:
            checkpoint = {"model": head.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "progress": progress,
                          "rng_by_rank": states, "config": config, "dataset": manifest,
                          "created_at": datetime.now(timezone.utc).isoformat()}
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                previous = path.with_name(path.stem + ".previous.pt")
                previous_temporary = path.with_name(path.stem + f".previous.{os.getpid()}.tmp")
                os.link(path, previous_temporary)
                previous_temporary.replace(previous)
            temporary.replace(path)
        except Exception as error:
            outcome[0] = repr(error)
    distributed.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(f"Collective checkpoint write failed; retained previous checkpoint: {outcome[0]}")


def coordinated_next(iterator, device, on_failure):
    """Make all ranks checkpoint at the same update if any loader cannot deliver data."""
    batch, error = None, None
    try:
        batch = next(iterator)
    except Exception as exception:
        error = repr(exception)
    failed = torch.tensor(int(error is not None), device=device)
    distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
    if int(failed):
        errors = [None] * distributed.get_world_size()
        distributed.all_gather_object(errors, error)
        on_failure(errors)
        raise RuntimeError(f"Data loader failed; all ranks saved a common checkpoint: {errors}")
    return batch
