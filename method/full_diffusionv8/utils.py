"""Atomic run records and small distributed-training helpers."""
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .prepare import atomic_json, sha


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def seed(value):
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def rng_state(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(), 'generator': generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())
    generator.set_state(state['generator'].cpu())


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
