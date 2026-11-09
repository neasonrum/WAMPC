"""Exact 0.5-second targets from immutable cached frozen TF++ observations."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from .interface import TIMES


def motion_targets(poses, anchor):
    future = anchor + np.arange(2, 13, 2)
    pose = poses[anchor]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    positions = ((poses[future, :2] - pose[:2]) @ rotation).astype(np.float32)
    delta = np.diff(np.concatenate((np.zeros((1, 2), np.float32), positions)), axis=0)
    indices = np.r_[np.arange(anchor - 2, anchor + 1), future]
    source = poses[indices]
    xy = (source[:, :2] - pose[:2]) @ rotation
    yaw = source[:, 2] - pose[2]
    relative = np.c_[xy, np.arctan2(np.sin(yaw), np.cos(yaw))].astype(np.float32)
    return delta, positions, relative, indices


class SequenceDataset(Dataset):
    def __init__(self, index, split, cache_root):
        self.index = Path(index)
        self.manifest = json.loads((self.index / 'manifest.json').read_text())
        self.routes = json.loads((self.index / 'routes.json').read_text())
        self.samples = np.load(self.index / f'{split}.npy', mmap_mode='r')
        if self.manifest['query_times_seconds'] != list(TIMES):
            raise ValueError('The dataset does not implement the fixed 3-second contract')
        identity = self.manifest['teacher_sha256'][:16] + '_' + self.manifest['teacher_config_sha256'][:12] + '_bf16_v1'
        self.cache_root = Path(cache_root) / identity
        complete = json.loads((self.cache_root / 'complete.json').read_text())
        if complete['missing_frames'] != 0 or complete['teacher_sha256'] != self.manifest['teacher_sha256']:
            raise ValueError('Teacher cache is incomplete or has a different teacher')

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=16)
    def metadata(self, rid):
        with np.load(self.index / 'metadata' / f'{rid:05}.npz') as f:
            return {k: f[k].copy() for k in f.files}

    def sensor(self, rid, frame):
        path = self.cache_root / self.routes[rid] / f'{frame:04}.npy'
        bits = np.load(path, allow_pickle=False)
        if bits.shape != (64, 64, 64) or bits.dtype != np.uint16:
            raise ValueError(f'Invalid frozen teacher feature: {path}')
        return {'key': str(path), 'bev': torch.from_numpy(bits).view(torch.bfloat16)}

    def __getitem__(self, index):
        rid, anchor = map(int, self.samples[index])
        m = self.metadata(rid)
        if not np.all(np.diff(m['frames'][anchor - 2:anchor + 13]) == 1):
            raise ValueError('Discontinuous input/target sequence')
        delta, positions, relative, indices = motion_targets(m['poses'], anchor)
        return {
            'sensors': [self.sensor(rid, int(m['frames'][i])) for i in indices],
            'source_in_anchor': torch.from_numpy(relative),
            'temporal_delta': torch.from_numpy(delta), 'path': torch.from_numpy(positions),
            'path_times': torch.tensor(TIMES), 'path_valid': torch.ones(6, dtype=torch.bool),
            'query_xy': torch.from_numpy(positions), 'query_times': torch.tensor(TIMES),
            'query_valid': torch.ones(6, dtype=torch.bool),
            'speed': torch.tensor(m['speed'][anchor], dtype=torch.float32),
            'route_id': rid, 'anchor_frame': int(m['frames'][anchor]),
        }


def collate(samples):
    return {**default_collate([{k: v for k, v in s.items() if k != 'sensors'} for s in samples]),
            'sensors': [s['sensors'] for s in samples]}


class TeacherCache:
    """Read only: no backbone GPU replica, no raw mount fallback, no in-RAM corpus."""
    def __init__(self, device):
        self.device, self.hits, self.misses = device, 0, 0

    @torch.no_grad()
    def __call__(self, sensors):
        flat = [f['bev'] for row in sensors for f in row]
        self.hits += len(flat)
        return torch.stack(flat).to(self.device, non_blocking=True).reshape(len(sensors), 9, 64, 64, 64)
