"""Retained 4Hz anchors, frozen teacher targets, and exact predecessor support."""
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from method.full_diffusion2.data import GarageTemporalDataset
from .legacy import module

motion_targets = module('data').motion_targets
ROOT = Path(__file__).resolve().parents[2]


class JointDataset(Dataset):
    def __init__(self, output, config, split='train'):
        self.directory = Path(output) / 'index'
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.routes = json.loads((self.directory / 'routes.json').read_text())
        self.samples = np.load(self.directory / f'{split}.npy', allow_pickle=False)
        self.world_masks = np.load(self.directory / f'{split}_world_mask.npy', allow_pickle=False)
        self.source = Path(self.manifest['source_index'])
        self.teacher_root = Path(self.manifest['teacher_cache'])
        self.loader = GarageTemporalDataset(ROOT / config['retained_index'], split='train',
                                            data_root=self.manifest['data_root'],
                                            backbone_directory=ROOT / config['backbone_directory'],
                                            training=False, sensor_snapshot=ROOT / config['sensor_snapshot'])
        self.needs_predecessor = np.ones(len(self.samples), dtype=bool)
        if len(self.samples) > 1:
            same_route = np.diff(self.samples[:, 0]) == 0
            adjacent = np.diff(self.samples[:, 2]) == 1
            self.needs_predecessor[1:] = ~(same_route & adjacent)

    def __len__(self):
        return len(self.samples)

    def chains(self):
        return self.samples[:, [0, 2]]

    @lru_cache(maxsize=40)
    def metadata(self, route):
        with np.load(self.source / 'metadata' / f'{route:05}.npz', allow_pickle=False) as handle:
            return {key: handle[key].copy() for key in handle.files}

    @lru_cache(maxsize=32)
    def feature(self, route, frame):
        path = self.teacher_root / self.routes[route] / f'{frame:04}.npy'
        bits = np.load(path, allow_pickle=False)
        if bits.dtype != np.uint16 or bits.shape != (64, 64, 64):
            raise ValueError(f'Invalid frozen teacher feature {path}')
        return torch.from_numpy(bits).view(torch.bfloat16)

    def observation(self, route, slot, frame):
        metadata = self.metadata(route)
        if int(metadata['frames'][slot]) != frame:
            raise ValueError('The route slot does not match its physical frame')
        raw = torch.stack([self.feature(route, f) for f in (frame - 2, frame - 1, frame)])
        current = self.loader.load_frame(Path(self.manifest['data_root']) / self.routes[route], frame, random.Random(901))
        poses = metadata['poses']
        selected = poses[slot - 2:slot + 1]
        current_pose = self.pose(poses[slot])
        transforms = np.stack([np.linalg.solve(current_pose, self.pose(pose)) for pose in selected])
        result = {key: current[key] for key in ('rgb', 'lidar', 'speed', 'command', 'target_point', 'path')}
        result.update(raw_bev=raw, current_pose=torch.from_numpy(current_pose),
                      history_transforms=torch.from_numpy(transforms.astype(np.float32)),
                      history_mask=torch.ones(3, dtype=torch.bool), timestamp=frame * .25)
        return result

    @staticmethod
    def pose(pose):
        x, y, yaw = map(float, pose)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]], dtype=np.float64)

    def __getitem__(self, index):
        route, slot, frame = map(int, self.samples[index])
        result = self.observation(route, slot, frame)
        metadata = self.metadata(route)
        delta, positions, relative, target_slots = motion_targets(metadata['poses'], slot)
        speeds = metadata['speed'][slot + np.array([0, 4, 8, 12])]
        change = np.diff(speeds)
        behavior = np.where(speeds[1:] < .3, 0, np.where(change < -.5, 1, np.where(change > .5, 3, 2)))
        teacher = torch.zeros(6, 64, 64, 64, dtype=torch.bfloat16)
        for horizon in np.flatnonzero(self.world_masks[index]):
            teacher[horizon] = self.feature(route, int(metadata['frames'][target_slots[3 + horizon]]))
        predecessor = None
        if self.needs_predecessor[index] and slot >= 3:
            try:
                predecessor = self.observation(route, slot - 1, frame - 1)
            except FileNotFoundError:
                pass
        result.update(temporal_delta=torch.from_numpy(delta), future_positions=torch.from_numpy(positions),
                      spatial_mask=torch.ones(10, dtype=torch.bool), temporal_mask=torch.ones(6, dtype=torch.bool),
                      behavior_targets=torch.from_numpy(behavior.astype(np.int64)),
                      behavior_mask=torch.ones(3, dtype=torch.bool),
                      world_target_bev=teacher, world_target_poses=torch.from_numpy(relative[3:]),
                      world_target_valid=torch.from_numpy(self.world_masks[index].copy()),
                      chain_id=self.routes[route], route_id=route, sample_index=index,
                      prior_observation=predecessor)
        return result


def collate_with_predecessors(samples):
    rows, observations, anchors = [], [], []
    for row, sample in enumerate(samples):
        sample = dict(sample)
        predecessor = sample.pop('prior_observation')
        if predecessor is not None:
            rows.append(row)
            observations.append(predecessor)
        anchors.append(sample)
    support = (torch.tensor(rows, dtype=torch.long), default_collate(observations)) if rows else None
    return default_collate(anchors), support
