"""Use the frozen route/frame protocol without WM targets or predecessor plans."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from method.full_diffusion2.data import GarageTemporalDataset

ROOT = Path(__file__).resolve().parents[2]


class PointDataset(Dataset):
    def __init__(self, config, split="train"):
        self.directory = Path(config["index_directory"])
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.routes = json.loads((self.directory / "routes.json").read_text())
        self.samples = np.load(self.directory / f"{split}.npy", allow_pickle=False)
        self.source = Path(self.manifest["source_index"])
        self.teacher_root = Path(self.manifest["teacher_cache"])
        self.loader = GarageTemporalDataset(ROOT / config["retained_index"], split="train",
            data_root=self.manifest["data_root"], backbone_directory=ROOT / config["backbone_directory"],
            training=False, sensor_snapshot=ROOT / config["sensor_snapshot"])

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=16)
    def metadata(self, route):
        with np.load(self.source / "metadata" / f"{route:05}.npz", allow_pickle=False) as handle:
            return {key: handle[key].copy() for key in handle.files}

    @lru_cache(maxsize=8)
    def feature(self, route, frame):
        path = self.teacher_root / self.routes[route] / f"{frame:04}.npy"
        bits = np.load(path, allow_pickle=False)
        if bits.dtype != np.uint16 or bits.shape != (64, 64, 64):
            raise ValueError(f"Invalid BF16 observed feature {path}")
        return torch.from_numpy(bits).view(torch.bfloat16)

    @staticmethod
    def pose(value):
        x, y, yaw = map(float, value)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]], dtype=np.float64)

    def __getitem__(self, index):
        route, slot, frame = map(int, self.samples[index])
        metadata = self.metadata(route)
        offsets = np.arange(2, 13, 2)
        physical = metadata["frames"][slot + np.r_[-2, -1, 0, offsets]]
        if not np.array_equal(physical, frame + np.r_[-2, -1, 0, offsets]):
            raise ValueError("Fixed protocol row does not have the required physical time window")
        current_pose = self.pose(metadata["poses"][slot])
        inverse = np.linalg.inv(current_pose)
        future_xy = metadata["poses"][slot + offsets, :2]
        positions = (np.c_[future_xy, np.ones(6)] @ inverse.T)[:, :2].astype(np.float32)
        delta = np.diff(np.concatenate((np.zeros((1, 2), dtype=np.float32), positions)), axis=0)
        speeds = metadata["speed"][slot + np.array([0, 4, 8, 12])]
        change = np.diff(speeds)
        behavior = np.where(speeds[1:] < .3, 0, np.where(change < -.5, 1, np.where(change > .5, 3, 2)))
        current = self.loader.load_frame(Path(self.manifest["data_root"]) / self.routes[route], frame, random.Random(901))
        result = {key: current[key] for key in ("rgb", "lidar", "speed", "command", "target_point", "path")}
        history_poses = metadata["poses"][slot - 2:slot]
        transforms = np.stack([inverse @ self.pose(value) for value in history_poses]).astype(np.float32)
        result.update(history_bev=torch.stack([self.feature(route, f) for f in (frame - 2, frame - 1)]),
                      history_transforms=torch.from_numpy(transforms), history_mask=torch.ones(2, dtype=torch.bool),
                      temporal_delta=torch.from_numpy(delta), future_positions=torch.from_numpy(positions),
                      spatial_mask=torch.ones(10, dtype=torch.bool), temporal_mask=torch.ones(6, dtype=torch.bool),
                      behavior_targets=torch.from_numpy(behavior.astype(np.int64)), behavior_mask=torch.ones(3, dtype=torch.bool),
                      route_id=route, sample_index=index)
        return result


class GlobalBatches:
    """Identical global shuffled epoch permutation across all eight ranks."""
    def __init__(self, size, batch_per_rank, rank, world, seed, epoch, offset=0):
        self.size, self.batch, self.rank, self.world = size, batch_per_rank, rank, world
        self.seed, self.epoch, self.offset = seed, epoch, offset

    def __len__(self):
        return self.size // (self.batch * self.world) - self.offset

    def __iter__(self):
        global_batch = self.batch * self.world
        count = self.size // global_batch
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        rows = order[:count * global_batch].view(count, self.world, self.batch)
        yield from rows[self.offset:, self.rank].tolist()


class ValidationBatches:
    """Keep the established eight validation shards, ordering, and batch geometry."""
    def __init__(self, dataset, rank, world, batch=8):
        grouped = defaultdict(list)
        for index, (route, _, frame) in enumerate(dataset.samples):
            grouped[int(route)].append((int(frame), index))
        self.routes = [sorted(grouped[key]) for key in sorted(grouped)[rank::world]]
        self.batch = batch

    def __iter__(self):
        active = [iter(route) for route in self.routes]
        while active:
            rows, keep = [], []
            for route in active[:self.batch]:
                item = next(route, None)
                if item is not None:
                    rows.append(item[1])
                    keep.append(route)
            active = keep + active[self.batch:]
            if rows:
                yield rows

