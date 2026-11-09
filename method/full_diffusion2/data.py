"""Causal three-observation windows on the unchanged Full Diffusion route split."""

import gzip
import hashlib
import io
import json
import multiprocessing
import random

import cv2
import laspy
import numpy as np
import torch

from method.full_diffusion.data import GarageGeoDataset, read_measurement
from method.full_diffusion2.snapshot import SensorSnapshot


def frame_pose(measurement, augmented=False):
    yaw = float(measurement["theta"])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    position = measurement["pos_global"]
    pose = np.array([[cosine, -sine, position[0]],
                     [sine, cosine, position[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    if augmented:
        angle = np.deg2rad(float(measurement["augmentation_rotation"]))
        cosine, sine = np.cos(angle), np.sin(angle)
        augmentation = np.array([[cosine, -sine, 0.0],
                                 [sine, cosine, float(measurement["augmentation_translation"])],
                                 [0.0, 0.0, 1.0]], dtype=np.float64)
        pose = pose @ augmentation
    return pose


class GarageTemporalDataset(GarageGeoDataset):
    def __init__(self, *args, history_frames=2, history_stride=1, sensor_snapshot=None, **kwargs):
        self._shared_epoch = multiprocessing.get_context("spawn").Value("q", 0)
        super().__init__(*args, **kwargs)
        self.history_frames = history_frames
        self.history_stride = history_stride
        index = args[0] if args else kwargs["index_directory"]
        self.sensor_snapshot = SensorSnapshot(sensor_snapshot, index) if sensor_snapshot is not None else None

    @property
    def epoch(self):
        return self._shared_epoch.value

    @epoch.setter
    def epoch(self, value):
        self._shared_epoch.value = int(value)

    def snapshot_name(self, path):
        relative = path.relative_to(self.data_root)
        return "/".join(relative.parts[:2]), "/".join(relative.parts[2:])

    def read_measurement_file(self, path):
        if self.sensor_snapshot is None:
            return super().read_measurement_file(path)
        return json.loads(gzip.decompress(self.sensor_snapshot.read(*self.snapshot_name(path))))

    def read_image_file(self, path):
        if self.sensor_snapshot is None:
            return super().read_image_file(path)
        payload = self.sensor_snapshot.read(*self.snapshot_name(path))
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    def read_lidar_file(self, path):
        if self.sensor_snapshot is None:
            return super().read_lidar_file(path)
        return laspy.read(io.BytesIO(self.sensor_snapshot.read(*self.snapshot_name(path)))).xyz

    def frame_file_exists(self, path):
        if self.sensor_snapshot is None:
            return super().frame_file_exists(path)
        return self.sensor_snapshot.exists(*self.snapshot_name(path))

    def __getitem__(self, index):
        route_id, frame_id = map(int, self.samples[index])
        digest = hashlib.sha256(f"901:{self.epoch}:{route_id}:{frame_id}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        route = self.data_root / self.routes[route_id]
        config = self.processor.config
        augmented = bool(config.augment and random.Random(seed).random() < config.augment_percentage)
        self.seed_augmenters(seed)
        current = self.load_frame(route, frame_id, random.Random(seed))
        measurement = self.read_measurement_file(route / "measurements" / f"{frame_id:04}.json.gz")
        current_pose = frame_pose(measurement, augmented)
        images, lidars, transforms, valid, frame_ids = [], [], [], [], []
        for offset in range(self.history_frames, 0, -1):
            previous = frame_id - offset * self.history_stride
            files = [route / "measurements" / f"{previous:04}.json.gz",
                     route / ("rgb_augmented" if augmented else "rgb") / f"{previous:04}.jpg",
                     route / "lidar" / f"{previous:04}.laz"]
            exists = previous >= 0 and all(self.frame_file_exists(path) for path in files)
            if exists:
                self.seed_augmenters((seed + offset * 104729) % 2**32)
                historical = self.load_frame(route, previous, random.Random(seed))
                source_measurement = self.read_measurement_file(files[0])
                transform = np.linalg.solve(current_pose, frame_pose(source_measurement, augmented))
            else:
                historical = current
                transform = np.eye(3)
            images.append(historical["rgb"])
            lidars.append(historical["lidar"])
            transforms.append(torch.tensor(transform, dtype=torch.float32))
            valid.append(exists)
            frame_ids.append(previous)
        return {**current, "history_rgb": torch.stack(images), "history_lidar": torch.stack(lidars),
                "history_transforms": torch.stack(transforms), "history_mask": torch.tensor(valid),
                "history_frame_ids": torch.tensor(frame_ids), "frame_id": frame_id}

    def seed_augmenters(self, seed):
        self.processor.image_augmenter_func.seed_(seed)
        self.processor.lidar_augmenter_func.seed_((seed + 1) % 2**32)