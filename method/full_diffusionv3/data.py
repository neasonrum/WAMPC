"""Causal observations, masked future targets, and training-only normalization."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from method.full_diffusion.data import geometric_path, read_measurement
from method.full_diffusion.garage import BACKBONE, load_config, make_preprocessor
from method.full_diffusion2.data import GarageTemporalDataset, frame_pose


BEHAVIOR_CLASSES = ("STOP", "DECELERATE", "KEEP", "ACCELERATE")
NORMALIZATION_PROCESSOR = None
NORMALIZATION_LABELS = None


def raw_offset(seconds, settings):
    interval = settings["raw_measurement_interval_seconds"]
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("The raw measurement interval must be finite and positive")
    offset = round(seconds / interval)
    if offset < 0 or not math.isclose(offset * interval, seconds, abs_tol=1e-8):
        raise ValueError("Future horizons must fall on raw measurement frames")
    return offset


def check_label_settings(settings, garage_config=None):
    if (settings["temporal_points"] != 6
            or settings["behavior_horizons_seconds"] != [1.0, 2.0, 3.0]
            or tuple(settings["behavior_classes"]) != BEHAVIOR_CLASSES
            or not math.isclose(settings["temporal_interval_seconds"], 0.5)):
        raise ValueError("V3 requires six half-second intervals and three fixed behavior horizons")
    for key in ("stop_speed_mps", "speed_change_mps"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Behavior thresholds must be finite and positive")
    for index in range(1, 7):
        raw_offset(index * settings["temporal_interval_seconds"], settings)
    if garage_config is not None:
        cadence = garage_config.data_save_freq / garage_config.carla_fps
        if not math.isclose(cadence, settings["raw_measurement_interval_seconds"]):
            raise ValueError("Future label cadence differs from the sensor checkpoint configuration")


def future_motion_targets(current, future_measurements, settings, augmented=False):
    check_label_settings(settings)
    inverse_pose = np.linalg.inv(frame_pose(current, augmented))
    positions = np.zeros((6, 2), dtype=np.float32)
    position_mask = np.zeros(6, dtype=bool)
    speeds = {0: float(current["speed"])}
    if not math.isfinite(speeds[0]):
        raise ValueError("Current expert speed is nonfinite")
    for index in range(6):
        offset = raw_offset((index + 1) * settings["temporal_interval_seconds"], settings)
        measurement = future_measurements.get(offset)
        if measurement is None:
            continue
        world = np.asarray(measurement["pos_global"], dtype=np.float64)
        speed = float(measurement["speed"])
        if world.shape != (2,) or not np.isfinite(world).all() or not math.isfinite(speed):
            raise ValueError("Future expert pose or speed is invalid")
        positions[index] = (inverse_pose @ np.append(world, 1.0))[:2]
        position_mask[index] = True
        speeds[offset] = speed
    previous_positions = np.concatenate((np.zeros((1, 2), dtype=np.float32), positions[:-1]))
    previous_mask = np.concatenate(([True], position_mask[:-1]))
    temporal_mask = position_mask & previous_mask
    temporal_delta = np.where(temporal_mask[:, None], positions - previous_positions, 0.0).astype(np.float32)
    behavior = np.full(3, -100, dtype=np.int64)
    behavior_mask = np.zeros(3, dtype=bool)
    previous_offset = 0
    for index, horizon in enumerate(settings["behavior_horizons_seconds"]):
        offset = raw_offset(horizon, settings)
        if previous_offset in speeds and offset in speeds:
            speed = speeds[offset]
            change = speed - speeds[previous_offset]
            if speed < settings["stop_speed_mps"]:
                category = 0
            elif change < -settings["speed_change_mps"]:
                category = 1
            elif change > settings["speed_change_mps"]:
                category = 3
            else:
                category = 2
            behavior[index] = category
            behavior_mask[index] = True
        previous_offset = offset
    return {"future_positions": positions, "future_position_mask": position_mask,
            "temporal_delta": temporal_delta, "temporal_mask": temporal_mask,
            "behavior_targets": behavior, "behavior_mask": behavior_mask}


class GarageSpatialTemporalDataset(GarageTemporalDataset):
    def __init__(self, *args, label_config, **kwargs):
        self._shared_epoch = multiprocessing.get_context("spawn").Value("q", 0)
        super().__init__(*args, **kwargs)
        self.label_config = dict(label_config)
        check_label_settings(self.label_config, self.processor.config)

    @property
    def epoch(self):
        return self._shared_epoch.value

    @epoch.setter
    def epoch(self, value):
        self._shared_epoch.value = int(value)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        route_id, frame_id = map(int, self.samples[index])
        route = self.data_root / self.routes[route_id]
        current = self.read_measurement_file(route / "measurements" / f"{frame_id:04}.json.gz")
        digest = hashlib.sha256(f"901:{self.epoch}:{route_id}:{frame_id}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        config = self.processor.config
        augmented = bool(config.augment and random.Random(seed).random() < config.augment_percentage)
        future = {}
        for interval in range(1, 7):
            offset = raw_offset(interval * self.label_config["temporal_interval_seconds"], self.label_config)
            path = route / "measurements" / f"{frame_id + offset:04}.json.gz"
            future[offset] = read_measurement(path) if path.is_file() else None
        targets = future_motion_targets(current, future, self.label_config, augmented)
        result.update({key: torch.from_numpy(value) for key, value in targets.items()})
        result["spatial_mask"] = torch.ones(10, dtype=torch.bool)
        return result


def normalization_worker_init(backbone_directory, label_config):
    global NORMALIZATION_PROCESSOR, NORMALIZATION_LABELS
    torch.set_num_threads(1)
    NORMALIZATION_PROCESSOR = make_preprocessor(load_config(backbone_directory), training=False)
    NORMALIZATION_LABELS = label_config
    check_label_settings(label_config, NORMALIZATION_PROCESSOR.config)


def collect_route_statistics(task):
    route, frame_ids = task
    route = Path(route)
    total = np.zeros((16, 2), dtype=np.float64)
    squared = np.zeros_like(total)
    counts = np.zeros(16, dtype=np.int64)
    class_counts = np.zeros((3, 4), dtype=np.int64)

    @lru_cache(maxsize=128)
    def measurement(frame_id):
        path = route / "measurements" / f"{frame_id:04}.json.gz"
        return read_measurement(path) if path.is_file() else None

    for frame_id in sorted(frame_ids):
        current = measurement(frame_id)
        if current is None:
            raise FileNotFoundError(route / "measurements" / f"{frame_id:04}.json.gz")
        future = {}
        for interval in range(1, 7):
            offset = raw_offset(interval * NORMALIZATION_LABELS["temporal_interval_seconds"], NORMALIZATION_LABELS)
            future[offset] = measurement(frame_id + offset)
        targets = future_motion_targets(current, future, NORMALIZATION_LABELS)
        spatial = geometric_path(NORMALIZATION_PROCESSOR, current)
        metric = np.concatenate((spatial, targets["temporal_delta"])).astype(np.float64)
        valid = np.concatenate((np.ones(10, dtype=bool), targets["temporal_mask"]))
        total += np.where(valid[:, None], metric, 0.0)
        squared += np.where(valid[:, None], metric * metric, 0.0)
        counts += valid
        for horizon in range(3):
            if targets["behavior_mask"][horizon]:
                class_counts[horizon, targets["behavior_targets"][horizon]] += 1
    return total, squared, counts, class_counts


def prepare_normalization(index_directory, output, label_config, backbone_directory=BACKBONE,
                          workers=4, std_floor=0.1):
    if workers < 1 or not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("CPU workers and normalization scale floor must be positive")
    check_label_settings(label_config, load_config(backbone_directory))
    index_directory, output = Path(index_directory), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalization: {output}")
    manifest = json.loads((index_directory / "manifest.json").read_text())
    if manifest["smoke_only"]:
        raise ValueError("V3 normalization requires the full training index")
    routes = json.loads((index_directory / "routes.json").read_text())
    samples = np.load(index_directory / "train.npy", mmap_mode="r")
    grouped = defaultdict(list)
    for route_id, frame_id in samples:
        grouped[int(route_id)].append(int(frame_id))
    tasks = [(str(Path(manifest["data_root"]) / routes[route_id]), frame_ids)
             for route_id, frame_ids in grouped.items()]
    total = np.zeros((16, 2), dtype=np.float64)
    squared = np.zeros_like(total)
    counts = np.zeros(16, dtype=np.int64)
    class_counts = np.zeros((3, 4), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=normalization_worker_init,
                             initargs=(str(backbone_directory), label_config)) as pool:
        for completed, (route_total, route_squared, route_counts, route_classes) in enumerate(pool.map(collect_route_statistics, tasks), 1):
            total += route_total
            squared += route_squared
            counts += route_counts
            class_counts += route_classes
            if completed % 100 == 0:
                print(json.dumps({"completed_routes": completed, "total_routes": len(tasks), "gpu_requested": False}), flush=True)
    if np.any(counts == 0):
        raise ValueError("At least one V3 slot has no valid training targets")
    mean = total / counts[:, None]
    variance = np.maximum(squared / counts[:, None] - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), std_floor)
    result = {
        "method": "full_diffusionv3", "label_config": label_config,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "valid_slot_counts": counts.tolist(), "behavior_class_counts": class_counts.tolist(),
        "train_frames": len(samples), "val_frames": manifest["val_frames"],
        "std_floor": std_floor, "source": "unaugmented valid retained training targets only",
        "source_sha256": {name: hashlib.sha256((index_directory / name).read_bytes()).hexdigest()
                          for name in ("train.npy", "val.npy", "routes.json", "manifest.json")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["method"] != "full_diffusionv3":
        raise ValueError("Normalization preparation requires a V3 configuration")
    output = args.output if args.output is not None else ROOT / config["normalization_file"]
    result = prepare_normalization(ROOT / config["index_directory"], output, config["labels"],
                                   ROOT / config["backbone_directory"], args.workers,
                                   config["normalization_std_floor"])
    print(json.dumps({"normalization_file": str(output), "train_frames": result["train_frames"],
                      "valid_slot_counts": result["valid_slot_counts"], "gpu_requested": False}))


if __name__ == "__main__":
    main()