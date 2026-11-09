"""Compact PDM-Lite frame index and upstream-compatible sensor preprocessing."""

import gzip
import hashlib
import json
import random
import re
from pathlib import Path

import cv2
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

from .garage import BACKBONE, load_config, make_preprocessor


DATA_ROOT = Path("/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/carla_garage/data")


def read_measurement(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def geometric_path(processor, measurement, translation=0.0, rotation=0.0):
    route = np.asarray(measurement["route"], dtype=np.float64)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] != 2:
        raise ValueError("An expert geometric route requires at least two 2D points")
    route = processor.augment_route(route, y_augmentation=translation, yaw_augmentation=rotation)
    path = processor.smooth_path(route)
    return np.asarray(path[:processor.config.predict_checkpoint_len], dtype=np.float32)


def expert_speed(measurement):
    return np.float32(0.0 if measurement["brake"] else measurement["target_speed"])


def route_is_validation(route, seed=901, fraction=0.05):
    digest = hashlib.sha256(f"{seed}:{route}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def index_worker_init(backbone_directory):
    global INDEX_PROCESSOR
    cv2.setNumThreads(0)
    INDEX_PROCESSOR = make_preprocessor(load_config(backbone_directory), training=False)


def index_route(task):
    route_path, seed, validation_fraction = task
    processor = INDEX_PROCESSOR
    validation = route_is_validation(f"{route_path.parent.name}/{route_path.name}", seed, validation_fraction)
    rng = random.Random(f"{seed}:{route_path.parent.name}/{route_path.name}")
    frames = []
    total = np.zeros((processor.config.predict_checkpoint_len + 1, 2), dtype=np.float64)
    squared = np.zeros_like(total)
    previous_speed = None
    previous_angles = None
    candidate_count = 0
    for measurement_path in sorted((route_path / "measurements").glob("*.json.gz")):
        frame_id = int(measurement_path.name.split(".")[0])
        if frame_id < processor.config.skip_first:
            continue
        measurement = read_measurement(measurement_path)
        path = geometric_path(processor, measurement)
        speed = float(expert_speed(measurement))
        angles = np.arctan2(path[:, 1], path[:, 0])
        significant = previous_speed is None or abs(speed - previous_speed) > 0.1
        if previous_angles is not None:
            difference = angles - previous_angles
            wrapped = np.arctan2(np.sin(difference), np.cos(difference))
            significant = significant or bool(np.any(np.abs(wrapped) > np.deg2rad(0.5)))
        previous_speed, previous_angles = speed, angles
        candidate_count += 1
        if not significant and rng.random() >= 0.14:
            continue
        frames.append(frame_id)
        if not validation:
            target = np.concatenate((path, [[speed, 0.0]]), axis=0).astype(np.float64)
            total += target
            squared += target * target
    return frames, validation, total, squared, candidate_count


def build_index(output, data_root=DATA_ROOT, backbone_directory=BACKBONE, workers=8,
                seed=901, validation_fraction=0.05, max_routes=None):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from tools.audit_full_diffusion import quality_passes
    config = load_config(backbone_directory)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing index: {output}")
    routes = []
    for scenario in sorted(Path(data_root).iterdir()):
        if not scenario.is_dir() or scenario.name.startswith("."):
            continue
        for route in sorted(scenario.iterdir()):
            if not route.is_dir() or route.name.startswith("FAILED_"):
                continue
            repetition = re.search(r"_Rep(\d+)", route.name)
            if repetition is None or int(repetition.group(1)) >= config.num_repetitions:
                continue
            results_path = route / "results.json.gz"
            if not results_path.exists() or not quality_passes(read_measurement(results_path)):
                continue
            routes.append(route)
    if max_routes is not None:
        routes = routes[:max_routes]
    tasks = [(route, seed, validation_fraction) for route in routes]
    training = []
    validation = []
    total = np.zeros((config.predict_checkpoint_len + 1, 2), dtype=np.float64)
    squared = np.zeros_like(total)
    candidates = 0
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=index_worker_init,
                             initargs=(backbone_directory,)) as pool:
        for route_id, result in enumerate(pool.map(index_route, tasks, chunksize=1)):
            frames, held_out, route_sum, route_squared, route_candidates = result
            destination = validation if held_out else training
            destination.extend((route_id, frame_id) for frame_id in frames)
            total += route_sum
            squared += route_squared
            candidates += route_candidates
            if (route_id + 1) % 100 == 0:
                print(json.dumps({"indexed_routes": route_id + 1, "total_routes": len(routes),
                                  "train_frames": len(training), "val_frames": len(validation)}), flush=True)
    if not training:
        raise ValueError("No training frames survived dataset indexing")
    mean = total / len(training)
    std = np.sqrt(np.maximum(squared / len(training) - mean * mean, 0.01))
    mean[-1, 1], std[-1, 1] = 0.0, 1.0
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "train.npy", np.asarray(training, dtype=np.int32).reshape(-1, 2))
    np.save(output / "val.npy", np.asarray(validation, dtype=np.int32).reshape(-1, 2))
    (output / "routes.json").write_text(json.dumps([str(route.relative_to(data_root)) for route in routes], indent=2))
    manifest = {
        "data_root": str(data_root), "seed": seed, "validation_fraction": validation_fraction,
        "split": "sha256 of whole route; no frame-level split", "routes": len(routes),
        "candidate_frames": candidates, "train_frames": len(training), "val_frames": len(validation),
        "num_repetitions": config.num_repetitions, "skip_first": config.skip_first,
        "filter": {"speed_delta_mps": 0.1, "path_angle_delta_deg": 0.5,
                   "unchanged_frame_keep_probability": 0.14, "reference": "previous raw frame"},
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "normalization_source": "unaugmented retained training frames only; std floor 0.1",
        "smoke_only": max_routes is not None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "normalization"}), flush=True)
    return manifest


class GarageGeoDataset(Dataset):
    def __init__(self, index_directory, split="train", data_root=DATA_ROOT,
                 backbone_directory=BACKBONE, training=None):
        self.data_root = Path(data_root)
        index_directory = Path(index_directory)
        self.routes = json.loads((index_directory / "routes.json").read_text())
        self.samples = np.load(index_directory / f"{split}.npy", mmap_mode="r")
        self.epoch = 0
        self.processor = make_preprocessor(load_config(backbone_directory),
                                           split == "train" if training is None else training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        route_id, frame_id = self.samples[index]
        digest = hashlib.sha256(f"901:{self.epoch}:{int(route_id)}:{int(frame_id)}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")
        self.processor.image_augmenter_func.seed_(seed)
        self.processor.lidar_augmenter_func.seed_((seed + 1) % 2**32)
        return self.load_frame(self.data_root / self.routes[int(route_id)], int(frame_id), random.Random(seed))

    def read_measurement_file(self, path):
        return read_measurement(path)

    def read_image_file(self, path):
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def read_lidar_file(self, path):
        return laspy.read(path).xyz

    def frame_file_exists(self, path):
        return path.is_file()

    def load_frame(self, route, frame_id, rng=random):
        import transfuser_utils
        cv2.setNumThreads(0)
        processor = self.processor
        config = processor.config
        frame = f"{frame_id:04}"
        measurement = self.read_measurement_file(route / "measurements" / f"{frame}.json.gz")
        augmented = bool(config.augment and rng.random() < config.augment_percentage)
        rotation = float(measurement["augmentation_rotation"]) if augmented else 0.0
        translation = float(measurement["augmentation_translation"]) if augmented else 0.0
        image_path = route / ("rgb_augmented" if augmented else "rgb") / f"{frame}.jpg"
        image = self.read_image_file(image_path)
        if image is None:
            raise OSError(f"Cannot decode {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transfuser_utils.crop_array(config, image)
        if config.use_color_aug:
            image = processor.image_augmenter_func(image=image)
        lidar = self.read_lidar_file(route / "lidar" / f"{frame}.laz")
        lidar = processor.align(lidar, measurement, measurement, translation, rotation)
        lidar = processor.lidar_to_histogram_features(lidar, use_ground_plane=config.use_ground_plane)
        lidar = processor.lidar_augmenter_func(image=lidar.transpose(1, 2, 0)).transpose(2, 0, 1)
        target_point = processor.augment_target_point(
            np.asarray(measurement["target_point"]), y_augmentation=translation, yaw_augmentation=rotation
        )
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "lidar": torch.from_numpy(np.ascontiguousarray(lidar)),
            "path": torch.from_numpy(geometric_path(processor, measurement, translation, rotation)),
            "target_speed": torch.tensor(expert_speed(measurement)),
            "speed": torch.tensor(float(measurement["speed"]), dtype=torch.float32),
            "command": torch.tensor(transfuser_utils.command_to_one_hot(measurement["command"]), dtype=torch.float32),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "sample_id": f"{route.parent.name}/{route.name}/{frame}",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--backbone-directory", type=Path, default=BACKBONE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--max-routes", type=int)
    args = parser.parse_args()
    build_index(args.output, args.data_root, args.backbone_directory, args.workers, args.seed,
                max_routes=args.max_routes)