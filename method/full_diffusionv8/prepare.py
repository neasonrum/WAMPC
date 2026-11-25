"""Build the filtered V8 index with the existing 44-route development holdout."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(config, output):
    output = Path(output)
    index = output / 'index'
    if (index / 'manifest.json').is_file():
        return json.loads((index / 'manifest.json').read_text())
    index.mkdir(parents=True, exist_ok=True)
    source = Path(config['world_source_index'])
    source_manifest = json.loads((source / 'manifest.json').read_text())
    routes = json.loads((source / 'routes.json').read_text())
    original = ROOT / config['retained_index']
    original_routes = json.loads((original / 'routes.json').read_text())
    reverse = {name: number for number, name in enumerate(routes)}
    old_train = np.load(source / 'train.npy', allow_pickle=False)
    old_val = np.load(source / 'val.npy', allow_pickle=False)
    train_routes, val_routes = set(old_train[:, 0].tolist()), set(old_val[:, 0].tolist())
    if train_routes & val_routes or len(val_routes) != 44:
        raise ValueError('The world-model development split must have 44 disjoint routes')
    candidates = np.concatenate([np.load(original / f'{split}.npy', allow_pickle=False)
                                 for split in ('train', 'val')])
    grouped = defaultdict(set)
    counts = Counter()
    for rid, frame in candidates:
        name = original_routes[int(rid)]
        if name not in reverse:
            counts['outside_existing_world_route_selection'] += 1
            continue
        grouped[reverse[name]].add(int(frame))
    cache = ROOT / config['teacher_cache']
    identity = source_manifest['teacher_sha256'][:16] + '_' + source_manifest['teacher_config_sha256'][:12] + '_bf16_v1'
    cache = cache / identity
    train, val, target_masks = [], [], {'train': [], 'val': []}
    started = time.monotonic()
    for completed, (rid, retained) in enumerate(sorted(grouped.items()), 1):
        with np.load(source / 'metadata' / f'{rid:05}.npz', allow_pickle=False) as metadata:
            frames = metadata['frames'].astype(np.int64)
        directory = cache / routes[rid]
        present = {int(entry.name[:-4]) for entry in os.scandir(directory)
                   if entry.name.endswith('.npy') and entry.name[:-4].isdigit()} if directory.is_dir() else set()
        split = 'val' if rid in val_routes else 'train'
        if rid not in train_routes and rid not in val_routes:
            continue
        destination = val if split == 'val' else train
        for frame in sorted(retained):
            slot = int(np.searchsorted(frames, frame))
            if slot < 2 or not np.array_equal(frames[slot - 2:slot + 13], np.arange(frame - 2, frame + 13)):
                counts[f'{split}_incomplete_physical_window'] += 1
                continue
            if not {frame - 2, frame - 1, frame}.issubset(present):
                counts[f'{split}_missing_observed_teacher'] += 1
                continue
            destination.append((rid, slot, frame))
            target_masks[split].append([frame + step in present for step in range(2, 13, 2)])
        if completed % 400 == 0:
            print(json.dumps({'phase': 'index', 'routes': completed, 'train': len(train), 'val': len(val),
                              'seconds': time.monotonic() - started}), flush=True)
    for split, rows in [('train', train), ('val', val)]:
        np.save(index / f'{split}.npy', np.asarray(rows, dtype=np.int64).reshape(-1, 3))
        np.save(index / f'{split}_world_mask.npy', np.asarray(target_masks[split], dtype=bool).reshape(-1, 6))
    if len({row[0] for row in val}) != 44:
        raise ValueError('The prepared validation index does not cover all 44 development routes')
    atomic_json(index / 'routes.json', routes)
    manifest = {'format': 'full_diffusionv8_ego12_index_v1', 'teacher_cache': str(cache),
                'source_index': str(source), 'data_root': source_manifest['data_root'],
                'teacher_sha256': source_manifest['teacher_sha256'],
                'teacher_config_sha256': source_manifest['teacher_config_sha256'],
                'source_manifest_sha256': sha(source / 'manifest.json'),
                'retained_index_sha256': {name: sha(original / name) for name in ('manifest.json', 'train.npy', 'val.npy')},
                'train_samples': len(train), 'validation_samples': len(val),
                'train_routes': len({row[0] for row in train}), 'validation_routes': 44,
                'excluded_counts': dict(counts), 'prior_age_seconds': .25,
                'selection': 'original retained PDM frames intersect existing WM route selection and continuous observed 3s windows',
                'validation_selection': 'existing 44 WM development routes, whole routes excluded from training',
                'target_missing_policy': 'mask the missing physical horizon, retain its action in rollout',
                'index_sha256': {name: sha(index / name) for name in
                                 ('train.npy', 'val.npy', 'train_world_mask.npy', 'val_world_mask.npy', 'routes.json')}}
    atomic_json(index / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
