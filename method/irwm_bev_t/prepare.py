"""Reuse the exact original 44-route split; require all six real future frames."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .interface import CONTRACT, TIMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    allowed = Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC').resolve()
    if not out.is_relative_to(allowed) or out.exists():
        raise ValueError('Index output must be NEW and under the authorized cache directory')
    out.mkdir(parents=True)
    old = json.loads((source / 'manifest.json').read_text())
    routes = json.loads((source / 'routes.json').read_text())
    for name in ('routes.json', 'split.json'):
        shutil.copy2(source / name, out / name)
    shutil.copytree(source / 'metadata', out / 'metadata')
    assert sha(out / 'split.json') == old['split_sha256']
    masks = {}
    for rid in range(len(routes)):
        with np.load(source / 'metadata' / f'{rid:05}.npz') as f:
            frames = f['frames']
        good = np.zeros(len(frames), dtype=bool)
        if len(frames) >= 15:
            windows = np.lib.stride_tricks.sliding_window_view(frames, 15)
            good[2:len(frames) - 12] = (np.diff(windows, axis=1) == 1).all(1)
        masks[rid] = good
    counts = {}
    route_sets = {}
    for split in ('train', 'val'):
        samples = np.load(source / f'{split}.npy')
        keep = np.array([masks[int(r)][int(a)] for r, a in samples], dtype=bool)
        selected = samples[keep]
        np.save(out / f'{split}.npy', selected)
        route_sets[split] = set(map(int, selected[:, 0]))
        counts[split] = {'old_samples': len(samples), 'samples': len(selected),
                         'removed_incomplete_3s': int((~keep).sum()), 'active_routes': len(route_sets[split])}
    split = json.loads((source / 'split.json').read_text())
    if {routes[i] for i in route_sets['val']} != set(split['validation_routes']):
        raise AssertionError('All original 44 validation routes must remain represented')
    assert not route_sets['train'].intersection(route_sets['val'])
    manifest = {k: v for k, v in old.items() if k not in ('query_progress_m', 'path_progress_m', 'max_future_frames')}
    manifest.update(version=2, method='irwm_bev_t', query_times_seconds=list(TIMES),
                    future_frames=12, action_contract=CONTRACT, source_index=str(source),
                    source_manifest_sha256=sha(source / 'manifest.json'),
                    train_samples=counts['train']['samples'], validation_samples=counts['val']['samples'],
                    active_train_routes=counts['train']['active_routes'], filtering=counts,
                    temporal_contract='six exact future observations at 0.5 to 3.0 s; full horizon only; no feature interpolation',
                    train_index_sha256=sha(out / 'train.npy'), val_index_sha256=sha(out / 'val.npy'))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'index_ready', **counts, 'validation_routes': 44, 'split_sha256': old['split_sha256']}), flush=True)


if __name__ == '__main__':
    main()
