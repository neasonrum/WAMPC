"""Read-only CARLA/PyTorch preflight inside an explicitly selected LSF allocation."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cgroup = Path('/proc/self/cgroup').read_text()
    if not re.search(rf'(?<!\d){re.escape(args.job_id)}(?!\d)', cgroup):
        raise RuntimeError('Preflight must run inside the designated battach allocation')
    result = {'job_id': args.job_id, 'host': socket.gethostname(), 'cgroup': cgroup,
              'affinity': sorted(os.sched_getaffinity(0)), 'python': sys.executable,
              'prefix': sys.prefix, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'packages': {}, 'errors': {}}
    for name in ('torch', 'torchvision', 'carla', 'numpy', 'cv2', 'pygame', 'py_trees',
                 'timm', 'diffusers', 'imgaug', 'filterpy', 'shapely', 'transforms3d'):
        try:
            module = importlib.import_module(name)
            result['packages'][name] = {'version': getattr(module, '__version__', None),
                                        'file': getattr(module, '__file__', None)}
        except Exception as exc:
            result['errors'][name] = repr(exc)
    import torch
    result['torch_cuda'] = torch.version.cuda
    result['devices'] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        x = torch.tensor([1., 2.], device=f'cuda:{i}')
        result['devices'].append({'logical_index': i, 'name': p.name,
                                  'uuid': str(p.uuid), 'total_memory': p.total_memory,
                                  'tensor_check': float((x * x).sum())})
    import carla
    settings = carla.WorldSettings()
    result['carla_distribution'] = importlib.metadata.version('carla')
    result['carla_world_settings'] = {k: hasattr(settings, k) for k in
                                    ('synchronous_mode', 'fixed_delta_seconds',
                                     'deterministic_ragdolls', 'spectator_as_ego')}
    result['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    cgroup_relative = cgroup.strip().split(':')[-1].lstrip('/')
    for name in ('memory.max', 'memory.current', 'memory.peak', 'memory.events'):
        p = Path('/sys/fs/cgroup') / cgroup_relative / name
        if p.exists():
            result[name] = p.read_text().strip()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return int(bool(result['errors']) or not result['devices'])


if __name__ == '__main__':
    raise SystemExit(main())
