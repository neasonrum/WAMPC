"""Record and verify the four user-supplied B200 allocations."""
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from method.full_diffusionv3_ego12.allocation import job_cgroup, job_memory, query_allocation


def inspect(job, config, require_idle=True):
    entry = config['jobs'][job]
    job_cgroup(job)
    if socket.gethostname().split('.')[0] != entry['host']:
        raise RuntimeError('The requested LSF job is attached to a different host')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 2 * (1 + entry['workers_per_rank']):
        raise RuntimeError('CPU affinity cannot accommodate two ranks and their loader workers')
    memory = job_memory(job)
    if memory['oom'] or memory['oom_kill'] or memory['noncache_bytes'] + 64 * 2**30 > .8 * memory['limit_bytes']:
        raise RuntimeError(f'The allocation lacks the expected memory headroom {memory}')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,uuid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [[value.strip() for value in row] for row in csv.reader(io.StringIO(raw))]
    if len(rows) != 2 or [row[0] for row in rows] != ['0', '1'] or any('B200' not in row[1] for row in rows):
        raise RuntimeError('Expected exactly two allocation-visible logical B200 GPUs')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True, timeout=15).strip()
    if require_idle and (apps or any(int(row[3]) >= 1024 for row in rows)):
        raise RuntimeError('The supplied allocation already has a GPU workload')
    proof_path = Path(config['output']) / 'allocation_plan.json'
    saved = json.loads(proof_path.read_text())['jobs'][job]
    try:
        scheduler = query_allocation(job, entry['host'])
    except subprocess.CalledProcessError:
        scheduler = saved
    if scheduler['deadline_unix'] < time.time() + 1800:
        raise RuntimeError('The allocation has less than thirty minutes left')
    return {**scheduler, 'gpu_uuids': [row[4] for row in rows], 'affinity': affinity,
            'memory': memory, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'checked_unix': time.time()}
