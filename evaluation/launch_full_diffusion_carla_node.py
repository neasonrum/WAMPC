"""Launch only the configured workers inside their existing LSF allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    check_job(args.job_id)
    run = args.run_root.resolve()
    run.relative_to(ROOT.resolve() / 'results')
    topology = json.loads((run / 'launch_topology.json').read_text())
    workers = [w for w in topology['workers'] if w['job_id'] == args.job_id]
    if len(workers) != 2 or {w['gpu'] for w in workers} != {0, 1}:
        raise RuntimeError('This node must have exactly two explicitly assigned workers')
    host = socket.gethostname().split('.')[0]
    if any(w['host'] != host for w in workers):
        raise RuntimeError('Wrong execution host')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(uuids) != 2 or any(uuids[w['gpu']] != w['gpu_uuid'] for w in workers):
        raise RuntimeError('Allocation-visible GPU UUID mapping changed')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError(f'GPUs are already occupied; inspect before launch: {apps}')
    checkpoint = Path(topology['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != topology['checkpoint_sha256']:
        raise RuntimeError('Checkpoint checksum differs from the shared topology')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError(f'Insufficient allocation-visible CPU affinity: {affinity}')
    group = Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    memory = {}
    for key in ('memory.max', 'memory.current', 'memory.events'):
        p = Path('/sys/fs/cgroup') / str(Path(group).parent) / key
        memory[key] = p.read_text().strip()
    metadata = {'job_id': args.job_id, 'host': host, 'unix': time.time(),
                'affinity': affinity, 'gpus': uuids, 'memory': memory,
                'expected_peak_host_memory_gib': 32, 'launches': []}
    limit = memory['memory.max']
    if limit != 'max' and int(limit) < 64 * 1024**3:
        raise RuntimeError('Unexpectedly low host memory limit; inspect before launch')
    sock = ROOT / f'.e10_{args.job_id}.sock'
    for worker in workers:
        index = worker['worker_index']
        command = [sys.executable, '-u', str(ROOT / 'evaluation/run_full_diffusion_carla.py'),
                   '--job-id', args.job_id, '--phase', 'full', '--gpu', str(worker['gpu']),
                   '--worker-index', str(index), '--num-workers', str(topology['num_workers']),
                   '--slot', str(worker['slot']), '--seed', str(topology['seed']),
                   '--checkpoint', str(checkpoint), '--output-root', str(run / f'full/worker{index}')]
        session = f'e10_worker_{index}'
        log = run / f'worker{index}.controller.log'
        shell_command = 'exec ' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1'
        subprocess.run(['tmux', '-S', str(sock), 'new-session', '-d', '-s', session, shell_command],
                       cwd=ROOT, check=True)
        metadata['launches'].append({'worker': index, 'session': session,
                                    'socket': str(sock), 'command': command, 'log': str(log)})
    (run / f'node_launch_{args.job_id}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
