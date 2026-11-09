"""Launch the two configured GPU groups inside an existing battach allocation."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--wall-budget', type=int, default=21600)
    a = p.parse_args()
    check_job(a.job_id)
    assert a.job_id in Path('/proc/self/cgroup').read_text()
    assert Path(sys.prefix).resolve() == (ROOT/'.venvs/full-diffusion-carla').resolve()
    run = a.run_root.resolve(); run.relative_to(ROOT.resolve()/'results')
    top = json.loads((run/'launch_topology.json').read_text())
    expected = json.loads((run/'source_sha256.json').read_text())
    changed = [n for n, h in expected.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != h]
    if changed:
        raise RuntimeError(f'Prepared sources differ: {changed}')
    workers = sorted([w for w in top['workers'] if w['job_id'] == a.job_id], key=lambda w: w['gpu'])
    assert len(workers) == 2 and [w['gpu'] for w in workers] == [0, 1]
    assert all(w['host'] == socket.gethostname().split('.')[0] for w in workers)
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    assert uuids == [w['gpu_uuid'] for w in workers]
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError(f'Allocation GPUs occupied; inspect before launch: {occupied}')
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert all(int(v) < 1024 for v in usage), 'Graphics/GPU memory already occupied'
    proof = run/'allocations'/f'allocation_{a.job_id}.json'
    evidence = json.loads(proof.read_text())
    assert 0 <= time.time()-evidence['checked_unix'] < 900, 'Refresh allocation evidence on login node'
    assert evidence['state'] == 'RUN' and evidence['host'] == workers[0]['host']
    assert evidence['gpu_uuids'] == uuids and evidence['job_id'] == a.job_id
    assert evidence['deadline_unix']-time.time() >= a.wall_budget+900, 'Insufficient LSF time headroom'
    checkpoint = Path(top['checkpoint']); checkpoint.resolve().relative_to(ROOT.resolve())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == top['checkpoint_sha256']
    sock = ROOT/f'.dq_{a.job_id}.sock'
    run_tag = hashlib.sha256(str(run).encode()).hexdigest()[:6]
    plans = []
    for w in workers:
        group = f'{a.job_id}_gpu{w["gpu"]}'
        if (run/'groups'/group).exists():
            raise FileExistsError(f'Group already exists: {group}; inspect logs instead of duplicating it')
        command = [sys.executable, '-u', str(ROOT/'evaluation/run_carla_parallel.py'),
                   '--job-id', a.job_id, '--gpu', str(w['gpu']), '--lanes', str(top['lanes']),
                   '--backend', top['backend'], '--group-id', group, '--route-ids', 'all',
                   '--run-root', str(run), '--checkpoint', str(checkpoint), '--seed', str(top['seed']),
                   '--slot-base', str(w['slot_base']), '--tm-base', str(w['tm_base']),
                   '--wall-budget', str(a.wall_budget), '--allocation-record', str(proof)]
        log = run/f'{group}.controller.log'
        plans.append({'group': group, 'command': command, 'session': f'dq_{run_tag}_gpu{w["gpu"]}', 'log': str(log)})
    for plan in plans:
        shell = 'exec '+shlex.join(plan['command'])+' > '+shlex.quote(plan['log'])+' 2>&1'
        subprocess.run(['tmux', '-S', str(sock), 'new-session', '-d', '-s', plan['session'], shell], cwd=ROOT, check=True)
    metadata = {'job_id': a.job_id, 'host': socket.gethostname(), 'unix': time.time(),
                'socket': str(sock), 'launches': plans, 'lsf_modified': False}
    (run/f'node_launch_{a.job_id}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
