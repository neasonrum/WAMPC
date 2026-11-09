"""Launch only the two benchmark cases assigned to an existing allocation."""
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
from evaluation.carla_eval_resources import resources, select_gpu_cpus, cpu_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    check_job(a.job_id)
    assert Path(sys.prefix).resolve() == (ROOT/'.venvs/full-diffusion-carla').resolve()
    run = a.run_root.resolve()
    run.relative_to(ROOT/'results')
    cases = [c for c in json.loads((run/'campaign.json').read_text())['cases'] if c['job_id'] == a.job_id]
    assert len(cases) == 2 and {c['gpu'] for c in cases} == {0, 1}
    affinity = sorted(os.sched_getaffinity(0))
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(uuids) == 2
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
    assert not occupied, f'GPU compute processes already active: {occupied}'
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert all(int(v) < 1024 for v in memory), 'GPU graphics memory already occupied'
    sample = resources()
    assert int(sample['job_memory.current'])+40*1024**3 < .85*int(sample['job_memory.max']), 'Insufficient RAM headroom'
    plans = []
    for case in cases:
        target = Path(case['run_root'])
        evidence = target/'allocations'/f'allocation_{a.job_id}.json'
        record = json.loads(evidence.read_text())
        assert record['state'] == 'RUN' and record['job_id'] == a.job_id
        assert record['host'] == socket.gethostname().split('.')[0]
        assert record['gpu_uuids'] == uuids and time.time()-record['checked_unix'] < 900
        assert record['deadline_unix']-time.time() > case['wall_budget_seconds']+900
        expected = json.loads((target/'source_sha256.json').read_text())
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n,h in expected.items())
        cpu_group = select_gpu_cpus(affinity, case['gpu'], case['cpus_per_gpu'])
        command = [sys.executable, '-u', str(ROOT/'evaluation/run_full_diffusion2_carla.py'),
                   '--job-id', a.job_id, '--run-root', str(target), '--gpu', str(case['gpu']),
                   '--checkpoint', case['checkpoint'], '--backend', 'direct', '--lanes', '1',
                   '--route-ids', ','.join(case['routes']), '--group-id', f'{a.job_id}_gpu{case["gpu"]}',
                   '--slot-base', str(case['worker']['slot_base']), '--tm-base', str(case['worker']['tm_base']),
                   '--seed', '901', '--cpus-per-gpu', str(case['cpus_per_gpu']),
                   '--server-mode', case['server_mode'], '--startup-wait', case['startup_wait'],
                   '--wall-budget', str(case['wall_budget_seconds']), '--allocation-record', str(evidence)]
        tag = hashlib.sha256(str(target).encode()).hexdigest()[:6]
        plans.append({'command': command, 'session': f'fd2speed_{tag}', 'cpu_affinity': cpu_group,
                      'log': str(target/'controller.log'), 'case': case['name']})
    assert not set(plans[0]['cpu_affinity']) & set(plans[1]['cpu_affinity'])
    sock = Path('results')/f'fd2speed_{a.job_id}.sock'
    for plan in plans:
        shell = 'exec '+shlex.join(plan['command'])+' > '+shlex.quote(plan['log'])+' 2>&1'
        subprocess.run(['tmux', '-S', str(sock), 'new-session', '-d', '-s', plan['session'], shell], cwd=ROOT, check=True)
    result = {'job_id': a.job_id, 'host': socket.gethostname(), 'socket': str(sock),
              'utc_unix': time.time(), 'cpu_cores': cpu_groups(affinity), 'launches': plans, 'initial_resources': sample}
    (run/f'node_launch_{a.job_id}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
