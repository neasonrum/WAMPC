"""One documented recovery attempt for the inspected route-23670 tick stall.

Uses the original evaluator, policy and 100-step checkpoint. Retains both failed
attempts, and adds passive Python stack dumps to diagnose a repeated stall.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.run_full_diffusion_carla import read, save, trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    check_job(args.job_id)
    assert args.job_id == '13053006'
    assert socket.gethostname().split('.')[0] == 'rng-dl01-w005'
    assert args.job_id in Path('/proc/self/cgroup').read_text()
    assert Path(sys.prefix).resolve() == (ROOT / '.venvs/full-diffusion-carla').resolve()
    base = ROOT / 'results/full_diffusion_carla_20261031_epoch10'
    route = base / 'full/worker2/route_23670'
    old = read(route / 'summary.json')
    assert old['classification'] == 'infrastructure_error'
    assert old['supervisor_reason'] == 'no_simulation_tick_progress'
    for index in (1, 2):
        previous = read(route / f'attempt_{index:02d}/summary.json')
        assert previous['classification'] == 'infrastructure_error'
        assert previous['supervisor_reason'] == 'no_simulation_tick_progress'
    expected = next(w for w in read(base / 'launch_topology.json')['workers'] if w['worker_index'] == 2)
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(uuids) == 2 and uuids[0] == expected['gpu_uuid']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    assert uuids[0] not in apps, 'Target logical GPU is occupied'
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) >= 8
    os.sched_setaffinity(0, affinity[:len(affinity)//2])
    checkpoint = base / 'checkpoint/epoch_10.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3'
    runtime = ROOT / '.runtime/carla_0.9.15_v1'
    assert (runtime / 'runtime_ready.json').is_file()
    os.environ['CARLA_ROOT'] = str(runtime)
    recovery = base / 'recovery_route_23670'
    recovery.mkdir(exist_ok=False)
    shutil.copy2(route / 'summary.json', recovery / 'route_summary_before_recovery.json')
    shutil.copy2(base / 'full/worker2/completed.json', recovery / 'worker_completed_before_recovery.json')
    save(recovery / 'recovery_plan.json', {
        'unix': time.time(), 'job_id': args.job_id, 'host': socket.gethostname(),
        'gpu_logical_index': 0, 'gpu_uuid': uuids[0], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'reason': 'Both prior attempts stopped producing simulation ticks near game time 71.9 seconds',
        'extra_attempt_limit': 1, 'attempt': str(route / 'attempt_03'),
        'runtime': str(runtime), 'checkpoint': str(checkpoint), 'inference_steps': 100,
        'changes': 'Private unchanged CARLA runtime and fresh port range; passive Python stack dumps every 180 wall seconds',
        'policy': 'No driving failure is rerun. Preserve all original attempts and publish only a scored driving result.'})
    original_popen = subprocess.Popen
    diagnostic = str(ROOT / 'evaluation/carla_diagnostics_evaluator.py')

    def instrumented_popen(command, *popen_args, **popen_kwargs):
        if isinstance(command, list) and len(command) > 1 and command[1] == diagnostic:
            code = ('import faulthandler,runpy,sys; '
                    'faulthandler.enable(); faulthandler.dump_traceback_later(180,repeat=True); '
                    'script=sys.argv.pop(1); sys.argv[0]=script; '
                    'runpy.run_path(script,run_name="__main__")')
            command = [command[0], '-c', code, *command[1:]]
            save(recovery / 'actual_evaluator_command.json', {'command': command})
        return original_popen(command, *popen_args, **popen_kwargs)

    settings = SimpleNamespace(job_id=args.job_id, checkpoint=checkpoint, output_root=recovery,
                               seed=901, slot=2452, gpu=0, phase='full', worker_index=2,
                               startup_timeout=600, no_progress_timeout=300, route_wall_timeout=5400)
    locks = base / 'locks'
    with (locks / f'{uuids[0]}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in range(26520, 26524):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
        subprocess.Popen = instrumented_popen
        try:
            result = trial(settings, '23670', route / 'attempt_03', uuids[0])
        finally:
            subprocess.Popen = original_popen
        save(recovery / 'result.json', result)
        if result['classification'] != 'driving_result':
            save(recovery / 'needs_attention.json', result)
            return 2
        result['infrastructure_recovery'] = str(recovery / 'recovery_plan.json')
        save(route / 'summary.json', result)
        completed = read(base / 'full/worker2/completed.json')
        completed.update(unix=time.time(), driving_results=28, errors=[],
                         infrastructure_recovery=str(recovery / 'recovery_plan.json'))
        save(base / 'full/worker2/completed.json', completed)
        save(recovery / 'completed.json', {'unix': time.time(), 'route_id': '23670', 'classification': 'driving_result'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
