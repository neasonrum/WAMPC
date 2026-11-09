"""Complete one inspected w002 recovery handoff after TCP sockets are released.

Does not repeat any route, change the evaluator, or stop any process. If the
ordinary controller is already running, it records that fact and exits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.run_full_diffusion_carla import read, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--worker', type=int, choices=(7,), required=True)
    parser.add_argument('--recovery-pid', type=int, required=True)
    args = parser.parse_args()
    check_job(args.job_id)
    assert args.job_id == '13047896'
    assert socket.gethostname().split('.')[0] == 'rng-dl01-w002'
    assert args.job_id in Path('/proc/self/cgroup').read_text()
    assert Path(sys.prefix).resolve() == (ROOT / '.venvs/full-diffusion-carla').resolve()
    base = ROOT / 'results/full_diffusion_carla_20261031_epoch10'
    output = base / 'full/worker7'
    recovery = output / 'recovery_storage_export_20261031'
    journal = recovery / 'port_handoff_status.json'

    def event(phase, **values):
        record = {'phase': phase, 'unix': time.time(), 'pid': os.getpid(), **values}
        save(journal, record)
        print(json.dumps(record), flush=True)

    event('waiting_for_existing_recovery', recovery_pid=args.recovery_pid)
    deadline = time.monotonic() + 6000
    controller_seen_at = None
    while time.monotonic() < deadline:
        try:
            command = Path(f'/proc/{args.recovery_pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        except FileNotFoundError:
            break
        if not command:
            break
        if str(ROOT / 'evaluation/run_full_diffusion_carla.py') in command and str(output) in command:
            if controller_seen_at is None:
                controller_seen_at = time.monotonic()
            if time.monotonic() - controller_seen_at >= 15:
                event('ordinary_controller_already_active', controller_pid=args.recovery_pid)
                return 0
            time.sleep(5)
            continue
        if str(ROOT / 'evaluation/recover_carla_storage_workers.py') not in command or '--worker 7' not in command:
            raise RuntimeError('Recovery PID identity differs; no action taken')
        time.sleep(5)
    else:
        raise TimeoutError('Original recovery has not handed off within the bounded wait')

    assert read(recovery / 'recovery_completed.json', {}).get('driving_result') is True
    assert read(output / 'route_2709/summary.json', {}).get('classification') == 'driving_result'
    log = (recovery / 'recovery.controller.log').read_text()
    assert 'OSError: [Errno 98] Address already in use' in log, 'Inspect the controller exit before resuming'
    event('waiting_for_socket_release')
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            for port in range(21070, 21074):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port))
            break
        except OSError:
            time.sleep(5)
    else:
        raise TimeoutError('The original four ports remain occupied; no process stopped')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                    '--format=csv,noheader'], text=True)
    assert 'GPU-f653bbe0-9522-f398-aff0-3931b9958f47' not in apps, 'Recovery GPU is occupied'
    checkpoint = base / 'checkpoint/epoch_10.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3'
    runtime = ROOT / '.runtime/carla_0.9.15_v1'
    assert (runtime / 'runtime_ready.json').is_file()
    os.environ['CARLA_ROOT'] = str(runtime)
    command = [sys.executable, '-u', str(ROOT / 'evaluation/run_full_diffusion_carla.py'),
               '--job-id', args.job_id, '--phase', 'full', '--gpu', '1', '--worker-index', '7',
               '--num-workers', '8', '--slot', '1907', '--seed', '901', '--checkpoint', str(checkpoint),
               '--output-root', str(output), '--max-attempts', '2']
    event('resuming_original_controller', command=command, preserve_driving_results=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
