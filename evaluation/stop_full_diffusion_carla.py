"""Stop only a named evaluation's owned processes, preserving its LSF allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def inspect(pid, job_id, run):
    proc = Path(f'/proc/{pid}')
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        cgroup = (proc / 'cgroup').read_text()
        if not re.search(rf'(?<!\d){re.escape(job_id)}(?!\d)', cgroup):
            return None
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = (proc / 'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
        cwd = (proc / 'cwd').resolve()
        def argument(flag):
            if flag not in argv:
                return None
            p = Path(argv[argv.index(flag) + 1])
            return (cwd / p).resolve() if not p.is_absolute() else p.resolve()
        names = {Path(x).name for x in argv}
        role = None
        if 'run_full_diffusion_carla.py' in names:
            p = argument('--output-root')
            if p and p.parent == run / 'full':
                role = 'controller'
        elif 'summarize_full_diffusion_carla.py' in names and argument('--run-root') == run:
            role = 'reporter'
        elif 'carla_diagnostics_evaluator.py' in names:
            p = argument('--checkpoint')
            if p and p.is_relative_to(run / 'full'):
                role = 'evaluator'
        elif argv and Path(argv[0]).name == 'CarlaUE4-Linux-Shipping':
            for value in argv:
                if value.startswith('-abslog='):
                    p = Path(value.split('=', 1)[1]).resolve()
                    if p.is_relative_to(run / 'full'):
                        role = 'server'
        if role:
            return {'pid': pid, 'role': role, 'argv': argv, 'start_ticks': fields[19],
                    'state': fields[0], 'cgroup': cgroup, 'pgid': os.getpgid(pid)}
    except (OSError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--replacement-run', type=Path, required=True)
    args = parser.parse_args()
    check_job(args.job_id)
    run = args.run_root.resolve()
    replacement = args.replacement_run.resolve()
    run.relative_to(ROOT.resolve() / 'results')
    replacement.relative_to(ROOT.resolve() / 'results')
    assert run != replacement
    integrity = read(replacement / 'checkpoint/integrity.json', {})
    assert integrity.get('updates') == 43210 and integrity.get('strict_head_load')
    records = [inspect(int(p.name), args.job_id, run) for p in Path('/proc').iterdir() if p.name.isdigit()]
    records = [r for r in records if r]
    controllers = [r for r in records if r['role'] == 'controller']
    if len(controllers) != 2:
        raise RuntimeError(f'Expected two old evaluation controllers, found {len(controllers)}')
    report = {'job_id': args.job_id, 'host': socket.gethostname(), 'unix': time.time(),
              'run': str(run), 'replacement_run': str(replacement),
              'reason': 'User requested final epoch-10 evaluation on all eight authorized A100 GPUs',
              'processes_before': records, 'actions': [], 'allocation_cancelled': False}
    target = run / 'evaluation_stopped.json'
    save(target, report)

    def send(record, sig):
        now = inspect(record['pid'], args.job_id, run)
        if now and now['start_ticks'] == record['start_ticks']:
            os.kill(record['pid'], sig)
            report['actions'].append({'pid': record['pid'], 'role': record['role'],
                                      'signal': signal.Signals(sig).name, 'unix': time.time()})
            save(target, report)

    # Freeze only the two Python route schedulers, so no new route can be spawned
    # while their evaluator processes are being shut down.
    for r in controllers:
        send(r, signal.SIGSTOP)
    frozen = True
    try:
        current = [inspect(int(p.name), args.job_id, run) for p in Path('/proc').iterdir() if p.name.isdigit()]
        current = [r for r in current if r]
        for r in current:
            if r['role'] == 'reporter':
                send(r, signal.SIGTERM)
        evaluators = [r for r in current if r['role'] == 'evaluator']
        for r in evaluators:
            send(r, signal.SIGTERM)
        deadline = time.time() + 25
        while time.time() < deadline and any(inspect(r['pid'], args.job_id, run) for r in evaluators):
            time.sleep(0.25)
        for r in evaluators:
            send(r, signal.SIGKILL)
        # The evaluator normally cleans up its server. Clean up only any exact
        # logged server belonging to this run if its parent could not do so.
        for r in current:
            if r['role'] == 'server':
                send(r, signal.SIGTERM)
        time.sleep(1)
        for r in current:
            if r['role'] == 'server':
                send(r, signal.SIGKILL)
        for r in controllers:
            send(r, signal.SIGTERM)
            send(r, signal.SIGCONT)
        frozen = False
        time.sleep(1)
        for r in controllers:
            send(r, signal.SIGKILL)
        survivors = [inspect(r['pid'], args.job_id, run) for r in records]
        report['survivors'] = [r for r in survivors if r]
        report['finished_unix'] = time.time()
        report['completed_routes_preserved'] = len(list((run / 'full').glob('worker*/route_*/summary.json')))
        report['gpu_after'] = subprocess.check_output(['nvidia-smi'], text=True)
        save(target, report)
        print(json.dumps({'stopped': not report['survivors'],
                          'completed_routes_preserved': report['completed_routes_preserved'],
                          'allocation_cancelled': False}), flush=True)
        if report['survivors']:
            raise RuntimeError('Some old evaluation processes require inspection')
    finally:
        if frozen:
            for r in controllers:
                send(r, signal.SIGCONT)


if __name__ == '__main__':
    main()
