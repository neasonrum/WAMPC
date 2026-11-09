"""Bounded, resumable closed-loop evaluation on a user-provided LSF allocation.

Each route gets a fresh server. Driving failures are kept; only infrastructure
failures are retried, with a fixed finite limit and all attempts retained.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import build_plan, check_job


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def stop_server(attempt, job_id):
    info = read(attempt / 'server_process.json', {})
    pid = info.get('pid')
    if not pid:
        return
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00', ' ')
        cgroup = Path(f'/proc/{pid}/cgroup').read_text()
        if str(attempt / 'CarlaUE4.log') not in command or job_id not in cgroup:
            return
        if os.getpgid(pid) != pid:
            return
        os.killpg(pid, signal.SIGTERM)
        for _ in range(20):
            if not Path(f'/proc/{pid}').exists():
                return
            time.sleep(0.25)
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
        pass


def resources():
    result = {'unix': time.time()}
    group = Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events'):
        p = Path('/sys/fs/cgroup') / group / name
        if p.exists():
            result[name] = p.read_text().strip()
        p = Path('/sys/fs/cgroup') / str(Path(group).parent) / name
        if p.exists():
            result['job_' + name] = p.read_text().strip()
    result['gpu'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15).stdout
    return result


def classify_result(result, record, server_exit, reason):
    status = record.get('status', '') if record else ''
    if any(message in status for message in
           ("Agent couldn't be set up", 'Agent crashed', "Agent's sensors were invalid")):
        return 'agent_error'
    if result.get('entry_status') == 'Rejected':
        return 'agent_error'
    if reason or 'Simulation crashed' in status or (server_exit and not server_exit.get('evaluator_finishing')):
        return 'infrastructure_error'
    if record and record.get('meta', {}).get('duration_game', 0) > 0 and status != 'Started':
        return 'driving_result'
    return 'evaluation_error'


def trial(args, route, attempt, gpu_uuid):
    attempt.mkdir(parents=True)
    for name in ('tmp', 'cache', 'matplotlib'):
        (attempt / name).mkdir()
    plan_args = SimpleNamespace(method='full_diffusion', experiment='carla_' + args.checkpoint.stem.replace('_', ''),
                                seed=args.seed, slot=args.slot, gpu_rank=args.gpu,
                                routes_subset=route, agent=str(ROOT / 'evaluation/full_diffusion_diagnostic_agent.py'),
                                agent_config=str(args.checkpoint), carla_root=None,
                                python=sys.executable, resume=False)
    plan = build_plan(plan_args)
    plan['output_dir'] = str(attempt)
    plan['command'][1] = str(ROOT / 'evaluation/carla_diagnostics_evaluator.py')
    command = plan['command']
    command[command.index('--checkpoint') + 1] = str(attempt / 'results.json')
    command[command.index('--debug-checkpoint') + 1] = str(attempt / 'live_results.txt')
    plan['environment'].update({
        'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'WMMPC_CARLA_LOG_DIR': str(attempt), 'PYTHONUNBUFFERED': '1',
        'SAVE_PATH': str(attempt / 'agent_data'), 'TMPDIR': str(attempt / 'tmp'),
        'XDG_CACHE_HOME': str(attempt / 'cache'), 'MPLCONFIGDIR': str(attempt / 'matplotlib'),
        'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'VK_ICD_FILENAMES': '/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json',
    })
    missing = [key for key, ready in plan['readiness'].items() if not ready]
    if missing:
        raise RuntimeError(f'Missing prerequisites: {missing}')
    environment = os.environ.copy()
    environment.update(plan['environment'])
    save(attempt / 'plan.json', plan)
    start = time.time()
    with (attempt / 'evaluation.log').open('wb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        save(attempt / 'evaluator_process.json', {'pid': proc.pid, 'unix': start})
        reason = None
        last_resource = 0
        while proc.poll() is None:
            now = time.time()
            telemetry = attempt / 'telemetry.jsonl'
            if telemetry.exists():
                if now - telemetry.stat().st_mtime > args.no_progress_timeout:
                    reason = 'no_simulation_tick_progress'
            elif now - start > args.startup_timeout:
                reason = 'startup_timeout'
            if now - start > args.route_wall_timeout:
                reason = 'route_wall_timeout'
            if now - last_resource > 60:
                sample = resources()
                with (attempt / 'resources.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample) + '\n')
                save(args.output_root / 'status.json', {
                    'phase': args.phase, 'worker': args.worker_index, 'gpu': args.gpu,
                    'job_id': args.job_id, 'host': socket.gethostname(), 'route_id': route,
                    'attempt': str(attempt), 'evaluator_pid': proc.pid,
                    'elapsed_seconds': now - start, 'unix': now,
                    'last_tick_unix': telemetry.stat().st_mtime if telemetry.exists() else None,
                    'resources': sample})
                last_resource = now
            if reason:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                break
            time.sleep(3)
        returncode = proc.wait()
    stop_server(attempt, args.job_id)
    result = read(attempt / 'results.json', {})
    records = result.get('_checkpoint', {}).get('records', [])
    record = records[0] if len(records) == 1 else None
    status = record.get('status', '') if record else ''
    server_exit = read(attempt / 'server_exit.json', {})
    classification = classify_result(result, record, server_exit, reason)
    telemetry_lines = []
    if (attempt / 'telemetry.jsonl').exists():
        telemetry_lines = (attempt / 'telemetry.jsonl').read_text().splitlines()
    summary = {'route_id': route, 'classification': classification, 'status': status,
               'returncode': returncode, 'supervisor_reason': reason,
               'elapsed_seconds': time.time() - start, 'attempt': str(attempt),
               'server_exit': server_exit, 'telemetry_records': len(telemetry_lines),
               'last_tick': json.loads(telemetry_lines[-1]) if telemetry_lines else None,
               'record': record, 'entry_status': result.get('entry_status')}
    save(attempt / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--slot', type=int, default=1900)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--route-id', default='1711')
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--no-progress-timeout', type=float, default=300)
    parser.add_argument('--route-wall-timeout', type=float, default=5400)
    args = parser.parse_args()
    args.worker_index = args.gpu if args.worker_index is None else args.worker_index
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError('Invalid global worker index/count')
    check_job(args.job_id)
    if Path(sys.prefix).resolve() != (ROOT / '.venvs/full-diffusion-carla').resolve():
        raise RuntimeError('Use the isolated full-diffusion-carla evaluation environment')
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.relative_to(ROOT.resolve())
    args.checkpoint.relative_to(ROOT.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_uuids = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    if len(gpu_uuids) != 2 or args.gpu not in (0, 1):
        raise RuntimeError(f'Expected the two allocation-visible GPUs, got {gpu_uuids}')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError(f'Unexpectedly small allocation CPU affinity: {affinity}')
    if args.phase == 'full':
        start = len(affinity) * args.gpu // len(gpu_uuids)
        end = len(affinity) * (args.gpu + 1) // len(gpu_uuids)
        os.sched_setaffinity(0, affinity[start:end])
    routes_xml = ROOT / 'leaderboard/data/bench2drive220.xml'
    available = [route.attrib['id'] for route in ET.parse(routes_xml).getroot().findall('route')]
    routes = [args.route_id] if args.phase == 'smoke' else available[args.worker_index::args.num_workers]
    if not set(routes) <= set(available):
        raise ValueError('Unknown route IDs')
    manifest = {'job_id': args.job_id, 'host': socket.gethostname(), 'phase': args.phase,
                'python': sys.executable, 'prefix': sys.prefix, 'gpu_logical_index': args.gpu,
                'gpu_uuid': gpu_uuids[args.gpu], 'affinity': sorted(os.sched_getaffinity(0)),
                'worker_index': args.worker_index, 'num_workers': args.num_workers,
                'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ('evaluation/run_full_diffusion_carla.py',
                                               'evaluation/carla_diagnostics_evaluator.py',
                                               'evaluation/full_diffusion_diagnostic_agent.py',
                                               'method/full_diffusion/inference/carla_agent.py',
                                               'method/full_diffusion/inference/policy.py')},
                'routes': routes, 'seed': args.seed, 'fresh_server_per_route': True,
                'retry_policy': 'Only infrastructure errors, maximum two attempts; driving failures are retained',
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save(args.output_root / 'manifest.json', manifest)
    locks = args.output_root.parents[1] / 'locks' if args.phase == 'full' else args.output_root.parent / 'locks'
    locks.mkdir(exist_ok=True)
    summaries = []
    with ExitStack() as stack:
        gpu_lock = stack.enter_context((locks / f'{gpu_uuids[args.gpu]}.lock').open('a'))
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in range(2000 + args.slot * 10, 2004 + args.slot * 10):
            lock = stack.enter_context((locks / f'{socket.gethostname()}_port_{port}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
        for route in routes:
            route_root = args.output_root / f'route_{route}'
            route_root.mkdir(exist_ok=True)
            finished = read(route_root / 'summary.json')
            if finished and finished['classification'] == 'driving_result':
                summaries.append(finished)
                continue
            summary = None
            for index in range(1, args.max_attempts + 1):
                attempt = route_root / f'attempt_{index:02d}'
                if attempt.exists():
                    previous = read(attempt / 'summary.json')
                    if previous:
                        summary = previous
                        if summary['classification'] != 'infrastructure_error':
                            break
                        continue
                    raise RuntimeError(f'Incomplete attempt requires inspection before resuming: {attempt}')
                summary = trial(args, route, attempt, gpu_uuids[args.gpu])
                if summary['classification'] != 'infrastructure_error':
                    break
            save(route_root / 'summary.json', summary)
            summaries.append(summary)
            save(args.output_root / 'progress.json', {
                'done': len(summaries), 'total': len(routes), 'unix': time.time(),
                'driving_results': sum(s['classification'] == 'driving_result' for s in summaries),
                'errors': [s['route_id'] for s in summaries if s['classification'] != 'driving_result']})
            if summary['classification'] in ('agent_error', 'evaluation_error'):
                save(args.output_root / 'needs_attention.json', summary)
                return 2
        save(args.output_root / 'completed.json', {
            'unix': time.time(), 'routes': len(summaries),
            'driving_results': sum(s['classification'] == 'driving_result' for s in summaries),
            'errors': [s['route_id'] for s in summaries if s['classification'] != 'driving_result']})
    return int(any(s['classification'] != 'driving_result' for s in summaries))


if __name__ == '__main__':
    raise SystemExit(main())
