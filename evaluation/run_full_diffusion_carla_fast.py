"""Bounded multi-route server reuse with the unchanged 100-step policy.

Publishes the same per-route result format as the baseline runner. A crashed
batch preserves completed routes and retries only unfinished infrastructure
failures, at most twice. Routes after the preloaded first episode reload the
world; every route reconstructs the original agent and its original sampler.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
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
from evaluation.run_full_diffusion_carla import read, save, resources, stop_server, classify_result
from evaluation.fast_carla_io import cooperative_pause
from evaluation.carla_lifecycle import ordered_route_file
from evaluation.carla_eval_resources import evaluation_runtime_root


def route_number(record):
    return record['route_id'].split('_')[1]


def stop_evaluator(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def publish_completed(result, routes, attempt, output, previous_finish):
    published = []
    for record in result.get('_checkpoint', {}).get('records', []):
        route = route_number(record)
        if route not in routes:
            raise RuntimeError(f'Unexpected route record: {route}')
        # entry_status is global to the batch; a later agent rejection must not
        # invalidate already completed earlier routes.
        if classify_result({}, record, {}, None) != 'driving_result':
            continue
        path = output / f'route_{route}/summary.json'
        if (read(path) or {}).get('classification') == 'driving_result':
            continue
        # The evaluator writes statistics before destroy() flushes metrics.
        # Publish only when both the record and its complete metric JSON exist.
        metric = attempt / 'agent_data' / record['save_name'] / 'metric_info.json'
        if not metric.exists():
            continue
        now = time.time()
        summary = {'route_id': route, 'classification': 'driving_result',
                   'status': record['status'], 'record': record, 'attempt': str(attempt),
                   'entry_status': result.get('entry_status'), 'returncode': None,
                   'elapsed_seconds': now - previous_finish,
                   'timing_basis': 'Checkpoint completion observation interval, at most 3s polling delay',
                   'shared_server': True, 'inference_steps': 100}
        path.parent.mkdir(exist_ok=True)
        save(path, summary)
        published.append(route)
        previous_finish = now
    return published, previous_finish


def batch_trial(args, routes, attempt, gpu_uuid):
    attempt.mkdir(parents=True)
    for name in ('tmp', 'cache', 'matplotlib', 'home'):
        (attempt / name).mkdir()
    plan_args = SimpleNamespace(
        method='full_diffusion', experiment='carla_fast_' + args.checkpoint.stem,
        seed=args.seed, slot=args.slot, gpu_rank=args.gpu, routes_subset=','.join(routes),
        agent=str(ROOT / 'evaluation/full_diffusion_fast_agent.py'),
        agent_config=str(args.checkpoint), carla_root=None, python=sys.executable, resume=False)
    plan_args.carla_root = str(args.carla_root) if args.carla_root else None
    save(attempt / 'batch.json', {'routes': routes, 'inference_steps': 100,
                                 'route_reset': 'Unused preloaded first episode; later routes full episode reload',
                                 'sampler': 'Original VPFullDiffusion, unmodified'})
    try:
        plan = build_plan(plan_args)
    except OSError as exc:
        failure = {'classification': 'infrastructure_error', 'routes': routes,
                   'supervisor_reason': 'runtime_preflight_oserror', 'error': repr(exc),
                   'elapsed_seconds': 0, 'inference_steps': 100}
        save(attempt / 'summary.json', failure)
        return {route: {**failure, 'route_id': route, 'attempt': str(attempt)} for route in routes}
    plan['output_dir'] = str(attempt)
    plan['command'][1] = str(ROOT / 'evaluation/carla_fast_evaluator.py')
    command = plan['command']
    if getattr(args,'_traffic_manager_port',None):
        command[command.index('--traffic-manager-port')+1]=str(args._traffic_manager_port)
    command[command.index('--checkpoint') + 1] = str(attempt / 'results.json')
    command[command.index('--debug-checkpoint') + 1] = str(attempt / 'live_results.txt')
    original_routes = Path(command[command.index('--routes') + 1])
    private_routes = attempt / 'routes.xml'
    ordered_route_file(original_routes, routes, private_routes)
    command[command.index('--routes') + 1] = str(private_routes)
    plan['environment'].update({
        'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'WMMPC_CARLA_LOG_DIR': str(attempt), 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'SAVE_PATH': str(attempt / 'agent_data'), 'TMPDIR': str(attempt / 'tmp'),
        'HOME': str(attempt / 'home'), 'HF_HOME': str(attempt / 'cache/huggingface'),
        'XDG_CACHE_HOME': str(attempt / 'cache'), 'MPLCONFIGDIR': str(attempt / 'matplotlib'),
        'TORCH_HOME': str(evaluation_runtime_root() / 'torch_cache'),
        'TRITON_CACHE_DIR': str(evaluation_runtime_root() / 'triton_cache'),
        'TORCH_EXTENSIONS_DIR': str(evaluation_runtime_root() / 'torch_extensions'),
        'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'WMMPC_CARLA_CACHE_DIR': str(evaluation_runtime_root() / 'carla_cache' / args.job_id / gpu_uuid),
        'WMMPC_CARLA_PRELOAD_FIRST_MAP': '0',
        'VK_ICD_FILENAMES': '/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json'})
    if not all(plan['readiness'].values()):
        raise RuntimeError(f'Missing prerequisites: {plan["readiness"]}')
    save(attempt / 'plan.json', plan)
    environment = os.environ.copy()
    environment.update(plan['environment'])
    started = time.time()
    last_finish = started
    last_resource = 0
    active_route = None
    active_started = started
    published = set()
    reason = None
    with (attempt / 'evaluation.log').open('wb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        args._active_evaluator=proc
        save(attempt / 'evaluator_process.json', {'pid': proc.pid, 'unix': started})
        while proc.poll() is None:
            now = time.time()
            external=getattr(args,'_external_server',None)
            if external is not None and external.poll() is not None:
                reason='external_carla_server_exit'
            telemetry = attempt / 'telemetry.jsonl'
            last = None
            if telemetry.exists():
                lines = telemetry.read_text().splitlines()
                if lines:
                    try:
                        last = json.loads(lines[-1])
                    except ValueError:
                        last = json.loads(lines[-2]) if len(lines) > 1 else None
            if last:
                match = re.match(r'RouteScenario_(\d+)', last.get('route', ''))
                route = match[1] if match else active_route
                if route != active_route:
                    active_route, active_started = route, now
                if now - last['unix'] > args.no_progress_timeout:
                    reason = 'no_simulation_tick_progress'
            elif now - started > args.startup_timeout:
                reason = 'startup_timeout'
            if now - active_started > args.route_wall_timeout:
                reason = 'route_wall_timeout'
            result = read(attempt / 'results.json', {})
            just_finished, last_finish = publish_completed(
                result, routes, attempt, args.output_root, last_finish)
            published.update(just_finished)
            if now - last_resource > 60:
                sample = resources()
                with (attempt / 'resources.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample) + '\n')
                save(args.output_root / 'status.json', {
                    'phase': args.phase, 'worker': args.worker_index, 'gpu': args.gpu,
                    'job_id': args.job_id, 'host': socket.gethostname(),
                    'route_id': active_route or routes[0], 'batch_routes': routes,
                    'attempt': str(attempt), 'evaluator_pid': proc.pid,
                    'elapsed_seconds': now - started, 'unix': now,
                    'last_tick_unix': last['unix'] if last else None, 'resources': sample})
                last_resource = now
            if reason:
                stop_evaluator(proc)
                break
            time.sleep(3)
        returncode = proc.wait()
    stop_server(attempt, args.job_id)
    result = read(attempt / 'results.json', {})
    just_finished, last_finish = publish_completed(result, routes, attempt, args.output_root, last_finish)
    published.update(just_finished)
    server_exit = read(attempt / 'server_exit.json', {})
    records = {route_number(r): r for r in result.get('_checkpoint', {}).get('records', [])}
    outcomes = {}
    for route in routes:
        if route in published:
            outcomes[route] = read(args.output_root / f'route_{route}/summary.json')
            continue
        record = records.get(route)
        # Missing records after a batch crash are unfinished infrastructure work.
        failure_reason = reason or ('batch_interrupted' if not record else None)
        classification = classify_result(result, record, server_exit, failure_reason)
        if classification == 'driving_result':
            classification = 'evaluation_error'  # Required final metric JSON is absent.
        outcomes[route] = {'route_id': route, 'classification': classification,
                           'status': record.get('status') if record else 'Not started',
                           'record': record, 'attempt': str(attempt), 'returncode': returncode,
                           'supervisor_reason': failure_reason, 'server_exit': server_exit,
                           'shared_server': True, 'inference_steps': 100}
        save(attempt / f'route_{route}.summary.json', outcomes[route])
    group_class = ('driving_result' if all(x['classification'] == 'driving_result' for x in outcomes.values())
                   else 'infrastructure_error' if any(x['classification'] == 'infrastructure_error' for x in outcomes.values())
                   else 'evaluation_error')
    save(attempt / 'summary.json', {'classification': group_class, 'unit': 'server_batch',
                                   'routes': routes, 'completed_routes': sorted(published, key=int),
                                   'returncode': returncode, 'supervisor_reason': reason,
                                   'elapsed_seconds': time.time() - started, 'server_exit': server_exit})
    return outcomes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job-id', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--phase', choices=('smoke', 'full'), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--slot', type=int, default=1950)
    p.add_argument('--seed', type=int, default=901)
    p.add_argument('--route-ids', default='1711,1773')
    p.add_argument('--routes-per-server', type=int, default=1,
                   help='Keep one for conservative scoring; multi-route server reuse is experimental')
    p.add_argument('--carla-root', type=Path)
    p.add_argument('--max-attempts', type=int, default=2)
    p.add_argument('--startup-timeout', type=float, default=600)
    p.add_argument('--no-progress-timeout', type=float, default=300)
    p.add_argument('--route-wall-timeout', type=float, default=5400)
    args = p.parse_args()
    check_job(args.job_id)
    if Path(sys.prefix).resolve() != (ROOT / '.venvs/full-diffusion-carla').resolve():
        raise RuntimeError('Use the isolated evaluation environment')
    if not 0 <= args.worker_index < args.num_workers or not 1 <= args.routes_per_server <= 64:
        raise ValueError('Invalid worker index or routes-per-server')
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.relative_to(ROOT.resolve())
    args.checkpoint.relative_to(ROOT.resolve())
    if args.carla_root:
        args.carla_root = args.carla_root.resolve()
        args.carla_root.relative_to(ROOT.resolve())
        if not (args.carla_root / 'runtime_ready.json').is_file():
            raise RuntimeError('Private runtime needs a completed copy and integrity manifest')
    args.output_root.mkdir(parents=True, exist_ok=True)
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True).strip().splitlines()
    if len(uuids) != 2 or args.gpu not in (0, 1):
        raise RuntimeError('Expected two allocation-visible GPUs')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError('Insufficient allocation CPU affinity')
    half = len(affinity) // 2
    os.sched_setaffinity(0, affinity[args.gpu * half:(args.gpu + 1) * half])
    elements = ET.parse(ROOT / 'leaderboard/data/bench2drive220.xml').getroot().findall('route')
    available = [x.attrib['id'] for x in elements]
    towns = {x.attrib['id']: x.attrib['town'] for x in elements}
    routes = args.route_ids.split(',') if args.phase == 'smoke' else available[args.worker_index::args.num_workers]
    if len(routes) != len(set(routes)) or not set(routes) <= set(available):
        raise ValueError('Unknown or duplicate routes')
    execution_order = sorted(routes, key=lambda route: towns[route]) if args.phase == 'full' else routes
    sources = ('evaluation/run_full_diffusion_carla_fast.py', 'evaluation/carla_fast_evaluator.py',
               'evaluation/full_diffusion_fast_agent.py', 'evaluation/fast_carla_io.py',
               'evaluation/carla_lifecycle.py', 'evaluation/carla_static_roads.py',
               'method/full_diffusion/diffusion.py',
               'method/full_diffusion/model.py', 'method/full_diffusion/inference/policy.py',
               'third_party/carla_garage/team_code/sensor_agent.py',
               'leaderboard/leaderboard/leaderboard_evaluator.py')
    manifest = {'job_id': args.job_id, 'host': socket.gethostname(), 'gpu_logical_index': args.gpu,
                'gpu_uuid': uuids[args.gpu], 'affinity': sorted(os.sched_getaffinity(0)),
                'worker_index': args.worker_index, 'num_workers': args.num_workers,
                'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'seed': args.seed, 'routes': routes, 'inference_steps': 100,
                'routes_per_server': args.routes_per_server, 'execution_order': execution_order,
                'route_reset': 'Unused preloaded first episode; later routes full episode reload',
                'sampler': 'Original VPFullDiffusion, unchanged',
                'carla_root': str(args.carla_root) if args.carla_root else None,
                'cooperative_pause_supported': True,
                'retry_policy': 'Only unfinished infrastructure failures, maximum two attempts; driving failures kept',
                'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    old = read(args.output_root / 'manifest.json')
    if old and any(old.get(k) != manifest[k] for k in ('checkpoint_sha256','seed','routes','execution_order','source_sha256','routes_per_server','carla_root')):
        raise RuntimeError('Resume identity differs from the saved run')
    save(args.output_root / 'manifest.json', manifest)
    done = {route: read(args.output_root / f'route_{route}/summary.json') for route in routes}
    queue = deque([execution_order[i:i+args.routes_per_server] for i in range(0,len(execution_order),args.routes_per_server)])
    tries = {route: 0 for route in routes}
    for path in args.output_root.glob('route_*/attempt_*/batch.json'):
        for route in read(path,{}).get('routes',[]):
            tries[route] = tries.get(route,0) + 1
        if not (path.parent / 'summary.json').exists():
            raise RuntimeError(f'Incomplete batch needs inspection before resuming: {path.parent}')
    locks = args.output_root.parent / 'locks'
    locks.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for name in (f'{uuids[args.gpu]}.lock', f'worker{args.worker_index}.lock'):
            fcntl.flock(stack.enter_context((locks/name).open('a')), fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared_locks = ROOT / '.runtime/carla_gpu_locks'
        shared_locks.mkdir(parents=True, exist_ok=True)
        fcntl.flock(stack.enter_context((shared_locks/f'{uuids[args.gpu]}.lock').open('a')),
                    fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in range(2000+args.slot*10,2004+args.slot*10):
            name=f'{socket.gethostname()}_port_{port}.lock'
            fcntl.flock(stack.enter_context((locks/name).open('a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',port))
        while queue:
            cooperative_pause(args.output_root, args.job_id)
            group = [route for route in queue.popleft()
                     if not done.get(route) or done[route]['classification']!='driving_result']
            if not group:
                continue
            if any(tries[route]>=args.max_attempts for route in group):
                raise RuntimeError('An exhausted route requires inspection before resuming')
            for route in group:
                tries[route]+=1
            attempt=args.output_root/f'route_{group[0]}'/f'attempt_{tries[group[0]]:02d}'
            if attempt.exists():
                raise RuntimeError(f'Attempt already exists: {attempt}')
            outcomes=batch_trial(args,group,attempt,uuids[args.gpu])
            retries=[]
            fatal=None
            for route, outcome in outcomes.items():
                if outcome['classification']=='infrastructure_error' and tries[route]<args.max_attempts:
                    retries.append([route])
                    continue
                path=args.output_root/f'route_{route}/summary.json';path.parent.mkdir(exist_ok=True)
                if outcome['classification']!='driving_result': save(path,outcome)
                done[route]=outcome
                if outcome['classification'] in ('agent_error','evaluation_error'): fatal=outcome
            queue.extendleft(reversed(retries))
            finished=[x for x in done.values() if x]
            save(args.output_root/'progress.json',{'done':len(finished),'total':len(routes),'unix':time.time(),
                                                  'driving_results':sum(x['classification']=='driving_result' for x in finished),
                                                  'errors':[x['route_id'] for x in finished if x['classification']!='driving_result']})
            if fatal:
                save(args.output_root/'needs_attention.json',fatal)
                return 2
        cooperative_pause(args.output_root, args.job_id)
        finished=[x for x in done.values() if x]
        result={'unix':time.time(),'routes':len(finished),'driving_results':sum(x['classification']=='driving_result' for x in finished),
                'errors':[x['route_id'] for x in finished if x['classification']!='driving_result']}
        save(args.output_root/'completed.json',result)
        return int(bool(result['errors']))


if __name__=='__main__':
    raise SystemExit(main())
