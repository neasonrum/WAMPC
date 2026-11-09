"""Dynamic route workers and bounded one-GPU concurrency experiments.

The parent owns the GPU lease, model broker and its lane controllers. Each lane
owns a CARLA server and launches a fresh original evaluator for every route.
Shared queues also support workers on other already-authorized allocations.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.carla_route_queue import RouteQueue
from evaluation.fast_carla_io import wait_for_server
from evaluation.run_full_diffusion_carla import read, save
from evaluation.carla_eval_resources import (
    checked_evaluation_path, checked_run_root, evaluation_runtime_root, resources, select_gpu_cpus,
)
from evaluation.run_full_diffusion_carla_fast import stop_evaluator
import evaluation.run_full_diffusion_carla_fast as fast


def configure_policy(configuration, method):
    agents = {'full_diffusion2': 'full_diffusion2_fast_agent.py',
              'full_diffusionv3': 'full_diffusionv3_fast_agent.py'}
    if method not in agents:
        raise ValueError(f'Unsupported temporal planner: {method}')
    configuration.method = method
    configuration.agent = str(ROOT / 'evaluation' / agents[method])
    return configuration


class LaneServer:
    def __init__(self, args, gpu_uuid, directory):
        self.directory = directory
        directory.mkdir(parents=True)
        cache = evaluation_runtime_root() / 'carla_cache' / args.job_id / gpu_uuid / 'fd2_speed'
        for name in ('home', 'cache', 'unreal_user'):
            (cache / name).mkdir(parents=True, exist_ok=True)
        (directory / 'tmp').mkdir()
        port = 2000 + args.slot * 10
        binary = args.carla_root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        command = [str(binary), 'CarlaUE4', '-RenderOffScreen', '-nosound', f'-carla-rpc-port={port}',
                   f'-graphicsadapter={args.gpu}', f'-abslog={directory / "CarlaUE4.log"}',
                   f'-userdir={cache / "unreal_user"}', '-unattended', '-nosplash']
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           HOME=str(cache / 'home'), XDG_CACHE_HOME=str(cache / 'cache'),
                           TMPDIR=str(directory / 'tmp'), OMP_NUM_THREADS='1',
                           VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json')
        with (directory / 'server_stdout.log').open('wb') as log:
            self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        self.owner = directory / 'server_process.json'
        save(self.owner, {'pid': self.process.pid, 'port': port, 'gpu_uuid': gpu_uuid,
                          'job_id': args.job_id, 'command': command, 'unix': time.time()})
        try:
            wait_for_server(self.process, 'localhost', port, directory, timeout=600)
        except BaseException:
            self.close()
            raise
        os.environ.update(WMMPC_EXTERNAL_CARLA_PID=str(self.process.pid),
                          WMMPC_EXTERNAL_CARLA_PORT=str(port), WMMPC_EXTERNAL_CARLA_OWNER=str(self.owner),
                          WMMPC_CACHE_STATIC_ROADS='0')

    def close(self):
        expected = self.process.poll() is None
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        save(self.directory / 'server_exit.json', {'pid': self.process.pid, 'returncode': self.process.returncode,
                                                  'owner_finishing': expected, 'unix': time.time()})
        for key in ('WMMPC_EXTERNAL_CARLA_PID', 'WMMPC_EXTERNAL_CARLA_PORT', 'WMMPC_EXTERNAL_CARLA_OWNER'):
            os.environ.pop(key, None)
        if getattr(self, 'port_lease', None) is not None:
            self.port_lease.close()


def start_lane_server(args, directory, sequence):
    """Reserve a free port block and retry bounded infrastructure startup failures."""
    lock_root = evaluation_runtime_root() / 'carla_port_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    errors = []
    for retry in range(3):
        sequence += 1
        target = directory / f'server_{sequence:03d}'
        lease = ExitStack()
        try:
            selected = None
            for offset in range(64):
                candidate = ExitStack()
                slot = args.slot_base + args.lane + offset
                try:
                    for port in range(2000 + slot*10, 2004 + slot*10):
                        handle = candidate.enter_context((lock_root / f'{socket.gethostname()}_{port}.lock').open('a'))
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        with socket.socket() as probe:
                            probe.bind(('0.0.0.0', port))
                    selected = slot
                    lease = candidate.pop_all()
                    break
                except OSError:
                    candidate.close()
            if selected is None:
                raise RuntimeError('No free CARLA port block in the assigned worker range')
            args.slot = selected
            owner = LaneServer(args, args.gpu_uuid, target)
            owner.port_lease = lease
            return owner, sequence
        except (OSError, RuntimeError, TimeoutError) as exc:
            lease.close()
            errors.append(repr(exc))
            target.mkdir(parents=True, exist_ok=True)
            save(target / 'startup_failure.json', {'error': repr(exc), 'retry': retry+1, 'unix': time.time()})
    raise RuntimeError(f'CARLA startup failed three times: {errors}')


def lane(args):
    args.run_root = checked_run_root(args.run_root)
    group = args.run_root / 'groups' / args.group_id
    args.output_root = group / f'lane_{args.lane}'
    args.output_root.mkdir(parents=True, exist_ok=True)
    affinity = sorted(os.sched_getaffinity(0))
    cpus = affinity[len(affinity)*args.lane//args.lanes:len(affinity)*(args.lane+1)//args.lanes]
    os.sched_setaffinity(0, cpus)
    args.slot = args.slot_base + args.lane
    args.worker_index = args.lane
    args.phase = 'full'
    args.startup_timeout = 600
    args.no_progress_timeout = 300
    args.route_wall_timeout = 5400
    args.num_workers = args.lanes
    args._active_evaluator = None
    owner = None
    worker = f'{args.group_id}/lane{args.lane}'
    queue = RouteQueue(args.run_root)
    original_build_plan = fast.build_plan

    def build_plan(configuration):
        configure_policy(configuration, args.method)
        plan = original_build_plan(configuration)
        plan['environment']['WMMPC_CARLA_STARTUP_WAIT'] = args.startup_wait
        return plan

    fast.build_plan = build_plan

    def terminate(signum, frame):
        active = args._active_evaluator
        if active is not None:
            stop_evaluator(active)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    last_town = None
    sequence = 0
    server_uses = 0
    try:
        while True:
            work = queue.claim(worker, last_town)
            if work is None:
                break
            route = work['id']
            route_root = args.run_root / 'routes' / f'route_{route}'
            route_root.mkdir(parents=True, exist_ok=True)
            attempt = route_root / f'attempt_{work["attempt"]:02d}'
            save(args.output_root / 'claim.json', {**work, 'owner': worker, 'unix': time.time()})
            if args.server_mode == 'warm' and (owner is None or server_uses >= args.routes_per_server):
                if owner is not None:
                    owner.close()
                owner = None
                try:
                    owner, sequence = start_lane_server(args, args.output_root / 'servers', sequence)
                except (OSError, RuntimeError, TimeoutError) as exc:
                    sequence += 3
                    result = {'route_id': route, 'classification': 'infrastructure_error',
                              'supervisor_reason': 'carla_server_startup', 'error': repr(exc),
                              'attempt': str(attempt), 'inference_steps': 100, 'queue_worker': worker}
                    save(route_root / 'summary.json', result)
                    queue.finish(route, worker, result)
                    continue
                server_uses = 0
            args._external_server = owner.process if owner is not None else None
            # Separate per-lane and per-route TM endpoints, including retries.
            args._traffic_manager_port = args.tm_base + args.lane*1024 + work['serial']
            result = fast.batch_trial(args, [route], attempt, args.gpu_uuid)[route]
            result['queue_worker'] = worker
            save(route_root / 'summary.json', result)
            queue.finish(route, worker, result)
            last_town = work['town']
            server_uses += 1
            if result['classification'] == 'infrastructure_error' and owner is not None:
                owner.close()
                owner = None
            if result['classification'] in ('agent_error', 'evaluation_error'):
                raise RuntimeError(f'Inspect route {route}: {result["classification"]}')
        save(args.output_root / 'completed.json', {'unix': time.time(), 'server_launches': sequence})
    finally:
        if owner is not None:
            owner.close()


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def parent(args):
    check_job(args.job_id)
    args.run_root = checked_run_root(args.run_root)
    args.checkpoint = checked_evaluation_path(args.checkpoint)
    assert args.job_id in Path('/proc/self/cgroup').read_text()
    assert Path(sys.prefix).resolve() == (ROOT / '.venvs/full-diffusion-carla').resolve()
    prepared_hashes = read(args.run_root / 'source_sha256.json')
    if prepared_hashes:
        changed = [name for name, digest in prepared_hashes.items()
                   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
        if changed:
            raise RuntimeError(f'Prepared source snapshot differs from active code: {changed}')
    try:
        details = subprocess.check_output(['bjobs', '-l', args.job_id], text=True, stderr=subprocess.PIPE, timeout=15)
        limit = float(re.search(r'RUNLIMIT\s*\n\s*([\d.]+)', details)[1]) * 60
        match = re.search(r'(?m)^(\w{3} \w{3}\s+\d+ [\d:]+)(?: (\d{4}))?: Started', details)
        year = int(match[2]) if match[2] else datetime.now().year
        started_dt = datetime.strptime(f'{match[1]} {year}', '%a %b %d %H:%M:%S %Y')
        if not match[2] and started_dt.timestamp() > time.time() + 60:
            started_dt = started_dt.replace(year=year-1)
        started = started_dt.timestamp()
        remaining = started + limit - time.time()
    except subprocess.CalledProcessError:
        # Compute-node tmux shells can lack the login node's LSF query context.
        # Accept only a freshly collected, explicit login-node allocation record.
        if args.allocation_record is None:
            raise
        proof = read(args.allocation_record)
        assert proof['job_id'] == args.job_id and proof['state'] == 'RUN'
        assert proof['host'] == socket.gethostname().split('.')[0]
        assert 0 <= time.time() - proof['checked_unix'] < 900
        assert proof['mem_per_slot_mb'] == 16000 and proof['slots'] == 16
        remaining = proof['deadline_unix'] - time.time()
    if remaining < args.wall_budget + 900:
        raise RuntimeError('Insufficient LSF run time left for this bounded experiment')
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) >= 8
    cpu_group = select_gpu_cpus(affinity, args.gpu, args.cpus_per_gpu)
    os.sched_setaffinity(0, cpu_group)
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(uuids) == 2
    if args.allocation_record is not None:
        assert read(args.allocation_record)['gpu_uuids'] == uuids
    args.gpu_uuid = uuids[args.gpu]
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    if args.gpu_uuid in occupied:
        raise RuntimeError('Selected allocation-visible GPU is occupied')
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if int(used[args.gpu]) >= 1024:
        raise RuntimeError('GPU has graphics or other memory allocations; inspect before launch')
    sample = resources()
    host_limit = sample.get('job_memory.max', 'max')
    if host_limit == 'max' or int(host_limit) < 64*1024**3:
        raise RuntimeError('A verified job cgroup memory limit of at least 64 GiB is required')
    current = int(sample['job_memory.current'])
    if current + 40*1024**3 > .85 * int(host_limit):
        raise RuntimeError('Insufficient cgroup RAM headroom for a 40 GiB node peak')
    assert (args.carla_root / 'runtime_ready.json').is_file()
    args.group_id = args.group_id or f'{args.job_id}_gpu{args.gpu}'
    group = args.run_root / 'groups' / args.group_id
    group.mkdir(parents=True, exist_ok=False)
    routes = [dict(id=r.attrib['id'], town=r.attrib['town']) for r in
              ET.parse(ROOT / 'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    requested = args.route_ids.split(',') if args.route_ids != 'all' else [r['id'] for r in routes]
    assert len(requested) == len(set(requested)) and set(requested) <= {r['id'] for r in routes}
    routes = sorted([r for r in routes if r['id'] in requested], key=lambda r: (r['town'], int(r['id'])))
    sources = ('method/full_diffusion/diffusion.py', 'method/full_diffusion/model.py',
               'method/full_diffusion/garage.py', 'method/full_diffusion/inference/policy.py',
               'method/full_diffusion/inference/carla_agent.py', 'third_party/carla_garage/team_code/sensor_agent.py',
               'evaluation/run_carla_parallel.py', 'evaluation/carla_route_queue.py',
               'evaluation/shared_gpu_inference.py', 'evaluation/full_diffusion_rpc_agent.py',
               'evaluation/full_diffusion_fast_agent.py', 'evaluation/full_diffusion_diagnostic_agent.py',
               'evaluation/full_diffusion_parallel_direct_agent.py', 'evaluation/carla_fast_evaluator.py',
               'evaluation/run_full_diffusion_carla_fast.py', 'evaluation/carla_lifecycle.py',
               'evaluation/fast_carla_io.py', 'leaderboard/data/bench2drive220.xml',
               'evaluation/run_full_diffusion2_carla.py', 'evaluation/full_diffusion2_fast_agent.py',
               'evaluation/carla_eval_resources.py', 'method/full_diffusion2/__init__.py',
               'method/full_diffusion2/inference/carla_agent.py', 'method/full_diffusion2/inference/policy.py',
               'method/full_diffusion2/model.py', 'method/full_diffusion2/perception.py',
               'method/full_diffusion2/data.py', 'method/full_diffusion2/snapshot.py')
    if args.method == 'full_diffusionv3':
        sources += ('method/full_diffusionv3/__init__.py', 'method/full_diffusionv3/model.py',
                    'method/full_diffusionv3/diffusion.py', 'method/full_diffusionv3/inference/policy.py',
                    'method/full_diffusionv3/inference/carla_agent.py',
                    'evaluation/full_diffusionv3_fast_agent.py')
    identity = {'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'seed': args.seed, 'inference_steps': 100, 'backend': args.backend, 'method': args.method,
                'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    if prepared_hashes:
        identity['prepared_source_sha256'] = prepared_hashes
    queue = RouteQueue(args.run_root)
    queue.initialize(routes, identity)
    manifest = {'job_id': args.job_id, 'host': socket.gethostname(), 'gpu_uuid': args.gpu_uuid,
                'logical_gpu': args.gpu, 'lanes': args.lanes, 'cpu_affinity': cpu_group,
                'server_mode': args.server_mode, 'startup_wait': args.startup_wait,
                'allocation_cpu_affinity': affinity, 'initial_cgroup_memory': sample,
                'lsf_remaining_seconds': remaining, 'lsf_memory_per_slot_mb': 16000,
                'lsf_slots': 16, 'expected_peak_host_memory_gib': 40,
                'soft_host_memory_limit_gib': 64, 'job_memory_limit_bytes': int(host_limit),
                'routes_per_server': args.routes_per_server, 'capture_first_policy_call': args.capture_first_policy_call,
                'identity': identity, 'unix': time.time()}
    save(group / 'manifest.json', manifest)
    children = []
    streams = []
    began = time.time()
    locks = evaluation_runtime_root() / 'carla_gpu_locks'
    locks.mkdir(parents=True, exist_ok=True)
    stopped = None
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as stack:
            for directory in dict.fromkeys((locks, ROOT / '.runtime/carla_gpu_locks',
                                           ROOT / 'results/full_diffusion_carla_20261031_epoch10/locks')):
                lock_path = directory / f'{args.gpu_uuid}.lock'
                if directory != locks and not lock_path.exists():
                    continue
                mode = 'a' if directory == locks else 'r'
                fcntl.flock(stack.enter_context(lock_path.open(mode)), fcntl.LOCK_EX | fcntl.LOCK_NB)
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                               PYTHONDONTWRITEBYTECODE='1')
            if args.backend == 'rpc':
                broker_command = [sys.executable, '-u', str(ROOT / 'evaluation/shared_gpu_inference.py'),
                                  '--directory', str(group/'broker'), '--checkpoint', str(args.checkpoint),
                                  '--max-sessions', str(args.lanes)]
                stream = (group / 'broker.log').open('wb'); streams.append(stream)
                broker = subprocess.Popen(broker_command, env=environment, cwd=ROOT, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                children.append(broker)
                deadline = time.time() + 60
                while not (group/'broker/ready.json').exists():
                    if broker.poll() is not None or time.time() > deadline:
                        raise RuntimeError('GPU policy broker did not become ready')
                    time.sleep(.2)
            lanes = []
            for index in range(args.lanes):
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--role', 'lane',
                           '--job-id', args.job_id, '--run-root', str(args.run_root), '--gpu', str(args.gpu),
                           '--gpu-uuid', args.gpu_uuid, '--group-id', args.group_id, '--lane', str(index),
                           '--lanes', str(args.lanes), '--backend', args.backend, '--checkpoint', str(args.checkpoint),
                           '--method', args.method,
                           '--carla-root', str(args.carla_root), '--slot-base', str(args.slot_base),
                           '--tm-base', str(args.tm_base), '--seed', str(args.seed),
                           '--routes-per-server', str(args.routes_per_server),
                           '--server-mode', args.server_mode, '--startup-wait', args.startup_wait,
                           '--cpus-per-gpu', str(args.cpus_per_gpu)]
                if args.capture_first_policy_call:
                    command.append('--capture-first-policy-call')
                stream = (group / f'lane{index}.controller.log').open('wb'); streams.append(stream)
                process = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                children.append(process); lanes.append(process)
            save(group/'processes.json', {'parent_pid': os.getpid(), 'children': [p.pid for p in children]})
            while any(p.poll() is None for p in lanes):
                sample = resources()
                with (group / 'resources.jsonl').open('a') as handle:
                    handle.write(json.dumps(sample) + '\n')
                memory = int(sample.get('job_memory.current', sample.get('memory.current', '0')))
                noncache = int(sample['job_memory.noncache'])
                if noncache > 64*1024**3:
                    stopped = 'soft_host_noncache_memory_limit'
                memory_limit = sample.get('job_memory.max', 'max')
                if memory_limit != 'max' and memory > .85 * int(memory_limit):
                    stopped = 'job_memory_headroom'
                for gpu_line in sample.get('gpu', '').splitlines():
                    if args.gpu_uuid in gpu_line and int(re.search(r'(\d+) MiB', gpu_line)[1]) > 60*1024:
                        stopped = 'soft_gpu_memory_limit'
                if time.time() - began > args.wall_budget:
                    stopped = 'experiment_wall_budget'
                if args.backend == 'rpc' and broker.poll() is not None:
                    stopped = 'policy_broker_exit'
                if any(p.poll() not in (None, 0) for p in lanes):
                    stopped = 'lane_controller_error'
                if stopped:
                    raise RuntimeError(stopped)
                time.sleep(5)
            if any(p.returncode != 0 for p in lanes):
                stopped = 'lane_controller_error'
                raise RuntimeError(stopped)
            state = queue.snapshot()
            records = [v['last_result']['record'] for v in state['items'].values() if v['state'] == 'done']
            done = {'elapsed_seconds': time.time()-began, 'routes': len(state['items']), 'scored': len(records),
                    'states': {r: v['state'] for r, v in state['items'].items()}, 'lanes': args.lanes,
                    'DS': sum(r['scores']['score_composed'] for r in records)/len(records) if records else None,
                    'RC': sum(r['scores']['score_route'] for r in records)/len(records) if records else None,
                    'backend': args.backend, 'unix': time.time(), 'inference_steps': 100}
            save(group/'completed.json', done)
            print(json.dumps(done), flush=True)
    finally:
        # First stop lane supervisors, which stop only their own evaluators and
        # renderers. The shared CUDA process is kept alive until that completes.
        for process in reversed(children):
            stop_owned(process)
        for stream in streams:
            stream.close()
        if stopped:
            save(group/'stopped.json', {'reason': stopped, 'unix': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('parent', 'lane'), default='parent')
    parser.add_argument('--method', choices=('full_diffusion2', 'full_diffusionv3'), default='full_diffusion2')
    parser.add_argument('--cpus-per-gpu', type=int, default=4)
    parser.add_argument('--server-mode', choices=('cold', 'warm'), default='warm')
    parser.add_argument('--startup-wait', choices=('fixed30', 'rpc'), default='rpc')
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path, default=ROOT/'.runtime/carla_0.9.15_v1')
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    parser.add_argument('--gpu-uuid', default='')
    parser.add_argument('--lanes', type=int, choices=(1, 2), default=1)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--backend', choices=('direct', 'rpc'), default='direct')
    parser.add_argument('--group-id')
    parser.add_argument('--route-ids', default='1711,1773,3464,3865')
    parser.add_argument('--slot-base', type=int, default=2700)
    parser.add_argument('--tm-base', type=int, default=47000)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--wall-budget', type=int, default=3600)
    parser.add_argument('--allocation-record', type=Path)
    parser.add_argument('--routes-per-server', type=int, default=32)
    parser.add_argument('--capture-first-policy-call', action='store_true')
    args = parser.parse_args()
    for name in ('run_root', 'checkpoint', 'carla_root'):
        path = checked_run_root(getattr(args, name)) if name == 'run_root' else checked_evaluation_path(getattr(args, name))
        setattr(args, name, path)
    if args.lanes != 1 or args.backend != 'direct':
        raise ValueError('FD2 evaluation requires one direct CUDA agent and one CARLA per GPU')
    if args.server_mode == 'warm' and args.startup_wait != 'rpc':
        raise ValueError('Persistent servers use RPC readiness')
    if args.lanes == 2 and args.backend != 'rpc':
        raise ValueError('The allocation requires one shared CUDA process for two lanes')
    if not 1 <= args.routes_per_server <= 32:
        raise ValueError('routes-per-server must be between one and 32')
    if args.role == 'lane':
        lane(args)
    else:
        parent(args)
