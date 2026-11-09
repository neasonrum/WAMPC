"""Launch the full FD2 route queue on two GPUs in an existing LSF allocation."""
import argparse
import fcntl
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
from evaluation.carla_eval_resources import checked_run_root, evaluation_runtime_root, resources, select_gpu_cpus


def selected_route_argument(topology):
    routes = [str(route) for route in topology['route_ids']]
    if not routes or len(routes) > 220 or len(routes) != len(set(routes)):
        raise ValueError('Prepared routes must contain one to 220 unique IDs')
    if topology['route_count'] != len(routes):
        raise ValueError('Prepared route count does not match route IDs')
    return ','.join(routes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--wall-budget', type=int, default=86400)
    args = parser.parse_args()
    check_job(args.job_id)
    assert Path(sys.prefix).resolve() == (ROOT/'.venvs/full-diffusion-carla').resolve()
    run = checked_run_root(args.run_root)
    top = json.loads((run/'launch_topology.json').read_text())
    assert top['method'] in ('full_diffusion2', 'full_diffusionv3') and top['inference_steps'] == 100
    route_argument = selected_route_argument(top)
    assert top['lanes'] == 1 and top['backend'] == 'direct'
    with (run/f'launch_{args.job_id}.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run/f'node_launch_{args.job_id}.json').exists():
            raise RuntimeError('This allocation was already launched; inspect its logs before recovery')
        expected = json.loads((run/'source_sha256.json').read_text())
        changed = [n for n,h in expected.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != h]
        assert not changed, f'Prepared source files changed: {changed}'
        checkpoint = Path(top['checkpoint']).resolve()
        checkpoint.relative_to(run/'checkpoint')
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == top['checkpoint_sha256']
        workers = sorted([w for w in top['workers'] if w['job_id'] == args.job_id], key=lambda w: w['gpu'])
        assert len(workers) == 2 and [w['gpu'] for w in workers] == [0, 1]
        assert all(w['host'] == socket.gethostname().split('.')[0] for w in workers)
        uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()
        assert uuids == [w['gpu_uuid'] for w in workers]
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
        assert not occupied, f'Allocation GPUs have active compute processes: {occupied}'
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        assert all(int(v) < 1024 for v in memory), 'GPU graphics memory is occupied'
        proof = run/'allocations'/f'allocation_{args.job_id}.json'
        evidence = json.loads(proof.read_text())
        assert evidence['job_id'] == args.job_id and evidence['state'] == 'RUN'
        assert evidence['host'] == workers[0]['host'] and evidence['gpu_uuids'] == uuids
        assert 0 <= time.time()-evidence['checked_unix'] < 900
        assert evidence['deadline_unix']-time.time() > args.wall_budget+900
        assert evidence['mem_per_slot_mb'] == 16000 and evidence['slots'] == 16
        sample = resources()
        assert int(sample['job_memory.current'])+40*1024**3 < .85*int(sample['job_memory.max'])
        affinity = sorted(os.sched_getaffinity(0))
        plans = []
        tag = hashlib.sha256(str(run).encode()).hexdigest()[:6]
        prefix = 'fd3full' if top['method'] == 'full_diffusionv3' else 'fd2full'
        socket_path = evaluation_runtime_root()/'ipc'/f'{prefix}_{args.job_id}.sock'
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        sock = str(socket_path.relative_to(ROOT)) if socket_path.is_relative_to(ROOT) else str(socket_path)
        assert len(os.fsencode(sock)) < 108, 'Configured tmux socket path exceeds the UNIX socket limit'
        for w in workers:
            group = f'{args.job_id}_gpu{w["gpu"]}'
            assert not (run/'groups'/group).exists(), f'Group already exists: {group}'
            cpu_count = len(affinity)//2
            cpus = select_gpu_cpus(affinity, w['gpu'], cpu_count)
            # Startup reserves one free block from this worker's own 64 slots.
            free_blocks = []
            for offset in range(64):
                port = 2000+(w['slot_base']+offset)*10
                try:
                    for endpoint in range(port,port+4):
                        with socket.socket() as probe:
                            probe.bind(('0.0.0.0',endpoint))
                    free_blocks.append(port)
                except OSError:
                    continue
            assert free_blocks, f'No available CARLA ports for {group}'
            command = [sys.executable, '-u', str(ROOT/'evaluation/run_full_diffusion2_carla.py'),
                       '--job-id', args.job_id, '--run-root', str(run), '--gpu', str(w['gpu']),
                       '--method', top['method'],
                       '--checkpoint', str(checkpoint), '--backend', 'direct', '--lanes', '1',
                       '--route-ids', route_argument, '--group-id', group, '--slot-base', str(w['slot_base']),
                       '--tm-base', str(w['tm_base']), '--seed', str(top['seed']),
                       '--cpus-per-gpu', str(cpu_count), '--server-mode', 'warm', '--startup-wait', 'rpc',
                       '--routes-per-server', '32', '--wall-budget', str(args.wall_budget),
                       '--allocation-record', str(proof)]
            if top.get('bridge_modules'):
                command = ['env', f'WMMPC_BRIDGE_MODULES={top["bridge_modules"]}', *command]
            plans.append({'group': group, 'command': command, 'cpu_affinity': cpus,
                          'session': f'{prefix}_{tag}_gpu{w["gpu"]}', 'log': str(run/f'{group}.controller.log'),
                          'first_free_rpc_port': free_blocks[0]})
        assert not set(plans[0]['cpu_affinity']) & set(plans[1]['cpu_affinity'])
        metadata = {'job_id': args.job_id, 'host': socket.gethostname(), 'unix': time.time(),
                    'socket': sock, 'initial_resources': sample, 'launches': plans, 'lsf_modified': False,
                    'inference_steps': 100, 'route_count': len(top['route_ids']), 'launched_sessions': []}
        for plan in plans:
            shell = 'exec '+shlex.join(plan['command'])+' > '+shlex.quote(plan['log'])+' 2>&1'
            subprocess.run(['tmux', '-S', sock, 'new-session', '-d', '-s', plan['session'], shell], cwd=ROOT, check=True)
            metadata['launched_sessions'].append(plan['session'])
            (run/f'node_launch_{args.job_id}.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({'job_id': args.job_id, 'host': socket.gethostname(), 'socket': sock,
                          'workers': [{k:p[k] for k in ['group','cpu_affinity','session','first_free_rpc_port']} for p in plans],
                          'memory_GiB': round(int(sample['job_memory.current'])/2**30,2)},indent=2))


if __name__ == '__main__':
    main()
