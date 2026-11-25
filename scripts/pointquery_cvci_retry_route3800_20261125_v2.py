"""Retry only unscored route 3800 once more on one existing A100."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.carla_route_queue import RouteQueue
from evaluation.summarize_parallel_carla import summarize
from method.full_diffusion_pointquery.utils import atomic_json, inspect_allocation, sha


def scored_digest(items):
    frozen = {rid: value for rid, value in items.items() if value['state'] == 'done'}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    config_path, job = Path(sys.argv[1]), sys.argv[2]
    config = json.loads(config_path.read_text())
    run = Path(config['output']) / 'carla25'
    recovery = run / 'retry_route3800_20261125_v2'
    lock = (recovery / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (recovery / 'started.json').exists():
        raise RuntimeError('This recovery was already started; inspect its recorded processes')
    proof = inspect_allocation(config, job, idle=True)
    scheduler = json.loads((recovery / 'scheduler.json').read_text())
    if scheduler['job_id'] != job or time.time() - scheduler['checked_unix'] > 900:
        raise RuntimeError('Fresh, matching scheduler proof is required')
    proof.update(scheduler)
    proof.update(gpu_uuids=config['jobs'][job]['gpu_uuids'], checked_unix=time.time())
    proof_path = recovery / 'allocation.json'
    atomic_json(proof_path, proof)
    selected = json.loads((run / 'ready.json').read_text())
    checkpoint = Path(selected['checkpoint'])
    if selected['selected_epoch'] != 10 or selected['inference_steps'] != 25 or sha(checkpoint) != selected['checkpoint_sha256']:
        raise RuntimeError('Selected epoch10 and 25 DDIM checkpoint changed')
    for name, digest in json.loads((run / 'source_sha256.json').read_text()).items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Previously verified evaluation source changed: {name}')
    targets = {'3800'}
    queue = RouteQueue(run)
    with queue.locked() as state:
        if any(value['state'] in ('pending', 'claimed') for value in state['items'].values()):
            raise RuntimeError('Another evaluation worker may still own routes')
        missing = {rid for rid, value in state['items'].items() if value['state'] != 'done'}
        if missing != targets or sum(value['state'] == 'done' for value in state['items'].values()) != 219:
            raise RuntimeError(f'Expected exactly 219 scores and route 3800 missing: {missing}')
        if state['identity']['checkpoint_sha256'] != selected['checkpoint_sha256'] or state['identity']['seed'] != 901:
            raise RuntimeError('The original checkpoint or evaluation seed differs')
        for rid in targets:
            value = state['items'][rid]
            if value['state'] != 'error' or value['last_result']['classification'] != 'infrastructure_error' or value['last_result']['record'] is not None:
                raise RuntimeError('Only unscored infrastructure errors are eligible')
        atomic_json(recovery / 'queue_before.json', state)
        original_scored_digest = scored_digest(state['items'])
        for rid in targets:
            # Preserve the original attempts/history. This authorizes one new
            # attempt, numbered 4; queue.finish will not auto-retry it again.
            state['items'][rid]['state'] = 'pending'
        state.setdefault('authorized_recoveries', []).append({'id': recovery.name, 'routes': sorted(targets),
            'reason': 'user requested another retry of the final unscored route 3800',
            'max_new_attempts_per_route': 1, 'unix': time.time()})
        queue.save(state)
    env = {**os.environ, 'WMMPC_CARLA_RESULTS_ROOT': str(run.parent.parent), 'SLOWER': '0'}
    os.environ['WMMPC_CARLA_RESULTS_ROOT'] = str(run.parent.parent)
    processes, logs = [], []
    try:
        for gpu in (0,):
            group = f'{job}_gpu{gpu}_3800retry2'
            log = recovery / f'gpu{gpu}.controller.log'
            handle = log.open('xb')
            logs.append(handle)
            command = [sys.executable, '-u', '-m', 'method.full_diffusion_pointquery_cvci.eval_controller',
                '--method', 'full_diffusion_pointquery_cvci', '--job-id', job, '--run-root', str(run),
                '--checkpoint', str(checkpoint), '--gpu', str(gpu), '--group-id', group,
                '--route-ids', 'all', '--slot-base', str(5300 + gpu * 100), '--tm-base', str(40000 + gpu * 2000),
                '--seed', '901', '--backend', 'direct', '--lanes', '1', '--cpus-per-gpu', '8',
                '--server-mode', 'warm', '--startup-wait', 'rpc', '--routes-per-server', '1',
                '--wall-budget', '7200', '--allocation-record', str(proof_path)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
        atomic_json(recovery / 'started.json', {'job': job, 'targets': sorted(targets), 'gpus': 1,
            'pids': [p.pid for p in processes], 'checkpoint': selected, 'original_scored_digest': original_scored_digest,
            'new_attempts_per_route': 1, 'unix': time.time()})
        print('RETRY_STARTED ' + json.dumps({'pids': [p.pid for p in processes], 'routes': sorted(targets)}), flush=True)
        while any(process.poll() is None for process in processes):
            atomic_json(run / 'summary.json', summarize(run))
            atomic_json(recovery / 'status.json', {'return_codes': [p.poll() for p in processes],
                'routes': {rid: queue.snapshot()['items'][rid]['state'] for rid in targets}, 'unix': time.time()})
            time.sleep(15)
        codes = [p.wait() for p in processes]
    finally:
        for handle in logs:
            handle.close()
    final_state = queue.snapshot()
    previous = json.loads((recovery / 'queue_before.json').read_text())
    preserved = all(final_state['items'][rid] == value for rid, value in previous['items'].items() if value['state'] == 'done')
    if not preserved:
        raise RuntimeError('A previously scored route record changed during recovery')
    final = summarize(run, extended=True, merge=True)
    atomic_json(run / 'summary.json', final)
    atomic_json(recovery / 'completed.json', {'return_codes': codes, 'previous_219_records_unchanged': preserved,
        'valid_driving_results': final['valid_driving_results'], 'missing_route_ids': final['missing_route_ids'],
        'metrics': final['metrics_observed_routes'], 'complete_220_routes': final['complete_220_routes'], 'unix': time.time()})
    print('RETRY_FINISHED ' + json.dumps(final), flush=True)
    if any(codes):
        raise RuntimeError(f'Recovery controllers exited with {codes}; preserve logs and allocation')


if __name__ == '__main__':
    main()
