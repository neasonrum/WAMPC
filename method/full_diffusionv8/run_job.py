"""Run V8 in an existing LSF allocation, then evaluate its validation-selected model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .allocation import inspect
from .prepare import ROOT, atomic_json, sha


def wait_file(path, failure=None, timeout=1800):
    deadline = time.time() + timeout
    while not path.is_file():
        if failure is not None and failure.is_file():
            raise RuntimeError(f'Another node reported setup failure in {failure}')
        if time.time() >= deadline:
            raise TimeoutError(f'Timed out waiting for {path}')
        time.sleep(5)


def prepare_evaluation(config, output):
    directory = output / 'carla25'
    if (directory / 'ready.json').is_file():
        return
    completed = json.loads((output / 'completed.json').read_text())
    best = json.loads((output / 'best.json').read_text())
    if completed['best'] != best or sha(output / 'best.pt') != best['checkpoint_sha256']:
        raise RuntimeError('The selected checkpoint and normal-completion marker disagree')
    directory.mkdir(exist_ok=True)
    (directory / 'allocations').mkdir(exist_ok=True)
    private = directory / 'selected_best.pt'
    shutil.copy2(output / 'best.pt', private)
    sources = list((ROOT / 'method/full_diffusionv8').glob('*.py'))
    for folder in ('method/full_diffusion', 'method/full_diffusion2', 'method/full_diffusionv3',
                   'method/full_diffusionv3_ego12'):
        sources.extend((ROOT / folder).rglob('*.py'))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    atomic_json(directory / 'source_sha256.json', hashes)
    atomic_json(directory / 'ready.json', {'checkpoint': str(private), 'checkpoint_sha256': sha(private),
                'selected_epoch': best['epoch'], 'validation': best['validation'],
                'routes': 220, 'inference_steps': 25, 'unix': time.time()})


def evaluate(config, output, job):
    entry = config['jobs'][job]
    directory = output / 'carla25'
    if entry['node_rank'] == 0:
        prepare_evaluation(config, output)
    wait_file(directory / 'ready.json', timeout=1800)
    proof = inspect(job, config)
    record = directory / 'allocations' / f'allocation_{job}.json'
    atomic_json(record, proof)
    budget = min(16 * 3600, int(proof['deadline_unix'] - time.time() - 1800))
    if budget < 1800:
        raise RuntimeError('Less than 30 minutes remains for closed-loop evaluation')
    env = {**os.environ, 'WMMPC_CARLA_RESULTS_ROOT': str(output.parent), 'SLOWER': '0'}
    os.environ['WMMPC_CARLA_RESULTS_ROOT'] = str(output.parent)
    processes, logs = [], []
    try:
        for gpu in (0, 1):
            index = entry['node_rank'] * 2 + gpu
            group = f'{job}_gpu{gpu}'
            handle = (directory / f'{group}.controller.log').open('xb')
            logs.append(handle)
            command = [sys.executable, '-u', '-m', 'method.full_diffusionv8.eval_controller',
                       '--method', 'full_diffusionv8', '--job-id', job, '--run-root', str(directory),
                       '--checkpoint', str(directory / 'selected_best.pt'), '--gpu', str(gpu), '--group-id', group,
                       '--route-ids', 'all', '--slot-base', str(4100 + index * 80), '--tm-base', str(33000 + index * 600),
                       '--seed', '901', '--backend', 'direct', '--lanes', '1', '--cpus-per-gpu', '8',
                       '--server-mode', 'warm', '--startup-wait', 'rpc', '--routes-per-server', '32',
                       '--wall-budget', str(budget), '--allocation-record', str(record)]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        atomic_json(directory / f'launch_{job}.json', {'job': job, 'pids': [p.pid for p in processes],
                    'selected_best': json.loads((directory / 'ready.json').read_text()), 'unix': time.time()})
        if entry['node_rank'] == 0:
            from evaluation.summarize_parallel_carla import summarize
            while any(process.poll() is None for process in processes):
                if (directory / 'route_queue.json').is_file():
                    try:
                        summary = summarize(directory)
                        atomic_json(directory / 'summary.json', summary)
                    except (OSError, ValueError, KeyError) as error:
                        print(f'Summary pending {error!r}', flush=True)
                time.sleep(30)
        codes = [process.wait() for process in processes]
    finally:
        for handle in logs:
            handle.close()
    atomic_json(directory / f'job_{job}.json', {'return_codes': codes, 'unix': time.time()})
    if entry['node_rank'] == 0:
        from evaluation.summarize_parallel_carla import summarize
        deadline = time.time() + 1800
        while not all((directory / f'job_{other}.json').is_file() for other in config['jobs']):
            if time.time() >= deadline:
                break
            time.sleep(30)
        atomic_json(directory / 'summary.json', summarize(directory, extended=True, merge=True))
    if any(codes):
        raise RuntimeError(f'Closed-loop controllers ended with return codes {codes}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config_joint.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    entry = config['jobs'][args.job]
    lock = (output / f'.job_{args.job}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = inspect(args.job, config)
    atomic_json(output / f'allocation_{args.job}.json', proof)
    if not (output / 'index/manifest.json').is_file():
        raise RuntimeError('Prepare the fixed training and 44-route validation index first')
    if entry['node_rank'] == 0 and not (output / 'preflight.json').is_file():
        command = [sys.executable, '-u', '-m', 'method.full_diffusionv8.preflight',
                   '--config', str(args.config), '--job-id', args.job]
        with (output / 'preflight.log').open('ab') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(output / 'preflight_failed.json', {'return_code': result.returncode, 'unix': time.time()})
            raise RuntimeError('V8 preflight failed, see preflight.log')
    wait_file(output / 'preflight.json', output / 'preflight_failed.json')
    report = json.loads((output / 'preflight.json').read_text())
    if report['status'] != 'PASS' or report['planner_tensors_receiving_world_loss'] != 0:
        raise RuntimeError('The V8 gradient-ownership preflight did not pass')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--nnodes', '4', '--nproc-per-node', '2',
               '--node-rank', str(entry['node_rank']), '--master-addr', config['master_address'],
               '--master-port', str(config['master_port']), '--max-restarts', '2',
               '-m', 'method.full_diffusionv8.train', '--job-id', args.job, '--config', str(args.config)]
    if args.resume:
        command.append('--resume')
    atomic_json(output / f'node_{args.job}.json', {'state': 'training', 'command': command, 'unix': time.time()})
    with (output / f'train_{args.job}.log').open('ab') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state = 'training_complete' if result.returncode == 0 and (output / 'completed.json').is_file() else 'paused' if result.returncode == 0 else 'failed'
    atomic_json(output / f'node_{args.job}.json', {'state': state, 'return_code': result.returncode, 'unix': time.time()})
    if state == 'training_complete':
        atomic_json(output / f'node_{args.job}.json', {'state': 'closed_loop', 'unix': time.time()})
        evaluate(config, output, args.job)
    elif result.returncode:
        raise RuntimeError(f'Training torchrun ended with return code {result.returncode}')


if __name__ == '__main__':
    main()
