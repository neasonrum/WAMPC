"""Switch one completed-route GPU worker to two isolated RPC lanes."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.pointquery_parallel.common import ROOT, OUTPUT, RUN, ARTIFACTS, JOBS, read, save
from evaluation.pointquery_parallel.drain import drain
from method.full_diffusion_pointquery.utils import inspect_allocation, sha
from method.full_diffusionv3_ego12.allocation import query_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=JOBS)
    parser.add_argument('gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    lock = (ARTIFACTS / f'.launch_{args.job}_{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(OUTPUT / 'config.json')
    entry = config['jobs'][args.job]
    inspect_allocation(config, args.job)
    drain(args.job, args.gpu)
    deadline = time.time() + 1800
    while not (ARTIFACTS / 'preflight.json').exists():
        if (ARTIFACTS / 'preflight_failed.json').exists() or time.time() > deadline:
            raise RuntimeError('Shared-inference preflight has not passed')
        time.sleep(2)
    acceptance = read(ARTIFACTS / 'preflight.json')
    ready = read(RUN / 'ready.json')
    if (not acceptance['passed'] or not acceptance.get('GPU_detection_decoding_exact')
            or acceptance['checkpoint_sha256'] != ready['checkpoint_sha256']):
        raise RuntimeError('Equivalence acceptance and benchmark checkpoint differ')
    hashes = read(RUN / 'source_sha256.json')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise RuntimeError('The original planner/evaluation sources changed')
    proof = {**query_allocation(args.job, entry['host']), **inspect_allocation(config, args.job),
             'gpu_uuids': entry['gpu_uuids'], 'checked_unix': time.time()}
    record = ARTIFACTS / f'allocation_{args.job}_{args.gpu}.json'
    save(record, proof)
    budget = min(10 * 3600, int(proof['deadline_unix'] - time.time() - 1800))
    if budget < 1800:
        raise RuntimeError('Insufficient allocation time')
    index = entry['node_rank'] * 2 + args.gpu
    group = f'{args.job}_gpu{args.gpu}_parallel14v1'
    command = [sys.executable, '-u', '-m', 'evaluation.pointquery_parallel.controller',
               '--method', 'full_diffusion_pointquery', '--job-id', args.job, '--run-root', str(RUN),
               '--checkpoint', str(RUN / 'selected_best.pt'), '--gpu', str(args.gpu), '--group-id', group,
               '--route-ids', 'all', '--slot-base', str(3500 + index * 80),
               '--tm-base', str(30000 + args.gpu * 4096), '--seed', '901', '--backend', 'rpc', '--lanes', '2',
               '--cpus-per-gpu', '8', '--server-mode', 'warm', '--startup-wait', 'rpc', '--routes-per-server', '32',
               '--wall-budget', str(budget), '--allocation-record', str(record)]
    log = ARTIFACTS / f'{group}.controller.log'
    save(ARTIFACTS / f'launch_{args.job}_{args.gpu}.json', {'state': 'launching', 'command': command,
              'log': str(log), 'expected_lanes': 2, 'checkpoint_sha256': ready['checkpoint_sha256'], 'unix': time.time()})
    with log.open('xb') as handle:
        status = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    save(ARTIFACTS / f'exit_{args.job}_{args.gpu}.json', {'return_code': status, 'unix': time.time()})
    if status:
        raise RuntimeError(f'Dual-lane controller returned {status}; inspect its logs')


if __name__ == '__main__':
    main()

