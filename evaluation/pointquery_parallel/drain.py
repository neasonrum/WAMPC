"""Stop an owned single-lane worker after a complete scored episode, preserving results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from evaluation.carla_route_queue import RouteQueue
from .common import ROOT, RUN, OUTPUT, ARTIFACTS, JOBS, live, owned, read, save


def completed(group, route):
    p = group / 'lane_0' / f'route_{route}' / 'summary.json'
    value = read(p)
    if not value or value.get('classification') != 'driving_result':
        return None
    if not value.get('record') or value['record']['meta'].get('duration_game', 0) <= 0:
        return None
    return value


def inferred(attempt):
    p = attempt / 'telemetry.jsonl'
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        if line.strip() and json.loads(line).get('network_seconds') is not None:
            return True
    return False


def stop_orphans(group, job):
    for path in (group / 'lane_0/servers').rglob('server_process.json'):
        value = read(path)
        pid = value['pid']
        if not owned(pid, job):
            continue
        cmd = (Path(f'/proc/{pid}') / 'cmdline').read_bytes().decode().split('\0')[:-1]
        if cmd != value['command'] or value['job_id'] != job:
            raise RuntimeError(f'Owned renderer identity changed: {pid}')
        save(ARTIFACTS / f'stop_renderer_{pid}.json', {'owner': value, 'reason': 'completed-route execution handoff', 'unix': time.time()})
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 25
        while live(pid) and time.time() < deadline:
            time.sleep(.2)
        if live(pid):
            raise RuntimeError(f'Renderer {pid} did not stop gracefully')


def drain(job, gpu):
    if job not in JOBS:
        raise ValueError('Only the three 16-CPU allocations switch to dual lane')
    if f'/job.{job}.' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('Enter the supplied LSF allocation with battach')
    config = read(OUTPUT / 'config.json')
    group_name = f'{job}_gpu{gpu}'
    group = RUN / 'groups' / group_name
    processes = read(group / 'processes.json')
    parent, lane = processes['parent_pid'], processes['children'][0]
    owner = group_name + '/lane0'
    marker = ARTIFACTS / f'drained_{job}_{gpu}.json'
    if marker.exists():
        return read(marker)
    if not owned(parent, job) or not owned(lane, job):
        raise RuntimeError('Original worker is already stopped; inspect before handoff')
    queue = RouteQueue(RUN)
    watched = None
    while True:
        snapshot = queue.snapshot()
        claims = [(rid, item) for rid, item in snapshot['items'].items()
                  if item['state'] == 'claimed' and item.get('owner') == owner]
        if watched is None and claims:
            watched = claims[0][0]
            save(ARTIFACTS / f'drain_wait_{job}_{gpu}.json', {'route': watched, 'parent': parent, 'lane': lane, 'unix': time.time()})
        result = completed(group, watched) if watched else None
        if result is None:
            if not live(parent):
                raise RuntimeError('Original worker exited before its watched route completed')
            time.sleep(.5)
            continue
        frozen = []
        with queue.locked() as state:
            claims = [(rid, item) for rid, item in state['items'].items()
                      if item['state'] == 'claimed' and item.get('owner') == owner]
            if len(claims) > 1:
                raise RuntimeError('Single-lane worker holds multiple active episodes')
            cancelled = None
            attempt = None
            if claims and claims[0][0] != watched:
                cancelled, item = claims[0]
                attempt = RUN / 'routes' / f'route_{cancelled}' / f'attempt_{item["attempts"]:02d}'
                # Never interrupt a rollout after its first actual model call.
                if inferred(attempt):
                    watched = cancelled
                    continue
                evaluator = read(attempt / 'evaluator_process.json', {}).get('pid')
                if evaluator and owned(evaluator, job):
                    os.kill(evaluator, signal.SIGSTOP)
                    frozen.append(evaluator)
                if inferred(attempt):
                    for pid in frozen:
                        os.kill(pid, signal.SIGCONT)
                    watched = cancelled
                    continue
            os.kill(lane, signal.SIGSTOP)
            frozen.append(lane)
            before = {rid: item for rid, item in state['items'].items() if item['state'] == 'done'}
            target = state['items'][watched]
            if target['state'] == 'claimed':
                record = {**result, 'queue_worker': owner}
                target['history'].append({'worker': owner, 'summary': record, 'unix': time.time()})
                target.update(state='done', last_result=record)
            elif target['state'] != 'done':
                raise RuntimeError('Completed route changed to an unexpected queue state')
            if cancelled:
                item = state['items'][cancelled]
                item['history'].append({'worker': owner, 'summary': {'classification': 'execution_handoff',
                    'supervisor_reason': 'dual_lane_switch_before_first_inference', 'attempt': str(attempt)}, 'unix': time.time()})
                item.update(state='pending')
                item.pop('owner', None)
                item.pop('claimed_unix', None)
            queue.save(state)
            assert all(state['items'][rid] == item for rid, item in before.items())
            # SIGTERM is pending before thaw: the old controllers exit through
            # their own shutdown handlers instead of claiming another route.
            for pid in frozen:
                if live(pid):
                    os.kill(pid, signal.SIGTERM)
            os.kill(parent, signal.SIGTERM)
            for pid in frozen:
                if live(pid):
                    os.kill(pid, signal.SIGCONT)
        deadline = time.time() + 50
        while any(live(pid) for pid in [parent, lane, *frozen]) and time.time() < deadline:
            time.sleep(.2)
        if any(live(pid) for pid in [parent, lane, *frozen]):
            raise RuntimeError('Old evaluator did not stop; preserve it for inspection')
        stop_orphans(group, job)
        value = {'job': job, 'gpu': gpu, 'old_group': group_name, 'completed_route': watched,
                 'startup_handoff_route': cancelled, 'scored_results_preserved': True,
                 'GPU_uuid': config['jobs'][job]['gpu_uuids'][gpu], 'unix': time.time()}
        save(marker, value)
        print(json.dumps(value), flush=True)
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=JOBS)
    parser.add_argument('gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    drain(args.job, args.gpu)
