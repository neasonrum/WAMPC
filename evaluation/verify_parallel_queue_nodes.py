"""CPU-only cross-node GPFS claim audit; synthetic routes never enter CARLA."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue

p = argparse.ArgumentParser()
p.add_argument('mode', choices=('initialize', 'worker', 'check'))
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--worker')
a = p.parse_args()
directory = a.directory.resolve(); directory.relative_to(ROOT.resolve()/'results')
q = RouteQueue(directory)
if a.mode == 'initialize':
    assert not directory.exists()
    q.initialize([{'id': str(i), 'town': f'SyntheticTown{i%3}'} for i in range(64)], {'test': 'cross_node_cpu_only'})
    (directory/'start.json').write_text(json.dumps({'unix': time.time()+20}))
elif a.mode == 'worker':
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    owner = socket.gethostname().split('.')[0]+'/'+a.worker
    start = json.loads((directory/'start.json').read_text())['unix']
    while time.time() < start:
        time.sleep(.05)
    previous = None
    while True:
        item = q.claim(owner, previous)
        if item is None:
            break
        time.sleep(.02+(int(item['id'])%4)*.01)
        q.finish(item['id'], owner, {'route_id': item['id'], 'classification': 'driving_result', 'status': 'Synthetic'})
        previous = item['town']
else:
    state = q.snapshot()
    items = list(state['items'].values())
    assert len(items) == 64 and all(v['state'] == 'done' and v['attempts'] == 1 and len(v['history']) == 1 for v in items)
    owners = {}
    for item in items:
        owner = item['owner']; owners[owner] = owners.get(owner, 0)+1
    assert len({o.split('/')[0] for o in owners}) == 2, owners
    assert len(owners) == 4, owners
    result = {'success': True, 'synthetic_routes': 64, 'claims': state['claim_serial'], 'owners': owners, 'duplicate_claims': 0, 'gpu_used': False}
    (directory/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
