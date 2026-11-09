"""Exercise real concurrent claims, bounded retries and failed-driving retention."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue


def worker(directory, index, barrier):
    queue = RouteQueue(directory)
    owner = f'contract_worker_{index}'
    barrier.wait()
    while True:
        task = queue.claim(owner)
        if task is None:
            return
        time.sleep(.01)
        route = task['id']
        infrastructure = route == '2' or (route == '0' and task['attempt'] == 1)
        summary = {'route_id': route,
                   'classification': 'infrastructure_error' if infrastructure else 'driving_result',
                   'status': 'Failed - Agent got blocked' if route == '1' else 'Completed',
                   'attempt_index': task['attempt']}
        queue.finish(route, owner, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve(); directory.relative_to(ROOT.resolve())
    assert not directory.exists()
    queue = RouteQueue(directory)
    queue.initialize([{'id': str(i), 'town': 'contract_test'} for i in range(24)], {'test': True})
    context = mp.get_context('spawn')
    barrier = context.Barrier(4)
    processes = [context.Process(target=worker, args=(directory, i, barrier)) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0, process.exitcode
    state = queue.snapshot()
    items = state['items']
    assert sum(v['state'] == 'done' for v in items.values()) == 23
    assert items['0']['attempts'] == 2 and items['0']['state'] == 'done'
    assert items['1']['attempts'] == 1 and items['1']['state'] == 'done'
    assert items['2']['attempts'] == 2 and items['2']['state'] == 'error'
    assert all(v['attempts'] == len(v['history']) for v in items.values())
    owners = {event['worker'] for v in items.values() for event in v['history']}
    assert len(owners) == 4
    result = {'success': True, 'workers': len(owners), 'routes': len(items),
              'claims': state['claim_serial'], 'bounded_infrastructure_retries': True,
              'driving_failure_retained_without_retry': True}
    (directory / 'contract_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
