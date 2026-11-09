"""A shared, locked route queue; scored driving failures are never retried."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time


class RouteQueue:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'route_queue.json'
        self.lock = self.directory / 'route_queue.lock'

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else None
            yield state

    def save(self, state):
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.path)

    def initialize(self, routes, identity):
        with self.locked() as state:
            if state is not None:
                if state['identity'] != identity or state['routes'] != routes:
                    raise RuntimeError('Queue identity differs; use a separate run directory')
                return
            self.save({'identity': identity, 'routes': routes, 'created_unix': time.time(),
                       'items': {r['id']: {'state': 'pending', 'attempts': 0, 'history': []}
                                 for r in routes}})

    def claim(self, worker, preferred_town=None):
        with self.locked() as state:
            pending = [r for r in state['routes'] if state['items'][r['id']]['state'] == 'pending']
            if not pending:
                return None
            route = next((r for r in pending if r['town'] == preferred_town), pending[0])
            item = state['items'][route['id']]
            state['claim_serial'] = state.get('claim_serial', 0) + 1
            item.update(state='claimed', owner=worker, claimed_unix=time.time(), attempts=item['attempts'] + 1)
            self.save(state)
            return {**route, 'attempt': item['attempts'], 'serial': state['claim_serial']}

    def finish(self, route, worker, summary):
        with self.locked() as state:
            item = state['items'][route]
            if item['state'] != 'claimed' or item['owner'] != worker:
                raise RuntimeError('Route is not owned by this worker')
            item['history'].append({'worker': worker, 'summary': summary, 'unix': time.time()})
            category = summary['classification']
            if category == 'driving_result':
                item['state'] = 'done'
            elif category == 'infrastructure_error' and item['attempts'] < 2:
                item['state'] = 'pending'
            else:
                item['state'] = 'error'
            item['last_result'] = summary
            self.save(state)

    def snapshot(self):
        with self.locked() as state:
            return state

