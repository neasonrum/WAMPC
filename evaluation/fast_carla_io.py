"""Small, auditable I/O optimizations; no planner or scoring changes."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import time


SENSOR_AGENT_SHA256 = '14dfe25b7d0d7444f8ac48df30cfc96722ba52b12a09ff627786da3deafcfbbe'


class MetricJournal:
    """Append each complete tick once and retain the original final JSON format."""
    def __init__(self, directory, serializer, flush_every=20):
        self.directory = Path(directory)
        self.serializer = serializer
        self.flush_every = flush_every
        self.buffer = []
        self.last_step = -1
        self.finalized = False
        self.journal = self.directory / 'metric_info.jsonl'
        if self.journal.exists():
            raise FileExistsError(f'Refusing to reuse a metric journal: {self.journal}')

    def append(self, step, metric):
        if self.finalized or step <= self.last_step:
            raise ValueError('Metric steps must be strictly increasing in an active route')
        self.buffer.append(self.serializer.dumps([step, metric]) + '\n')
        self.last_step = step
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buffer:
            with self.journal.open('a') as handle:
                handle.writelines(self.buffer)
                handle.flush()
            self.buffer.clear()

    def finalize(self, metrics):
        if self.finalized:
            return
        self.flush()
        path = self.directory / 'metric_info.json'
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('w') as handle:
            self.serializer.dump(metrics, handle, indent=4)
        temporary.replace(path)
        self.finalized = True

    @staticmethod
    def recover(path):
        recovered = {}
        lines = Path(path).read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                step, metric = json.loads(line)
            except ValueError:
                # A killed append can leave only the final line incomplete.
                if index != len(lines) - 1:
                    raise
                break
            if str(step) in recovered:
                raise ValueError('Duplicate metric step in journal')
            recovered[str(step)] = metric
        return recovered


def replace_metric_write(function_source):
    """Change exactly one history-file write, preserving its execution position."""
    tree = ast.parse(textwrap.dedent(function_source))
    changed = 0
    for parent in ast.walk(tree):
        for field, children in ast.iter_fields(parent):
            if not isinstance(children, list):
                continue
            for index, node in enumerate(children):
                if not isinstance(node, ast.With) or len(node.items) != 1:
                    continue
                expression = node.items[0].context_expr
                if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                        and expression.func.id == 'open' and expression.args
                        and any(isinstance(x, ast.Constant) and x.value == 'metric_info.json'
                                for x in ast.walk(expression.args[0]))):
                    continue
                call = ast.Expr(ast.Call(
                    func=ast.Attribute(ast.Name('self', ast.Load()), '_store_metric', ast.Load()),
                    args=[ast.Name('metric_info', ast.Load())], keywords=[]))
                children[index] = ast.copy_location(call, node)
                changed += 1
    if changed != 1:
        raise RuntimeError(f'Expected exactly one metric-file write, found {changed}')
    return ast.fix_missing_locations(tree)


def compile_buffered_run_step(sensor_module):
    import inspect
    path = Path(sensor_module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != SENSOR_AGENT_SHA256:
        raise RuntimeError('The upstream sensor agent changed; review the logging adapter first')
    source = inspect.getsource(sensor_module.SensorAgent.run_step)
    namespace = dict(vars(sensor_module))
    exec(compile(replace_metric_write(source), str(path) + ':buffered_metrics', 'exec'), namespace)
    return namespace['run_step']


def wait_for_server(server, host, port, log_dir, timeout=120, client_factory=None):
    """Replace a fixed sleep with successful read-only CARLA RPCs."""
    if client_factory is None:
        import carla
        client_factory = carla.Client
    started = time.monotonic()
    attempts = 0
    error = None
    while time.monotonic() - started < timeout:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f'CARLA exited before RPC readiness: {server.returncode}')
        attempts += 1
        try:
            client = client_factory(host, int(port))
            client.set_timeout(1.0)
            version = client.get_server_version()
            snapshot = client.get_world().get_snapshot()
            record = {'ready': True, 'elapsed_seconds': time.monotonic() - started,
                      'attempts': attempts, 'version': version, 'frame': snapshot.frame,
                      'unix': time.time()}
            if log_dir is not None:
                (Path(log_dir) / 'startup_readiness.json').write_text(json.dumps(record, indent=2) + '\n')
            return record
        except RuntimeError as exc:
            error = repr(exc)
            time.sleep(min(0.5, max(0, timeout - (time.monotonic() - started))))
    raise TimeoutError(f'CARLA RPC readiness timed out after {attempts} attempts: {error}')


def replace_startup_sleep(source):
    target = '        time.sleep(30)'
    if source.count(target) != 1:
        raise RuntimeError('Expected one known CARLA startup sleep; review evaluator changes')
    return source.replace(target, '        _wmmpc_wait_for_server(self.server, args.host, args.port)', 1)


def cooperative_pause(directory, job_id, poll_seconds=0.2):
    """Acknowledge a pause only between batches, with no evaluator or server."""
    import os
    directory = Path(directory)
    request = directory / 'pause_request.json'
    if not request.exists():
        return
    data = json.loads(request.read_text())
    if data.get('job_id') != str(job_id) or not data.get('token'):
        raise RuntimeError('Pause request identity does not match this worker')
    token = data['token']
    state = {'job_id': str(job_id), 'token': token, 'pid': os.getpid(),
             'state': 'paused', 'unix': time.time()}
    acknowledgment = directory / 'pause_ack.json'
    def save_state():
        temporary = acknowledgment.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(acknowledgment)
    save_state()
    while request.exists():
        current = json.loads(request.read_text())
        if current.get('token') != token or current.get('job_id') != str(job_id):
            raise RuntimeError('Pause request was replaced while the GPU was reserved')
        time.sleep(poll_seconds)
    state.update(state='resumed', resumed_unix=time.time())
    save_state()
