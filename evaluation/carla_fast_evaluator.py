"""Reuse CARLA processes and preload the first map; keep original policy/scoring."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import threading
import time
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.fast_carla_io import replace_startup_sleep, wait_for_server
from evaluation.carla_lifecycle import Lifecycle, first_town, instrument_evaluator, map_asset
from evaluation.carla_eval_resources import checked_evaluation_path
LOG = checked_evaluation_path(os.environ['WMMPC_CARLA_LOG_DIR'])
LOG.mkdir(parents=True, exist_ok=True)
_popen = subprocess.Popen
_finishing = threading.Event()
_servers = []
_threads = []
INITIAL_TOWN = first_town(sys.argv)
PRELOAD = os.environ.get('WMMPC_CARLA_PRELOAD_FIRST_MAP', '0') == '1'
LIFECYCLE = Lifecycle(LOG, INITIAL_TOWN, use_initial_world=PRELOAD)
EXTERNAL_PID = int(os.environ.get('WMMPC_EXTERNAL_CARLA_PID','0'))
EXTERNAL_PORT = int(os.environ.get('WMMPC_EXTERNAL_CARLA_PORT','0'))


class BorrowedServer:
    def __init__(self,pid):self.pid=pid;self.returncode=None

    def poll(self):
        try:os.kill(self.pid,0)
        except ProcessLookupError:self.returncode=1
        return self.returncode


def write(name, data):
    target = LOG / name
    temporary = target.with_suffix(target.suffix + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)


def watch(server):
    code = server.wait()
    write('server_exit.json', {'pid': server.pid, 'returncode': code, 'unix': time.time(),
                               'evaluator_finishing': _finishing.is_set()})
    if not _finishing.is_set():
        print(f'CARLA_SERVER_EXIT pid={server.pid} returncode={code}', flush=True)
        os.kill(os.getpid(), signal.SIGTERM)


def logged_popen(command, *args, **kwargs):
    if isinstance(command, (list, tuple)) and command and Path(command[0]).name == 'CarlaUE4.sh':
        if EXTERNAL_PID:
            owner=checked_evaluation_path(os.environ['WMMPC_EXTERNAL_CARLA_OWNER'])
            info=json.loads(owner.read_text())
            if info['pid']!=EXTERNAL_PID or info['port']!=EXTERNAL_PORT:
                raise RuntimeError('External CARLA ownership record differs')
            if info['gpu_uuid']!=os.environ['CUDA_VISIBLE_DEVICES']:
                raise RuntimeError('External CARLA GPU differs from evaluator GPU')
            commandline=Path(f'/proc/{EXTERNAL_PID}/cmdline').read_bytes().decode().replace('\x00',' ')
            if f'-carla-rpc-port={EXTERNAL_PORT}' not in commandline or str(owner.parent/'CarlaUE4.log') not in commandline:
                raise RuntimeError('External CARLA process identity differs')
            server=BorrowedServer(EXTERNAL_PID)
            if server.poll() is not None:raise RuntimeError('External CARLA has exited')
            write('server_lease.json',{'pid':EXTERNAL_PID,'port':EXTERNAL_PORT,'owner':str(owner),
                                       'gpu_uuid':info['gpu_uuid'],'unix':time.time(),
                                       'scope':'Server owned by the same worker; fresh evaluator for this route'})
            return server
        carla_root = Path(command[0]).parent
        binary = carla_root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        cache = checked_evaluation_path(os.environ.get('WMMPC_CARLA_CACHE_DIR', str(LOG / 'server_cache')))
        cache.mkdir(parents=True, exist_ok=True)
        initial_map = [map_asset(INITIAL_TOWN)] if PRELOAD else []
        command = [str(binary), 'CarlaUE4', *initial_map, *command[1:],
                   f'-abslog={LOG / "CarlaUE4.log"}', f'-userdir={cache / "unreal_user"}',
                   '-unattended', '-nosplash']
        # Avoid the shared shell script's chmod and keep all writable files here.
        environment = kwargs.get('env', os.environ).copy()
        environment['HOME'] = str(cache / 'home')
        environment['XDG_CACHE_HOME'] = str(cache / 'cache')
        for directory in ('home', 'cache', 'unreal_user'):
            (cache / directory).mkdir(exist_ok=True)
        kwargs['env'] = environment
        stream = (LOG / 'server_stdout.log').open('ab', buffering=0)
        kwargs.update(stdout=stream, stderr=subprocess.STDOUT)
        server = _popen(command, *args, **kwargs)
        stream.close()
        _servers.append(server)
        write('server_process.json', {'pid': server.pid, 'command': command,
                                      'unix': time.time(), 'cuda_visible_devices':
                                      environment.get('CUDA_VISIBLE_DEVICES'),
                                      'preloaded_town': INITIAL_TOWN if PRELOAD else None,
                                      'persistent_cache': str(cache)})
        thread = threading.Thread(target=watch, args=(server,), daemon=True)
        _threads.append(thread)
        thread.start()
        return server
    return _popen(command, *args, **kwargs)


def finish():
    _finishing.set()
    for server in _servers:
        if server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for thread in _threads:
        thread.join(timeout=3)


def terminate(signum, frame):
    write('evaluator_signal.json', {'signal': signum, 'unix': time.time()})
    raise SystemExit(128 + signum)


atexit.register(finish)
signal.signal(signal.SIGTERM, terminate)
subprocess.Popen = logged_popen
write('diagnostic_source.json', {'path': str(Path(__file__).resolve()),
                                 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
try:
    evaluator = ROOT / 'leaderboard/leaderboard/leaderboard_evaluator.py'
    source = evaluator.read_text()
    write('upstream_evaluator.json', {
        'path': str(evaluator), 'sha256': hashlib.sha256(source.encode()).hexdigest(),
        'transformation': 'RPC readiness; unused preloaded first episode; route-state isolation; full reload for later routes',
        'simulation_hz': 20, 'inference_steps': 100})
    namespace = {'__name__': '_wmmpc_lifecycle_upstream', '__file__': str(evaluator),
                 '_wmmpc_load_world': LIFECYCLE.load_world,
                 '_wmmpc_wait_for_server': lambda server, host, port:
                     wait_for_server(server, host, port, LOG)}
    transformed = instrument_evaluator(source)
    if os.environ.get('WMMPC_CARLA_STARTUP_WAIT', 'rpc') == 'rpc':
        transformed = replace_startup_sleep(transformed)
    elif os.environ['WMMPC_CARLA_STARTUP_WAIT'] != 'fixed30':
        raise ValueError('Unknown startup wait mode')
    exec(compile(transformed, str(evaluator), 'exec'), namespace)
    if EXTERNAL_PID:
        original_find_port=namespace['find_free_port']
        namespace['find_free_port']=lambda port:EXTERNAL_PORT if port==EXTERNAL_PORT else original_find_port(port)
        original_stop=namespace['stop_server']
        namespace['stop_server']=lambda server:None if isinstance(server,BorrowedServer) else original_stop(server)
    LIFECYCLE.install(namespace)
    namespace['main']()
finally:
    _finishing.set()
