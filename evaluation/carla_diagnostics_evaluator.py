"""Run the existing benchmark evaluator with isolated UE logs and exit diagnostics.

Scoring, scenarios, watchdog settings and driving behavior remain in the existing
leaderboard implementation. Only the server command's storage/logging is changed.
"""
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


ROOT = Path(__file__).resolve().parents[1]
LOG = Path(os.environ['WMMPC_CARLA_LOG_DIR']).resolve()
LOG.relative_to(ROOT.resolve())
LOG.mkdir(parents=True, exist_ok=True)
_popen = subprocess.Popen
_finishing = threading.Event()
_servers = []
_threads = []


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
        carla_root = Path(command[0]).parent
        binary = carla_root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        command = [str(binary), 'CarlaUE4', *command[1:],
                   f'-abslog={LOG / "CarlaUE4.log"}', f'-userdir={LOG / "unreal_user"}',
                   '-unattended', '-nosplash']
        # Avoid the shared shell script's chmod and keep all writable files here.
        environment = kwargs.get('env', os.environ).copy()
        environment['HOME'] = str(LOG / 'home')
        environment['XDG_CACHE_HOME'] = str(LOG / 'cache')
        for directory in ('home', 'cache', 'unreal_user'):
            (LOG / directory).mkdir(exist_ok=True)
        kwargs['env'] = environment
        stream = (LOG / 'server_stdout.log').open('ab', buffering=0)
        kwargs.update(stdout=stream, stderr=subprocess.STDOUT)
        server = _popen(command, *args, **kwargs)
        stream.close()
        _servers.append(server)
        write('server_process.json', {'pid': server.pid, 'command': command,
                                      'unix': time.time(), 'cuda_visible_devices':
                                      environment.get('CUDA_VISIBLE_DEVICES')})
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
    runpy.run_path(str(ROOT / 'leaderboard/leaderboard/leaderboard_evaluator.py'), run_name='__main__')
finally:
    _finishing.set()
