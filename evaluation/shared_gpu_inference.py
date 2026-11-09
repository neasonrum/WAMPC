"""One CUDA process serves isolated original policies, without changing sampling.

Uses inline pickle bytes over authenticated local sockets. No shared-memory
tensor transport, batching, altered precision, or modified diffusion code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import queue
import signal
import sys
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pack(value):
    import torch
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        is_bfloat = tensor.dtype == torch.bfloat16
        array = tensor.view(torch.uint16).numpy() if is_bfloat else tensor.numpy()
        return {'__tensor__': True, 'bfloat16': is_bfloat, 'array': array}
    if isinstance(value, tuple):
        return tuple(pack(x) for x in value)
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value, device='cpu'):
    import torch
    if isinstance(value, dict) and value.get('__tensor__') is True:
        tensor = torch.from_numpy(value['array'])
        if value['bfloat16']:
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(device)
    if isinstance(value, tuple):
        return tuple(unpack(x, device) for x in value)
    if isinstance(value, list):
        return [unpack(x, device) for x in value]
    if isinstance(value, dict):
        return {k: unpack(v, device) for k, v in value.items()}
    return value


def send(connection, value):
    connection.send_bytes(pickle.dumps(value, protocol=5))


def receive(connection):
    return pickle.loads(connection.recv_bytes(64 * 1024 * 1024))


class PolicyClient:
    def __init__(self, address, token):
        self.connection = Client(address, family='AF_UNIX', authkey=Path(token).read_bytes())

    def call(self, operation, **values):
        send(self.connection, {'operation': operation, **values})
        if not self.connection.poll(180):
            raise TimeoutError('Shared GPU inference did not respond within 180 seconds')
        result = receive(self.connection)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result['value']

    def close(self):
        try:
            self.call('close')
        finally:
            self.connection.close()


def load_policy(checkpoint):
    from method.full_diffusion.inference.carla_agent import FullDiffusionGeoAgent
    from method.full_diffusion.diffusion import VPFullDiffusion
    agent = FullDiffusionGeoAgent.__new__(FullDiffusionGeoAgent)
    agent.setup(str(checkpoint), None, None)
    net = agent.nets[0]
    assert type(net.diffusion) is VPFullDiffusion
    assert net.run_config['diffusion']['inference_steps'] == 100
    return net


def serve(args):
    import torch
    torch.set_num_threads(1)
    directory = args.directory.resolve()
    directory.relative_to(ROOT.resolve())
    directory.mkdir(parents=True, exist_ok=True)
    token = directory / 'token'
    with token.open('xb') as handle:
        handle.write(os.urandom(32))
    token.chmod(0o600)
    socket_id = hashlib.sha256(str(directory).encode()).hexdigest()[:8]
    address = str(ROOT / f'.pi_{socket_id}.sock')
    os.environ.pop('SAVE_PATH', None)
    os.environ['IS_BENCH2DRIVE'] = 'True'
    listener = Listener(address, family='AF_UNIX', authkey=token.read_bytes())
    Path(address).chmod(0o600)
    pending = queue.Queue(maxsize=8)
    stop = threading.Event()
    clients = []

    def handle(connection):
        session = {'net': None, 'forwards': 0}
        try:
            while not stop.is_set():
                request = receive(connection)
                event = threading.Event()
                task = {'request': request, 'event': event, 'session': session,
                        'enqueued': time.monotonic()}
                pending.put(task)
                while not event.wait(.5):
                    if stop.is_set():
                        return
                send(connection, task['response'])
                if request['operation'] == 'close':
                    return
        except (EOFError, ConnectionError, OSError):
            if not stop.is_set():
                pending.put({'request': {'operation': 'close'}, 'event': threading.Event(),
                             'session': session, 'enqueued': time.monotonic()})
        finally:
            connection.close()

    def accept():
        while not stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            worker = threading.Thread(target=handle, args=(connection,), daemon=True)
            clients.append(worker)
            worker.start()

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=accept, daemon=True).start()
    ready = {'pid': os.getpid(), 'address': address, 'token': str(token),
             'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
             'sampler': 'Original VPFullDiffusion', 'inference_steps': 100,
             'max_sessions': args.max_sessions, 'unix': time.time()}
    (directory / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    active = 0
    calls = 0
    try:
        while not stop.is_set():
            try:
                task = pending.get(timeout=.5)
            except queue.Empty:
                continue
            session = task['session']
            request = task['request']
            started = time.monotonic()
            try:
                operation = request['operation']
                if operation == 'open':
                    if session['net'] is not None or active >= args.max_sessions:
                        raise RuntimeError('Invalid or excessive policy session')
                    session['net'] = load_policy(args.checkpoint)
                    active += 1
                    value = {'inference_steps': 100, 'sampler': 'VPFullDiffusion',
                             'checkpoint_sha256': ready['checkpoint_sha256']}
                elif operation == 'forward':
                    assert session['net'] is not None
                    # SensorAgent.run_step is decorated with inference_mode.
                    # Preserve that outer execution context across the RPC.
                    with torch.inference_mode():
                        inputs = unpack(request['inputs'], 'cuda:0')
                        output = session['net'].forward(**inputs)
                    value = {'output': pack(output), 'last_target_speed': session['net'].last_target_speed,
                             'queue_seconds': started - task['enqueued']}
                    session['forwards'] += 1
                    calls += 1
                elif operation == 'close':
                    if session['net'] is not None:
                        session['net'] = None
                        active -= 1
                    value = {'forwards': session['forwards']}
                elif operation == 'ping':
                    value = ready
                else:
                    raise ValueError(operation)
                task['response'] = {'ok': True, 'value': value}
            except Exception:
                task['response'] = {'ok': False, 'error': traceback.format_exc()}
                print(task['response']['error'], flush=True)
            finally:
                task['event'].set()
            if calls % 20 == 0 or request['operation'] in ('open', 'close'):
                status = {'active_sessions': active, 'forward_calls': calls,
                          'allocated_bytes': torch.cuda.memory_allocated(),
                          'reserved_bytes': torch.cuda.memory_reserved(), 'unix': time.time()}
                (directory / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    finally:
        stop.set()
        listener.close()
        # The socket is owned by this process and stays under its run directory.
        if Path(address).is_socket():
            Path(address).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-sessions', type=int, choices=(1, 2), required=True)
    serve(parser.parse_args())

