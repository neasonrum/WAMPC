"""One CUDA process, two independent original 25-DDIM point-query policies."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback
from multiprocessing.connection import Listener

from evaluation.pointquery_parallel.common import ROOT, OUTPUT, save
from evaluation.pointquery_parallel.codec import pack, unpack, send, receive


def load_policy(checkpoint):
    from method.full_diffusion_pointquery.carla_agent import PointQueryAgent
    from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
    agent = PointQueryAgent.__new__(PointQueryAgent)
    agent.setup(str(checkpoint), None, None)
    net = agent.nets[0]
    if type(net.diffusion) is not SpatialTemporalVPFullDiffusion or net.run_config['diffusion']['inference_steps'] != 25:
        raise RuntimeError('Shared service must use the original trained 25-DDIM planner')
    return net


def forward(net, request):
    import torch
    # Match the outer SensorAgent.run_step inference_mode, including H2D.
    with torch.inference_mode():
        inputs = unpack(request['inputs'], 'cuda:0')
        pose, timestamp = request['pose'], request['timestamp']
        net.set_observation(pose, timestamp)
        output = net.forward(**inputs)
        # Preserve GPU detection decoding as well as the neural forward; CPU
        # clients retain only the original route-local filtering/controllers.
        decoded_boxes = net.convert_features_to_bb_metric(output[6]) if output[6] is not None else None
        return {'output': pack(output), 'last_target_speed': net.last_target_speed,
                'last_temporal_delta': pack(net.last_temporal_delta),
                'last_behavior_probabilities': pack(net.last_behavior_probabilities),
                'decoded_boxes': decoded_boxes, 'history_entries': len(net.history.entries)}


def serve(args):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    directory = args.directory.resolve()
    directory.relative_to(OUTPUT.resolve())
    directory.mkdir(parents=True, exist_ok=True)
    token = directory / 'token'
    with token.open('xb') as handle:
        handle.write(os.urandom(32))
    token.chmod(0o600)
    address = str(ROOT / ('.pq14_' + hashlib.sha256(str(directory).encode()).hexdigest()[:8] + '.sock'))
    os.environ.pop('SAVE_PATH', None)
    os.environ['IS_BENCH2DRIVE'] = 'True'
    listener = Listener(address, family='AF_UNIX', authkey=token.read_bytes())
    Path(address).chmod(0o600)
    pending, stop = queue.Queue(maxsize=8), threading.Event()

    def handle(connection):
        session = {'net': None, 'forwards': 0}
        try:
            while not stop.is_set():
                request = receive(connection)
                task = {'request': request, 'session': session, 'event': threading.Event(), 'enqueued': time.monotonic()}
                pending.put(task)
                while not task['event'].wait(.5):
                    if stop.is_set():
                        return
                send(connection, task['response'])
                if request['operation'] == 'close':
                    return
        except (EOFError, ConnectionError, OSError):
            if not stop.is_set():
                pending.put({'request': {'operation': 'close'}, 'session': session,
                             'event': threading.Event(), 'enqueued': time.monotonic()})
        finally:
            connection.close()

    def accept():
        while not stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=handle, args=(connection,), daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=accept, daemon=True).start()
    metadata = {'pid': os.getpid(), 'address': address, 'token': str(token),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'sampler': 'SpatialTemporalVPFullDiffusion', 'method': 'full_diffusion_pointquery',
                'inference_steps': 25, 'max_sessions': args.max_sessions, 'unix': time.time()}
    save(directory / 'ready.json', metadata)
    active, calls = 0, 0
    try:
        while not stop.is_set():
            try:
                task = pending.get(timeout=.5)
            except queue.Empty:
                continue
            request, session = task['request'], task['session']
            started = time.monotonic()
            try:
                op = request['operation']
                if op == 'open':
                    if session['net'] is not None or active >= args.max_sessions:
                        raise RuntimeError('Invalid or excessive independent planner session')
                    session['net'] = load_policy(args.checkpoint)
                    active += 1
                    value = metadata
                elif op == 'forward':
                    if session['net'] is None:
                        raise RuntimeError('Session is not initialized')
                    value = forward(session['net'], request)
                    value['queue_seconds'] = started - task['enqueued']
                    session['forwards'] += 1
                    calls += 1
                elif op == 'close':
                    if session['net'] is not None:
                        session['net'] = None
                        active -= 1
                        gc.collect()
                    value = {'forwards': session['forwards']}
                elif op == 'ping':
                    value = metadata
                else:
                    raise ValueError(op)
                task['response'] = {'ok': True, 'value': value}
            except Exception:
                task['response'] = {'ok': False, 'error': traceback.format_exc()}
                print(task['response']['error'], flush=True)
            finally:
                task['event'].set()
            if calls % 20 == 0 or request['operation'] in ('open', 'close'):
                save(directory / 'status.json', {'active_sessions': active, 'forward_calls': calls,
                     'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(), 'unix': time.time()})
    finally:
        stop.set()
        listener.close()
        if Path(address).is_socket():
            Path(address).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-sessions', type=int, choices=(1, 2), required=True)
    serve(parser.parse_args())

