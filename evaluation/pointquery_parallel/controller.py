"""Original warm-CARLA evaluation with a verified two-session point-query backend."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluation.pointquery_parallel.common import ROOT, OUTPUT, RUN, ARTIFACTS, read, save


def main():
    from scripts import pointquery_monitor_recovery_20261124 as monitor
    monitor.install()
    import method.full_diffusion_pointquery.eval_controller as base
    from evaluation.carla_route_queue import RouteQueue
    real_popen = monitor.REAL_POPEN
    original_configure = base.configure_policy
    original_controller = ROOT / 'method/full_diffusion_pointquery/eval_controller.py'
    original_broker = ROOT / 'evaluation/shared_gpu_inference.py'
    extension = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('parent', 'lane'), default='parent')
    parser.add_argument('--method', default='full_diffusion_pointquery')
    parser.add_argument('--cpus-per-gpu', type=int, default=8)
    parser.add_argument('--server-mode', default='warm')
    parser.add_argument('--startup-wait', default='rpc')
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path, default=ROOT / '.runtime/carla_0.9.15_v1')
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gpu-uuid', default='')
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--backend', default='rpc')
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--route-ids', default='all')
    parser.add_argument('--slot-base', type=int, required=True)
    parser.add_argument('--tm-base', type=int, required=True)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--wall-budget', type=int, default=36000)
    parser.add_argument('--allocation-record', type=Path)
    parser.add_argument('--routes-per-server', type=int, default=32)
    parser.add_argument('--capture-first-policy-call', action='store_true')
    args = parser.parse_args()
    if args.method != 'full_diffusion_pointquery' or args.lanes != 2 or args.backend != 'rpc' or args.cpus_per_gpu != 8:
        raise ValueError('This extension is the authorized 8-CPU/GPU, two-lane point-query configuration')
    proof = read(ARTIFACTS / 'preflight.json')
    if not proof or not proof['passed'] or proof['inference_steps'] != 25 or not proof.get('GPU_detection_decoding_exact'):
        raise RuntimeError('Independent state/coordinate transport preflight did not pass')
    args.inference_steps, args.sampler_name = 25, 'Original SpatialTemporalVPFullDiffusion (DDIM 25), independent RPC sessions'

    class CompatibleQueue(RouteQueue):
        def initialize(self, routes, identity):
            baseline = self.snapshot()
            normalized = copy.deepcopy(identity)
            normalized['backend'] = baseline['identity']['backend']
            # Only the transport label differs; all model, source, seed and
            # sampler identities must still equal the existing scored run.
            super().initialize(routes, normalized)
            with self.locked() as state:
                state.setdefault('execution_variants', {})[args.group_id] = {
                    'backend': 'pointquery_rpc14', 'lanes': 2, 'inference_steps': 25,
                    'extension_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in extension.glob('*.py')},
                    'equivalence_preflight': str(ARTIFACTS / 'preflight.json'), 'unix': time.time()}
                self.save(state)

        def finish(self, route, worker, summary):
            with self.locked() as state:
                item = state['items'][route]
                if item['state'] != 'claimed' or item['owner'] != worker:
                    raise RuntimeError('Route is not owned by this worker')
                item['history'].append({'worker': worker, 'summary': summary, 'unix': time.time()})
                category = summary['classification']
                infra = sum(h['summary']['classification'] == 'infrastructure_error' for h in item['history'])
                item['state'] = ('done' if category == 'driving_result' else
                                 'pending' if category == 'infrastructure_error' and infra < 2 else 'error')
                item['last_result'] = summary
                self.save(state)

    base.RouteQueue = CompatibleQueue
    original_save = base.save

    def save_manifest(path, value):
        if Path(path).name == 'manifest.json' and 'identity' in value:
            # Keep the original semantic identity as the anchor expected by
            # existing summary readers. Explicitly retain the actual RPC
            # execution identity beside it; no original manifests are edited.
            value = dict(value)
            value['execution_identity'] = value['identity']
            value['identity'] = read(args.run_root / 'route_queue.json')['identity']
            value['execution_backend'] = 'pointquery_rpc14'
            value['equivalence_preflight'] = str(ARTIFACTS / 'preflight.json')
        return original_save(path, value)

    base.save = save_manifest

    def configure(configuration, method):
        configuration = original_configure(configuration, method)
        configuration.agent = str(extension / 'agent.py')
        return configuration

    base.configure_policy = configure

    def popen(command, *values, **kwargs):
        if isinstance(command, (list, tuple)):
            cmd = list(command)
            if len(cmd) > 2 and cmd[0] == sys.executable and cmd[1] == '-u':
                if Path(cmd[2]).resolve() == original_controller:
                    command = [cmd[0], '-u', str(extension / 'controller.py'), *cmd[3:]]
                elif Path(cmd[2]).resolve() == original_broker:
                    command = [cmd[0], '-u', str(extension / 'broker.py'), *cmd[3:]]
        return real_popen(command, *values, **kwargs)

    subprocess.Popen = popen

    class IsolatedLaneServer(base.LaneServer):
        def __init__(self, settings, uuid, directory):
            self.directory = directory
            directory.mkdir(parents=True)
            cache = base.evaluation_runtime_root() / 'carla_cache' / settings.job_id / uuid / f'parallel14_lane{settings.lane}'
            for name in ('home', 'cache', 'unreal_user'):
                (cache / name).mkdir(parents=True, exist_ok=True)
            (directory / 'tmp').mkdir()
            port = 2000 + settings.slot * 10
            binary = settings.carla_root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
            command = [str(binary), 'CarlaUE4', '-RenderOffScreen', '-nosound', f'-carla-rpc-port={port}',
                       f'-graphicsadapter={settings.gpu}', f'-abslog={directory / "CarlaUE4.log"}',
                       f'-userdir={cache / "unreal_user"}', '-unattended', '-nosplash']
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                           'HOME': str(cache / 'home'), 'XDG_CACHE_HOME': str(cache / 'cache'),
                           'TMPDIR': str(directory / 'tmp'), 'OMP_NUM_THREADS': '1',
                           'VK_ICD_FILENAMES': '/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json'}
            with (directory / 'server_stdout.log').open('wb') as log:
                self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
            self.owner = directory / 'server_process.json'
            base.save(self.owner, {'pid': self.process.pid, 'port': port, 'gpu_uuid': uuid,
                                  'job_id': settings.job_id, 'command': command, 'unix': time.time()})
            try:
                base.wait_for_server(self.process, 'localhost', port, directory, timeout=600)
            except BaseException:
                self.close()
                raise
            os.environ.update(WMMPC_EXTERNAL_CARLA_PID=str(self.process.pid), WMMPC_EXTERNAL_CARLA_PORT=str(port),
                              WMMPC_EXTERNAL_CARLA_OWNER=str(self.owner), WMMPC_CACHE_STATIC_ROADS='0')

    base.LaneServer = IsolatedLaneServer

    def stop_owned(process):
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    base.stop_owned = stop_owned
    if args.role == 'lane':
        ready = read(args.run_root / 'groups' / args.group_id / 'broker/ready.json')
        os.environ.update(WMMPC_POLICY_SOCKET=ready['address'], WMMPC_POLICY_TOKEN=ready['token'])
        base.lane(args)
    else:
        base.parent(args)


if __name__ == '__main__':
    main()
