"""Original CPU sensor/control state, with isolated GPU pose/history/RNG sessions."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np
import torch

from method.full_diffusion_pointquery.fast_agent import PointQueryFastAgent
from evaluation.pointquery_parallel.codec import PolicyClient, pack, unpack


def get_entry_point():
    return 'PointQueryRPCAgent'


class PointQueryRPCAgent(PointQueryFastAgent):
    @property
    def device(self):
        return torch.device('cpu')

    @device.setter
    def device(self, requested):
        self._requested_device = str(requested)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        with patch.object(torch.nn.Module, 'cuda', lambda module, device=None: module.to('cpu')), \
                patch.object(torch.cuda, 'empty_cache', lambda: None):
            super().setup(path_to_conf_file, route_index, traffic_manager)
        self._policy_client = PolicyClient(os.environ['WMMPC_POLICY_SOCKET'], os.environ['WMMPC_POLICY_TOKEN'])
        metadata = self._policy_client.call('open')
        checkpoint = Path(path_to_conf_file.split('+')[0])
        if (metadata['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                or metadata['inference_steps'] != 25 or metadata['method'] != 'full_diffusion_pointquery'):
            raise RuntimeError('Shared GPU service checkpoint or sampling protocol differs')
        net = self.nets[0]
        self._rpc_observation = None

        def observation(pose, timestamp):
            self._rpc_observation = (np.asarray(pose, dtype=np.float64).copy(), float(timestamp))

        def remote_forward(*args, **kwargs):
            if args or self._rpc_observation is None:
                raise RuntimeError('Point-query RPC needs named sensor inputs and the current real ego pose')
            started = time.perf_counter()
            pose, timestamp = self._rpc_observation
            response = self._policy_client.call('forward', inputs=pack(kwargs), pose=pose, timestamp=timestamp)
            output = unpack(response['output'])
            net.last_target_speed = response['last_target_speed']
            net.last_temporal_delta = unpack(response['last_temporal_delta'])
            net.last_behavior_probabilities = unpack(response['last_behavior_probabilities'])
            self._decoded_boxes = response['decoded_boxes']
            net.history = SimpleNamespace(entries=range(response['history_entries']))
            self._network_seconds = time.perf_counter() - started
            self._rpc_queue_seconds = response['queue_seconds']
            if self.step < 3 or self.step % 20 == 0:
                self._last_path = output[2][0].float().tolist()
            return output

        net.set_observation, net.forward = observation, remote_forward
        net.convert_features_to_bb_metric = lambda _: self._decoded_boxes
        print('POINTQUERY_RPC_READY steps=25 independent_history_pose_rng=True', flush=True)

    def extra_telemetry(self):
        value = super().extra_telemetry()
        value.update(inference_backend='pointquery_rpc14', gpu_queue_seconds=getattr(self, '_rpc_queue_seconds', None))
        return value

    def destroy(self, results=None):
        try:
            super().destroy(results)
        finally:
            client = getattr(self, '_policy_client', None)
            if client is not None:
                client.close()
