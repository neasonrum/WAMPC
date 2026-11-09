"""Fresh original CPU agent/controller with a GPU-only policy RPC boundary."""
import hashlib
import os
import time
from unittest.mock import patch

import torch

from evaluation.full_diffusion_diagnostic_agent import FullDiffusionDiagnosticAgent
from evaluation.shared_gpu_inference import PolicyClient, pack, unpack


def get_entry_point():
    return 'FullDiffusionRPCGeoAgent'


class FullDiffusionRPCGeoAgent(FullDiffusionDiagnosticAgent):
    @property
    def device(self):
        return torch.device('cpu')

    @device.setter
    def device(self, requested):
        self._original_requested_device = str(requested)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        # Only this fresh CPU evaluator is patched during model construction.
        # Control/state/filtering use the original implementation thereafter.
        with patch.object(torch.nn.Module, 'cuda', lambda module, device=None: module.to('cpu')), \
                patch.object(torch.cuda, 'empty_cache', lambda: None):
            super().setup(path_to_conf_file, route_index, traffic_manager)
        self._policy_client = PolicyClient(os.environ['WMMPC_POLICY_SOCKET'], os.environ['WMMPC_POLICY_TOKEN'])
        metadata = self._policy_client.call('open')
        from pathlib import Path
        checkpoint = Path(path_to_conf_file.split('+')[0])
        assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert metadata['inference_steps'] == 100 and metadata['sampler'] == 'VPFullDiffusion'
        net = self.nets[0]

        def remote_forward(*args, **kwargs):
            if args:
                raise TypeError('Use the original named policy input interface')
            started = time.perf_counter()
            response = self._policy_client.call('forward', inputs=pack(kwargs))
            output = unpack(response['output'])
            net.last_target_speed = response['last_target_speed']
            self._network_seconds = time.perf_counter() - started
            if self.step < 3 or self.step % 20 == 0:
                self._last_path = output[2][0].detach().float().cpu().tolist()
            return output

        net.forward = remote_forward
        print('SHARED_GPU_READY original_sampler=VPFullDiffusion inference_steps=100 cpu_controller=True', flush=True)

    def destroy(self, results=None):
        try:
            super().destroy(results)
        finally:
            client = getattr(self, '_policy_client', None)
            if client is not None:
                client.close()
