"""Original policy with passive setup timings for CARLA loading tests."""
import json
import os
from pathlib import Path
import time

from evaluation.full_diffusion_diagnostic_agent import FullDiffusionDiagnosticAgent
from method.full_diffusion.diffusion import VPFullDiffusion


def get_entry_point():
    return 'FullDiffusionFastAgent'


class FullDiffusionFastAgent(FullDiffusionDiagnosticAgent):
    def _event(self, event, **values):
        directory = Path(os.environ['WMMPC_CARLA_LOG_DIR'])
        with (directory / 'route_events.jsonl').open('a') as handle:
            handle.write(json.dumps({'event': event, 'route': getattr(self, '_fast_route', None),
                                     'unix': time.time(), **values}) + '\n')

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self._fast_route = route_index or path_to_conf_file.split('+')[-1]
        self._event('agent_setup_started')
        super().setup(path_to_conf_file, route_index, traffic_manager)
        if self.nets[0].run_config['diffusion']['inference_steps'] != 100:
            raise ValueError('The checkpoint must require exactly 100 denoising steps')
        if type(self.nets[0].diffusion) is not VPFullDiffusion:
            raise TypeError('CARLA lifecycle evaluation requires the original VPFullDiffusion')
        self._event('agent_setup_ready', inference_steps=100,
                    sampler='VPFullDiffusion', sampler_replaced=False)

    def destroy(self, results=None):
        super().destroy(results)
        self._event('agent_destroy_finished')
