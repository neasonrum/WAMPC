"""Passive telemetry around the unchanged V3 sampler and controller."""

from evaluation.full_diffusion2_fast_agent import FullDiffusionTiming
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.inference.carla_agent import FullDiffusionV3Agent


def get_entry_point():
    return 'FullDiffusionV3FastAgent'


class FullDiffusionV3FastAgent(FullDiffusionTiming, FullDiffusionV3Agent):
    method = 'full_diffusionv3'
    sampler_type = SpatialTemporalVPFullDiffusion
    log_prefix = 'FULL_DIFFUSIONV3'

    def extra_telemetry(self):
        network = self.nets[0]
        delta = network.last_temporal_delta
        behavior = network.last_behavior_probabilities
        return {'architecture': network.run_config['architecture'],
            'longitudinal_controller': network.run_config['controller'],
            'reference_interval_seconds': [0.5, 1.0],
            'speed_limit_kmh': 35,
            'legacy_speed_bins_control': False,
            'temporal_delta_xy': delta[0].cpu().tolist() if delta is not None else None,
                'behavior_probabilities': behavior[0].cpu().tolist() if behavior is not None else None}