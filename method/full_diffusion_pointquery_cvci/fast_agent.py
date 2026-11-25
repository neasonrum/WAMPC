"""Passive timing and prior-age diagnostics for V8's 25-step DDIM policy."""
from evaluation.full_diffusion2_fast_agent import FullDiffusionTiming
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusion_pointquery_cvci.carla_agent import V8Agent


def get_entry_point():
    return 'V8FastAgent'


class V8FastAgent(FullDiffusionTiming, V8Agent):
    method = 'full_diffusion_pointquery_cvci'
    inference_steps = 25
    sampler_type = SpatialTemporalVPFullDiffusion
    log_prefix = 'V8_EGO12'

    def extra_telemetry(self):
        network = self.nets[0]
        return {'architecture': network.run_config['architecture'], 'ego_condition_tokens': 12,
                'direct_scene_memory_attention': False, 'world_jointly_trained': True,
                'prior_tick': network.last_prior_tick, 'history_ticks': network.last_history_ticks,
                'future_ready': network.last_future_ready, 'prior_age_seconds': .25,
                'temporal_delta_xy': network.last_temporal_delta[0].cpu().tolist()
                if network.last_temporal_delta is not None else None}
