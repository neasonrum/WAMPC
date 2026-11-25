"""Timing instrumentation for the point-query 25-step DDIM planner."""
from evaluation.full_diffusion2_fast_agent import FullDiffusionTiming
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusion_pointquery.carla_agent import PointQueryAgent


def get_entry_point():
    return "PointQueryFastAgent"


class PointQueryFastAgent(FullDiffusionTiming, PointQueryAgent):
    method = "full_diffusion_pointquery"
    inference_steps = 25
    sampler_type = SpatialTemporalVPFullDiffusion
    log_prefix = "POINTQUERY_E2E"

    def extra_telemetry(self):
        return {"architecture": self.nets[0].run_config["architecture"], "ego_condition_tokens": 12,
                "direct_scene_memory_attention": False, "world_jointly_trained": False,
                "spatial_point_queries": 10, "temporal_point_queries": 6, "samples_per_point": 5}

