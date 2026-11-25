"""The existing 20Hz ego12 controller with the new point-query diffusion head."""
import torch
from torch import nn

from method.full_diffusion2.inference.policy import TemporalBEVHistory
from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3_ego12.inference.policy import Ego12Network
from . import FORMAT, METHOD
from .model import PointQueryHead


class PointQueryNetwork(Ego12Network):
    def __init__(self, garage_net, checkpoint_path, device):
        nn.Module.__init__(self)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT or state.get("method") != METHOD or state["progress"]["updates"] < 1:
            raise ValueError("Point-query inference requires its own trained checkpoint")
        self.run_config = state["config"]
        if (self.run_config["architecture"] != "independent_point_query_navigation_concat_v1"
                or self.run_config["conditioning"] != "tfpp_ego11_plus_state1"
                or self.run_config["diffusion"]["inference_steps"] != 25):
            raise ValueError("The trained point-query architecture and 25-step sampler must agree")
        normalization = state["dataset"]["normalization"]["normalization"]
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.head = PointQueryHead(**self.run_config["model"], normalization=normalization)
        self.head.load_state_dict(state["model"], strict=True)
        self.diffusion = SpatialTemporalVPFullDiffusion(self.head.codec, **{
            k: v for k, v in self.run_config["diffusion"].items() if k != "inference_steps"})
        self.history = TemporalBEVHistory(2, .25)
        self.generator = torch.Generator(device=device).manual_seed(self.run_config["seed"])
        self.last_target_speed = self.last_temporal_delta = self.last_behavior_probabilities = None
        if (garage_net.config.speed_kp, garage_net.config.speed_ki, garage_net.config.speed_kd,
                garage_net.config.speed_n) != (1.75, 1., 2., 20):
            raise ValueError("Preserve the existing AutoMoT longitudinal PID")
        self.requires_grad_(False)
        self.to(device)
        self.eval()

