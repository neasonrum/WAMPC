"""Full Diffusion outputs with the existing Garage controller interface."""

from contextlib import nullcontext

import torch
from torch import nn

from ..diffusion import VPFullDiffusion
from ..garage import FrozenGaragePerception
from ..model import TrajectoryFullDiffuGeoHead


class FullDiffusionNetwork(nn.Module):
    def __init__(self, garage_net, checkpoint_path, device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.run_config = checkpoint["config"]
        self.perception = FrozenGaragePerception.from_net(garage_net)
        self.head = TrajectoryFullDiffuGeoHead(**self.run_config["model"])
        self.head.load_state_dict(checkpoint["model"], strict=True)
        self.diffusion = VPFullDiffusion(self.head.codec, **{
            key: value for key, value in self.run_config["diffusion"].items() if key != "inference_steps"
        })
        self.last_target_speed = None
        self.generator = torch.Generator(device=device).manual_seed(self.run_config["seed"])
        self.to(device)
        self.eval()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("perception").net, name)

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        batch = {"rgb": rgb, "lidar": lidar_bev, "target_point": target_point,
                 "target_point_next": target_point_next, "speed": ego_vel, "command": command}
        context = torch.autocast("cuda", dtype=torch.bfloat16) if rgb.is_cuda else nullcontext()
        with context:
            bev, memory, ego = self.perception(batch)
            condition = self.head.prepare_condition(bev, memory, ego, target_point)
            sample = self.diffusion.sample(
                lambda noisy, timestep: self.head(noisy, timestep, condition)[-1],
                len(rgb), rgb.device, self.generator, self.run_config["diffusion"]["inference_steps"],
            )
            path, speed = self.head.codec.decode(sample)
            if not torch.isfinite(path).all() or not torch.isfinite(speed).all():
                raise FloatingPointError("Full diffusion generated a nonfinite control target")
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        speed = speed.clamp(0, max(self.config.target_speeds))
        self.last_target_speed = float(speed[0])
        bins = torch.tensor(self.config.target_speeds, device=speed.device)
        upper = torch.searchsorted(bins, speed).clamp(1, len(bins) - 1)
        weight = (speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        logits = probabilities.clamp_min(1e-12).log()
        return None, logits, path.float(), None, None, None, boxes, None, None, None

    def control_pid_direct(self, checkpoints, predicted_class_speed, current_speed):
        if self.last_target_speed is None:
            raise RuntimeError("The diffusion policy must predict before control")
        return self.perception.net.control_pid_direct(checkpoints, predicted_class_speed, current_speed)