"""Pure Gaussian V3 planning with causal observed BEVs and geometric control."""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from method.full_diffusion2.inference.policy import TemporalBEVHistory
from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead


class FullDiffusionV3Network(nn.Module):
    def __init__(self, garage_net, checkpoint_path, device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.run_config = checkpoint["config"]
        dataset = checkpoint["dataset"]
        if self.run_config.get("method") != "full_diffusionv3" or dataset.get("method") != "full_diffusionv3":
            raise ValueError("V3 inference requires its own trained checkpoint")
        if (self.run_config.get("architecture") != "independent_dual_query64"
                or self.run_config.get("controller") != "automot_temporal_pid_v1"):
            raise ValueError("V3 requires the independent dual-query checkpoint and AutoMoT temporal PID")
        artifact = dataset["v3_normalization"]
        if artifact.get("method") != "full_diffusionv3" or artifact["label_config"] != self.run_config["labels"]:
            raise ValueError("V3 checkpoint normalization and label settings differ")
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.head = SpatialTemporalFullDiffusionHead(**self.run_config["model"],
                                                    normalization=artifact["normalization"])
        self.head.load_state_dict(checkpoint["model"], strict=True)
        self.diffusion = SpatialTemporalVPFullDiffusion(self.head.codec, **{
            key: value for key, value in self.run_config["diffusion"].items() if key != "inference_steps"
        })
        self.history = TemporalBEVHistory(self.run_config["model"]["history_frames"],
                                          self.run_config["observation_interval_seconds"])
        self.generator = torch.Generator(device=device).manual_seed(self.run_config["seed"])
        self.last_target_speed = None
        self.last_temporal_delta = None
        self.last_behavior_probabilities = None
        if (garage_net.config.speed_kp, garage_net.config.speed_ki,
                garage_net.config.speed_kd, garage_net.config.speed_n) != (1.75, 1.0, 2.0, 20):
            raise ValueError("V3 requires the AutoMoT longitudinal PID gains and history length")
        self.to(device)
        self.eval()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("perception").net, name)

    def set_observation(self, pose, timestamp):
        self.history.set_observation(pose, timestamp)

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        batch = {"rgb": rgb, "lidar": lidar_bev, "target_point": target_point,
                 "target_point_next": target_point_next, "speed": ego_vel, "command": command}
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if rgb.is_cuda and self.run_config["precision"] == "bf16" else nullcontext())
        with context:
            bev, memory, ego = self.perception(batch)
            history_bev, transforms, mask = self.history.get_inputs(bev)
            condition = self.head.prepare_condition(bev, memory, ego, target_point, history_bev,
                                                    transforms, mask, ego_vel, command)
            sample = self.diffusion.sample(
                lambda noisy, time_step: self.head(noisy, time_step, condition)["predictions"][-1],
                len(rgb), rgb.device, self.generator, self.run_config["diffusion"]["inference_steps"],
            )
            path, delta = self.head.codec.decode(sample)
            speed = self.head.codec.target_speed(delta)
            if not torch.isfinite(path).all() or not torch.isfinite(delta).all() or not torch.isfinite(speed).all():
                raise FloatingPointError("V3 generated nonfinite path or temporal displacement")
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        self.last_target_speed = float(speed[0])
        self.last_temporal_delta = delta.float().detach()
        self.last_behavior_probabilities = condition["behavior_logits"].float().softmax(dim=-1).detach()
        bins = torch.tensor(self.config.target_speeds, device=speed.device, dtype=speed.dtype)
        compatibility_speed = speed.clamp(0, max(self.config.target_speeds))
        upper = torch.searchsorted(bins, compatibility_speed).clamp(1, len(bins) - 1)
        weight = (compatibility_speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device, dtype=speed.dtype)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None

    def control_pid_direct(self, checkpoints, predicted_class_speed, current_speed):
        if self.last_target_speed is None:
            raise RuntimeError("V3 must predict before invoking the controller")
        speed = float(current_speed.reshape(-1)[0].detach().cpu())
        reference_speed = self.last_target_speed
        if not np.isfinite(speed) or not np.isfinite(reference_speed):
            raise FloatingPointError("V3 control requires finite current and temporal reference speeds")
        brake = reference_speed < 0.4 or speed / max(reference_speed, 1e-5) > 1.1
        delta = np.clip(reference_speed - speed, 0.0, 1.0)
        throttle = np.clip(self.perception.net.speed_controller.step(delta), 0.0, 1.0)
        throttle = 0.0 if brake else float(throttle)
        steer = self.perception.net.lateral_pid_controller.step(checkpoints, speed, 0, 0)
        steer = round(float(np.clip(steer, -1.0, 1.0)), 3)
        return steer, throttle, bool(brake)

    @staticmethod
    def apply_speed_limit(control, current_speed):
        if current_speed * 3.6 > 35:
            control.throttle = 0.0
            control.brake = 1.0
        return control