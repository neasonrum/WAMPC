"""Causal online BEV history with the unchanged Garage control/detection interface."""

from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from method.full_diffusion.diffusion import VPFullDiffusion
from method.full_diffusion2.data import frame_pose
from method.full_diffusion2.model import TemporalFullDiffusionHead
from method.full_diffusion2.perception import FrozenPlannerPerception


class TemporalBEVHistory:
    def __init__(self, frames=2, interval=0.25, tolerance=0.075):
        self.frames = frames
        self.interval = interval
        self.tolerance = tolerance
        self.entries = deque(maxlen=32)
        self.pose = None
        self.timestamp = None

    def set_observation(self, pose, timestamp):
        timestamp = float(timestamp)
        if self.entries and timestamp <= self.entries[-1][0]:
            self.entries.clear()
        self.pose = frame_pose({"theta": float(pose[2]), "pos_global": pose[:2]})
        self.timestamp = timestamp

    def get_inputs(self, current):
        if self.pose is None or self.timestamp is None or current.shape[0] != 1:
            raise RuntimeError("Online BEV history requires the current online ego pose, timestamp and batch1")
        features, transforms, valid = [], [], []
        for offset in range(self.frames, 0, -1):
            target = self.timestamp - offset * self.interval
            candidates = [entry for entry in self.entries if entry[0] <= target + 1e-6]
            selected = candidates[-1] if candidates else None
            exists = selected is not None and target - selected[0] <= self.tolerance
            features.append(selected[1] if exists else current)
            transforms.append(np.linalg.solve(self.pose, selected[2]) if exists else np.eye(3))
            valid.append(exists)
        self.entries.append((self.timestamp, current.detach(), self.pose.copy()))
        return (torch.stack(features, dim=1),
                torch.tensor(np.stack(transforms), device=current.device, dtype=torch.float32)[None],
                torch.tensor([valid], device=current.device))


class FullDiffusion2Network(nn.Module):
    def __init__(self, garage_net, checkpoint_path, device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.run_config = checkpoint["config"]
        if self.run_config.get("method") != "full_diffusion2":
            raise ValueError("Full Diffusion2 inference requires its own trained checkpoint")
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.head = TemporalFullDiffusionHead(**self.run_config["model"])
        self.head.load_state_dict(checkpoint["model"], strict=True)
        self.diffusion = VPFullDiffusion(self.head.codec, **{
            key: value for key, value in self.run_config["diffusion"].items() if key != "inference_steps"
        })
        self.history = TemporalBEVHistory(self.run_config["model"]["history_frames"],
                                          self.run_config["observation_interval_seconds"])
        self.last_target_speed = None
        self.generator = torch.Generator(device=device).manual_seed(self.run_config["seed"])
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
        context = torch.autocast("cuda", dtype=torch.bfloat16) if rgb.is_cuda else nullcontext()
        with context:
            bev, memory, ego = self.perception(batch)
            history_bev, transforms, mask = self.history.get_inputs(bev)
            condition = self.head.prepare_condition(bev, memory, ego, target_point, history_bev,
                                                    transforms, mask, ego_vel, command)
            sample = self.diffusion.sample(
                lambda noisy, timestep: self.head(noisy, timestep, condition)[-1],
                len(rgb), rgb.device, self.generator, self.run_config["diffusion"]["inference_steps"],
            )
            path, speed = self.head.codec.decode(sample)
            if not torch.isfinite(path).all() or not torch.isfinite(speed).all():
                raise FloatingPointError("Full Diffusion2 generated nonfinite control targets")
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        speed = speed.clamp(0, max(self.config.target_speeds))
        self.last_target_speed = float(speed[0])
        bins = torch.tensor(self.config.target_speeds, device=speed.device)
        upper = torch.searchsorted(bins, speed).clamp(1, len(bins) - 1)
        weight = (speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None

    def control_pid_direct(self, checkpoints, predicted_class_speed, current_speed):
        if self.last_target_speed is None:
            raise RuntimeError("The diffusion policy must predict before control")
        return self.perception.net.control_pid_direct(checkpoints, predicted_class_speed, current_speed)