"""Planner-only temporal BEV adaptation and clean-state query conditioning."""

import torch
from torch import nn
from torch.nn import functional

from method.full_diffusion.model import TrajectoryFullDiffuGeoHead, blocks


def align_bev(bev, source_to_current, extent=32.0):
    batch, channels, height, width = bev.shape
    with torch.autocast(device_type=bev.device.type, enabled=False):
        inverse = torch.linalg.inv(source_to_current.float())
        theta = inverse[:, :2].clone()
        theta[:, :, 2] /= extent
        grid = functional.affine_grid(theta, (batch, channels, height, width), align_corners=False)
        aligned = functional.grid_sample(bev.float(), grid, align_corners=False, padding_mode="zeros")
    return aligned.to(bev.dtype)


class TemporalBEVAdapter(nn.Module):
    def __init__(self, channels=64, frames=3, state_dim=9, extent=32.0):
        super().__init__()
        self.frames = frames
        self.extent = extent
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * frames, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.ReLU(),
        )
        self.gate = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                  nn.Linear(128, channels), nn.Sigmoid())
        self.output_projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, sequence, transforms, valid, state):
        batch, frames, channels, height, width = sequence.shape
        if frames != self.frames or transforms.shape != (batch, frames, 3, 3):
            raise ValueError("Temporal BEV sequence or transform shape differs from the configured window")
        if valid.shape != (batch, frames):
            raise ValueError("Temporal BEV validity mask has the wrong shape")
        aligned = align_bev(sequence.reshape(-1, channels, height, width),
                            transforms.reshape(-1, 3, 3), self.extent)
        aligned = aligned.reshape(batch, frames, channels, height, width)
        aligned = aligned * valid[:, :, None, None, None].to(aligned.dtype)
        fused = self.fusion(aligned.flatten(1, 2))
        gated = fused * self.gate(state)[:, :, None, None]
        return sequence[:, -1] + self.output_projection(gated)


class TemporalFullDiffusionHead(TrajectoryFullDiffuGeoHead):
    def __init__(self, state_width=128, history_frames=2, query_state_conditioning=True, **kwargs):
        super().__init__(**kwargs)
        self.history_frames = history_frames
        self.query_state_conditioning = bool(query_state_conditioning)
        self.temporal = TemporalBEVAdapter(kwargs.get("bev_channels", 64), history_frames + 1)
        if self.query_state_conditioning:
            self.state_encoder = nn.Sequential(nn.Linear(9, state_width), nn.ReLU(), nn.LayerNorm(state_width))
            self.query_norm = nn.LayerNorm(256)
            self.query_condition = nn.Linear(256 + state_width, 256)
            nn.init.zeros_(self.query_condition.weight)
            nn.init.zeros_(self.query_condition.bias)

    def prepare_condition(self, bev, memory, ego, target_point, history_bev,
                          history_transforms, history_mask, speed, command):
        batch = bev.shape[0]
        if history_bev.shape[:2] != (batch, self.history_frames):
            raise ValueError("The planner requires exactly the configured number of historical BEVs")
        state = torch.cat((speed.reshape(-1, 1) / self.codec.speed_scale,
                           command, target_point / self.codec.path_scale), dim=-1).to(bev.dtype)
        sequence = torch.cat((history_bev, bev[:, None]), dim=1)
        identity = torch.eye(3, device=bev.device, dtype=history_transforms.dtype)
        transforms = torch.cat((history_transforms, identity.expand(batch, 1, 3, 3)), dim=1)
        valid = torch.cat((history_mask, history_mask.new_ones(batch, 1)), dim=1)
        planner_bev = self.temporal(sequence, transforms, valid, state)
        condition = super().prepare_condition(planner_bev, memory, ego, target_point)
        if self.query_state_conditioning:
            condition["state"] = self.state_encoder(state)[:, None]
        return condition

    def forward(self, noisy, time, condition):
        if isinstance(condition, tuple):
            condition = self.prepare_condition(*condition)
        metric_points = self.to_metric(noisy)
        embedding = blocks.gen_sineembed_for_position(metric_points, hidden_dim=64).flatten(-2)
        query = self.noisy_path_encoder(embedding)[:, None]
        if self.query_state_conditioning:
            query = query + self.query_condition(torch.cat((self.query_norm(query), condition["state"]), dim=-1))
        time_embedding = self.time_mlp(time * 1000)[:, None]
        predictions = []
        for layer, bev_values in zip(self.layers, condition["bev"]):
            prediction = layer(query, metric_points, bev_values, condition["memory"],
                               condition["ego"], time_embedding)
            normalized = self.to_normalized(prediction)
            predictions.append(normalized)
            metric_points = self.to_metric(normalized.detach())
        return torch.stack(predictions)