"""Geo adaptation of BridgeDrive's NAVSIM TrajectoryFullDiffuHead."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional

from .diffusion import GeoCodec


BRIDGE_MODULES = (
    Path(__file__).resolve().parents[3] / "BridgeDrive" /
    "BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules"
)
BRIDGE_MODULES = Path(os.environ.get("WMMPC_BRIDGE_MODULES", str(BRIDGE_MODULES)))


def load_bridge_module(name):
    spec = importlib.util.spec_from_file_location(f"wmmpc_bridge_{name}", BRIDGE_MODULES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blocks = load_bridge_module("blocks")
SinusoidalPosEmb = load_bridge_module("conditional_unet1d").SinusoidalPosEmb


class GeoDecoderLayer(nn.Module):
    def __init__(self, path_points=10, bev_channels=64, d_model=256, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.path_points = path_points
        config = SimpleNamespace(lidar_max_x=32, lidar_max_y=32)
        self.cross_bev_attention = blocks.GridSampleCrossBEVAttention(
            d_model, 8, in_bev_dims=bev_channels, num_points=path_points, config=config
        )
        self.cross_agent_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.cross_ego_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
        self.time_modulation = nn.Sequential(nn.Mish(), nn.Linear(d_model, 2 * d_model))
        self.task_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, (path_points + 1) * 2),
        )

    def forward(self, query, metric_points, bev_values, memory, ego, time_embedding):
        attention = self.cross_bev_attention
        grid = metric_points[:, None, :self.path_points] / 32.0
        sampled = functional.grid_sample(bev_values, grid, align_corners=False, padding_mode="zeros")
        weights = attention.attention_weights(query).softmax(-1)
        sampled = (sampled * weights[:, None]).sum(-1).transpose(1, 2)
        query = query + attention.dropout(attention.output_proj(sampled))
        query = self.norm1(query + self.dropout(
            self.cross_agent_attention(query, memory, memory, need_weights=False)[0]
        ))
        query = self.norm2(query + self.dropout(
            self.cross_ego_attention(query, ego, ego, need_weights=False)[0]
        ))
        query = self.norm3(self.ffn(query))
        scale, shift = self.time_modulation(time_embedding).chunk(2, dim=-1)
        query = query * (1 + scale) + shift
        return metric_points + self.task_decoder(query).reshape(metric_points.shape)


class TrajectoryFullDiffuGeoHead(nn.Module):
    def __init__(self, path_points=10, bev_channels=64, d_model=256, d_ffn=1024,
                 layers=2, dropout=0.1, path_scale=20.0, speed_scale=20.0, normalization=None):
        super().__init__()
        if d_model != 256:
            raise ValueError("The upstream BEV projection expects d_model=256")
        self.codec = GeoCodec(path_points, path_scale, speed_scale, **(normalization or {}))
        self.noisy_path_encoder = nn.Sequential(
            *blocks.linear_relu_ln(d_model, 1, 1, 64 * (path_points + 1)),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, 4 * d_model), nn.Mish(),
            nn.Linear(4 * d_model, d_model),
        )
        self.target_point_encoder = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList([
            GeoDecoderLayer(path_points, bev_channels, d_model, d_ffn, dropout) for _ in range(layers)
        ])

    def to_metric(self, normalized):
        path, speed = self.codec.decode(normalized)
        speed_token = torch.stack((speed, torch.zeros_like(speed)), dim=-1)
        return torch.cat((path, speed_token[:, None]), dim=1)

    def to_normalized(self, metric):
        return self.codec.encode(metric[:, :-1], metric[:, -1, 0])

    def prepare_condition(self, bev, memory, ego, target_point):
        target_token = self.target_point_encoder(target_point / self.codec.path_scale)[:, None]
        return {
            "bev": [layer.cross_bev_attention.value_proj(bev) for layer in self.layers],
            "memory": memory,
            "ego": torch.cat((ego, target_token), dim=1),
        }

    def forward(self, noisy, time, condition):
        if isinstance(condition, tuple):
            condition = self.prepare_condition(*condition)
        metric_points = self.to_metric(noisy)
        embedding = blocks.gen_sineembed_for_position(metric_points, hidden_dim=64).flatten(-2)
        query = self.noisy_path_encoder(embedding)[:, None]
        time_embedding = self.time_mlp(time * 1000)[:, None]
        predictions = []
        for layer, bev_values in zip(self.layers, condition["bev"]):
            metric_prediction = layer(
                query, metric_points, bev_values, condition["memory"], condition["ego"], time_embedding
            )
            normalized_prediction = self.to_normalized(metric_prediction)
            predictions.append(normalized_prediction)
            metric_points = self.to_metric(normalized_prediction.detach())
        return torch.stack(predictions)

    def loss(self, predictions, clean):
        metric_clean = self.to_metric(clean)
        return sum(self.codec.loss(self.to_metric(prediction), metric_clean) for prediction in predictions)