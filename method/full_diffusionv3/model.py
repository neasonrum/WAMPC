"""Separate 256D spatial and temporal queries with 64D point encodings."""

import torch
from torch import nn
from torch.nn import functional

from method.full_diffusion.model import GeoDecoderLayer, SinusoidalPosEmb, blocks
from method.full_diffusion2.model import TemporalBEVAdapter
from method.full_diffusionv3.diffusion import SpatialTemporalCodec


class SpatialTemporalDecoderLayer(GeoDecoderLayer):
    def __init__(self, points, temporal=False, **kwargs):
        super().__init__(path_points=points, **kwargs)
        d_model = kwargs.get("d_model", 256)
        self.temporal = temporal
        self.task_decoder[-1] = nn.Linear(d_model, points * 2)

    def forward(self, query, metric_points, bev_values, memory, ego, time_embedding):
        attention = self.cross_bev_attention
        references = metric_points.cumsum(dim=1) if self.temporal else metric_points
        grid = references[:, None] / 32.0
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


class SpatialTemporalFullDiffusionHead(nn.Module):
    def __init__(self, path_points=10, temporal_points=6, slot_width=64,
                 d_model=256, bev_channels=64, d_ffn=1024, layers=2, dropout=0.1,
                 history_frames=2, behavior_width=128, path_scale=20.0,
                 speed_scale=20.0, temporal_interval_seconds=0.5, normalization=None):
        super().__init__()
        if (path_points, temporal_points, slot_width, d_model) != (10, 6, 64, 256):
            raise ValueError("V3 requires 64D point encodings and two independent 256D queries")
        if history_frames != 2 or layers < 1 or path_scale <= 0 or speed_scale <= 0:
            raise ValueError("Invalid V3 observation window, decoder depth, or state scales")
        self.history_frames = history_frames
        self.slot_width = slot_width
        self.path_scale = path_scale
        self.speed_scale = speed_scale
        self.codec = SpatialTemporalCodec(path_points, temporal_points, temporal_interval_seconds,
                                          **(normalization or {}))
        self.register_buffer("type_ids", torch.tensor([0] * path_points + [1] * temporal_points))
        self.register_buffer("slot_ids", torch.arange(self.codec.total_points))
        self.type_embedding = nn.Embedding(2, slot_width)
        self.slot_embedding = nn.Embedding(self.codec.total_points, slot_width)
        self.path_query_encoder = nn.Sequential(
            nn.Linear(path_points * slot_width, d_model), nn.ReLU(), nn.LayerNorm(d_model),
        )
        self.temporal_query_encoder = nn.Sequential(
            nn.Linear(temporal_points * slot_width, d_model), nn.ReLU(), nn.LayerNorm(d_model),
        )
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(d_model), nn.Linear(d_model, 4 * d_model),
                                      nn.Mish(), nn.Linear(4 * d_model, d_model))
        self.temporal_time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, 4 * d_model), nn.Mish(),
            nn.Linear(4 * d_model, d_model),
        )
        self.target_point_encoder = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.temporal = TemporalBEVAdapter(bev_channels, history_frames + 1)
        self.behavior_head = nn.Sequential(
            nn.Linear(bev_channels + 2 * d_model + 9, behavior_width), nn.ReLU(),
            nn.LayerNorm(behavior_width), nn.Linear(behavior_width, 12),
        )
        self.behavior_condition = nn.Linear(12, d_model)
        self.temporal_behavior_condition = nn.Linear(12, d_model)
        nn.init.zeros_(self.behavior_condition.weight)
        nn.init.zeros_(self.behavior_condition.bias)
        nn.init.zeros_(self.temporal_behavior_condition.weight)
        nn.init.zeros_(self.temporal_behavior_condition.bias)
        self.layers = nn.ModuleList([
            SpatialTemporalDecoderLayer(path_points, bev_channels=bev_channels,
                                        d_model=d_model, d_ffn=d_ffn, dropout=dropout)
            for _ in range(layers)
        ])
        self.temporal_layers = nn.ModuleList([
            SpatialTemporalDecoderLayer(temporal_points, temporal=True, bev_channels=bev_channels,
                                        d_model=d_model, d_ffn=d_ffn, dropout=dropout)
            for _ in range(layers)
        ])

    def prepare_condition(self, bev, memory, ego, target_point, history_bev,
                          history_transforms, history_mask, speed, command):
        batch = len(bev)
        if history_bev.shape[:2] != (batch, self.history_frames):
            raise ValueError("V3 requires exactly two historical BEVs")
        state = torch.cat((speed.reshape(-1, 1) / self.speed_scale,
                           command, target_point / self.path_scale), dim=-1).to(bev.dtype)
        if state.shape != (batch, 9):
            raise ValueError("Expected speed, six command dimensions, and target XY")
        sequence = torch.cat((history_bev, bev[:, None]), dim=1)
        identity = torch.eye(3, device=bev.device, dtype=history_transforms.dtype)
        transforms = torch.cat((history_transforms, identity.expand(batch, 1, 3, 3)), dim=1)
        valid = torch.cat((history_mask, history_mask.new_ones(batch, 1)), dim=1)
        planner_bev = self.temporal(sequence, transforms, valid, state)
        context = torch.cat((planner_bev.mean(dim=(-2, -1)), memory.mean(dim=1),
                             ego.mean(dim=1), state), dim=-1)
        logits = self.behavior_head(context).reshape(batch, 3, 4)
        probabilities = logits.softmax(dim=-1)
        target_token = self.target_point_encoder(target_point / self.path_scale)[:, None]
        return {
            "bev": [layer.cross_bev_attention.value_proj(planner_bev) for layer in self.layers],
            "temporal_bev": [layer.cross_bev_attention.value_proj(planner_bev) for layer in self.temporal_layers],
            "memory": memory,
            "ego": torch.cat((ego, target_token), dim=1),
            "behavior_logits": logits,
            "behavior_offset": self.behavior_condition(probabilities.flatten(1))[:, None],
            "temporal_behavior_offset": self.temporal_behavior_condition(probabilities.flatten(1))[:, None],
        }

    def forward(self, noisy, time, condition):
        if isinstance(condition, tuple):
            condition = self.prepare_condition(*condition)
        self.codec.check_shape(noisy)
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=self.slot_width)
        features = features + self.type_embedding(self.type_ids)[None]
        features = features + self.slot_embedding(self.slot_ids)[None]
        path_query = self.path_query_encoder(features[:, :self.codec.path_points].flatten(start_dim=1))[:, None]
        temporal_query = self.temporal_query_encoder(features[:, self.codec.path_points:].flatten(start_dim=1))[:, None]
        path_query = path_query + condition["behavior_offset"]
        temporal_query = temporal_query + condition["temporal_behavior_offset"]
        time_embedding = self.time_mlp(time * 1000)[:, None]
        temporal_time_embedding = self.temporal_time_mlp(time * 1000)[:, None]
        metric_points = self.codec.to_metric(noisy)
        predictions = []
        for layer, temporal_layer, bev_values, temporal_bev in zip(
                self.layers, self.temporal_layers, condition["bev"], condition["temporal_bev"]):
            spatial = layer(path_query, metric_points[:, :self.codec.path_points], bev_values,
                            condition["memory"], condition["ego"], time_embedding)
            temporal = temporal_layer(temporal_query, metric_points[:, self.codec.path_points:], temporal_bev,
                                      condition["memory"], condition["ego"], temporal_time_embedding)
            metric_prediction = torch.cat((spatial, temporal), dim=1)
            normalized_prediction = self.codec.encode_metric(metric_prediction)
            predictions.append(normalized_prediction)
            metric_points = self.codec.to_metric(normalized_prediction.detach())
        return {"predictions": torch.stack(predictions),
                "behavior_logits": condition["behavior_logits"]}

    def loss_components(self, output, clean, spatial_mask, temporal_mask,
                        behavior_targets, behavior_mask, loss_weights=None):
        weights = loss_weights or {}
        temporal_weights = weights.get("temporal_interval_weights", [3, 3, 2, 2, 1, 1])
        metric_clean = self.codec.to_metric(clean)
        spatial_loss = output["predictions"].sum() * 0.0
        temporal_loss = spatial_loss
        for prediction in output["predictions"]:
            metric = self.codec.to_metric(prediction)
            spatial_loss = spatial_loss + self.codec.masked_l1(
                metric[:, :10], metric_clean[:, :10], spatial_mask,
            )
            temporal_loss = temporal_loss + self.codec.masked_l1(
                metric[:, 10:], metric_clean[:, 10:], temporal_mask, temporal_weights,
            )
        logits = output["behavior_logits"].float()
        if logits.shape[:-1] != behavior_targets.shape or behavior_mask.shape != behavior_targets.shape:
            raise ValueError("Expected B x 3 behavior labels and masks")
        valid = behavior_mask.to(device=logits.device, dtype=torch.bool)
        targets = behavior_targets.to(device=logits.device, dtype=torch.long)
        targets = torch.where(valid, targets, -100)
        errors = functional.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1),
                                           reduction="none", ignore_index=-100).reshape(targets.shape)
        behavior_loss = errors.sum() / valid.sum().clamp_min(1)
        loss = (weights.get("spatial_weight", 1.0) * spatial_loss
                + weights.get("temporal_weight", 2.0) * temporal_loss
                + weights.get("behavior_weight", 0.4) * behavior_loss)
        return {"loss": loss, "spatial_l1": spatial_loss,
                "temporal_delta_l1": temporal_loss, "behavior_ce": behavior_loss}