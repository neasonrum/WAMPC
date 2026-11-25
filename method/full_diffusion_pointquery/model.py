"""Independent 10/6 point queries with bounded, learned local BEV reads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from method.full_diffusionv3_ego12.model import (
    SpatialTemporalDecoderLayer, SpatialTemporalFullDiffusionHead, blocks,
)


def masked_softmax(logits, valid):
    """All-invalid neighborhoods contribute zero, including their gradients."""
    weights = logits.float().masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
    weights = weights * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


class PointDecoderLayer(SpatialTemporalDecoderLayer):
    def __init__(self, points, temporal=False, samples_per_point=5,
                 sampling_radius_m=2.0, initial_offset_m=1.0, **kwargs):
        super().__init__(points, temporal=temporal, **kwargs)
        if samples_per_point != 5 or not 0 < initial_offset_m < sampling_radius_m:
            raise ValueError("Use the center and four bounded learned neighbors")
        width = kwargs.get("d_model", 256)
        self.samples_per_point = samples_per_point
        self.sampling_radius_m = float(sampling_radius_m)
        self.self_attention = nn.MultiheadAttention(width, 8, dropout=kwargs.get("dropout", .1), batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.sampling_offsets = nn.Linear(width, (samples_per_point - 1) * 2)
        self.cross_bev_attention.attention_weights = nn.Linear(width, samples_per_point)
        self.task_decoder[-1] = nn.Linear(width, 2)
        nn.init.normal_(self.task_decoder[-1].weight, std=.01)
        nn.init.zeros_(self.task_decoder[-1].bias)
        nn.init.zeros_(self.sampling_offsets.weight)
        v = math.atanh(initial_offset_m / sampling_radius_m)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(torch.tensor([v, 0., -v, 0., 0., v, 0., -v]))
            self.cross_bev_attention.attention_weights.weight.zero_()
            # Initial center probability .8; each neighbor .05.
            self.cross_bev_attention.attention_weights.bias.copy_(torch.tensor([math.log(16.), 0., 0., 0., 0.]))
        self.debug = False
        self.last_read = None

    def local_read(self, query, metric_points, bev_values):
        references = metric_points.cumsum(1) if self.temporal else metric_points
        # Grid x/y and +/-32m extent exactly follow the existing B2D ego12 reader.
        # Coordinates stay FP32: BF16 rounding at 32m would erase small offsets.
        offsets = self.sampling_offsets(query).float().reshape(*references.shape[:2], 4, 2)
        offsets = self.sampling_radius_m * offsets.tanh()
        offsets = torch.cat((torch.zeros_like(offsets[:, :, :1]), offsets), 2)
        coordinates = references.float()[:, :, None] + offsets
        grid = coordinates / 32.
        valid = torch.isfinite(grid).all(-1) & (grid.abs() < 1.).all(-1)
        # CPU and CUDA use the same differentiable bilinear interpolation contract.
        with torch.autocast(device_type=bev_values.device.type, enabled=False):
            sampled = F.grid_sample(bev_values.float(), grid.float(), align_corners=False,
                                    padding_mode="zeros").permute(0, 2, 3, 1)
        weights = masked_softmax(self.cross_bev_attention.attention_weights(query), valid)
        pooled = (sampled * weights[..., None]).sum(2).to(query.dtype)
        update = self.cross_bev_attention.output_proj(pooled)
        # Suppress output-projection bias when no BEV position is supported.
        update = update * valid.any(-1, keepdim=True).to(update.dtype)
        if self.debug:
            self.last_read = {"coordinates": coordinates.detach(), "offsets": offsets.detach(),
                              "weights": weights.detach(), "valid": valid.detach(),
                              "pooled": pooled.detach()}
        return update

    def forward(self, query, metric_points, bev_values, ego, time_embedding):
        query = self.self_norm(query + self.dropout(
            self.self_attention(query, query, query, need_weights=False)[0]))
        query = self.norm1(query + self.cross_bev_attention.dropout(
            self.local_read(query, metric_points, bev_values)))
        query = self.norm2(query + self.dropout(
            self.cross_ego_attention(query, ego, ego, need_weights=False)[0]))
        # Preserve the original ego12 FFN and timestep modulation semantics.
        query = self.norm3(self.ffn(query))
        scale, shift = self.time_modulation(time_embedding).chunk(2, dim=-1)
        query = query * (1 + scale) + shift
        return query, metric_points + self.task_decoder(query).float()


class PointQueryHead(SpatialTemporalFullDiffusionHead):
    def __init__(self, samples_per_point=5, sampling_radius_m=2., initial_offset_m=1., **kwargs):
        super().__init__(**kwargs)
        width = kwargs.get("d_model", 256)
        self.path_query_encoder = nn.Sequential(nn.Linear(self.slot_width, width), nn.ReLU(), nn.LayerNorm(width))
        self.temporal_query_encoder = nn.Sequential(nn.Linear(self.slot_width, width), nn.ReLU(), nn.LayerNorm(width))
        settings = {k: kwargs[k] for k in ("bev_channels", "d_model", "d_ffn", "dropout") if k in kwargs}
        settings.update(samples_per_point=samples_per_point, sampling_radius_m=sampling_radius_m,
                        initial_offset_m=initial_offset_m)
        depth = kwargs.get("layers", 2)
        self.layers = nn.ModuleList([PointDecoderLayer(10, **settings) for _ in range(depth)])
        self.temporal_layers = nn.ModuleList([PointDecoderLayer(6, temporal=True, **settings) for _ in range(depth)])

    def forward(self, noisy, time, condition):
        if isinstance(condition, tuple):
            condition = self.prepare_condition(*condition)
        self.codec.check_shape(noisy)
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=self.slot_width)
        features = features + self.type_embedding(self.type_ids)[None] + self.slot_embedding(self.slot_ids)[None]
        spatial_query = self.path_query_encoder(features[:, :10]) + condition["behavior_offset"]
        temporal_query = self.temporal_query_encoder(features[:, 10:]) + condition["temporal_behavior_offset"]
        time_embedding = self.time_mlp(time * 1000)[:, None]
        temporal_time = self.temporal_time_mlp(time * 1000)[:, None]
        metric = self.codec.to_metric(noisy)
        predictions = []
        for layer, temporal_layer, values, temporal_values in zip(
                self.layers, self.temporal_layers, condition["bev"], condition["temporal_bev"]):
            spatial_query, spatial = layer(spatial_query, metric[:, :10], values, condition["ego"], time_embedding)
            temporal_query, temporal = temporal_layer(temporal_query, metric[:, 10:], temporal_values,
                                                       condition["ego"], temporal_time)
            prediction = self.codec.encode_metric(torch.cat((spatial, temporal), 1))
            predictions.append(prediction)
            # Refine references between stages while carrying the trainable point tokens.
            metric = self.codec.to_metric(prediction.detach())
        return {"predictions": torch.stack(predictions), "behavior_logits": condition["behavior_logits"]}


def load_compatible_ego12(head, checkpoint):
    source, target = checkpoint["model"], head.state_dict()
    compatible = {k: v for k, v in source.items() if k in target and v.shape == target[k].shape}
    # Coordinate normalization may never be silently replaced by another protocol.
    for key in ("codec.mean", "codec.std"):
        if not torch.equal(source[key].cpu(), target[key].cpu()):
            raise ValueError("Pretrained and fixed-protocol coordinate normalization differ")
    result = head.load_state_dict(compatible, strict=False)
    return {"loaded_tensors": len(compatible), "loaded_parameters": sum(v.numel() for v in compatible.values()),
            "new_or_resized": result.missing_keys,
            "not_reused": [k for k in source if k not in compatible],
            "mode": "compatible ego12 epoch10 tensors; point encoders/output heads/offsets/self-attention initialized separately"}

