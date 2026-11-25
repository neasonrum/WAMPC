"""V8 future-BEV readouts inserted into the existing ego12 denoiser."""
import torch
from torch import nn
from torch.nn import functional as F

from method.full_diffusion.model import blocks
from method.full_diffusionv3_ego12.model import SpatialTemporalFullDiffusionHead
from .future import FutureNeighborhoodSampler
from .readout import CachedFutureReadout


def decode_layer(layer, query, points, bev, ego, time_embedding, readout, future):
    attention = layer.cross_bev_attention
    references = points.cumsum(1) if layer.temporal else points
    sampled = F.grid_sample(bev, references[:, None] / 32., align_corners=False, padding_mode='zeros')
    weights = attention.attention_weights(query).softmax(-1)
    sampled = (sampled * weights[:, None]).sum(-1).transpose(1, 2)
    query = query + attention.dropout(attention.output_proj(sampled))
    query = layer.norm1(query)
    query = layer.norm2(query + layer.dropout(layer.cross_ego_attention(query, ego, ego, need_weights=False)[0]))
    if future is not None:
        query = readout(query, references, future)
    query = layer.norm3(layer.ffn(query))
    scale, shift = layer.time_modulation(time_embedding).chunk(2, dim=-1)
    query = query * (1 + scale) + shift
    return points + layer.task_decoder(query).reshape(points.shape)


class FutureAdaptedPlanner(nn.Module):
    def __init__(self, base, sampler_config):
        super().__init__()
        self.base = base
        self.base_training_mode = True
        self.adapter = nn.ModuleDict({
            'sampler': FutureNeighborhoodSampler(64, 256, sampler_config),
            'spatial': nn.ModuleList([CachedFutureReadout(256) for _ in base.layers]),
            'temporal': nn.ModuleList([CachedFutureReadout(256, temporal=True) for _ in base.temporal_layers]),
        })

    @property
    def codec(self):
        return self.base.codec

    def tokenize(self, maps, prior, valid, ego, motion):
        return self.adapter['sampler'](maps, prior, valid, ego.mean(1), motion)

    def condition(self, bev, memory, ego, target_point, history_bev, transforms, mask, speed, command, future=None):
        result = self.base.prepare_condition(bev, memory, ego, target_point, history_bev, transforms, mask, speed, command)
        result['future'] = [reader.prepare(future) for reader in self.adapter['spatial']]
        result['temporal_future'] = [reader.prepare(future) for reader in self.adapter['temporal']]
        return result

    def forward(self, noisy, time, condition):
        base, count = self.base, self.codec.path_points
        self.codec.check_shape(noisy)
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=base.slot_width)
        features = features + base.type_embedding(base.type_ids)[None] + base.slot_embedding(base.slot_ids)[None]
        path_query = base.path_query_encoder(features[:, :count].flatten(1))[:, None] + condition['behavior_offset']
        temporal_query = base.temporal_query_encoder(features[:, count:].flatten(1))[:, None] + condition['temporal_behavior_offset']
        path_time = base.time_mlp(time * 1000)[:, None]
        temporal_time = base.temporal_time_mlp(time * 1000)[:, None]
        metric, predictions = self.codec.to_metric(noisy), []
        for index, (layer, temporal_layer) in enumerate(zip(base.layers, base.temporal_layers)):
            path = decode_layer(layer, path_query, metric[:, :count], condition['bev'][index], condition['ego'],
                                path_time, self.adapter['spatial'][index], condition['future'][index])
            temporal = decode_layer(temporal_layer, temporal_query, metric[:, count:], condition['temporal_bev'][index],
                                    condition['ego'], temporal_time, self.adapter['temporal'][index],
                                    condition['temporal_future'][index])
            normalized = self.codec.encode_metric(torch.cat((path, temporal), 1))
            predictions.append(normalized)
            metric = self.codec.to_metric(normalized.detach())
        return {'predictions': torch.stack(predictions), 'behavior_logits': condition['behavior_logits']}

    def gates(self):
        return {f'{branch}{index}': float(reader.gate.detach().tanh())
                for branch in ('spatial', 'temporal') for index, reader in enumerate(self.adapter[branch])}


def build_planner(config, normalization):
    base = SpatialTemporalFullDiffusionHead(**config['model'], normalization=normalization)
    return FutureAdaptedPlanner(base, config['future_sampler'])
