"""Preserve the completed point-query baseline and append CVCI future K/V reads."""
import torch
from torch import nn

from method.full_diffusion_pointquery import FORMAT as BASE_FORMAT
from method.full_diffusion_pointquery.model import PointQueryHead, blocks
from .future import SharedFutureSampler, PointFutureReadout
from .prepare import sha


class FuturePointPlanner(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base, self.base_training_mode = base, True
        self.adapter = nn.ModuleDict({'sampler': SharedFutureSampler(config['future_sampler']),
            'spatial': nn.ModuleList([PointFutureReadout(False, config['future_readout']['initial_probability']) for _ in base.layers]),
            'temporal': nn.ModuleList([PointFutureReadout(True, config['future_readout']['initial_probability']) for _ in base.temporal_layers])})
        self.initialization = None

    @property
    def codec(self):
        return self.base.codec

    def load_pretrained(self, config, artifact):
        if sha(config['planner_checkpoint']) != config['planner_checkpoint_sha256']:
            raise ValueError("Completed point-query epoch10 checksum changed")
        state = torch.load(config['planner_checkpoint'], map_location='cpu', weights_only=False)
        if (state['format'] != BASE_FORMAT or state['progress']['epoch'] != 10
                or state['config']['model'] != config['model']
                or state['dataset']['normalization']['normalization'] != artifact['normalization']):
            raise ValueError("Initialization must be the exact completed point-query baseline")
        self.base.load_state_dict(state['model'], strict=True)
        self.initialization = {'checkpoint': config['planner_checkpoint'],
            'sha256': config['planner_checkpoint_sha256'], 'strict': True,
            'loaded_tensors': len(state['model']), 'progress': state['progress'],
            'future_extension': 'new parameters; sigmoid channel gates at .01'}

    def tokenize(self, maps, prior, valid, ego, motion):
        return self.adapter['sampler'](maps, prior, valid, ego.mean(1), motion)

    def condition(self, bev, memory, ego, target_point, history_bev, transforms, mask, speed, command, future=None):
        result = self.base.prepare_condition(bev, memory, ego, target_point, history_bev, transforms, mask, speed, command)
        if future is not None:
            future = {**future, 'map_ids': self.adapter['sampler'].map_ids}
        result['future'] = [reader.prepare(future) for reader in self.adapter['spatial']]
        result['temporal_future'] = [reader.prepare(future) for reader in self.adapter['temporal']]
        return result

    @staticmethod
    def decode(layer, query, points, values, ego, embedding, reader, future):
        query = layer.self_norm(query + layer.dropout(layer.self_attention(query, query, query, need_weights=False)[0]))
        query = layer.norm1(query + layer.cross_bev_attention.dropout(layer.local_read(query, points, values)))
        query = layer.norm2(query + layer.dropout(layer.cross_ego_attention(query, ego, ego, need_weights=False)[0]))
        references = points.cumsum(1) if layer.temporal else points
        query = reader(query, references, future)
        query = layer.norm3(layer.ffn(query))
        scale, shift = layer.time_modulation(embedding).chunk(2, dim=-1)
        query = query * (1 + scale) + shift
        return query, points + layer.task_decoder(query).float()

    def forward(self, noisy, time, condition):
        base = self.base
        self.codec.check_shape(noisy)
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=base.slot_width)
        features = features + base.type_embedding(base.type_ids)[None] + base.slot_embedding(base.slot_ids)[None]
        path_query = base.path_query_encoder(features[:, :10]) + condition['behavior_offset']
        temporal_query = base.temporal_query_encoder(features[:, 10:]) + condition['temporal_behavior_offset']
        path_time, temporal_time = base.time_mlp(time * 1000)[:, None], base.temporal_time_mlp(time * 1000)[:, None]
        metric, predictions = self.codec.to_metric(noisy), []
        for index, (layer, temporal_layer) in enumerate(zip(base.layers, base.temporal_layers)):
            path_query, path = self.decode(layer, path_query, metric[:, :10], condition['bev'][index],
                condition['ego'], path_time, self.adapter['spatial'][index], condition['future'][index])
            temporal_query, temporal = self.decode(temporal_layer, temporal_query, metric[:, 10:],
                condition['temporal_bev'][index], condition['ego'], temporal_time,
                self.adapter['temporal'][index], condition['temporal_future'][index])
            prediction = self.codec.encode_metric(torch.cat((path, temporal), 1))
            predictions.append(prediction)
            metric = self.codec.to_metric(prediction.detach())
        return {'predictions': torch.stack(predictions), 'behavior_logits': condition['behavior_logits']}

    def gates(self):
        return {f'{branch}{index}': float(reader.gate.detach().sigmoid().mean())
                for branch in ('spatial', 'temporal') for index, reader in enumerate(self.adapter[branch])}

    def monitor(self, enabled):
        for branch in ('spatial', 'temporal'):
            for reader in self.adapter[branch]:
                reader.monitor = enabled

    def diagnostics(self):
        return {f'{branch}{index}': reader.last_stats
                for branch in ('spatial', 'temporal') for index, reader in enumerate(self.adapter[branch])}


def build_planner(config, normalization):
    return FuturePointPlanner(PointQueryHead(**config['model'], normalization=normalization), config)
