"""Sample once in fixed B2D ego axes, then read independently with point queries."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from method.full_diffusionv8.future import FutureNeighborhoodSampler


class SharedFutureSampler(FutureNeighborhoodSampler):
    def __init__(self, config):
        super().__init__(64, 256, config)
        # Future maps have their own dense projection, shared only within this branch.
        self.future_projection = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1), nn.ReLU())
        self.value = nn.Identity()

    def forward(self, maps, prior_delta, prior_valid, ego, motion):
        if maps.shape[1:] != (6, 64, 64, 64):
            raise ValueError("Expected six fixed-input-ego BEV predictions")
        projected = self.future_projection(maps[:, :5].flatten(0, 1)).unflatten(0, (len(maps), 5))
        # The inherited sampler's geometry and validity are unchanged; only values
        # are sampled from the already projected 256-channel maps.
        b = len(maps)
        delta = prior_delta[:, :5].float()
        positions = delta.cumsum(1)
        times = torch.arange(1, 6, device=maps.device).float()[None, :, None] / 6
        geometry = torch.cat((positions / 32, delta / delta.new_tensor([8.1, 3.]),
                              F.normalize(delta, dim=-1, eps=1e-3), times.expand(b, -1, -1)), -1)
        controller = self.controller_norm(self.geometry(torch.cat(
            (geometry, motion[:, None].expand(-1, 5, -1)), -1)) + self.ego(ego)[:, None])
        raw = self.offset(controller)
        units = self.template + .25 * raw[..., :-2].reshape(b, 5, -1, 2).tanh() * self.movable
        radii = (1 + .5 * raw[..., -2:].tanh()) * maps.new_tensor(self.config['window_m'])
        radii = radii * maps.new_tensor(self.config['horizon_window_scale'])[None, :, None]
        from method.full_diffusionv8.future import path_axes
        tangent, normal = path_axes(delta, prior_valid[:, :5])
        features, valids, coordinates, offsets, priors, axes, token_radii = [], [], [], [], [], [], []
        for horizon, count in enumerate(self.config['points_per_map']):
            local = units[:, horizon, :count] * radii[:, horizon:horizon + 1]
            offset = local[..., :1] * tangent[:, horizon:horizon + 1] + local[..., 1:] * normal[:, horizon:horizon + 1]
            points = positions[:, horizon:horizon + 1] + offset
            grid = points.float() / 32
            valid = (grid.abs() <= 1 - 1 / 64).all(-1) & torch.isfinite(grid).all(-1)
            valid &= prior_valid[:, horizon:horizon + 1]
            safe = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            with torch.autocast(device_type=maps.device.type, enabled=False):
                values = F.grid_sample(projected[:, horizon].float(), safe[:, :, None],
                                       align_corners=False, mode='bilinear', padding_mode='zeros')
                values = values.squeeze(-1).transpose(1, 2)
            features.append(values.masked_fill(~valid[..., None], 0))
            valids.append(valid)
            coordinates.append(points)
            offsets.append(local)
            # Each horizon has equal initial group mass, regardless of token count.
            priors.append(-valid.sum(-1, keepdim=True).clamp_min(1).float().log().expand(-1, count))
            axes.append(torch.stack((tangent[:, horizon], normal[:, horizon]), 1)[:, None].expand(-1, count, -1, -1))
            token_radii.append(radii[:, horizon:horizon + 1].expand(-1, count, -1))
        valid = torch.cat(valids, 1)
        token_time = times.expand(b, -1, -1)[:, self.map_ids]
        encoding = torch.cat((geometry[:, self.map_ids, :2],
                              torch.cat(offsets, 1) / maps.new_tensor(self.config['window_m']),
                              token_time, token_time), -1)
        tokens = self.output_norm(torch.cat(features, 1) + self.position(encoding)
                                  + self.time(self.map_ids)[None] + self.waypoint(self.map_ids)[None])
        return {'tokens': tokens.masked_fill(~valid[..., None], 0), 'valid': valid,
                'log_prior': torch.cat(priors, 1), 'positions': torch.cat(coordinates, 1),
                'axes': torch.cat(axes, 1), 'radii': torch.cat(token_radii, 1),
                'centres': positions, 'window_axes': torch.stack((tangent, normal), 2),
                'window_radii': radii, 'map_ids': self.map_ids,
                'window_fraction': torch.stack([v.float().mean(-1) for v in valids], -1)}


class PointFutureReadout(nn.Module):
    """A separate reader per branch/layer; never pool over the trajectory points."""
    def __init__(self, temporal=False, initial_probability=.01):
        super().__init__()
        self.temporal, self.heads = temporal, 8
        self.norm = nn.LayerNorm(256)
        self.query = nn.Linear(256, 256)
        self.position = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256))
        self.timing = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256))
        self.key, self.value, self.output = (nn.Linear(256, 256) for _ in range(3))
        self.gate = nn.Parameter(torch.full((256,), math.log(initial_probability / (1 - initial_probability))))
        self.distance_logit = nn.Parameter(torch.tensor(math.log(.01 / .99)))
        self.decay_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1 / 3))))
        if temporal:
            self.time_logit = nn.Parameter(torch.zeros(()))
        self.monitor = False
        self.last_stats = {}

    def prepare(self, future):
        if future is None:
            return None
        b, n, d = future['tokens'].shape
        timing = torch.stack((future['token_time'] / 3., future['age'][:, None].expand(-1, n) / .5), -1)
        timed = future['tokens'] + self.timing(timing.to(future['tokens']))
        return {**future, 'key': self.key(timed).reshape(b, n, 8, d // 8),
                'value': self.value(timed).reshape(b, n, 8, d // 8)}

    def forward(self, query, positions, future):
        if future is None:
            return query
        b, count, width = query.shape
        if positions.shape != (b, count, 2) or count != (6 if self.temporal else 10):
            raise ValueError("Future readout must preserve the independent 10/6 point queries")
        projected = self.query(self.norm(query) + self.position(positions.float() / 32))
        projected = projected.reshape(b, count, 8, width // 8)
        with torch.autocast(device_type=query.device.type, enabled=False):
            offset = positions.float()[:, :, None] - future['positions'].float()[:, None]
            local = torch.einsum('bpni,bnai->bpna', offset, future['axes'].float())
            distance = (local / future['radii'].float()[:, None].clamp_min(.1)).square().sum(-1).clamp(max=9)
            score = torch.einsum('bphd,bnhd->bhpn', projected.float(), future['key'].float()) / math.sqrt(width // 8)
            score = score + future['log_prior'].float()[:, None, None] - self.distance_logit.sigmoid() * distance[:, None]
            score = score - F.softplus(self.decay_raw) * future['token_time'].float()[:, None, None]
            if self.temporal:
                query_times = torch.arange(1, count + 1, device=query.device).float() * .5
                gap = (future['token_time'][:, None] - query_times[None, :, None]) / .5
                score = score - F.softplus(self.time_logit) * gap.square().clamp(max=9)[:, None]
            valid = future['valid']
            any_valid = valid.any(-1)
            safe_valid = valid.clone()
            safe_valid[:, 0] |= ~any_valid
            weights = score.masked_fill(~safe_valid[:, None, None], float('-inf')).softmax(-1)
            context = torch.einsum('bhpn,bnhd->bphd', weights, future['value'].float()).reshape(b, count, width)
        update = self.output(context.to(query))
        update = F.layer_norm(update, (width,)) * self.gate.sigmoid()
        update = update * any_valid[:, None, None].to(update)
        if self.monitor:
            with torch.no_grad():
                mass = weights.mean((0, 1, 2))
                self.last_stats = {'gate_mean': float(self.gate.sigmoid().mean()),
                    'gate_min': float(self.gate.sigmoid().min()), 'gate_max': float(self.gate.sigmoid().max()),
                    'residual_ratio': float((update.float().norm(dim=-1) / query.float().norm(dim=-1).clamp_min(1e-6)).mean()),
                    'valid_tokens_mean': float(valid.sum(-1).float().mean()),
                    'time_mass': [float(mass[future['map_ids'] == i].sum()) for i in range(5)],
                    'attention_shape': list(weights.shape)}
        return query + update.to(query.dtype)
