import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_FUTURE_CONFIG = {"points_per_map": [9,9,16,25,36], "window_m": [6.,3.], "horizon_window_scale": [.5,.625,.75,.875,1.]}

def path_axes(delta, valid=None):
    """Nearest usable direction; a stopped trajectory uses observed ego-forward."""
    b, n, _ = delta.shape
    lengths = delta.float().norm(dim=-1)
    moving = lengths > 1e-3
    if valid is not None:
        moving = moving & valid
    index = torch.arange(n, device=delta.device)
    distance = (index[:, None] - index[None]).abs()[None] + (~moving[:, None]).long() * (n + 1)
    nearest = distance.argmin(-1)
    directions = delta.float() / lengths[..., None].clamp_min(1e-3)
    selected = directions.gather(1, nearest[..., None].expand(-1, -1, 2))
    fallback = torch.zeros_like(selected)
    fallback[..., 0] = 1
    tangent = torch.where(moving.any(-1)[:, None, None], selected, fallback)
    return tangent, torch.stack((-tangent[..., 1], tangent[..., 0]), -1)



class FutureNeighborhoodSampler(nn.Module):
    def __init__(self, channels=64, d_model=256, config=None):
        super().__init__()
        self.config = {k: list(v) for k, v in (config or DEFAULT_FUTURE_CONFIG).items()}
        if set(self.config) != set(DEFAULT_FUTURE_CONFIG):
            raise ValueError('Future sampler configuration fields differ')
        counts = self.config['points_per_map']
        if len(counts) != 5 or any(type(n) is not int or n < 5 or n > 64 for n in counts):
            raise ValueError('Five future maps need valid point budgets')
        if len(self.config['window_m']) != 2 or min(self.config['window_m']) <= 0:
            raise ValueError('Expected positive tangent/normal window radii')
        if len(self.config['horizon_window_scale']) != 5 or min(self.config['horizon_window_scale']) <= 0:
            raise ValueError('Expected five positive horizon window scales')
        self.geometry = nn.Sequential(nn.Linear(27, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.ego = nn.Linear(d_model, d_model)
        self.controller_norm = nn.LayerNorm(d_model)
        maximum = max(counts)
        self.offset = nn.Linear(d_model, maximum * 2 + 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.value = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, d_model))
        self.position = nn.Sequential(nn.Linear(6, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time = nn.Embedding(6, d_model)
        self.waypoint = nn.Embedding(6, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        template = torch.zeros(5, maximum, 2)
        movable = torch.zeros(5, maximum, 1)
        map_ids = []
        for horizon, count in enumerate(counts):
            inner = max(4, (count - 1) // 3)
            outer = count - 1 - inner
            cursor = 1
            for number, radius, phase in ((inner, .25, 0.), (outer, .75, math.pi / max(outer, 1))):
                angle = torch.arange(number).float() * (2 * math.pi / max(number, 1)) + phase
                template[horizon, cursor:cursor + number] = radius * torch.stack((angle.cos(), angle.sin()), -1)
                cursor += number
            movable[horizon, 1:count] = 1
            map_ids.extend([horizon] * count)
        self.register_buffer('template', template, persistent=False)
        self.register_buffer('movable', movable, persistent=False)
        self.register_buffer('map_ids', torch.tensor(map_ids), persistent=False)

    def forward(self, maps, prior_delta, prior_valid, ego, motion):
        b, t, c, h, w = maps.shape
        if (t, c, h, w) != (6, 64, 64, 64) or prior_delta.shape != (b, 6, 2):
            raise ValueError('Expected six fixed-frame TF++ future maps and six metric actions')
        if prior_valid.shape != (b, 6) or motion.shape != (b, 20):
            raise ValueError('Future validity or observed motion shape differs')
        delta = prior_delta[:, :5].float()
        positions = delta.cumsum(1)
        times = torch.arange(1, 6, device=maps.device).float()[None, :, None] / 6
        geometry = torch.cat((positions / 32, delta / delta.new_tensor([8.1, 3.]),
                              F.normalize(delta, dim=-1, eps=1e-3), times.expand(b, -1, -1)), -1)
        controller = self.controller_norm(self.geometry(torch.cat((geometry, motion[:, None].expand(-1, 5, -1)), -1))
                                          + self.ego(ego)[:, None])
        raw = self.offset(controller)
        residual = raw[..., :-2].reshape(b, 5, -1, 2).tanh()
        radii = (1 + .5 * raw[..., -2:].tanh()) * maps.new_tensor(self.config['window_m'])
        radii = radii * maps.new_tensor(self.config['horizon_window_scale'])[None, :, None]
        units = self.template + .25 * residual * self.movable
        tangent, normal = path_axes(delta, prior_valid[:, :5])
        features, valids, coordinates, offsets, priors, axes, token_radii = [], [], [], [], [], [], []
        for horizon, count in enumerate(self.config['points_per_map']):
            local = units[:, horizon, :count] * radii[:, horizon:horizon + 1]
            offset = local[..., :1] * tangent[:, horizon:horizon + 1] + local[..., 1:] * normal[:, horizon:horizon + 1]
            points = positions[:, horizon:horizon + 1] + offset
            # Fixed input-ego maps: add the predicted centre once, use width=x.
            grid = points.float() / 32
            valid = (grid.abs() <= 1 - 1 / 64).all(-1) & torch.isfinite(grid).all(-1)
            valid = valid & prior_valid[:, horizon:horizon + 1]
            safe = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            with torch.autocast(device_type=maps.device.type, enabled=False):
                values = F.grid_sample(maps[:, horizon].float(), safe[:, :, None], align_corners=False,
                                       mode='bilinear', padding_mode='zeros').squeeze(-1).transpose(1, 2)
            features.append(values.masked_fill(~valid[..., None], 0))
            valids.append(valid)
            coordinates.append(points)
            offsets.append(local)
            priors.append(-valid.sum(-1, keepdim=True).clamp_min(1).float().log().expand(-1, count))
            axes.append(torch.stack((tangent[:, horizon], normal[:, horizon]), 1)[:, None].expand(-1, count, -1, -1))
            token_radii.append(radii[:, horizon:horizon + 1].expand(-1, count, -1))
        valid = torch.cat(valids, 1)
        token_geometry = geometry[:, self.map_ids]
        token_time = times.expand(b, -1, -1)[:, self.map_ids]
        encoding = torch.cat((token_geometry[..., :2],
                              torch.cat(offsets, 1) / maps.new_tensor(self.config['window_m']),
                              token_time, token_time), -1)
        tokens = self.output_norm(self.value(torch.cat(features, 1)) + self.position(encoding)
                                  + self.time(self.map_ids)[None] + self.waypoint(self.map_ids)[None])
        return {'tokens': tokens.masked_fill(~valid[..., None], 0), 'valid': valid,
                'log_prior': torch.cat(priors, 1), 'positions': torch.cat(coordinates, 1),
                'axes': torch.cat(axes, 1), 'radii': torch.cat(token_radii, 1),
                'centres': positions, 'window_axes': torch.stack((tangent, normal), 2),
                'window_radii': radii, 'map_ids': self.map_ids,
                'window_fraction': torch.stack([v.float().mean(-1) for v in valids], -1)}

