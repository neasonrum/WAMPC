import math
import torch
from torch import nn
from torch.nn import functional as F

MAP_TIMES = (.5,1.,1.5,2.,2.5)

def align_future(future, transform, age):
    """Generation-frame future -> current frame; `transform` maps generation to current."""
    with torch.autocast(device_type=transform.device.type, enabled=False):
        rotation = transform[:, :2, :2].float()
        translation = transform[:, :2, 2].float()
        age = age.float().to(rotation.device)
        times = torch.tensor(MAP_TIMES, device=rotation.device)
        map_time = times[None] - age[:, None]
        token_time = times[future['map_ids']][None] - age[:, None]
        map_valid = (future['window_fraction'] > 0) & (map_time > 1e-3)
        positions = torch.einsum('bij,bnj->bni', rotation, future['positions'].float()) + translation[:, None]
        axes = torch.einsum('bij,bnaj->bnai', rotation, future['axes'].float())
        centres = torch.einsum('bij,bkj->bki', rotation, future['centres'].float()) + translation[:, None]
        window_axes = torch.einsum('bij,bkaj->bkai', rotation, future['window_axes'].float())
    return {'tokens': future['tokens'], 'log_prior': future['log_prior'],
            'valid': future['valid'] & (token_time > 1e-3),
            'positions': positions, 'axes': axes, 'radii': future['radii'], 'centres': centres,
            'window_axes': window_axes, 'window_radii': future['window_radii'],
            'window_fraction': future['window_fraction'] * map_valid,
            'map_valid': map_valid, 'map_time': map_time, 'token_time': token_time, 'age': age}



class CachedFutureReadout(nn.Module):
    """Geometry/time-aware attention to cached future tokens; exact no-op at gate 0."""
    def __init__(self, d_model=256, heads=8, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.heads = heads
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.position = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.timing = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.reliability = nn.Sequential(nn.Linear(d_model + 8, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.distance_logit = nn.Parameter(torch.tensor(math.log(.01 / .99)))
        if temporal:
            self.time_logit = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.zeros(()))

    def prepare(self, future):
        if future is None:
            return None
        b, n, d = future['tokens'].shape
        timing = torch.stack((future['token_time'] / 3., future['age'][:, None].expand(-1, n) / .5), -1)
        timed = future['tokens'] + self.timing(timing.to(future['tokens'].dtype))
        return {**future, 'key': self.key(timed).reshape(b, n, self.heads, d // self.heads),
                'value': self.value(timed).reshape(b, n, self.heads, d // self.heads)}

    def forward(self, query, positions, future):
        if future is None:
            return query
        b, count, _ = positions.shape
        hidden = self.norm(query).expand(-1, count, -1)
        queries = self.query(hidden + self.position(positions / 32)).reshape(b, count, self.heads, -1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            offset = positions.float()[:, :, None] - future['positions'].float()[:, None]
            local = torch.einsum('bpni,bnai->bpna', offset, future['axes'].float())
            distance = (local / future['radii'].float()[:, None].clamp_min(.1)).square().sum(-1).clamp(max=9)
            score = torch.einsum('bphd,bnhd->bhpn', queries.float(), future['key'].float()) / math.sqrt(queries.shape[-1])
            score = score + future['log_prior'].float()[:, None, None] - self.distance_logit.sigmoid() * distance[:, None]
            map_valid = future['map_valid']
            if self.temporal:
                query_time = torch.arange(1, count + 1, device=query.device).float() * .5
                gap = (future['token_time'][:, None] - query_time[None, :, None]) / .5
                score = score - F.softplus(self.time_logit) * gap.square().clamp(max=9)[:, None]
                gaps = (future['map_time'][:, None] - query_time[None, :, None]).abs()
                match = gaps.masked_fill(~map_valid[:, None], float('inf')).argmin(-1)
                time_gap = gaps.gather(-1, match[..., None]).squeeze(-1)
            else:
                spread = (positions.float()[:, :, None] - future['centres'][:, None]).square().sum(-1)
                match = spread.masked_fill(~map_valid[:, None], float('inf')).argmin(-1)
                time_gap = torch.zeros(b, count, device=query.device)
            valid = future['valid']
            any_valid = valid.any(-1)
            safe_valid = valid.clone()
            safe_valid[:, 0] |= ~any_valid
            weights = score.masked_fill(~safe_valid[:, None, None], float('-inf')).softmax(-1)
            context = torch.einsum('bhpn,bnhd->bphd', weights, future['value'].float()).reshape(b, count, -1)
            has_map = map_valid.gather(1, match)
            centre = future['centres'].gather(1, match[..., None].expand(-1, -1, 2))
            axes = future['window_axes'].gather(1, match[..., None, None].expand(-1, -1, 2, 2))
            radii = future['window_radii'].float().gather(1, match[..., None].expand(-1, -1, 2))
            window_local = torch.einsum('bpi,bpai->bpa', positions.float() - centre, axes) / radii.clamp_min(.1)
            window_distance = window_local.abs().amax(-1)
            fraction = future['window_fraction'].float().gather(1, match)
            supported = (window_distance <= 1) & (fraction > 0) & has_map & any_valid[:, None]
            time_gap = torch.where(has_map, time_gap, 0.)
            extra = torch.cat(((positions.float() - centre) / 32, window_distance[..., None].clamp(max=4),
                               fraction[..., None], supported.float()[..., None], has_map.float()[..., None],
                               (future['age'][:, None, None] / .5).expand(-1, count, -1),
                               (time_gap[..., None] / .5).clamp(max=4)), -1)
        reliability = self.reliability(torch.cat((hidden, extra.to(hidden)), -1)).sigmoid()
        update = self.output(context.to(query.dtype)) * reliability * supported[..., None]
        update = update.sum(1, keepdim=True) / supported.sum(1, keepdim=True)[..., None].clamp_min(1)
        return query + self.gate.tanh() * update

