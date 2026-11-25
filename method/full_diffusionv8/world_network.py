"""BCDE: wider/deeper recurrent dynamics with multiscale history and detail skip."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .legacy import module

LegacyModel = module('model').ActionBEVWorldModel
Residual = module('model').Residual
metric_grid = module('geometry').metric_grid


class HistoryEncoder(nn.Module):
    def __init__(self, inputs, hidden, skip_channels, coarse_hidden, blocks, coarse_blocks):
        super().__init__()
        self.high = nn.Sequential(nn.Conv2d(inputs, skip_channels, 3, padding=1),
                                  nn.GroupNorm(8, skip_channels), nn.SiLU(), Residual(skip_channels))
        self.middle = nn.Sequential(nn.Conv2d(skip_channels, hidden, 3, stride=2, padding=1),
                                    *[Residual(hidden) for _ in range(blocks)])
        self.coarse = nn.Sequential(nn.Conv2d(hidden, coarse_hidden, 3, stride=2, padding=1),
                                    *[Residual(coarse_hidden) for _ in range(coarse_blocks)])
        self.context = nn.Conv2d(coarse_hidden, hidden, 1)
        self.fuse = Residual(hidden)

    def forward(self, inputs):
        high = self.high(inputs)                         # 64 x 64: detail skip
        middle = self.middle(high)                       # 32 x 32: recurrent state
        coarse = self.coarse(middle)                     # 16 x 16: context
        context = F.interpolate(self.context(coarse), size=middle.shape[-2:],
                                mode='bilinear', align_corners=False)
        return self.fuse(middle + context), high


class DetailDecoder(nn.Module):
    def __init__(self, hidden, skip_channels, blocks):
        super().__init__()
        # Expensive hidden->hidden convolution stays at 32 x 32.
        self.low = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, 64, 1))
        self.refine = nn.Sequential(nn.Conv2d(128 + skip_channels, 64, 3, padding=1),
                                    nn.SiLU(), *[Residual(64) for _ in range(blocks)])
        self.output = nn.Conv2d(64, 64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, observed_detail, previous_prediction):
        low = F.interpolate(self.low(state), size=(64, 64), mode='bilinear', align_corners=False)
        return self.output(self.refine(torch.cat((low, observed_detail, previous_prediction), 1)))


class ActionBEVWorldModel(LegacyModel):
    """Same six-action/fixed-frame API as irwm_bev_t; separate checkpoint identity."""
    def __init__(self, hidden=192, history=3, path_nodes=6, skip_channels=64,
                 coarse_hidden=384, history_residuals=4, transition_residuals=2,
                 coarse_residuals=2, decoder_residuals=1):
        widths = (hidden, skip_channels, coarse_hidden)
        depths = (history_residuals, transition_residuals, coarse_residuals, decoder_residuals)
        if history != 3 or any(n <= 0 or n % 8 for n in widths) or any(n < 1 for n in depths):
            raise ValueError('Expected three observations, positive widths divisible by eight, and positive depths')
        super().__init__(hidden=hidden, history=history, path_nodes=path_nodes)
        self.history_encoder = HistoryEncoder(history * 65 + 2, hidden, skip_channels,
                                              coarse_hidden, history_residuals, coarse_residuals)
        self.transition = nn.Sequential(*[Residual(hidden) for _ in range(transition_residuals)])
        self.decode = DetailDecoder(hidden, skip_channels, decoder_residuals)

    def _evolve(self, history, history_valid, path, path_times, path_valid,
                query_xy, query_times, query_valid, speed, target_speed):
        b, nh, c, h, w = history.shape
        if (nh, c, h, w) != (3, 64, 64, 64) or path.shape != (b, 6, 2):
            raise ValueError('Expected Bx3x64x64x64 history and Bx6x2 metric path')
        if history_valid.shape != (b, 3, 1, 64, 64) or query_xy.shape != (b, 6, 2):
            raise ValueError('Unexpected history mask or query shape')
        spatial = metric_grid(history.device).permute(2, 0, 1)[None].expand(b, -1, -1, -1) / 32.
        visible = torch.cat((torch.where(history_valid.bool(), history, 0.), history_valid.float()), 2).flatten(1, 2)
        state, observed_detail = self.history_encoder(torch.cat((visible, spatial), 1))
        path = torch.where(path_valid[..., None], path, 0.)
        path_times = torch.where(path_valid, path_times, 0.)
        query_xy = torch.where(query_valid[..., None], query_xy, 0.)
        query_times = torch.where(query_valid, query_times, 0.)
        origin = path.new_zeros(b, 1, 2)
        nodes = torch.cat((origin, path), 1)
        deltas = torch.cat((origin, nodes[:, 1:] - nodes[:, :-1]), 1)
        times = torch.cat((path_times.new_zeros(b, 1), path_times), 1)
        valid = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=path.device), path_valid), 1)
        inputs = torch.cat((nodes / 32., deltas / 5., times[..., None] / 6., valid[..., None].float()), -1)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            tokens = self.path_encoder(self.path_input(inputs) + self.path_position, src_key_padding_mask=~valid)
            flat = state.flatten(2).transpose(1, 2)
            context, _ = self.cross(self.cross_norm(flat), tokens, tokens, key_padding_mask=~valid, need_weights=False)
        state = state + context.transpose(1, 2).reshape_as(state)
        predicted = torch.where(history_valid[:, -1].bool(), history[:, -1], 0.)
        outputs = []
        previous_xy = query_xy.new_zeros(b, 2)
        previous_time = query_times.new_zeros(b)
        for k in range(6):
            dt = (query_times[:, k] - previous_time).clamp_min(0.)
            action = torch.cat((query_xy[:, k] / 32., (query_xy[:, k] - previous_xy) / 5.,
                                query_times[:, k, None] / 6., dt[:, None] / 2.,
                                speed[:, None] / 15., target_speed[:, None] / 15.), 1)
            scale, shift = self.action(action).chunk(2, 1)
            encoded = self.state_input(predicted)
            encoded = encoded * (1 + .1 * scale[:, :, None, None].tanh()) + shift[:, :, None, None]
            reset, update = self.gates(torch.cat((state, encoded), 1)).sigmoid().chunk(2, 1)
            proposal = self.candidate(torch.cat((reset * state, encoded), 1))
            next_state = self.transition((1 - update) * state + update * proposal)
            delta = self.decode(next_state, observed_detail, predicted)
            next_prediction = predicted + delta * dt.clamp(max=4.)[:, None, None, None]
            keep = query_valid[:, k, None, None, None]
            state = torch.where(keep, next_state, state)
            predicted = torch.where(keep, next_prediction, predicted)
            outputs.append(predicted)
            previous_xy = torch.where(query_valid[:, k, None], query_xy[:, k], previous_xy)
            previous_time = torch.where(query_valid[:, k], query_times[:, k], previous_time)
        return torch.stack(outputs, 1)
