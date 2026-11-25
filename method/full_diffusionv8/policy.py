"""20Hz online planning with an exact 0.25s-old action prior and jointly trained WM."""
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3_ego12.inference.policy import Ego12Network
from . import METHOD, FORMAT
from .clock import PlanningClock
from .data import JointDataset
from .history import TickBEVHistory, LaggedPlanHistory
from .train import build


class V8Network(Ego12Network):
    def __init__(self, garage_net, checkpoint_path, device):
        nn.Module.__init__(self)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state.get('format') != FORMAT or state.get('method') != METHOD or state['progress']['updates'] < 1:
            raise ValueError('V8 inference needs a trained V8 ego12 joint checkpoint')
        self.run_config = state['config']
        if self.run_config['conditioning'] != 'tfpp_ego11_plus_state1' or self.run_config['diffusion']['inference_steps'] != 25:
            raise ValueError('The V8 ego12 conditioning and 25-step sampler must match training')
        self.policy, _ = build(self.run_config, device)
        self.policy.planner.base.load_state_dict(state['planner_base'], strict=True)
        self.policy.planner.adapter.load_state_dict(state['adapter'], strict=True)
        self.policy.world.model.load_state_dict(state['world_model'], strict=True)
        self.policy.requires_grad_(False)
        self.perception = FrozenPlannerPerception.from_net(garage_net)
        self.diffusion = self.policy.diffusion
        self.history, self.rolling, self.clock = TickBEVHistory(), LaggedPlanHistory(), PlanningClock()
        self.generator = torch.Generator(device=device).manual_seed(self.run_config['seed'])
        self.last_target_speed = self.last_temporal_delta = self.last_behavior_probabilities = None
        self.last_prior_tick, self.last_history_ticks = None, []
        self.last_future_ready = False
        self._pose = self._timestamp = None
        self.to(device)
        self.eval()

    def set_observation(self, pose, timestamp):
        self.history.set_observation(pose, timestamp)
        self._pose = torch.from_numpy(JointDataset.pose(np.asarray(pose, dtype=np.float64)))
        self._timestamp = float(timestamp)

    @torch.no_grad()
    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
        if self._pose is None or len(rgb) != 1 or not self.clock.due(self._timestamp):
            raise RuntimeError('V8 requires one fresh CARLA observation per planning tick')
        context = torch.autocast('cuda', dtype=torch.bfloat16) if rgb.is_cuda else nullcontext()
        with context:
            bev, memory, ego = self.perception({'rgb': rgb, 'lidar': lidar_bev, 'target_point': target_point,
                                               'target_point_next': target_point_next, 'speed': ego_vel, 'command': command})
            history, transforms, mask = self.history.get_inputs(bev)
            pose = self._pose.to(rgb.device)[None]
            prior, valid, ages = self.rolling.prior(['route'], pose, [self._timestamp])
            tick = PlanningClock.tick(self._timestamp)
            selected = self.rolling.last_selected_ticks[0]
            if selected is not None and selected != tick - 5:
                raise AssertionError('V8 selected a plan other than k-5')
            raw = torch.cat((history, bev[:, None]), 1)
            identity = torch.eye(3, device=rgb.device)[None, None]
            batch = {'raw_bev': raw, 'history_transforms': torch.cat((transforms, identity), 1),
                     'history_mask': torch.cat((mask, mask.new_ones(1, 1)), 1),
                     'memory': memory, 'ego': ego, 'target_point': target_point,
                     'speed': ego_vel, 'command': command}
            future = self.policy.future(batch, prior, valid, ages)
            condition = self.policy.condition(batch, future)
            path, delta = self.policy.sample(condition, 1, rgb.device, self.generator, 25)
            speed = self.policy.planner.codec.target_speed(delta)
            if not bool(torch.isfinite(path).all() and torch.isfinite(delta).all() and torch.isfinite(speed).all()):
                raise FloatingPointError('V8 online sampling produced a nonfinite action')
            self.rolling.update(['route'], pose, [self._timestamp], delta)
            self.clock.mark(self._timestamp)
            boxes = self.perception.net.head(bev) if self.config.detect_boxes else None
        self.last_target_speed = float(speed[0])
        self.last_temporal_delta = delta.float().detach()
        self.last_behavior_probabilities = condition['behavior_logits'].float().softmax(-1).detach()
        self.last_prior_tick = selected
        self.last_history_ticks = self.history.last_ticks + [tick]
        self.last_future_ready = bool(future is not None and future['valid'].any())
        bins = torch.tensor(self.config.target_speeds, device=speed.device, dtype=speed.dtype)
        compatibility_speed = speed.clamp(0, max(self.config.target_speeds))
        upper = torch.searchsorted(bins, compatibility_speed).clamp(1, len(bins) - 1)
        weight = (compatibility_speed - bins[upper - 1]) / (bins[upper] - bins[upper - 1])
        probabilities = torch.zeros(len(speed), len(bins), device=speed.device, dtype=speed.dtype)
        probabilities.scatter_(1, (upper - 1)[:, None], (1 - weight)[:, None])
        probabilities.scatter_(1, upper[:, None], weight[:, None])
        return None, probabilities.clamp_min(1e-12).log(), path.float(), None, None, None, boxes, None, None, None
