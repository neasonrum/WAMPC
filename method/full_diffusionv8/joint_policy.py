"""V8's two gradient paths, using the ego12 future-adapted planner."""
import torch
from torch import nn
from .geometry import align_history, cumulative_valid, motion_features
from .readout import align_future
from .world_geometry import align_to_anchor, path_corridor
from .world_loss import loss_components
from .speed import speed_supervision
from .utils import move

def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    return value


def generate_future(world, planner, raw_bev, transforms, history_mask, prior, prior_valid, ages, speed, ego):
    """One world rollout driven by the aligned previous plan; tokens stay in this frame."""
    maps, valid, aligned = world(raw_bev, transforms, history_mask, prior, prior_valid, speed)
    if maps is None:
        return None
    motion = motion_features(aligned.detach(), history_mask, speed, prior, ages)
    future = planner.tokenize(maps, prior, valid, ego, motion)
    future['valid'] = future['valid'] & valid[:, :5].any(-1)[:, None]
    return future


class FreshWorldPlanner(nn.Module):
    """The detached t-0.25 prediction drives a new rollout at the current t."""
    def __init__(self, planner, world, diffusion, config):
        super().__init__()
        self.planner, self.world, self.config = planner, world, config
        self.diffusion = diffusion

    def train(self, mode=True):
        super().train(mode)
        self.planner.base.train(mode and self.planner.base_training_mode)
        self.world.eval()
        return self

    def future(self, x, prior, prior_valid, ages):
        available = prior_valid.any(-1)
        if bool((available & ((ages - .25).abs() > 1e-6)).any()):
            raise ValueError('A usable training prior must be exactly 0.25s old')
        if bool(prior_valid[:, -1].any()) or bool((available & ~prior_valid[:, :5].all(-1)).any()):
            raise ValueError('The advanced previous plan covers five future points; its sixth tail is invalid')
        future = generate_future(self.world, self.planner, x['raw_bev'], x['history_transforms'],
                                 x['history_mask'], prior.detach(), prior_valid, ages.detach(), x['speed'], x['ego'])
        if future is None:
            return None
        identity = torch.eye(3, device=prior.device).expand(len(prior), -1, -1)
        # Generation and conditioning happen at the same observation: cache age is always zero.
        return align_future(future, identity, torch.zeros(len(prior), device=prior.device))

    def condition(self, x, future):
        return self.planner.condition(x['raw_bev'][:, -1], x['memory'], x['ego'], x['target_point'],
                                      x['raw_bev'][:, :2], x['history_transforms'][:, :2],
                                      x['history_mask'][:, :2], x['speed'], x['command'], future)

    @torch.no_grad()
    def sample(self, condition, batch_size, device, generator, steps):
        was_training = self.planner.training
        self.planner.eval()  # Roll the prior with inference-time dropout, also during scratch training.
        try:
            result = self.diffusion.sample(lambda x, t: self.planner(x, t, condition)['predictions'][-1],
                                           batch_size, device, generator, steps)
            return self.planner.codec.decode(result)
        finally:
            self.planner.train(was_training)

    @torch.no_grad()
    def predict_predecessor(self, observation, generator):
        device = observation['raw_bev'].device
        was_training = self.planner.training
        self.planner.eval()
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                condition = self.condition(observation, future=None)
                _, delta = self.sample(condition, len(observation['raw_bev']), device, generator,
                                       self.config['training']['rolling_inference_steps'])
            return delta.float()
        finally:
            self.planner.train(was_training)

    def forward(self, batch, prior, prior_valid, ages, generator):
        settings, weights = self.config['training'], self.config['loss']
        device, b = prior.device, len(prior)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = self.future(batch, prior, prior_valid, ages)
            ready = (future['valid'].any(-1) if future is not None
                     else torch.zeros(b, dtype=torch.bool, device=device))
            condition = self.condition(batch, future)
            clean = self.planner.codec.encode(batch['path'], batch['temporal_delta'])
            steps = torch.randint(self.diffusion.train_steps, (b,), device=device)
            steps = torch.where(torch.rand(b, device=device) < settings['max_noise_fraction'],
                                self.diffusion.train_steps - 1, steps)
            noisy, t = self.diffusion.corrupt(clean, steps)
            output = self.planner(noisy, t, condition)
            parts = self.planner.base.loss_components(
                output, clean, batch['spatial_mask'], batch['temporal_mask'],
                batch['behavior_targets'], batch['behavior_mask'], weights)
            _, predicted_delta = self.planner.codec.decode(output['predictions'][-1].float())
            speed, speed_absolute_error, speed_valid = speed_supervision(
                self.planner.codec, predicted_delta, batch['temporal_delta'], batch['temporal_mask'],
                batch['speed'], weights['speed_supervision'], self.config['labels']['stop_speed_mps'])
            total = parts['loss'] + weights['speed_supervision']['weight'] * speed
            anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            _, delta = self.sample(detach_tree(condition), b, device, generator, settings['rolling_inference_steps'])
        return {'loss': total + anchor, 'spatial_l1': parts['spatial_l1'].detach(),
                'temporal_l1': parts['temporal_delta_l1'].detach(), 'behavior_ce': parts['behavior_ce'].detach(),
                'base_loss': parts['loss'].detach(), 'speed_huber': speed.detach(),
                'speed_abs_error_sum': (speed_absolute_error * speed_valid).sum().detach(),
                'speed_valid': speed_valid.float().sum(),
                'sampled_delta': delta.detach().float(),
                'future_ready': ready.float().sum(), 'prior_ready': prior_valid.any(-1).float().sum(),
                'prior_age_sum_seconds': ages.sum(), 'cache_age_max_seconds': ages.new_zeros(()),
                'valid_prior_points': prior_valid.float().sum()}



@torch.no_grad()
def training_prior(rolling, policy, batch, support, generator):
    prior, valid, ages = rolling.prior(batch['chain_id'], batch['current_pose'], batch['timestamp'])
    if support is None:
        return prior, valid, ages, 0
    rows, observations = support
    needed = ~valid.any(-1).cpu()[rows]
    rows = rows[needed]
    if not len(rows):
        return prior, valid, ages, 0
    observation = move({k: v[needed] for k, v in observations.items()}, prior.device)
    now = batch['timestamp'][rows]
    if not torch.equal(observation['timestamp'], now - .25):
        raise ValueError('Backfill must come from the same route exactly one raw frame before the anchor')
    scenes = [batch['chain_id'][i] for i in rows.tolist()]
    prediction = policy.predict_predecessor(observation, generator)
    rolling.update(scenes, observation['current_pose'], observation['timestamp'], prediction)
    targets = rows.to(prior.device)
    filled = rolling.prior(scenes, batch['current_pose'][targets], now)
    prior[targets], valid[targets], ages[targets] = filled
    return prior, valid, ages, len(rows)


class JointPlanner(FreshWorldPlanner):
    def planning_step(self, batch, prior, prior_valid, ages, generator):
        return super().forward(batch, prior, prior_valid, ages, generator)

    def world_losses(self, batch):
        # Everything here comes from the observed or expert data. Never use the current planner
        # output as a condition for matching future BEV labels from the expert trajectory.
        with torch.no_grad():
            raw = batch['raw_bev'].detach()
            aligned, history_valid = align_history(raw, batch['history_transforms'], batch['history_mask'])
            action = batch['temporal_delta'].detach().float()
            query_valid = cumulative_valid(batch['temporal_mask'].bool()) & batch['history_mask'].all(-1)[:, None]
            path = torch.where(query_valid[..., None], action, 0.).cumsum(1)
            corridor = path_corridor(path, query_valid, path, query_valid)
            teacher = batch['world_target_bev'].detach()
            target, valid = align_to_anchor(teacher.flatten(0, 1), batch['world_target_poses'].flatten(0, 1))
            target = target.unflatten(0, (len(raw), 6))
            valid = valid.unflatten(0, (len(raw), 6))
            # Missing teacher observations only mask supervision at their own physical horizon.
            # Their actions still participate in the recurrent rollout.
            target_available = query_valid & batch['world_target_valid'].bool()
        prediction = self.world.rollout(aligned, history_valid, action, query_valid, batch['speed'])
        return loss_components(prediction, target, valid, target_available, corridor,
                               aligned[:, -1], history_valid[:, -1], self.config['world_supervision']['loss'])

    def forward(self, batch, prior, prior_valid, ages, generator):
        planner = self.planning_step(batch, prior, prior_valid, ages, generator)
        world = self.world_losses(batch)
        return {**planner, 'planner_loss': planner['loss'].detach(),
                'loss': planner['loss'] + self.config['world_supervision']['weight'] * world['loss'],
                'world_loss': world['loss'].detach(), 'world_nmse': world['nmse'].detach(),
                'world_target_count': batch['world_target_valid'].float().sum()}

