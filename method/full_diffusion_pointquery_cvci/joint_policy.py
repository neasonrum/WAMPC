"""Planner loss trains planner and WM; feature loss has only the original WM path."""
import torch

from method.full_diffusionv8.joint_policy import JointPlanner as BaseJointPlanner, detach_tree, training_prior


class JointPlanner(BaseJointPlanner):
    def planning_step(self, batch, prior, prior_valid, ages, generator):
        device, b = prior.device, len(prior)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            future = self.future(batch, prior, prior_valid, ages)
            ready = future['valid'].any(-1) if future is not None else torch.zeros(b, device=device, dtype=torch.bool)
            condition = self.condition(batch, future)
            clean = self.planner.codec.encode(batch['path'], batch['temporal_delta'])
            steps = torch.randint(self.diffusion.train_steps, (b,), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy, t = self.diffusion.corrupt(clean, steps, noise)
            output = self.planner(noisy, t, condition)
            parts = self.planner.base.loss_components(output, clean, batch['spatial_mask'], batch['temporal_mask'],
                batch['behavior_targets'], batch['behavior_mask'], self.config['loss'])
            anchor = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            monitoring = any(reader.monitor for reader in self.planner.adapter['spatial'])
            self.planner.monitor(False)
            _, delta = self.sample(detach_tree(condition), b, device, generator,
                                   self.config['training']['rolling_inference_steps'])
            self.planner.monitor(monitoring)
        zero = parts['loss'].detach() * 0
        return {'loss': parts['loss'] + anchor, 'spatial_l1': parts['spatial_l1'].detach(),
            'temporal_l1': parts['temporal_delta_l1'].detach(), 'behavior_ce': parts['behavior_ce'].detach(),
            'base_loss': parts['loss'].detach(), 'speed_huber': zero,
            'speed_abs_error_sum': zero, 'speed_valid': zero, 'sampled_delta': delta.detach().float(),
            'future_ready': ready.float().sum(), 'prior_ready': prior_valid.any(-1).float().sum(),
            'prior_age_sum_seconds': ages.sum(), 'cache_age_max_seconds': ages.new_zeros(()),
            'valid_prior_points': prior_valid.float().sum()}
