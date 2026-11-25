"""V8 differentiable BCDE with the original frozen-teacher feature contract."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .geometry import align_history, cumulative_valid
from .prepare import sha
from .world_network import ActionBEVWorldModel


class JointBCDE(nn.Module):
    def __init__(self, checkpoint_path, teacher_path, checkpoint_sha256):
        super().__init__()
        digest = sha(checkpoint_path)
        if digest != checkpoint_sha256:
            raise ValueError('Pretrained BCDE checksum changed')
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        dataset, config = state['dataset'], state['config']
        teacher_hash = sha(teacher_path)
        if dataset['teacher_sha256'] != teacher_hash:
            raise ValueError('World and planner must use the same frozen TF++ teacher')
        if dataset.get('teacher_config_sha256') != sha(Path(teacher_path).with_name('config.json')):
            raise ValueError('World and planner teacher configurations differ')
        if (config['method'] != 'irwm_bev_t_large' or config['architecture'] != 'multiscale_recurrent_192_v1'
                or dataset['query_times_seconds'] != [.5, 1., 1.5, 2., 2.5, 3.]):
            raise ValueError('Expected the pretrained six-action BCDE architecture')
        self.model = ActionBEVWorldModel(**config['model'])
        self.model.load_state_dict(state['model'], strict=True)
        self.model.requires_grad_(True)
        self.provenance = {'checkpoint': str(checkpoint_path), 'sha256': digest,
                           'teacher_sha256': teacher_hash, 'teacher_config_sha256': dataset['teacher_config_sha256'],
                           'method': config['method'], 'architecture': config['architecture'],
                           'model_config': config['model'], 'action_contract': dataset.get('action_contract'),
                           'progress': state.get('progress'), 'tuning': 'all_bcde_parameters'}
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def rollout(self, history, spatial_valid, delta, action_valid, speed):
        delta = torch.where(action_valid[..., None], delta.detach().float(), 0.)
        path = delta.cumsum(1)
        times = self.model.times[None].expand(len(path), -1)
        speed = speed.detach().float().reshape(len(path))
        target_speed = delta[:, 1].norm(dim=-1) / .5
        inputs = (history, spatial_valid, path, times, action_valid, path, times,
                  action_valid, speed, target_speed)
        with torch.autocast(device_type=history.device.type, dtype=torch.bfloat16, enabled=history.is_cuda):
            if torch.is_grad_enabled():
                return checkpoint(self.model._evolve, *inputs, use_reentrant=False, preserve_rng_state=False)
            return self.model._evolve(*inputs)

    def forward(self, history, transforms, history_valid, prior_delta, prior_valid, speed):
        aligned, spatial_valid = align_history(history.detach(), transforms.detach(), history_valid)
        valid = cumulative_valid(prior_valid.bool()) & history_valid.all(-1)[:, None]
        ready = valid[:, :5].any(-1)
        if not bool(ready.any()):
            return None, valid, aligned
        predicted = self.rollout(aligned[ready], spatial_valid[ready], prior_delta[ready], valid[ready], speed[ready])
        maps = predicted.new_zeros(len(history), *predicted.shape[1:])
        return maps.index_copy(0, ready.nonzero().flatten(), predicted), valid, aligned
