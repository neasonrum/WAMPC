import torch
from torch.nn import functional as F

def speed_group(target, current, stop_mps, transition_mps):
    """Braking to a stop is a transition, distinct from holding a stop."""
    change = target - current
    return torch.where(change <= -transition_mps, 1, torch.where(
        change >= transition_mps, 2, torch.where(target < stop_mps, 0, 3)))



def speed_supervision(codec, predicted_delta, target_delta, mask, current, settings, stop_mps):
    valid = mask.bool()
    # Mask before the norm so malformed padding cannot produce NaN gradients.
    predicted_delta = torch.where(valid[..., None], predicted_delta.float(), 0.0)
    target_delta = torch.where(valid[..., None], target_delta.float(), 0.0)
    dt = codec.temporal_interval_seconds
    pred, target = (x.norm(dim=-1) / dt for x in (predicted_delta, target_delta))
    group = speed_group(target[:, :2], current.float().reshape(-1, 1), stop_mps,
                        settings['transition_mps'])
    boost = torch.where(group != 3, settings['event_multiplier'], 1.0)
    active = torch.cat((boost, torch.ones_like(target[:, 2:])), dim=1)
    factors = valid * target.new_tensor(settings['interval_weights']) * active
    error = F.smooth_l1_loss(pred, target, reduction='none', beta=settings['huber_beta_mps'])
    loss = (error * factors).sum() / factors.sum().clamp_min(1e-6)
    control_abs_error = (codec.target_speed(predicted_delta) - codec.target_speed(target_delta)).abs()
    return loss, control_abs_error, valid[:, 1]

