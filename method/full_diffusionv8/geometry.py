import torch
from torch.nn import functional as F

FUTURE_TIMES = (.5, 1., 1.5, 2., 2.5, 3.)

def align_history(history, source_to_current, frame_valid, extent=32.0):
    """Pull observed BEVs into the fixed current ego frame, with pixel masks."""
    b, n, c, h, w = history.shape
    if source_to_current.shape != (b, n, 3, 3) or frame_valid.shape != (b, n):
        raise ValueError('History transform or validity shape differs')
    with torch.autocast(device_type=history.device.type, enabled=False):
        inverse = torch.linalg.inv(source_to_current.float()).reshape(b * n, 3, 3)
        axis_x = (torch.arange(w, device=history.device).float() + .5) * (2 * extent / w) - extent
        axis_y = (torch.arange(h, device=history.device).float() + .5) * (2 * extent / h) - extent
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing='ij')
        grid = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        source = torch.einsum('bij,hwj->bhwi', inverse, grid)[..., :2]
        limits = source.new_tensor([extent - extent / w, extent - extent / h])
        valid = (source.abs() <= limits + 1e-5).all(-1)
        valid &= frame_valid.reshape(b * n, 1, 1)
        aligned = F.grid_sample(history.float().reshape(b * n, c, h, w), source / extent,
                                align_corners=False, padding_mode='zeros')
        aligned = aligned * valid[:, None]
    return aligned.reshape(b, n, c, h, w), valid.reshape(b, n, 1, h, w)



def cumulative_valid(valid):
    """An increment/recursive prediction is usable only with its full prefix."""
    return valid.to(torch.int64).cumprod(-1).bool()



def motion_features(history, valid, speed, prior_delta, prior_age):
    """Twenty observed/prior features; no future labels or future poses."""
    b = len(history)
    pooled = history.float().mean((-1, -2, -3))
    differences = history[:, 1:].float() - history[:, :-1].float()
    mean_change = differences.mean((-1, -2, -3))
    rms_change = differences.square().mean((-1, -2, -3)).sqrt()
    interval_speed = prior_delta.float().norm(dim=-1) / .5
    inputs = torch.cat((speed.float().reshape(b, 1) / 15, prior_age.float().reshape(b, 1),
                        valid.float().reshape(b, 3), pooled, mean_change, rms_change,
                        interval_speed / 15, prior_delta[:, 0].float() / 5), -1)
    if inputs.shape != (b, 20):
        raise ValueError('Observed motion descriptor must have twenty dimensions')
    return inputs

