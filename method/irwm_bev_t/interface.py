"""The metric V3 -> timed BEV world-model contract (no normalization guessing)."""
import torch

TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
CONTRACT = {
    'version': 'irwm_bev_t_v1', 'action': 'six consecutive XY displacements in metres',
    'action_shape': [6, 2], 'times_seconds': list(TIMES),
    'action_frame': 'all displacements expressed in the input observation ego frame',
    'feature_shape': [6, 64, 64, 64], 'feature_frame': 'fixed input observation ego frame',
    'axes': 'width=x forward; height=y right', 'extent_metres': [-32, 32],
    'history_times_seconds': [-0.5, -0.25, 0.0],
    'speed': 'current measured speed; target speed derived from second displacement / 0.5',
}


def check_motion(value):
    if value.ndim != 3 or value.shape[1:] != (6, 2):
        raise ValueError('Expected B x 6 x 2 METRIC temporal motion, not ten spatial points')


def positions_from_deltas(temporal_delta):
    check_motion(temporal_delta)
    return temporal_delta.cumsum(1)


def deltas_from_positions(positions):
    check_motion(positions)
    return torch.cat((positions[:, :1], positions[:, 1:] - positions[:, :-1]), 1)


def from_v3_normalized(normalized, codec):
    """Decode once with the actual V3 checkpoint codec; do NOT difference delta twice."""
    if codec.temporal_points != 6 or abs(codec.temporal_interval_seconds - 0.5) > 1e-9:
        raise ValueError('Incompatible V3 temporal contract')
    _, temporal_delta = codec.decode(normalized)
    check_motion(temporal_delta)
    return temporal_delta.float()


def from_v3_policy(policy):
    """Read after policy.forward(): this field is already decoded, in metres."""
    delta = policy.last_temporal_delta
    if delta is None:
        raise ValueError('The planner has not produced temporal motion')
    check_motion(delta)
    return delta.detach().float()


def rebase_positions(positions, source_to_world, current_to_world):
    """Transform endpoints first. Translation must NOT be added to every increment."""
    # Geometry must not inherit TF32/BF16 matrix multiplication from model inference.
    with torch.autocast(device_type=positions.device.type, enabled=False):
        transform = torch.linalg.solve(current_to_world.double(), source_to_world.double())
        points = torch.cat((positions.double(), torch.ones_like(positions[..., :1], dtype=torch.float64)), -1)
        result = (points @ transform.transpose(-1, -2))[..., :2]
    return result.to(torch.float64 if positions.dtype == torch.float64 else torch.float32)
