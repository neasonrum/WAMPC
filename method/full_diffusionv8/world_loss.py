"""Masked feature losses; future observations enter supervision only."""
import torch
import torch.nn.functional as F

DEFAULTS = {'nmse': 1., 'corridor_nmse': .5, 'cosine': .05,
            'dynamic_nmse': .25, 'temporal_nmse': .1, 'multiscale_nmse': .1,
            'pool_scales': [2, 4], 'dynamic_weight_cap': 5.}
METRICS = ('nmse', 'mse', 'corridor_nmse', 'cosine', 'dynamic_nmse',
           'temporal_nmse', 'multiscale_nmse')


def masked_mean(value, weight):
    return (value * weight).sum((-1, -2, -3)) / weight.sum((-1, -2, -3)).clamp_min(1e-8)


def feature_error(prediction, target, weight, reference=None):
    active = weight > 0
    # Sanitize before subtraction so invalid NaNs do not poison gradients.
    prediction = torch.where(active, prediction, 0.).float()
    target = torch.where(active, target, 0.).float()
    energy = target.square().mean(2, keepdim=True) if reference is None else reference
    error = (prediction - target).square().mean(2, keepdim=True)
    mse = masked_mean(error, weight)
    power = masked_mean(energy, weight)
    return mse, mse / power.clamp_min(1e-4)


def error_metrics(prediction, target, spatial_valid, query_valid, corridor,
                  current, current_valid, settings=None):
    options = {**DEFAULTS, **(settings or {})}
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[1:] != (6, 64, 64, 64):
        raise ValueError('Expected matching Bx6x64x64x64 feature tensors')
    b = len(prediction)
    if (spatial_valid.shape != (b, 6, 1, 64, 64) or query_valid.shape != (b, 6)
            or current.shape != (b, 64, 64, 64) or current_valid.shape != (b, 1, 64, 64)):
        raise ValueError('Unexpected temporal/spatial validity or current teacher shape')
    mask = spatial_valid.bool() & query_valid[:, :, None, None, None].bool()
    prediction = torch.where(mask, prediction, 0.).float()
    target = torch.where(mask, target.detach(), 0.).float()
    current = torch.where(current_valid.bool(), current.detach(), 0.).float()
    weight = mask.float()
    mse, nmse = feature_error(prediction, target, weight)
    _, corridor_nmse = feature_error(prediction, target, weight * corridor[:, None].float())
    target_energy = target.square().mean(2, keepdim=True)
    cosine_weight = weight * (target_energy > 1e-8)
    cosine = masked_mean(F.cosine_similarity(prediction, target, dim=2, eps=1e-6)[:, :, None], cosine_weight)
    # No eligible target vector means no cosine penalty, including fully masked queries.
    cosine = torch.where(cosine_weight.sum((-1, -2, -3)) > 0, cosine, 1.)

    previous_target = torch.cat((current[:, None], target[:, :-1]), 1)
    previous_prediction = torch.cat((current[:, None], prediction[:, :-1]), 1)
    previous_valid = torch.cat((current_valid[:, None].bool(), mask[:, :-1]), 1)
    pair_mask = (mask & previous_valid).float()
    target_delta = target - previous_target
    change = target_delta.square().mean(2, keepdim=True)
    average_change = masked_mean(change, pair_mask)[:, :, None, None, None]
    cap = float(options['dynamic_weight_cap'])
    if not cap > 0:
        raise ValueError('dynamic_weight_cap must be positive')
    dynamic_weight = (change / average_change.clamp_min(1e-8)).clamp(0, cap).detach() * pair_mask
    _, dynamic_nmse = feature_error(prediction, target, dynamic_weight)
    # Normalize change error by feature power, avoiding unstable near-zero motion energy.
    reference = .5 * (target.square().mean(2, keepdim=True) + previous_target.square().mean(2, keepdim=True))
    _, temporal_nmse = feature_error(prediction - previous_prediction, target_delta, pair_mask, reference)

    pooled_errors = []
    for scale in options['pool_scales']:
        if not isinstance(scale, int) or scale < 2 or 64 % scale:
            raise ValueError('Pooling scales must divide 64 and be at least two')
        pooled_mask = F.avg_pool2d(weight.flatten(0, 1), scale, scale)
        pooled_mask = (pooled_mask >= 1 - 1e-6).float().reshape(b, 6, 1, 64 // scale, 64 // scale)
        pred = F.avg_pool2d(prediction.flatten(0, 1), scale, scale).reshape(b, 6, 64, 64 // scale, 64 // scale)
        truth = F.avg_pool2d(target.flatten(0, 1), scale, scale).reshape_as(pred)
        pooled_errors.append(feature_error(pred, truth, pooled_mask)[1])
    if not pooled_errors:
        raise ValueError('At least one pooled supervision scale is required')
    multiscale_nmse = torch.stack(pooled_errors).mean(0)
    return dict(zip(METRICS, (nmse, mse, corridor_nmse, cosine, dynamic_nmse,
                              temporal_nmse, multiscale_nmse)))


def reduce_queries(value, query_valid, spatial_valid):
    available = query_valid.bool() & spatial_valid.flatten(2).any(-1)
    per_sequence = (value * available).sum(1) / available.sum(1).clamp_min(1)
    return per_sequence.mean()


def loss_components(prediction, target, spatial_valid, query_valid, corridor,
                    current, current_valid, settings=None):
    options = {**DEFAULTS, **(settings or {})}
    metrics = error_metrics(prediction, target, spatial_valid, query_valid, corridor,
                            current, current_valid, options)
    reduced = {name: reduce_queries(value, query_valid, spatial_valid) for name, value in metrics.items()}
    per_query = (float(options['nmse']) * metrics['nmse']
                 + float(options['corridor_nmse']) * metrics['corridor_nmse']
                 + float(options['cosine']) * (1 - metrics['cosine'])
                 + float(options['dynamic_nmse']) * metrics['dynamic_nmse']
                 + float(options['temporal_nmse']) * metrics['temporal_nmse']
                 + float(options['multiscale_nmse']) * metrics['multiscale_nmse'])
    reduced['loss'] = reduce_queries(per_query, query_valid, spatial_valid)
    return reduced
