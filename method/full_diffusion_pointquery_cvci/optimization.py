"""Keep the baseline planner LR/decay/schedule, and the established WM LR groups."""
import math
from method.full_diffusionv8.optimization import WORLD_GROUPS, world_group


def optimizer_groups(policy, settings):
    groups = {}
    seen = set()
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("Duplicate optimizer ownership")
        seen.add(id(parameter))
        if name.startswith('world.model.'):
            group = world_group(name.removeprefix('world.model.'))
            rate, decay = settings['learning_rates'][group], settings['world_weight_decay']
            if parameter.ndim <= 1:
                decay = 0.
        elif name.startswith('planner.base.'):
            group, rate, decay = 'pretrained_pointquery', settings['learning_rate'], settings['weight_decay']
        elif name.startswith('planner.adapter.'):
            group, rate = 'future_adapter', settings['learning_rate']
            decay = settings['weight_decay'] if parameter.ndim > 1 else 0.
        else:
            raise ValueError(f"Unexpected trainable parameter {name}")
        key = group, decay
        record = groups.setdefault(key, {'params': [], 'param_names': [], 'lr': rate,
                                       'weight_decay': decay, 'module_group': group})
        record['params'].append(parameter)
        record['param_names'].append(name)
    return list(groups.values())


def make_scheduler(optimizer, settings, total):
    from torch.optim.lr_scheduler import LambdaLR
    steps_per_epoch = total // settings['epochs']
    return LambdaLR(optimizer, lambda update: .5 * (1 + math.cos(math.pi * ((update / steps_per_epoch) % 10) / 10)))
