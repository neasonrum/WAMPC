"""The donor V8 parameter groups and scratch-planner learning-rate schedule."""
import math
from torch import nn

WORLD_GROUPS = ('world_history', 'world_action', 'world_dynamics', 'world_decoder')


def world_group(name):
    if name.startswith('history_encoder.'):
        return 'world_history'
    if name.startswith(('path_', 'cross.', 'cross_norm.', 'action.')):
        return 'world_action'
    if name.startswith(('state_input.', 'gates.', 'candidate.', 'transition.')):
        return 'world_dynamics'
    if name.startswith('decode.'):
        return 'world_decoder'
    raise ValueError(f'Unclassified BCDE parameter {name}')


def parameter_group(name, planner):
    if name.startswith('planner.adapter.'):
        if name.endswith('.gate'):
            return 'gate'
        if name.endswith(('.distance_logit', '.time_logit')):
            return 'distance'
        return 'new'
    relative = name.removeprefix('planner.base.')
    module = relative.split('.')[0]
    if module in ('layers', 'temporal_layers'):
        if '.task_decoder.' in relative:
            return 'base_output'
        stack = planner.base.layers if module == 'layers' else planner.base.temporal_layers
        return 'base_decoder_late' if int(relative.split('.')[1]) == len(stack) - 1 else 'base_decoder_early'
    if module in ('type_embedding', 'slot_embedding', 'path_query_encoder', 'temporal_query_encoder',
                  'time_mlp', 'temporal_time_mlp'):
        return 'base_query'
    if module in ('temporal', 'target_point_encoder'):
        return 'base_context'
    if module in ('behavior_head', 'behavior_condition', 'temporal_behavior_condition'):
        return 'base_behavior'
    raise ValueError(f'Unclassified planner parameter {name}')


def optimizer_groups(policy, settings):
    groups, owners = {}, dict(policy.named_modules())
    seen = set()
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError('An optimizer parameter was listed twice')
        seen.add(id(parameter))
        if name.startswith('world.model.'):
            group = world_group(name.removeprefix('world.model.'))
            decay = settings['world_weight_decay']
        else:
            group = parameter_group(name, policy.planner)
            decay = settings['weight_decay'] if group == 'new' or group.startswith('base_') else 0.
        if parameter.ndim <= 1 or isinstance(owners[name.rpartition('.')[0]], (nn.LayerNorm, nn.GroupNorm, nn.Embedding)):
            decay = 0.
        key = (group, decay)
        if key not in groups:
            groups[key] = {'params': [], 'param_names': [], 'lr': settings['learning_rates'][group],
                           'weight_decay': decay, 'module_group': group, 'name': f'{group}/wd{decay}'}
        groups[key]['params'].append(parameter)
        groups[key]['param_names'].append(name)
    if {group for group, _ in groups} != set(settings['learning_rates']):
        raise ValueError('Optimizer groups do not cover the complete model')
    return list(groups.values())


def make_scheduler(optimizer, settings, total):
    from torch.optim.lr_scheduler import LambdaLR
    def multiplier(step):
        warmup = settings['warmup_updates']
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, total - warmup - 1))
        minimum = settings['minimum_lr_ratio']
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)
