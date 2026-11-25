"""Exercise real cached inputs, V8 gradient ownership, and the online time contract."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from method.full_diffusion2.perception import FrozenPlannerPerception
from .data import JointDataset, collate_with_predecessors
from .joint_policy import training_prior
from .optimization import WORLD_GROUPS, world_group, optimizer_groups
from .prepare import ROOT, atomic_json
from .rolling import QuarterSecondPlanStore, advance_plan
from .train import build, prepare_batch, checkpoint_state
from .utils import atomic_save


def run(config, output, job):
    from .allocation import inspect
    proof = inspect(job, config)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dataset = JointDataset(output, config)
    candidates = np.flatnonzero(dataset.needs_predecessor)
    sample = next((item for index in candidates[:128] if
                   (item := dataset[int(index)])['prior_observation'] is not None and item['world_target_valid'].any()), None)
    if sample is None:
        raise RuntimeError('Preflight could not find an anchor with a physical t-0.25 predecessor')
    perception = FrozenPlannerPerception(ROOT / config['backbone_directory']).to(device).eval()
    policy, artifact = build(config, device)
    batch, support = prepare_batch(collate_with_predecessors([sample]), perception, config, device)
    torch.testing.assert_close(batch['world_target_poses'][..., :2], batch['future_positions'], atol=.01, rtol=1e-4)
    generator = torch.Generator(device=device).manual_seed(914)
    prior, valid, ages, filled = training_prior(QuarterSecondPlanStore(), policy, batch, support, generator)
    if filled != 1 or not valid[:, :5].all() or valid[:, -1].any() or not torch.equal(ages, ages.new_full(ages.shape, .25)):
        raise RuntimeError('The real predecessor did not exercise five supported future intervals')
    policy.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        future = policy.future(batch, prior, valid, ages)
        condition = policy.condition(batch, future)
        no_future = policy.condition(batch, None)
        clean = policy.planner.codec.encode(batch['path'], batch['temporal_delta'])
        time = torch.full((1,), .7, device=device)
        expected = policy.planner.base(clean, time, no_future)['predictions']
        torch.testing.assert_close(policy.planner(clean, time, condition)['predictions'], expected, rtol=0, atol=0)
        altered = dict(batch)
        altered['memory'] = batch['memory'].clone()
        altered['memory'][:, :64] = torch.randn_like(altered['memory'][:, :64]) * 100
        perturbed = policy.condition(altered, None)
        torch.testing.assert_close(policy.planner(clean, time, perturbed)['predictions'], expected, rtol=0, atol=0)
    if any(hasattr(layer, 'cross_agent_attention') for layer in list(policy.planner.base.layers) + list(policy.planner.base.temporal_layers)):
        raise RuntimeError('The planner still contains direct scene-memory attention')
    world_named = list(policy.world.model.named_parameters())
    world_parameters = [parameter for _, parameter in world_named]
    planner_parameters = list(policy.planner.parameters())
    gates = [reader.gate for branch in ('spatial', 'temporal') for reader in policy.planner.adapter[branch]]
    with torch.no_grad():
        for gate in gates:
            gate.fill_(.01)
    planning = policy.planning_step(batch, prior, valid, ages, generator)
    action_gradients = torch.autograd.grad(planning['loss'], world_parameters + planner_parameters, allow_unused=True)
    reconstruction = policy.world_losses(batch)
    world_gradients = torch.autograd.grad(reconstruction['loss'], world_parameters + planner_parameters, allow_unused=True)
    count = len(world_parameters)
    if any(gradient is not None for gradient in world_gradients[count:]):
        raise RuntimeError('BEV reconstruction loss reached planner parameters')
    def norm(gradients):
        terms = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
        result = float(torch.stack(terms).sum().sqrt()) if terms else 0.
        if not np.isfinite(result) or result <= 0:
            raise RuntimeError('An expected gradient path is missing or nonfinite')
        return result
    action_to_world, reconstruction_to_world = {}, {}
    for group in WORLD_GROUPS:
        action_to_world[group] = norm([gradient for (name, _), gradient in zip(world_named, action_gradients[:count])
                                      if world_group(name) == group])
        reconstruction_to_world[group] = norm([gradient for (name, _), gradient in zip(world_named, world_gradients[:count])
                                              if world_group(name) == group])
    norm(action_gradients[count:])
    with torch.no_grad():
        for gate in gates:
            gate.zero_()
    groups = optimizer_groups(policy, config['training'])
    ids = [id(parameter) for group in groups for parameter in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in policy.parameters() if p.requires_grad}:
        raise RuntimeError('The optimizer does not own every trainable parameter exactly once')
    # A constant-velocity plan has the same motion after advancing and applying real ego translation.
    delta = torch.tensor([[2., 0.]] * 6, device=device)
    source = torch.eye(3, device=device)
    current = source.clone()
    current[0, 2] = 1.
    advanced, supported = advance_plan(delta, source, current, .25)
    torch.testing.assert_close(advanced[:5], delta[:5])
    if supported.tolist() != [True] * 5 + [False]:
        raise RuntimeError('The unsupported sixth interval was not masked')
    state = checkpoint_state(policy, config, artifact, dataset.manifest,
                             {'epoch': 0, 'updates': 1, 'step_in_epoch': 0, 'validation_pending': False})
    atomic_save(Path(output) / 'interface_test.pt', state)
    from .policy import V8Network
    online = V8Network(perception.net, Path(output) / 'interface_test.pt', device)
    observation = sample
    with torch.no_grad():
        for tick in range(11):
            pose = np.array([0., 0., 0.])
            online.set_observation(pose, tick * .05)
            online(observation['rgb'][None].to(device), observation['lidar'][None].to(device),
                   observation['target_point'][None].to(device), observation['speed'][None].to(device),
                   observation['command'][None].to(device))
            if tick >= 5 and online.last_prior_tick != tick - 5:
                raise RuntimeError('Online inference did not select its exact fifth-previous plan')
        if not online.last_future_ready or online.last_history_ticks != [0, 5, 10]:
            raise RuntimeError('Online inference did not use the exact k-10/k-5/k history')
    report = {'status': 'PASS', 'inference_steps': 25, 'conditioning': config['conditioning'],
              'zero_gate_equals_ego12': True, 'spatial_memory_invariance': True,
              'planner_loss_to_world': action_to_world, 'world_loss_to_world': reconstruction_to_world,
              'planner_tensors_receiving_world_loss': 0, 'online_ticks_checked': 11,
              'online_last_history_ticks': online.last_history_ticks, 'online_last_prior_tick': online.last_prior_tick,
              'all_future_gates_restored_zero': True, 'training_updates': 0,
              'interface_test_checkpoint_only': True, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'allocation': proof}
    atomic_json(Path(output) / 'preflight.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, config['output'], args.job_id)
