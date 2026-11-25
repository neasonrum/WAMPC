"""Check real-data geometry, both gradient paths, online timing and batch32 peaks."""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3_ego12.allocation import job_memory
from .data import JointDataset, collate_with_predecessors
from .joint_policy import training_prior
from .optimization import WORLD_GROUPS, world_group, optimizer_groups
from .rolling import QuarterSecondPlanStore, advance_plan
from .train import build, prepare_batch, checkpoint_state
from .utils import ROOT, atomic_json, atomic_save, inspect_allocation, verify_data, sha


def norm(gradients):
    values = [g.float().square().sum() for g in gradients if g is not None]
    value = float(torch.stack(values).sum().sqrt()) if values else 0.
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Expected a finite, nonzero gradient, got {value}")
    return value


def run(config, job):
    output = Path(config['output'])
    proof = inspect_allocation(config, job, idle=True)
    metadata = verify_data(config)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda', 0)
    dataset = JointDataset(output, config)
    candidates = np.flatnonzero(dataset.needs_predecessor)
    sample = next((item for index in candidates[:128]
                   if (item := dataset[int(index)])['prior_observation'] is not None
                   and item['world_target_valid'].any()), None)
    if sample is None:
        raise RuntimeError("No real predecessor was available for interface preflight")
    perception = FrozenPlannerPerception(ROOT / config['backbone_directory']).to(device).eval()
    policy, artifact = build(config, device)
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(914)
    batch, support = prepare_batch(collate_with_predecessors([sample]), perception, config, device)
    prior, valid, ages, filled = training_prior(QuarterSecondPlanStore(), policy, batch, support, generator)
    if filled != 1 or not valid[:, :5].all() or valid[:, -1].any() or not torch.equal(ages, ages.new_full(ages.shape, .25)):
        raise RuntimeError("The physical t-0.25 plan must support five future intervals")
    torch.testing.assert_close(batch['world_target_poses'][..., :2], batch['future_positions'], atol=.01, rtol=1e-4)
    readers = list(policy.planner.adapter['spatial']) + list(policy.planner.adapter['temporal'])
    gates = [reader.gate for reader in readers]
    original_gates = [gate.detach().clone() for gate in gates]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        future = policy.future(batch, prior, valid, ages)
        condition = policy.condition(batch, future)
        no_future = policy.condition(batch, None)
        clean = policy.planner.codec.encode(batch['path'], batch['temporal_delta'])
        timestamp = torch.full((1,), .7, device=device)
        expected = policy.planner.base(clean, timestamp, no_future)['predictions']
        torch.testing.assert_close(policy.planner(clean, timestamp, no_future)['predictions'], expected, rtol=0, atol=0)
        if condition['future'][0]['tokens'].data_ptr() != condition['temporal_future'][0]['tokens'].data_ptr():
            raise RuntimeError("The branches are not using the same sampled future tensor")
        if readers[0].key.weight.data_ptr() == readers[2].key.weight.data_ptr():
            raise RuntimeError("The two branches unexpectedly share attention parameters")
        for gate in gates:
            gate.fill_(-100.)
        torch.testing.assert_close(policy.planner(clean, timestamp, condition)['predictions'], expected, rtol=0, atol=0)
        for gate, saved in zip(gates, original_gates):
            gate.copy_(saved)
        invalid = {**future, 'valid': torch.zeros_like(future['valid'])}
        unavailable = policy.condition(batch, invalid)
        torch.testing.assert_close(policy.planner(clean, timestamp, unavailable)['predictions'], expected, rtol=0, atol=0)
        altered = dict(batch)
        altered['memory'] = batch['memory'].clone()
        altered['memory'][:, :64] = torch.randn_like(altered['memory'][:, :64]) * 100
        perturbed = policy.condition(altered, None)
        torch.testing.assert_close(policy.planner(clean, timestamp, perturbed)['predictions'], expected, rtol=0, atol=0)
    current_projection = policy.planner.base.layers[0].cross_bev_attention.value_proj
    future_projection = policy.planner.adapter['sampler'].future_projection
    if {id(p) for p in current_projection.parameters()} & {id(p) for p in future_projection.parameters()}:
        raise RuntimeError("Current and future BEV projections must have separate parameters")
    world_named = list(policy.world.model.named_parameters())
    world_parameters = [p for _, p in world_named]
    planner_parameters = list(policy.planner.parameters())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        differentiable = policy.condition(batch, policy.future(batch, prior, valid, ages))
        result = policy.planner(clean, timestamp, differentiable)
        parts = policy.planner.base.loss_components(result, clean, batch['spatial_mask'], batch['temporal_mask'],
            batch['behavior_targets'], batch['behavior_mask'], config['loss'])
    branch_norms = {}
    for field in ('spatial_l1', 'temporal_delta_l1'):
        branch_norms[field] = norm(torch.autograd.grad(parts[field], list(future_projection.parameters()),
                                                       retain_graph=True, allow_unused=True))
    planning_gradients = torch.autograd.grad(parts['loss'], world_parameters + planner_parameters, allow_unused=True)
    reconstruction = policy.world_losses(batch)
    reconstruction_gradients = torch.autograd.grad(reconstruction['loss'], world_parameters + planner_parameters, allow_unused=True)
    n = len(world_parameters)
    if any(g is not None for g in reconstruction_gradients[n:]):
        raise RuntimeError("Feature loss must never reach the planner or its future sampler")
    planning_to_world, feature_to_world = {}, {}
    for group in WORLD_GROUPS:
        planning_to_world[group] = norm([g for (name, _), g in zip(world_named, planning_gradients[:n]) if world_group(name) == group])
        feature_to_world[group] = norm([g for (name, _), g in zip(world_named, reconstruction_gradients[:n]) if world_group(name) == group])
    norm(planning_gradients[n:])
    del planning_gradients, reconstruction_gradients, differentiable, result, parts, reconstruction, condition, no_future, future
    gc.collect()
    optimizer = torch.optim.AdamW(optimizer_groups(policy, config['training']), fused=True)
    profiles = []
    policy.train()
    for size in (4, 32):
        # The first anchor of a route can be a genuine cold start. For capacity
        # measurement, exercise real predecessor observations on every batch row.
        value = collate_with_predecessors([dict(sample, chain_id=f'capacity_probe_{i}') for i in range(size)])
        probe_batch, probe_support = prepare_batch(value, perception, config, device)
        probe_prior, probe_valid, probe_ages, _ = training_prior(QuarterSecondPlanStore(), policy, probe_batch, probe_support, generator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        policy.planner.monitor(True)
        result = policy(probe_batch, probe_prior, probe_valid, probe_ages, generator)
        if int(result['future_ready']) != size:
            raise RuntimeError("Capacity probes must activate the future branch for every row")
        if not torch.isfinite(result['loss']):
            raise FloatingPointError("Nonfinite full joint training probe")
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        memory = job_memory(job)
        if memory['noncache_bytes'] > .7 * memory['limit_bytes'] or memory['oom'] or memory['oom_kill']:
            raise RuntimeError(f"Allocation RAM does not have the required headroom: {memory}")
        profiles.append({'batch': size, 'seconds': time.perf_counter() - started,
            'loss': float(result['loss'].detach()), 'future_ready': int(result['future_ready']),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'memory': memory,
            'future_attention': policy.planner.diagnostics()})
        print('BATCH_PROBE ' + json.dumps(profiles[-1]), flush=True)
        del value, probe_batch, probe_support, result
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
    atomic_save(output / 'preflight_optimizer.pt', {'optimizer': optimizer.state_dict(), 'probe_only': True})
    saved_optimizer = torch.load(output / 'preflight_optimizer.pt', map_location='cpu', weights_only=False)
    if not saved_optimizer['optimizer']['state']:
        raise RuntimeError("The optimizer probe did not materialize its resumable state")
    del saved_optimizer
    delta = torch.tensor([[2., 0.]] * 6, device=device)
    source = torch.eye(3, device=device)
    current = source.clone()
    current[0, 2] = 1.
    advanced, supported = advance_plan(delta, source, current, .25)
    torch.testing.assert_close(advanced[:5], delta[:5])
    if supported.tolist() != [True] * 5 + [False]:
        raise RuntimeError("Unsupported previous-plan tail was not masked")
    atomic_save(output / 'interface_test.pt', checkpoint_state(policy, config, artifact, dataset.manifest,
        {'epoch': 0, 'updates': 1, 'step_in_epoch': 0, 'validation_pending': False, 'interface_test_only': True}))
    from .policy import V8Network
    online = V8Network(perception.net, output / 'interface_test.pt', device)
    with torch.no_grad():
        for tick in range(11):
            online.set_observation(np.array([0., 0., 0.]), tick * .05)
            online(sample['rgb'][None].to(device), sample['lidar'][None].to(device),
                   sample['target_point'][None].to(device), sample['speed'][None].to(device), sample['command'][None].to(device))
            if tick >= 5 and online.last_prior_tick != tick - 5:
                raise RuntimeError("Online inference did not select exactly k-5")
        if not online.last_future_ready or online.last_history_ticks != [0, 5, 10]:
            raise RuntimeError("Online inference did not use exact k-10/k-5/k history")
    report = {'passed': True, 'inference_steps': 25, 'online_acceptance_passed': True,
        'strict_pointquery_epoch10_initialization': policy.planner.initialization,
        'future_disabled_equals_baseline_exactly': True, 'closed_gate_equals_baseline_exactly': True,
        'invalid_future_equals_baseline_exactly': True, 'scene_memory_invariant': True,
        'shared_future_tokens': True, 'independent_attention_parameters': True,
        'independent_current_and_future_projections': True, 'branch_loss_to_future_projection': branch_norms,
        'planner_loss_to_world': planning_to_world, 'feature_loss_to_world': feature_to_world,
        'planner_tensors_receiving_world_loss': 0, 'online_ticks_checked': 11,
        'online_last_prior_tick': online.last_prior_tick, 'online_last_history_ticks': online.last_history_ticks,
        'batch_profiles': profiles, 'optimizer_state_saved_and_read': True,
        'training_updates': 0, 'probe_only': True, 'data_protocol_sha256': metadata['protocol_sha256'],
        'allocation': proof, 'unix': time.time()}
    atomic_json(output / 'preflight.json', report)
    print('PREFLIGHT_PASS ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.job_id)
