"""Eight-GPU V8 joint training, fixed-route validation, and resumable checkpoints."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
from method.full_diffusionv3_ego12.allocation import job_cgroup, job_memory
from . import METHOD, FORMAT
from .data import JointDataset, collate_with_predecessors
from .joint_policy import JointPlanner, training_prior
from .model import build_planner
from .optimization import optimizer_groups, make_scheduler
from .prepare import ROOT, atomic_json, sha
from .rolling import QuarterSecondPlanStore
from .streams import DistributedRouteStreams
from .utils import atomic_save, barrier, move, seed, rng_state, restore_rng, verify_data
from .world import JointBCDE

STOP = False
FIELDS = ('loss', 'planner_loss', 'world_loss', 'world_nmse', 'spatial_l1', 'temporal_l1',
          'behavior_ce', 'speed_huber', 'speed_abs_error_sum', 'speed_valid',
          'future_ready', 'prior_ready', 'world_target_count')
COUNTS = {'speed_abs_error_sum', 'speed_valid', 'future_ready', 'prior_ready', 'world_target_count'}


def request_stop(*_):
    global STOP
    STOP = True


def build(config, device):
    seed(config['seed'])
    artifact = json.loads((ROOT / config['normalization_file']).read_text())
    if artifact['label_config'] != config['labels']:
        raise ValueError('The planner normalization and V8 label contract differ')
    planner = build_planner(config, artifact['normalization'])
    planner.load_pretrained(config, artifact)
    world = JointBCDE(config['world']['checkpoint'], ROOT / config['backbone_directory'] / 'model_0030_1.pth',
                      config['world']['checkpoint_sha256'])
    diffusion = SpatialTemporalVPFullDiffusion(planner.codec, **{
        key: value for key, value in config['diffusion'].items() if key != 'inference_steps'})
    return JointPlanner(planner, world, diffusion, config).to(device), artifact


@torch.no_grad()
def encode_context(batch, perception, config):
    bevs, memories, egos = [], [], []
    for start in range(0, len(batch['rgb']), config['perception_chunk']):
        inputs = {key: batch[key][start:start + config['perception_chunk']]
                  for key in ('rgb', 'lidar', 'speed', 'command', 'target_point')}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            bev, memory, ego = perception(inputs)
        bevs.append(bev)
        memories.append(memory)
        egos.append(ego)
    batch['memory'], batch['ego'] = torch.cat(memories), torch.cat(egos)
    batch['raw_bev'] = torch.cat((batch['raw_bev'][:, :2], torch.cat(bevs)[:, None]), 1)
    del batch['rgb'], batch['lidar']
    return batch


def prepare_batch(value, perception, config, device):
    batch, support = value
    batch = encode_context(move(batch, device), perception, config)
    if support is not None:
        rows, observation = support
        support = (rows, encode_context(move(observation, device), perception, config))
    return batch, support


def checkpoint_state(policy, config, artifact, metadata, progress):
    return {'format': FORMAT, 'method': METHOD, 'experiment': config['experiment'], 'config': config,
            'normalization': artifact['normalization'], 'dataset': metadata,
            'planner_base': policy.planner.base.state_dict(), 'adapter': policy.planner.adapter.state_dict(),
            'world_model': policy.world.model.state_dict(), 'world': policy.world.provenance,
            'progress': dict(progress)}


def save_latest(output, policy, optimizer, scheduler, config, artifact, metadata, progress, sampler, rolling, generator):
    rank, size = dist.get_rank(), dist.get_world_size()
    local = {'sampler': sampler.state_dict(), 'rolling': rolling.state_dict(), 'rng': rng_state(generator)}
    states = [None] * size if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    if rank == 0:
        state = checkpoint_state(policy, config, artifact, metadata, progress)
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rank_states=states)
        atomic_save(output / 'latest.pt', state)
        atomic_json(output / 'latest.json', {'progress': progress, 'sha256': sha(output / 'latest.pt'),
                                           'optimizer_saved': True, 'scheduler_saved': True,
                                           'bytes': (output / 'latest.pt').stat().st_size, 'unix': time.time()})
    barrier()


class ValidationBatches:
    def __init__(self, dataset, rank, world, batch):
        grouped = defaultdict(list)
        for index, (route, _, frame) in enumerate(dataset.samples):
            grouped[int(route)].append((int(frame), index))
        self.routes = [sorted(grouped[key]) for key in sorted(grouped)[rank::world]]
        self.batch = batch

    def __iter__(self):
        active = [iter(route) for route in self.routes]
        while active:
            rows, keep = [], []
            for route in active[:self.batch]:
                item = next(route, None)
                if item is not None:
                    rows.append(item[1])
                    keep.append(route)
            active = keep + active[self.batch:]
            if rows:
                yield rows


@torch.no_grad()
def validate(dataset, perception, policy, config, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    batches = ValidationBatches(dataset, rank, world, config['training']['validation_batch_per_rank'])
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0, collate_fn=collate_with_predecessors,
                        pin_memory=True)
    rolling = QuarterSecondPlanStore()
    generator = torch.Generator(device=device).manual_seed(config['validation']['fixed_noise_seed'] + rank)
    prior_generator = torch.Generator(device=device).manual_seed(config['validation']['fixed_noise_seed'] + rank + 1000000)
    was_training = policy.training
    policy.eval()
    routes, world_nmse_sum, count = {}, 0., 0
    for value in loader:
        batch, support = prepare_batch(value, perception, config, device)
        prior, valid, ages, _ = training_prior(rolling, policy, batch, support, prior_generator)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            future = policy.future(batch, prior, valid, ages)
            condition = policy.condition(batch, future)
            path, delta = policy.sample(condition, len(prior), device, generator, config['validation']['inference_steps'])
            reconstruction = policy.world_losses(batch)
        if not bool(torch.isfinite(path).all() and torch.isfinite(delta).all()):
            raise FloatingPointError('Validation sampling generated a nonfinite trajectory')
        path_error = (path.float() - batch['path']).norm(dim=-1).mean(-1)
        future_error = (delta.float().cumsum(1) - batch['future_positions']).norm(dim=-1).mean(-1)
        speed_error = (policy.planner.codec.target_speed(delta) - policy.planner.codec.target_speed(batch['temporal_delta'])).abs()
        values = torch.stack((path_error, future_error, speed_error), -1).cpu().tolist()
        for route, metrics in zip(batch['chain_id'], values):
            previous = routes.setdefault(route, [0., 0., 0., 0])
            for index, metric in enumerate(metrics):
                previous[index] += metric
            previous[3] += 1
        world_nmse_sum += float(reconstruction['nmse']) * len(prior)
        count += len(prior)
        rolling.update(batch['chain_id'], batch['current_pose'], batch['timestamp'], delta)
    results = [None] * world
    dist.all_gather_object(results, {'routes': routes, 'world_nmse_sum': world_nmse_sum, 'samples': count})
    combined = {}
    for result in results:
        if combined.keys() & result['routes'].keys():
            raise ValueError('A validation route was processed by more than one rank')
        combined.update(result['routes'])
    if len(combined) != 44 or sum(result['samples'] for result in results) != 2223:
        raise ValueError('Validation must cover the complete fixed 44-route development set')
    macro = np.mean([np.asarray(value[:3]) / value[3] for value in combined.values()], axis=0)
    total = sum(result['samples'] for result in results)
    report = dict(zip(('route_macro_spatial_ade_m', 'route_macro_future_ade_m', 'route_macro_control_speed_mae_mps'), macro.tolist()))
    report.update(routes=44, samples=total, inference_steps=25,
                  world_nmse_sample_mean=sum(result['world_nmse_sum'] for result in results) / total,
                  prior_age_seconds=.25, per_route=combined, unix=time.time())
    policy.train(was_training)
    return report


def run_update(ddp, policy, batch, support, optimizer, scheduler, rolling, generator, settings):
    prior, valid, ages, filled = training_prior(rolling, policy, batch, support, generator)
    if len(prior) != settings['batch_per_gpu'] or settings['microbatch_per_gpu'] != settings['batch_per_gpu']:
        raise ValueError('V8 uses the complete local batch in one forward/backward pass')
    optimizer.zero_grad(set_to_none=True)
    result = ddp(batch, prior, valid, ages, generator)
    if not bool(torch.isfinite(result['loss'])):
        raise FloatingPointError('V8 joint loss is nonfinite')
    result['loss'].backward()
    totals = {name: result[name].detach().float() for name in FIELDS}
    planner_norm = torch.nn.utils.clip_grad_norm_(policy.planner.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
    world_norm = torch.nn.utils.clip_grad_norm_(policy.world.parameters(), settings['world_gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    rolling.update(batch['chain_id'], batch['current_pose'], batch['timestamp'], result['sampled_delta'])
    return totals, float(planner_norm), float(world_norm), filled


def compatible_resume_config(saved, current):
    saved, current = deepcopy(saved), deepcopy(current)
    # Execution batching can change while preserving the optimizer batch and its data stream.
    for value in (saved, current):
        value['training'].pop('microbatch_per_gpu', None)
    return saved == current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    job_cgroup(args.job_id)
    config = json.loads(args.config.read_text())
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    if (world != 8 or world != config['training']['world_size']
            or rank != config['jobs'][args.job_id]['node_rank'] * 2 + local):
        raise ValueError('Rank does not belong to the authorized four-allocation topology')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = Path(config['output'])
    verify_data(config)
    dataset, validation = JointDataset(output, config), JointDataset(output, config, 'val')
    metadata = dataset.manifest
    policy, artifact = build(config, device)
    perception = FrozenPlannerPerception(ROOT / config['backbone_directory']).to(device).eval()
    settings = config['training']
    steps = len(dataset) // (settings['batch_per_gpu'] * world)
    total = settings['epochs'] * steps
    optimizer = torch.optim.AdamW(optimizer_groups(policy, settings), fused=True,
                                  betas=tuple(settings['adam_betas']), eps=settings['adam_eps'])
    scheduler = make_scheduler(optimizer, settings, total)
    sampler = DistributedRouteStreams(dataset.chains(), settings['batch_per_gpu'], rank, world, config['seed'])
    rolling = QuarterSecondPlanStore()
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank + 100003)
    seed(config['seed'] + rank + 200003)
    progress = {'epoch': 0, 'step_in_epoch': 0, 'updates': 0, 'best_score': None, 'best_epoch': None,
                'validation_pending': False}
    resume = args.resume or int(os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')) > 0
    if resume:
        state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        if state['format'] != FORMAT or not compatible_resume_config(state['config'], config) or state['dataset'] != metadata:
            raise ValueError('Resume checkpoint/config/data identity mismatch')
        policy.planner.base.load_state_dict(state['planner_base'], strict=True)
        policy.planner.adapter.load_state_dict(state['adapter'], strict=True)
        policy.world.model.load_state_dict(state['world_model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        sampler.load_state_dict(state['rank_states'][rank]['sampler'])
        rolling.load_state_dict(state['rank_states'][rank]['rolling'])
        restore_rng(state['rank_states'][rank]['rng'], generator)
        progress = state['progress']
        del state
    elif (output / 'latest.pt').exists():
        raise RuntimeError('An existing training checkpoint requires --resume')
    policy.train()
    ddp = DDP(policy, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    workers = config['jobs'][args.job_id]['workers_per_rank']
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        pin_memory=True, persistent_workers=True, collate_fn=collate_with_predecessors,
                        multiprocessing_context='spawn', prefetch_factor=settings['prefetch_factor'], timeout=180)
    iterator = iter(loader)
    if rank == 0:
        atomic_json(output / 'training_plan.json', {'steps_per_epoch': steps, 'epochs': settings['epochs'],
                    'expected_updates': total, 'global_batch': settings['batch_per_gpu'] * world,
                    'microbatch_per_gpu': settings['microbatch_per_gpu'], 'world_size': world,
                    'train_samples': len(dataset), 'validation_samples': len(validation),
                    'planner_initialization': policy.planner.initialization, 'world_initialization': policy.world.provenance,
                    'optimizer_groups': [{key: value for key, value in group.items() if key != 'params'}
                                         for group in optimizer.param_groups], 'unix': time.time()})
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for epoch in range(progress['epoch'], settings['epochs']):
        first = progress['step_in_epoch'] if epoch == progress['epoch'] else 0
        for step in range(first, steps):
            began = time.perf_counter()
            value = next(iterator)
            data_wait = time.perf_counter() - began
            batch, support = prepare_batch(value, perception, config, device)
            policy.planner.monitor(progress['updates'] < 4 or (progress['updates'] + 1) % settings['log_every'] == 0)
            parts, planner_norm, world_norm, filled = run_update(ddp, policy, batch, support, optimizer, scheduler,
                                                               rolling, generator, settings)
            sampler.commit()
            progress.update(epoch=epoch, step_in_epoch=step + 1, updates=progress['updates'] + 1)
            update = progress['updates']
            if update <= 4 or update % settings['log_every'] == 0:
                torch.cuda.synchronize()
                numbers = torch.stack([parts[name] for name in FIELDS] + [torch.tensor(filled, device=device)])
                dist.all_reduce(numbers)
                if rank == 0:
                    metrics = {name: float(value) / (1 if name in COUNTS else world)
                               for name, value in zip(FIELDS, numbers[:-1])}
                    record = {**progress, **metrics, 'prior_backfilled': float(numbers[-1]),
                              'seconds_per_update': time.perf_counter() - began, 'data_wait_seconds': data_wait,
                              'planner_grad_norm': planner_norm, 'world_grad_norm': world_norm,
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                              'future_gates': policy.planner.gates(), 'future_attention': policy.planner.diagnostics(),
                              'world_size': world, 'global_batch': world * settings['batch_per_gpu'],
                              'workers_per_rank': workers, 'unix': time.time()}
                    with (output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, allow_nan=False) + '\n')
                    atomic_json(output / 'status.json', {'state': 'training', **record})
                    print(json.dumps(record), flush=True)
            memory = job_memory(args.job_id)
            stop = torch.tensor(int(STOP or (output / 'STOP').is_file()
                                    or time.time() >= config['train_stop_unix']
                                    or memory['noncache_bytes'] > .75 * memory['limit_bytes']), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if update == 1 or update % settings['checkpoint_every'] == 0 or bool(stop):
                save_latest(output, policy, optimizer, scheduler, config, artifact, metadata,
                            progress, sampler, rolling, generator)
            if bool(stop):
                if rank == 0:
                    atomic_json(output / 'paused.json', {'progress': progress, 'memory': memory, 'unix': time.time()})
                dist.destroy_process_group()
                return
        progress['validation_pending'] = True
        save_latest(output, policy, optimizer, scheduler, config, artifact, metadata, progress, sampler, rolling, generator)
        if rank == 0:
            atomic_json(output / 'status.json', {'state': 'validating', **progress, 'unix': time.time()})
        report = validate(validation, perception, policy, config, device)
        score = report[config['validation']['selection_metric']]
        improved = progress['best_score'] is None or score < progress['best_score']
        progress.update(epoch=epoch + 1, step_in_epoch=0, validation_pending=False)
        if improved:
            progress.update(best_score=score, best_epoch=epoch + 1)
        if rank == 0:
            report.update(epoch=epoch + 1, updates=progress['updates'], selected_best=improved)
            with (output / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(report, allow_nan=False) + '\n')
            state = checkpoint_state(policy, config, artifact, metadata, progress)
            state['validation'] = report
            atomic_save(output / f'epoch_{epoch + 1:02}.pt', state)
            if improved:
                atomic_save(output / 'best.pt', state)
                atomic_json(output / 'best.json', {'epoch': epoch + 1, 'score': score,
                            'selection_metric': config['validation']['selection_metric'],
                            'checkpoint_sha256': sha(output / 'best.pt'), 'validation': report})
            print('VALIDATION ' + json.dumps(report), flush=True)
        save_latest(output, policy, optimizer, scheduler, config, artifact, metadata, progress, sampler, rolling, generator)
    if rank == 0:
        if progress['updates'] != total:
            raise ValueError('Training update count differs from the requested ten-epoch budget')
        atomic_json(output / 'completed.json', {'progress': progress, 'best': json.loads((output / 'best.json').read_text()),
                    'unix': time.time(), 'expected_updates': total})
        atomic_json(output / 'status.json', {'state': 'training_complete', **progress, 'unix': time.time()})
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
