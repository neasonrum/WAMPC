"""Verify and time full local-batch updates from the saved training state."""
import argparse
import json
from pathlib import Path
import time

import torch

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv3_ego12.allocation import job_memory
from .data import JointDataset, collate_with_predecessors
from .optimization import optimizer_groups, make_scheduler
from .prepare import ROOT, atomic_json
from .rolling import QuarterSecondPlanStore
from .streams import DistributedRouteStreams
from .train import build, prepare_batch, run_update
from .utils import restore_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda', 0)
    state = torch.load(Path(config['output']) / 'latest.pt', map_location='cpu', weights_only=False)
    policy, _ = build(config, device)
    policy.planner.base.load_state_dict(state['planner_base'], strict=True)
    policy.planner.adapter.load_state_dict(state['adapter'], strict=True)
    policy.world.model.load_state_dict(state['world_model'], strict=True)
    optimizer = torch.optim.AdamW(optimizer_groups(policy, config['training']), fused=True,
                                  betas=tuple(config['training']['adam_betas']), eps=config['training']['adam_eps'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler = make_scheduler(optimizer, config['training'], 9280)
    scheduler.load_state_dict(state['scheduler'])
    rolling = QuarterSecondPlanStore()
    rolling.load_state_dict(state['rank_states'][0]['rolling'])
    generator = torch.Generator(device=device)
    restore_rng(state['rank_states'][0]['rng'], generator)
    dataset = JointDataset(config['output'], config)
    sampler = DistributedRouteStreams(dataset.chains(), 32, 0, 8, config['seed'])
    sampler.load_state_dict(state['rank_states'][0]['sampler'])
    start_update = state['progress']['updates']
    del state
    perception = FrozenPlannerPerception(ROOT / config['backbone_directory']).to(device).eval()
    policy.train()
    records = []
    for indices in iter(sampler):
        started = time.perf_counter()
        value = collate_with_predecessors([dataset[index] for index in indices])
        loaded = time.perf_counter()
        batch, support = prepare_batch(value, perception, config, device)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        result, planner_norm, world_norm, filled = run_update(policy, policy, batch, support, optimizer, scheduler,
                                                             rolling, generator, config['training'])
        torch.cuda.synchronize()
        record = {'local_batch': 32, 'accumulation_steps': 1, 'source_update': start_update,
                  'data_seconds': loaded - started, 'perception_seconds': prepared - loaded,
                  'update_seconds': time.perf_counter() - prepared,
                  'loss': float(result['loss']), 'planner_grad_norm': planner_norm, 'world_grad_norm': world_norm,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'host_memory': job_memory(args.job_id)}
        records.append(record)
        print(json.dumps(record), flush=True)
        sampler.commit()
        if len(records) == 4:
            break
    atomic_json(Path(config['output']) / 'batch32_probe.json', {'status': 'PASS', 'records': records,
                'training_checkpoint_modified': False})


if __name__ == '__main__':
    main()
