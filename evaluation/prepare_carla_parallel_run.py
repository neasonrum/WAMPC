"""Prepare an immutable shared-queue benchmark without launching GPU work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'evaluation/prepare_carla_parallel_run.py', 'evaluation/launch_carla_parallel_node.py',
    'evaluation/refresh_carla_parallel_allocations.py', 'evaluation/summarize_parallel_carla.py',
    'evaluation/run_carla_parallel.py', 'evaluation/carla_route_queue.py',
    'evaluation/shared_gpu_inference.py', 'evaluation/full_diffusion_rpc_agent.py',
    'evaluation/full_diffusion_parallel_direct_agent.py', 'evaluation/full_diffusion_fast_agent.py',
    'evaluation/full_diffusion_diagnostic_agent.py', 'evaluation/run_full_diffusion_carla.py',
    'evaluation/summarize_full_diffusion_carla.py',
    'evaluation/run_full_diffusion_carla_fast.py', 'evaluation/carla_fast_evaluator.py',
    'evaluation/carla_lifecycle.py', 'evaluation/carla_static_roads.py',
    'evaluation/fast_carla_io.py', 'evaluation/bench2drive.py',
    'method/full_diffusion/diffusion.py', 'method/full_diffusion/model.py',
    'method/full_diffusion/garage.py', 'method/full_diffusion/inference/policy.py',
    'method/full_diffusion/inference/carla_agent.py',
    'third_party/carla_garage/team_code/sensor_agent.py',
    'leaderboard/leaderboard/leaderboard_evaluator.py', 'leaderboard/data/bench2drive220.xml',
    'leaderboard/leaderboard/utils/statistics_manager.py',
    'tools/ability_benchmark.py', 'tools/efficiency_smoothness_benchmark.py',
    'ckps/full_diffusion/backbone/all_towns_seed1/config.json',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--backend', choices=('direct', 'rpc'), default='direct')
    p.add_argument('--lanes', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    checkpoint = a.checkpoint.resolve(); baseline = a.baseline_run.resolve(); run = a.run_root.resolve()
    checkpoint.relative_to(ROOT.resolve()); baseline.relative_to(ROOT.resolve())
    run.relative_to(ROOT.resolve()/'results')
    if run.exists():
        raise FileExistsError('Use a new run directory')
    if a.lanes == 2 and a.backend != 'rpc':
        raise ValueError('Two lanes require the experimental shared CUDA process')
    topology = json.loads((baseline/'launch_topology.json').read_text())
    workers = topology['workers']
    assert len(workers) == 8 and len({w['job_id'] for w in workers}) == 4
    assert {w['worker_index'] for w in workers} == set(range(8))
    for job in {w['job_id'] for w in workers}:
        node = [w for w in workers if w['job_id'] == job]
        assert {w['gpu'] for w in node} == {0, 1} and len({w['host'] for w in node}) == 1
    expected = [r.attrib['id'] for r in ET.parse(ROOT/'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    assert len(expected) == 220 and len(set(expected)) == 220
    for name in SOURCES:
        assert (ROOT/name).is_file(), name
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['config']['diffusion']['inference_steps'] == 100
    epoch = state['progress']['epoch']
    run.mkdir(parents=True); (run/'checkpoint').mkdir()
    private = run/'checkpoint'/f'epoch_{epoch}.pt'
    shutil.copy2(checkpoint, private)
    digest = hashlib.sha256(private.read_bytes()).hexdigest()
    for w in workers:
        w.pop('routes', None)
        w['slot_base'] = 3000 + w['worker_index']*4
        w['tm_base'] = 30000 + w['gpu']*8000
    topology.update(profile='dynamic_queue_original_100', checkpoint=str(private),
                    checkpoint_sha256=digest, epoch=epoch, inference_steps=100,
                    backend=a.backend, lanes=a.lanes, route_ids=expected,
                    created_utc=datetime.now(timezone.utc).isoformat())
    (run/'launch_topology.json').write_text(json.dumps(topology, indent=2)+'\n')
    hashes = {}
    for name in SOURCES:
        source = ROOT/name; target = run/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    config = {'epoch': epoch, 'checkpoint_sha256': digest, 'inference_steps': 100,
              'sampler': 'Original VPFullDiffusion', 'simulation_hz': 20, 'seed': topology['seed'],
              'backend': a.backend, 'lanes_per_gpu': a.lanes, 'gpu_count': 8,
              'route_count': 220, 'scheduling': 'Shared locked queue; same-town preference; no fixed route shards',
              'episode_reset': 'Full world reload; fresh evaluator, controller, history and traffic manager per route',
              'infrastructure_attempt_limit': 2, 'driving_failure_retry': False,
              'unexpected_worker_exit': 'Claim retained for inspection; no automatic duplicate execution',
              'cpu_threads_per_gpu': 4, 'host_memory_soft_limit_gib': 64,
              'default_wall_budget_seconds': 21600, 'lsf_deadline_headroom_seconds': 900,
              'server_route_limit': 32, 'captures_enabled': False,
              'status': 'Prepared only; no GPU work launched'}
    (run/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (run/'checkpoint/integrity.json').write_text(json.dumps({'sha256': digest, 'progress': state['progress'],
              'inference_steps': 100, 'bytes': private.stat().st_size}, indent=2)+'\n')
    print(json.dumps({'prepared': str(run), 'backend': a.backend, 'lanes_per_gpu': a.lanes,
                      'route_count': 220, 'gpu_count': 8, 'launched': False}, indent=2))


if __name__ == '__main__':
    main()
