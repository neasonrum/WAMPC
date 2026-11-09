"""Freeze FD2 checkpoint/source and prepare bounded speed comparisons or 220 routes."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.carla_eval_resources import checked_run_root


def inference_source_matches(name, original, current):
    training_classes = {'method/full_diffusion/data.py': 'GarageGeoDataset',
                        'method/full_diffusion2/data.py': 'GarageTemporalDataset'}
    allowed = []
    if name == 'method/full_diffusion/model.py':
        allowed = ['import os', 'BRIDGE_MODULES = Path(os.environ.get("WMMPC_BRIDGE_MODULES", str(BRIDGE_MODULES)))']
    elif name == 'method/full_diffusion2/data.py':
        allowed = ['import gzip', 'import io', 'import json', 'import multiprocessing',
                   'import cv2', 'import laspy', 'from method.full_diffusion2.snapshot import SensorSnapshot']
    elif name not in training_classes:
        return original == current
    signatures = {ast.dump(ast.parse(statement).body[0]) for statement in allowed}
    before, after = ast.parse(original), ast.parse(current)
    for tree in (before, after):
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ClassDef) and node.name == training_classes.get(name))]
    after.body = [node for node in after.body if ast.dump(node) not in signatures]
    return ast.dump(before) == ast.dump(after)


def prepare(run, checkpoint, routes, method='full_diffusion2'):
    if method not in ('full_diffusion2', 'full_diffusionv3'):
        raise ValueError(f'Unsupported temporal planner: {method}')
    run = checked_run_root(run)
    run.mkdir(exist_ok=False)
    baseline = ROOT/'results/full_diffusion_carla_dynamic_epoch10_20261031_v1/launch_topology.json'
    top = json.loads(baseline.read_text())
    private = run/'checkpoint/epoch_10.pt'
    private.parent.mkdir()
    shutil.copy2(checkpoint, private)
    profile = 'fd3_100_direct_carla' if method == 'full_diffusionv3' else 'fd2_100_direct_carla'
    top.update(method=method, profile=profile, checkpoint=str(private),
               checkpoint_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
               route_ids=routes, lanes=1, backend='direct', seed=901,
               created_utc=datetime.now(timezone.utc).isoformat())
    for worker in top['workers']:
        worker.pop('routes', None)
        worker['slot_base'] = 3700+worker['worker_index']*4
        worker['tm_base'] = 30000+worker['gpu']*8000
    (run/'launch_topology.json').write_text(json.dumps(top, indent=2)+'\n')
    sources = set()
    folders = ['method/full_diffusion', 'method/full_diffusion2', 'evaluation']
    if method == 'full_diffusionv3':
        folders.append('method/full_diffusionv3')
    for folder in folders:
        sources.update(p for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts)
    for name in ['third_party/carla_garage/team_code/sensor_agent.py',
                 'third_party/carla_garage/team_code/model.py',
                 'third_party/carla_garage/team_code/nav_planner.py',
                 'third_party/carla_garage/team_code/transfuser_utils.py',
                 'third_party/carla_garage/team_code/config.py',
                 'leaderboard/leaderboard/leaderboard_evaluator.py', 'leaderboard/data/bench2drive220.xml',
                 'configs/paths.json', 'ckps/full_diffusion/backbone/all_towns_seed1/config.json']:
        sources.add(ROOT/name)
    hashes = {}
    for source in sorted(sources):
        name = str(source.relative_to(ROOT))
        target = run/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    return top


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--method', choices=('full_diffusion2', 'full_diffusionv3'), default='full_diffusion2')
    p.add_argument('--routes', default='1711,1773')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--retry-baseline', type=Path)
    a = p.parse_args()
    run = a.run_root.resolve()
    available = [r.attrib['id'] for r in ET.parse(ROOT/'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    routes = available if a.full else a.routes.split(',')
    baseline_reference = None
    if a.retry_baseline:
        from evaluation.prepare_carla_retest import strict, collision_free
        baseline = a.retry_baseline.resolve()
        baseline.relative_to(ROOT.resolve()/'results')
        old_top = json.loads((baseline/'launch_topology.json').read_text())
        state = json.loads((baseline/'route_queue.json').read_text())
        digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        identity = {'method': a.method, 'checkpoint_sha256': digest, 'seed': 901,
                    'inference_steps': 100, 'backend': 'direct'}
        assert all(old_top[key] == value and state['identity'][key] == value for key, value in identity.items())
        assert set(state['items']) == set(available) and len(available) == 220
        frozen = json.loads((baseline/'source_sha256.json').read_text())
        policy_sources = [name for name in frozen if name.startswith(('method/full_diffusion/', 'method/full_diffusion2/', 'method/full_diffusionv3/'))
                          or name == 'third_party/carla_garage/team_code/sensor_agent.py']
        differences = [name for name in policy_sources if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != frozen[name]]
        changed = [name for name in differences if not inference_source_matches(name,
               (baseline/'source_snapshot'/name).read_text(), (ROOT/name).read_text())]
        assert not changed, f'Baseline policy or controller changed: {changed}'
        records = {route: item['last_result'] for route, item in state['items'].items() if item['state'] == 'done'}
        routes = sorted([route for route in available if route not in records or not strict(records[route]['record'])], key=int)
        if not routes:
            raise ValueError('The baseline has no missing or unsuccessful routes to repeat')
        baseline_reference = {'run': str(baseline), 'identity': identity, 'records': records,
                              'benchmark_route_ids': available,
                              'inference_compatible_source_changes': differences,
                              'missing_route_ids': [route for route in available if route not in records]}
    if len(routes) != len(set(routes)) or not set(routes) <= set(available):
        raise ValueError('Unknown or duplicate routes')
    if a.full or a.retry_baseline:
        top = prepare(run, a.checkpoint.resolve(), routes, a.method)
        for worker in top['workers']:
            # Disjoint 64-slot RPC pools on each host; per-route TM ports use
            # separate ranges and the shared queue's monotonically increasing serial.
            worker['slot_base'] = 4400 + worker['gpu']*200
            worker['tm_base'] = 31000 + worker['gpu']*2000
        top.update(route_count=len(routes), server_mode='warm', startup_wait='rpc',
                   routes_per_server=32, wall_budget_seconds=86400,
                   cpu_policy='Disjoint allocation CPU halves, 4 or 8 CPUs per GPU',
                   retry_policy='Up to three server starts; only infrastructure route failures retry, maximum two route attempts',
                   ram_guard='85% of job cgroup total; 64 GiB noncache; 40 GiB startup headroom')
        if baseline_reference is not None:
            top.update(profile='same_seed_repeat_missing_or_unsuccessful', baseline_run=str(baseline), repeat_count=1,
                       bridge_modules=str(ROOT.parent/'BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules'))
            (run/'baseline_reference.json').write_text(json.dumps(baseline_reference, indent=2)+'\n')
            selection = {'route_ids': routes, 'count': len(routes),
                         'criterion': 'Missing valid result, or not Completed/Perfect without upstream SR infractions; min_speed excluded',
                         'original_valid_routes': len(records), 'original_strict_successes': sum(strict(row['record']) for row in records.values()),
                         'original_collision_free_completed': sum(collision_free(row['record']) for row in records.values()),
                         'missing_route_ids': baseline_reference['missing_route_ids'],
                         'original_simulation_seed': 901, 'repeat_simulation_seed': 901,
                         'repeat_count': 1, 'original_results_preserved': True,
                         'best_record_rule': 'Higher DS, then strict success, collision-free completion, then RC; original wins exact ties'}
            (run/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
            (run/'run_config.json').write_text(json.dumps({key: value for key, value in top.items() if key != 'workers'}, indent=2)+'\n')
        (run/'launch_topology.json').write_text(json.dumps(top, indent=2)+'\n')
    else:
        # The top-level directory already holds the original source archive.
        cases = [
            ('cold_fixed30_cpu4', '13053006', 0, 4, 'cold', 'fixed30'),
            ('warm_rpc_cpu4_a', '13053006', 1, 4, 'warm', 'rpc'),
            ('cold_rpc_cpu4', '13047896', 0, 4, 'cold', 'rpc'),
            ('warm_rpc_cpu4_b', '13047896', 1, 4, 'warm', 'rpc'),
            ('warm_rpc_cpu4_c', '13047898', 0, 4, 'warm', 'rpc'),
            ('warm_rpc_cpu8_a', '13047898', 1, 8, 'warm', 'rpc'),
            ('cold_fixed30_cpu8', '13046830', 0, 8, 'cold', 'fixed30'),
            ('warm_rpc_cpu8_b', '13046830', 1, 8, 'warm', 'rpc'),
        ]
        plans = []
        for name, job, gpu, cpus, mode, wait in cases:
            case = run/'cases'/name
            case.parent.mkdir(exist_ok=True)
            top = prepare(case, a.checkpoint.resolve(), routes, a.method)
            worker = next(w for w in top['workers'] if w['job_id'] == job and w['gpu'] == gpu)
            plan = {'name': name, 'job_id': job, 'gpu': gpu, 'cpus_per_gpu': cpus,
                    'server_mode': mode, 'startup_wait': wait, 'run_root': str(case),
                    'checkpoint': top['checkpoint'], 'worker': worker, 'routes': routes,
                    'inference_steps': 100, 'wall_budget_seconds': 5400}
            (case/'case_config.json').write_text(json.dumps(plan, indent=2)+'\n')
            plans.append(plan)
        (run/'campaign.json').write_text(json.dumps({'cases': plans, 'purpose': 'Same-checkpoint paired CARLA overhead and CPU-affinity measurements',
            'limits': 'One CARLA plus one direct agent per GPU; 85% total cgroup RAM, 64 GiB working-set cap; no LSF changes',
            'caveat': 'Different GPUs within a host; two routes, not a full-220 timing or bitwise trajectory equivalence claim'}, indent=2)+'\n')
    print(json.dumps({'prepared': str(run), 'full': a.full, 'repeat': bool(a.retry_baseline), 'routes': len(routes)}))


if __name__ == '__main__':
    main()
