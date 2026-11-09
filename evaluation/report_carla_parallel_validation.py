"""Aggregate the paired throughput experiment without changing any route result."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.summarize_parallel_carla import summarize
from evaluation.summarize_full_diffusion_carla import get_infraction_status

p = ROOT/'results/full_diffusion_carla_parallel_validation_20261031'
report = {'unix': time.time(), 'inference_steps': 100, 'sampler': 'Original VPFullDiffusion',
          'route_ids': ['1711', '1773', '3464', '3865'], 'cases': {}}
for name in ('direct_one', 'rpc_one', 'rpc_one_pair', 'rpc_two'):
    case = p/name
    groups = list((case/'groups').glob('*'))
    if not groups:
        continue
    group = groups[0]
    manifest = json.loads((group/'manifest.json').read_text())
    done = json.loads((group/'completed.json').read_text()) if (group/'completed.json').exists() else None
    metrics = summarize(case)['metrics_observed_routes']
    rows = [json.loads(s) for s in (group/'resources.jsonl').read_text().splitlines()]
    memories = []
    for sample in rows:
        for line in sample.get('gpu', '').splitlines():
            if manifest['gpu_uuid'] in line:
                memories.append(int(re.search(r'(\d+) MiB', line)[1]))
    exits = [json.loads(f.read_text()) for f in group.glob('lane*/servers/*/server_exit.json')]
    route_details = {}
    for path in case.glob('routes/*/summary.json'):
        result = json.loads(path.read_text())
        route_details[result['route_id']] = {
            'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
            'scenario': result['record']['scenario_name'],
            'simulation_seconds': result['record']['meta']['duration_game'],
            'evaluator_seconds': result['record']['meta']['duration_system'],
            'DS': result['record']['scores']['score_composed'],
        }
    report['cases'][name] = {
        'host': manifest['host'], 'gpu_uuid': manifest['gpu_uuid'], 'lanes': manifest['lanes'],
        'cpu_affinity': manifest['cpu_affinity'], 'complete': done is not None,
        'elapsed_seconds': done['elapsed_seconds'] if done else None, 'metrics': metrics,
        'job_memory_peak_gib_during_case': max(int(r['job_memory.current']) for r in rows)/1024**3,
        'job_memory_limit_bytes': int(rows[-1]['job_memory.max']),
        'selected_gpu_peak_mib': max(memories), 'job_memory_events': rows[-1]['job_memory.events'],
        'unexpected_owned_carla_exits': [d for d in exits if not d['owner_finishing']],
        'routes': route_details,
        'note': 'Early RPC trial before matching outer inference_mode' if name == 'rpc_one' else None,
    }
one = report['cases'].get('rpc_one_pair', {})
two = report['cases'].get('rpc_two', {})
if one.get('complete') and two.get('complete'):
    assert one['host'] == two['host']
    assert len(one['cpu_affinity']) == len(two['cpu_affinity']) == 4
    report['paired_comparison'] = {
        'speedup_factor': one['elapsed_seconds']/two['elapsed_seconds'],
        'wall_time_reduction_percent': 100*(1-two['elapsed_seconds']/one['elapsed_seconds']),
        'same_node': True, 'logical_cpu_threads_per_gpu': 4,
        'caution': 'Four short routes only; not a repeated full 220-route benchmark',
    }
for name in ('policy_replay_context01.json', 'policy_layout01.json'):
    if (p/name).exists():
        report[name.removesuffix('.json')] = json.loads((p/name).read_text())
source = json.loads((p/'source_snapshot_pair_v3/sha256.json').read_text())
report['original_core_source_unchanged'] = {
    n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == digest
    for n, digest in source.items() if n.startswith('method/')
}
formal = ROOT/'results/full_diffusion_carla_20261031_epoch10'
results = [json.loads(f.read_text()) for f in (formal/'full').glob('worker*/route_*/summary.json')]
records = [d['record'] for d in results if d['classification'] == 'driving_result']
report['original_formal_run'] = {'valid_routes': len(records),
    'unrecovered_errors': sum(d['classification'] != 'driving_result' for d in results),
    'DS': sum(r['scores']['score_composed'] for r in records)/len(records),
    'RC': sum(r['scores']['score_route'] for r in records)/len(records),
    'SR_percent': 100*sum(r['status'] in ('Completed', 'Perfect') and not get_infraction_status(r) for r in records)/len(records)}
(p/'parallel_validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('policy_replay_context01', 'policy_layout01')}, indent=2))
