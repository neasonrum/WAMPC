"""Freeze the 77 completed-but-unsuccessful routes for a same-seed repeat."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.prepare_carla_parallel_run import SOURCES


def strict(record):
    return record['status'] in ('Completed', 'Perfect') and not any(v for k, v in record['infractions'].items() if k != 'min_speed_infractions')


def collision_free(record):
    return record['status'] in ('Completed', 'Perfect') and not any(record['infractions'][k] for k in ('collisions_layout', 'collisions_vehicle', 'collisions_pedestrian'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    baseline = a.baseline_run.resolve(); run = a.run_root.resolve()
    baseline.relative_to(ROOT.resolve()/'results'); run.relative_to(ROOT.resolve()/'results')
    assert not run.exists(), 'Use a new run directory'
    summary = json.loads((baseline/'summary.json').read_text())
    assert summary['complete_220_routes'] and not summary['unrecovered_errors']
    rows = [json.loads(f.read_text()) for f in (baseline/'full').glob('worker*/route_*/summary.json')]
    assert len(rows) == 220 and len({d['route_id'] for d in rows}) == 220
    selected = [d for d in rows if d['status'] in ('Completed', 'Perfect') and not strict(d['record'])]
    assert len(selected) == 77
    topology = json.loads((baseline/'launch_topology.json').read_text())
    assert topology['epoch'] == 10 and topology['seed'] == 901 and topology['inference_steps'] == 100
    checkpoint = Path(topology['checkpoint']); checkpoint.resolve().relative_to(ROOT.resolve())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == summary['checkpoint_sha256'] == '031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3'
    names = (*SOURCES, 'evaluation/prepare_carla_retest.py', 'evaluation/launch_carla_retest_node.py', 'evaluation/report_carla_retest.py')
    assert all((ROOT/n).is_file() for n in names)
    run.mkdir(parents=True); (run/'checkpoint').mkdir()
    private = run/'checkpoint/epoch_10.pt'; shutil.copy2(checkpoint, private)
    ids = sorted([d['route_id'] for d in selected], key=int)
    for w in topology['workers']:
        w.pop('routes', None)
        w['slot_base'] = 3200+w['worker_index']*4
        w['tm_base'] = 30000+w['gpu']*8000
    topology.update(profile='same_seed_retest_77_completed_with_infractions', checkpoint=str(private),
                    checkpoint_sha256=digest, backend='direct', lanes=1, route_count=77,
                    route_ids=ids, created_utc=datetime.now(timezone.utc).isoformat())
    (run/'launch_topology.json').write_text(json.dumps(topology, indent=2)+'\n')
    baseline_map = {d['route_id']: d for d in rows}
    (run/'baseline_reference.json').write_text(json.dumps({'run': str(baseline), 'records': baseline_map}, indent=2)+'\n')
    (run/'selection.json').write_text(json.dumps({'route_ids': ids, 'count': 77,
        'criterion': 'Completed/Perfect and at least one upstream SR infraction; min_speed excluded',
        'original_strict_successes': sum(strict(d['record']) for d in rows),
        'original_collision_free_completed': sum(collision_free(d['record']) for d in rows),
        'original_simulation_seed': 901, 'repeat_simulation_seed': 901,
        'network_rng': 'Original checkpoint-seeded per-route generator, unchanged',
        'original_driving_hours_for_selected': sum(d['record']['meta']['duration_system'] for d in selected)/3600,
        'original_total_gpu_hours_for_selected': sum(d['elapsed_seconds'] for d in selected)/3600}, indent=2)+'\n')
    hashes = {}
    for name in names:
        target = run/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target); hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (run/'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (run/'run_config.json').write_text(json.dumps({'epoch': 10, 'checkpoint_sha256': digest,
        'seed': 901, 'inference_steps': 100, 'simulation_hz': 20, 'route_count': 77,
        'gpus': 8, 'lanes_per_gpu': 1, 'backend': 'direct', 'scheduling': 'Shared dynamic queue',
        'original_results_preserved': True, 'repeat_count': 1,
        'comparison': ['paired cohort', '220-route replacement', '220-route best of original and repeat'],
        'host_memory_soft_limit_gib': 64, 'wall_budget_seconds': 10800}, indent=2)+'\n')
    print(json.dumps({'prepared': str(run), 'routes': len(ids), 'seed': 901, 'inference_steps': 100,
                      'baseline_strict_SR': 100*sum(strict(d['record']) for d in rows)/220,
                      'baseline_collision_free_SR': 100*sum(collision_free(d['record']) for d in rows)/220,
                      'selected_original_total_gpu_hours': sum(d['elapsed_seconds'] for d in selected)/3600}, indent=2))
