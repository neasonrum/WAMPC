"""Compare the same-seed repeat, preserving original, replacement and best results."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue
from evaluation.prepare_carla_retest import strict, collision_free


def metrics(records):
    return {'routes': len(records), 'DS': sum(r['scores']['score_composed'] for r in records)/len(records),
            'RC': sum(r['scores']['score_route'] for r in records)/len(records),
            'strict_SR_percent': 100*sum(map(strict, records))/len(records),
            'collision_free_SR_percent': 100*sum(map(collision_free, records))/len(records)} if records else None


def summarize(run):
    reference = json.loads((run/'baseline_reference.json').read_text())
    baseline = reference['records']
    selected = json.loads((run/'selection.json').read_text())['route_ids']
    top = json.loads((run/'launch_topology.json').read_text())
    state = RouteQueue(run).snapshot()
    items = state['items'] if state else {}
    if state and 'identity' in reference:
        assert set(items) == set(selected), 'Retry queue differs from frozen selection'
        assert all(state['identity'].get(key) == value == top.get(key)
                   for key, value in reference['identity'].items()), 'Retry protocol differs from baseline'
    new = {r: v['last_result'] for r, v in items.items() if v['state'] == 'done'}
    assert set(new) <= set(selected)
    paired = sorted(set(new) & set(baseline), key=int)
    before = [baseline[r]['record'] for r in paired]
    after = [new[r]['record'] for r in paired]
    replacement = {**baseline, **new}
    rank = lambda d: (d['record']['scores']['score_composed'], strict(d['record']), collision_free(d['record']), d['record']['scores']['score_route'])
    best = {r: max([row for row in (baseline.get(r), new.get(r)) if row is not None], key=rank)
            for r in replacement}
    complete = len(new) == len(selected)
    finished = bool(items) and len(items) == len(selected) and all(item['state'] in ('done', 'error') for item in items.values())
    benchmark = set(reference.get('benchmark_route_ids', baseline))
    coverage_complete = len(best) == 220 and set(best) == benchmark
    deltas = []
    for r in paired:
        old = baseline[r]['record']; now = new[r]['record']
        deltas.append({'route_id': r, 'scenario': old['scenario_name'], 'old_DS': old['scores']['score_composed'],
            'new_DS': now['scores']['score_composed'], 'DS_delta': now['scores']['score_composed']-old['scores']['score_composed'],
            'old_strict_success': strict(old), 'new_strict_success': strict(now),
            'old_collision_free_success': collision_free(old), 'new_collision_free_success': collision_free(now),
            'new_status': now['status'], 'chosen_attempt_for_best_DS': best[r]['attempt'],
            'old_infractions': {k: len(v) for k, v in old['infractions'].items() if k != 'min_speed_infractions' and v},
            'new_infractions': {k: len(v) for k, v in now['infractions'].items() if k != 'min_speed_infractions' and v}})
    report = {'unix': time.time(), 'complete_77_routes': complete and len(selected) == 77,
        'complete_selected_routes': complete, 'campaign_finished': finished, 'complete_220_routes': coverage_complete,
        'selected_routes': len(selected), 'valid_repeat_results': len(new),
        'missing_route_ids': [r for r in selected if r not in new],
        'missing_benchmark_route_ids': sorted(benchmark-set(best), key=int),
        'new_coverage_route_ids': sorted(set(new)-set(baseline), key=int),
        'unrecovered_errors': {r: v.get('last_result') for r, v in items.items() if v['state'] == 'error'},
        'seed': 901, 'inference_steps': 100, 'checkpoint_sha256': top['checkpoint_sha256'],
        'baseline_220': metrics([d['record'] for d in baseline.values()]) if len(baseline) == 220 else None,
        'baseline_observed': metrics([d['record'] for d in baseline.values()]),
        'paired_original_observed': metrics(before), 'paired_repeat_observed': metrics(after),
        'repeat_observed': metrics([d['record'] for d in new.values()]),
        'replacement_220_provisional_until_complete': metrics([d['record'] for d in replacement.values()]),
        'best_of_2_220_provisional_until_complete': metrics([d['record'] for d in best.values()]),
        'final_best_of_2_220': metrics([d['record'] for d in best.values()]) if finished and coverage_complete else None,
        'best_record_rule': 'Higher DS, then strict success, collision-free completion, then RC; original wins exact ties',
        'strict_failure_to_success': sum(not x['old_strict_success'] and x['new_strict_success'] for x in deltas),
        'collision_failure_to_success': sum(not x['old_collision_free_success'] and x['new_collision_free_success'] for x in deltas),
        'collision_success_to_failure': sum(x['old_collision_free_success'] and not x['new_collision_free_success'] for x in deltas),
        'routes_DS_improved': sum(x['DS_delta'] > 1e-6 for x in deltas),
        'routes_DS_regressed': sum(x['DS_delta'] < -1e-6 for x in deltas), 'route_comparisons': deltas}
    tmp = run/'comparison.json.tmp'; tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(run/'comparison.json')
    if finished and coverage_complete:
        from evaluation.summarize_full_diffusion_carla import StatisticsManager
        for label, chosen in [('replacement', replacement), ('best_of_2', best)]:
            destination = run/f'{label}_merged_results.json'
            manager = StatisticsManager(str(destination), str(run/f'{label}_live_results.txt'))
            for d in chosen.values():
                manager.add_file_records(str(Path(d['attempt'])/'results.json'))
            manager.sort_records()
            first = json.loads((Path(next(iter(chosen.values()))['attempt'])/'results.json').read_text())
            manager.save_sensors(first.get('sensors', [])); manager.save_progress(220, 220)
            manager.compute_global_statistics(); manager.validate_and_write_statistics(True, False)
            (run/f'{label}_provenance.json').write_text(json.dumps({r: d['attempt'] for r, d in chosen.items()}, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('route_comparisons', 'missing_route_ids')}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-root', type=Path, required=True); p.add_argument('--watch', action='store_true')
    a = p.parse_args(); run = a.run_root.resolve(); run.relative_to(ROOT.resolve()/'results')
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    previous = None
    while True:
        queue = run/'route_queue.json'; version = queue.stat().st_mtime_ns if queue.exists() else 0
        if version != previous:
            report = summarize(run); previous = version
            if report['campaign_finished']:
                break
        if not a.watch:
            break
        time.sleep(30)
