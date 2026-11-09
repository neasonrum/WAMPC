"""Merge only valid, fixed-checkpoint route evaluations and audit infrastructure.

Uses the existing leaderboard statistics manager and Bench2Drive metric helpers.
Partial means are explicitly separate from a complete 220-route benchmark.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
paths = json.loads((ROOT / 'configs/paths.json').read_text())
for directory in (ROOT, ROOT / 'leaderboard', ROOT / 'scenario_runner',
                  Path(paths['carla_root']) / 'PythonAPI/carla'):
    sys.path.insert(0, str(directory))

import numpy as np
from leaderboard.utils.statistics_manager import StatisticsManager
from tools.ability_benchmark import get_infraction_status
from tools.efficiency_smoothness_benchmark import seg_compute_comfort_metric


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def summarize(run, worker_count=2):
    expected = [route.attrib['id'] for route in
                ET.parse(ROOT / 'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    summaries = [read(p) for p in sorted((run / 'full').glob('worker*/route_*/summary.json'))]
    summaries = [s for s in summaries if s]
    manifest_paths = sorted((run / 'full').glob('worker*/manifest.json'))
    manifests = [read(p) for p in manifest_paths]
    manifests = [m for m in manifests if m]
    identities = {(m['checkpoint_sha256'], m['seed']) for m in manifests}
    if len(identities) > 1:
        raise RuntimeError('Refusing to merge different checkpoints/seeds')
    planned = [route for m in manifests for route in m['routes']]
    if len(planned) != len(set(planned)):
        raise RuntimeError('Overlapping route shards')
    valid = [s for s in summaries if s['classification'] == 'driving_result']
    route_ids = [s['route_id'] for s in valid]
    if len(route_ids) != len(set(route_ids)) or not set(route_ids) <= set(expected):
        raise RuntimeError('Duplicate or unexpected evaluated routes')
    for summary in valid:
        record_id = summary['record']['route_id'].split('_')[1]
        if record_id != summary['route_id']:
            raise RuntimeError('Route manifest and record ID mismatch')
    missing = sorted(set(expected) - set(route_ids), key=int)
    failures = [s for s in summaries if s['classification'] != 'driving_result']
    worker_states = []
    for index in range(worker_count):
        worker = run / f'full/worker{index}'
        worker_states.append({'worker': index,
                              'completed': read(worker / 'completed.json'),
                              'needs_attention': read(worker / 'needs_attention.json'),
                              'progress': read(worker / 'progress.json'),
                              'status': read(worker / 'status.json')})
    complete = not missing and len(valid) == len(expected) and not failures
    attempts = []
    for p in sorted((run / 'full').glob('worker*/route_*/attempt_*/summary.json')):
        data = read(p)
        if data:
            attempts.append(data)
    unexpected_exits = []
    incomplete_exit_logs = []
    exception_logs = []
    for p in sorted((run / 'full').glob('worker*/route_*/attempt_*/server_exit.json')):
        data = read(p)
        if not data or 'returncode' not in data or 'evaluator_finishing' not in data:
            incomplete_exit_logs.append(str(p))
        elif not data['evaluator_finishing']:
            unexpected_exits.append({'file': str(p), **data})
    for p in sorted((run / 'full').glob('worker*/route_*/attempt_*/evaluation.log')):
        text = p.read_text(errors='replace')
        if 'Traceback (most recent call last):' in text or 'Exception in thread' in text:
            exception_logs.append(str(p))
    report = {'unix': time.time(), 'complete_220_routes': complete, 'worker_count': worker_count,
              'checkpoint_sha256': next(iter(identities))[0] if identities else None,
              'expected_routes': len(expected), 'valid_driving_results': len(valid),
              'missing_route_ids': missing, 'unrecovered_errors': failures,
              'attempt_classifications': dict(Counter(a['classification'] for a in attempts)),
              'unexpected_carla_exits': unexpected_exits, 'exception_logs': exception_logs,
              'incomplete_server_exit_logs': incomplete_exit_logs,
              'worker_states': worker_states, 'metrics_observed_routes': None,
              'metrics_full_benchmark': None}
    if valid:
        records = [s['record'] for s in valid]
        success = [r['status'] in ('Completed', 'Perfect') and not get_infraction_status(r)
                   for r in records]
        efficiency = []
        comfort = []
        metric_errors = []
        for summary in valid:
            record = summary['record']
            percentages = []
            for message in record['infractions']['min_speed_infractions']:
                number = re.search(r'\b\d+\.?\d*%', message)
                if number and float(number.group().rstrip('%')) <= 1000:
                    percentages.append(float(number.group().rstrip('%')))
            if percentages:
                efficiency.append(float(np.mean(percentages)))
            metric = Path(summary['attempt']) / 'agent_data' / record['save_name'] / 'metric_info.json'
            try:
                data = read(metric)
                if not data:
                    raise ValueError('Missing or empty metric_info.json')
                rows = [data[key] for key in sorted(data, key=int)]
                fields = ('acceleration', 'angular_velocity', 'forward_vector',
                          'right_vector', 'location', 'rotation')
                arrays = {key: np.asarray([row[key] for row in rows]) for key in fields}
                comfort.append(float(seg_compute_comfort_metric(**arrays)))
            except Exception as exc:
                metric_errors.append({'route_id': summary['route_id'], 'error': repr(exc)})
        metrics = {'DS': float(np.mean([r['scores']['score_composed'] for r in records])),
                   'route_completion_percent': float(np.mean([r['scores']['score_route'] for r in records])),
                   'SR_percent': 100.0 * float(np.mean(success)),
                   'Efficiency': float(np.mean(efficiency)) if efficiency else None,
                   'Comfortness_percent_upstream_tool': 100.0 * float(np.mean(comfort)) if comfort else None,
                   'routes': len(records), 'efficiency_routes': len(efficiency),
                   'comfort_routes': len(comfort), 'metric_errors': metric_errors}
        report['metrics_observed_routes'] = metrics
        report['comfort_definition'] = {
            'implementation': 'tools/efficiency_smoothness_benchmark.py:seg_compute_comfort_metric',
            'script_time_interval_seconds': 0.1,
            'actual_simulation_step_seconds': 0.05,
            'angular_velocity': 'Raw CARLA values, following the existing upstream metric tool',
            'comparability_note': 'Logging interval and angular-velocity conventions need checking before treating this as an exact paper Comfortness reproduction.'}
        if complete:
            report['metrics_full_benchmark'] = metrics
        merged = run / ('merged_results.json' if complete else 'partial_results.json')
        manager = StatisticsManager(str(merged), str(run / 'merged_live_results.txt'))
        for summary in valid:
            manager.add_file_records(str(Path(summary['attempt']) / 'results.json'))
        manager.sort_records()
        first = read(Path(valid[0]['attempt']) / 'results.json', {})
        manager.save_sensors(first.get('sensors', []))
        manager.save_progress(len(valid), len(valid))
        manager.compute_global_statistics()
        manager.validate_and_write_statistics(True, False)
        report['merged_observed_results'] = str(merged)
    save(run / 'summary.json', report)
    print(json.dumps({'complete_220_routes': complete, 'valid_driving_results': len(valid),
                      'missing': len(missing), 'errors': len(failures),
                      'unexpected_carla_exits': len(unexpected_exits),
                      'metrics': report['metrics_observed_routes']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run = args.run_root.resolve()
    run.relative_to(ROOT.resolve())
    signature = None
    while True:
        tracked = list((run / 'full').glob('worker*/route_*/summary.json'))
        tracked += list((run / 'full').glob('worker*/completed.json'))
        tracked += list((run / 'full').glob('worker*/needs_attention.json'))
        new_signature = tuple(sorted((str(p), p.stat().st_mtime_ns) for p in tracked))
        if new_signature != signature:
            report = summarize(run, args.workers)
            signature = new_signature
            terminal = all(w['completed'] or w['needs_attention'] for w in report['worker_states'])
            if terminal:
                save(run / 'evaluation_finished.json', {
                    'unix': time.time(), 'complete_220_routes': report['complete_220_routes'],
                    'valid_driving_results': report['valid_driving_results'],
                    'unrecovered_error_count': len(report['unrecovered_errors'])})
                return 0 if report['complete_220_routes'] else 2
        if not args.watch:
            return 0
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
