"""Report a dynamic queue with the same upstream driving metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue
from evaluation.carla_eval_resources import checked_run_root


def summarize(run, extended=False, merge=False):
    from evaluation.summarize_full_diffusion_carla import get_infraction_status, StatisticsManager
    state = RouteQueue(run).snapshot()
    if state is None:
        return {'status': 'prepared', 'valid_driving_results': 0, 'complete_220_routes': False}
    valid = []
    for route, item in state['items'].items():
        if item['state'] == 'done':
            result = item['last_result']
            assert result['classification'] == 'driving_result'
            assert result['route_id'] == route == result['record']['route_id'].split('_')[1]
            valid.append(result)
    records = [s['record'] for s in valid]
    expected = [r['id'] for r in state['routes']]
    missing = sorted(set(expected)-{s['route_id'] for s in valid}, key=int)
    attempts = [h['summary'] for item in state['items'].values() for h in item['history']]
    manifests = [json.loads(p.read_text()) for p in (run/'groups').glob('*/manifest.json')]
    assert all(m['identity'] == state['identity'] for m in manifests)
    complete = not missing and len(valid) == 220
    report = {'unix': time.time(), 'complete_220_routes': complete, 'expected_routes': len(expected),
              'valid_driving_results': len(valid), 'missing_route_ids': missing,
              'states': dict(Counter(item['state'] for item in state['items'].values())),
              'unrecovered_errors': {r: v.get('last_result') for r, v in state['items'].items() if v['state'] == 'error'},
              'attempt_classifications': dict(Counter(a['classification'] for a in attempts)),
              'checkpoint_sha256': state['identity']['checkpoint_sha256'], 'inference_steps': 100,
              'group_count': len(manifests), 'metrics_observed_routes': None, 'metrics_full_benchmark': None}
    if manifests:
        start = min(m['unix'] for m in manifests)
        end = max([h['unix'] for item in state['items'].values() for h in item['history']] or [time.time()])
        report['wall_seconds_to_latest_route_result'] = end-start
    if valid:
        metrics = {'DS': sum(r['scores']['score_composed'] for r in records)/len(records),
                   'route_completion_percent': sum(r['scores']['score_route'] for r in records)/len(records),
                   'SR_percent': 100*sum(r['status'] in ('Completed', 'Perfect') and not get_infraction_status(r) for r in records)/len(records),
                   'routes': len(records)}
        if extended:
            import numpy as np
            from evaluation.summarize_full_diffusion_carla import seg_compute_comfort_metric
            efficiency = []; comfort = []; errors = []
            for s in valid:
                percentages = []
                for message in s['record']['infractions']['min_speed_infractions']:
                    number = re.search(r'\b\d+\.?\d*%', message)
                    if number and float(number.group().rstrip('%')) <= 1000:
                        percentages.append(float(number.group().rstrip('%')))
                if percentages:
                    efficiency.append(float(np.mean(percentages)))
                try:
                    path = Path(s['attempt'])/'agent_data'/s['record']['save_name']/'metric_info.json'
                    data = json.loads(path.read_text())
                    rows = [data[k] for k in sorted(data, key=int)]
                    arrays = {k: np.asarray([r[k] for r in rows]) for k in
                              ('acceleration', 'angular_velocity', 'forward_vector', 'right_vector', 'location', 'rotation')}
                    comfort.append(float(seg_compute_comfort_metric(**arrays)))
                except Exception as e:
                    errors.append({'route_id': s['route_id'], 'error': repr(e)})
            metrics.update(Efficiency=float(np.mean(efficiency)) if efficiency else None,
                           Comfortness_percent_upstream_tool=100*float(np.mean(comfort)) if comfort else None,
                           metric_errors=errors, efficiency_routes=len(efficiency), comfort_routes=len(comfort))
            report['comfort_comparability_note'] = 'Uses the existing upstream script with its 0.1s convention; actual simulator step is 0.05s. Exact paper comparability is not established.'
        report['metrics_observed_routes'] = metrics
        if complete:
            report['metrics_full_benchmark'] = metrics
        if merge:
            destination = run/('merged_results.json' if complete else 'partial_results.json')
            manager = StatisticsManager(str(destination), str(run/'merged_live_results.txt'))
            for s in valid:
                manager.add_file_records(str(Path(s['attempt'])/'results.json'))
            manager.sort_records()
            first = json.loads((Path(valid[0]['attempt'])/'results.json').read_text())
            manager.save_sensors(first.get('sensors', [])); manager.save_progress(len(valid), len(expected))
            manager.compute_global_statistics(); manager.validate_and_write_statistics(True, False)
            report['merged_observed_results'] = str(destination)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--extended', action='store_true')
    p.add_argument('--merge', action='store_true')
    a = p.parse_args(); run = checked_run_root(a.run_root)
    previous = None
    while True:
        path = run/'route_queue.json'
        version = path.stat().st_mtime_ns if path.exists() else 0
        if version != previous:
            report = summarize(run, a.extended, a.merge)
            temporary = run/'summary.json.tmp'
            temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(run/'summary.json')
            print(json.dumps({k: report.get(k) for k in ('complete_220_routes', 'valid_driving_results', 'states', 'metrics_observed_routes')}), flush=True)
            previous = version
            if report['complete_220_routes'] or report.get('states', {}).get('error'):
                break
        if not a.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
