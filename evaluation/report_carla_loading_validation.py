"""Summarize the completed CARLA loading comparison without launching a GPU job."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/full_diffusion_carla_20261031_lifecycle_validation'
ROUTES = ('1711', '1773')


def read(path):
    return json.loads(path.read_text())


def profile(directory):
    done = read(directory / 'completed.json')
    assert done['routes'] == done['driving_results'] == 2 and not done['errors']
    rows = []
    for route in ROUTES:
        summary = read(directory / f'route_{route}/summary.json')
        assert summary['classification'] == 'driving_result'
        record = summary['record']
        assert record['status'] == 'Completed'
        assert record['scores'] == {'score_route': 100, 'score_penalty': 1.0, 'score_composed': 100.0}
        rows.append({'route': route, 'scores': record['scores'], 'meta': record['meta'],
                     'elapsed_seconds': summary['elapsed_seconds'], 'attempt': summary['attempt']})
    driving = sum(row['meta']['duration_system'] for row in rows)
    return {'directory': str(directory), 'routes': rows, 'elapsed_seconds': done['elapsed_seconds'],
            'driving_system_seconds': driving,
            'non_driving_overhead_seconds': done['elapsed_seconds'] - driving}


def main():
    baseline = profile(RUN / 'paired01/baseline')
    warm = profile(RUN / 'warm01')
    manifest = read(RUN / 'warm01/manifest.json')
    assert manifest['inference_steps'] == 100
    assert manifest['fresh_python_per_route'] and manifest['fresh_agent_per_route']
    assert manifest['full_world_reload_per_route'] and manifest['fresh_traffic_manager_per_route']
    assert not manifest['static_road_cache']
    identities = []
    for route in ROUTES:
        attempt = Path(next(row['attempt'] for row in warm['routes'] if row['route'] == route))
        events = [json.loads(line) for line in (attempt / 'route_events.jsonl').read_text().splitlines()]
        setup = next(event for event in events if event['event'] == 'agent_setup_ready')
        assert setup['inference_steps'] == 100 and setup['sampler'] == 'VPFullDiffusion'
        assert setup['sampler_replaced'] is False
        lifecycle = [json.loads(line) for line in (attempt / 'lifecycle_events.jsonl').read_text().splitlines()]
        world = next(event for event in lifecycle if event['event'] == 'world_load_finished')
        assert world['mode'] == 'full_episode_reload'
        identities.append({'route': route, 'server': read(attempt / 'server_lease.json'),
                           'evaluator': read(attempt / 'evaluator_process.json'),
                           'rpc_readiness': read(attempt / 'startup_readiness.json'),
                           'world_load': world, 'agent_setup': setup})
    assert len({row['server']['pid'] for row in identities}) == 1
    assert len({row['evaluator']['pid'] for row in identities}) == 2
    done = read(RUN / 'warm01/completed.json')
    assert done['server_launches'] == 1
    sources = manifest['source_sha256']
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in sources.items())
    prepared = ROOT / 'results/full_diffusion_carla_20261031_epoch10_carla_warm'
    snapshot = read(prepared / 'source_sha256.json')
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in snapshot.items())
    saved = baseline['elapsed_seconds'] - warm['elapsed_seconds']
    overhead_saved = baseline['non_driving_overhead_seconds'] - warm['non_driving_overhead_seconds']
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'baseline': baseline, 'warm': warm,
              'elapsed_seconds_saved': saved,
              'elapsed_reduction_percent': 100 * saved / baseline['elapsed_seconds'],
              'non_driving_overhead_seconds_saved': overhead_saved,
              'non_driving_overhead_reduction_percent':
                  100 * overhead_saved / baseline['non_driving_overhead_seconds'],
              'overhead_definition': 'Overall controller wall time minus sum of CARLA duration_system; includes startup, loading, setup and teardown, not a pure map-load timer',
              'checkpoint_sha256': manifest['checkpoint_sha256'], 'inference_steps': 100,
              'source_sha256': sources, 'prepared_source_snapshot_matches': True,
              'server_startup': read(RUN / 'warm01/servers/server_001/startup_readiness.json'),
              'route_process_and_timing_evidence': identities,
              'limitation': 'Two closed-loop routes with matching scores; driving durations vary. This does not establish identical trajectories or full-220 runtime.',
              'full_220_warm_run_launched': False}
    target = RUN / 'warm01_loading_report.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(target), 'elapsed_reduction_percent': report['elapsed_reduction_percent'],
                      'overhead_reduction_percent': report['non_driving_overhead_reduction_percent'],
                      'baseline_overhead_seconds': baseline['non_driving_overhead_seconds'],
                      'warm_overhead_seconds': warm['non_driving_overhead_seconds'],
                      'prepared_sources_match': True}, indent=2))


if __name__ == '__main__':
    main()
