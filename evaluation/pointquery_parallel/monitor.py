"""Keep a complete-run summary after original single-lane drivers hand off."""
import time
from evaluation.pointquery_parallel.common import RUN, ARTIFACTS, read, save
from evaluation.summarize_parallel_carla import summarize


def main():
    while True:
        report = summarize(RUN)
        if report['complete_220_routes']:
            save(RUN / 'summary.json', summarize(RUN, extended=True, merge=True))
            save(ARTIFACTS / 'completed.json', {'valid_driving_results': 220, 'unix': time.time()})
            return
        save(RUN / 'summary.json', report)
        now = time.time()
        active = {}
        for group in (RUN / 'groups').iterdir():
            p = group / 'resources.jsonl'
            if not p.exists():
                continue
            rows = p.read_text().splitlines()
            if not rows:
                continue
            latest = __import__('json').loads(rows[-1])
            if now - latest['unix'] > 30:
                continue
            lanes = []
            for lane in group.glob('lane_*'):
                if (lane / 'completed.json').exists():
                    continue
                claim = read(lane / 'claim.json')
                if claim:
                    lanes.append(claim['owner'])
            active[group.name] = lanes
        save(ARTIFACTS / 'progress.json', {'configured_maximum_lanes': 14,
             'active_claimed_lanes': sum(map(len, active.values())), 'active_groups': active,
             'valid_routes': report['valid_driving_results'], 'states': report['states'], 'unix': now})
        time.sleep(15)


if __name__ == '__main__':
    main()
