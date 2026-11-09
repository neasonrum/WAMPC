"""Collect fresh read-only LSF evidence on the login node; never submit jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_eval_resources import checked_run_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    run = checked_run_root(a.run_root)
    workers = json.loads((run/'launch_topology.json').read_text())['workers']
    target = run/'allocations'; target.mkdir(exist_ok=True)
    records = []
    for job in sorted({w['job_id'] for w in workers}):
        node = sorted([w for w in workers if w['job_id'] == job], key=lambda w: w['gpu'])
        details = subprocess.check_output(['bjobs', '-l', job], text=True, timeout=30)
        if not re.search(r'Sta\s*tus <RUN>', details):
            raise RuntimeError(f'Allocation {job} is no longer RUN; no replacement submitted')
        host = node[0]['host']
        if not re.search(r'<16\*'+re.escape(host)+r'>', details):
            raise RuntimeError(f'Allocation {job}: execution host/slot count changed')
        if not re.search(r'rusage\[mem=16000(?:\D|$)', details):
            raise RuntimeError(f'Allocation {job}: memory request changed')
        m = re.search(r'(?m)^(\w{3} \w{3}\s+\d+ [\d:]+)(?: (\d{4}))?: Started', details)
        if not m:
            raise RuntimeError(f'Allocation {job}: cannot establish start time')
        year = int(m[2]) if m[2] else datetime.now().year
        started = datetime.strptime(f'{m[1]} {year}', '%a %b %d %H:%M:%S %Y')
        if not m[2] and started.timestamp() > time.time()+60:
            started = started.replace(year=year-1)
        limit = float(re.search(r'RUNLIMIT\s*\n\s*([\d.]+)', details)[1])*60
        record = {'job_id': job, 'host': host, 'state': 'RUN', 'checked_unix': time.time(),
                  'deadline_unix': started.timestamp()+limit, 'mem_per_slot_mb': 16000,
                  'slots': 16, 'gpu_uuids': [w['gpu_uuid'] for w in node], 'lsf_details': details,
                  'timezone': time.strftime('%Z %z')}
        (target/f'allocation_{job}.json').write_text(json.dumps(record, indent=2)+'\n')
        records.append({'job': job, 'host': host, 'remaining_hours': (record['deadline_unix']-time.time())/3600})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
