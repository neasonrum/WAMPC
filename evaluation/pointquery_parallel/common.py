import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/full_diffusion_pointquery_8a100_20261124_v1')
RUN = OUTPUT / 'carla25'
ARTIFACTS = OUTPUT / 'parallel14'
JOBS = ('13188135', '13188133', '13188134')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def live(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ', 1)[1][0] != 'Z'


def owned(pid, job):
    p = Path(f'/proc/{pid}')
    if not live(pid):
        return False
    if f'/job.{job}.' not in (p / 'cgroup').read_text() or p.stat().st_uid != os.getuid():
        raise RuntimeError(f'Process {pid} is outside the selected user/allocation')
    return True

