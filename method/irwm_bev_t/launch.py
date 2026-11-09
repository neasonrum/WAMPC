"""Single-node, two-GPU launch inside the explicitly authorized LSF allocation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ.get('WMMPC_ROOT', '/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC')).resolve()
OUTPUT_ROOT = Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC').resolve()
AUTHORIZED = {'13053006': 'rng-dl01-w005'}


def guard_context(job):
    if job not in AUTHORIZED or socket.gethostname().split('.')[0] != AUTHORIZED[job]:
        raise RuntimeError('Wrong authorized allocation host')
    if not re.search(rf'\b{job}\b', Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Not inside the requested LSF cgroup; use battach')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def memory_snapshot():
    rel = Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    base = Path('/sys/fs/cgroup') / rel.lstrip('/')
    candidates = [p for p in (base, base.parent) if (p / 'memory.max').exists() and (p / 'memory.max').read_text().strip() != 'max']
    if not candidates:
        return {}
    base = min(candidates, key=lambda p: int((p / 'memory.max').read_text()))
    limit = int((base / 'memory.max').read_text())
    current = int((base / 'memory.current').read_text())
    stat = dict(line.split() for line in (base / 'memory.stat').read_text().splitlines())
    unreclaimable = int(stat.get('anon', 0)) + int(stat.get('shmem', 0)) + int(stat.get('kernel', 0))
    return {'limit_gib': limit / 2**30, 'current_gib': current / 2**30,
            'anon_gib': int(stat.get('anon', 0)) / 2**30, 'file_cache_gib': int(stat.get('file', 0)) / 2**30,
            'unreclaimable_fraction': unreclaimable / limit}


def allocation_deadline(job):
    detail = subprocess.check_output(['bjobs', '-l', job], text=True)
    if not re.search(r'Sta\s*tus\s*<RUN>', detail):
        raise RuntimeError('The selected allocation is no longer RUN')
    start = re.search(r'([A-Z][a-z]{2} [A-Z][a-z]{2}\s+\d+ \d+:\d+:\d+): Started', detail)
    duration = re.search(r'RUNLIMIT\s+(\d+(?:\.\d+)?) min', detail)
    if not start or not duration:
        raise RuntimeError('Cannot establish allocation expiry; refusing unattended launch')
    year = time.localtime().tm_year
    started = time.mktime(time.strptime(f'{year} {start.group(1)}', '%Y %a %b %d %H:%M:%S'))
    expires = started + float(duration.group(1)) * 60
    stop_at = expires - 25 * 60
    if stop_at <= time.time():
        raise RuntimeError('Allocation has less than 25 minutes left')
    return stop_at, expires


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', choices=AUTHORIZED, required=True)
    p.add_argument('--phase', choices=('acceptance', 'train'), required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tmux', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-runtime-change', action='store_true')
    args = p.parse_args()
    guard_context(args.job)
    args.output = args.output.resolve()
    if not args.output.is_relative_to(OUTPUT_ROOT):
        raise RuntimeError('Outputs must stay under the user-authorized common_ai/WMMPC directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tmux:
        log = args.output / 'node0.log'
        command = [sys.executable, '-u', '-m', 'method.irwm_bev_t.launch', '--job', args.job,
                   '--phase', args.phase, '--config', str(args.config.resolve()), '--output', str(args.output)]
        if args.resume:
            command.append('--resume')
        if args.allow_runtime_change:
            command.append('--allow-runtime-change')
        shell = f'cd {shlex.quote(str(ROOT))} && exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1'
        session = 'irwmt-' + args.phase + '-' + args.output.name
        subprocess.run(['tmux', '-L', 'irwmt' + args.job, 'new-session', '-d', '-s', session, shell], check=True)
        print(json.dumps({'job': args.job, 'tmux': session, 'socket': 'irwmt' + args.job, 'log': str(log)}), flush=True)
        return
    with (args.output / '.node0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop_at, expires = allocation_deadline(args.job)
        config = json.loads(args.config.read_text())
        cores = len(os.sched_getaffinity(0))
        if 2 * (config['workers_per_rank'] + 1) > cores:
            raise RuntimeError('Loader workers exceed measured CPU affinity')
        probe = "import torch,json; assert torch.cuda.device_count()==2; print(json.dumps([{'logical':i,'name':torch.cuda.get_device_name(i),'total_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(2)]))"
        gpu_info = json.loads(subprocess.check_output([sys.executable, '-c', probe], text=True))
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if processes:
            raise RuntimeError('Allocation GPUs have existing compute processes; refusing to overlap')
        memory = memory_snapshot()
        if memory.get('unreclaimable_fraction', 1) > 0.5:
            raise RuntimeError('Insufficient RAM headroom')
        files=sorted(p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json','.md'))
        source_hash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()[:16]
        source = args.output / 'source_revisions' / source_hash
        package = source / 'method' / 'irwm_bev_t'
        if not package.exists():
            package.mkdir(parents=True)
            for path in Path(__file__).parent.iterdir():
                if path.suffix in ('.py', '.json', '.md'):
                    shutil.copy2(path, package / path.name)
            (source / 'method' / '__init__.py').write_text('__path__.append(' + repr(str(ROOT / 'method')) + ')\n')
        launch_name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        frozen_config=args.output/'launches'/launch_name/'config.json'
        frozen_config.parent.mkdir(parents=True,exist_ok=False)
        shutil.copy2(args.config,frozen_config)
        temporary = Path('/fs/scratch/projects/common_ai/nix3sgh_cache/tmp') / ('iwmt' + args.job)
        temporary.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONPATH=str(source) + ':' + str(ROOT), WMMPC_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1', TMPDIR=str(temporary), NCCL_IB_DISABLE='1', NCCL_NVLS_ENABLE='0',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1', IRWM_STOP_AT_UNIX=str(stop_at))
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=2', '--max_restarts=0', '-m', 'method.irwm_bev_t.train',
                   '--config', str(frozen_config), '--output', str(args.output), '--phase', args.phase]
        if args.resume:
            command.append('--resume')
        if args.allow_runtime_change:
            command.append('--allow-runtime-change')
        status = {'state': 'running', 'job': args.job, 'host': socket.gethostname(), 'pid': os.getpid(),
                  'gpus': gpu_info, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'memory': memory,
                  'stop_before_expiry_utc': datetime.fromtimestamp(stop_at, timezone.utc).isoformat(),
                  'allocation_expiry_utc': datetime.fromtimestamp(expires, timezone.utc).isoformat(),
                  'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                  'source_snapshot':str(source),'frozen_config':str(frozen_config)}
        print(json.dumps({'event': 'launch', **status}), flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        status['torchrun_pid'] = child.pid
        write_json(args.output / 'node0.status.json', status)
        rc = child.wait()
        status.update(return_code=rc, state='completed' if rc == 0 else 'failed',
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / 'node0.status.json', status)
        sys.exit(rc)


if __name__ == '__main__':
    main()
