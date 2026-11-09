"""CPU topology, cgroup RAM and owned-process telemetry for CARLA evaluation."""
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def evaluation_results_root():
    return Path(os.environ.get('WMMPC_CARLA_RESULTS_ROOT', ROOT / 'results')).resolve()


def checked_run_root(path):
    resolved = Path(path).resolve()
    resolved.relative_to(evaluation_results_root())
    return resolved


def checked_evaluation_path(path):
    resolved = Path(path).resolve()
    for root in (ROOT.resolve(), evaluation_results_root()):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f'Evaluation path is outside the project and configured results root: {resolved}')


def evaluation_runtime_root():
    root = evaluation_results_root() / '.runtime' if 'WMMPC_CARLA_RESULTS_ROOT' in os.environ else ROOT / '.runtime'
    return checked_evaluation_path(root)


def cpu_groups(allowed, topology=None):
    groups = {}
    for cpu in sorted(allowed):
        if topology is None:
            base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
            key = (int((base/'physical_package_id').read_text()),
                   int((base/'core_id').read_text()))
        else:
            key = topology[cpu]
        groups.setdefault(key, []).append(cpu)
    return list(groups.values())


def select_gpu_cpus(allowed, gpu, count, topology=None):
    if gpu not in (0, 1):
        raise ValueError('Expected an allocation-visible GPU ordinal')
    groups = cpu_groups(allowed, topology)
    half = len(groups) // 2
    cores = groups[gpu*half:(gpu+1)*half]
    # Give each GPU disjoint physical cores, then add available SMT siblings.
    ordered = [core[level] for level in range(max(map(len, cores), default=0))
               for core in cores if len(core) > level]
    if count < 2 or count > len(ordered):
        raise ValueError(f'Requested {count} CPUs, but GPU {gpu} has {len(ordered)} disjoint logical CPUs')
    return sorted(ordered[:count])


def memory_snapshot():
    group = Path('/proc/self/cgroup').read_text().strip().split(':')[-1].lstrip('/')
    leaf = Path('/sys/fs/cgroup') / group
    ancestors = []
    current = leaf
    while current != Path('/sys/fs/cgroup'):
        limit = current/'memory.max'
        if limit.is_file() and limit.read_text().strip() != 'max':
            ancestors.append((int(limit.read_text()), current))
        current = current.parent
    if not ancestors:
        raise RuntimeError('No bounded cgroup RAM limit found')
    limit, job = min(ancestors)
    stats = {k: int(v) for k, v in (line.split() for line in (job/'memory.stat').read_text().splitlines())}
    used = int((job/'memory.current').read_text())
    return {'job_memory.max': str(limit), 'job_memory.current': str(used),
            'job_memory.working_set': str(max(0, used-stats.get('inactive_file', 0))),
            'job_memory.noncache': str(sum(stats.get(k, 0) for k in ('anon','kernel','shmem','sock','file_dirty','file_writeback'))),
            'job_memory.anon': stats.get('anon', 0), 'job_memory.file': stats.get('file', 0),
            'job_memory.inactive_file': stats.get('inactive_file', 0),
            'job_memory.events': (job/'memory.events').read_text().strip(),
            'cgroup': str(job)}


def process_tree(pid=None):
    todo = [pid or os.getpid()]
    found = []
    while todo:
        current = todo.pop()
        try:
            base = Path(f'/proc/{current}')
            raw = (base/'stat').read_text()
            fields = raw[raw.rfind(')')+2:].split()
            status = dict(line.split(':', 1) for line in (base/'status').read_text().splitlines() if ':' in line)
            io = dict(line.split(':', 1) for line in (base/'io').read_text().splitlines() if ':' in line)
            found.append({'pid': current, 'name': status['Name'].strip(),
                          'cpu_seconds': (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                          'rss_kib': int(status.get('VmRSS', '0 kB').split()[0]),
                          'threads': int(status['Threads']),
                          'read_bytes': int(io.get('read_bytes', 0)),
                          'write_bytes': int(io.get('write_bytes', 0))})
            todo.extend(map(int, (base/f'task/{current}/children').read_text().split()))
        except (OSError, ProcessLookupError):
            continue
    return found


def resources():
    result = {'unix': time.time(), **memory_snapshot(), 'processes': process_tree(),
              'cpu_affinity': sorted(os.sched_getaffinity(0))}
    result['gpu'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15).stdout
    return result
