"""Measure worker heap retention while reading real route-stream samples."""
import argparse
import ctypes
import gc
import json
from pathlib import Path
import resource
import time
import tracemalloc

from .data import JointDataset
from .prepare import atomic_json
from .streams import DistributedRouteStreams


def rss_bytes():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return int(fields['VmRSS'].split()[0]) * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dataset = JointDataset(config['output'], config)
    sampler = iter(DistributedRouteStreams(dataset.chains(), 32, 0, 8, config['seed']))
    tracemalloc.start(5)
    started = time.monotonic()
    records = []
    for step in range((args.samples + 31) // 32):
        for index in next(sampler):
            item = dataset[index]
        del item
        current, peak = tracemalloc.get_traced_memory()
        record = {'samples': (step + 1) * 32, 'rss_bytes': rss_bytes(), 'python_traced_bytes': current,
                  'python_peak_bytes': peak, 'seconds': time.monotonic() - started}
        records.append(record)
        print(json.dumps(record), flush=True)
    before = rss_bytes()
    gc.collect()
    after_gc = rss_bytes()
    ctypes.CDLL(None).malloc_trim(0)
    after_trim = rss_bytes()
    snapshot = tracemalloc.take_snapshot()
    report = {'records': records, 'before_gc_bytes': before, 'after_gc_bytes': after_gc,
              'after_trim_bytes': after_trim, 'python_top': [str(value) for value in snapshot.statistics('lineno')[:12]],
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_json(Path(config['output']) / 'worker_memory_probe.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
