"""Persistent loading, CPU fault control, and bounded node-local phase traces."""

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import struct
import threading
import time

import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, get_worker_info


PHASES = ("idle", "get_batch", "batch_control", "to_device", "perception", "forward",
          "loss_control", "backward", "gradient_nccl_begin", "gradient_nccl_end", "clip",
          "optimizer", "stop_control", "metrics_control", "checkpoint_sync", "checkpoint_rng",
          "checkpoint_copy", "checkpoint_write", "checkpoint_commit", "validation_get",
          "validation_forward", "validation_sample", "validation_reduce", "completed", "failed")
HEADER = struct.Struct("<QQQQQQQQ")
EVENT = struct.Struct("<QQQId")
EVENT_OFFSET = 1024
EVENT_CAPACITY = 512
TRACE_BYTES = EVENT_OFFSET + EVENT.size * EVENT_CAPACITY


def local_directory(output, attempt):
    identity = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:16]
    directory = Path("/dev/shm") / f"wmmpc_v3_{os.getuid()}_{identity}_{attempt}"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


class PhaseTrace:
    def __init__(self, directory, rank):
        self.path = Path(directory) / f"rank{rank}.trace"
        self.abort_path = Path(directory) / "abort.request"
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(descriptor, TRACE_BYTES)
        self.memory = mmap.mmap(descriptor, TRACE_BYTES)
        os.close(descriptor)
        self.lock = threading.Lock()
        self.version = self.sequence = self.epoch = self.updates = 0
        self.phase = 0
        self.began = time.monotonic_ns()
        self.cuda_events = []
        self.metadata = {"rank": rank, "pid": os.getpid()}
        self._header()

    def _header(self):
        self.version += 2
        now = time.monotonic_ns()
        HEADER.pack_into(self.memory, 0, self.version - 1, os.getpid(), self.epoch, self.updates,
                         self.phase, self.began, now, self.sequence)
        metadata = json.dumps(self.metadata).encode()[:HEADER.size * 8 - 4]
        struct.pack_into("<I", self.memory, HEADER.size, len(metadata))
        self.memory[HEADER.size + 4:HEADER.size + 4 + len(metadata)] = metadata
        struct.pack_into("<Q", self.memory, 0, self.version)

    def progress(self, epoch, updates):
        with self.lock:
            self.epoch, self.updates = int(epoch), int(updates)
            self._header()

    def event(self, phase, duration=0.0):
        with self.lock:
            self.sequence += 1
            offset = EVENT_OFFSET + ((self.sequence - 1) % EVENT_CAPACITY) * EVENT.size
            EVENT.pack_into(self.memory, offset, self.sequence, time.monotonic_ns(), self.updates,
                            PHASES.index(phase), float(duration))
            self._header()

    @contextmanager
    def stage(self, phase, cuda=False):
        started = time.monotonic()
        with self.lock:
            self.phase = PHASES.index(phase)
            self.began = time.monotonic_ns()
            self._header()
        cuda_pair = None
        if cuda and torch.cuda.is_available():
            cuda_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            cuda_pair[0].record()
        try:
            yield
        except BaseException as error:
            with self.lock:
                self.metadata["error"] = repr(error)[:300]
                self.metadata["failed_phase"] = phase
                self.phase = PHASES.index("failed")
                self._header()
            raise
        finally:
            self.event(phase, time.monotonic() - started)
            if cuda_pair is not None:
                cuda_pair[1].record()
                self.cuda_events.append((phase, cuda_pair))
                self.cuda_events = self.cuda_events[-64:]

    def cuda_timings(self):
        complete, pending = {}, []
        for phase, (start, end) in self.cuda_events:
            if end.query():
                complete[phase] = complete.get(phase, 0.0) + start.elapsed_time(end) / 1000.0
            else:
                pending.append((phase, (start, end)))
        self.cuda_events = pending
        return complete

    def finish(self, failed=False):
        with self.lock:
            self.phase = PHASES.index("failed" if failed else "completed")
            self._header()

    def close(self):
        self.memory.close()


def read_trace(path, limit=32):
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            memory = mmap.mmap(descriptor, TRACE_BYTES, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
        with memory:
            for unused in range(3):
                before = HEADER.unpack_from(memory)
                if before[0] % 2:
                    continue
                count = struct.unpack_from("<I", memory, HEADER.size)[0]
                metadata = json.loads(memory[HEADER.size + 4:HEADER.size + 4 + count])
                events = []
                for sequence in range(max(1, before[-1] - min(limit, EVENT_CAPACITY) + 1), before[-1] + 1):
                    entry = EVENT.unpack_from(memory, EVENT_OFFSET + ((sequence - 1) % EVENT_CAPACITY) * EVENT.size)
                    if entry[0] == sequence:
                        events.append({"sequence": entry[0], "monotonic_ns": entry[1], "updates": entry[2],
                                       "phase": PHASES[entry[3]], "host_seconds": entry[4]})
                if HEADER.unpack_from(memory)[0] == before[0]:
                    now = time.monotonic_ns()
                    return {**metadata, "epoch": before[2], "updates": before[3], "phase": PHASES[before[4]],
                            "phase_seconds": (now - before[5]) / 1e9,
                            "heartbeat_age_seconds": (now - before[6]) / 1e9,
                            "event_sequence": before[7], "events": events}
    except (OSError, ValueError):
        return None
    return None


class ControlGroup:
    def __init__(self, group=None, world=1):
        self.group, self.world = group, world
        self.failed = False

    @classmethod
    def create(cls, timeout=120):
        group = distributed.new_group(backend="gloo", timeout=timedelta(seconds=timeout))
        return cls(group, distributed.get_world_size())

    def any(self, value):
        flag = torch.tensor(int(bool(value)), dtype=torch.int64)
        if self.world > 1:
            distributed.all_reduce(flag, op=distributed.ReduceOp.MAX, group=self.group)
        return bool(flag.item())

    def gather(self, value):
        result = [None] * self.world
        if self.world > 1:
            distributed.all_gather_object(result, value, group=self.group)
        else:
            result[0] = value
        return result

    def reduce(self, value):
        result = value.detach().cpu().clone()
        if self.world > 1:
            distributed.all_reduce(result, group=self.group)
        return result

    def barrier(self):
        if self.world > 1:
            distributed.monitored_barrier(group=self.group, timeout=timedelta(seconds=240), wait_all_ranks=True)


class RemainingBatches:
    def __init__(self, batches, offset=0):
        self.batches, self.offset = batches, offset

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.offset, None)

    def __len__(self):
        return max(0, len(self.batches) - self.offset)


def initialize_worker(unused):
    torch.set_num_threads(1)
    dataset = get_worker_info().dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    for name, expected in getattr(dataset, "source_identity", {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Worker source changed after launch: {name}")


def make_training_loader(dataset, batches, workers, seed, rank=0, timeout=180):
    generator = torch.Generator().manual_seed(seed + 1000003 * (rank + 1))
    extra = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    return DataLoader(dataset, batch_sampler=batches, num_workers=workers, pin_memory=True,
                      persistent_workers=bool(workers), timeout=timeout if workers else 0,
                      generator=generator, worker_init_fn=initialize_worker, **extra)


def coordinated_next(iterator, control, trace):
    error, batch = None, None
    with trace.stage("get_batch"):
        try:
            batch = next(iterator)
        except BaseException as exception:
            error = repr(exception)
    with trace.stage("batch_control"):
        if control.any(error is not None):
            errors = control.gather(error)
            raise RuntimeError(f"Coordinated DataLoader failure; no new checkpoint will be saved: {errors}")
    return batch


def gradient_hook(trace, bucket):
    trace.event("gradient_nccl_begin")
    began = time.monotonic()
    future = distributed.all_reduce(bucket.buffer(), async_op=True).get_future()

    def finished(result):
        value = result.value()[0]
        value.div_(distributed.get_world_size())
        trace.event("gradient_nccl_end", time.monotonic() - began)
        return value

    return future.then(finished)