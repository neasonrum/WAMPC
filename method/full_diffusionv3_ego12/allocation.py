"""Exact-batch sampler and checks for the five supplied two-card LSF jobs."""

import csv
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import torch

RESULTS = Path("/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/results")
RUN_PREFIX = "full_diffusionv3_ego12_10a100_"


class GlobalBatches:
    """The same epoch permutations and groups of 64 as the original V3."""
    def __init__(self, length, rank_batches, rank, seed):
        self.size, self.rank, self.seed = length, rank, seed
        self.total = sum(rank_batches)
        self.begin, self.end = sum(rank_batches[:rank]), sum(rank_batches[:rank + 1])
        self.epoch = 0
        if self.total != 64 or self.end <= self.begin:
            raise ValueError("Expected a positive share of the V3 global batch of 64")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size // self.total

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        yield from order[:len(self) * self.total].view(-1, self.total)[:, self.begin:self.end].tolist()


def read(path):
    return json.loads(Path(path).read_text())


def run_path(output):
    original = Path(output)
    resolved = original.resolve(strict=True)
    if original != resolved or resolved.parent != RESULTS or not resolved.name.startswith(RUN_PREFIX):
        raise ValueError(f"Invalid run path: {original}")
    return resolved


def job_cgroup(job):
    raw = Path("/proc/self/cgroup").read_text()
    leaf = Path("/sys/fs/cgroup") / next(line.split(":", 2)[2].lstrip("/") for line in raw.splitlines()
                                        if line.startswith("0::"))
    matches = [p for p in (leaf, *leaf.parents) if p.name.startswith(f"job.{job}.")]
    if len(matches) != 1 or not (matches[0] / "memory.max").is_file():
        raise RuntimeError(f"Not inside LSF allocation {job}")
    return matches[0]


def job_memory(job):
    p = job_cgroup(job)
    stats = dict(line.split() for line in (p / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (p / "memory.events").read_text().splitlines())
    return {"cgroup": str(p), "limit_bytes": int((p / "memory.max").read_text()),
            "current_bytes": int((p / "memory.current").read_text()),
            "noncache_bytes": sum(int(stats.get(k, 0)) for k in
                                  ("anon", "shmem", "kernel", "sock", "file_dirty", "file_writeback")),
            "oom": int(events["oom"]), "oom_kill": int(events["oom_kill"]),
            "unix": time.time()}


def query_allocation(job, expected_host):
    raw = subprocess.check_output(["bjobs", "-l", job], text=True, timeout=15)
    joined = "".join(raw.split())
    start = re.search(r"(?m)^(\w{3} \w{3}\s+\d+ [\d:]+)(?: (\d{4}))?: Started 16 Task\(s\) on Host\(s\) <16\*([^>]+)>", raw)
    limit = re.search(r"RUNLIMIT\s*\n\s*([\d.]+)", raw)
    if not (f"Job<{job}>" in joined and "Status<RUN>" in joined
            and "RequestedGPU<num=2:" in joined and "rusage[mem=16000" in joined
            and start and limit and start[3] == expected_host):
        raise RuntimeError(f"Scheduler does not confirm the expected live 2xA100 allocation {job}")
    from datetime import datetime
    started = datetime.strptime(f"{start[1]} {start[2] or datetime.now().year}", "%a %b %d %H:%M:%S %Y").timestamp()
    return {"job_id": job, "state": "RUN", "host": expected_host, "slots": 16, "mem_per_slot_mb": 16000,
            "deadline_unix": started + float(limit[1]) * 60, "checked_unix": time.time()}


def inspect(job, entry, idle=True):
    notes = query_allocation(job, entry["host"])
    if (socket.gethostname().split(".")[0] != entry["host"]
            or abs(notes["deadline_unix"] - entry["deadline_unix"]) > 1):
        raise RuntimeError(f"LSF allocation identity changed: {job}")
    mem = job_memory(job)
    if mem["limit_bytes"] < 16000 * 16 * 2**20 or mem["noncache_bytes"] + 64 * 2**30 > .80 * mem["limit_bytes"]:
        raise RuntimeError(f"Insufficient verified host-memory headroom: {mem}")
    if mem["oom"] or mem["oom_kill"]:
        raise RuntimeError(f"Existing job OOM events require inspection: {mem}")
    affinity = sorted(os.sched_getaffinity(0))
    if 2 * (1 + entry["workers_per_rank"]) > len(affinity):
        raise RuntimeError(f"The CPU affinity no longer supports the requested workers: {affinity}")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,uuid",
                                   "--format=csv,noheader,nounits"], text=True, timeout=12)
    rows = [[s.strip() for s in row] for row in csv.reader(io.StringIO(raw))]
    if (len(rows) != 2 or [r[0] for r in rows] != ["0","1"] or
            [r[-1] for r in rows] != entry["gpu_uuids"] or
            any("A100" not in r[1] or int(r[2]) < 79000 for r in rows)):
        raise RuntimeError(f"Allocation does not expose the two expected logical A100s: {rows}")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                    "--format=csv,noheader"], text=True, timeout=12).strip()
    if idle and (apps or any(int(r[3]) >= 1024 for r in rows)):
        raise RuntimeError("The requested GPUs are in use; workloads will be left running")
    return {**notes, "gpu_uuids": [r[-1] for r in rows], "affinity": affinity,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "memory": mem}


def check_rank(job, output, config, rank, local_rank, world):
    plan = read(run_path(output) / "plan.json")
    entry = plan["jobs"][job]
    job_cgroup(job)
    if (entry["host"] != socket.gethostname().split(".")[0]
            or world != config["world_size"] or len(config["rank_batches"]) != world
            or world != 2 * len(plan["jobs"]) or (sum(config["rank_batches"]) != 64 or config["global_batch"] != 64)
            or rank != 2 * entry["node_rank"] + local_rank or local_rank not in (0, 1)
            or int(os.environ["LOCAL_WORLD_SIZE"]) != 2
            or 2 * (1 + entry["workers_per_rank"]) > len(os.sched_getaffinity(0))):
        raise RuntimeError("Rank layout or resource binding does not match the run plan")
    if time.time() + 300 >= plan["train_stop_unix"]:
        raise RuntimeError("Insufficient time to start training before the shared checkpoint deadline")
    return entry, plan["train_stop_unix"]
