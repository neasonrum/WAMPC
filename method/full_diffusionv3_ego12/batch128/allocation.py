"""Batch-128 continuation with an explicit first-epoch sample offset."""

import os
import socket
import time

import torch

from method.full_diffusionv3_ego12.allocation import job_cgroup, read, run_path


class GlobalBatches:
    def __init__(self, length, rank_batches, rank, seed, switch):
        self.length, self.seed, self.switch = length, seed, switch
        self.epoch = switch["epoch"]
        self.total = sum(rank_batches)
        self.begin, self.end = sum(rank_batches[:rank]), sum(rank_batches[:rank + 1])
        if self.total != 128 or self.end <= self.begin or not 0 <= switch["sample_offset"] <= length:
            raise ValueError("Invalid batch-128 continuation")

    @property
    def skip(self):
        return self.switch["sample_offset"] if self.epoch == self.switch["epoch"] else 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (self.length - self.skip) // self.total

    def __iter__(self):
        perm = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        yield from perm[self.skip:self.skip + len(self) * self.total].view(-1, self.total)[:, self.begin:self.end].tolist()


def check_rank(job, output, config, rank, local_rank, world):
    plan = read(run_path(output) / "plan.json")
    entry = plan["jobs"][job]
    job_cgroup(job)
    if (entry["host"] != socket.gethostname().split(".")[0]
            or world != config["world_size"] or len(config["rank_batches"]) != world
            or world != 2 * len(plan["jobs"]) or sum(config["rank_batches"]) != 128 or config["global_batch"] != 128
            or rank != 2 * entry["node_rank"] + local_rank or local_rank not in (0, 1)
            or int(os.environ["LOCAL_WORLD_SIZE"]) != 2
            or 2 * (1 + entry["workers_per_rank"]) > len(os.sched_getaffinity(0))):
        raise RuntimeError("Rank layout or resource binding does not match batch-128 plan")
    if time.time() + 300 >= plan["train_stop_unix"]:
        raise RuntimeError("Insufficient time before the shared checkpoint deadline")
    return entry, plan["train_stop_unix"]
