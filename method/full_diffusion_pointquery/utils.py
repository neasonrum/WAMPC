"""Atomic artifacts and validation of the fixed data/allocation identity."""
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch

from method.full_diffusionv3_ego12.allocation import job_cgroup, job_memory

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def move(value, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in value.items()}


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(), "generator": generator.get_state()}


def restore_rng(value, generator):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state(value["cuda"])
    generator.set_state(value["generator"])


def verify_data(config):
    protocol_path = Path(config["data_protocol"])
    protocol = json.loads(protocol_path.read_text())
    directory = Path(config["index_directory"])
    if directory.resolve() != Path(protocol["index_directory"]).resolve():
        raise ValueError("Data directory differs from the frozen protocol")
    for name, expected in protocol["files_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Fixed data identity changed: {name}")
    normalization = Path(protocol["reference_normalization"]["path"])
    if normalization != ROOT / config["normalization_file"] or sha(normalization) != protocol["reference_normalization"]["sha256"]:
        raise ValueError("Normalization differs from the fixed protocol")
    train, val = [np.load(directory / f"{split}.npy", allow_pickle=False) for split in ("train", "val")]
    if len(train) != 237351 or len(val) != 2223 or len(set(val[:, 0])) != 44 or set(train[:, 0]) & set(val[:, 0]):
        raise ValueError("Training and validation route/sample identities differ")
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha(ROOT / config["backbone_directory"] / "model_0030_1.pth") != manifest["teacher_sha256"]:
        raise ValueError("Observed historical cache and frozen perception weights differ")
    return {"protocol": protocol, "manifest": manifest, "normalization": json.loads(normalization.read_text()),
            "train_samples": len(train), "validation_samples": len(val), "train_routes": len(set(train[:, 0])),
            "validation_routes": len(set(val[:, 0])), "protocol_sha256": sha(protocol_path)}


def inspect_allocation(config, job, idle=False):
    import subprocess
    entry = config["jobs"][job]
    job_cgroup(job)
    affinity = sorted(os.sched_getaffinity(0))
    if socket.gethostname().split(".")[0] != entry["host"] or 2 * (entry["workers_per_rank"] + 1) > len(affinity):
        raise RuntimeError("Allocation host/CPU affinity cannot support the requested workers")
    memory = job_memory(job)
    if memory["limit_bytes"] != 268435456000 or memory["noncache_bytes"] > .7 * memory["limit_bytes"]:
        raise RuntimeError(f"Insufficient allocation RAM headroom: {memory}")
    if memory["oom"] or memory["oom_kill"]:
        raise RuntimeError("Allocation OOM events require inspection")
    if time.time() > config["train_stop_unix"] - 900:
        raise RuntimeError("Insufficient allocation time")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=20).strip().splitlines()
    values = [[x.strip() for x in row.split(",")] for row in rows]
    if len(values) != 2 or [v[0] for v in values] != ["0", "1"] or [v[1] for v in values] != entry["gpu_uuids"]:
        raise RuntimeError(f"Allocation logical GPU identity differs: {values}")
    if any("A100" not in v[2] or int(v[3]) < 79000 for v in values):
        raise RuntimeError("Expected two 80GB A100 GPUs")
    if idle and any(int(v[4]) > 512 for v in values):
        raise RuntimeError("Allocation GPUs already have a workload")
    return {"job": job, "host": entry["host"], "affinity": affinity, "gpus": values,
            "memory": memory, "cgroup": str(job_cgroup(job)), "unix": time.time()}

