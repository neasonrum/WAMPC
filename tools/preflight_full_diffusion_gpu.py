"""Finite GPU smoke test restricted to the two user-authorized LSF jobs."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED = {"13047898": "rng-dl01-w007", "13047896": "rng-dl01-w002"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, choices=AUTHORIZED)
    args = parser.parse_args()
    hostname = socket.gethostname().split(".")[0]
    if hostname != AUTHORIZED[args.job_id]:
        raise RuntimeError("Wrong host: attach to the designated LSF job before GPU work")
    cgroup = Path("/proc/self/cgroup").read_text()
    if os.environ.get("LSB_JOBID") != args.job_id and not re.search(rf"\b{args.job_id}\b", cgroup):
        raise RuntimeError("Wrong LSF job context; refusing GPU work")
    job = "".join(subprocess.check_output(["bjobs", "-l", args.job_id], text=True).split())
    allocation = re.search(r"gpus=([0-9,]+);", job)
    if f"Job<{args.job_id}>" not in job or "Status<RUN>" not in job or allocation is None:
        raise RuntimeError("Cannot verify the live scheduler GPU allocation")
    physical_ids = allocation.group(1).split(",")
    if len(physical_ids) != 2 or len(set(physical_ids)) != 2:
        raise RuntimeError("Expected exactly two scheduler-allocated GPUs")
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits",
    ], text=True)
    devices = [[field.strip() for field in row] for row in csv.reader(io.StringIO(output))]
    visible_ids = {row[0] for row in devices}
    if len(devices) != 2 or visible_ids not in (set(physical_ids), {"0", "1"}):
        raise RuntimeError("GPU inventory does not match the scheduler allocation")
    if any("A100" not in row[2] or int(row[3]) >= 1000 for row in devices):
        raise RuntimeError("Allocated A100 GPUs are unavailable or busy; no workload touched")
    allocated_uuids = ",".join(row[1] for row in devices)
    processes = subprocess.check_output([
        "nvidia-smi", "--id=" + allocated_uuids,
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    if processes:
        raise RuntimeError("Allocated GPUs already have compute processes; no workload touched")
    os.environ["CUDA_VISIBLE_DEVICES"] = allocated_uuids
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("CUDA does not expose exactly the two allocated GPUs")
    results = []
    for ordinal in range(2):
        matrix = torch.ones((128, 128), device=f"cuda:{ordinal}")
        product = matrix @ matrix
        torch.cuda.synchronize(ordinal)
        if product.mean().item() != 128.0:
            raise RuntimeError("GPU matrix multiplication failed")
        props = torch.cuda.get_device_properties(ordinal)
        results.append({"ordinal": ordinal, "name": props.name,
                        "memory_gib": props.total_memory / 1024**3, "matmul_pass": True})
    report = {
        "job_id": args.job_id, "hostname": hostname, "scheduler_physical_ids": physical_ids,
        "nvidia_visible_ids": [row[0] for row in devices],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "devices": results, "training_started": False,
    }
    destination = ROOT / "results" / f"full_diffusion_gpu_{args.job_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()