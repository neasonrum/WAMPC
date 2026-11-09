"""Durable four-node launch restricted to the user's eight allocated A100 GPUs."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.launch_full_diffusion import write_status


AUTHORIZED = {"13053006": "rng-dl01-w005", "13047896": "rng-dl01-w002",
              "13047898": "rng-dl01-w007", "13046830": "rng-dl01-w010"}


def guard_context(job_id=None):
    hostname = socket.gethostname().split(".")[0]
    if job_id is None:
        job_id = next((job for job, host in AUTHORIZED.items() if host == hostname), None)
    if job_id not in AUTHORIZED or AUTHORIZED[job_id] != hostname:
        raise RuntimeError("Wrong host; attach to one of the four designated LSF jobs")
    if not re.search(rf"\b{job_id}\b", Path("/proc/self/cgroup").read_text()):
        raise RuntimeError("Wrong LSF cgroup; refusing to touch GPUs")
    return job_id


def inspect_allocation(job_id):
    guard_context(job_id)
    scheduler = "".join(subprocess.check_output(["bjobs", "-l", job_id], text=True).split())
    match = re.search(r"gpus=([0-9,]+);", scheduler)
    if f"Job<{job_id}>" not in scheduler or "Status<RUN>" not in scheduler or match is None:
        raise RuntimeError("Cannot verify the live scheduler GPU allocation")
    physical = match.group(1).split(",")
    if len(physical) != 2 or len(set(physical)) != 2:
        raise RuntimeError("Expected exactly two scheduler-allocated GPUs")
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"
    ], text=True)
    devices = [[field.strip() for field in row] for row in csv.reader(io.StringIO(output))]
    if len(devices) != 2 or {row[0] for row in devices} not in (set(physical), {"0", "1"}):
        raise RuntimeError("Visible GPU namespace differs from the scheduler allocation")
    if any("A100" not in row[2] for row in devices):
        raise RuntimeError("The authorized experiment requires A100 GPUs")
    uuids = ",".join(row[1] for row in devices)
    processes = subprocess.check_output([
        "nvidia-smi", "--id=" + uuids, "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits"
    ], text=True).strip()
    return {"job_id": job_id, "hostname": socket.gethostname(), "scheduler_physical_ids": physical,
            "devices": devices, "cuda_visible_devices": uuids, "compute_processes": processes,
            "available": not processes and all(int(row[3]) < 1000 for row in devices),
            "checked_at": datetime.now(timezone.utc).isoformat()}


def gpu_preflight(job_id, output):
    report = inspect_allocation(job_id)
    if not report["available"]:
        raise RuntimeError("Allocated GPUs are busy; no existing workload will be stopped")
    os.environ["CUDA_VISIBLE_DEVICES"] = report["cuda_visible_devices"]
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("CUDA must expose exactly the two allocated GPUs")
    results = []
    for ordinal in range(2):
        matrix = torch.ones((128, 128), device=f"cuda:{ordinal}")
        if float((matrix @ matrix).mean()) != 128.0:
            raise RuntimeError("Allocated GPU matrix multiplication failed")
        torch.cuda.synchronize(ordinal)
        results.append({"ordinal": ordinal, "name": torch.cuda.get_device_name(ordinal), "matmul_pass": True})
    report.update({"torch": torch.__version__, "cuda_build": torch.version.cuda, "probes": results})
    write_status(output / f"gpu_{job_id}.json", report)
    print(json.dumps(report), flush=True)


def check_master_port(address, port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((address, port))


def runtime_arguments(args):
    arguments = []
    if args.workers_per_rank is not None:
        arguments.extend(("--workers-per-rank", str(args.workers_per_rank)))
    if args.trial_updates is not None:
        arguments.extend(("--trial-updates", str(args.trial_updates)))
    return arguments


def run_controller(args, job_id, node_rank):
    config = json.loads(args.config.read_text())
    with (args.output / f".node{node_rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--phase", "preflight",
                        "--job-id", job_id, "--output", str(args.output)], check=True)
        report = json.loads((args.output / f"gpu_{job_id}.json").read_text())
        import psutil
        address = socket.gethostbyname(socket.gethostname())
        interfaces = [name for name, addresses in psutil.net_if_addrs().items()
                      if any(item.family == socket.AF_INET and item.address == address for item in addresses)]
        if len(interfaces) != 1:
            raise RuntimeError(f"Cannot identify inter-node interface for {address}")
        if node_rank == 0:
            check_master_port(address, config["master_port"])
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": report["cuda_visible_devices"],
                            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
                            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT),
                            "NCCL_SOCKET_IFNAME": interfaces[0], "GLOO_SOCKET_IFNAME": interfaces[0],
                            "NCCL_IB_DISABLE": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_SOCKET_FAMILY": "AF_INET",
                            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"})
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--nnodes=4", "--nproc_per_node=2",
                   f"--node_rank={node_rank}", f"--master_addr={config['master_address']}",
                   f"--master_port={config['master_port']}", "--max_restarts=0",
                   str(Path(__file__).with_name("train.py")), "--config", str(args.config),
                   "--output", str(args.output), "--phase", args.phase]
        if args.resume:
            command.append("--resume")
        command.extend(runtime_arguments(args))
        status_path = args.output / f"node{node_rank}.status.json"
        status = {"controller_pid": os.getpid(), "job_id": job_id, "node_rank": node_rank,
                  "hostname": socket.gethostname(), "phase": args.phase, "command": command,
                  "started_at": datetime.now(timezone.utc).isoformat(), "state": "starting"}
        write_status(status_path, status)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL)
        signal.signal(signal.SIGTERM, lambda *_: process.send_signal(signal.SIGTERM))
        status.update({"worker_parent_pid": process.pid, "state": "running",
                       "cuda_visible_devices": report["cuda_visible_devices"], "network_interface": interfaces[0]})
        write_status(status_path, status)
        return_code = process.wait()
        status.update({"return_code": return_code, "state": "completed" if return_code == 0 else "failed",
                       "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status(status_path, status)
        return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", choices=AUTHORIZED)
    parser.add_argument("--phase", choices=("inspect", "preflight", "acceptance", "train"), required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers-per-rank", type=int)
    parser.add_argument("--trial-updates", type=int)
    args = parser.parse_args()
    if ((args.workers_per_rank is not None and args.workers_per_rank < 0)
            or (args.trial_updates is not None and args.trial_updates <= 0)):
        parser.error("Workers must be nonnegative and trial updates must be positive")
    job_id = guard_context(args.job_id)
    node_rank = list(AUTHORIZED).index(job_id)
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "inspect":
        report = inspect_allocation(job_id)
        write_status(args.output / f"node{node_rank}.inspect.json", report)
        print(json.dumps(report), flush=True)
        return
    if args.phase == "preflight":
        gpu_preflight(job_id, args.output)
        return
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--job-id", job_id,
                   "--phase", args.phase, "--config", str(args.config), "--output", str(args.output)]
        if args.resume:
            command.append("--resume")
        command.extend(runtime_arguments(args))
        log_path = args.output / f"node{node_rank}.controller.log"
        with log_path.open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        result = {"controller_pid": process.pid, "job_id": job_id, "phase": args.phase,
                  "node_rank": node_rank, "hostname": socket.gethostname(), "log": str(log_path)}
        write_status(args.output / f"node{node_rank}.launch.json", result)
        print(json.dumps(result), flush=True)
        return
    sys.exit(run_controller(args, job_id, node_rank))


if __name__ == "__main__":
    main()