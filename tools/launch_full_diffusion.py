"""Job-guarded durable launchers; the controller never initializes CUDA."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED = {"13047898": "rng-dl01-w007", "13047896": "rng-dl01-w002"}


def guard_context(job_id):
    if socket.gethostname().split(".")[0] != AUTHORIZED[job_id]:
        raise RuntimeError("Wrong host; use battach on the designated job")
    cgroup = Path("/proc/self/cgroup").read_text()
    if os.environ.get("LSB_JOBID") != job_id and not re.search(rf"\b{job_id}\b", cgroup):
        raise RuntimeError("Wrong LSF cgroup; refusing to run")


def write_status(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", choices=AUTHORIZED, required=True)
    parser.add_argument("--phase", choices=("index", "acceptance", "train"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "method/full_diffusion/configs/geo.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--master-port", type=int)
    args = parser.parse_args()
    guard_context(args.job_id)
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text())
    node_rank = list(AUTHORIZED).index(args.job_id)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "index" and node_rank != 0:
        raise RuntimeError("Index preparation runs once on node 0, without CUDA")
    if args.detach:
        log_path = args.output / f"node{node_rank}.controller.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--job-id", args.job_id, "--phase", args.phase,
                   "--config", str(args.config), "--output", str(args.output)]
        if args.resume:
            command.append("--resume")
        if args.master_port:
            command.extend(("--master-port", str(args.master_port)))
        with log_path.open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        result = {"controller_pid": process.pid, "job_id": args.job_id, "phase": args.phase,
                  "hostname": socket.gethostname(), "log": str(log_path)}
        write_status(args.output / f"node{node_rank}.launch.json", result)
        print(json.dumps(result), flush=True)
        return
    with (args.output / f".node{node_rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)})
        if args.phase == "index":
            command = [sys.executable, "-u", "-m", "method.full_diffusion.data", "--output",
                       str(ROOT / config["index_directory"]), "--workers", "8"]
        else:
            subprocess.run([sys.executable, str(ROOT / "tools/preflight_full_diffusion_gpu.py"),
                            "--job-id", args.job_id], check=True, env=environment)
            report = json.loads((ROOT / f"results/full_diffusion_gpu_{args.job_id}.json").read_text())
            environment["CUDA_VISIBLE_DEVICES"] = report["cuda_visible_devices"]
            import psutil
            node_address = socket.gethostbyname(socket.gethostname())
            interfaces = [name for name, addresses in psutil.net_if_addrs().items()
                          if any(address.family == socket.AF_INET and address.address == node_address
                                 for address in addresses)]
            if len(interfaces) != 1:
                raise RuntimeError(f"Cannot identify inter-node interface for {node_address}")
            environment.update({"NCCL_SOCKET_IFNAME": interfaces[0], "GLOO_SOCKET_IFNAME": interfaces[0],
                                "NCCL_IB_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"})
            command = [sys.executable, "-u", "-m", "torch.distributed.run", "--nnodes=2",
                       "--nproc_per_node=2", f"--node_rank={node_rank}",
                       f"--master_addr={config['master_address']}",
                       f"--master_port={args.master_port or config['master_port']}",
                       "--max_restarts=0", str(ROOT / "train/train_full_diffusion.py"),
                       "--config", str(args.config), "--output", str(args.output), "--phase", args.phase]
            if args.resume:
                command.append("--resume")
        status_path = args.output / f"node{node_rank}.status.json"
        status = {"controller_pid": os.getpid(), "job_id": args.job_id, "node_rank": node_rank,
                  "hostname": socket.gethostname(), "phase": args.phase, "command": command,
                  "started_at": datetime.now(timezone.utc).isoformat(), "state": "starting"}
        write_status(status_path, status)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL)
        signal.signal(signal.SIGTERM, lambda *_: process.send_signal(signal.SIGTERM))
        status.update({"worker_parent_pid": process.pid, "state": "running",
                       "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
                       "network_interface": environment.get("NCCL_SOCKET_IFNAME")})
        write_status(status_path, status)
        return_code = process.wait()
        status.update({"return_code": return_code, "state": "completed" if return_code == 0 else "failed",
                       "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status(status_path, status)
        sys.exit(return_code)


if __name__ == "__main__":
    main()