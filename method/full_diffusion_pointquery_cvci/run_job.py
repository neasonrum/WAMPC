"""Launch only inside a supplied, verified allocation; never submit/cancel LSF jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .utils import ROOT, atomic_json, inspect_allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = Path(config["output"])
    entry = config["jobs"][args.job]
    lock = (run / f".job_{args.job}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = inspect_allocation(config, args.job, idle=True)
    preflight = json.loads((run / "preflight.json").read_text())
    if not preflight["passed"] or preflight["inference_steps"] != 25:
        raise RuntimeError("Real-data preflight has not passed")
    if not preflight['online_acceptance_passed'] or preflight['planner_tensors_receiving_world_loss'] != 0:
        raise RuntimeError("The actual online wrapper acceptance check has not passed")
    if (run / "completed.json").exists():
        raise RuntimeError("Training has already completed")
    if (run / "latest.pt").exists() != args.resume:
        raise RuntimeError("Explicit resume must agree with existing optimizer state")
    atomic_json(run / f"allocation_launch_{args.job}.json", proof)
    env = {**os.environ, "NCCL_SOCKET_IFNAME": "=bond0", "GLOO_SOCKET_IFNAME": "bond0",
           "NCCL_SOCKET_FAMILY": "AF_INET", "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1",
           "NCCL_CUMEM_HOST_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
           "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "CUDA_MODULE_LOADING": "LAZY"}
    cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--nnodes", "4", "--nproc-per-node", "2",
           "--node-rank", str(entry["node_rank"]), "--master-addr", config["master_address"],
           "--master-port", str(config["master_port"]), "--max-restarts", "0",
           "-m", "method.full_diffusion_pointquery_cvci.train", "--config", str(args.config), "--job-id", args.job]
    if args.resume:
        cmd.append("--resume")
    stamp = time.time_ns()
    log = run / f"train_{args.job}_{stamp}.log"
    atomic_json(run / f"node_{args.job}.json", {"state": "launching", "job": args.job,
                "log": str(log), "command": cmd, "unix": time.time()})
    with log.open("xb") as handle:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    state = "completed" if result == 0 and (run / "completed.json").exists() else "paused" if result == 0 else "failed"
    atomic_json(run / f"node_{args.job}.json", {"state": state, "return_code": result, "log": str(log), "unix": time.time()})
    if result:
        raise RuntimeError(f"Training worker group exited with code {result}; preserve allocation and logs")
    if state == "completed" and config["validation"].get("automatic_closed_loop", False):
        from .evaluate import evaluate
        atomic_json(run / f"node_{args.job}.json", {"state": "closed_loop", "unix": time.time()})
        evaluate(config, run, args.job)


if __name__ == "__main__":
    main()

