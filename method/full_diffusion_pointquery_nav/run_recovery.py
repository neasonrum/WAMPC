"""Run verified six-GPU recovery inside the three existing allocations."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = Path(config["output"])
    lock = (run / f".job_{args.job}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = inspect_allocation(config, args.job, idle=True)
    if not json.loads((run / "recovery_preflight.json").read_text())["passed"]:
        raise RuntimeError("Recovery preflight has not passed")
    if (run / "completed.json").exists() or (run / "latest.pt").exists() != args.resume:
        raise RuntimeError("Recovery/resume state does not agree with the launch request")
    entry = config["jobs"][args.job]
    atomic_json(run / f"allocation_launch_{args.job}.json", proof)
    env = {**os.environ, "NCCL_SOCKET_IFNAME": "=bond0", "GLOO_SOCKET_IFNAME": "bond0",
           "NCCL_SOCKET_FAMILY": "AF_INET", "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1",
           "NCCL_CUMEM_HOST_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
           "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "CUDA_MODULE_LOADING": "LAZY"}
    launch = [sys.executable, "-u", "-m", "torch.distributed.run", "--nnodes", "3", "--nproc-per-node", "2",
              "--node-rank", str(entry["node_rank"]), "--master-addr", config["master_address"],
              "--master-port", str(config["master_port"]), "--max-restarts", "0", "-m"]
    test_launch = list(launch)
    test_launch[test_launch.index("--master-port") + 1] = str(config["master_port"] + 1)
    test = test_launch + ["method.full_diffusion_pointquery_nav.recovery_ddp_check", "--config", str(args.config)]
    with (run / f"ddp_check_{args.job}.log").open("ab") as handle:
        subprocess.run(test, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    deadline = time.time() + 120
    while not (run / "ddp_recovery_check.json").is_file():
        if time.time() > deadline:
            raise TimeoutError("Six-rank gradient/RNG validation did not complete")
        time.sleep(1)
    if not json.loads((run / "ddp_recovery_check.json").read_text())["passed"]:
        raise RuntimeError("Distributed recovery checks failed")
    command = launch + ["method.full_diffusion_pointquery_nav.recovery", "--config", str(args.config), "--job-id", args.job]
    if args.resume:
        command.append("--resume")
    log = run / f"train_{args.job}_{time.time_ns()}.log"
    atomic_json(run / f"node_{args.job}.json", {"state": "launching", "job": args.job,
                "log": str(log), "command": command, "unix": time.time()})
    with log.open("xb") as handle:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    state = "completed" if code == 0 and (run / "completed.json").exists() else "paused" if code == 0 else "failed"
    atomic_json(run / f"node_{args.job}.json", {"state": state, "return_code": code, "log": str(log), "unix": time.time()})
    if code:
        raise RuntimeError(f"Recovery workers exited {code}; preserve allocation and logs")
    if state == "completed" and config["validation"].get("automatic_closed_loop", False):
        from .evaluate import evaluate
        atomic_json(run / f"node_{args.job}.json", {"state": "closed_loop", "unix": time.time()})
        evaluate(config, run, args.job)


if __name__ == "__main__":
    main()
