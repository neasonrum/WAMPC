"""Run one authorized LSF group, then hand off its two GPUs to CARLA if scheduled."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from method.full_diffusion2.snapshot import atomic_json, sha_file
from method.full_diffusionv3.checkpoint import select_complete
from method.full_diffusionv3_ego12.allocation import RESULTS, inspect, read, run_path

ROOT = Path(__file__).resolve().parents[2]
METHOD = "full_diffusionv3_ego12"


def final_checkpoint(run, plan):
    marker = run / "completed.json"
    if not marker.is_file():
        raise RuntimeError("No normal training completion marker; do not start CARLA")
    result = read(marker)
    if (result["epoch"], result["updates"], result["next_batch"], result["validation_pending"]) != (
            plan["epochs"], plan["expected_updates"], 0, False):
        raise RuntimeError("Training did not finish the requested epoch and its validation")
    checkpoint = run / f"epoch_{plan['epochs']:02d}.pt"
    final = select_complete(run, full_hash=False)
    if (final["progress"]["epoch"] != plan["epochs"] or final["progress"].get("validation_pending", True)
            or final["progress"]["updates"] != plan["expected_updates"]
            or not checkpoint.is_file() or sha_file(checkpoint) != final["sha256"]):
        raise RuntimeError("The completed epoch checkpoint is missing or does not match its committed checksum")
    return checkpoint


def wait_carla(run, job, procs, plan, deadline):
    if plan["jobs"][job]["node_rank"] != 0:
        return [p.wait() for p in procs]
    try:
        from evaluation.summarize_parallel_carla import summarize
    except (ImportError, RuntimeError) as error:
        print(f"Summary monitor import failed: {error!r}", flush=True)
        return [p.wait() for p in procs]
    ids = [key for key, settings in plan["jobs"].items() if settings["eval"]]
    last = None
    while True:
        q = run / "route_queue.json"
        exits = {j: read(run / f"job_{j}.json") for j in ids if (run / f"job_{j}.json").is_file()}
        local_finished = all(p.poll() is not None for p in procs)
        terminal = local_finished and (all(j in exits for j in ids if j != job) or time.time() >= deadline)
        fingerprint = (q.stat().st_mtime_ns if q.exists() else 0, len(exits), local_finished)
        if fingerprint != last or terminal:
            try:
                metrics = summarize(run)
                if metrics["complete_220_routes"] or terminal:
                    metrics = summarize(run, extended=True, merge=True)
                atomic_json(run / "summary.json", metrics)
                if metrics["complete_220_routes"] or terminal:
                    return [p.wait() for p in procs]
            except Exception as error:
                print(f"Summary update failed: {error!r}", flush=True)
                if terminal:
                    return [p.wait() for p in procs]
            last = fingerprint
        time.sleep(20)


def evaluate(run, job, plan, env):
    directory = run / "carla25"
    entry = plan["jobs"][job]
    final = final_checkpoint(run, plan)
    proof_path = directory / "allocations" / f"allocation_{job}.json"
    proof = inspect(job, entry)
    wall = min(9 * 3600, int(proof["deadline_unix"] - time.time() - 1800))
    if wall < 30 * 60:
        raise RuntimeError("Less than 30 minutes remains for CARLA")
    atomic_json(proof_path, proof)
    eval_env = {**env, "WMMPC_CARLA_RESULTS_ROOT": str(RESULTS), "SLOWER": "0"}
    # summarize() reads the same results namespace as the child evaluators.
    os.environ["WMMPC_CARLA_RESULTS_ROOT"] = str(RESULTS)
    started = time.time()
    procs, logs = [], []
    try:
        for gpu in (0, 1):
            index = entry["node_rank"] * 2 + gpu
            group = f"{job}_gpu{gpu}"
            handle = (directory / f"{group}.controller.log").open("xb")
            logs.append(handle)
            cmd = [sys.executable, "-u", str(ROOT / "evaluation/run_full_diffusion2_carla.py"),
                   "--method", METHOD, "--job-id", job, "--run-root", str(directory),
                   "--checkpoint", str(final), "--gpu", str(gpu), "--group-id", group,
                   "--route-ids", "all", "--slot-base", str(3620 + index * 90),
                   "--tm-base", str(23500 + index * 550), "--seed", "901",
                   "--backend", "direct", "--lanes", "1",
                   "--cpus-per-gpu", str(entry["eval_cpus_per_gpu"]), "--server-mode", "warm",
                   "--startup-wait", "rpc", "--routes-per-server", "32",
                   "--wall-budget", str(wall), "--allocation-record", str(proof_path)]
            procs.append(subprocess.Popen(cmd, cwd=ROOT, env=eval_env, stdout=handle, stderr=subprocess.STDOUT))
        atomic_json(directory / f"launch_{job}.json", {"job": job, "pids": [p.pid for p in procs],
                    "checkpoint": str(final), "inference_steps": 25, "unix": started})
        results = wait_carla(directory, job, procs, plan, started + wall + 180)
    finally:
        # Scope shutdown to children owned by this local driver.
        for process in procs:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=90)
        for log in logs:
            log.close()
    atomic_json(directory / f"job_{job}.json", {"job": job, "return_codes": results, "unix": time.time()})
    if any(results):
        raise RuntimeError(f"CARLA controllers ended with {results}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run, job = run_path(args.output), args.job
    plan, config = read(run / "plan.json"), read(run / "config.json")
    entry = plan["jobs"][job]
    attempt = 1 if args.resume else 0
    lock = (run / f".job_{job}.lock").open("a")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = inspect(job, entry)
        assert time.time() + 900 < plan["train_stop_unix"]
        preflight = read(run / "preflight.json")
        assert preflight["inference_steps"] == 25 and preflight["spatial_memory_invariance"]
        assert (run / "latest.pt").exists() == args.resume
        if (run / "completed.json").exists():
            raise RuntimeError("Training already completed")
        atomic_json(run / f"allocation_{job}.json", proof)
        env = {**os.environ, "NCCL_SOCKET_IFNAME": "=bond0", "GLOO_SOCKET_IFNAME": "bond0",
               "NCCL_SOCKET_FAMILY": "AF_INET", "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1",
               "NCCL_CUMEM_HOST_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "CUDA_MODULE_LOADING": "LAZY"}
        cmd = [sys.executable, "-u", "-m", "torch.distributed.run",
               "--nnodes", str(len(plan["jobs"])), "--nproc-per-node", "2",
               "--node-rank", str(entry["node_rank"]), "--master-addr", plan["master_address"],
               "--master-port", str(plan["master_port"]), "--max-restarts", "0",
               "-m", "method.full_diffusionv3_ego12.train", "--gpu-authorized", "--job-id", job,
               "--config", str(run / "config.json"), "--output", str(run), "--attempt", str(attempt)]
        if args.resume:
            cmd.append("--resume")
        atomic_json(run / f"node_{job}.json", {"state": "training", "job": job, "node_rank": entry["node_rank"],
                    "local_batch_sizes": config["rank_batches"][entry["node_rank"] * 2: entry["node_rank"] * 2 + 2],
                    "torchrun_command": cmd, "unix": time.time()})
        with (run / f"train_{job}_attempt{attempt}.log").open("xb") as log:
            status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        state = "complete" if status == 0 and (run / "completed.json").is_file() else "paused" if status == 0 else "failed"
        atomic_json(run / f"node_{job}.json", {"state": state, "job": job, "node_rank": entry["node_rank"],
                    "torchrun_return_code": status, "unix": time.time()})
        if state == "complete" and entry["eval"]:
            try:
                evaluate(run, job, plan, env)
            except Exception as error:
                atomic_json(run / "carla25" / f"job_{job}.json",
                            {"job": job, "error": repr(error), "unix": time.time()})
                raise
        elif state == "failed":
            raise RuntimeError(f"torchrun exited with {status}; see the per-job training log")


if __name__ == "__main__":
    main()
