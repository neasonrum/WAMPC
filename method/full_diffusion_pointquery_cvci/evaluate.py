"""After normal completion, evaluate the fixed-validation-selected checkpoint."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from method.full_diffusionv3_ego12.allocation import query_allocation
from .utils import ROOT, atomic_json, inspect_allocation, sha


def prepare(config, output):
    completed = json.loads((output / "completed.json").read_text())
    best = json.loads((output / "best.json").read_text())
    progress = completed['progress']
    if progress["epoch"] != 10 or progress["validation_pending"] or progress["best_epoch"] != best["epoch"]:
        raise RuntimeError("Training did not normally complete all ten epochs and validation")
    if sha(output / "best.pt") != best["checkpoint_sha256"]:
        raise RuntimeError("Selected best checkpoint checksum differs")
    directory = output / "carla25"
    directory.mkdir(exist_ok=True)
    (directory / "allocations").mkdir(exist_ok=True)
    if (directory / "ready.json").exists():
        raise RuntimeError("Existing closed-loop run must be inspected before restart")
    private = directory / "selected_best.pt"
    shutil.copy2(output / "best.pt", private)
    hashes = json.loads((output / "source_sha256.json").read_text())
    changed = [name for name, expected in hashes.items() if sha(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"Code changed since this experiment was launched: {changed}")
    atomic_json(directory / "source_sha256.json", hashes)
    atomic_json(directory / "ready.json", {"checkpoint": str(private), "checkpoint_sha256": sha(private),
                "selected_epoch": best["epoch"], "selection_metric": best["selection_metric"],
                "validation_score": best["score"], "routes": 220, "inference_steps": 25, "unix": time.time()})


def evaluate(config, output, job):
    entry = config["jobs"][job]
    directory = output / "carla25"
    if entry["node_rank"] == 0:
        prepare(config, output)
    deadline = time.time() + 1800
    while not (directory / "ready.json").exists():
        if time.time() > deadline:
            raise TimeoutError("The selected-checkpoint evaluation setup did not complete")
        time.sleep(5)
    verified = inspect_allocation(config, job, idle=True)
    scheduler = query_allocation(job, entry["host"])
    proof = {**scheduler, **verified, "job_id": job, "state": "RUN", "slots": 16,
             "mem_per_slot_mb": 16000, "gpu_uuids": entry["gpu_uuids"], "checked_unix": time.time()}
    record = directory / "allocations" / f"allocation_{job}.json"
    atomic_json(record, proof)
    budget = min(16 * 3600, int(scheduler["deadline_unix"] - time.time() - 1800))
    if budget < 1800:
        raise RuntimeError("Less than thirty minutes remains in the supplied allocation")
    env = {**os.environ, "WMMPC_CARLA_RESULTS_ROOT": str(output.parent), "SLOWER": "0"}
    os.environ["WMMPC_CARLA_RESULTS_ROOT"] = str(output.parent)
    processes, handles = [], []
    try:
        for gpu in (0, 1):
            index = entry["node_rank"] * 2 + gpu
            group = f"{job}_gpu{gpu}"
            handle = (directory / f"{group}.controller.log").open("xb")
            handles.append(handle)
            command = [sys.executable, "-u", "-m", "method.full_diffusion_pointquery_cvci.eval_controller",
                "--method", "full_diffusion_pointquery_cvci", "--job-id", job, "--run-root", str(directory),
                "--checkpoint", str(directory / "selected_best.pt"), "--gpu", str(gpu), "--group-id", group,
                "--route-ids", "all", "--slot-base", str(3500 + index * 80), "--tm-base", str(45000 + index * 600),
                "--seed", "901", "--backend", "direct", "--lanes", "1", "--cpus-per-gpu", str(entry["eval_cpus_per_gpu"]),
                "--server-mode", "warm", "--startup-wait", "rpc", "--routes-per-server", "32",
                "--wall-budget", str(budget), "--allocation-record", str(record)]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        atomic_json(directory / f"launch_{job}.json", {"job": job, "pids": [p.pid for p in processes],
                    "selected_best": json.loads((directory / "ready.json").read_text()), "unix": time.time()})
        if entry["node_rank"] == 0:
            from evaluation.summarize_parallel_carla import summarize
            while any(process.poll() is None for process in processes):
                if (directory / "route_queue.json").exists():
                    try:
                        atomic_json(directory / "summary.json", summarize(directory))
                    except (OSError, ValueError, KeyError) as error:
                        print(f"Summary pending: {error!r}", flush=True)
                time.sleep(30)
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    atomic_json(directory / f"job_{job}.json", {"return_codes": codes, "unix": time.time()})
    if entry["node_rank"] == 0:
        from evaluation.summarize_parallel_carla import summarize
        deadline = time.time() + 1800
        while not all((directory / f"job_{other}.json").exists() for other in config["jobs"]):
            if time.time() > deadline:
                break
            time.sleep(30)
        atomic_json(directory / "summary.json", summarize(directory, extended=True, merge=True))
    if any(codes):
        raise RuntimeError(f"Closed-loop controllers returned {codes}; retain logs and allocation")

