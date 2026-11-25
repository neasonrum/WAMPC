"""Recover only the two workers interrupted by a diagnosed NVML-monitor timeout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.carla_route_queue import RouteQueue
from method.full_diffusion_pointquery.utils import atomic_json, inspect_allocation, sha
from method.full_diffusionv3_ego12.allocation import query_allocation


def live(pid):
    p = Path(f"/proc/{pid}/stat")
    return p.exists() and p.read_text().split(") ", 1)[1][0] != "Z"


def close_owned_orphan(owner_file, job, audit):
    owner = json.loads(owner_file.read_text())
    pid = int(owner["pid"])
    if not live(pid):
        return
    p = Path(f"/proc/{pid}")
    command = (p / "cmdline").read_bytes().decode().split("\0")[:-1]
    status = (p / "status").read_text()
    uid = int(next(s.split()[1] for s in status.splitlines() if s.startswith("Uid:")))
    parent = int(next(s.split()[1] for s in status.splitlines() if s.startswith("PPid:")))
    if (owner["job_id"] != job or uid != os.getuid() or parent != 1
            or f"/job.{job}." not in (p / "cgroup").read_text()
            or command != owner["command"] or "CarlaUE4-Linux-Shipping" not in command[0]):
        raise RuntimeError(f"Cannot verify this as the stopped worker's orphan renderer: {pid}")
    record = {"pid": pid, "owner_file": str(owner_file), "command": command, "signal": "SIGTERM", "unix": time.time()}
    atomic_json(audit / f"orphan_{pid}_before_stop.json", record)
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 20
    while live(pid):
        if time.time() > deadline:
            raise RuntimeError(f"Owned orphan {pid} did not stop; preserve it for inspection")
        time.sleep(.2)
    atomic_json(audit / f"orphan_{pid}_stopped.json", {**record, "stopped_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("13177034",))
    args = parser.parse_args()
    output = Path("/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/full_diffusion_pointquery_8a100_20261124_v1")
    run = output / "carla25"
    config = json.loads((output / "config.json").read_text())
    entry = config["jobs"][args.job]
    lock = (run / f".monitor_recovery_{args.job}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = output / "monitor_recovery"
    audit.mkdir(exist_ok=True)
    inspect_allocation(config, args.job)
    if (run / f"queue_before_monitor_recovery_{args.job}.json").exists():
        raise RuntimeError("This recovery has already been attempted")
    for name, expected in json.loads((run / "source_sha256.json").read_text()).items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Prepared planner/evaluation source changed: {name}")
    ready = json.loads((run / "ready.json").read_text())
    if sha(run / "selected_best.pt") != ready["checkpoint_sha256"]:
        raise RuntimeError("The fixed selected checkpoint changed")
    if (output / "monitor_recovery_selftest.exit").read_text().strip() != "0":
        raise RuntimeError("Monitor recovery preflight has not passed")
    for gpu in (0, 1):
        group = run / "groups" / f"{args.job}_gpu{gpu}"
        text = (run / f"{args.job}_gpu{gpu}.controller.log").read_text()
        if "subprocess.TimeoutExpired" not in text or "nvidia-smi" not in text or "after 15 seconds" not in text:
            raise RuntimeError("Old worker does not match the inspected monitor failure")
        processes = json.loads((group / "processes.json").read_text())
        if any(live(pid) for pid in [processes["parent_pid"], *processes["children"]]):
            raise RuntimeError("An original controller is still running")
        status = json.loads((group / "lane_0/status.json").read_text())
        if live(status["evaluator_pid"]):
            raise RuntimeError("An original evaluator is still running")
        for owner in (group / "lane_0/servers").rglob("server_process.json"):
            close_owned_orphan(owner, args.job, audit)
    proof = {**query_allocation(args.job, entry["host"]), **inspect_allocation(config, args.job, idle=True),
             "gpu_uuids": entry["gpu_uuids"], "checked_unix": time.time()}
    budget = min(12 * 3600, int(proof["deadline_unix"] - time.time() - 1800))
    if budget < 1800:
        raise RuntimeError("Insufficient allocation time")
    queue = RouteQueue(run)
    snapshot = queue.snapshot()
    atomic_json(run / f"queue_before_monitor_recovery_{args.job}.json", snapshot)
    recovered = []
    for rid, item in snapshot["items"].items():
        if item["state"] != "claimed" or item.get("owner") not in [f"{args.job}_gpu{i}/lane0" for i in (0, 1)]:
            continue
        attempt = run / "routes" / f"route_{rid}" / f"attempt_{item['attempts']:02d}"
        records = json.loads((attempt / "results.json").read_text()).get("_checkpoint", {}).get("records", [])
        if records:
            raise RuntimeError(f"Interrupted route {rid} has a driving result; reconcile it without rerunning")
        summary = {"route_id": rid, "classification": "infrastructure_error",
                   "supervisor_reason": "periodic_gpu_monitor_timeout", "attempt": str(attempt),
                   "error": "NVML monitor exceeded 15 seconds; original controller/evaluator exited before a scored outcome",
                   "inference_steps": 25, "queue_worker": item["owner"]}
        atomic_json(attempt / "monitor_recovery_classification.json", summary)
        queue.finish(rid, item["owner"], summary)
        recovered.append(rid)
    after = queue.snapshot()
    if set(recovered) != {"1825", "2283"}:
        raise RuntimeError(f"Unexpected recovery route set: {recovered}")
    assert all(after["items"][rid] == item for rid, item in snapshot["items"].items() if item["state"] == "done")
    record = run / "allocations" / f"allocation_{args.job}_monitorfix1.json"
    atomic_json(record, proof)
    processes, handles = [], []
    for gpu in (0, 1):
        index = entry["node_rank"] * 2 + gpu
        group = f"{args.job}_gpu{gpu}_monitorfix1"
        handle = (run / f"{group}.controller.log").open("xb")
        handles.append(handle)
        command = [sys.executable, "-u", str(ROOT / "scripts/pointquery_monitor_recovery_20261124.py"),
                   "--method", "full_diffusion_pointquery", "--job-id", args.job, "--run-root", str(run),
                   "--checkpoint", str(run / "selected_best.pt"), "--gpu", str(gpu), "--group-id", group,
                   "--route-ids", "all", "--slot-base", str(3500 + index * 80), "--tm-base", str(45000 + index * 600),
                   "--seed", "901", "--backend", "direct", "--lanes", "1", "--cpus-per-gpu", "4",
                   "--server-mode", "warm", "--startup-wait", "rpc", "--routes-per-server", "32",
                   "--wall-budget", str(budget), "--allocation-record", str(record)]
        processes.append(subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
    atomic_json(run / "monitor_recovery_13177034.json", {"state": "running", "recovered_routes": recovered,
                "controller_pids": [p.pid for p in processes], "checkpoint_sha256": ready["checkpoint_sha256"],
                "model_source_unchanged": True, "already_scored_routes_unchanged": True,
                "monitor_wrapper_sha256": sha(ROOT / "scripts/pointquery_monitor_recovery_20261124.py"), "unix": time.time()})
    atomic_json(output / f"node_{args.job}.json", {"state": "closed_loop_recovery", "unix": time.time()})
    from evaluation.summarize_parallel_carla import summarize
    while any(p.poll() is None for p in processes):
        atomic_json(run / "summary.json", summarize(run))
        time.sleep(20)
    codes = [p.wait() for p in processes]
    for handle in handles:
        handle.close()
    atomic_json(run / "job_13177034_monitorfix1.json", {"return_codes": codes, "unix": time.time()})
    if any(codes):
        raise RuntimeError(f"Recovered workers exited {codes}")
    deadline = time.time() + budget
    while time.time() < deadline:
        summary = summarize(run)
        atomic_json(run / "summary.json", summary)
        if summary["complete_220_routes"]:
            atomic_json(run / "summary.json", summarize(run, extended=True, merge=True))
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
