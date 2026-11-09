"""Owned node controllers and an independent, bounded CPU telemetry process."""

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from method.full_diffusion2.launch import AUTHORIZED, guard_context, inspect_allocation
from method.full_diffusion2.snapshot import atomic_json
from method.full_diffusionv3.runtime import local_directory, read_trace


GPU_UUIDS = {
    "13053006": {"GPU-bff11038-816b-34d5-1a5a-5506e873362e", "GPU-995a83a0-4897-fc64-3550-2e8cd068ee31"},
    "13047896": {"GPU-178dbb9f-517a-16af-9122-031d88c03b5e", "GPU-f653bbe0-9522-f398-aff0-3931b9958f47"},
    "13047898": {"GPU-e096e26f-e4ce-456e-4b87-ad3b7cfe3008", "GPU-d349a998-9556-3217-2b00-461f6c910b60"},
    "13046830": {"GPU-7bf1e1c2-f957-24b1-78a8-30a803781df7", "GPU-249dd7ce-bdb4-3cb5-77e2-8ee3bed6ed34"},
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def process_info(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "uid": path.stat().st_uid, "state": fields[0], "ppid": int(fields[1]),
                "pgrp": int(fields[2]), "start_ticks": int(fields[19]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                "wchan": (path / "wchan").read_text().strip(),
                "cgroup": (path / "cgroup").read_text().strip(),
                "argv": (path / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")}
    except (OSError, ValueError, IndexError):
        return None


def job_processes(job):
    records = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                if path.stat().st_uid != os.getuid():
                    continue
                record = process_info(int(path.name))
                if record and f"job.{job}." in record["cgroup"]:
                    records.append(record)
            except OSError:
                continue
    return records


def verify_owned(pid, job, required, ticks):
    record = process_info(pid)
    if record is None:
        return None
    if (record["uid"] != os.getuid() or f"job.{job}." not in record["cgroup"]
            or record["start_ticks"] != ticks):
        raise RuntimeError(f"PID ownership/job/start-time changed: {pid}")
    if record["state"] in ("Z", "X"):
        return None
    if any(value not in record["argv"] for value in required):
        raise RuntimeError(f"Refusing to signal a different command: {pid}")
    return record


def verified_allocation(job):
    report = inspect_allocation(job)
    if {row[1] for row in report["devices"]} != GPU_UUIDS[job]:
        raise RuntimeError("The original authorized GPU allocation changed")
    return report


def memory_sample():
    try:
        host_pressure = Path("/proc/pressure/memory").read_text().strip()
    except OSError:
        host_pressure = None
    groups = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            groups.append(Path("/sys/fs/cgroup") / relative.lstrip("/"))
        elif "memory" in controllers.split(","):
            groups.append(Path("/sys/fs/cgroup/memory") / relative.lstrip("/"))
    fallback = None
    for start in groups:
        for path in (start, *start.parents):
            if "job." not in str(path):
                continue
            filenames = ("memory.current", "memory.peak", "memory.max", "memory.high", "memory.events",
                         "memory.pressure", "memory.swap.current", "memory.usage_in_bytes",
                         "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt")
            result = {"cgroup_path": str(path), "host_memory_pressure": host_pressure}
            for name in filenames:
                try:
                    text = (path / name).read_text().strip()
                    result[name] = int(text) if text.isdigit() else text
                except OSError:
                    continue
            if len(result) > 2:
                fallback = fallback or result
                limit = result.get("memory.max", result.get("memory.limit_in_bytes"))
                if isinstance(limit, int) and 0 < limit < 2**60:
                    return result
    return fallback or {"cgroup_path": None, "job_memory_unavailable": True, "host_memory_pressure": host_pressure}


def training_root(config):
    return Path(config["source_identity_file"]).parent if config.get("source_identity_file") else ROOT


def node_environment(config, report, directory):
    source = training_root(config)
    navigation = source / "third_party/carla_navigation_0_9_15"
    return dict(os.environ, CUDA_VISIBLE_DEVICES=report["cuda_visible_devices"],
                OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
                PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                PYTHONPATH=os.pathsep.join((str(source), str(navigation))),
                WMMPC_CARLA_NAVIGATION_PATH=str(navigation), WMMPC_BRIDGE_MODULES=config["bridge_modules"],
                NCCL_SOCKET_IFNAME="bond0", GLOO_SOCKET_IFNAME="bond0",
                NCCL_IB_DISABLE="1", NCCL_NVLS_ENABLE="0", NCCL_SOCKET_FAMILY="AF_INET",
                NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                TORCH_NCCL_TRACE_BUFFER_SIZE="8192", TORCH_NCCL_DUMP_ON_TIMEOUT="1",
                TORCH_NCCL_DESYNC_DEBUG="1", TORCH_NCCL_ENABLE_TIMING="1",
                TORCH_NCCL_DEBUG_INFO_TEMP_FILE=str(directory / "node.nccl_trace"))


def node_command(args, node, resume=None):
    config = json.loads(args.config.read_text())
    command = [config.get("training_python", sys.executable), "-u", "-m", "torch.distributed.run", "--nnodes=4", "--nproc_per_node=2",
               f"--node_rank={node}", f"--master_addr={config['master_address']}",
               f"--master_port={config['master_port']}", "--max_restarts=0",
               str(training_root(config) / "method/full_diffusionv3/train.py"), "--config", str(args.config),
               "--output", str(args.output), "--gpu-authorized", "--attempt", str(args.attempt), "--phase", args.phase]
    if resume:
        command.extend(("--resume", "--resume-checkpoint", str(resume)))
    return command


def reset_local_traces(directory):
    if directory.stat().st_uid != os.getuid() or directory.is_symlink():
        raise RuntimeError("Refusing to reset another owner's local trace directory")
    for path in directory.iterdir():
        if path.name != "abort.request" and not re.fullmatch(r"rank[0-9]+\.trace", path.name):
            raise RuntimeError(f"Unexpected local trace member: {path}")
        if path.stat().st_uid != os.getuid() or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Refusing to reset an unowned trace: {path}")
    for path in directory.iterdir():
        path.unlink()


def monitor(args, job, node):
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    config = json.loads(args.config.read_text())
    settings = config.get("runtime", {})
    interval = settings.get("monitor_interval_seconds", 5)
    export_interval = settings.get("monitor_export_seconds", 30)
    samples = deque(maxlen=settings.get("monitor_buffer_samples", 72))
    traces = local_directory(args.output, args.attempt)
    directory = args.output / "attempts" / f"attempt_{args.attempt:03}"
    baseline, sequence, last_export = None, 0, 0.0
    while True:
        sequence += 1
        ranks = [read_trace(traces / f"rank{rank}.trace", limit=8) for rank in (node * 2, node * 2 + 1)]
        ranks = [record for record in ranks if record]
        memory = memory_sample()
        counters = {}
        for line in str(memory.get("memory.events", "")).splitlines():
            name, count = line.split()
            counters[name] = int(count)
        if baseline is None:
            baseline = counters.copy()
        records = job_processes(job)
        summary = {"utc": utc(), "sequence": sequence, "memory": memory,
                   "memory_event_delta_since_monitor_start": {name: value - baseline.get(name, 0) for name, value in counters.items()},
                   "ranks": [{key: value for key, value in record.items() if key != "events"} for record in ranks],
                   "process_states": dict(Counter(record["state"] for record in records)),
                   "blocked_processes": [{key: record[key] for key in ("pid", "ppid", "state", "wchan", "rss_bytes")}
                                         for record in records if record["state"] == "D"]}
        samples.append(summary)
        if time.monotonic() - last_export >= export_interval or stop.is_set():
            gpu = None
            try:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used",
                                               "--format=csv,noheader,nounits"], text=True, timeout=8).strip()
            except (subprocess.SubprocessError, OSError) as error:
                gpu = repr(error)
            current = {"job": job, "host": socket.gethostname(), "node": node, "attempt": args.attempt,
                       "monitor_pid": os.getpid(), "sequence": sequence, "sample_utc": utc(),
                       "ranks": ranks, "processes": records, "job_memory": memory,
                       "gpu_external_query": gpu, "recent_samples": list(samples),
                       "bounded_samples": samples.maxlen, "sample_seconds": interval, "shared_export_seconds": export_interval}
            atomic_json(directory / f"node{node}.telemetry.json", current)
            last_export = time.monotonic()
        if stop.is_set():
            break
        stop.wait(interval)
    for record in ranks:
        full = read_trace(traces / f"rank{record['rank']}.trace", limit=512)
        if full:
            atomic_json(directory / f"rank{record['rank']}.final_trace.json", full)
    atomic_json(directory / f"node{node}.monitor.complete.json", {"job": job, "attempt": args.attempt,
                                                               "final_sequence": sequence, "utc": utc()})


def run_node(args, job, node):
    directory = args.output / "attempts" / f"attempt_{args.attempt:03}"
    config = json.loads(args.config.read_text())
    path = directory / f"node{node}.status.json"
    status = {"job": job, "node": node, "attempt": args.attempt, "phase": args.phase,
              "controller_pid": os.getpid(), "controller_start_ticks": process_info(os.getpid())["start_ticks"],
              "host": socket.gethostname(), "state": "preflight", "started_at": utc()}
    stop = threading.Event()
    children = {}

    def abort(*unused):
        stop.set()
        (local_directory(args.output, args.attempt) / "abort.request").touch()
        child = children.get("torchrun")
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

            def force():
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)

            timer = threading.Timer(25, force)
            timer.daemon = True
            timer.start()

    signal.signal(signal.SIGTERM, abort)
    signal.signal(signal.SIGINT, abort)
    with (args.output / f".node{node}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(path, status)
        try:
            report = verified_allocation(job)
            if not report["available"]:
                raise RuntimeError("The original allocation is not idle; do not reset traces or launch")
            reset_local_traces(local_directory(args.output, args.attempt))
            subprocess.run([config["training_python"], str(Path(__file__).resolve()), "--action", "preflight",
                            "--job-id", job, "--config", str(args.config), "--output", str(args.output),
                            "--attempt", str(args.attempt), "--phase", args.phase], check=True, timeout=150)
            report = json.loads((directory / f"gpu_{job}.json").read_text())
            monitor_command = [config["training_python"], "-u", str(Path(__file__).resolve()), "--action", "monitor",
                               "--job-id", job, "--config", str(args.config), "--output", str(args.output),
                               "--attempt", str(args.attempt), "--phase", args.phase]
            children["monitor"] = subprocess.Popen(monitor_command, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdin=subprocess.DEVNULL, start_new_session=True)
            status.update({"state": "ready", "monitor_pid": children["monitor"].pid,
                           "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                           "gpu_uuids": sorted(row[1] for row in report["devices"])})
            atomic_json(path, status)
            gate = directory / "start.json"
            began = time.monotonic()
            while not gate.exists():
                if stop.wait(2) or time.monotonic() - began > 600:
                    raise RuntimeError("Global attempt start was cancelled or timed out")
            launch = json.loads(gate.read_text())
            if launch["attempt"] != args.attempt or launch["phase"] != args.phase:
                raise RuntimeError("Wrong global launch gate")
            environment = node_environment(config, report, directory)
            environment["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = str(directory / f"node{node}.nccl_trace")
            command = node_command(args, node, launch.get("resume_checkpoint"))
            children["torchrun"] = subprocess.Popen(command, cwd=training_root(config), env=environment,
                                                    stdin=subprocess.DEVNULL, start_new_session=True)
            child = children["torchrun"]
            record = process_info(child.pid)
            status.update({"state": "running", "worker_parent_pid": child.pid,
                           "worker_parent_start_ticks": record["start_ticks"], "command": command,
                           "resume_checkpoint": launch.get("resume_checkpoint"), "started_workers_at": utc()})
            atomic_json(path, status)
            result = child.wait()
            if result != 0:
                raise RuntimeError(f"torchrun failed with return code {result}; no per-node automatic restart")
            status.update({"state": "completed", "return_code": 0, "finished_at": utc()})
        except BaseException as error:
            status.update({"state": "failed", "error": repr(error), "return_code": 1, "finished_at": utc()})
            abort()
            if children.get("torchrun") is not None:
                children["torchrun"].wait(timeout=35)
        finally:
            if children.get("monitor") is not None:
                children["monitor"].send_signal(signal.SIGTERM)
                try:
                    children["monitor"].wait(timeout=35)
                except subprocess.TimeoutExpired:
                    children["monitor"].kill()
                    children["monitor"].wait()
            atomic_json(path, status)
    return status.get("return_code", 1)


def inspect_node(args, job, node):
    locked = False
    entry = str(training_root(json.loads(args.config.read_text())) / "method/full_diffusionv3/train.py")
    with (args.output / f".node{node}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
    live = []
    groups = set()
    for path in (args.output / "attempts").glob(f"attempt_*/node{node}.status.json"):
        status = json.loads(path.read_text())
        if status.get("job") == job and status.get("worker_parent_pid"):
            groups.add(status["worker_parent_pid"])
    for record in job_processes(job):
        if record["state"] in ("Z", "X"):
            continue
        argv = record["argv"]
        if record["pgrp"] in groups or (str(args.output) in argv and entry in argv):
            live.append(record)
    allocation = verified_allocation(job)
    return {"job": job, "node": node, "request": args.request, "attempt": args.attempt,
            "allocation": allocation, "controller_lock_free": not locked,
            "live_training_processes": live, "exited_and_idle": not locked and not live and allocation["available"], "utc": utc()}


def stop_node(args, job, node):
    import psutil
    directory = args.output / "attempts" / f"attempt_{args.attempt:03}"
    path = directory / f"node{node}.status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    (local_directory(args.output, args.attempt) / "abort.request").touch()
    pid = status.get("controller_pid")
    if pid:
        required = [str(Path(__file__).resolve()), "--action", "run", str(args.output)]
        owned = verify_owned(pid, job, required, status["controller_start_ticks"])
        if owned:
            os.kill(pid, signal.SIGTERM)
            try:
                psutil.Process(pid).wait(timeout=45)
            except psutil.TimeoutExpired:
                owned = verify_owned(pid, job, required, status["controller_start_ticks"])
                if owned:
                    os.kill(pid, signal.SIGKILL)
                    psutil.Process(pid).wait(timeout=10)
    pid = status.get("worker_parent_pid")
    if pid:
        entry = training_root(json.loads(args.config.read_text())) / "method/full_diffusionv3/train.py"
        required = ["torch.distributed.run", str(entry), str(args.output)]
        owned = verify_owned(pid, job, required, status["worker_parent_start_ticks"])
        if owned:
            os.killpg(pid, signal.SIGKILL)
            psutil.Process(pid).wait(timeout=15)
    return inspect_node(args, job, node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("start", "run", "inspect", "stop", "monitor", "preflight"), required=True)
    parser.add_argument("--job-id", choices=AUTHORIZED, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--phase", choices=("acceptance", "train"), default="train")
    parser.add_argument("--request", default="manual")
    args = parser.parse_args()
    job = guard_context(args.job_id)
    node = list(AUTHORIZED).index(job)
    args.output, args.config = args.output.resolve(), args.config.resolve()
    directory = args.output / "attempts" / f"attempt_{args.attempt:03}"
    directory.mkdir(parents=True, exist_ok=True)
    if args.action == "preflight":
        from method.full_diffusion2.launch import gpu_preflight
        import torch
        if not torch.__version__.startswith("2.5.0") or torch.version.cuda != "12.1":
            raise RuntimeError("V3 requires the pinned Torch2.5.0/CUDA12.1 training environment")
        verified_allocation(job)
        gpu_preflight(job, directory)
        return
    if args.action == "monitor":
        monitor(args, job, node)
        return
    if args.action == "run":
        sys.exit(run_node(args, job, node))
    if args.action in ("stop", "inspect"):
        report = stop_node(args, job, node) if args.action == "stop" else inspect_node(args, job, node)
        atomic_json(directory / f"node{node}.{args.action}.json", report)
        print(json.dumps(report), flush=True)
        return
    config = json.loads(args.config.read_text())
    command = [config["training_python"], "-u", str(Path(__file__).resolve()), "--action", "run", "--job-id", job,
               "--config", str(args.config), "--output", str(args.output), "--attempt", str(args.attempt), "--phase", args.phase]
    with (directory / f"node{node}.controller.log").open("a") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    report = {"controller_pid": child.pid, "job": job, "node": node, "attempt": args.attempt,
              "request": args.request, "phase": args.phase, "utc": utc()}
    atomic_json(directory / f"node{node}.launch.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()