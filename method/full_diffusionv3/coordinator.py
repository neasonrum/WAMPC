"""One bounded recovery authority for all four existing LSF allocations."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from method.full_diffusion2.launch import AUTHORIZED
from method.full_diffusion2.snapshot import atomic_json
from method.full_diffusionv3.checkpoint import select_complete


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze_sources(output, config_path):
    destination = output / "source_snapshot"
    if destination.exists():
        config = destination / "method/full_diffusionv3/config.json"
        if not config.exists():
            raise RuntimeError("Existing source snapshot is incomplete; preserve it and use a new run")
        return config
    paths = []
    for name in ("method/full_diffusion", "method/full_diffusion2", "method/full_diffusionv3", "train", "tools"):
        paths.extend(path for path in (ROOT / name).rglob("*.py") if "__pycache__" not in path.parts)
    paths += list((ROOT / "third_party/carla_garage/team_code").rglob("*.py"))
    paths += list((ROOT / "third_party/carla_navigation_0_9_15").rglob("*.py"))
    paths += [ROOT / "third_party/carla_navigation_0_9_15/source.json"]
    paths += [ROOT / "configs/paths.json"]
    for name in ("method/__init__.py", "train/__init__.py", "tools/__init__.py"):
        if (ROOT / name).exists():
            paths.append(ROOT / name)
    for original in paths:
        target = destination / original.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    bridge = ROOT.parent / "BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules"
    bridge_snapshot = destination / "third_party/bridge_modules"
    bridge_snapshot.mkdir(parents=True)
    for name in ("blocks.py", "conditional_unet1d.py"):
        shutil.copy2(bridge / name, bridge_snapshot / name)
    sources = {str((destination / path.relative_to(ROOT)).resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths if path.suffix == ".py"}
    sources.update({str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in bridge_snapshot.glob("*.py")})
    atomic_json(destination / "code_identity.json", sources)
    config = json.loads(Path(config_path).read_text())
    config["training_python"] = os.path.abspath(ROOT / config["training_python"])
    for name in ("index_directory", "normalization_file", "backbone_directory", "sensor_snapshot"):
        if config.get(name):
            config[name] = str((ROOT / config[name]).resolve())
    config["source_identity_file"] = str(destination / "code_identity.json")
    config["bridge_modules"] = str(bridge_snapshot)
    frozen = destination / "method/full_diffusionv3/config.json"
    atomic_json(frozen, config)
    atomic_json(output / "frozen_source_receipt.json", {"source_root": str(destination), "files": len(sources),
                                                       "config_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(), "utc": utc()})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(destination),
                       WMMPC_BRIDGE_MODULES=str(bridge_snapshot),
                       WMMPC_CARLA_NAVIGATION_PATH=str(destination / "third_party/carla_navigation_0_9_15"))
    probe = "import sys; from method.full_diffusionv3.train import training_metadata; import json; training_metadata(json.load(open(sys.argv[1]))); print('Frozen V3 training imports and metadata: PASS')"
    subprocess.run([config["training_python"], "-c", probe, str(frozen)], cwd=destination,
                   env=environment, check=True, timeout=120)
    return frozen


def all_nodes_verified(reports):
    if len(reports) != len(AUTHORIZED) or {record.get("job") for record in reports} != set(AUTHORIZED):
        return False
    return all(record.get("exited_and_idle") and record.get("controller_lock_free")
               and not record.get("live_training_processes") and record.get("allocation", {}).get("available")
               for record in reports)


def node_request(args, output, attempt, phase, action, job):
    node = list(AUTHORIZED).index(job)
    directory = output / "attempts" / f"attempt_{attempt:03}"
    directory.mkdir(parents=True, exist_ok=True)
    request = uuid.uuid4().hex
    entry = ROOT / "method/full_diffusionv3/supervisor.py" if getattr(args, "resume", False) else args.config.parent / "supervisor.py"
    python = json.loads(args.config.read_text())["training_python"]
    command = ["exec", "env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1",
               python, str(entry), "--action", action, "--job-id", job, "--output", str(output),
               "--config", str(args.config), "--attempt", str(attempt), "--phase", phase, "--request", request]
    result = subprocess.run(["battach", "-L", "/bin/bash", job], input=shlex.join(command) + "\n",
                            text=True, capture_output=True, timeout=180)
    with (directory / f"node{node}.attach.log").open("a") as handle:
        handle.write(f"{utc()} action={action} request={request}\n" + result.stdout + result.stderr)
    result.check_returncode()
    suffix = "launch" if action == "start" else action
    response = json.loads((directory / f"node{node}.{suffix}.json").read_text())
    if response.get("request") != request or response.get("job") != job or response.get("attempt") != attempt:
        raise RuntimeError("Attachment returned without a fresh, matching node acknowledgement")
    return response


def verify_cluster(args, output, attempt, phase, stop=False):
    reports = []
    errors = []
    for job in AUTHORIZED:
        try:
            reports.append(node_request(args, output, attempt, phase, "stop" if stop else "inspect", job))
        except BaseException as error:
            errors.append({"job": job, "error": repr(error)})
    record = {"attempt": attempt, "phase": phase, "stop_requested": stop, "reports": reports,
              "errors": errors, "all_nodes_exited_and_original_allocations_idle": all_nodes_verified(reports), "utc": utc()}
    atomic_json(output / "attempts" / f"attempt_{attempt:03}" / ("stop_verification.json" if stop else "preflight_verification.json"), record)
    if errors or not record["all_nodes_exited_and_original_allocations_idle"]:
        raise RuntimeError(f"Cannot verify every original allocation idle; no restart permitted: {record}")
    return record


def checkpoint_for_restart(output, config):
    import torch
    record = select_complete(output, config)
    if record is None:
        return None
    checkpoint = torch.load(record["absolute_path"], map_location="cpu", weights_only=False)
    if (checkpoint["config"] != config or checkpoint["progress"] != record["progress"]
            or len(checkpoint["rng_by_rank"]) != config["world_size"] or not checkpoint["model"]
            or not checkpoint["optimizer"]["state"]):
        raise RuntimeError("Checkpoint metadata/model/optimizer/RNG completeness check failed")
    expected = checkpoint["progress"]["updates"]
    steps = {int(state["step"]) for state in checkpoint["optimizer"]["state"].values()}
    if steps != {expected}:
        raise RuntimeError(f"Checkpoint Adam steps differ from completed updates: {steps} versus {expected}")
    dataset_path = output / "dataset.json"
    if dataset_path.exists() and checkpoint["dataset"] != json.loads(dataset_path.read_text()):
        raise RuntimeError("Checkpoint dataset/source identity differs from the frozen run")
    return record


def first_attempt(output, resume):
    previous = [int(path.name.removeprefix("attempt_"))
                for path in (output / "attempts").glob("attempt_[0-9][0-9][0-9]") if path.is_dir()]
    if previous and not resume:
        raise RuntimeError("Existing attempts require explicit --resume; prior attempts cannot be reused")
    return max(previous) + 1 if previous else 0


def run_cluster(args, output, phase, stop_event, recovery_limit, start_attempt=0):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    settings = config["runtime"]
    for recovery in range(recovery_limit + 1):
        attempt = start_attempt + recovery
        directory = output / "attempts" / f"attempt_{attempt:03}"
        if (directory / "start.json").exists():
            raise RuntimeError("An attempt directory cannot be reused; preserve prior runs")
        verify_cluster(args, output, attempt, phase)
        checkpoint = checkpoint_for_restart(output, config) if phase == "train" else None
        atomic_json(directory / "checkpoint_selected.json", checkpoint or {"fresh_seed": config["seed"], "updates": 0})
        try:
            for job in AUTHORIZED:
                node_request(args, output, attempt, phase, "start", job)
        except BaseException:
            verify_cluster(args, output, attempt, phase, stop=True)
            raise
        state = {"state": "starting", "phase": phase, "attempt": attempt, "max_recoveries": recovery_limit,
                 "recovery": recovery, "start_attempt": start_attempt,
                 "config": str(args.config), "resume_checkpoint": checkpoint, "utc": utc()}
        atomic_json(args.output / "coordinator.status.json", state)
        began, ready, failure = time.monotonic(), False, None
        last_monitor = {}
        while not stop_event.is_set():
            reports = []
            for node in range(4):
                path = directory / f"node{node}.status.json"
                reports.append(json.loads(path.read_text()) if path.exists() else {})
            failed = [record for record in reports if record.get("state") == "failed"]
            if failed:
                failure = {"reason": "node_attempt_failed", "nodes": failed}
                break
            if not ready and all(record.get("state") == "ready" for record in reports):
                expected = hashlib.sha256(args.config.read_bytes()).hexdigest()
                if any(record.get("config_sha256") != expected for record in reports):
                    failure = {"reason": "node_configuration_disagrees"}
                    break
                atomic_json(directory / "start.json", {"attempt": attempt, "phase": phase,
                            "resume_checkpoint": checkpoint["absolute_path"] if checkpoint else None,
                            "checkpoint_sha256": checkpoint["sha256"] if checkpoint else None, "utc": utc()})
                ready = True
                state.update({"state": "running", "started_at": utc()})
                atomic_json(args.output / "coordinator.status.json", state)
            if all(record.get("state") == "completed" for record in reports):
                verify_cluster(args, output, attempt, phase)
                if phase == "acceptance":
                    results = [json.loads((output / f"acceptance_rank{rank}.json").read_text()) for rank in range(8)]
                    if not all(record.get("passed") for record in results):
                        raise RuntimeError("Real eight-rank resume acceptance did not pass")
                    atomic_json(output / "accepted.json", {"passed": True, "ranks": results, "utc": utc()})
                elif not (output / "completed.json").exists():
                    raise RuntimeError("Nodes exited without complete formal training results")
                return
            if not ready and time.monotonic() - began > settings["startup_timeout_seconds"]:
                failure = {"reason": "startup_timeout", "nodes": reports}
                break
            if ready:
                for node in range(4):
                    path = directory / f"node{node}.telemetry.json"
                    if not path.exists():
                        continue
                    telemetry = json.loads(path.read_text())
                    sequence = telemetry.get("sequence")
                    previous, observed = last_monitor.get(node, (None, time.monotonic()))
                    if sequence != previous:
                        last_monitor[node] = (sequence, time.monotonic())
                    elif time.monotonic() - observed > settings["monitor_stale_seconds"]:
                        failure = {"reason": "independent_monitor_stale", "node": node}
                    stalled = [record for record in telemetry.get("ranks", [])
                               if record.get("phase") not in ("completed", "failed")
                               and record.get("heartbeat_age_seconds", 0) > settings["rank_stall_seconds"]]
                    if stalled:
                        failure = {"reason": "rank_heartbeat_stalled", "node": node, "ranks": stalled}
                if failure:
                    break
            stop_event.wait(settings["coordinator_interval_seconds"])
        failure = failure or {"reason": "coordinator_stop_requested"}
        atomic_json(directory / "failure.json", {**failure, "attempt": attempt, "phase": phase, "utc": utc(),
                                                  "new_uncertain_checkpoint_requested": False})
        verify_cluster(args, output, attempt, phase, stop=True)
        atomic_json(directory / "failure_preserved.json", {"all_node_logs_and_traces_preserved": True,
                                                          "next_attempt_allowed": not stop_event.is_set() and recovery < recovery_limit, "utc": utc()})
        if stop_event.is_set() or recovery >= recovery_limit:
            raise RuntimeError(f"Finite recovery stopped at attempt {attempt}: {failure}")
    raise RuntimeError("No attempt completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.gpu_authorized and not args.prepare_only:
        parser.error("Explicit approval is required before using the original GPU jobs")
    if not socket.gethostname().split(".")[0].startswith("rng-dl01-login"):
        raise RuntimeError("The CPU-only global coordinator must run on the login host")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.config = freeze_sources(args.output, args.config.resolve())
    if args.prepare_only:
        print(json.dumps({"frozen_config": str(args.config)}), flush=True)
        return
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(args.config),
                   "--output", str(args.output), "--gpu-authorized"]
        if args.resume:
            command.append("--resume")
        with (args.output / "coordinator.log").open("a") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        atomic_json(args.output / "coordinator.launch.json", {"pid": child.pid, "command": command, "GPU_context_in_parent": False, "utc": utc()})
        print(json.dumps({"coordinator_pid": child.pid, "run": str(args.output)}), flush=True)
        return
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    with (args.output / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            accepted = args.output / "acceptance/accepted.json"
            config = json.loads(args.config.read_text())
            start_attempt = first_attempt(args.output, args.resume)
            if args.resume and (not accepted.exists() or checkpoint_for_restart(args.output, config) is None):
                raise RuntimeError("Explicit resume requires prior acceptance and a complete formal checkpoint")
            if (args.output / "completed.json").exists():
                raise RuntimeError("Formal training is already complete; no restart permitted")
            if not accepted.exists():
                run_cluster(args, args.output / "acceptance", "acceptance", stop_event, recovery_limit=0)
            run_cluster(args, args.output, "train", stop_event, config["runtime"]["max_recoveries"], start_attempt)
            atomic_json(args.output / "coordinator.status.json", {"state": "completed", "utc": utc()})
        except BaseException as error:
            atomic_json(args.output / "coordinator.status.json", {"state": "blocked", "error": repr(error), "utc": utc()})
            raise


if __name__ == "__main__":
    main()