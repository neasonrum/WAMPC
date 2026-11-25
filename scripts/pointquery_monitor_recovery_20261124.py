"""Resume failed point-query evaluator workers without changing planner/queue identity.

The original evaluator source remains untouched. Only the exact periodic NVML
monitor query tolerates timeout; CPU/RAM guards still read live cgroup data.
"""
import ast
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROLLER = ROOT / "method/full_diffusion_pointquery/eval_controller.py"
WRAPPER = Path(__file__).resolve()
QUERY = ("nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader")
REAL_RUN, REAL_POPEN = subprocess.run, subprocess.Popen
STATE = {"stdout": "", "last_success_unix": None, "last_error": None, "fallbacks": 0}


def event(kind, **fields):
    directory = Path(os.environ["WMMPC_MONITOR_RECOVERY_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"monitor_{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps({"kind": kind, "unix": time.time(), "pid": os.getpid(), **fields}) + "\n")


def bounded_run(command, *args, **kwargs):
    if not isinstance(command, (list, tuple)) or tuple(map(str, command)) != QUERY:
        return REAL_RUN(command, *args, **kwargs)
    kwargs["timeout"] = min(float(kwargs.get("timeout", 15)), 3.)
    try:
        result = REAL_RUN(command, *args, **kwargs)
        if result.returncode == 0 and result.stdout.strip():
            STATE.update(stdout=result.stdout, last_success_unix=time.time(), last_error=None)
            return result
        raise RuntimeError(f"NVML monitor returned code {result.returncode}")
    except (subprocess.TimeoutExpired, OSError, RuntimeError) as error:
        STATE["last_error"] = repr(error)
        STATE["fallbacks"] += 1
        age = None if STATE["last_success_unix"] is None else time.time() - STATE["last_success_unix"]
        # Stop presenting an old memory measurement as current after a minute.
        cached = STATE["stdout"] if age is not None and age <= 60 else ""
        event("nvml_monitor_unavailable", error=repr(error), last_success_age_seconds=age,
              cached_snapshot_used=bool(cached), driving_evaluation_continues=True)
        return subprocess.CompletedProcess(command, 0, stdout=cached, stderr=repr(error))


def inherited_popen(command, *args, **kwargs):
    # The original parent starts each lane with its own source filename. Route
    # that child through this same monitor wrapper; keep all lane CLI settings.
    if isinstance(command, (list, tuple)):
        cmd = list(command)
        if (len(cmd) > 3 and cmd[0] == sys.executable and cmd[1] == "-u"
                and Path(cmd[2]).resolve() == CONTROLLER and "--role" in cmd):
            command = [cmd[0], "-u", str(WRAPPER), *cmd[3:]]
    return REAL_POPEN(command, *args, **kwargs)


def install():
    import evaluation.carla_eval_resources as resource_module
    original_resources = resource_module.resources

    def live_resources():
        result = original_resources()
        result.update(gpu_monitor_fallback=STATE["last_error"] is not None,
                      gpu_monitor_last_success_unix=STATE["last_success_unix"],
                      gpu_monitor_error=STATE["last_error"], gpu_monitor_fallback_count=STATE["fallbacks"])
        return result

    subprocess.run = bounded_run
    subprocess.Popen = inherited_popen
    resource_module.resources = live_resources
    return resource_module


def self_test():
    global REAL_RUN
    resource_module = install()
    actual = REAL_RUN

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    STATE.update(stdout="0, test-gpu, 100 MiB, 10 %\n", last_success_unix=time.time())
    REAL_RUN = timeout
    result = bounded_run(list(QUERY), capture_output=True, text=True, timeout=15)
    assert result.stdout == STATE["stdout"] and STATE["fallbacks"] == 1
    STATE["last_success_unix"] = time.time() - 120
    assert bounded_run(list(QUERY), capture_output=True, text=True, timeout=15).stdout == ""
    REAL_RUN = actual
    snapshot = resource_module.resources()
    assert int(snapshot["job_memory.max"]) == 268435456000
    assert int(snapshot["job_memory.noncache"]) > 0 and not snapshot["gpu_monitor_fallback"]
    print(json.dumps({"status": "PASS", "NVML_timeout_does_not_abort": True,
                      "stale_GPU_snapshot_is_marked": True, "live_cgroup_RAM_guards": True}), flush=True)


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    install()
    module = importlib.import_module("method.full_diffusion_pointquery.eval_controller")

    def stop_owned(process):
        if process.poll() is not None:
            return
        # A lane closes its owned CARLA server on SIGTERM. Give that cleanup
        # time to complete instead of orphaning the renderer after five seconds.
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    module.stop_owned = stop_owned
    event("recovery_monitor_installed", controller=str(CONTROLLER), argv=sys.argv[1:])
    # Reuse the original, verified CLI verbatim and its unmodified parent/lane
    # implementations. Only the resource monitor and owned shutdown are wrapped.
    tree = ast.parse(CONTROLLER.read_text(), filename=str(CONTROLLER))
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                 and node.test.left.id == "__name__")
    cli = ast.Module(body=guard.body, type_ignores=[])
    exec(compile(cli, str(CONTROLLER), "exec"), module.__dict__)


if __name__ == "__main__":
    main()

