import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def component(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid path component: {value!r}")
    return value


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plan or execute an isolated Bench2Drive evaluation.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--gpu-rank", type=int, default=0)
    parser.add_argument("--routes-subset", default="")
    parser.add_argument("--agent")
    parser.add_argument("--agent-config")
    parser.add_argument("--carla-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--job-id")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_plan(args):
    paths = json.loads((ROOT / "configs/paths.json").read_text())
    method = component(args.method)
    experiment = component(args.experiment)
    method_root = ROOT / "method" / method
    if not method_root.is_dir():
        raise ValueError(f"Unknown method directory: {method_root}")
    if args.seed < 0 or args.slot < 0 or args.gpu_rank < 0:
        raise ValueError("Seed, slot and GPU rank must be nonnegative")
    port = 2000 + args.slot * 10
    if port + 3 > 65535:
        raise ValueError("Slot exceeds the available TCP port range")
    routes = resolve_path(paths["routes"])
    available = {route.attrib["id"] for route in ET.parse(routes).getroot().findall("route")}
    selected = args.routes_subset.split(",") if args.routes_subset else []
    if selected and (len(selected) != len(set(selected)) or not set(selected) <= available):
        raise ValueError("Route subset must contain unique, comma-separated IDs from bench2drive220.xml")
    run_suffix = Path(method) / experiment / f"seed_{args.seed}"
    checkpoint = resolve_path(paths["checkpoint_root"]) / run_suffix
    output = resolve_path(paths["results_root"]) / run_suffix / f"worker_{args.slot}"
    agent = resolve_path(args.agent) if args.agent else method_root / "inference/carla_agent.py"
    agent_config = resolve_path(args.agent_config) if args.agent_config else checkpoint
    carla_root = resolve_path(args.carla_root or os.environ.get("CARLA_ROOT") or paths["carla_root"])
    command = [
        args.python, str(ROOT / "leaderboard/leaderboard/leaderboard_evaluator.py"),
        "--routes", str(routes), "--agent", str(agent), "--agent-config", str(agent_config),
        "--checkpoint", str(output / "results.json"),
        "--debug-checkpoint", str(output / "live_results.txt"),
        "--port", str(port), "--traffic-manager-port", str(port + 3),
        "--traffic-manager-seed", str(args.seed), "--gpu-rank", str(args.gpu_rank),
        "--track", "SENSORS", "--repetitions", "1",
    ]
    if args.routes_subset:
        command.extend(["--routes-subset", args.routes_subset])
    if args.resume:
        command.extend(["--resume", "True"])
    python_paths = [ROOT, ROOT / "leaderboard", ROOT / "scenario_runner", method_root,
                    method_root / "inference", carla_root / "PythonAPI", carla_root / "PythonAPI/carla"]
    if os.environ.get("PYTHONPATH"):
        python_paths.append(os.environ["PYTHONPATH"])
    environment = {
        "CARLA_ROOT": str(carla_root),
        "CARLA_SERVER": str(carla_root / "CarlaUE4.sh"),
        "LEADERBOARD_ROOT": str(ROOT / "leaderboard"),
        "SCENARIO_RUNNER_ROOT": str(ROOT / "scenario_runner"),
        "TEAM_CODE_ROOT": str(agent.parent),
        "TEAM_AGENT": str(agent),
        "TEAM_CONFIG": str(agent_config),
        "IS_BENCH2DRIVE": "True",
        "PLANNER_TYPE": "only_traj",
        "SAVE_PATH": str(output / "agent_data"),
        "GPU_RANK": str(args.gpu_rank),
        "PYTHONPATH": os.pathsep.join(map(str, python_paths)),
    }
    readiness = {
        "agent": agent.is_file(),
        "agent_config": agent_config.exists(),
        "carla_server": os.access(carla_root / "CarlaUE4.sh", os.X_OK),
        "carla_navigation": (carla_root / "PythonAPI/carla/agents").is_dir(),
    }
    return {
        "method": method, "experiment": experiment, "seed": args.seed, "slot": args.slot,
        "route_count": len(selected) if selected else len(available),
        "checkpoint_dir": str(checkpoint), "output_dir": str(output),
        "ports": list(range(port, port + 4)), "command": command,
        "environment": environment, "readiness": readiness,
    }


def check_job(job_id):
    if not job_id or not job_id.isdigit():
        raise ValueError("Execution requires an explicitly authorized --job-id inside that LSF job")
    if os.environ.get("LSB_JOBID") == job_id:
        return
    cgroup = Path("/proc/self/cgroup").read_text()
    if not re.search(rf"(?<!\d){re.escape(job_id)}(?!\d)", cgroup):
        raise ValueError("Current process is not inside the designated LSF job; use battach first")


def execute(plan, args):
    check_job(args.job_id)
    missing = [name for name, ready in plan["readiness"].items() if not ready]
    if missing:
        raise ValueError(f"Execution prerequisites missing: {', '.join(missing)}")
    output = Path(plan["output_dir"])
    lock_root = Path(tempfile.gettempdir()) / f"wmmpc-bench2drive-{os.getuid()}"
    lock_root.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for port in plan["ports"]:
            lock = stack.enter_context((lock_root / f"port_{port}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", port))
        output.mkdir(parents=True, exist_ok=True)
        run_lock = stack.enter_context((output / ".run.lock").open("a"))
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "plan.json").exists() and not args.resume:
            raise ValueError("Run already exists; use a new experiment/seed/slot or explicitly --resume")
        if args.resume and (output / "plan.json").exists():
            previous = json.loads((output / "plan.json").read_text())
            current_command = [value for value in plan["command"] if value not in ("--resume", "True")]
            previous_command = [value for value in previous["command"] if value not in ("--resume", "True")]
            if previous_command != current_command or previous["environment"] != plan["environment"]:
                raise ValueError("Resume parameters differ from the saved run plan")
        (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        environment = os.environ.copy()
        environment.update(plan["environment"])
        with (output / "evaluation.log").open("a") as log:
            completed = subprocess.run(plan["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
        (output / "exit.json").write_text(json.dumps({"returncode": completed.returncode}) + "\n")
        return completed.returncode


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        print(json.dumps(plan, indent=2))
        return execute(plan, args) if args.execute else 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())