import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench2drive_launcher", ROOT / "evaluation/bench2drive.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class StructureTests(unittest.TestCase):
    def plan(self, *extra):
        args = LAUNCHER.parse_args(["--method", "WMMPC", "--experiment", "smoke", *extra])
        return LAUNCHER.build_plan(args)

    def test_reserved_directories_exist(self):
        for method in ("WMMPC", "diffusiondrive"):
            self.assertTrue((ROOT / "ckps" / method).is_dir())
            for directory in ("modeling", "inference", "configs"):
                self.assertTrue((ROOT / "method" / method / directory).is_dir())
        for directory in ("data", "train", "preprocess", "results", "tools"):
            self.assertTrue((ROOT / directory).is_dir())

    def test_all_python_files_parse_without_automot_imports(self):
        for path in ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                self.assertFalse(any(name.lower().startswith(("automot", "mot.")) for name in names), str(path))

    def test_routes_and_scenarios_are_complete(self):
        routes = ET.parse(ROOT / "leaderboard/data/bench2drive220.xml").getroot().findall("route")
        route_ids = {int(route.attrib["id"]) for route in routes}
        self.assertEqual((len(routes), len(route_ids)), (220, 220))
        split_ids = []
        for name in ("b2d_all_routes_split1.json", "b2d_all_routes_split2.json"):
            split = json.loads((ROOT / "eval_json" / name).read_text())
            split_ids.extend(split.get("ids") or [int(route["id"]) for route in split.get("routes", [])])
        self.assertEqual(set(split_ids) | {1711, 1773}, route_ids)
        classes = set()
        for path in (ROOT / "scenario_runner/srunner/scenarios").glob("*.py"):
            classes.update(node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        required = {scenario.attrib["type"] for route in routes for scenario in route.findall("scenarios/scenario")}
        self.assertTrue(required)
        self.assertFalse(required - classes, required - classes)

    def test_method_checkpoint_and_worker_isolation(self):
        first = self.plan()
        other_method = LAUNCHER.build_plan(LAUNCHER.parse_args(["--method", "diffusiondrive", "--experiment", "smoke"]))
        other_seed = self.plan("--seed", "1")
        other_slot = self.plan("--slot", "1")
        self.assertEqual(len({plan["output_dir"] for plan in (first, other_method, other_seed, other_slot)}), 4)
        self.assertEqual(len({plan["checkpoint_dir"] for plan in (first, other_method, other_seed)}), 3)
        self.assertFalse(set(first["ports"]) & set(other_slot["ports"]))
        self.assertNotIn("--resume", first["command"])

    def test_route_subset_and_path_validation(self):
        route = ET.parse(ROOT / "leaderboard/data/bench2drive220.xml").getroot().find("route").attrib["id"]
        self.assertEqual(self.plan("--routes-subset", route)["route_count"], 1)
        for extra in (("--routes-subset", "missing"), ("--experiment", "../escape"), ("--slot", "-1")):
            with self.assertRaises(ValueError):
                self.plan(*extra)

    def test_planning_does_not_launch_or_create_output(self):
        with patch.object(LAUNCHER.subprocess, "run") as process:
            plan = self.plan("--experiment", "dry_run_must_not_create_output")
            self.assertFalse(Path(plan["output_dir"]).exists())
            process.assert_not_called()
        self.assertFalse(plan["readiness"]["agent"])

    def test_execution_requires_explicit_job_context(self):
        with self.assertRaises(ValueError):
            LAUNCHER.check_job(None)
        with patch.dict(os.environ, {"LSB_JOBID": "12345"}):
            LAUNCHER.check_job("12345")

    def test_execution_guards_existing_runs_and_resume_identity(self):
        args = LAUNCHER.parse_args(["--method", "WMMPC", "--experiment", "smoke", "--job-id", "12345"])
        with tempfile.TemporaryDirectory(prefix="wmmpc-test-") as temporary:
            plan = LAUNCHER.build_plan(args)
            plan["output_dir"] = str(Path(temporary) / "run")
            plan["readiness"] = {key: True for key in plan["readiness"]}
            with patch.object(LAUNCHER, "check_job"), patch.object(LAUNCHER.socket, "socket"), \
                    patch.object(LAUNCHER.tempfile, "gettempdir", return_value=temporary), \
                    patch.object(LAUNCHER.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as process:
                self.assertEqual(LAUNCHER.execute(plan, args), 0)
                process.assert_called_once()
                with self.assertRaises(ValueError):
                    LAUNCHER.execute(plan, args)
                self.assertEqual(process.call_count, 1)
                args.resume = True
                plan["command"].extend(["--resume", "True"])
                self.assertEqual(LAUNCHER.execute(plan, args), 0)
                self.assertEqual(process.call_count, 2)
                plan["command"].extend(["--routes-subset", "different"])
                with self.assertRaises(ValueError):
                    LAUNCHER.execute(plan, args)
                self.assertEqual(process.call_count, 2)
            self.assertEqual(json.loads((Path(temporary) / "run/exit.json").read_text())["returncode"], 0)

    def test_evaluator_cleanup_is_scoped_to_owned_process(self):
        path = ROOT / "leaderboard/leaderboard/leaderboard_evaluator.py"
        source = path.read_text()
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "stop_server")
        mock_os = SimpleNamespace(killpg=Mock())
        namespace = {"os": mock_os, "signal": signal}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
        namespace["stop_server"](SimpleNamespace(pid=12345))
        mock_os.killpg.assert_called_once_with(12345, signal.SIGKILL)
        mock_os.killpg.side_effect = ProcessLookupError
        namespace["stop_server"](SimpleNamespace(pid=12345))
        self.assertNotIn("ps -ef", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()