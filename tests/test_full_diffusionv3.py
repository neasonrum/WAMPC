import ast
import hashlib
import multiprocessing
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.full_diffusionv3.runtime import (
    ControlGroup, PhaseTrace, RemainingBatches, TRACE_BYTES, make_training_loader, read_trace,
)
from method.full_diffusionv3.checkpoint import load_checkpoint, save_checkpoint, select_complete


class EpochDataset(Dataset):
    def __init__(self, length=12):
        self.length = length
        self.shared_epoch = multiprocessing.get_context("spawn").Value("q", 0)

    @property
    def epoch(self):
        return self.shared_epoch.value

    @epoch.setter
    def epoch(self, value):
        self.shared_epoch.value = value

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = int.from_bytes(hashlib.sha256(f"901:{self.epoch}:{index}".encode()).digest()[:4], "big")
        augmentation = random.Random(seed).random() + np.random.default_rng(seed).random()
        return {"index": index, "epoch": self.epoch, "augmentation": augmentation}


class V3RuntimeTests(unittest.TestCase):
    def test_six_point_targets_and_loss_match_automot(self):
        import json
        from method.full_diffusion2.data import frame_pose
        from method.full_diffusionv3.data import future_motion_targets
        from method.full_diffusionv3.diffusion import SpatialTemporalCodec
        reference = ROOT.parent / "AutoMoT/Automot/mot/modeling/automot/automot.py"
        tree = ast.parse(reference.read_text())
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name in ("position_to_offset", "vad_traj_loss")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(reference), "exec"), namespace)
        settings = json.loads((ROOT / "method/full_diffusionv3/config.json").read_text())["labels"]
        current = {"pos_global": [12.0, -4.0], "theta": np.pi / 2, "speed": 5.0}
        expected = np.array([[2.0 * index, 0.1 * index ** 2] for index in range(1, 7)], dtype=np.float32)
        pose = frame_pose(current)
        future = {2 * index: {"pos_global": (pose @ np.append(position, 1.0))[:2], "speed": 5.0}
                  for index, position in enumerate(expected, 1)}
        targets = future_motion_targets(current, future, settings)
        positions = torch.from_numpy(targets["future_positions"])[None]
        offsets = namespace["position_to_offset"](None, positions)
        torch.testing.assert_close(torch.from_numpy(targets["temporal_delta"])[None], offsets)
        torch.testing.assert_close(offsets.cumsum(dim=1), torch.from_numpy(expected)[None])
        prediction = offsets + torch.arange(12, dtype=torch.float32).reshape(1, 6, 2) / 10
        actual = SpatialTemporalCodec.masked_l1(prediction, offsets, torch.ones(1, 6, dtype=torch.bool),
                                                [3, 3, 2, 2, 1, 1])
        torch.testing.assert_close(actual, namespace["vad_traj_loss"](None, prediction, offsets))
        self.assertAlmostEqual(float(SpatialTemporalCodec().target_speed(offsets)[0]),
                               float((positions[:, 1] - positions[:, 0]).norm(dim=-1)[0] * 2), places=5)
        del future[4]
        missing = future_motion_targets(current, future, settings)
        self.assertEqual(missing["temporal_mask"].tolist(), [True, False, False, True, True, True])
        self.assertTrue(np.all(missing["temporal_delta"][1:3] == 0))

    def test_v3_longitudinal_control_matches_automot_and_ignores_legacy_bins(self):
        from types import SimpleNamespace
        from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
        reference = ROOT.parent / "AutoMoT/leaderboard/team_code"
        namespace = {"np": np, "deque": __import__("collections").deque}
        for filename, name in (("automot_utils.py", "PIDController"), ("mot_b2d_agent.py", "control_pid")):
            tree = ast.parse((reference / filename).read_text())
            nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                     and node.name == name]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference / filename), "exec"), namespace)
        make_pid = lambda: namespace["PIDController"](k_p=1.75, k_i=1.0, k_d=2.0, n=20)
        garage_source = ROOT / "third_party/carla_garage/team_code/transfuser_utils.py"
        garage_tree = ast.parse(garage_source.read_text())
        garage_nodes = [node for node in garage_tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "PIDController"]
        garage_namespace = {"np": np, "deque": namespace["deque"]}
        exec(compile(ast.Module(body=garage_nodes, type_ignores=[]), str(garage_source), "exec"), garage_namespace)
        old_speed_mapping = lambda *args: self.fail("V3 must not use the Garage longitudinal mapping")
        net = SimpleNamespace(speed_controller=garage_namespace["PIDController"](k_p=1.75, k_i=1.0, k_d=2.0, n=20),
                              lateral_pid_controller=SimpleNamespace(step=lambda *args: 0.0),
                              control_pid_direct=old_speed_mapping)
        policy = SimpleNamespace(perception=SimpleNamespace(net=net), last_target_speed=None)
        agent = SimpleNamespace(speed_controller=make_pid(), brake_speed=0.4, brake_ratio=1.1,
                                clip_delta=1.0, clip_throttle=1.0, interpolate_waypoints=lambda path: path,
                                turn_controller=SimpleNamespace(step=lambda *args: 0.0))
        route = torch.tensor([[[float(index), 0.0] for index in range(1, 11)]])
        for speed, reference_speed in ((0.0, 0.2), (5.0, 5.0), (5.0, 5.1), (5.25, 5.0),
                                       (0.0, 2.0), (10.0, 12.0), (1.0, 0.0)):
            policy.last_target_speed = reference_speed
            motion = torch.tensor([[[reference_speed * 0.5 * index, 0.0] for index in range(1, 7)]])
            expected = namespace["control_pid"](agent, route, speed, motion)
            actual = FullDiffusionV3Network.control_pid_direct(policy, route[0].numpy(), 0.0,
                                                               torch.tensor([speed]))
            self.assertAlmostEqual(actual[1], float(expected[1]), places=5)
            self.assertEqual(actual[2], bool(expected[2]))
        action = SimpleNamespace(throttle=0.7, brake=False, steer=0.1)
        self.assertIs(FullDiffusionV3Network.apply_speed_limit(action, 9.0), action)
        self.assertEqual(action.throttle, 0.7)
        FullDiffusionV3Network.apply_speed_limit(action, 10.0)
        self.assertEqual((action.throttle, action.brake, action.steer), (0.0, 1.0, 0.1))

    def test_dual_queries_use_64d_slots_and_separate_heads(self):
        from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead
        head = SpatialTemporalFullDiffusionHead(layers=1, dropout=0.0)
        self.assertEqual(head.path_query_encoder[0].in_features, 10 * 64)
        self.assertEqual(head.temporal_query_encoder[0].in_features, 6 * 64)
        self.assertEqual(head.layers[0].task_decoder[-1].out_features, 20)
        self.assertEqual(head.temporal_layers[0].task_decoder[-1].out_features, 12)
        self.assertEqual(head.codec.total_points, 16)
        query_shapes = []
        path_hook = head.layers[0].register_forward_pre_hook(
            lambda module, inputs: query_shapes.append(tuple(inputs[0].shape))
        )
        temporal_hook = head.temporal_layers[0].register_forward_pre_hook(
            lambda module, inputs: query_shapes.append(tuple(inputs[0].shape))
        )
        condition = head.prepare_condition(
            torch.randn(2, 64, 8, 8), torch.randn(2, 5, 256), torch.randn(2, 1, 256),
            torch.randn(2, 2), torch.randn(2, 2, 64, 8, 8),
            torch.eye(3).expand(2, 2, 3, 3).clone(), torch.ones(2, 2, dtype=torch.bool),
            torch.ones(2), torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 6).float(),
        )
        noisy = torch.randn(2, 16, 2, requires_grad=True)
        output = head(noisy, torch.rand(2), condition)
        path_hook.remove()
        temporal_hook.remove()
        self.assertEqual(query_shapes, [(2, 1, 256), (2, 1, 256)])
        self.assertEqual(tuple(output["predictions"].shape), (1, 2, 16, 2))
        spatial_gradient = torch.autograd.grad(output["predictions"][:, :, :10].sum(), noisy,
                            retain_graph=True)[0]
        temporal_gradient = torch.autograd.grad(output["predictions"][:, :, 10:].sum(), noisy,
                             retain_graph=True)[0]
        self.assertEqual(int(torch.count_nonzero(spatial_gradient[:, 10:])), 0)
        self.assertEqual(int(torch.count_nonzero(temporal_gradient[:, :10])), 0)
        loss = head.loss_components(
            output, torch.zeros(2, 16, 2), torch.ones(2, 10, dtype=torch.bool),
            torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool),
        )["loss"]
        loss.backward()
        for parameter in (head.path_query_encoder[0].weight, head.temporal_query_encoder[0].weight,
                          head.layers[0].task_decoder[-1].weight, head.temporal_layers[0].task_decoder[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_carla_external_results_root_keeps_path_boundaries(self):
        import os
        from unittest.mock import patch
        from evaluation.carla_eval_resources import checked_evaluation_path, checked_run_root, evaluation_runtime_root
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary) / 'storage'
            outside = Path(temporary) / 'outside'
            storage.mkdir()
            outside.mkdir()
            (storage / 'escaped').symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, {'WMMPC_CARLA_RESULTS_ROOT': str(storage)}):
                self.assertEqual(checked_run_root(storage / 'v3'), storage / 'v3')
                self.assertEqual(evaluation_runtime_root(), storage / '.runtime')
                self.assertEqual(checked_evaluation_path(storage / 'v3/checkpoint.pt'),
                                 storage / 'v3/checkpoint.pt')
                self.assertEqual(checked_evaluation_path(ROOT / 'leaderboard/data/bench2drive220.xml'),
                                 ROOT / 'leaderboard/data/bench2drive220.xml')
                for target in (outside, storage / '../outside', storage / 'escaped/result.json'):
                    with self.assertRaises(ValueError):
                        checked_run_root(target)
                    with self.assertRaises(ValueError):
                        checked_evaluation_path(target)
            with patch.dict(os.environ):
                os.environ.pop('WMMPC_CARLA_RESULTS_ROOT', None)
                self.assertEqual(checked_run_root(ROOT / 'results/v3'), ROOT / 'results/v3')
                self.assertEqual(evaluation_runtime_root(), ROOT / '.runtime')
                with self.assertRaises(ValueError):
                    checked_run_root(storage / 'v3')

    def test_carla_external_lane_server_keeps_all_writes_in_storage(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.run_full_diffusion2_carla import LaneServer
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            args = SimpleNamespace(job_id='13053006', gpu=0, slot=4400,
                                   carla_root=ROOT / '.runtime/carla_0.9.15_v1')
            with patch.dict(os.environ, {'WMMPC_CARLA_RESULTS_ROOT': str(storage)}), \
                    patch('evaluation.run_full_diffusion2_carla.subprocess.Popen') as launch, \
                    patch('evaluation.run_full_diffusion2_carla.wait_for_server'):
                launch.return_value.pid = 1234
                launch.return_value.poll.return_value = 0
                launch.return_value.returncode = 0
                owner = LaneServer(args, 'GPU-test', storage / 'servers/server_001')
                command = launch.call_args.args[0]
                environment = launch.call_args.kwargs['env']
                for key in ('HOME', 'XDG_CACHE_HOME', 'TMPDIR'):
                    Path(environment[key]).resolve().relative_to(storage)
                    self.assertTrue(Path(environment[key]).is_dir())
                for argument in command:
                    if argument.startswith(('-abslog=', '-userdir=')):
                        Path(argument.split('=', 1)[1]).resolve().relative_to(storage)
                self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'GPU-test')
                self.assertEqual(os.environ['WMMPC_EXTERNAL_CARLA_OWNER'], str(owner.owner))
                owner.close()

    def test_carla_external_batch_keeps_implicit_caches_in_storage(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.run_full_diffusion_carla_fast import batch_trial
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            checkpoint = storage / 'epoch_10.pt'
            checkpoint.write_bytes(b'checkpoint')
            args = SimpleNamespace(job_id='13053006', gpu=0, slot=4400, seed=901,
                                   checkpoint=checkpoint, carla_root=ROOT / '.runtime/carla_0.9.15_v1',
                                   output_root=storage / 'lane')
            args.output_root.mkdir()
            plan = {'command': [sys.executable, 'original_evaluator.py', '--checkpoint', 'unused',
                                '--debug-checkpoint', 'unused', '--routes',
                                str(ROOT / 'leaderboard/data/bench2drive220.xml')],
                    'environment': {}, 'readiness': {'test': True}}
            with patch.dict(os.environ, {'WMMPC_CARLA_RESULTS_ROOT': str(storage)}), \
                    patch('evaluation.run_full_diffusion_carla_fast.build_plan', return_value=plan), \
                    patch('evaluation.run_full_diffusion_carla_fast.subprocess.Popen') as launch:
                launch.return_value.pid = 1234
                launch.return_value.poll.return_value = 0
                launch.return_value.wait.return_value = 0
                outcomes = batch_trial(args, ['1711'], storage / 'attempt_01', 'GPU-test')
                environment = launch.call_args.kwargs['env']
                for key in ('SAVE_PATH', 'TMPDIR', 'HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR',
                            'HF_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                            'WMMPC_CARLA_CACHE_DIR'):
                    Path(environment[key]).resolve().relative_to(storage)
                self.assertEqual(environment['PYTHONDONTWRITEBYTECODE'], '1')
                self.assertEqual(outcomes['1711']['inference_steps'], 100)

    def test_carla_repeat_compatibility_rejects_inference_changes(self):
        from evaluation.prepare_full_diffusion2_carla import inference_source_matches
        original = 'import torch\nclass GarageTemporalDataset:\n    pass\ndef frame_pose(value):\n    return value\n'
        current = 'import gzip\n' + original.replace('    pass', '    def training_cache(self): return 1')
        self.assertTrue(inference_source_matches('method/full_diffusion2/data.py', original, current))
        self.assertFalse(inference_source_matches('method/full_diffusion2/data.py', original,
                                                 current.replace('return value', 'return value + 1')))
        original = 'from pathlib import Path\nBRIDGE_MODULES = Path("original")\ndef predict(value):\n    return value\n'
        current = 'import os\n' + original + 'BRIDGE_MODULES = Path(os.environ.get("WMMPC_BRIDGE_MODULES", str(BRIDGE_MODULES)))\n'
        self.assertTrue(inference_source_matches('method/full_diffusion/model.py', original, current))
        self.assertFalse(inference_source_matches('method/full_diffusion/model.py', original,
                                                 current.replace('return value', 'return value + 1')))

    def test_carla_repeat_fills_missing_route_without_replacing_better_record(self):
        import json
        from evaluation.report_carla_retest import summarize
        def result(route, score, attempt):
            return {'route_id': route, 'attempt': attempt,
                    'record': {'scenario_name': 'test', 'status': 'Completed',
                               'scores': {'score_composed': score, 'score_route': 100},
                               'infractions': {'collisions_layout': [] if score == 100 else ['collision'],
                                               'collisions_vehicle': [], 'collisions_pedestrian': [],
                                               'min_speed_infractions': []}}}
        identity = {'method': 'full_diffusion2', 'checkpoint_sha256': 'same-checkpoint',
                    'seed': 901, 'inference_steps': 100, 'backend': 'direct'}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            baseline = {'1711': result('1711', 50, 'old1711'), '1773': result('1773', 100, 'old1773')}
            new = {'1711': result('1711', 30, 'new1711'), '23687': result('23687', 80, 'new23687')}
            (run/'baseline_reference.json').write_text(json.dumps({'records': baseline, 'identity': identity,
                                                                  'benchmark_route_ids': ['1711', '1773', '23687']}))
            (run/'selection.json').write_text(json.dumps({'route_ids': ['1711', '23687']}))
            (run/'launch_topology.json').write_text(json.dumps(identity))
            state = {'identity': identity.copy(), 'items': {route: {'state': 'done', 'last_result': row}
                                                          for route, row in new.items()}}
            (run/'route_queue.json').write_text(json.dumps(state))
            report = summarize(run)
            self.assertTrue(report['complete_selected_routes'])
            self.assertTrue(report['campaign_finished'])
            self.assertFalse(report['complete_220_routes'])
            self.assertIsNone(report['final_best_of_2_220'])
            self.assertEqual(report['new_coverage_route_ids'], ['23687'])
            self.assertEqual(report['paired_original_observed']['routes'], 1)
            self.assertEqual(report['paired_repeat_observed']['DS'], 30)
            self.assertAlmostEqual(report['best_of_2_220_provisional_until_complete']['DS'], 230/3)
            self.assertAlmostEqual(report['best_of_2_220_provisional_until_complete']['strict_SR_percent'], 100/3)
            self.assertEqual(report['route_comparisons'][0]['chosen_attempt_for_best_DS'], 'old1711')
            state['identity']['checkpoint_sha256'] = 'different-checkpoint'
            (run/'route_queue.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(AssertionError, 'protocol'):
                summarize(run)

    def test_carla_node_launch_selects_only_prepared_routes(self):
        from evaluation.launch_full_diffusion2_carla_node import selected_route_argument
        for routes in (['23687'], ['1711', '23687'], [str(index) for index in range(220)]):
            self.assertEqual(selected_route_argument({'route_ids': routes, 'route_count': len(routes)}),
                             ','.join(routes))
        for topology in ({'route_ids': [], 'route_count': 0},
                         {'route_ids': ['1711', '1711'], 'route_count': 2},
                         {'route_ids': ['1711'], 'route_count': 220},
                         {'route_ids': [str(index) for index in range(221)], 'route_count': 221}):
            with self.assertRaises(ValueError):
                selected_route_argument(topology)

    def test_carla_method_routing_preserves_benchmark_plan(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.bench2drive import build_plan
        from evaluation.run_full_diffusion2_carla import configure_policy
        for method, filename in (("full_diffusion2", "full_diffusion2_fast_agent.py"),
                                 ("full_diffusionv3", "full_diffusionv3_fast_agent.py")):
            args = SimpleNamespace(method=method, experiment="routing_check", seed=901, slot=4400,
                                   gpu_rank=0, routes_subset="1711", agent=None,
                                   agent_config=str(ROOT / "method/full_diffusionv3/config.json"),
                                   carla_root=None, python=sys.executable, resume=False)
            configure_policy(args, method)
            plan = build_plan(args)
            self.assertEqual(plan["method"], method)
            self.assertEqual(plan["route_count"], 1)
            self.assertEqual(plan["seed"], 901)
            self.assertEqual(Path(plan["command"][plan["command"].index("--agent") + 1]).name, filename)
            self.assertTrue(plan["readiness"]["agent"])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'epoch_10.pt'
            checkpoint.write_bytes(b'checkpoint')
            with patch.dict(os.environ, {'WMMPC_CARLA_RESULTS_ROOT': temporary}):
                args.agent_config = str(checkpoint)
                plan = build_plan(args)
                self.assertEqual(plan['command'][plan['command'].index('--agent-config') + 1], str(checkpoint))
        with self.assertRaises(ValueError):
            configure_policy(SimpleNamespace(), "unknown")

    def test_trace_is_bounded_and_records_pre_operation_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = PhaseTrace(directory, 0)
            trace.progress(2, 41)
            with trace.stage("get_batch"):
                record = read_trace(trace.path)
                self.assertEqual(record["phase"], "get_batch")
                self.assertEqual(record["updates"], 41)
            for unused in range(1200):
                trace.event("forward", 0.01)
            record = read_trace(trace.path, limit=1024)
            self.assertEqual(len(record["events"]), 512)
            self.assertEqual(trace.path.stat().st_size, TRACE_BYTES)
            trace.close()

    def test_checkpoint_recovery_rejects_incomplete_and_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = PhaseTrace(directory, 0)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            control = ControlGroup()
            for updates in (1, 2):
                optimizer.zero_grad()
                model(torch.ones(2, 2)).square().mean().backward()
                optimizer.step()
                scheduler.step()
                save_checkpoint(Path(directory) / "latest.pt", model, optimizer, scheduler,
                                {"epoch": 0, "next_batch": updates, "updates": updates}, {}, {}, control, trace)
            latest = select_complete(directory)
            target = Path(latest["absolute_path"])
            target.write_bytes(target.read_bytes() + b"incomplete-write")
            complete = select_complete(directory)
            self.assertEqual(complete["progress"]["updates"], 1)
            control.failed = True
            with self.assertRaisesRegex(RuntimeError, "Refusing checkpoint"):
                save_checkpoint(Path(directory) / "latest.pt", model, optimizer, scheduler,
                                {"epoch": 0, "next_batch": 3, "updates": 3}, {}, {}, control, trace)
            self.assertEqual(select_complete(directory)["progress"]["updates"], 1)
            trace.close()

    def test_persistent_loader_cross_epoch_and_resume_order(self):
        dataset = EpochDataset()
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=901)
        batches = RemainingBatches(BatchSampler(sampler, 3, drop_last=True))
        loader = make_training_loader(dataset, batches, workers=2, seed=901)
        torch.manual_seed(72)
        state = torch.get_rng_state().clone()
        first = list(loader)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        workers = [worker.pid for worker in loader._iterator._workers]
        dataset.epoch = 1
        sampler.set_epoch(1)
        second = list(loader)
        self.assertEqual(workers, [worker.pid for worker in loader._iterator._workers])
        self.assertTrue(all(bool((batch["epoch"] == 1).all()) for batch in second))
        batches.offset = 2
        resumed = list(loader)
        for actual, expected in zip(resumed, second[2:]):
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(before["index"], after["index"]) for before, after in zip(first, second)))
        loader._iterator._shutdown_workers()

    def test_control_group_uses_cpu_state(self):
        control = ControlGroup()
        self.assertTrue(control.any(True))
        self.assertEqual(control.gather({"failed": False}), [{"failed": False}])
        value = torch.tensor([1.0, 3.0])
        self.assertEqual(control.reduce(value).device.type, "cpu")

    def test_supervised_launch_uses_one_global_attempt(self):
        import json
        from types import SimpleNamespace
        from method.full_diffusionv3.supervisor import AUTHORIZED, node_command
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"master_address": "rng-dl01-w005", "master_port": 29693,
                                          "training_python": "/required/venv/bin/python"}))
            args = SimpleNamespace(config=config, output=Path(directory), attempt=2, phase="train")
            commands = [node_command(args, node, "/same/complete.pt") for node in range(len(AUTHORIZED))]
        for command in commands:
            self.assertEqual(command[0], "/required/venv/bin/python")
            self.assertIn("--max_restarts=0", command)
            self.assertEqual(command[command.index("--attempt") + 1], "2")
            self.assertEqual(command[command.index("--resume-checkpoint") + 1], "/same/complete.pt")

    def test_supervised_launch_rejects_wrong_owner(self):
        import os
        from unittest.mock import patch
        from method.full_diffusionv3 import supervisor
        record = {"uid": os.getuid() + 1, "cgroup": "job.13053006.x", "start_ticks": 123,
                  "state": "R", "argv": ["train.py"]}
        with patch.object(supervisor, "process_info", return_value=record):
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                supervisor.verify_owned(10, "13053006", ["train.py"], 123)

    def test_supervised_attempt_resets_only_owned_local_trace_members(self):
        from method.full_diffusionv3.supervisor import reset_local_traces
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "abort.request").touch()
            trace = PhaseTrace(path, 0)
            trace.close()
            reset_local_traces(path)
            self.assertEqual(list(path.iterdir()), [])
            (path / "unrelated.txt").touch()
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                reset_local_traces(path)
            self.assertTrue((path / "unrelated.txt").exists())

    def test_global_recovery_requires_every_original_node_idle(self):
        from method.full_diffusionv3.coordinator import AUTHORIZED, all_nodes_verified
        reports = [{"job": job, "exited_and_idle": True, "controller_lock_free": True,
                    "live_training_processes": [], "allocation": {"available": True}} for job in AUTHORIZED]
        self.assertTrue(all_nodes_verified(reports))
        self.assertFalse(all_nodes_verified(reports[:-1]))
        reports[-1]["controller_lock_free"] = False
        self.assertFalse(all_nodes_verified(reports))

    def test_explicit_global_resume_preserves_attempts_and_bounds_retries(self):
        import json
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch
        from method.full_diffusionv3 import coordinator
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for attempt in range(4):
                previous = output / "attempts" / f"attempt_{attempt:03}"
                previous.mkdir(parents=True)
                (previous / "start.json").write_text("preserved")
            with self.assertRaisesRegex(RuntimeError, "explicit --resume"):
                coordinator.first_attempt(output, resume=False)
            start_attempt = coordinator.first_attempt(output, resume=True)
            self.assertEqual(start_attempt, 4)
            config = output / "config.json"
            config.write_text(json.dumps({"runtime": {}, "seed": 901}))
            args = SimpleNamespace(config=config, output=output)
            started = []

            def verified_cluster(args, output, attempt, phase, stop=False):
                (output / "attempts" / f"attempt_{attempt:03}").mkdir(parents=True, exist_ok=True)

            def fail_node(args, output, attempt, phase, action, job):
                self.assertEqual(action, "start")
                started.append((attempt, job))
                node = list(coordinator.AUTHORIZED).index(job)
                coordinator.atomic_json(output / "attempts" / f"attempt_{attempt:03}" / f"node{node}.status.json",
                                        {"state": "failed", "job": job})

            with patch.object(coordinator, "verify_cluster", side_effect=verified_cluster) as verify, \
                    patch.object(coordinator, "checkpoint_for_restart", return_value=None), \
                    patch.object(coordinator, "node_request", side_effect=fail_node):
                with self.assertRaisesRegex(RuntimeError, "Finite recovery stopped at attempt 7"):
                    coordinator.run_cluster(args, output, "train", threading.Event(), 3, start_attempt)
            self.assertEqual(started, [(attempt, job) for attempt in range(4, 8) for job in coordinator.AUTHORIZED])
            self.assertEqual(verify.call_count, 8)
            preserved = json.loads((output / "attempts/attempt_007/failure_preserved.json").read_text())
            self.assertFalse(preserved["next_attempt_allowed"])
            for attempt in range(4):
                self.assertEqual((output / "attempts" / f"attempt_{attempt:03}/start.json").read_text(), "preserved")

    def test_resumed_worker_environment_imports_frozen_navigation(self):
        import json
        import os
        import subprocess
        from types import SimpleNamespace
        from method.full_diffusionv3.coordinator import freeze_sources
        from method.full_diffusionv3.supervisor import node_command, node_environment, training_root
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = freeze_sources(output, ROOT / "method/full_diffusionv3/config.json")
            config = json.loads(path.read_text())
            source = training_root(config)
            environment = node_environment(config, {"cuda_visible_devices": ""}, output)
            self.assertEqual(environment["PYTHONPATH"].split(os.pathsep),
                             [str(source), str(source / "third_party/carla_navigation_0_9_15")])
            args = SimpleNamespace(config=path, output=output, attempt=4, phase="train")
            self.assertIn(str(source / "method/full_diffusionv3/train.py"), node_command(args, 0, "/complete.pt"))
            probe = "import sys; from method.full_diffusion.garage import load_config; load_config(sys.argv[1]); from model import LidarCenterNet; from agents.navigation.global_route_planner import GlobalRoutePlanner"
            subprocess.run([config["training_python"], "-c", probe, config["backbone_directory"]],
                           cwd=source, env=environment, check=True, capture_output=True, text=True, timeout=120)

    def test_frozen_source_contains_navigation_dependency(self):
        import json
        from method.full_diffusionv3.coordinator import freeze_sources
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = freeze_sources(output, ROOT / "method/full_diffusionv3/config.json")
            navigation = output / "source_snapshot/third_party/carla_navigation_0_9_15/agents/navigation"
            self.assertTrue(navigation.is_dir())
            self.assertTrue((navigation / "local_planner.py").is_file())
            frozen = json.loads(config.read_text())
            self.assertTrue(Path(frozen["normalization_file"]).is_absolute())
            self.assertIn("ENV/wmmpc-full-diffusion/bin/python", frozen["training_python"])
            sources = json.loads(Path(frozen["source_identity_file"]).read_text())
            self.assertIn(str((navigation / "local_planner.py").resolve()), sources)
            self.assertTrue((Path(frozen["bridge_modules"]) / "blocks.py").is_file())

    def test_formal_v3_update_learns_temporal_gate_and_behavior(self):
        from method.full_diffusionv3.diffusion import SpatialTemporalVPFullDiffusion
        from method.full_diffusionv3.model import SpatialTemporalFullDiffusionHead
        from method.full_diffusionv3.train import update
        torch.set_num_threads(2)
        torch.manual_seed(901)
        head = SpatialTemporalFullDiffusionHead(bev_channels=8, d_ffn=64, layers=1, dropout=0.1)
        policy = torch.nn.DataParallel(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003)
        diffusion = SpatialTemporalVPFullDiffusion(head.codec)

        class Perception:
            def __call__(self, batch):
                return torch.randn(2, 8, 8, 8), torch.randn(2, 5, 256), torch.randn(2, 11, 256)

            def encode_history(self, batch, chunk):
                return torch.randn(2, 2, 8, 8, 8)

        batch = {"path": torch.randn(2, 10, 2), "temporal_delta": torch.randn(2, 6, 2),
                 "spatial_mask": torch.ones(2, 10, dtype=torch.bool), "temporal_mask": torch.ones(2, 6, dtype=torch.bool),
                 "behavior_targets": torch.tensor([[0, 1, 2], [2, 3, 1]]), "behavior_mask": torch.ones(2, 3, dtype=torch.bool),
                 "target_point": torch.randn(2, 2), "speed": torch.tensor([2.0, 4.0]),
                 "command": torch.nn.functional.one_hot(torch.tensor([0, 3]), 6).float(),
                 "history_mask": torch.ones(2, 2), "history_transforms": torch.eye(3).expand(2, 2, 3, 3)}
        config = {"precision": "fp32", "history_encode_chunk": 8, "gradient_clip": 1.0,
                  "loss": {"spatial_weight": 1.0, "temporal_weight": 2.0, "behavior_weight": 0.4,
                           "temporal_interval_weights": [3, 3, 2, 2, 1, 1]}}
        before = {name: value.detach().clone() for name, value in head.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            trace = PhaseTrace(directory, 0)
            for unused in range(3):
                losses, norm = update(batch, Perception(), policy, optimizer, diffusion, config, ControlGroup(), trace)
                self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
                self.assertTrue(bool(torch.isfinite(norm)))
            trace.close()
        for prefix in ("temporal.gate", "behavior_head"):
            changed = [not torch.equal(value, before[name]) for name, value in head.named_parameters() if name.startswith(prefix)]
            self.assertTrue(changed and any(changed), prefix)

    def test_true_mid_epoch_resume_preserves_subsequent_updates(self):
        def setup():
            torch.manual_seed(901)
            np.random.seed(901)
            random.seed(901)
            model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(),
                                        torch.nn.Dropout(0.15), torch.nn.Linear(8, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            dataset = EpochDataset()
            sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=901)
            batches = RemainingBatches(BatchSampler(sampler, 3, drop_last=True))
            loader = make_training_loader(dataset, batches, workers=2, seed=901)
            return model, optimizer, scheduler, dataset, sampler, batches, loader

        def step(model, optimizer, scheduler, batch, epoch, batch_id):
            features = torch.stack((batch["index"].float() / 12, batch["augmentation"].float()), dim=-1)
            optimizer.zero_grad(set_to_none=True)
            noise = torch.rand_like(features)
            loss = model(features + noise).square().mean() * (np.random.random() + random.random() + 1)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + (batch_id + 1) / 4)
            return {"batch": batch, "parameters": {name: value.detach().clone() for name, value in model.named_parameters()},
                    "optimizer": {index: {name: value.clone() for name, value in state.items()}
                                  for index, state in optimizer.state_dict()["state"].items()},
                    "lr": optimizer.param_groups[0]["lr"]}

        with tempfile.TemporaryDirectory() as directory:
            trace = PhaseTrace(directory, 0)
            path = Path(directory) / "latest.pt"
            objects = setup()
            model, optimizer, scheduler, dataset, sampler, batches, loader = objects
            expected = []
            for epoch in range(2):
                dataset.epoch = epoch
                sampler.set_epoch(epoch)
                for batch_id, batch in enumerate(loader):
                    record = step(model, optimizer, scheduler, batch, epoch, batch_id)
                    if epoch == 0 and batch_id == 1:
                        save_checkpoint(path, model, optimizer, scheduler,
                                        {"epoch": 0, "next_batch": 2, "updates": 2}, {"seed": 901}, {},
                                        ControlGroup(), trace, loader)
                    elif epoch > 0 or batch_id > 1:
                        expected.append(record)
            loader._iterator._shutdown_workers()
            objects = setup()
            model, optimizer, scheduler, dataset, sampler, batches, loader = objects
            complete = select_complete(directory, {"seed": 901}, {})
            progress = load_checkpoint(complete["absolute_path"], model, optimizer, scheduler, {"seed": 901}, {}, loader)
            actual = []
            for epoch in range(progress["epoch"], 2):
                dataset.epoch = epoch
                sampler.set_epoch(epoch)
                batches.offset = progress["next_batch"] if epoch == progress["epoch"] else 0
                for batch_id, batch in enumerate(loader, start=batches.offset):
                    actual.append(step(model, optimizer, scheduler, batch, epoch, batch_id))
            self.assertEqual(len(actual), len(expected))
            for resumed, uninterrupted in zip(actual, expected):
                self.assertEqual(resumed["lr"], uninterrupted["lr"])
                for group in ("batch", "parameters"):
                    for name in uninterrupted[group]:
                        torch.testing.assert_close(resumed[group][name], uninterrupted[group][name], rtol=0, atol=0)
                for index, state in uninterrupted["optimizer"].items():
                    for name in state:
                        torch.testing.assert_close(resumed["optimizer"][index][name], state[name], rtol=0, atol=0)
            loader._iterator._shutdown_workers()
            trace.close()


if __name__ == "__main__":
    unittest.main()