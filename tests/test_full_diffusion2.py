import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.full_diffusion.diffusion import VPFullDiffusion
from method.full_diffusion.model import TrajectoryFullDiffuGeoHead
from method.full_diffusion2.model import TemporalBEVAdapter, TemporalFullDiffusionHead, align_bev


class FullDiffusion2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(901)

    def condition_inputs(self):
        return (torch.randn(2, 8, 8, 8), torch.randn(2, 5, 256),
                torch.randn(2, 11, 256), torch.randn(2, 2),
                torch.randn(2, 2, 8, 8, 8), torch.eye(3).expand(2, 2, 3, 3).clone(),
                torch.ones(2, 2), torch.tensor([2.0, 8.0]),
                functional_one_hot(torch.tensor([0, 3]), 6))

    def test_alignment_identity_and_translation_axes(self):
        feature = torch.zeros(1, 1, 8, 8)
        feature[0, 0, 3, 3] = 1
        transform = torch.eye(3)[None]
        torch.testing.assert_close(align_bev(feature, transform, extent=4), feature)
        transform[:, 0, 2] = 1
        expected = torch.zeros_like(feature)
        expected[0, 0, 3, 4] = 1
        torch.testing.assert_close(align_bev(feature, transform, extent=4), expected)
        transform[:, 0, 2] = 0
        transform[:, 1, 2] = 1
        expected.zero_()
        expected[0, 0, 4, 3] = 1
        torch.testing.assert_close(align_bev(feature, transform, extent=4), expected)

    def test_zero_residual_initialization_preserves_original_head(self):
        head = TemporalFullDiffusionHead(bev_channels=8, d_ffn=64, layers=1, dropout=0).eval()
        baseline = TrajectoryFullDiffuGeoHead(bev_channels=8, d_ffn=64, layers=1, dropout=0).eval()
        state = head.state_dict()
        baseline.load_state_dict({name: state[name] for name in baseline.state_dict()})
        inputs = self.condition_inputs()
        original_bev = inputs[0].clone()
        clean = head.codec.encode(torch.randn(2, 10, 2), torch.tensor([1.0, 4.0]))
        noisy, time = VPFullDiffusion(head.codec).corrupt(clean)
        expected = baseline(noisy, time, baseline.prepare_condition(*inputs[:4]))
        actual = head(noisy, time, head.prepare_condition(*inputs))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(inputs[0], original_bev)

    def test_missing_history_is_masked(self):
        adapter = TemporalBEVAdapter(channels=8)
        torch.nn.init.normal_(adapter.output_projection.weight, std=0.1)
        sequence = torch.randn(2, 3, 8, 8, 8)
        transforms = torch.eye(3).expand(2, 3, 3, 3)
        valid = torch.tensor([[0, 0, 1], [0, 0, 1]])
        state = torch.randn(2, 9)
        expected = adapter(sequence, transforms, valid, state)
        changed = sequence.clone()
        changed[:, :2] = torch.randn_like(changed[:, :2]) * 100
        torch.testing.assert_close(adapter(changed, transforms, valid, state), expected)

    def test_history_state_and_all_parameter_gradients(self):
        head = TemporalFullDiffusionHead(bev_channels=8, d_ffn=64, layers=1, dropout=0)
        torch.nn.init.normal_(head.temporal.output_projection.weight, std=0.02)
        torch.nn.init.normal_(head.query_condition.weight, std=0.02)
        inputs = self.condition_inputs()
        clean = head.codec.encode(torch.randn(2, 10, 2), torch.tensor([1.0, 4.0]))
        noisy, time = VPFullDiffusion(head.codec).corrupt(clean)
        predictions = head(noisy, time, inputs)
        changed_history = list(inputs)
        changed_history[4] = torch.randn_like(inputs[4])
        self.assertGreater(float((head(noisy, time, tuple(changed_history)) - predictions).abs().max()), 1e-6)
        changed_state = list(inputs)
        changed_state[7] = inputs[7] + 10
        self.assertGreater(float((head(noisy, time, tuple(changed_state)) - predictions).abs().max()), 1e-6)
        head.loss(predictions, clean).backward()
        self.assertEqual([name for name, parameter in head.named_parameters() if parameter.grad is None], [])
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in head.parameters()))
        self.assertGreater(float(head.temporal.fusion[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.temporal.gate[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.state_encoder[0].weight.grad.abs().sum()), 0)

    def test_augmented_pose_transform_matches_world_geometry(self):
        import numpy as np
        from method.full_diffusion.garage import load_config, make_preprocessor
        from method.full_diffusion2.data import frame_pose
        processor = make_preprocessor(load_config(), training=False)
        source = {"theta": 0.7, "pos_global": [3.0, 5.0],
                  "augmentation_rotation": 12.0, "augmentation_translation": 0.8}
        current = {"theta": 1.2, "pos_global": [4.0, 6.0],
                   "augmentation_rotation": -5.0, "augmentation_translation": -0.3}
        cloud = np.array([[5.0, 2.0, 0.0], [8.0, -1.0, 0.0]])
        source_augmented = processor.align(cloud, source, source, 0.8, 12.0)
        source_rotation = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
        current_rotation = np.array([[np.cos(1.2), -np.sin(1.2)], [np.sin(1.2), np.cos(1.2)]])
        world_points = cloud[:, :2] @ source_rotation.T + source["pos_global"]
        current_points = (world_points - current["pos_global"]) @ current_rotation
        expected = processor.augment_route(current_points, -0.3, -5.0)
        transform = np.linalg.solve(frame_pose(current, True), frame_pose(source, True))
        homogeneous = np.column_stack((source_augmented[:, :2], np.ones(len(cloud))))
        np.testing.assert_allclose((homogeneous @ transform.T)[:, :2], expected, atol=1e-6)

    def test_real_history_window_is_causal_and_deterministic(self):
        from method.full_diffusion2.data import GarageTemporalDataset
        index = ROOT / "data/full_diffusion/pdm_lite_seed901"
        if not (index / "manifest.json").exists():
            self.skipTest("Production Garage index is not installed")
        dataset = GarageTemporalDataset(index, split="train")
        first, second = dataset[0], dataset[0]
        self.assertEqual(first["history_rgb"].shape[0], 2)
        self.assertEqual(first["history_lidar"].shape, (2, 1, 256, 256))
        self.assertEqual(first["history_transforms"].shape, (2, 3, 3))
        self.assertTrue(bool((first["history_frame_ids"] < first["frame_id"]).all()))
        torch.testing.assert_close(first["history_frame_ids"],
                                   torch.tensor([first["frame_id"] - 2, first["frame_id"] - 1]))
        for key in ("rgb", "lidar", "path", "history_rgb", "history_lidar", "history_transforms"):
            torch.testing.assert_close(first[key], second[key])

    def test_job_guard_requires_both_exact_host_and_cgroup(self):
        from unittest.mock import patch
        from method.full_diffusion2 import launch
        with patch.object(launch.socket, "gethostname", return_value="rng-dl01-w005"), \
             patch.object(Path, "read_text", return_value="0::/lsf/13053006"):
            self.assertEqual(launch.guard_context(), "13053006")
            with self.assertRaises(RuntimeError):
                launch.guard_context("13047898")
        with patch.object(launch.socket, "gethostname", return_value="rng-dl01-w005"), \
             patch.object(Path, "read_text", return_value="0::/lsf/99999999"):
            with self.assertRaises(RuntimeError):
                launch.guard_context("13053006")

    def test_busy_gpu_preflight_refuses_without_stopping_workloads(self):
        from unittest.mock import patch
        from method.full_diffusion2 import launch
        with patch.object(launch, "inspect_allocation", return_value={"available": False}), \
             patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                launch.gpu_preflight("13053006", ROOT)
            run.assert_not_called()

    def test_detached_launch_is_durable_and_uses_absolute_paths(self):
        import tempfile
        from unittest.mock import patch
        from method.full_diffusion2 import launch
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["launch.py", "--job-id", "13053006", "--phase", "train",
                         "--output", directory, "--detach", "--resume",
                         "--workers-per-rank", "4", "--trial-updates", "300"]
            with patch.object(sys, "argv", arguments), \
                 patch.object(launch, "guard_context", return_value="13053006"), \
                 patch.object(launch.subprocess, "Popen") as process:
                process.return_value.pid = 123
                launch.main()
            command = process.call_args.args[0]
            self.assertTrue(process.call_args.kwargs["start_new_session"])
            self.assertTrue(process.call_args.kwargs["close_fds"])
            self.assertTrue(Path(command[command.index("--config") + 1]).is_absolute())
            self.assertTrue(Path(command[command.index("--output") + 1]).is_absolute())
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--workers-per-rank") + 1], "4")
            self.assertEqual(command[command.index("--trial-updates") + 1], "300")

    def test_master_port_probe_accepts_time_wait_but_rejects_listener(self):
        import socket
        from method.full_diffusion2 import launch
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            address, port = listener.getsockname()
            listener.listen(1)
            with self.assertRaises(OSError):
                launch.check_master_port(address, port)
            with socket.create_connection((address, port)) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b"")
        launch.check_master_port(address, port)

    def test_worker_override_preserves_checkpoint_training_config(self):
        from contextlib import redirect_stdout
        import io
        import json
        from unittest.mock import patch
        from method.full_diffusion2 import train
        path = ROOT / "method/full_diffusion2/config.json"
        original = path.read_text()
        captured = io.StringIO()
        arguments = ["train.py", "--output", "/tmp/full_diffusion2_config_check",
                     "--check-config", "--workers-per-rank", "4", "--trial-updates", "300"]
        with patch.object(sys, "argv", arguments), redirect_stdout(captured):
            train.main()
        result = json.loads(captured.getvalue())
        self.assertEqual(result["workers_per_rank"], 4)
        self.assertEqual(result["checkpoint_config_workers_per_rank"], 2)
        self.assertEqual(result["trial_updates"], 300)
        self.assertEqual(result["global_batch"], 64)
        self.assertEqual(path.read_text(), original)

    def test_two_and_four_workers_produce_identical_training_batches(self):
        import os
        if os.environ.get("WMMPC_WORKER_PARITY_TEST") != "1":
            self.skipTest("Enable explicitly for the production multi-process data parity check")
        from torch.utils.data import DataLoader, Subset
        from method.full_diffusion2.data import GarageTemporalDataset
        dataset = GarageTemporalDataset(ROOT / "data/full_diffusion/pdm_lite_seed901")
        dataset.epoch = 3
        subset = Subset(dataset, [0, 101, 501, 1001, 2001, 3001, 4001, 5001])
        batches = []
        for workers in (2, 4):
            loader = DataLoader(subset, batch_size=2, num_workers=workers,
                                multiprocessing_context="spawn", prefetch_factor=2)
            batches.append(list(loader))
        self.assertEqual(len(batches[0]), len(batches[1]))
        for expected, actual in zip(*batches):
            self.assertEqual(expected.keys(), actual.keys())
            for key, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
                else:
                    self.assertEqual(actual[key], value)

    def test_online_history_is_causal_aligned_and_episode_local(self):
        from method.full_diffusion2.inference.policy import TemporalBEVHistory
        history = TemporalBEVHistory()
        feature = torch.ones(1, 8, 8, 8)
        history.set_observation([0, 0, 0], 0)
        _, _, mask = history.get_inputs(feature)
        self.assertFalse(bool(mask.any()))
        history.set_observation([1, 0, 0], 0.25)
        previous, transforms, mask = history.get_inputs(feature * 2)
        torch.testing.assert_close(mask, torch.tensor([[False, True]]))
        self.assertEqual(float(transforms[0, 1, 0, 2]), -1)
        history.set_observation([2, 0, 0], 0.5)
        previous, transforms, mask = history.get_inputs(feature * 3)
        self.assertTrue(bool(mask.all()))
        torch.testing.assert_close(previous[:, 0], feature)
        torch.testing.assert_close(previous[:, 1], feature * 2)
        torch.testing.assert_close(transforms[0, :, 0, 2], torch.tensor([-2.0, -1.0]))
        history.set_observation([0, 0, 0], 0)
        _, _, mask = history.get_inputs(feature)
        self.assertFalse(bool(mask.any()))
        self.assertEqual(len(history.entries), 1)

    def test_online_history_never_uses_a_later_or_stale_observation(self):
        from method.full_diffusion2.inference.policy import TemporalBEVHistory
        history = TemporalBEVHistory()
        feature = torch.ones(1, 8, 8, 8)
        history.set_observation([0, 0, 0], 0.1)
        history.get_inputs(feature)
        history.set_observation([0, 0, 0], 0.3)
        _, _, mask = history.get_inputs(feature)
        self.assertFalse(bool(mask.any()))
        history.set_observation([0, 0, 0], 2.0)
        _, _, mask = history.get_inputs(feature)
        self.assertFalse(bool(mask.any()))

    def test_checkpointed_policy_preserves_raw_detection_interface(self):
        import tempfile
        from types import SimpleNamespace
        from method.full_diffusion2.inference.policy import FullDiffusion2Network

        class GarageStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(detect_boxes=True, target_speeds=[0.0, 5.0, 10.0], tp_attention=False)
                self.extra_sensors = False
                self.change_channel = torch.nn.Identity()
                self.checkpoint_query = torch.nn.Parameter(torch.randn(1, 11, 256))

            def backbone(self, rgb, lidar):
                return torch.ones(len(rgb), 8, 8, 8), torch.ones(len(rgb), 256, 2, 2), None

            def encoder_pos_encoding(self, feature):
                return torch.zeros_like(feature)

            def join(self, query, memory):
                return query + memory.mean(1, keepdim=True)

            def head(self, bev):
                return bev.clone()

        config = {"method": "full_diffusion2", "seed": 901, "observation_interval_seconds": 0.25,
                  "model": {"bev_channels": 8, "d_ffn": 64, "layers": 1, "dropout": 0,
                            "history_frames": 2},
                  "diffusion": {"train_steps": 1000, "beta_min": 0.1, "beta_d": 19.9,
                                "inference_steps": 2}}
        head = TemporalFullDiffusionHead(**config["model"])
        torch.nn.init.normal_(head.temporal.output_projection.weight, std=0.1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            torch.save({"config": config, "model": head.state_dict()}, checkpoint)
            policy = FullDiffusion2Network(GarageStub(), checkpoint, "cpu")
            for timestamp in (0.0, 0.25, 0.5):
                policy.set_observation([timestamp, 0, 0], timestamp)
                result = policy(torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8),
                                torch.zeros(1, 2), torch.ones(1, 1), torch.tensor([[1., 0, 0, 0, 0, 0]]))
                self.assertEqual(len(result), 10)
                self.assertEqual(result[1].shape, (1, 3))
                self.assertEqual(result[2].shape, (1, 10, 2))
                self.assertTrue(bool(torch.isfinite(result[2]).all()))
                torch.testing.assert_close(result[6], torch.ones(1, 8, 8, 8))
            self.assertTrue(all(not parameter.requires_grad for parameter in policy.perception.parameters()))


def functional_one_hot(indices, classes):
    return torch.nn.functional.one_hot(indices, classes).float()


if __name__ == "__main__":
    unittest.main()