import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.full_diffusion.diffusion import GeoCodec, VPFullDiffusion


class FullDiffusionTests(unittest.TestCase):
    def test_geo_speed_roundtrip(self):
        codec = GeoCodec()
        path = torch.randn(3, 10, 2) * 10
        speed = torch.tensor([0.0, 4.0, 13.8889])
        encoded = codec.encode(path, speed)
        decoded_path, decoded_speed = codec.decode(encoded)
        torch.testing.assert_close(decoded_path, path)
        torch.testing.assert_close(decoded_speed, speed)
        self.assertEqual(encoded.shape, (3, 11, 2))
        self.assertEqual(encoded[:, -1, 1].count_nonzero().item(), 0)

    def test_vp_matches_continuous_schedule(self):
        diffusion = VPFullDiffusion()
        clean = torch.randn(4, 11, 2)
        clean[:, -1, 1] = 0
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0, 99, 499, 999])
        noisy, times = diffusion.corrupt(clean, timesteps, noise)
        alpha_squared = torch.exp(-0.1 * times - 0.5 * 19.9 * times.square())[:, None, None]
        expected = alpha_squared.sqrt() * clean + (1 - alpha_squared).sqrt() * noise
        expected[:, -1, 1] = 0
        torch.testing.assert_close(noisy, expected, atol=2e-6, rtol=2e-5)
        self.assertLess(float(diffusion.scheduler.alphas_cumprod[-1]), 5e-5)

    def test_l1_has_path_and_speed_gradients(self):
        prediction = torch.ones(2, 11, 2, requires_grad=True)
        loss = GeoCodec.loss(prediction, torch.zeros_like(prediction))
        loss.backward()
        self.assertEqual(float(loss), 1.0)
        self.assertGreater(float(prediction.grad[:, :-1].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[:, -1, 0].abs().sum()), 0)
        self.assertEqual(float(prediction.grad[:, -1, 1].abs().sum()), 0)

    def test_hundred_step_gaussian_start_oracle(self):
        diffusion = VPFullDiffusion()
        clean = torch.randn(2, 11, 2)
        clean[:, -1, 1] = 0
        calls = []

        def oracle(noisy, time):
            calls.append((noisy.clone(), time.clone()))
            return clean

        generator = torch.Generator().manual_seed(901)
        result = diffusion.sample(oracle, 2, "cpu", generator)
        expected_start = torch.randn((2, 11, 2), generator=torch.Generator().manual_seed(901))
        expected_start[:, -1, 1] = 0
        torch.testing.assert_close(calls[0][0], expected_start)
        torch.testing.assert_close(result, clean)
        self.assertEqual(len(calls), 100)
        self.assertEqual(float(calls[0][1][0]), 1.0)

    def test_upstream_geo_head_and_all_trainable_gradients(self):
        from method.full_diffusion.model import TrajectoryFullDiffuGeoHead
        torch.manual_seed(901)
        head = TrajectoryFullDiffuGeoHead()
        condition = head.prepare_condition(
            torch.randn(2, 64, 8, 8), torch.randn(2, 16, 256),
            torch.randn(2, 11, 256), torch.randn(2, 2),
        )
        clean = head.codec.encode(torch.randn(2, 10, 2), torch.tensor([0.0, 8.0]))
        noisy, time = VPFullDiffusion(head.codec).corrupt(clean)
        predictions = head(noisy, time, condition)
        self.assertEqual(predictions.shape, (2, 2, 11, 2))
        head.loss(predictions, clean).backward()
        missing = [name for name, parameter in head.named_parameters() if parameter.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in head.parameters()))
        self.assertFalse(any("anchor" in name or "cls_branch" in name for name, _ in head.named_parameters()))

    def test_bev_grid_uses_garage_histogram_axes(self):
        from method.full_diffusion.model import GeoDecoderLayer
        layer = GeoDecoderLayer()
        grid = torch.tensor([[[[16.0, 0.0]]]]) / 32.0
        longitudinal = torch.arange(8).float()[None, None, None].expand(1, 1, 8, 8)
        lateral = longitudinal.transpose(-1, -2)
        sampled_x = torch.nn.functional.grid_sample(longitudinal, grid, align_corners=False)
        sampled_y = torch.nn.functional.grid_sample(lateral, grid, align_corners=False)
        self.assertAlmostEqual(float(sampled_x), 5.5)
        self.assertAlmostEqual(float(sampled_y), 3.5)
        self.assertEqual(layer.path_points, 10)

    def test_real_garage_frame_and_labels(self):
        from method.full_diffusion.data import DATA_ROOT, GarageGeoDataset
        from method.full_diffusion.garage import load_config, make_preprocessor
        dataset = GarageGeoDataset.__new__(GarageGeoDataset)
        dataset.processor = make_preprocessor(load_config(), training=False)
        route = DATA_ROOT / "Accident/Town12_Rep0_10_0_route0_11_08_23_53_07"
        sample = dataset.load_frame(route, 10)
        self.assertEqual(sample["path"].shape, (10, 2))
        self.assertEqual(sample["rgb"].shape[0], 3)
        self.assertEqual(sample["lidar"].shape, (1, 256, 256))
        self.assertTrue(all(torch.isfinite(value).all() for value in sample.values() if torch.is_tensor(value)))
        distances = torch.linalg.vector_norm(sample["path"][1:] - sample["path"][:-1], dim=-1)
        torch.testing.assert_close(distances, torch.ones_like(distances), atol=1e-4, rtol=1e-4)
        self.assertGreaterEqual(float(sample["target_speed"]), 0)

    def test_normalization_is_checkpointed_and_train_only(self):
        codec = GeoCodec(mean=[[1.0, 0.0]] * 10 + [[4.0, 0.0]], std=[[2.0, 3.0]] * 10 + [[4.0, 1.0]])
        path = torch.randn(2, 10, 2)
        speed = torch.tensor([0.0, 8.0])
        normalized = codec.encode(path, speed)
        restored = GeoCodec()
        restored.load_state_dict(codec.state_dict())
        decoded_path, decoded_speed = restored.decode(normalized)
        torch.testing.assert_close(path, decoded_path)
        torch.testing.assert_close(speed, decoded_speed)
        from method.full_diffusion.data import route_is_validation
        self.assertEqual(route_is_validation("scenario/route"), route_is_validation("scenario/route"))

    def test_detached_launcher_uses_absolute_paths(self):
        import tempfile
        from unittest.mock import patch
        from tools import launch_full_diffusion
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["launch_full_diffusion.py", "--job-id", "13047898", "--phase", "index",
                         "--output", "relative_output", "--detach"]
            with patch.object(sys, "argv", arguments), patch.object(launch_full_diffusion, "guard_context"), \
                 patch("os.getcwd", return_value=directory), patch.object(launch_full_diffusion.subprocess, "Popen") as launch:
                launch.return_value.pid = 123
                launch_full_diffusion.main()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--output") + 1], str(Path(directory) / "relative_output"))
            self.assertTrue(Path(command[command.index("--config") + 1]).is_absolute())

    def test_real_dataset_augmentation_is_epoch_deterministic(self):
        from method.full_diffusion.data import GarageGeoDataset
        dataset = GarageGeoDataset(ROOT / "results/full_diffusion_index_smoke")
        first = dataset[0]
        second = dataset[0]
        for key in ("rgb", "lidar", "path", "target_point"):
            torch.testing.assert_close(first[key], second[key])


if __name__ == "__main__":
    unittest.main()