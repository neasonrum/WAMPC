"""Exercise the actual online wrapper on real observations, without starting CARLA."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from method.full_diffusion2.perception import FrozenPlannerPerception
from .data import PointDataset
from .policy import PointQueryNetwork
from .utils import ROOT, atomic_json, move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    data = PointDataset(config)
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device).eval()
    network = PointQueryNetwork(perception.net, Path(config["output"]) / "preflight_model.pt", device)
    checks = []
    for index in (0, 1):
        sample = move(default_collate([data[index]]), device)
        route, slot, frame = map(int, data.samples[index])
        network.set_observation(data.metadata(route)["poses"][slot], frame * .25)
        output = network(sample["rgb"], sample["lidar"], sample["target_point"], sample["speed"], sample["command"])
        if len(output) != 10 or output[2].shape != (1, 10, 2) or network.last_temporal_delta.shape != (1, 6, 2):
            raise RuntimeError("Online 10-point/6-increment interface mismatch")
        speed = network.head.codec.target_speed(network.last_temporal_delta)
        torch.testing.assert_close(speed[0], speed.new_tensor(network.last_target_speed))
        controller = network.control_pid_direct(output[2][0].detach().cpu().numpy(), output[1], sample["speed"])
        if not all(torch.isfinite(torch.as_tensor(value)).all() for value in controller):
            raise RuntimeError("The existing longitudinal/lateral controller produced nonfinite control")
        checks.append({"frame": frame, "path_shape": list(output[2].shape),
                       "temporal_shape": list(network.last_temporal_delta.shape),
                       "target_speed": network.last_target_speed, "control": list(controller)})
    atomic_json(Path(config["output"]) / "online_acceptance.json", {"passed": True,
                "actual_online_wrapper": True, "real_sensor_observations": len(checks),
                "CARLA_engine_started": False, "inference_steps": 25, "checks": checks, "unix": time.time()})
    print(json.dumps({"online_acceptance": "PASS", "observations": len(checks)}), flush=True)


if __name__ == "__main__":
    main()

