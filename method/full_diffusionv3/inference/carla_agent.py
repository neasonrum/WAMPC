"""Per-tick Bench2Drive sensor agent for the pure-E2E V3 planner."""

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "leaderboard"))
sys.path.insert(0, str(ROOT / "scenario_runner"))

from method.full_diffusion.garage import BACKBONE, load_config

load_config()

import sensor_agent
import timm

from method.full_diffusionv3.inference.policy import FullDiffusionV3Network


def get_entry_point():
    return "FullDiffusionV3Agent"


class FullDiffusionV3Agent(sensor_agent.SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        checkpoint = Path(path_to_conf_file.split("+")[0])
        if checkpoint.is_dir():
            checkpoint = checkpoint / "latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        create_model = timm.create_model

        def offline_model(*args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(*args, **kwargs)

        with patch.object(timm, "create_model", side_effect=offline_model):
            super().setup(str(BACKBONE), route_index, traffic_manager)
        if self.model_count != 1:
            raise RuntimeError("V3 requires exactly one frozen all_towns perception model")
        self.nets = [FullDiffusionV3Network(self.nets[0], checkpoint, self.device)]
        self.config.inference_direct_controller = 1
        self.compile = False
        self.observation_timestamp = 0.0

    def tick(self, input_data):
        result = super().tick(input_data)
        pose = self.state_log[-1][:3].copy()
        for net in self.nets:
            net.set_observation(pose, self.observation_timestamp)
        return result

    def run_step(self, input_data, timestamp, sensors=None):
        self.observation_timestamp = float(timestamp)
        control = super().run_step(input_data, timestamp, sensors)
        current_speed = float(input_data["speed"][1]["speed"])
        if current_speed * 3.6 > 35:
            control = self.nets[0].apply_speed_limit(control, current_speed)
            self.control = control
        return control