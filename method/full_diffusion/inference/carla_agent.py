"""Leaderboard entrypoint for the Garage/PDM-Lite Full Diffusion Geo run."""

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

from method.full_diffusion.inference.policy import FullDiffusionNetwork


def get_entry_point():
    return "FullDiffusionGeoAgent"


class FullDiffusionGeoAgent(sensor_agent.SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        checkpoint_path = Path(path_to_conf_file.split("+")[0])
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "latest.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        create_model = timm.create_model

        def offline_model(*args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(*args, **kwargs)

        with patch.object(timm, "create_model", side_effect=offline_model):
            super().setup(str(BACKBONE), route_index, traffic_manager)
        if self.model_count != 1:
            raise RuntimeError("This reproduction uses one frozen all_towns seed-1 perception model")
        self.nets = [FullDiffusionNetwork(self.nets[0], checkpoint_path, self.device)]
        self.config.inference_direct_controller = 1
        self.compile = False