"""Read-only integration with the pinned CARLA Garage perception checkpoint."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import jsonpickle
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE = ROOT / "third_party/carla_garage/team_code"
BACKBONE = ROOT / "ckps/full_diffusion/backbone/all_towns_seed1"


def load_config(directory=BACKBONE):
    if str(TEAM_CODE) not in sys.path:
        sys.path.insert(0, str(TEAM_CODE))
    paths = json.loads((ROOT / "configs/paths.json").read_text())
    navigation = str(Path(paths["carla_root"]) / "PythonAPI/carla")
    if navigation not in sys.path:
        sys.path.insert(0, navigation)
    from config import GlobalConfig
    config = jsonpickle.decode((Path(directory) / "config.json").read_text())
    if not isinstance(config, GlobalConfig):
        raise TypeError("The official checkpoint config did not decode as GlobalConfig")
    return config


def make_preprocessor(config, training=False):
    from copy import deepcopy
    from data import CARLA_Data
    config = deepcopy(config)
    config.num_route_points = config.predict_checkpoint_len
    if not training:
        config.augment = False
        config.use_color_aug = False
        config.color_aug_prob = 0.0
        config.lidar_aug_prob = 0.0
    return CARLA_Data(root=[], config=config, rank=1)


class FrozenGaragePerception(nn.Module):
    def __init__(self, directory=BACKBONE):
        super().__init__()
        self.config = load_config(directory)
        from model import LidarCenterNet
        import timm
        create_model = timm.create_model

        def offline_model(*args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(*args, **kwargs)

        with patch.object(timm, "create_model", side_effect=offline_model):
            self.net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        state = torch.load(Path(directory) / "model_0030_1.pth", map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        net = self.net
        bev, fused, _ = net.backbone(batch["rgb"].float(), batch["lidar"].float())
        fused = net.change_channel(fused)
        fused = fused + net.encoder_pos_encoding(fused)
        fused = fused.flatten(start_dim=2)
        if net.extra_sensors:
            extra = []
            if self.config.use_velocity:
                extra.append(net.velocity_normalization(batch["speed"].reshape(-1, 1)))
            if self.config.use_discrete_command:
                extra.append(batch["command"])
            extra = net.extra_sensor_encoder(torch.cat(extra, dim=1)) + net.extra_sensor_pos_embed
            fused = torch.cat((fused, extra[:, :, None]), dim=2)
        memory = fused.transpose(1, 2)
        if self.config.tp_attention:
            target_point = batch["target_point"]
            if self.config.two_tp_input:
                target_point = torch.cat((target_point, batch["target_point_next"]), dim=-1)
            target_token = net.tp_encoder(target_point) + net.tp_pos_embed
            memory = torch.cat((memory, target_token[:, None]), dim=1)
        ego = net.join(net.checkpoint_query.expand(memory.shape[0], -1, -1), memory)
        if isinstance(ego, tuple):
            ego = ego[0]
        return bev, memory, ego