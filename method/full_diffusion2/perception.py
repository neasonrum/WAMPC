"""Frozen current-frame context and backbone-only historical feature extraction."""

import torch
from torch import nn

from method.full_diffusion.garage import FrozenGaragePerception


class FrozenPlannerPerception(FrozenGaragePerception):
    @classmethod
    def from_net(cls, net):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.net = net
        instance.config = net.config
        instance.requires_grad_(False)
        instance.eval()
        return instance

    @torch.no_grad()
    def encode_history(self, batch, chunk=8):
        images = batch["history_rgb"]
        batch_size, frames = images.shape[:2]
        images = images.flatten(0, 1)
        lidar = batch["history_lidar"].flatten(0, 1)
        encoded = []
        for start in range(0, len(images), chunk):
            bev, _, _ = self.net.backbone(images[start:start + chunk].float(),
                                          lidar[start:start + chunk].float())
            encoded.append(bev)
        features = torch.cat(encoded)
        return features.reshape(batch_size, frames, *features.shape[1:])