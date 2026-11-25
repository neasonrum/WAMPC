"""Point-query E2E with identity-initialized per-point navigation fusion."""
import torch
from torch import nn
from torch.nn import functional as F

from method.full_diffusion_pointquery.model import (
    PointDecoderLayer, PointQueryHead as BaselinePointQueryHead,
    blocks, masked_softmax,
)


class PointQueryHead(BaselinePointQueryHead):
    def __init__(self, navigation_input_dim=8, navigation_width=256, **kwargs):
        super().__init__(**kwargs)
        width = kwargs.get("d_model", 256)
        if navigation_input_dim != 8 or navigation_width != width:
            raise ValueError("Navigation requires six command dimensions and normalized target XY")
        self.navigation_encoder = nn.Sequential(
            nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.path_navigation_fusion = nn.Linear(2 * width, width)
        self.temporal_navigation_fusion = nn.Linear(2 * width, width)
        self.register_buffer("_navigation_identity", torch.cat((torch.eye(width), torch.zeros(width, width)), 1),
                             persistent=False)
        for fusion in (self.path_navigation_fusion, self.temporal_navigation_fusion):
            with torch.no_grad():
                fusion.weight.zero_()
                fusion.weight[:, :width].copy_(torch.eye(width))
                fusion.bias.zero_()

    def prepare_condition(self, bev, memory, ego, target_point, history_bev,
                          history_transforms, history_mask, speed, command):
        condition = super().prepare_condition(
            bev, memory, ego, target_point, history_bev,
            history_transforms, history_mask, speed, command,
        )
        navigation = torch.cat((command.float(), target_point.float() / self.path_scale), -1)
        if navigation.shape != (len(bev), 8):
            raise ValueError("Expected [B,6] command and current ego-frame [B,2] target")
        condition["navigation_token"] = self.navigation_encoder(navigation)
        return condition

    def fuse_navigation(self, query, navigation, projection):
        context = navigation[:, None].expand(-1, query.shape[1], -1)
        # Equivalent to Linear([query,nav], W, b), but computes only W-[I,0]
        # under autocast. The original query bypasses BF16/TF32 rounding, making
        # the [I,0] initialization exactly preserve the pretrained planner.
        residual = F.linear(torch.cat((query, context), -1),
                            projection.weight - self._navigation_identity, projection.bias)
        return query + residual.to(query.dtype)

    def forward(self, noisy, time, condition):
        if isinstance(condition, tuple):
            condition = self.prepare_condition(*condition)
        self.codec.check_shape(noisy)
        features = blocks.gen_sineembed_for_position(noisy, hidden_dim=self.slot_width)
        features = features + self.type_embedding(self.type_ids)[None] + self.slot_embedding(self.slot_ids)[None]
        spatial_query = self.fuse_navigation(
            self.path_query_encoder(features[:, :10]), condition["navigation_token"],
            self.path_navigation_fusion,
        ) + condition["behavior_offset"]
        temporal_query = self.fuse_navigation(
            self.temporal_query_encoder(features[:, 10:]), condition["navigation_token"],
            self.temporal_navigation_fusion,
        ) + condition["temporal_behavior_offset"]
        time_embedding = self.time_mlp(time * 1000)[:, None]
        temporal_time = self.temporal_time_mlp(time * 1000)[:, None]
        metric = self.codec.to_metric(noisy)
        predictions = []
        for layer, temporal_layer, values, temporal_values in zip(
                self.layers, self.temporal_layers, condition["bev"], condition["temporal_bev"]):
            spatial_query, spatial = layer(spatial_query, metric[:, :10], values, condition["ego"], time_embedding)
            temporal_query, temporal = temporal_layer(temporal_query, metric[:, 10:], temporal_values,
                                                       condition["ego"], temporal_time)
            prediction = self.codec.encode_metric(torch.cat((spatial, temporal), 1))
            predictions.append(prediction)
            metric = self.codec.to_metric(prediction.detach())
        return {"predictions": torch.stack(predictions), "behavior_logits": condition["behavior_logits"]}

    def navigation_statistics(self):
        width = self.path_navigation_fusion.out_features
        return {
            name: float(layer.weight[:, width:].detach().float().norm())
            for name, layer in (("path_navigation_weight_norm", self.path_navigation_fusion),
                                ("temporal_navigation_weight_norm", self.temporal_navigation_fusion))
        }


def load_compatible_ego12(head, checkpoint):
    """Strictly reuse every completed point-query E2E tensor, adding only navigation."""
    if (checkpoint.get("format") != "wmmpc_pointquery_e2e_v1"
            or checkpoint.get("method") != "full_diffusion_pointquery"
            or checkpoint["progress"]["epoch"] != 10
            or checkpoint["config"]["architecture"] != "independent_point_query_learned_local_bev_v1"):
        raise ValueError("Initialize only from the agreed pure E2E point-query epoch10")
    source = checkpoint["model"]
    target = head.state_dict()
    new_keys = {key for key in target if key.startswith(
        ("navigation_encoder.", "path_navigation_fusion.", "temporal_navigation_fusion."))}
    if set(source) != set(target) - new_keys:
        raise ValueError("The E2E initializer must match every original model tensor")
    for key, tensor in source.items():
        if tensor.shape != target[key].shape:
            raise ValueError(f"Baseline tensor shape changed: {key}")
    for key in ("codec.mean", "codec.std"):
        if not torch.equal(source[key].cpu(), target[key].cpu()):
            raise ValueError("Pretrained and fixed-protocol coordinate normalization differ")
    result = head.load_state_dict(source, strict=False)
    if set(result.missing_keys) != new_keys or result.unexpected_keys:
        raise ValueError("Unexpected checkpoint loading discrepancy")
    return {"loaded_tensors": len(source), "loaded_parameters": sum(v.numel() for v in source.values()),
            "new_or_resized": sorted(new_keys), "not_reused": [],
            "source_epoch": checkpoint["progress"]["epoch"],
            "source_updates": checkpoint["progress"]["updates"],
            "new_parameters": sum(target[k].numel() for k in new_keys),
            "fusion_initialization": "W=[I,0], bias=0; original query preserved",
            "mode": "strict pure point-query E2E epoch10 + new shared navigation encoder and branch-specific fusion"}
