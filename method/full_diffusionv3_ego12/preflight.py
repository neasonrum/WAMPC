"""Check the new conditioning and 25-step inference on one real PDM-Lite frame."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import default_collate

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusion2.snapshot import atomic_json
from method.full_diffusionv3.data import GarageSpatialTemporalDataset
from method.full_diffusionv3_ego12.allocation import read, run_path
from method.full_diffusionv3_ego12.inference.policy import Ego12Network
from method.full_diffusionv3_ego12.train import ROOT, SpatialTemporalFullDiffusionHead, condition_inputs, compute_losses, training_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    path = run_path(parser.parse_args().output)
    config = read(path / "config.json")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    manifest = training_metadata(config)
    backbone = ROOT / config["backbone_directory"]
    dataset = GarageSpatialTemporalDataset(
        ROOT / config["index_directory"], split="val", backbone_directory=backbone,
        data_root=manifest["data_root"], label_config=config["labels"], history_frames=2,
        history_stride=config["history_stride"], sensor_snapshot=ROOT / config["sensor_snapshot"])
    batch = default_collate([dataset[0]])
    perception = FrozenPlannerPerception(backbone)
    assert perception.config.use_velocity and perception.config.use_discrete_command and not perception.config.tp_attention
    inputs = condition_inputs(batch, perception, config)
    head = SpatialTemporalFullDiffusionHead(**config["model"], normalization=manifest["v3_normalization"]["normalization"])
    head.eval()
    condition = head.prepare_condition(*inputs)
    assert condition["ego"].shape == (1, 12, 256) and "memory" not in condition
    torch.testing.assert_close(condition["ego"][:, -1], inputs[1][:, -1], rtol=0, atol=0)
    changed_memory = inputs[1].clone()
    changed_memory[:, :64] += 17
    perturb = list(inputs)
    perturb[1] = changed_memory
    changed = head.prepare_condition(*perturb)
    clean = head.codec.encode(batch["path"], batch["temporal_delta"])
    t = torch.tensor([0.5])
    first = head(clean, t, condition)
    second = head(clean, t, changed)
    for key in ("predictions", "behavior_logits"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"])
    param = head.layers[0].cross_ego_attention.in_proj_weight
    before = param.detach().clone()
    loss = compute_losses(head, first, clean, batch, config)["loss"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    optimizer.step()
    assert not torch.equal(before, param)

    # Use the same strictly loaded inference policy as the CARLA agent.
    # The sensor agent sets these gains during setup, before building Ego12Network.
    perception.net.config.speed_kp = 1.75
    perception.net.config.speed_ki = 1.0
    perception.net.config.speed_kd = 2.0
    perception.net.config.speed_n = 20
    with TemporaryDirectory(prefix="cpu-preflight-", dir=path) as temp:
        checkpoint = Path(temp) / "probe.pt"
        torch.save({"config": config, "dataset": manifest, "model": head.state_dict()}, checkpoint)
        network = Ego12Network(perception.net, checkpoint, torch.device("cpu"))
        network.set_observation([0.0, 0.0, 0.0], 0.0)
        output = network(batch["rgb"], batch["lidar"], batch["target_point"], batch["speed"], batch["command"])
        assert output[2].shape == (1, 10, 2) and torch.isfinite(output[2]).all()
        assert network.last_temporal_delta.shape == (1, 6, 2)
        assert torch.isfinite(network.last_temporal_delta).all()
        assert network.run_config["diffusion"]["inference_steps"] == 25
    result = {"real_validation_frame": True, "ego_shape": list(condition["ego"].shape),
              "spatial_memory_invariance": True, "finite_optimizer_update": True,
              "strict_checkpoint_reload": True, "inference_steps": 25, "target_speed_mps": network.last_target_speed}
    atomic_json(path / "preflight.json", result)
    print("EGO12_PREFLIGHT", result, flush=True)


if __name__ == "__main__":
    main()
