"""Real-data update, sampler/gradient checks, inference and optimizer round-trip."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from method.full_diffusion2.perception import FrozenPlannerPerception
from method.full_diffusionv8.legacy import module
from . import FORMAT, METHOD
from .data import PointDataset
from .model import BaselinePointQueryHead, PointDecoderLayer, masked_softmax
from .train import build, encode_context, condition_inputs, losses_for, worker_init
from .utils import (ROOT, atomic_json, atomic_save, inspect_allocation, move, sha,
                    verify_data)


def architecture_checks(head, device):
    # The initializer already has trained offsets and weights. Test the reader's
    # analytic geometry using fresh instances without changing pretrained weights.
    readers = [PointDecoderLayer(10).to(device), PointDecoderLayer(6, temporal=True).to(device)]
    for reader in readers:
        reader.debug = True
    mask = torch.tensor([[[True, False, False, False, False], [False] * 5]], device=device)
    result = masked_softmax(torch.zeros(1, 2, 5, device=device), mask)
    torch.testing.assert_close(result[0, 0], result.new_tensor([1., 0., 0., 0., 0.]))
    torch.testing.assert_close(result[0, 1], result.new_zeros(5))
    # Test metric-to-grid orientation against an analytic, non-symmetric ramp.
    y, x = torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing="ij")
    ramp = ((x + .5) / 64. * 64. - 32.) + 3. * ((y + .5) / 64. * 64. - 32.)
    values = ramp[None, None].expand(1, 256, 64, 64).contiguous()
    queries = torch.randn(1, 10, 256, device=device)
    points = torch.tensor([[2., 3.]], device=device).expand(1, 10, 2).clone()
    reader = readers[0]
    reader.local_read(queries, points, values)
    torch.testing.assert_close(reader.last_read["coordinates"][:, :, 0], points)
    torch.testing.assert_close(reader.last_read["pooled"], torch.full_like(reader.last_read["pooled"], 11.), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(reader.last_read["weights"].sum(-1), torch.ones(1, 10, device=device))
    output = reader.local_read(queries, torch.full_like(points, 1000.), values)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    temporal = readers[1]
    delta = torch.tensor([[1., 2.]], device=device).expand(1, 6, 2).clone()
    temporal.local_read(queries[:, :6], delta, values)
    torch.testing.assert_close(temporal.last_read["coordinates"][:, :, 0], delta.cumsum(1))
    # Sampling coordinates and softmax weights both have actual BEV-gradient paths.
    reader.zero_grad(set_to_none=True)
    features = torch.randn(1, 256, 64, 64, device=device)
    reader.local_read(queries, points, features).square().mean().backward()
    gradients = {key: float(value.grad.float().norm()) for key, value in reader.named_parameters()
                 if key in ("sampling_offsets.weight", "cross_bev_attention.attention_weights.weight")}
    if len(gradients) != 2 or any(not np.isfinite(v) or v <= 0 for v in gradients.values()):
        raise RuntimeError(f"Learned offsets/weights have no finite gradient: {gradients}")
    reader.zero_grad(set_to_none=True)
    for reader in readers:
        reader.debug, reader.last_read = False, None
    return {"masked_softmax": True, "all_invalid_zero_residual": True,
            "metric_xy_grid_ramp": True, "temporal_cumsum": True, "sampling_gradients": gradients}


@torch.no_grad()
def initialization_checks(head, batch, config, device):
    settings = {k: v for k, v in config["model"].items() if not k.startswith("navigation_")}
    baseline = BaselinePointQueryHead(**settings, normalization={
        "mean": head.codec.mean.detach().cpu().tolist(),
        "std": head.codec.std.detach().cpu().tolist(),
    }).to(device).eval()
    source = {k: v for k, v in head.state_dict().items() if not k.startswith(
        ("navigation_encoder.", "path_navigation_fusion.", "temporal_navigation_fusion."))}
    baseline.load_state_dict(source, strict=True)
    head.eval()
    noisy = torch.randn(len(batch["path"]), 16, 2, device=device)
    time_value = torch.full((len(noisy),), .5, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        condition = head.prepare_condition(*condition_inputs(batch))
        expected = baseline(noisy, time_value, condition)["predictions"]
        actual = head(noisy, time_value, condition)["predictions"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        changed = {**condition, "navigation_token": torch.randn_like(condition["navigation_token"]) * 10}
        torch.testing.assert_close(head(noisy, time_value, changed)["predictions"], actual, atol=0, rtol=0)
    assert condition["navigation_token"].shape == (len(noisy), 256)
    del baseline
    return {"baseline_denoising_outputs_bitwise_equal_bf16": True,
            "initial_navigation_branch_zero_effect": True, "new_parameters": sum(
                p.numel() for n, p in head.named_parameters() if n.startswith(
                    ("navigation_encoder.", "path_navigation_fusion.", "temporal_navigation_fusion.")))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    proof = inspect_allocation(config, args.job_id, idle=True)
    metadata = verify_data(config)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    head, diffusion, artifact, initialization = build(config, device)
    architecture = architecture_checks(head, device)
    dataset = PointDataset(config)
    # Verify labels against the already-trained WM's exact action interface.
    route, slot, frame = map(int, dataset.samples[0])
    sample = dataset[0]
    delta, positions, _, _ = module("data").motion_targets(dataset.metadata(route)["poses"], slot)
    np.testing.assert_allclose(sample["temporal_delta"].numpy(), delta, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(sample["future_positions"].numpy(), positions, atol=1e-5, rtol=1e-5)
    perception = FrozenPlannerPerception(ROOT / config["backbone_directory"]).to(device).eval()
    if any(p.requires_grad for p in perception.parameters()):
        raise RuntimeError("TF++ is not frozen")
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"], fused=True)
    cases, last_batch, identity = [], None, None
    for size in (8, config["training"]["batch_per_gpu"]):
        raw = default_collate([dataset[index] for index in range(size)])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = encode_context(move(raw, device), perception, config)
        torch.cuda.synchronize()
        perception_seconds = time.perf_counter() - started
        if identity is None:
            identity = initialization_checks(head, batch, config, device)
        for iteration in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            head.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                clean = head.codec.encode(batch["path"], batch["temporal_delta"])
                noisy, timestep = diffusion.corrupt(clean)
                result = head(noisy, timestep, condition_inputs(batch))
                losses = losses_for(head, result, clean, batch, config)
            losses["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            navigation_gradients = {name: float(value.grad.float().norm())
                for name, value in head.named_parameters()
                if name in ("navigation_encoder.0.weight", "path_navigation_fusion.weight",
                            "temporal_navigation_fusion.weight") and value.grad is not None}
            if iteration == 1 and (len(navigation_gradients) != 3 or any(
                    not np.isfinite(v) or v <= 0 for v in navigation_gradients.values())):
                raise RuntimeError(f"Navigation fusion has no finite learning path: {navigation_gradients}")
            if any(p.grad is not None for p in perception.parameters()):
                raise RuntimeError("Trajectory gradients reached frozen TF++")
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
        case = {"batch_per_gpu": size, "perception_seconds": perception_seconds,
                "planner_update_seconds": duration, "loss": float(losses["loss"]), "gradient_norm": float(norm),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "navigation_gradients": navigation_gradients, **head.navigation_statistics()}
        cases.append(case)
        print(json.dumps({"batch_probe": case}), flush=True)
        last_batch = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        del result, losses, clean, noisy, batch, raw
    head.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition = head.prepare_condition(*condition_inputs(last_batch))
        noisy = torch.randn(1, 16, 2, device=device)
        t = torch.tensor([.5], device=device)
        original = head(noisy, t, condition)["predictions"]
        changed_navigation = {**condition, "navigation_token": torch.zeros_like(condition["navigation_token"])}
        navigation_effect = float((head(noisy, t, changed_navigation)["predictions"] - original).abs().max())
        if not np.isfinite(navigation_effect) or navigation_effect <= 0:
            raise RuntimeError("The trained navigation route has no effect on denoising")
        other = noisy.clone()
        other[:, 10:] += 2
        torch.testing.assert_close(head(other, t, condition)["predictions"][:, :, :10], original[:, :, :10], atol=0, rtol=0)
        altered = dict(last_batch)
        altered["memory"] = last_batch["memory"].clone()
        altered["memory"][:, :64] = torch.randn_like(altered["memory"][:, :64]) * 100
        changed_condition = head.prepare_condition(*condition_inputs(altered))
        torch.testing.assert_close(head(noisy, t, changed_condition)["predictions"], original, atol=0, rtol=0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        sampled = diffusion.sample(lambda x, time: head(x, time, condition)["predictions"][-1],
                                   1, device, torch.Generator(device=device).manual_seed(901700), 25)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        path, temporal = head.codec.decode(sampled)
        if path.shape != (1, 10, 2) or temporal.shape != (1, 6, 2) or not torch.isfinite(sampled).all():
            raise RuntimeError("Full-noise 25-step inference did not produce valid 10/6 outputs")
    # Exercise spawn workers with the actual dataset before launching eight ranks.
    loader = DataLoader(dataset, batch_size=4, num_workers=2, prefetch_factor=2, timeout=180,
                        multiprocessing_context="spawn", worker_init_fn=worker_init)
    iterator = iter(loader)
    worker_batch = next(iterator)
    if worker_batch["history_bev"].shape != (4, 2, 64, 64, 64):
        raise RuntimeError("Real spawned DataLoader returned invalid observation tensors")
    iterator._shutdown_workers()
    del iterator, loader, worker_batch
    path = output / "preflight_model.pt"
    atomic_save(path, {"format": FORMAT, "method": METHOD, "model": head.state_dict(), "optimizer": optimizer.state_dict(),
                       "config": config, "dataset": metadata, "progress": {"updates": 4, "epoch": 0},
                       "initialization": initialization})
    restored = torch.load(path, map_location="cpu", weights_only=False)
    if not restored["optimizer"]["state"]:
        raise RuntimeError("Optimizer states were not saved")
    expected = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(restored["model"], strict=True)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    optimizer.load_state_dict(restored["optimizer"])
    report = {"passed": True, "allocation": proof, "architecture_checks": architecture,
              "navigation_initialization": identity, "navigation_effect_after_probe": navigation_effect,
              "WM_temporal_label_match": True, "direct_memory_invariance": True,
              "independent_geometry_temporal_queries": True, "spawn_dataloader": True,
              "optimizer_checkpoint_roundtrip": True, "initialization": initialization,
              "batch_probes": cases, "full_noise_inference_steps": 25,
              "inference_seconds_single_sample": inference_seconds,
              "probe_checkpoint": path.name,
              "checkpoint_sha256": sha(path), "unix": time.time()}
    atomic_json(output / "preflight.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

