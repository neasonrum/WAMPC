"""Real eight-rank data/augmentation and fresh-model mid-epoch resume acceptance."""

import hashlib
import json
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import BatchSampler, DistributedSampler, Subset

from method.full_diffusion2.snapshot import atomic_json
from method.full_diffusionv3.checkpoint import cpu_copy, load_checkpoint, rng_state, save_checkpoint, select_complete
from method.full_diffusionv3.runtime import RemainingBatches, coordinated_next, make_training_loader
from method.full_diffusionv3.train import autocast, condition_inputs, move_batch, new_head, update


def batch_digest(batch):
    digest = hashlib.sha256()
    for name in sorted(batch):
        digest.update(name.encode())
        value = batch[name]
        if torch.is_tensor(value):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def max_difference(actual, expected):
    if torch.is_tensor(expected):
        return float((actual.cpu().double() - expected.double()).abs().max()) if expected.numel() else 0.0
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("Resumed optimizer/model structure differs")
        return max((max_difference(actual[name], value) for name, value in expected.items()), default=0.0)
    if isinstance(expected, (list, tuple)):
        return max((max_difference(first, second) for first, second in zip(actual, expected)), default=0.0)
    if actual != expected:
        raise AssertionError(f"Resumed scalar state differs: {actual} versus {expected}")
    return 0.0


def run_acceptance(args, config, manifest, dataset, perception, head, policy, optimizer, scheduler,
                   diffusion, device, control, trace):
    rank, world = distributed.get_rank(), distributed.get_world_size()
    batch_count = 3
    window = Subset(dataset, range(world * config["batch_per_gpu"] * batch_count))

    def loader_components():
        sampler = DistributedSampler(window, seed=config["seed"], drop_last=True)
        batches = RemainingBatches(BatchSampler(sampler, config["batch_per_gpu"], drop_last=True))
        loader = make_training_loader(window, batches, config["workers_per_rank"], config["seed"], rank)
        return sampler, batches, loader

    before = cpu_copy(head.state_dict())
    sampler, batches, loader = loader_components()
    expected = []
    started, updates, gate_gradient = time.monotonic(), 0, 0.0
    previous_pids = None
    cross_epoch_same_workers = False
    for epoch in range(2):
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        pids = [worker.pid for worker in loader._iterator._workers]
        if previous_pids is not None:
            cross_epoch_same_workers = pids == previous_pids
        previous_pids = pids
        for batch_id in range(batch_count):
            trace.progress(epoch, updates)
            cpu_batch = coordinated_next(iterator, control, trace)
            digest = batch_digest(cpu_batch)
            batch = move_batch(cpu_batch, device)
            losses, norm = update(batch, perception, policy, optimizer, diffusion, config, control, trace)
            updates += 1
            scheduler.step(epoch + (batch_id + 1) / batch_count)
            gate_gradient = max(gate_gradient, sum(float(parameter.grad.abs().sum()) for name, parameter in head.named_parameters()
                                                     if name.startswith("temporal.gate") and parameter.grad is not None))
            if epoch == 0 and batch_id == 1:
                progress = {"epoch": epoch, "next_batch": batch_id + 1, "updates": updates}
                save_checkpoint(args.output / "latest.pt", head, optimizer, scheduler, progress,
                                config, manifest, control, trace, loader)
            elif epoch > 0 or batch_id > 1:
                expected.append({"digest": digest, "epoch": epoch, "batch_id": batch_id,
                                 "model": cpu_copy(head.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
                                 "scheduler": scheduler.state_dict(), "lr": optimizer.param_groups[0]["lr"],
                                 "rng": rng_state(), "loss": float(losses["loss"]), "gradient_norm": float(norm)})
    loader._iterator._shutdown_workers()
    del iterator, loader, optimizer, scheduler, policy, head, diffusion
    torch.manual_seed(config["seed"])
    head, policy, optimizer, scheduler, diffusion = new_head(config, manifest, device, int(device.index), trace)
    sampler, batches, loader = loader_components()
    complete = select_complete(args.output, config, manifest)
    progress = load_checkpoint(complete["absolute_path"], head, optimizer, scheduler, config, manifest, loader)
    model_error = optimizer_error = loss_error = 0.0
    matched = 0
    try:
        for epoch in range(progress["epoch"], 2):
            dataset.epoch = epoch
            sampler.set_epoch(epoch)
            batches.offset = progress["next_batch"] if epoch == progress["epoch"] else 0
            iterator = iter(loader)
            for batch_id in range(batches.offset, batch_count):
                cpu_batch = coordinated_next(iterator, control, trace)
                expected_step = expected[matched]
                if batch_digest(cpu_batch) != expected_step["digest"]:
                    raise AssertionError("Real resumed sample order, augmentation, history, or labels differ")
                batch = move_batch(cpu_batch, device)
                losses, norm = update(batch, perception, policy, optimizer, diffusion, config, control, trace)
                scheduler.step(epoch + (batch_id + 1) / batch_count)
                model_error = max(model_error, max_difference(head.state_dict(), expected_step["model"]))
                optimizer_error = max(optimizer_error, max_difference(optimizer.state_dict(), expected_step["optimizer"]))
                loss_error = max(loss_error, abs(float(losses["loss"]) - expected_step["loss"]))
                if optimizer.param_groups[0]["lr"] != expected_step["lr"] or scheduler.state_dict() != expected_step["scheduler"]:
                    raise AssertionError("Real resumed LR/scheduler state differs")
                state = rng_state()
                if (not torch.equal(state["torch"], expected_step["rng"]["torch"])
                        or not torch.equal(state["cuda"], expected_step["rng"]["cuda"])
                        or random.getstate() != expected_step["rng"]["python"]
                        or not np.array_equal(state["numpy"][1], expected_step["rng"]["numpy"][1])):
                    raise AssertionError("Main/model RNG changed across real resume")
                matched += 1
        if control.any(model_error > 2e-6 or optimizer_error > 2e-6 or matched != 4 or not cross_epoch_same_workers):
            raise AssertionError(f"Real resume diverged: model={model_error}, Adam={optimizer_error}, steps={matched}")
        head.eval()
        with torch.no_grad(), autocast(config), trace.stage("validation_sample", cuda=True):
            condition = head.prepare_condition(*condition_inputs(batch, perception, config))
            sample = diffusion.sample(lambda noisy, timestep: head(noisy, timestep, condition)["predictions"][-1],
                                      config["batch_per_gpu"], device,
                                      inference_steps=config["diffusion"]["inference_steps"])
        if sample.shape != (config["batch_per_gpu"], 16, 2) or not torch.isfinite(sample).all():
            raise AssertionError("Real 100-step V3 sampling is nonfinite or has the wrong sixteen-slot shape")
        changed_gate = max(float((value.detach().cpu() - before[name]).abs().max()) for name, value in head.named_parameters()
                           if name.startswith("temporal.gate"))
        changed_behavior = max(float((value.detach().cpu() - before[name]).abs().max()) for name, value in head.named_parameters()
                               if name.startswith("behavior_head"))
        changed_queries = {
            prefix: max(float((value.detach().cpu() - before[name]).abs().max())
                        for name, value in head.named_parameters() if name.startswith(prefix))
            for prefix in ("path_query_encoder", "temporal_query_encoder")
        }
        result = {"passed": True, "rank": rank, "world_size": world, "host": socket.gethostname(),
                  "real_sensor_loader": True, "observed_frames": 3,
                  "architecture": config["architecture"], "controller": config["controller"],
                  "query_count": 2, "query_width": 256, "cross_query_attention": False,
                  "slots": 16, "slot_width": config["model"]["slot_width"],
                  "query_parameter_max_delta": changed_queries, "baseline_updates": 6, "resumed_updates": matched,
                  "resume_checkpoint_update": complete["progress"]["updates"],
                  "sample_augmentation_history_label_hashes_exact": True,
                  "persistent_worker_cross_epoch": cross_epoch_same_workers,
                  "resume_model_max_error": model_error, "resume_optimizer_max_error": optimizer_error,
                  "resume_loss_max_error": loss_error, "LR_and_main_RNG_exact": True,
                  "gate_gradient_abs_sum": gate_gradient, "gate_parameter_max_delta": changed_gate,
                  "behavior_parameter_max_delta": changed_behavior,
                  "frozen_perception_no_grad": all(parameter.grad is None for parameter in perception.parameters()),
                  "gaussian_sampling_steps": config["diffusion"]["inference_steps"],
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "elapsed_seconds": time.monotonic() - started}
        if control.any(changed_gate == 0 or changed_behavior == 0 or gate_gradient == 0
                       or any(value == 0 for value in changed_queries.values())):
            raise AssertionError("Real gate/behavior or independent query parameters did not receive learning signal")
        atomic_json(args.output / f"acceptance_rank{rank}.json", result)
        print(json.dumps(result), flush=True)
        control.barrier()
    finally:
        loader._iterator._shutdown_workers()