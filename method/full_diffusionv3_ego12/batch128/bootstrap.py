"""Fork a paused batch-64 checkpoint into a new batch-128 run."""

import hashlib
import json
from pathlib import Path
import sys

import torch

from method.full_diffusion2.snapshot import atomic_json
from method.full_diffusionv3.checkpoint import commit, select_complete
from method.full_diffusionv3_ego12.allocation import RESULTS, RUN_PREFIX, read, run_path
from method.full_diffusionv3_ego12.batch128.train import ROOT, training_metadata


def main():
    prev = run_path(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    assert out.parent == RESULTS and out.parent.resolve(strict=True) == RESULTS and out.name.startswith(RUN_PREFIX)
    assert not (out.exists() or out.is_symlink())
    actual = read(prev / "paused.json")
    plan = read(prev / "plan.json")
    assert actual["epoch"] < plan["epochs"] and "signal" in actual["reasons"]
    assert all(read(prev / f"node_{job}.json")["state"] == "paused" for job in plan["jobs"])
    record = select_complete(prev, full_hash=True)
    state = torch.load(record["absolute_path"], map_location="cpu", weights_only=False)
    old = state["config"]
    progress = state["progress"]
    assert old["global_batch"] == 64 and old["world_size"] == len(state["rng_by_rank"]) == 10
    assert {k:actual[k] for k in ("epoch","next_batch","updates")} == progress
    assert state["optimizer"]["state"] and state["scheduler"]
    config = read(prev / "config.json")
    assert config == old
    switch = {"epoch": progress["epoch"], "sample_offset": 64 * progress["next_batch"],
              "prior_optimizer_updates": progress["updates"], "source_sha256": record["sha256"],
              "source_run": str(prev), "source_checkpoint": record["absolute_path"]}
    config.update(global_batch=128, rank_batches=[13]*8+[12]*2, batch_switch=switch,
                  master_port=29932, history_encode_chunk=8, source_identity_file=str(out / "train_sources.json"))
    size = state["dataset"]["train_frames"]
    remaining = (size - switch["sample_offset"]) // 128 + (plan["epochs"] - 1 - switch["epoch"]) * (size // 128)
    plan.update(master_port=config["master_port"], expected_updates=switch["prior_optimizer_updates"] + remaining,
                batch_switch=switch, remaining_updates=remaining, global_batch=128)
    sources = read(prev / "train_sources.json")
    for path in Path(__file__).parent.glob("*"):
        if path.is_file() and path.suffix in (".py",".sh"):
            sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    out.mkdir(mode=0o700)
    ev = out / "carla25"
    ev.mkdir(mode=0o700)
    (ev / "allocations").mkdir(mode=0o700)
    atomic_json(out / "plan.json",plan)
    atomic_json(out / "train_sources.json", sources)
    atomic_json(out / "config.json", config)
    atomic_json(out / "preflight.json", read(prev / "preflight.json"))
    atomic_json(ev / "source_sha256.json",read(prev / "carla25/source_sha256.json"))
    manifest = training_metadata(config)
    same_inputs = lambda m: {k:v for k,v in m.items() if k != "runtime_source_sha256"}
    if same_inputs(manifest) != same_inputs(state["dataset"]):
        raise ValueError("Training data/labels changed while forking batch size")
    # Reuse the trained model, optimizer/scheduler and per-rank RNG. The new
    # sampler seeks to the exact unconsumed sample in this epoch.
    state.update(config=config, dataset=manifest,
                 progress={"epoch": switch["epoch"], "next_batch": 0, "updates": switch["prior_optimizer_updates"]})
    report = commit(out / "latest.pt", state)
    atomic_json(out / "batch_switch.json", {**switch, "migrated":report, "remaining_updates":remaining})
    print("EGO128_BOOTSTRAP", json.dumps(switch), "remaining_updates", remaining, flush=True)


if __name__ == "__main__":
    main()
