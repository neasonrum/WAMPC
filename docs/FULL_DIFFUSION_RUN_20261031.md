# Full Diffusion geo training run (2026-10-31)

## Current status

Production training is running in the two user-provided allocations, under named tmux sessions. The initial audit confirmed 520 real optimizer updates, finite loss, and a saved update-500 checkpoint. This document is a launch-time record; read `progress.json` for live progress. No CARLA closed-loop score has been measured.

Run directory: `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC/results/full_diffusion_geo_20261031_seed901`.
Python: `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/ENV/wmmpc-full-diffusion/bin/python`.

## Paper target and training length

The target is **Full Diffusion geo** in Table 2 of arXiv 2509.23589v4: DS 83.85, SR 67.27%, Efficiency 238.90, Comfortness 21.40. Appendix C.3 says models are trained for **10 epochs**, with frozen pretrained perception, AdamW, initial lr 3e-4, and cosine warm restarts T_0=10, T_mult=2. The paper reports about 10 hours on one H20. Table 7 gives **100 denoising steps** for Full Diffusion inference.

This run follows that 10-epoch schedule. On the current four A100s the recent update time is 0.387 seconds, implying about 4.59 remaining training hours before validation overhead at the recorded update. Allow roughly 5-6 hours including validation; filesystem load can change this. This is an early estimate, not a measured full-epoch time.

## Implementation status and remaining uncertainty

This is a **paper-based reimplementation**, not a verified copy of the authors' exact PDM-Lite Full Diffusion ablation. The public BridgeDrive checkout provides NAVSIM and LEAD adaptations, but no complete Garage/PDM-Lite Full Diffusion ablation configuration or trained head. The published target scores are reference values, not achieved results.

The full list of implementation choices is preserved in `run_config.json` under `assumptions`. Material choices include:

- Global batch 64 (16 per GPU), bf16, weight decay 1e-4, gradient clip 1.0, seed 901, and a deterministic whole-route 5% validation split. The paper does not specify these exact choices.
- Gaussian-start, anchor-free VP diffusion with beta_min=0.1 and beta_d=19.9, x0 prediction, L1 supervision, and 100 deterministic DDIM steps. beta_d=19.9 ensures an almost Gaussian endpoint; the paper's beta_d=2 is reported for its diffusion bridge. The exact Full Diffusion ablation schedule is unpublished.
- The public NAVSIM Full Diffusion head/attention structure is adapted to frozen Garage fusion features, 10 geometric path points plus a speed scalar, train-only per-point normalization, and a target-point token. These adaptation and speed-representation details are not confirmed author settings.
- Adjacent-frame critical-change filtering (speed >0.1 m/s or path angle >0.5 degrees; retain 14% of unchanged frames) and official Garage route-quality/repetition/initial-frame rules give 276,547 training frames and 15,017 validation frames over 5,060 usable routes.

To claim numerical reproduction of Table 2, complete the 220-route closed-loop evaluation and compare all four metrics. Validation L1/ADE is not a substitute for DS/SR. Differences from the target can require matching author preprocessing, sampler, controller and batch settings, rather than simply adding epochs.

## Backbone and fixes

The official all_towns seed-1 TransFuser++ checkpoint `model_0030_1.pth` is already present, so no duplicate download was made. This run rechecked SHA256 `0ff5062a234f5919385957c4b46df8a10adcf09a88a427198e0e2e655b5935cb` and strict state-dict loading. It is a perception checkpoint, not a pretrained Full Diffusion head. The official archive and provenance remain under `ckps/full_diffusion/backbone`.

The audit added `FrozenGaragePerception.from_net` for the evaluator's already loaded model and fixed controller forwarding so the official agent's uncertainty braking and slower-speed choices reach the controller. Original files are preserved in `results/code_audit_20261031`. The training source and config snapshot, hashes, and assumptions are in the run's `source_snapshot/` and `provenance.json`.

## Validation performed

- Ten existing Full Diffusion CPU tests passed, including a real Garage sensor frame, deterministic augmentation, normalization, all trainable gradients, VP schedule consistency and a 100-step oracle.
- Four-rank NCCL training acceptance passed on real data. All ranks agreed on parameter checksums, four distinct optimizer updates changed the head, the backbone stayed frozen, and the resumed next update had maximum parameter error **0.0**. Evidence: `results/full_diffusion_acceptance_20261031/acceptance_rank*.json`.
- CPU integration loaded the saved acceptance checkpoint with the real TF++ backbone, produced finite 100-step sampled path/speed and box outputs with the expected interfaces, and verified controller speed forwarding. Evidence: `inference_integration.json`. This is an interface check using an untrained head, not a driving score.
- Production passed update 500 and wrote a fresh checkpoint. It logs every 10 updates, checkpoints every 500, validates once per epoch, and preserves `epoch_01.pt` through `epoch_10.pt` plus `latest.pt`.
- One-route and 220-route evaluation plans pass all file/dependency readiness checks. Real CARLA startup/episodes are still unverified.

## Allocations and resource envelope

| Job | Node | Local GPUs | Ranks |
| --- | --- | --- | --- |
| 13047898 | rng-dl01-w007 | 0,1 (A100 80GB) | 0,1 |
| 13047896 | rng-dl01-w002 | 0,1 (A100 80GB) | 2,3 |

Both jobs remain RUN. No job was submitted, canceled, resized or replaced. Each job requests mem=16000 per slot across 16 slots; the live cgroup limit is 268,435,456,000 bytes (250 GiB) per host. Initial production memory was about 8 GiB per host and peak PyTorch GPU allocation about 1.26 GiB per rank. Thread pools and two data workers per rank are bounded; w002's actual CPU affinity exposes eight logical CPUs. The launcher uses allocation-visible GPU UUIDs, not physical IDs from LSF.

## Inspect and resume

From the run's project root:

```bash
cd /fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC
cat results/full_diffusion_geo_20261031_seed901/progress.json
tail -f results/full_diffusion_geo_20261031_seed901/metrics.jsonl
```

For the existing session, use `battach 13047898` (or 13047896) and then:

```bash
tmux -S /fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC/results/tmux_full_geo_13047898.sock attach -t full_geo
# On w002, use tmux_full_geo_13047896.sock instead.
# Detach without stopping the experiment: Ctrl-b, then d.
```

If training has stopped and both allocations are still RUN/idle, resume on **both** attached nodes in their existing tmux sessions, substituting that node's authorized JOB_ID. Do not launch a second copy while the current run is active:

```bash
../ENV/wmmpc-full-diffusion/bin/python -u tools/launch_full_diffusion.py \
  --job-id JOB_ID --phase train --resume \
  --config results/full_diffusion_geo_20261031_seed901/run_config.json \
  --output results/full_diffusion_geo_20261031_seed901 --master-port 29582
```

## Closed-loop evaluation after training

Wait for `completed.json` and use the fixed `epoch_10.pt` checkpoint for the final evaluation, or explicitly record a different validation-selected epoch. Run one smoke route before the benchmark. Inside one designated allocation with an idle selected GPU:

```bash
../ENV/wmmpc-full-diffusion/bin/python evaluation/bench2drive.py \
  --method full_diffusion --experiment geo_20261031_seed901_smoke \
  --seed 901 --slot 0 --gpu-rank 0 --routes-subset 1711 \
  --agent-config results/full_diffusion_geo_20261031_seed901/epoch_10.pt \
  --execute --job-id 13047898
```

`evaluation_smoke_plan.json`, `evaluation_bench220_plan.json` and `evaluation_worker0_plan.json` through `evaluation_worker3_plan.json` record ready plans against the current checkpoint path. Before final execution, change `latest.pt` to the fixed epoch checkpoint. The four worker plans contain disjoint 55-route shards: workers 0/1 on 13047898, workers 2/3 on 13047896, using local GPUs 0/1. They are plans only, not scheduled evaluation processes.

After all 220 unique routes complete, aggregate DS and SR from evaluator records and compute Efficiency/Comfortness from per-route metric traces with `tools/efficiency_smoothness_benchmark.py`. Keep route coverage, failure reasons, seeds and checkpoint identity with the result. Do not count training completion as benchmark reproduction.
