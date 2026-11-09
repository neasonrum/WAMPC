# Full Diffusion2: Pure E2E Temporal Conditioning

This method adds two trainable paths to the unchanged Full Diffusion Geo baseline:

- Current plus two past frozen TF++ BEVs, spaced 0.25 seconds apart, are ego-motion aligned, channel-concatenated, convolutionally fused and conditioned by a channel gate. A residual projection modifies only the planner BEV input.
- Current speed, six-dimensional navigation command and goal XY are normalized and encoded to 128 dimensions. Concatenation with the 256-dimensional noisy trajectory query is projected 384-to-256 and added residually before decoding.

Both residual projections start at zero. The initial function matches the original planner. The frozen backbone, current memory/checkpoint-query context, raw-BEV CenterNet detector, Gaussian VP process, 100-step DDIM sampling and controller interface remain unchanged. No world model, future observation, prior-plan warm start or diffusion anchor is used.

## Training

Configuration: `config.json`. Same 276547 retained training frames and 15017 whole-route validation frames as the baseline; seed 901; 10 epochs; AdamW LR 3e-4; bf16; global batch 64 = 8 GPUs x 8 examples. The planner starts from scratch, not the baseline epoch 10 checkpoint.

Only these existing LSF jobs are authorized: 13053006 (w005), 13047896 (w002), 13047898 (w007), 13046830 (w010), two A100 GPUs each. `launch.py` checks the exact host, cgroup, scheduler allocation and GPU occupancy. Its CUDA probe is a finite subprocess; the durable parent controller never initializes CUDA.

`attach.sh` now defaults to read-only allocation inspection. It does not submit a second training run. To launch or resume, invoke `launch.py --phase train` inside the designated job with the appropriate output directory and explicit `--resume` when needed.

Production output: `WMMPC/results/full_diffusion2_geo_20261102_seed901/train`. Acceptance output is separate. Checkpoints are atomic, resume restores per-rank RNG and optimizer/scheduler state, and each epoch preserves its checkpoint. `progress.json`, `metrics.jsonl`, `validation.jsonl` and per-node logs describe training; only a closed-loop evaluation can establish DS/SR improvement.

`launch.py --workers-per-rank 4` overrides only the runtime training DataLoader, without modifying the canonical checkpoint configuration or its strict resume comparison. `runtime.json` and `runtime_history.jsonl` record actual workers and resume steps; metrics include a runtime session identifier so elapsed times are not mixed across restarts. Validation retains zero workers. Worker overrides must be passed on every resume; omitting the option uses the configured default of two.

The 2026-11-02 trial retained four workers per rank (32 total). A 300-update trial preserved steps 2670 through 2970; its final 200 updates averaged 0.467 s/update versus a clean 500-update two-worker baseline of 0.589 s/update, about 26% higher throughput. All four jobs exited normally with zero cgroup OOM events. Full training resumed at step 2970 with four workers and no trial limit. Structured evidence is in `results/full_diffusion2_geo_20261102_seed901/inspect/worker_trial/`.

For a finite throughput trial, add `--trial-updates 300 --resume` on all nodes. The trial performs real training updates, collectively saves its final checkpoint, writes `worker_trial_completed.json` and exits without marking the ten-epoch run completed. Resume without `--trial-updates` to continue the full run. Signal only a verified training rank to request a graceful collective checkpoint; do not terminate its controller or LSF job. Initial source hashes are retained separately before resumed code hashes are refreshed.

## Closed-Loop Inference

Entrypoint: `inference/carla_agent.py`, class `FullDiffusion2GeoAgent`. The checkpoint path may be an individual checkpoint or a directory containing `latest.pt`.

The agent supplies online UKF pose and simulator time. The policy caches already-observed raw BEVs and selects observations at or before the two requested historical times, with at most 75 ms extra age. Missing history is masked without waiting. Cache state is local to the agent/episode and clears if simulator time resets. Detection always receives the current raw BEV, not the temporal/gated planner BEV.

## Verification

CPU: `CUDA_VISIBLE_DEVICES='' ENV/wmmpc-full-diffusion/bin/python -m unittest discover -s WMMPC/tests -p test_full_diffusion2.py -v`.

Eight-rank acceptance checks real updates, frozen perception, nonzero temporal/gate/state gradients, replica agreement, exact next-update resume and finite 100-step Gaussian generation. It does not initialize production training from acceptance weights.