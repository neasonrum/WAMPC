# IR-WM adaptation for TransFuser++ / Bench2Drive

This is a native PyTorch B2D adaptation of IR-WM's action-conditioned, recurrent
residual latent-world prediction idea. It does not reproduce the original
nuScenes OCC detector, its legacy MMCV environment, or its published metrics.
Implementation lives in WMMPC/method/irwm_bev; IR-WM/b2d provides an entrypoint.
The existing Full Diffusion planner (including its 100 denoising steps) is unchanged.

## Representation and action contract

* Frozen teacher: exactly WMMPC/ckps/full_diffusion/backbone/all_towns_seed1/model_0030_1.pth.
* Unaugmented existing RGB/LiDAR preprocessing; teacher runs in bf16 autocast.
* Latent BEV is 64 channels x 64 x 64 cells, extent [-32,32] m on both axes.
  Width is x forward, height is y right. Ego is at the center for this config.
* Three observations (current and previous two 0.25 s frames) are pose-aligned to
  the current ego frame. All future predictions and targets use that same frame.
* The planner supplies ten geometric XY points and one target speed. Its point
  index is not a timestamp. Current actual speed is a separate input.
* Actions encode the full path, displacement, current/target speeds, and query
  XY plus elapsed arrival time. Path cross-attention conditions a convolutional
  recurrent residual predictor. Rollout feeds its own predicted BEV back in;
  future GT features never enter the model forward call.
* Training uses the actual executed path. Spatial goals 2.5,3.5,...,11.5 m select
  nearest real future observations inside a 6 s window. Query inputs contain
  those observations' actual XY and elapsed times. Duplicate/unavailable frames
  are masked; feature targets are never temporally interpolated. Slow/dwell
  samples query actual future observations at varying elapsed times, including
  unchanged ego positions. Interpolation is used only to describe the action path.
* True future yaw/pose is used solely to warp the teacher targets, not as a hidden
  future state input. Grid validity masks exclude zero padding and partial cells.
* `ActionBEVWorldModel.rollout` returns ten dense BEVs for supplied planner nodes.
  Explicit estimated arrival times are preferred. Its fallback average-speed
  timing is an approximation, rejects stopped trajectories and times >6 s.
  After a control tick, rebase the unexecuted path using actual odometry; do not
  automatically remove its first geometric node after a single 0.05 s tick.

## Split, loss, and selection

Index: `data/irwm_bev/pdm_seed901_dev44_v1`. The 44 fixed dev routes were drawn from
the raw driving corpus, not from the official 220 evaluation routes. Source route
families (including weather/repetition variants) are excluded from training.
Available source XMLs are compared with official route scenario triggers using
town/type and 5 m distance; matched families are excluded. The split records
source templates that could not be mapped: those cannot become dev routes, but
may remain training data. This is not a claim of a comprehensive geographic
overlap audit for the pretrained teacher or all raw training data.

Validation averages valid queries per sequence, then sequences per route, then
the 44 route scores equally. `best.pt` minimizes that route-macro normalized
feature MSE. Also report raw MSE, cosine similarity, 3 m path-corridor error,
per-query/route metrics, and copying current BEV as a baseline. Training loss is
NMSE + 0.5 corridor NMSE + 0.05 (1 - cosine). No OCC/semantic supervision.
A better feature score is not a guarantee of better downstream closed-loop DS/SR.
Counterfactual paths are accepted as actions, but factual logged training alone
does not provide ground truth or validated accuracy for unobserved actions.

## Environment and training

Python: `../ENV/wmmpc-irwm-bev/bin/python`, a separate venv with explicit read-only
package reuse from `wmmpc-full-diffusion`. Its own installed packages take precedence.
Base environments are unchanged. This supports Full Diffusion and the B2D adaptation;
it does not claim support for the old nuScenes MMCV project. Dependency lock and
environment provenance are saved in the new venv.

Initial budget: 24 epochs, 8 GPUs, 4 examples per GPU, global optimizer batch 32.
AdamW lr 2e-4, weight decay .01, 500-update warmup and cosine decay to 10% lr.
Teacher chunking does not change the optimizer batch size. Two loader workers/rank,
one CPU compute thread/rank. Teacher features are cached as bf16 uint16 arrays
atomically, keyed by route/frame plus checkpoint/config hashes and preprocessing
version. Fresh and cached tensors use identical bf16 rounding. A full cache is
bounded by the indexed raw corpus (approximately 270 GiB), not stored in RAM.

Launch only after `battach JOB`; do not submit or resize jobs:

```bash
cd /fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC
../ENV/wmmpc-irwm-bev/bin/python -m method.irwm_bev.launch \
  --job JOB --phase train --output results/irwm_bev/train_20261101 --tmux
```

Run on all four existing allocations (two GPUs each): 13053006, 13047896,
13047898, 13046830. The launcher checks cgroup and host, uses allocation-visible
logical GPUs and a named tmux session, and records launch/status logs.
Acceptance uses a separate output and port; it tests actual updates, continuation,
replica consistency, target-cache equality, geometry, masks, action intervention,
ten-node inference, and one validation sample from each of the 44 routes.

Checkpoint policy:

* `latest.pt`: full model, optimizer, scheduler, progress, rank RNG states,
  config/split/teacher provenance; first update, every 250 updates, every epoch.
* `latest.previous.pt`: previous complete checkpoint retained atomically.
* `epoch_NNNN.pt`: model/config/metrics every epoch, without old optimizer copies.
* `best.pt`: full checkpoint for best validation epoch, retaining optimizer.
* bf16 autocast uses no GradScaler; parameters and optimizer remain fp32.
* Resume all four nodes with the same command plus `--resume`. Progress includes
  the exact next sampler batch. Config, dataset and world size must match.
* A file named `STOP` inside the run directory requests a synchronized stop after
  the current update and full checkpoint. It does not cancel LSF allocations.

Record measured throughput after caching warms up before estimating final runtime.

## Runtime recovery, 2026-11-01

The world-model package sets WMMPC_CARLA_NAVIGATION_PATH to the byte-identical
workspace copy in third_party/carla_navigation_0_9_15, avoiding the simulator
installation mount in spawned data workers. Full Diffusion uses its existing
default unless this optional variable is set. Validation uses no subprocess
workers; rank*.stage.json records validation progress. A complete latest.pt is
now also saved immediately before validation so a validation failure loses no
completed optimizer updates. Recovery details and original logs are preserved
in the run recovery_20261101_1250UTC directory.

The same node also lost access to the raw dataset mount. All 514,365 potentially
required frames (frame index >=8 across the indexed routes) are therefore audited
and cached under WMMPC. The cache completion tool fills only missing frozen-teacher
features on healthy allocations. Once complete.json exists, a missing feature
raises an explicit error instead of falling back to the raw mount. This does not
change labels, split, model, optimizer, batch size or validation coverage.
