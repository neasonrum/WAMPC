# Timed action-conditioned BEV world model

This new module trains from random initialization for 24 epochs. It leaves
`irwm_bev`, the V3 planner, and the original IR-WM repository unchanged.
It adapts IR-WM's fixed-cadence displacement-conditioned recurrent prediction
idea; it is NOT an identical reproduction of its nuScenes architecture or OCC task.

## Contract with V3

V3's normalized diffusion state is B x 16 x 2: ten spatial positions followed by
six temporal displacement increments. Use the checkpoint's codec.decode first.
The deployed policy.last_temporal_delta is ALREADY decoded in metres and can be
passed directly to this model. Do not difference it again, swap axes, rotate each
increment into a future ego frame, or treat ten geometric points as timestamps.

`model(history, history_valid, temporal_delta, speed)` takes raw frozen TF++ BEVs
at -0.5, -0.25, 0 seconds, pose-aligned to the current anchor; B x 6 x 2 actions;
and current measured speed. It cumulatively sums the increments internally.
The auxiliary target-speed value is derived exactly as in V3: norm(delta[1])/0.5.
There is no future GT speed, steering, command, yaw, or image in model.forward.

Output: B x 6 x 64 x 64 x 64, at 0.5, 1, 1.5, 2, 2.5, 3 seconds. All maps remain
in the INPUT observation's ego frame: width=x forward, height=y right, +/-32 m.
Future true poses are used only to construct aligned teacher targets and masks.
The same frozen teacher/cache identity as V3 is preserved; no OCC supervision.

`interface.py` contains explicit conversion helpers. To rebase old actions,
cumulatively sum, transform endpoints with inverse(T_current) @ T_source,
then difference if increments are needed. Time advancement is a separate step;
this training module does not implement the asynchronous planner/cache yet.

Reference checked: IR-WM/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py
uses cumsum(plan_traj) for cumulative displacement; its world head uses the latest
increment as an action. Its original coordinate/backbone/occupancy conventions are
not copied into this TF++ feature-prediction adaptation.

## Dataset and validation

Reuse the original split.json byte for byte, including its 44 development routes
and route-family/official-evaluation exclusions. Filter existing anchors to require
all observations from -0.5 through +3 seconds continuously, instead of padding
missing future targets or temporally interpolating features. Future targets use
raw-frame offsets 2,4,6,8,10,12 at the verified 0.25-second recording cadence.
The actual post-filter sample/route counts are in index/manifest.json.

The completed bf16 frozen-teacher cache is read only. No backbone is loaded during
training and there is no fallback to unavailable raw sensor mounts. Target tensors
are never fed back into the recurrent predictor. Model selection minimizes the
44-route macro NMSE across all six horizons, also reporting corridor NMSE, MSE,
cosine, each future time, and the copy-current baseline. Validation runs every epoch.

## Training and memory

Authorized allocation: 13053006, rng-dl01-w005, two logical A100 80GB GPUs.
Old batch: 8 GPUs x 4 = 32. New batch: 2 GPUs x 4 x 4 accumulated microbatches = 32.
Loss is divided by four; DDP no_sync covers the first three microbatches; clipping,
AdamW.step and scheduler.step occur once after the fourth. AdamW lr 2e-4,
weight decay .01, 500-update warmup, cosine decay, bf16 autocast/FP32 optimizer.

Measured CPU affinity is 8 total CPUs, despite the LSF request listing 16 slots.
Start with two loader workers per rank, prefetch_factor=2, and one compute thread
per rank. Approximate queued raw BEV payload: 2 ranks x 2 workers x 2 prefetched
batches x 4 examples x 9 frames x 64^3 x 2 bytes = 144 MiB (plus copies/processes).
No in-memory feature corpus is retained. The job cgroup RAM limit is 250 GiB;
reclaimable file cache and anonymous memory are logged separately. A conservative
unreclaimable-memory guard requests a checkpointed stop before exhaustion.

Launch in a named tmux session using the existing wmmpc-irwm-bev environment,
after battach 13053006. The launcher checks cgroup, GPU idleness, actual affinity,
RAM, and LSF expiration. It stops after an update and saves 25 minutes before
allocation expiry. It never submits, cancels, or replaces allocations.

## Checkpoint and recovery

All new indexes, logs, snapshots and checkpoints go to the configured run under
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC. Existing teacher data is reused.

latest.pt is atomic and includes model, AdamW state, scheduler, both ranks' RNG,
epoch and next microbatch, exact config, and dataset/split/teacher provenance.
It is saved after update 1, every 250 updates, before validation, every epoch,
and on graceful stop. latest.previous.pt retains the previous complete checkpoint.
epoch_XXXX.pt keeps per-epoch weights/metrics without old optimizer copies;
best.pt retains a full resumable checkpoint. Resume with the same launch command
plus --resume. Resume only at a completed accumulation boundary.

Acceptance checks actual two-GPU accumulated updates, bit-level checkpoint
continuation tolerance, optimizer step counts, replica agreement, action effects,
six-step inference, V3 codec/real-label agreement, coordinate transforms, cache
equality, and one validation example from each of the 44 held-out routes.

## Throughput tuning, 2026-11-06

Continued the same run at optimizer update 4597, with unchanged global batch 32,
AdamW moments, scheduler, seed, sampler order, split, targets, and 24-epoch budget.
Current runtime: two GPUs x batch 16, accumulation 1, workers 3/rank, prefetch 4.
Batch 4/8/16 and loader settings were benchmarked on the same checkpoint and
samples; benchmark updates were discarded. Warm baseline 0.1971 s/update,
selected config 0.1277 s/update (1.544x throughput, 35.2% less update time).
Higher batch shapes can change BF16 rounding; this preserves optimizer batch
semantics, not bitwise identity of the future training trajectory.

Runtime changes require --resume --allow-runtime-change. Only microbatch size,
accumulation, workers and prefetch may change. Effective batch must remain 32.
The sampler offset is converted through consumed examples, not reused as a raw
batch index: 18388 old microbatches x 4 = 4597 new batches x 16 per rank.
The next ten update groups were checked for identical sample membership/order.

The launcher now runs immutable source_revisions/<hash> snapshots and stores
per-launch frozen configs under launches/. The original source/ snapshot,
pre-tuning checkpoint and benchmark reports are retained in the run directory.
The read-only cache is large and OS file pages are reclaimable; model/loader
anonymous memory and LSF process memory must be distinguished from page cache.
Raw prefetched feature payload is bounded at ~1.69 GiB for this configuration,
plus transient copies and worker processes. No whole-corpus RAM cache is used.
