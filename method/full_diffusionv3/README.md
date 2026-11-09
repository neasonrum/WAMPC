# Full Diffusion V3: Independent Spatial and Temporal Queries

Status: train-only V3 normalization has been prepared on 276,547 retained
training samples. The current architecture is `independent_dual_query64` with
the V3-only `automot_temporal_pid_v1` controller. Focused CPU checks cover
cross-query isolation, AutoMoT target/loss/controller equivalence, the actual V3 update, persistent
workers, independent loader RNG, complete checkpoints, and exact subsequent
updates across a fresh-model mid-epoch resume and an epoch boundary. The
four-node controller requires real eight-rank sensor-data resume acceptance
before formal training. Per-run acceptance and training status are recorded
in the selected output directory. Existing experiment results are preserved.

## Implemented Scope

The following checklist describes the independently versioned V3 method.

1. Keep the existing ten spatial path labels. Replace the ONE diffused speed
  scalar with SIX consecutive 0.5-second XY displacement increments.
2. The diffusion state is `[B, 16, 2]`. Encode every point into 64 features.
  Independently compress the spatial `10 * 64` and temporal `6 * 64` features
  into two separate `[B, 1, 256]` queries. There is no cross-query attention.
3. Distinguish spatial positions from temporal increments with fixed slot
  order and additive type/slot embeddings. Keep per-group normalization.
4. Add a clean-observation behavior head: three horizons (1/2/3 seconds),
  four classes (STOP/DECELERATE/KEEP/ACCELERATE). Its predicted probabilities
  condition denoising; labels are supervision only, never inference inputs.
5. Supervise spatial coordinates, temporal increments, and behavior separately.
  Missing future data must not become a fabricated stationary/STOP target.
6. For control, keep spatial points for steering and compute target speed as
  `norm(temporal_delta[:, 1]) / 0.5`. No separate diffused speed slot remains.
  Keep per-tick replanning and spatial steering; use AutoMoT longitudinal PID
  and its final 35 km/h speed cap in V3 only, not the Garage oil/throttle map.
7. Remove the old sampler's final-coordinate zeroing and its 21-coordinate
  loss denominator. All 32 V3 coordinates are real supervised quantities.
8. Use a new method ID, configuration, normalization, and checkpoint format.
  Do not resume old planner/optimizer checkpoints as if they were compatible.
  Existing V3 sixteen-slot normalization is reusable because its labels did not change.
  Reuse the frozen perception checkpoint and compatible observation helpers.

Motivation: supervise how far the vehicle should move over future time
intervals, and explicitly teach longitudinal decisions, while retaining the
independent spatial and temporal query structure. Better closed-loop performance remains
an experimental hypothesis, not a promised outcome.

## Training Runtime

Use `coordinator.py --gpu-authorized --detach --output <new-run-directory>`
from the login node after explicit approval. It uses only the original four
two-A100 LSF jobs (13053006, 13047896, 13047898, 13046830), freezes the planner
and Garage Python sources, and verifies each original GPU UUID/allocation.
Preflight CUDA work runs in finite children, never in the parent controller.

After a stopped run, newly authorized continuation uses
`coordinator.py --gpu-authorized --resume --detach --output <existing-run-directory>`.
It preserves old attempts, starts at the next attempt number, and retains a
finite three-recovery budget. The current control layer launches the original
frozen trainer/config/checkpoint, with frozen CARLA navigation explicitly on
the rank and worker `PYTHONPATH`; supervision and optimizer state are unchanged.

Current plus two consecutive past observations feed the temporal BEV fusion
and gate. The persistent spawn loader reads the verified sensor snapshot,
shares the augmentation epoch, and has its own RNG generator. Future raw
measurements supply targets only; missing endpoints remain masked.

GPU gradients use NCCL. Data-fetch failures, finite-loss checks, stop votes,
statistics, rank RNG exchange, and checkpoint outcomes use an independent
CPU/Gloo group. NCCL/attempt failure aborts the attempt without saving fresh
uncertain state. Complete checkpoint generations have SHA256, fsync, an
atomic completeness index, and a retained previous generation. Every epoch
is saved before validation and revalidated on resume when necessary.

Each rank records phase entry and host timings in a fixed 512-event node-local
shared-memory ring. CUDA event timings are read only when complete, without
added per-stage synchronization. An independent CPU process samples rank
heartbeats, worker/process states, job memory/events, and memory pressure every
five seconds; it exports at most 72 recent samples every 30 seconds. Failure
directories retain full bounded rank traces and node logs.

One global controller permits at most three recovery attempts. It keeps every
torchrun `--max_restarts=0`, verifies all node/rank/loader groups exited and the
original allocations remain idle/RUN, validates the last complete model,
optimizer, RNG, config and dataset, then starts all four together. Diagnostics
and recovery reduce downtime; they do not prove or fix an underlying storage,
memory, hardware, or communication stall. Historical memory peaks alone are
not evidence of failure-time causality.

## Motivation

The current CARLA Full Diffusion target contains ten geometric path points and
one target-speed scalar stored as `(speed, 0)`. The path does not specify when
the vehicle should reach each point. Longitudinal behavior therefore depends
on a single current speed target, while spatial ADE/FDE do not measure the
quality of braking, stopping, or the executed motion.

This is a hypothesis about a useful improvement, not an established cause of
the earlier closed-loop regression. In particular, the existing approximately
1 Hz telemetry does not establish that FD2 has more speed jitter. Jointly
diffusing a speed scalar is also supported by BridgeDrive's LEAD adaptation;
the representation is not inherently invalid.

V3 borrows temporal-motion supervision from AutoMoT without adopting its large
action transformer or its point-level attention queries. The intended changes
are:

- Replace the diffused speed scalar with six future displacement increments.
- Keep ten spatial path points for steering.
- Encode each of these sixteen 2D quantities into 64 features.
- Independently compress the spatial and temporal features into two 256D queries.
- Use separate decoder stacks and time/behavior projections, with no exchange
  of queries, predictions, or BEV sampling references between the branches.
- Predict longitudinal behavior from clean observations and use the predicted
  probabilities as an additional denoising condition.

Keep pure Gaussian initialization, VP diffusion, 100-step DDIM inference, the
frozen perception backbone, and the existing sensor interface. Do not add a
world model, anchor trajectories, or previous-plan initialization.

## 1. Output Contract

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `spatial_path` | `[B, 10, 2]` | Current ego-frame geometric path |
| `temporal_delta` | `[B, 6, 2]` | Consecutive 0.5-second displacement increments |
| `future_positions` | `[B, 6, 2]` | Predicted cumulative positions; target endpoints have their own mask |
| `behavior_logits` | `[B, 3, 4]` | Longitudinal classes for horizons 1, 2, 3 seconds |
| `diffusion_target` | `[B, 16, 2]` | Spatial points followed by temporal increments |

The spatial labels keep their current definition: the first point is 2.5 m
from the planning origin and consecutive labels are 1 m apart in chord
distance. These are label-construction rules, not hard output constraints.

Let `future_positions[:, k]` correspond to `(k + 1) * 0.5` seconds. With the
planning origin at `(0, 0)`, define:

```text
delta_1 = position_at_0.5s - planning_origin
delta_2 = position_at_1.0s - position_at_0.5s
...
delta_6 = position_at_3.0s - position_at_2.5s

diffusion_target = concat(spatial_path, temporal_delta, dim=1)
```

There are six increments, not five. They encode displacement, not six
instantaneous speeds. All positions and displacement vectors use the SAME
current planning coordinate frame; increments are not expressed in successive
future vehicle frames.

The original target-speed scalar is not a seventeenth diffused slot. A separate
future-speed auxiliary head is optional follow-up work, not required for V3.

## 2. Two Independent Queries, 64 Features Per Point

Use the existing sine/cosine coordinate embedding with `hidden_dim=64`. It
returns 64 features for each 2D input, not 64 per coordinate.

```python
point_features = gen_sineembed_for_position(noisy_normalized, hidden_dim=64)
point_features = point_features + type_embedding(type_ids)[None]
point_features = point_features + slot_embedding(slot_ids)[None]
path_query = path_query_encoder(point_features[:, :10].flatten(1)).unsqueeze(1)
temporal_query = temporal_query_encoder(point_features[:, 10:].flatten(1)).unsqueeze(1)
```

| Stage | Shape |
| --- | --- |
| Noisy normalized target | `[B, 16, 2]` |
| Coordinate embedding plus type/slot embeddings | `[B, 16, 64]` |
| Flatten spatial points | `[B, 640]` |
| Flatten temporal increments | `[B, 384]` |
| Independent Linear, ReLU, LayerNorm, then unsqueeze | two `[B, 1, 256]` tensors |

Use two type IDs: `SPATIAL_POSITION` for slots 0-9 and `TEMPORAL_DELTA` for
slots 10-15. Use sixteen distinct slot IDs. Each type/slot embedding has
64 features and is ADDED, not concatenated. The temporal slot IDs represent
fixed future intervals. Neither branch contains a speed-placeholder point.

Fixed slot order already distinguishes the quantities after flattening;
type/slot embeddings make the identity explicit. There is no point-level or
cross-query self-attention. Each branch has its own decoder, time MLP and
behavior projection. The clean scene/BEV context is shared, but neither branch
reads the other branch's query, predicted points or BEV sampling references.
CPU tests require both cross-input Jacobians to be exactly zero.

The diffusion timestep embedding remains a separate 256-dimensional condition.
Diffusion noise time is NOT the future driving horizon. The type and slot
embeddings are model parameters, never noised training targets.

Each spatial decoder predicts 20 coordinate residuals and each temporal decoder
predicts 12 displacement residuals. Concatenate only their output states for
the diffusion sampler. Preserve iterative refinement and per-layer supervision.

## 3. Labels and Causality

Reuse the route-level train/validation split and current-frame spatial labels.
Build future motion targets from raw expert measurements in the SAME route,
not from neighboring entries in the filtered training index.

- Confirm timestamps and the raw measurement cadence when preparing labels.
  With the current 0.25-second cadence, temporal positions use raw offsets
  `+2, +4, +6, +8, +10, +12`; do not assume this for another dataset.
- Transform future world positions with the current ego transform and the same
  current-frame geometric augmentation used for spatial labels.
- Set the planning origin to zero in that augmented frame. Do not apply each
  future frame's independent augmentation to its label.
- Future positions and future expert speeds are training targets ONLY. Never
  load future RGB, LiDAR, BEV, or labels into the conditioning path.
- Retain causal current/past observations for the temporal BEV adapter. Reuse
  the FD2 history convention: current observation plus two past observations.
- Emit position, increment, and behavior validity masks for missing futures.
  An increment is valid only if both endpoints are valid, with the current
  planning origin considered valid. Do not bridge a missing interval.
- Numerical padding for invalid targets is allowed only with the associated
  masks. Exclude it from normalization statistics and all relevant losses;
  do not interpret it as zero motion or a STOP label. Masks are not policy
  conditions at inference. Handle an all-invalid loss group without NaNs.

Compute new per-slot/per-coordinate normalization statistics from valid,
retained TRAINING targets only. Spatial positions and temporal increments
have different distributions. Do not reuse the old `[11, 2]` statistics or
estimate them from validation data. Store the statistics and label-generation
settings with the V3 checkpoint.

The raw files inspected here have no timestamp fields. Their sensor checkpoint
configuration records `carla_fps=20` and `data_save_freq=5`, giving 0.25 seconds
per saved measurement. The dataset and preparation entry point reject a cadence
mismatch with that configuration; this is not a per-route timestamp audit.

Target `future_positions` stores each available transformed world endpoint
directly. Do not reconstruct ground truth by cumulatively summing padded
invalid increments. Predicted increments are always cumulatively summed for
time-position metrics, evaluated only at valid target endpoints.

## 4. Clean Longitudinal Behavior Head

This is a proposed lightweight alternative to AutoMoT's decision-text interface,
not a literal copy of its token decoder.

Use four class IDs in fixed order:

```text
0 STOP
1 DECELERATE
2 KEEP
3 ACCELERATE
```

Use measured expert speeds at `0, 1, 2, 3` seconds, not the expert's desired
target speed. For each one-second interval, assign STOP first when its endpoint
speed is below `stop_speed_mps`; otherwise use the speed change from the previous
endpoint to distinguish DECELERATE, KEEP, and ACCELERATE.

Initial configurable thresholds: `stop_speed_mps=0.3` and
`speed_change_mps=0.5`. These are proposed starting values, not tuned constants
or verified AutoMoT thresholds. Record class counts and inspect their semantics
before training. Do not automatically use inverse-frequency class weights.

STOP at a future horizon does not imply that the vehicle is already stopped or
should immediately apply full brake. Likewise, an expert brake command while
slowing from 10 to 8 m/s is not itself a STOP label.

Build a small trainable MLP from clean planner context: pooled BEV features,
pooled memory/ego features, and normalized current speed, navigation command,
and target point. It outputs twelve logits reshaped to `[B, 3, 4]`. No future
target, noisy trajectory, or diffusion timestep enters this classifier.

Compute softmax per horizon, flatten the twelve probabilities, and project to
256 dimensions through separate additive projections for each branch. The
zero-initialized projections introduce this condition gradually. Train the classifier with
masked cross-entropy; use PREDICTED probabilities for denoising during both
training and inference. Feeding ground-truth classes during training would
create an avoidable train/inference mismatch.

Compute this clean condition once per observation, outside the DDIM loop. The
behavior head does not directly override throttle or brake in the initial V3.

## 5. Diffusion and Loss Changes

The existing [codec and sampler](../full_diffusion/diffusion.py) are specific to
`path + (speed, 0)` and cannot be reused unchanged:

- Replace the `[B, 11, 2]` state with `[B, 16, 2]`.
- Remove EVERY `sample[:, -1, 1] = 0` and `noisy[:, -1, 1] = 0` operation in
  the V3 implementation. That coordinate is now a real lateral displacement.
- Noise and denoise all 32 continuous coordinates. Do not diffuse class IDs.
- Split decoded metric outputs into ten positions and six increments.
- Replace the old loss that omits the final dummy coordinate and divides by 21.
- Keep the VP schedule, pure Gaussian start, x0 prediction, and 100 DDIM steps.

Proposed initial objective:

```text
loss = sum_over_decoder_layers(spatial_L1 + 2.0 * temporal_delta_L1)
       + 0.4 * behavior_CE
```

Compute the spatial and temporal losses separately in metric units and with
their own valid-element denominators. Weight temporal intervals by
`[3, 3, 2, 2, 1, 1]`, normalizing by the sum of valid weights. Compute behavior
loss once per observation, not once per denoising decoder layer. Start with
equal horizon/class weights for cross-entropy unless the label audit motivates
a change. These coefficients are design defaults, not demonstrated optima.

Log each loss independently. Also distinguish geometric path errors, cumulative
time-position errors, early-interval speed error, and per-class behavior metrics.
If adding a derived-speed loss later, compare predicted displacement speed to
the SAME expert interval-displacement speed, not to an instantaneous speed at
the interval endpoint.

## 6. BEV Sampling and Online Control

The spatial decoder samples BEV at its own ten geometric points. The temporal
decoder independently samples BEV at its own six cumulatively summed future
positions. A temporal increment is NOT an absolute BEV location. Neither branch
uses the other's reference points.

At inference:

```python
spatial_path, temporal_delta = codec.decode(sample)
future_positions = temporal_delta.cumsum(dim=1)
target_speed = temporal_delta[:, 1].norm(dim=-1) / 0.5
```

The selected second increment represents the 0.5-1.0 second interval. It gives
a scalar average-displacement speed, not a commanded acceleration or a required
arrival speed at a spatial checkpoint.

Feed `spatial_path` to the existing lateral controller. Feed the raw continuous
derived speed to the AutoMoT longitudinal PID: gains `(1.75, 1.0, 2.0)`, history
20, positive speed-error clip `[0, 1]`, throttle clip `[0, 1]`. Brake when the
reference speed is below 0.4 m/s or current/reference speed exceeds 1.1. The
speed PID updates its history even on braking ticks, as AutoMoT does. The old
Garage polynomial throttle map and its internal 1.0313 overspeed ratio are not
used by V3. Compatibility speed-bin outputs remain for the SensorAgent return
format only and cannot alter longitudinal control. Predicted future behavior
classes do not hard-override brake or throttle.

Continue replanning every normal simulation tick. Do not execute a full
0.5-second motion segment before replanning. V3 applies AutoMoT's final 35 km/h
brake cap after the inherited emergency/creep handling. Do not import parking
escape or other unrelated heuristics. The gate-BEV and earlier method controllers
are unchanged. This matches AutoMoT longitudinal control, not its entire agent.

## 7. Implementation Map

All new behavior belongs under this independent method directory. Existing
methods remain unchanged. Reuse compatible utilities rather than copying an
entire training stack.

| Implemented file in this V3 directory | Behavior |
| --- | --- |
| [data.py](data.py) | Reuse current/past observation loading; add six raw-future displacement labels, three behavior labels, validity masks and training-only normalization preparation. Future data is targets only. |
| [diffusion.py](diffusion.py) | Implement the `[B,16,2]` codec and sampler; remove speed-placeholder masking; retain the VP schedule and all 32 noise coordinates. |
| [model.py](model.py) | Encode each point with 64 features; independently compress 10 spatial and 6 temporal points into two 256D queries and decoder stacks; add the clean behavior condition. |
| [train.py](train.py) | Wire the dataset/codec/head; compute masked losses separately; log components, geometric and time-position errors, six interval speed errors and behavior confusion. Store V3 statistics and source snapshots. |
| [inference/policy.py](inference/policy.py) | Decode ten spatial points and six increments; derive continuous speed from increment index1; use AutoMoT longitudinal PID and ignore legacy speed probabilities for control. |
| [inference/carla_agent.py](inference/carla_agent.py) | Load only V3 planner checkpoints; preserve observation history, raw-BEV detection, sensor return format and per-tick control. |
| [config.json](config.json) | Record architecture/controller IDs, 10+6 layout, 64D point encodings, two independent 256D queries, future cadence/horizons, labels, normalization, losses and 100 DDIM steps. |

CPU checks cover branch isolation, real update gradients and numerical agreement
with the local AutoMoT offset/loss/PID source. The shared V1/V2 modules have not
been changed. New encoder/decoder shapes require a fresh training run; old
single-query V3 model/optimizer states must not be loaded into this architecture.

Useful existing references: [dataset](../full_diffusion/data.py),
[denoising head](../full_diffusion/model.py),
[temporal adapter](../full_diffusion2/model.py),
[frozen perception](../full_diffusion2/perception.py), and
[online history/control wrapper](../full_diffusion2/inference/policy.py).

Old planner checkpoints are not shape-compatible with this target/encoder.
Reuse the frozen perception weights, but do not silently load old planner,
optimizer, or normalization states. Freeze source/config identities before any
eventual launch; do not alter a running experiment's imported sources.

This revision does not add per-point transformer queries or change the perception
backbone. It separately records the independent-query architecture and the
AutoMoT longitudinal adapter. No improvement in closed-loop score is assumed
or claimed by this implementation.

## 8. Entry Points and Current Limitations

The following CPU-only preparation command is for a new index/label definition.
The existing verified sixteen-slot V3 artifact is reused for this revision:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  ../ENV/wmmpc-full-diffusion/bin/python \
  -m method.full_diffusionv3.data --workers 4
```

It reads only retained training targets, writes the configured V3 normalization
artifact, records valid slot/class counts and index hashes, and refuses to
overwrite an existing artifact. It does not load RGB/LiDAR or request a GPU.
Codec fallback scales are construction defaults, not audited statistics;
training requires the prepared artifact and checks its identity and hashes.

The training entry point is `method.full_diffusionv3.train`. It requires an
explicit `--gpu-authorized` flag, `--output`, eight distributed ranks/global
batch64, and the existing designated LSF job/host guards. The coordinator freezes
the sources, checks the original four allocations and runs real eight-rank
resume acceptance before training. Every launch requires explicit user approval;
do not submit replacement GPU jobs or touch unrelated workloads. Resume compares the saved full
configuration/dataset identity and strictly loads V3 model/optimizer states.

The CARLA entry point is [inference/carla_agent.py](inference/carla_agent.py),
with `get_entry_point()` returning `FullDiffusionV3Agent`. Its configuration
argument is a compatible dual-query V3 checkpoint file or a directory containing
`latest.pt`. Old single-query checkpoints are rejected. CPU equivalence is not
a closed-loop performance result; real eight-rank acceptance and formal training
status live in each new run directory. CARLA scores require a separately
authorized evaluation of the new checkpoint.