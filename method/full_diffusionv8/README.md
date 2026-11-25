# V8 joint world training with ego12

The reference is the friend's `full_diffusionv8` at commit
`e28d3fffeecfe39f52ea0972e069618dd0d33fea`. Tensor implementations and their
source identities are recorded in `donor.json`.

The planner uses the existing `full_diffusionv3_ego12` architecture, with ten
geometric points and six half-second XY increments. Each branch reads the
observed BEV and twelve TF++ ego/status tokens. The V8 sampler extracts 95
future tokens and adds gated readouts after the ego attention in each layer.

The planner starts randomly, as in the reference V8. TF++ is frozen. All BCDE
parameters start from the pretrained epoch-21 checkpoint and receive both
planning gradients and the independent expert-action BEV reconstruction loss.
The reconstruction branch has no path to planner parameters.

The filtered PDM anchors use the existing WM route selection. Its 44 development
routes are held out for checkpoint selection by route-macro future trajectory
ADE with 25-step DDIM. An epoch is the reference's fixed route-stream sample
budget, with streams continuing across epochs.

`config_joint.json` records the four authorized two-B200 allocations, ten epochs,
global batch 256, one local batch of 32 without gradient accumulation, and four
loader workers per rank. The allocator thresholds in `launch_job.sh` release
large temporary sensor-decoding buffers as workers continue reading frames.
Latest checkpoints contain optimizer, scheduler, committed route-stream state,
rolling predictions, and per-rank RNG. Epoch and best checkpoints contain the
planner and jointly updated WM weights.

After successful training, each allocation's driver runs two CARLA workers using
the selected `best.pt`. They share the existing dynamic 220-route queue and use
25 denoising steps. CARLA uses 20Hz observations and an exact five-tick-old plan,
with observed BEVs at k-10/k-5/k. Unsupported 3-second prior tails are masked.

Run `bash method/full_diffusionv8/launch_job.sh JOB_ID` inside the matching LSF
allocation and a named tmux session. Add `--resume` for an existing run. The
configured output directory contains `status.json`, `train.jsonl`,
`validation.jsonl`, `best.json`, `latest.pt`, and `carla25/summary.json`.
