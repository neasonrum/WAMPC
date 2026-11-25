# Point-query E2E and future-BEV K/V source handoff

This branch snapshots the implementation evaluated on 2026-11-25. The source
workspace on the original server is:

```text
/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC
```

## Which code to read

| Path | Purpose |
| --- | --- |
| `method/full_diffusion_pointquery_cvci/` | Latest joint planner; shared sampled future token pool, independent spatial/temporal attention readers. |
| `method/full_diffusion_pointquery/` | Pure E2E baseline: one 256D query per geometric/temporal point, learned local current-BEV reads, ego12. |
| `method/full_diffusionv3_ego12/` | Shared planner codec integration, ego12 conditioning, controller and allocation checks. |
| `method/full_diffusionv8/` | Timed predecessor handling, joint WM wrapper, world dynamics, geometry and feature supervision used by the latest method. |
| `method/irwm_bev_t/` | Tensor-only timed-world implementation dynamically loaded by the V8 world dynamics. |
| `method/full_diffusion2/`, `method/full_diffusionv3/`, `method/full_diffusion/` | Frozen TF++ perception, observation history, diffusion/normalization and shared Garage integration. |
| `evaluation/`, `leaderboard/`, `scenario_runner/` | CARLA route queue, scoring, simulator lifecycle and benchmark scenarios. |
| `evaluation/pointquery_parallel/` | The pure E2E baseline's separately verified two-world CUDA broker; not enabled for the joint WM agent. |
| `data_protocols/pdm_dev44_fixed_v1/` | Exact frozen route/frame lists, future-label masks and hashes; no sensor data or BEV feature cache. |

Main entry points in `method/full_diffusion_pointquery_cvci/`:

- `model.py`: point-query planner and independent future attention per branch/stage.
- `future.py`: future-only dense projection, shared neighborhood sampling and gated readers.
- `joint_policy.py`: original E2E planning losses with WM conditioning.
- `train.py`: joint training, validation, best selection and resumable checkpoints.
- `policy.py`, `carla_agent.py`, `fast_agent.py`: exact 20Hz online interface.
- `config_joint.json`, `launch_job.sh`, `run_job.py`: original-server experiment settings and launch flow.
- `preflight.py`: real-data gradient ownership, online lag, baseline equivalence and full batch32 checks.

The navigation-command plus target-point embedding concatenated to each query
was only discussed after this experiment. It is **not implemented** here.
Current command/target conditioning and ego12 are the existing baseline paths.

## Model and gradient contracts

Both branches diffuse coordinates: 10 geometric XY positions and six XY
increments at half-second intervals. The latter cumulatively define 0.5–3.0s
positions. Every point retains a 256D hidden query. Current-BEV reads sample the
center plus four learned neighbors; sampling pools those neighbors without
collapsing trajectory points. Frozen TF++ provides observed BEV and ego12 tokens.

At time t, the detached plan from exactly t-0.25s is advanced using observed ego
pose and physical time interpolation. At 20Hz this is k-5, with observed BEV at
k-10/k-5/k. The unsupported sixth interval is masked. A fresh WM rollout at t
uses this advanced prior; no asynchronous future cache is introduced here.

Five usable maps at 0.5…2.5s use a separate Conv3x3+ReLU 64-to-256 projection.
The projection is shared within the future branch across time, but never with
current-BEV projections. Neighborhood budgets [9,9,16,25,36] produce up to 95
tokens. Both trajectory branches read this exact pool using separate Q/K/V,
geometry/time biases and channel sigmoid gates initialized at 0.01. Future
attention is inserted after ego12 attention and before the existing FFN.

Trajectory loss updates planner, future sampler and all WM parameters. Teacher
BEV reconstruction loss updates only WM parameters; it has no planner gradient
path. TF++ remains frozen. No additional speed loss is introduced in this method.

## Training and fixed validation

- Strict initialization from the completed point-query E2E epoch10.
- WM initialization from the existing timed BCDE checkpoint, with its own lower LR groups.
- Eight A100, local batch32, global batch256, no gradient accumulation.
- Ten epochs, 927 updates per epoch, 9270 total updates.
- 237351 training anchors over 4796 routes; 2223 validation anchors over 44 disjoint routes.
- Fixed eight validation shards, route-macro aggregation, 25-step DDIM.
- Select best checkpoint by route-macro temporal-trajectory ADE.
- Latest checkpoint stores optimizer, scheduler, per-rank RNG, route-stream cursor and rolling plans.

The route-stream ordering differs from the shuffled pure E2E run even though
their frozen train/validation rows are identical. This method also receives ten
additional fine-tuning epochs. A comparison to the original pure E2E checkpoint
therefore does not isolate the causal effect of WM alone.

## Selected results

Best checkpoint: epoch10, SHA256
`e8b426cc23c8150a8a6466a824c77619298d1679698b79417621ca18ce989114`.

| Fixed dev44 metric | Pure E2E epoch10 | Joint WM epoch10 |
| --- | ---: | ---: |
| Geometric ADE (m) | 0.0351996 | 0.0315247 |
| Temporal ADE (m) | 1.6475200 | 1.4916661 |
| Control-speed MAE (m/s) | 0.8350517 | 0.7672197 |

On the same 219 valid closed-loop routes, both with 25-step DDIM and seed901:

| Metric | Pure E2E | Joint WM |
| --- | ---: | ---: |
| DS | 86.1088 | 87.3749 |
| SR | 147/219 = 67.1233% | 154/219 = 70.3196% |
| Route completion | 94.9248% | 96.3647% |

Route3800 has no valid driving record: CARLA0.9.15 segfaulted around 75 simulation
seconds on four attempts. It is excluded, not assigned zero. Route23687 obtained
DS100 on the first non-crashing retry, but has a route-deviation infraction and
does not count as success under the existing upstream SR helper. Scored driving
outcomes were not retried or selected by score. There is no complete220 score.

The machine-readable [handoff manifest](pointquery_cvci_handoff_20261125.json)
contains original source hashes, configuration provenance, checkpoint identity,
validation history and the partial closed-loop summary. Its file hashes refer
to the server snapshot before adding these handoff documentation changes.

## External files and original-server paths

This source upload adds no checkpoint weights, raw sensor data, teacher-BEV
features, CARLA binaries or Python environments. The repository's older V3
checkpoint was already in main/history and is not the current method's weight.

Original joint output:

```text
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/full_diffusion_pointquery_cvci_sharedkv_8a100_20261124_v1
```

Use `best.pt` for the selected model and `latest.pt` for exact training resume.
The strict planner initializer is:

```text
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/full_diffusion_pointquery_8a100_20261124_v1/best.pt
SHA256 f1b29877c050c3666d482e0fc2efca1d6417d743620a920325db24327036d66f
```

The pretrained WM path and SHA256 are in `config_joint.json`. The frozen TF++
teacher is `ckps/full_diffusion/backbone/all_towns_seed1/model_0030_1.pth`, SHA256
`0ff5062a234f5919385957c4b46df8a10adcf09a88a427198e0e2e655b5935cb`.
Keep its matching config and the pinned Garage submodule.

The actual launcher uses `../ENV/recogdrive-B200/bin/python`, supplemented by
`../ENV/wmmpc-full-diffusion` and `.venvs/full-diffusion-carla` site-packages.
Despite the environment name, this experiment ran on A100. Configuration retains
the original paths, job topology, cgroup checks and runtime receipts; cloning on
another server requires explicit path/environment/allocation setup. Do not run
the recorded jobs against the existing output or bypass their guards.

Read training/inference through the files above. On the original cluster, the
verified launcher interface is:

```bash
bash method/full_diffusion_pointquery_cvci/launch_job.sh JOB_ID CONFIG_JSON
```

It must run inside the selected live LSF allocation. Fresh training requires its
real-data preflight receipt; resume requires the matching latest checkpoint.
Raw data, teacher metadata and the matching BEV cache are needed for training.
Closed-loop inference additionally requires the full CARLA0.9.15 runtime/maps.
