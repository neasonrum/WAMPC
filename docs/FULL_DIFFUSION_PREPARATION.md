# Full Diffusion Preparation

Audit date: 2026-10-30. Preparation only; no training has been launched.

## Paper Target

The confirmed target is **Full Diffusion geo**, not DiffusionDrive's truncated
diffusion or BridgeDrive's anchor-conditioned diffusion bridge. The user
explicitly selected the geometric row of Table 2; temporal waypoints are a
separate ablation, not an interchangeable implementation.

| Variant | Driving score | Success rate | Denoising steps |
| --- | --- | --- | --- |
| Full Diffusion, temporal | 79.75 | 58.18% | 100 |
| Full Diffusion, geometric | 83.85 | 67.27% | 100 |

Sources: [BridgeDrive Table 2 and Appendix B.2/C](https://arxiv.org/html/2509.23589v4).
Full Diffusion starts from Gaussian noise, uses no anchors, and removes
anchor-specific cross-attention. The perception backbone is TransFuser++ from
CARLA Garage, not the NAVSIM TransFuser adaptation. Appendix C.3 describes a
frozen pretrained perception module. Exact Full Diffusion configuration still
needs confirmation from the authors' development code.

## Existing Resources

- Training root: `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/carla_garage/data`.
  This is the correct Zimmerlin et al. Garage/PDM-Lite dataset family, not the
  Think2Drive-collected Bench2Drive training dataset or SimLingo.
- Download status reports 39/39 complete, 379,040,860,160 archive bytes.
- All 39 original archive sizes match the manifest; all 39 extraction
  verification records are present. No dataset re-download is needed.
- Live inventory: **5,368 routes**, with **696,391 frames each** in RGB, LiDAR,
  measurements, and boxes. Frame IDs align on every nonempty route. The
  Garage loader's route-quality rule accepts **5,060 routes**; this is not yet
  the final training-window count or the author's exact filtered manifest.
- **22 routes lack results metadata, including 2 empty routes.** All 22 are
  missing the same metadata in the original tar files, and their core sensor
  counts match the archives exactly. These are upstream release defects, not
  incomplete local extraction. The official Garage loader skips such routes.
  Do not fabricate metadata, delete shared data, or re-download these archives.
- First/last RGB and LAZ frames from one quality-passing route per scenario
  were decoded successfully: **78 samples**, including measurement/box JSON.
  This is not a full image/point-cloud integrity scan or a new byte/hash audit.
- CARLA root: `/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/carla/0.9.15`.
- Bench2Drive has **220 unique routes**; all referenced scenario classes and
  the required **12 town maps** are present, with valid Unreal map headers.
  AdditionalMaps are already imported; do not import them again.
- Leaderboard, scenario runner, metric tools, and the job-guarded launcher are
  present. The evaluator imports in the new environment. CARLA server/map
  loading, a real agent episode, and exact author protocol equivalence remain
  unverified. No closed-loop benchmark has been run.

Machine-readable evidence is in `results/full_diffusion_preflight_20261030.json`
and `results/full_diffusion_archive_exceptions_20261030.json`. The strict audit
returns nonzero because it reports the upstream data defects rather than hiding
them; this does not mean the archives need downloading again.

## Exact-Reproduction Blocker

The local public BridgeDrive checkout is commit
`85aa089a321a7b7cbc42aa58b3be508eb223d9e2`. Its documented adaptations are LEAD
and NAVSIM/DiffusionDrive, not a released Garage/PDM-Lite Full Diffusion pipeline.

In [author issue #8](https://github.com/ShuLiu-ETHZ/BridgeDrive/issues/8), the
author offers the PDM-Lite development code on request and identifies Garage's
`model_0030_1.pth` as the backbone. Request the **Full Diffusion ablation** too:
training/inference entrypoints, exact config and checkpoint, data filtering and
split, trajectory/speed normalization, sampler, controller, and evaluator
revision. A LEAD BridgeDrive checkpoint is not a Full Diffusion checkpoint.

## Installed Code and Weights

- Official Garage `leaderboard_2` source is in `third_party/carla_garage`, pinned
  at `48e6a6dbcb133220333bd65e242fbc2b492bb15b`. Existing BridgeDrive code is unchanged.
- Official `pretrained_models.zip` is retained under
  `ckps/full_diffusion/backbone/`: **2,686,570,563 bytes**, all ZIP CRCs verified.
- Extracted **all_towns, seed 1** weights/config/args are in
  `ckps/full_diffusion/backbone/all_towns_seed1/`. Do not substitute
  `town13_withheld` for this Bench2Drive target. The checkpoint is 483,676,856
  bytes; SHA256 is
  `0ff5062a234f5919385957c4b46df8a10adcf09a88a427198e0e2e655b5935cb`.
- All **1,332 state-dict entries load with strict=True** into the official
  120,656,247-parameter TF++ model. Both encoders are `regnety_032`.
  See `results/full_diffusion_backbone_load_20261030.json` and the checkpoint's
  `source.json` for provenance. This checkpoint does **not** contain a trained
  Full Diffusion head.
- Upstream construction requests ImageNet initialization even when a full
  checkpoint will immediately replace it. The strict-load check disabled that
  redundant download during construction only. A future offline agent must
  likewise avoid it or prepare the corresponding cache explicitly.

## Independent Environment

The new environment is `../ENV/wmmpc-full-diffusion`:

```bash
source ../ENV/wmmpc-full-diffusion/bin/activate
python -m pip check
```

Python 3.10.20, PyTorch 2.5.0+cu121, torchvision 0.20.0+cu121, timm 1.0.11,
NumPy 1.26.4, and CARLA API 0.9.15 are installed. `pip check` passes. CPU imports
of TF++ model, Garage dataset/augmentations/LAZ support, and Bench2Drive evaluator
pass. No existing environment was modified. The new venv uses the existing
Python 3.10 interpreter read-only and does not inherit its site-packages.

`requirements-full-diffusion.txt` combines Garage's core model/data dependencies
with the existing benchmark requirements. It is not the unpublished author's
exact environment: several evaluator/scientific-library pins follow WMMPC's
existing requirements. Actual installed versions are frozen in
`../ENV/wmmpc-full-diffusion/requirements.lock.txt`. Dependencies specific to the
unreleased denoiser may still need adding once its code is obtained.

## Authorized GPU Checks

Only the following user-designated jobs may be used for GPU preparation:

| Job | Node | Allocation |
| --- | --- | --- |
| 13047898 | rng-dl01-w007 | 2 A100-SXM4-80GB GPUs, CUDA matmul passed |
| 13047896 | rng-dl01-w002 | 2 A100-SXM4-80GB GPUs, CUDA matmul passed |

These are two nodes, not a single four-GPU allocation. Use `battach` and verify
the job context; do not submit new GPU jobs or touch unrelated workloads.
Four-rank distributed training requires a separate cross-node communication
check once the exact model pipeline is available. The finite preflight processes
have exited; no training was started or left running.

These jobs expose only their two allocated GPUs as local `0,1`, despite LSF
reporting host GPU IDs `4,0` and `7,2`; neither `LSB_JOBID` nor
`CUDA_VISIBLE_DEVICES` was set in the attachment. The GPU preflight verifies
hostname, job cgroup, live scheduler allocation, exactly two visible GPUs, and
idle compute state, then binds their UUIDs before importing PyTorch.
See `tools/preflight_full_diffusion_gpu.py` and `results/full_diffusion_gpu_*.json`.

## Still Needed Before Training

1. Request the **Garage/PDM-Lite Full Diffusion geo ablation development code**
  from the BridgeDrive authors, not just LEAD's BridgeDrive adaptation. Include
  the denoiser, Gaussian-start sampler, trajectory/speed scaling, data filtering
  and split, augmentation, optimizer/batch/epoch settings, and CARLA controller.
2. Obtain the exact Full Diffusion geo checkpoint if the goal is to evaluate
  the reported model directly. It is not required for training a new denoiser
  once the exact training pipeline is known, but is valuable for comparison.
3. Confirm the author's backbone fine-tuning/seed history and the precise
  Bench2Drive evaluator/scenario-runner revisions and control settings.
4. Integrate the obtained pipeline under `method/full_diffusion/`, check a real
  batch and training update, verify two-node communication, then run a CARLA
  smoke episode before launching the 220-route evaluation.

No LEAD dataset/TFv6 weights, NAVSIM/nuPlan data, anchor K-means file, AutoMoT
checkpoint, VLM, or repeated CARLA/AdditionalMaps download is required for this
target. A paper-based reimplementation is possible, but must be labeled as such
until the missing author settings are resolved; it is not already an exact
reproduction.

## Recheck

From WMMPC:

```bash
../ENV/wmmpc-full-diffusion/bin/python tools/audit_full_diffusion.py \
  --inventory-frames --decode --output results/full_diffusion_preflight_20261030.json
```

The original nine infrastructure tests passed seven checks, including route/
scenario completeness and owned-process cleanup. Two fail because the earlier
reserved `method/diffusiondrive` directories are absent; those unrelated
directories were not recreated or substituted for the new target. New tool
syntax/editor checks, the GPU namespace mock, and login-host rejection passed.