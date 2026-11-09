# WMMPC

Multi-method CARLA research workspace. As of 2026-11-09, the handoff target is
Full Diffusion V3 (`independent_dual_query64`, `automot_temporal_pid_v1`), trained
for 10 epochs / 43,210 updates using Garage/PDM-Lite and frozen TransFuser++
perception. Its evaluated epoch-10 checkpoint has a complete 220-route CARLA
report: DS 85.359683, strict success 65.909091%, route completion 95.557818%.
The report combines 216 original routes, two ordinary retries, and two retries
using a version-specific native sensor-dormancy guard. It is not an unmodified,
single-pass evaluation or an exact BridgeDrive reproduction.

V3 model, training, and CARLA agent implementations are included. See
[the V3 method description](method/full_diffusionv3/README.md) for architecture
details. The [Full Diffusion preparation audit](docs/FULL_DIFFUSION_PREPARATION.md)
is historical preparation documentation, not the current training/result status.

## Handoff Inventory

Code delivery does not include all inference/reproduction dependencies.

| Item | Delivery status / action |
| --- | --- |
| V3 source and evaluation code | In this repository; clone with submodules. The verified source baseline before this documentation update is `d98a90b16619d585d6db400d3cfd924c35b225b6`. |
| Our V3 planner checkpoint | Not published in Git. Deliver the evaluated epoch-10 / step-43,210 file identified below; do not substitute an older V1/V2/single-query model. |
| Frozen perception source and weights | Source is the pinned CARLA Garage submodule. Weights/config can be obtained from the exact official archive below; re-uploading this backbone is unnecessary while that link is accessible. |
| Environment | Installation recipe below; also attach current training and evaluation environment exports. Do not transfer a virtual environment as a portable installation. |
| CARLA 0.9.15 and additional maps | Official download links below. The LSF evaluator additionally needs the validated private runtime configuration. |
| Results and native guard | Not published in Git. Deliver the selected report, route records/provenance, guard scripts, debugger/build identity, and runtime configuration. |
| Training data and indexes | Required only for training/resume, not for closed-loop inference. Preserve exact splits and normalization. |

The public repository is named `WAMPC`; the original local folder was `WMMPC`.
The clone directory name does not require a source rename.

### Evaluated planner identity

This is the final evaluated checkpoint, not a best-across-epochs selection.
Its size is 93,222,266 bytes and its SHA256 is:

```text
ea3eea848318c86cef17b9bc59a5a8c7fa17eed345123a5d359967a5323eb027
```

Original-server source file, for the maintainer to deliver:

```text
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/results/full_diffusionv3_dual_query64_automot_pid_train_20261104_v1/checkpoints/step_00043210_1793838564964641800.pt
```

On the receiving server, use `ckps/full_diffusionv3/release/epoch_10.pt` as the
documented destination and pass the file explicitly to the agent/evaluator.
The checkpoint contains V3 model/config/dataset-normalization and resume state;
it does not replace the separate frozen perception checkpoint.

```text
method/
  full_diffusion/      Frozen Garage integration and original baseline
  full_diffusion2/     Shared temporal perception/history utilities
  full_diffusionv3/    V3 data, model, diffusion, trainer and CARLA agent
ckps/
  full_diffusion/backbone/all_towns_seed1/  Official perception files
  full_diffusionv3/release/epoch_10.pt      Our delivered planner
evaluation/           V3 fast agent, CARLA lifecycle and route-queue drivers
configs/paths.json    Simulator/data and output paths to customize
data/full_diffusion/ Training indexes/normalization; not the raw dataset
leaderboard/          Bench2Drive evaluator and 220-route definitions
scenario_runner/      Scenario implementations
eval_json/            Bundled route splits
docs/                 Method/provenance and historical preparation notes
tests/                CPU-only checks
```

## Receiving-Server Setup

All commands below start at the repository root. Replace `/path/to/...` with
receiving-server paths. Use a disk with enough space for downloads, extraction,
and results; do not place large caches in Git. Linux, a CUDA-compatible NVIDIA
driver, and Vulkan support are required for actual CARLA/GPU evaluation.

### 1. Clone and install Python dependencies

The perception implementation comes from the
[CARLA Garage repository](https://github.com/autonomousvision/carla_garage), pinned
as the `third_party/carla_garage` submodule at commit
`48e6a6dbcb133220333bd65e242fbc2b492bb15b`. Clone this project with submodules:

```bash
git clone --recurse-submodules https://github.com/neasonrum/WAMPC.git
cd WAMPC
```

For an existing ordinary clone, run `git submodule update --init --recursive`.
The original evaluation used Python 3.10, PyTorch 2.5.0+cu121,
torchvision 0.20.0+cu121, diffusers 0.30.3, NumPy 1.26.4, and CARLA 0.9.15.
Rebuild the evaluation environment at the path required by the current driver:

```bash
python3.10 -m venv .venvs/full-diffusion-carla
source .venvs/full-diffusion-carla/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-full-diffusion.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip check

export WMMPC_DOWNLOAD_ROOT=/path/to/wmmpc-download-cache
export WMMPC_CARLA_RESULTS_ROOT=/path/to/wmmpc-results
export TMPDIR=/path/to/wmmpc-tmp
mkdir -p "$WMMPC_DOWNLOAD_ROOT" "$WMMPC_CARLA_RESULTS_ROOT" "$TMPDIR"
```

The [full dependency list](requirements-full-diffusion.txt) includes the
[benchmark dependencies](requirements-bench2drive.txt). These are an installation
recipe, not a claim that a clean receiving-server environment has been tested.
The maintainer should export both actual environments after all V3 dependencies
are installed; an older training lock file may predate later additions.

### 2. Download the frozen upstream perception backbone

Git clone obtains the source code, not the pretrained perception checkpoint.
The existing local provenance record identifies this official download:

https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip

The archive is 2,686,570,563 bytes. Select the `pretrained_models/all_towns/`
members `model_0030_1.pth`, `config.json`, and `args.txt`, and place them in
`ckps/full_diffusion/backbone/all_towns_seed1/`. The required weight is
483,676,856 bytes, with SHA256:

```text
0ff5062a234f5919385957c4b46df8a10adcf09a88a427198e0e2e655b5935cb
```

Download and extract only the matching members. `unzip -n` preserves existing
files; use an empty destination for a fresh handoff and never silently replace
a mismatched model/config pair.

```bash
BACKBONE_ROOT="$PWD/ckps/full_diffusion/backbone/all_towns_seed1"
mkdir -p "$BACKBONE_ROOT" "$WMMPC_DOWNLOAD_ROOT"
curl --fail --location --retry 3 --continue-at - \
  --output "$WMMPC_DOWNLOAD_ROOT/pretrained_models.zip" \
  https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip
unzip -n -j "$WMMPC_DOWNLOAD_ROOT/pretrained_models.zip" \
  pretrained_models/all_towns/model_0030_1.pth \
  pretrained_models/all_towns/config.json \
  pretrained_models/all_towns/args.txt \
  -d "$BACKBONE_ROOT"
printf '%s  %s\n' \
  0ff5062a234f5919385957c4b46df8a10adcf09a88a427198e0e2e655b5935cb \
  "$BACKBONE_ROOT/model_0030_1.pth" | sha256sum --check -
```

On 2026-11-09, the official URL returned HTTP 200 with the recorded size.
The local weight SHA256 and the byte contents of `config.json` and `args.txt`
were checked against this exact local official archive and matched. No locally
modified backbone configuration needs to be supplied in place of those files.

The official link, exact archive members, pinned source commit, and checksum
are sufficient to identify the frozen backbone; it need not be re-uploaded
if the recipient can download the upstream artifact. The V3 planner is our
separate trained checkpoint and must still be delivered. V3 checkpoint
normalization is embedded; closed-loop inference does not need the full
Garage/PDM-Lite training dataset.

### 3. Place our V3 checkpoint

The V3 weight is not in the upstream Garage archive and has no published release
URL yet. Obtain the file from the maintainer, then copy it without overwriting
an existing release file and verify its identity:

```bash
mkdir -p ckps/full_diffusionv3/release
cp -n /path/to/delivered/epoch_10.pt ckps/full_diffusionv3/release/epoch_10.pt
printf '%s  %s\n' \
  ea3eea848318c86cef17b9bc59a5a8c7fa17eed345123a5d359967a5323eb027 \
  ckps/full_diffusionv3/release/epoch_10.pt | sha256sum --check -
```

### 4. CARLA, maps, and receiving-server paths

Official simulator downloads:

- [CARLA 0.9.15 Linux](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz): 8,386,636,048 bytes.
- [Additional maps 0.9.15](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz): 7,375,946,087 bytes.

The archives total about 15.76 GB before extraction. Both must be installed,
including Town13 used by the benchmark. Use the official 0.9.15 installation
procedure; the Python package alone is not the simulator.

[configs/paths.json](configs/paths.json) contains original-server absolute
paths. Set `CARLA_ROOT` to the receiving installation for Python navigation
sources. `GARAGE_DATA_ROOT` is needed only when preparing/training on the dataset.
Update the JSON structurally, not by replacing a remembered username:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
python - <<'PY'
import json
import os
from pathlib import Path

path = Path("configs/paths.json")
paths = json.loads(path.read_text())
paths["carla_root"] = os.environ["CARLA_ROOT"]
paths["garage_root"] = os.environ.get("GARAGE_DATA_ROOT", "")
paths["checkpoint_root"] = "ckps"
paths["results_root"] = os.environ["WMMPC_CARLA_RESULTS_ROOT"]
path.write_text(json.dumps(paths, indent=2) + "\n")
PY
```

`WMMPC_CARLA_RESULTS_ROOT` also controls the fast driver's result/runtime path
checks. Its checkpoint and CARLA runtime must resolve under the project or this
results root. The navigation installation and private evaluation runtime are
separate dependencies; changing an environment variable does not relocate every
historical controller path.

### 5. CPU-only readiness checks

After verifying both weight hashes and paths, load only the trusted project
checkpoint on CPU and check the import/payload contract:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from pathlib import Path
import torch
from method.full_diffusionv3.inference.carla_agent import get_entry_point

assert get_entry_point() == "FullDiffusionV3Agent"
checkpoint = torch.load(
    Path("ckps/full_diffusionv3/release/epoch_10.pt"),
    map_location="cpu", weights_only=False,
)
assert {"config", "dataset", "model"} <= checkpoint.keys()
assert "v3_normalization" in checkpoint["dataset"]
print("V3 agent import and checkpoint payload: OK")
PY

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python evaluation/run_full_diffusion2_carla.py --help
```

These checks do not start CARLA or request a GPU, and do not prove closed-loop
performance. Existing CPU tests can be run with
`PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`.

## V3 Closed-Loop Evaluation

The actual evaluation driver is
[evaluation/run_full_diffusion2_carla.py](evaluation/run_full_diffusion2_carla.py).
Despite the filename, it selects V3 with `--method full_diffusionv3` and uses
[evaluation/full_diffusionv3_fast_agent.py](evaluation/full_diffusionv3_fast_agent.py).
Always set this method explicitly; the default is V2. Sampling uses 100 steps,
seed 901, the V3 AutoMoT longitudinal PID, and its final 35 km/h cap. Keep the
original 650 m actor/tile distances and benchmark scoring when comparing runs.

The existing driver is LSF-specific, not a portable one-command bootstrap. It
requires an authorized attached job, exactly two allocation-visible GPU UUIDs
(one selected GPU per worker), at least eight visible CPUs, verified cgroup
RAM/headroom, an idle selected GPU, and the fixed evaluation environment path.
The private CARLA runtime must contain the genuine `runtime_ready.json` receipt
and the validated renderer/map configuration. Plain official archive extraction
does not create that receipt. Do not fabricate it or remove authorization checks
to bypass setup. Runtime reconstruction and non-LSF launch adaptation on a new
server remain unvalidated; deliver the original runtime receipt/configuration
and adapt those host-specific assumptions before executing this template.

Attach using `battach YOUR_APPROVED_JOB_ID` on the original cluster. Then, inside
that allocation, restore the repository/environment/paths and use a new run:

```bash
cd /path/to/WAMPC
source .venvs/full-diffusion-carla/bin/activate
export WMMPC_CARLA_RESULTS_ROOT=/path/to/wmmpc-results
export TMPDIR=/path/to/wmmpc-tmp
WMMPC_JOB_ID=YOUR_APPROVED_JOB_ID
WMMPC_CARLA_RUNTIME="$WMMPC_CARLA_RESULTS_ROOT/.runtime/carla_0.9.15_v1"

python evaluation/run_full_diffusion2_carla.py \
  --method full_diffusionv3 \
  --job-id "$WMMPC_JOB_ID" \
  --checkpoint "$PWD/ckps/full_diffusionv3/release/epoch_10.pt" \
  --run-root "$WMMPC_CARLA_RESULTS_ROOT/v3_handoff_$(date +%Y%m%d_%H%M%S)" \
  --carla-root "$WMMPC_CARLA_RUNTIME" \
  --gpu 0 --lanes 1 --backend direct \
  --route-ids 1711 --seed 901 \
  --server-mode warm --startup-wait rpc \
  --routes-per-server 32 --slot-base 2700 --tm-base 47000 \
  --wall-budget 5400
```

This is a single-route readiness run. For all 220 routes, use `--route-ids all`
with a sufficient approved allocation/wall budget. Use disjoint route subsets
and port ranges for simultaneous workers; preserve all attempts. The driver's
budget is a bounded evaluation timeout, not permission to submit or repurpose
jobs. Historical job IDs in method/coordinator notes are not current approvals.

## Training or Resume Handoff

Training data is Garage/PDM-Lite, not the Think2Drive dataset or SimLingo. The
[pinned upstream download script](https://github.com/autonomousvision/carla_garage/blob/48e6a6dbcb133220333bd65e242fbc2b492bb15b/tools/download_data.sh)
lists 39 archives from
`https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset/`.
Their total download size is 379,040,860,160 bytes (about 379.04 GB); extraction
requires additional space. This raw dataset and large derived sensor/BEV caches
are not needed for closed-loop inference and should not be added to Git.

For exact training/resume, also deliver the original index directory
`data/full_diffusion/pdm_lite_seed901/` (`manifest.json`, `routes.json`,
`train.npy`, `val.npy`), the matching V3 normalization, sensor snapshot identity,
frozen source/config snapshots, and checkpoint completeness metadata. The
normalization was prepared on 276,547 retained training samples. Do not regenerate
different splits or silently substitute dataset/config identities during resume.

The trainer is [method/full_diffusionv3/train.py](method/full_diffusionv3/train.py);
the historical multi-node supervisor is
[method/full_diffusionv3/coordinator.py](method/full_diffusionv3/coordinator.py).
The completed run used eight A100 ranks, global batch 64, and 10 epochs. See the
[V3 runtime/architecture notes](method/full_diffusionv3/README.md). Their original
LSF job/host guards and training environment path must be adapted to newly
authorized allocations; copying old job IDs is not a valid resume procedure.

## Final Results and Known Limitations

Authoritative original-server result directory:

```text
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/results/full_diffusionv3_complete220_native_guard_20261105_r6
```

Deliver `aggregate_results.py`, `merged_results.json`, `final_report.json`,
`route_scores.csv`, `route_provenance.json`, and `merged_provenance.json`, plus
the four referenced source queues and selected raw route/guard records. These
artifacts are currently external/ignored, not bundled with this repository.
The merged JSON SHA256 is:

```text
9670c85e2fbfcf45353c06a49abf996c30fdff0fdf7a90414a35ccdfbbb7581a
```

| Metric | Full 220-route result |
| --- | --- |
| Driving score (DS) | 85.35968342727273 |
| Strict success rate | 65.9090909090909% (145/220) |
| Route completion (RC) | 95.55781818181819% |
| Collision-free rate | 72.72727272727273% (160/220) |
| Mean score penalty | 0.8874112681818181 |

All 220 unique routes are included, including valid driving failures. Status
counts are 196 Completed, 16 Failed-TickRuntime, and 8 Failed-Agent-blocked;
Completed is not the strict success definition. The merge uses 216 original
records, ordinary R2 retries for 24041/25845, and guarded R4/R5 retries for
3800/23670. It does not choose the best score across attempts.

The two guarded routes hit a native Town13 sensor-dormancy lifecycle failure.
The temporary debugger guard skips an empty moved-stream access during sensor
EndPlay and retains deregistration. It is pinned to ELF BuildID
`12cdff35da2c8067` and is not a universal CARLA 0.9.15 patch. Guard scripts,
debugger/runtime dependencies, matching build identity, and diagnostic records
must be delivered separately if reproducing this exact recovered report.
Route 23670 also had a post-score shutdown SIGSEGV despite a valid saved driving
record. A permanent CARLA source repair has not been compiled/validated.

Consequently, full coverage is not a claim of an unmodified single-pass run,
guaranteed identical scores on a new server, or universal improvement over
gate-BEV. Keep the result provenance and retry/selection rules with every
comparison. See [migration provenance](docs/MIGRATION.md) for the shared
evaluator's origins.

## Maintainer Publication Checklist

Publish our planner with its exact checksum and replace the pending-delivery
description with a real release/storage URL. The frozen backbone can remain
an upstream link. Publish the selected result/guard/runtime materials and
current environment exports; do not upload the whole result/cache/venv tree.
Keep third-party attribution and applicable redistribution permissions.

Original-server environment export commands, to run when preparing the handoff:

```bash
HANDOFF_DIR=/path/to/wmmpc-handoff
mkdir -p "$HANDOFF_DIR"
../ENV/wmmpc-full-diffusion/bin/python -m pip freeze \
  > "$HANDOFF_DIR/requirements-training.lock.txt"
.venvs/full-diffusion-carla/bin/python -m pip freeze \
  > "$HANDOFF_DIR/requirements-evaluation.lock.txt"
git rev-parse HEAD > "$HANDOFF_DIR/source_commit.txt"
git submodule status > "$HANDOFF_DIR/submodule_commits.txt"
```

Review exports for credentials/private URLs before public release. Our evaluated
planner is about 93 MB; the official backbone is about 484 MB. Use suitable
release/storage artifacts (or configured Git LFS), not an indiscriminate
`git add .`. Check storage/bandwidth limits before publishing large files.