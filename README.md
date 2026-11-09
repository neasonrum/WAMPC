# WMMPC

Multi-method CARLA research workspace. The current preparation target is the
Full Diffusion baseline in BridgeDrive, using the Garage/PDM-Lite dataset and
the TransFuser++ perception backbone. This is not truncated DiffusionDrive,
BridgeDrive's anchor-guided bridge, or AutoMoT. The earlier `diffusiondrive/`
directories remain reserved for a separate baseline.

See [the Full Diffusion preparation audit](docs/FULL_DIFFUSION_PREPARATION.md)
for the paper target, resource checks, and reproduction blockers.

```text
method/
  WMMPC/               modeling/, inference/, configs/
  diffusiondrive/      modeling/, inference/, configs/
ckps/
  WMMPC/               <experiment>/seed_<seed>/
  diffusiondrive/      <experiment>/seed_<seed>/
results/               <method>/<experiment>/seed_<seed>/worker_<slot>/
configs/paths.json     External simulator/data and local output paths
data/                 Shared dataset code, not raw data copies
preprocess/           Shared preprocessing code
train/                Shared training utilities
evaluation/           Method-independent evaluation launcher
leaderboard/          Migrated Bench2Drive evaluator and 220-route definitions
scenario_runner/      Migrated scenario implementations
eval_json/            Bundled route ID splits
tools/                Migrated benchmark analysis utilities
scripts/              Convenience launchers
tests/                CPU-only infrastructure checks
```

Method, training, and agent implementations are intentionally not supplied.
Add a method under `method/<name>/`; its default CARLA entrypoint is
`inference/carla_agent.py`, exposing `get_entry_point()` and the Leaderboard
AutonomousAgent interface. Shared evaluation does not import any method eagerly.
An agent can also be selected explicitly with `--agent` and `--agent-config`.

## Resources

`configs/paths.json` points to the shared CARLA 0.9.15 installation and Garage
data. No simulator, training data, or model weights are copied into this project.
Override the simulator with `CARLA_ROOT` or `--carla-root`.
The Python 3.10 benchmark dependency list is `requirements-bench2drive.txt`.
The independent `../ENV/wmmpc-full-diffusion` environment includes these and
`requirements-full-diffusion.txt`; model/data/evaluator imports and finite CUDA
checks passed on the four designated A100 GPUs. Real CARLA episodes and the
Full Diffusion training pipeline are not yet available or validated.

## Plan an evaluation

From this directory, without CARLA, model weights, or GPUs:

```bash
bash scripts/eval_bench2drive.sh --method WMMPC --experiment smoke --seed 0 --slot 0 --dry-run
bash scripts/eval_bench2drive.sh --method diffusiondrive --experiment baseline --seed 0 --slot 1 --dry-run
```

The default is planning only, even without `--dry-run`. The printed JSON lists
the command, paths, ports, and missing prerequisites; it does not create run
outputs. The future checkpoint default is
`ckps/<method>/<experiment>/seed_<seed>/`. Each worker gets a separate result
directory, log, statistics JSON, and saved invocation plan.

## Execute later

Only after an agent, weights/config, dependencies and simulator/maps are ready,
attach to an explicitly authorized LSF GPU job using `battach`. Add `--execute`
and `--job-id JOB_ID` to a plan inside that job. The launcher never submits jobs,
never changes `CUDA_VISIBLE_DEVICES`, and never launches CARLA during planning.
`--gpu-rank` selects a job-visible GPU ordinal. Use distinct slots across all
concurrent workers on the same host, including workers from different methods.
Slot N uses ports `2000 + 10*N` through `2003 + 10*N`; TrafficManager uses the
last port, separate from CARLA's streaming ports. Host-local locks and port
checks reject overlapping workers using this launcher.

Use comma-separated route IDs via `--routes-subset` to shard the 220 routes.
Different workers must receive disjoint subsets for one combined benchmark.
Use a new experiment/seed/slot rather than overwriting a run; `--resume` is
explicit and verifies the stored invocation. GPU capacity and job authorization
remain the user's responsibility. No workloads are started automatically.

## Check infrastructure

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

See `docs/MIGRATION.md` for source provenance and evaluator safety changes.
This migration is not a claim of exact BridgeDrive protocol equivalence or a
successful CARLA closed-loop evaluation.