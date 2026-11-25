#!/usr/bin/env bash
set -euo pipefail
job=$1
config=$2
root=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
grep -Fq "/job.$job." /proc/self/cgroup
cd -- "$root"
runtime="$root/../ENV/recogdrive-B200"
garage="$root/../ENV/wmmpc-full-diffusion"
eval_runtime="$root/.venvs/full-diffusion-carla"
export LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
export WMMPC_CARLA_NAVIGATION_PATH="$root/third_party/carla_navigation_0_9_15"
export WMMPC_BRIDGE_MODULES="$root/../BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules"
export PYTHONPATH="$root:$runtime/lib/python3.10/site-packages:$garage/lib/python3.10/site-packages:$eval_runtime/lib/python3.10/site-packages"
exec "$runtime/bin/python" -u scripts/pointquery_cvci_retry_route3800_20261125_v2.py "$config" "$job"
