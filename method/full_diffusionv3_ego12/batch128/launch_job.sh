#!/usr/bin/env bash
set -euo pipefail
job=$1
output=$2
shift 2
root=$(cd -- "$(dirname -- "$0")/../../.." && pwd -P)
[[ "$job" =~ ^[0-9]+$ ]]
grep -Fq "/job.$job." /proc/self/cgroup
cd -- "$root"
runtime="$root/../ENV/recogdrive-B200"
garage="$root/../ENV/wmmpc-full-diffusion"
eval_runtime="$root/.venvs/full-diffusion-carla"
export LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WMMPC_CARLA_NAVIGATION_PATH="$root/third_party/carla_navigation_0_9_15"
export WMMPC_BRIDGE_MODULES="$root/../BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules"
export PYTHONPATH="$root:$runtime/lib/python3.10/site-packages:$garage/lib/python3.10/site-packages:$eval_runtime/lib/python3.10/site-packages"
exec "$runtime/bin/python" -u -m method.full_diffusionv3_ego12.batch128.run_job "$job" "$output" "$@"
