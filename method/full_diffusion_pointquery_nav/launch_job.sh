#!/usr/bin/env bash
set -euo pipefail
job=$1
config=$2
shift 2
root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
[[ "$job" =~ ^[0-9]+$ ]]
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
if [[ "${1:-}" == "--preflight" ]]; then
    exec "$runtime/bin/python" -u -m method.full_diffusion_pointquery_nav.preflight --job-id "$job" --config "$config"
fi
if [[ "${1:-}" == "--acceptance" ]]; then
    exec "$runtime/bin/python" -u -m method.full_diffusion_pointquery_nav.acceptance --config "$config"
fi
exec "$runtime/bin/python" -u -m method.full_diffusion_pointquery_nav.run_job "$job" "$config" "$@"

