#!/usr/bin/env bash
set -euo pipefail
job=$1
shift
[[ "$job" =~ ^[0-9]+$ ]]
grep -Fq "/job.$job." /proc/self/cgroup
root=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
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
export WMMPC_CARLA_RESULTS_ROOT=/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC
export WMMPC_MONITOR_RECOVERY_DIR="$WMMPC_CARLA_RESULTS_ROOT/full_diffusion_pointquery_8a100_20261124_v1/monitor_recovery"
export SLOWER=0
if [[ "${1:-}" == "--recover" ]]; then
    shift
    exec "$runtime/bin/python" -u "$root/scripts/pointquery_resume_workers_20261124.py" "$job" "$@"
fi
exec "$runtime/bin/python" -u "$root/scripts/pointquery_monitor_recovery_20261124.py" "$@"
