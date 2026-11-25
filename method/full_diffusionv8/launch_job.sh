#!/usr/bin/env bash
set -euo pipefail
job=$1
shift
root=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
grep -Fq "/job.$job." /proc/self/cgroup
cd -- "$root"
runtime="$root/../ENV/recogdrive-B200"
garage="$root/../ENV/wmmpc-full-diffusion"
eval_runtime="$root/.venvs/full-diffusion-carla"
export LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072
export WMMPC_CARLA_NAVIGATION_PATH="$root/third_party/carla_navigation_0_9_15"
export WMMPC_BRIDGE_MODULES="$root/../BridgeDrive/BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/modules"
export PYTHONPATH="$root:$runtime/lib/python3.10/site-packages:$garage/lib/python3.10/site-packages:$eval_runtime/lib/python3.10/site-packages"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SOCKET_IFNAME='=bond0' GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_MODULE_LOADING=LAZY
exec "$runtime/bin/python" -u -m method.full_diffusionv8.run_job "$job" "$@"
