#!/bin/bash
set -euo pipefail
workspace=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh
phase=inspect
exec "$workspace/ENV/wmmpc-full-diffusion/bin/python" \
  "$workspace/WMMPC/method/full_diffusion2/launch.py" \
  --phase "$phase" --detach \
  --output "$workspace/WMMPC/results/full_diffusion2_geo_20261102_seed901/$phase"