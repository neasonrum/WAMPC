"""Trajectory-conditioned latent BEV world model for the WMMPC B2D pipeline."""
import os
from pathlib import Path
import sys

# Keep newly spawned data workers independent of the simulator installation mount.
_root = Path(__file__).resolve().parents[2]
_navigation = _root / 'third_party/carla_navigation_0_9_15'
if not (_navigation / 'agents/navigation/global_route_planner.py').is_file():
    raise RuntimeError('The workspace CARLA navigation dependency is missing')
os.environ.setdefault('WMMPC_CARLA_NAVIGATION_PATH', str(_navigation))
_shared = '/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/carla/0.9.15/PythonAPI/carla'
sys.path[:] = [p for p in sys.path if p != _shared]
