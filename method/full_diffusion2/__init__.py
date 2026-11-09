"""Training uses the verified workspace copy of CARLA navigation sources."""

import os
from pathlib import Path

_navigation = Path(__file__).resolve().parents[2] / "third_party/carla_navigation_0_9_15"
if not (_navigation / "agents/navigation").is_dir():
    raise FileNotFoundError(f"Verified workspace CARLA navigation sources are missing: {_navigation}")
os.environ.setdefault("WMMPC_CARLA_NAVIGATION_PATH", str(_navigation))
