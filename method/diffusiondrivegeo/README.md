# DiffusionDrive-Geo

This directory is reserved for the CARLA DiffusionDrive-Geo baseline described
in BridgeDrive section 4.2 and appendix C.2: TransFuser++ perception,
DiffusionDrive truncated diffusion, and geometric path waypoints with speed.
It is not the public NAVSIM DiffusionDrive adaptation or BridgeDrive itself.

Place model code in `modeling/`, method configuration in `configs/`, and the
future Leaderboard agent in `inference/carla_agent.py`. No baseline model or
agent is implemented yet. Weights belong in
`../../ckps/diffusiondrive/<experiment>/seed_<seed>/`.