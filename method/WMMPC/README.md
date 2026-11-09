# WMMPC

Method-specific implementation belongs here: `modeling/`, `inference/`, and
`configs/`. The future CARLA agent entrypoint is `inference/carla_agent.py`.
It must expose `get_entry_point()` and implement the Leaderboard AutonomousAgent
interface. No model, training algorithm, or runnable agent is implemented yet.

Weights belong in `../../ckps/WMMPC/<experiment>/seed_<seed>/`, not this folder.
Shared datasets, training utilities and Bench2Drive infrastructure stay outside
the method directory.