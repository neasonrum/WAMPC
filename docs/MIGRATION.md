# Bench2Drive migration

Source: local `../AutoMoT`, https://github.com/OscarHuangWind/AutoMoT.git,
commit `278fa5ad703f64d2104c163d00b886b4545a6186` (clean worktree at migration).
The source repository is preserved. Existing upstream file headers are retained.

Copied:
- `leaderboard/` runtime and data, excluding AutoMoT launch scripts and team_code.
- `scenario_runner/`, including scenario configurations and implementations.
- `eval_json/` route ID splits.
- Benchmark ability, efficiency/smoothness and supporting utilities under `tools/`.

Not copied: AutoMoT models, training code, inference agent/helpers, checkpoints,
NuSync/JSONL data, weight-merging and upload utilities, raw Garage data, simulator.

Local evaluator changes are restricted to process lifecycle:
- Launch CARLA with an argument list and a new process session.
- Pass the explicit job-visible graphics adapter index.
- Stop only the CARLA process group created by this evaluator, including on crash.
- Remove the upstream crash handler's global process search by graphics adapter.

Route definitions, scenarios, scoring, simulation timing, and the inherited
free-port lookup remain unchanged. The shared launcher supplies isolated ports,
per-run output paths, absolute Python import paths, and explicit LSF job checks.

Verification covers Python syntax, method-independent imports, 220 unique route
IDs, split coverage, scenario class availability, planning without GPU imports,
output/port isolation, and process cleanup scope. It does not verify CARLA runtime
dependencies, map availability, controller behavior, or exact compatibility with
the original BridgeDrive PDM-Lite evaluator revision. These require later audit.