# Full Diffusion Geo: CARLA closed-loop evaluation

当前已跑通真实 CARLA 闭环，并在独立测试环境中运行完整 220 条路线。
首次汇报快照为 **4/220 条，DS 84.00，SR 75.00%**；这是阶段结果。
其中一条路线发生两次车辆碰撞，DS 为 36，已正常计入；其余三条 DS 为 100。
尚未发现确认的 CARLA 意外崩溃或 Python traceback。完整评测仍在运行，
后续进展和最终结果以远端运行目录的 `summary.json` 为准。

This run evaluates an intermediate **epoch 7** checkpoint of the paper-based
reimplementation. It is not a verified exact author reproduction, and the
single-route result below is not the complete Bench2Drive result.

## Confirmed smoke result

- Route: `1711`, Town12, `ParkingCutIn_1`, weather 15.
- Checkpoint: `epoch_07.pt` (fixed private evaluation copy).
- SHA256: `5174d07e4e681bf3915a98053fad73cff9d11f41a10f0e72024bf633610fa4b4`.
- Result: `Completed`, Driving Score **100**, route completion **100%**.
- No collision, red-light, stop-sign, lane, route-deviation, or blocked-vehicle infraction.
- Bench2Drive success on this single route: yes; its success helper excludes
  `min_speed_infractions` from the failure check.
- Simulated driving duration: 19.5 seconds; driving wall time: 203.275 seconds.
- Total attempt time including initial imports, server startup and world loading:
  382.10 seconds.
- Observed steady policy/control step: approximately 0.43–0.45 seconds for the
  100-step diffusion configuration. This is slower than real time in synchronous
  simulation; tick telemetry distinguishes this from a hung simulator.
- No unexpected CARLA exit observed. `server_exit.json` records return code `-9`
  with `evaluator_finishing=true`: the existing evaluator intentionally kills its
  server on normal shutdown.
- Existing upstream metric script on this route reports Efficiency 130.0085 and
  Smoothness 0.0. These are one-route diagnostics. In particular, comfort needs
  its logging-rate and angular-velocity conventions checked before a paper-level
  comparison: the existing tool assumes 0.1 seconds and uses raw CARLA angular
  velocities, whereas the evaluator ticks at 0.05 seconds. Speed and brake/throttle
  changes also warrant checking across the full run.

## Allocation and environment

Evaluation uses only user-provided LSF allocation **13046830** on
`rng-dl01-w010`, with two allocation-visible A100 80GB GPUs. Entry was through
`battach`. The two original training allocations are separate and were left
running.

The allocation has 16 logical CPUs in its actual affinity and a parent cgroup
memory limit of **268435456000 bytes (250 GiB)**. The displayed LSF per-slot
15.6 GiB MEMLIMIT is not the whole allocation limit. Full evaluation splits the
CPU affinity into two groups of 8, one per worker. Measured initial combined
memory remained far below the limit.

Authorized project root:

`/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC`

Dedicated evaluation Python, relative to that root:

`.venvs/full-diffusion-carla/bin/python`

It is a separate copy of the compatible training virtual environment. It does
not modify the training environment. Interpreter versions were retained:
Python 3.10.20, PyTorch 2.5.0+cu121, torchvision 0.20.0+cu121, CARLA Python client
0.9.15, numpy 1.26.4, timm 1.0.11 and diffusers 0.30.3. See
`evaluation_requirements.lock` and `allocation_preflight.json` in the run root.
Package imports, real CUDA tensor operations on both visible GPUs and the
required CARLA world settings passed preflight.
An RPC check against the running simulator confirmed **client 0.9.15 and server
0.9.15**, and all **12 towns required by the 220-route XML** are available,
including Town11, Town12, Town13 and Town15. Evidence:
`maps_and_versions.json` in the run root. No map download was required.

The packaged CARLA 0.9.15 Unreal executable is independent of the Python venv.
Evaluation runs it directly, headlessly, avoiding the shared launcher script's
`chmod`. Unreal user files, logs, caches and temporary files are directed into
the experiment directory. Shared CARLA and dataset files are not edited.

## Full 220-route run

Run root, relative to the project:

`results/full_diffusion_carla_20261031_epoch07`

- Both workers use the same immutable epoch 7 checkpoint and seed 901.
- Worker 0: alternating XML routes starting at index 0, 110 routes, GPU 0,
  ports 21000–21003.
- Worker 1: alternating XML routes starting at index 1, 110 routes, GPU 1,
  ports 21010–21013.
- GPU binding uses allocation-visible UUIDs, not host physical device IDs.
- A fresh CARLA process is started for every route to bound resource growth.
- Driving failures (collisions, deviation, blocked vehicle, simulation route
  timeout or the existing benchmark tick cap) are retained as driving results.
  They are not rerun for a better score.
- Infrastructure failures receive at most two total attempts. All attempts and
  errors are retained. Agent initialization/runtime errors stop that worker for
  investigation instead of producing a misleading sequence of scores.
- Startup timeout: 600 wall seconds; no tick progress timeout: 300 seconds;
  route wall-time bound: 5400 seconds. These are infrastructure guards; the
  benchmark's driving criteria, simulation rate, scenarios and score rules are
  unchanged.

Durable tmux socket: `results/tmux_carla_13046830.sock`.
Sessions: `carla_full0`, `carla_full1`, `carla_summary`.

Useful files in the run root:

- `smoke/route_1711/summary.json`: confirmed initial smoke result.
- `full/worker{0,1}/manifest.json`: fixed checkpoint hash, source hashes, seed,
  selected routes, GPU UUID and CPU affinity.
- `full/worker{0,1}/status.json`: latest active route and resource snapshot.
- `full/worker{0,1}/progress.json`: completed route count and errors.
- `full.worker{0,1}.log`: worker controller output.
- `full/worker*/route_*/attempt_*/evaluation.log`: evaluator output and errors.
- `full/worker*/route_*/attempt_*/server_stdout.log`: Unreal stdout/stderr.
- `full/worker*/route_*/attempt_*/server_process.json` and `server_exit.json`:
  server PID, command and expected/unexpected shutdown evidence.
- `full/worker*/route_*/attempt_*/telemetry.jsonl`: real simulation time,
  observed speed, target speed, controls, ten predicted path points and latency.
- `full/worker*/route_*/attempt_*/agent_data/*/metric_info.json`: per-tick motion
  measurements needed for downstream Efficiency/Comfortness calculation.
- `full/worker*/route_*/attempt_*/resources.jsonl`: GPU use, job memory and OOM
  counters.
- `summary.json`: automatically updated coverage, observed-route metrics,
  missing routes, failures, unexpected server exits and exception log paths.
- `partial_results.json`: official-statistics merge of currently valid results.
- `merged_results.json`: produced once all 220 routes have valid driving results.
- `evaluation_finished.json`: emitted when both workers reach a terminal state;
  check `complete_220_routes` before calling the full benchmark complete.

DS and SR from an incomplete set are explicitly marked as observed-route
metrics. `metrics_full_benchmark` remains null until all 220 routes are covered.

## Implementation and validation

New files are inside `WMMPC`:

- `tools/preflight_carla_eval.py`
- `evaluation/run_full_diffusion_carla.py`
- `evaluation/carla_diagnostics_evaluator.py`
- `evaluation/full_diffusion_diagnostic_agent.py`
- `evaluation/summarize_full_diffusion_carla.py`

The diagnostic agent forwards Bench2Drive's `+save_name` to Garage's route log
interface so motion metric files are actually written. Its telemetry wrapper
observes the existing forward output and control without changing the policy.
The diagnostic evaluator adds server log/storage isolation and unexpected-exit
reporting. Existing scenario and scoring implementations remain in use.

Validation completed: Python compilation, allocation-aware dependency/CUDA
preflight, nine outcome-classification checks (including retaining blocked,
deviated and timed-out driving failures), and an actual complete CARLA route.
The original training/inference integration checks alone had not established
this closed-loop result; the smoke run does.

Code snapshots are retained under `source_smoke_v1` and `source_full`. The full
run's passive trajectory logger fixes the initial smoke logger's use of PyTorch
hooks, because Garage calls `net.forward(...)` directly; the policy output is
unchanged.

During early full-run auditing, one normal-shutdown `server_exit.json` was empty:
its daemon log writer could be interrupted when the evaluator exited. This is
recorded as an incomplete diagnostic log, not evidence of a CARLA crash. The
server logger was hardened with atomic file replacement and a short writer join
on shutdown; later evaluator instances record their own diagnostic source hash.
The report helper was updated to separate unreadable exit logs from confirmed
unexpected exits. The two driving workers were left running during this change;
only the lightweight summary process was restarted. Diagnostic code snapshots
are retained under `source_diagnostics_v3`.
The normal-exit writer regression check passed using a real short-lived child
process: the writer was joined and its atomic JSON contained a complete normal
exit record. Evidence: `diagnostic_writer_check/passed.json`.
