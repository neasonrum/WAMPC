# Point-query 14-lane CARLA evaluation

Topology: two worlds on each of the six GPUs in allocations 13188135 (w005),
13188133 (w080), and 13188134 (w012), using eight logical CPUs per GPU. The two
GPUs in allocation 13177034 (w006) retain one world per GPU and four CPUs each.
No allocation requests, compute mode, MPS settings, checkpoint weights, model
code, sensor settings, controller gains, simulator timestep, or 25-DDIM sampler
are changed.

Each dual-lane GPU has one CUDA process and two separate original
PointQueryNetwork instances. Each local authenticated socket connection owns
one network, its explicit CUDA RNG generator, real ego pose/timestamp and
observed-BEV history. Requests are serviced serially without inference batching.
Every route constructs a fresh session with the same original seed. Both CARLA
instances, sensor agents, Kalman filters, PIDs, Traffic Managers, ports and
Unreal cache/user directories remain independent. GPU bounding-box decoding is
kept on the GPU; the CPU clients receive the decoded boxes and original outputs.
Tensor values, dtypes and non-overlapping strides are preserved by inline
pickle-byte transport; there is no Torch shared-memory tensor transport.

Preflight alternates two sessions for three real observations each against two
independent direct GPU references, with differing velocities between sessions.
Trajectory, speed, temporal increments and detector outputs have zero observed
differences. Tensor values/strides and each session's generator state match
exactly. History lengths progress independently. This tests numerical and state
semantics, not a closed-loop speedup or closed-loop outcome guarantee.

The old worker drains after a complete scored episode. If it has just claimed
the next episode, startup is paused and checked for any actual network call;
an episode with inference is allowed to finish. A startup-only handoff retains
its attempt files and is recorded as execution_handoff, rather than a scored
driving failure. Infrastructure retry accounting excludes this non-driving
handoff. Existing scored results are never deleted, retried or selected by score.

The existing queue semantic identity and old manifests remain immutable. The
legacy summary reader requires new manifest.identity to equal that anchor, so
new manifests explicitly add execution_identity, execution_backend and the
equivalence-preflight path. Queue.execution_variants and the parallel14 source
snapshot record the real RPC transport and extension hashes. Thus the baseline
identity remains a comparison anchor; execution_backend=pointquery_rpc14 is
the actual backend for these new workers. All model/sampler/source/seed identity
fields are checked, and only the transport label is normalized for compatibility.

Periodic NVML query failures are logged and do not abort driving; CPU/RAM guards
continue reading live cgroups. GPU snapshots older than 60 seconds are not shown
as live values. The existing 64GiB noncache RAM and 60GiB GPU soft limits remain.
Each job has a verified 250GiB hard RAM limit. A summary monitor remains active
through the handoff and merges all 220 raw official route records when complete.

Artifacts: the original run's parallel14 directory contains plan.json,
preflight.json, extension_source_sha256.json, extension_source_snapshot.tgz,
drained_JOB_GPU.json, controller logs and progress.json. Raw scores and the shared
queue remain in the original carla25 directory.
