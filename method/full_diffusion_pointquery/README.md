# Point-query pure E2E experiment

This is a new architecture experiment, not an exact DiffusionDrive reproduction.
It keeps the ego12 baseline's coordinate diffusion, label codec, 10 geometric XY
points, six half-second XY increments, observation history, controller, and
frozen TF++ perception. It has no action-conditioned future world model and no
anchor bank.

Each branch retains one 256D query per point. Two decoder stages each perform
within-branch point self-attention, learned local BEV reading, ego12
cross-attention, the existing FFN/time modulation, and a shared-per-point XY
refinement head. Updated hidden tokens continue to the next decoder stage;
coordinate references are detached between stages, as in the existing decoder.

Each local BEV read uses the point itself and four learned neighboring offsets,
bounded to +/-2m per axis. Offsets initially form a 1m cardinal stencil.
Query-conditioned softmax weights start at .8 for the center and .05 for each
neighbor. Softmax aggregates only the five neighbors of each individual point,
preserving [B,10,256] and [B,6,256]. Out-of-bounds samples are masked and an
entirely unsupported neighborhood produces a zero BEV residual. Temporal
increments are cumulatively summed before spatial sampling. Grid x/y and the
32m half-extent follow the existing B2D ego12 sampler; metric coordinates and
bilinear interpolation use FP32 to preserve small offsets under BF16 training.

Compatible ego12 epoch10 weights are reused. The per-point encoder's first
linear layer, the XY output heads, self-attention, offset prediction, and
five-neighbor weighting layers are new. This initialization is not numerically
equivalent to the old one-query planner; initialization.json records every
reused, resized, and newly initialized tensor.

Data use the immutable pdm_dev44_fixed_v1 protocol: 237351 train anchors on 4796
routes, and all 2223 validation anchors on 44 disjoint routes. Current TF++
features/ego tokens are computed live. The two historical observed features use
the existing cache from the same frozen teacher. No future features, WM targets,
or predecessor predictions are read. Normalization and supervision retain the
protocol's recorded identities. Validation uses eight fixed shards, the existing
route ordering, seed 901700 + shard, 25 DDIM steps, and route-macro aggregation.
Unlike joint runs, pure E2E validation does not consume RNG for predecessor
backfill; paired-noise ablations should explicitly supply the same noise.

The optimizer batch is 8 GPUs x 32 samples x 1 accumulation = 256, with ten
epochs, AdamW 3e-4/1e-4 decay and the original ego12 cosine-restart schedule.
Each epoch uses a shared global permutation and drops its 39-sample tail.
Latest checkpoints include optimizer, scheduler, all eight RNG states and exact
epoch/update/data cursor. Each epoch also saves weights and validates; best.pt
minimizes the fixed validation temporal-position ADE. All run artifacts live
under /fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC.

After normal training completion, all eight GPUs automatically evaluate the
validation-selected checkpoint on the 220 closed-loop routes with 25 DDIM
steps. Existing dynamic town-sorted queues, persistent CARLA servers, free-port
reservation, bounded infrastructure retries, and allocation RAM guards are
reused in a method-local evaluation controller. Route outcomes remain raw
official results; there is no best-of-repeats score selection.

Use only the four supplied LSF allocations through battach and a dedicated tmux
server per allocation. The launcher verifies allocation cgroups, GPU UUIDs,
CPU affinity, real job RAM limits, and real-data preflight. It neither submits
nor cancels jobs, and never stops unrelated workloads.

