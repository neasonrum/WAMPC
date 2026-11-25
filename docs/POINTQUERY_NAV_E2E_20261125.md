# Point-query E2E with direct per-point navigation fusion

This experiment adds navigation to the completed pure point-query E2E planner.
It does not use an action-conditioned world model or future BEV features.
The existing full_diffusion_pointquery and full_diffusion_pointquery_cvci source
are preserved. The new checkpoint has a separate method/format identity.

At the current observation, concatenate the six-way one-hot command and target
XY in the current ego frame, normalizing XY by the existing path_scale=20m:
[B,6] + [B,2] -> [B,8]. A shared Linear(8,256), ReLU, Linear(256,256), LayerNorm
encoder produces [B,256]. Broadcast this vector to the 10 geometric queries or
six temporal queries. Each branch has its own Linear(512,256) fusion:
[B,N,256] concat [B,N,256] -> [B,N,512] -> [B,N,256]. Apply this once before the
first point self-attention block; preserve the baseline behavioral offsets.

The fusion starts with W=[I,0] and zero bias. An equivalent
query + Linear([query,nav], W-[I,0], bias) calculation prevents autocast from
rounding the original query through an extra matrix multiply. Every original
tensor is loaded strictly from pure point-query E2E epoch10. Preflight verifies bitwise-identical
BF16 denoising outputs before training and real navigation gradients after
optimization. The encoder initially has zero gradient because the navigation
half of the fusion is zero; it receives gradients after that half learns.
Existing ego12 conditioning still includes current speed/command status; the
experimental change is this additional direct navigation path into point queries.

The baseline 10 geometric XY points, six 0.5s XY increments, coordinate-space
VP diffusion, two decoder stages, five learned current-BEV samples per point,
12 ego/status conditioning tokens, frozen TF++, controller, supervision and
loss weights are unchanged. No anchors or direct scene-memory attention are added.

Use the frozen pdm_dev44_fixed_v1 protocol: 237351 training anchors on 4796
routes and 2223 validation anchors on 44 disjoint routes. Keep the eight fixed
validation shards, seed 901700+rank, 25 DDIM steps, and route-macro aggregation.
Validate the initializer before any training. The new optimizer/scheduler starts
fresh for 10 additional epochs: 8 A100 x32 samples, accumulation=1, global256,
927 updates/epoch, AdamW 3e-4, weight decay1e-4, the original cosine-restart
schedule. Compared with the published epoch10 baseline, improvement also
includes additional optimization; a matched continuation control is needed to
attribute improvement exclusively to navigation.

latest.pt saves model, optimizer, scheduler, all eight RNG states and exact
training/data cursors at the first update, every250 updates and each epoch.
Each epoch validates and saves epoch weights. best.pt minimizes the fixed
validation temporal-position ADE. Artifacts are under the configured
/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC output directory.

After normal completion, evaluate the selected checkpoint on 220 closed-loop
routes using 25 DDIM steps, a dynamic queue and persistent CARLA servers.
Use the existing direct CUDA evaluator with one lane per GPU on the eight GPUs.
Respect each allocation's actual deadline and RAM limits. Missing routes after
infrastructure failure are reported, never silently counted as completed.

Launch only through the existing supplied LSF allocations using battach and
dedicated experiment tmux servers. Preflight runs real local batch8 and32
updates, gradients, inference, DataLoader workers and optimizer round-trip.
The online acceptance check exercises the actual inference/controller wrapper.
Never submit/cancel jobs or change MPS/compute mode as part of this experiment.
