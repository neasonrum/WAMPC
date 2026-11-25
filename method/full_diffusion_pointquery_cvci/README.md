# Point-query E2E with CVCI-style future K/V

The complete, strictly loaded initialization is the finished point-query epoch10
checkpoint. Current BEV reads, 10 geometric point queries, 6 temporal increment
queries, ego12 attention, supervised labels, planner loss weights and 25 DDIM
steps are preserved. Frozen TF++ remains frozen.

One action-conditioned WM is initialized from the established timed BCDE
checkpoint. Its prior is the model prediction at exactly t-0.25 seconds,
advanced using the observed ego pose and time interpolation. The final unsupported
interval is masked. Online 20Hz inference selects k-5, with k-10/k-5/k BEV history.

Five usable fixed-input-ego future maps have their own shared Conv3x3+ReLU
64-to-256 projection. They do not share projection parameters with current BEV.
The established V8 neighborhood budgets are [9,9,16,25,36], producing 95 tokens
before out-of-bounds masks. Each sampled 256D vector retains its spatial and
temporal encoding. Equal initial horizon mass avoids bias from unequal budgets.

Both trajectory branches reuse this exact sampled tensor. Every decoder stage
and branch owns separate Q/K/V/output projections, position/time biases, and
256 independent sigmoid gate channels initialized at 0.01. Spatial queries use
actual XY positions; temporal queries use cumulative XY positions and the six
actual query times. A future cross-attention residual is inserted after ego12
attention and before the existing FFN. No pooling collapses trajectory queries.

Trajectory loss updates planner, sampler and all WM parameters. The original
teacher-feature loss updates WM only. No additional speed loss is introduced.
The original point-query planner loss weights and 10-epoch cosine LR trajectory
are retained, with the established lower WM learning rates. Training uses 8 A100,
local32/global256, no accumulation and 927 updates per sample-budget epoch.
Disjoint route streams preserve exact predecessor predictions, so data ordering
differs from the pure E2E shuffled baseline even though the frozen rows are equal.

All experiments reuse the immutable pdm_dev44_fixed_v1 lists: 237351 train anchors,
4796 train routes and 2223 validation anchors over 44 disjoint routes. Validation
retains the original eight shards and current-plan noise seeds; predecessor noise
uses a separate generator. Checkpoint selection uses route-macro temporal ADE.
Latest state includes optimizer, scheduler, per-rank route-stream cursor, rolling
plans and RNG. Best/epoch checkpoints include the trained planner, adapter and WM.
Normal completion automatically evaluates the selected checkpoint on the 220 B2D
routes with 25 DDIM, persistent CARLA servers and a shared dynamic queue.

Shared sampling can miss off-prior geometric alternatives. Current-BEV point
reads remain available, and each branch controls its own future contribution.
Per-branch/layer future horizon mass, sigmoid gates, residual norm ratios and valid
token counts are logged. Attention alone is not proof of benefit; assess fixed
validation and closed-loop driving scores against the point-query baseline.
