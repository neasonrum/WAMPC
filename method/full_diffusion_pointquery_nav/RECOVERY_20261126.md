# Six-GPU recovery of the navigation E2E experiment

The original run failed at update2222 while reading rgb/0069.jpg in a verified
sensor snapshot: zipfile reported CRC32 mismatch. Reopening that same member
subsequently passed CRC and the immutable per-member SHA256 three times.
The snapshot reader retried selected OSError values but not BadZipFile.
The last complete resumable checkpoint is update2000 (epoch3, step146), with
the original model, AdamW, scheduler and all eight RNG streams.

Do not overwrite the original run or modify the shared snapshot/base/CVCI code.
This recovery adds a method-local reader which closes and reopens on CRC,
decompression, EOF or SHA256 errors, with a maximum of four independent opens.
Every successful read retains the existing CRC and SHA256 verification. A
persistent failure stops training; no sample is skipped and no raw-data fallback
or replacement label is used.

The original allocation13177034 is no longer RUN. Use only the three remaining
two-A100 allocations13188135/13188133/13188134; do not submit or cancel jobs.
Preserve the exact original 256-sample epoch permutation, 927 updates/epoch,
data cursor, LR/scheduler and optimizer state. Six physical ranks own eight
logical batches of32, with virtual assignments [[0,2],[1,3],[4],[5],[6],[7]].
Each original logical batch retains its own restored noise and dropout RNG
state. The first two physical ranks execute two logical microbatches, the other
four execute one. Use no_sync for each non-final local microbatch, scale each
logical loss by physical_world_size/8, and perform one AdamW update after the
single DDP synchronization. This preserves the original global loss average.
Floating-point collective reduction order can differ from eight-GPU training.

The fixed44-route/2223-anchor validation retains all eight original virtual
shards, each shard's original ordering, batch geometry and noise seed, and25
DDIM steps. Model architecture, current speed/navigation encoding, supervision,
TF++ freezing and controller are unchanged. Carry over existing best/validation
records; complete the original total of10 epochs, then automatically run220
closed-loop routes with the selected checkpoint on the available allocations.

Preflight exercises the actual failing sample, original sample partitions,
bounded CRC recovery and persistent-error failure. A real six-GPU distributed
probe compares uneven no_sync gradients with an eight-batch reference and
checks RNG restoration before launching the actual resumed optimizer updates.

The parent source and original failure logs remain in the v1 output. Recovery
artifacts use a separate common_ai/nix3sgh_cache/WMMPC directory. latest.pt
continues to save AdamW, scheduler, all eight logical RNG states and the exact
update/data cursor. The original latest.pt remains untouched.
