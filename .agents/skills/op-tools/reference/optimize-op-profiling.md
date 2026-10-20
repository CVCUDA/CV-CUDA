[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — profiling and leads

Read during workflow steps 3–4 of `.agents/guidance/OPTIMIZATION_GUIDELINES.md`
(profile before coding; the optimization loop). The core guide's Golden Rules
and accept/drop/stop criteria still apply.

## Profiling

Classify the bottleneck before choosing a technique. Every number in the work
log, commit body, or MR description must come from benchmark or profiler output.
Do not fabricate or extrapolate metrics.

Start with the benchmark's CPU-time and GPU-time columns. If GPU time dominates,
use Nsight Compute (`ncu`) on the hot kernel. If CPU time is high, launches are
fragmented, or CPU/GPU times diverge, use Nsight Systems (`nsys`) to diagnose
launch overhead, synchronization, allocator overhead, or Python binding cost.

```bash
ncu --launch-skip 10 --launch-count 5 --kernel-name regex:"<hot_kernel>" --set full -o pre.ncu-rep ./bench_<op>
nsys profile -t cuda,nvtx,osrt -o sys.nsys-rep ./bench_<op>
nsys stats sys.nsys-rep
```

Use `--kernel-name` with `ncu`. Without it, `--launch-skip` can land on warmup
or setup work. If the profiled `Duration`, grid, or kernel name does not match
the benchmark configuration, the SOL numbers are not evidence.

| Profile signal | Bottleneck | Primary evidence | Typical next step |
|---|---|---|---|
| Memory SOL higher than Compute SOL, low `BWUtil`, memory stalls | Memory-bound | `SpeedOfLight`, `MemoryWorkloadAnalysis`, `WarpStateStatistics` | Improve access pattern, coalescing, vector width, or memory-level parallelism |
| Compute SOL higher than Memory SOL, one math pipe saturated | Compute-bound | `SpeedOfLight`, `ComputeWorkloadAnalysis`, roofline | Reduce or change the dominant math |
| Compute SOL high, no single pipe saturated, high issue-slot pressure | Issue-bound | `Issue Slots Busy`, `Executed IPC`, pipe utilization | Hoist invariants or reduce instruction count |
| Many small launches or CPU time dominates GPU time | Launch-overhead-bound | nvbench CPU/GPU columns, `nsys` timeline | Batch work, collapse per-channel/per-plane launches |
| Host gaps, stream waits, CPU/GPU synchronization | Sync-bound | `nsys` timeline | Remove unnecessary synchronization or reuse stream-ordered resources |

Read `SpeedOfLight` first for kernel work. The higher of Compute SOL% and
Memory SOL% names the bound; `100 / max(SOL%)` is a rough speedup ceiling for
the hypothesis. A memory-bound kernel already near ridge is normally triaged
instead of chasing noise.

After a change, capture `post.ncu-rep` with the same command and diff it:

```bash
ncu --import pre.ncu-rep --import post.ncu-rep --page diff
```

Report only the summary evidence, not raw profiler files. A useful profile
summary includes the bottleneck, the technique, the moved metric from the diff,
and the benchmark delta. For memory-bound wins, also include `BWUtil` before and
after.

## Technique Leads

These are advisory leads for optimization-loop step 2. Pick only after profiling names the
bottleneck. `NIX` means elements processed per thread; `DPT` is the local
data-packing helper for vectorized writes.

| Profile evidence | Candidate lead | Validation signal | Watch out for |
|---|---|---|---|
| Memory or latency bound; small dtype moves too few bytes per thread | Vectorize loads/stores per thread with NIX and DPT | `BWUtil` rises, memory stalls fall, benchmark clears noise | Wide elements may already be at ridge; contraction kernels can regress, so keep a measured scalar fallback if needed |
| Compute or issue bound; repeated invariant work per element | Hoist loop-invariant or redundant compute, then amortize with NIX | Issue pressure or dominant math pipe falls; output remains covered by pixel tests | Any arithmetic change that affects rounding is a precision trade, not a silent test update |
| Fixed per-thread cost dominates, such as repeated var-shape pointer lookup | Resolve invariant state once per thread and process more elements | Same body improves on expensive wrappers and remains stable on cheap wrappers | Raising bytes/thread with too many registers can reduce occupancy |
| Memory-bound kernel has similar absolute GB/s across low- and high-bandwidth SKUs | Increase memory-level parallelism with grid-stride loops, wide accesses, and enough occupancy | Reference SKU `BWUtil` improves and dev GPU does not regress | Prefer one SKU-generic kernel; only fork per architecture after proving a Pareto conflict |
| Cross-thread reductions cause shared-memory traffic or barriers | Use warp shuffle reductions, then one shared write per warp and a final first-warp reduce | Barrier/shared-memory stalls fall; tests cover edge counts and masks | Incorrect masks or tail handling create silent data errors |
| Writes are poorly coalesced | Make block X dimension at least 32 when compatible with the algorithm | Sectors/request and write throughput improve | Recheck occupancy and shared-memory layout |
| Per-channel, per-plane, or per-image dispatch dominates | Collapse launches and batch work inside one kernel | `nsys` shows fewer launches; benchmark clears noise | Preserve stream ordering, error handling, and per-image parameter semantics |
| Transient allocation, event creation, or synchronization dominates | Reuse cached stream-ordered resources such as `cudaMallocAsync`, cached events, or `PerDeviceResource<T>` | CPU time, sync gaps, or allocator cost falls | Check multi-GPU behavior and resource lifetime |
| A local helper already covers the operation | Reuse local primitives before adding generic abstractions | Existing tests and edge cases carry over | Grep first for `InterpolationWrap`, `BorderWrap`, `filter_utils`, NIX/DPT helpers, `PerDeviceResource`, and cached-event helpers |
