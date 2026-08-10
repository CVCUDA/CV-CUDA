[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Operator Optimization Guidelines

How to land a performance change in CV-CUDA -- kernels, host-side overhead,
caches, allocators, and Python bindings -- without regressing correctness or
neighbouring code paths. These guidelines apply to humans and AI agents.

For *reviewing* an optimization MR, see [REVIEW_PR_GUIDELINES.md](REVIEW_PR_GUIDELINES.md).
For bug fixes that may incidentally improve performance, see
[BUGFIX_GUIDELINES.md](BUGFIX_GUIDELINES.md).

## Golden Rules

These rules override every workflow shortcut, technique, or convenience.
An optimization MR is not ready unless all four hold.

1. **Correctness is mandatory.** Do not ignore, disable, weaken, or silently
   retune tests. Any deliberate precision or correctness trade requires explicit
   human-maintainer review. Before touching a kernel, confirm pixel-level
   regression coverage against an independent reference: integer outputs use
   bit-exact checks such as `EXPECT_EQ`; floating-point outputs use a stated
   tight tolerance such as `EXPECT_NEAR(tol)` when device FMA contraction makes
   bit-exact host comparison invalid. If coverage is missing, add it before
   optimizing or leave that kernel untouched and record the gap as a required
   follow-up.
2. **Benchmark evidence is mandatory.** Every claimed win needs a named-hardware
   before/after, and no benchmark in the regression surface may slow down beyond
   its noise band. The regression surface is the existing benchmark suite,
   including sibling configurations and other operators that share the modified
   kernel or code path. Any committed baseline update that replaces a previous
   comparable baseline row with a slower same-key value is blocking by default.
3. **CI evidence is mandatory.** Local benchmarks guide iteration. The
   review-ready MR needs a passing CI run on established reference hardware,
   with the benchmark baseline updated when CI flags real improvements.
4. **Limit memory growth.** Operational memory should remain unchanged by
   default. Up to **10 MB** of peak attributable growth is
   automatically allowed if no new runtime CUDA allocation/free path is added.
   Anything larger, or any new allocation/free path, requires explicit human
   review before submission.

## Definitions

- **Full benchmark surface**: the union of every exact case expanded by
  `expected_case_keys_for_entry` from every declared operator config entry.
  Each entry's authoritative `tier` must be `basic` or `advanced`; an operator
  need not declare both. The nested baseline keys must exactly equal that union,
  covering all benchmarked layouts, dtypes, sizes, and variants. Use
  `bench_<op> --list` and `run_bench.py ... --operator <op> --tier basic,advanced`.
- **Optimization-target surface**: rows in the full benchmark surface that are
  real operator execution paths. Layout-conversion comparison rows such as
  `NCHW_FAKE` are reference-only: include them in regression evidence and the
  full-operator Impact statistics, but do not pick them as optimization leads
  or spend strike/triage budget on them. Every FakePlanar row is advanced-tier
  and Tensor-only, and must have exactly one same-tier native `NCHW`/`CHW` row
  whose other expanded axes are identical; unmatched or ambiguous rows are not
  a comparison surface and block preflight, final evidence, and summary output.
- **Regression surface**: the full existing benchmark suite, not a selected
  subset. Include the target operator's other configurations and any other
  operator that shares changed code.
- **Profile/lead coverage**: every row in the optimization-target surface is
  either directly profiled or mapped to a representative profiled kernel group,
  and every group has an optimization-lead disposition: improved, at ridge,
  triaged after struck leads, or unchanged/no-regression with a measured reason.
  Reference-only rows still need harness-verified no-regression evidence, but
  they are not optimization leads.
- **Noise band**: the per-config noise/parity gate reported by the benchmark
  harness; see [bench/README.md](../../bench/README.md). A benchmark change is real
  only when it clears this band. A regression is a slowdown beyond this band.
- **At ridge**: profiled near the hardware limit, typically about 90% or higher
  `BWUtil` / SOL for memory-bound kernels, with no meaningful headroom left.
- **Triaged**: left unoptimized for a measurement-backed reason, never for an
  unprofiled "low ROI" estimate.

## Workflow

Follow these phases for every optimization campaign.

### 1. Survey And Scope

Baseline the full benchmark surface before narrowing the work. Rank
optimization-target configurations by measured headroom, weighted by absolute
cost: a 36% `BWUtil` case taking 1.5 ms outranks a 39% case taking 0.7 ms.
Do not optimize layout-conversion reference rows (`NCHW_FAKE`/`CHW_FAKE`);
they compare a native layout path against the equivalent reformat-and-run path.

Build a profile/lead coverage map before coding. For each benchmark row, record
the kernel group or host path it exercises, the representative profile that
covers it, and the initial candidate leads. Grouping rows is allowed only when
they share the same implementation path and bottleneck; otherwise profile them
separately.

Treat layouts and variants as separate until proven otherwise. In CV-CUDA,
NHWC vs. NCHW and Tensor vs. VarShape often use different kernels; optimizing
one does not prove the other is done.

Each attempt stays narrow: one operator, one variant, one input type, and one
optimization idea. The campaign stays exhaustive: every surveyed configuration
with material headroom is either optimized or measurement-backed-triaged.
Out-of-scope ideas start a separate branch and MR.

Public C, C++, and Python API/ABI must remain unchanged. Signature or behavior
changes are feature work, not optimization work.

### 2. Establish Coverage And Baseline

If benchmarks do not cover the scoped work, add benchmark coverage first and
push it separately from optimization changes. Do not put the benchmark and the
optimization in the same commit; doing so erases the pre-optimization baseline.
New benchmarks must pass the noise and parity gates.

For each kernel to be optimized, verify pixel-level regression tests against an
independent reference across the kernel branches being changed: vector body,
scalar tail, and each broadcast or parameter mode. Coverage of one layout or
variant is not coverage of another kernel.

Capture the baseline before coding. Local numbers are acceptable for iteration;
CI on the benchmark-only SHA is the reference for new benchmarks. Existing
benchmarks may already have a usable CI baseline.

### 3. Profile Before Coding

Profile the baseline and name the bottleneck before writing code. Classify it as
memory-bound, compute-bound, issue-bound, launch-overhead-bound, or sync-bound,
then derive the hypothesis target from the SOL ceiling or host-side profile.
See [Profiling](#profiling).

When available, run `compute-sanitizer --tool memcheck ./bench_<op>` on the
baseline before optimizing. If the tool or required GPU environment is
unavailable, record that fact in the MR.

### 4. Run The Optimization Loop

Run this loop until no new promising leads remain. Failed attempts are dropped;
the MR is the sequence of kept attempts.

1. **Profile** the current hot configuration and name the bottleneck. If a kept
   win changed the profile, re-profile before choosing the next lead.
2. **Pick the most promising lead** from [Technique Leads](#technique-leads) or
   from deeper profiling. Do not choose by ease alone; choose by measured
   headroom and likely impact on the full regression surface.
3. **State the hypothesis** in the work log: evidence, named metric, and target,
   for example `Memory SOL 38% -> >=70%, about 1.8x`.
4. **Implement and test one change** at the level the bottleneck dictates:
   kernel, launch config, host-side path, or shared utility. Reuse local
   primitives before adding generic abstractions.
5. **Evaluate** correctness, profile movement, benchmark delta, and operational
   memory footprint against the baseline. Re-profile with the same flags and use
   `ncu --diff` when `ncu` drove the hypothesis. An unchanged-memory claim may
   use code inspection. Any positive peak attributable increase requires a
   before/after measurement on a representative workload. Record whether the
   change adds a runtime CUDA allocation or free path, and apply Golden Rule 4.
6. **Review/refactor gate** before lock-in. Inspect the attempted diff like a
   reviewer, then run `python3 tools/refactor_op.py <Operator> --phase assess`
   to catch operator-scoped redundancy such as duplicated Tensor/VarShape
   kernels, local helpers that should use shared utilities, and dead code. Fix
   every relevant recommendation introduced or exposed by the attempt before
   committing, or document why intentional duplication is required for
   performance, translation-unit locality, ABI, or another concrete technical
   reason. If a refactor is applied, run `refactor_op.py --phase verify`, the
   frozen operator tests, and the targeted benchmark again before committing.
7. **Lock in or strike** by the criteria below. Keep a successful attempt as a
   `perf:` commit. Drop or revert a struck lead and record the evidence. Never
   keep a change whose only evidence is an unprofiled wall-time improvement.
8. **Rinse and repeat**: after a kept win, re-profile and pick the next most
   promising lead. Update the profile/lead coverage map after each kept win or
   strike. Stop only after the new-lead strike budget is exhausted.

A numerical-test failure is either a logic regression or an explicit precision
trade. Drop logic regressions. Document precision trades in the commit body and
request human-maintainer review. Do not silently bump tolerances.

### 5. Finish The MR

After a kept win, push for CI evidence. If CI flags an expected improvement,
update the benchmark baseline in the same MR using the baseline tooling in
[bench/README.md](../../bench/README.md). Import collected JSON run artifacts
with `bench/_internal/update_baseline.py`; do not hand-edit baseline blocks. Before review,
`tools/optimize_op.py <Op> --phase evidence --base <ref>` must pass ODO-7,
which runs `bench/_internal/validate_baselines.py --reject-regressions-from <ref>` across the
committed baseline set and rejects same-key committed baseline slowdowns. New
benchmark rows with no previous comparable baseline are seeded through the
normal CI baseline workflow; existing rows must not be reset slower unless
there is an explicit maintainer-reviewed exception outside the optimization
done-gate.

Use `perf:` as the subject prefix for kept optimization commits. Keep profiler
artifacts such as `.ncu-rep` and `.nsys-rep` files local; summarize their
evidence in the commit body or MR description.

If a correctness bug in the operator kernel is fixed during the campaign,
preserve the bug-fix workflow as an immediate `test(<operator>):` then
`fix(<operator>):` pair. The test commit must change only `tests/` and include
an operator-specific regression; unscoped, unpaired, or implementation-bearing
test commits do not bypass the `perf:` hygiene gate.

Maintain the MR results summary throughout development. Local numbers must be
clearly labeled provisional. Before review, add complete CI results for every
reference SKU and set the block to `final`. Local rows may remain only with the
prominent non-reference warning. See
[Results Summary Format](#results-summary-format).
The MR description is the authoritative home for the Results Summary. Keep
optimization evidence as summary text in the MR, and keep any
`tools/optimize_op.py --phase evidence` results file as a temporary local
artifact generated from that MR description.

Close the campaign only when every optimization-target configuration is
optimized or measurement-backed-triaged, and the remaining headroom has
exhausted the new-lead strike budget below. Every row in the full benchmark
surface must be present in the result data validated by the harness;
layout-conversion reference rows need no-regression evidence but no
optimization-lead disposition. Do not expand the concise MR summary with a
per-configuration detail table.

## Accept, Drop, And Stop Criteria

Use these definitions for every attempt, lead, and campaign decision.
An attempt is one atomic optimization change, committed separately if kept.
A lead is one optimization idea matched to a profiled bottleneck. A campaign is
one MR's optimization effort for one operator.

**Success**: keep the attempt as its own commit only when all are true:

- functional and unit tests stay green;
- the targeted metric moved in the hypothesized direction;
- the benchmark gain clears the config's noise band; and
- the regression surface has no slowdown beyond noise; and
- operational memory grows by no more than 10 MB with no new runtime CUDA
  allocation/free path; any other result has explicit human review before
  submission; and
- the review/refactor gate is clean: relevant `refactor-op` recommendations are
  fixed, or intentional duplication is documented with a concrete technical or
  measured reason.

A gain that clears the noise band is a success even if it misses the hypothesis
target. The hypothesis sizes the opportunity; it is not the pass/fail line.

**Strike**: drop or revert the lead when the result is within noise, the
targeted metric does not move, tests fail, any regression surface benchmark
slows down beyond noise, or an increase above 10 MB or new runtime CUDA
allocation/free path does not receive explicit human review. Do not submit such
a memory result while review is unresolved. Log one line: what was tried, what
metric failed to move, which config regressed, or the measured memory increase.
A mechanical correction to the same idea, such as fixing a launch-bound bug or
instrumentation mistake, may be retested, but do not turn one lead into an
open-ended search.

**New-lead strike budget**: after each kept win, re-profile and start a fresh
search for the next most promising lead. A successful lead resets the strike
count. Three struck new leads after the last kept win exhaust the campaign for
the currently scoped operator/configuration surface. A dropped lead does not
finish a configuration that still has measured headroom; it must be covered by
another successful lead, triaged at ridge, or counted toward this exhaustion
rule.

**Triage**: leave a configuration unoptimized only with measurement-backed
evidence: it is already at ridge, or the applicable leads struck out. Estimated
effort, implementation risk, or "low ROI" is not sufficient until the config has
been profiled.

**Stop**: the campaign is exhausted only when every surveyed configuration is
optimized or triaged, and the new-lead strike budget for remaining headroom is
spent.

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

Further reading: NVIDIA TensorRT-LLM's performance-analysis skill has a broader
profiler-agnostic methodology:
<https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/skills/perf-analysis/SKILL.md>.

## Results Summary Format

The MR description is the human-readable report and the input to the
deterministic done-gate. Keep exactly one bounded, versioned summary block. The
visible portion is deliberately concise; full per-case data remains in the
benchmark artifacts and is checked by `tools/optimize_op.py` rather than copied
into the description.

This contract describes one operator-optimization campaign. Use the exact MR
scope `perf(bench)` for aggregate benchmark-suite, harness, configuration, or CI
maintenance that has no runtime implementation changes; CI verifies that scope
against the complete diff. A live v1 marker always opts into this contract.

### Metadata Contract

Begin the block with a standalone `cvcuda-optimize-summary:v1` HTML-comment
line containing compact JSON. End it with the matching standalone end marker.
Markers inside fenced code examples do not count; duplicate or nested live
blocks are invalid. The object has the following required fields and may add
the optional `impact_metric` and `secondary_operators` fields described below:

```markdown
<!-- cvcuda-optimize-summary:v1 {"operator":"ExampleOp","state":"final","baseline_commit":"<baseline commit SHA>","candidate_commit":"<candidate commit SHA>","optimized_cases":["<exact expanded benchmark case key>"]} -->
...
<!-- /cvcuda-optimize-summary:v1 -->
```

`state` is exactly `provisional` or `final`. `optimized_cases` contains the
unique exact case keys intentionally covered by the optimization; it is not a
list of only the cases that got faster. The harness rejects duplicate, missing,
or unknown keys, a candidate SHA that is not the revision being validated, and
a baseline SHA that is not its ancestor. Do not put calculated statistics or
human-authored evidence in the hidden object. The visible Markdown remains the
authoritative report for those claims.

`impact_metric` is optional and defaults to `cpp_time`, preserving summaries
that predate the field. Its only other allowed value is `python_overhead`. The
generator omits the default from canonical metadata and emits
`"impact_metric":"python_overhead"` only when the campaign targets Python
binding overhead.

`secondary_operators` is an optional array of operator names whose generated
baseline config must change because the primary implementation shares the
optimized runtime path. A declaration does not authorize broader scope by
itself: the internal code-reviewed policy must allow the exact MR IID, primary
operator, and complete secondary set, and CI requires the changed operator
configs to match that declaration exactly. Missing policy, an undeclared or
unchanged secondary, or any additional operator fails closed. The normal rule
remains one MR and one summary per operator. When exercising an approved
exception locally, pass the exact MR IID to the evidence gate with `--mr-iid`.

`baseline_commit` identifies the revision that supplies the before benchmark
measurements. It is distinct from `--base`, which identifies the changed-set
reference for the MR evidence gate; they may resolve to the same commit but are
not interchangeable. The baseline must be an ancestor that precedes the
candidate, never the candidate itself. Both metadata revisions are full
lowercase 40-character commit SHAs; the generator resolves symbolic refs before
writing them.

A summary configuration is one unique expanded benchmark case key, independent
of SKU and benchmark driver. The full set is the union expanded from every
declared `basic` or `advanced` candidate entry, including reference-only
comparison cases; it is never only the optimized, touched, or result-present
rows. Candidate nested baseline keys must equal that union exactly. The
optimized set is `optimized_cases`. A fake-planar comparison case is
reference-only and cannot be in that set. Summary generation and validation
reject a partial FakePlanar surface instead of silently omitting unmatched
cases from the layout-comparison statistics.

### Canonical Visible Format

Use the following headings, tables, labels, and order. Generator-owned values
must be regenerated rather than hand-edited.

```markdown
## ExampleOp optimization summary

> Final reference-hardware evidence · base `<baseline SHA>` · candidate `<candidate SHA>`

**Primary bottleneck: Memory-bound**

Representative profiles showed 88–94% Memory SOL versus 23–38% Compute SOL,
with `long-scoreboard` as the dominant stall.

### Scope

**Configurations optimized:** 18 / 42 total

**Optimized categories:**

- `Tensor Interleaved RGBF32`
- `Tensor Planar RGBF32`
- `VarShape Interleaved RGB8`
- `VarShape Planar RGB8`

### Impact

Speedup is `before / after`; `2.00x` means twice as fast.

| SKU | Configuration scope | Configurations | Min | Median | Max |
|---|---|---:|---:|---:|---:|
| A100 | Optimized | 18 | 1.18x | 1.74x | 3.02x |
| A100 | Full operator | 42 | 0.99x | 1.11x | 3.02x |
| H100 | Optimized | 18 | 1.12x | 1.58x | 2.71x |
| H100 | Full operator | 42 | 0.99x | 1.08x | 2.71x |

### Layout comparison

Timing ratio is `numerator / Planar`; values above `1.00x` mean Planar is faster.

| SKU | Comparison | Before min / median / max | After min / median / max |
|---|---|---:|---:|
| A100 | Interleaved / Planar | 0.81x / 0.94x / 1.08x | 1.14x / 1.39x / 1.67x |
| A100 | FakePlanar / Planar | 0.96x / 1.10x / 1.27x | 1.32x / 1.57x / 1.93x |
| H100 | Interleaved / Planar | 0.84x / 0.97x / 1.11x | 1.09x / 1.31x / 1.54x |
| H100 | FakePlanar / Planar | 0.98x / 1.13x / 1.30x | 1.27x / 1.48x / 1.82x |

### Evidence checklist

- [x] **Pixelwise equality to reference** — `<named test/command, independent reference, assertion, branches, and passing result>`
- [x] **Memory-footprint checks** — `<ODO-9 evidence and passing result>`
- [x] **Baselines updated** — `<baseline artifacts imported and generated files>`
- [x] **Baseline validation** — `<command against the previous baseline and zero regressions>`
- [x] **Lead exhaustion** — `<at-ridge profile evidence or three measured post-win strikes>`
- [x] **Review/refactor gate** — `<review, refactor assess/verify as applicable, tests, and result>`

### Top learnings

- Vectorized RGB loads made Tensor Interleaved RGBF32 configurations up to 3.02x faster.
- Wider launch blocks increased register spilling and degraded VarShape performance, so that attempt was dropped.
```

The primary bottleneck value is exactly `Memory-bound` or `Compute-bound` and
must appear before Scope. Follow it with measured profiler evidence supporting
the classification. The binary top-level assessment does not replace the more
specific issue, launch, or synchronization diagnosis used in the campaign.

`Configurations optimized: X / Y total` reports intentional scope, not how many
cases improved. `X` is the number of unique optimized case keys; `Y` is the
candidate's complete union of expanded keys from all declared `basic` and
`advanced` entries. List each distinct optimized category once in
`Container Layout Type` order, for example
`Tensor Interleaved RGBF32`. The harness derives these categories from the exact
keys, sorts the complete category strings lexicographically, and rejects
missing, extra, or differently ordered categories.

Derive Container from `inputKind` (`Tensor`, `TensorBatch`, or `VarShape`),
defaulting to `Tensor` only for benchmarks without a container axis. Map
`NHWC`/`HWC` to
`Interleaved`, `NCHW`/`CHW` to `Planar`, and an absent layout axis to
`NoLayout`. Render the type as a friendly image type such as `RGB8`, `RGBA8`,
or `RGBF32` when channel semantics are known; otherwise use the canonical
scalar/vector type such as `U8`, `F32`, or `2S16`. Use `A→B` for conversions.
Do not list shape, tier, mode, or other narrow benchmark axes as categories.

The Impact table always contains two rows for every reported SKU: `Optimized`
and `Full operator`. With the default `cpp_time` metric, calculate each case's
speedup as the baseline C++ GPU time divided by candidate
`gpu_time_us_cpp`. Report the minimum, standard median, and maximum across the
named case set as factors rounded to two decimals. Do not substitute geomean,
Python time, percentage improvement, or only improved cases. This default uses
the canonical explanation and table shown above, so existing v1 summaries
remain valid without an `impact_metric` field.

With `impact_metric` set to `python_overhead`, derive each case's overhead as
`gpu_time_us_python - gpu_time_us_cpp` separately for the baseline and
candidate. Derive reduction as `before - after`, so positive values mean less
Python overhead. The canonical Impact section instead uses this deterministic
microsecond table:

```markdown
Python overhead is `gpu_time_us_python - gpu_time_us_cpp`; reduction is `before - after`, so positive values mean less overhead.

| SKU | Configuration scope | Configurations | Before min / median / max | After min / median / max | Reduction min / median / max |
|---|---|---:|---:|---:|---:|
| A100 | Optimized | 18 | 12.00 µs / 18.00 µs / 24.00 µs | 2.00 µs / 4.00 µs / 8.00 µs | 8.00 µs / 14.00 µs / 22.00 µs |
| A100 | Full operator | 42 | 10.00 µs / 17.00 µs / 25.00 µs | 1.00 µs / 5.00 µs / 9.00 µs | 7.00 µs / 12.00 µs / 23.00 µs |
```

Report before, after, and reduction minimum, standard median, and maximum,
rounded to two decimal places. Every Full operator row requires before/after
`gpu_time_us_cpp` for `cpp_time`, or both `gpu_time_us_cpp` and
`gpu_time_us_python` for `python_overhead`, for every key in the full union on
that SKU. A missing tier, entry, case, or required timing is blocking; the
harness recomputes the values and counts from the benchmark data. Every SKU
present in either artifact must have that full coverage and appear in the
tables; a partially measured local SKU is blocking, not silently omitted.

For a binding-only campaign, the evidence gate additionally requires every
optimized case on every configured reference SKU to show a positive overhead
reduction larger than its combined standard error. Within each independently
collected before/after wave, pair C++ and Python from the same artifact and use
the unbiased sample standard deviation of those Python-minus-C++ gaps
(`gpu_gap_stddev_us`). The combined error is the root sum of the before and
after gap variances divided by their respective artifact counts. Do not treat
nvbench's within-process `gpu_noise_us_*` as between-artifact uncertainty, and
do not pair before/after artifacts by burn-in index. The candidate
Python-minus-C++ gap must also remain within the shared absolute parity limit.
A reduction inside that error band is not a proven optimization even when the
summary's aggregate median is positive.

The binding-only evidence gate derives this result from the committed raw
artifact fields; it does not trust the summary's selected Impact-table metric.
An MR that introduces a new optional summary metric may therefore retain the
previous parser's canonical `cpp_time` block while that MR is under review,
provided the exact Python-overhead table remains visible next to the block and
the raw-artifact significance/parity gate passes. After the schema extension is
merged, use `impact_metric=python_overhead` for subsequent binding campaigns.

The Layout comparison table also has two rows per reported SKU. Form pairs from
the same full union, matching by tier and every case axis except layout, and
require equal data volumes. For both baseline and candidate data, calculate
`Interleaved / Planar` and
`FakePlanar / Planar` timing ratios, then report min / median / max. A factor
above `1.00x` means native Planar is faster. If no valid pairs exist for a
comparison, put `n/a — no matched equivalent configurations` in both statistic
cells; the harness must verify that claim. A signature with no counterpart
layout produces no comparison, but its complete workload identity must still
be unique. When both sides exist, never silently omit the comparison or choose
arbitrarily between multiple same-layout cases; make the benchmark axes
unambiguous first.

Layout comparison remains based on `gpu_time_us_cpp` for both impact metrics.
Expanded workload identities must be unique by `(tier, complete raw axes)`,
independent of their config-entry names and whether a matching layout exists.
The summary tool carries a closed transitional inventory of exact aliases that
predate this rule. It canonicalizes each listed pair to one workload so legacy
configs are not double-weighted, rejects any new alias pair, and the inventory
entry must be removed when that operator config is migrated.

Every checklist item is mandatory. A box may be checked only when its line
contains concrete hard evidence and its corresponding done-gate check passes.
A bare `PASS`, a checked label without a named command/test/artifact and result,
or placeholder text is not evidence. Pixelwise evidence names the independent
reference, assertion, branch coverage, and passing test. Baseline evidence names
the imported artifacts/updated generated files and the passing
`bench/_internal/validate_baselines.py --reject-regressions-from <base>` result. Lead exhaustion
names at-ridge measurements or the three struck leads since the last kept win.
Review/refactor evidence includes the assessment, disposition, and, when a
refactor was applied, verification, frozen tests, and benchmark result.

Memory-footprint acceptance is owned by ODO-9 and its canonical companion
`## Memory footprint` section below. Cite that gate's hard evidence and passing
result in the checklist; do not duplicate or weaken its policy. The memory item
remains unchecked until ODO-9 passes.

Top learnings contains one to five bullets. Each bullet is one concise sentence
describing a noteworthy technique and its measured success or failure. Do not
recreate per-configuration results, split successes and failures into separate
sections, or add a sixth bullet.

### Provisional And Final States

A provisional summary uses the same visible format. Put
`Provisional local evidence` in the status line, leave any unsupported checklist
items unchecked, and show this warning immediately below the title for every
non-reference GPU represented in its statistics:

```markdown
> ⚠️ **Non-reference local GPU:** Results from NVIDIA RTX 6000 Ada are provisional and cannot satisfy final readiness; final A100 and H100 statistics are still required.
```

Reference SKUs come from `bench/config/sku_map.json`; do not hard-code them in
the validator. They are currently A100 and H100. A final summary has complete
baseline and candidate coverage for every key in the full union, plus separate
Impact and Layout comparison rows for every configured reference SKU. It also
has all six evidence items checked with hard evidence, one to five learnings,
the current candidate SHA, and no placeholders or unresolved `MANUAL` findings.
A local-SKU row may remain for context only if the warning remains visible; it
never substitutes for a reference SKU.

Use `tools/optimize_op.py` to initialize or refresh this block from benchmark
artifacts and to validate it through the evidence phase. The generator preserves
surrounding MR prose and the human-authored assessment, checklist evidence, and
learnings while regenerating scope, categories, Impact, and Layout comparison.

**Memory footprint (ODO-9)**: include this canonical section with exactly these
fields (replace the placeholders). Place it before or after, never inside, the
bounded v1 summary block; summary refresh preserves it as surrounding prose.

```markdown
## Memory footprint
Peak attributable increase: <integer> B
New runtime CUDA allocations/frees: <no|yes>
Evidence: <non-placeholder evidence>
```

Use `0 B` when there is no increase. An unchanged claim may cite code
inspection; a positive increase requires a before/after measurement on a
representative workload, summarized on the `Evidence` line. Report the
non-negative aggregate peak-live increase; do not net growth in one path
against an unrelated decrease. Do not use scaled units. The checker returns
`MANUAL` when the increase is above 10 MB or the allocation/free field is
`yes`. Such a result requires explicit human review and must not be submitted
while that review is unresolved.

## Technique Leads

These are advisory leads for loop step 2. Pick only after profiling names the
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

## Final Checklist

Each item is mandatory. An MR with any unchecked item is not ready for review.

- [ ] [Golden Rules](#golden-rules) are satisfied: pixel-wise equality evidence
  for all modified configurations/kernels, benchmark evidence, CI evidence, and
  the canonical memory-footprint evidence. Any increase above 10 MB or new
  runtime CUDA allocation/free path has explicit human review before submission.
- [ ] [Workflow](#workflow) is complete, including survey, coverage, baseline,
  full-surface profile/lead coverage, scoped attempts, review/refactor gates,
  API/ABI stability, baseline updates, and a passing comparable-baseline
  regression gate across the committed baseline set
  (`bench/_internal/validate_baselines.py --reject-regressions-from <base>` via ODO-7).
- [ ] [Accept, Drop, And Stop Criteria](#accept-drop-and-stop-criteria) are
  satisfied for every attempt, lead, and campaign decision.
- [ ] The versioned [Results Summary Format](#results-summary-format) is final,
  complete for every configured reference SKU, and has all six hard-evidence
  items checked.
