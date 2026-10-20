[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — MR results summary

Read when writing, refreshing, or verifying a perf MR's optimization summary.
The generator in `.agents/tools/optimization_summary.py` renders every numeric
table and every evidence line from recorded tool output
(`.agents/tools/optimization_evidence.py`); the author writes only the
generalization review and optional surrounding prose.

## Results summary format

The MR description is the human-readable report and the input to the
deterministic done-gate. Keep exactly one bounded, versioned summary block. Do
not copy per-case data into it; `tools/optimize_op.py` checks that in the
benchmark artifacts.

This contract describes one operator-optimization campaign. Use the exact MR
scope `perf(bench)` for aggregate benchmark-suite, harness, configuration, or CI
maintenance that has no runtime implementation changes; CI verifies that scope
against the complete diff. A live summary marker always opts into this contract;
any marker version other than `v3` is rejected.

### Metadata Contract

Begin the block with a standalone `cvcuda-optimize-summary:v3` HTML-comment
line containing compact JSON. End it with the matching standalone end marker.
Markers inside fenced code examples do not count; duplicate or nested live
blocks are invalid. The object has the following required fields and may add
the optional `impact_metric` and `baseline_retention` fields described below:

```markdown
<!-- cvcuda-optimize-summary:v3 {"operators":["ExampleOp"],"state":"final","baseline_commit":"<baseline commit SHA>","candidate_commit":"<candidate commit SHA>","evidence":{...}} -->
...
<!-- /cvcuda-optimize-summary:v3 -->
```

`operators` lists every operator whose generated baseline config this MR
changes, primary first. The primary is the operator that scope authorization
and the evidence gates key on. `state` is exactly
`provisional` or `final`. Do not put calculated statistics or authored claims in
the hidden object.

`evidence` is the machine-readable record every evidence line is rendered from.
Its sections are `tests`, `profile`, `attempts`, `memory`, `review`,
`baselines`, and `reference_repair`; `optimization_evidence.py` defines and
validates each. Every value in it comes from a test report, benchmark artifact,
profiler report, or command exit status, and each record names the commit it
was measured at. Validation re-renders the checklist, bottleneck, profile line,
learnings, and Memory footprint section from the record and rejects any
difference, so the visible report cannot drift from the evidence.

There is no field naming the configurations an optimization "covers". Scope is
measured, not declared: a configuration is Improved or Regressed when its
committed baseline moved by at least the threshold the contract fixes, and every
other configuration is reported under All configurations. Nothing in the
authoring path can narrow the set a summary reports on.

`impact_metric` is optional and defaults to `cpp_time`. Its only other allowed
value is `python_overhead`, for campaigns targeting Python binding overhead; the
generator omits the default from canonical metadata.

`baseline_retention` is optional and names a code-reviewed policy that allows an
MR to keep committed baselines unchanged; the parser never authorizes it, and
the evidence gate and CI enforce the named policy independently.

Declaring more than one operator does not authorize broader scope by itself: the
internal code-reviewed policy must allow the exact MR IID, primary operator, and
complete secondary set, and CI requires the changed operator configs to match the
declaration exactly. An undeclared or unchanged operator, or any additional one,
fails closed. The normal rule remains one MR and one summary per operator. When
exercising an approved exception locally, pass the exact MR IID to the evidence
gate with `--mr-iid`.

`baseline_commit` identifies the revision that supplies the before benchmark
measurements. It is distinct from `--base`, which identifies the changed-set
reference for the MR evidence gate; they may resolve to the same commit but are
not interchangeable. The baseline must be an ancestor that precedes the
candidate, never the candidate itself. Both metadata revisions are full
lowercase 40-character commit SHAs; the generator resolves symbolic refs before
writing them.

A case present at only one commit is never compared, so a renamed or newly
added case cannot appear as a speedup.

### Canonical Visible Format

Use the following headings, tables, labels, and order. Generator-owned values
must be regenerated rather than hand-edited.

```markdown
## Optimization summary

> Final reference-hardware evidence · base `<baseline SHA>` · candidate `<candidate SHA>`

**Primary bottleneck: Memory-bound**

Nsight Compute: `ExampleKernel` takes 97.0% of profiled time at 93.0% Memory SOL versus 21.0% Compute SOL. Top finding: UncoalescedGlobalAccess (estimated 40% speedup).

### Scope

Counted per configuration, not per measurement; Improved and Regressed count configurations that moved at least 5% on any reported SKU.

| Operator | Configurations | Improved | Regressed |
|---|---:|---:|---:|
| ExampleOp | 42 | 18 | 1 |
| SiblingOp | 24 | 6 | 0 |

### Impact

Speedup is `before / after`; `2.00x` means twice as fast.

| Operator | SKU | Scope | Configurations | Min | Median | Max |
|---|---|---|---:|---:|---:|---:|
| ExampleOp | A100 | All configurations | 42 | 0.92x | 1.11x | 3.02x |
| ExampleOp | A100 | Improved | 18 | 1.18x | 1.74x | 3.02x |
| ExampleOp | A100 | Regressed | 1 | 0.92x | 0.92x | 0.92x |
| SiblingOp | A100 | All configurations | 24 | 0.99x | 1.31x | 2.04x |
| SiblingOp | A100 | Improved | 6 | 1.22x | 1.55x | 2.04x |

### Layout comparison

Timing ratio is `numerator / Planar`; values above `1.00x` mean Planar is faster.

| Operator | SKU | Comparison | Before min / median / max | After min / median / max |
|---|---|---|---:|---:|
| ExampleOp | A100 | Interleaved / Planar | 0.81x / 0.94x / 1.08x | 1.14x / 1.39x / 1.67x |
| ExampleOp | A100 | FakePlanar / Planar | 0.96x / 1.10x / 1.27x | 1.32x / 1.57x / 1.93x |
| SiblingOp | A100 | Interleaved / Planar | 0.88x / 0.97x / 1.12x | 1.05x / 1.22x / 1.44x |
| SiblingOp | A100 | FakePlanar / Planar | n/a — no matched equivalent configurations | n/a — no matched equivalent configurations |

### Generalization review

| Area | Assessment | Decision |
|---|---|---|
| Layouts / input kinds | `LoadVector` also applies to the Tensor planar path. | Applied: `ExampleOp.planar` tests and basic,advanced benchmarks pass. |
| Dtypes / channels | `ConvertF16` is a separate conversion path that could use wider loads. | Deferred: keep this MR scoped to the F32 path. |
| Shapes / parameters | `LoadVector` and its scalar tail cover small and odd widths. | Applied: `ExampleOp.tail` tests and basic,advanced benchmarks pass. |
| Shared code / sibling operators | `LoadVector` is operator-private; sibling kernels already use vector loads. | Not applicable: sibling paths already implement this technique. |

### Evidence checklist

- [x] **Pixelwise equality to reference** — 38 passing referenced tests cover all 12 benchmarked categories
- [x] **Memory-footprint checks** — attributable peak increase 0 B
- [x] **Baselines updated** — imported 2 baseline file(s) measured at 222222222222
- [x] **Baseline validation** — validate_baselines --reject-regressions-from origin/main passed
- [x] **Lead exhaustion** — hot kernel at ridge: 93.0% SOL >= 90%
- [x] **Review/refactor gate** — refactor assess PASS; no refactor applied

### Top learnings

- Vectorized loads was kept at 1.74x median (1.18x to 3.02x over 18 cases).
- Wider launch blocks was struck at 0.94x median over 18 cases.
```

The bottleneck and the line after it are derived from the `profile` record:
`Memory-bound` when the hot kernel's Memory SOL is at least its Compute SOL,
otherwise `Compute-bound`. A final summary requires the record; a provisional
one without it renders a pending profile line and warns. The binary top-level
assessment does not replace the more specific diagnosis used in the campaign.

The summary generator derives Scope, Impact, and Layout comparison from the
committed baselines (C++ GPU time for `cpp_time`; `gpu_time_us_python -
gpu_time_us_cpp` before/after for `python_overhead`), and
`.agents/tools/optimization_summary.py` defines the exact tables and rejects
hand edits. Every SKU present in either
artifact needs before/after timings for every key in the full union; a missing
tier, entry, case, or timing is blocking, not silently omitted.

For a binding-only campaign, ODO-12 requires every case in the full before/after
union to show an overhead reduction larger than its standard error on every
reference SKU. Generalization deferrals do not waive this gate. Require repeated
candidate artifacts with paired C++/Python gap dispersion (`gpu_gap_stddev_us`).
Use the combined before/after standard error when the committed baseline has
dispersion data: `sqrt(before_stddev² / before_runs + after_stddev² / after_runs)`.
Otherwise use `after_stddev / sqrt(after_runs)` against the fixed committed
reference and label it accordingly; do not remeasure the before revision.
Within-process `gpu_noise_us_*` is not between-artifact dispersion. The candidate
gap must also pass the shared absolute parity limit. The gate reads committed
artifact fields independently of the summary's selected Impact-table metric.

Expanded workload identities must be unique by `(tier, complete raw axes)`; when
two same-layout cases share a signature, make the benchmark axes unambiguous
rather than letting the pairing choose.

Every checklist line is rendered from the evidence record; a box is checked
only when the same gate function the done-gate runs passes on that record:

- **Pixelwise equality** passes when every recorded case ran at the candidate
  commit and passed, and the proving cases (an equality assertion against an
  independent reference) cover every (container, layout, dtype) category the
  benchmark surface requires.
- **Memory-footprint checks** passes when every allocation is attributed to a
  CV-CUDA call site, no site is new, and the aggregate peak increase is at most
  10 MB; anything else is `MANUAL` and needs human review before submission.
- **Baselines updated** names the guarded import and the committed files it
  wrote, whose digests CI compares with the source revision. An approved
  retention exception renders **Baselines verified unchanged** instead.
- **Baseline validation** is the recorded
  `bench/_internal/validate_baselines.py --reject-regressions-from <base>` run.
- **Lead exhaustion** passes when the hot kernel is at the ridge (at least 90%
  SOL), three leads were struck after the last kept win, or the generalization
  review defers the remaining opportunities.
- **Review/refactor gate** is the recorded refactor assessment and, when a
  refactor was applied, its verification and post-refactor proof.

Top learnings are drafted from the recorded attempts, one per lead, at most
five. Do not edit any rendered line; record new evidence and refresh instead.

### Generalization Review

Inside the summary block, place `### Generalization review` after Layout
comparison. Use the `Area | Assessment | Decision` table shown above, with at
least one row for each of the four exact area labels. Assess every kept technique;
group opportunities when they share a decision and add rows otherwise.

The assessment names the opportunity or evidence that none applies, citing a
code/config location or existing profile measurement. The decision is one of
`Applied: <validation>`, `Deferred: <scope reason>`, or
`Not applicable: <reason>`. Applied decisions name validation and its result.
Deferral is an explicit choice of MR scope, not a claim that the opportunity
failed. No separate approval or implementation trial is required. Final
summaries require a complete review; provisional summaries may retain
placeholders. ODO-3 validates this section without another checkbox.

### Provisional And Final States

A provisional summary uses the same visible format. Put
`Provisional local evidence` in the status line, renders unchecked items for
missing or failing records, and shows this warning immediately below the title for every
non-reference GPU represented in its statistics:

```markdown
> ⚠️ **Non-reference local GPU:** Results from NVIDIA RTX 6000 Ada are provisional and cannot satisfy final readiness; final A100 and H100 statistics are still required.
```

Reference SKUs come from `bench/config/sku_map.json`; do not hard-code them in
the validator. A final summary has complete
baseline and candidate coverage for every key in the full union, plus separate
Impact and Layout comparison rows for every configured reference SKU. It also
has every evidence item checked from its record, a completed generalization
review, the current candidate SHA, and no placeholders or unresolved `MANUAL`
findings. A local-SKU row may remain for context only if the warning remains
visible; it never substitutes for a reference SKU.

Initialize or refresh this block from benchmark artifacts and the evidence
record, then validate it with `tools/optimize_op.py --phase evidence`. Refresh
preserves surrounding MR prose and the generalization review while regenerating
everything else.

### Memory footprint

ODO-9 requires this section. The generator renders it immediately after the
bounded block from the `memory` record:

```markdown
## Memory footprint
Peak attributable increase: <integer> B
New runtime CUDA allocations/frees: <no|yes>
Evidence: nsys per-site peak on `<workload>` (baseline <report>, candidate <report>).
```

The increase is the non-negative sum of per-site peak-live growth, so growth in
one call site is never netted against an unrelated decrease. Sites are
attributed by the innermost CV-CUDA NVTX range active at allocation time.
