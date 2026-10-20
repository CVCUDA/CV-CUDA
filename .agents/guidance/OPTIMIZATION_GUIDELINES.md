[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Operator Optimization Guidelines

How to land a performance change in CV-CUDA -- kernels, host-side overhead,
caches, allocators, and Python bindings -- without regressing correctness or
neighbouring code paths. These guidelines apply to humans and AI agents.

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
- **Profile/lead coverage**: every row in the optimization-target surface maps
  to a kernel or host-path group with a disposition: improved, at ridge,
  triaged after struck leads, unchanged/no-regression with a measured reason,
  or deferred in the generalization review.
  Profile each group in the retained scope; deferred generalizations may use
  code/config inspection without profiling or implementation.
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

Survey the full benchmark surface using the committed baselines. Rank
optimization-target configurations by measured headroom, weighted by absolute
cost: a 36% `BWUtil` case taking 1.5 ms outranks a 39% case taking 0.7 ms.
Do not optimize layout-conversion reference rows (`NCHW_FAKE`/`CHW_FAKE`);
they compare a native layout path against the equivalent reformat-and-run path.

Build a profile/lead coverage map before coding. For each benchmark row, record
the kernel group or host path it exercises and the initial candidate leads.
Profile leads selected for implementation. A representative profile may cover
rows sharing the same implementation path and bottleneck.

Treat layouts and variants as separate until proven otherwise. In CV-CUDA,
NHWC vs. NCHW and Tensor vs. VarShape often use different kernels; optimizing
one does not prove the other is done.

Each attempt stays narrow: one operator, one variant, one input type, and one
optimization idea. Assess the full surface; generalizations may be explicitly
deferred in the MR. Assessing sibling operators does not authorize changing
them in the same MR.

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

Use the committed `bench/config/operators/<op>.json` numbers at the
pre-optimization revision as the before measurements. Do not regenerate existing
baselines in pre-change CI. Preflight checks coverage and availability, not
measurement age, history, sample counts, or partial-import markers.

Seed missing reference-SKU rows for new benchmarks on the benchmark-only
revision before optimizing those paths. Local measurements and profiles may
guide iteration; committed numbers remain the reference for CI comparisons.

After changing an operator, run its full `basic,advanced` surface for C++ and
Python against the committed baselines. Include affected siblings when shared
code changes. Import candidate baseline updates through the guarded workflow in
`bench/README.md`; do not hand-edit baseline blocks.

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
2. **Pick the most promising lead** from the technique-lead table in
   [optimize-op-profiling.md](../skills/op-tools/reference/optimize-op-profiling.md#technique-leads) or
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
   strike. Apply the stop criteria below.

A numerical-test failure is either a logic regression or an explicit precision
trade. Drop logic regressions. Document precision trades in the commit body and
request human-maintainer review. Do not silently bump tolerances.

### 5. Review Generalization

Review each kept technique beyond the configuration that motivated it: layouts
and input kinds, dtypes and channels, shapes and parameters, and shared code or
sibling operators. Inspect the corresponding code, benchmark configs, and
available profiles to identify where the same idea applies or why it does not.

Record the assessment in the MR's generalization review
([optimize-op-summary.md](../skills/op-tools/reference/optimize-op-summary.md#generalization-review)).
Developers may defer implementation to keep the MR scoped; state the opportunity
and scope reason. Deferred opportunities require no implementation trial or new
benchmark run. All changed paths still require correctness tests and the full
`basic,advanced` regression surface.

### 6. Finish The MR

After a kept win, push for CI evidence. If CI flags an expected improvement,
update the benchmark baseline in the same MR using the guarded workflow in
[bench/README.md](../../bench/README.md) under "Import, Validate, and Recheck";
do not hand-edit baseline blocks. Before review,
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

Formatting-only edits, `test:`/`fix:` pairs, and compiler-only `build:`
corrections have their own rules; see [Commit hygiene (ODO-6)](#commit-hygiene-odo-6).

Maintain the MR results summary throughout development. Local numbers must be
clearly labeled provisional. Before review, add complete CI results for every
reference SKU and set the block to `final`. Local rows may remain only with the
prominent non-reference warning. See
[optimize-op-summary.md](../skills/op-tools/reference/optimize-op-summary.md). Any
`tools/optimize_op.py --phase evidence` results file is a temporary local
artifact generated from the MR description.

Close the campaign when the generalization review is complete and every
optimization-target configuration is optimized, measurement-backed-triaged,
or explicitly deferred there. Exhaust leads for the retained scope using the
criteria below. Every row in the full benchmark surface must be present in
the result data validated by the harness; layout-conversion reference rows
need no-regression evidence but no
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
the retained operator/configuration scope. Deferred generalizations do not
consume this budget. A dropped lead does not finish a configuration with
measured headroom; it must be covered by another successful lead, triaged at
ridge, or counted toward this exhaustion rule.

**Triage**: requires measured evidence that the configuration is at ridge or
its applicable leads struck out. Scope, effort, or risk decisions belong in the
generalization review as deferrals.

**Stop**: every surveyed configuration is optimized, triaged, or covered by a
documented generalization deferral. Remaining leads in the retained scope are at
ridge or have exhausted the strike budget. If all remaining opportunities are
deferred generalizations, no additional strikes are required.

## Profiling

Profiling requirements, tool flags, and evidence expectations live in
[optimize-op-profiling.md](../skills/op-tools/reference/optimize-op-profiling.md), together with the
advisory technique-lead table. Read it for workflow steps 3–4.

## Results Summary Format

The MR description carries exactly one versioned optimization-summary block, the
input to the done-gate. Its metadata, canonical format, provisional/final rules,
and the companion memory-footprint section live in
[optimize-op-summary.md](../skills/op-tools/reference/optimize-op-summary.md). Read it when writing,
refreshing, or verifying the summary.

## Gate index

`tools/optimize_op.py` links every finding to the section that owns it; read
that section only when the gate reports `GAP` or `MANUAL`.

| Gate | Checks | Owner |
|---|---|---|
| PRE-1 | Correctness/regression coverage | [Establish Coverage And Baseline](#2-establish-coverage-and-baseline) |
| PRE-2, PRE-3 | Benchmark coverage and committed baselines | [Establish Coverage And Baseline](#2-establish-coverage-and-baseline) |
| PRE-4 | Profiler availability | [optimize-op-profiling.md](../skills/op-tools/reference/optimize-op-profiling.md) |
| PRE-5, ODO-5 | Public API/ABI unchanged | [Survey And Scope](#1-survey-and-scope) |
| ODO-0, ODO-1 | Implementation changed; pixelwise equality | [Golden Rules](#golden-rules) |
| ODO-2, ODO-7 | Reference baselines present; no committed same-key slowdown | [Finish The MR](#6-finish-the-mr) |
| ODO-3, ODO-10, ODO-11, ODO-12 | Summary format, declared operators, FakePlanar pairing, binding overhead | [optimize-op-summary.md](../skills/op-tools/reference/optimize-op-summary.md) |
| ODO-4 | Lead exhaustion or reviewed deferrals | [Accept, Drop, And Stop Criteria](#accept-drop-and-stop-criteria) |
| ODO-6 | `perf:` commit hygiene | [Commit hygiene (ODO-6)](#commit-hygiene-odo-6) |
| ODO-8 | Review/refactor lock-in | [Run The Optimization Loop](#4-run-the-optimization-loop) |
| ODO-9 | Memory footprint | [optimize-op-summary.md](../skills/op-tools/reference/optimize-op-summary.md#memory-footprint) |

### Commit hygiene (ODO-6)

Formatting-only C++/CUDA implementation edits do not require a `perf:` subject;
ODO-6 detects a conservative token-equivalent subset. New files, mode changes,
and semantic edits keep the normal requirement.

If a correctness bug in the operator kernel is fixed during the campaign,
preserve the bug-fix workflow as an immediate `test(<operator>):` then
`fix(<operator>):` pair. The test commit must change only `tests/` and include
an operator-specific regression; unscoped, unpaired, or implementation-bearing
test commits do not bypass the `perf:` hygiene gate.

Compiler-only compatibility corrections may use `build(<exact operator>):` without
inventing a runtime regression test. ODO-6 routes these to **MANUAL**, never automatic
PASS, only when the commit has one parent and modifies existing private operator
translation units exclusively (no shared files, headers, tests or other changes).
Supply exactly one of each Git trailer below; indented continuation lines are supported:

```text
Compiler-Failure: <full immediate-parent SHA> <failing CI job URL>
Compiler-Verification: parent-fail/candidate-pass/codegen <report SHA256> <report reference>
```

The job URL must identify a GitLab `/-/jobs/<id>` job. The reviewer must verify that
the report matches its hash and records the supported compiler's actual parent
failure, candidate success and semantic/device-code equivalence. These trailers
record claims, not validated compiler results; unresolved review still blocks completion.
Missing or malformed evidence and broader `build:` changes remain GAP.

## Final Checklist

Each item is mandatory. An MR with any unchecked item is not ready for review.

- [ ] [Golden Rules](#golden-rules) are satisfied: pixel-wise equality evidence
  for all modified configurations/kernels, benchmark evidence, CI evidence, and
  the canonical memory-footprint evidence. Any increase above 10 MB or new
  runtime CUDA allocation/free path has explicit human review before submission.
- [ ] [Workflow](#workflow) is complete, including survey, coverage, baseline,
  full-surface profile/lead coverage, generalization review and any deferrals,
  scoped attempts, review/refactor gates, API/ABI stability, baseline updates,
  and a passing comparable-baseline
  regression gate across the committed baseline set
  (`bench/_internal/validate_baselines.py --reject-regressions-from <base>` via ODO-7).
- [ ] [Accept, Drop, And Stop Criteria](#accept-drop-and-stop-criteria) are
  satisfied for every attempt, lead, and campaign decision.
- [ ] The versioned [Results Summary Format](#results-summary-format) is final,
  complete for every configured reference SKU, and has every evidence item
  checked from its recorded tool output.
