[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Operator Refactoring Guidelines (`/refactor-op`)

The single source of truth for the per-operator refactoring / redundancy-reduction harness. It is
**implemented by** `tools/refactor_op.py` (the deterministic checker) and **cited by** the thin
skill wrapper (`.agents/skills/refactor-op/SKILL.md`). The catalog substance lives here once; tools
delegate to it instead of duplicating the workflow.

This document does not restate house policy; it *cites* the authoritative docs and turns their
requirements into concrete, machine-checkable items: `AGENTS.md` ("Follow existing local patterns",
"do not mix unrelated refactors into functional changes", SPDX),
`.agents/guidance/REVIEW_OP_GUIDELINES.md` (the support/test matrices reused for parity), and
`.agents/guidance/OPTIMIZATION_GUIDELINES.md` (bit-exact default).

## Scope, and the relationship to SonarQube

`refactor-op` finds and safely applies **operator-scoped, semantic** redundancy that generic
copy-paste detection cannot express: near-duplicate Tensor/VarShape kernels that should share a
template, re-implemented validation that should call a shared helper, a local helper that shadows a
canonical `cuda_tools` utility, dead code. It is **complementary to SonarQube CPD**
(`sonar-project.properties` + the `sonarqube` CI job), which already owns generic, whole-corpus
duplication *amount and location*. For raw duplication metrics, read Sonar; `refactor-op` does not
re-implement clone detection across the tree. It also honors Sonar's `sonar.cpd.exclusions`
philosophy: per-op benchmark and sibling-test scaffolds are near-identical *by construction* and are
out of scope here (they are part of the frozen surface — see below).

## How to use

```bash
python3 tools/refactor_op.py <Operator> [--phase assess|verify] [--domain impl|api|xcut|all] \
                             [--base <ref>] [--format md|json] [--out PATH] [--apply]
```
- **assess** (default) — a read-only candidate report. Every finding carries a **status**, literal
  **evidence** (`file:line`), its **id**, and a named **fix**.
- **verify** — the strict parity gate for an applied refactor (see "The parity gate").
- **Completion** of an assess-then-apply cycle is defined as: `--phase verify` shows **zero `GAP`**
  and every `MANUAL` has been resolved green (the VER-6/VER-7 proofs, plus any VER-3 header/binding
  inspection).

The checker is **deterministic & idempotent**: no network, no clocks, no randomness; the changed-set
comes from `git diff <base>`. The same tree yields a byte-identical report.

### Status vocabulary

| status | meaning | affects exit code |
|--------|---------|-------------------|
| `PASS` | check satisfied | no |
| `GAP` | a parity violation (verify only); an actionable hard failure | **yes (non-zero)** |
| `N-A` | not applicable (e.g. no priv files, no binding) | no |
| `MANUAL` | needs a step the checker can't take (build+run, human read); it points at the spot | no (but surfaced) |
| `RECOMMENDATION` | advisory refactoring opportunity (assess); human judges worth | no |

Assess emits only `RECOMMENDATION`/`MANUAL`/`PASS`/`N-A` — **refactoring is improvement, not a
gate**, so an assess run is always exit 0. `GAP` arises **only** in `--phase verify`, where a
parity violation must block the refactor.

### Two modes: assess → apply → verify

The agent that *performs* a refactor must also produce local proof that it changed nothing
observable. Verification is this deterministic checker, run locally when the MR scope includes a
semantic refactor; the agent applies, then the checker + the frozen tests prove parity. `--apply` is
a wrapper-level flag that emphasizes the `fix` actions — the Python checker **never writes files**.

1. **assess** — run the checker, choose a `RECOMMENDATION`/`MANUAL` to act on.
2. **apply** — the agent applies that finding's named fix on the **implementation / binding only**
   (never tests, never benchmarks), reusing existing shared utilities (`AGENTS.md`).
3. **verify** — when the MR scope includes a refactor, run `--phase verify` locally to prove
   parity; then build + run the frozen tests (bit-exact) and re-run assess (redundancy gone). If the
   same MR also changes performance-sensitive implementation, use the operator optimization workflow
   to provide benchmark evidence.

### Inviolable rules

- **Frozen test surface** — a refactor must not change the operator's *tested feature set*. Do not
  edit `tests/cvcuda/system/TestOp<Op>.cpp` or `tests/cvcuda/python/test_op<op>.py`; move shared
  helpers outside the operator test files or split the test change into a separate MR.
- **Frozen bench surface** — do not touch anything under `bench/` (sources, configs, **baselines**).
  The benchmarks are the unbiased measurement instrument; touching them could mask a regression.
- **API/ABI unchanged** — public C, C++, and Python signatures and ABI stay identical (VER-3). A
  signature change is feature work, not a refactor — split it into its own MR.
- **Bit-exact** — output must be unchanged; the frozen `EXPECT_EQ` tests are the oracle (VER-6).
  Never weaken a test, widen a tolerance, or fabricate a baseline to make a refactor "pass".
- **Scoped commits** — kept refactors are `refactor:` commits with nothing unrelated folded in.

---

## Domain: impl  (the priv implementation — kernels, validation, indexing)

| id | check | probe | status | fix |
|----|-------|-------|--------|-----|
| RED-1 | Near-duplicate Tensor/VarShape kernels or functions in priv | shingle-hash each `__global__`/function body in `P.priv`; report pairs with `jaccard ≥ threshold` | `RECOMMENDATION` (near-dup); `PASS` if none | Unify behind a templated kernel; put the addressing difference in the accessor (cf. `OpBrightnessContrast.cu` → `DoBrightnessContrast<isPlanar>`). |
| RED-4 | Re-implemented layout/dtype/channel validation | regex for manual `TENSOR_NCHW \|\| … TENSOR_NHWC` chains | `MANUAL` if present; else `PASS` | Replace hand-rolled layout checks with `nvcv` `TensorDataAccess` helpers (cf. `OpStack.cpp`). |
| RED-5 | Manual index/stride arithmetic duplicating accessors | stride math in a priv file that uses no `TensorWrap`/`ImageBatchVarShapeWrap`/`TensorDataAccess` | `MANUAL` if present; else `PASS` | Use `TensorWrap` / `TensorDataAccessStridedImagePlanar` accessors instead of hand-rolled stride math. |

## Domain: api  (the Python binding)

| id | check | probe | status | fix |
|----|-------|-------|--------|-----|
| RED-6 | Duplicated binding bodies (Tensor↔VarShape, allocating↔`_into`) | shingle-hash the function/lambda bodies in `Op<Op>.cpp`; report `jaccard ≥ threshold` | `RECOMMENDATION`; `PASS` if none | Extract a shared submit helper for the Tensor/VarShape & allocating/`_into` paths; reuse `VarShapeUtils.hpp` (`CreateSameShapeImageBatch`). |

## Domain: xcut  (cross-cutting)

| id | check | probe | status | fix |
|----|-------|-------|--------|-----|
| RED-10 | Local helper shadowing a canonical shared util | a local definition whose name ∈ the shared-util reference set | `MANUAL` if present; else `PASS` | Replace the reinvented helper with `cuda_tools/{SaturateCast,StaticCast,TypeTraits}.hpp`. |
| RED-11 | Dead code | a `static` function whose name occurs exactly once (definition only) across the op surface | `RECOMMENDATION` if present; else `PASS` | Remove the unreferenced function. |

Source: `AGENTS.md` (reuse shared utilities; don't mix refactors into functional changes),
the `cuda_tools` accessor/cast headers, and `python/mod_cvcuda/operators/VarShapeUtils.hpp`.

---

## The parity gate (`--phase verify`)

Run locally when an MR's scope includes an operator refactor. This is not a standing CI job:
optimization, feature, baseline, and bug-fix MRs should run it only when they deliberately include a
refactoring change that must preserve observable behavior. All deterministic legs are
artifact-derived (`git diff <base>` and parsed matrices); the bit-exact leg is delegated to the
frozen tests.

| id | check | PASS condition |
|----|-------|----------------|
| VER-1 | Frozen test surface | no diff under the op's test files |
| VER-2 | Frozen bench surface | no diff under the op's `bench/` sources, config, or baselines |
| VER-3 | API/ABI unchanged | no public-header signature diff; binding registration, callable-signature, and reachable-type-alias snapshots are identical |
| VER-4 | Feature-matrix parity | declared layouts/channels/dtypes identical base-vs-working |
| VER-5 | Coverage-matrix parity | test macros + parametrized value-row count identical base-vs-working |
| VER-6 | Bit-exact (`MANUAL`) | `build-rel/bin/cvcuda_test_system --gtest_filter='Op<Op>*'` green (frozen `EXPECT_EQ`) |
| VER-7 | Redundancy resolved (`MANUAL`) | re-running assess no longer reports the applied finding |

A `GAP` on any of VER-1..5 is a hard failure: revert the offending change (it is feature/measurement
drift, not a refactor) or split it into its own MR. The gate passes when VER-1..5 are `PASS` and the
VER-6/VER-7 proofs have been run green.

### Refactor summary (deterministic, appended to `--phase verify`)

Every verify run ends with a quantified impact summary — the headline deliverable of the refactor,
computed from artifacts (not narrated by the agent):

- **LOC delta** on the implementation/binding files vs `<base>` (`git diff --numstat`):
  `+insertions / -deletions (net)`. A unification should net-reduce lines.
- **Redundancy resolved** — the `RED-*` finding ids that assess reported at `<base>` and no longer
  reports on the working tree (the before→after of VER-7). `introduced` should be empty.
- **Redundancy still open** — `RED-*` ids assess still reports (remaining opportunities).

Example: `RED-6 resolved; impl/binding net -38 LOC; parity OK`.

---

## Curated data (reviewed material)

These values tune the checker without code edits.

similarity-threshold = 0.80

### Shared-util reference set
Canonical helpers a local definition must not shadow (RED-10). Header homes:
`src/cvcuda/include/cvcuda/cuda_tools/{TensorWrap,TypeTraits,SaturateCast,StaticCast,ImageBatchVarShapeWrap,BorderWrap,InterpolationWrap}.hpp`;
`src/nvcv/src/include/nvcv/TensorDataAccess.hpp` (`TensorDataAccessStridedImagePlanar`);
`python/mod_cvcuda/operators/VarShapeUtils.hpp` (`CreateSameShapeImageBatch`).
```text
SaturateCast
StaticCast
ConvertBaseTypeTo
TensorWrap
CreateSameShapeImageBatch
```

### Duplicate allowlist
Block names that are legitimately duplicated for a given operator and should be suppressed from
RED-1/RED-6 (e.g. two kernels that must stay separate for a measured reason). Format:
`op: <name>, <name>` (one line per operator). Extend deliberately, with justification.
```text
# resize: someIntentionallyDuplicatedKernel
# (populate per operator as confirmed-intentional duplications are reviewed)
```
