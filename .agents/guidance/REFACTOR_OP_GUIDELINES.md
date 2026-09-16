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

## Scope

`refactor-op` finds and safely applies **operator-scoped, semantic** redundancy that generic
copy-paste detection cannot express: near-duplicate Tensor/VarShape kernels that should share a
template, re-implemented validation that should call a shared helper, a local helper that shadows a
canonical `cuda_tools` utility, dead code. It complements generic, whole-corpus copy-paste
detection, while keeping its own findings operator-scoped and semantic.

`refactor-op` does look across operators, but only to answer an **operator-anchored** question it
does not: *which of **this** operator's bodies have a twin in a **sibling operator's** priv, and
therefore belong in a shared priv header* (RED-2). The comparison is scoped to `src/cvcuda/priv/**`
and reported as an advisory `RECOMMENDATION` against the operator being assessed — it is a
hoist-candidate finder, not a duplication metric. Per-op benchmark and sibling-test scaffolds are
near-identical *by construction* and are out of scope here (they are part of the frozen surface —
see below).

## How to use

```bash
python3 tools/refactor_op.py <Operator> [--phase assess|verify] [--domain impl|api|xcut|all] \
                             [--base <ref>] [--format md|json] [--out PATH] [--apply]
```
- **assess** (default) — a read-only candidate report. Every finding carries a **status**, literal
  **evidence** (`file:line`), its **id**, and a named **fix**.
- **`--all-operators`** — assess every operator in one process and emit a ranked roll-up. Prefer it
  over a shell loop: it is the same analysis, it keeps the corpus parse warm, and the roll-up is
  asserted against the per-operator reports.
- **`--variants BLOCK`** — group every definition of `BLOCK` across `src/cvcuda/priv/**` into
  equivalence classes. Settles "how many genuinely different versions exist, and who shares each"
  before a hoist is planned, instead of reading N copies side by side. Note it groups
  *syntactically*: two members of one class can still differ semantically in a way that matters, so
  read the largest class before hoisting it.
- **`--simulate-hoist BLOCK[,BLOCK...]`** — predict the RED-2 delta if those blocks moved to a
  shared header, per operator, with the residual names that keep an id open. Run this **before**
  writing the plan: an operator that drops but does not reach zero reports `redundancy reduced`,
  not `resolved`, and a plan that assumed otherwise is wrong.
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
| RED-2 | The same body in **another operator's** priv | shingle-hash `P.priv`'s bodies and every other `src/cvcuda/priv/**` body; report this operator's block *names* with a `jaccard ≥ threshold` twin outside `P.priv` | `RECOMMENDATION` per duplicated name (uncapped — every twin is a lead); `PASS` if none | Hoist the shared body into `src/cvcuda/priv/<Name>.cuh` (namespace `cvcuda::priv::<area>`) and include it from every consumer; priv headers need no CMake edit (cf. `AdjustColorCommon.cuh`, `OpHQResizePlanar.cuh`). |
| RED-3 | Dispatch guard narrower than the declared dtype support | compare each `sizeof(T) == N` admission gate in `P.priv` against the byte widths of the dtypes the public header declares | `MANUAL` if a declared width is excluded; else `PASS` | Check whether the excluded widths work behind the gate; widening one removes a fallback path rather than adding code. If they genuinely cannot, record why. |
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
| RED-12 | Shared util reimplemented under a different name | match the *shape* of a canonical utility rather than its name (RED-10 is name-only, so it is blind whenever the local spelling differs — the usual case) | `MANUAL` per matched shape; else `PASS` | Replace the local reimplementation with the canonical `cuda_tools` utility. |

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
| VER-6 | Bit-exact (`MANUAL`) | `build-rel/bin/cvcuda_test_system --gtest_filter='Op<Op>*'` green (frozen `EXPECT_EQ`); with no build, gather device-code evidence first (see below) |
| VER-7 | Redundancy resolved (`MANUAL`) | re-running assess no longer reports the applied finding, **or reports it with a strictly smaller count** (`redundancy_reduced`) |

A `GAP` on any of VER-1..5 is a hard failure: revert the offending change (it is feature/measurement
drift, not a refactor) or split it into its own MR. The gate passes when VER-1..5 are `PASS` and the
VER-6/VER-7 proofs have been run green.

### VER-6 device-code evidence when there is no build (`NEEDS-LOCAL-PROOF`)

VER-6 is `MANUAL` because it needs a build and a GPU, so verify reports `NEEDS-LOCAL-PROOF` and the
leg is routinely deferred to CI. `tools/device_code_proof.py <Op> --base <ref>` (the
`/device-code-proof` skill) closes most of that gap in seconds: it compiles the changed translation
unit on both sides — no build tree, no GPU — and classifies the per-kernel difference as
`identical` / `signature-only` / `schedule-only` / `changed` / `added` / `removed`. Flags come from
`compile_commands.json` when one exists (`ninja -C build-rel -t compdb > build-rel/compile_commands.json`);
otherwise it synthesizes a representative set and stubs the CMake-generated version headers.

Read it as **evidence for** VER-6, never as a substitute for the frozen tests. It proves device code
for the arch(es) compiled and nothing else: not other arches, and nothing host-side (validation
order, exception types, message text, workspace sizing, API/ABI — those are VER-1..VER-5 and
review). `signature-only` and `schedule-only` are expected for some legitimate changes and are also
what a silently reordered parameter list looks like, so they exit non-zero and must be explained,
not accepted. A non-empty diff is not a failure; an *unexplained* non-empty diff is not a pass —
isolate the cause with a control compile (`--revert <file>`, then narrow to the hunk) and name it.

Stronger tiers exist and should be preferred when a build is available: comparing the built object's
device symbol table (`cuobjdump --dump-elf-symbols`, the `MIG-11` identity proof in
`.agents/guidance/MODERNIZE_OP_GUIDELINES.md`) covers every arch the build emits, and `cmp` on the
extracted `.nv_fatbin` section shows a change was pure relocation. `tools/tu_cost.py` answers a
different question — compile time and `.text` size — and the two are complementary.

### Refactor summary (deterministic, appended to `--phase verify`)

Every verify run ends with a quantified impact summary — the headline deliverable of the refactor,
computed from artifacts (not narrated by the agent):

- **LOC delta** on the implementation/binding files vs `<base>` (`git diff --numstat`):
  `+insertions / -deletions (net)`. A unification should net-reduce lines.
- **Redundancy resolved** — the `RED-*` finding ids that assess reported at `<base>` and no longer
  reports on the working tree (the before→after of VER-7). `introduced` should be empty.
- **Redundancy reduced (still open)** — ids assess still reports, but *fewer times* than at `<base>`,
  as `id base->now`. An id can be legitimately partially open: RED-2 emits one finding per duplicated
  body, so hoisting four of an operator's five shared helpers leaves the id open while still being
  the entire point of the change. Without this line such a refactor reads as `resolved: none`.
- **Redundancy still open** — `RED-*` ids assess still reports (remaining opportunities).

Example: `RED-6 resolved; impl/binding net -38 LOC; parity OK`.
Example: `RED-2 5->1 reduced; impl net -182 LOC; parity OK`.

---

## Curated data (reviewed material)

These values tune the checker without code edits.

similarity-threshold = 0.80
min-block-lines = 6

`min-block-lines` is the smallest body the checker fingerprints; `--min-block-lines N` overrides it
per run. Sweep it downwards when planning a refactor (the `/refactor-op` skill does this): the
default hides small shared helpers, and a one-line helper defined in five operators still belongs
in a shared header. Two properties make the low end safe to read:

- Below the 5-line shingle window a body is hashed **whole**, so short blocks match only on exact
  equality — jaccard 1.0 or 0.0, nothing between. A floor-2 hit is a verbatim twin. *Do not delete
  that fallback in `shingle_hashes` as dead code: it is unreachable only while the floor is at or
  above the window, and every sweep below 5 depends on it.*
- `similarity-threshold` does almost nothing for RED-2 — measured across the tree, 39 of 41 findings
  sit at exactly 1.00, and the operator set is unchanged anywhere from 0.70 to 1.00. It is retained
  because **RED-1** genuinely needs it: RED-1 finds *near*-duplicate Tensor/VarShape kernels that
  differ in their addressing, and its finding count moves 82 → 30 → 28 across 0.60 → 0.80 → 1.00.

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
RED-1/RED-2/RED-6 (e.g. two kernels that must stay separate for a measured reason). Format:
`op: <name>, <name>` (one line per operator). Extend deliberately, with justification.
```text
# resize: someIntentionallyDuplicatedKernel
# (populate per operator as confirmed-intentional duplications are reviewed)
```
