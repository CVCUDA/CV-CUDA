[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA New-Operator Guidelines (`/make-op`)

The single source of truth for adding a **new** operator to CV-CUDA end-to-end. It is
**implemented by** `tools/make_op.py` (the deterministic checker) and **cited by** the thin
per-tool wrappers (`.claude/commands/make-op*.md`, `.codex/skills/make-op*/SKILL.md`). The
checklist substance lives here once; both Claude and Codex delegate to it — no duplication.

This document also **documents the skill** (purpose, modes, workflow, invocation), exactly as
`OPTIMIZATION_GUIDELINES.md` documents `/optimize-op`. It does not restate house policy; it
*cites* the authoritative docs and turns their requirements into machine-checkable items:
`AGENTS.md`, `docs/sphinx/advanced/make_operator.rst` (the narrative tutorial),
`REVIEW_OP_GUIDELINES.md`, and `OPTIMIZATION_GUIDELINES.md`.

`make_op.py` **composes, does not duplicate**: it imports `review_op.py` (all four review
domains) and `optimize_op.py --phase preflight`, then adds the new-operator-specific spec /
scaffold / coverage / execution gates on top.

```
review_op.py        coverage primitive (support / test / bench / docs)
   └─ optimize_op.py --phase preflight   = review_op(test+bench) + baseline + profiling readiness
        └─ make_op.py                     = review_op(all 4) + preflight(RDY) + SPEC + SCF + IMP + COV + DOC-REL + EXEC
```

## The skill

Two modes:

- **(A) Full end-to-end** — `/make-op <Name>`: propose a spec + reference oracle → **get user
  approval** → scaffold (`tools/mkop/mkop.sh`) → record the approved contract in the C-API header
  → `/make-op-scaffold` (structural gate) → implement (guided by `make_operator.rst` + this doc)
  → `/make-op-verify` (the deterministic done-gate) → hand off to `/optimize-op` for the
  performance campaign.
- **(B) Scaffold-only, implementation delegated** — `/make-op-scaffold <Name> [--bare]`: produce
  a complete, wired, *building* skeleton and stop, leaving implementation to a human or another
  AI. *Spec'd* (default) records the approved contract in the header; `--bare` skips the spec
  (definition + implementation fully delegated). The done-gate stays red until someone implements
  it and runs `/make-op-verify`.

### Part 0 — Spec & approval (the operator's definition)

The checker gates completeness *against a declared contract*; this interactive step (in the
skill, not the checker) authors that contract so it is deliberate, not a guess:

1. The agent **proposes a spec**: name; one-line semantics; a **cited reference oracle** (e.g.
   "mimic `torchvision.transforms.v2.functional.invert`: `out = dtype_max − in`", or OpenCV
   `cv::bitwise_not`, or an explicit formula); API parameters; the target **support matrix**
   (dtype × channel × layout × container).
2. The agent **queries the user for approval** before scaffolding. The user may amend the
   semantics/oracle/matrix, or **explicitly waive the oracle** for a genuinely-novel custom op
   (waiver + rationale recorded).
3. On approval, the contract is recorded in the C-API header (`Op<Name>.h`): `@brief` = semantics;
   a `Reference:` line = the cited oracle (or `Reference: custom — oracle waived (<rationale>)`);
   the **Limitations table** = the approved matrix. This header **is** the single source of truth
   that COV, the gold reference, and DOC all mirror. Image operators include interleaved and
   planar layouts by default. If the inputs are not images, record `Planar image layouts: Not
   applicable` and a non-empty `Reason` beside the Limitations table.

## How to use

```bash
python3 tools/make_op.py <Operator> --phase scaffold|done \
                         [--bare] [--format md|json] [--out report.json] [--run]
```
- Each checklist item is reported with a **status**, literal **evidence**, and its **id**.
- **Completion** of the `done` phase = a re-run shows **zero `GAP` and zero unresolved `MANUAL`**.
  Final green requires `--run` (tests compile + execute + pass) and CI-seeded baselines.

### Status vocabulary

| status | meaning | affects exit code |
|--------|---------|-------------------|
| `PASS` | check satisfied | no |
| `GAP` | check failed; an actionable deficiency | **yes (non-zero)** |
| `N-A` | not applicable to this operator | no |
| `MANUAL` | needs human reading; checker points at the location | no (but surfaced) |
| `RECOMMENDATION` | advisory follow-up | no |

The static checks are **deterministic & idempotent** (no network, no clocks, no randomness);
only `--run` executes (opt-in, like `review_op --run`).

### Inviolable rules (shared with the sibling skills)
- **Never fabricate baselines** — a missing baseline is `GAP "requires CI"` → regen fan-out.
- **Bit-exact is the default** for correctness — never silently introduce or widen a
  tolerance; an unexplained `EXPECT_NEAR` is a `GAP` here (stronger than `review_op`).
- **Layout-support gaps are author work** — the checker reports missing operator capability
  and does not implement it. Planar image layouts are required by default; only an operator-local
  C-header declaration with a reason makes them not applicable.

### Operator name resolution
Same as `REVIEW_OP_GUIDELINES.md`: `<Operator>` is PascalCase; the checker derives `op`
(lowercase stem), `Op` (C++/test stem), and `pyname` (Python function), reusing
`review_op.resolve_op`.

---

## Phase `scaffold` — the wired skeleton (run right after `mkop.sh`)

### Domain: spec  (the approved contract)

| id | check | probe | PASS condition |
|----|-------|-------|----------------|
| SPEC-BRIEF | Header `@brief` authored | `Op<Name>.h` Doxygen brief | not the mkop stub ("Defines types and functions to handle…" / `TBD`); `--bare` → `MANUAL` "spec delegated" |
| SPEC-ORACLE | Reference oracle cited | header brief | a `Reference:`/`matches`/`mimics` line naming an external API (OpenCV/PIL/TorchVision/formula) **or** an explicit `oracle waived`/`custom — no external reference` token; `--bare` → `MANUAL` |
| SPEC-MATRIX | Limitations matrix declared | header Limitations table | real layout/channel/dtype rows, no `TODO`/`[TODO]`; `--bare` → `MANUAL` |
| SPEC-CORRECT | Semantics match the oracle; gold implements it independently | — | **MANUAL** (design/review judgment) |

### Domain: scaffold  (files + wiring)

| id | check | probe | PASS condition |
|----|-------|-------|----------------|
| SCF-1 | Public C API header | `src/cvcuda/include/cvcuda/Op<Name>.h` | present |
| SCF-2 | Public C++ header | `src/cvcuda/include/cvcuda/Op<Name>.hpp` | present |
| SCF-3 | C API impl | `src/cvcuda/Op<Name>.cpp` | present |
| SCF-4 | Private impl (`.cpp` or `.cu`) | `src/cvcuda/priv/Op<Name>.{cpp,cu}` | present (`.cpp`↔`.cu` is the author's choice → N-A either way) |
| SCF-5 | Private header | `src/cvcuda/priv/Op<Name>.hpp` | present |
| SCF-6 | C++ system test | `tests/cvcuda/system/TestOp<Name>.cpp` | present |
| SCF-7 | Python binding under `operators/` | `python/mod_cvcuda/operators/Op<Name>.cpp` | present (not in `python/mod_cvcuda/`) |
| SCF-8 | Python test | `tests/cvcuda/python/test_op<name>.py` | present |
| SCF-9 | Bench C++ + Python + config | `bench/cpp/ops/Bench<Name>.cpp`, `bench/python/ops/bench_<name>.py`, `bench/config/operators/<name>.json` | all present |
| SCF-10 | Lib + priv + test CMake wiring | `Op<Name>.cpp` in `src/cvcuda{,/priv}/CMakeLists.txt`; `TestOp<Name>.cpp` in `tests/cvcuda/system/CMakeLists.txt` | all present |
| SCF-11 | Python module wiring | `ExportOp<Name>` in `Main.cpp`; decl in `operators/Operators.hpp`; `operators/Op<Name>.cpp` in `python/mod_cvcuda/CMakeLists.txt` | all present |
| SCF-12 | Bench wiring | `ops/Bench<Name>.cpp` in `bench/cpp/CMakeLists.txt`; `ops/bench_<name>.py` in `bench/python/CMakeLists.txt`; `<name>` in `bench_params.json` | all present |
| SCF-13 | Docs rows | `operator_list.rst` row + `operators.rst` autofunction (fn + `_into`) + latest-relnote bullet | all present |
| SCF-14 | SPDX headers | the op's new source/test/bench/doc files | present (`.json` exempt) |

A green scaffold phase = the skeleton is complete and wired (and, unless `--bare`, the contract
is authored). The done-gate (below) stays red until it is implemented.

---

## Phase `done` — the deterministic final regression checklist

Runs the scaffold checks **plus** the groups below. Green ⇒ correct, complete, documented, in
the relnotes, and optimization-ready.

### Domain: implementation

| id | check | probe | PASS condition |
|----|-------|-------|----------------|
| IMP-1 | No stub markers | `TODO`, `t.fail`, `noop`, the template's `std::generate(goldVec…)` / placeholder `ASSERT_EQ(goldVec, testVec)` | none remain in the op's src/test |
| IMP-2 | Limitations table filled | header | no `TODO`/`[TODO]` rows (also SPEC-MATRIX) |
| IMP-3 | Multi-GPU safety | if priv has `cudaMalloc`, it uses `PerDeviceResource<>` | satisfied / N-A if no device alloc |
| NVTX-1 | NVTX marker present **and** registered | `CVCUDA_NVTX_RANGE("cvcuda<Name>…Submit")` in `Op<Name>.cpp` (the scaffold templates emit the submit / priv `operator()` / Python `NvtxTrace` markers — this guards they survived) **and** a `"<name>": ("cvcuda<Name>Submit", _<name>)` row in `tests/cvcuda/python/test_nvtx_markers.py` `OPERATORS` | both present → else GAP. Every public operator (`fn` + `_into`) must be in the registry or `test_all_operators_registered` fails once always-on NVTX is merged; the invocation helper must issue a minimal real call so the submit range actually fires |

### Domain: coverage + test-rigor (the teeth — stricter than `review_op`)

Declared-matrix mirroring with **hard GAPs**; the header Limitations table is the source of
truth (an integer-only op is never forced to add float). Uses a fixed `FMT_`/`TYPE_`/bench-name
→ canonical-dtype table. COV iterates **only declared dtypes/channels**.

| id | check | probe | PASS condition |
|----|-------|-------|----------------|
| COV-GOLD | Independent CPU gold reference | `Gold`/`Reference`/`Ref`/`naive`/`CPU` fn in `TestOp<Name>.cpp` | present (hard; independence stays MANUAL via SPEC-CORRECT) |
| COV-BITEXACT | Results compared bit-exact | `EXPECT_EQ`/`ASSERT_EQ` of gold vs result; any unexplained `EXPECT_NEAR`/`ASSERT_NEAR` | bit-exact; a NEAR site → **GAP** |
| COV-1 | Every declared **dtype** tested | `FMT_*`/`nvcv::TYPE_*`/type-list tokens in positive cases | each declared dtype detected → else GAP (strict, no MANUAL) |
| COV-CHAN | Every declared **channel count** tested | channel of the `FMT_*` tokens | each declared channel (1/3/4) detected → else GAP |
| COV-2 | Every declared dtype **benched** | `dtypes` in `bench/config/operators/<name>.json` (canonical map) | each declared dtype present → else GAP |
| BEN-DRV | Bench **drivers** implemented (not the NHWC-only stub) | `bench/cpp/ops/Bench<Op>.cpp` + `bench/python/ops/bench_<op>.py` carry no `TODO(make-op)` markers | both real → else GAP (the structural BEN-* checks pass on a skip-everything stub, so this guards the drivers actually exercise the declared layouts/containers; see `BenchFlip.cpp`) |
| BEN-GUIDE | RGB-benchmark-guideline tests pass | runs `bench/tests/test_bench_rgb_guidelines.py` + `test_run_bench_config_key.py` (static, no GPU) | green → else GAP. These gate the **CI baseline burn-in** *before* any benchmark runs, so a config that is unclassified in `bench/config/operator_categories.json`, mis-tiered (Cat-A scalar in basic), or that drifts the tripwire key/row counts fails the regen silently. `mkop.sh` auto-adds a category-`A` entry; **verify** A (general per-pixel op) vs B (inherently single-channel) vs C (intrinsic channel semantics), keep Cat-A basic RGB-only (single-channel + RGBA in advanced, R2/R3/R4), and bump the counts in `test_run_bench_config_key.py` for the op's added profiles |
| COV-3 | Each declared **container** tested + benched | TST-2/3 + BEN-14 promoted | Tensor (+ VarShape unless tensor-only) covered |
| COV-PARITY | Equivalent image-layout parity | `PlanarParityUtils`/`matches_interleaved`, or `Reformat`+planar+`EQ` in the test | native planar == reformat→interleaved-op→reformat, bit-exact |
| COV-5 | Image-layout completeness | header declares planar layouts + COV-PARITY + bench `NCHW` & `NCHW_FAKE` | satisfied; **only** escape = the public C header declares planar image layouts not applicable and gives a reason |
| COV-NEG | Complement rejected | `Op<Name>_Negative` asserting `NVCV_ERROR_INVALID_ARGUMENT` for an unsupported dtype + layout + channel + in/out mismatch | each represented |
| COV-MATRIX | Per-dtype × per-container exhaustiveness | — | **MANUAL** (tokens can't fully prove) |

### Domain: docs

Runs `review_op`'s **docs domain directly** (DOC-1 operator_list row, DOC-2 autofunction
fn+`_into`, DOC-5 pybind docstrings, DOC-7 SPDX; DOC-3/4/6 MANUAL) — these would otherwise be
missed, since `optimize_op` preflight only pulls test+bench. Plus:

| id | check | probe | PASS condition |
|----|-------|-------|----------------|
| DOC-REL | Operator in the latest release notes | bullet in `docs/sphinx/relnotes/vX.Y.Z-*.rst` (X.Y.Z = `CMakeLists.txt` `VERSION`) | present → else GAP |

### Domain: execution (`--run`, mandatory for final green)

| id | check | PASS condition |
|----|-------|----------------|
| EXEC-1 | C++ tests build + run + pass | `--run` builds `cvcuda_test_system` and runs `Op<Name>.*` / `TestOp<Name>` → all pass; no GPU → `GAP`/`MANUAL`, defer to CI |
| EXEC-2 | Python tests run + pass | `--run` runs `tests/cvcuda/python/test_op<name>.py` → pass; no GPU → `GAP`/`MANUAL` |

### Domain: readiness (reuses `optimize_op.py --phase preflight`)

| id | check | PASS condition |
|----|-------|----------------|
| RDY-1 | Optimization-readiness | `optimize_op.py --phase preflight` green — correctness + bench coverage (re-runs `review_op`) + baseline captured (`GAP [requires CI]` if absent) + profiling available |

### Domain: review (reuses `review_op.py`, all four domains)

`make_op.py --phase done` runs `review_op.run(P, all 4 domains)` so the full support / test /
bench / docs checklist (`SUP-*`/`TST-*`/`BEN-*`/`DOC-*`) is enforced. The COV/IMP/EXEC groups
above are make-op-specific *additions on top* (they tighten `review_op`'s MANUAL/REC items into
GAPs for a brand-new operator).

---

## Notes

- **Python scope (repo split):** C++ owns numerical bit-exact regression; Python tests stay
  API-surface (`make_op_tests`: layouts / `_into` / negative). COV-BITEXACT + the gold-reference
  mandate apply to the **C++** suite only.
- **Benchmark coverage (representative-sampled, deliberate):** tests are exhaustive over the
  declared matrix; image-operator benchmarks mandate the `layout` axis on every config plus native
  `NCHW` and the `NCHW_FAKE` reference
  + the basic-tier floor (BEN-14) + every declared dtype benched once (COV-2) + baselines
  (`GAP [requires CI]`). No full cross-product (it explodes the suite and breaks row parity).
- **Benchmark calibration (1–2 ms interleaved, mandatory):** because runtime scales with per-pixel
  bytes (1–16 across u8…float4), **a single shape cannot calibrate all dtypes** — use **per-dtype
  configs** with per-dtype batch sizes, as `flip.json` does. Within a dtype, **calibrate so the
  INTERLEAVED (`NHWC`) config runs 1–2 ms of nvbench GPU (kernel) time** on the target SKU — long
  enough that timing noise is low, short enough that the suite stays fast.
  **Apples-to-apples (mandatory): the interleaved, native-planar (`NCHW`) and fake-planar
  (`NCHW_FAKE`) configs of the same dtype must share the SAME input size** (`shape`), so the three
  layouts are directly comparable. Calibrate the *interleaved* config to 1–2 ms; the planar and
  fake-planar configs reuse that same shape and **may legitimately run longer** (planar moves
  1 element/thread; fake-planar adds reformat traffic) — that is expected and must not be "fixed" by
  shrinking their batch. (This is checked statically by `review_op`'s `BEN-SIZE` and is why the
  `BenchConfig.json` template shares one `shape` per dtype across the three layouts.)
  Calibrate by measuring: `./build-rel/bin/bench_<op> -a "shape=NxHxW" -a "InOutDataType=…" …`
  reports `Cold: <t> ms GPU`; adjust the batch `N` until the **NHWC** `t ∈ [1,2] ms`. Memory-bound
  element-wise ops can reuse `flip.json`'s batches directly (same traffic profile).
- **Benchmark drivers + run (`BEN-DRV`, run-validated):** the C++ and Python drivers must be real
  (no `TODO(make-op)` stub; `BEN-DRV` GAPs otherwise) and must be **exercised**: build
  `bench_<op>` + run it (all configs Pass, none unintentionally Skip), and run
  `python3 bench/run_bench.py --operator <op>` to confirm **noise < 5%** and **C++/Python parity**
  (BEN-11; run-dependent, so verified with `--run`/CI, not statically).
- **Baseline regen (CI, closes `BEN-7`/`RDY-1`):** once the configs are calibrated and run-green
  **locally**, trigger the named CI `baseline-regen` workflow to seed baselines on the **reference
  SKUs** and import its artifacts with
  `python3 bench/_internal/update_baseline.py --from <artifact-dir> --operator <op>` (never fabricate
  baselines).
  Local absolute timings are SKU-specific, so the gating baselines come from CI, not the dev box;
  after import, `make_op.py --phase done` shows `BEN-7`/`RDY-1` green.
  Current trigger, artifact, and import mechanics live in `bench/README.md` under "Regenerating
  baselines via CI"; CI selection semantics live in `ci/README.md`.
- **SonarQube "Sonar way" gate (hard CI gate, `allow_failure=false`):** the MR pipeline runs a
  SonarQube quality gate that fails on **any** new issue (`new_violations > 0`) and on unreviewed
  new security hotspots. Existing operators are grandfathered, so a brand-new op's code is judged
  against a zero-tolerance bar and the idiomatic-but-noncompliant boilerplate trips it. Write the
  op's `.cpp`/`.hpp`/test/bench **Sonar-clean from the start** (CUDA `.cu` files are excluded from
  analysis). The recurring rules and their fixes:
  - **S3608** default lambda capture — use explicit captures (`[handle, stream, in, out]`), never `[&]`.
  - **S5025** manual `new` — the operator's `Create` transfers ownership to the C handle; keep the
    `// NOSONAR` on that one line (NOSONAR is honored on NVIDIA's server) rather than fighting it.
  - **S5817** non-mutating method should be `const` — the public `operator()` is `const`.
  - **S3471/S3576** redundant `virtual` on an override — write `… handle() const noexcept override;`.
  - **S1659** multiple declarations per line — one identifier per statement (e.g. split
    `TensorWrapHandle input(in), output(out);`).
  - **S6012** redundant class-template args — rely on CTAD (`std::uniform_int_distribution dist(0, m);`).
  - **S5827** repeated type — `auto`. **S1301** two-case `switch` — use `if`. **S1481** unnecessary
    lambda capture of a `constexpr`/const-integral — drop it.
  - **S924** >1 `break`/`goto` per loop — GTest `ASSERT_NO_THROW`/`EXPECT_NO_THROW` expand to `goto`
    labels, so **don't wrap calls in `*_NO_THROW` inside a loop**; call the helper directly (a throw
    still fails the test). The `mkop.sh` C-API + C++ templates are already Sonar-clean; this is on
    the agent-written test/bench code. Verify locally by reading the `sonarqube` job log on the MR.
- **Samples:** not gated (consistent with `review_op`); optional/manual.
