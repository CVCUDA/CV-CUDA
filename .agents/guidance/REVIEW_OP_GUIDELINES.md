[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA Operator Review Guidelines (`/review-op`)

The single source of truth for the per-operator review harness. It is **implemented by**
`tools/review_op.py` (the deterministic checker) and **cited by** the thin skill
wrappers (`.agents/skills/review-op*/SKILL.md`, symlinked into `.claude/skills` for
Claude Code). The checklist substance lives here once; every tool's skill delegates
to it — no duplication.

This document does not restate house policy; it *cites* the authoritative docs and turns
their requirements into concrete, machine-checkable items:
`AGENTS.md`, `REVIEW_PR_GUIDELINES.md`, `docs/sphinx/advanced/make_operator.rst`,
and `OPTIMIZATION_GUIDELINES.md`.

## How to use

```bash
python3 tools/review_op.py <Operator> [--domain support|test|bench|docs|all] \
                           [--format md|json] [--out report.json] [--run] [--fix]
```
- Each checklist item is reported with a **status**, literal **evidence**, and its **id**.
- **Completion** is defined as: a re-run of the checker shows **zero `GAP` and zero
  unresolved `MANUAL`** for the selected domains. That re-run is the deterministic
  evidence of completion.

### Status vocabulary

| status | meaning | affects exit code |
|--------|---------|-------------------|
| `PASS` | check satisfied | no |
| `GAP` | check failed; an actionable deficiency | **yes (non-zero)** |
| `N-A` | not applicable to this operator (e.g. no VarShape surface) | no |
| `MANUAL` | needs human reading; checker points at the location | no (but surfaced) |
| `RECOMMENDATION` | advisory follow-up (bench coverage analysis); human may override | no |

The checker is **deterministic & idempotent**: no network, no clocks, no randomness — the
same tree yields a byte-identical report.

### `--fix`
The checker itself stays read-only; `--fix` directs the wrapper/agent to apply the
per-item corrective action below where mechanical, then **re-run the checker to prove
the item is now `PASS`**. Inviolable fix rules:
- **Never fabricate baselines** (a missing baseline is `GAP "requires CI"`; use the baseline
  workflow in `bench/README.md`).
- **Never silently introduce or widen a test tolerance** — bit-exact is the default; a
  needed `EXPECT_NEAR` emits rationale + a `needs-human` flag (see `OPTIMIZATION_GUIDELINES.md`).
- **A missing layout capability is author work** — `--fix` reports it and does not
  reimplement operator functionality.

### Operator name resolution
`<Operator>` is PascalCase (e.g. `CenterCrop`). The checker derives:
- `op` — lowercase bench/config stem (e.g. `centercrop`), the key in `bench/config/bench_params.json`.
- `Op` — C++ class / test-suite stem (e.g. `CenterCrop`), the `cvcuda<Op>Submit` infix.
- `pyname` — the Python function name (e.g. `cvcuda.hq_resize` for `HqResize`). It can
  differ from `op`; resolve via the `m.def("<pyname>", …)` in the Python binding /
  `operator_list.rst` link, MANUAL fallback if ambiguous.

---

## Domain: support  (matrix + enforcement + consistency)

Owns the support matrix `container × layout × dtype × channels` and its enforcement and
cross-surface consistency. Does **not** verify tests (that's the test domain).

| id | check | probe | PASS condition | fix |
|----|-------|-------|----------------|-----|
| SUP-1 | Tensor container declared | primary data input of `cvcuda<Op>Submit` is `NVCVTensorHandle` or `NVCVTensorBatchHandle` in `src/cvcuda/include/cvcuda/Op<Op>.h` | present | report (rare) |
| SUP-2 | VarShape container declared | primary data input of a Submit entry point is `NVCVImageBatchHandle` (including legacy generic `cvcuda<Op>Submit` APIs), or an established `cvcuda<Op>VarShapeSubmit` is present | present → PASS; absent → **N-A iff deterministic tool data or operator-local evidence marks the op tensor-only, else GAP** | report |
| SUP-3 | C++ overloads match C-API containers | `Op<Op>.hpp` operator() overloads | same container set as SUP-1/2 | report |
| SUP-4 | Python binds allocating + `_into` per container | `m.def("<pyname>"…)` / `"<pyname>_into"` in `python/mod_cvcuda/operators/Op<Op>.cpp` | both variants per supported container | add missing binding (mechanical) |
| SUP-5 | Limitations tables present | header Doxygen `Data Layout:`/`Channels:`/`Data Type` (Input + Output) | all three present, in & out | add table row (mechanical) |
| SUP-6 | Declared layouts parsed | parse `Data Layout: [..]` | layout set extracted | — |
| SUP-7 | Declared dtypes parsed | parse `Data Type \| Allowed` rows | dtype set extracted | — |
| SUP-8 | Declared channels parsed | parse `Channels: [..]` | channel set extracted | — |
| SUP-9 | Enforcement present | guards in `src/cvcuda/priv/**` reject unsupported layout/dtype/channel with `ERROR_INVALID_ARGUMENT`; **declared == enforced** | guards found & consistent | report (often MANUAL — needs a code guard) |
| SUP-10 | Default image-layout policy | header layout set plus optional `Planar image layouts: Not applicable` and required `Reason` beside the Limitations table | `NCHW`/`CHW` declared; or a well-formed operator-local inapplicability declaration; never both | author |
| SUP-11 | Cross-surface consistency | diff parsed sets across header ↔ `.hpp` ↔ python | agree | report |

SUP-6/7/8 use a **full structured parse with MANUAL fallback** (emit MANUAL + the table
location if a Doxygen table doesn't parse cleanly).

SUP-1/2 classify the **primary data input**, meaning the first Tensor/TensorBatch/ImageBatch
handle after the operator and stream parameters. Tensor outputs and auxiliary tensors do not
declare Tensor-input support for mixed-container legacy APIs.

Source: `make_operator.rst` (Limitations table, runtime validation), `REVIEW_PR_GUIDELINES.md`
(input-type support documented & tested), and `OPTIMIZATION_GUIDELINES.md`.

---

## Domain: test  (correctness coverage + rigor)

Consumes the support matrix. **C++ owns numerical correctness; Python owns the API
surface** (`make_operator.rst`). Every declared layout needs correctness coverage; equivalent
interleaved and planar image layouts additionally require bit-exact parity.

| id | check | probe | PASS condition | fix |
|----|-------|-------|----------------|-----|
| TST-1 | Independent CPU reference | reference fn in `tests/cvcuda/system/TestOp<Op>.cpp` | present (independence = MANUAL) | report |
| TST-2 | `tensor_correct_output` | grep | present | author (mirror existing) |
| TST-3 | `varshape_correct_output` | grep | present / N-A if tensor-only | author |
| TST-4 | `NVCV_TEST_SUITE_P` + parse cases | grep/parse | present (parse → MANUAL fallback) | — |
| TST-5 | Matrix-mirror (axis-coverage) | each supported dtype/channel/layout/container/**mode** value appears in ≥1 positive case | all axis values covered; GAP lists uncovered | author missing case |
| TST-6 | Negative suite | `Op<Op>_Negative` asserting `NVCV_ERROR_INVALID_ARGUMENT` for unsupported matrix combinations | present | author |
| TST-7 | Equivalent-layout parity | `PlanarParityUtils` + `Op<Op>Planar.*_matches_interleaved` | present for image-layout operators; N-A only with SUP-10 declaration | author |
| TST-8 | Tolerance discipline | `EXPECT_EQ` vs `EXPECT_NEAR(tol)` | **bit-exact `EXPECT_EQ` default**; every `EXPECT_NEAR` w/o adjacent rationale → MANUAL; `EXPECT_NEAR`-on-int → stronger flag | never auto-loosen; emit `needs-human` |
| TST-9 | Reference independence + edge adequacy | — | MANUAL with pointers | report |
| TST-10 | Deterministic inputs | grep for seeded/fixed fill vs unseeded RNG | seeded; unseeded RNG → flag | report |
| TST-11 | Python Tensor NHWC + HWC | `test_op<op>.py` | exercised | author |
| TST-12 | Python VarShape | grep | exercised / N-A if tensor-only | author |
| TST-13 | Python allocating + `_into` | grep | both called | author |
| TST-14 | Python negative | `pytest.raises` | present | author |

**Mode axes** for TST-5 are parsed from the op's C-API enums (e.g. `NVCVInterpolationType`,
`NVCVBorderType`), MANUAL fallback if unparseable.

Source: `make_operator.rst` (test structure, C++/Python split), `OPTIMIZATION_GUIDELINES.md`
(bit-exact default, no silent tolerance bumps).

---

## Domain: bench  (structural + baselines + the basic-tier floor + advisory coverage)

### Hard layer (PASS/GAP)

| id | check | probe | PASS condition | fix |
|----|-------|-------|----------------|-----|
| BEN-1 | C++ bench present + registered | `bench/cpp/ops/Bench<Op>.cpp` + `bench/cpp/CMakeLists.txt` | both | scaffold |
| BEN-2 | Python bench present + registered | `bench/python/ops/bench_<op>.py` + `bench/python/CMakeLists.txt` | both | scaffold |
| BEN-3 | Manifest entry | `<op>` in `bench/config/bench_params.json` (config+cpp+python) | present | add |
| BEN-4 | Config + tiers | `bench/config/operators/<op>.json`; every config has a valid `tier` | present | — |
| BEN-5 | truthful `layout` axis on every layout-bearing config | every config `string_axes.layout` present, unless deterministic review data marks the benchmark layout axis N-A | all applicable configs / N-A | add the real layout axis; never add a dummy axis |
| BEN-6 | Exact planar comparison coverage (if planar) | native `NCHW`/`CHW` configs; every `NCHW_FAKE`/`CHW_FAKE` expanded case is advanced-tier, Tensor-only, and has exactly one same-tier native case with every non-layout axis identical | exact per-signature pairs / N-A when no Tensor-image FakePlanar surface applies | add the advanced native/FakePlanar pair together |
| BEN-7 | Baseline completeness | every case-key × every `sku_map.json` SKU has metrics | complete; missing → **GAP "requires CI"** | baseline workflow in `bench/README.md` (never fabricate) |
| BEN-8 | Config expansion consistency | recompute `max(len(dtypes),1)·Π len(axis)` for the operator config and compare against generated benchmark rows/baselines where available | match | fix the operator config or baselines |
| BEN-9 | Internal baseline validation | `python3 bench/_internal/validate_baselines.py --operator <op>` | **MANUAL** — run the probe command; PASS = exit 0 | fix case keys |
| BEN-10 | C++/Python config parity | both drivers handle the configured axes | structural PASS / full = `--run` | — |
| BEN-11 | Noise/parity quality + currency | `run_bench.py` | **"requires GPU run"** (MANUAL; `--run` checks) | — |

### Basic-tier minimum floor (HARD → GAP)

| id | check | PASS condition |
|----|-------|----------------|
| BEN-14 | `basic` tier covers **≥ RGB × Tensor(if supported) × VarShape(if supported) × interleaved `NHWC` × planar `NCHW`(if planar)** | each *applicable* element is present in a `basic` config; applicability comes from the support matrix, operator-local layout policy, and deterministic review data. Any applicable element missing → GAP |

BEN-14 is an axis-coverage floor, not a demand for the full Cartesian product.
An unsupported container/layout combination must remain absent rather than being
added as a skipped or misleading row (for example, a VarShape API that rejects
planar image formats).

(Bench expresses container via the `inputKind` axis = `Tensor`/`VarShape` and, for an
operator with a distinct batch-of-tensors API, `TensorBatch`. The universal floor applies
to Tensor and VarShape; reviewed TensorBatch profiles may remain advanced. Layout uses
`NHWC`/`NCHW`/`NCHW_FAKE`. "RGB" = the primary 3-channel dtype, e.g. `uchar3`/`RGB8`.)

### Advisory coverage analysis (`RECOMMENDATION`, report-only)

| id | check |
|----|-------|
| BEN-13 | Coverage statistics vs the support matrix, built from the per-tier (`basic`/`advanced`) benched matrix of `<op>.json`: per-axis (layout/inputKind/dtypes(benched)/channels → basic/advanced/none) |
| BEN-15 | Tiering check vs curated operator-local or deterministic tool expectations: listed combos in `basic`; remainder in `advanced` or flagged |
| BEN-16 | Emit advisory follow-ups (`RECOMMENDATION`); report-only, no persistence (human override wins) |
| BEN-SIZE | Apples-to-apples sizes: same-dtype `NHWC`/`NCHW`/`NCHW_FAKE` configs share one input `shape` (calibrate the interleaved config to 1-2 ms); mismatch → `RECOMMENDATION` |

Source: `bench/README.md` and `OPTIMIZATION_GUIDELINES.md` (layout axis, NCHW_FAKE,
baseline regen).

---

## Domain: docs  (published artifacts + semantic consistency + SPDX)

Owns docs artifacts + Limitations-vs-code semantics + docstrings + SPDX. Does **not** re-do
container/overload parity (support SUP-3/4).

| id | check | probe | PASS condition | fix |
|----|-------|-------|----------------|-----|
| DOC-1 | `operator_list.rst` row | `:py:func:`<pyname>`` row in `docs/sphinx/operator_list.rst` | present | add row |
| DOC-2 | Python autofunction (fn + `_into`) | `.. cvcuda-autofunction:: cvcuda.<pyname>` and `…_into` in `docs/sphinx/modules/python/operators.rst` | both | add directive |
| DOC-3 | C++ API-reference entry | operator listed in the C++ API docs | present (MANUAL fallback) | report |
| DOC-4 | Limitations table consistent with code | **diff** declared (SUP-6/7/8) vs enforced (SUP-9) vs tested (TST-5) | three agree; divergence → GAP naming the mismatch; nuance → MANUAL | correct table iff diff unambiguous |
| DOC-5 | Python docstrings (op + `_into`) | grep binding | present (quality MANUAL) | report |
| DOC-6 | Doxygen param docs explain "why" | — | MANUAL | report |
| DOC-7 | SPDX 2026 headers | the op's new/changed src/test/bench/doc files | present | add header (mechanical) |

Source: `make_operator.rst` (docs steps), `AGENTS.md` (SPDX), `REVIEW_PR_GUIDELINES.md`
(docs consistency).

---

## Operator-specific review data

Keep this guidance generic. Do not add operator names, reviewed exception lists,
campaign status, or operator-specific benchmark expectations here.

Operator-specific facts belong in deterministic tool data, the active campaign
backlog/status, or operator-local source/test/bench artifacts. This keeps
unrelated operator work from conflicting in the shared guidance file.
