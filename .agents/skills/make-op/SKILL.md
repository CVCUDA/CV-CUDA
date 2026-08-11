---
name: make-op
description: Add a new CV-CUDA operator end-to-end per .agents/guidance/MAKE_OP_GUIDELINES.md, with a deterministically-enforced definition-of-done. Use when asked to create/add a new operator, scaffold one, or verify that a new operator is complete (approved spec, wired scaffold, bit-exact regression tests across the declared support matrix, required layout parity, complement negatives, docs + relnote, benched dtypes, tests that run and pass, optimization-ready).
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op

Thin entry point. The workflow + rules live in `.agents/guidance/MAKE_OP_GUIDELINES.md` and the
narrative how-to in `docs/sphinx/advanced/make_operator.rst`; the definition-of-done is gated by the
deterministic checker `tools/make_op.py`. Keep this skill thin; do not duplicate the guidelines.

## Workflow

1. **Spec & approval (Part 0)** — propose the operator's semantics + a **cited reference oracle**
   (e.g. mimic TorchVision/OpenCV, or an explicit formula) + API + the support matrix
   (dtype × channel × layout × container); **get user approval** (or an explicit oracle waiver for
   a novel custom op), then record the contract in `Op<Name>.h` (`@brief` + `Reference:` +
   Limitations matrix).
2. **Scaffold** — `tools/mkop/mkop.sh <Name>` → the wired skeleton; gate with
   `python3 tools/make_op.py <Name> --phase scaffold` (SCF-green).
3. **Implement** — kernel + an independent CPU gold reference + tests/bench per the `COV-*` rules:
   bit-exact for every declared variant, required equivalent-layout parity, complement negatives,
   and every declared dtype benched. Image operators support interleaved and planar layouts by
   default; record an operator-local reason when image layouts do not apply. The scaffold emits the
   always-on NVTX markers (C-API submit / priv `operator()` / Python `NvtxTrace`); keep them and add
   the op to the `OPERATORS` registry in `tests/cvcuda/python/test_nvtx_markers.py` (`NVTX-1`).
4. **Done gate** — `python3 tools/make_op.py <Name> --phase done --run`: composes `/review-op`
   (all domains) + `/optimize-op` preflight and adds COV/EXEC/DOC-REL. Must be green; loop on the
   verdict — for each `GAP` apply its named fix and re-run.
   **Inviolable:** never fabricate baselines (missing → `GAP [requires CI]` → CI regen); bit-exact
   is the default (an `EXPECT_NEAR` on an interleaved path is a GAP); layout-support gaps are
   author work.
5. **Calibrate + seed baselines** — calibrate each bench config to **1–2 ms nvbench GPU time**
   (per-dtype; measure with `bench_<op>` and adjust the batch), exercise with
   `bench/run_bench.py --operator <op> --lang both` (noise < 5% + C++/Python parity), then trigger
   the named CI `baseline-regen` workflow to seed baselines on the reference SKUs and import its
   artifacts with `bench/_internal/update_baseline.py --from <artifact-dir> --operator <op>`. This closes
   `BEN-7`/`RDY-1` (never fabricate baselines; absolute timings are SKU-specific, so baselines come
   from CI). Follow `bench/README.md` for the workflow and `ci/README.md` for CI selection.
6. **Hand off** to `/optimize-op <Name>` for the performance campaign.

For a wired skeleton with the implementation delegated to a human/other AI, use `make-op-scaffold`
(optionally `--bare`); `make-op-verify` is the done-gate. Findings-first.
