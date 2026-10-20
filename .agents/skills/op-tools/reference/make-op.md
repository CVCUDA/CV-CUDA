[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op

Procedure for the `/make-op` dispatcher (`.agents/skills/op-tools/skills/make-op/SKILL.md`); the shared operating
contract is `conventions.md`. The workflow + rules live in
`.agents/guidance/MAKE_OP_GUIDELINES.md` and the narrative how-to in
`docs/sphinx/advanced/make_operator.rst`; the definition-of-done is gated by the deterministic
checker `tools/make_op.py`. Do not duplicate the guidelines here.

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
   The inviolables in `conventions.md` apply in full.
5. **Calibrate + seed baselines** — calibrate each bench config to **1–2 ms nvbench GPU time**
   (per-dtype; measure with `bench_<op>` and adjust the batch), exercise with
   `bench/run_bench.py --operator <op> --lang both` (noise < 5% + C++/Python parity), then seed
   baselines from reference-SKU runs and import their artifacts with
   the guarded workflow in `bench/README.md` under "Import, Validate, and Recheck". This
   closes `BEN-7`/`RDY-1`; never fabricate baselines because absolute timings are SKU-specific.
6. **Hand off** to `/optimize-op <Name>` for the performance campaign.

For a wired skeleton with the implementation delegated to a human/other AI, use `make-op-scaffold`
(optionally `--bare`); `make-op-verify` is the done-gate.
