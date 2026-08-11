---
name: make-op-verify
description: Verify a new CV-CUDA operator against the deterministic final regression checklist (the /make-op done-gate). Use when asked whether a new operator is complete/done, or to gate it before merge - gold reference, bit-exact coverage across the declared support matrix, required layout parity, complement negatives, docs + relnote, benched dtypes, tests that run and pass, and optimization-readiness.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op — verify

Thin entry point. Run `python3 tools/make_op.py <Operator> --phase done --run` and interpret per
`.agents/guidance/MAKE_OP_GUIDELINES.md`. This is the strict-regression done-gate: it reuses `review-op` (all four
domains) and `optimize-op` preflight, then adds the make-op teeth — independent CPU gold reference
(`COV-GOLD`), **bit-exact** coverage across the declared support matrix
(`COV-1`/`COV-CHAN`/`COV-3`/`COV-BITEXACT`), required equivalent-layout parity
(`COV-PARITY`/`COV-5`),
complement negatives (`COV-NEG`), every declared dtype benched (`COV-2`), the always-on NVTX
marker present and registered in the runtime marker test (`NVTX-1`), the operator in the
latest relnote (`DOC-REL`), and tests that compile/run/pass (`EXEC-*`). Resolve every `GAP` with
its named fix and re-run; resolve each `MANUAL` at the cited location. Inviolable: never fabricate
baselines (`GAP [requires CI]` → CI regen); never loosen tolerances (bit-exact default; any
`EXPECT_NEAR` in the C++ test is a GAP); layout-support gaps are author work.
Findings-first; completion = a re-run shows zero `GAP` and zero unresolved `MANUAL`.
