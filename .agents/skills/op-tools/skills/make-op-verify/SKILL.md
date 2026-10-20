---
name: make-op-verify
description: Verify a new CV-CUDA operator against the deterministic final regression checklist (the /make-op done-gate). Use when asked whether a new operator is complete/done, or to gate it before merge - gold reference, bit-exact coverage across the declared support matrix, required layout parity, complement negatives, docs + relnote, benched dtypes, tests that run and pass, and optimization-readiness.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op — verify

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/make-op-verify.md`.

- **Run:** `python3 tools/make_op.py <Operator> --phase done --run`.
- **Interpret per:** `.agents/guidance/MAKE_OP_GUIDELINES.md` (composes `/review-op`, all four
  domains, and `/optimize-op` preflight, then adds the `COV-*`/`NVTX-1`/`DOC-REL`/`EXEC-*` teeth).
  This gate is stricter than `conventions.md` on tolerances — preserve verbatim: never loosen
  tolerances (bit-exact default; any `EXPECT_NEAR` in the C++ test is a GAP).
