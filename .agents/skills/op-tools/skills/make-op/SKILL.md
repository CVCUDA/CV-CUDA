---
name: make-op
description: Add a new CV-CUDA operator end-to-end per .agents/guidance/MAKE_OP_GUIDELINES.md, with a deterministically-enforced definition-of-done. Use when asked to create/add a new operator, scaffold one, or verify that a new operator is complete (approved spec, wired scaffold, bit-exact regression tests across the declared support matrix, required layout parity, complement negatives, docs + relnote, benched dtypes, tests that run and pass, optimization-ready).
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/make-op.md`.

- **Spec first:** propose semantics + a cited reference oracle + the support matrix and get user
  approval, then scaffold with `tools/mkop/mkop.sh <Name>`.
- **Gate:** `python3 tools/make_op.py <Name> --phase scaffold`, implement, then
  `python3 tools/make_op.py <Name> --phase done --run`.
- **Interpret per:** `.agents/guidance/MAKE_OP_GUIDELINES.md`.

**Delegation rule.** There is deliberately no `make-op-worker` sub-agent: spec approval is
human-in-loop and the scaffold is deterministic, so this procedure runs inline. The stage
sub-skills (`/make-op-scaffold`, `/make-op-verify`) stay single-checker.
