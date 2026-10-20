---
name: refactor-op
description: Find and safely apply per-operator refactoring / redundancy-reduction opportunities in a CV-CUDA operator (near-duplicate Tensor/VarShape kernels, reinvented shared utilities, dead code). Use when asked to reduce code duplication, de-duplicate or unify an operator's kernels/bindings, remove dead code, or verify a refactor changed nothing observable. Produces a deterministic findings-first report; the apply path is gated by a strict bit-exact / feature-set / test-coverage parity check.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Refactor Op

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/refactor-op.md`.

- **Run:** `python3 tools/refactor_op.py <Operator>` to assess (`--domain impl|api|xcut`;
  `--format json`); `python3 tools/refactor_op.py <Operator> --phase verify --base <ref>` to gate
  an applied refactor.
- **Interpret per:** `.agents/guidance/REFACTOR_OP_GUIDELINES.md` (`RED-*` findings, `VER-*` parity
  items). When verify reports `NEEDS-LOCAL-PROOF` and no build is available, use
  `/device-code-proof`.

**Delegation rule.** At top level, delegate the full procedure to the `refactor-op-worker`
sub-agent (`.agents/agents/refactor-op-worker.md`) and read back its verdict; when you are already
inside a delegated sub-agent run, or the task is a single checker invocation, run inline. The
worker is an optional accelerator — `.agents/agents/` is internal-only and the release trim
removes it, so the reference doc above remains the complete procedure.
