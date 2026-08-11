---
name: refactor-op
description: Find and safely apply per-operator refactoring / redundancy-reduction opportunities in a CV-CUDA operator (near-duplicate Tensor/VarShape kernels, reinvented shared utilities, dead code). Use when asked to reduce code duplication, de-duplicate or unify an operator's kernels/bindings, remove dead code, or verify a refactor changed nothing observable. Produces a deterministic findings-first report; the apply path is gated by a strict bit-exact / feature-set / test-coverage parity check.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Refactor Op

Thin entry point. The catalog substance and the checks are shared repo artifacts: the deterministic
checker `tools/refactor_op.py` and the spec `.agents/guidance/REFACTOR_OP_GUIDELINES.md`. Keep this
skill thin; do not duplicate the catalog here. `refactor-op` is **complementary to SonarQube CPD**
(which owns generic, whole-corpus duplication) — it surfaces operator-scoped *semantic* redundancy
Sonar cannot express.

## Workflow

1. **assess** — `python3 tools/refactor_op.py <Operator>` (scope with `--domain impl|api|xcut`;
   `--format json` for machine output). Read-only and deterministic. Interpret each finding by its
   `RED-*` id in `.agents/guidance/REFACTOR_OP_GUIDELINES.md`.
2. **apply** (only if asked to fix) — apply the finding's named corrective action on the
   **implementation / binding only**, reusing existing shared utilities. **Never** edit the
   operator's tests or anything under `bench/`, and **never** change public API/ABI.
3. **verify** — when the MR scope includes a refactor, run
   `python3 tools/refactor_op.py <Operator> --phase verify --base <ref>` locally and require zero
   `GAP` (frozen test surface, frozen bench surface, API/ABI, feature-matrix and coverage-matrix
   parity), then build + run the frozen `Op<Operator>` tests (bit-exact, VER-6) and re-run assess
   (redundancy gone, VER-7). The verify report ends with a deterministic **refactor summary** (LOC
   delta + `RED-*` resolved before→after) — report it as the outcome. If the MR also changes
   performance-sensitive implementation, provide benchmark evidence through the optimization
   workflow.

Inviolable: refactoring changes nothing observable — bit-exact output, identical feature set, frozen
tests and benchmarks, unchanged API/ABI; never weaken a test or fabricate a baseline to pass. There
is no standing `refactor-parity` CI job; invoke the local verify flow when the MR scope calls for it.

## Prompt Handling

Requests like "reduce duplication in Flip", "/refactor-op BrightnessContrast", "unify the Tensor and
VarShape kernels for Resize", "is there dead code in OpStack?", or "verify my refactor is bit-exact"
trigger this skill. Use `--domain` to scope; use `--phase verify` to gate an applied refactor.
