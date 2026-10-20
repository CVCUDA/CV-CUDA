---
name: review-op
description: Review a CV-CUDA operator end-to-end (support / test / bench / docs coverage). Use when the user asks to review an operator, audit its input-type/layout/dtype support, test coverage, benchmark coverage, or docs/API, or to find & fix per-operator coverage gaps. Produces a deterministic findings-first report.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/review-op.md`.

- **Run:** `python3 tools/review_op.py <Operator>` (scope with `--domain support|test|bench|docs`;
  `--format json` for machine output; `--run` for the GPU-run-dependent bench checks).
- **Interpret per:** `.agents/guidance/REVIEW_OP_GUIDELINES.md` (item ids `SUP-*`/`TST-*`/`BEN-*`/`DOC-*`).

**Delegation rule.** At top level, delegate the full procedure to the `review-op-worker`
sub-agent (`.agents/agents/review-op-worker.md`) and read back its verdict; when you are already
inside a delegated sub-agent run, or the task is a single checker invocation, run inline. The
worker is an optional accelerator — `.agents/agents/` is internal-only and the release trim
removes it, so the reference doc above remains the complete procedure.
