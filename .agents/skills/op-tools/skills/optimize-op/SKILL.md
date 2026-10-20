---
name: optimize-op
description: Drive a single-operator optimization campaign per .agents/guidance/OPTIMIZATION_GUIDELINES.md. Use when asked to optimize an operator, run or finish a campaign, update its MR summary, or verify correctness, benchmarks, generalization review, lead disposition, memory, and API/ABI gates.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/optimize-op.md`.

- **Run:** `python3 tools/optimize_op.py <Op> --phase preflight` before optimizing;
  `python3 tools/optimize_op.py <Op> --phase evidence --base <ref> --results <description>` as
  the campaign done-gate.
- **Interpret per:** `.agents/guidance/OPTIMIZATION_GUIDELINES.md` (the source of truth for the
  campaign loop and the `PRE-*`/`ODO-*` gate index); the MR summary block is in
  `.agents/skills/op-tools/reference/optimize-op-summary.md`.

**Delegation rule.** At top level, delegate the full procedure to the `optimize-op-worker`
sub-agent (`.agents/agents/optimize-op-worker.md`) and read back its verdict; when you are already
inside a delegated sub-agent run, or the task is a single checker invocation, run inline. The
worker is an optional accelerator — `.agents/agents/` is internal-only and the release trim
removes it, so the reference doc above remains the complete procedure.
