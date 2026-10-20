---
name: optimize-op-preflight
description: Check whether a CV-CUDA operator is READY to optimize (correctness + bench coverage + committed baselines + profiling) per .agents/guidance/OPTIMIZATION_GUIDELINES.md. Use before starting an optimization campaign to confirm the readiness gate is clean.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — preflight

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/optimize-op-preflight.md`.

- **Run:** `python3 tools/optimize_op.py <Op> --phase preflight`.
- **Interpret per:** the readiness items (`PRE-*`) against
  `.agents/guidance/OPTIMIZATION_GUIDELINES.md`. Any `GAP` means the operator is not ready.
