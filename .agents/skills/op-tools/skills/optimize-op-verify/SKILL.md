---
name: optimize-op-verify
description: Verify a CV-CUDA optimization campaign and MR summary per .agents/guidance/OPTIMIZATION_GUIDELINES.md. Check reference-SKU statistics, checklist evidence, baseline validation, generalization review, lead disposition, memory, and API/ABI stability before review.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — verify

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/optimize-op-verify.md`.

- **Run:** `python3 tools/optimize_op.py <Op> --phase evidence --base <ref> --results <description>`.
- **Interpret per:** each finding's `ref` section (the `ODO-*` gate index lives in
  `.agents/guidance/OPTIMIZATION_GUIDELINES.md`; the summary contract in
  `.agents/skills/op-tools/reference/optimize-op-summary.md`).
