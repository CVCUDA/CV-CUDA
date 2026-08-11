---
name: review-op-bench-coverage
description: Review a CV-CUDA operator's BENCHMARK coverage — drivers, layout axis, baselines, the basic-tier floor, row counts, and coverage statistics. Use when asked whether an operator's benchmarks/baselines are complete or to find/fill bench gaps.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — bench coverage

Thin entry point. Run `python3 tools/review_op.py <Operator> --domain bench` (add `--run`
for the GPU-run-dependent checks) and interpret per the **bench** section (BEN-*) of
`.agents/guidance/REVIEW_OP_GUIDELINES.md`: structural rules + the hard basic-tier minimum floor
(BEN-14) + advisory coverage statistics and `RECOMMENDATION`s. Resolve `MANUAL` items at the cited locations. When fixing
(`--fix`), add the layout axis and required native/reference layout configs, then recompute row counts - a missing-SKU
**baseline gap is reported "requires CI" and never fabricated** — and re-run to confirm.
Findings-first.
