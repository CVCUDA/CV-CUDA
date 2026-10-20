---
name: review-op-bench-coverage
description: Review a CV-CUDA operator's BENCHMARK coverage — drivers, layout axis, baselines, the basic-tier floor, row counts, and coverage statistics. Use when asked whether an operator's benchmarks/baselines are complete or to find/fill bench gaps.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — bench coverage

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/review-op-bench-coverage.md`.

- **Run:** `python3 tools/review_op.py <Operator> --domain bench` (add `--run` for the
  GPU-run-dependent checks).
- **Interpret per:** the bench section (`BEN-*`) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`.
