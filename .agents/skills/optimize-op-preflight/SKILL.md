---
name: optimize-op-preflight
description: Check whether a CV-CUDA operator is READY to optimize (correctness + bench coverage + captured baseline + profiling) per .agents/guidance/OPTIMIZATION_GUIDELINES.md. Use before starting an optimization campaign to confirm the readiness gate is clean.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — preflight

Thin entry point. Run `python3 tools/optimize_op.py <Op> --phase preflight` and interpret per
the tool's readiness (PRE-*) findings against `.agents/guidance/OPTIMIZATION_GUIDELINES.md`: correctness/regression coverage and
benchmark coverage (reused from `review_op.py` test+bench), a captured baseline, and profiling
tools. Any `GAP` means the operator is not ready — add the missing coverage **first**, in a
separate non-`perf:` commit, before starting the optimization. Resolve `MANUAL` items at the
cited locations. Findings-first.
