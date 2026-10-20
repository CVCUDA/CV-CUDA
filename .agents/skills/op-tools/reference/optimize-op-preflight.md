[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — preflight

Procedure for the `/optimize-op-preflight` dispatcher
(`.agents/skills/op-tools/skills/optimize-op-preflight/SKILL.md`); the shared operating contract is
`conventions.md`.

Run `python3 tools/optimize_op.py <Op> --phase preflight` and interpret per
the tool's readiness (PRE-*) findings against `.agents/guidance/OPTIMIZATION_GUIDELINES.md`: correctness/regression coverage and
benchmark coverage (reused from `review_op.py` test+bench), committed baselines, and profiling
tools. Any `GAP` means the operator is not ready — add the missing coverage **first**, in a
separate non-`perf:` commit, before starting the optimization. Resolve `MANUAL` items at the
cited locations.
