---
name: review-op-test-coverage
description: Review a CV-CUDA operator's test coverage, including C++ correctness, required cross-layout parity, correctness rigor, and the Python API surface. Use when asked whether an operator is adequately tested or to find and fill test gaps.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — test coverage

Thin entry point. Run `python3 tools/review_op.py <Operator> --domain test` and interpret
per the **test** section (TST-*) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`: independent gold,
required cross-layout parity, axis-coverage mirror, negative tests, tolerance discipline (bit-exact by
default), and the Python API surface. Resolve `MANUAL` items at the cited locations. When
fixing (`--fix`), author the missing positive, negative, or layout-parity cases - **never introduce
or widen a tolerance silently** (emit rationale + needs-human) — and re-run to confirm.
Findings-first.
