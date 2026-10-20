---
name: review-op-test-coverage
description: Review a CV-CUDA operator's test coverage, including C++ correctness, required cross-layout parity, correctness rigor, and the Python API surface. Use when asked whether an operator is adequately tested or to find and fill test gaps.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — test coverage

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/review-op-test-coverage.md`.

- **Run:** `python3 tools/review_op.py <Operator> --domain test`.
- **Interpret per:** the test section (`TST-*`) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`.
