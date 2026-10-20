---
name: review-op-docs-api
description: Review a CV-CUDA operator's DOCS & API artifacts — operator_list row, Python autofunction (fn + _into), Limitations-table-vs-code consistency, docstrings, and SPDX headers. Use when asked whether an operator's docs/API surface is complete and consistent.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — docs & API

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/review-op-docs-api.md`.

- **Run:** `python3 tools/review_op.py <Operator> --domain docs`.
- **Interpret per:** the docs section (`DOC-*`) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`.
