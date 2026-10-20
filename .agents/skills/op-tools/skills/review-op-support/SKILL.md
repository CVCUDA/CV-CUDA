---
name: review-op-support
description: Review a CV-CUDA operator's input-type, layout, dtype, and channel support matrix. Use when asked which inputs an operator supports, whether required layouts are complete, or whether support is enforced consistently across C, C++, and Python APIs.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — support

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/review-op-support.md`.

- **Run:** `python3 tools/review_op.py <Operator> --domain support`.
- **Interpret per:** the support section (`SUP-*`) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`.
