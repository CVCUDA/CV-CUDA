[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — support

Procedure for the `/review-op-support` dispatcher (`.agents/skills/op-tools/skills/review-op-support/SKILL.md`);
the shared operating contract is `conventions.md`.

Run `python3 tools/review_op.py <Operator> --domain support` and interpret
per the **support** section (SUP-*) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`: the container × layout ×
dtype × channel matrix, runtime enforcement, cross-surface (C-API/C++/Python) consistency,
and layout completeness. Resolve `MANUAL` items at the cited locations. When fixing (`--fix`),
apply the per-item corrective action and keep missing layout capability as explicit author work.
Re-run to confirm.
