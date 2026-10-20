[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — bench coverage

Procedure for the `/review-op-bench-coverage` dispatcher
(`.agents/skills/op-tools/skills/review-op-bench-coverage/SKILL.md`); the shared operating contract is
`conventions.md`.

Run `python3 tools/review_op.py <Operator> --domain bench` (add `--run`
for the GPU-run-dependent checks) and interpret per the **bench** section (BEN-*) of
`.agents/guidance/REVIEW_OP_GUIDELINES.md`: structural rules + the hard basic-tier minimum floor
(BEN-14) + advisory coverage statistics and `RECOMMENDATION`s. Resolve `MANUAL` items at the cited locations. When fixing
(`--fix`), add the layout axis and required native/reference layout configs, then recompute row counts - a missing-SKU
**baseline gap is reported "requires CI" and never fabricated** — and re-run to confirm.
