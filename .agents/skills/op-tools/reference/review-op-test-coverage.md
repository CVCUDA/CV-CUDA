[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — test coverage

Procedure for the `/review-op-test-coverage` dispatcher
(`.agents/skills/op-tools/skills/review-op-test-coverage/SKILL.md`); the shared operating contract is
`conventions.md`.

Run `python3 tools/review_op.py <Operator> --domain test` and interpret
per the **test** section (TST-*) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`: independent gold,
required cross-layout parity, axis-coverage mirror, negative tests, tolerance discipline (bit-exact by
default), and the Python API surface. Resolve `MANUAL` items at the cited locations. When
fixing (`--fix`), author the missing positive, negative, or layout-parity cases - **never introduce
or widen a tolerance silently** (emit rationale + needs-human) — and re-run to confirm.
