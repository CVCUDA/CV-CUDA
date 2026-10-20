[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — docs & API

Procedure for the `/review-op-docs-api` dispatcher (`.agents/skills/op-tools/skills/review-op-docs-api/SKILL.md`);
the shared operating contract is `conventions.md`.

Run `python3 tools/review_op.py <Operator> --domain docs` and interpret
per the **docs** section (DOC-*) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`: the `operator_list.rst` row,
Python `cvcuda-autofunction` directives (fn + `_into`), the Limitations-table-vs-code
cross-domain diff, docstrings, and SPDX headers. Resolve `MANUAL` items at the cited
locations. When fixing (`--fix`), add the missing docs row / autofunction / SPDX header;
correct a stale Limitations entry only where the declared-vs-enforced-vs-tested diff is
unambiguous; re-run to confirm.
