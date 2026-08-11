---
name: review-op-docs-api
description: Review a CV-CUDA operator's DOCS & API artifacts — operator_list row, Python autofunction (fn + _into), Limitations-table-vs-code consistency, docstrings, and SPDX headers. Use when asked whether an operator's docs/API surface is complete and consistent.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op — docs & API

Thin entry point. Run `python3 tools/review_op.py <Operator> --domain docs` and interpret
per the **docs** section (DOC-*) of `.agents/guidance/REVIEW_OP_GUIDELINES.md`: the `operator_list.rst` row,
Python `cvcuda-autofunction` directives (fn + `_into`), the Limitations-table-vs-code
cross-domain diff, docstrings, and SPDX headers. Resolve `MANUAL` items at the cited
locations. When fixing (`--fix`), add the missing docs row / autofunction / SPDX header;
correct a stale Limitations entry only where the declared-vs-enforced-vs-tested diff is
unambiguous; re-run to confirm. Findings-first.
