---
name: op-tools
description: Hub of the CV-CUDA op-tools bundle — the operator-lifecycle command families (make-op, review-op, optimize-op, refactor-op, device-code-proof) and their shared operating contract. Use when asked which op command applies to a task or how the op tooling fits together; the per-command skills remain the entry points for the work itself.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# op-tools

One home for the CV-CUDA operator-lifecycle tooling. Every command is a thin dispatcher
over a deterministic checker in `tools/` and a guidance document in `.agents/guidance/`; the
shared operating contract (checker loop, findings-first reporting, inviolables) lives in
`reference/conventions.md`, and each command's procedure lives in `reference/<command>.md`.

## Layout

The 13 per-command skills nest under `skills/` — skill discovery walks their `SKILL.md`
recursively, so `/make-op`, `/review-op`, … keep their names and trigger as before. The four
verb skills `review-op`, `optimize-op`, `refactor-op`, `device-code-proof` may delegate to an
optional internal `<verb>-worker` sub-agent; the reference docs remain the complete procedure.

## Command map

| Command | Checker | Guidance | Purpose |
|---------|---------|----------|---------|
| `/make-op` | `tools/make_op.py` | `MAKE_OP_GUIDELINES.md` | Add a new operator end-to-end |
| `/make-op-scaffold` | `tools/make_op.py --phase scaffold` | `MAKE_OP_GUIDELINES.md` | Wired skeleton; implementation delegated |
| `/make-op-verify` | `tools/make_op.py --phase done --run` | `MAKE_OP_GUIDELINES.md` | New-operator done-gate |
| `/review-op` | `tools/review_op.py` | `REVIEW_OP_GUIDELINES.md` | End-to-end coverage review |
| `/review-op-support` | `tools/review_op.py --domain support` | `REVIEW_OP_GUIDELINES.md` (SUP-*) | Support matrix |
| `/review-op-test-coverage` | `tools/review_op.py --domain test` | `REVIEW_OP_GUIDELINES.md` (TST-*) | Test coverage |
| `/review-op-bench-coverage` | `tools/review_op.py --domain bench` | `REVIEW_OP_GUIDELINES.md` (BEN-*) | Benchmark coverage |
| `/review-op-docs-api` | `tools/review_op.py --domain docs` | `REVIEW_OP_GUIDELINES.md` (DOC-*) | Docs & API surface |
| `/optimize-op` | `tools/optimize_op.py` | `OPTIMIZATION_GUIDELINES.md` | Drive a performance campaign |
| `/optimize-op-preflight` | `tools/optimize_op.py --phase preflight` | `OPTIMIZATION_GUIDELINES.md` | Readiness gate |
| `/optimize-op-verify` | `tools/optimize_op.py --phase evidence` | `OPTIMIZATION_GUIDELINES.md` | Campaign done-gate |
| `/refactor-op` | `tools/refactor_op.py` | `REFACTOR_OP_GUIDELINES.md` | Redundancy reduction + parity verify |
| `/device-code-proof` | `tools/device_code_proof.py` | `REFACTOR_OP_GUIDELINES.md` (VER-6) | No-build device-code bit-exactness evidence |

The lifecycle composes: `/make-op` scaffolds and gates a new operator, hands off to
`/optimize-op` for the performance campaign, `/review-op` audits coverage at any point, and
`/refactor-op` (+ `/device-code-proof` when no build is available) keeps implementation
duplication down without changing observable behavior.

## Prompt handling

Requests like "which op tool should I use", "what op commands exist", or "how do the make-op /
review-op / optimize-op skills fit together" trigger this skill. A request that already names
the task (review/optimize/refactor/add an operator) should go straight to that command.
