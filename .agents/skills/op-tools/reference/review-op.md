[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Review Op

Procedure for the `/review-op` dispatcher (`.agents/skills/op-tools/skills/review-op/SKILL.md`); the shared
operating contract is `conventions.md`. The checklist substance and the checks are shared repo
artifacts: the deterministic checker `tools/review_op.py` and the spec
`.agents/guidance/REVIEW_OP_GUIDELINES.md`. Do not duplicate the checklist here.

## Workflow

1. Run `python3 tools/review_op.py <Operator>` (scope with
   `--domain support|test|bench|docs`; `--format json` for machine output; `--run` for the
   GPU-run-dependent bench checks). The checker is read-only and deterministic.
2. Interpret each finding by its cited item id in `.agents/guidance/REVIEW_OP_GUIDELINES.md`. Resolve every
   `MANUAL` item by inspecting the location it points to.
3. If the user asked to fix (`--fix`), apply the corrective action named per `GAP` in
   `.agents/guidance/REVIEW_OP_GUIDELINES.md`, then **re-run the checker to prove the item is `PASS`**.
   Follow the inviolable `--fix` rules in `.agents/guidance/REVIEW_OP_GUIDELINES.md` ("### `--fix`").
4. Return findings-first: `GAP`s, then unresolved `MANUAL`s, then `RECOMMENDATION`s, then the
   per-domain + overall verdict. Completion = a re-run shows zero GAP and zero unresolved MANUAL.

## Prompt Handling

Requests like "review the Resize operator", "/review-op CenterCrop", "audit Flip's bench
coverage", or "does Remap cover every declared layout?" trigger this skill. Use `--domain` to
scope to a single domain when the request is domain-specific.
