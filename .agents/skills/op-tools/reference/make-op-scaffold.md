[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op — scaffold

Procedure for the `/make-op-scaffold` dispatcher (`.agents/skills/op-tools/skills/make-op-scaffold/SKILL.md`); the
shared operating contract is `conventions.md`.

Produce a wired, building skeleton and stop. By default, propose the operator's
semantics + a cited reference oracle + the support matrix, **get user approval**, and record the
contract in `Op<Name>.h`; pass `--bare` to delegate the definition too (`SPEC-*` then report
`MANUAL` "spec delegated", not GAP). Run `tools/mkop/mkop.sh <Name>`, then gate with
`python3 tools/make_op.py <Name> --phase scaffold [--bare]` and resolve every `SCF-*`/`SPEC-*`
GAP per `.agents/guidance/MAKE_OP_GUIDELINES.md`. Report the outstanding work (IMP/COV/EXEC) and that
`make-op-verify` is the bar for "done" — it stays red until the operator is implemented.
