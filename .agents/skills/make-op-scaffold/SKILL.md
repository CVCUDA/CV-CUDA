---
name: make-op-scaffold
description: Scaffold a new CV-CUDA operator — a complete, wired, building skeleton — and delegate the implementation to a human or another AI. Use when asked to set up / stub out a new operator without implementing it, or to bootstrap one for someone else to finish. Supports --bare (skip the spec when the definition itself is delegated).
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op — scaffold

Thin entry point. Produce a wired, building skeleton and stop. By default, propose the operator's
semantics + a cited reference oracle + the support matrix, **get user approval**, and record the
contract in `Op<Name>.h`; pass `--bare` to delegate the definition too (`SPEC-*` then report
`MANUAL` "spec delegated", not GAP). Run `tools/mkop/mkop.sh <Name>`, then gate with
`python3 tools/make_op.py <Name> --phase scaffold [--bare]` and resolve every `SCF-*`/`SPEC-*`
GAP per `.agents/guidance/MAKE_OP_GUIDELINES.md`. Report the outstanding work (IMP/COV/EXEC) and that
`make-op-verify` is the bar for "done" — it stays red until the operator is implemented.
Findings-first.
