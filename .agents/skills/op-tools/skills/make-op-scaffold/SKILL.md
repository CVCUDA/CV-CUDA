---
name: make-op-scaffold
description: Scaffold a new CV-CUDA operator — a complete, wired, building skeleton — and delegate the implementation to a human or another AI. Use when asked to set up / stub out a new operator without implementing it, or to bootstrap one for someone else to finish. Supports --bare (skip the spec when the definition itself is delegated).
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Make Op — scaffold

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/make-op-scaffold.md`.

- **Run:** `tools/mkop/mkop.sh <Name>`.
- **Gate:** `python3 tools/make_op.py <Name> --phase scaffold [--bare]`. Pass `--bare` (to the gate,
  not `mkop.sh`) only when the spec itself is delegated.
- **Interpret per:** `.agents/guidance/MAKE_OP_GUIDELINES.md` (`SCF-*`/`SPEC-*`).
