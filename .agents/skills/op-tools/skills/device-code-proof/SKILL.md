---
name: device-code-proof
description: Prove an operator refactor changed nothing in the device code, without a build tree and without a GPU, by compiling the changed translation unit before and after and classifying the SASS/PTX difference. Use when asked to show a refactor is bit-exact, to satisfy VER-6 / NEEDS-LOCAL-PROOF from `tools/refactor_op.py --phase verify`, or to explain why codegen changed. Produces a classified per-kernel report; a non-empty diff is not a failure and an unexplained one is not a pass.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Device Code Proof

Nested skill of the **op-tools** bundle (`.agents/skills/op-tools/`). Follow the shared operating
contract in `.agents/skills/op-tools/reference/conventions.md`; the full procedure for this
command is `.agents/skills/op-tools/reference/device-code-proof.md`.

- **Run:** `python3 tools/device_code_proof.py <Operator> --base <ref>` (or `--file <TU>`;
  `--stage sass` for the strongest nvcc-only tier).
- **Interpret per:** the VER-6 leg of `.agents/guidance/REFACTOR_OP_GUIDELINES.md` — evidence for
  the leg, never a replacement for the frozen test run.

**Delegation rule.** At top level, delegate the full procedure to the `device-code-proof-worker`
sub-agent (`.agents/agents/device-code-proof-worker.md`) and read back its verdict; when you are already
inside a delegated sub-agent run, or the task is a single checker invocation, run inline. The
worker is an optional accelerator — `.agents/agents/` is internal-only and the release trim
removes it, so the reference doc above remains the complete procedure.
