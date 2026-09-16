---
name: refactor-op
description: Find and safely apply per-operator refactoring / redundancy-reduction opportunities in a CV-CUDA operator (near-duplicate Tensor/VarShape kernels, reinvented shared utilities, dead code). Use when asked to reduce code duplication, de-duplicate or unify an operator's kernels/bindings, remove dead code, or verify a refactor changed nothing observable. Produces a deterministic findings-first report; the apply path is gated by a strict bit-exact / feature-set / test-coverage parity check.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Refactor Op

Thin entry point. The catalog substance and the checks are shared repo artifacts: the deterministic
checker `tools/refactor_op.py` and the spec `.agents/guidance/REFACTOR_OP_GUIDELINES.md`. Keep this
skill thin; do not duplicate the catalog here. `refactor-op` complements generic, whole-corpus
copy-paste detection by surfacing operator-scoped *semantic* redundancy it cannot express.

## Workflow

1. **assess** — `python3 tools/refactor_op.py <Operator>` (scope with `--domain impl|api|xcut`;
   `--format json` for machine output). Read-only and deterministic. Interpret each finding by its
   `RED-*` id in `.agents/guidance/REFACTOR_OP_GUIDELINES.md`.

   **Sweep the block floor downwards while planning.** The default floor hides small shared
   helpers, and a one-line helper defined in five operators is still a helper that belongs in a
   shared header. Re-run assess at successively smaller floors and collect the union of the
   `RED-2` leads:
   ```bash
   for n in 6 5 4 3 2 1; do
     python3 tools/refactor_op.py <Operator> --domain impl --min-block-lines "$n" --format json
   done
   ```
   Floor 1 is the point of the sweep, not an extreme: it is the only floor that can surface a
   genuinely one-line helper, and the tree has several duplicated across operators
   (`IsPlanarLayout`, `IsInterleaved`, `SameLayoutFamily`). Below the shingle window a body is
   fingerprinted whole, so short blocks match only on **exact** equality — a low-floor match is a
   verbatim twin, not a fuzzy one. Triage what the sweep adds:
   macro definitions (`NVCV_*_RUN_TYPED`) and one-line accessors named `read`/`write`/`release`
   are usually per-operator by design, while named helpers (`CheckedWorkspaceAdd`,
   `WorkspaceReleaseGuard`, `CurrentDeviceSMOrZero`) are real hoist candidates. Report the floor
   each kept lead was found at.

   Note the tool only ever names **function-like** bodies, so duplicated `struct`/trait
   definitions are invisible at every floor — check those by hand.

   **Let the tool do the mechanical analysis.** Three modes replace work that is otherwise done by
   reading files side by side, and they are cheap — prefer them over your own reasoning, then apply
   judgement to what they return:
   ```bash
   python3 tools/refactor_op.py --all-operators --domain impl   # where the work is, ranked
   python3 tools/refactor_op.py --variants ValidateSrcDstTensors # how many versions really exist
   python3 tools/refactor_op.py --simulate-hoist A,B,C           # the RED-2 delta, before planning
   ```
   `--simulate-hoist` in particular belongs **before** the plan is written: it reports which
   operators reach zero and which merely drop, and a plan that assumes the wrong one is wrong.
   `--variants` groups syntactically, so read the largest class before hoisting it — members can
   still differ semantically in a way that matters.

   For an operator whose translation units you are about to change, `tools/tu_cost.py <Op>
   --build-dir <dir> --out base.json` before and `--compare base.json` after gives compile-time and
   `.text` deltas to cite as simplification evidence.
2. **apply** (only if asked to fix) — apply the finding's named corrective action on the
   **implementation / binding only**, reusing existing shared utilities. **Never** edit the
   operator's tests or anything under `bench/`, and **never** change public API/ABI.
3. **verify** — when the MR scope includes a refactor, run
   `python3 tools/refactor_op.py <Operator> --phase verify --base <ref>` locally and require zero
   `GAP` (frozen test surface, frozen bench surface, API/ABI, feature-matrix and coverage-matrix
   parity), then build + run the frozen `Op<Operator>` tests (bit-exact, VER-6) and re-run assess
   (redundancy gone, or reported fewer times, VER-7). When the verdict is `NEEDS-LOCAL-PROOF` and
   you cannot build, `python3 tools/device_code_proof.py <Operator> --base <ref>` (the
   `/device-code-proof` skill) compiles the changed TU on both sides with no build tree and no GPU
   and classifies the per-kernel device-code difference. It is evidence *for* VER-6 covering only
   the arch(es) compiled and nothing host-side — it never replaces the frozen test run.
   The verify report ends with a deterministic **refactor summary** (LOC delta + `RED-*`
   resolved/reduced before→after) — report it as the outcome. If the MR also changes
   performance-sensitive implementation, provide benchmark evidence through the optimization
   workflow.

Inviolable: refactoring changes nothing observable — bit-exact output, identical feature set, frozen
tests and benchmarks, unchanged API/ABI; never weaken a test or fabricate a baseline to pass. There
is no standing `refactor-parity` CI job; invoke the local verify flow when the MR scope calls for it.

## Prompt Handling

Requests like "reduce duplication in Flip", "/refactor-op BrightnessContrast", "unify the Tensor and
VarShape kernels for Resize", "is there dead code in OpStack?", or "verify my refactor is bit-exact"
trigger this skill. Use `--domain` to scope; use `--phase verify` to gate an applied refactor.
