[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# op-tools shared operating contract

Every op-tools command is a **thin dispatcher** over two shared repo artifacts: a deterministic
checker under `tools/` and a guidance document under `.agents/guidance/`. The checklist substance
lives in those artifacts once; dispatchers and these reference docs route to them and never
restate the checklists. Where a guidance doc delegates a phase to a sibling reference doc here
(for example `optimize-op-summary.md`), that doc is the owner and each checker finding's `ref`
names the one section to read. When a dispatcher and its guidance disagree, the guidance wins — fix the
dispatcher, do not fork the rules.

## The checker loop

1. Run the command's checker. The checkers are read-only and deterministic; pass
   `--format json` for machine output and `--run` where a phase has GPU-run-dependent checks.
2. Interpret each finding by its cited item id in the command's guidance document.
3. Resolve every `MANUAL` item by inspecting the location it points to.
4. When asked to fix, apply the corrective action the guidance names for that `GAP`, then
   **re-run the checker to prove the item is `PASS`**. A fix claimed without a green re-run is
   not done.
5. Completion = a re-run shows zero `GAP` and zero unresolved `MANUAL`.

## Findings-first reporting

Report `GAP`s first, then unresolved `MANUAL`s, then `RECOMMENDATION`s, then the per-domain or
per-phase verdicts and the overall verdict.

## Inviolables

- **Never fabricate baselines.** A missing baseline is `GAP [requires CI]` → CI regeneration;
  absolute timings are SKU-specific, so a locally invented number is worse than a gap.
- **Never weaken or silently retune a test or tolerance.** Bit-exact is the default; an
  `EXPECT_NEAR` on an interleaved path is a `GAP`, and widening a tolerance needs a recorded
  rationale plus a needs-human flag.
- **Layout-support gaps are author work.** Image operators support interleaved and planar
  layouts by default; where image layouts do not apply, record the operator-local reason in the
  Limitations contract instead of waiving the gap.
- **No public API/ABI change to pass a gate.** Refactor and optimization campaigns must leave
  the public surface unchanged; new-operator work declares its API in the approved spec.
