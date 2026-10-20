[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op

Procedure for the `/optimize-op` dispatcher (`.agents/skills/op-tools/skills/optimize-op/SKILL.md`); the shared
operating contract is `conventions.md`.

Read `.agents/guidance/OPTIMIZATION_GUIDELINES.md` (the core rules and workflow) and
follow it as the source of truth; read `optimize-op-profiling.md` for profiling and
leads, and `optimize-op-summary.md` when writing the MR summary. Use `tools/optimize_op.py` for the deterministic gates and the versioned
MR summary. Do not duplicate or weaken the guidance here.

## Workflow

1. Run `python3 tools/optimize_op.py <Op> --phase preflight`. Resolve every
   readiness `GAP` in a separate non-`perf:` coverage commit before optimizing.
2. Survey the full union expanded from all declared `basic` and `advanced`
   configurations, using the committed before numbers per the guidance's
   baseline policy. Run both tiers for every affected operator after changes.
   Iterate profile → lead → one change →
   correctness/performance evaluation → review/refactor. Run
   `python3 tools/refactor_op.py <Op> --phase assess` before every kept `perf:`
   commit. Re-profile after each win. Before finishing, review each kept
   technique for generalization across the required areas; record applied,
   deferred, or inapplicable opportunities in the MR. Apply the guidance's
   stop criteria to the retained scope; documented deferrals need no trials.
3. Record evidence from tool output as the campaign runs: executed tests with
   their references and covered categories, the Nsight Compute profile, each
   kept or struck lead's before/after benchmark artifacts, per-call-site memory
   peaks, the refactor assessment, and the guarded baseline import and
   validation. Each record names the commit it was measured at and carries no
   authored values.


4. Generate the MR summary block in the canonical format documented in
   `optimize-op-summary.md`, with `state` `provisional`, and refresh it with
   `state` `final` once every configured reference SKU is measured. Scope,
   Impact, and Layout comparison derive from the committed baselines at the
   pre-optimization revision and at HEAD; the bottleneck, checklist, learnings,
   and Memory footprint section derive from the evidence record. None of them
   may be hand-edited. The author writes the generalization review, applying,
   deferring, or ruling out each area.


   Keep the visible warning for any non-reference local SKU retained for
   context.
5. Run the final gate:

   ```bash
   python3 tools/optimize_op.py <Op> --phase evidence \
     --base <ref> --results <description>
   ```

   Require pixelwise-equality evidence for every changed configuration, updated
   and non-regressing comparable baselines, full-surface profile/lead coverage
   (agent-verified per the guidance; the gate does not check it),
   a complete generalization review, record-backed checklist evidence, lead
   exhaustion or documented remaining deferrals, stable API/ABI, and `perf:`
   commit hygiene. ODO-3 validates the review. ODO-9 owns memory-footprint
   acceptance; its `MANUAL` result requires human review before submission.
   Resolve every `GAP` and unresolved `MANUAL`, then rerun. Never weaken or
   silently retune tests, fabricate baselines or records, edit a rendered
   evidence line, weaken the ODO-9 memory policy, or change public API/ABI.
