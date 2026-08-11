---
name: optimize-op
description: Drive a single-operator optimization campaign per .agents/guidance/OPTIMIZATION_GUIDELINES.md, with a deterministically enforced definition-of-done and versioned MR summary. Use when asked to optimize an operator, run or finish a performance campaign, generate or refresh its MR performance summary, or verify that a perf MR satisfies correctness, benchmark, baseline, lead-exhaustion, memory, and API/ABI gates.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op

Read `.agents/guidance/OPTIMIZATION_GUIDELINES.md` and follow it as the source of
truth. Use `tools/optimize_op.py` for the deterministic gates and the versioned
MR summary. Do not duplicate or weaken the guidance here.

## Workflow

1. Run `python3 tools/optimize_op.py <Op> --phase preflight`. Resolve every
   readiness `GAP` in a separate non-`perf:` coverage commit before optimizing.
2. Survey and baseline the full union expanded from all declared `basic` and
   `advanced` configurations. Iterate profile → lead → one change →
   correctness/performance evaluation → review/refactor. Run
   `python3 tools/refactor_op.py <Op> --phase assess` before every kept `perf:`
   commit. Re-profile after each win and stop only at the guidance's measured
   ridge or three-post-win-strike condition.
3. Initialize the concise v1 MR summary from exact optimized case keys:

   ```bash
   python3 tools/optimize_op.py <Op> --phase summary \
     --benchmark-base <ref> --optimized-cases-file <file> \
     --state provisional --bottleneck Memory-bound \
     --profile-evidence '<measured evidence>' --out <description>
   ```

   Use `Compute-bound` when supported by the profile. Keep the human-authored
   assessment, six hard-evidence checklist lines, and one to five learnings
   current. Regenerate scope, categories, per-SKU Impact, and Layout comparison;
   do not hand-edit those derived fields. Maintain the canonical companion
   `## Memory footprint` section outside the bounded v1 block; check its summary
   item only after ODO-9 passes.
4. Refresh the description with reference-SKU evidence and finalize it:

   ```bash
   python3 tools/optimize_op.py <Op> --phase summary \
     --results <description> --state final --out <description>
   ```

   Require every configured reference SKU. Keep the visible warning for any
   non-reference local SKU retained for context.
5. Run the final gate:

   ```bash
   python3 tools/optimize_op.py <Op> --phase evidence \
     --base <ref> --results <description>
   ```

   Require pixelwise-equality evidence for every changed configuration, updated
   and non-regressing comparable baselines, full-surface profile/lead coverage
   (agent-verified per the guidance; the gate does not check it),
   hard evidence for all six checklist items, exhausted leads, stable API/ABI,
   and `perf:` commit hygiene. Treat ODO-9 as the memory-footprint authority;
   its `MANUAL` result requires explicit human review before submission.
   Resolve every `GAP` and unresolved `MANUAL`, then rerun. Never weaken or
   silently retune tests, fabricate baselines, check an evidence item without
   its hard proof, weaken the ODO-9 memory policy, or change public API/ABI.
