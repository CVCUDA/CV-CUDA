[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — verify

Procedure for the `/optimize-op-verify` dispatcher (`.agents/skills/op-tools/skills/optimize-op-verify/SKILL.md`);
the shared operating contract is `conventions.md`.

Read `optimize-op-summary.md`; open a gate's section only when the done-gate reports it
(each finding's `ref`). Re-derive the summary tables
from the committed baselines with `state` `final`, then run the done-gate:


```bash
python3 tools/optimize_op.py <Op> --phase evidence \
  --base <ref> --results <description>
```

Report findings first. Require the exact full-union key set expanded from every
declared `basic` and `advanced` entry, before/after C++ timing for every key on
every configured reference SKU, per-operator scope counts that match the Impact
rows, verified
min/median/max statistics, verified layout ratios or verified n/a rows, and all
checklist items checked from the embedded evidence record and a complete
Generalization review.
Check every kept technique across the four required areas; accept explicit
implementation deferrals with scope reasons. ODO-3 validates that review;
remaining deferrals may satisfy Lead exhaustion without additional trials.
Require the canonical companion `## Memory footprint` section outside the
bounded block, and treat ODO-9 as the memory-footprint authority; its `MANUAL`
result requires explicit human
review before submission. Declare the campaign done only after a rerun shows
zero `GAP` and zero unresolved `MANUAL`. Never weaken tests, fabricate
baselines or evidence records, edit rendered evidence lines, weaken the ODO-9 memory policy, or
change public API/ABI.
