---
name: optimize-op-verify
description: Verify a CV-CUDA optimization campaign's deterministic definition-of-done and concise versioned MR summary per .agents/guidance/OPTIMIZATION_GUIDELINES.md. Use to gate whether a perf campaign or performance MR is ready, including reference-SKU statistics, hard checklist evidence, baseline validation, lead exhaustion, memory checks, and API/ABI stability.
---

[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Optimize Op — verify

Read `.agents/guidance/OPTIMIZATION_GUIDELINES.md`, then refresh the v1 summary
and run the done-gate:

```bash
python3 tools/optimize_op.py <Op> --phase summary \
  --results <description> --state final --out <description>
python3 tools/optimize_op.py <Op> --phase evidence \
  --base <ref> --results <description>
```

Report findings first. Require the exact full-union key set expanded from every
declared `basic` and `advanced` entry, before/after C++ timing for every key on
every configured reference SKU, exact optimized/full counts, verified
min/median/max statistics, verified layout ratios or verified n/a rows, and all
six checklist items checked with hard evidence. Require the canonical companion
`## Memory footprint` section outside the bounded v1 block, and treat ODO-9 as
the memory-footprint authority; its `MANUAL` result requires explicit human
review before submission. Declare the campaign done only after a rerun shows
zero `GAP` and zero unresolved `MANUAL`. Never weaken tests, fabricate
baselines, check unsupported evidence, weaken the ODO-9 memory policy, or
change public API/ABI.
