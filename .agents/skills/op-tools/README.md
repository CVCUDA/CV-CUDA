[//]: # "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# op-tools

One home for the CV-CUDA operator-lifecycle tooling: the `make-op`, `review-op`,
`optimize-op`, and `refactor-op` command families plus `device-code-proof`.

## Layout

```
.agents/skills/op-tools/
├── SKILL.md                  # hub: command map + routing
├── README.md
├── reference/
│   ├── conventions.md
│   └── <command>.md
└── skills/                   # the 13 op-series skills
    ├── make-op/SKILL.md
    ├── make-op-scaffold/SKILL.md      … one dir per command …
    └── device-code-proof/SKILL.md
```

The slash commands keep their names: skill discovery walks each nested
`skills/<command>/SKILL.md` recursively, so `/make-op`, `/review-op`, … trigger exactly as
before. A nested skill carries only the routing frontmatter, the checker invocation, and
pointers here; the procedure text lives in `reference/<command>.md` and the shared contract in
`reference/conventions.md`.

The deterministic checkers (`tools/make_op.py`, `tools/review_op.py`, `tools/optimize_op.py`,
`tools/refactor_op.py`, `tools/device_code_proof.py`) and the guidance documents
(`.agents/guidance/*_GUIDELINES.md`) are shared repo artifacts and deliberately stay where they
are — op-tools routes to them, it does not wrap or move them.

## Publishing

This tree ships in the public OSS mirror: every file here carries an Apache-2.0 SPDX header
and must stay free of internal infrastructure names. The publish whitelist entry is the single
`op-tools` glob in `ci/publish_whitelist.sh` — the nested `skills/` and `reference/` trees ship
with it. Internal-only helpers must not live here; put them under `.agents/tools/`, which the
trim prunes by SPDX header. The optional `<verb>-worker` sub-agents the verb skills may
delegate to live under `.agents/agents/`, which the release trim removes; they are internal
accelerators only, never the procedure's home.
