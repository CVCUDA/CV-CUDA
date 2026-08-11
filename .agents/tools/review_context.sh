#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Emit the shared "git context" block every review wrapper prepends to the shared
# review prompt. Injecting the committed diff as leading context makes all the
# independent agents (claude/codex/cursor) reason over the identical, correct
# scope, instead of each re-deriving it from whatever its own tool access happens
# to gather (which previously drifted -- e.g. an agent with no shell reviewing the
# staged working tree rather than the branch's committed diff).
#
# The diff is the merge-base ("three-dot") diff against the base, so it is scoped
# to this branch's own work and is not polluted by base-side changes made since
# the fork point (a two-dot diff would be). Uncommitted changes are surfaced under
# git status and left for the agent to treat as uncommitted.
#
# Args: <repo_root> [base_ref] (base defaults to main).

set -euo pipefail

repo_root="$1"
base="${2:-main}"

# Never let a git hiccup abort the wrapper: the agents retain full read access and
# can gather context themselves if a command here comes back empty.
run() { git -C "${repo_root}" "$@" 2>&1 || true; return 0; }

cat <<EOF
## Git context (base: ${base})

The committed diff of this branch against \`${base}\` is provided below as latent
context for the code-review instructions that follow. It is the merge-base
("three-dot") \`git diff ${base}...HEAD\`, scoped to this branch's own work
(base-side changes since the fork point are excluded). Any uncommitted working-tree
changes are listed under git status; treat those as uncommitted. You retain full
read access to the repository, so open any file beyond the diff if you need more
context.

### git status --short --branch
\`\`\`
$(run status --short --branch)
\`\`\`

### git log ${base}..HEAD --oneline
\`\`\`
$(run log "${base}..HEAD" --oneline)
\`\`\`

### git diff ${base}...HEAD --stat
\`\`\`
$(run diff "${base}...HEAD" --stat)
\`\`\`

### git diff ${base}...HEAD
\`\`\`diff
$(run diff "${base}...HEAD")
\`\`\`
EOF
