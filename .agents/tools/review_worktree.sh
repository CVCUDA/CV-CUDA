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

# shellcheck shell=bash

# Sourced by the review wrappers to run each agent in its own throwaway git
# worktree, so a full-access agent's writes -- and the other agents launched
# concurrently -- never touch the shared working tree. The worktree is a detached
# checkout of the repo's current commit (the committed snapshot under review);
# uncommitted changes are surfaced through the injected git context, not here.
#
# Split into create + remove so the wrapper owns the cleanup trap. The worktree
# must outlive the command substitution that captures its path, so creation must
# not register the trap itself: a $(...) subshell fires an EXIT trap the instant
# it returns, which would destroy the worktree before the agent runs. Usage:
#
#     source "${SCRIPT_DIR}/review_worktree.sh"
#     agent_cwd="$(review_worktree_create "${REPO_ROOT}")" || exit 0  # skip if none
#     trap 'review_worktree_remove "${REPO_ROOT}" "${agent_cwd}"' EXIT
#     cd "${agent_cwd}"   # codex instead passes --cd "${agent_cwd}"

# Create a detached worktree at the repo's current commit and echo its path, or
# return non-zero if one cannot be created so the caller can skip the agent.
review_worktree_create() {
    local repo_root="$1" parent wt
    parent="$(mktemp -d "${TMPDIR:-/tmp}/mrcr-agent.XXXXXX")"
    wt="${parent}/tree"
    if git -C "${repo_root}" worktree add --detach --quiet "${wt}" HEAD 2>/dev/null; then
        printf '%s\n' "${wt}"
    else
        rm -rf "${parent}" 2>/dev/null || true
        return 1
    fi
}

# Remove a worktree created by review_worktree_create; a no-op on an empty path or
# the repo root, so the real working tree is never removed.
review_worktree_remove() {
    local repo_root="$1" wt="$2"
    [[ -n "${wt}" && "${wt}" != "${repo_root}" ]] || return 0
    git -C "${repo_root}" worktree remove --force "${wt}" 2>/dev/null || true
    rm -rf "$(dirname "${wt}")" 2>/dev/null || true
}
