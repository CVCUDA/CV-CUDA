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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
readonly REPO_ROOT

# Missing CLI: write a sentinel the orchestrator skips, don't fail the review.
if ! command -v codex >/dev/null 2>&1; then
    echo "MISSING HARNESS: 'codex' CLI not found on PATH -- skipping this review agent."
    exit 0
fi

# Prompt = injected git context, then the shared review contract, then any focus.
# `codex review` rejects `--base` with a custom PROMPT, so the diff is passed here.
BASE="${REVIEW_BASE:-main}"
review_prompt="$(bash "${SCRIPT_DIR}/review_context.sh" "${REPO_ROOT}" "${BASE}")"
review_prompt+=$'\n\n---\n\n'
review_prompt+="$(cat "${SCRIPT_DIR}/mr_code_review_prompt.md")"
if (($# > 0)); then
    review_prompt+=$'\n\nSpecial instructions:\n'
    review_prompt+="$*"
fi

# Run in a throwaway worktree so a full-access agent stays isolated from the
# shared tree; this shell owns the cleanup trap (see review_worktree.sh).
source "${SCRIPT_DIR}/review_worktree.sh"
if ! agent_cwd="$(review_worktree_create "${REPO_ROOT}")"; then
    echo "MISSING HARNESS: could not create an isolated worktree -- skipping this review agent."
    exit 0
fi
trap 'review_worktree_remove "${REPO_ROOT}" "${agent_cwd}"' EXIT

# Full access relying on the sandboxed environment; read-only by contract.
codex \
    --sandbox danger-full-access \
    --ask-for-approval never \
    --cd "${agent_cwd}" \
    review \
    - <<<"${review_prompt}"
