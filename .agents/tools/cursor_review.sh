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
if ! command -v cursor-agent >/dev/null 2>&1; then
    echo "MISSING HARNESS: 'cursor-agent' CLI not found on PATH -- skipping this review agent."
    exit 0
fi

# Model-agnostic: forward an optional --model; everything else is the review focus.
model=""
focus_args=()
while (($# > 0)); do
    case "$1" in
        --model)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "cursor_review: --model requires a non-empty value" >&2
                exit 2
            fi
            model="$2"
            shift 2
            ;;
        --model=*)
            model="${1#*=}"
            if [[ -z "${model}" ]]; then
                echo "cursor_review: --model requires a non-empty value" >&2
                exit 2
            fi
            shift
            ;;
        *)
            focus_args+=("$1")
            shift
            ;;
    esac
done

# Prompt = injected git context, then the shared review contract, then any focus.
BASE="${REVIEW_BASE:-main}"
review_prompt="$(bash "${SCRIPT_DIR}/review_context.sh" "${REPO_ROOT}" "${BASE}")"
review_prompt+=$'\n\n---\n\n'
review_prompt+="$(cat "${SCRIPT_DIR}/mr_code_review_prompt.md")"
if ((${#focus_args[@]} > 0)); then
    review_prompt+=$'\n\nSpecial instructions:\n'
    review_prompt+="${focus_args[*]}"
fi

model_args=()
if [[ -n "${model}" ]]; then
    model_args+=(--model "${model}")
fi

# Run in a throwaway worktree so a full-access agent stays isolated from the
# shared tree; this shell owns the cleanup trap (see review_worktree.sh).
source "${SCRIPT_DIR}/review_worktree.sh"
if ! agent_cwd="$(review_worktree_create "${REPO_ROOT}")"; then
    echo "MISSING HARNESS: could not create an isolated worktree -- skipping this review agent."
    exit 0
fi
trap 'review_worktree_remove "${REPO_ROOT}" "${agent_cwd}"' EXIT
cd "${agent_cwd}"

# Full access relying on the sandboxed environment; read-only by contract.
# --force auto-allows commands (no --mode ask/plan: ask can't run git); --trust
# skips the workspace-trust prompt a headless run cannot answer.
cursor-agent \
    --print \
    --force \
    --trust \
    "${model_args[@]}" \
    <<<"${review_prompt}"
