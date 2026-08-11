#!/bin/bash
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

# Generate requirements .txt files from *.template sources and versions.env.
# Requirements outputs are written next to their templates across the tree.
# Each *.template file uses ${var_name} placeholders defined in versions.env.
#
# Usage:
#   bash generate_requirements.sh          # generate all managed dependency files
#   bash generate_requirements.sh --check  # verify files are up to date (exits 1 if stale)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEMPLATES=(
    docker/requirements.build.all_pythons.template
    docker/requirements.build.sys_python.template
    tests/requirements.tests.common.template
    tests/requirements.tests.cu12.template
    tests/requirements.tests.cu12.numpy1.template
    tests/requirements.tests.cu13.template
    tests/requirements.tests.numpy1.template
    tests/requirements.tests.numpy2.template
    bench/python/requirements.bench.common.template
    bench/python/requirements.bench.cu12.template
    bench/python/requirements.bench.cu13.template
    samples/requirements.samples.common.template
    samples/requirements.samples.cu12.template
    samples/requirements.samples.cu13.template
    samples/requirements.samples.hello_world_cu12.template
    samples/requirements.samples.hello_world_cu13.template
    docs/requirements.docs.template
)

# Render a template to stdout: strip the "Template for …" comment block and
# substitute ${var_name} placeholders with values from versions.env.
_render() {
    local tpl="$1"
    local tpl_name sed_args=()
    tpl_name=$(basename "$tpl")

    while IFS='=' read -r key value; do
        # Strip inline comments (# and everything after) and trailing whitespace
        value="${value%%#*}"
        value="${value%"${value##*[^[:space:]]}"}"
        [[ -z "$value" ]] && continue
        sed_args+=(-e "s|\${${key}}|${value}|g")
    done < <(grep -v '^[[:space:]]*#' "$REPO_ROOT/versions.env" | grep '=')

    printf '# AUTO-GENERATED \xe2\x80\x94 do not edit directly.\n'
    printf '# Edit %s or versions.env, then run: bash generate_requirements.sh\n' "$tpl_name"
    printf '\n'

    awk '
        /^# Template for /              { skip = 1 }
        skip && /generate_requirements/ { skip = 2; next }
        skip == 2 && /^$/               { skip = 0; next }
        skip                            { next }
        { print }
    ' "$tpl" | sed "${sed_args[@]}"
    return $?
}


# ── Main ──────────────────────────────────────────────────────────────────────

check_mode=0
[[ "${1:-}" == "--check" ]] && check_mode=1

stale=()
skipped=0
for tpl_rel in "${TEMPLATES[@]}"; do
    tpl="$REPO_ROOT/$tpl_rel"
    out="${tpl%.template}.txt"

    # Tolerate sparse checkouts: CI test pods that only need a subset of
    # the tree (e.g. test-benchmarks pulls just bench/) won't have every
    # template on disk.  Skip what isn't present in both modes.
    if [[ ! -f "$tpl" ]]; then
        printf '  skip %s (template not present — partial checkout?)\n' "$tpl_rel"
        skipped=$((skipped + 1))
        continue
    fi

    if (( check_mode )); then
        tmp=$(mktemp)
        _render "$tpl" > "$tmp"
        if ! cmp -s "$tmp" "$out" 2>/dev/null; then
            stale+=("${out#"$REPO_ROOT/"}")
        fi
        rm -f "$tmp"
    else
        _render "$tpl" > "$out"
        printf '  wrote %s\n' "${out#"$REPO_ROOT/"}"
    fi
done


if (( check_mode )); then
    if (( ${#stale[@]} > 0 )); then
        printf '\n%d managed dependency file(s) are out of date or missing:\n' "${#stale[@]}" >&2
        printf '  %s\n' "${stale[@]}" >&2
        printf '\nRun: bash generate_requirements.sh\n' >&2
        exit 1
    else
        checked=$(( ${#TEMPLATES[@]} - skipped ))
        if (( skipped > 0 )); then
            printf 'OK \xe2\x80\x94 %d/%d requirements files up to date (%d skipped — partial checkout).\n' \
                   "$checked" "${#TEMPLATES[@]}" "$skipped"
        else
            printf 'OK \xe2\x80\x94 all %d requirements files are up to date.\n' "${#TEMPLATES[@]}"
        fi
    fi
fi
