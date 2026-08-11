#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEPARATOR_LINE="======================================================================"

# Function to run samples and report status
run_samples() {
    local category="$1"
    local directory="$2"
    local failed_tests=()
    local temp_output

    for script in "$SCRIPT_DIR"/"$directory"/*.py; do
        echo -n "Running $category: $script ... "
        if [[ -f "$script" ]]; then
            # Create a temporary file to capture output
            temp_output=$(mktemp)

            if python3 "$script" > "$temp_output" 2>&1; then
                # Check if output contains warnings or skip messages
                if grep -qiE "(warning|skipping)" "$temp_output"; then
                    echo "PASSED (with warnings)"
                    echo "$SEPARATOR_LINE"
                    echo "OUTPUT from $(basename "$script"):"
                    echo "$SEPARATOR_LINE"
                    cat "$temp_output"
                    echo "$SEPARATOR_LINE"
                    echo ""
                else
                    echo "PASSED"
                fi
                rm -f "$temp_output"
            else
                exit_code=$?
                echo "FAILED (exit code: $exit_code)"
                failed_tests+=("$(basename "$script")")

                # Display the error output for debugging
                echo "$SEPARATOR_LINE"
                echo "ERROR OUTPUT from $(basename "$script"):" >&2
                echo "$SEPARATOR_LINE"
                cat "$temp_output"
                echo "$SEPARATOR_LINE"
                echo ""

                rm -f "$temp_output"
            fi
        fi
    done

    # Print summary if there were failures
    if [[ ${#failed_tests[@]} -gt 0 ]]; then
        echo ""
        echo "FAILED ${#failed_tests[@]} TEST(S): ${failed_tests[*]}"
        return 1
    fi

    return 0
}

# Only run samples if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Run samples for each category
    run_samples "Data Types" "datatypes"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    run_samples "Object Cache" "object_cache"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    run_samples "Operator" "operators"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    run_samples "Application" "applications"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    run_samples "Interoperability" "interoperability"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi
fi
