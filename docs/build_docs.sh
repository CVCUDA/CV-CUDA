#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Builds documentation based on sphinx and doxygen
# Usage: docs/build_docs.sh [build folder]

REPO_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

build_type="release"
build_dir="build-rel"

if [[ $# -ge 1 ]]; then
   build_dir=$1
fi

# Get the system's default Python version
DEFAULT_PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# (warning): Use "$@" (with quotes) to prevent whitespace problems.
# shellcheck disable=SC2048

# Check if build directory already has a valid CMake configuration
if [ -f "$build_dir/CMakeCache.txt" ]; then
    echo "Build directory already configured. Enabling documentation with minimal changes..."

    # Read existing PYTHON_VERSIONS from CMakeCache.txt
    EXISTING_PYTHON_VERSIONS=$(grep "^PYTHON_VERSIONS:" "$build_dir/CMakeCache.txt" | cut -d'=' -f2 || echo "")

    # Check if DEFAULT_PYTHON_VER is in EXISTING_PYTHON_VERSIONS
    if [[ "$EXISTING_PYTHON_VERSIONS" == *"$DEFAULT_PYTHON_VER"* ]]; then
        # Default version already present, no need to modify PYTHON_VERSIONS
        "$REPO_ROOT/build.sh" $build_type $build_dir "-DBUILD_DOCS=1 -DBUILD_PYTHON=1 -DDOC_PYTHON_VERSION=$DEFAULT_PYTHON_VER"
    else
        # Add DEFAULT_PYTHON_VER to existing list
        if [ -n "$EXISTING_PYTHON_VERSIONS" ]; then
            NEW_PYTHON_VERSIONS="${EXISTING_PYTHON_VERSIONS};${DEFAULT_PYTHON_VER}"
        else
            NEW_PYTHON_VERSIONS="$DEFAULT_PYTHON_VER"
        fi
        echo "Adding Python $DEFAULT_PYTHON_VER to existing versions: $NEW_PYTHON_VERSIONS"
        "$REPO_ROOT/build.sh" $build_type $build_dir "-DBUILD_DOCS=1 -DBUILD_PYTHON=1 -DPYTHON_VERSIONS=$NEW_PYTHON_VERSIONS -DDOC_PYTHON_VERSION=$DEFAULT_PYTHON_VER"
    fi
else
    echo "Build directory not configured. Running full configuration for docs..."
    # Set all flags explicitly for clean docs-only build with single Python version
    "$REPO_ROOT/build.sh" $build_type $build_dir "-DBUILD_DOCS=1 -DBUILD_TESTS=0 -DBUILD_TESTS_CPP=0 -DBUILD_TESTS_WHEELS=0 -DBUILD_TESTS_PYTHON=0 -DBUILD_PYTHON=1 -DPYTHON_VERSIONS=$DEFAULT_PYTHON_VER -DDOC_PYTHON_VERSION=$DEFAULT_PYTHON_VER"
fi
