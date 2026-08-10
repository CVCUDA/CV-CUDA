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

# Creates the build tree and builds CV-CUDA

# Usage: build.sh [build type] [build dir] [additional cmake args]
# build type:
#   - debug or release
#   - If not specified, defaults to release
# build dir:
#   - Where build tree will be created
#   - If not specified, defaults to either build-rel or build-deb, depending on the build type

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

# Defaults
build_type="release"
build_dir=""
source_dir="$SDIR"
num_jobs="${CVCUDA_BUILD_JOBS:-$(nproc)}" # Allow callers to cap build parallelism.

if ! [[ "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid CVCUDA_BUILD_JOBS='${CVCUDA_BUILD_JOBS:-}'; falling back to nproc." >&2
    num_jobs="$(nproc)"
fi

# Command line parsing
if [[ $# -ge 1 ]]; then
    case $1 in
    debug|release)
        build_type=$1
        if [[ $# -ge 2 ]]; then
            build_dir=$2
            shift
        fi
        shift
        ;;
    *)
        build_dir=$1
        shift
        ;;
    esac
fi

# Create build directory
build_dir=${build_dir:-"build-${build_type:0:3}"}
mkdir -p "$build_dir"

# Initialize cmake_args with any additional args passed from command line
cmake_args="$*"

# Specific configurations for build type
case $build_type in
    release)
        cmake_args="$cmake_args -DCMAKE_BUILD_TYPE=Release"
        ;;
    debug)
        cmake_args="$cmake_args -DCMAKE_BUILD_TYPE=Debug"
        ;;
    *)
        ;;
esac

# Configure build toolchain
CC=${CC:-$(find /usr/bin/gcc-1* | sort -rV | head -n 1)}
CXX=${CXX:-$(find /usr/bin/g++-1* | sort -rV | head -n 1)}
cmake_args="$cmake_args -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"

# Use ninja if available
if which ninja > /dev/null; then
    cmake_args="$cmake_args -G Ninja"
    export NINJA_STATUS="[%f/%t %r %es] "
fi

# Configure ccache
if which ccache > /dev/null; then
    ccache_stats=$(pwd)/$build_dir/ccache_stats.log
    rm -rf "$ccache_stats"
    cmake_args="$cmake_args -DCCACHE_STATSLOG=${ccache_stats}"
fi

# Configure CUDA - use default installation
if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    cmake_args="$cmake_args -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
fi

# Generate requirements files from templates and versions.env
bash "$source_dir/generate_requirements.sh"

# Create build tree and build. Forward the job budget so the Python wheel
# sub-builds can cap their combined fan-out (see cmake/BuildPython.cmake).
cmake -B "$build_dir" "$source_dir" $cmake_args -DCVCUDA_BUILD_JOBS="${num_jobs}"
cmake --build "$build_dir" -- "-j${num_jobs}"

# Show ccache status
if which ccache > /dev/null; then
    ccache --show-stats -V
fi
