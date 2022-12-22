#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Builds samples
# Usage: build_samples.sh [build folder]

build_type="release"
build_dir="build"

if [[ $# -ge 1 ]]; then
   build_dir=$1
   shift
fi

# (warning): Use "$@" (with quotes) to prevent whitespace problems.
# shellcheck disable=SC2048
 ./ci/build.sh $build_type $build_dir "-DBUILD_SAMPLES=ON -DBUILD_TESTS=OFF -DBUILD_PYTHON=1" $*
