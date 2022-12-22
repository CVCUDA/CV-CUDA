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

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

args="$*"

do_push=0

if [[ $# == 1 && $1 == "--push" ]]; then
    do_push=1
    shift
elif [[ $# != 0 ]]; then
    echo "Usage: $(basename "$0") [--push]"
    exit 1
fi

cd "$SDIR"

# load up configuration variables
. ./config

cd build

image=$IMAGE_URL_BASE/build-linux:$TAG_IMAGE

docker build \
    --build-arg "VER_CUDA=$VER_CUDA" \
    --build-arg "VER_UBUNTU=$VER_UBUNTU" \
    . -t "$image"

if [[ $do_push == 1 ]]; then
    docker push "$image"
fi

cd ..

# must update all other dependent images
./update_devel_image.sh $args

cd ..
