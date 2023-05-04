#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

do_push=0

if [[ $# == 1 && $1 == "--push" ]]; then
    do_push=1
    shift
elif [[ $# != 0 ]]; then
    echo "Usage: $(basename "$0") [--push]"
    exit 1
fi

cd "$SDIR"
# Copy install_dependencies script from the samples folder to the samples' docker folder
# so that it can be added and used inside the image.
cp $SDIR/../samples/scripts/install_dependencies.sh $SDIR/samples/

# load up configuration variables
. ./config
cd samples
image=$IMAGE_URL_BASE/samples-linux-x64:$TAG_IMAGE_SAMPLES

nvidia-docker build --network=host --no-cache \
    --build-arg "VER_TRT=$VER_TRT" \
    . -t "$image"

if [[ $do_push == 1 ]]; then
    docker push "$image"
fi

cd ../..
