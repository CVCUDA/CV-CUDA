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

if [[ -z "${PRIVATE_TOKEN}" ]]; then
    echo "PRIVATE_TOKEN environment variable was not set. Please set it to your private Gitlab access token with api, read_api scopes."
    echo "Private token is used to fetch assets needed to build samples docker image from the package registry."
    echo "Usage:  PRIVATE_TOKEN=... $(basename "$0") [--push]"
    exit 1
fi

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

do_push=0

if [[ $# == 1 && $1 == "--push" ]]; then
    do_push=1
    shift
elif [[ $# != 0 ]]; then
    echo "Usage:  PRIVATE_TOKEN=... $(basename "$0") [--push]"
    exit 1
fi

cd "$SDIR"

# load up configuration variables
. ./config

cd samples

image=$IMAGE_URL_BASE/samples-linux-x64:$TAG_IMAGE

CACHEDIR=/tmp/samples/pkg_cache "$SDIR/../ci/download_assets.sh" ffmpeg-master-latest-linux64-gpl-shared 22.12
tar -xvf "/tmp/samples/pkg_cache/ffmpeg-master-latest-linux64-gpl-shared/22.12/ffmpeg-master-latest-linux64-gpl-shared.tar.gz" -C $SDIR/samples

CACHEDIR=/tmp/samples/pkg_cache "$SDIR/../ci/download_assets.sh" video_codec_sdk 22.12
tar -xvf "/tmp/samples/pkg_cache/video_codec_sdk/22.12/Video_Codec_SDK.tar.gz" -C $SDIR/samples

CACHEDIR=/tmp/samples/pkg_cache "$SDIR/../ci/download_assets.sh" video_processing_framework 22.12
tar -xvf "/tmp/samples/pkg_cache/video_processing_framework/22.12/VideoProcessingFramework.tar.gz" -C $SDIR/samples

nvidia-docker build --network=host --no-cache \
    --build-arg "VER_TRT=$VER_TRT" \
    . -t "$image"

if [[ $do_push == 1 ]]; then
    docker push "$image"
fi

cd ../..
