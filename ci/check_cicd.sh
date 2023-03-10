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

if [ $# != 1 ]; then
    echo "Invalid arguments"
    echo "Usage: $(basename "$0") <container tag id>"
    exit 1
fi

tag_used=$1
shift

# shellcheck source=docker/config
. $SDIR/../docker/config

if [ "$TAG_IMAGE" != "$tag_used" ]; then
    echo "Tag of docker image used, $IMAGE_URL_BASE:$tag_used, must be $TAG_IMAGE. Please update the .gitlab-ci.yml" && false
fi
