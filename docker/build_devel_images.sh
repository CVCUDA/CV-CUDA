#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Auto-detect architecture for tagging
host_arch=$(uname -m)

# Validate supported architectures for tagging
case $host_arch in
    x86_64|aarch64)
        # Supported architecture
        ;;
    *)
        # Use the detected arch name directly for unsupported, but warn
        echo "Warning: Unsupported host architecture '$host_arch' for tagging. Using architecture name as tag."
        ;;
esac

echo "Detected host architecture: $host_arch, tagging images with suffix: -$host_arch"

cd "$SDIR"

IMAGE_URL_BASE='gitlab-master.nvidia.com:5005/cv/cvcuda'

# Define the versions to loop over
CUDA_VERSIONS=("11.7.1" "12.2.0")
UBUNTU_VERSIONS=("20.04" "22.04")
NUMPY_VERSIONS=("1.26.2" "2.0.1")

# Loop over all combinations
for VER_CUDA in "${CUDA_VERSIONS[@]}"; do
    for VER_UBUNTU in "${UBUNTU_VERSIONS[@]}"; do
        for VER_NUMPY in "${NUMPY_VERSIONS[@]}"; do
            cd "devel$VER_UBUNTU"

            # Construct image tag including the auto-detected architecture (x86_64 or aarch64)
            image=$IMAGE_URL_BASE/devel-linux-${host_arch}:$VER_UBUNTU-$VER_CUDA-$VER_NUMPY

            echo "Building image: $image"

            # NOTE: The following builds the devel image. The image size can be huge > 50GB.
            # Dockerfile internally handles architecture-specific package downloads.
            docker build --network=host \
                --build-arg "VER_CUDA=$VER_CUDA" \
                --build-arg "VER_UBUNTU=$VER_UBUNTU" \
                --build-arg "VER_NUMPY=$VER_NUMPY" \
                . -t "$image"

            if [[ $do_push == 1 ]]; then
                docker push "$image"
            fi

            cd "$SDIR"
        done
    done
done

echo "Build complete for architecture: $host_arch"
