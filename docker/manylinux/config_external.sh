#!/bin/bash -ex

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}

export REGISTRY_MANYLINUX_PREFIX=${REGISTRY_MANYLINUX_PREFIX:-"quay.io/pypa/"}
export REGISTRY_CUDA_PREFIX=${REGISTRY_CUDA_PREFIX:-"nvidia/"}
export REGISTRY_HOST_PREFIX=${REGISTRY_HOST_PREFIX:-""}

export MANYLINUX_VERSION="2014"
export MANYLINUX_BASE_OS="centos7"
export MANYLINUX_IMAGE_TAG="2024.10.26-1"

export ARCHITECTURES=(
    "x86_64"
    "aarch64"
)

export GCC_VERSIONS=(
    "10"
)

export CUDA_VERSIONS=(
    "11.7.1"
    "12.2.0"
)

export TEST_OS_VERSIONS=(
    "ubuntu20.04"
    "ubuntu22.04"
)
