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

# Ensure failures are caught when commands are piped
set -o pipefail

# Parse command-line options
PUSH_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH_FLAG="--push"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default version if not provided
export VERSION="${VERSION:-1}"

# Get the directory of the script
SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"

# Move to the script directory
pushd "${SCRIPT_DIR}" >/dev/null

# Source configuration files
if ! source "${SCRIPT_DIR}/config_internal.sh"; then
    source "${SCRIPT_DIR}/config_external.sh"
fi

# Initialize variables
BUILDER_NAME="cvcuda_builder"

# Initialize buildx instance
docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1 || docker buildx create --name "${BUILDER_NAME}"
docker buildx use "${BUILDER_NAME}"
docker buildx inspect --bootstrap

# Function to convert an arch to a platform
function arch_to_platform() {
    case "$1" in
        "x86_64")
            echo "linux/amd64"
            ;;
        "aarch64")
            echo "linux/arm64"
            ;;
        *)
            echo "Unknown architecture: $1"
            exit 1
            ;;
    esac
}

for ARCH in "${ARCHITECTURES[@]}"; do
    PLATFORM="$(arch_to_platform "${ARCH}")"
    export DOCKER_DEFAULT_PLATFORM="${PLATFORM}"

    ####### BASE IMAGES #######

    # Build Manylinux images with different GCC versions
    for GCC_VERSION in "${GCC_VERSIONS[@]}"; do
        IMAGE_NAME="${REGISTRY_HOST_PREFIX}manylinux${MANYLINUX_VERSION}-${ARCH}.gcc${GCC_VERSION}"
        DOCKERFILE="${SCRIPT_DIR}/Dockerfile.gcc.manylinux${MANYLINUX_VERSION}.deps"
        FROM_IMAGE_NAME="${REGISTRY_MANYLINUX_PREFIX}manylinux${MANYLINUX_VERSION}_${ARCH}:${MANYLINUX_IMAGE_TAG}"

        docker buildx build \
            --cache-to type=inline \
            --cache-from type=registry,ref="${IMAGE_NAME}" \
            -t "${IMAGE_NAME}" \
            -t "${IMAGE_NAME}:v${VERSION}" \
            -f "${DOCKERFILE}" \
            --build-arg "FROM_IMAGE_NAME=${FROM_IMAGE_NAME}" \
            --build-arg "GCC_VERSION=${GCC_VERSION}" \
            --platform "${PLATFORM}" \
            --provenance=false \
            ${PUSH_FLAG} \
            .
    done

    # Build CUDA images on manylinux platform
    for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
        IMAGE_NAME="${REGISTRY_HOST_PREFIX}cuda${CUDA_VERSION}-${MANYLINUX_BASE_OS}-${ARCH}"
        DOCKERFILE="${SCRIPT_DIR}/Dockerfile.cuda.${MANYLINUX_BASE_OS}.deps"
        FROM_IMAGE_NAME="${REGISTRY_CUDA_PREFIX}cuda:${CUDA_VERSION}-devel-${MANYLINUX_BASE_OS}"
        # A special case for CUDA on aarch64 is that FROM_IMAGE_NAME will need to be from rockylinux8
        if [ "${ARCH}" == "aarch64" ]; then
            FROM_IMAGE_NAME="${REGISTRY_CUDA_PREFIX}cuda:${CUDA_VERSION}-devel-rockylinux8"
        fi

        docker buildx build \
            --no-cache \
            --cache-to type=inline \
            --cache-from type=registry,ref="${IMAGE_NAME}" \
            -t "${IMAGE_NAME}" \
            -t "${IMAGE_NAME}:v${VERSION}" \
            -f "${DOCKERFILE}" \
            --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
            --platform "${PLATFORM}" \
            --provenance=false \
            ${PUSH_FLAG} \
            .
    done

    # Build CUDA images on test OS platforms
    for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
        for OS_VERSION in "${TEST_OS_VERSIONS[@]}"; do
            IMAGE_NAME="${REGISTRY_HOST_PREFIX}cuda${CUDA_VERSION}-${OS_VERSION}-${ARCH}"
            DOCKERFILE="${SCRIPT_DIR}/Dockerfile.cuda.${OS_VERSION}.deps"
            FROM_IMAGE_NAME="${REGISTRY_CUDA_PREFIX}cuda:${CUDA_VERSION}-devel-${OS_VERSION}"

            docker buildx build \
                --cache-to type=inline \
                --cache-from type=registry,ref="${IMAGE_NAME}" \
                -t "${IMAGE_NAME}" \
                -t "${IMAGE_NAME}:v${VERSION}" \
                -f "${DOCKERFILE}" \
                --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
                --platform "${PLATFORM}" \
                --provenance=false \
                ${PUSH_FLAG} \
                .
        done
    done

    # Build base images for building dependencies
    for GCC_VERSION in "${GCC_VERSIONS[@]}"; do
        IMAGE_NAME="${REGISTRY_HOST_PREFIX}cvcuda_deps-${ARCH}.gcc${GCC_VERSION}"
        DOCKERFILE="${SCRIPT_DIR}/Dockerfile.build.manylinux${MANYLINUX_VERSION}.deps"
        FROM_IMAGE_NAME="${REGISTRY_HOST_PREFIX}manylinux${MANYLINUX_VERSION}-${ARCH}.gcc${GCC_VERSION}"

        docker buildx build \
            --cache-to type=inline \
            --cache-from type=registry,ref="${IMAGE_NAME}" \
            -t "${IMAGE_NAME}" \
            -t "${IMAGE_NAME}:v${VERSION}" \
            -f "${DOCKERFILE}" \
            --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
            --build-arg ARCH="${ARCH}" \
            --platform "${PLATFORM}" \
            --provenance=false \
            ${PUSH_FLAG} \
            .
    done

    ####### BUILDER IMAGES #######

    # Generate the builder image over cuda and compiler versions
    for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
        for GCC_VERSION in "${GCC_VERSIONS[@]}"; do
            IMAGE_NAME="${REGISTRY_HOST_PREFIX}builder-cuda${CUDA_VERSION}-gcc${GCC_VERSION}-${ARCH}"
            DOCKERFILE="${SCRIPT_DIR}/Dockerfile.builder.deps"
            FROM_IMAGE_NAME="${REGISTRY_HOST_PREFIX}cvcuda_deps-${ARCH}.gcc${GCC_VERSION}"
            CUDA_IMAGE="${REGISTRY_HOST_PREFIX}cuda${CUDA_VERSION}-${MANYLINUX_BASE_OS}-${ARCH}"

            docker buildx build \
                --cache-to type=inline \
                --cache-from type=registry,ref="${IMAGE_NAME}" \
                -t "${IMAGE_NAME}" \
                -t "${IMAGE_NAME}:v${VERSION}" \
                -f "${DOCKERFILE}" \
                --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
                --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
                --platform "${PLATFORM}" \
                --provenance=false \
                ${PUSH_FLAG} \
                .
        done
    done

    ####### RUNNER IMAGES #######

    # Generate the runner image over cuda and os versions
    for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
        for OS_VERSION in "${TEST_OS_VERSIONS[@]}"; do
            IMAGE_NAME="${REGISTRY_HOST_PREFIX}runner-cuda${CUDA_VERSION}-${OS_VERSION}-${ARCH}"
            DOCKERFILE="${SCRIPT_DIR}/Dockerfile.runner.deps"
            FROM_IMAGE_NAME="${REGISTRY_HOST_PREFIX}cuda${CUDA_VERSION}-${OS_VERSION}-${ARCH}"

            docker buildx build \
                --cache-to type=inline \
                --cache-from type=registry,ref="${IMAGE_NAME}" \
                -t "${IMAGE_NAME}" \
                -t "${IMAGE_NAME}:v${VERSION}" \
                -f "${DOCKERFILE}" \
                --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
                --platform "${PLATFORM}" \
                --provenance=false \
                ${PUSH_FLAG} \
                .
        done
    done
done

popd >/dev/null
