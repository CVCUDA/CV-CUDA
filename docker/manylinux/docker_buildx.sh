#!/bin/bash -ex

# SPDX-License-Identifier: Apache-2.0

# Ensure failures are caught when commands are piped
set -o pipefail

# Set default version if not provided
export VERSION="${VERSION:-1}"

# Get the directory of the script
SCRIPT_DIR="$(readlink -f $(dirname "$0"))"

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
        --push \
        .
done

# Build CUDA images on manylinux platform
for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
    IMAGE_NAME="${REGISTRY_HOST_PREFIX}cuda${CUDA_VERSION}-${MANYLINUX_BASE_OS}-${ARCH}"
    DOCKERFILE="${SCRIPT_DIR}/Dockerfile.cuda.${MANYLINUX_BASE_OS}.deps"
    FROM_IMAGE_NAME="${REGISTRY_CUDA_PREFIX}cuda:${CUDA_VERSION}-devel-${MANYLINUX_BASE_OS}"

    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref="${IMAGE_NAME}" \
        -t "${IMAGE_NAME}" \
        -t "${IMAGE_NAME}:v${VERSION}" \
        -f "${DOCKERFILE}" \
        --build-arg FROM_IMAGE_NAME="${FROM_IMAGE_NAME}" \
        --platform "${PLATFORM}" \
        --provenance=false \
        --push \
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
            --push \
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
        --push \
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
            --push \
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
            --push \
            .
    done
done

popd >/dev/null
