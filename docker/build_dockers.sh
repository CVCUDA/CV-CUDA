#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -ex

REPO_ROOT=$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)

# Main script used to build the Docker images for the CV-CUDA project
# Usage: ./build_dockers.sh [REGISTRY_PREFIX] [MODE]
# REGISTRY_PREFIX: Optional registry prefix for Docker images (e.g., "myregistry.com/")
#                  If empty or not provided, images will be built locally for native architecture
# MODE: Optional build mode - "multiarch" or "local" (default: "multiarch" if REGISTRY_PREFIX set, "local" otherwise)
#       - "multiarch": Build multi-architecture images (x86_64 + aarch64) and push to registry (requires REGISTRY_PREFIX)
#       - "local": Build for native architecture only and load into local Docker

export VERSION=${VERSION:-14}  # Update version when changing anything in the Dockerfiles
export TEGRA_VERSION=${TEGRA_VERSION:-1} # Update version when changing anything in the Dockerfile.tegra-aarch64-linux.builder

export REGISTRY_PREFIX=${1:-${REGISTRY_PREFIX:-}}
export PYVER=${PYVER:-"py310"}
export MANYLINUX_IMAGE_TAG="2025.10.10-1"
readonly MODE_MULTIARCH="multiarch"
readonly MODE_LOCAL="local"

# Python versions for the multi-Python devel image
export PYTHON_VERSIONS_310_TO_314="3.10 3.11 3.12 3.13 3.14"

# Parse MODE from command line argument, or auto-detect based on REGISTRY_PREFIX
MODE="${2:-}"
if [[ -z "$MODE" ]]; then
    if [[ -n "$REGISTRY_PREFIX" ]]; then
        MODE="$MODE_MULTIARCH"
    else
        MODE="$MODE_LOCAL"
    fi
fi

# Validate MODE
if [[ "$MODE" != "$MODE_MULTIARCH" && "$MODE" != "$MODE_LOCAL" ]]; then
    echo "Error: Unsupported mode '$MODE'. Supported values are: $MODE_MULTIARCH, $MODE_LOCAL" >&2
    exit 1
fi

# Multiarch mode requires a registry
if [[ "$MODE" == "$MODE_MULTIARCH" && -z "$REGISTRY_PREFIX" ]]; then
    echo "Error: $MODE_MULTIARCH mode requires REGISTRY_PREFIX to be set" >&2
    echo "Usage: $0 <REGISTRY_PREFIX> $MODE_MULTIARCH" >&2
    exit 1
fi

echo "Build mode: $MODE"

# Regenerate requirements files from versions.env before building images
echo "Regenerating requirements files from versions.env..."
bash "$REPO_ROOT/generate_requirements.sh"

# Detect native architecture (used for local builds and context directory naming)
DETECTED_ARCH=$(uname -m)
case "$DETECTED_ARCH" in
    x86_64)
        NATIVE_ARCH="x86_64"
        NATIVE_PLATFORM="linux/amd64"
        ;;
    aarch64|arm64)
        NATIVE_ARCH="aarch64"
        NATIVE_PLATFORM="linux/arm64"
        ;;
    *)
        echo "Error: Unsupported detected architecture '$DETECTED_ARCH'" >&2
        exit 1
        ;;
esac
echo "Native architecture: $NATIVE_ARCH"

# Set platform(s) based on mode
if [[ "$MODE" == "$MODE_MULTIARCH" ]]; then
    PLATFORMS="linux/amd64,linux/arm64"
    echo "Building for platforms: $PLATFORMS"
else
    PLATFORMS="$NATIVE_PLATFORM"
    echo "Building for platform: $PLATFORMS (local native only)"
fi
export PLATFORMS

# Create isolated build context
BUILD_CONTEXT_DIR="/tmp/cvcuda_build_$$"
echo "Creating isolated build context at: $BUILD_CONTEXT_DIR"
mkdir -p "$BUILD_CONTEXT_DIR"

# Copy all necessary files to the isolated build context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR"/Dockerfile.* "$BUILD_CONTEXT_DIR/"
# Copy build requirements (docker/ is the source of truth for build dependencies)
cp "$SCRIPT_DIR"/requirements.build.sys_python.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/requirements.build.all_pythons.txt "$BUILD_CONTEXT_DIR/"

# Copy docs requirements (single source of truth for documentation dependencies)
cp "$SCRIPT_DIR"/../docs/requirements.docs.txt "$BUILD_CONTEXT_DIR/"

# Copy test requirements (single source of truth for test dependencies)
cp "$SCRIPT_DIR"/../tests/requirements.tests.common.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../tests/requirements.tests.numpy1.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../tests/requirements.tests.numpy2.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../tests/requirements.tests.cu12.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../tests/requirements.tests.cu12.numpy1.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../tests/requirements.tests.cu13.txt "$BUILD_CONTEXT_DIR/"

# Copy bench requirements (single source of truth for benchmark dependencies)
cp "$SCRIPT_DIR"/../bench/python/requirements.bench.common.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../bench/python/requirements.bench.cu12.txt "$BUILD_CONTEXT_DIR/"
cp "$SCRIPT_DIR"/../bench/python/requirements.bench.cu13.txt "$BUILD_CONTEXT_DIR/"

# Cleanup function to remove build context on exit
cleanup_build_context() {
    echo "Cleaning up isolated build context: $BUILD_CONTEXT_DIR"
    rm -rf "$BUILD_CONTEXT_DIR"
    return $?
}
trap cleanup_build_context EXIT

# Determine push/load strategy based on mode
if [[ "$MODE" == "$MODE_MULTIARCH" ]]; then
    echo "Multiarch mode: pushing images to registry '$REGISTRY_PREFIX'"
    PUSH_OR_LOAD="--push"
else
    echo "Local mode: loading images into local Docker"
    PUSH_OR_LOAD="--load"
fi

# Disable provenance and SBOM attestations to prevent hangs, especially with QEMU emulation
# These features can cause Docker buildx to hang during manifest push phase
ATTESTATION_FLAGS="--provenance=false --sbom=false"

# Set builder name based on mode
if [[ "$MODE" == "$MODE_MULTIARCH" ]]; then
    BUILDER_NAME="cvcuda_multiarch_builder"
else
    BUILDER_NAME="cvcuda_builder_${NATIVE_ARCH}"
fi

# Check if builder already exists
if docker buildx ls | grep -q "^${BUILDER_NAME}[* ]"; then
    echo "Buildx builder '$BUILDER_NAME' already exists, reusing it"
else
    echo "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME"
fi

# Don't use 'docker buildx use' as it sets global state that conflicts with parallel builds
# Instead, we'll use --builder flag in each build command
docker buildx inspect --bootstrap "$BUILDER_NAME"

# Note about QEMU emulation for multiarch builds
if [[ "$MODE" == "$MODE_MULTIARCH" ]]; then
    echo "Note: Multi-arch builds will use QEMU emulation for non-native architectures"
    echo "      This may be slower than native builds"

    # Optimize QEMU performance
    export QEMU_CPU=max
    echo "      QEMU_CPU set to 'max' for better performance"
fi

####### BASE IMAGES #######

# Manylinux2_28 with GCC 10
export MANYLINUX_GCC10="${REGISTRY_PREFIX}manylinux2_28_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${MANYLINUX_GCC10}:v${VERSION}} \
    -t ${MANYLINUX_GCC10} -t ${MANYLINUX_GCC10}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.gcc10.deps" \
    --build-arg "MANYLINUX_IMAGE_TAG=${MANYLINUX_IMAGE_TAG}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

####### BUILDER IMAGES #######
# Manylinux-based, various GCC versions
# Dockerfile: Dockerfile.builder.deps
# CUDA toolkit is copied from official NVIDIA Docker images (multi-arch, no .run installer needed)

# GCC 10, CUDA 12.2
export BUILDER_CUDA_122="${REGISTRY_PREFIX}builder_cu12.2.0_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_122}:v${VERSION}} \
    -t ${BUILDER_CUDA_122} -t ${BUILDER_CUDA_122}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.builder.deps" \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# GCC 10, CUDA 12.5
export BUILDER_CUDA_125="${REGISTRY_PREFIX}builder_cu12.5.0_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_125}:v${VERSION}} \
    -t ${BUILDER_CUDA_125} -t ${BUILDER_CUDA_125}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.builder.deps" \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=nvidia/cuda:12.5.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# GCC 10, CUDA 13.0.1
export BUILDER_CUDA_1301="${REGISTRY_PREFIX}builder_cu13.0.1_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_1301}:v${VERSION}} \
    -t ${BUILDER_CUDA_1301} -t ${BUILDER_CUDA_1301}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.builder.deps" \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# GCC 10, CUDA 13.3.0
export BUILDER_CUDA_1330="${REGISTRY_PREFIX}builder_cu13.3.0_gcc10"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${BUILDER_CUDA_1330}:v${VERSION}} \
    -t ${BUILDER_CUDA_1330} -t ${BUILDER_CUDA_1330}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.builder.deps" \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

####### DEVEL IMAGES #######
# Ubuntu-based, various Python versions


# UBUNTU 22.04, CUDA 12.5, NUMPY 1
# Python 3.13 and 3.14 not supported by numpy 1
export DEVEL_U22_CU125_NUM1="${REGISTRY_PREFIX}devel_u22.04_cu12.5.0_num1"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U22_CU125_NUM1}:v${VERSION}} \
    -t ${DEVEL_U22_CU125_NUM1} -t ${DEVEL_U22_CU125_NUM1}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.devel.deps" \
    --build-arg "BASE=nvidia/cuda:12.5.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=3.10" \
    --build-arg "VER_CUDA=12.5.0" \
    --build-arg "VER_NUMPY_MAJOR=1" \
    --build-arg "TORCH_CUDA_SUFFIX=cu12" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# UBUNTU 22.04, CUDA 12.5, NUMPY 2, ALL PYTHON VERSIONS (3.10-3.14)
export DEVEL_U22_PY310_314_CU125_NUM2="${REGISTRY_PREFIX}devel_u22.04_py310-314_cu12.5.0_num2"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U22_PY310_314_CU125_NUM2}:v${VERSION}} \
    -t ${DEVEL_U22_PY310_314_CU125_NUM2} -t ${DEVEL_U22_PY310_314_CU125_NUM2}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.devel.deps" \
    --build-arg "BASE=nvidia/cuda:12.5.0-devel-ubuntu22.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    --build-arg "VER_CUDA=12.5.0" \
    --build-arg "VER_NUMPY_MAJOR=2" \
    --build-arg "TORCH_CUDA_SUFFIX=cu12" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"


# UBUNTU 26.04, CUDA 13.3.0, NUMPY 2
export DEVEL_U26_CU1330_NUM2="${REGISTRY_PREFIX}devel_u26.04_cu13.3.0_num2"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U26_CU1330_NUM2}:v${VERSION}} \
    -t ${DEVEL_U26_CU1330_NUM2} -t ${DEVEL_U26_CU1330_NUM2}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.devel.deps" \
    --build-arg "BASE=nvidia/cuda:13.3.0-devel-ubuntu26.04" \
    --build-arg "PYTHON_VERSIONS=3.14" \
    --build-arg "VER_CUDA=13.3.0" \
    --build-arg "VER_NUMPY_MAJOR=2" \
    --build-arg "TORCH_CUDA_SUFFIX=cu13" \
    --build-arg "INSTALL_NSIGHT_COMPUTE=1" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"

# UBUNTU 26.04, CUDA 13.3.0, NUMPY 2, ALL PYTHON VERSIONS (3.10-3.14)
export DEVEL_U26_PY310_314_CU1330_NUM2="${REGISTRY_PREFIX}devel_u26.04_py310-314_cu13.3.0_num2"
docker buildx build \
    --builder "$BUILDER_NAME" \
    ${REGISTRY_PREFIX:+--cache-from type=registry,ref=${DEVEL_U26_PY310_314_CU1330_NUM2}:v${VERSION}} \
    -t ${DEVEL_U26_PY310_314_CU1330_NUM2} -t ${DEVEL_U26_PY310_314_CU1330_NUM2}:v${VERSION} \
    -f "$BUILD_CONTEXT_DIR/Dockerfile.devel.deps" \
    --build-arg "BASE=nvidia/cuda:13.3.0-devel-ubuntu26.04" \
    --build-arg "PYTHON_VERSIONS=${PYTHON_VERSIONS_310_TO_314}" \
    --build-arg "VER_CUDA=13.3.0" \
    --build-arg "VER_NUMPY_MAJOR=2" \
    --build-arg "TORCH_CUDA_SUFFIX=cu13" \
    --build-arg "INSTALL_NSIGHT_COMPUTE=1" \
    ${REGISTRY_PREFIX:+--cache-to type=inline} \
    --platform ${PLATFORMS} \
    ${ATTESTATION_FLAGS} \
    ${PUSH_OR_LOAD} \
    "$BUILD_CONTEXT_DIR"


docker buildx stop "$BUILDER_NAME"
docker buildx rm "$BUILDER_NAME"
