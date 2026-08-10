#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [[ "$#" -ne 1 ]]; then
    echo "Usage: build_wheels.sh <python_build_dir>"
    exit 1
fi

PYTHON_BUILD_DIR=$(realpath "$1")
BUILD_DIR=$(dirname "${PYTHON_BUILD_DIR}")
WHEEL_DIR="${PYTHON_BUILD_DIR}/dist"
REPAIRED_WHEEL_DIR="${PYTHON_BUILD_DIR}/repaired_wheels"
WHEEL_BUILD_DIR="${PYTHON_BUILD_DIR}/build_wheel"
LIB_DIR="${PYTHON_BUILD_DIR}/cvcuda_cu${CUDA_VERSION_MAJOR}.libs"
SUPPORTED_PYTHONS=("310" "311" "312" "313" "314")

# Option to force universal wheel creation even if not all Python versions are built
FORCE_UNIVERSAL=${FORCE_UNIVERSAL:-false}

detect_platform_tag() {
    if [[ -n "${AUDITWHEEL_PLAT}" ]]; then
        echo "${AUDITWHEEL_PLAT}"
    else
        echo "auto"
    fi
    return 0
}

PLATFORM_TAG=$(detect_platform_tag)
echo "Detected Platform Tag: ${PLATFORM_TAG}"

LIBRARIES=(
    "libcvcuda.so"
    "libnvcv_types.so"
)

mkdir -p "${WHEEL_DIR}" "${REPAIRED_WHEEL_DIR}" "${WHEEL_BUILD_DIR}" "${LIB_DIR}"

# Detect available Python bindings
AVAILABLE_PYTHONS=()
for py_ver in "${SUPPORTED_PYTHONS[@]}"; do
    py_exec="python3.${py_ver:1}"
    if command -v "${py_exec}" &> /dev/null && compgen -G "${PYTHON_BUILD_DIR}/cvcuda/*.cpython-${py_ver}-*.so" > /dev/null; then
        AVAILABLE_PYTHONS+=("cp${py_ver}")
    fi
done
PYTHON_EXECUTABLE=python3

# Print the available Python bindings
echo "Available Python Bindings: ${AVAILABLE_PYTHONS[*]}"

if [[ "${#AVAILABLE_PYTHONS[@]}" -eq 0 ]]; then
    echo "Error: No Python bindings detected." >&2
    exit 1
fi

# Copy and patch shared libraries
echo "Copying and patching shared libraries..."
for lib in "${LIBRARIES[@]}"; do
    src_path="${BUILD_DIR}/lib/${lib}"
    if [[ -f "${src_path}" ]]; then
        cp "${src_path}" "${LIB_DIR}/"
        echo "Copied: ${src_path} -> ${LIB_DIR}/"
        patchelf --force-rpath --set-rpath '$ORIGIN/../cvcuda_cu${CUDA_VERSION_MAJOR}.libs' "${LIB_DIR}/${lib}"
    else
        echo "Warning: Shared library ${src_path} not found. Skipping."
    fi
done

# Create wheel structure
ln -sf "${PYTHON_BUILD_DIR}/setup.py" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/README.md" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/pyproject.toml" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/MANIFEST.in" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/cvcuda" "${WHEEL_BUILD_DIR}/"
ln -sf "${LIB_DIR}" "${WHEEL_BUILD_DIR}/cvcuda_cu${CUDA_VERSION_MAJOR}.libs"

echo "Printing currently installed python packages from v-env: $VIRTUAL_ENV and dir: $(pwd)."
${PYTHON_EXECUTABLE} -m pip list

# Build wheel
echo "Building wheel..."
pushd "${WHEEL_BUILD_DIR}" > /dev/null
${PYTHON_EXECUTABLE} -m build --wheel --outdir="${WHEEL_DIR}" || ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir="${WHEEL_DIR}"

# Modify the wheel's Python and ABI tags for detected versions
# If all supported Python versions are available or FORCE_UNIVERSAL is set, use py3-none for universal compatibility
if [[ "${#AVAILABLE_PYTHONS[@]}" -eq "${#SUPPORTED_PYTHONS[@]}" ]] || [[ "${FORCE_UNIVERSAL}" == "true" ]]; then
    echo "Creating universal py3-none wheel..."
    echo "  Available Python versions: ${AVAILABLE_PYTHONS[*]}"
    echo "  Supported Python versions: ${SUPPORTED_PYTHONS[*]}"
    python_tag="py3"
    abi_tag="none"

    # Verify that we have bindings for the most common Python versions
    if [[ "${FORCE_UNIVERSAL}" == "true" ]] && [[ "${#AVAILABLE_PYTHONS[@]}" -lt 3 ]]; then
        echo "Warning: Creating universal wheel with only ${#AVAILABLE_PYTHONS[@]} Python version(s). This may cause compatibility issues."
    fi
else
    echo "Creating multi-version wheel for available Python versions..."
    echo "  Available: ${AVAILABLE_PYTHONS[*]}"
    python_tag=$(IFS=. ; echo "${AVAILABLE_PYTHONS[*]}")
    abi_tag="${python_tag}"
fi

# Check the auditwheel version and handle "auto" platform tag
auditwheel_version=$(${PYTHON_EXECUTABLE} -m pip list | grep auditwheel | awk '{print $2}')
echo "Auditwheel version: ${auditwheel_version}"

version_check() {
    local version1=$1
    local version2=$2
    local i
    IFS=. read -ra ver1 <<< "$version1"
    IFS=. read -ra ver2 <<< "$version2"

    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=${#ver2[@]}; i<${#ver1[@]}; i++)); do
        ver2[i]=0
    done

    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 0
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 1
        fi
    done
    return 0
}

# Handle "auto" platform tag based on auditwheel version
if [[ "${PLATFORM_TAG}" == "auto" ]]; then
    if ! version_check "${auditwheel_version}" "6.4.0"; then
        echo "Auditwheel version ${auditwheel_version} is below requirement (>= 6.4.0) and PLATFORM_TAG is auto, set PLATFORM_TAG to linux_$(uname -m)"
        PLATFORM_TAG="linux_$(uname -m)"
    else
        echo "Auditwheel version ${auditwheel_version} supports auto platform detection - will auto-detect appropriate manylinux/musllinux tag"
        # Keep PLATFORM_TAG as "auto" for auditwheel >= 6.4.0 to auto-detect the appropriate platform
    fi
fi

# Apply Python and ABI tags to the wheel
# Only set platform-tag if it's not "auto" (auditwheel repair will set it)
for whl in "${WHEEL_DIR}"/*.whl; do
    if [[ "${PLATFORM_TAG}" == "auto" ]]; then
        ${PYTHON_EXECUTABLE} -m wheel tags --remove \
                            --python-tag "${python_tag}" \
                            --abi-tag "${abi_tag}" \
                            "${whl}"
    else
        ${PYTHON_EXECUTABLE} -m wheel tags --remove \
                            --python-tag "${python_tag}" \
                            --abi-tag "${abi_tag}" \
                            --platform-tag "${PLATFORM_TAG}" \
                            "${whl}"
    fi
done
popd > /dev/null

echo "Repairing wheel for compliance..."

for whl in "${WHEEL_DIR}"/*.whl; do
    echo "Auditing wheel: ${whl}"
    ${PYTHON_EXECUTABLE} -m auditwheel show "${whl}"
    ${PYTHON_EXECUTABLE} -m auditwheel repair "${whl}" --plat "${PLATFORM_TAG}" --exclude libcuda.so.1 -w "${REPAIRED_WHEEL_DIR}"
    rm "${whl}"
done

echo "Verifying wheel filenames..."
for repaired_whl in "${REPAIRED_WHEEL_DIR}"/*.whl; do
    repaired_whl_name="$(basename "${repaired_whl}")"
    echo "Wheel: ${repaired_whl_name}"
    IFS='-' read -r dist_name version python_tag abi_tag platform_tag <<< "$(echo "${repaired_whl_name}" | sed 's/\.whl$//')"
    echo "  Distribution Name: ${dist_name}"
    echo "  Version: ${version}"
    echo "  Python Tag: ${python_tag}"
    echo "  ABI Tag: ${abi_tag}"
    echo "  Platform Tag: ${platform_tag}"
done

echo "Repaired wheels are located in: ${REPAIRED_WHEEL_DIR}"
