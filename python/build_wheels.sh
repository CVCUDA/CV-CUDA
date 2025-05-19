#!/bin/bash -e

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

if [ "$#" -ne 1 ]; then
    echo "Usage: build_wheels.sh <python_build_dir>"
    exit 1
fi

PYTHON_BUILD_DIR=$(realpath "$1")
BUILD_DIR=$(dirname "${PYTHON_BUILD_DIR}")
WHEEL_DIR="${PYTHON_BUILD_DIR}/dist"
REPAIRED_WHEEL_DIR="${PYTHON_BUILD_DIR}/repaired_wheels"
WHEEL_BUILD_DIR="${PYTHON_BUILD_DIR}/build_wheel"
LIB_DIR="${PYTHON_BUILD_DIR}/cvcuda_cu${CUDA_VERSION_MAJOR}.libs"
SUPPORTED_PYTHONS=("38" "39" "310" "311" "312" "313")

detect_platform_tag() {
    if [ -n "${AUDITWHEEL_PLAT}" ]; then
        echo "${AUDITWHEEL_PLAT}"
    else
        echo "linux_$(uname -m)"
    fi
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
PYTHON_EXECUTABLES=()
for py_ver in "${SUPPORTED_PYTHONS[@]}"; do
    py_exec="python3.${py_ver:1}"
    if command -v "${py_exec}" &> /dev/null; then
        if compgen -G "${PYTHON_BUILD_DIR}/cvcuda/_bindings/cvcuda.cpython-${py_ver}-*.so" > /dev/null; then
            AVAILABLE_PYTHONS+=("cp${py_ver}")
            PYTHON_EXECUTABLES+=("${py_exec}")
        fi
    fi
done
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLES[0]}"

# Print the available Python bindings
echo "Available Python Bindings: ${AVAILABLE_PYTHONS[*]}"

if [ "${#AVAILABLE_PYTHONS[@]}" -eq 0 ]; then
    echo "Error: No Python bindings detected."
    exit 1
fi

# Copy and patch shared libraries
echo "Copying and patching shared libraries..."
for lib in "${LIBRARIES[@]}"; do
    src_path="${BUILD_DIR}/lib/${lib}"
    if [ -f "${src_path}" ]; then
        cp "${src_path}" "${LIB_DIR}/"
        echo "Copied: ${src_path} -> ${LIB_DIR}/"
        patchelf --force-rpath --set-rpath '$ORIGIN/../cvcuda_cu${CUDA_VERSION_MAJOR}.libs' "${LIB_DIR}/${lib}"
    else
        echo "Warning: Shared library ${src_path} not found. Skipping."
    fi
done

# Create wheel structure
ln -sf "${PYTHON_BUILD_DIR}/setup.py" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/cvcuda" "${WHEEL_BUILD_DIR}/"
ln -sf "${PYTHON_BUILD_DIR}/nvcv" "${WHEEL_BUILD_DIR}/"
ln -sf "${LIB_DIR}" "${WHEEL_BUILD_DIR}/cvcuda_cu${CUDA_VERSION_MAJOR}.libs"

echo "Printing currently installed python packages from v-env: $VIRTUAL_ENV and dir: `pwd`."
${PYTHON_EXECUTABLE} -m pip list

# Build wheel
echo "Building wheel..."
pushd "${WHEEL_BUILD_DIR}" > /dev/null
${PYTHON_EXECUTABLE} -m build --wheel --outdir="${WHEEL_DIR}" || ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir="${WHEEL_DIR}"

# Modify the wheel's Python and ABI tags for detected versions
# Ensuring the tag is propagated to the wheel
${PYTHON_EXECUTABLE} -m pip install --upgrade wheel
python_tag=$(IFS=. ; echo "${AVAILABLE_PYTHONS[*]}")
for whl in "${WHEEL_DIR}"/*.whl; do
    ${PYTHON_EXECUTABLE} -m wheel tags --remove \
                        --python-tag "${python_tag}" \
                        --abi-tag "${python_tag}" \
                        --platform-tag "${PLATFORM_TAG}" \
                        "${whl}"
done
popd > /dev/null

echo "Repairing wheel for compliance..."
${PYTHON_EXECUTABLE} -m pip install --upgrade auditwheel
for whl in "${WHEEL_DIR}"/*.whl; do
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
