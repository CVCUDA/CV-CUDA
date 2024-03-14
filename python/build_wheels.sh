#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Creates the Python self contained wheels

# Usage: build_wheels.sh [build_artifacts_dir] [python_versions]
# Note: This script is automatically called by cmake/make. The proper way to
# build python wheels is to issue the command:
#
# Do not run this script outside of cmake.

set -e  # Stops this script if any one command fails.

if [ "$#" -lt 2 ]; then
    echo "Usage: build_wheels.sh <build_dir> [python_versions,...]"
    exit 1
fi

BUILD_DIR=$(realpath "$1"); shift
PY_VERSIONS=("$@")
LIB_DIR="${BUILD_DIR}/lib"

echo "BUILD_DIR: $BUILD_DIR"
echo "Python Versions: ${PY_VERSIONS[*]}"

for py_version in "${PY_VERSIONS[@]}"
do
    py_version_flat="${py_version//./}"  # Gets the non dotted version string
    echo "Building Python wheels for: Python${py_version}"

    # Step 1: Create a directories to store all wheels related files for this python version
    py_dir="${BUILD_DIR}/python${py_version}"
    wheel_dir="${py_dir}/wheel"
    mkdir -p "${wheel_dir}"
    rm -rf ${wheel_dir:?}/*
    mkdir -p "${wheel_dir}/cvcuda.libs"

    cd "${wheel_dir}"

    # Step 2: Copy necessary .so files under one directory
    # We will copy the target of the linked file and not the symlink only.
    # Also the new file-name of the .so will be the actual so-name present inside the header of the .so
    # This can be retrieved by using patchelf.
    # This allows us to copy .so files without knowing their versions and also making sure they still
    # work after copying.
    # Copy the core .so files first
    for so_file_name in libcvcuda.so libnvcv_types.so
    do
        cp -L "${LIB_DIR}/${so_file_name}" \
            "${wheel_dir}/cvcuda.libs/`patchelf --print-soname "${LIB_DIR}/${so_file_name}"`"
    done

    # Copy the bindings .so files + patch them in their rpath.
    # This allows the bindings to find the core .so files in a directory named cvcuda.libs only.
    for so_file_path in ${LIB_DIR}/python/*.cpython-${py_version_flat}*.so
    do
        so_file_name=$(basename ${so_file_path})
        cp -L "${so_file_path}" \
            "${wheel_dir}/"

        patchelf --force-rpath --set-rpath '$ORIGIN'/cvcuda.libs "${wheel_dir}/${so_file_name}"
    done

    # Step 3: Copy the setup.py corresponding to current python version to our wheels directory.
    cp "${py_dir}/setup.py" "${wheel_dir}"

    # Step 3: Create wheel
    python${py_version} setup.py bdist_wheel --dist-dir="${wheel_dir}"

done
