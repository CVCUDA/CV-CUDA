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

# Create build tree

# System tests for nvcv_types public API
add_subdirectory(system)

# Unit tests for internal functions from nvcv_types
add_subdirectory(unit)

# System tests for nvcv CUDA tools public API
add_subdirectory(cudatools_system)

# Unit tests for nvcv CUDA tools
add_subdirectory(cudatools_unit)

if(BUILD_PYTHON)
    # System tests for nvcv python
    add_subdirectory(python)
endif()

# Test NVCV can be used standalone

# The idea is to copy NVCV root source dir outside of CVCUDA tree prior to build it to make sure it has no hard
# dependencies betweeen NVCV and CVCUDA.  Then the NVCV standalone test is build using external project to ensure
# it is not taking advantage of CVCUDA build.  The installation is disabled on the NVCV standalone build so it does
# not interfere with CVCUDA installation of NVCV.  After NVCV standalone is built, its executable is imported here,
# this is done to allow it to be added as an NVCV test.

include(ExternalProject)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../src/nvcv DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/standalone/)

ExternalProject_Add(
    nvcv_standalone
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/standalone
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/standalone
    INSTALL_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        -DNVCV_DIR=${CMAKE_CURRENT_BINARY_DIR}/standalone/nvcv
        -DNVCV_ENABLE_INSTALL=OFF
        -DEXPOSE_CODE=OFF
        -DWARNINGS_AS_ERRORS=${WARNINGS_AS_ERRORS}
        -DENABLE_TEGRA=${ENABLE_TEGRA}
        -DENABLE_COMPAT_OLD_GLIBC=${ENABLE_COMPAT_OLD_GLIBC}
)

add_executable(nvcv_test_standalone IMPORTED)
set_target_properties(nvcv_test_standalone PROPERTIES
    IMPORTED_LOCATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nvcv_test_standalone
)
add_dependencies(nvcv_test_standalone nvcv_standalone)

nvcv_add_test(${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/nvcv_test_standalone nvcv)
