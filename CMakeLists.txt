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

cmake_minimum_required(VERSION 3.22)

# We need to check this variable before starting a CUDA project - otherwise it will appear
# as set, with the default value pointing to the oldest supported architecture (52 as of CUDA 11.8)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(USE_CMAKE_CUDA_ARCHITECTURES TRUE)
endif()

project(cvcuda
        LANGUAGES C CXX
        VERSION 0.2.0
        DESCRIPTION "CUDA-accelerated Computer Vision algorithms"
)

# Make sure the cuda host compiler agrees with what we're using,
# unless user overwrites it (at their own risk).
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

enable_language(CUDA)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "-alpha")

# if user didn't set install prefix,
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Allow cv-cuda libraries with different major versions to be
    # installed in parallel
    set(CMAKE_INSTALL_PREFIX "/opt/nvidia/cvcuda${PROJECT_VERSION_MAJOR}" CACHE PATH "where cvcuda will be installed" FORCE)
endif()

# Options to configure the build tree =======
option(BUILD_TESTS "Enable testsuite" OFF)
option(BUILD_PYTHON "Build python bindings" OFF)
option(ENABLE_SANITIZER "Enabled sanitized build" OFF)

# Configure build tree ======================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigVersion)
include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigCUDA)
include(ConfigCCache)
if(BUILD_PYTHON)
    include(ConfigPython)
endif()

# Define the build tree ====================

add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(BUILD_PYTHON)
    include(BuildPython)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

# Must be done after build tree is defined
include(ConfigCPack)

# Print build tree configuration ===========

message(STATUS "")
include(PrintConfig)
