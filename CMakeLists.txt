# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20.1)

project(cvcuda
        LANGUAGES C CXX
        VERSION 0.17.0
        DESCRIPTION "CUDA-accelerated Computer Vision algorithms"
)

# CUDA initializes CMAKE_CUDA_ARCHITECTURES while enabling the language. Record
# its provenance after project() loads any toolchain file, but before enabling
# CUDA, so toolchain/cache values (including false values such as OFF) and
# CUDAARCHS are distinguishable from CV-CUDA's generated defaults.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDAArchitecturePolicy.cmake")
cvcuda_detect_cuda_architecture_source()

# Make sure the cuda host compiler agrees with what we're using,
# unless user overwrites it (at their own risk).
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

enable_language(CUDA)

# if user didn't set install prefix,
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Allow cv-cuda libraries with different major versions to be
    # installed in parallel
    set(CMAKE_INSTALL_PREFIX "/opt/nvidia/cvcuda${PROJECT_VERSION_MAJOR}" CACHE PATH "where cvcuda will be installed" FORCE)
endif()

# Save command-line values before option() resets them
set(_BUILD_TESTS_CPP_CMDLINE ${BUILD_TESTS_CPP})
set(_BUILD_TESTS_WHEELS_CMDLINE ${BUILD_TESTS_WHEELS})
set(_BUILD_TESTS_PYTHON_CMDLINE ${BUILD_TESTS_PYTHON})

# Options to configure the build tree =======
option(BUILD_LIB "Build cvcuda + nvcv_types libraries (and their consumers: Python bindings, tests, docs). When OFF, downstream sub-projects (bench, samples) build against an installed cvcuda-dev via find_package." ON)
option(BUILD_TESTS "Enable testsuite" ON)
option(BUILD_TESTS_CPP "Build C++ tests" OFF)
option(BUILD_TESTS_WHEELS "Generate test_wheels.sh script" OFF)
option(BUILD_TESTS_PYTHON "Build Python tests" OFF)
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_PYTHON_WHEEL "Package Python bindings into a wheel" ON)
option(BUILD_BENCH "Build benchmark" OFF)
option(BUILD_SAMPLES "Build samples" OFF)
option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_SANITIZER "Enabled sanitized build" OFF)

# BUILD_TESTS enables all test sub-options by default, but respects explicit overrides
# Normalize command-line values to uppercase for proper boolean comparison
string(TOUPPER "${_BUILD_TESTS_CPP_CMDLINE}" _BUILD_TESTS_CPP_CMDLINE_UPPER)
string(TOUPPER "${_BUILD_TESTS_WHEELS_CMDLINE}" _BUILD_TESTS_WHEELS_CMDLINE_UPPER)
string(TOUPPER "${_BUILD_TESTS_PYTHON_CMDLINE}" _BUILD_TESTS_PYTHON_CMDLINE_UPPER)

# Define patterns for CMake boolean values (supports both numeric and string forms)
set(_CMAKE_BOOL_TRUE_PATTERN "^(1|ON|YES|TRUE|Y)$")
set(_CMAKE_BOOL_FALSE_PATTERN "^(0|OFF|NO|FALSE|N)$")

if(BUILD_TESTS)
    # Enable sub-options unless explicitly set to a false value on command line
    if(NOT _BUILD_TESTS_CPP_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_FALSE_PATTERN}")
        set(BUILD_TESTS_CPP ON CACHE BOOL "Build C++ tests (enabled by BUILD_TESTS)" FORCE)
    endif()
    if(NOT _BUILD_TESTS_WHEELS_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_FALSE_PATTERN}")
        set(BUILD_TESTS_WHEELS ON CACHE BOOL "Generate test_wheels.sh (enabled by BUILD_TESTS)" FORCE)
    endif()
    if(NOT _BUILD_TESTS_PYTHON_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_FALSE_PATTERN}")
        set(BUILD_TESTS_PYTHON ON CACHE BOOL "Build Python tests (enabled by BUILD_TESTS)" FORCE)
    endif()
else()
    # If BUILD_TESTS is OFF, disable all test sub-options unless explicitly set to a true value
    if(_BUILD_TESTS_CPP_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_TRUE_PATTERN}")
        set(BUILD_TESTS_CPP ON CACHE BOOL "Build C++ tests (explicitly enabled)" FORCE)
    else()
        set(BUILD_TESTS_CPP OFF CACHE BOOL "Build C++ tests (disabled by BUILD_TESTS=OFF)" FORCE)
    endif()
    if(_BUILD_TESTS_WHEELS_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_TRUE_PATTERN}")
        set(BUILD_TESTS_WHEELS ON CACHE BOOL "Generate test_wheels.sh (explicitly enabled)" FORCE)
    else()
        set(BUILD_TESTS_WHEELS OFF CACHE BOOL "Generate test_wheels.sh (disabled by BUILD_TESTS=OFF)" FORCE)
    endif()
    if(_BUILD_TESTS_PYTHON_CMDLINE_UPPER MATCHES "${_CMAKE_BOOL_TRUE_PATTERN}")
        set(BUILD_TESTS_PYTHON ON CACHE BOOL "Build Python tests (explicitly enabled)" FORCE)
    else()
        set(BUILD_TESTS_PYTHON OFF CACHE BOOL "Build Python tests (disabled by BUILD_TESTS=OFF)" FORCE)
    endif()
endif()

# Configure build tree ======================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigVersion)
include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigCUDA)
include(ConfigCCache)

add_library(cvcuda_nvtx_config INTERFACE)
find_path(CVCUDA_NVTX_INCLUDE_DIR
    NAMES nvtx3/nvToolsExt.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_TARGET_DIR}/include
    NO_DEFAULT_PATH
)
if(NOT CVCUDA_NVTX_INCLUDE_DIR)
    message(FATAL_ERROR "nvtx3/nvToolsExt.h not found in the CUDA Toolkit include path")
endif()
target_include_directories(cvcuda_nvtx_config INTERFACE ${CVCUDA_NVTX_INCLUDE_DIR})

if(BUILD_LIB AND BUILD_PYTHON)
    # ConfigPython generates Find*.cmake files that reference the in-tree
    # nvcv_types + cvcuda targets via $<TARGET_FILE:...>. Only meaningful when
    # we're actually building the lib in this tree; with BUILD_LIB=OFF, those
    # targets don't exist and the generator expressions error out.
    include(ConfigPython)
endif()

# Define the build tree ====================

# In-tree library + everything that links against it directly. Skipping this
# block (BUILD_LIB=OFF) lets a tree configure for downstream-only builds —
# e.g. a CI test pod that compiles bench against an installed cvcuda-dev
# without rebuilding the lib itself.
if(BUILD_LIB)
    add_subdirectory(src)

    if(BUILD_PYTHON)
        include(BuildPython)
    endif()

    if(BUILD_TESTS_CPP OR BUILD_TESTS_WHEELS OR BUILD_TESTS_PYTHON)
        add_subdirectory(tests)
    endif()

    if(BUILD_DOCS)
        add_subdirectory(docs)
    endif()
endif()

# Bench + samples are downstream consumers — they work whether the lib was
# built in this tree (BUILD_LIB=ON) or imported from an installed cvcuda-dev
# (BUILD_LIB=OFF, see bench/cpp/CMakeLists.txt for the find_package fallback).
if(BUILD_BENCH)
    find_package(nvbench REQUIRED)
    add_library(cvcuda::nvbench::main ALIAS nvbench::main)
    add_subdirectory(bench)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

# Must be done after build tree is defined
if(BUILD_LIB)
    include(ConfigCPack)
endif()

# Print build tree configuration ===========

message(STATUS "")
include(PrintConfig)
