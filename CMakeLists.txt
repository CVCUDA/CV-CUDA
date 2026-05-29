# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# We need to check this variable before starting a CUDA project - otherwise it will appear
# as set, with the default value pointing to the oldest supported architecture (52 as of CUDA 11.8)
if(CMAKE_CUDA_ARCHITECTURES)
    set(USE_CMAKE_CUDA_ARCHITECTURES TRUE)
endif()

project(cvcuda
	LANGUAGES C CXX
	VERSION 0.15.0
	DESCRIPTION "CUDA-accelerated Computer Vision algorithms"
)

# Make sure the cuda host compiler agrees with what we're using,
# unless user overwrites it (at their own risk).
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

enable_language(CUDA)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "")

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
option(BUILD_TESTS "Enable testsuite" ON)
option(BUILD_TESTS_CPP "Build C++ tests" OFF)
option(BUILD_TESTS_WHEELS "Generate test_wheels.sh script" OFF)
option(BUILD_TESTS_PYTHON "Build Python tests" OFF)
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_BENCH "Build benchmark" OFF)
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
if(BUILD_PYTHON)
    include(ConfigPython)
endif()

# Define the build tree ====================

add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

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

if(BUILD_BENCH)
    add_subdirectory(bench)
endif()

# Must be done after build tree is defined
include(ConfigCPack)

# Print build tree configuration ===========

message(STATUS "")
include(PrintConfig)
