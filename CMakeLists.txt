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
        VERSION 0.16.0
        DESCRIPTION "CUDA-accelerated Computer Vision algorithms"
)

# Make sure the cuda host compiler agrees with what we're using,
# unless user overwrites it (at their own risk).
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

# Build the GPU kernels for AMD via HIP/ROCm instead of CUDA. Additive: the
# NVIDIA path below is unchanged when USE_HIP is OFF.
option(USE_HIP "Build the GPU code with HIP for AMD GPUs (ROCm)" OFF)

if(USE_HIP)
    # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects
    # the host GPU(s) via rocm_agent_enumerator and errors if none is found (a
    # no-GPU build host must then set the arch explicitly).
    enable_language(HIP)

    # Host .cpp files also use the CUDA runtime/library spellings (cudaStream_t,
    # cudaMalloc, ...); steer every C/C++/HIP TU through the ROCm compat layer.
    add_compile_definitions(USE_HIP __HIP_PLATFORM_AMD__)
    set(CVCUDA_HIP_COMPAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip")
    include_directories(BEFORE "${CVCUDA_HIP_COMPAT_DIR}")
    # Force-include the compat header on every C++/HIP TU so the cuda*->hip*
    # aliases and the 64-bit warp mask precede any include, regardless of file
    # order. C TUs do not use the CUDA runtime, so they are left alone.
    set(_cvcuda_force_include "-include ${CVCUDA_HIP_COMPAT_DIR}/CvCudaHipCompat.h")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cvcuda_force_include}")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${_cvcuda_force_include}")

    # clang (HIP) defaults to -ffp-contract=fast, which forms FMAs across separate
    # statements/expressions; nvcc only contracts within a single expression
    # (--fmad=true). That extra contraction makes HIP float results drift by ~1 ULP
    # from the CUDA build and the CPU gold references (e.g. the bicubic weight chain
    # in InterpolationWrap), failing the bit-exact gtests. Pin contraction to the
    # CUDA/host semantics so float math is reproducible across the two toolchains.
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffp-contract=on")

    find_package(hip REQUIRED)
    find_package(hipcub REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipsolver REQUIRED)

    # The compat header is force-included on every C++/HIP TU and pulls in the
    # HIP runtime, so the ROCm headers must be on the include path globally (not
    # only on targets that link hip::host).
    get_target_property(_hip_inc hip::host INTERFACE_INCLUDE_DIRECTORIES)
    if(_hip_inc)
        include_directories(SYSTEM ${_hip_inc})
    else()
        include_directories(SYSTEM "/opt/rocm/include")
    endif()

    # Map the CUDA::* link targets the build uses onto their HIP equivalents so
    # the per-target target_link_libraries() lines stay untouched.
    foreach(_cudalib cudart_static cublas_static cublasLt_static cusolver_static cuda_driver)
        if(NOT TARGET CUDA::${_cudalib})
            add_library(CUDA::${_cudalib} INTERFACE IMPORTED)
            target_link_libraries(CUDA::${_cudalib} INTERFACE hip::host)
        endif()
    endforeach()
    target_link_libraries(CUDA::cusolver_static INTERFACE roc::hipsolver)
    target_link_libraries(CUDA::cublas_static   INTERFACE roc::hipblas)

    # Retag every .cu source LANGUAGE HIP without editing each per-target
    # CMakeLists. add_library/add_executable are overridden at top scope to
    # forward to the real command then flip the target's .cu files to HIP and
    # set HIP_ARCHITECTURES. The NVIDIA build never defines
    # these macros, so it is a pure passthrough.
    macro(_cvcuda_hipify_target _tgt)
        get_target_property(_t_type ${_tgt} TYPE)
        if(NOT _t_type STREQUAL "INTERFACE_LIBRARY")
            get_target_property(_t_srcs ${_tgt} SOURCES)
            if(_t_srcs)
                foreach(_s ${_t_srcs})
                    if(_s MATCHES "\\.cu$")
                        set_source_files_properties(${_s} PROPERTIES LANGUAGE HIP)
                    endif()
                endforeach()
            endif()
            set_target_properties(${_tgt} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        endif()
    endmacro()

    macro(add_library _tgt)
        _add_library(${_tgt} ${ARGN})
        set(_args "${ARGN}")
        if(NOT "ALIAS" IN_LIST _args AND NOT "IMPORTED" IN_LIST _args AND NOT "INTERFACE" IN_LIST _args)
            _cvcuda_hipify_target(${_tgt})
        endif()
    endmacro()

    macro(add_executable _tgt)
        _add_executable(${_tgt} ${ARGN})
        set(_args "${ARGN}")
        if(NOT "ALIAS" IN_LIST _args AND NOT "IMPORTED" IN_LIST _args)
            _cvcuda_hipify_target(${_tgt})
        endif()
    endmacro()
else()
    enable_language(CUDA)
endif()

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
