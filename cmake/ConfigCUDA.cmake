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

string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)

find_package(CUDAToolkit ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} REQUIRED)

# CUDA version requirement:
# - to use gcc-11 (11.7)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.7")
    message(FATAL_ERROR "Minimum CUDA version supported is 11.7")
endif()

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

# Compress kernels to generate smaller executables
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=--compress-all")

if(NOT USE_CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")

    # All architectures we build sass for
    list(APPEND CMAKE_CUDA_ARCHITECTURES
         70-real # Volta  - gv100/Tesla
         75-real # Turing - tu10x/GeForce
         80-real # Ampere - ga100/Tesla
         86-real # Ampere - ga10x/GeForce
    )

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
        list(APPEND CMAKE_CUDA_ARCHITECTURES
            89-real # Ada    - ad102/GeForce
            90-real # Hopper - gh100/Tesla
        )
    endif()

    # Required compute capability:
    # * compute_70: fast fp16 support + PTX for forward compatibility
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)

    # We must set the cache to the correct values, or else cmake will write its default there,
    # which is the old architecture supported by nvcc. We don't want that.
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build for" FORCE)
endif()
