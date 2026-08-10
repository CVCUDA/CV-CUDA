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

string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_LIST 1 CUDA_VERSION_MINOR)
list(GET CUDA_VERSION_LIST 2 CUDA_VERSION_PATCH)

find_package(CUDAToolkit ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} REQUIRED)

# CUDA version requirement:
# - to use gcc-9 (11.4)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.2")
    message(FATAL_ERROR "Minimum CUDA version supported is 12.2")
endif()

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

# Compress kernels to generate smaller executables. NVCC supports compression
# modes starting with CUDA 12.8, so older supported toolkits retain the default.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=--compress-all")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compress-mode=size")
endif()

# Enable device lambdas
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Compile multiple GPU architectures in parallel within each nvcc invocation.
# Cap at 4 threads per nvcc process to avoid oversaturating with outer build parallelism.
set(CVCUDA_NVCC_THREADS 4 CACHE STRING "Max threads per nvcc invocation for multi-arch builds")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${CVCUDA_NVCC_THREADS}")

# see https://developer.nvidia.com/cuda-gpus
option(CVCUDA_AARCH64_JETSON "Build for Jetson Orin platforms only (aarch64)" OFF)

cvcuda_configure_cuda_architecture_policy()
