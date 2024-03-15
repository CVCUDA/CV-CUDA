# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_FOLDER 3rdparty)

# disable all warnings when compiling objects of 3rdparty
# libraries included here. It *doesn't* affect warnings in public
# header files that are included by cvcuda code. For that, see
# solution employed with nvbench.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# GoogleTest -----------------------------
set(INSTALL_GTEST off)
set(BUILD_GMOCK on)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if(BUILD_TESTS)
    add_subdirectory(googletest)
endif()

# pybind11 -----------------------------
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11" PARENT_SCOPE)

# dlpack -----------------------------
set(DLPACK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dlpack" PARENT_SCOPE)

# cuOSD -----------------------------
set(CUOSD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuOSD" PARENT_SCOPE)

# NVBench --------------------------------
if(BUILD_BENCH)
    set(NVBench_ENABLE_CUPTI off)
    set(BUILD_SHARED_LIBS off)
    add_subdirectory(nvbench)
    # Because nvbench::main is an object library, cmake<=3.20 doesn't treat it
    # like regular libraries, and just creating an cvcuda_nvbench interface
    # library that depends on it doesn't work. We need to create an static
    # library and pull in the objects created by nvbench::main, as per cmake
    # docs.
    add_library(cvcuda_nvbench_main STATIC $<TARGET_OBJECTS:nvbench::main>)
    target_link_libraries(cvcuda_nvbench_main PUBLIC nvbench::nvbench)
    target_include_directories(cvcuda_nvbench_main SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/nvbench)
    add_library(cvcuda::nvbench::main ALIAS cvcuda_nvbench_main)
endif()
