# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(nvcv_test_standalone)

set(CMAKE_CXX_STANDARD 17)

enable_testing()

find_package(GTest REQUIRED)

if(NOT NVCV_DIR)
    message(FATAL_ERROR "NVCV_DIR is empty!  Path to NVCV directory must be given.")
endif()

add_subdirectory(${NVCV_DIR} nvcv_dir)

if(NOT NVCV_EXE)
    message(FATAL_ERROR "NVCV_EXE is empty!  NVCV executable must be given.")
endif()

add_executable(${NVCV_EXE}
    TestNVCVStandalone.cpp
)

target_link_libraries(${NVCV_EXE}
    PUBLIC
        nvcv_types
        GTest::gtest_main
)
