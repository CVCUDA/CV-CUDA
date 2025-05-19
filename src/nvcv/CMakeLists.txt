# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(nvcv
        LANGUAGES C CXX
        VERSION 0.15.0
        DESCRIPTION "NVCV is NVIDIA Computer Vision library"
)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "")

include(CMakeDependentOption)

option(NVCV_BUILD_SHARED_LIBS "Builds NVCV as shared library" ON)

if(NVCV_ENABLE_INSTALL AND NOT NVCV_BUILD_SHARED_LIBS)
    message(FATAL_ERROR "NVCV installers can only be enabled when building NVCV as a shared library")
endif()

# Enables install targets by default if building shared libs. User can disable it.
# If building static libs, forcibly disables install targets.
cmake_dependent_option(NVCV_ENABLE_INSTALL "Enables creation of NVCV installers using cpack" ON NVCV_BUILD_SHARED_LIBS OFF)

# Configure build tree ======================

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ConfigVersion)
include(ConfigBuildTree)

add_subdirectory(util)
add_subdirectory(src)
