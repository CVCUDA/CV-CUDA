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

project(nvcv
        LANGUAGES C CXX
        VERSION 0.10.0
        DESCRIPTION "NVCV is NVIDIA Computer Vision library"
)

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "-beta")

option(NVCV_ENABLE_INSTALL "Enables creation of NVCV installers using cpack" ON)

# Configure build tree ======================

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ConfigVersion)
include(ConfigBuildTree)

# NVCV currently supports only shared build
set(CMAKE_POSITION_INDEPENDENT_CODE on)

add_subdirectory(util)
add_subdirectory(src)
