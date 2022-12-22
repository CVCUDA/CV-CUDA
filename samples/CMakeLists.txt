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
project(nvcv_samples LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(nvcv_types)

set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

set(CMAKE_FOLDER samples)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(ConfigTensorRT)

set(CPPSAMPLES common
	    classification
        cropandresize)

set(PYSAMPLES classification
    segmentation)

foreach(sample ${CPPSAMPLES})
    add_subdirectory(${sample})
endforeach()

install(DIRECTORY assets
    COMPONENT samples
    DESTINATION samples)

install(DIRECTORY scripts
    COMPONENT samples
    DESTINATION samples)

install(FILES "CMakeLists.txt" "ConfigTensorRT.cmake" "README.md"
    COMPONENT samples
    DESTINATION samples)

install(DIRECTORY ${CPPSAMPLES}
    COMPONENT samples
    DESTINATION samples)

install(DIRECTORY ${PYSAMPLES}
    COMPONENT samples
    DESTINATION samples)
