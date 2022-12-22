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

if(TensorRT_FIND_REQUIRED)
    find_package(CUDA REQUIRED)
else()
    find_package(CUDA)
    if(NOT CUDA_FOUND)
        return()
    endif()
endif()

if (TensorRT_FOUND)
    return()
endif()

# for some reason this isn't being set correctly.
# tested with cmake-3.12 with Visual Studio 2017
if(NOT CMAKE_LIBRARY_ARCHITECTURE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(CMAKE_LIBRARY_ARCHITECTURE x32)
    else()
        set(CMAKE_LIBRARY_ARCHITECTURE x64)
    endif()
endif()

if(NOT TensorRT_ROOT AND TRT_ROOT)
    set(TensorRT_ROOT "${TRT_ROOT}")
endif()

set(NV_INFER_VERSION_FILE "NvInferVersion.h")

find_path(TensorRT_INCLUDE_DIR NAMES ${NV_INFER_VERSION_FILE} PATH_SUFFIXES include HINTS "${TensorRT_ROOT}" "${CUDA_TOOLKIT_ROOT_DIR}")

foreach(lib nvinfer nvinfer_plugin nvparsers nvparsers nvonnxparser)
    find_library(TensorRT_${lib}_LIBRARY NAMES ${lib} HINTS "${TensorRT_ROOT}/lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endforeach()

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/${NV_INFER_VERSION_FILE}")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/${NV_INFER_VERSION_FILE}" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/${NV_INFER_VERSION_FILE}" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/${NV_INFER_VERSION_FILE}" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvparsers_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
    add_library(TensorRT::nvinfer INTERFACE IMPORTED GLOBAL)
    target_link_libraries(TensorRT::nvinfer INTERFACE "${TensorRT_nvinfer_LIBRARY}")
    target_include_directories(TensorRT::nvinfer INTERFACE "${TensorRT_INCLUDE_DIR}")

    set(_found_libs nvinfer)

    foreach(lib nvinfer_plugin nvparsers nvonnxparser)
        if(TensorRT_${lib}_LIBRARY)
            add_library(TensorRT::${lib} INTERFACE IMPORTED GLOBAL)
            target_link_libraries(TensorRT::${lib} INTERFACE "${TensorRT_${lib}_LIBRARY}")
            set(_found_libs "${_found_libs} ${lib}")
        endif()
    endforeach()

    if(NOT TensorRT_FIND_QUIETLY)
        message(STATUS "Found TensorRT-${TensorRT_VERSION_STRING} libs: ${_found_libs}")
    endif()
endif()
