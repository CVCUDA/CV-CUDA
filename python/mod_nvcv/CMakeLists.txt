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

set(imgformat_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonImageFormatDefs.inc)
set(imgformat_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_imgformat_list.sh)
set(imgformat_header ${NVCV_TYPES_SOURCE_DIR}/include/nvcv/ImageFormat.h)
add_custom_command(OUTPUT ${imgformat_defs}
    COMMAND ${imgformat_script} "${imgformat_header}" > ${imgformat_defs}
    DEPENDS ${imgformat_script} ${imgformat_header}
)

set(dtype_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonDataTypeDefs.inc)
set(dtype_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_dtype_list.sh)
set(dtype_header ${NVCV_TYPES_SOURCE_DIR}/include/nvcv/DataType.h)
add_custom_command(OUTPUT ${dtype_defs}
    COMMAND ${dtype_script} "${dtype_header}" > ${dtype_defs}
    DEPENDS ${dtype_script} ${dtype_header}
)

nvcv_python_add_module(
    TARGET nvcv_module_python SHARED
    OUTPUT_NAME nvcv
    SOURCES
        ${imgformat_defs}
        ${dtype_defs}
        Main.cpp
        ImageFormat.cpp
        DataType.cpp
        Stream.cpp
        StreamStack.cpp
        Cache.cpp
        Resource.cpp
        Container.cpp
        Tensor.cpp
        Image.cpp
        ImageBatch.cpp
        CudaBuffer.cpp
        Rect.cpp
        Object.cpp
        CAPI.cpp
)

target_include_directories(nvcv_module_python
    PRIVATE
        include
        # for imgformat_defs
        ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(nvcv_module_python
    PRIVATE
        nvcv_types
        nvcv_util_compat
        nvcv_python_common
        CUDA::cudart_static
)

add_library(pynvcv INTERFACE)
target_include_directories(pynvcv
    INTERFACE include
)

set_target_properties(nvcv_module_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python)
