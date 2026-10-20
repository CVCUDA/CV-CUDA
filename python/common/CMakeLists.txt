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

find_package(cvcuda REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(cvcuda_python_common STATIC
    Assert.cpp
    CheckError.cpp
    PyUtil.cpp
    String.cpp
)

if(NB_FREE_THREADED)
    target_compile_definitions(cvcuda_python_common PUBLIC NB_FREE_THREADED)
endif()

target_include_directories(cvcuda_python_common
    PUBLIC
        ..
        ${NB_DIR}/include
)

target_link_libraries(cvcuda_python_common
    PUBLIC
        nvcv_types
        nvcv_util
        cvcuda
        CUDA::cudart_static
        Python::Module
)
