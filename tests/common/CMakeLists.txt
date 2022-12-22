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

# nvcv_test_common ================================
# Utilities used in unit and system tests. They don't
# call the public API functions.

add_library(nvcv_test_common STATIC
    Printers.cpp
    TensorDataUtils.cpp
)

target_include_directories(nvcv_test_common
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(nvcv_test_common
    INTERFACE
        -DNVCV_EXPOSE_CODE=1
)

target_link_libraries(nvcv_test_common
    INTERFACE
        nvcv_util
    PUBLIC
        nvcv_types_headers
        nvcv_util
        nvcv_util_compat
)

# nvcv_test_common_system ==============================
# Utilities for system tests, they call the public API functions.

add_library(nvcv_test_common_system STATIC
    ObjectBag.cpp
    BorderUtils.cpp
)

target_include_directories(nvcv_test_common_system
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(nvcv_test_common_system
    PUBLIC
        nvcv_types
        nvcv_test_common
        cvcuda_headers
)
