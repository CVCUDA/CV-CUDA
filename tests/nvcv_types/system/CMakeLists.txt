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

project(nvcv_test_types_system)
set(CMAKE_FOLDER tests)

# system core -------------------------------------------------
add_executable(nvcv_test_types_system
    TestVersion.cpp
    TestStatus.cpp
    TestSize.cpp
    TestColorSpec.cpp
    TestDataLayout.cpp
    TestImageFormat.cpp
    TestDataType.cpp
    TestAllocatorC.cpp
    TestAllocatorCpp.cpp
    TestRequirements.cpp
    TestImage.cpp
    TestImageBatch.cpp
    TestTensor.cpp
    TestTensorLayout.cpp
    TestTensorLayoutInfo.cpp
    TestTensorShapeInfo.cpp
    TestTensorDataAccess.cpp
    TestTensorDataUtils.cpp
    TestExceptions.cpp
    TestConfig.cpp
)

target_link_libraries(nvcv_test_types_system
    PUBLIC
        nvcv_test_main
        nvcv_test_common_system
        nvcv_types
)

nvcv_add_test(nvcv_test_types_system)

# header compatibility tests ---------------------------------------------

get_target_property(NVCV_SOURCE_DIR nvcv_types SOURCE_DIR)

# Gather C headers
file(GLOB_RECURSE CAPI_HEADERS RELATIVE "${NVCV_SOURCE_DIR}/include" CONFIGURE_DEPENDS "${NVCV_SOURCE_DIR}/include/*.h")
add_header_compat_test(TARGET nvcv_test_capi_header_compat
                       SOURCE TestAPI.c
                       STANDARD c11
                       DEPENDS nvcv_types
                       HEADERS ${CAPI_HEADERS})

# Gather C++ headers
file(GLOB_RECURSE CXXAPI_HEADERS RELATIVE "${NVCV_SOURCE_DIR}/include" CONFIGURE_DEPENDS "${NVCV_SOURCE_DIR}/include/*.hpp")
# remove optools files, they are c++17
list(FILTER CXXAPI_HEADERS EXCLUDE REGEX "nvcv/cuda/")

add_header_compat_test(TARGET nvcv_test_cxxapi_header_compat
                       SOURCE TestAPI.cpp
                       STANDARD c++11
                       DEPENDS nvcv_types
                       HEADERS ${CXXAPI_HEADERS})

# system core version commit ----------------------------------

# Separate test for versioning API so that only one target depends on
# vpi/detail/VersionUtil.h, which gets updated whenever the commit changes.
# This avoids spurious recompilation of all projects.
add_executable(nvcv_test_types_system_version_commit
    TestVersionCommit.cpp
)

target_link_libraries(nvcv_test_types_system_version_commit
    PRIVATE
        nvcv_test_main
        nvcv_test_common
        nvcv_types
)

nvcv_add_test(nvcv_test_types_system_version_commit)
