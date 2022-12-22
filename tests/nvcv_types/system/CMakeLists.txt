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
    TestPixelType.cpp
    TestAllocator.cpp
    TestRequirements.cpp
    TestImage.cpp
    TestImageBatch.cpp
    TestTensor.cpp
    TestTensorLayout.cpp
    TestTensorLayoutInfo.cpp
    TestTensorShapeInfo.cpp
    TestTensorDataAccess.cpp
    TestExceptions.cpp
    TestConfig.cpp
    TestCAPI.c
)

target_link_libraries(nvcv_test_types_system
    PUBLIC
        nvcv_test_main
        nvcv_test_common_system
        nvcv_types
)

nvcv_add_test(nvcv_test_types_system)

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
