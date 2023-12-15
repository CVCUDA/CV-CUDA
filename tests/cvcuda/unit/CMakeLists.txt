# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_executable(cvcuda_test_unit
    TestWorkspaceAllocator.cpp
    TestWorkspaceEstimator.cpp
)

target_compile_definitions(cvcuda_test_unit
    PRIVATE
        -DNVCV_UNIT_TEST=1
)

target_link_libraries(cvcuda_test_unit
    PRIVATE
        nvcv_test_main
        nvcv_util
        nvcv_test_common
        cvcuda_priv
)

nvcv_add_test(cvcuda_test_unit cvcuda)
