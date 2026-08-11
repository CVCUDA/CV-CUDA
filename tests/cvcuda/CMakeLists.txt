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

# Create build tree

if(BUILD_TESTS_CPP)
    # System tests for cvcuda public API
    add_subdirectory(system)

    # Unit tests for cvcuda utilities and operator internals
    add_subdirectory(unit)
endif()

if(BUILD_TESTS_PYTHON AND BUILD_PYTHON)
    # NVTX injection probe used by the python NVTX marker test
    add_subdirectory(nvtx_probe)

    # System tests for cvcuda python
    add_subdirectory(python)
endif()
