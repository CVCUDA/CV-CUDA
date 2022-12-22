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

# Create build tree

# System tests for nvcv_types public API
add_subdirectory(system)

# Unit tests for internal functions from nvcv_types
add_subdirectory(unit)

# System tests for nvcv CUDA tools public API
add_subdirectory(cudatools_system)

# Unit tests for nvcv CUDA tools
add_subdirectory(cudatools_unit)

if(BUILD_PYTHON)
    # System tests for nvcv python
    add_subdirectory(python)
endif()
