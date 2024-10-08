# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# We're creating a shared lib, all projects it depends on must have
# PIC enabled
set(CMAKE_POSITION_INDEPENDENT_CODE on)

if(LTO_ENABLED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION on)
endif()

# Setting NVCV lib to be a shared lib
set(NVCV_BUILD_SHARED_LIBS ON)

add_subdirectory(nvcv)
add_subdirectory(cvcuda)
