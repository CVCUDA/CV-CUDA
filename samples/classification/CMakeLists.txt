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

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "-Wno-deprecated-enum-enum-conversion")

# tag: Build classification sample
add_executable(nvcv_samples_classification Main.cpp)
target_link_libraries(nvcv_samples_classification nvcv_types cvcuda CUDA::cudart nvcv_samples_common)

target_include_directories(nvcv_samples_classification
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

install(TARGETS nvcv_samples_classification
        EXPORT nvcv_samples_classification
        COMPONENT samples
        DESTINATION samples/bin)
