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

project(nvcv_test_cudatools_system)
set(CMAKE_FOLDER tests)

add_executable(nvcv_test_cudatools_system
    TestAtomics.cpp
    DeviceAtomics.cu
    TestInterpolationVarShapeWrap.cpp
    DeviceInterpolationVarShapeWrap.cu
    TestInterpolationWrap.cpp
    DeviceInterpolationWrap.cu
    TestFullTensorWrap.cpp
    DeviceFullTensorWrap.cu
    TestBorderVarShapeWrap.cpp
    DeviceBorderVarShapeWrap.cu
    TestImageBatchVarShapeWrap.cpp
    DeviceImageBatchVarShapeWrap.cu
    TestBorderWrap.cpp
    DeviceBorderWrap.cu
    TestLinAlg.cpp
    TestTensorWrap.cpp
    DeviceTensorWrap.cu
    TestSaturateCast.cpp
    DeviceSaturateCast.cu
    TestRangeCast.cpp
    TestMathWrappers.cpp
    DeviceMathWrappers.cu
    TestMathOps.cpp
    TestStaticCast.cpp
    TestDropCast.cpp
    TestTypeTraits.cpp
    TestMetaprogramming.cpp
    TestArrayWrap.cpp
    TestTensorBatchWrap.cpp
    DeviceTensorBatchWrap.cu
)

target_link_libraries(nvcv_test_cudatools_system
    PUBLIC
        nvcv_types
        nvcv_util
        nvcv_test_common_system
)

target_include_directories(nvcv_test_cudatools_system
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

nvcv_add_test(nvcv_test_cudatools_system nvcv)
