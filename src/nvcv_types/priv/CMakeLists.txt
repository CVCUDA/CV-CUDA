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

add_library(nvcv_types_priv STATIC
    Context.cpp
    TLS.cpp
    Status.cpp
    CustomAllocator.cpp
    DefaultAllocator.cpp
    IAllocator.cpp
    Requirements.cpp
    Exception.cpp
    Image.cpp
    ImageBatchVarShape.cpp
    Tensor.cpp
    TensorWrapDataStrided.cpp
    TensorLayout.cpp
    TensorData.cpp
    TensorShape.cpp
    ColorSpec.cpp
    DataLayout.cpp
    ColorFormat.cpp
    DataType.cpp
    ImageFormat.cpp
)

target_include_directories(nvcv_types_priv
    PUBLIC
        ../..
)

target_compile_definitions(nvcv_types_priv PUBLIC -DNVCV_EXPORTING=1)

target_link_libraries(nvcv_types_priv
    PUBLIC
        nvcv_types_headers
        nvcv_util_sanitizer
        CUDA::cudart_static
)
