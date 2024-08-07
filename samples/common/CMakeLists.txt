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

project(cvcuda_samples_common LANGUAGES CXX)

add_library(cvcuda_samples_common SHARED
                                TRTUtils.cpp
			        NvDecoder.cpp)
target_compile_options(cvcuda_samples_common PRIVATE -Wno-deprecated-declarations -Wno-missing-declarations)
target_link_libraries(cvcuda_samples_common nvcv_types cvcuda CUDA::cudart TensorRT::nvinfer CUDA::nvjpeg)

install(TARGETS cvcuda_samples_common
        EXPORT cvcuda_samples_common
        COMPONENT samples
	DESTINATION samples/lib)
