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

add_subdirectory(legacy)

add_library(cvcuda_priv STATIC
    IOperator.cpp
    OpReformat.cpp
    OpResize.cpp
    OpCustomCrop.cpp
    OpNormalize.cpp
    OpPadAndStack.cpp
    OpConvertTo.cpp
    OpCopyMakeBorder.cpp
    OpCenterCrop.cpp
    OpRotate.cpp
    OpLaplacian.cpp
    OpGaussian.cpp
    OpErase.cpp
    OpAverageBlur.cpp
    OpConv2D.cpp
    OpMedianBlur.cpp
    OpMorphology.cpp
    OpBilateralFilter.cpp
    OpCvtColor.cpp
    OpWarpAffine.cpp
    OpWarpPerspective.cpp
    OpComposite.cpp
    OpChannelReorder.cpp
    OpFlip.cpp
    OpGammaContrast.cpp
    OpPillowResize.cpp
)

target_link_libraries(cvcuda_priv
    PUBLIC
        nvcv_types
        nvcv_util
        cvcuda_headers
        nvcv_util_sanitizer
        cvcuda_legacy
        CUDA::cudart_static
)
