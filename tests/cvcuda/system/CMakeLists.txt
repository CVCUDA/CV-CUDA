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

# system core -------------------------------------------------
add_executable(cvcuda_test_system
    FlipUtils.cpp
    ConvUtils.cpp
    TestCAPI.c
    TestOpReformat.cpp
    TestOpResize.cpp
    TestOpCustomCrop.cpp
    TestOpNormalize.cpp
    TestOpPadAndStack.cpp
    TestOpConvertTo.cpp
    TestOpCopyMakeBorder.cpp
    TestOpCenterCrop.cpp
    TestOpRotate.cpp
    TestOpLaplacian.cpp
    TestOpGaussian.cpp
    TestOpErase.cpp
    TestOpAverageBlur.cpp
    TestOpConv2D.cpp
    TestOpMedianBlur.cpp
    TestOpMorphology.cpp
    TestOpCvtColor.cpp
    TestOpWarpAffine.cpp
    TestOpBilateralFilter.cpp
    TestOpWarpPerspective.cpp
    TestOpComposite.cpp
    TestOpChannelReorder.cpp
    TestOpFlip.cpp
    TestOpGammaContrast.cpp
    TestOpPillowResize.cpp
)

target_link_libraries(cvcuda_test_system
    PUBLIC
        cvcuda
        nvcv_test_common_system
)

nvcv_add_test(cvcuda_test_system)
