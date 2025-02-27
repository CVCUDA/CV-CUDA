# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(cvcuda REQUIRED)
find_package(CUDAToolkit REQUIRED)

nvcv_python_add_module(
    TARGET cvcuda_module_python MODULE
    OUTPUT_NAME cvcuda
    SOURCES
        Main.cpp
        ChannelManipType.cpp
        OpResizeCropConvertReformat.cpp
        OpPairwiseMatcher.cpp
        PairwiseMatcherType.cpp
        NormType.cpp
        OpStack.cpp
        WorkspaceCache.cpp
        OpLabel.cpp
        LabelType.cpp
        ConnectivityType.cpp
        OpHistogramEq.cpp
        OpOSD.cpp
        OpAdvCvtColor.cpp
        OpSIFT.cpp
        SIFTFlagType.cpp
        OpMinMaxLoc.cpp
        OpHistogram.cpp
        OpMinAreaRect.cpp
        OpBndBox.cpp
        OpBoxBlur.cpp
        OpBrightnessContrast.cpp
        OpColorTwist.cpp
        OpHQResize.cpp
        OsdElement.cpp
        OpRemap.cpp
        RemapMapValueType.cpp
        OpCropFlipNormalizeReformat.cpp
        InterpolationType.cpp
        BorderType.cpp
        ColorConversionCode.cpp
        MorphologyType.cpp
        ThresholdType.cpp
        AdaptiveThresholdType.cpp
        OpNonMaximumSuppression.cpp
        OpReformat.cpp
        OpResize.cpp
        OpCustomCrop.cpp
        OpNormalize.cpp
        OpConvertTo.cpp
        OpPadAndStack.cpp
        OpCopyMakeBorder.cpp
        OpRotate.cpp
        OpErase.cpp
        OpGaussian.cpp
        OpMedianBlur.cpp
        OpLaplacian.cpp
        OpAverageBlur.cpp
        OpConv2D.cpp
        OpBilateralFilter.cpp
        OpJointBilateralFilter.cpp
        OpCenterCrop.cpp
        OpWarpAffine.cpp
        OpWarpPerspective.cpp
        OpChannelReorder.cpp
        OpMorphology.cpp
        OpFlip.cpp
        OpCvtColor.cpp
        OpComposite.cpp
        OpGammaContrast.cpp
        OpPillowResize.cpp
        OpThreshold.cpp
        OpAdaptiveThreshold.cpp
        OpRandomResizedCrop.cpp
        OpGaussianNoise.cpp
        OpInpaint.cpp
        CvtColorUtil.cpp
        OpFindHomography.cpp
        NormType.cpp
)

target_link_libraries(cvcuda_module_python
    PRIVATE
        CUDA::cudart_static
        pynvcv
        nvcv_util_compat
        cvcuda
        cvcuda_util
        nvcv_python_common
        nvcv_util
        cuda
        -lrt
)

# use exports file to expose only the symbol dl-loaded by python,
# and nothing else.
target_link_options(cvcuda_module_python
    PRIVATE
        -Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/exports.ldscript
)

set_target_properties(cvcuda_module_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python)
