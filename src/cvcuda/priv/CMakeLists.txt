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


set(CV_CUDA_PRIV_FILES IOperator.cpp)

set(CV_CUDA_PRIV_OP_FILES
    ChannelCopy.cu
    OpMerge.cpp
    OpSplit.cpp
    OpTvResize.cu
    OpAdaptiveThreshold.cu
    OpAverageBlur.cu
    OpBilateralFilter.cu
    OpBndBox.cu
    OpConv2D.cu
    OpGaussian.cu
    OpInpaint.cu
    OpJointBilateralFilter.cu
    OpJpegCompressionDistortion.cu
    OpAdjustHue.cu
    OpAdjustSaturation.cu
    OpAdjustSharpness.cu
    OpAdjustContrast.cu
    OpInvert.cu
    OpLaplacian.cu
    OpMedianBlur.cu
    OpOSD.cu
    OpPillowResize.cu
    OpRandomResizedCrop.cu
    OpSolarize.cu
    OpPosterize.cu
    OpAutoContrast.cu
    OpCLAHE.cu
    OpHistogramEq.cu
    OpAdvCvtColor.cu
    OpCvtColor.cu
    OpSIFT.cu
    OpMinMaxLoc.cu
    OpHistogram.cu
    OpMinAreaRect.cu
    OpBoxBlur.cu
    OpBrightnessContrast.cu
    OpRemap.cu
    OpColorTwist.cu
    OpCropFlipNormalizeReformat.cu
    OpHQResize.cu
    OpHQResize2D.cu
    OpHQResize3D.cu
    OpNonMaximumSuppression.cu
    OpReformat.cu
    OpResize.cu
    OpCustomCrop.cu
    OpNormalize.cu
    OpPadAndStack.cu
    OpConvertTo.cu
    OpCopyMakeBorder.cu
    OpCenterCrop.cu
    OpRotate.cu
    OpErase.cu
    OpEraseRegion.cu
    OpMorphology.cu
    OpComposite.cu
    OpChannelReorder.cu
    OpFlip.cu
    OpGammaContrast.cu
    OpThreshold.cu
    OpGaussianNoise.cu
    OpLabel.cu
    OpPairwiseMatcher.cu
    OpStack.cpp
    OpStackKernels.cu
    OpFindHomography.cu
    OpResizeCropConvertReformat.cu
    OpWarpAffine.cu
    OpWarpPerspective.cu
    OsdRenderer.cu
    textbackend/backend.cpp
    textbackend/stb.cpp
)

# filter only one that matches the patern (case insensitive), should be set on the global level
# usage:
# set(CV_CUDA_SRC_PATERN medianblur median_blur"
#
if (NOT "${CV_CUDA_SRC_PATERN}" STREQUAL "")
    foreach(PATTERN ${CV_CUDA_SRC_PATERN})
        string(TOLOWER ${PATTERN} PATTERN)
        foreach(FILENAME ${CV_CUDA_PRIV_OP_FILES})
            string(TOLOWER ${FILENAME} FILENAME_LOWERCASE)
            if (${FILENAME_LOWERCASE} MATCHES ${PATTERN})
                list(APPEND CV_CUDA_PRIV_FILES ${FILENAME})
            endif()
        endforeach()
    endforeach()
else()
    list(APPEND CV_CUDA_PRIV_FILES ${CV_CUDA_PRIV_OP_FILES})
endif()

cvcuda_add_targeted_cuda_architectures_to_sources(
    ARCHITECTURES 86 89
    SOURCES
        OpAdjustContrast.cu
        OpAdvCvtColor.cu
        OpChannelReorder.cu
        OpEraseRegion.cu
        OpHQResize2D.cu
        OpInvert.cu
        OpJpegCompressionDistortion.cu
        OpMinMaxLoc.cu
        OpReformat.cu
        OpResize.cu
        OpRotate.cu
    SELECTED_SOURCES_VAR CV_CUDA_PRIV_FILES
    PARENT_SOURCES_VAR CV_CUDA_PRIV_OP_FILES
)

cvcuda_add_targeted_cuda_architectures_to_sources(
    ARCHITECTURES 86
    SOURCES
        OpGammaContrast.cu
        OpResizeCropConvertReformat.cu
    SELECTED_SOURCES_VAR CV_CUDA_PRIV_FILES
    PARENT_SOURCES_VAR CV_CUDA_PRIV_OP_FILES
)

cvcuda_add_targeted_cuda_architectures_to_sources(
    ARCHITECTURES 89
    SOURCES
        OpBrightnessContrast.cu
        OpFlip.cu
        OpPairwiseMatcher.cu
        OpWarpAffine.cu
        OpWarpPerspective.cu
    SELECTED_SOURCES_VAR CV_CUDA_PRIV_FILES
    PARENT_SOURCES_VAR CV_CUDA_PRIV_OP_FILES
)

add_library(cvcuda_priv STATIC
    ${CV_CUDA_PRIV_FILES}
)

target_link_libraries(cvcuda_priv
    PUBLIC
        nvcv_types
        nvcv_util
        cvcuda_headers
        nvcv_util_sanitizer
        CUDA::cudart_static
	    CUDA::cusolver_static
        CUDA::cublas_static
        CUDA::cublasLt_static
        -lrt
)

target_link_libraries(cvcuda_priv PRIVATE cvcuda_nvtx_config)
# The stb-based text rasterizer under textbackend/ is the OSD operator's glyph backend.
add_compile_definitions(ENABLE_TEXT_BACKEND_STB)
# will compile only files relevant to themedian blur op "OpMedianBlur.cu"
