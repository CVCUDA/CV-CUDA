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

add_subdirectory(legacy)

set(CV_CUDA_PRIV_FILES IOperator.cpp)

set(CV_CUDA_PRIV_OP_FILES
    OpOSD.cpp
    OpHistogramEq.cpp
    OpAdvCvtColor.cu
    OpSIFT.cu
    OpMinMaxLoc.cu
    OpHistogram.cpp
    OpMinAreaRect.cpp
    OpBndBox.cpp
    OpBoxBlur.cpp
    OpBrightnessContrast.cu
    OpRemap.cu
    OpColorTwist.cu
    OpCropFlipNormalizeReformat.cu
    OpHQResize.cu
    OpNonMaximumSuppression.cu
    OpReformat.cpp
    OpResize.cpp
    OpResize.cu
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
    OpJointBilateralFilter.cpp
    OpCvtColor.cpp
    OpWarpAffine.cpp
    OpWarpPerspective.cpp
    OpComposite.cpp
    OpChannelReorder.cpp
    OpFlip.cpp
    OpGammaContrast.cpp
    OpPillowResize.cpp
    OpThreshold.cpp
    OpAdaptiveThreshold.cpp
    OpRandomResizedCrop.cpp
    OpGaussianNoise.cpp
    OpInpaint.cpp
    OpLabel.cu
    OpPairwiseMatcher.cu
    OpStack.cpp
    OpFindHomography.cu
    OpResizeCropConvertReformat.cu
)

# filter only one that matches the patern (case insensitive), should be set on the global level
# usage:
# set(CV_CUDA_SRC_PATERN medianblur median_blur"
#
# will compile only files relevant to themedian blur op "OpMedianBlur.cpp"
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

add_library(cvcuda_priv STATIC
    ${CV_CUDA_PRIV_FILES}
)

target_link_libraries(cvcuda_priv
    PUBLIC
        nvcv_types
        nvcv_util
        cvcuda_headers
        nvcv_util_sanitizer
        cvcuda_legacy
        CUDA::cudart_static
	    CUDA::cusolver_static
        CUDA::cublas_static
        CUDA::cublasLt_static
        -lrt
)
