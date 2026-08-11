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

# system core -------------------------------------------------

set(CVCUDA_TEST_SOURCES
    TestOpJpegCompressionDistortion.cpp
    TestOpAdjustHue.cpp
    TestOpAdjustSaturation.cpp
    TestOpAdjustSharpness.cpp
    TestOpAdjustContrast.cpp
    TestOpInvert.cpp
    TestOpSolarize.cpp
    TestOpPosterize.cpp
    TestOpAutoContrast.cpp
    TestOpPairwiseMatcher.cpp
    TestOpStack.cpp
    TestOpLabel.cpp
    TestOpHistogramEq.cpp
    TestOpAdvCvtColor.cpp
    TestOpMinMaxLoc.cpp
    TestOpHistogram.cpp
    TestOpMinAreaRect.cpp
    TestOpMinMaxLoc.cpp
    TestOpBrightnessContrast.cpp
    TestOpColorTwist.cpp
    FlipUtils.cpp
    ConvUtils.cpp
    CvtColorUtils.cpp
    ResizeUtils.cpp
    TestUtils.cpp
    TestOpNonMaximumSuppression.cpp
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
    TestOpJointBilateralFilter.cpp
    TestOpWarpPerspective.cpp
    TestOpComposite.cpp
    TestOpChannelReorder.cpp
    TestOpFlip.cpp
    TestOpGammaContrast.cpp
    TestOpPillowResize.cpp
    TestOpThreshold.cpp
    TestOpAdaptiveThreshold.cpp
    TestOpRandomResizedCrop.cpp
    TestOpGaussianNoise.cpp
    GaussianNoiseUtils.cu
    TestOpInpaint.cpp
    TestOpFindHomography.cpp
    TestOpHQResize.cpp
    TestOpOSD.cpp
    TestOpBndBox.cpp
    TestOpBoxBlur.cpp
    TestOpCLAHE.cpp
    TestOperatorWrapperCopy.cpp
)

set(CVCUDA_SMOKE_TEST_SOURCES
    TestOpBndBox_Smoke.cpp
    TestOpBoxBlur_Smoke.cpp
)

# Test files that require C++20 (use floating-point values as template non-type parameters)
# These are incompatible with GCC-10 which doesn't fully support NTTPs for floating-point types
set(CVCUDA_TEST_SOURCES_CPP20
    TestOpResizeCropConvertReformat.cpp
    TestOpRemap.cpp
    TestOpCropFlipNormalizeReformat.cpp
    TestOpSIFT.cpp
)

# Build smoke tests unconditionally - they work on all compilers
add_executable(cvcuda_test_system_smoke ${CVCUDA_SMOKE_TEST_SOURCES})

target_link_libraries(cvcuda_test_system_smoke
    PUBLIC
        cvcuda
        nvcv_test_common_system
)

nvcv_add_test(cvcuda_test_system_smoke cvcuda cpp)

# Include C++20-requiring tests only if compiler supports them
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0))
    list(APPEND CVCUDA_TEST_SOURCES ${CVCUDA_TEST_SOURCES_CPP20})
endif()

add_executable(cvcuda_test_system ${CVCUDA_TEST_SOURCES})

target_link_libraries(cvcuda_test_system
    PUBLIC
        cvcuda
        nvcv_test_common_system
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
    message(STATUS "GCC-${CMAKE_CXX_COMPILER_VERSION}: Building cvcuda_test_system without C++20 NTTP tests.")
endif()

nvcv_add_test(cvcuda_test_system cvcuda cpp)

# header compatibility tests ---------------------------------------------

get_target_property(CVCUDA_SOURCE_DIR cvcuda SOURCE_DIR)

# Gather C headers
file(GLOB_RECURSE CAPI_HEADERS RELATIVE "${CVCUDA_SOURCE_DIR}/include" CONFIGURE_DEPENDS "${CVCUDA_SOURCE_DIR}/include/*.h")
add_header_compat_test(TARGET cvcuda_test_capi_header_compat
                       SOURCE TestAPI.c
                       STANDARD c11
                       DEPENDS cvcuda
                       HEADERS ${CAPI_HEADERS})

# Gather C++ headers
file(GLOB_RECURSE CXXAPI_HEADERS RELATIVE "${CVCUDA_SOURCE_DIR}/include" CONFIGURE_DEPENDS "${CVCUDA_SOURCE_DIR}/include/*.hpp")
# remove optools files, they are c++17
list(FILTER CXXAPI_HEADERS EXCLUDE REGEX "cuda_tools/")

add_header_compat_test(TARGET cvcuda_test_cxxapi_header_compat
                       SOURCE TestAPI.cpp
                       STANDARD c++11
                       DEPENDS cvcuda
                       HEADERS ${CXXAPI_HEADERS})
