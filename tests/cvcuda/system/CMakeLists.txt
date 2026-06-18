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

if(ARCH_AARCH64)
    include_directories(
        ${CUOSD_SOURCE_DIR}/aarch64/include
    )
    link_directories(
        ${CUOSD_SOURCE_DIR}/aarch64/lib
    )
else()
    include_directories(
        ${CUOSD_SOURCE_DIR}/x86_64/include
    )
    link_directories(
        ${CUOSD_SOURCE_DIR}/x86_64/lib
    )
endif()

# system core -------------------------------------------------

# Base source files that work with C++17 and don't require libcuosd
set(CVCUDA_TEST_SOURCES
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
)

# Smoke tests that don't require libcuosd - these work on all compilers including GCC-10
set(CVCUDA_SMOKE_TEST_SOURCES
    TestOpBndBox_Smoke.cpp
    TestOpBoxBlur_Smoke.cpp
    TestOpOSD_Smoke.cpp
)

# Test files that require C++20 (use floating-point values as template non-type parameters)
# These are incompatible with GCC-10 which doesn't fully support NTTPs for floating-point types
set(CVCUDA_TEST_SOURCES_CPP20
    TestOpResizeCropConvertReformat.cpp
    TestOpRemap.cpp
    TestOpCropFlipNormalizeReformat.cpp
    TestOpSIFT.cpp
)

# Test files that require libcuosd (OSD library)
# libcuosd has ABI incompatibility with GCC-10, so these tests are excluded for GCC-10
set(CVCUDA_TEST_SOURCES_CUOSD
    TestOpOSD.cpp
    TestOpBndBox.cpp
    TestOpBoxBlur.cpp
    OsdUtils.cu
)

# The smoke tests exercise the OSD/BndBox/BoxBlur operators, which depend on the
# prebuilt CUDA-only cuOSD lib and are scoped out of the ROCm build.
if(NOT USE_HIP)
    # Build smoke tests unconditionally - they work on all compilers
    add_executable(cvcuda_test_system_smoke ${CVCUDA_SMOKE_TEST_SOURCES})

    target_link_libraries(cvcuda_test_system_smoke
        PUBLIC
            cvcuda
            nvcv_test_common_system
            # Note: Does NOT link against cuosd
    )

    nvcv_add_test(cvcuda_test_system_smoke cvcuda cpp)
endif()

# Include C++20-requiring tests and libcuosd tests only if compiler supports them properly
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0))
    list(APPEND CVCUDA_TEST_SOURCES ${CVCUDA_TEST_SOURCES_CPP20})
    if(NOT USE_HIP)
        list(APPEND CVCUDA_TEST_SOURCES ${CVCUDA_TEST_SOURCES_CUOSD})
    endif()
endif()

# Build cvcuda_test_system (works with GCC-10+ but with reduced test coverage for GCC-10)
add_executable(cvcuda_test_system ${CVCUDA_TEST_SOURCES})

# Link against libcuosd only for GCC-11+ (needed for OSD tests)
if(NOT USE_HIP AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0))
    target_link_libraries(cvcuda_test_system
        PUBLIC
            cvcuda
            nvcv_test_common_system
            cuosd
    )
else()
    # GCC-10: Build without libcuosd, excluding OSD and C++20 NTTP tests
    target_link_libraries(cvcuda_test_system
        PUBLIC
            cvcuda
            nvcv_test_common_system
    )
    message(STATUS "GCC-${CMAKE_CXX_COMPILER_VERSION}: Building cvcuda_test_system without libcuosd. OSD tests and C++20 NTTP tests excluded.")
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
