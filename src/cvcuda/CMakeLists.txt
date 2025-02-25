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

add_subdirectory(util)

# cvcuda private implementation
add_subdirectory(priv)

set(CV_CUDA_LIB_FILES Operator.cpp)

set(CV_CUDA_OP_FILES
    OpOSD.cpp
    OpHistogramEq.cpp
    OpAdvCvtColor.cpp
    OpSIFT.cpp
    OpMinMaxLoc.cpp
    OpHistogram.cpp
    OpMinAreaRect.cpp
    OpBndBox.cpp
    OpBoxBlur.cpp
    OpBrightnessContrast.cpp
    OpRemap.cpp
    OpColorTwist.cpp
    OpCropFlipNormalizeReformat.cpp
    OpHQResize.cpp
    OpNonMaximumSuppression.cpp
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
    OpBilateralFilter.cpp
    OpJointBilateralFilter.cpp
    OpMorphology.cpp
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
    OpLabel.cpp
    OpPairwiseMatcher.cpp
    OpFindHomography.cpp
    OpStack.cpp
    OpResizeCropConvertReformat.cpp
)

# filter only one that matches the patern (case insensitive), should be set on the global level
# usage:
# set(CV_CUDA_SRC_PATERN medianblur median_blur"
#
# will compile only files relevant to themedian blur op "OpMedianBlur.cpp"
if (NOT "${CV_CUDA_SRC_PATERN}" STREQUAL "")
    foreach(PATTERN ${CV_CUDA_SRC_PATERN})
        string(TOLOWER ${PATTERN} PATTERN)
        foreach(FILENAME ${CV_CUDA_OP_FILES})
            string(TOLOWER ${FILENAME} FILENAME_LOWERCASE)
            if (${FILENAME_LOWERCASE} MATCHES ${PATTERN})
                list(APPEND CV_CUDA_LIB_FILES ${FILENAME})
            endif()
        endforeach()
    endforeach()
else()
    list(APPEND CV_CUDA_LIB_FILES ${CV_CUDA_OP_FILES})
endif()

add_library(cvcuda SHARED
    ${CV_CUDA_LIB_FILES}
)

target_link_libraries(cvcuda
    PUBLIC
        CUDA::cudart_static
        nvcv_types
        -lrt
    PRIVATE
        nvcv_util
        nvcv_util_symver
        cvcuda_priv
)

target_include_directories(cvcuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# target used when only public headers are needed
add_library(cvcuda_headers INTERFACE)
target_include_directories(cvcuda_headers INTERFACE include ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(cvcuda_headers INTERFACE nvcv_types_headers)

configure_version(cvcuda CVCUDA cvcuda ${PROJECT_VERSION})
configure_symbol_versioning(cvcuda CVCUDA cvcuda)
setup_dso(cvcuda ${PROJECT_VERSION})

target_compile_definitions(cvcuda PRIVATE -DCVCUDA_EXPORTING=1)

# Installer

install(TARGETS cvcuda
        EXPORT cvcuda
        COMPONENT lib
        LIBRARY NAMELINK_COMPONENT dev)

install(DIRECTORY include/cvcuda
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(cvcuda-config-version.cmake
                                 COMPATIBILITY SameMajorVersion)

if(UNIX)
    install(EXPORT cvcuda
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cvcuda"
        FILE cvcuda-config.cmake
        COMPONENT dev)

    # WAR for https://gitlab.kitware.com/cmake/cmake/-/issues/23563
    install(CODE "set(CVCUDA_CONFIG_PATH \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/cvcuda/cvcuda-config.cmake\")"
        COMPONENT dev)

    install(CODE [=[
        file(READ "${CVCUDA_CONFIG_PATH}" contents)
        string(REPLACE "get_filename_component(_IMPORT_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)"
[[
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" REALPATH)
]]
            contents "${contents}")
        file(WRITE "${CVCUDA_CONFIG_PATH}" "${contents}")
        ]=]
        COMPONENT dev)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cvcuda-config-version.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cvcuda"
        COMPONENT dev)
endif()
