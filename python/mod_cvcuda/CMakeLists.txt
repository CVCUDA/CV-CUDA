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

find_package(cvcuda REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(dlpack CONFIG REQUIRED)

# dlpack's CMake project version is stuck at 0.6 regardless of release,
# so check the actual version from header macros.
get_target_property(_dlpack_inc dlpack::dlpack INTERFACE_INCLUDE_DIRECTORIES)
file(STRINGS "${_dlpack_inc}/dlpack/dlpack.h" _dlpack_major REGEX "#define DLPACK_MAJOR_VERSION")
file(STRINGS "${_dlpack_inc}/dlpack/dlpack.h" _dlpack_minor REGEX "#define DLPACK_MINOR_VERSION")
string(REGEX MATCH "[0-9]+" _dlpack_major "${_dlpack_major}")
string(REGEX MATCH "[0-9]+" _dlpack_minor "${_dlpack_minor}")
if(_dlpack_major LESS 1 OR (_dlpack_major EQUAL 1 AND _dlpack_minor LESS 3))
    message(FATAL_ERROR "DLPack >= 1.3 required, found ${_dlpack_major}.${_dlpack_minor}")
endif()

# Generate NVCV type definitions
set(imgformat_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonImageFormatDefs.inc)
set(imgformat_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_imgformat_list.sh)
set(imgformat_header ${NVCV_TYPES_SOURCE_DIR}/include/nvcv/ImageFormat.h)
add_custom_command(OUTPUT ${imgformat_defs}
    COMMAND ${imgformat_script} "${imgformat_header}" > ${imgformat_defs}
    DEPENDS ${imgformat_script} ${imgformat_header}
)

set(dtype_defs ${CMAKE_CURRENT_BINARY_DIR}/NVCVPythonDataTypeDefs.inc)
set(dtype_script ${CMAKE_CURRENT_SOURCE_DIR}/gen_dtype_list.sh)
set(dtype_header ${NVCV_TYPES_SOURCE_DIR}/include/nvcv/DataType.h)
add_custom_command(OUTPUT ${dtype_defs}
    COMMAND ${dtype_script} "${dtype_header}" > ${dtype_defs}
    DEPENDS ${dtype_script} ${dtype_header}
)

cvcuda_python_add_module(
    TARGET cvcuda_module_python MODULE
    OUTPUT_NAME _cvcuda
    PACKAGE_DIR cvcuda
    SOURCES
        # NVCV core types
        operators/OpJpegCompressionDistortion.cpp
        operators/OpAdjustHue.cpp
        operators/OpAdjustSaturation.cpp
        operators/OpAdjustSharpness.cpp
        operators/OpAdjustContrast.cpp
        operators/OpInvert.cpp
        operators/OpSolarize.cpp
        operators/OpPosterize.cpp
        operators/OpAutoContrast.cpp
        ${imgformat_defs}
        ${dtype_defs}
        nvcv/ImageFormat.cpp
        nvcv/DataType.cpp
        nvcv/Stream.cpp
        nvcv/StreamStack.cpp
        nvcv/Cache.cpp
        nvcv/Resource.cpp
        nvcv/Container.cpp
        nvcv/Tensor.cpp
        nvcv/Image.cpp
        nvcv/ImageBatch.cpp
        nvcv/TensorBatch.cpp
        nvcv/ExternalBuffer.cpp
        nvcv/Rect.cpp
        nvcv/Object.cpp
        nvcv/CAPI.cpp
        nvcv/DLPackUtils.cpp
        nvcv/ColorSpec.cpp
        nvcv/Array.cpp
        nvcv/ThreadScope.cpp
        # CV-CUDA
        Main.cpp
        ChannelManipType.cpp
        PairwiseMatcherType.cpp
        NormType.cpp
        WorkspaceCache.cpp
        LabelType.cpp
        ConnectivityType.cpp
        SIFTFlagType.cpp
        OsdElement.cpp
        operators/OpRemap.cpp
        RemapMapValueType.cpp
        InterpolationType.cpp
        BorderType.cpp
        RoundMode.cpp
        ColorConversionCode.cpp
        MorphologyType.cpp
        ThresholdType.cpp
        AdaptiveThresholdType.cpp
        CvtColorUtil.cpp
        NormType.cpp
        # CV-CUDA operators
        operators/OpResizeCropConvertReformat.cpp
        operators/OpPairwiseMatcher.cpp
        operators/OpStack.cpp
        operators/OpLabel.cpp
        operators/OpHistogramEq.cpp
        operators/OpOSD.cpp
        operators/OpAdvCvtColor.cpp
        operators/OpSIFT.cpp
        operators/OpMinMaxLoc.cpp
        operators/OpHistogram.cpp
        operators/OpMinAreaRect.cpp
        operators/OpBndBox.cpp
        operators/OpBoxBlur.cpp
        operators/OpBrightnessContrast.cpp
        operators/OpColorTwist.cpp
        operators/OpHQResize.cpp
        operators/OpCropFlipNormalizeReformat.cpp
        operators/OpNonMaximumSuppression.cpp
        operators/OpReformat.cpp
        operators/OpResize.cpp
        operators/OpCustomCrop.cpp
        operators/OpNormalize.cpp
        operators/OpConvertTo.cpp
        operators/OpPadAndStack.cpp
        operators/OpCopyMakeBorder.cpp
        operators/OpRotate.cpp
        operators/OpErase.cpp
        operators/OpGaussian.cpp
        operators/OpMedianBlur.cpp
        operators/OpLaplacian.cpp
        operators/OpAverageBlur.cpp
        operators/OpConv2D.cpp
        operators/OpBilateralFilter.cpp
        operators/OpJointBilateralFilter.cpp
        operators/OpCenterCrop.cpp
        operators/OpWarpAffine.cpp
        operators/OpWarpPerspective.cpp
        operators/OpChannelReorder.cpp
        operators/OpMorphology.cpp
        operators/OpFlip.cpp
        operators/OpCvtColor.cpp
        operators/OpComposite.cpp
        operators/OpGammaContrast.cpp
        operators/OpPillowResize.cpp
        operators/OpThreshold.cpp
        operators/OpAdaptiveThreshold.cpp
        operators/OpRandomResizedCrop.cpp
        operators/OpGaussianNoise.cpp
        operators/OpInpaint.cpp
        operators/OpFindHomography.cpp
        operators/OpCLAHE.cpp
)

target_include_directories(cvcuda_module_python
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(cvcuda_module_python
    PRIVATE
        CUDA::cudart_static
        nvcv_types
        nvcv_util_compat
        cvcuda
        cvcuda_util
        cvcuda_python_common
        nvcv_util
        CUDA::cuda_driver
        dlpack::dlpack
        -lrt
)

# use exports file to expose only the symbol dl-loaded by python,
# and nothing else.
target_link_options(cvcuda_module_python
    PRIVATE
        -Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/exports.ldscript
)

set_target_properties(cvcuda_module_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python)
