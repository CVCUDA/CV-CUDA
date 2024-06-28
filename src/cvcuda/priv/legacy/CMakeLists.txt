# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CV_CUDA_PRIV_LEGACY_FILES CvCudaLegacyHelpers.cpp)

set(CV_CUDA_PRIV_LEGACY_OP_FILES
    filter_utils.cu
    custom_crop.cu
    reformat.cu
    resize_var_shape.cu
    convert_to.cu
    normalize.cu
    pad_and_stack.cu
    normalize_var_shape.cu
    copy_make_border.cu
    copy_make_border_var_shape.cu
    center_crop.cu
    rotate.cu
    rotate_var_shape.cu
    filter.cu
    erase.cu
    erase_var_shape.cu
    filter_var_shape.cu
    median_blur.cu
    bilateral_filter.cu
    bilateral_filter_var_shape.cu
    joint_bilateral_filter.cu
    joint_bilateral_filter_var_shape.cu
    median_blur_var_shape.cu
    morphology.cu
    cvt_color.cu
    warp.cu
    warp_var_shape.cu
    cvt_color_var_shape.cu
    composite.cu
    morphology_var_shape.cu
    channel_reorder_var_shape.cu
    flip.cu
    flip_or_copy_var_shape.cu
    composite_var_shape.cu
    custom_crop.cu
    reformat.cu
    resize_var_shape.cu
    convert_to.cu
    normalize.cu
    pad_and_stack.cu
    normalize_var_shape.cu
    gamma_contrast_var_shape.cu
    pillow_resize.cu
    pillow_resize_var_shape.cu
    threshold.cu
    adaptive_threshold.cu
    adaptive_threshold_var_shape.cu
    threshold_var_shape.cu
    threshold_util.cu
    box_blur.cu
    osd.cu
    textbackend/backend.cpp
    textbackend/stb.cpp
    random_resized_crop.cu
    random_resized_crop_var_shape.cu
    gaussian_noise.cu
    gaussian_noise_var_shape.cu
    gaussian_noise_util.cu
    inpaint.cu
    inpaint_var_shape.cu
    inpaint_utils.cu
    calc_hist.cu
    histogram_eq.cu
    histogram_eq_var_shape.cu
    min_area_rect.cu
)

add_compile_definitions(ENABLE_TEXT_BACKEND_STB)

# filter only one that matches the patern (case insensitive), should be set on the global level
# usage:
# set(CV_CUDA_SRC_PATERN medianblur median_blur"
#
# will compile only files relevant to themedian blur op "median_blur.cu;median_blur_var_shape.cu"
if (NOT "${CV_CUDA_SRC_PATERN}" STREQUAL "")
    foreach(PATTERN ${CV_CUDA_SRC_PATERN})
        string(TOLOWER ${PATTERN} PATTERN)
        foreach(FILENAME ${CV_CUDA_PRIV_LEGACY_OP_FILES})
            string(TOLOWER ${FILENAME} FILENAME_LOWERCASE)
            if (${FILENAME_LOWERCASE} MATCHES ${PATTERN})
                list(APPEND CV_CUDA_PRIV_LEGACY_FILES ${FILENAME})
            endif()
        endforeach()
    endforeach()
else()
    list(APPEND CV_CUDA_PRIV_LEGACY_FILES ${CV_CUDA_PRIV_LEGACY_OP_FILES})
endif()

add_library(cvcuda_legacy STATIC
    ${CV_CUDA_PRIV_LEGACY_FILES}
)

target_link_libraries(cvcuda_legacy
    PUBLIC
        CUDA::cudart_static
        nvcv_types
        nvcv_util
        cvcuda_headers
        -lrt
)
