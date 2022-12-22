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

add_library(cvcuda_legacy STATIC
    custom_crop.cu
    reformat.cu
    resize.cu
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
    CvCudaLegacyHelpers.cpp
    custom_crop.cu
    reformat.cu
    resize.cu
    resize_var_shape.cu
    convert_to.cu
    normalize.cu
    pad_and_stack.cu
    normalize_var_shape.cu
    gamma_contrast_var_shape.cu
    pillow_resize.cu
    pillow_resize_var_shape.cu
)

target_link_libraries(cvcuda_legacy
    PUBLIC
        CUDA::cudart_static
        nvcv_types
        nvcv_util
        cvcuda_headers
)
