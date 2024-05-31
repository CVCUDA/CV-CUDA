/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file OpResizeCropConvertReformat.h
 *
 * @brief Defines functions that fuses resize, crop, data type conversion, channel manipulation, and layout reformat operations to optimize pipelines.
 * @defgroup NVCV_C_ALGORITHM__RESIZE_CROP Resize Crop
 * @{
 */

#ifndef CVCUDA__RESIZE_CROP_H
#define CVCUDA__RESIZE_CROP_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Rect.h>
#include <nvcv/Size.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the ResizeCropConvertReformat operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaResizeCropConvertReformatCreate(NVCVOperatorHandle *handle);

/** Executes the ResizeCropConvertReformat operation on the given cuda stream. This operation
 *  does not wait for completion.
 *
 *  ResizeCropConvertReformat performs the following operations in order:
 *    1) Resize either a single tensor or each image in an ImageBatchVarShape
 *       to a specified width and height (other dimensions are unchanged).
 *    2) Crops a specified region of size width x height (determined by the
 *       output tensor's width & height) starting at the pixel position
 *       (cropPos.x, cropPos.y) out of the resized tensor.
 *    3) Convert the element data type to the output tensor's data type. For
 *       example, convert uchar elements to float. Limited options availble.
 *    4) Optional channel manipulation--i.e., re-order the channels
 *       of a tensor (e.g., RGB to BGR). Limited options available.
 *    5) If output tensor's layout doesn't match the input's layout, reshape
 *       the layout to match output layout (e.g., NHWC to NCHW). Limited
 *       options available.
 *  NOTE: Since all images in an ImageBatchVarShape are resized to the
 *        same size, the resulting collection now fits in a single tensor.
 *
 *  Limitations:
 *
 *  Input: STILL NEED TO FILL THIS IN
 *       Data Layout:    [NVCV_TENSOR_HWC, NVCV_TENSOR_NHWC]
 *       Channels:       [1, 3]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC,
 *                        NVCV_TENSOR_NCHW, NVCV_TENSOR_CHW]
 *       Channels:       [1, 3]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | No (Limited)
 *       Data Type     | No (Limited)
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | No
 *       Height        | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor or image batch. The images in an image batch can be of different
 *                sizes, but all images must have the same data type, channels, and layout.
 *
 * @param [in] resizeDim Dimensions, {width, height}, to resize the tensor method to be used,
 *                       see \ref NVCVSize2D for more details.
 *
 * @param [in] interpolation Interpolation method to be used, see \ref NVCVInterpolationType for
 *                           more details. Currently, only NVCV_INTERP_NEAREST and NVCV_INTERP_LINEAR
 *                           are available.
 *
 * @param [in] cropPos Crop position, (x, y), specifying the top-left corner of the crop region.
 *                     The crop region's width and height is specified by the output tensor's
 *                     width & height.
 *                     @note: The crop must fall within the resized image. Let (x, y, w, h)
 *                     represent the crop rectangle, where x & y are the cropPos coordinates
 *                     and w & h are the output tensor's width and height, then the following
 *                     must all be true:
 *                        x >= 0
 *                        y >= 0
 *                        x + w <= resizeDim.w
 *                        y + h <= resizeDim.h
 *
 *
 * @param [in] manip Channel manipulation to be used (e.g., reshuffle RGB to BGR),
 *                   see \ref NVCVChannelManip for more details.
 *
 * @param [out] out Output tensor. In addition to the output tensor determining the crop width
 *                  and height, the output tensor also specifies the data type (e.g., uchar3 or
 *                  float) and tensor layout (NHWC or NCHW), with limitations.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaResizeCropConvertReformatSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVTensorHandle in, NVCVTensorHandle out,
                                                               const NVCVSize2D            resizeDim,
                                                               const NVCVInterpolationType interpolation,
                                                               const int2 cropPos, const NVCVChannelManip manip);

CVCUDA_PUBLIC NVCVStatus cvcudaResizeCropConvertReformatVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                       NVCVImageBatchHandle in, NVCVTensorHandle out,
                                                                       const NVCVSize2D            resizeDim,
                                                                       const NVCVInterpolationType interpolation,
                                                                       const int2                  cropPos,
                                                                       const NVCVChannelManip      manip);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__RESIZE_CROP_H */
