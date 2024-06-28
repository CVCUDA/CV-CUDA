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
 * @defgroup NVCV_C_ALGORITHM__RESIZE_CROP Resize Crop Convert
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

/** Executes the fused ResizeCropConvertReformat operation on the given cuda
 *  stream. This operation does not wait for completion.
 *
 *  ResizeCropConvertReformat is a fused operator that performs the following
 *  operations in order:
 *
 *    1. Resize either a single tensor or each image in an ImageBatchVarShape
 *       to a specified width and height (other dimensions are unchanged).
 *       This step is identical to the stand-alone Resize operation with the
 *       exception of optionally not type-casting the interpolation results
 *       back to the input data type (see the srcCast parameter for details).
 *
 *    2. Crops a specified region out of the resized tensor.
 *
 *    3. Apply a scale and offset to output result (after resizing and
 *       cropping). This can be used to normalize to a new range of values.
 *       For example, if the input is unsigned 8-bit values and the output is
 *       floating point, setting scale = 1.0/127.5 and offset = -1.0 will
 *       convert the 8-bit input values (ranging from 0 to 255) to floating
 *       point output values between -1.0 and 1.0.
 *
 *    4. Optional channel manipulation--i.e., re-order the channels
 *       of a tensor (e.g., RGB to BGR). Limited options available.
 *
 *    5. Convert the element data type to the output tensor's data type. For
 *       example, convert uchar elements to float. Limited options availble.
 *
 *    6. If output tensor's layout doesn't match the input's layout, reshape
 *       the layout to match output layout (e.g., NHWC to NCHW). Limited
 *       options available.
 *
 *  NOTES:
 *    + Since all images in an ImageBatchVarShape are resized to the same size,
 *      the resulting collection now fits in a single tensor.
 *    + Except for nearest-neighbor interpolation (NVCV_INTERP_NEAREST),
 *      interpolation (e.g., NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC, and
 *      NVCV_INTERP_AREA) computes resized pixel values using floating point
 *      math. However, the stand-alone resize operation (i.e., running the
 *      standard Resize operator independently) converts interpolated pixel
 *      values back to the source data type since its input and output types
 *      must be the same. As an option, this fused operator can either cast
 *      the resized pixel values back to the source type (to match results
 *      from running the steps independently), or leave them in the
 *      interpolated floating-point space to avoid quantization issues that
 *      occur from casting back to an integer source type (e.g., uchar). See
 *      the srcCast parameter for details.
 *
 *  Limitations:
 *
 *  Input:
 *       + Data Layout: [NVCV_TENSOR_HWC, NVCV_TENSOR_NHWC]
 *       + Channels: [1, 3]
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
 *       + Data Layout: [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC,
 *                       NVCV_TENSOR_NCHW, NVCV_TENSOR_CHW]
 *       + Channels: [1, 3]
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
 *  Input/Output dependency:
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
 * @param [in] handle Handle to the operator. Must not be NULL.
 *
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor or image batch. The images in an image batch can
 *                be of different sizes, but all images must have the same data
 *                type, channels, and layout.
 *
 * @param [in] resizeDim Dimensions, {width, height}, that tensor or image
 *                       batch images are resized to prior to cropping, see
 *                       \ref NVCVSize2D for more details.
 *
 * @param [in] interpolation Interpolation method to be used, (see \ref
 *                           NVCVInterpolationType). Currently, only
 *                           NVCV_INTERP_NEAREST and NVCV_INTERP_LINEAR are
 *                           available.
 *
 * @param [in] cropPos Crop position, (x, y), specifying the top-left corner of
 *                     the crop region. The crop region's width and height is
 *                     specified by the output tensor's width & height. The crop
 *                     must fall within the resized image. Let (x, y, w, h)
 *                     represent the crop rectangle, where x & y are the cropPos
 *                     coordinates and w and h are the output tensor's width and
 *                     height, respectively, then it must be true that:
 *                     + x >= 0,
 *                     + y >= 0,
 *                     + x + w <= resizeDim.w, and
 *                     + y + h <= resizeDim.h.
 *
 * @param [in] manip Channel manipulation to be used--e.g., reshuffle RGB to
 *                   BGR (see \ref NVCVChannelManip).
 *
 * @param [in] scale Scale (i.e., multiply) the resized and cropped output
 *                   values by this amount. 1.0 results in no scaling of the
 *                   output values.
 *
 * @param [in] offset Offset (i.e., add to) the output values by this amount.
 *                    This is applied after scaling--if v is a resized and
 *                    cropped value, then scale * v + offset is the final output
 *                    value. 0.0 results in no offset being added to the output.
 *
 * @param [in] srcCast Boolean value indicating whether or not the interpolation
 *                     results during the resize are re-cast back to the input
 *                     (or source) data type. Most interpolation methods (e.g.,
 *                     NVCV_INTERP_LINEAR) compute resized pixel values using
 *                     floating point math. This parameter determines if the
 *                     interpolation result is cast to the source data type
 *                     before computing the remaining steps in this operator:
 *                     + true: the interpolation result is cast back to the
 *                       source type prior to computing the remaining steps --
 *                       as if calling the stand-alone Resize operator (since
 *                       its input and output types must be the same). Note:
 *                       this option can produce quantized outputs (e.g., the
 *                       input source type is uchar3), even if the destination
 *                       data type is floating point.
 *                     + false: the interpolation result is NOT cast back to
 *                       the source type. Rather, the floating-point
 *                       interpolation results are directly passed on to the
 *                       remaining steps in the fused operator.
 *                     + Note: in either case (true or false) the final (fused)
 *                       result is still cast to the destination data type
 *                       before writing values into the output tensor.
 *
 * @param [out] out Output tensor. In addition to the output tensor determining
 *                  the crop width and height, the output tensor also specifies
 *                  the data type (e.g., uchar3 or float) and tensor layout
 *                  (NHWC or NCHW), with limitations.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is invalid or outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaResizeCropConvertReformatSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVTensorHandle in, NVCVTensorHandle out,
                                                               const NVCVSize2D            resizeDim,
                                                               const NVCVInterpolationType interpolation,
                                                               const int2 cropPos, const NVCVChannelManip manip,
                                                               const float scale, const float offset, bool srcCast);

CVCUDA_PUBLIC NVCVStatus cvcudaResizeCropConvertReformatVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                       NVCVImageBatchHandle in, NVCVTensorHandle out,
                                                                       const NVCVSize2D            resizeDim,
                                                                       const NVCVInterpolationType interpolation,
                                                                       const int2 cropPos, const NVCVChannelManip manip,
                                                                       const float scale, float offset, bool srcCast);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__RESIZE_CROP_H */
