/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpChannelReorder.h
 *
 * @brief Defines types and functions to handle the channel reorder operation.
 * @defgroup NVCV_C_ALGORITHM_CHANNEL_REORDER Channel Reorder
 * @{
 */

#ifndef CVCUDA_CHANNEL_REORDER_H
#define CVCUDA_CHANNEL_REORDER_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the channel reorder operator.
 * The operator copies input channels to output channels according to an order tensor.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaChannelReorderCreate(NVCVOperatorHandle *handle);

/** Executes channel reorder on a tensor on the given CUDA stream. This operation does not
 *  wait for completion.
 *
 *  For every output channel @p c, the operator copies input channel @p order[c]. A negative
 *  order entry writes zero to the corresponding output channel. Repeated non-negative entries
 *  are allowed.
 *
 *  Reference: the non-negative gather mapping matches
 *  torchvision.transforms.v2.functional.permute_channels. Native ChannelReorder preserves its
 *  established negative-entry zero-fill behavior; callers that need Python negative indexing
 *  must normalize those indices before submission.
 *
 *  Limitations:
 *
 *  Input/Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (planar kNCHW/kCHW: [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input and output must have identical shape, layout, data type, and channel count.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor.
 * @param [out] out Output tensor.
 * @param [in] order Host pointer to @p orderLength signed 32-bit channel indices. The values are
 *                   copied synchronously during this call and may be released after it returns.
 * @param [in] orderLength Number of entries in @p order; must equal the tensor channel count.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside the valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval #NVCV_SUCCESS                Operation submitted successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaChannelReorderSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                    NVCVTensorHandle out, const int32_t *order, int32_t orderLength);

/** Executes the reformat operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (planar kNCHW/kCHW: [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (planar kNCHW/kCHW: [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Layout family
 *       Data Type     | Yes
 *       Number        | Yes
 *       Channels      | No
 *
 *  * Input and output image formats must have the same layout family: interleaved
 *    formats have one plane, and planar formats have one plane per channel.
 *    Channels can be swizzled (i.e. RGBA8, BGRA8, RGBA8p, BGRA8p, etc).
 *
 *  * The number of samples in the input and output ImageBatch must be the same
 *
 *  * The \p orders_in tensor must have 2 dimensions. First dimension correspond to the
 *    number of images being, and the second the number of channels.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input varshape image batch.
 *
 * @param [out] out output varshape image batch.
 *
 * @param [in] orders_in 2D tensor with layout "NC" which specifies, for each output image sample in the batch,
 *                       the index of the input channel to copy to the output channel.
 *                       Negative indices will map to '0' value written to the corresponding output channel.
 *
 *                       @note The output images' format isn't updated to reflect the new channel ordering.
 *
 *                       Example:
 *                          let input be RGBA8 with a pixel = [108,63,18,214],
 *                               output be YUV8,
 *                               orders_in = [3,-1,1]
 *
 *                          The corresponding pixel in the output will be [214,0,63].
 *
 *                       + Must not be NULL.
 *                       + A non-negative order value must be less than the number of channels in the input image.
 *                         Negative values write zero to the corresponding output channel.
 *                       + Tensor dimensions must be NxC, where N is the number of images in the input varshape,
 *                         and C is the number of channels in the output images.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaChannelReorderVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                            NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                            NVCVTensorHandle orders_in);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* CVCUDA_CHANNEL_REORDER_H */
