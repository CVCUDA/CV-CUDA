/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @defgroup NVCV_C_ALGORITHM_CHANNEL_REORDER ChannelReorder
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

/** Constructs and an instance of the channel reorder operator.
 * The operator is used to cha
 * reformats the input from kNHWC format to kNCHW format or from kNCHW format to
 * kNHWC format.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaChannelReorderCreate(NVCVOperatorHandle *handle);

/** Executes the reformat operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  * Input and output image formats must all have the same number of channels and
 *    have only one plane, although the channels can be swizzled (i.e. RGBA8, BGRA8, etc).
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
 *                       Negative indices will map to '0' value written to the correspondign output channel..
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
 *                       + The order value must be >= 0 and < the number of channels in input image.
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

#endif /* CVCUDA_CHANNEL_REORDER_H */
