/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpImageArea.h
 *
 * @brief Writes the spatial pixel count (area) of every image in a batch: @f$ out[n] = W * H @f$.
 *
 * ImageArea reports the spatial extent of each sample image as its number of pixels, i.e.
 * @f$ out[n] = W_n \times H_n @f$, with the channel dimension excluded. Only the tensor geometry
 * is consulted; pixel values, image layout (interleaved or planar) and data type never change the
 * result, but the input must still be a valid image tensor.
 *
 * Reference: custom — no external reference; @f$ out[n] = W_n \times H_n @f$ (spatial pixel count,
 * channels excluded). This is the area of each image, not the number of stored elements.
 *
 * @defgroup NVCV_C_ALGORITHM__IMAGEAREA ImageArea
 * @{
 */

#ifndef CVCUDA__IMAGEAREA_H
#define CVCUDA__IMAGEAREA_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the ImageArea operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaImageAreaCreate(NVCVOperatorHandle *handle);

/** Executes the ImageArea operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Output:
 *       Data Layout:    [N]
 *       Channels:       [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       16bit Float    | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Planar image layouts: supported (NCHW, CHW), alongside interleaved (NHWC, HWC).
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | No
 *       Data Type     | No
 *       Number        | Yes (see below)
 *       Channels      | No
 *       Width         | No
 *       Height        | No
 *
 *  The output holds exactly one element per input sample, so a batched input ([kNHWC] or
 *  [kNCHW], rank 4) requires an [N] output with the same N, and an unbatched input ([kHWC] or
 *  [kCHW], rank 3) requires an [N] output with N == 1.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaImageAreaSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__IMAGEAREA_H */
