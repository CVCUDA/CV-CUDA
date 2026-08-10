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
 * @file OpPosterize.h
 *
 * @brief Posterizes an image — reduces each channel to its top @p bits.
 *
 * Posterize keeps the @p bits most-significant bits of every channel value and zeros the rest,
 * computing @f$ out = in \& \sim((1 \ll (W - bits)) - 1) @f$ per element, where @p W is the data
 * type bit width (8 for 8-bit unsigned, 16 for 16-bit unsigned). It is a pure bitwise mask, applied
 * channel-independently. @p bits must be in @f$ [0, W] @f$ for the input data type.
 *
 * Reference: mimics torchvision.transforms.v2.functional.posterize / PIL ImageOps.posterize (8-bit:
 * keep the top @p bits), generalized to 16-bit. The mask is exact, so the result is bit-exact.
 *
 * @defgroup NVCV_C_ALGORITHM__POSTERIZE Posterize
 * @{
 */

#ifndef CVCUDA__POSTERIZE_H
#define CVCUDA__POSTERIZE_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the Posterize operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPosterizeCreate(NVCVOperatorHandle *handle);

/** Executes the Posterize operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] bits Number of most-significant bits to keep per channel; must be in [0, W] where W
 *                  is the data type bit width (8 for 8-bit, 16 for 16-bit). bits == W is a no-op;
 *                  bits == 0 zeros the image. Applied to all images.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPosterizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out, int32_t bits);

/** Executes the Posterize operation on a batch of variable-shaped images on the given cuda stream.
 *  Same limitations as cvcudaPosterizeSubmit.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in     input image batch.
 * @param [out] out   output image batch.
 * @param [in] bits   Number of most-significant bits to keep per channel (see cvcudaPosterizeSubmit).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaPosterizeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVImageBatchHandle out, int32_t bits);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__POSTERIZE_H */
