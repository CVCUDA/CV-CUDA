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
 * @file OpSolarize.h
 *
 * @brief Solarizes an image — inverts every pixel at or above a threshold, per element.
 *
 * Solarize computes, element-wise, @f$ out = (in \ge threshold) ? (bound - in) : in @f$, where
 * @p bound is the maximum representable value of the data type (255 for 8-bit unsigned, 65535 for
 * 16-bit unsigned, and 1.0 for 32-bit float). The above-threshold branch is the photometric
 * negative; the below-threshold branch passes the pixel through unchanged. The operation is
 * channel-independent.
 *
 * Reference: mimics torchvision.transforms.v2.functional.solarize
 * (out = where(x >= threshold, invert(x), x)). The mapping is exactly representable, so the result
 * is bit-exact with no rounding.
 *
 * @defgroup NVCV_C_ALGORITHM__SOLARIZE Solarize
 * @{
 */

#ifndef CVCUDA__SOLARIZE_H
#define CVCUDA__SOLARIZE_H

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

/** Constructs an instance of the Solarize operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSolarizeCreate(NVCVOperatorHandle *handle);

/** Executes the Solarize operation on the given cuda stream. This operation does not
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
 *       32bit Float    | Yes
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
 *       32bit Float    | Yes
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
 * @param [in] threshold Pixels whose value is greater than or equal to this threshold are inverted;
 *                       the threshold is expressed in the pixel value domain (e.g. 0..255 for 8-bit
 *                       unsigned, 0..1 for normalized float) and applied to all images.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSolarizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                              NVCVTensorHandle out, double threshold);

/** Executes the Solarize operation on a batch of variable-shaped images on the given cuda stream.
 *  Same limitations as cvcudaSolarizeSubmit.
 *
 * @param [in] handle    Handle to the operator. Must not be NULL.
 * @param [in] stream    Handle to a valid CUDA stream.
 * @param [in] in        input image batch.
 * @param [out] out      output image batch.
 * @param [in] threshold Inversion threshold in the pixel value domain (see cvcudaSolarizeSubmit).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSolarizeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                      NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                      double threshold);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__SOLARIZE_H */
