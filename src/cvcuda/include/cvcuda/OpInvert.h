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
 * @file OpInvert.h
 *
 * @brief Inverts an image — the photometric negative @f$ out = bound - in @f$, per element.
 *
 * Invert computes the per-element photometric negative @f$ out = bound - in @f$, where @p bound
 * is the maximum representable value for integer types and 1.0 for 16-, 32-, and 64-bit float.
 * Unsigned integer subtraction is exact. Signed integer results saturate to the data type maximum
 * for negative inputs. Floating-point subtraction is rounded to the output type. The operation is
 * element-wise and channel-independent.
 *
 * Reference: mimics torchvision.transforms.v2.functional.invert (equivalently OpenCV
 * cv::bitwise_not for the unsigned-integer cases).
 *
 * @defgroup NVCV_C_ALGORITHM__INVERT Invert
 * @{
 */

#ifndef CVCUDA__INVERT_H
#define CVCUDA__INVERT_H

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

/** Constructs an instance of the Invert operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaInvertCreate(NVCVOperatorHandle *handle);

/** Executes the Invert operation on the given cuda stream. This operation does not
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
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       64bit Unsigned | No
 *       64bit Signed   | Yes
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
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
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       64bit Unsigned | No
 *       64bit Signed   | Yes
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
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
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaInvertSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                            NVCVTensorHandle out);

/** Executes the Invert operation on a batch of variable-shaped images on the given cuda stream.
 *  Same limitations as cvcudaInvertSubmit.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in     input image batch.
 * @param [out] out   output image batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaInvertVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                    NVCVImageBatchHandle in, NVCVImageBatchHandle out);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__INVERT_H */
