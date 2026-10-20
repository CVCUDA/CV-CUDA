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
 * @file OpHistogramEq.h
 *
 * @brief Defines types and functions to handle the HistogramEq operation.
 * @defgroup NVCV_C_ALGORITHM__HISTOGRAM_EQ Histogram Eq
 * @{
 */

#ifndef CVCUDA__HISTOGRAM_EQ_H
#define CVCUDA__HISTOGRAM_EQ_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the HistogramEq operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxBatchSize the maximum batch size.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHistogramEqCreate(NVCVOperatorHandle *handle, uint32_t maxBatchSize);

/** Executes the HistogramEq operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (kNCHW/kCHW support [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *
 *
 *
 *
 *
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 2, 3, 4] (kNCHW/kCHW support [1, 3, 4])
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       16bit Float    | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *       A floating-point image is binned to 8 bits, equalized and scaled back: a 256-bin
 *       histogram has no other reading of a continuous value. The scale factors are
 *       torchvision's, which reaches equalize through the same conversion -- multiply by
 *       `255 + 1 - 1e-3` and truncate on the way in, divide by 255 on the way out. 16-bit float
 *       bins through 32-bit float, because the multiplier is not representable in half.
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
CVCUDA_PUBLIC NVCVStatus cvcudaHistogramEqSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out);
/**
 * Executes the HistogramEq operation on a batch of images.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 *
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHistogramEqVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         NVCVImageBatchHandle in, NVCVImageBatchHandle out);

/** Executes HistogramEq on a tensor, choosing how the equalization table is built.
 *
 * The histogram, the per-channel semantics, the layouts and the data types are identical to
 * \ref cvcudaHistogramEqSubmit; only the lookup table differs, cf. \ref NVCVHistogramEqMethod.
 * Calling this with #NVCV_HISTOGRAM_EQ_OPENCV is bit-exact with \ref cvcudaHistogramEqSubmit.
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input tensor.
 * @param [out] out Output tensor.
 * @param [in] method Lookup-table construction, cf. \ref NVCVHistogramEqMethod. Use
 *                    #NVCV_HISTOGRAM_EQ_OPENCV to preserve \ref cvcudaHistogramEqSubmit's
 *                    behavior, or #NVCV_HISTOGRAM_EQ_PILLOW to match Pillow and torchvision.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHistogramEqMethodSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVTensorHandle in, NVCVTensorHandle out,
                                                       NVCVHistogramEqMethod method);

/** Executes the method-selecting HistogramEq on a batch of variable-shaped images.
 *
 * Apart from input and output image batches, all parameters are the same as
 * \ref cvcudaHistogramEqMethodSubmit.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaHistogramEqVarShapeMethodSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                               NVCVHistogramEqMethod method);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* CVCUDA__HISTOGRAM_EQ_H */
