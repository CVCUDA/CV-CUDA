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
 * @file OpConvertTo.h
 *
 * @brief Defines types and functions to handle the ConvertTo operation.
 * @defgroup NVCV_C_ALGORITHM_CONVERT_TO ConvertTo
 * @{
 */

#ifndef CVCUDA_CONVERT_TO_H
#define CVCUDA_CONVERT_TO_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the ConvertTo operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaConvertToCreate(NVCVOperatorHandle *handle);

/** Executes the ConvertTo operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  outputs(x,y) = saturate_cast<out_type>(α * inputs(x, y) + β)
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1-4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1-4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in intput tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] alpha Scalar for output data.
 *
 * @param [in] beta Offset for the data.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaConvertToSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out, const double alpha, const double beta);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_CONVERT_TO_H */
