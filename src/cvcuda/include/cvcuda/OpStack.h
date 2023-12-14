/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpStack.h
 *
 * @brief Defines types and functions to handle the Stack operation.
 * @defgroup NVCV_C_ALGORITHM__STACK Stack
 * @{
 */

#ifndef CVCUDA__STACK_H
#define CVCUDA__STACK_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Stack operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaStackCreate(NVCVOperatorHandle *handle);

/**
 *
 *  Executes the Stack operation on the given cuda stream. This operation does not
 *  wait for completion. The stack operation copies source tensors from into an output tensor.
 *  The output tensor is a concatenation of the source tensors, with each source tensor copied into
 *  the output tensor. All of the source tensors must have the same data type and number of channels width and height.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NHWC, NCHW, CHW, HWC]
 *       Channels:       [1,2,3,4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Output:
 *       Data Layout:    [NHWC, NCHW]
 *       Channels:       [1,2,3,4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | Yes
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | Yes
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Number        | No
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensors batch.
 *
 * @param [out] out output tensor NHWC/CHW where N is equal to the number of all input tensors.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaStackSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in,
                                           NVCVTensorHandle out);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__STACK_H */
