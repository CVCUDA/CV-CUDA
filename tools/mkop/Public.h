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
 * @file Op__OPNAME__.h
 *
 * @brief Defines types and functions to handle the __OPNAME__ operation.
 * @defgroup NVCV_C_ALGORITHM___OPNAMECAP__ __OPNAMESPACE__
 * @{
 */

#ifndef CVCUDA___OPNAMECAP___H
#define CVCUDA___OPNAMECAP___H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the __OPNAME__ operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcuda__OPNAME__Create(NVCVOperatorHandle *handle);

/** Executes the __OPNAME__ operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [TODO]
 *       Channels:       [TODO]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | TODO
 *       8bit  Signed   | TODO
 *       16bit Unsigned | TODO
 *       16bit Signed   | TODO
 *       32bit Unsigned | TODO
 *       32bit Signed   | TODO
 *       32bit Float    | TODO
 *       64bit Float    | TODO
 *
 *  Output:
 *       Data Layout:    [TODO]
 *       Channels:       [TODO]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | TODO
 *       8bit  Signed   | TODO
 *       16bit Unsigned | TODO
 *       16bit Signed   | TODO
 *       32bit Unsigned | TODO
 *       32bit Signed   | TODO
 *       32bit Float    | TODO
 *       64bit Float    | TODO
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | TODO
 *       Data Type     | TODO
 *       Number        | TODO
 *       Channels      | TODO
 *       Width         | TODO
 *       Height        | TODO
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
CVCUDA_PUBLIC NVCVStatus cvcuda__OPNAME__Submit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                NVCVTensorHandle out);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA___OPNAMECAP___H */
