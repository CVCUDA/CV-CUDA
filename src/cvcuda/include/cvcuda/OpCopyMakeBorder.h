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
 * @file OpCopyMakeBorder.h
 *
 * @brief Defines types and functions to handle the copy make border operation.
 * @defgroup NVCV_C_ALGORITHM_COPYMAKEBORDER Copy make border
 * @{
 */

#ifndef CVCUDA_COPYMAKEBORDER_H
#define CVCUDA_COPYMAKEBORDER_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the copy make border.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCopyMakeBorderCreate(NVCVOperatorHandle *handle);

/** Executes the copy make border operation on the given cuda stream. This operation does not wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *       Note: 2 channels can only support 8bit Unsigned data type.
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
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
 *       Width         | No
 *       Height        | No
 *
 *  Top/left Tensors
 *
 *      Must be kNHWC where N=H=C=1 with W = N (N in reference to input and output tensors).
 *      Data Type must be 32bit Signed.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] top The top pixels
 *
 * @param [in] left The left pixels.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @param [in] borderValue Border value to be used for constant border mode \p NVCV_BORDER_CONSTANT.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCopyMakeBorderSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                    NVCVTensorHandle out, int32_t top, int32_t left,
                                                    NVCVBorderType borderMode, const float4 borderValue);

CVCUDA_PUBLIC NVCVStatus cvcudaCopyMakeBorderVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                            NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                            NVCVTensorHandle top, NVCVTensorHandle left,
                                                            NVCVBorderType borderMode, const float4 borderValue);

CVCUDA_PUBLIC NVCVStatus cvcudaCopyMakeBorderVarShapeStackSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                 NVCVImageBatchHandle in, NVCVTensorHandle out,
                                                                 NVCVTensorHandle top, NVCVTensorHandle left,
                                                                 NVCVBorderType borderMode, const float4 borderValue);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_COPYMAKEBORADER_H */
