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
 * @file OpBilateralFilter.h
 *
 * @brief Defines types and functions to handle the bilateral filter operation.
 * @defgroup NVCV_C_ALGORITM_BILATERAL_FILTER bilateral filter
 * @{
 */

#ifndef CVCUDA_BILATERAL_FILTER_H
#define CVCUDA_BILATERAL_FILTER_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Rect.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Bilateral Filter operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBilateralFilterCreate(NVCVOperatorHandle *handle);

/** Executes the BilateralFilter operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Destination must be same format and size as source
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 * @param [in] diameter bilateral filter diameter.
 *
 * @param [in] sigmaColor Gaussian exponent for color difference
 *
 * @param [in] sigmaSpace Gaussian exponent for position difference
 *
 * @param [in] borderMode texture border mode for input tensor
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBilateralFilterSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVTensorHandle in, NVCVTensorHandle out, int diameter,
                                                     float sigmaColor, float sigmaSpace, NVCVBorderType borderMode);

CVCUDA_PUBLIC NVCVStatus cvcudaBilateralFilterVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                             NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                             NVCVTensorHandle diameterData,
                                                             NVCVTensorHandle sigmaColorData,
                                                             NVCVTensorHandle sigmaSpaceData,
                                                             NVCVBorderType   borderMode);
#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_BILATERAL_FILTER_H */
