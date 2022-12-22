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
 * @file OpComposite.h
 *
 * @brief Defines types and functions to handle the composite operation.
 * @defgroup NVCV_C_ALGORITHM_COMPOSITE Composite
 * @{
 */

#ifndef CVCUDA_COMPOSITE_H
#define CVCUDA_COMPOSITE_H

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

/** Constructs and an instance of the composite operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCompositeCreate(NVCVOperatorHandle *handle);

/** Executes the composite operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [3]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
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
 *       Channels      | No
 *       Width         | Yes
 *       Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] foreground input foreground tensor. Each image is BGR (3-channel) 8-bit.
 *
 * @param [in] background input background tensor. Each image is BGR (3-channel) 8-bit.
 *
 * @param [in] fgMask input foreground mask tensor. Each mask image is grayscale 8-bit
 *
 * @param [out] output output tensor. Each output image is BGR(A) (3-channel for BGR, 4-channel for BGRA) 8-bit.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaCompositeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                               NVCVTensorHandle foreground, NVCVTensorHandle background,
                                               NVCVTensorHandle fgMask, NVCVTensorHandle output);

CVCUDA_PUBLIC NVCVStatus cvcudaCompositeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle foreground, NVCVImageBatchHandle background,
                                                       NVCVImageBatchHandle fgMask, NVCVImageBatchHandle output);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_COMPOSITE_H */
