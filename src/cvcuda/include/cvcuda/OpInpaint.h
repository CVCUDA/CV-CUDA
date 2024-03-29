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
 * @file OpInpaint.h
 *
 * @brief Defines types and functions to handle the inpaint operation.
 * @defgroup NVCV_C_ALGORITHM_INPAINT Inpaint
 * @{
 */

#ifndef CVCUDA_INPAINT_H
#define CVCUDA_INPAINT_H

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

/* Constructs and an instance of the inpaint operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] maxBatchSize the maximum batch size.
 *
 * @param [in] maxHeight the maximum image height.
 *
 * @param [in] maxWidth the maximum image width.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/

CVCUDA_PUBLIC NVCVStatus cvcudaInpaintCreate(NVCVOperatorHandle *handle, int32_t maxBatchSize, int32_t maxHeight,
                                             int32_t maxWidth);

/* Executes the inpaint operation on the given cuda stream. This operation does not
 *  wait for completion.
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
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Mask:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1]
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
 *       Channels:       [1, 2, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
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
 * @param [in] in input tensor / image batch.
 *
 * @param [in] masks mask tensor, 8-bit 1-channel images. Non-zero pixels indicate the area that needs to be inpainted.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] inpaintRadius radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaInpaintSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                             NVCVTensorHandle masks, NVCVTensorHandle out, double inpaintRadius);

CVCUDA_PUBLIC NVCVStatus cvcudaInpaintVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVImageBatchHandle in, NVCVImageBatchHandle masks,
                                                     NVCVImageBatchHandle out, double inpaintRadius);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_INPAINT_H */
