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
 * @file OpErase.h
 *
 * @brief Defines types and functions to handle the erase operation.
 * @defgroup NVCV_C_ALGORITHM_ERASE Erase
 * @{
 */

#ifndef CVCUDA_ERASE_H
#define CVCUDA_ERASE_H

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

/* Constructs and an instance of the erase operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] max_num_erasing_area the maximum number of areas that will be erased.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/

CVCUDA_PUBLIC NVCVStatus cvcudaEraseCreate(NVCVOperatorHandle *handle, int32_t max_num_erasing_area);

/* Executes the erase operation on the given cuda stream. This operation does not
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
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
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
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
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
 *  anchor Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of eraing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_2S32.
 *
 *  erasing Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of eraing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_3S32.
 *
 *  imgIdx Tensor
 *
 *      Must be 'N' (dim = 1) with N = number of eraing area.
 *      Data Type must be 32bit Signed.
 *      DataType must be TYPE_S32.
 *
 *  values Tensor
 *
 *      Must be 'N' (dim = 1) with W = number of eraing area * 4.
 *      Data Type must be 32bit Float.
 *      DataType must be TYPE_F32.
 *
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor / image batch.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] anchor an array of size num_erasing_area that gives the x coordinate and y coordinate of the top left point in the eraseing areas.
 *
 * @param [in] eraisng an array of size num_erasing_area that gives the widths of the eraseing areas, the heights of the eraseing areas and
 *              integers in range 0-15, each of whose bits indicates whether or not the corresponding channel need to be erased.
 *
 * @param [in] values an array of size num_erasing_area*4 that gives the filling value for each erase area.
 *
 * @param [in] imgIdx an array of size num_erasing_area that maps a erase area idx to img idx in the batch.
 *
 * @param [in] random an boolean for random op.
 *
 * @param [in] seed random seed for random filling erase area.
 *
 * @param [in] inplace for perform inplace op.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
*/
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaEraseSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out, NVCVTensorHandle anchor, NVCVTensorHandle erasing,
                                           NVCVTensorHandle values, NVCVTensorHandle imgIdx, int8_t random,
                                           uint32_t seed);

CVCUDA_PUBLIC NVCVStatus cvcudaEraseVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                   NVCVTensorHandle anchor, NVCVTensorHandle erasing,
                                                   NVCVTensorHandle values, NVCVTensorHandle imgIdx, int8_t random,
                                                   uint32_t seed);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_ERASE_H */
