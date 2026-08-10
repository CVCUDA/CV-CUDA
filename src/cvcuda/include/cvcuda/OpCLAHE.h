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
 * @file OpCLAHE.h
 *
 * @brief Defines types and functions to handle CLAHE operation.
 * @defgroup NVCV_C_ALGORITHM__CLAHE CLAHE
 * @{
 */

#ifndef CVCUDA__CLAHE_H
#define CVCUDA__CLAHE_H

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

/** Constructs an instance of the CLAHE operator.
 *
 * @param [out] handle Where the operator handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxBatchSize The maximum batch size this operator will process.
 *                          + Must be >= 1.
 * @param [in] tilesX Number of tiles along width.
 *                    + Must be >= 1.
 * @param [in] tilesY Number of tiles along height.
 *                    + Must be >= 1.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null or maxBatchSize is invalid.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCLAHECreate(NVCVOperatorHandle *handle, int32_t maxBatchSize, int32_t tilesX,
                                           int32_t tilesY);

/** Executes CLAHE on a tensor input/output on the given cuda stream.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
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
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
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
 * @param [in] in Input tensor.
 * @param [out] out Output tensor.
 * @param [in] clipLimit CLAHE clip limit value. Must be > 0.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCLAHESubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorHandle out, float clipLimit);

/** Executes CLAHE on a varshape image batch on the given cuda stream.
 *
 *  Input and output images must be single-channel U8 and have matching per-image
 *  shape and format.
 *
 * @param [in] handle Handle to the operator.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in Input image batch.
 * @param [out] out Output image batch.
 * @param [in] clipLimit CLAHE clip limit value. Must be > 0.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaCLAHEVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   NVCVImageBatchHandle in, NVCVImageBatchHandle out, float clipLimit);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* CVCUDA__CLAHE_H */
