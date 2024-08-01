/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpFindHomography.h
 *
 * @brief Defines types and functions to handle the Find-Homography operation.
 * @defgroup NVCV_C_ALGORITHM_FIND_HOMOGRAPHY Find-Homography
 * @{
 */

#ifndef CVCUDA__FIND_HOMOGRAPHY_H
#define CVCUDA__FIND_HOMOGRAPHY_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/BorderType.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the Find-Homography operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in]  batchSize number of samples in the batch
 * @param [in]  numPoints maximum number of coordinates that in the batch
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaFindHomographyCreate(NVCVOperatorHandle *handle, int batchSize, int maxNumPoints);

/** Executes the Find-Homography operation on the given cuda stream.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NW]
 *       Channel count:  [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NHW]
 *       Channel count:  [1]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | No
 *       Data Type     | Yes
 *       Batches (N)   | Yes
 *       Channels      | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] srcPts Input tensor, srcPts[i, j] is the set of coordinates for the source image where i ranges
 *                from 0 to batch-1, j ranges from 4 to number of coordinates per image, and the data type being
 *                float2 for (x=x, y=y)
 *                + Number of coordinates must be >= 4
 *                + Must have data type 2F32 or F32
 *                + Must have rank 2 or 3
 *
 * * @param [in] dstPts Input tensor, dstPts[i, j] is the set of coordinates for the destination image where i ranges
 *                from 0 to batch-1, j ranges from 4 to number of coordinates per image, and the data type being
 *                float2 for (x=x, y=y)
 *                + Number of coordinates must be >= 4
 *                + Must have data type 2F32 or F32
 *                + Must have rank 2 or 3
 *
 * @param [out] out Output tensor, models[i, j, k] is the output model tensor which maps the src points to dst points
 *                  in image i, where i ranges from 0 to batch-1, j ranges from 0 to 2 and k ranges from 0 to 2, and
 *                  the data type being F32.
 *                  + Must have data type F32
 *                  + Must have rank 3
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaFindHomographySubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                    NVCVTensorHandle srcPts, NVCVTensorHandle dstPts,
                                                    NVCVTensorHandle models);

/**
 * Executes the FindHomography operation on a batch of images.
 *
 * Apart from input and output image batches, all parameters are the same as \ref cvcudaFindHomographySubmit.
 *
 * @param[in] srcPts batch of coordinates in the source image.
 * @param[out] dstPts batch of coordinates in the destination image.
 * @param [in] models model tensor batch.
 *
 */
CVCUDA_PUBLIC NVCVStatus cvcudaFindHomographyVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                            NVCVTensorBatchHandle srcPts, NVCVTensorBatchHandle dstPts,
                                                            NVCVTensorBatchHandle models);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__FIND_HOMOGRAPHY_H */
