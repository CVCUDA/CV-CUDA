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
 * @file OpNonMaximumSuppression.h
 *
 * @brief Defines types and functions to handle the Non-Maximum-Suppression operation.
 * @defgroup NVCV_C_ALGORITHM_NON_MAXIMUM_SUPPRESSION Non-Maximum Suppression
 * @{
 */

#ifndef CVCUDA__NON_MAXIMUM_SUPPRESSION_H
#define CVCUDA__NON_MAXIMUM_SUPPRESSION_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/BorderType.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Non-Maximum-Suppression operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaNonMaximumSuppressionCreate(NVCVOperatorHandle *handle);

/** Executes the Non-Maximum Suppression operation on the given cuda stream.
 *
 *  Non-Maximum-Suppression operation finds the set of non-overlapping bounding
 *  boxes, for a given set of bounding boxes and scores, based on a score
 *  threshold and Intersection-over-Union (IoU) threshold.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NCW]
 *       Width:          [4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NCW]
 *       Width:          [4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
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
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor batch, in[i, j] is the set of input bounding
 *     box proposals for an image where i ranges from 0 to batch-1, j ranges
 *     from 0 to number of bounding box proposals, and the data type being int4
 *     for (x=x, y=y, z=width, w=height) anchored at the top-left of the
 *     bounding box area
 *
 * @param [out] out output tensor bactch, out[i, j, k] is the set of output
 *     bounding box proposals for an image where i ranges from 0 to batch-1, j
 *     ranges from 0 to the reduced number of bounding box proposals, and the
 *     data type being int4 for (x=x, y=y, z=width, w=height) anchored at the
 *     top-left of the bounding box area
 *
 * @param [in] scores input tensor batch, scores[i, j] are the associated scores
 *     for each bounding box proposal in ``in`` considered during the reduce
 *     operation
 *
 * @param [in] score_threshold Minimum score of a bounding box proposals
 *
 * @param [in] iou_threshold Maximum overlap between bounding box proposals
 *      covering the same effective image region as calculated by Intersection-
 *      over-Union (IoU)
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaNonMaximumSuppressionSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                           NVCVTensorHandle in, NVCVTensorHandle out,
                                                           NVCVTensorHandle scores, float scoreThreshold,
                                                           float iouThreshold);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__NON_MAXIMUM_SUPPRESSION_H */
