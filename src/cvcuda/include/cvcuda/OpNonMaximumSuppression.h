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
 *  Non-Maximum-Suppression operation finds the set of non-overlapping bounding boxes (bboxes), for a given set of
 *  bboxes and scores, based on a score threshold and Intersection-over-Union (IoU) threshold.  First, all input
 *  bboxes with scores less than the score threshold are discarded.  Then, all input bboxes with sufficient overlap
 *  and lower score than another input bbox are discarded.  The overlap is calculated via IoU fraction, i.e. the
 *  intersection area divided by the area of the union, where sufficient overlap means greater than IoU threshold.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NW]
 *       Channel count:  [4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | No
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | No
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NW]
 *       Channel count:  [1]
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
 *       Data Type     | No
 *       Batches (N)   | Yes
 *       Bboxes (W)    | Yes
 *       Channels      | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor, in[i, j] is the set of input bbox proposals for an image where i ranges
 *                from 0 to batch-1, j ranges from 0 to number of bbox proposals, and the data type being
 *                short4 for (x=x, y=y, z=width, w=height) anchored at the top-left of the bounding box area
 *                + Must have data type 4S16 or S16, in case of S16 last shape must be 4
 *                + Must have rank 2 or 3, in case of 2 last shape must be 1 and data type must be 4S16
 *
 * @param [out] out Output tensor, out[i, j] is the output boolean mask marking the selected bboxes for an image
 *                  where i ranges from 0 to batch-1, j ranges from 0 to the number of bbox proposals, and the data
 *                  type being uint8_t marking selected bboxes as ones and discarded bboxes as zeros
 *                  + Must have data type U8
 *                  + Must have rank 2 or 3, in case of 3 last shape must be 1
 *
 * @param [in] scores Input tensor, scores[i, j] are the associated scores for each bounding box proposal in ``in``
 *                    considered during the reduce operation
 *                    + Must have data type F32
 *                    + Must have rank 2 or 3, in case of 3 last shape must be 1
 *
 * @param [in] scoreThreshold Minimum score an input bbox proposal need to have to be kept
 *
 * @param [in] iouThreshold Maximum overlap between bbox proposals covering the same effective image region as
 *                          calculated by Intersection-over-Union (IoU) fraction
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
