/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpFindContours.h
 *
 * @brief Defines types and functions to handle the resize operation.
 * @defgroup NVCV_C_ALGORITHM_FIND_CONTOURS Find Contours
 * @{
 */

#ifndef CVCUDA_FIND_CONTOURS_H
#define CVCUDA_FIND_CONTOURS_H

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

/** Constructs and an instance of the resize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaFindContoursCreate(NVCVOperatorHandle *handle, int32_t maxWidth, int32_t maxHeight,
                                                  int32_t maxBatchSize);

/**
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | No
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | No
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes
 *      Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in GPU pointer to input data. Represents an 8-bit, unsigned,
 *     single-channel image. Non-zero pixels are treated as 1's, and zero
 *     pixels remain as 0's, which makes the image binary.
 * @param [out] points GPU pointer to output data. It contains the detected
 *     contours for the input image. The data is structured as: [x_c0_p0,
 *     y_c0_p0, ..., x_ci_pj, y_ci_pj, ...], where "ci" denotes a contour's
 *     index in the output array and "pj" is a point's index within a
 *     contour.
 * @param [out] numPoints Holds the number of contour points for each image.
 *     Specifically, numPoints[i] gives the number of contours for the i-th
 *     image, while numPoints[i][j] gives the number of points in the j-th
 *     contour of i-th image.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaFindContoursSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                  NVCVTensorHandle points, NVCVTensorHandle numPoints);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_FIND_CONTOURS_H */
