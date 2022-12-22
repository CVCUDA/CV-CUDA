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
 * @file OpRotate.h
 *
 * @brief Defines types and functions to handle the rotate operation.
 * @defgroup NVCV_C_ALGORITHM_ROTATE Rotate
 * @{
 */

#ifndef CVCUDA_ROTATE_H
#define CVCUDA_ROTATE_H

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

/** Constructs and an instance of the rotate operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] maxVarShapeBatchSize maximum batch size for var shape operator
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaRotateCreate(NVCVOperatorHandle *handle, const int32_t maxVarShapeBatchSize);

/** Executes the rotate operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
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
 *       Width         | Yes
 *       Height        | Yes
 *
 *  Interpolation type
 *
 *       Type                 |  Allowed
 *      --------------------- | -------------
 *       NVCV_INTERP_NEAREST  | Yes
 *       NVCV_INTERP_LINEAR   | Yes
 *       NVCV_INTERP_CUBIC    | Yes
 *       NVCV_INTERP_AREA     | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor / image batch.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] angleDeg angle used for rotation in degrees.
 *
 * @param [in] shift value of shift in {x, y} directions to move the center at the same coord after rotation.
 *
 * @param [in] interpolation Interpolation method to be used, see \ref NVCVInterpolationType for more details.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaRotateSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                            NVCVTensorHandle out, const double angleDeg, const double2 shift,
                                            const NVCVInterpolationType interpolation);

CVCUDA_PUBLIC NVCVStatus cvcudaRotateVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                    NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                    NVCVTensorHandle angleDeg, NVCVTensorHandle shift,
                                                    const NVCVInterpolationType interpolation);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_ROTATE_H */
