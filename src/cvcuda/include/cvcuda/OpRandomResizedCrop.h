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
 * @file OpRandomResizedCrop.h
 *
 * @brief Defines types and functions to handle the random resized crop operation.
 * @defgroup NVCV_C_ALGORITHM_RANDOMRESIZEDCROP Random Resized Crop
 * @{
 */

#ifndef CVCUDA_RANDOMRESIZEDCROP_H
#define CVCUDA_RANDOMRESIZEDCROP_H

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

/** Constructs an instance of the random resized crop.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] minScale Lower bound for the random area of the crop, before resizing.
 *                      The scale is defined with respect to the area of the original image.
 *                      + Positive value.
 * @param [in] maxScale Upper bound for the random area of the crop, before resizing.
 *                      The scale is defined with respect to the area of the original image.
 *                      + Positive value.
 * @param [in] minRatio Lower bound for the random aspect ratio of the crop, before resizing.
 *                      + Positive value.
 * @param [in] maxRatio Upper bound for the random aspect ratio of the crop, before resizing.
 *                      + Positive value.
 * @param [in] maxBatchSize The maximum batch size that will be used by the operator.
 *                          + Positive value.
 * @param [in] seed The random seed that will be used by the operator. Pass 0 to use std::random_device.
 *                  + Positive value.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaRandomResizedCropCreate(NVCVOperatorHandle *handle, double minScale, double maxScale,
                                                       double minRatio, double maxRatio, int32_t maxBatchSize,
                                                       uint32_t seed);

/** Executes the random resized crop operation on the given cuda stream. This operation does not wait for completion.
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
 *       Width         | No
 *       Height        | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] interpolation Interpolation method to be used, see \ref NVCVInterpolationType for more details.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
CVCUDA_PUBLIC NVCVStatus cvcudaRandomResizedCropSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVTensorHandle in, NVCVTensorHandle out,
                                                       const NVCVInterpolationType interpolation);

CVCUDA_PUBLIC NVCVStatus cvcudaRandomResizedCropVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                               const NVCVInterpolationType interpolation);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_RANDOMRESIZEDCROP_H */
