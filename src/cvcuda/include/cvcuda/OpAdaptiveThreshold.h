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
 * @file OpAdaptiveThreshold.h
 *
 * @brief Defines types and functions to handle the adaptive threshold operation.
 * @defgroup NVCV_C_ALGORITHM_ADAPTIVETHRESHOLD Adaptive Threshold
 * @{
 */

#ifndef CVCUDA_ADAPTIVETHRESHOLD_H
#define CVCUDA_ADAPTIVETHRESHOLD_H

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

/** Constructs an instance of the adaptive threshold.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxBlockSize The maximum block size that will be used by the operator.
 *                          + Positive value.
 * @param [in] maxVarShapeBatchSize The maximum batch size that will be used by the var-shape operator.
 *                                  + Positive value.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdaptiveThresholdCreate(NVCVOperatorHandle *handle, int32_t maxBlockSize,
                                                       int32_t maxVarShapeBatchSize);

/** Executes the adaptive threshold operation on the given cuda stream. This operation does not wait for completion.
 *
 *  Limitations:
 *
 *  Input:
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
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] maxValue Non-zero value assigned to the pixels for which the condition is satisfied.
 *
 * @param [in] adaptiveMethod Adaptive thresholding algorithm to use. \p NVCVAdaptiveThresholdType.
 *
 * @param [in] thresholdType Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV. \p NVCVThresholdType.
 *
 * @param [in] blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
 *
 * @param [in] c Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdaptiveThresholdSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVTensorHandle in, NVCVTensorHandle out, double maxValue,
                                                       NVCVAdaptiveThresholdType adaptiveMethod,
                                                       NVCVThresholdType thresholdType, int32_t blockSize, double c);

CVCUDA_PUBLIC NVCVStatus cvcudaAdaptiveThresholdVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                               NVCVTensorHandle          maxValue,
                                                               NVCVAdaptiveThresholdType adaptiveMethod,
                                                               NVCVThresholdType         thresholdType,
                                                               NVCVTensorHandle blockSize, NVCVTensorHandle c);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_ADAPTIVETHRESHOLD_H */
