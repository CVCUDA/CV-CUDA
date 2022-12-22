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
 * @file OpAverageBlur.h
 *
 * @brief Defines types and functions to handle the AverageBlur operation.
 * @defgroup NVCV_C_ALGORITHM_AVERAGEBLUR Average Blur
 * @{
 */

#ifndef CVCUDA_AVERAGEBLUR_H
#define CVCUDA_AVERAGEBLUR_H

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

/** Constructs an instance of the AverageBlur.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxKernelWidth The maximum kernel width that will be used by the operator.
 *                            + Positive value.
 * @param [in] maxKernelHeight The maximum kernel height that will be used by the operator.
 *                            + Positive value.
 * @param [in] maxVarShapeBatchSize The maximum batch size that will be used by the var-shape operator.
 *                                  + Positive value.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAverageBlurCreate(NVCVOperatorHandle *handle, int32_t maxKernelWidth,
                                                 int32_t maxKernelHeight, int32_t maxVarShapeBatchSize);

/** Executes the AverageBlur operation on the given cuda stream.  This operation does not wait for completion.
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
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
 *      16bit Signed   | Yes
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
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] kernelWidth AverageBlur kernel width.
 *
 * @param [in] kernelHeight AverageBlur kernel height.
 *
 * @param [in] kernelAnchorX Kernel anchor in X direction.  Indicates the relative position of a filtered point
 * within the kernel.  Use (-1, -1) to indicate kernel center.
 *
 * @param [in] kernelAnchorY Kernel anchor in Y direction.  Indicates the relative position of a filtered point
 * within the kernel.  Use (-1, -1) to indicate kernel center.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAverageBlurSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, int32_t kernelWidth, int32_t kernelHeight,
                                                 int32_t kernelAnchorX, int32_t kernelAnchorY,
                                                 NVCVBorderType borderMode);

/**
 * Executes the AverageBlur operation on a batch of images.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 * @param[in] kernelSize Average blur kernel size as a Tensor of int2.
 *                       + Must be of pixel type NVCV_DATA_TYPE_2S32
 * @param[in] kernelAnchor Average blur kernel anchor as a Tensor of int2.
 *                         + Must be of pixel type NVCV_DATA_TYPE_2S32
 * @param[in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAverageBlurVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                         NVCVTensorHandle kernelSize, NVCVTensorHandle kernelAnchor,
                                                         NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_AVERAGEBLUR_H */
