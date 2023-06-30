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
 * @file OpColorTwist.h
 *
 * @brief Defines types and functions to handle the ColorTwist operation.
 * @defgroup NVCV_C_ALGORITHM_COLOR_TWIST Color Twist
 * @{
 */

#ifndef CVCUDA_COLOR_TWIST_H
#define CVCUDA_COLOR_TWIST_H

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

/** Constructs an instance of the ColorTwist operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */

CVCUDA_PUBLIC NVCVStatus cvcudaColorTwistCreate(NVCVOperatorHandle *handle);

/** Executes the ColorTwist operation on the given cuda stream. This operation does not wait for completion.
 *
 * ColorTwist modifies colors in the image by applying custom affine transformation to the channels.
 * For the 3x4 twist matrix and 3 channel input, the output channels are:
 *
 *       output[0] = twist[0][0] * input[0] + twist[0][1] * input[1] + twist[0][2] * input[2] + twist[0][3]
 *       output[1] = twist[1][0] * input[0] + twist[1][1] * input[1] + twist[1][2] * input[2] + twist[1][3]
 *       output[2] = twist[2][0] * input[0] + twist[2][1] * input[1] + twist[2][2] * input[2] + twist[2][3]
 *
 * If the input images have 4 channels, the last channel is assumed to be alpha and is copied unmodified
 * to the output.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC]
 *       Channels:       [3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC]
 *       Channels:       [3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | Yes
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Twist matrix:
 *       Data Layout:    [NVCV_TENSOR_NHW, NVCV_TENSOR_HW]
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
 *       64bit Float    | Yes
 *
 *  Input/Output dependency
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *       Samples       | Yes
 *
 * Input/Twist matrix type dependency
 *       Input type     |  Accepted twist type
 *      --------------- | -------------
 *       uint8          | float32
 *       int16, uint16  | float32
 *       int32, uint32  | float64
 *       float32        | float32
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor to get values from.
 *                + Must not be NULL.
 *
 * @param [out] out Output tensor to set values to.
 *                  + Must not be NULL.
 *
 * @param [in] twist Tensor describing the 3x4 affine transformation to apply to the channels extent.
 *                   It can either be a 2D 3x4 tensor describing a single transformation for all samples
 *                   or 3D tensor that defines the transformations for each sample separately.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaColorTwistSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                NVCVTensorHandle out, NVCVTensorHandle twist);

/**
 * Executes the ColorTwist operation on a batch of images.
 *
 * Apart from input and output image batches, all parameters are the same as \ref cvcudaColorTwistSubmit.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 * @param [in] twist Tensor describing the affine transformation to apply to the channels extent.
 *
 */
CVCUDA_PUBLIC NVCVStatus cvcudaColorTwistVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                        NVCVTensorHandle twist);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_COLOR_TWIST_H */
