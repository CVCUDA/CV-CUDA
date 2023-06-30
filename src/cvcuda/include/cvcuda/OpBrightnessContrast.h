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
 * @file OpBrightnessContrast.h
 *
 * @brief Defines types and functions to handle the BrightnessContrast operation.
 * @defgroup NVCV_C_ALGORITHM_BRIGHTNESS_CONTRAST Brightness Contrast
 * @{
 */

#ifndef CVCUDA_BRIGHTNESS_CONTRAST_H
#define CVCUDA_BRIGHTNESS_CONTRAST_H

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

/** Constructs an instance of the BrightnessContrast operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */

CVCUDA_PUBLIC NVCVStatus cvcudaBrightnessContrastCreate(NVCVOperatorHandle *handle);

/** Executes the BrightnessContrast operation on the given cuda stream. This operation does not wait for completion.
 *
 * The brightness and contrast are adjusted based on the following formula:
 *
 *       out = brightnessShift + brightness * (contrastCenter + contrast * (in - contrastCenter))
 *
 * If input or output type is (u)int32, the brightness, contrast, brightness shift and contrast center must
 * have float64 type, otherwise they are expected to be of float32 type.
 *
 *  Limitations:
 *
 *  Input:
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | Yes
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | No
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *       Samples       | Yes
 *
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
 * @param [in] brightness Optional tensor describing brightness multiplier.
 *                        If specified, it must contain either 1 or N elements where N
 *                        is the number of input images. If it contains a single element,
 *                        the same value is used for all input images.
 *                        If not specified, the neutral `1.` is used.
 *
 * @param [in] contrast Optional tensor describing contrast multiplier.
 *                      If specified, it must contain either 1 or N elements where N
 *                      is the number of input images. If it contains a single element,
 *                      the same value is used for all input images.
 *                      If not specified, the neutral `1.` is used.
 *
 * @param [in] brightnessShift Optional tensor describing brightness shift.
 *                             If specified, it must contain either 1 or N elements where N
 *                             is the number of input images. If it contains a single element,
 *                             the same value is used for all input images.
 *                             If not specified, the neutral `0.` is used.
 *
 * @param [in] contrastCenter Optional tensor describing contrast center.
 *                            If specified, it must contain either 1 or N elements where N
 *                            is the number of input images. If it contains a single element,
 *                            the same value is used for all input images.
 *                            If not specified, the middle of the assumed input type range is used.
 *                            For floats it is `0.5`, for unsigned integer types it is
 *                            `2 ** (number_of_bits - 1)`, for signed integer types it is
 *                            `2 ** (number_of_bits - 2)`.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBrightnessContrastSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        NVCVTensorHandle in, NVCVTensorHandle out,
                                                        NVCVTensorHandle brightness, NVCVTensorHandle contrast,
                                                        NVCVTensorHandle brightnessShift,
                                                        NVCVTensorHandle contrastCenter);

/**
 * Executes the BrightnessContrast operation on a batch of images.
 *
 * Apart from input and output image batches, all parameters are the same as \ref cvcudaBrightnessContrastSubmit.
 *
 * @param[in] in Input image batch.
 *
 * @param[out] out Output image batch.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBrightnessContrastVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                                NVCVTensorHandle brightness, NVCVTensorHandle contrast,
                                                                NVCVTensorHandle brightnessShift,
                                                                NVCVTensorHandle contrastCenter);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA_BRIGHTNESS_CONTRAST_H */
