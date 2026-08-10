/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OpAdjustHue.h
 *
 * @brief Rotates the hue of an image in HSV space.
 *
 * AdjustHue converts each RGB pixel to HSV, shifts the hue channel by a scalar @p hue factor
 * (H is normalized to [0, 1)), and converts back to RGB: @f$ H' = (H + hue) \bmod 1 @f$. @p hue = 0
 * leaves the image unchanged; +/-0.5 is a full 180-degree hue rotation in either direction.
 * Single-channel images are returned unchanged.
 *
 * The pixel is processed in single precision: 8-bit unsigned input is scaled to [0, 1] before the
 * HSV round-trip and scaled back (truncated toward zero) afterwards. Float input is used directly;
 * input in [0, 1] remains in [0, 1], while out-of-range value-channel data is preserved as in torchvision.
 *
 * Reference: mimics torchvision.transforms.v2.functional.adjust_hue
 * (RGB -> HSV, H = (H + hue_factor) % 1, HSV -> RGB).
 *
 * @defgroup NVCV_C_ALGORITHM__ADJUST_HUE Adjust Hue
 * @{
 */

#ifndef CVCUDA__ADJUST_HUE_H
#define CVCUDA__ADJUST_HUE_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the AdjustHue operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustHueCreate(NVCVOperatorHandle *handle);

/** Executes the AdjustHue operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC, NVCV_TENSOR_NCHW, NVCV_TENSOR_CHW]
 *       Channels:       [1, 3]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       16bit Float    | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [NVCV_TENSOR_NHWC, NVCV_TENSOR_HWC, NVCV_TENSOR_NCHW, NVCV_TENSOR_CHW]
 *       Channels:       [1, 3]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | No
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       16bit Float    | No
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
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] hue Hue-rotation factor applied to all images. Must be in [-0.5, 0.5]; 0 leaves the
 *                 image unchanged, +/-0.5 is a full 180-degree rotation.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustHueSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out, double hue);

/** Executes the AdjustHue operation on a batch of variable-shaped images on the given cuda stream.
 *  Same limitations as cvcudaAdjustHueSubmit.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in     input image batch.
 * @param [out] out   output image batch.
 * @param [in] hue    Hue-rotation factor applied to all images (see cvcudaAdjustHueSubmit).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustHueVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVImageBatchHandle out, double hue);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__ADJUST_HUE_H */
