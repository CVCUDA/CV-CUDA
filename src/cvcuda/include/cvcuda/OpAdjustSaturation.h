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
 * @file OpAdjustSaturation.h
 *
 * @brief Adjusts the color saturation of an image by blending it toward its grayscale.
 *
 * AdjustSaturation blends each RGB pixel toward its luminance (grayscale) by a scalar
 * @p saturation factor: @f$ out_c = saturation \cdot in_c + (1 - saturation) \cdot gray @f$, where
 * @f$ gray = 0.2989 \cdot R + 0.587 \cdot G + 0.114 \cdot B @f$ (computed in single precision, and
 * floored for integer dtypes before blending). @p saturation = 1 leaves the image unchanged,
 * 0 produces the grayscale image, and values > 1 over-saturate. For integer dtypes the blended
 * value is clamped to the dtype range and truncated toward zero (matching torchvision's cast); for
 * float the value is clamped to [0, 1]. Single-channel images are returned unchanged.
 *
 * The luminance coefficients (0.2989 / 0.587 / 0.114) follow torchvision and intentionally differ
 * from cvcuda.cvtcolor's RGB2GRAY (0.299 / 0.587 / 0.114); the two operators answer to different
 * references.
 *
 * Reference: mimics torchvision.transforms.v2.functional.adjust_saturation
 * (out = blend(image, rgb_to_grayscale(image), saturation_factor)).
 *
 * @defgroup NVCV_C_ALGORITHM__ADJUST_SATURATION Adjust Saturation
 * @{
 */

#ifndef CVCUDA__ADJUST_SATURATION_H
#define CVCUDA__ADJUST_SATURATION_H

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

/** Constructs an instance of the AdjustSaturation operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSaturationCreate(NVCVOperatorHandle *handle);

/** Executes the AdjustSaturation operation on the given cuda stream. This operation does not
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
 * @param [in] saturation Saturation factor applied to all images. Must be >= 0; 1 leaves the image
 *                        unchanged, 0 yields grayscale, values > 1 over-saturate.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSaturationSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                      NVCVTensorHandle in, NVCVTensorHandle out, double saturation);

/** Executes the AdjustSaturation operation on a batch of variable-shaped images on the given cuda
 *  stream. Same limitations as cvcudaAdjustSaturationSubmit.
 *
 * @param [in] handle     Handle to the operator. Must not be NULL.
 * @param [in] stream     Handle to a valid CUDA stream.
 * @param [in] in         input image batch.
 * @param [out] out       output image batch.
 * @param [in] saturation Saturation factor applied to all images (see cvcudaAdjustSaturationSubmit).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSaturationVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                              NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                              double saturation);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__ADJUST_SATURATION_H */
