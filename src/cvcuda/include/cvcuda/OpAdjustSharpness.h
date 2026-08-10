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
 * @file OpAdjustSharpness.h
 *
 * @brief Adjusts image sharpness by blending each image with a 3x3-smoothed copy of itself.
 *
 * AdjustSharpness enhances or reduces local contrast by blending the input with a "degenerate"
 * (smoothed) version of itself. The smoothed image is produced, per channel independently, by a
 * depthwise 3x3 convolution with the normalized smoothing kernel @f$ \frac{1}{13}
 * \begin{bmatrix} 1 & 1 & 1 \\ 1 & 5 & 1 \\ 1 & 1 & 1 \end{bmatrix} @f$. The output is the blend
 * @f$ out = factor \cdot in + (1 - factor) \cdot blur @f$, where @p factor is
 * @p sharpnessFactor: 1.0 leaves the image unchanged, 0.0 yields the fully-smoothed image, and
 * values greater than 1.0 sharpen. For integer types the smoothed value is rounded to nearest
 * (ties to even) before blending, the blend is clamped to @f$ [0, bound] @f$ (255 for 8-bit
 * unsigned, 65535 for 16-bit unsigned) and truncated toward zero on the final cast; for float it
 * is clamped to @f$ [0, 1] @f$.
 *
 * The convolution only rewrites the image interior: the 1-pixel border is copied through
 * unchanged (there is no border extension). Consequently, when the image height or width is less
 * than 3 there is no interior and the entire image is copied unchanged.
 *
 * Reference: mimics torchvision.transforms.v2.functional.adjust_sharpness
 * (out = blend(image, blur(image), sharpness_factor), interior-only, border unchanged).
 *
 * @defgroup NVCV_C_ALGORITHM__ADJUST_SHARPNESS Adjust Sharpness
 * @{
 */

#ifndef CVCUDA__ADJUST_SHARPNESS_H
#define CVCUDA__ADJUST_SHARPNESS_H

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

/** Constructs an instance of the AdjustSharpness operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSharpnessCreate(NVCVOperatorHandle *handle);

/** Executes the AdjustSharpness operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
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
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] sharpnessFactor Blend weight applied to the original image; the smoothed image
 *                             receives weight (1 - sharpnessFactor). 1.0 leaves the image
 *                             unchanged, 0.0 yields the fully-smoothed image, values above 1.0
 *                             sharpen. Must be non-negative. Only the image interior is blended;
 *                             the 1-pixel border is copied unchanged, and images with height or
 *                             width below 3 are copied unchanged in full.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSharpnessSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVTensorHandle in, NVCVTensorHandle out, float sharpnessFactor);

/** Executes the AdjustSharpness operation on a batch of variable-shaped images on the given cuda
 *  stream. Same limitations as cvcudaAdjustSharpnessSubmit.
 *
 * @param [in] handle          Handle to the operator. Must not be NULL.
 * @param [in] stream          Handle to a valid CUDA stream.
 * @param [in] in              input image batch.
 * @param [out] out            output image batch.
 * @param [in] sharpnessFactor Blend weight (see cvcudaAdjustSharpnessSubmit).
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustSharpnessVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                             NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                             float sharpnessFactor);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__ADJUST_SHARPNESS_H */
