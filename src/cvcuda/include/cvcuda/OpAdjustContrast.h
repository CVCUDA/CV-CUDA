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
 * @file OpAdjustContrast.h
 *
 * @brief Blends each image toward its grayscale mean by a scalar factor (torchvision-compatible
 *        adjust_contrast): out = clamp(contrastFactor * in + (1 - contrastFactor) * mean, 0, bound).
 * @defgroup NVCV_C_ALGORITHM_ADJUST_CONTRAST Adjust Contrast
 * @{
 */

#ifndef CVCUDA_ADJUST_CONTRAST_H
#define CVCUDA_ADJUST_CONTRAST_H

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

/** Constructs an instance of the AdjustContrast operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustContrastCreate(NVCVOperatorHandle *handle);

/** Executes the AdjustContrast operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 * Blends each image toward its grayscale mean by a single scalar factor, matching
 * torchvision.transforms.v2.functional.adjust_contrast:
 *
 *       out = clamp(contrastFactor * in + (1 - contrastFactor) * mean, 0, bound)
 *
 * where `bound` is 1.0 for floating-point images and the dtype maximum for integer images, and
 * `mean` is the per-image mean of its grayscale conversion. Grayscale uses the BT.601 luma weights
 * `0.2989 R + 0.587 G + 0.114 B` (floored for integer 3-channel input, matching torchvision); a
 * single-channel image is its own grayscale. A factor of 0 produces a flat gray image, 1 leaves
 * the image unchanged, and values > 1 increase contrast. Integer results round-to-nearest, so they
 * may differ from torchvision (which truncates) by at most 1 LSB.
 * Three-channel tensors are interpreted in RGB component order. Three-channel image batches use
 * their RGB format semantics; formats without a color model are interpreted as RGB. The format
 * swizzle is honored, including BGR storage order.
 *
 * Reference: torchvision.transforms.v2.functional.adjust_contrast
 *            (out = clamp(factor * in + (1 - factor) * grayscale_mean, 0, bound)).
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
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
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
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
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *       Samples       | Yes
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
 * @param [in] contrastFactor Non-negative scalar contrast multiplier.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustContrastSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                    NVCVTensorHandle out, double contrastFactor);

/**
 * Executes the AdjustContrast operation on a batch of images.
 *
 * Apart from input and output image batches, all parameters are the same as \ref cvcudaAdjustContrastSubmit.
 *
 * @param[in] in Input image batch.
 *
 * @param[out] out Output image batch.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAdjustContrastVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                            NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                            double contrastFactor);

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* CVCUDA_ADJUST_CONTRAST_H */
