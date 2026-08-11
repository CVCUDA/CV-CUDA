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
 * @file OpAutoContrast.h
 *
 * @brief Maximizes per-channel contrast by stretching each channel to the full dynamic range.
 *
 * AutoContrast remaps every channel independently so its spatial minimum maps to 0 and its spatial
 * maximum maps to the data-type maximum (255 for 8-bit, 65535 for 16-bit, 1.0 for float):
 * @f$ out = \mathrm{clamp}\big((in - lo) \cdot bound / (hi - lo)\big) @f$, where @p lo and @p hi
 * are the per-(image, channel) minimum and maximum over the spatial extent and @p bound is the
 * data-type maximum. Integer results are truncated toward zero after clamping. A channel that is
 * flat (@f$ hi == lo @f$) is left unchanged. For floating-point inputs, only finite pixels
 * contribute to @p lo and @p hi; NaN and infinity pixels are copied unchanged. A channel with no
 * finite pixels is therefore left unchanged.
 *
 * Reference: mimics torchvision.transforms.v2.functional.autocontrast / PIL ImageOps.autocontrast
 * (cutoff = 0) for finite inputs. The explicit non-finite handling above provides deterministic
 * floating-point behavior where those references do not define a useful contrast transform. The
 * remap is exact, so the result is bit-exact with an independent reference.
 *
 * @defgroup NVCV_C_ALGORITHM__AUTO_CONTRAST Auto Contrast
 * @{
 */

#ifndef CVCUDA__AUTO_CONTRAST_H
#define CVCUDA__AUTO_CONTRAST_H

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

/** Constructs an instance of the AutoContrast operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAutoContrastCreate(NVCVOperatorHandle *handle);

/** Executes the AutoContrast operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  CUDA stream capture is not supported.
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC, kNCHW, kCHW]
 *       Number:         At most 65535 samples
 *       Channels:       [1, 3, 4]
 *       Element Type:   Scalar; channels are represented by the C dimension
 *       Pixel Strides:  Width and channel dimensions must be packed; rows and samples may be padded
 *       Addressing:     Dynamic byte strides and the maximum byte offset must fit signed 32-bit
 *       Width:          At most 2147483647 pixels
 *       Height:         At most 262140 pixels
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
 *       Addressing:     Dynamic byte strides and the maximum byte offset must fit signed 32-bit
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
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INVALID_OPERATION The CUDA stream is being captured.
 * @retval #NVCV_ERROR_OVERFLOW         A dynamic stride or maximum byte offset exceeds signed 32-bit addressing.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAutoContrastSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                  NVCVTensorHandle out);

/** Executes the AutoContrast operation on a batch of variable-shaped images on the given cuda stream.
 *  Same data type, channel, and maximum-height (262140 pixels) limitations as cvcudaAutoContrastSubmit.
 *  A batch may contain at most 65535 images.
 *  Images must be either single-plane packed or full-resolution planar with one scalar plane per channel.
 *  Chroma-subsampled, semi-planar, macro-pixel, and extra-channel formats are not supported.
 *  Each image plane's maximum byte offset must fit signed 32-bit addressing.
 *  CUDA stream capture is not supported.
 *
 * @param [in] handle Handle to the operator. Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 * @param [in] in     input image batch.
 * @param [out] out   output image batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INVALID_OPERATION The CUDA stream is being captured.
 * @retval #NVCV_ERROR_OVERFLOW         An image-plane maximum byte offset exceeds signed 32-bit addressing.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaAutoContrastVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                          NVCVImageBatchHandle in, NVCVImageBatchHandle out);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__AUTO_CONTRAST_H */
