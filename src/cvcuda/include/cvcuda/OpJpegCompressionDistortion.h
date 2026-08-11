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
 * @file OpJpegCompressionDistortion.h
 *
 * @brief C API for JpegCompressionDistortion: simulates JPEG compression artifacts (full-range
 *        JFIF YCbCr, 4:2:0 chroma subsampling, per-8x8-block DCT quantization) without a codec.
 * @defgroup NVCV_C_ALGORITHM__JPEG_COMPRESSION_DISTORTION Jpeg Compression Distortion
 * @{
 */

#ifndef CVCUDA__JPEG_COMPRESSION_DISTORTION_H
#define CVCUDA__JPEG_COMPRESSION_DISTORTION_H

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

/** Constructs an instance of the JpegCompressionDistortion operator.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaJpegCompressionDistortionCreate(NVCVOperatorHandle *handle);

/** Executes the JpegCompressionDistortion operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  JpegCompressionDistortion simulates the artifacts of a JPEG compression/decompression round
 *  trip: 3-channel RGB images are converted to full-range JFIF YCbCr, chroma is 4:2:0 subsampled
 *  (2x2 box average on RGB, nearest-neighbor upsampling on reconstruction), and every 8x8 block of
 *  each plane goes through a DCT, quantization with the JPEG Annex-K tables scaled by the libjpeg
 *  quality mapping, dequantization and inverse DCT. 1-channel images are treated as a bare luma
 *  plane (DCT/quantization only, no color conversion or chroma path). Entropy coding is not
 *  simulated, so results approximate — but do not bit-match — a real JPEG codec round trip.
 *
 *  Reference: mimics `torchvision.transforms.v2.functional.jpeg` (approximately; torchvision runs
 *  a real libjpeg round trip on the CPU). Algorithm ported from NVIDIA DALI's
 *  `JpegCompressionDistortion` GPU kernel (dali/kernels/imgproc/jpeg, Apache-2.0).
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
 *       32bit Float    | No
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
 *  quality Tensor
 *
 *      Must be rank-1 ('N') and packed, with one value per image (length == batch size).
 *      Data Type must be TYPE_S32.
 *      Values are clamped to [1, 100] on the device (matching NVIDIA DALI); they are not
 *      validated on the host.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] quality Per-image JPEG quality tensor, from 1 (strongest distortion) to 100
 *                     (weakest). See the quality Tensor requirements above.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaJpegCompressionDistortionSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                               NVCVTensorHandle in, NVCVTensorHandle out,
                                                               NVCVTensorHandle quality);

/** Executes the JpegCompressionDistortion operation with a single quality for the whole batch.
 *
 *  This parameter-tensor-free variant passes \p quality by value, avoiding a device parameter
 *  tensor. Semantics and Limitations are identical to #cvcudaJpegCompressionDistortionSubmit,
 *  except that the scalar quality is validated on the host: values outside [1, 100] are rejected
 *  with #NVCV_ERROR_INVALID_ARGUMENT (matching torchvision's argument validation) instead of being
 *  clamped.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] quality JPEG quality applied to all images, from 1 (strongest distortion) to 100
 *                     (weakest). Must be in [1, 100].
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaJpegCompressionDistortionScalarSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                     NVCVTensorHandle in, NVCVTensorHandle out,
                                                                     int32_t quality);

/** Executes the JpegCompressionDistortion operation on a variable-shape image batch.
 *
 *  Semantics, data-type and channel constraints match #cvcudaJpegCompressionDistortionSubmit.
 *  All images in a batch must share one image format, which must be RGB(8) for 3 channels (packed
 *  or planar) or U8/Y8 for 1 channel, without chroma subsampling or extra channels.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input image batch.
 *
 * @param [out] out Output image batch.
 *
 * @param [in] quality Per-image JPEG quality tensor, from 1 (strongest distortion) to 100
 *                     (weakest). Must be rank-1, packed, TYPE_S32, with one value per image;
 *                     values are clamped to [1, 100] on the device.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaJpegCompressionDistortionVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                                       NVCVImageBatchHandle in,
                                                                       NVCVImageBatchHandle out,
                                                                       NVCVTensorHandle     quality);

/** Executes the JpegCompressionDistortion operation on a variable-shape image batch with a single
 *  quality for the whole batch.
 *
 *  Semantics match #cvcudaJpegCompressionDistortionVarShapeSubmit; the scalar quality is validated
 *  on the host like #cvcudaJpegCompressionDistortionScalarSubmit.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input image batch.
 *
 * @param [out] out Output image batch.
 *
 * @param [in] quality JPEG quality applied to all images, from 1 (strongest distortion) to 100
 *                     (weakest). Must be in [1, 100].
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaJpegCompressionDistortionVarShapeScalarSubmit(NVCVOperatorHandle   handle,
                                                                             cudaStream_t         stream,
                                                                             NVCVImageBatchHandle in,
                                                                             NVCVImageBatchHandle out, int32_t quality);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__JPEG_COMPRESSION_DISTORTION_H */
