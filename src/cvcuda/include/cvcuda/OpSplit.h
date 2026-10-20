/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file OpSplit.h
 * @brief Copies tensor or variable-shape image channels into separate single-channel outputs, without conversion.
 * Reference: Pillow 12.3.0 Image.Image.split(), https://github.com/python-pillow/Pillow/blob/12.3.0/src/PIL/Image.py
 * @defgroup NVCV_C_ALGORITHM_SPLIT Split
 * @{
 */

#ifndef CVCUDA_SPLIT_H
#define CVCUDA_SPLIT_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Tensor.h>
#include <nvcv/TensorBatch.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Creates a stateless Split operator.
 * @param[out] handle Non-null destination for the operator handle.
 * @retval NVCV_SUCCESS Operator successfully created.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY Insufficient memory to create the operator.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSplitCreate(NVCVOperatorHandle *handle);

/**
 * Copies input channel c to output tensor c, preserving every element's bits.
 * The output TensorBatch contains channels, not samples: each tensor retains the
 * input's sample count, height and width, with an explicit channel dimension of one.
 * C=1 still copies data. There is no color conversion or alpha processing.
 * This entry point operates on tensors; see cvcudaSplitVarShapeSubmit for image batches.
 * Submission is asynchronous; all tensor storage must remain alive until completion.
 *
 * Limitations:
 *
 * Tensor input:
 *      Data Layout: [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels: [1, 2, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | Yes
 *      32bit Signed   | Yes
 *      16bit Float    | Yes
 *      32bit Float    | Yes
 *      64bit Float    | Yes
 *
 * Tensor output (each band):
 *      Data Layout: [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels: [1]
 *      Data Type: same scalar type as input; vector/packed/complex types are not supported.
 *
 * All dimensions must be positive. Input and output ranks must match. Output
 * bands must have identical shape, layout and dtype; their layout may differ
 * from the input's layout. The number of bands must equal the input channel count.
 * CUDA-accessible, pitch-linear tensors are required. Base addresses and strides
 * must be aligned to the scalar element size, and all strides must be positive.
 * Each non-singleton dimension's stride must cover its inner dimensions' byte
 * span in layout order; singleton strides need not. Row/sample padding is allowed.
 * Input/output byte spans and output/output byte spans must not overlap, including
 * padding inside those spans. Padding is not written. No resizing or broadcasting occurs.
 * Floating-point and multichannel non-U8 support extend Pillow's pixel-copy semantics.
 *
 * @param[in] handle Split operator handle, not null.
 * @param[in] stream CUDA stream, including the default stream.
 * @param[in] in Input tensor, not null.
 * @param[out] out Prepopulated batch of destination band tensors, not null.
 * @retval NVCV_SUCCESS Work successfully submitted.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid metadata, unsupported storage, or overlapping buffers.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSplitSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                           NVCVTensorBatchHandle out);

/**
 * Copies physical storage channel c of input image n to image n in output batch c.
 * All element bits are preserved, including floating-point payloads. C=1 still
 * copies data; outputs are not views. Channel order is the component order within
 * an interleaved pixel or plane order for planar images. Color models, swizzles
 * and alpha metadata are not interpreted; no conversion or alpha processing occurs.
 * Submission is asynchronous. All batches, images, image/batch descriptors and
 * backing storage must remain alive and must not be mutated by the caller until
 * completion. The host array of batch handles need only survive this call.
 *
 * Limitations:
 *
 * ImageBatchVarShape input:
 *      Data Layout: one interleaved plane, or C full-resolution single-channel planes.
 *      Channels: [1, 2, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | Yes
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | Yes
 *      32bit Signed   | Yes
 *      16bit Float    | Yes
 *      32bit Float    | Yes
 *      64bit Float    | Yes
 *
 * ImageBatchVarShape output (each band):
 *      Data Layout: one single-channel plane.
 *      Channels: [1]
 *      Data Type: same scalar type as input.
 *
 * Each batch must be nonempty and have a uniform image format. Every batch has
 * the same number N of images; numOutputs equals the input channel count C.
 * Semantic format tags may differ across band batches if their scalar dtype matches.
 * Image sizes must be positive and match at each corresponding index across
 * batches, but may vary across indices. Pitches and allocations may vary across
 * images and planes. Plane data kinds and packings must describe contiguous,
 * byte-aligned, equal-width components of exactly one of the listed scalar types,
 * without intra-pixel padding. Subsampled, sub-byte, unequal-component and hybrid
 * plane packings are not supported. Extra-channel metadata must have
 * NVCVExtraChannelInfo.numChannels == 0. CUDA-accessible,
 * pitch-linear storage is required; CUDA arrays are not supported. Plane bases
 * and row pitches must be aligned to the scalar size, and each positive row pitch
 * must cover a full row. Every write/read and write/write overlap across the
 * entire submission is rejected, including across image indices, bands and planes
 * and padding inside their byte spans. Repeated read-only sources are permitted.
 * Padding is not written. No resizing or broadcasting occurs.
 *
 * @param[in] handle Split operator handle, not null.
 * @param[in] stream CUDA stream, including the default stream.
 * @param[in] in Nonempty source image batch, not null.
 * @param[out] out Host array of numOutputs prepopulated destination band batch handles, none null.
 * @param[in] numOutputs Number of band batches, equal to C (1-4).
 * @retval NVCV_SUCCESS Work successfully submitted.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid metadata, unsupported storage, or overlapping buffers.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaSplitVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   NVCVImageBatchHandle in, const NVCVImageBatchHandle *out,
                                                   int32_t numOutputs);

#ifdef __cplusplus
}
#endif

/** @} */
#endif
