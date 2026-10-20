/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file OpMerge.h
 * @brief Copies ordered single-channel tensors or variable-shape image batches into channels, without conversion.
 * Reference: Pillow 12.3.0 Image.merge(), https://github.com/python-pillow/Pillow/blob/12.3.0/src/PIL/Image.py
 * @defgroup NVCV_C_ALGORITHM_MERGE Merge
 * @{
 */

#ifndef CVCUDA_MERGE_H
#define CVCUDA_MERGE_H

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

/** Creates a stateless Merge operator.
 * @param[out] handle Non-null destination for the operator handle.
 * @retval NVCV_SUCCESS Operator successfully created.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval NVCV_ERROR_OUT_OF_MEMORY Insufficient memory to create the operator.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMergeCreate(NVCVOperatorHandle *handle);

/**
 * Copies input tensor c to output channel c, preserving every element's bits.
 * TensorBatch entries represent channels, not samples. Each input has an explicit
 * channel dimension of one. Repeated read-only inputs are permitted, including
 * merging the same grayscale tensor three times. C=1 still copies data.
 * There is no blending, color conversion, metadata interpretation or alpha processing.
 * This entry point operates on tensors; see cvcudaMergeVarShapeSubmit for image batches.
 * Submission is asynchronous; all tensor storage must remain alive until completion.
 *
 * Limitations:
 *
 * Tensor input (each band):
 *      Data Layout: [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels: [1]
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
 * Tensor output:
 *      Data Layout: [kNHWC, kHWC, kNCHW, kCHW]
 *      Channels: [1, 2, 3, 4]
 *      Data Type: same scalar type as input; vector/packed/complex types are not supported.
 *
 * All dimensions must be positive. All inputs must have identical shape, layout
 * and dtype. Output rank, sample count, height, width and dtype must match the
 * inputs; its layout may differ. The number of inputs must equal the output channel count.
 * CUDA-accessible, pitch-linear tensors are required. Base addresses and strides
 * must be aligned to the scalar element size, and all strides must be positive.
 * Each non-singleton dimension's stride must cover its inner dimensions' byte
 * span in layout order; singleton strides need not. Row/sample padding is allowed.
 * Input/output byte spans must not overlap, including padding inside those spans.
 * Padding is not written. No resizing or broadcasting occurs.
 * Floating-point and multichannel non-U8 support extend Pillow's pixel-copy semantics.
 *
 * @param[in] handle Merge operator handle, not null.
 * @param[in] stream CUDA stream, including the default stream.
 * @param[in] in Prepopulated batch of source band tensors, not null.
 * @param[out] out Destination tensor, not null.
 * @retval NVCV_SUCCESS Work successfully submitted.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid metadata, unsupported storage, or overlapping buffers.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMergeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in,
                                           NVCVTensorHandle out);

/**
 * Copies image n in input batch c to physical storage channel c of output image n.
 * All element bits are preserved, including floating-point payloads. C=1 still
 * copies data; output images are not views. Channel order is the component order
 * within an interleaved pixel or plane order for planar images. Color models,
 * swizzles and alpha metadata are not interpreted; no conversion or alpha processing
 * occurs. Repeated read-only inputs are permitted, including the same band batch
 * in multiple positions. Submission is asynchronous. All batches, images,
 * image/batch descriptors and backing storage must remain alive and must not be
 * mutated by the caller until completion. The host array of batch handles need
 * only survive this call.
 *
 * Limitations:
 *
 * ImageBatchVarShape input (each band):
 *      Data Layout: one single-channel plane.
 *      Channels: [1]
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
 * ImageBatchVarShape output:
 *      Data Layout: one interleaved plane, or C full-resolution single-channel planes.
 *      Channels: [1, 2, 3, 4]
 *      Data Type: same scalar type as input.
 *
 * Each batch must be nonempty and have a uniform image format. Every batch has
 * the same number N of images; numInputs equals the output channel count C.
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
 * and padding inside their byte spans. Padding is not written. No resizing or
 * broadcasting occurs.
 *
 * @param[in] handle Merge operator handle, not null.
 * @param[in] stream CUDA stream, including the default stream.
 * @param[in] in Host array of numInputs nonempty source band batch handles, none null.
 * @param[in] numInputs Number of band batches, equal to C (1-4).
 * @param[out] out Prepopulated destination image batch, not null.
 * @retval NVCV_SUCCESS Work successfully submitted.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid metadata, unsupported storage, or overlapping buffers.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaMergeVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                   const NVCVImageBatchHandle *in, int32_t numInputs,
                                                   NVCVImageBatchHandle out);

#ifdef __cplusplus
}
#endif

/** @} */
#endif
