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
 * @file OpTvResize.h
 * @brief TorchVision-style spatial resize with half-pixel sampling and optional antialiasing.
 * @defgroup NVCV_C_ALGORITHM_TV_RESIZE TvResize
 * @{
 * Reference: torchvision.transforms.v2.functional.resize (v0.28.0), with the
 * explicit rounding and saturation semantics documented below.
 */

#ifndef CVCUDA_TV_RESIZE_H
#define CVCUDA_TV_RESIZE_H

#include "Operator.h"
#include "Types.h"
#include "Workspace.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs a stateless TvResize operator.
 * @param [out] handle Nonnull location receiving the owned operator handle.
 * @retval NVCV_SUCCESS Success.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Null handle pointer.
 * @retval NVCV_ERROR_OUT_OF_MEMORY Allocation failed.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaTvResizeCreate(NVCVOperatorHandle *handle);

/** Queries caller-owned scratch for equally sized tensor samples.
 * @param [in] handle TvResize operator handle.
 * @param [in] batchSize Sample count in [0,65535].
 * @param [in] srcHeight Positive input height.
 * @param [in] srcWidth Positive input width.
 * @param [in] dstHeight Positive output height.
 * @param [in] dstWidth Positive output width.
 * @param [in] numChannels Positive scalar channel count.
 * @param [in] interpolation NEAREST, NEAREST_FLOOR, LINEAR or CUBIC.
 * @param [in] antialias Enable independently dilated, clipped and normalized axis filters.
 * @param [out] reqOut Nonnull requirements output.
 * All supplied dimensions and modes are validated even for zero workspace. N=0
 * requires zero workspace. Queries have no layout/dtype and cannot enforce their limits.
 * @retval NVCV_SUCCESS Success.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid handle, null output, dimensions, mode or size overflow.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaTvResizeTensorGetWorkspaceRequirements(
    NVCVOperatorHandle handle, int batchSize, int32_t srcHeight, int32_t srcWidth, int32_t dstHeight, int32_t dstWidth,
    int32_t numChannels, NVCVInterpolationType interpolation, bool antialias, NVCVWorkspaceRequirements *reqOut);

/** Queries scratch for a uniform-format variable-shape image batch.
 * The four extent arrays contain batchSize positive entries and must be nonnull
 * even when batchSize=0. Other parameters/statuses follow the tensor query.
 * @param [in] handle TvResize operator handle.
 * @param [in] batchSize Sample count in [0,65535].
 * @param [in] srcHeights Input heights.
 * @param [in] srcWidths Input widths.
 * @param [in] dstHeights Output heights.
 * @param [in] dstWidths Output widths.
 * @param [in] numChannels Channel count of 1, 3 or 4, including when batchSize=0.
 * @param [in] interpolation Supported interpolation mode.
 * @param [in] antialias Enable antialiasing.
 * @param [out] reqOut Nonnull requirements output.
 * @retval NVCV_SUCCESS Success.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid handle, null arrays/output, dimensions, mode or size overflow.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaTvResizeImageBatchGetWorkspaceRequirements(
    NVCVOperatorHandle handle, int batchSize, const int32_t *srcHeights, const int32_t *srcWidths,
    const int32_t *dstHeights, const int32_t *dstWidths, int32_t numChannels, NVCVInterpolationType interpolation,
    bool antialias, NVCVWorkspaceRequirements *reqOut);

/** Enqueues spatial resize on the caller stream without waiting for completion.
 *
 * Limitations:
 * Input:
 *      Data Layout: [HWC, NHWC, CHW, NCHW]
 *      Channels: [1, 2, 3, 4] for HWC/NHWC; any positive C with N*C<=65535 for CHW/NCHW.
 *      1<=N<=65535; unbatched layouts imply N=1. H/W/C must be positive.
 *
 *      Data Type      | Allowed
 *      -------------- | -------
 *      8bit Unsigned  | Yes
 *      8bit Signed    | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      16bit Float    | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | No
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output: same layouts, channels and scalar types as input. Vector element types
 * are not supported. CUDA-accessible positive padded strides are supported with
 * checked 64-bit address arithmetic; negative/overlapping-element strides are invalid.
 *
 * Input/Output dependency:
 *      Property      | Input == Output
 *      ------------- | ---------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | No
 *      Height        | No
 *
 * NEAREST gathers floor((dst+0.5)*scale), NEAREST_FLOOR gathers floor(dst*scale),
 * with float32 scale=S/D and coordinates, clamped to S-1. Both preserve sampled bits
 * and ignore antialias. LINEAR/CUBIC use half-pixel centers and float32 accumulation.
 * Plain cubic uses Keys a=-0.75 and clamped taps. Antialias uses per-axis dilation
 * max(1,S/D), clips taps to the image, and normalizes surviving weights; cubic uses
 * a=-0.5 even when enlarging. Singleton axes are supported; identity is bit-exact.
 * Integers are rounded ties-to-even and saturated, F16 rounds to nearest-even.
 * Interpolation guarantees apply to finite inputs with finite representable results;
 * no nonfinite pixel scan is performed. Unlike external TorchVision casting, U16/S16
 * cubic overshoot saturates rather than wraps.
 *
 * Source/destination storage must not overlap, even for identity (caller precondition).
 * The shared Tensor API rejects zero extents at construction; empty Tensor submits
 * are not representable. N=0 workspace queries still return zero workspace.
 * All metadata is validated before output work. The caller keeps inputs, outputs
 * and workspace alive until stream completion. No hidden device allocation or host synchronization.
 *
 * @param [in] handle TvResize operator handle.
 * @param [in] stream CUDA stream.
 * @param [in] workspace Queried scratch with sufficient size/alignment; may be null only when zero scratch is needed.
 * @param [in] in Source tensor.
 * @param [out] out Destination tensor.
 * @param [in] interpolation NEAREST, NEAREST_FLOOR, LINEAR or CUBIC.
 * @param [in] antialias Enable antialiasing (ignored for nearest).
 * @retval NVCV_SUCCESS Work enqueued successfully.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid handle, metadata, mode, workspace or resource-size overflow.
 * @retval NVCV_ERROR_OUT_OF_MEMORY Resource allocation failed.
 * @retval NVCV_ERROR_INTERNAL Internal error. Device execution errors may be asynchronous.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaTvResizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                              const NVCVWorkspace *workspace, NVCVTensorHandle in, NVCVTensorHandle out,
                                              NVCVInterpolationType interpolation, bool antialias);

/** Enqueues variable-shape resize with the sampling/workspace/lifetime contract of cvcudaTvResizeSubmit.
 * Limitations: CUDA-accessible pitch-linear single-plane interleaved images, channels
 * [1,3,4], scalar U8/U16/S16/F16/F32, N<=65535. Each batch must have a uniform format,
 * matching between input and output. Counts match; only spatial sizes may differ.
 * Two-channel, multiplane and mixed-format batches are rejected. Matched empty
 * batches succeed with zero workspace and no launches.
 * @param [in] handle TvResize operator handle.
 * @param [in] stream CUDA stream.
 * @param [in] workspace Queried scratch, or null if no scratch is needed.
 * @param [in] in Source image batch.
 * @param [out] out Destination image batch.
 * @param [in] interpolation Supported interpolation mode.
 * @param [in] antialias Enable antialiasing.
 * @retval NVCV_SUCCESS Work enqueued successfully.
 * @retval NVCV_ERROR_INVALID_ARGUMENT Invalid handle, metadata, mode, workspace or size overflow.
 * @retval NVCV_ERROR_OUT_OF_MEMORY Resource allocation failed.
 * @retval NVCV_ERROR_INTERNAL Internal error; device execution errors may be asynchronous.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaTvResizeImageBatchSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        const NVCVWorkspace *workspace, NVCVImageBatchHandle in,
                                                        NVCVImageBatchHandle out, NVCVInterpolationType interpolation,
                                                        bool antialias);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // CVCUDA_TV_RESIZE_H
