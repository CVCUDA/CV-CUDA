/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file OsdRenderer.hpp
 *
 * @brief Host interface to the cuOSD rasterizer shared by the OSD and BndBox operators.
 */

#ifndef CVCUDA_PRIV_OSD_RENDERER_HPP
#define CVCUDA_PRIV_OSD_RENDERER_HPP

#include "OsdContext.hpp"
#include "Types.hpp"

#include <cuda_runtime.h>
#include <nvcv/DataType.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/TensorShape.hpp>

namespace cvcuda::priv::osd {

/**
 * @brief Whether @p layout is planar, the case both operators bridge to the rasterizer by
 *        reformatting into the matching interleaved layout.
 */
inline bool IsPlanarLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

/**
 * @brief Whether @p layout is interleaved, the only kind the rasterizer draws on directly.
 */
inline bool IsInterleavedLayout(nvcv::TensorLayout layout)
{
    return layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC;
}

/**
 * @brief The interleaved shape a planar @p access bridges through: NCHW becomes NHWC, CHW becomes
 *        HWC. @p opName only names the operator in the rejection message for any other layout.
 */
nvcv::TensorShape InterleavedShapeFor(nvcv::TensorLayout layout, const nvcv::TensorDataAccessStridedImagePlanar &access,
                                      const char *opName);

/**
 * @brief Whether @p dtype is the 8-bit unsigned element type the rasterizer accepts.
 *
 * Reproduces nvcv::legacy::helpers::GetLegacyDataType()'s classification exactly: it rejects a
 * non-uniform channel bit-depth first, then classifies on (data kind, bits per channel) and
 * rejects every pair its enum has no name for. Only (UNSIGNED, 8) mapped to the kCV_8U that both
 * operators required, so this returns true for that pair alone -- and, like the helper, throws
 * rather than returning false for the pairs the helper never classified.
 */
bool IsLegacyU8(nvcv::DataType dtype);

/**
 * @brief Reject, in the order the legacy operator rejected them, every input/output pair the
 *        rasterizer cannot draw on -- a non-interleaved layout, differing data types, non-8-bit
 *        unsigned channels, a non-image-like input -- and return the validated input accessor.
 *
 * This prefix is common to both operators; the checks that follow it (channel count, element or
 * box batch, output shape) are ordered differently per operator and stay at the call site.
 */
nvcv::TensorDataAccessStridedImagePlanar ValidateInterleavedU8Pair(const nvcv::TensorDataStridedCuda &inData,
                                                                   const nvcv::TensorDataStridedCuda &outData);

/**
 * @brief Queue the OSD draw commands for every element in @p elements onto @p context.
 */
void DrawElements(cuOSDContext_t context, int width, int height, NVCVElementsImpl *elements);

/**
 * @brief Queue one rectangle draw command per bounding box in @p bboxes onto @p context.
 */
void DrawBndBoxes(cuOSDContext_t context, int width, int height, NVCVBndBoxesImpl *bboxes);

/**
 * @brief Upload the queued commands, rasterize them over @p inData into @p outData, and reset
 *        @p context's command list.
 *
 * @p inData and @p outData must be interleaved 8-bit tensors of identical shape with three or
 * four channels; drawing happens in place when they share a base pointer.
 */
void Render(cuOSDContext_t context, const nvcv::TensorDataStridedCuda &inData,
            const nvcv::TensorDataStridedCuda &outData, int batch, int channels, int rows, int cols,
            cudaStream_t stream);

} // namespace cvcuda::priv::osd

#endif // CVCUDA_PRIV_OSD_RENDERER_HPP
