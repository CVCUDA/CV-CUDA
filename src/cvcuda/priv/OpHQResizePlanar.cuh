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

#ifndef CVCUDA_PRIV_OP_HQ_RESIZE_PLANAR_CUH
#define CVCUDA_PRIV_OP_HQ_RESIZE_PLANAR_CUH

#include "PlanarTensorView.hpp"

// Shared helpers for planar (NCHW/CHW) HQResize support. HQResize resizes spatial axes only and
// treats channels independently, so a C-plane planar sample is equivalent to C single-channel
// samples. The tensor/tensor-batch paths reinterpret each plane as a single-channel image
// (PlanarAsSingleChannelView); the var-shape path decodes the flattened plane index in-adapter
// (DecodePlane). Both feed the existing single-channel kernels, so the result is bit-identical to
// the interleaved path and no new kernels are needed.
namespace cvcuda::priv::hq_resize::planar {

// Map an expanded sample index `nc` (running over images * channels, plane-minor) to its
// {image, plane} pair. Used by the var-shape adapter when it decodes grid.y / CurrentSample().
__host__ __device__ __forceinline__ int2 DecodePlane(int nc, int channels)
{
    return int2{nc / channels, nc % channels};
}

// Build an (N*C, H, W, 1) NHWC view of a packed planar (NCHW/CHW) tensor so the interleaved
// single-channel resize path can process each channel plane as an independent image. Requires the
// channel planes to be tightly packed for batched tensors (sampleStride == numChannels * chStride);
// a single sample may legitimately have a larger sampleStride due to allocation alignment.
inline nvcv::TensorDataStridedCuda PlanarAsSingleChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t numSamples  = access.numSamples();
    const int64_t numChannels = access.numChannels();

    if (numSamples > 1 && access.sampleStride() != numChannels * access.chStride())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar HQResize of a batched tensor requires tightly packed channel planes");
    }

    return ::cvcuda::priv::PlanarAsSingleChannelView(data, access);
}

} // namespace cvcuda::priv::hq_resize::planar

#endif // CVCUDA_PRIV_OP_HQ_RESIZE_PLANAR_CUH
