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

#ifndef CVCUDA_PRIV_PLANAR_TENSOR_VIEW_HPP
#define CVCUDA_PRIV_PLANAR_TENSOR_VIEW_HPP

#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <optional>
#include <utility>

namespace cvcuda::priv {

// Build a single-channel (N*C, H, W, 1) NHWC view of a packed planar (NCHW/CHW) tensor.
//
// Operators that treat color channels independently (Resize, ConvertTo, Remap, ...) can process a
// planar image by viewing each (sample, channel) plane as a single-channel image: flattening the
// planes into N*C "samples" lets the existing interleaved single-channel kernel run unchanged. The
// view treats plane (n, c) as sample index n*C + c at byte offset (n*C + c) * chStride, which
// matches the real layout only when the channel planes are tightly packed
// (sampleStride == numChannels * chStride). That always holds within a single sample (n == 0); a
// batched tensor additionally requires tight packing across samples, which is validated here.
inline nvcv::TensorDataStridedCuda PlanarAsSingleChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t numSamples  = access.numSamples();
    const int64_t numChannels = access.numChannels();
    const int64_t numRows     = access.numRows();
    const int64_t numCols     = access.numCols();

    // A single sample may legitimately have sampleStride > C*chStride due to allocation alignment,
    // so the uniform-stride flatten is only valid for batches when the planes are tightly packed.
    if (numSamples > 1 && access.sampleStride() != numChannels * access.chStride())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar processing of a batched tensor requires tightly packed channel planes");
    }

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access.chStride();  // N*C flattened planes
    buf.strides[1] = access.rowStride(); // H
    buf.strides[2] = access.colStride(); // W
    buf.strides[3] = access.colStride(); // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{numSamples * numChannels, numRows, numCols, 1}, "NHWC"},
        data.dtype(), buf
    };
}

// Validate an input/output tensor pair for an independent-channel op that supports planar layout,
// and, when the pair is planar, return flattened single-channel (N*C, H, W, 1) views of each so the
// caller can run its interleaved single-channel kernel unchanged. Returns std::nullopt for the
// interleaved case (the caller runs its normal path). Throws ERROR_INVALID_ARGUMENT on a mixed
// planar/interleaved pair, an unsupported channel count (only 1/3/4; 2-channel planar is
// unsupported), a sample/channel-count mismatch, or a flattened plane count above the CUDA grid-z
// limit. Layout is inferred from the tensors, so this adds planar as a capability with no API change.
inline std::optional<std::pair<nvcv::TensorDataStridedCuda, nvcv::TensorDataStridedCuda>>
    PlanarSingleChannelViews(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData)
{
    const nvcv::TensorLayout inLayout  = inData.layout();
    const nvcv::TensorLayout outLayout = outData.layout();
    const bool               inPlanar  = (inLayout == nvcv::TENSOR_NCHW || inLayout == nvcv::TENSOR_CHW);

    if (const bool outPlanar = (outLayout == nvcv::TENSOR_NCHW || outLayout == nvcv::TENSOR_CHW);
        inPlanar != outPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved (N)HWC or both planar (N)CHW");
    }
    if (!inPlanar)
    {
        return std::nullopt;
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar processing requires image-like NCHW/CHW tensors");
    }

    const int64_t numChannels = inAccess->numChannels();
    if (numChannels < 1 || numChannels > 4 || numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels (2-channel planar is unsupported)");
    }
    if (inAccess->numSamples() != outAccess->numSamples() || numChannels != outAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same sample and channel counts");
    }

    if (const int64_t planarBatch = numChannels * inAccess->numSamples(); planarBatch > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar processing requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
    }

    return std::make_pair(PlanarAsSingleChannelView(inData, *inAccess), PlanarAsSingleChannelView(outData, *outAccess));
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_PLANAR_TENSOR_VIEW_HPP
