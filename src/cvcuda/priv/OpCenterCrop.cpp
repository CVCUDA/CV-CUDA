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

#include "OpCenterCrop.hpp"

#include "Nvtx.hpp"
#include "PlanarTensorView.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cuda_runtime_api.h>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstddef>

namespace cvcuda::priv {

namespace legacy = nvcv::legacy::cuda_op;

namespace {

constexpr int kBitsPerByte = 8;

bool TryCopyDenseByteNHWCCenterCrop(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                    const nvcv::TensorDataStridedCuda &outData, const nvcv::Size2D &cropSize)
{
    if (inData.rank() != 4 || outData.rank() != 4 || inData.layout() != nvcv::TENSOR_NHWC
        || outData.layout() != nvcv::TENSOR_NHWC || inData.dtype() != outData.dtype())
    {
        return false;
    }

    const auto bitsPerChannel = inData.dtype().bitsPerChannel();
    // Reference CI showed that cudaMemcpy3DAsync regresses the A100 float32 rows by 22% and the
    // H100 byte-wide three-channel rows by 8%, while local SM86 benchmarking showed the byte-wide
    // four-channel row regressing by 8%. Keep every interleaved non-scalar case on the legacy
    // kernel. Native-planar U8 is flattened to scalar planes before reaching this check.
    if (bitsPerChannel[0] != kBitsPerByte)
    {
        return false;
    }
    const int64_t bytesPerChannel = bitsPerChannel[0] / kBitsPerByte;

    const int64_t inBatch = inData.shape(0);
    const int64_t inRows  = inData.shape(1);
    const int64_t inCols  = inData.shape(2);
    const int64_t inCh    = inData.shape(3);

    if (inCh != 1)
    {
        return false;
    }

    const int64_t outBatch = outData.shape(0);
    const int64_t outRows  = outData.shape(1);
    const int64_t outCols  = outData.shape(2);

    if (const int64_t outCh = outData.shape(3); cropSize.h <= 0 || cropSize.w <= 0 || inBatch != outBatch
                                                || inCh != outCh || inCh < 1 || outRows != cropSize.h
                                                || outCols != cropSize.w || cropSize.h > inRows || cropSize.w > inCols)
    {
        return false;
    }

    const int64_t pixelBytes = inCh * bytesPerChannel;
    if (inData.stride(3) != bytesPerChannel || outData.stride(3) != bytesPerChannel || inData.stride(2) != pixelBytes
        || outData.stride(2) != pixelBytes || inData.stride(1) < inCols * pixelBytes
        || outData.stride(1) < outCols * pixelBytes || inData.stride(0) != inData.stride(1) * inRows
        || outData.stride(0) != outData.stride(1) * outRows)
    {
        return false;
    }

    const int64_t top  = (inRows - cropSize.h) / 2;
    const int64_t left = (inCols - cropSize.w) / 2;

    cudaMemcpy3DParms params{};
    params.srcPtr
        = make_cudaPitchedPtr(inData.basePtr(), static_cast<std::size_t>(inData.stride(1)),
                              static_cast<std::size_t>(inCols * pixelBytes), static_cast<std::size_t>(inRows));
    params.dstPtr
        = make_cudaPitchedPtr(outData.basePtr(), static_cast<std::size_t>(outData.stride(1)),
                              static_cast<std::size_t>(outCols * pixelBytes), static_cast<std::size_t>(outRows));
    params.srcPos = make_cudaPos(static_cast<std::size_t>(left * pixelBytes), static_cast<std::size_t>(top), 0);
    params.dstPos = make_cudaPos(0, 0, 0);
    params.extent = make_cudaExtent(static_cast<std::size_t>(cropSize.w * pixelBytes),
                                    static_cast<std::size_t>(cropSize.h), static_cast<std::size_t>(inBatch));
    params.kind   = cudaMemcpyDeviceToDevice;

    NVCV_CHECK_THROW(cudaMemcpy3DAsync(&params, stream));
    return true;
}

} // namespace

CenterCrop::CenterCrop()
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<legacy::CenterCrop>(maxIn, maxOut);
}

void CenterCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const nvcv::Size2D &cropSize) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CenterCrop::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    if (TryCopyDenseByteNHWCCenterCrop(stream, *inData, *outData, cropSize))
    {
        return;
    }

    // Center-cropping copies each channel plane independently and identically, so a planar (NCHW/CHW)
    // image is just N*C single-channel planes: flatten them into the sample dimension and reuse the
    // dense-copy path or legacy interleaved single-channel kernel unchanged, producing bit-exact
    // planar output.
    if (auto planarViews = PlanarSingleChannelViews(*inData, *outData))
    {
        if (TryCopyDenseByteNHWCCenterCrop(stream, planarViews->first, planarViews->second, cropSize))
        {
            return;
        }

        NVCV_CHECK_THROW(m_legacyOp->infer(planarViews->first, planarViews->second, cropSize.h, cropSize.w, stream));
        return;
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, cropSize.h, cropSize.w, stream));
}

} // namespace cvcuda::priv
