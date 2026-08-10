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

#include "OpBoxBlur.hpp"

#include "Nvtx.hpp"
#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <cvcuda/priv/Types.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace cvcuda::priv {

namespace legacy  = nvcv::legacy::cuda_op;
namespace helpers = nvcv::legacy::helpers;

namespace {

enum class DenseCopyLayout
{
    INTERLEAVED,
    PLANAR
};

std::optional<DenseCopyLayout> GetDenseCopyLayout(legacy::DataFormat inputFormat, legacy::DataFormat outputFormat)
{
    if (inputFormat == legacy::kNHWC && outputFormat == legacy::kNHWC)
    {
        return DenseCopyLayout::INTERLEAVED;
    }
    if (inputFormat == legacy::kNCHW && outputFormat == legacy::kNCHW)
    {
        return DenseCopyLayout::PLANAR;
    }
    return std::nullopt;
}

bool HaveSameImageShape(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                        const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    return inAccess.numSamples() == outAccess.numSamples() && inAccess.numRows() == outAccess.numRows()
        && inAccess.numCols() == outAccess.numCols() && inAccess.numChannels() == outAccess.numChannels();
}

bool HaveSupportedStrides(DenseCopyLayout layout, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int64_t rows, int64_t channels,
                          int64_t rowBytes)
{
    if (layout == DenseCopyLayout::INTERLEAVED)
    {
        return inAccess.chStride() == 1 && outAccess.chStride() == 1 && inAccess.colStride() == channels
            && outAccess.colStride() == channels;
    }

    return inAccess.colStride() == 1 && outAccess.colStride() == 1 && inAccess.chStride() == rows * inAccess.rowStride()
        && outAccess.chStride() == rows * outAccess.rowStride() && inAccess.rowStride() >= rowBytes
        && outAccess.rowStride() >= rowBytes;
}

int64_t RowsPerSample(int64_t sampleStride, int64_t rowStride, int64_t copyRows, int64_t samples)
{
    if (sampleStride == 0)
    {
        return samples == 1 ? copyRows : 0;
    }
    if (rowStride <= 0 || sampleStride % rowStride != 0)
    {
        return 0;
    }
    return sampleStride / rowStride;
}

bool BuffersOverlap(const nvcv::Byte *src, const nvcv::Byte *dst,
                    const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                    const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int64_t samples, int64_t copyRows,
                    int64_t rowBytes)
{
    const auto srcStart = reinterpret_cast<std::uintptr_t>(src);
    const auto dstStart = reinterpret_cast<std::uintptr_t>(dst);
    const auto srcSpan  = static_cast<std::uintptr_t>((samples - 1) * inAccess.sampleStride()
                                                     + (copyRows - 1) * inAccess.rowStride() + rowBytes);
    const auto dstSpan  = static_cast<std::uintptr_t>((samples - 1) * outAccess.sampleStride()
                                                     + (copyRows - 1) * outAccess.rowStride() + rowBytes);
    return srcStart < dstStart + dstSpan && dstStart < srcStart + srcSpan;
}

std::optional<int64_t> DenseCopyByteCount(int64_t copyRows, int64_t rowBytes, int64_t samples)
{
    if (copyRows > std::numeric_limits<int64_t>::max() / rowBytes)
    {
        return std::nullopt;
    }

    const int64_t sampleBytes = copyRows * rowBytes;
    if (samples > std::numeric_limits<int64_t>::max() / sampleBytes)
    {
        return std::nullopt;
    }
    return sampleBytes * samples;
}

bool CanUseLinearCopy(DenseCopyLayout layout, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                      const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int64_t samples, int64_t rowBytes,
                      int64_t sampleBytes)
{
    return layout == DenseCopyLayout::INTERLEAVED && inAccess.rowStride() == rowBytes
        && outAccess.rowStride() == rowBytes
        && (samples == 1 || (inAccess.sampleStride() == sampleBytes && outAccess.sampleStride() == sampleBytes));
}

bool TryDenseCopy(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                  const nvcv::TensorDataStridedCuda &outData, NVCVBlurBoxesI bboxes)
{
    if (bboxes == nullptr || inData.dtype() != outData.dtype())
    {
        return false;
    }

    legacy::DataFormat inputFormat;
    legacy::DataFormat outputFormat;
    legacy::DataType   dataType;
    try
    {
        inputFormat  = helpers::GetLegacyDataFormat(inData.layout());
        outputFormat = helpers::GetLegacyDataFormat(outData.layout());
        dataType     = helpers::GetLegacyDataType(inData.dtype());
    }
    catch (const nvcv::Exception &)
    {
        return false;
    }

    const auto copyLayout = GetDenseCopyLayout(inputFormat, outputFormat);
    if (dataType != legacy::kCV_8U || !copyLayout)
    {
        return false;
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess || !HaveSameImageShape(*inAccess, *outAccess))
    {
        return false;
    }

    if (const auto *blurBoxes = reinterpret_cast<const NVCVBlurBoxesImpl *>(bboxes);
        blurBoxes->batch() != inAccess->numSamples())
    {
        return false;
    }

    const int64_t samples  = inAccess->numSamples();
    const int64_t rows     = inAccess->numRows();
    const int64_t cols     = inAccess->numCols();
    const int64_t channels = inAccess->numChannels();
    if (channels != 3 && channels != 4)
    {
        return false;
    }

    const bool    isInterleaved = *copyLayout == DenseCopyLayout::INTERLEAVED;
    const int64_t rowBytes      = isInterleaved ? cols * channels : cols;
    const int64_t copyRows      = isInterleaved ? rows : rows * channels;

    if (samples <= 0 || rows <= 0 || cols <= 0 || rowBytes <= 0 || copyRows <= 0 || inAccess->rowStride() < rowBytes
        || outAccess->rowStride() < rowBytes)
    {
        return false;
    }

    if (!HaveSupportedStrides(*copyLayout, *inAccess, *outAccess, rows, channels, rowBytes))
    {
        return false;
    }

    const int64_t srcRowsPerSample = RowsPerSample(inAccess->sampleStride(), inAccess->rowStride(), copyRows, samples);
    const int64_t dstRowsPerSample
        = RowsPerSample(outAccess->sampleStride(), outAccess->rowStride(), copyRows, samples);
    if (srcRowsPerSample < copyRows || dstRowsPerSample < copyRows)
    {
        return false;
    }

    auto *src = inAccess->sampleData(0);
    auto *dst = outAccess->sampleData(0);
    if (src == dst)
    {
        return false;
    }

    if (BuffersOverlap(src, dst, *inAccess, *outAccess, samples, copyRows, rowBytes))
    {
        return false;
    }

    const auto byteCount = DenseCopyByteCount(copyRows, rowBytes, samples);
    if (!byteCount.has_value())
    {
        return false;
    }

    if (const int64_t denseSampleBytes = *byteCount / samples;
        CanUseLinearCopy(*copyLayout, *inAccess, *outAccess, samples, rowBytes, denseSampleBytes))
    {
        NVCV_CHECK_THROW(cudaMemcpyAsync(dst, src, static_cast<size_t>(*byteCount), cudaMemcpyDeviceToDevice, stream));
        return true;
    }

    cudaMemcpy3DParms copyParams{};
    copyParams.srcPtr = make_cudaPitchedPtr(src, static_cast<size_t>(inAccess->rowStride()),
                                            static_cast<size_t>(rowBytes), static_cast<size_t>(srcRowsPerSample));
    copyParams.dstPtr = make_cudaPitchedPtr(dst, static_cast<size_t>(outAccess->rowStride()),
                                            static_cast<size_t>(rowBytes), static_cast<size_t>(dstRowsPerSample));
    copyParams.extent
        = make_cudaExtent(static_cast<size_t>(rowBytes), static_cast<size_t>(copyRows), static_cast<size_t>(samples));
    copyParams.kind = cudaMemcpyDeviceToDevice;

    NVCV_CHECK_THROW(cudaMemcpy3DAsync(&copyParams, stream));
    return true;
}

} // namespace

BoxBlur::BoxBlur()
{
    legacy::DataShape maxIn;
    legacy::DataShape maxOut; //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<legacy::BoxBlur>(maxIn, maxOut);
}

void BoxBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                         const NVCVBlurBoxesI &bboxes) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BoxBlur::operator()[Tensor]");
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

    if (TryDenseCopy(stream, *inData, *outData, bboxes))
    {
        // Keep the immutable input as the blur source so concurrent boxes cannot observe each other's output writes.
        NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, bboxes, stream, true));
    }
    else
    {
        NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, bboxes, stream));
    }
}

} // namespace cvcuda::priv
