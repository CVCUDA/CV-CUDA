/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2023, Bytedance Inc. All rights reserved.
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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpReformat.hpp"
#include "TensorAccessValidation.hpp"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// The single-channel byte copy crossover, calibrated by measurement: a pitched copy costs one CUDA
// operation per sample, so below these sizes the setup cost exceeds one reformat kernel launch.
inline constexpr std::size_t kSingleSampleMinRowBytes = 256;
inline constexpr std::size_t kMultiSampleMinRowBytes  = 288;
inline constexpr std::size_t kLimitedBatchMinPixels   = 1600 * 900;
inline constexpr std::size_t kAnyBatchMinPixels       = 1792 * 1056;

constexpr bool ShouldUsePitchedCopy(int numSamples, std::size_t pixelsPerSample, std::size_t rowBytes) noexcept
{
    if (numSamples == 0)
    {
        return true;
    }

    if (numSamples == 1)
    {
        return rowBytes >= kSingleSampleMinRowBytes;
    }

    return rowBytes >= kMultiSampleMinRowBytes
        && (pixelsPerSample >= kAnyBatchMinPixels || (numSamples <= 8 && pixelsPerSample >= kLimitedBatchMinPixels));
}

// Ordinals are printed in the unsupported-pairing rejection message.
using Layout = cvcuda::priv::legacy_format::DataFormatCode;

template<Layout L>
constexpr int LayoutDimensions = (L == Layout::kNCHW || L == Layout::kNHWC) ? 4 : 3;

using cvcuda::priv::legacy_format::ClassifyDataFormat;
using cvcuda::priv::legacy_format::ClassifyDataType;
using cvcuda::priv::legacy_format::DataTypeCode;

__device__ __forceinline__ void packRGB8x4(uint32_t r, uint32_t g, uint32_t b, uint32_t &out0, uint32_t &out1,
                                           uint32_t &out2)
{
    uint32_t tmp = __byte_perm(r, g, 0x1040);
    out0         = __byte_perm(tmp, b, 0x3410);
    tmp          = __byte_perm(g, b, 0x2251);
    out1         = __byte_perm(tmp, r, 0x3610);
    tmp          = __byte_perm(b, r, 0x3372);
    out2         = __byte_perm(tmp, g, 0x3710);
}

__global__ void transformFormatNCHWToNHWCRGB8(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width,
                                              int height, int64_t srcRowStride, int64_t srcPlaneStride,
                                              int64_t srcSampleStride, int64_t dstRowStride, int64_t dstSampleStride,
                                              int groupsPerRow, int64_t totalRows)
{
    const int64_t tid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = totalRows * groupsPerRow;
    if (tid >= total)
        return;

    const int64_t rowIdx = tid / groupsPerRow;
    const int     group  = static_cast<int>(tid - rowIdx * groupsPerRow);
    const int     sample = static_cast<int>(rowIdx / height);
    const int     y      = static_cast<int>(rowIdx - static_cast<int64_t>(sample) * height);
    const int     x      = group * 8;

    const uint8_t *red = src + static_cast<int64_t>(sample) * srcSampleStride + static_cast<int64_t>(y) * srcRowStride;
    const uint8_t *green = red + srcPlaneStride;
    const uint8_t *blue  = green + srcPlaneStride;
    uint8_t       *out
        = dst + static_cast<int64_t>(sample) * dstSampleStride + static_cast<int64_t>(y) * dstRowStride + x * 3;

    if (x + 8 <= width)
    {
        const uint2 red8   = *reinterpret_cast<const uint2 *>(red + x);
        const uint2 green8 = *reinterpret_cast<const uint2 *>(green + x);
        const uint2 blue8  = *reinterpret_cast<const uint2 *>(blue + x);

        uint2 out0, out1, out2;
        packRGB8x4(red8.x, green8.x, blue8.x, out0.x, out0.y, out1.x);
        packRGB8x4(red8.y, green8.y, blue8.y, out1.y, out2.x, out2.y);

        *reinterpret_cast<uint2 *>(out)      = out0;
        *reinterpret_cast<uint2 *>(out + 8)  = out1;
        *reinterpret_cast<uint2 *>(out + 16) = out2;
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            if (x + i < width)
            {
                out[i * 3]     = red[x + i];
                out[i * 3 + 1] = green[x + i];
                out[i * 3 + 2] = blue[x + i];
            }
        }
    }
}

__global__ void transformFormatNCHWToNHWCU8C1(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int width,
                                              int height, int srcRowStride, int srcSampleStride, int dstRowStride,
                                              int dstSampleStride)
{
    const int column = (blockIdx.x * blockDim.x + threadIdx.x) * sizeof(uint4);
    const int row    = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample = blockIdx.z;
    if (column >= width || row >= height)
        return;

    src += sample * srcSampleStride + row * srcRowStride + column;
    dst += sample * dstSampleStride + row * dstRowStride + column;
    if (column + sizeof(uint4) <= width)
    {
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
    }
    else
    {
#pragma unroll
        for (int byte = 0; byte < sizeof(uint4); ++byte)
        {
            if (column + byte < width)
                dst[byte] = src[byte];
        }
    }
}

template<Layout SrcLayout, int RowsPerThread, class SrcWrapper, class DstWrapper>
__global__ void transformFormat(const SrcWrapper src, DstWrapper dst, int3 inout_size)
{
    const int x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int firstY = blockIdx.y * blockDim.y * RowsPerThread + threadIdx.y;

    if (x >= inout_size.x)
        return;

    using DimType = cuda::MakeType<int, SrcWrapper::kNumDimensions>;
    DimType srcCoord, dstCoord;

#pragma unroll
    for (int row = 0; row < RowsPerThread; ++row)
    {
        const int y = firstY + row * blockDim.y;
        if (y >= inout_size.y)
            continue;

#pragma unroll 4
        for (int c = 0; c < inout_size.z; c++)
        {
            if constexpr (SrcLayout == Layout::kNCHW)
            {
                srcCoord = {x, y, c, static_cast<int>(blockIdx.z)};
                dstCoord = {c, x, y, static_cast<int>(blockIdx.z)};
            }
            else if constexpr (SrcLayout == Layout::kNHWC)
            {
                srcCoord = {c, x, y, static_cast<int>(blockIdx.z)};
                dstCoord = {x, y, c, static_cast<int>(blockIdx.z)};
            }
            else if constexpr (SrcLayout == Layout::kCHW)
            {
                srcCoord = {x, y, c};
                dstCoord = {c, x, y};
            }
            else if constexpr (SrcLayout == Layout::kHWC)
            {
                srcCoord = {c, x, y};
                dstCoord = {x, y, c};
            }

            dst[dstCoord] = src[srcCoord];
        }
    }
}

template<Layout SrcLayout, typename T>
inline void Transform(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                      const nvcv::TensorDataStridedCuda              &outData,
                      const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                      const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    const int3        inout_size = {inAccess.numCols(), inAccess.numRows(), outAccess.numChannels()};
    constexpr int64_t kMaxGridZ  = 65535;

    const int64_t inMaxStride  = inAccess.sampleStride() * inAccess.numSamples();
    const int64_t outMaxStride = outAccess.sampleStride() * outAccess.numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    bool usedRGB8FastPath = false;

    if constexpr (SrcLayout == Layout::kNCHW && std::is_same_v<T, unsigned char>)
    {
        const uintptr_t alignment
            = reinterpret_cast<uintptr_t>(inData.basePtr()) | reinterpret_cast<uintptr_t>(outData.basePtr())
            | static_cast<uintptr_t>(inAccess.rowStride()) | static_cast<uintptr_t>(inAccess.planeStride())
            | static_cast<uintptr_t>(inAccess.sampleStride()) | static_cast<uintptr_t>(outAccess.rowStride())
            | static_cast<uintptr_t>(outAccess.sampleStride());

        constexpr int kBlockRows = 8;
        constexpr int kMaxGridY  = 65535;
        if (inAccess.numSamples() > 0 && inout_size.x > 0 && inout_size.y > 0 && inout_size.y <= kBlockRows * kMaxGridY
            && inAccess.rowStride() > 0 && inAccess.sampleStride() > 0 && outAccess.rowStride() > 0
            && outAccess.sampleStride() > 0 && inData.rank() == 4 && outData.rank() == 4 && inout_size.z == 1
            && inData.stride(3) == 1 && outData.stride(2) == 1 && outData.stride(3) == 1
            && (alignment & (alignof(uint4) - 1)) == 0)
        {
            const dim3 block{32, kBlockRows};
            dim3       grid(util::DivUp(inout_size.x, static_cast<int>(block.x * sizeof(uint4))),
                            util::DivUp(inout_size.y, static_cast<int>(block.y)));
            int64_t    sample = 0;
            do
            {
                grid.z = static_cast<unsigned int>(std::min(inAccess.numSamples() - sample, kMaxGridZ));
                transformFormatNCHWToNHWCU8C1<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const uint8_t *>(inData.basePtr() + sample * inAccess.sampleStride()),
                    reinterpret_cast<uint8_t *>(outData.basePtr() + sample * outAccess.sampleStride()), inout_size.x,
                    inout_size.y, static_cast<int>(inAccess.rowStride()), static_cast<int>(inAccess.sampleStride()),
                    static_cast<int>(outAccess.rowStride()), static_cast<int>(outAccess.sampleStride()));
                NVCV_CHECK_THROW(cudaGetLastError());
                sample += grid.z;
            }
            while (sample < inAccess.numSamples());
            return;
        }

        usedRGB8FastPath = inAccess.numSamples() > 0 && inout_size.x > 0 && inout_size.y > 0 && inAccess.rowStride() > 0
                        && inAccess.planeStride() > 0 && inAccess.sampleStride() > 0 && outAccess.rowStride() > 0
                        && outAccess.sampleStride() > 0 && inData.rank() == 4 && outData.rank() == 4
                        && inout_size.z == 3 && inData.stride(3) == 1 && outData.stride(2) == 3
                        && outData.stride(3) == 1 && (alignment & 7U) == 0;

        if (usedRGB8FastPath)
        {
            constexpr int kPixelsPerThread = 8;
            constexpr int kBlockSize       = 128;
            const int     groupsPerRow     = util::DivUp(inout_size.x, kPixelsPerThread);
            const int64_t totalRows = static_cast<int64_t>(inAccess.numSamples()) * static_cast<int64_t>(inout_size.y);
            const int64_t totalGroups = totalRows * groupsPerRow;
            const int64_t numBlocks   = util::DivUp(totalGroups, static_cast<int64_t>(kBlockSize));

            transformFormatNCHWToNHWCRGB8<<<static_cast<uint32_t>(numBlocks), kBlockSize, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(inData.basePtr()), reinterpret_cast<uint8_t *>(outData.basePtr()),
                inout_size.x, inout_size.y, inAccess.rowStride(), inAccess.planeStride(), inAccess.sampleStride(),
                outAccess.rowStride(), outAccess.sampleStride(), groupsPerRow, totalRows);
        }
    }

    if (!usedRGB8FastPath)
    {
        // Bytes are the only type wide enough to profit from an eight-row-per-thread block; every
        // other element width keeps the legacy 32x8 single-row geometry.
        constexpr bool kUseWideBlock  = std::is_same_v<T, unsigned char>;
        constexpr int  kRowsPerThread = kUseWideBlock ? 8 : 1;
        const dim3     block          = kUseWideBlock ? dim3{128, 2} : dim3{32, 8};
        dim3           grid(util::DivUp(inout_size.x, static_cast<int>(block.x)),
                            util::DivUp(inout_size.y, static_cast<int>(block.y) * kRowsPerThread));

        cuda::TensorNDWrap<const T, LayoutDimensions<SrcLayout>, int32_t> src(inData);
        cuda::TensorNDWrap<T, LayoutDimensions<SrcLayout>, int32_t>       dst(outData);

        // Rebase non-owning wraps so oversized batches can reuse the unchanged kernel indexing.
        int64_t sample = 0;
        do
        {
            grid.z = static_cast<unsigned int>(std::min(inAccess.numSamples() - sample, kMaxGridZ));
            const decltype(src) chunkSrc(inData.basePtr() + sample * inAccess.sampleStride(), src.strides());
            const decltype(dst) chunkDst(outData.basePtr() + sample * outAccess.sampleStride(), dst.strides());
            transformFormat<SrcLayout, kRowsPerThread><<<grid, block, 0, stream>>>(chunkSrc, chunkDst, inout_size);
            NVCV_CHECK_THROW(cudaGetLastError());
            sample += grid.z;
        }
        while (sample < inAccess.numSamples());
        return;
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// The legacy 4x8 function-pointer table indexed by [input format][data type code] becomes two
// nested switches; the instantiated set is identical.
template<Layout SrcLayout>
inline void TransformByElementType(DataTypeCode type, cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda              &outData,
                                   const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                   const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    switch (type)
    {
    case DataTypeCode::kU8:
        Transform<SrcLayout, unsigned char>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kS8:
        Transform<SrcLayout, signed char>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kU16:
        Transform<SrcLayout, unsigned short>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kS16:
        Transform<SrcLayout, short>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kS32:
        Transform<SrcLayout, int>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kF32:
        Transform<SrcLayout, float>(stream, inData, outData, inAccess, outAccess);
        return;
    case DataTypeCode::kF64:
        Transform<SrcLayout, double>(stream, inData, outData, inAccess, outAccess);
        return;
    // Reformat relocates elements without arithmetic, so 16-bit float rides the 16-bit integer
    // instantiation and copies the bit patterns through unchanged. Sharing the instantiation with
    // kU16 rather than adding a half specialization is what makes the transfer bit-exact.
    case DataTypeCode::kF16:
        Transform<SrcLayout, unsigned short>(stream, inData, outData, inAccess, outAccess);
        return;
    }
}

inline void RunTransform(Layout srcLayout, DataTypeCode type, cudaStream_t stream,
                         const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                         const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                         const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    switch (srcLayout)
    {
    case Layout::kNCHW:
        TransformByElementType<Layout::kNCHW>(type, stream, inData, outData, inAccess, outAccess);
        return;
    case Layout::kNHWC:
        TransformByElementType<Layout::kNHWC>(type, stream, inData, outData, inAccess, outAccess);
        return;
    case Layout::kCHW:
        TransformByElementType<Layout::kCHW>(type, stream, inData, outData, inAccess, outAccess);
        return;
    case Layout::kHWC:
        TransformByElementType<Layout::kHWC>(type, stream, inData, outData, inAccess, outAccess);
        return;
    }
}

inline void RunReformat(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                        const nvcv::TensorDataStridedCuda &outData)
{
    const Layout inLayout  = ClassifyDataFormat(inData.layout());
    const Layout outLayout = ClassifyDataFormat(outData.layout());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input DataFormat");
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output DataFormat");
    }

    if (inAccess->numSamples() != outAccess->numSamples() || inAccess->numChannels() != outAccess->numChannels()
        || inAccess->numRows() != outAccess->numRows() || inAccess->numCols() != outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output logical image extents must match");
    }

    if (inData.dtype() == outData.dtype() && inData.shape() == outData.shape())
    {
        for (int i = 0; i < inAccess->numSamples(); ++i)
        {
            nvcv::Byte *inSampData  = inAccess->sampleData(i);
            nvcv::Byte *outSampData = outAccess->sampleData(i);

            for (int p = 0; p < inAccess->numPlanes(); ++p)
            {
                NVCV_CHECK_THROW(cudaMemcpy2DAsync(outAccess->planeData(p, outSampData), outAccess->rowStride(),
                                                   inAccess->planeData(p, inSampData), inAccess->rowStride(),
                                                   inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                                   cudaMemcpyDeviceToDevice, stream));
            }
        }
        return;
    }

    const bool inverseLayouts = (inLayout == Layout::kNHWC && outLayout == Layout::kNCHW)
                             || (inLayout == Layout::kNCHW && outLayout == Layout::kNHWC)
                             || (inLayout == Layout::kHWC && outLayout == Layout::kCHW)
                             || (inLayout == Layout::kCHW && outLayout == Layout::kHWC);
    if (!inverseLayouts)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid combination of input format %d and output format %d", static_cast<int>(inLayout),
                              static_cast<int>(outLayout));
    }

    const DataTypeCode inType  = ClassifyDataType(inData.dtype());
    const DataTypeCode outType = ClassifyDataType(outData.dtype());

    if (inType != outType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DataType of input and output must be equal, but got %d and %d", static_cast<int>(inType),
                              static_cast<int>(outType));
    }

    if (inType == DataTypeCode::kU8 && inAccess->numChannels() == 1 && outAccess->numChannels() == 1
        && inAccess->numPlanes() == 1 && outAccess->numPlanes() == 1 && inAccess->numRows() == outAccess->numRows()
        && inAccess->numCols() == outAccess->numCols() && inAccess->colStride() == outAccess->colStride())
    {
        const auto rowBytes = inAccess->numCols() * inAccess->colStride();

        if (inAccess->rowStride() == rowBytes && outAccess->rowStride() == rowBytes
            && inAccess->sampleStride() == rowBytes * inAccess->numRows()
            && outAccess->sampleStride() == rowBytes * outAccess->numRows())
        {
            if (inAccess->numSamples() > 0)
            {
                const auto totalBytes = static_cast<size_t>(rowBytes) * static_cast<size_t>(inAccess->numRows())
                                      * static_cast<size_t>(inAccess->numSamples());
                NVCV_CHECK_THROW(cudaMemcpyAsync(outAccess->sampleData(0), inAccess->sampleData(0), totalBytes,
                                                 cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }

        const size_t pixelsPerSample
            = static_cast<size_t>(inAccess->numRows()) * static_cast<size_t>(inAccess->numCols());
        const int  numSamples     = inAccess->numSamples();
        const bool usePitchedCopy = ShouldUsePitchedCopy(numSamples, pixelsPerSample, static_cast<size_t>(rowBytes));

        if (usePitchedCopy)
        {
            if (numSamples > 1 && inAccess->numRows() > 0 && rowBytes > 0 && inAccess->rowStride() >= rowBytes
                && outAccess->rowStride() >= rowBytes)
            {
                const auto rows            = static_cast<size_t>(inAccess->numRows());
                const auto samples         = static_cast<size_t>(numSamples);
                const auto inPitch         = static_cast<size_t>(inAccess->rowStride());
                const auto outPitch        = static_cast<size_t>(outAccess->rowStride());
                const auto inSampleStride  = static_cast<size_t>(inAccess->sampleStride());
                const auto outSampleStride = static_cast<size_t>(outAccess->sampleStride());
                // Bound both conservative buffer spans before forming any products, including copy height.
                if (std::max(inPitch, outPitch) <= std::numeric_limits<size_t>::max() / rows / samples
                    && inSampleStride == inPitch * rows && outSampleStride == outPitch * rows)
                {
                    const auto src = reinterpret_cast<uintptr_t>(inAccess->sampleData(0));
                    const auto dst = reinterpret_cast<uintptr_t>(outAccess->sampleData(0));
                    // Keep ordered per-sample behavior for aliases without forming overflowing end pointers.
                    const bool disjoint
                        = src < dst ? dst - src >= samples * inSampleStride : src - dst >= samples * outSampleStride;
                    if (disjoint)
                    {
                        NVCV_CHECK_THROW(cudaMemcpy2DAsync(outAccess->sampleData(0), outPitch, inAccess->sampleData(0),
                                                           inPitch, static_cast<size_t>(rowBytes), rows * samples,
                                                           cudaMemcpyDeviceToDevice, stream));
                        return;
                    }
                }
            }

            for (int i = 0; i < numSamples; ++i)
            {
                nvcv::Byte *inSampData  = inAccess->sampleData(i);
                nvcv::Byte *outSampData = outAccess->sampleData(i);

                NVCV_CHECK_THROW(cudaMemcpy2DAsync(
                    outAccess->planeData(0, outSampData), outAccess->rowStride(), inAccess->planeData(0, inSampData),
                    inAccess->rowStride(), static_cast<size_t>(rowBytes), static_cast<size_t>(inAccess->numRows()),
                    cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }
    }

    RunTransform(inLayout, inType, stream, inData, outData, *inAccess, *outAccess);
}

} // namespace

namespace cvcuda::priv {

void Reformat::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Reformat::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RunReformat(stream, *inData, *outData);
}

} // namespace cvcuda::priv
