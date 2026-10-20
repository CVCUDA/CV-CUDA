/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpBoxBlur.hpp"
#include "TensorAccessValidation.hpp"

#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using cvcuda::priv::BoxBlurCommand;
using cvcuda::priv::NVCVBlurBoxesImpl;

// Every blur region is resampled into a fixed 32x32 shared-memory crop and one thread owns one
// crop texel, so the blur block size and the crop extent are the same constant.
constexpr int kCropSize = 32;

// Command capacity preallocated by the constructor. The legacy operator sized this buffer as
// PREALLOC_CMD_NUM (100) * sizeof(BoxBlurCommand) (28) *entries*; the same command capacity is
// carried over so no workload that avoided a growth reallocation before can trigger one now.
constexpr size_t kPreallocatedCommandCount = 2800;

// The four tensor layouts the operator accepts. Kept as a four-way distinction rather than a
// planar/interleaved flag because the dense-copy fast path only admits the batched spellings.
enum class FormatKind
{
    kNHWC,
    kHWC,
    kNCHW,
    kCHW
};

enum class ByteKind
{
    kUnsigned,
    kSigned
};

inline bool IsPlanar(FormatKind format)
{
    return format == FormatKind::kNCHW || format == FormatKind::kCHW;
}

inline FormatKind GetFormatKind(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return FormatKind::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return FormatKind::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return FormatKind::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return FormatKind::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

// Classifies on bits-per-channel plus data kind, which is exactly what the legacy
// GetLegacyDataType did before the kCV_8U/kCV_8S gate: the accepted set therefore still includes
// the packed multi-channel byte types (TYPE_2U8, TYPE_3U8, TYPE_4U8 and their signed siblings),
// not just scalar TYPE_U8/TYPE_S8.
inline std::optional<ByteKind> GetByteKind(const nvcv::DataType &dtype)
{
    const std::array<int32_t, 4> bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            return std::nullopt;
        }
    }

    if (bpc[0] != 8)
    {
        return std::nullopt;
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        return ByteKind::kUnsigned;
    case nvcv::DataKind::SIGNED:
        return ByteKind::kSigned;
    default:
        return std::nullopt;
    }
}

struct ImageShape
{
    int samples;
    int channels;
    int rows;
    int cols;
};

template<typename T>
__forceinline__ __device__ T LimitValue(T value, T low, T high)
{
    return value < low ? low : (value > high ? high : value);
}

// Pixel-to-pixel copy of the whole image, issued when the operator is asked to write a different
// buffer than it reads and the dense host-side copy could not take the transfer.
template<int Channels, class SrcWrapper, class DstWrapper>
__global__ void RenderCopyKernel(SrcWrapper src, DstWrapper dst, int batch, int height, int width)
{
    int       ix       = blockDim.x * blockIdx.x + threadIdx.x;
    int       iy       = blockDim.y * blockIdx.y + threadIdx.y;
    const int batchIdx = blockIdx.z;

    if (ix >= width || iy >= height || batchIdx >= batch)
        return;

    if constexpr (Channels == 3)
    {
        *reinterpret_cast<uchar3 *>(dst.ptr(batchIdx, iy, ix, 0))
            = *reinterpret_cast<const uchar3 *>(src.ptr(batchIdx, iy, ix, 0));
    }
    else
    {
        *reinterpret_cast<uchar4 *>(dst.ptr(batchIdx, iy, ix, 0))
            = *reinterpret_cast<const uchar4 *>(src.ptr(batchIdx, iy, ix, 0));
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void RenderCopyPlanarKernel(SrcWrapper src, DstWrapper dst, int batch, int height, int width, int channels)
{
    int       ix       = blockDim.x * blockIdx.x + threadIdx.x;
    int       iy       = blockDim.y * blockIdx.y + threadIdx.y;
    const int batchIdx = blockIdx.z;

    if (ix >= width || iy >= height || batchIdx >= batch)
        return;

    for (int c = 0; c < channels; ++c)
    {
        *dst.ptr(batchIdx, c, iy, ix) = *src.ptr(batchIdx, c, iy, ix);
    }
}

// One block per box: the box is resampled down to a 32x32 crop in shared memory, box-filtered
// there, then scattered back over the box. The alpha channel of a four-channel image is not
// blurred, it is set opaque.
template<int Channels, class SrcWrapper, class DstWrapper>
__global__ void RenderBlurKernel(SrcWrapper src, DstWrapper dst, const BoxBlurCommand *commands, int numCommands,
                                 int imageBatch, int imageWidth, int imageHeight)
{
    if (blockIdx.x >= numCommands)
        return;
    const BoxBlurCommand &box = commands[blockIdx.x];
    if (box.batchIndex >= imageBatch)
        return;

    using ChannelType = std::remove_const_t<typename SrcWrapper::ValueType>;
    using Pixel3Type  = cuda::MakeType<ChannelType, 3>;

    __shared__ Pixel3Type crop[kCropSize][kCropSize];
    int                   ix = threadIdx.x;
    int                   iy = threadIdx.y;

    int boxWidth  = box.boundingRight - box.boundingLeft;
    int boxHeight = box.boundingBottom - box.boundingTop;
    int sx        = LimitValue((int)(ix / 32.0f * (float)boxWidth + 0.5f + box.boundingLeft), 0, imageWidth);
    int sy        = LimitValue((int)(iy / 32.0f * (float)boxHeight + 0.5f + box.boundingTop), 0, imageHeight);

    crop[iy][ix] = *reinterpret_cast<const Pixel3Type *>(src.ptr(box.batchIndex, sy, sx, 0));
    __syncthreads();

    int3 color = make_int3(0, 0, 0);
    int  n     = 0;
    for (int i = -box.kernelSize / 2; i <= box.kernelSize / 2; ++i)
    {
        for (int j = -box.kernelSize / 2; j <= box.kernelSize / 2; ++j)
        {
            int u = i + iy;
            int v = j + ix;
            if (u >= 0 && u < kCropSize && v >= 0 && v < kCropSize)
            {
                auto &c = crop[u][v];
                color.x += c.x;
                color.y += c.y;
                color.z += c.z;
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = Pixel3Type{static_cast<ChannelType>(color.x / n), static_cast<ChannelType>(color.y / n),
                              static_cast<ChannelType>(color.z / n)};
    __syncthreads();

    int gapWidth  = (boxWidth + kCropSize - 1) / kCropSize;
    int gapHeight = (boxHeight + kCropSize - 1) / kCropSize;
    for (int i = 0; i < gapHeight; ++i)
    {
        for (int j = 0; j < gapWidth; ++j)
        {
            int fx = ix * gapWidth + j + box.boundingLeft;
            int fy = iy * gapHeight + i + box.boundingTop;
            if (fx >= 0 && fx < imageWidth && fy >= 0 && fy < imageHeight)
            {
                int cx = static_cast<int>((ix * gapWidth + j) / (float)boxWidth * kCropSize);
                int cy = static_cast<int>((iy * gapHeight + i) / (float)boxHeight * kCropSize);
                if (cx < kCropSize && cy < kCropSize)
                {
                    auto &pix = crop[cy][cx];
                    if constexpr (Channels == 3)
                    {
                        *reinterpret_cast<Pixel3Type *>(dst.ptr(box.batchIndex, fy, fx, 0)) = pix;
                    }
                    else
                    {
                        using Pixel4Type = cuda::MakeType<ChannelType, 4>;

                        *reinterpret_cast<Pixel4Type *>(dst.ptr(box.batchIndex, fy, fx, 0))
                            = Pixel4Type{pix.x, pix.y, pix.z, cuda::TypeTraits<ChannelType>::max};
                    }
                }
            }
        }
    }
}

// Planar counterpart: one block per (box, plane). Plane 3 is the alpha plane and is filled
// opaque instead of blurred.
template<class SrcWrapper, class DstWrapper>
__global__ void RenderBlurPlanarKernel(SrcWrapper src, DstWrapper dst, const BoxBlurCommand *commands, int numCommands,
                                       int imageBatch, int imageWidth, int imageHeight, int channels)
{
    if (blockIdx.x >= numCommands)
        return;
    const BoxBlurCommand &box = commands[blockIdx.x];
    if (box.batchIndex >= imageBatch)
        return;

    using ChannelType = std::remove_const_t<typename SrcWrapper::ValueType>;

    int plane = blockIdx.y;
    if (plane >= channels)
        return;

    int ix = threadIdx.x;
    int iy = threadIdx.y;

    int boxWidth  = box.boundingRight - box.boundingLeft;
    int boxHeight = box.boundingBottom - box.boundingTop;

    int gapWidth  = (boxWidth + kCropSize - 1) / kCropSize;
    int gapHeight = (boxHeight + kCropSize - 1) / kCropSize;

    if (plane == 3)
    {
        for (int i = 0; i < gapHeight; ++i)
        {
            for (int j = 0; j < gapWidth; ++j)
            {
                int fx = ix * gapWidth + j + box.boundingLeft;
                int fy = iy * gapHeight + i + box.boundingTop;
                if (fx >= 0 && fx < imageWidth && fy >= 0 && fy < imageHeight)
                {
                    int cx = static_cast<int>((ix * gapWidth + j) / (float)boxWidth * kCropSize);
                    int cy = static_cast<int>((iy * gapHeight + i) / (float)boxHeight * kCropSize);
                    if (cx < kCropSize && cy < kCropSize)
                    {
                        *dst.ptr(box.batchIndex, plane, fy, fx) = cuda::TypeTraits<ChannelType>::max;
                    }
                }
            }
        }
        return;
    }

    __shared__ ChannelType crop[kCropSize][kCropSize];

    int sx = LimitValue((int)(ix / 32.0f * (float)boxWidth + 0.5f + box.boundingLeft), 0, imageWidth);
    int sy = LimitValue((int)(iy / 32.0f * (float)boxHeight + 0.5f + box.boundingTop), 0, imageHeight);

    crop[iy][ix] = *src.ptr(box.batchIndex, plane, sy, sx);
    __syncthreads();

    int color = 0;
    int n     = 0;
    for (int i = -box.kernelSize / 2; i <= box.kernelSize / 2; ++i)
    {
        for (int j = -box.kernelSize / 2; j <= box.kernelSize / 2; ++j)
        {
            int u = i + iy;
            int v = j + ix;
            if (u >= 0 && u < kCropSize && v >= 0 && v < kCropSize)
            {
                color += crop[u][v];
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = static_cast<ChannelType>(color / n);
    __syncthreads();

    for (int i = 0; i < gapHeight; ++i)
    {
        for (int j = 0; j < gapWidth; ++j)
        {
            int fx = ix * gapWidth + j + box.boundingLeft;
            int fy = iy * gapHeight + i + box.boundingTop;
            if (fx >= 0 && fx < imageWidth && fy >= 0 && fy < imageHeight)
            {
                int cx = static_cast<int>((ix * gapWidth + j) / (float)boxWidth * kCropSize);
                int cy = static_cast<int>((iy * gapHeight + i) / (float)boxHeight * kCropSize);
                if (cx < kCropSize && cy < kCropSize)
                {
                    *dst.ptr(box.batchIndex, plane, fy, fx) = crop[cy][cx];
                }
            }
        }
    }
}

template<typename T, int Channels>
inline void LaunchInterleaved(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              const ImageShape &shape, const BoxBlurCommand *commands, int numCommands,
                              cudaStream_t stream, bool skipCopy)
{
    auto src = cuda::CreateTensorWrapNHWC<T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHWC<T, int32_t>(outData);

    if (!skipCopy && src.ptr(0) != dst.ptr(0))
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(util::DivUp(shape.cols + 1, static_cast<int>(blockSize.x)),
                      util::DivUp(shape.rows + 1, static_cast<int>(blockSize.y)), shape.samples);

        RenderCopyKernel<Channels><<<gridSize, blockSize, 0, stream>>>(src, dst, shape.samples, shape.rows, shape.cols);

        NVCV_CHECK_THROW(cudaGetLastError());
    }

    if (numCommands > 0)
    {
        dim3 blockSize(kCropSize, kCropSize);
        dim3 gridSize(numCommands, 1);

        RenderBlurKernel<Channels><<<gridSize, blockSize, 0, stream>>>(src, dst, commands, numCommands, shape.samples,
                                                                       shape.cols, shape.rows);

        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename T>
inline void LaunchPlanar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                         const ImageShape &shape, const BoxBlurCommand *commands, int numCommands, cudaStream_t stream,
                         bool skipCopy)
{
    auto src = cuda::CreateTensorWrapNCHW<T, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);

    if (!skipCopy && src.ptr(0, 0, 0, 0) != dst.ptr(0, 0, 0, 0))
    {
        dim3 blockSize(32, 32);
        dim3 gridSize(util::DivUp(shape.cols + 1, static_cast<int>(blockSize.x)),
                      util::DivUp(shape.rows + 1, static_cast<int>(blockSize.y)), shape.samples);

        RenderCopyPlanarKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, shape.samples, shape.rows, shape.cols,
                                                                   shape.channels);

        NVCV_CHECK_THROW(cudaGetLastError());
    }

    if (numCommands > 0)
    {
        dim3 blockSize(kCropSize, kCropSize);
        dim3 gridSize(numCommands, shape.channels);

        RenderBlurPlanarKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, commands, numCommands, shape.samples,
                                                                   shape.cols, shape.rows, shape.channels);

        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

inline void BuildCommands(std::vector<BoxBlurCommand> &commands, int width, int height, const NVCVBlurBoxesImpl &bboxes)
{
    commands.clear();

    for (int n = 0; n < bboxes.batch(); n++)
    {
        const int numBoxes = bboxes.numBoxesAt(n);

        for (int i = 0; i < numBoxes; i++)
        {
            const NVCVBlurBoxI bbox   = bboxes.boxAt(n, i);
            const int          left   = std::max(std::min(bbox.box.x, width - 1), 0);
            const int          top    = std::max(std::min(bbox.box.y, height - 1), 0);
            const int          right  = std::max(std::min(left + bbox.box.width - 1, width - 1), 0);
            const int          bottom = std::max(std::min(top + bbox.box.height - 1, height - 1), 0);

            // A degenerate region, a region smaller than the 3x3 the crop resampling assumes, or a
            // non-positive kernel leaves the image untouched instead of being an error.
            if (left == right || top == bottom || bbox.box.width < 3 || bbox.box.height < 3 || bbox.kernelSize < 1)
            {
                continue;
            }

            commands.push_back(BoxBlurCommand{left, top, right, bottom, n, bbox.kernelSize});
        }
    }
}

// The command list belongs to a long-lived operator, so it is cleared however the call ends; a
// later submit must not replay this call's boxes.
class CommandScope
{
public:
    explicit CommandScope(std::vector<BoxBlurCommand> &commands)
        : m_commands(commands)
    {
    }

    ~CommandScope()
    {
        m_commands.clear();
    }

    CommandScope(const CommandScope &)            = delete;
    CommandScope &operator=(const CommandScope &) = delete;

private:
    std::vector<BoxBlurCommand> &m_commands;
};

inline void ValidateOutputShape(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    if (outAccess.numRows() != inAccess.numRows() || outAccess.numCols() != inAccess.numCols()
        || outAccess.numSamples() != inAccess.numSamples() || outAccess.numChannels() != inAccess.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape");
    }
}

inline void ValidateInterleavedStrides(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                       const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    const int64_t srcMaxStride = inAccess.sampleStride() * inAccess.numSamples();
    const int64_t dstMaxStride = outAccess.sampleStride() * outAccess.numSamples();

    if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }
}

inline void ValidatePlanarStrides(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                  const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    const int64_t maxStride
        = std::max({inAccess.sampleStride() * inAccess.numSamples(), inAccess.chStride() * inAccess.numChannels(),
                    inAccess.rowStride(), outAccess.sampleStride() * outAccess.numSamples(),
                    outAccess.chStride() * outAccess.numChannels(), outAccess.rowStride()});

    if (maxStride > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output stride exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
}

enum class DenseCopyLayout
{
    INTERLEAVED,
    PLANAR
};

inline std::optional<DenseCopyLayout> GetDenseCopyLayout(FormatKind inputFormat, FormatKind outputFormat)
{
    if (inputFormat == FormatKind::kNHWC && outputFormat == FormatKind::kNHWC)
    {
        return DenseCopyLayout::INTERLEAVED;
    }
    if (inputFormat == FormatKind::kNCHW && outputFormat == FormatKind::kNCHW)
    {
        return DenseCopyLayout::PLANAR;
    }
    return std::nullopt;
}

inline bool HaveSameImageShape(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                               const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    return inAccess.numSamples() == outAccess.numSamples() && inAccess.numRows() == outAccess.numRows()
        && inAccess.numCols() == outAccess.numCols() && inAccess.numChannels() == outAccess.numChannels();
}

inline bool HaveSupportedStrides(DenseCopyLayout layout, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                 const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int64_t rows,
                                 int64_t channels, int64_t rowBytes)
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

inline int64_t RowsPerSample(int64_t sampleStride, int64_t rowStride, int64_t copyRows, int64_t samples)
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

inline bool BuffersOverlap(const nvcv::Byte *src, const nvcv::Byte *dst,
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

inline std::optional<int64_t> DenseCopyByteCount(int64_t copyRows, int64_t rowBytes, int64_t samples)
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

inline bool CanUseLinearCopy(DenseCopyLayout layout, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                             const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int64_t samples,
                             int64_t rowBytes, int64_t sampleBytes)
{
    return layout == DenseCopyLayout::INTERLEAVED && inAccess.rowStride() == rowBytes
        && outAccess.rowStride() == rowBytes
        && (samples == 1 || (inAccess.sampleStride() == sampleBytes && outAccess.sampleStride() == sampleBytes));
}

inline bool TryDenseCopy(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                         const nvcv::TensorDataStridedCuda &outData, NVCVBlurBoxesI bboxes)
{
    if (bboxes == nullptr || inData.dtype() != outData.dtype())
    {
        return false;
    }

    FormatKind inputFormat;
    FormatKind outputFormat;
    try
    {
        inputFormat  = GetFormatKind(inData.layout());
        outputFormat = GetFormatKind(outData.layout());
    }
    catch (const nvcv::Exception &)
    {
        return false;
    }

    const auto copyLayout = GetDenseCopyLayout(inputFormat, outputFormat);
    if (GetByteKind(inData.dtype()) != ByteKind::kUnsigned || !copyLayout)
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

namespace cvcuda::priv {

BoxBlur::BoxBlur()
{
    ReserveCommandBuffer(kPreallocatedCommandCount);
}

BoxBlur::~BoxBlur()
{
    FreeCommandBuffer();
}

void BoxBlur::ReserveCommandBuffer(size_t count) const
{
    if (m_commandCapacity >= count)
    {
        return;
    }

    FreeCommandBuffer();

    NVCV_CHECK_THROW(cudaMallocHost(&m_hostCommands, count * sizeof(BoxBlurCommand)));
    NVCV_CHECK_THROW(cudaMalloc(&m_deviceCommands, count * sizeof(BoxBlurCommand)));
    m_commandCapacity = count;
}

void BoxBlur::FreeCommandBuffer() const noexcept
{
    NVCV_CHECK_LOG(cudaFreeHost(m_hostCommands));
    NVCV_CHECK_LOG(cudaFree(m_deviceCommands));

    m_hostCommands    = nullptr;
    m_deviceCommands  = nullptr;
    m_commandCapacity = 0;
}

void BoxBlur::UploadCommands(cudaStream_t stream) const
{
    if (m_commands.empty())
    {
        return;
    }

    ReserveCommandBuffer(m_commands.size());

    const size_t bytes = m_commands.size() * sizeof(BoxBlurCommand);
    std::memcpy(m_hostCommands, m_commands.data(), bytes);

    NVCV_CHECK_THROW(cudaMemcpyAsync(m_deviceCommands, m_hostCommands, bytes, cudaMemcpyHostToDevice, stream));
}

void BoxBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                         const NVCVBlurBoxesI &bboxes) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BoxBlur::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    // Keep the immutable input as the blur source so concurrent boxes cannot observe each other's
    // output writes: when the dense copy takes the transfer, the kernels skip their own copy pass.
    const bool skipCopy = TryDenseCopy(stream, *inData, *outData, bboxes);

    const FormatKind inputFormat  = GetFormatKind(inData->layout());
    const FormatKind outputFormat = GetFormatKind(outData->layout());

    if (IsPlanar(inputFormat) != IsPlanar(outputFormat))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat both Input and Output must be kNHWC, kHWC, kNCHW or kCHW");
    }

    RequireSameDtype(*inData, *outData);

    const std::optional<ByteKind> byteKind = GetByteKind(inData->dtype());
    if (!byteKind)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataType, only 8-bit signed and unsigned types are supported");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input tensor");
    }

    const int batch    = inAccess->numSamples();
    const int channels = inAccess->numChannels();
    const int rows     = inAccess->numRows();
    const int cols     = inAccess->numCols();

    if (channels > 4 || channels < 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number ch = %d", channels);
    }

    if (bboxes == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "bboxes must not be NULL");
    }

    const auto *blurBoxes = reinterpret_cast<const NVCVBlurBoxesImpl *>(bboxes);
    if (blurBoxes->batch() != batch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "bboxes batch %d != input batch %d",
                              blurBoxes->batch(), batch);
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output tensor");
    }

    BuildCommands(m_commands, cols, rows, *blurBoxes);
    CommandScope commandScope(m_commands);

    ValidateOutputShape(*inAccess, *outAccess);

    UploadCommands(stream);

    const ImageShape shape{batch, channels, rows, cols};
    const auto       numCommands = static_cast<int>(m_commands.size());

    // The legacy dispatch was a funcs[dataTypeIdx][channelIdx] table: rows selected the element
    // type (kCV_8U -> uint8_t, kCV_8S -> int8_t) and columns the channel count (channels - 3, so
    // three then four), with the planar layout dispatched separately on the element type alone.
    // Both axes are closed by the validation above, so the switch below covers exactly the same
    // entries.
    if (IsPlanar(inputFormat))
    {
        ValidatePlanarStrides(*inAccess, *outAccess);

        if (*byteKind == ByteKind::kSigned)
        {
            LaunchPlanar<int8_t>(*inData, *outData, shape, m_deviceCommands, numCommands, stream, skipCopy);
        }
        else
        {
            LaunchPlanar<uint8_t>(*inData, *outData, shape, m_deviceCommands, numCommands, stream, skipCopy);
        }
    }
    else
    {
        ValidateInterleavedStrides(*inAccess, *outAccess);

        if (*byteKind == ByteKind::kSigned)
        {
            if (channels == 3)
            {
                LaunchInterleaved<int8_t, 3>(*inData, *outData, shape, m_deviceCommands, numCommands, stream, skipCopy);
            }
            else
            {
                LaunchInterleaved<int8_t, 4>(*inData, *outData, shape, m_deviceCommands, numCommands, stream, skipCopy);
            }
        }
        else
        {
            if (channels == 3)
            {
                LaunchInterleaved<uint8_t, 3>(*inData, *outData, shape, m_deviceCommands, numCommands, stream,
                                              skipCopy);
            }
            else
            {
                LaunchInterleaved<uint8_t, 4>(*inData, *outData, shape, m_deviceCommands, numCommands, stream,
                                              skipCopy);
            }
        }
    }
}

} // namespace cvcuda::priv
