/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Assert.h"
#include "Nvtx.hpp"
#include "OpChannelReorder.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

struct TensorByteView
{
    unsigned char *base;
    int64_t        sampleStride;
    int64_t        rowStride;
    int64_t        colStride;
    int64_t        channelStride;
};

inline __device__ int GetOrder(int4 order, int channel)
{
    switch (channel)
    {
    case 0:
        return order.x;
    case 1:
        return order.y;
    case 2:
        return order.z;
    default:
        return order.w;
    }
}

template<typename T>
__global__ void ChannelReorderTensorKernel(TensorByteView src, TensorByteView dst, int64_t totalPixels, int rows,
                                           int cols, int channels, int4 order)
{
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step   = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t pixel = thread; pixel < totalPixels; pixel += step)
    {
        const int     x = static_cast<int>(pixel % cols);
        const int64_t q = pixel / cols;
        const int     y = static_cast<int>(q % rows);
        const int64_t n = q / rows;

        const int64_t srcPixelOffset = n * src.sampleStride + y * src.rowStride + x * src.colStride;
        const int64_t dstPixelOffset = n * dst.sampleStride + y * dst.rowStride + x * dst.colStride;

#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            if (c >= channels)
            {
                break;
            }
            const int srcChannel = GetOrder(order, c);
            T        *dstValue
                = reinterpret_cast<T *>(dst.base + dstPixelOffset + static_cast<int64_t>(c) * dst.channelStride);
            if (srcChannel < 0)
            {
                *dstValue = T{};
            }
            else
            {
                const T *srcValue = reinterpret_cast<const T *>(src.base + srcPixelOffset
                                                                + static_cast<int64_t>(srcChannel) * src.channelStride);
                *dstValue         = *srcValue;
            }
        }
    }
}

inline int4 PackOrder(const int32_t *order, int32_t orderLength, int channels)
{
    if (order == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Order pointer must not be NULL");
    }
    if (orderLength != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Order length must match the input channel count");
    }

    int values[4]{-1, -1, -1, -1};
    for (int i = 0; i < channels; ++i)
    {
        if (order[i] >= channels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Non-negative order entries must be smaller than the channel count");
        }
        values[i] = order[i];
    }
    return {values[0], values[1], values[2], values[3]};
}

inline int ElementSize(nvcv::DataType dtype)
{
    if (dtype == nvcv::TYPE_U8)
    {
        return 1;
    }
    // Channel reorder moves values without arithmetic, so F16 shares the 16-bit path bit-exactly.
    if (dtype == nvcv::TYPE_U16 || dtype == nvcv::TYPE_S16 || dtype == nvcv::TYPE_F16)
    {
        return 2;
    }
    if (dtype == nvcv::TYPE_S32 || dtype == nvcv::TYPE_F32)
    {
        return 4;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "ChannelReorder supports U8, U16, S16, S32, F16, and F32 tensors");
}

template<typename T>
inline void LaunchChannelReorderTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                       const nvcv::TensorDataStridedCuda              &dstData,
                                       const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                       const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int channels,
                                       int4 order)
{
    const int64_t samples = srcAccess.numSamples();
    const int64_t rows    = srcAccess.numRows();
    const int64_t cols    = srcAccess.numCols();
    if (samples == 0 || rows == 0 || cols == 0)
    {
        return;
    }
    if (samples > std::numeric_limits<int64_t>::max() / rows
        || samples * rows > std::numeric_limits<int64_t>::max() / cols)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Tensor element count overflows int64");
    }
    const int64_t totalPixels  = samples * rows * cols;
    constexpr int threads      = 256;
    const int64_t neededBlocks = (totalPixels + threads - 1) / threads;
    const int     blocks       = static_cast<int>(std::min<int64_t>(neededBlocks, 65535));

    TensorByteView src{reinterpret_cast<unsigned char *>(srcData.basePtr()), srcAccess.sampleStride(),
                       srcAccess.rowStride(), srcAccess.colStride(), srcAccess.chStride()};
    TensorByteView dst{reinterpret_cast<unsigned char *>(dstData.basePtr()), dstAccess.sampleStride(),
                       dstAccess.rowStride(), dstAccess.colStride(), dstAccess.chStride()};
    ChannelReorderTensorKernel<T><<<blocks, threads, 0, stream>>>(src, dst, totalPixels, static_cast<int>(rows),
                                                                  static_cast<int>(cols), channels, order);
    NVCV_CHECK_THROW(cudaGetLastError());
}

constexpr int kVarShapeBlock = 32;

// Pixels per thread in the packed u8 kernel.
constexpr int kNix = 4;

template<typename T>
__global__ void ChannelReorderVarShapeKernel(const cuda::ImageBatchVarShapeWrapNHWC<const T> src,
                                             cuda::ImageBatchVarShapeWrapNHWC<T>             dst,
                                             const cuda::Tensor2DWrap<int>                   orders)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = blockIdx.z;
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int *chOrder = orders.ptr(batch_idx);

    for (int ch = 0; ch < dst.numChannels(); ch++)
    {
        int src_ch = chOrder[ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = 0;
        }
        else
        {
            NVCV_CUDA_ASSERT(0 <= src_ch && src_ch < src.numChannels(),
                             "Index to source channel %d is out of bounds (%d)", src_ch, src.numChannels());
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = *src.ptr(batch_idx, dst_y, dst_x, src_ch);
        }
    }
}

template<typename T>
__global__ void ChannelReorderVarShapePlanarKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                                   cuda::ImageBatchVarShapeWrap<T>             dst,
                                                   const cuda::Tensor2DWrap<int> orders, int numSrcChannels,
                                                   int numDstChannels)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = blockIdx.z;
    int       out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);
    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int *chOrder = orders.ptr(batch_idx);

    for (int ch = 0; ch < numDstChannels; ch++)
    {
        int src_ch = chOrder[ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = 0;
        }
        else
        {
            NVCV_CUDA_ASSERT(0 <= src_ch && src_ch < numSrcChannels, "Index to source channel %d is out of bounds (%d)",
                             src_ch, numSrcChannels);
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = *src.ptr(batch_idx, src_ch, dst_y, dst_x);
        }
    }
}

template<typename T>
inline void ReorderVarShape(bool isPlanar, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                            const nvcv::TensorDataStridedCuda &orderData, int numSrcChannels, int numDstChannels,
                            cudaStream_t stream)
{
    const auto maxSize = isPlanar ? outData.maxSize() : inData.maxSize();

    dim3 blockSize(kVarShapeBlock, kVarShapeBlock / 4, 1);
    // DivUp is computed in int64_t: the int32_t form overflows once maxSize is within blockSize of
    // INT32_MAX, which would turn the block count negative.
    dim3 gridSize(static_cast<unsigned>(util::DivUp<int64_t>(maxSize.w, blockSize.x)),
                  static_cast<unsigned>(util::DivUp<int64_t>(maxSize.h, blockSize.y)), inData.numImages());

    cuda::Tensor2DWrap<int> order_ptr(orderData);

    if (isPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const T> src_ptr(inData);
        cuda::ImageBatchVarShapeWrap<T>       dst_ptr(outData);

        ChannelReorderVarShapePlanarKernel<T>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr, numSrcChannels, numDstChannels);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<const T> src_ptr(inData, numSrcChannels);
        cuda::ImageBatchVarShapeWrapNHWC<T>       dst_ptr(outData, numDstChannels);

        ChannelReorderVarShapeKernel<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// uint3 is 12 bytes but only 4-byte aligned, so it must be masked against its alignment rather
// than its size; using sizeof(uint3) here would over-constrain and silently disable the
// 3-channel fast path.
template<typename PackT>
constexpr uintptr_t kPackAlignmentMask = (sizeof(PackT) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(PackT)) - 1;

template<bool IsPlanar, int NumChannels, class SrcWrapper, class DstWrapper>
__global__ void ChannelReorderVarShapeU8Kernel(const SrcWrapper src, DstWrapper dst,
                                               const cuda::Tensor2DWrap<int> orders)
{
    const int dst_x0     = (blockIdx.x * blockDim.x + threadIdx.x) * kNix;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = blockIdx.z;
    const int out_height = dst.height(batch_idx);
    const int out_width  = dst.width(batch_idx);
    if (dst_x0 >= out_width || dst_y >= out_height)
        return;

    int chOrder[NumChannels];
#pragma unroll
    for (int ch = 0; ch < NumChannels; ++ch)
    {
        chOrder[ch] = orders.ptr(batch_idx)[ch];
        NVCV_CUDA_ASSERT(chOrder[ch] < NumChannels, "Index to source channel %d is out of bounds (%d)", chOrder[ch],
                         NumChannels);
    }

    if constexpr (IsPlanar)
    {
        using PackT = unsigned int;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch)
        {
            const int      src_ch = chOrder[ch];
            unsigned char *dp     = dst.ptr(batch_idx, ch, dst_y, dst_x0);

            if (dst_x0 + kNix <= out_width && (reinterpret_cast<uintptr_t>(dp) & kPackAlignmentMask<PackT>) == 0)
            {
                PackT out{};
                if (src_ch >= 0)
                {
                    const unsigned char *sp = src.ptr(batch_idx, src_ch, dst_y, dst_x0);
                    if ((reinterpret_cast<uintptr_t>(sp) & kPackAlignmentMask<PackT>) == 0)
                    {
                        out                            = *reinterpret_cast<const PackT *>(sp);
                        *reinterpret_cast<PackT *>(dp) = out;
                        continue;
                    }
                }
                else
                {
                    *reinterpret_cast<PackT *>(dp) = out;
                    continue;
                }
            }

#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
                if (dst_x0 + i < out_width)
                {
                    dp[i] = src_ch < 0 ? 0 : *src.ptr(batch_idx, src_ch, dst_y, dst_x0 + i);
                }
            }
        }
    }
    else
    {
        using PackT = std::conditional_t<NumChannels == 3, uint3, uint4>;

        const unsigned char *sp = src.ptr(batch_idx, dst_y, dst_x0);
        unsigned char       *dp = dst.ptr(batch_idx, dst_y, dst_x0);
        if (dst_x0 + kNix <= out_width && (reinterpret_cast<uintptr_t>(sp) & kPackAlignmentMask<PackT>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & kPackAlignmentMask<PackT>) == 0)
        {
            PackT in = *reinterpret_cast<const PackT *>(sp);
            PackT out;

            const unsigned char *inBytes  = reinterpret_cast<const unsigned char *>(&in);
            unsigned char       *outBytes = reinterpret_cast<unsigned char *>(&out);
#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
#pragma unroll
                for (int ch = 0; ch < NumChannels; ++ch)
                {
                    const int src_ch               = chOrder[ch];
                    outBytes[i * NumChannels + ch] = src_ch < 0 ? 0 : inBytes[i * NumChannels + src_ch];
                }
            }
            *reinterpret_cast<PackT *>(dp) = out;
        }
        else
        {
#pragma unroll
            for (int i = 0; i < kNix; ++i)
            {
                if (dst_x0 + i < out_width)
                {
#pragma unroll
                    for (int ch = 0; ch < NumChannels; ++ch)
                    {
                        const int src_ch         = chOrder[ch];
                        dp[i * NumChannels + ch] = src_ch < 0 ? 0 : *src.ptr(batch_idx, dst_y, dst_x0 + i, src_ch);
                    }
                }
            }
        }
    }
}

template<bool IsPlanar, int NumChannels>
inline void ReorderVarShapeU8(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                              const nvcv::TensorDataStridedCuda &orderData, cudaStream_t stream)
{
    int                     batch_size = inData.numImages();
    const auto              maxSize    = IsPlanar ? outData.maxSize() : inData.maxSize();
    dim3                    blockSize(kVarShapeBlock, kVarShapeBlock / 4, 1);
    // See ReorderVarShape: DivUp is computed in int64_t to keep the block count from overflowing.
    dim3                    gridSize(static_cast<unsigned>(util::DivUp<int64_t>(maxSize.w, blockSize.x * kNix)),
                                     static_cast<unsigned>(util::DivUp<int64_t>(maxSize.h, blockSize.y)), batch_size);
    cuda::Tensor2DWrap<int> order_ptr(orderData);

    if constexpr (IsPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const unsigned char> src_ptr(inData);
        cuda::ImageBatchVarShapeWrap<unsigned char>       dst_ptr(outData);
        ChannelReorderVarShapeU8Kernel<true, NumChannels>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<const unsigned char> src_ptr(inData, NumChannels);
        cuda::ImageBatchVarShapeWrapNHWC<unsigned char>       dst_ptr(outData, NumChannels);
        ChannelReorderVarShapeU8Kernel<false, NumChannels>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, order_ptr);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

// Classification is by bits-per-channel plus data kind, never by data-type identity: the var-shape
// path classifies ImageFormat plane types, which are packed (FMT_RGB8's is TYPE_3U8, not TYPE_U8),
// so an exact-dtype test would reject every multi-channel interleaved format this operator accepts.
enum class SampleType
{
    U8,
    S8,
    U16,
    S16,
    S32,
    F32,
    F64,
    F16,
};

inline SampleType GetSampleType(int32_t bitsPerChannel, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bitsPerChannel == 64)
            return SampleType::F64;
        if (bitsPerChannel == 32)
            return SampleType::F32;
        if (bitsPerChannel == 16)
            return SampleType::F16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ",
                              bitsPerChannel);

    case nvcv::DataKind::SIGNED:
        if (bitsPerChannel == 8)
            return SampleType::S8;
        if (bitsPerChannel == 16)
            return SampleType::S16;
        if (bitsPerChannel == 32)
            return SampleType::S32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ",
                              bitsPerChannel);

    case nvcv::DataKind::UNSIGNED:
        if (bitsPerChannel == 8)
            return SampleType::U8;
        if (bitsPerChannel == 16)
            return SampleType::U16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ",
                              bitsPerChannel);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline SampleType GetSampleType(nvcv::DataType dtype)
{
    const std::array<int32_t, 4> bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return GetSampleType(bpc[0], dtype.dataKind());
}

inline SampleType GetSampleType(nvcv::ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return GetSampleType(fmt.planeDataType(0));
}

// The legacy DataFormat enum also distinguished batched from single-image layouts (kNHWC from
// kHWC), but that distinction is a function of the batch size alone, which is common to every
// image, so every comparison the operator made reduced to planar-vs-interleaved.
enum class VarShapeLayout
{
    Interleaved,
    Planar,
};

inline VarShapeLayout GetVarShapeLayout(nvcv::ImageFormat fmt, int32_t numImages)
{
    if (fmt.numPlanes() == 1)
    {
        return VarShapeLayout::Interleaved;
    }
    if (fmt.numChannels() == fmt.numPlanes())
    {
        return VarShapeLayout::Planar;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only planar or packed formats supported CH = %d, planes = %d, batch = %d", fmt.numChannels(),
                          fmt.numPlanes(), numImages);
}

inline void RunChannelReorderVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                      const nvcv::TensorDataStridedCuda &orderData, cudaStream_t stream)
{
    if (inData.numImages() != outData.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output batches must have the same number of images");
    }

    if (inData.numImages() == 0)
    {
        // Must stay ahead of hostFormatList()[0] below, which would index an empty list.
        return;
    }

    const int32_t numImages = inData.numImages();

    const nvcv::ImageFormat fmt(inData.hostFormatList()[0]);
    const nvcv::ImageFormat outFmt(outData.hostFormatList()[0]);

    const SampleType     sampleType  = GetSampleType(fmt);
    const VarShapeLayout inLayout    = GetVarShapeLayout(fmt, numImages);
    const VarShapeLayout outLayout   = GetVarShapeLayout(outFmt, numImages);
    const int            channels    = fmt.numChannels();
    const int            outChannels = outFmt.numChannels();

    if (inLayout != outLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output image layouts must both be planar or both be interleaved");
    }

    const bool isPlanar = inLayout == VarShapeLayout::Planar;

    if (!(sampleType == SampleType::U8 || sampleType == SampleType::U16 || sampleType == SampleType::S16
          || sampleType == SampleType::S32 || sampleType == SampleType::F32 || sampleType == SampleType::F16))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    if (orderData.rank() != 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "order tensor must have 2 dimensions, not %d",
                              orderData.rank());
    }

    if (GetSampleType(orderData.dtype()) != SampleType::S32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Order tensor DataType");
    }

    if (orderData.layout()[0] != nvcv::LABEL_BATCH)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Label of the first dimension of order tensor must be N");
    }

    if (orderData.shape(0) != numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Order tensor must have same number of samples as number of input images");
    }

    if (orderData.shape(1) > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Second dimension of order tensor must be at most 4, not %ld", orderData.shape(1));
    }

    for (int i = 0; i < numImages; ++i)
    {
        const nvcv::ImageFormat inFmt(inData.hostFormatList()[i]);
        const nvcv::ImageFormat imageOutFmt(outData.hostFormatList()[i]);

        const VarShapeLayout imageInLayout  = GetVarShapeLayout(inFmt, numImages);
        const VarShapeLayout imageOutLayout = GetVarShapeLayout(imageOutFmt, numImages);

        if (imageInLayout != inLayout)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input image #%d has an unexpected layout", i);
        }

        if (imageInLayout != imageOutLayout)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input image #%d and output image #%d must both be planar or both be interleaved", i,
                                  i);
        }

        // Image #0's channel count is the reference for the whole batch: a batch with mixed channel
        // counts is rejected rather than silently reinterpreted through image #0's count.
        if (inFmt.numChannels() != channels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input channel %d differs from %d",
                                  inFmt.numChannels(), channels);
        }

        if (inFmt.numChannels() > 4 || (isPlanar && inFmt.numChannels() == 2))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d",
                                  inFmt.numChannels());
        }

        if (imageOutFmt.numChannels() != outChannels)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output channel %d differs from %d",
                                  imageOutFmt.numChannels(), outChannels);
        }

        if (imageOutFmt.numChannels() > 4 || (isPlanar && imageOutFmt.numChannels() == 2))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d",
                                  imageOutFmt.numChannels());
        }

        if (imageOutFmt.numChannels() > orderData.shape(1))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid output channel number %d, must be at most %ld", imageOutFmt.numChannels(),
                                  orderData.shape(1));
        }
        // Deliberately not host-validating order[i] < channels here: the order tensor lives on the
        // device, so checking it would cost a copy back. The kernels assert instead, which only
        // fires in debug builds and is unrecoverable (CUDA kernel errors are sticky).

        if (GetSampleType(inFmt) != sampleType)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Format of input images must all have the same data type");
        }

        if (GetSampleType(imageOutFmt) != sampleType)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Format of output images must all have the same data type");
        }
    }

    if (isPlanar && numImages > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar ChannelReorder requires numImages <= 65535 (CUDA grid-z limit)");
    }

    // Vectorized 4-pixel-at-a-time path; note the different grid width it needs.
    if (sampleType == SampleType::U8 && channels == outChannels)
    {
        if (channels == 3)
        {
            isPlanar ? ReorderVarShapeU8<true, 3>(inData, outData, orderData, stream)
                     : ReorderVarShapeU8<false, 3>(inData, outData, orderData, stream);
            return;
        }
        if (channels == 4)
        {
            isPlanar ? ReorderVarShapeU8<true, 4>(inData, outData, orderData, stream)
                     : ReorderVarShapeU8<false, 4>(inData, outData, orderData, stream);
            return;
        }
    }

    // S8 and F64 are unreachable -- the data-type gate above rejects them -- so they get no kernel
    // rather than being aliased onto a neighbouring width. F16 shares the 16-bit integer kernel:
    // channel reorder moves values without arithmetic, so that copy is bit-exact.
    switch (sampleType)
    {
    case SampleType::U8:
        ReorderVarShape<unsigned char>(isPlanar, inData, outData, orderData, channels, outChannels, stream);
        return;

    case SampleType::U16:
    case SampleType::F16:
        ReorderVarShape<unsigned short>(isPlanar, inData, outData, orderData, channels, outChannels, stream);
        return;

    case SampleType::S16:
        ReorderVarShape<short>(isPlanar, inData, outData, orderData, channels, outChannels, stream);
        return;

    case SampleType::S32:
        ReorderVarShape<int>(isPlanar, inData, outData, orderData, channels, outChannels, stream);
        return;

    case SampleType::F32:
        ReorderVarShape<float>(isPlanar, inData, outData, orderData, channels, outChannels, stream);
        return;

    case SampleType::S8:
    case SampleType::F64:
        break;
    }

    NVCV_ASSERT(false);
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
}

} // namespace

namespace cvcuda::priv {

void ChannelReorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                const int32_t *order, int32_t orderLength) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ChannelReorder::operator()[Tensor]");
    auto inData  = in.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (!inData || !outData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must be CUDA-accessible pitch-linear tensors");
    }
    if (inData->layout() != outData->layout()
        || !(inData->layout() == nvcv::TENSOR_HWC || inData->layout() == nvcv::TENSOR_NHWC
             || inData->layout() == nvcv::TENSOR_CHW || inData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same (N)HWC or (N)CHW layout");
    }
    if (inData->rank() != outData->rank())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output ranks must match");
    }
    for (int i = 0; i < inData->rank(); ++i)
    {
        if (inData->shape(i) != outData->shape(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output shapes must match");
        }
    }
    if (inData->dtype() != outData->dtype() || inData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same scalar data type");
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must support strided planar image access");
    }
    const int channels = inAccess->numChannels();
    if (channels < 1 || channels > 4 || (inAccess->numPlanes() > 1 && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "ChannelReorder supports 1-4 interleaved channels and 1, 3, or 4 planar channels");
    }
    const int4 packedOrder = PackOrder(order, orderLength, channels);
    const int  elementSize = ElementSize(inData->dtype());
    if (elementSize == 1)
    {
        LaunchChannelReorderTensor<uint8_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
    else if (elementSize == 2)
    {
        LaunchChannelReorderTensor<uint16_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
    else
    {
        LaunchChannelReorderTensor<uint32_t>(stream, *inData, *outData, *inAccess, *outAccess, channels, packedOrder);
    }
}

void ChannelReorder::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &orders) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ChannelReorder::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto ordersData = orders.exportData<nvcv::TensorDataStridedCuda>();
    if (!ordersData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input channel order tensor must be cuda-accessible, pitch-linear tensor");
    }

    RunChannelReorderVarShape(*inData, *outData, *ordersData, stream);
}

} // namespace cvcuda::priv
