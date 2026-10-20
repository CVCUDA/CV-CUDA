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
#include "OpHistogramEq.hpp"
#include "TensorAccessValidation.hpp"

#include "ElementStreamCommon.cuh"

#include <cuda_fp16.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <type_traits>

namespace cuda        = nvcv::cuda;
namespace util        = nvcv::util;
namespace elem_stream = cvcuda::priv::elem_stream;

namespace {

using uchar = unsigned char;

// Equalization is defined over the whole 8-bit intensity range: one bin per value, per channel.
constexpr int kHistogramBins = 256;

// The bin store is sized for the widest supported image so a single allocation serves every call.
constexpr int kMaxChannelCount = 4;

// gridDim.z carries the sample index, so the batch cannot exceed the CUDA grid limit.
constexpr int kMaxGridBatch = 65535;

// Coarsen histogram work to reduce the number of block histograms merged into the global one.
constexpr int kPixelsPerThread = 8;

// Coarsen lookup to amortize each block's shared CDF load and synchronization.
constexpr int kLookupPixelsPerThreadX = 8;

// A 256-bin histogram has no reading of a continuous value other than binning it, so a
// floating-point image is quantized to 8 bits, equalized, and scaled back. The factors are the
// reference's: torchvision reaches equalize through to_dtype(uint8, scale=True), which multiplies
// by `255 + 1 - 1e-3` and truncates, and returns through to_dtype(float, scale=True), which
// divides by 255. An 8-bit image is already its own bin.
//
// half bins through float: `255.999` is not representable in half -- it rounds to 256 -- so
// binning in half would map 1.0 outside the table. float and double bin in their own precision,
// which is what makes them bit-exact against the reference.
template<typename T>
inline __device__ uchar ToBin(T v)
{
    if constexpr (std::is_same_v<std::remove_cv_t<T>, uchar>)
    {
        return v;
    }
    else
    {
        using AT            = std::conditional_t<std::is_same_v<std::remove_cv_t<T>, double>, double, float>;
        constexpr AT kScale = AT{255} + AT{1} - AT{1e-3};

        AT scaled = static_cast<AT>(v) * kScale;
        scaled    = scaled < AT{0} ? AT{0} : (scaled > AT{255} ? AT{255} : scaled);
        return static_cast<uchar>(scaled);
    }
}

template<typename T>
inline __device__ T FromBin(uchar v)
{
    if constexpr (std::is_same_v<std::remove_cv_t<T>, uchar>)
    {
        return v;
    }
    else
    {
        return cuda::SaturateCast<T>(static_cast<float>(v) * (1.0f / 255.0f));
    }
}

// Double-buffered inclusive prefix sum over kHistogramBins elements. buf must be
// kHistogramBins * 2 elements of shared memory, with the initial histogram in buf[0..255].
// Returns the output half index (0 or 1): buf[pout * kHistogramBins + tid] is each thread's
// cumulative count on return.
__device__ inline int InclusiveScan256(int *buf, int tid)
{
    int pout = 0, pin = 1;
    for (int offset = 1; offset < kHistogramBins; offset *= 2)
    {
        pout = 1 - pout;
        pin  = 1 - pout;
        if (tid >= offset)
            buf[pout * kHistogramBins + tid]
                = buf[pin * kHistogramBins + tid] + buf[pin * kHistogramBins + tid - offset];
        else
            buf[pout * kHistogramBins + tid] = buf[pin * kHistogramBins + tid];
        __syncthreads();
    }
    return pout;
}

// The Tensor and var-shape paths differ only in how a (sample, row, column, channel) coordinate
// becomes an address and where the per-sample extent comes from; the three kernels below are
// written once against this accessor pair. Each keeps the exact coordinate convention its wrapper
// expects -- the four spellings are not interchangeable.
template<typename T, bool IsPlanar, class Wrapper>
struct TensorImageAccessor
{
    using value_type = T;

    Wrapper wrap;
    int     imgWidth;
    int     imgHeight;

    inline __device__ int width(int) const
    {
        return imgWidth;
    }

    inline __device__ int height(int) const
    {
        return imgHeight;
    }

    inline __device__ T &at(int batch, int y, int x, int ch) const
    {
        if constexpr (IsPlanar)
        {
            return wrap[int4{x, y, ch, batch}];
        }
        else
        {
            return wrap[int4{ch, x, y, batch}];
        }
    }
};

template<typename T, bool IsPlanar, class Wrapper>
struct VarShapeImageAccessor
{
    using value_type = T;

    Wrapper wrap;

    inline __device__ int width(int batch) const
    {
        return wrap.width(batch);
    }

    inline __device__ int height(int batch) const
    {
        return wrap.height(batch);
    }

    inline __device__ T &at(int batch, int y, int x, int ch) const
    {
        if constexpr (IsPlanar)
        {
            return wrap[int4{x, y, ch, batch}];
        }
        else
        {
            return wrap[int4{batch, y, x, ch}];
        }
    }
};

template<int PixelsPerThreadX, typename T, class SrcAccessor, class HistWrapper>
__global__ void HistogramEqHistKernel(const SrcAccessor src, HistWrapper histogram, int channels)
{
    const int src_x0    = blockIdx.x * blockDim.x * PixelsPerThreadX + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    const int local_id  = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ int shist[];
    if (local_id < kHistogramBins * channels)
    {
        shist[local_id] = 0;
    }
    __syncthreads();

    const int imgWidth  = src.width(batch_idx);
    const int imgHeight = src.height(batch_idx);

#pragma unroll
    for (int i = 0; i < PixelsPerThreadX; ++i)
    {
        const int src_x = src_x0 + i * blockDim.x;

        // Skip widened X positions that extend beyond the final partial tile.
        if (src_x < imgWidth && src_y < imgHeight)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                uchar out = ToBin(src.at(batch_idx, src_y, src_x, ch));
                int   idx = out + (kHistogramBins * ch);
                atomicAdd(&shist[idx], 1);
            }
        }
    }
    __syncthreads();

    if (local_id < kHistogramBins * channels)
    {
        int hist_val = shist[local_id];
        if (hist_val > 0)
        {
            int2 coordHisto{local_id, batch_idx};
            atomicAdd(&histogram[coordHisto], hist_val);
        }
    }
}

template<class CdfWrapper, class DstAccessor>
__global__ void HistogramEqPrefixSumWithNormKernel(CdfWrapper histogram, const DstAccessor dst)
{
    const int tid       = threadIdx.x; // thread id in the block 0-255
    const int batch_idx = blockIdx.z;

    const int  hist_idx = threadIdx.x + (blockIdx.x * kHistogramBins); // index into the histogram 0-255*channels
    const int2 coordHisto{hist_idx, batch_idx};

    __shared__ int temp[kHistogramBins * 2];
    int           *reduce_buf = &temp[kHistogramBins];

    temp[tid] = histogram[coordHisto];

    if (temp[tid])
    {
        reduce_buf[tid] = tid;
    }
    else
    {
        reduce_buf[tid] = kHistogramBins - 1;
    }
    __syncthreads();

    //min-reduce
    for (int s = kHistogramBins / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            reduce_buf[tid] = min(reduce_buf[tid], reduce_buf[tid + s]);
        }
        __syncthreads();
    }

    const int min_idx   = reduce_buf[0]; // this is the first non-zero element in the histogram
    const int min_count = temp[min_idx];
    // Every warp must finish reading the reduction before the scan reuses either half.
    __syncthreads();

    const int total_pixels = dst.width(batch_idx) * dst.height(batch_idx);

    // compute and normalize cdf put into histogram
    if (min_count == total_pixels)
    {
        // all pixels have same value
        histogram[coordHisto] = min_idx;
    }
    else
    {
        if (tid == 0)
        {
            temp[min_idx] = 0;
        }
        __syncthreads();
        const int pout        = InclusiveScan256(temp, tid);
        histogram[coordHisto] = cuda::SaturateCast<int>(1.0 * temp[pout * kHistogramBins + tid]
                                                        / temp[pout * kHistogramBins + kHistogramBins - 1] * 255);
    }
}

template<class CdfWrapper, class DstAccessor>
__global__ void HistogramEqPillowLutKernel(CdfWrapper histogram, const DstAccessor dst)
{
    const int tid       = threadIdx.x;
    const int batch_idx = blockIdx.z;

    const int  hist_idx = threadIdx.x + (blockIdx.x * kHistogramBins);
    const int2 coordHisto{hist_idx, batch_idx};

    __shared__ int scan[kHistogramBins * 2];

    scan[tid] = histogram[coordHisto];
    __syncthreads();

    const int  pout       = InclusiveScan256(scan, tid);
    const int *cumulative = &scan[pout * kHistogramBins];
    int       *reduce_buf = &scan[(1 - pout) * kHistogramBins];

    // The cumulative histogram's maximum is its last entry, and argmax is the first bin reaching
    // it -- the last non-empty bin. Its predecessor counts Pillow's num_non_max_pixels.
    // Parallel min-reduce (same pattern as the OpenCV kernel) over the freed scan half.
    const int total = cumulative[kHistogramBins - 1];
    reduce_buf[tid] = (cumulative[tid] == total) ? tid : (kHistogramBins - 1);
    __syncthreads();
    for (int s = kHistogramBins / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            reduce_buf[tid] = min(reduce_buf[tid], reduce_buf[tid + s]);
        }
        __syncthreads();
    }
    const int argmax = reduce_buf[0];
    const int step   = argmax == 0 ? 0 : cumulative[argmax - 1] / 255;

    int lut;
    if (step == 0)
    {
        lut = tid;
    }
    else if (tid == 0)
    {
        lut = 0;
    }
    else
    {
        lut = (cumulative[tid - 1] + step / 2) / step;
        lut = lut > 255 ? 255 : lut;
    }

    histogram[coordHisto] = lut;
}

template<int PixelsPerThreadX, typename T, class SrcAccessor, class DstAccessor, class CdfWrapper>
__global__ void HistogramEqLookupKernel(const SrcAccessor src, DstAccessor dst, CdfWrapper cdf, int channels)
{
    const int             src_x0    = blockIdx.x * blockDim.x * PixelsPerThreadX + threadIdx.x;
    const int             src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int             batch_idx = blockIdx.z;
    const int             local_id  = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ int temp[];

    //copy the cdf to the block shared memory
    if (local_id < kHistogramBins * channels)
    {
        int2 coordHisto{local_id, batch_idx};
        temp[local_id] = cdf[coordHisto];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < PixelsPerThreadX; ++i)
    {
        const int src_x = src_x0 + i * blockDim.x;

        if (src_x < dst.width(batch_idx) && src_y < dst.height(batch_idx))
        {
            for (int ch = 0; ch < channels; ch++)
            {
                const int offset = kHistogramBins * ch;
                dst.at(batch_idx, src_y, src_x, ch)
                    = FromBin<T>(cuda::SaturateCast<uchar>(temp[ToBin(src.at(batch_idx, src_y, src_x, ch)) + offset]));
            }
        }
    }
}

// Integer histogram counts are order-independent; matching Plan and carrier width preserves channel mapping.

constexpr int kStreamBlock       = 256;
constexpr int kStreamGroups      = 4;
constexpr int kStreamBlocksPerSM = 8;

template<int C, bool IsPlanar>
struct HistogramEqCdfTable
{
    static constexpr int kTables = IsPlanar ? 1 : C;

    struct Param
    {
        const int *cdf;
        int32_t    histoPitch;
    };

    inline static __device__ uchar Entry(const Param &p, int sample, int plane, int table, int value)
    {
        const int base = sample * p.histoPitch + (IsPlanar ? plane : table) * kHistogramBins;
        return cuda::SaturateCast<uchar>(p.cdf[base + value]);
    }
};

template<int C, bool IsPlanar>
using HistogramEqCarrier = elem_stream::LutCarrier<HistogramEqCdfTable<C, IsPlanar>>;

template<int C, bool IsPlanar>
__global__ __launch_bounds__(kStreamBlock) void HistogramEqStreamHistKernel(const uchar *__restrict__ src,
                                                                            int *__restrict__ histo, int32_t histoPitch,
                                                                            int32_t runCarriers, int32_t runTailBytes,
                                                                            int32_t regionsPerPlane,
                                                                            int64_t regionStride, int64_t sampleStride)
{
    using Carrier      = HistogramEqCarrier<C, IsPlanar>;
    using Vec          = typename Carrier::type;
    constexpr int kNT  = Carrier::kTables;
    constexpr int kCar = Carrier::kBytes;

    __shared__ int sh[kNT * kHistogramBins];
    for (int t = threadIdx.x; t < kNT * kHistogramBins; t += kStreamBlock)
    {
        sh[t] = 0;
    }
    __syncthreads();

    const int     region = static_cast<int>(blockIdx.y);
    const int     plane  = IsPlanar ? region / regionsPerPlane : 0;
    const int64_t off = static_cast<int64_t>(blockIdx.z) * sampleStride + static_cast<int64_t>(region) * regionStride;

    const uchar *p = src + off;
    const Vec   *s = reinterpret_cast<const Vec *>(p);

    const int32_t sweep = static_cast<int32_t>(gridDim.x * kStreamBlock);
    for (int32_t i = static_cast<int32_t>(blockIdx.x * kStreamBlock + threadIdx.x); i < runCarriers; i += sweep)
    {
        Vec   v = s[i];
        uchar b[kCar];
        memcpy(b, &v, kCar);
#pragma unroll
        for (int k = 0; k < kCar; ++k)
        {
            atomicAdd(&sh[elem_stream::LutTableIndex(kNT, k, b[k])], 1);
        }
    }

    // The tail starts at channel 0 because runs do and carrier width is a multiple of the table count.
    if (blockIdx.x == 0)
    {
        const int32_t tail = runCarriers * kCar;
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < runTailBytes; i += kStreamBlock)
        {
            atomicAdd(&sh[elem_stream::LutTableIndex(kNT, i, p[tail + i])], 1);
        }
    }

    __syncthreads();
    const int chBase = IsPlanar ? plane * kHistogramBins : 0;
    for (int t = threadIdx.x; t < kNT * kHistogramBins; t += kStreamBlock)
    {
        const int v = sh[t];
        if (v > 0)
        {
            atomicAdd(&histo[blockIdx.z * histoPitch + chBase + t], v);
        }
    }
}

template<int C, bool IsPlanar, class CdfWrapper, class DstAccessor>
inline bool TryRunHistogramEqStream(const nvcv::TensorDataStridedCuda              &inData,
                                    const nvcv::TensorDataStridedCuda              &outData,
                                    const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                    const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int *histo,
                                    CdfWrapper histoWrap, const DstAccessor &dst, int32_t histoPitch, int batch,
                                    cudaStream_t stream)
{
    using ValueT  = std::conditional_t<IsPlanar, uchar, cuda::MakeType<uchar, C>>;
    using Carrier = HistogramEqCarrier<C, IsPlanar>;
    using Traits  = elem_stream::CarrierTraits<Carrier>;

    elem_stream::Plan plan;
    if (!elem_stream::MakePlan<IsPlanar, ValueT>(srcAccess, dstAccess, plan, Traits::kSplitPlanes, Traits::kStateful))
    {
        return false;
    }
    if (!elem_stream::IsCarrierAligned(inData.basePtr(), outData.basePtr(), plan))
    {
        return false;
    }

    const int32_t runCarriers = static_cast<int32_t>(plan.runBytes / Carrier::kBytes);
    const int32_t runTail = static_cast<int32_t>(plan.runBytes - static_cast<int64_t>(runCarriers) * Carrier::kBytes);

    // Cap blocks by device capacity to limit block-local histogram merges into global memory.
    const int64_t regions = std::max<int64_t>(static_cast<int64_t>(plan.numRegions) * plan.numSamples, 1);
    const int64_t budget  = elem_stream::DeviceBlockBudget(kStreamBlocksPerSM, regions);
    const int64_t blocksX = std::max<int64_t>(std::min<int64_t>(util::DivUp(runCarriers, kStreamBlock), budget), 1);

    const dim3 histGrid(static_cast<unsigned int>(blocksX), static_cast<unsigned int>(plan.numRegions),
                        static_cast<unsigned int>(plan.numSamples));
    HistogramEqStreamHistKernel<C, IsPlanar><<<histGrid, kStreamBlock, 0, stream>>>(
        reinterpret_cast<const uchar *>(inData.basePtr()), histo, histoPitch, runCarriers, runTail,
        plan.regionsPerPlane, plan.srcRegionStride, plan.srcSampleStride);
    NVCV_CHECK_THROW(cudaGetLastError());

    HistogramEqPrefixSumWithNormKernel<<<dim3(C, 1, batch), dim3(kHistogramBins, 1, 1), 0, stream>>>(histoWrap, dst);
    NVCV_CHECK_THROW(cudaGetLastError());

    const typename Carrier::Param param{histo, histoPitch};
    const bool ranApply = elem_stream::TryRun<kStreamGroups, Carrier, IsPlanar, ValueT>(stream, inData, outData,
                                                                                        srcAccess, dstAccess, param);
    // Matching Plan/alignment must prevent decline here: fallback would re-accumulate a nonzero histogram.
    if (!ranApply)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL,
                              "HistogramEq apply pass declined a plan the histogram pass accepted");
    }
    return true;
}

template<bool IsPlanar, class CdfWrapper, class DstAccessor>
inline bool TryRunHistogramEqStreamDispatch(const nvcv::TensorDataStridedCuda              &inData,
                                            const nvcv::TensorDataStridedCuda              &outData,
                                            const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                            const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int *histo,
                                            CdfWrapper histoWrap, const DstAccessor &dst, int32_t histoPitch, int batch,
                                            int channels, cudaStream_t stream)
{
#define HISTOGRAM_EQ_STREAM_CASE(C)                                                                               \
    case C:                                                                                                       \
        return TryRunHistogramEqStream<C, IsPlanar>(inData, outData, srcAccess, dstAccess, histo, histoWrap, dst, \
                                                    histoPitch, batch, stream)

    switch (channels)
    {
        HISTOGRAM_EQ_STREAM_CASE(1);
    case 2:
        if constexpr (IsPlanar)
        {
            return false;
        }
        else
        {
            return TryRunHistogramEqStream<2, IsPlanar>(inData, outData, srcAccess, dstAccess, histo, histoWrap, dst,
                                                        histoPitch, batch, stream);
        }
        HISTOGRAM_EQ_STREAM_CASE(3);
        HISTOGRAM_EQ_STREAM_CASE(4);
    default:
        return false;
    }
#undef HISTOGRAM_EQ_STREAM_CASE
}

// gridExtent is the extent the launch is sized against: the common image size for a Tensor, the
// batch's maximum image size for a var-shape batch.
//
// util::DivUp is a deliberate divergence from the legacy divUp, which was ceil((float)a/b). Above
// 2^24 a float32 numerator is no longer exact -- the spacing is 2 -- so an odd extent rounds down
// and the trailing tile is never launched, leaving the last pixel column out of the histogram and
// unwritten in the output. This is reachable at U8: a 16777217x1 tensor is 16 MB, ~0.8% of the
// int32 stride guard. Being a power of two does not protect a divisor here; the precision is lost
// in the numerator, before the division. Do not restore the float spelling.
//
// The rounding is done in 64-bit because only extent * elementSize has to fit int32, so a one-row
// U8 image can be within 255 of INT32_MAX wide and `num + den - 1` would overflow int32 there.
// For every extent below that window the 32-bit and 64-bit results are identical.
inline int DivUpExtent(int32_t extent, int32_t tile)
{
    return static_cast<int>(util::DivUp(static_cast<int64_t>(extent), static_cast<int64_t>(tile)));
}

template<int PixelsPerThreadX, typename T, class SrcAccessor, class DstAccessor, class HistWrapper>
inline void LaunchHistogramEq(const SrcAccessor &src, const DstAccessor &dst, HistWrapper histo, int batch,
                              int channels, dim3 histBlockSize, nvcv::Size2D gridExtent, NVCVHistogramEqMethod method,
                              cudaStream_t stream)
{
    const size_t sharedMemSize = kHistogramBins * channels * sizeof(int);

    {
        const int histTileW = static_cast<int>(histBlockSize.x) * PixelsPerThreadX;
        const int histTileH = static_cast<int>(histBlockSize.y);
        dim3      histGridSize(DivUpExtent(gridExtent.w, histTileW), DivUpExtent(gridExtent.h, histTileH), batch);

        HistogramEqHistKernel<PixelsPerThreadX, T>
            <<<histGridSize, histBlockSize, sharedMemSize, stream>>>(src, histo, channels);
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    // compute cdf / lut
    {
        dim3 lutBlockSize(kHistogramBins, 1, 1);
        dim3 lutGridSize(channels, 1, batch);

        if (method == NVCV_HISTOGRAM_EQ_OPENCV)
        {
            HistogramEqPrefixSumWithNormKernel<<<lutGridSize, lutBlockSize, 0, stream>>>(histo, dst);
        }
        else
        {
            HistogramEqPillowLutKernel<<<lutGridSize, lutBlockSize, 0, stream>>>(histo, dst);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    {
        dim3 lookupBlockSize(32, 32, 1);
        dim3 lookupGridSize(DivUpExtent(gridExtent.w, 32 * kLookupPixelsPerThreadX), DivUpExtent(gridExtent.h, 32),
                            batch);

        HistogramEqLookupKernel<kLookupPixelsPerThreadX, T>
            <<<lookupGridSize, lookupBlockSize, sharedMemSize, stream>>>(src, dst, histo, channels);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

inline bool IsSupportedLayout(const nvcv::TensorLayout &layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW || layout == nvcv::TENSOR_NHWC
        || layout == nvcv::TENSOR_HWC;
}

inline bool IsPlanarLayout(const nvcv::TensorLayout &layout)
{
    return layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW;
}

// The legacy gate compared GetLegacyDataType(dtype) against a DataType code, and that classifier
// keys on bits-per-channel plus data kind rather than on the exact nvcv type. Its kCV_8U code
// therefore also covered the packed TYPE_2U8/TYPE_3U8/TYPE_4U8 types. Nothing upstream narrows the
// dtype to a scalar type first -- the channel count comes from the shape's C axis, not from the
// data type -- so width plus kind is the equivalent test and `dtype() == TYPE_U8` would not be.
inline bool HasElementFormat(nvcv::DataType dtype, nvcv::DataKind kind, int32_t bitsPerChannel)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        // Legacy rejected mixed-depth types outright, with the same status.
        if (bpc[i] != bpc[0])
        {
            return false;
        }
    }

    return dtype.dataKind() == kind && bpc[0] == bitsPerChannel;
}

// Legacy compared the two batches' GetLegacyDataType codes, which are exactly (bits-per-channel,
// data kind) pairs, not the image formats themselves: U8 and RGB8 planes both classify as kCV_8U.
inline bool HaveSameElementFormat(nvcv::DataType a, nvcv::DataType b)
{
    return a.dataKind() == b.dataKind() && a.bitsPerChannel()[0] == b.bitsPerChannel()[0];
}

inline bool IsSupportedElem(nvcv::DataType dtype)
{
    return HasElementFormat(dtype, nvcv::DataKind::UNSIGNED, 8) || HasElementFormat(dtype, nvcv::DataKind::FLOAT, 16)
        || HasElementFormat(dtype, nvcv::DataKind::FLOAT, 32) || HasElementFormat(dtype, nvcv::DataKind::FLOAT, 64);
}

template<class Cb>
inline void DispatchElem(nvcv::DataType dtype, const Cb &cb)
{
    if (HasElementFormat(dtype, nvcv::DataKind::UNSIGNED, 8))
    {
        return cb(uchar{});
    }
    if (HasElementFormat(dtype, nvcv::DataKind::FLOAT, 16))
    {
        return cb(__half{});
    }
    if (HasElementFormat(dtype, nvcv::DataKind::FLOAT, 32))
    {
        return cb(float{});
    }
    if (HasElementFormat(dtype, nvcv::DataKind::FLOAT, 64))
    {
        return cb(double{});
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
}

inline void ValidateMethod(NVCVHistogramEqMethod method)
{
    if (method != NVCV_HISTOGRAM_EQ_OPENCV && method != NVCV_HISTOGRAM_EQ_PILLOW)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid HistogramEq method");
    }
}

// The legacy var-shape data format was derived from the batch's unique image format and its image
// count: planar when every channel has its own plane, and "batched" (kNCHW/kNHWC rather than
// kCHW/kHWC) from two images up. Input and output had to agree on both, so both are carried here.
struct VarShapeFormatClass
{
    bool planar;
    bool batched;

    bool operator==(const VarShapeFormatClass &that) const
    {
        return planar == that.planar && batched == that.batched;
    }

    bool operator!=(const VarShapeFormatClass &that) const
    {
        return !(*this == that);
    }
};

inline VarShapeFormatClass ClassifyVarShapeFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &data)
{
    nvcv::ImageFormat fmt = data.uniqueFormat();

    for (int32_t p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planeDataType(p) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (fmt.numPlanes() >= 2 && fmt.numPlanes() != fmt.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
    }

    return VarShapeFormatClass{fmt.numPlanes() >= 2, data.numImages() >= 2};
}

inline void ValidateChannelCount(int channels, bool isPlanar)
{
    if (channels > kMaxChannelCount || channels < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number");
    }
    if (isPlanar && channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar HistogramEq is unsupported");
    }
}

inline void ValidateGridBatch(int batch)
{
    if (batch > kMaxGridBatch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "HistogramEq input exceeds the 65535 batch launch limit");
    }
}

// Occupancy tables carried over verbatim: the block is sized so its thread count covers the
// 256 * channels block-local histogram bins. The two tables agree except at one channel; both are
// kept as-is so each path keeps the launch geometry it had.
inline dim3 TensorHistBlockSize(int channels)
{
    switch (channels)
    {
    case 1:
    case 2:
        return dim3(32, 16, 1); // 512 threads
    case 3:
        return dim3(32, 24, 1); // 768 threads
    default:
        return dim3(32, 32, 1); // 1024 threads, enough for 4 channels of 256 bins
    }
}

inline dim3 VarShapeHistBlockSize(int channels)
{
    switch (channels)
    {
    case 1:
        return dim3(16, 16, 1); // 256 threads
    case 2:
        return dim3(32, 16, 1); // 512 threads
    case 3:
        return dim3(32, 24, 1); // 768 threads
    default:
        return dim3(32, 32, 1); // 1024 threads, enough for 4 channels of 256 bins
    }
}

} // namespace

namespace cvcuda::priv {

HistogramEqScratch::HistogramEqScratch(int32_t maxBatchSize)
{
    if (maxBatchSize < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }

    //histogram is 256 * channels per image
    m_sizeBytes = static_cast<size_t>(maxBatchSize) * kMaxChannelCount * kHistogramBins * sizeof(int32_t);

    if (m_sizeBytes > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&m_data), m_sizeBytes));
    }
}

HistogramEqScratch::~HistogramEqScratch()
{
    if (m_data)
    {
        NVCV_CHECK_LOG(cudaFree(m_data));
        m_data = nullptr;
    }
}

HistogramEq::HistogramEq(uint32_t maxBatchSize)
    // HistogramEqScratch owns a device allocation and has no device-switching logic of its own.
    // PerDeviceResource creates one instance per CUDA device for transparent multi-GPU support.
    : m_maxBatchSize(maxBatchSize)
    , m_scratch([maxBatchSize](int) { return std::make_unique<HistogramEqScratch>(maxBatchSize); })
    , m_scratchVarShape([maxBatchSize](int) { return std::make_unique<HistogramEqScratch>(maxBatchSize); })
{
    if (maxBatchSize == 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "HistogramEq: maxBatchSize must be >= 1");
    }
}

void HistogramEq::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                             NVCVHistogramEqMethod method) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HistogramEq::operator()[Tensor]");
    ValidateMethod(method);

    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    if (!inAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be image-compatible tensor");
    }

    if (inAccess->numSamples() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    HistogramEqScratch &scratch = m_scratch.get();

    if (!IsSupportedLayout(inData->layout()) || !IsSupportedLayout(outData->layout()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }

    RequireSameLayout(*inData, *outData);

    RequireSameDtype(*inData, *outData);

    if (!IsSupportedElem(inData->dtype()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    const bool isPlanar = IsPlanarLayout(inData->layout());

    const int          channels = inAccess->numChannels();
    const int          batch    = inAccess->numSamples();
    const int          width    = inAccess->numCols();
    const int          height   = inAccess->numRows();
    const nvcv::Size2D dstSize{width, height};

    ValidateChannelCount(channels, isPlanar);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be image-compatible tensor");
    }
    if (outAccess->numSamples() != batch || outAccess->numChannels() != channels || outAccess->numCols() != width
        || outAccess->numRows() != height)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output tensor shapes must match");
    }

    ValidateGridBatch(batch);

    const int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    const int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor is too large");
    }

    NVCV_CHECK_THROW(cudaMemsetAsync(scratch.data(), 0, scratch.sizeBytes(), stream));

    // 2d wrap since its an array of [256 * channels] = width, height = samples
    auto histo = cuda::Tensor2DWrap<int, int32_t>(scratch.data(),
                                                  static_cast<int32_t>(kHistogramBins * channels * sizeof(int)));

    const dim3 histBlockSize = TensorHistBlockSize(channels);

    auto launch = [&](auto planarIC, const auto &src, const auto &dst)
    {
        constexpr bool kPlanar = decltype(planarIC)::value;
        using T                = typename std::decay_t<decltype(src)>::value_type;
        if constexpr (std::is_same_v<T, uchar>)
        {
            if (method == NVCV_HISTOGRAM_EQ_OPENCV)
            {
                if (TryRunHistogramEqStreamDispatch<kPlanar>(*inData, *outData, *inAccess, *outAccess,
                                                             reinterpret_cast<int *>(scratch.data()), histo, dst,
                                                             kHistogramBins * channels, batch, channels, stream))
                {
                    return;
                }
            }
        }
        LaunchHistogramEq<kPixelsPerThread, T>(src, dst, histo, batch, channels, histBlockSize, dstSize, method,
                                               stream);
    };

    DispatchElem(
        inData->dtype(),
        [&](auto elem)
        {
            using T = decltype(elem);

            if (isPlanar)
            {
                auto srcWrap = cuda::CreateTensorWrapNCHW<T, int32_t>(*inData);
                auto dstWrap = cuda::CreateTensorWrapNCHW<T, int32_t>(*outData);

                launch(std::true_type{}, TensorImageAccessor<T, true, decltype(srcWrap)>{srcWrap, width, height},
                       TensorImageAccessor<T, true, decltype(dstWrap)>{dstWrap, width, height});
            }
            else
            {
                auto srcWrap = cuda::CreateTensorWrapNHWC<T, int32_t>(*inData);
                auto dstWrap = cuda::CreateTensorWrapNHWC<T, int32_t>(*outData);

                launch(std::false_type{}, TensorImageAccessor<T, false, decltype(srcWrap)>{srcWrap, width, height},
                       TensorImageAccessor<T, false, decltype(dstWrap)>{dstWrap, width, height});
            }
        });
}

void HistogramEq::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                             const nvcv::ImageBatchVarShape &out, NVCVHistogramEqMethod method) const
{
    CVCUDA_NVTX_RANGE("cvcuda::HistogramEq::operator()[VarShape]");
    ValidateMethod(method);

    if (in.numImages() > m_maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    HistogramEqScratch &scratch = m_scratchVarShape.get();

    RequireUniqueFormat(*inData, "input");

    RequireUniqueFormat(*outData, "output");

    // Sequenced so the input batch is diagnosed before the output batch, as legacy did.
    const VarShapeFormatClass inFormatClass  = ClassifyVarShapeFormat(*inData);
    const VarShapeFormatClass outFormatClass = ClassifyVarShapeFormat(*outData);

    if (inFormatClass != outFormatClass)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }

    const nvcv::ImageFormat inFormat  = inData->uniqueFormat();
    const nvcv::ImageFormat outFormat = outData->uniqueFormat();

    const bool isPlanar = inFormatClass.planar;
    const int  channels = inFormat.numChannels();
    const int  batch    = inData->numImages();

    if (!HaveSameElementFormat(inFormat.planeDataType(0), outFormat.planeDataType(0)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and Output formats must be same data type");
    }

    if (!IsSupportedElem(inFormat.planeDataType(0)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid IDataType");
    }

    if (inFormat.numChannels() != outFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and Output must have the same number of channels, got %d and %d",
                              inFormat.numChannels(), outFormat.numChannels());
    }

    ValidateChannelCount(channels, isPlanar);

    if (batch != outData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and Output batch size must be same");
    }

    RequireSameVarShapeImageSizes(in, out);

    ValidateGridBatch(batch);

    auto histo = cuda::Tensor2DWrap<int, int32_t>(scratch.data(),
                                                  static_cast<int32_t>(kHistogramBins * channels * sizeof(int)));

    NVCV_CHECK_THROW(cudaMemsetAsync(scratch.data(), 0, scratch.sizeBytes(), stream));

    const dim3         histBlockSize = VarShapeHistBlockSize(channels);
    const nvcv::Size2D gridExtent    = inData->maxSize();

    auto launch = [&](const auto &src, const auto &dst)
    {
        using T = typename std::decay_t<decltype(src)>::value_type;
        LaunchHistogramEq<kPixelsPerThread, T>(src, dst, histo, batch, channels, histBlockSize, gridExtent, method,
                                               stream);
    };

    DispatchElem(inFormat.planeDataType(0),
                 [&](auto elem)
                 {
                     using T = decltype(elem);

                     if (isPlanar)
                     {
                         cuda::ImageBatchVarShapeWrap<T> srcWrap(*inData);
                         cuda::ImageBatchVarShapeWrap<T> dstWrap(*outData);

                         launch(VarShapeImageAccessor<T, true, decltype(srcWrap)>{srcWrap},
                                VarShapeImageAccessor<T, true, decltype(dstWrap)>{dstWrap});
                     }
                     else
                     {
                         cuda::ImageBatchVarShapeWrapNHWC<T> srcWrap(*inData, channels);
                         cuda::ImageBatchVarShapeWrapNHWC<T> dstWrap(*outData, channels);

                         launch(VarShapeImageAccessor<T, false, decltype(srcWrap)>{srcWrap},
                                VarShapeImageAccessor<T, false, decltype(dstWrap)>{dstWrap});
                     }
                 });
}

} // namespace cvcuda::priv
