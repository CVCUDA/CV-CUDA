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
#include "OpMedianBlur.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// The general kernel stages a 32x32 tile of the image in shared memory; the small kernels use a
// 16x16 tile so that each thread can also own a kW*kH scratch array in dynamic shared memory.
constexpr int kGeneralKernelBlock = 32;
constexpr int kSmallKernelBlock   = 16;

// A block of the dynamic-shared-memory kernel needs 16*16*kW*kH*sizeof(T) bytes of scratch. Past
// the 48 KiB per-block limit that launch would simply fail, so the general kernel — which keeps no
// per-thread array — takes over for the larger windows.
constexpr size_t kMaxDynamicSharedMemBytes = 48 * 1024;

// Accessors -------------------------------------------------------------------------------------
//
// The Tensor and VarShape paths differ only in how a (sample, channel, row, column) address is
// resolved, so they go through one of these views and share every kernel below. The legacy layer
// instead carried a near-verbatim copy of all three kernels in a second translation unit.

template<typename T>
class TensorImages
{
public:
    TensorImages(const nvcv::TensorDataStridedCuda &data, const nvcv::TensorDataAccessStridedImagePlanar &access)
        : m_data(cuda::CreateTensorWrapNHW<T, int32_t>(data))
        , m_channels(static_cast<int>(access.numChannels()))
        , m_rows(static_cast<int>(access.numRows()))
        , m_cols(static_cast<int>(access.numCols()))
    {
    }

    inline __device__ int width(int) const
    {
        return m_cols;
    }

    inline __device__ int height(int) const
    {
        return m_rows;
    }

    inline __device__ T *ptr(int sample, int channel, int y, int x) const
    {
        return m_data.ptr(sample, y, x * m_channels + channel);
    }

private:
    // A row-wrap plus an explicit interleaved-channel index, with 32-bit strides. That is the
    // address arithmetic the legacy Ptr2dNHWC accessor emitted; a 4D wrap would multiply by a
    // 64-bit column stride instead, and the fetch sits in the innermost loop of every kernel.
    cuda::Tensor3DWrap<T, int32_t> m_data;
    int                            m_channels;
    int                            m_rows;
    int                            m_cols;
};

// Interleaved and planar batches differ only in where the channel sits in the wrapper's index
// order, so they are one adapter parameterized on the layout, matching same_shape::GetCoordForLayout.
template<typename T, bool IsPlanar>
class VarShapeImages
{
    using Wrap = std::conditional_t<IsPlanar, cuda::ImageBatchVarShapeWrap<T>, cuda::ImageBatchVarShapeWrapNHWC<T>>;

public:
    VarShapeImages(const nvcv::ImageBatchVarShapeDataStridedCuda &data, int numChannels)
        : m_data(MakeWrap(data, numChannels))
    {
    }

    inline __device__ int width(int sample) const
    {
        return m_data.width(sample);
    }

    inline __device__ int height(int sample) const
    {
        return m_data.height(sample);
    }

    inline __device__ T *ptr(int sample, int channel, int y, int x) const
    {
        if constexpr (IsPlanar)
        {
            return m_data.ptr(sample, channel, y, x);
        }
        else
        {
            return m_data.ptr(sample, y, x, channel);
        }
    }

private:
    // A planar batch carries one channel per plane, so the wrapper derives the channel count itself.
    static __host__ Wrap MakeWrap(const nvcv::ImageBatchVarShapeDataStridedCuda &data, int numChannels)
    {
        if constexpr (IsPlanar)
        {
            return Wrap(data);
        }
        else
        {
            return Wrap(data, numChannels);
        }
    }

    Wrap m_data;
};

// The Tensor overload takes a single kernel size for the whole batch; the VarShape overload reads
// one per sample out of a device tensor. Both kernels index a source of this shape.
struct UniformKernelSize
{
    int2 value;

    inline __device__ int2 operator[](int) const
    {
        return value;
    }
};

// Window addressing -----------------------------------------------------------------------------

// The kW*kH neighbourhood of one output pixel, seen as a flat array. Elements inside the block's
// staged tile are served from shared memory and the rest from global memory; out-of-image
// coordinates are clamped, which is BORDER_REPLICATE. Legacy spelled this as two function-like
// macros repeated in both translation units.
template<int BlockSize, typename T, class SrcView>
class MedianWindow
{
public:
    // The coordinate pairs are grouped so that a transposed argument is a type error rather than a
    // silent wrong-pixel read: this is built with eleven values that are otherwise all plain ints.
    inline __device__ MedianWindow(const T *tile, SrcView src, int batchIdx, int channel, int2 imgSize,
                                   int2 blockOrigin, int2 pixel, int2 kernelSize)
        : m_tile(tile)
        , m_src(src)
        , m_batchIdx(batchIdx)
        , m_channel(channel)
        , m_imgSize(imgSize)
        , m_blockOrigin(blockOrigin)
        , m_pixel(pixel)
        , m_kernelSize(kernelSize)
    {
    }

    inline __device__ T operator[](int idx) const
    {
        const int gx = min(max(m_pixel.x - (m_kernelSize.x / 2) + (idx % m_kernelSize.x), 0), m_imgSize.x - 1);
        const int gy = min(max(m_pixel.y - (m_kernelSize.y / 2) + (idx / m_kernelSize.x), 0), m_imgSize.y - 1);

        const int sx = gx - m_blockOrigin.x;
        const int sy = gy - m_blockOrigin.y;
        if (sx < 0 || sx >= BlockSize || sy < 0 || sy >= BlockSize)
        {
            return *m_src.ptr(m_batchIdx, m_channel, gy, gx);
        }
        return m_tile[sy * BlockSize + sx];
    }

private:
    const T *m_tile;
    SrcView  m_src;
    int      m_batchIdx;
    int      m_channel;
    int2     m_imgSize;
    int2     m_blockOrigin;
    int2     m_pixel;
    int2     m_kernelSize;
};

// Kernels ---------------------------------------------------------------------------------------

// General kernel: narrows [min_, max_) by repeated three-way partitioning around a median-of-three
// pivot until the median's rank falls inside the equal bucket. Holds no per-thread array, so it is
// the only kernel that can serve windows too large for the dynamic shared-memory budget.
template<typename T, class SrcView, class DstView, class KernelSizeSource>
__global__ void MedianGeneral(const SrcView src, DstView dst, KernelSizeSource ksize, const int channels)
{
    constexpr int kBlock = kGeneralKernelBlock;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int blockX   = blockIdx.x * blockDim.x;
    const int blockY   = blockIdx.y * blockDim.y;
    const int x        = blockX + tx;
    const int y        = blockY + ty;
    const int channel  = blockIdx.z % channels;
    const int batchIdx = blockIdx.z / channels;
    const int h = src.height(batchIdx), w = src.width(batchIdx);

    const int2 kernelSize = ksize[batchIdx];
    const int  kWidth     = kernelSize.x;
    const int  kHeight    = kernelSize.y;

    __shared__ T tails[kBlock * kBlock];
    if (x < w && y < h)
    {
        tails[ty * kBlock + tx] = *src.ptr(batchIdx, channel, y, x);
    }
    __syncthreads();

    if (x >= w || y >= h)
    {
        return;
    }

    const MedianWindow<kBlock, T, SrcView> window(tails, src, batchIdx, channel, int2{w, h}, int2{blockX, blockY},
                                                  int2{x, y}, kernelSize);

    // min_ and max_ set up a range that we are looking for
    // only elements in that range could be median
    T    tmp, pivot0, pivot1, pivot2, min_, max_;
    // In the 1st and possibly several following iterations, min_ or max_ is not assigned.
    // use isMinReady and isMaxReady to control from comparison on them.
    bool isMinReady = false, isMaxReady = false;
    int  numOfEq = 0, numOfGt = 0, numOfLt = 0, numOfTaken = 0;
    int  median = (kWidth * kHeight) / 2;
    int  start = 0, end = kWidth * kHeight, t;
    bool isAllPreviousOutOfRange = true;

    // loop until we rule out all possible elements, and the last pivot is the median.
    while (numOfTaken < (kWidth * kHeight))
    {
        pivot0 = window[start];
        while ((isMinReady && (min_ >= pivot0)) || (isMaxReady && (max_ <= pivot0)))
        {
            start++;
            pivot0 = window[start];
        }

        pivot2 = window[end - 1];
        while ((isMinReady && (min_ >= pivot2)) || (isMaxReady && (max_ <= pivot2)))
        {
            end--;
            pivot2 = window[end - 1];
        }

        int idx = (start + end) / 2;
        pivot1  = window[idx];
        // check if the pivot is in the range defined by min_ and max_.
        // if not, go to the next until we find one that is in the range.
        while ((isMinReady && (min_ >= pivot1)) || (isMaxReady && (max_ <= pivot1)))
        {
            idx++;
            if (idx >= end)
            {
                idx = start;
            }
            pivot1 = window[idx];
        }

        if (pivot0 < pivot1 && pivot1 < pivot2)
        {
            pivot0 = pivot1;
        }
        else if (pivot0 < pivot2 && pivot2 < pivot1)
        {
            pivot0 = pivot2;
        }

        // use the pivot to partition the array.
        t = end;
        for (int i = start; i < t; i++)
        {
            tmp = window[i];
            // only consider the element in the range defined by min_ and max_.
            // because others are already ruled out.
            if ((!isMinReady || min_ < tmp) && (!isMaxReady || tmp < max_))
            {
                if (tmp > pivot0)
                {
                    numOfGt++;
                }
                else if (tmp < pivot0)
                {
                    numOfLt++;
                }
                else
                {
                    numOfEq++;
                }
                if (isAllPreviousOutOfRange)
                {
                    start                   = i;
                    isAllPreviousOutOfRange = false;
                }
                end = i + 1;
            }
        }

        // if the index of median is less than numOfLt,
        // use max_ to rule out elements greater than or equal to pivot.
        if (median < numOfLt)
        {
            max_       = pivot0;
            numOfTaken = numOfTaken + numOfEq + numOfGt;
            isMaxReady = true;
            // if the index of median is in between numOfLt and (numOfLt + numOfEq).
            // the median is found. we are lucky:)
        }
        else if (median < (numOfLt + numOfEq))
        {
            break;
            // if the index of median is greater than (numOfLt + numOfEq),
            // use min_ to rule out elements greater than or equal to pivot.
        }
        else
        {
            min_       = pivot0;
            median     = median - numOfLt - numOfEq;
            numOfTaken = numOfTaken + numOfLt + numOfEq;
            isMinReady = true;
        }
        numOfLt = 0;
        numOfEq = 0;
        numOfGt = 0;
    }
    *dst.ptr(batchIdx, channel, y, x) = pivot0;
}

template<typename T>
__device__ int Partition(T *arr, int length, T pvt, int *numOfEq)
{
    T val;
    *numOfEq = 1;
    int i    = 1;
    for (int j = 1; j < length; j++)
    {
        val = arr[j];
        if (val == pvt)
        {
            (*numOfEq) += 1;
        }
        if (val < pvt)
        {
            arr[j] = arr[i];
            arr[i] = val;
            i += 1;
        }
    }
    val        = arr[0];
    arr[0]     = arr[i - 1];
    arr[i - 1] = val;
    return i - 1;
}

template<typename T>
__device__ __forceinline__ T PlacePivot(T *arr, int length)
{
    int mid    = length / 2;
    T   pivot0 = arr[0], pivot1 = arr[mid], pivot2 = arr[length - 1];
    if (pivot0 < pivot1 && pivot1 <= pivot2)
    {
        arr[0]   = pivot1;
        arr[mid] = pivot0;
        return pivot1;
    }
    if (pivot0 < pivot2 && pivot2 <= pivot1)
    {
        arr[0]          = pivot2;
        arr[length - 1] = pivot0;
        return pivot2;
    }
    return pivot0;
}

// Small kernel: copies the window into a per-thread dynamic-shared-memory array and quickselects
// in place. Cheaper than the general kernel because the window is read once, but it needs
// 16*16*kW*kH*sizeof(T) bytes of shared memory per block.
template<typename T, class SrcView, class DstView, class KernelSizeSource>
__global__ void MedianSmall(const SrcView src, DstView dst, KernelSizeSource ksize, const int channels)
{
    constexpr int kBlock = kSmallKernelBlock;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int blockX   = blockIdx.x * blockDim.x;
    const int blockY   = blockIdx.y * blockDim.y;
    const int x        = blockX + tx;
    const int y        = blockY + ty;
    const int channel  = blockIdx.z % channels;
    const int batchIdx = blockIdx.z / channels;
    const int h = src.height(batchIdx), w = src.width(batchIdx);

    const int2 kernelSize = ksize[batchIdx];
    const int  kWidth     = kernelSize.x;
    const int  kHeight    = kernelSize.y;

    __shared__ T tails[kBlock * kBlock];
    if (x < w && y < h)
    {
        tails[ty * kBlock + tx] = *src.ptr(batchIdx, channel, y, x);
    }
    __syncthreads();

    if (x >= w || y >= h)
    {
        return;
    }

    extern __shared__ char arrays[];
    int                    length = kWidth * kHeight;
    T                     *arr    = ((T *)arrays) + ((tx * kBlock) + ty) * length;
    T                      pivot;
    int                    numOfEq, k = length / 2;

    const MedianWindow<kBlock, T, SrcView> window(tails, src, batchIdx, channel, int2{w, h}, int2{blockX, blockY},
                                                  int2{x, y}, kernelSize);

    for (int i = 0; i < length; i++)
    {
        arr[i] = window[i];
    }
    while (length > 1)
    {
        pivot      = PlacePivot(arr, length);
        int middle = Partition(arr, length, pivot, &numOfEq);
        if (k < middle)
        {
            length = middle;
        }
        else if (k < (middle + numOfEq))
        {
            *dst.ptr(batchIdx, channel, y, x) = pivot;
            return;
        }
        else
        {
            k      = k - middle - 1;
            length = length - middle - 1;
            arr    = arr + middle + 1;
        }
    }
    *dst.ptr(batchIdx, channel, y, x) = arr[0];
}

template<typename T, int LENGTH>
__device__ __forceinline__ T MedianFromSortedWindow(T (&arr)[LENGTH])
{
#pragma unroll
    for (int i = 0; i < LENGTH - 1; ++i)
    {
#pragma unroll
        for (int j = i + 1; j < LENGTH; ++j)
        {
            if (arr[j] < arr[i])
            {
                T tmp  = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }

    return arr[LENGTH / 2];
}

template<typename T, int LENGTH>
__device__ __forceinline__ bool IsMedianRank(const T (&arr)[LENGTH], T candidate)
{
    constexpr int median = LENGTH / 2;

    int numLess        = 0;
    int numLessOrEqual = 0;
#pragma unroll
    for (int i = 0; i < LENGTH; ++i)
    {
        numLess += arr[i] < candidate;
        numLessOrEqual += arr[i] <= candidate;
    }

    return numLess <= median && median < numLessOrEqual;
}

// Fixed kernel: the window size is a compile-time constant, so it lives in registers and the
// selection unrolls into a sorting network. U8 3x3 windows reuse a shared halo.
template<typename T, int KWidth, int KHeight, class SrcView, class DstView>
__global__ __launch_bounds__(kSmallKernelBlock *kSmallKernelBlock) void MedianFixed(const SrcView src, DstView dst,
                                                                                    const int channels)
{
    constexpr int length = KWidth * KHeight;

    const int x        = blockIdx.x * blockDim.x + threadIdx.x;
    const int y        = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel  = blockIdx.z % channels;
    const int batchIdx = blockIdx.z / channels;
    const int h = src.height(batchIdx), w = src.width(batchIdx);

    constexpr bool kStageHalo = std::is_same_v<T, unsigned char> && KWidth == 3 && KHeight == 3
                             && std::is_same_v<SrcView, TensorImages<const T>>;
    constexpr int kTile = kSmallKernelBlock + 2;
    __shared__ T  tile[kStageHalo ? kTile * kTile : 1];
    if constexpr (kStageHalo)
    {
        // Tail threads also load the replicated halo before any thread exits.
#pragma unroll
        for (int offset = 0; offset < kTile * kTile; offset += kSmallKernelBlock * kSmallKernelBlock)
        {
            const int i = threadIdx.y * kSmallKernelBlock + threadIdx.x + offset;
            if (i < kTile * kTile)
            {
                const int gx = min(max(static_cast<int>(blockIdx.x) * kSmallKernelBlock - 1 + i % kTile, 0), w - 1);
                const int gy = min(max(static_cast<int>(blockIdx.y) * kSmallKernelBlock - 1 + i / kTile, 0), h - 1);
                tile[i]      = *src.ptr(batchIdx, channel, gy, gx);
            }
        }
        __syncthreads();
    }

    if (x >= w || y >= h)
    {
        return;
    }

    T arr[length];
#pragma unroll
    for (int i = 0; i < length; i++)
    {
        if constexpr (kStageHalo)
        {
            arr[i] = tile[(threadIdx.y + i / KWidth) * kTile + threadIdx.x + i % KWidth];
        }
        else
        {
            int gx = x - (KWidth / 2) + (i % KWidth);
            int gy = y - (KHeight / 2) + (i / KWidth);

            gx = min(max(gx, 0), w - 1);
            gy = min(max(gy, 0), h - 1);

            arr[i] = *src.ptr(batchIdx, channel, gy, gx);
        }
    }

    if constexpr (std::is_same_v<T, unsigned char> && KWidth == 5 && KHeight == 5)
    {
        T candidate = arr[length / 2];
        if (IsMedianRank(arr, candidate))
        {
            *dst.ptr(batchIdx, channel, y, x) = candidate;
            return;
        }
    }

    *dst.ptr(batchIdx, channel, y, x) = MedianFromSortedWindow(arr);
}

// Launch ----------------------------------------------------------------------------------------

struct MedianLaunchParams
{
    int  numSamples;
    int  channels;
    int  maxWidth;
    int  maxHeight;
    int  maxKWidth;
    int  maxKHeight;
    bool uniformKernelSize;
};

// Kernel selection, in the order established by the operator's optimization campaign: the fixed
// sorting-network kernels first (they need a compile-time window, so a var-shape batch qualifies
// only when every image shares one kernel size), then the dynamic shared-memory kernel, then the
// general kernel for windows that no longer fit in shared memory.
template<typename T, class SrcView, class DstView, class KernelSizeSource>
void LaunchMedian(cudaStream_t stream, const SrcView &src, const DstView &dst, const KernelSizeSource &ksize,
                  const MedianLaunchParams &p)
{
    // Both smallGrid and generalGrid use p.channels * p.numSamples for their z dimension.
    // CUDA limits grid.z to 65535, so reject oversized batches before building either grid.
    if (static_cast<int64_t>(p.channels) * p.numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "MedianBlur requires channels * numSamples <= 65535 (CUDA grid-z limit)");
    }
    const dim3 smallBlock(kSmallKernelBlock, kSmallKernelBlock);
    const dim3 smallGrid(util::DivUp(p.maxWidth, kSmallKernelBlock), util::DivUp(p.maxHeight, kSmallKernelBlock),
                         p.channels * p.numSamples);

    auto runDynamicKernel = [&]
    {
        const size_t sharedMemSize
            = static_cast<size_t>(kSmallKernelBlock) * kSmallKernelBlock * p.maxKWidth * p.maxKHeight * sizeof(T);
        if (sharedMemSize < kMaxDynamicSharedMemBytes)
        {
            MedianSmall<T><<<smallGrid, smallBlock, sharedMemSize, stream>>>(src, dst, ksize, p.channels);
        }
        else
        {
            const dim3 generalBlock(kGeneralKernelBlock, kGeneralKernelBlock);
            const dim3 generalGrid(util::DivUp(p.maxWidth, kGeneralKernelBlock),
                                   util::DivUp(p.maxHeight, kGeneralKernelBlock), p.channels * p.numSamples);
            MedianGeneral<T><<<generalGrid, generalBlock, 0, stream>>>(src, dst, ksize, p.channels);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    };

    if (p.uniformKernelSize && p.maxKWidth == 3 && p.maxKHeight == 3)
    {
        MedianFixed<T, 3, 3><<<smallGrid, smallBlock, 0, stream>>>(src, dst, p.channels);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    // __half shares float's fixed-5x5 sorting-network envelope: the dynamic shared-memory kernel is
    // ~2.7x slower per image for half, and the fixed kernel is the same template the unconditional
    // 3x3 path already instantiates and validates for __half.
    else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, unsigned char> || std::is_same_v<T, __half>)
    {
        if (p.uniformKernelSize && p.maxKWidth == 5 && p.maxKHeight == 5)
        {
            MedianFixed<T, 5, 5><<<smallGrid, smallBlock, 0, stream>>>(src, dst, p.channels);
            NVCV_CHECK_THROW(cudaGetLastError());
        }
        else if constexpr (std::is_same_v<T, unsigned char>)
        {
            if (p.uniformKernelSize && p.maxKWidth == 7 && p.maxKHeight == 7)
            {
                MedianFixed<T, 7, 7><<<smallGrid, smallBlock, 0, stream>>>(src, dst, p.channels);
                NVCV_CHECK_THROW(cudaGetLastError());
            }
            else
            {
                runDynamicKernel();
            }
        }
        else
        {
            runDynamicKernel();
        }
    }
    else
    {
        runDynamicKernel();
    }
}

// Validation ------------------------------------------------------------------------------------

// Resolve the element type the kernels run on. The median only compares and selects input values,
// so __half needs no arithmetic of its own and reuses the same templates.
template<class Cb>
void RunTypeSwitch(nvcv::DataType dtype, const Cb &cb)
{
    const auto bpc         = dtype.bitsPerChannel();
    const int  numChannels = dtype.numChannels();
    for (int i = 1; i < numChannels; ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    const nvcv::DataKind kind = dtype.dataKind();
    if (kind == nvcv::DataKind::UNSIGNED)
    {
        if (bpc[0] == 8)
        {
            return cb(static_cast<unsigned char>(0));
        }
        if (bpc[0] == 16)
        {
            return cb(static_cast<unsigned short>(0));
        }
    }
    else if (kind == nvcv::DataKind::FLOAT)
    {
        if (bpc[0] == 16)
        {
            return cb(__half{});
        }
        if (bpc[0] == 32)
        {
            return cb(0.f);
        }
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Invalid data type: MedianBlur supports 8-bit unsigned, 16-bit unsigned, 16-bit float and "
                          "32-bit float");
}

// Reject an unsupported element type where the legacy layer did — ahead of the kernel-size and
// channel checks — so an input that fails several of them still reports the same one.
inline void ValidateDataType(nvcv::DataType dtype)
{
    RunTypeSwitch(dtype, [](auto) {});
}

inline void ValidateLayout(const nvcv::TensorLayout &layout)
{
    if (!(layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_NCHW
          || layout == nvcv::TENSOR_CHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid layout, the valid layouts are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
    }
}

inline void ValidateChannels(int channels)
{
    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid channel number %d, MedianBlur supports 1, 3 or 4 channels", channels);
    }
}

inline void ValidateKernelSize(int kWidth, int kHeight)
{
    if (!(kWidth > 0 && kWidth % 2 == 1 && kHeight > 0 && kHeight % 2 == 1))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid ksize %dx%d, both must be positive and odd", kWidth, kHeight);
    }
}

// Tensor path -----------------------------------------------------------------------------------

template<typename T>
void RunMedianTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                     const nvcv::TensorDataStridedCuda &outData, const nvcv::Size2D &ksize)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    cvcuda::priv::RequireImageAccess(inAccess, "Input");
    cvcuda::priv::RequireImageAccess(outAccess, "Output");
    cvcuda::priv::RequireInt32Addressable(*inAccess, *outAccess);
    cvcuda::priv::RequireSameImageShape(*inAccess, *outAccess);

    const TensorImages<const T> src(inData, *inAccess);
    const TensorImages<T>       dst(outData, *outAccess);

    LaunchMedian<T>(stream, src, dst,
                    UniformKernelSize{
                        int2{ksize.w, ksize.h}
    },
                    MedianLaunchParams{static_cast<int>(inAccess->numSamples()),
                                       static_cast<int>(inAccess->numChannels()), static_cast<int>(inAccess->numCols()),
                                       static_cast<int>(inAccess->numRows()), ksize.w, ksize.h, true});
}

inline void MedianBlurTensor(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                             const nvcv::TensorDataStridedCuda &outData, const nvcv::Size2D &ksize)
{
    ValidateLayout(inData.layout());
    ValidateLayout(outData.layout());
    if (inData.layout() != outData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }

    // The public header's "Input/Output dependency" table requires input and output to agree on
    // every one of these; see OpMedianBlur_Negative.tensor_output_data_type_mismatch.
    if (inData.dtype() != outData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must be image-like tensors");
    }
    if (inAccess->numSamples() != outAccess->numSamples() || inAccess->numChannels() != outAccess->numChannels()
        || inAccess->numRows() != outAccess->numRows() || inAccess->numCols() != outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of samples, channels, rows and columns");
    }

    ValidateDataType(inData.dtype());
    ValidateKernelSize(ksize.w, ksize.h);
    ValidateChannels(static_cast<int>(inAccess->numChannels()));

    // Each channel plane is blurred independently, so a packed planar tensor can be viewed as an
    // (N*C, H, W, 1) interleaved batch and fed to the very same kernels. Interleaved input is
    // returned unflattened.
    const auto                         views = cvcuda::priv::PlanarSingleChannelViews(inData, outData);
    const nvcv::TensorDataStridedCuda &src   = views ? views->first : inData;
    const nvcv::TensorDataStridedCuda &dst   = views ? views->second : outData;

    RunTypeSwitch(inData.dtype(), [&](auto value) { RunMedianTensor<decltype(value)>(stream, src, dst, ksize); });
}

// VarShape path ---------------------------------------------------------------------------------

template<typename T>
void RunMedianVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const nvcv::TensorDataStridedCuda &ksize,
                       bool isPlanar, const MedianLaunchParams &params)
{
    const cuda::Tensor1DWrap<const int2> ksizeWrap(ksize);

    if (isPlanar)
    {
        LaunchMedian<T>(stream, VarShapeImages<const T, true>(inData, params.channels),
                        VarShapeImages<T, true>(outData, params.channels), ksizeWrap, params);
    }
    else
    {
        LaunchMedian<T>(stream, VarShapeImages<const T, false>(inData, params.channels),
                        VarShapeImages<T, false>(outData, params.channels), ksizeWrap, params);
    }
}

// Classify a var-shape batch's layout the way the legacy layer did: one plane is interleaved, one
// plane per channel is planar, anything else the operator cannot address.
inline bool IsPlanarBatch(const nvcv::ImageFormat &format)
{
    const int            numPlanes = format.numPlanes();
    const nvcv::DataType planeType = format.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (format.planeDataType(i) != planeType)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (numPlanes < 2)
    {
        return false;
    }
    if (numPlanes != format.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
    }
    return true;
}

// Read the per-image kernel sizes back to the host and reduce them to the launch parameters the
// fixed-window fast paths are selected from. Each size must be positive and odd.
inline void ReadKernelSizes(cudaStream_t stream, const nvcv::TensorDataStridedCuda &ksize, int numImages,
                            MedianLaunchParams &params)
{
    auto ksizeAccess = nvcv::TensorDataAccessStrided::Create(ksize);
    NVCV_ASSERT(ksizeAccess);

    std::vector<int2> kernelSizes(numImages);
    NVCV_CHECK_THROW(cudaMemcpy2DAsync(kernelSizes.data(), sizeof(int2), ksizeAccess->sampleData(0),
                                       ksizeAccess->sampleStride(), sizeof(int2), numImages, cudaMemcpyDeviceToHost,
                                       stream));
    NVCV_CHECK_THROW(cudaStreamSynchronize(stream));

    for (const int2 &kernelSize : kernelSizes)
    {
        ValidateKernelSize(kernelSize.x, kernelSize.y);

        params.maxKWidth  = std::max(params.maxKWidth, kernelSize.x);
        params.maxKHeight = std::max(params.maxKHeight, kernelSize.y);
        params.uniformKernelSize
            = params.uniformKernelSize && kernelSize.x == kernelSizes[0].x && kernelSize.y == kernelSizes[0].y;
    }
}

inline void MedianBlurVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                               const nvcv::ImageBatchVarShape                &out,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                               const nvcv::TensorDataStridedCuda &ksize, int maxVarShapeBatchSize)
{
    if (maxVarShapeBatchSize <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator median blur var shape is not initialized properly, maxVarShapeBatchSize: %d",
                              maxVarShapeBatchSize);
    }
    const int numImages = inData.numImages();
    if (maxVarShapeBatchSize < numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of images, it should not exceed %d",
                              maxVarShapeBatchSize);
    }

    const nvcv::ImageFormat inFormat = inData.uniqueFormat();
    if (!inFormat || !outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    // The public header's "Input/Output dependency" table requires input and output to agree on
    // layout, data type, channels, number and size; see OpMedianBlurVarshape_Negative's
    // varshape_output_data_type_mismatch, varshape_output_image_count_mismatch and
    // varshape_output_image_size_mismatch.
    if (inFormat != outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }
    if (numImages != outData.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images");
    }
    cvcuda::priv::RequireSameVarShapeImageSizes(in, out);

    const bool           isPlanar  = IsPlanarBatch(inFormat);
    const int            channels  = inFormat.numChannels();
    const nvcv::DataType planeType = inFormat.planeDataType(0);

    ValidateDataType(planeType);
    ValidateChannels(channels);

    // Validate ksize: each sample must supply exactly two S32 values (one int2). Accept both the
    // compound form (rank-1 TYPE_2S32) used by the C++ bench and the NC interop form (rank-2
    // TYPE_S32 with 2 columns) used by PyTorch-sourced int32 tensors; both present sizeof(int2)
    // bytes per sample to ReadKernelSizes and Tensor1DWrap<const int2>. Reject anything else.
    const bool ksizeIs2S32  = ksize.rank() == 1 && ksize.dtype() == nvcv::TYPE_2S32;
    const bool ksizeIsNCS32 = ksize.rank() == 2 && ksize.dtype() == nvcv::TYPE_S32 && ksize.shape(1) == 2;
    if (!ksizeIs2S32 && !ksizeIsNCS32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "ksize must be a 1D TYPE_2S32 tensor or a rank-2 NC TYPE_S32 tensor with 2 columns");
    }
    if (ksize.shape(0) < numImages)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ksize must contain at least %d elements",
                              numImages);
    }

    MedianLaunchParams params{numImages, channels, outData.maxSize().w, outData.maxSize().h, 0, 0, true};
    ReadKernelSizes(stream, ksize, numImages, params);

    RunTypeSwitch(planeType, [&](auto value)
                  { RunMedianVarShape<decltype(value)>(stream, inData, outData, ksize, isPlanar, params); });
}

} // namespace

namespace cvcuda::priv {

MedianBlur::MedianBlur(const int maxVarShapeBatchSize)
    : m_maxVarShapeBatchSize(maxVarShapeBatchSize)
{
}

void MedianBlur::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            const nvcv::Size2D ksize) const
{
    CVCUDA_NVTX_RANGE("cvcuda::MedianBlur::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    MedianBlurTensor(stream, *inData, *outData, ksize);
}

void MedianBlur::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &ksize) const
{
    CVCUDA_NVTX_RANGE("cvcuda::MedianBlur::operator()[ImageBatchVarShape]");
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

    auto ksizeData = ksize.exportData<nvcv::TensorDataStridedCuda>();
    if (ksizeData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "ksize must be a tensor");
    }

    MedianBlurVarShape(stream, in, out, *inData, *outData, *ksizeData, m_maxVarShapeBatchSize);
}

} // namespace cvcuda::priv
