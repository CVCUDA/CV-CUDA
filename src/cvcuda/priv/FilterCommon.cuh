/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CVCUDA_PRIV_FILTER_COMMON_CUH
#define CVCUDA_PRIV_FILTER_COMMON_CUH

#include "Assert.h"
#include "LegacyFormatClassify.hpp"
#include "PlanarSampleDecode.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"

#include <cuda_runtime.h>
// The kernels take their source as a template parameter, but every consumer builds it from one of
// these two wrappers, so the header carries them rather than making each operator repeat the include.
#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/HalfTypes.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/BorderType.h>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

// One 2D-correlation implementation shared by Conv2D, AverageBlur, Gaussian and Laplacian.
//
// The four operators only disagree about where the kernel coefficients come from: Conv2D carries one
// kernel image per sample, Gaussian materializes coefficients on the device from (kernelSize, sigma),
// AverageBlur uses a uniform 1/(kw*kh), and Laplacian uses one of two fixed 3x3 stencils. Everything
// downstream of that -- the tap loops, the shared-memory tiling ladder, the output-batching widths,
// the border handling and the saturating store -- is identical, so it lives here once and each
// operator supplies a coefficient source. That is also what lets legacy/filter.cu and
// legacy/filter_var_shape.cu, which the four previously shared, be deleted together.
namespace cvcuda::priv::filter {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

using uchar  = unsigned char;
using ushort = unsigned short;

// Shared memory per block that the tiling ladder is allowed to request. 48 KiB is the portable
// static limit; above it a launch would need an opt-in carve-out, so the ladder steps down to a
// narrower tile (or to the untiled kernel) instead.
constexpr size_t kMaxSharedBytes = 48 * 1024;

// CUDA caps gridDim.z; planar paths that fold (sample, plane) into z must stay under it.
constexpr int64_t kMaxGridZ = 65535;

// ---------------------------------------------------------------------------------------------
// Destination adaptors
// ---------------------------------------------------------------------------------------------

// Everything the kernels need to know about their destination: how gridDim.z splits into a sample
// and a plane, the extent of that plane, and the address of one pixel. All three adaptors share one
// signature -- the two interleaved ones ignore the plane index, exactly as
// `same_shape::GetCoordForLayout` does -- so one kernel body serves tensors, var-shape batches and
// planar var-shape batches. `kPlanar` is what the body branches on; nothing here adds state beyond
// the wrapper the kernel already carried.
template<class W>
struct UniformDst
{
    using ValueType = typename W::ValueType;

    static constexpr bool kPlanar = false;

    W    dst;
    int2 size;

    inline __device__ int Sample(int z) const
    {
        return z;
    }

    inline __device__ int Plane(int, int) const
    {
        return 0;
    }

    inline __device__ int Width(int, int) const
    {
        return size.x;
    }

    inline __device__ int Height(int, int) const
    {
        return size.y;
    }

    inline __device__ ValueType *Ptr(int b, int, int y, int x) const
    {
        return dst.ptr(b, y, x);
    }
};

template<class W>
struct VarShapeDst
{
    using ValueType = typename W::ValueType;

    static constexpr bool kPlanar = false;

    W dst;

    inline __device__ int Sample(int z) const
    {
        return z;
    }

    inline __device__ int Plane(int, int) const
    {
        return 0;
    }

    inline __device__ int Width(int b, int) const
    {
        return dst.width(b);
    }

    inline __device__ int Height(int b, int) const
    {
        return dst.height(b);
    }

    inline __device__ ValueType *Ptr(int b, int, int y, int x) const
    {
        return dst.ptr(b, y, x);
    }
};

// Planar var-shape batches fold (sample, plane) into gridDim.z, because per-image extents rule out
// the flatten a packed planar tensor gets.
template<class W>
struct VarShapePlanarDst
{
    using ValueType = typename W::ValueType;

    static constexpr bool kPlanar = true;

    W        dst;
    int      channels;
    uint32_t sampleRecip;

    // Multiply-high in place of `z / channels`: the runtime division landed on the vector datapath
    // where the legacy planar kernel had it on the uniform one, and that gap measured ~5.9% on the
    // planar var-shape filters. PlanarSampleDecode.hpp derives the multiplier and states the range
    // over which it is exact.
    inline __device__ int Sample(int z) const
    {
        return static_cast<int>(__umulhi(static_cast<uint32_t>(z) << kPlanarSampleShift, sampleRecip));
    }

    inline __device__ int Plane(int z, int sample) const
    {
        return z - sample * channels;
    }

    inline __device__ int Width(int b, int p) const
    {
        return dst.width(b, p);
    }

    inline __device__ int Height(int b, int p) const
    {
        return dst.height(b, p);
    }

    inline __device__ ValueType *Ptr(int b, int p, int y, int x) const
    {
        return dst.ptr(b, p, y, x);
    }
};

// Address a source sample in either layout from one kernel body: interleaved sources are already
// indexed by (x, y, sample); planar sources splice the plane index in.
template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> SrcCoord(int x, int y, int plane, int sample)
{
    if constexpr (IsPlanar)
    {
        return int4{x, y, plane, sample};
    }
    else
    {
        return int3{x, y, sample};
    }
}

template<class W>
inline UniformDst<W> MakeUniformDst(W dst, int2 size)
{
    return UniformDst<W>{dst, size};
}

template<class W>
inline VarShapeDst<W> MakeVarShapeDst(W dst)
{
    return VarShapeDst<W>{dst};
}

template<class W>
inline VarShapePlanarDst<W> MakeVarShapePlanarDst(W dst, int channels)
{
    return VarShapePlanarDst<W>{dst, channels, PlanarSampleReciprocal(channels)};
}

// ---------------------------------------------------------------------------------------------
// Coefficient sources
// ---------------------------------------------------------------------------------------------

// A coefficient source answers, for sample `b`: the kernel window size, the anchor within it, and a
// coefficient addressed either as (row, column) or by its row-major index. `Linear` exists because
// the tiled kernels stage the whole window into shared memory with a flat, block-strided loop.

// Host-resolved coefficients: one window shared by the whole batch, already normalized. The wrapper
// is anything indexable by a flat offset -- a device pointer for Gaussian, a math::Vector for
// Laplacian's fixed stencils.
template<class KernelWrapper>
struct UniformCoefs
{
    KernelWrapper kernel;
    int2          size;
    int2          anchor;

    inline __device__ int2 Size(int) const
    {
        return size;
    }

    inline __device__ int2 Anchor(int, int2) const
    {
        return anchor;
    }

    inline __device__ float At(int, int i, int j) const
    {
        return kernel[i * size.x + j];
    }

    inline __device__ float Linear(int, int idx, int2) const
    {
        return kernel[idx];
    }
};

template<class KernelWrapper>
inline UniformCoefs<KernelWrapper> MakeUniformCoefs(KernelWrapper kernel, int2 size, int2 anchor)
{
    return UniformCoefs<KernelWrapper>{kernel, size, anchor};
}

// Conv2D: one kernel image per sample, sized by the image itself, with a per-sample anchor tensor
// in which (-1, -1) means "kernel center".
struct ImageBatchCoefs
{
    cuda::ImageBatchVarShapeWrap<float> kernel;
    cuda::Tensor1DWrap<int2, int32_t>   anchors;

    inline __device__ int2 Size(int b) const
    {
        return int2{kernel.width(b), kernel.height(b)};
    }

    inline __device__ int2 Anchor(int b, int2 kernelSize) const
    {
        int2 anchor = anchors[b];
        if (anchor.x < 0)
            anchor.x = kernelSize.x / 2;
        if (anchor.y < 0)
            anchor.y = kernelSize.y / 2;
        return anchor;
    }

    inline __device__ float At(int b, int i, int j) const
    {
        return *kernel.ptr(b, i, j);
    }

    inline __device__ float Linear(int b, int idx, int2 kernelSize) const
    {
        return *kernel.ptr(b, idx / kernelSize.x, idx % kernelSize.x);
    }
};

// Gaussian var-shape: coefficients precomputed into a (sample, row, column) tensor, with the window
// size re-derived per sample so an all-zero kernelSize falls back to the sigma-driven size. The
// derivation is repeated here rather than read back from the host because it is per sample.
struct GaussianBatchCoefs
{
    cuda::Tensor3DWrap<float, int32_t>   kernel;
    cuda::Tensor1DWrap<int2, int32_t>    kernelSizes;
    cuda::Tensor1DWrap<double2, int32_t> sigmas;
    int2                                 maxKernelSize;
    int                                  dataKernelSize;

    inline __device__ int2 Size(int b) const
    {
        int2    kernelSize = kernelSizes[b];
        double2 sigma      = sigmas[b];

        // automatic detection of kernel size from sigma
        if (kernelSize.x <= 0 && sigma.x > 0)
            kernelSize.x = cuda::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
        if (kernelSize.y <= 0 && sigma.y > 0)
            kernelSize.y = cuda::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

        NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.x,
                         "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.x);
        NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.y,
                         "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.y);

        return kernelSize;
    }

    inline __device__ int2 Anchor(int, int2 kernelSize) const
    {
        return int2{kernelSize.x / 2, kernelSize.y / 2};
    }

    inline __device__ float At(int b, int i, int j) const
    {
        return *kernel.ptr(b, i, j);
    }

    inline __device__ float Linear(int b, int idx, int2 kernelSize) const
    {
        return *kernel.ptr(b, idx / kernelSize.x, idx % kernelSize.x);
    }
};

// ---------------------------------------------------------------------------------------------
// Correlation kernels
// ---------------------------------------------------------------------------------------------

// The tiled kernels carve one dynamic shared-memory allocation into the input tile followed by the
// kernel window. The window is float-aligned, so the split must be too -- and it has to agree
// exactly with TiledSharedBytes below, which sizes the allocation.
inline __device__ float *KernelTile(unsigned char *shared, size_t tileBytes)
{
    return reinterpret_cast<float *>(shared + util::RoundUpPowerOfTwo(tileBytes, sizeof(float)));
}

// gridDim.z carries the sample for a tensor or an interleaved batch, and (sample, plane) for a
// planar var-shape batch; the destination adaptor owns that decode so the bodies below do not.
#define NVCV_FILTER_RESOLVE_SAMPLE()              \
    const int batch_idx = dst.Sample(blockIdx.z); \
    const int plane     = dst.Plane(blockIdx.z, batch_idx)

template<class SrcWrapper, class Dst, class Coefs>
__global__ void Filter2D(SrcWrapper src, Dst dst, Coefs coefs)
{
    using T         = typename Dst::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    NVCV_FILTER_RESOLVE_SAMPLE();

    if (x >= dst.Width(batch_idx, plane) || y >= dst.Height(batch_idx, plane))
        return;

    const int2 kernelSize = coefs.Size(batch_idx);
    const int2 anchor     = coefs.Anchor(batch_idx, kernelSize);

    for (int i = 0; i < kernelSize.y; ++i)
    {
        const int cy = y - anchor.y + i;

        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + src[SrcCoord<Dst::kPlanar>(x - anchor.x + j, cy, plane, batch_idx)] * coefs.At(batch_idx, i, j);
        }
    }

    *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
}

template<int Outputs, bool Slide = true, class T>
__device__ __forceinline__ void AccumulateTiled(const T *tile, const float *kernelTile, int tileW, int2 kernelSize,
                                                cuda::ConvertBaseTypeTo<float, T> (&res)[Outputs])
{
    // Reuse overlapping shared loads without changing each output's tap order. Narrow values
    // only benefit at two outputs; longer shift chains cost more than the packed loads they save.
    constexpr bool kSlide
        = Slide && Outputs >= 2 && (sizeof(cuda::BaseType<T>) >= 4 || (Outputs == 2 && cuda::NumElements<T> < 4));
    int kInd = 0;
    for (int i = 0; i < kernelSize.y; ++i)
    {
        if constexpr (kSlide)
        {
            const T *row = tile + (threadIdx.y + i) * tileW + threadIdx.x * Outputs;
            T        window[Outputs];
#pragma unroll
            for (int output = 0; output < Outputs - 1; ++output) window[output] = row[output];
            for (int j = 0; j < kernelSize.x; ++j)
            {
                window[Outputs - 1] = row[j + Outputs - 1];
                const float k       = kernelTile[kInd++];
#pragma unroll
                for (int output = 0; output < Outputs; ++output) res[output] = res[output] + window[output] * k;
#pragma unroll
                for (int output = 0; output < Outputs - 1; ++output) window[output] = window[output + 1];
            }
        }
        else
        {
            const int tileY = threadIdx.y + i;
            for (int j = 0; j < kernelSize.x; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];
#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
            }
        }
    }
}

// Stage the input tile and the kernel window in shared memory, then let every thread reuse them.
// `Outputs` columns per thread amortize the staging cost; the caller picks the width that fits both
// the shared-memory budget and the value type's register pressure.
template<int Outputs, bool Slide = false, class SrcWrapper, class Dst, class Coefs>
__global__ void Filter2DTiled(SrcWrapper src, Dst dst, Coefs coefs)
{
    using T         = typename Dst::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int outputW = blockDim.x * Outputs;
    const int x0      = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y       = blockIdx.y * blockDim.y + threadIdx.y;
    NVCV_FILTER_RESOLVE_SAMPLE();

    const int2 kernelSize = coefs.Size(batch_idx);
    const int2 anchor     = coefs.Anchor(batch_idx, kernelSize);

    const int tileW       = outputW + kernelSize.x - 1;
    const int tileH       = blockDim.y + kernelSize.y - 1;
    const int baseX       = blockIdx.x * outputW - anchor.x;
    const int baseY       = blockIdx.y * blockDim.y - anchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    T                                            *tile       = reinterpret_cast<T *>(tileRaw);
    float                                        *kernelTile = KernelTile(tileRaw, tileElems * sizeof(T));

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    for (int idx = tid; idx < tileElems; idx += blockThreads)
    {
        tile[idx] = src[SrcCoord<Dst::kPlanar>(baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx)];
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = coefs.Linear(batch_idx, idx, kernelSize);
    }

    __syncthreads();

    if (x0 >= dst.Width(batch_idx, plane) || y >= dst.Height(batch_idx, plane))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    AccumulateTiled<Outputs, Slide>(tile, kernelTile, tileW, kernelSize, res);

    *dst.Ptr(batch_idx, plane, y, x0) = cuda::SaturateCast<T>(res[0]);
#pragma unroll
    for (int output = 1; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.Width(batch_idx, plane))
        {
            *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

// Square kernels of a size known at compile time: the tap loops unroll fully and the tile geometry
// becomes constant. `KernelSizeIsExact` is true when the caller has already proved the runtime
// window equals (KernelW, KernelH) -- the tensor paths gate on it on the host -- so the fallback
// loop is compiled out entirely there. `LoadCoefsBeforeBounds` lets latency-sensitive coefficient
// sources issue metadata loads early enough to overlap them with block-coordinate work.
template<int Outputs, int KernelW, int KernelH, bool KernelSizeIsExact, bool LoadCoefsBeforeBounds = false,
         bool BlockLevelBoundsCheck = true, bool StageHalfAsFloat = false, bool PackedHalfLoads = false,
         class SrcWrapper, class Dst, class Coefs>
__global__ void Filter2DTiledFixed(SrcWrapper src, Dst dst, Coefs coefs)
{
    using T         = typename Dst::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    // Convert scalar halves once during staging instead of at every overlapping tap.
    using SharedT = std::conditional_t<StageHalfAsFloat, float, T>;
    static_assert(!StageHalfAsFloat || std::is_same_v<T, __half>);

    const int outputW = blockDim.x * Outputs;
    const int x0      = blockIdx.x * outputW + threadIdx.x * Outputs;
    const int y       = blockIdx.y * blockDim.y + threadIdx.y;
    NVCV_FILTER_RESOLVE_SAMPLE();

    const int outBaseX = blockIdx.x * outputW;
    const int outBaseY = blockIdx.y * blockDim.y;

    int2 kernelSize;
    int2 anchor;

    if constexpr (LoadCoefsBeforeBounds)
    {
        kernelSize = coefs.Size(batch_idx);
        anchor     = coefs.Anchor(batch_idx, kernelSize);
    }

    // Retire a block whose whole tile falls outside the image before it stages anything. Kept on by
    // default: it is what every caller had before this became selectable, and dropping it is not a
    // free win. On conv2d's var-shape 7x7 half3 benchmark the same grid barely overhangs the image,
    // so almost nothing is actually skipped -- yet removing the test costs 12.9% (434.8 -> 490.9 us
    // on a GB300). The test is doing more for codegen than it is for work avoidance.
    //
    // Gaussian's var-shape launches measure the other way, which is why this is a per-call-site
    // choice and not a property of the destination: there the two extra per-image extent loads sit
    // on the near side of the staging barrier, cannot be shared with the per-thread test on the far
    // side (__syncthreads() blocks the forwarding), and cost 1.24% on a GB300 and 4-6% on A100.
    if constexpr (BlockLevelBoundsCheck)
    {
        if (outBaseX >= dst.Width(batch_idx, plane) || outBaseY >= dst.Height(batch_idx, plane))
            return;
    }

    if constexpr (!LoadCoefsBeforeBounds)
    {
        kernelSize = coefs.Size(batch_idx);
        anchor     = coefs.Anchor(batch_idx, kernelSize);
    }

    const int tileW       = outputW + KernelW - 1;
    const int tileH       = blockDim.y + KernelH - 1;
    const int baseX       = outBaseX - anchor.x;
    const int baseY       = outBaseY - anchor.y;
    const int tileElems   = tileW * tileH;
    const int kernelElems = KernelSizeIsExact ? KernelW * KernelH : kernelSize.x * kernelSize.y;

    extern __shared__ __align__(16) unsigned char tileRaw[];
    SharedT                                      *tile       = reinterpret_cast<SharedT *>(tileRaw);
    float                                        *kernelTile = KernelTile(tileRaw, tileElems * sizeof(SharedT));

    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;

    if constexpr (PackedHalfLoads)
    {
        static_assert(std::is_same_v<T, __half> && (Outputs == 2 || Outputs == 4));
        const int width  = dst.Width(batch_idx, plane);
        const int height = dst.Height(batch_idx, plane);
        for (int loadY = threadIdx.y; loadY < tileH; loadY += blockDim.y)
        {
            const int srcY = baseY + loadY;
            const T  *row  = nullptr;
            if (srcY >= 0 && srcY < height)
            {
                if constexpr (KernelSizeIsExact)
                    row = src.tensorWrap().ptr(batch_idx, srcY, 0);
                else
                    row = src.imageBatchWrap().ptr(batch_idx, plane, srcY, 0);
            }
            // Dense pitches can move the first aligned group on every row.
            const int begin = row ? (-(reinterpret_cast<uintptr_t>(row) / sizeof(T) + baseX) & 3) : 0;
            const int end   = begin + cuda::min(outputW, (tileW - begin) / 4 * 4);
            for (int groupX = begin + threadIdx.x * 4; groupX < end; groupX += blockDim.x * 4)
            {
                const int srcX = baseX + groupX;
                const int idx  = loadY * tileW + groupX;
                if (row && srcX >= 0 && srcX <= width - 4)
                {
                    const uint2 value = *reinterpret_cast<const uint2 *>(row + srcX);
                    tile[idx]         = __ushort_as_half(value.x);
                    tile[idx + 1]     = __ushort_as_half(value.x >> 16);
                    tile[idx + 2]     = __ushort_as_half(value.y);
                    tile[idx + 3]     = __ushort_as_half(value.y >> 16);
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        tile[idx + i] = src[SrcCoord<Dst::kPlanar>(srcX + i, srcY, plane, batch_idx)];
                }
            }
            if (threadIdx.x < begin + tileW - end)
            {
                const int edgeX             = threadIdx.x < begin ? threadIdx.x : end + threadIdx.x - begin;
                tile[loadY * tileW + edgeX] = src[SrcCoord<Dst::kPlanar>(baseX + edgeX, srcY, plane, batch_idx)];
            }
        }
    }
    else
    {
        for (int idx = tid; idx < tileElems; idx += blockThreads)
        {
            const auto coord = SrcCoord<Dst::kPlanar>(baseX + idx % tileW, baseY + idx / tileW, plane, batch_idx);
            // Value access removes RGB8 border-value stack traffic. The schedule without block
            // bounds checks uses more registers without reducing runtime.
            if constexpr (BlockLevelBoundsCheck && SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT
                          && std::is_same_v<T, uchar3>)
                tile[idx] = src.value(coord);
            else
                tile[idx] = src[coord];
        }
    }
    for (int idx = tid; idx < kernelElems; idx += blockThreads)
    {
        kernelTile[idx] = coefs.Linear(batch_idx, idx, kernelSize);
    }

    __syncthreads();

    if (x0 >= dst.Width(batch_idx, plane) || y >= dst.Height(batch_idx, plane))
        return;

    work_type res[Outputs];
#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        res[output] = cuda::SetAll<work_type>(0);
    }

    int kInd = 0;
    if (KernelSizeIsExact || (kernelSize.x == KernelW && kernelSize.y == KernelH))
    {
#pragma unroll
        for (int i = 0; i < KernelH; ++i)
        {
            const int tileY = threadIdx.y + i;

#pragma unroll
            for (int j = 0; j < KernelW; ++j)
            {
                const int   tileX = threadIdx.x * Outputs + j;
                const float k     = kernelTile[kInd++];

#pragma unroll
                for (int output = 0; output < Outputs; ++output)
                {
                    res[output] = res[output] + tile[tileY * tileW + tileX + output] * k;
                }
            }
        }
    }
    else
    {
        // Keep fixed-window register allocation and scheduling independent of the runtime optimization.
        AccumulateTiled<Outputs, false>(tile, kernelTile, tileW, kernelSize, res);
    }

#pragma unroll
    for (int output = 0; output < Outputs; ++output)
    {
        const int x = x0 + output;
        if (x < dst.Width(batch_idx, plane))
        {
            *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res[output]);
        }
    }
}

// The untiled planar form is the one kernel the layout adaptor cannot absorb: it walks the planes
// inside one thread against a gridDim.z of samples only, so the tiled kernels' (sample, plane)
// decode does not apply.
template<class SrcWrapper, class Dst, class Coefs>
__global__ void Filter2DPlanar(SrcWrapper src, Dst dst, Coefs coefs, int channels)
{
    using T         = typename Dst::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (x >= dst.Width(batch_idx, 0) || y >= dst.Height(batch_idx, 0))
        return;

    const int2 kernelSize = coefs.Size(batch_idx);
    const int2 anchor     = coefs.Anchor(batch_idx, kernelSize);

    for (int plane = 0; plane < channels; ++plane)
    {
        work_type res = cuda::SetAll<work_type>(0);

        for (int i = 0; i < kernelSize.y; ++i)
        {
            const int cy = y - anchor.y + i;

            for (int j = 0; j < kernelSize.x; ++j)
            {
                res = res + src[int4{x - anchor.x + j, cy, plane, batch_idx}] * coefs.At(batch_idx, i, j);
            }
        }

        *dst.Ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(res);
    }
}

#undef NVCV_FILTER_RESOLVE_SAMPLE

// Shared-memory budgeting
// ---------------------------------------------------------------------------------------------

// Bytes a tiled launch needs: the input tile followed by the kernel window, the latter aligned up to
// float so the two views of the same allocation stay naturally aligned.
inline size_t TiledSharedBytes(int outputs, dim3 block, int2 kernelSize, size_t elemSize)
{
    const size_t kernelBytes = static_cast<size_t>(kernelSize.x) * kernelSize.y * sizeof(float);
    const size_t tileElems   = static_cast<size_t>(block.x * outputs + kernelSize.x - 1) * (block.y + kernelSize.y - 1);

    return util::RoundUpPowerOfTwo(tileElems * elemSize, sizeof(float)) + kernelBytes;
}

// Compile-time twin of TiledSharedBytes for the fixed-size launches, so a specialization that would
// not fit the budget is a build error rather than a launch failure at run time.
template<int Outputs, int KSize, typename T, int BlockW, int BlockH>
constexpr size_t FixedSharedBytes()
{
    constexpr size_t tileBytes = static_cast<size_t>(BlockW * Outputs + KSize - 1) * (BlockH + KSize - 1) * sizeof(T);
    return util::RoundUpPowerOfTwo(tileBytes, sizeof(float)) + static_cast<size_t>(KSize) * KSize * sizeof(float);
}

// ---------------------------------------------------------------------------------------------
// Validation and dispatch
// ---------------------------------------------------------------------------------------------

using ElemType = cvcuda::priv::legacy_format::DataTypeCode;

using cvcuda::priv::legacy_format::ClassifyDataType;

// The set Conv2D, AverageBlur and Gaussian accept. Laplacian declares a narrower one and classifies
// locally.
inline ElemType ClassifyFilterElemType(nvcv::DataType dtype)
{
    const ElemType elemType = ClassifyDataType(dtype);
    if (elemType == ElemType::kS8 || elemType == ElemType::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type");
    }
    return elemType;
}

inline ElemType ClassifyFilterElemType(nvcv::ImageFormat fmt)
{
    const ElemType elemType = ClassifyDataType(fmt);
    if (elemType == ElemType::kS8 || elemType == ElemType::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type");
    }
    return elemType;
}

using cvcuda::priv::ValidateBorderMode;

inline void ValidateChannelCount(int channels)
{
    // 2-channel: see Op{AverageBlur,Gaussian,Laplacian}Varshape_Negative.rejects_two_channel_interleaved_batch.
    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels (%d): 1, 3 or 4 are supported", channels);
    }
}

inline void ValidateTensorLayout(const nvcv::TensorLayout &layout)
{
    if (!(layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC || layout == nvcv::TENSOR_NCHW
          || layout == nvcv::TENSOR_CHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor layout not supported, the valid layouts are: NHWC, HWC, NCHW, CHW");
    }
}

// The historical layout contract for var-shape batches is a pair, not a bool: a single-image batch
// and a multi-image batch are different "formats" and were rejected against each other, which is the
// only way a mismatched image count was ever caught. Kept as a pair so the accepted set is unchanged.
struct BatchLayout
{
    bool planar;
    bool batched;
};

inline BatchLayout ClassifyBatchLayout(const nvcv::ImageBatchVarShapeDataStridedCuda &data)
{
    nvcv::ImageFormat fmt = cvcuda::priv::legacy_format::RequireUniformVarShapeFormat(data);

    return BatchLayout{fmt.numPlanes() >= 2, data.numImages() >= 2};
}

inline void RequireSameBatchLayout(BatchLayout in, BatchLayout out)
{
    if (in.planar != out.planar || in.batched != out.batched)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data format");
    }
}

// Resolve an element type to its base C++ type and hand it to `cb`. Replaces the leading index of
// the legacy [DataType][channels] function-pointer tables.
template<class Cb>
inline void RunBaseTypeSwitch(ElemType type, const Cb &cb)
{
    switch (type)
    {
    case ElemType::kU8:
        cb(uchar{});
        return;
    case ElemType::kU16:
        cb(ushort{});
        return;
    case ElemType::kS16:
        cb(short{});
        return;
    case ElemType::kS32:
        cb(int{});
        return;
    case ElemType::kF32:
        cb(float{});
        return;
    case ElemType::kF16:
        cb(__half{});
        return;
    default:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type");
}

// Resolve an element type plus a channel count to a value type and hand it to `cb`. Replaces the
// legacy [DataType][channels] tables; the instantiated set is the same, minus the null cells those
// tables carried for the widths and channel counts nothing dispatched to. A single channel resolves
// to the bare scalar (MakeType index 0), not to the one-component vector, matching what the tables
// named.
template<class Cb>
inline void RunTypeSwitch(ElemType type, int channels, const Cb &cb)
{
    RunBaseTypeSwitch(type,
                      [channels, &cb](auto base)
                      {
                          using Base = decltype(base);
                          if (channels == 1)
                              cb(cuda::MakeType<Base, 0>{});
                          else if (channels == 3)
                              cb(cuda::MakeType<Base, 3>{});
                          else if (channels == 4)
                              cb(cuda::MakeType<Base, 4>{});
                          else
                              throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                    "Invalid number of channels (%d): 1, 3 or 4 are supported",
                                                    channels);
                      });
}

// Bind a square kernel size to a template parameter when it matches one of the compile-time sizes
// the caller specialized for, and report whether any did, so the caller can fall through to its
// dynamic-size ladder. Replaces the chained `if (size == K) ... else` macro each launcher grew.
template<int... KSizes, class Launch>
inline bool TryFixedKernelSizes(int2 kernelSize, const Launch &launch)
{
    return (
        ...
        || (kernelSize.x == KSizes && kernelSize.y == KSizes && (launch(std::integral_constant<int, KSizes>{}), true)));
}

// The square window sizes the convolution-style filters give a compile-time specialization to.
//
// A specialized size unrolls its tap loops and folds its tile geometry to constants; anything else
// reads the window from memory and loops at run time. The gap is not small -- Gaussian's 7x7
// benchmarks ran 19.5% faster once 7 was on this list -- so the set has to stay ahead of the sizes
// callers actually use, and each launcher naming its own was how it drifted: Gaussian specialized
// {3, 5} while half its benchmark suite asked for 7.
//
// 3, 5 and 7 are the sizes the operators are benchmarked at and the ones users reach for. Adding
// one costs roughly 19% of a launcher's object size and 40% of its compile time, so this is a list
// to extend deliberately, not a range. Sizes outside it still work -- they take the dynamic kernel.
//
// Laplacian is deliberately not a caller: its ksize selects between two stencils inside a fixed 3x3
// window rather than sizing the window, and it binds that to a template parameter already.
template<class Launch>
inline bool TryCommonKernelSizes(int2 kernelSize, const Launch &launch)
{
    return TryFixedKernelSizes<3, 5, 7>(kernelSize, launch);
}

// Bind a border mode to a template parameter. The value is validated up front, so the default arm
// only guards against a caller that skipped ValidateBorderMode.
using cvcuda::priv::DispatchBorderMode;

// Every launcher needs a value type and a border mode bound together, so pair the two switches.
template<class Cb>
inline void RunTypeBorderSwitch(ElemType type, int channels, NVCVBorderType borderMode, const Cb &cb)
{
    RunTypeSwitch(type, channels,
                  [&](auto value) { DispatchBorderMode(borderMode, [&](auto border) { cb(value, border); }); });
}

template<class Cb>
inline void RunBaseTypeBorderSwitch(ElemType type, NVCVBorderType borderMode, const Cb &cb)
{
    RunBaseTypeSwitch(type, [&](auto base) { DispatchBorderMode(borderMode, [&](auto border) { cb(base, border); }); });
}

using cvcuda::priv::RequireInt32Addressable;
using cvcuda::priv::RequireSameImageShape;

inline void RequirePlanarGridZ(int64_t numImages)
{
    if (numImages > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar var-shape processing requires numImages <= %ld (CUDA grid-z limit)",
                              (long)kMaxGridZ);
    }
}

// Gaussian coefficient for one window position, normalised over the window. Shared by Gaussian
// (which materialises a whole kernel from it) and AdaptiveThreshold (which weights its mean
// window with it); both previously carried an identical copy.
// The normalizing sum is recomputed by every thread rather than reduced once, so each coefficient is
// the ratio of two values produced by the same instruction sequence.
template<typename CoordType>
inline __device__ float ComputeSingleGaussianValue(CoordType coord, int2 half, double2 sigma)
{
    float sx = 2.f * sigma.x * sigma.x;
    float sy = 2.f * sigma.y * sigma.y;
    float s  = 2.f * sigma.x * sigma.y * M_PI;

    float sum = 0.f;

    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            sum += cuda::exp(-((x * x) / sx + (y * y) / sy)) / s;
        }
    }

    int x = coord.x - half.x;
    int y = coord.y - half.y;

    return cuda::exp(-((x * x) / sx + (y * y) / sy)) / (s * sum);
}

} // namespace cvcuda::priv::filter

#endif // CVCUDA_PRIV_FILTER_COMMON_CUH
