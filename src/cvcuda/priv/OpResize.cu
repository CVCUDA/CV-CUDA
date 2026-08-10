/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CudaDeviceUtils.hpp"
#include "OpResize.hpp"
#include "PlanarTensorView.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/Printer.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

namespace {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

// Destination pack type given the source and destination type T
template<typename T>
using DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;

// Row alignment mask to determine if data pointer is aligned for vectorized write:
// note that in CUDA, uint3 has 4-byte (uint) alignment.
template<typename T>
constexpr uint MSK = (sizeof(DPT<T>) == sizeof(uint3) ? sizeof(uint) : sizeof(DPT<T>)) - 1;

// Number of items in x written by each thread
template<typename T>
constexpr int NIX = sizeof(DPT<T>) / sizeof(T);

// Write a pack of N elements of type T as a different pack type DPT
template<typename T>
__device__ __forceinline__ void WritePack(T &u, const T (&v)[NIX<T>])
{
    reinterpret_cast<DPT<T> &>(u) = reinterpret_cast<const DPT<T> &>(v);
}

// Check if destination row pointer is aligned for vector writes.
template<typename T>
__device__ __forceinline__ bool CheckRowAlign(T *row)
{
    return (static_cast<uint>(reinterpret_cast<size_t>(row)) & MSK<T>) == 0;
}

// Nearest ---------------------------------------------------------------------

template<bool INTERSECT, typename T, class SrcWrapper>
inline __device__ void NearestInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, float srcCoordX, int srcSizeX,
                                              int dstCoordX, int dstSizeX, float scaleRatioX)
{
    int iPrevCoordX;
    T   srcPack;

    if (dstCoordX + NIX<T> - 1 < dstSizeX)
    {
        T dstPack[NIX<T>];
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            iSrcCoord.x = floor(srcCoordX + x * scaleRatioX);
            iSrcCoord.x = cuda::min(iSrcCoord.x, srcSizeX - 1);

            if constexpr (INTERSECT)
            {
                if (x == 0 || iSrcCoord.x != iPrevCoordX)
                {
                    srcPack = src[iSrcCoord];
                }

                dstPack[x] = srcPack;

                iPrevCoordX = iSrcCoord.x;
            }
            else
            {
                dstPack[x] = src[iSrcCoord];
            }
        }

        if (CheckRowAlign(dstRow))                 // Branch is the same for all threads in warp.
            WritePack(dstRow[dstCoordX], dstPack); // If row is aligned, write vector pack;
        else
        {
            T *dstPtr = dstRow + dstCoordX; // otherwise, write individual elements.
#pragma unroll
            for (uint i = 0; i < NIX<T>; ++i) dstPtr[i] = dstPack[i];
        }
        // writePack(dstRow + dstCoordX, dstPack);
    }
    else
    {
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            if (dstCoordX + x < dstSizeX)
            {
                iSrcCoord.x = floor(srcCoordX + x * scaleRatioX);
                iSrcCoord.x = cuda::min(iSrcCoord.x, srcSizeX - 1);

                if constexpr (INTERSECT)
                {
                    if (x == 0 || iSrcCoord.x != iPrevCoordX)
                    {
                        srcPack = src[iSrcCoord];
                    }

                    dstRow[dstCoordX + x] = srcPack;

                    iPrevCoordX = iSrcCoord.x;
                }
                else
                {
                    dstRow[dstCoordX + x] = src[iSrcCoord];
                }
            }
        }
    }
}

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void NearestResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (dstCoord.y < dstSize.y)
    {
        dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

        float2 srcCoord = (cuda::DropCast<2>(dstCoord) + 0.5f) * scaleRatio;
        int3   iSrcCoord{0, (int)floor(srcCoord.y), dstCoord.z};

        iSrcCoord.y = cuda::min(iSrcCoord.y, srcSize.y - 1);

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        NearestInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcCoord.x, srcSize.x, dstCoord.x, dstSize.x,
                                          scaleRatio.x);
    }
}

// Linear ----------------------------------------------------------------------

// The cached INTERSECT pack holds the taps as work-type floats: float(byte) conversion is exact,
// so converting once at read time (instead of inside every blend that reuses the pack) is
// value-identical while removing the per-output conversions the 4x-class upscales were
// instruction-bound on. For float element types the conversion is a no-op. Scalar byte elements
// keep the raw byte pack: the float cache regressed the NIX=16 U8 upscale rows +24% on A100
// (+5% H100) against reference baselines while multi-channel bytes gained.
template<typename T>
using LinearFT = std::conditional_t<cuda::NumElements<T> == 1 && sizeof(cuda::BaseType<T>) == 1, T,
                                    cuda::ConvertBaseTypeTo<float, T>>;

template<typename T>
__device__ __forceinline__ LinearFT<T> LinearToPack(T v)
{
    return cuda::StaticCast<cuda::BaseType<LinearFT<T>>>(v);
}

template<class SrcWrapper, typename FT>
inline __device__ void LinearReadPack(SrcWrapper src, FT (&srcPack)[4], int3 iSrcCoord)
{
    srcPack[0] = LinearToPack(src[int3{iSrcCoord.x, iSrcCoord.y, iSrcCoord.z}]);
    srcPack[1] = LinearToPack(src[int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}]);
    srcPack[2] = LinearToPack(src[int3{iSrcCoord.x, iSrcCoord.y + 1, iSrcCoord.z}]);
    srcPack[3] = LinearToPack(src[int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}]);
}

template<typename T>
__device__ __forceinline__ void LinearReadPack(const T *srcRow0, const T *srcRow1, LinearFT<T> (&srcPack)[4],
                                               int srcCoordX)
{
    srcPack[0] = LinearToPack(srcRow0[srcCoordX]);
    srcPack[1] = LinearToPack(srcRow0[srcCoordX + 1]);
    srcPack[2] = LinearToPack(srcRow1[srcCoordX]);
    srcPack[3] = LinearToPack(srcRow1[srcCoordX + 1]);
}

template<bool USE_ROW_PTR, class SrcWrapper, typename T>
__device__ __forceinline__ void LinearReadPackMaybe(SrcWrapper src, const T *srcRow0, const T *srcRow1,
                                                    LinearFT<T> (&srcPack)[4], int3 iSrcCoord)
{
    if constexpr (USE_ROW_PTR)
    {
        LinearReadPack(srcRow0, srcRow1, srcPack, iSrcCoord.x);
    }
    else
    {
        LinearReadPack(src, srcPack, iSrcCoord);
    }
}

template<bool USE_ROW_PTR, class SrcWrapper, typename T>
__device__ __forceinline__ T LinearReadMaybe(SrcWrapper src, const T *srcRow, int3 iSrcCoord)
{
    if constexpr (USE_ROW_PTR)
    {
        return srcRow[iSrcCoord.x];
    }
    else
    {
        return src[iSrcCoord];
    }
}

template<typename T>
__device__ __forceinline__ T LinearBlend(T p00, T p10, T p01, T p11, float2 w)
{
    return cuda::SaturateCast<T>(p00 * ((1.f - w.x) * (1.f - w.y)) + p10 * (w.x * (1.f - w.y))
                                 + p01 * ((1.f - w.x) * w.y) + p11 * (w.x * w.y));
}

// Blend from the float-cached INTERSECT pack: the expression tree matches LinearBlend exactly and
// float(byte) conversion is exact, so the result is bit-identical to blending the raw taps.
template<typename T>
__device__ __forceinline__ T LinearBlendPack(const LinearFT<T> (&p)[4], float2 w)
{
    return cuda::SaturateCast<T>(p[0] * ((1.f - w.x) * (1.f - w.y)) + p[1] * (w.x * (1.f - w.y))
                                 + p[2] * ((1.f - w.x) * w.y) + p[3] * (w.x * w.y));
}

template<bool USE_ROW_PTR, class SrcWrapper, typename T>
__device__ __forceinline__ void LinearReadNextPackMaybe(SrcWrapper src, const T *srcRow0, const T *srcRow1,
                                                        LinearFT<T> (&srcPack)[4], int3 iSrcCoord)
{
    srcPack[0] = srcPack[1];
    srcPack[2] = srcPack[3];
    srcPack[1]
        = LinearToPack(LinearReadMaybe<USE_ROW_PTR>(src, srcRow0, int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}));
    srcPack[3]
        = LinearToPack(LinearReadMaybe<USE_ROW_PTR>(src, srcRow1, int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}));
}

template<bool USE_ROW_PTR, class SrcWrapper, typename T>
__device__ __forceinline__ void LinearUpdatePackMaybe(SrcWrapper src, const T *srcRow0, const T *srcRow1,
                                                      LinearFT<T> (&srcPack)[4], int3 iSrcCoord, int iPrevCoordX, int x)
{
    if (x == 0)
    {
        LinearReadPackMaybe<USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack, iSrcCoord);
    }
    else if (iSrcCoord.x != iPrevCoordX)
    {
        if (iSrcCoord.x == (iPrevCoordX + 1))
        {
            LinearReadNextPackMaybe<USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack, iSrcCoord);
        }
        else
        {
            LinearReadPackMaybe<USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack, iSrcCoord);
        }
    }
}

template<bool INTERSECT, bool USE_ROW_PTR, class SrcWrapper, typename T>
__device__ __forceinline__ T LinearSampleMaybe(SrcWrapper src, const T *srcRow0, const T *srcRow1,
                                               LinearFT<T> (&srcPack)[4], int &iPrevCoordX, int3 iSrcCoord, int x,
                                               float2 w)
{
    if constexpr (INTERSECT)
    {
        LinearUpdatePackMaybe<USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack, iSrcCoord, iPrevCoordX, x);
        iPrevCoordX = iSrcCoord.x;
        return LinearBlendPack<T>(srcPack, w);
    }
    else
    {
        return LinearBlend(
            LinearReadMaybe<USE_ROW_PTR>(src, srcRow0, int3{iSrcCoord.x, iSrcCoord.y, iSrcCoord.z}),
            LinearReadMaybe<USE_ROW_PTR>(src, srcRow0, int3{iSrcCoord.x + 1, iSrcCoord.y, iSrcCoord.z}),
            LinearReadMaybe<USE_ROW_PTR>(src, srcRow1, int3{iSrcCoord.x, iSrcCoord.y + 1, iSrcCoord.z}),
            LinearReadMaybe<USE_ROW_PTR>(src, srcRow1, int3{iSrcCoord.x + 1, iSrcCoord.y + 1, iSrcCoord.z}), w);
    }
}

template<bool INTERSECT, typename T, class SrcWrapper>
inline __device__ void LinearInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, float srcCoordX, int srcSizeX,
                                             int dstCoordX, int dstSizeX, float scaleRatioX, float2 w)
{
    float       sx;
    int         iPrevCoordX;
    LinearFT<T> srcPack[4];

    // Row pointers reduce index math for byte-vector LINEAR; float vectors keep wrapper loads
    // to preserve bit-exact parity with the flattened scalar planar path.
    constexpr int  NUM_ELEMENTS = cuda::NumElements<T>;
    constexpr bool USE_ROW_PTR  = NUM_ELEMENTS > 1 && sizeof(cuda::BaseType<T>) == 1;

    const T *srcRow0 = nullptr;
    const T *srcRow1 = nullptr;
    if constexpr (USE_ROW_PTR)
    {
        srcRow0 = src.ptr(iSrcCoord.z, iSrcCoord.y);
        srcRow1 = src.ptr(iSrcCoord.z, iSrcCoord.y + 1);
    }

    if (dstCoordX + NIX<T> - 1 < dstSizeX)
    {
        T dstPack[NIX<T>];
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            sx          = srcCoordX + x * scaleRatioX;
            iSrcCoord.x = floor(sx);

            w.x = ((iSrcCoord.x < 0) ? 0 : ((iSrcCoord.x > srcSizeX - 2) ? 1 : sx - iSrcCoord.x));

            iSrcCoord.x = cuda::max(0, cuda::min(iSrcCoord.x, srcSizeX - 2));

            dstPack[x] = LinearSampleMaybe<INTERSECT, USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack, iPrevCoordX,
                                                                   iSrcCoord, x, w);
        }

        if (CheckRowAlign(dstRow))                 // Branch is the same for all threads in warp.
            WritePack(dstRow[dstCoordX], dstPack); // If row is aligned, write vector pack;
        else
        {
            T *dstPtr = dstRow + dstCoordX; // otherwise, write individual elements.
#pragma unroll
            for (uint i = 0; i < NIX<T>; ++i) dstPtr[i] = dstPack[i];
        }
        // writePack<true>(dstRow + dstCoordX, dstPack, reinterpret_cast<uint>(dstRow) & DstMask) == 0);
    }
    else
    {
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            if (dstCoordX + x < dstSizeX)
            {
                sx          = srcCoordX + x * scaleRatioX;
                iSrcCoord.x = floor(sx);

                w.x = ((iSrcCoord.x < 0) ? 0 : ((iSrcCoord.x > srcSizeX - 2) ? 1 : sx - iSrcCoord.x));

                iSrcCoord.x = cuda::max(0, cuda::min(iSrcCoord.x, srcSizeX - 2));

                dstRow[dstCoordX + x] = LinearSampleMaybe<INTERSECT, USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack,
                                                                                  iPrevCoordX, iSrcCoord, x, w);
            }
        }
    }
}

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void LinearResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (dstCoord.y < dstSize.y)
    {
        dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

        float2 srcCoord = (cuda::DropCast<2>(dstCoord) + .5f) * scaleRatio - .5f;
        int3   iSrcCoord{0, (int)floor(srcCoord.y), dstCoord.z};

        float2 w;

        w.y = ((iSrcCoord.y < 0) ? 0 : ((iSrcCoord.y > srcSize.y - 2) ? 1 : srcCoord.y - iSrcCoord.y));

        iSrcCoord.y = cuda::max(0, cuda::min(iSrcCoord.y, srcSize.y - 2));

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        LinearInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcCoord.x, srcSize.x, dstCoord.x, dstSize.x,
                                         scaleRatio.x, w);
    }
}

// Exact-2x bilinear upscale for 1-byte base types, mirroring CubicResizeExpand2x: at dst == 2*src
// the x/y weights are exactly 0.25f/0.75f (the same fx/fy the general kernel derives), so each
// thread owns NIX output columns x 2 output rows, stages the shared clamped 3-row x (NIX/2+2)-col
// source window in registers, and writes two vector packs. Border outputs replicate the general
// kernel's weight override (w = 0 when iSrc < 0, w = 1 when iSrc > size-2): the overridden weight
// zeroes the tap whose unclamped-window value differs from the clamped-coordinate load, so the
// blend -- evaluated with LinearBlend's exact expression -- stays bit-exact.
template<class SrcWrapper, class DstWrapper>
__global__ void LinearResizeExpand2x(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize)
{
    using T = typename DstWrapper::ValueType;

    constexpr int KC = NIX<T> / 2; // source columns owned per thread
    constexpr int WC = KC + 2;     // staged window columns (taps k0-1 .. k0+KC)

    const int k0    = (blockIdx.x * blockDim.x + threadIdx.x) * KC; // first owned source column
    const int j     = blockIdx.y * blockDim.y + threadIdx.y;        // owned source row
    const int z     = blockIdx.z;
    const int dstX0 = k0 * 2;
    const int dstY0 = j * 2;

    if (dstX0 >= dstSize.x || dstY0 >= dstSize.y)
        return;

    // Clamped source window shared by all 2*NIX outputs: win[r][c] = src[clamp(k0-1+c), clamp(j-1+r)].
    T win[3][WC];
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        const T *srcRow = src.ptr(z, cuda::clamp(j - 1 + r, 0, srcSize.y - 1));
#pragma unroll
        for (int c = 0; c < WC; ++c)
        {
            win[r][c] = srcRow[cuda::clamp(k0 - 1 + c, 0, srcSize.x - 1)];
        }
    }

    // Even output row 2j: iSrcY = j-1, fy = 0.75 (taps window rows 0,1); odd row 2j+1: iSrcY = j,
    // fy = 0.25 (taps rows 1,2) -- with the general kernel's border overrides.
    const float wyE = (j - 1 < 0) ? 0.f : ((j - 1 > srcSize.y - 2) ? 1.f : 0.75f);
    const float wyO = (j > srcSize.y - 2) ? 1.f : 0.25f;

    T out0[NIX<T>], out1[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        const bool  odd   = (i & 1) != 0;
        const int   k     = k0 + i / 2;
        const int   iSrcX = odd ? k : k - 1;
        const float wx    = (iSrcX < 0) ? 0.f : ((iSrcX > srcSize.x - 2) ? 1.f : (odd ? 0.25f : 0.75f));
        const int   cb    = (i / 2) + (odd ? 1 : 0); // window column of the first x tap

        out0[i] = LinearBlend(win[0][cb], win[0][cb + 1], win[1][cb], win[1][cb + 1], float2{wx, wyE});
        out1[i] = LinearBlend(win[1][cb], win[1][cb + 1], win[2][cb], win[2][cb + 1], float2{wx, wyO});
    }

    const int validCount = cuda::min(NIX<T>, dstSize.x - dstX0);
    // dstSize.y = 2*srcSize.y is even, so the odd output row always exists.
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        T *dstRow              = dst.ptr(z, dstY0 + p);
        const T(&outp)[NIX<T>] = p == 0 ? out0 : out1;

        if (validCount == NIX<T> && CheckRowAlign(dstRow)) // uniform across the warp
            WritePack(dstRow[dstX0], outp);
        else
        {
            T *dstPtr = dstRow + dstX0;
#pragma unroll
            for (int c = 0; c < NIX<T>; ++c)
                if (c < validCount)
                    dstPtr[c] = outp[c];
        }
    }
}

// Cubic -----------------------------------------------------------------------

inline __device__ void GetCubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3)
{
    constexpr float A = -0.75f;

    w0 = ((A * (delta + 1) - 5 * A) * (delta + 1) + 8 * A) * (delta + 1) - 4 * A;
    w1 = ((A + 2) * delta - (A + 3)) * delta * delta + 1;
    w2 = ((A + 2) * (1 - delta) - (A + 3)) * (1 - delta) * (1 - delta) + 1;
    w3 = 1.f - w0 - w1 - w2;
}

// Each thread owns COLS output columns of a single destination row. The cubic y-axis pipeline
// (source coordinate, fractional offset, the 4 cubic coefficients, and the 4 clamped tap rows)
// depends only on the row, so it is computed once and reused across all COLS columns -- removing
// the redundant per-pixel y-coefficient and y-clamp ALU work that dominated the issue-bound
// single-pixel-per-thread kernel. The columns are grid-strided by blockDim.x so each store step
// stays coalesced. The per-pixel arithmetic is byte-identical to the scalar kernel (the hoisted
// wy/sy values are the same), so the result is bit-exact.
// Shared-memory-tiled bicubic for wide-float (float3/float4) EXPAND (upscale).
//
// Reference-SKU ncu shows the per-output 4x4 gather is L1/TEX-throughput bound (~77-87% on A100/
// H100) with DRAM far below (~28-61%): the 12-/16-byte unaligned float3/float4 taps thrash L1 while
// the data is being re-read, not streamed. For an upscale, a thread block's output tile maps to a
// SMALL source tile (output_extent * scale + cubic support), so we stage that tile once in shared
// memory -- with replicate-border clamp baked in at load time -- and serve all 16 taps/pixel from
// smem instead of L1. tileW/tileH are sized on the host from the scale ratio (generous upper bound).
//
// Bit-exact with CubicResize: each smem slot holds src[clamp(tileOrigin + slot)], so indexing a tap
// by its UNCLAMPED position minus the tile origin yields tile[tap - origin] == src[clamp(tap)] --
// identical values, weights and accumulation order. EXPAND-only (CONTRACT would need a huge tile)
// and gated to float3/float4 (the at-ridge wide-element path); every other CUBIC case keeps
// CubicResize<COLS> / CubicResizePlanar.
template<class SrcWrapper, class DstWrapper>
__global__ void CubicResizeSharedExpand(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio,
                                        int tileW, int tileH)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    extern __shared__ __align__(16) unsigned char smemRaw[];
    T                                            *tile = reinterpret_cast<T *>(smemRaw);

    const int z   = blockIdx.z;
    const int ox0 = blockIdx.x * blockDim.x; // block output-tile origin
    const int oy0 = blockIdx.y * blockDim.y;

    // Source tile origin = first cubic tap (iSrc-1) of the block's first output pixel.
    const int tx0 = (int)floorf(((float)ox0 + .5f) * scaleRatio.x - .5f) - 1;
    const int ty0 = (int)floorf(((float)oy0 + .5f) * scaleRatio.y - .5f) - 1;

    // Cooperative, border-clamped load: tile[j*tileW + i] = src[clamp(tx0+i), clamp(ty0+j), z].
    const int nThreads = blockDim.x * blockDim.y;
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileW * tileH; idx += nThreads)
    {
        const int i  = idx % tileW;
        const int j  = idx / tileW;
        const int sx = cuda::clamp(tx0 + i, 0, srcSize.x - 1);
        const int sy = cuda::clamp(ty0 + j, 0, srcSize.y - 1);
        tile[idx]    = src[int3{sx, sy, z}];
    }
    __syncthreads();

    const int dstX = ox0 + threadIdx.x;
    const int dstY = oy0 + threadIdx.y;
    if (dstX >= dstSize.x || dstY >= dstSize.y)
        return;

    const float srcX  = ((float)dstX + .5f) * scaleRatio.x - .5f;
    const float srcY  = ((float)dstY + .5f) * scaleRatio.y - .5f;
    const int   iSrcX = (int)floorf(srcX);
    const int   iSrcY = (int)floorf(srcY);

    float wx[4], wy[4];
    GetCubicCoeffs(srcX - iSrcX, wx[0], wx[1], wx[2], wx[3]);
    GetCubicCoeffs(srcY - iSrcY, wy[0], wy[1], wy[2], wy[3]);

    FT sum = FT{};
#pragma unroll
    for (int cy = 0; cy < 4; cy++)
    {
        // tile rows/cols hold the clamped source, so index by the unclamped tap minus the tile
        // origin -- tile[(tap)-origin] == src[clamp(tap)], matching CubicResize exactly.
        const int jj = (iSrcY + cy - 1) - ty0;
#pragma unroll
        for (int cx = 0; cx < 4; cx++)
        {
            const int ii = (iSrcX + cx - 1) - tx0;
            sum += tile[jj * tileW + ii] * (wx[cx] * wy[cy]);
        }
    }
    dst[int3{dstX, dstY, z}] = cuda::SaturateCast<T>(sum);
}

// General cubic upscale for byte and scalar elements with NIX consecutive output columns per
// thread: for any expand scale the source column index advances by at most one between adjacent
// outputs, so the 4x4 tap window is kept as work-type floats (float(byte) conversion is exact) and
// shifted by a single converted column load when it advances -- the grid-strided general kernel
// re-gathers and re-converts all 16 taps per output. The y coefficients/rows stay hoisted as in
// CubicResize; x coefficients remain per-output (fractional phases). Tap selection, weight values,
// and the cy-outer/cx-inner accumulation order match CubicResize, so the result is bit-exact. The
// exact-2x cases peel off to their phase-fused kernels before this dispatch.
template<class SrcWrapper, class DstWrapper>
__global__ void CubicResizeUpscale(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int z    = blockIdx.z;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0   = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (dstY >= dstSize.y || x0 >= dstSize.x)
        return;

    // Row-invariant y terms, identical to CubicResize.
    const float srcY  = ((float)dstY + .5f) * scaleRatio.y - .5f;
    const int   iSrcY = (int)floor(srcY);
    const float fy    = srcY - iSrcY;

    float wy[4];
    GetCubicCoeffs(fy, wy[0], wy[1], wy[2], wy[3]);

    const T *rows[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) rows[k] = src.ptr(z, cuda::clamp(iSrcY + k - 1, 0, srcSize.y - 1));

    // Float-cached 4x4 tap window of the current source column set [prevSx-1, prevSx+2].
    FT  win[4][4];
    int prevSx = -srcSize.x - 4; // never matches or adjoins a valid first column

    T out[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        const int   dstX  = x0 + i;
        const float srcX  = ((float)dstX + .5f) * scaleRatio.x - .5f;
        const int   iSrcX = (int)floor(srcX);
        const float fx    = srcX - iSrcX;

        float wx[4];
        GetCubicCoeffs(fx, wx[0], wx[1], wx[2], wx[3]);

        if (iSrcX == prevSx + 1)
        {
#pragma unroll
            for (int cy = 0; cy < 4; ++cy)
            {
                win[cy][0] = win[cy][1];
                win[cy][1] = win[cy][2];
                win[cy][2] = win[cy][3];
                win[cy][3] = cuda::StaticCast<float>(rows[cy][cuda::clamp(iSrcX + 2, 0, srcSize.x - 1)]);
            }
        }
        else if (iSrcX != prevSx)
        {
#pragma unroll
            for (int cy = 0; cy < 4; ++cy)
            {
#pragma unroll
                for (int cx = 0; cx < 4; ++cx)
                {
                    win[cy][cx] = cuda::StaticCast<float>(rows[cy][cuda::clamp(iSrcX + cx - 1, 0, srcSize.x - 1)]);
                }
            }
        }
        prevSx = iSrcX;

        FT sum = FT{};
#pragma unroll
        for (int cy = 0; cy < 4; ++cy)
        {
#pragma unroll
            for (int cx = 0; cx < 4; ++cx)
            {
                sum += win[cy][cx] * (wx[cx] * wy[cy]);
            }
        }
        out[i] = cuda::SaturateCast<T>(sum);
    }

    T        *dstRow     = dst.ptr(z, dstY);
    const int validCount = cuda::min(NIX<T>, dstSize.x - x0);

    if (validCount == NIX<T> && CheckRowAlign(dstRow)) // uniform across the warp
        WritePack(dstRow[x0], out);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int c = 0; c < NIX<T>; ++c)
            if (c < validCount)
                dstPtr[c] = out[c];
    }
}

// Exact-2x cubic upscale for wide float elements (float3/float4), which the register-staged
// CubicResizeExpand2x excludes: a full NIX-column window would need 75-100 registers of staged
// data. Instead each thread owns one output column x 2 output rows -- the vertical pair shares its
// 4-column x-window, so the staged window is 5 rows x 4 columns (20 wide elements) and each output
// costs 10 staged loads instead of 16 gathered ones, with the per-pixel coefficient math replaced
// by the two per-axis phase tables. Tap selection, weight values, and accumulation order match
// CubicResize (and the smem kernel), so the result is bit-exact.
template<class SrcWrapper, class DstWrapper>
__global__ void CubicResizeExpand2xWide(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int j    = blockIdx.y * blockDim.y + threadIdx.y; // owned source row
    const int z    = blockIdx.z;
    const int dstY = j * 2;

    if (dstX >= dstSize.x || dstY >= dstSize.y)
        return;

    const bool odd = (dstX & 1) != 0;
    const int  k   = dstX / 2;
    // First x tap: even outputs of source col k tap k-2, odd outputs tap k-1.
    const int  cb = k - 2 + (odd ? 1 : 0);

    // Clamped 5-row x 4-column source window shared by the vertical output pair.
    T win[5][4];
#pragma unroll
    for (int r = 0; r < 5; ++r)
    {
        const T *srcRow = src.ptr(z, cuda::clamp(j - 2 + r, 0, srcSize.y - 1));
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            win[r][c] = srcRow[cuda::clamp(cb + c, 0, srcSize.x - 1)];
        }
    }

    float wx[4], wyE[4], wyO[4];
    GetCubicCoeffs(odd ? 0.25f : 0.75f, wx[0], wx[1], wx[2], wx[3]);
    GetCubicCoeffs(0.75f, wyE[0], wyE[1], wyE[2], wyE[3]);
    GetCubicCoeffs(0.25f, wyO[0], wyO[1], wyO[2], wyO[3]);

    FT sum0 = FT{}; // even output row (taps j-2..j+1 -> window rows 0..3)
    FT sum1 = FT{}; // odd output row (taps j-1..j+2 -> window rows 1..4)
#pragma unroll
    for (int cy = 0; cy < 4; ++cy)
    {
#pragma unroll
        for (int cx = 0; cx < 4; ++cx)
        {
            sum0 += win[cy][cx] * (wx[cx] * wyE[cy]);
            sum1 += win[cy + 1][cx] * (wx[cx] * wyO[cy]);
        }
    }

    dst[int3{dstX, dstY, z}]     = cuda::SaturateCast<T>(sum0);
    dst[int3{dstX, dstY + 1, z}] = cuda::SaturateCast<T>(sum1);
}

inline bool UseSharedMemoryCubicExpand()
{
    int dev = 0;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    int major = 0, minor = 0;
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

    const int sm = major * 10 + minor;

    // The smem-tiled float CUBIC EXPAND path was validated on the reference CI SKUs
    // (A100/H100: SM80/SM90). GA102/AD104 (SM86/SM89) regress and keep the direct gather.
    return sm == 80 || sm == 90;
}

// Exact-2x cubic upscale for 1-byte base types. The benched EXPAND configs (and any dst == 2*src
// resize) have only two coefficient phases per axis: srcX = (dstX+.5f)*.5f - .5f lands on k-0.25
// for even dstX (fx = 0.75, taps k-2..k+1) and k+0.25 for odd dstX (fx = 0.25, taps k-1..k+2), and
// likewise per row. The general kernel is issue-bound on that per-pixel coefficient/clamp/index
// work (SM 83%, IPC 3.3, DRAM 11%), so this kernel hoists it wholesale: each thread owns NIX
// output columns x 2 output rows, computes the four 4-tap coefficient sets once (GetCubicCoeffs at
// exactly 0.25f/0.75f -- the same fx the general kernel derives, so the weights are identical
// floats), stages the shared 5-row x (NIX/2+4)-col clamped source window in registers, and emits
// two WritePack vector stores. Tap selection, weight values, and the cy-outer/cx-inner
// sum += tap * (wx*wy) accumulation order match CubicResize exactly, so the output is bit-exact.
// All loops fully unroll: window/output indices stay compile-time (runtime register indexing
// would spill to local memory).
template<class SrcWrapper, class DstWrapper>
__global__ void CubicResizeExpand2x(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    constexpr int KC = NIX<T> / 2; // source columns owned per thread
    constexpr int WC = KC + 4;     // staged window columns (taps k0-2 .. k0+KC+1)

    const int k0    = (blockIdx.x * blockDim.x + threadIdx.x) * KC; // first owned source column
    const int j     = blockIdx.y * blockDim.y + threadIdx.y;        // owned source row
    const int z     = blockIdx.z;
    const int dstX0 = k0 * 2;
    const int dstY0 = j * 2;

    if (dstX0 >= dstSize.x || dstY0 >= dstSize.y)
        return;

    // Clamped source window shared by all 2*NIX outputs: win[r][c] = src[clamp(k0-2+c), clamp(j-2+r)].
    T win[5][WC];
#pragma unroll
    for (int r = 0; r < 5; ++r)
    {
        const T *srcRow = src.ptr(z, cuda::clamp(j - 2 + r, 0, srcSize.y - 1));
#pragma unroll
        for (int c = 0; c < WC; ++c)
        {
            win[r][c] = srcRow[cuda::clamp(k0 - 2 + c, 0, srcSize.x - 1)];
        }
    }

    // The two per-axis coefficient phases; identical values to the general kernel's per-pixel
    // GetCubicCoeffs(fx) because fx is exactly 0.75f/0.25f for dst == 2*src.
    float wxE[4], wxO[4], wyE[4], wyO[4];
    GetCubicCoeffs(0.75f, wxE[0], wxE[1], wxE[2], wxE[3]);
    GetCubicCoeffs(0.25f, wxO[0], wxO[1], wxO[2], wxO[3]);
    GetCubicCoeffs(0.75f, wyE[0], wyE[1], wyE[2], wyE[3]);
    GetCubicCoeffs(0.25f, wyO[0], wyO[1], wyO[2], wyO[3]);

    T out0[NIX<T>], out1[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        const bool odd      = (i & 1) != 0;
        const float(&wx)[4] = odd ? wxO : wxE;
        // First tap column within the window: even outputs of source col k tap k-2 (window index
        // k-k0), odd outputs tap k-1 (window index k-k0+1).
        const int cb = (i / 2) + (odd ? 1 : 0);

        FT sum0 = FT{}; // even output row (taps j-2..j+1 -> window rows 0..3)
        FT sum1 = FT{}; // odd output row (taps j-1..j+2 -> window rows 1..4)
#pragma unroll
        for (int cy = 0; cy < 4; ++cy)
        {
#pragma unroll
            for (int cx = 0; cx < 4; ++cx)
            {
                sum0 += win[cy][cb + cx] * (wx[cx] * wyE[cy]);
                sum1 += win[cy + 1][cb + cx] * (wx[cx] * wyO[cy]);
            }
        }
        out0[i] = cuda::SaturateCast<T>(sum0);
        out1[i] = cuda::SaturateCast<T>(sum1);
    }

    const int validCount = cuda::min(NIX<T>, dstSize.x - dstX0);
    // dstSize.y = 2*srcSize.y is even, so the odd output row always exists.
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        T *dstRow              = dst.ptr(z, dstY0 + p);
        const T(&outp)[NIX<T>] = p == 0 ? out0 : out1;

        if (validCount == NIX<T> && CheckRowAlign(dstRow)) // uniform across the warp
            WritePack(dstRow[dstX0], outp);
        else
        {
            T *dstPtr = dstRow + dstX0;
#pragma unroll
            for (int c = 0; c < NIX<T>; ++c)
                if (c < validCount)
                    dstPtr[c] = outp[c];
        }
    }
}

template<int COLS, class SrcWrapper, class DstWrapper>
__global__ void CubicResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int z    = blockIdx.z;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstY >= dstSize.y)
        return;

    // y-axis terms are row-invariant: compute once, reuse across this thread's COLS columns.
    const float srcY  = ((float)dstY + .5f) * scaleRatio.y - .5f;
    const int   iSrcY = (int)floor(srcY);
    const float fy    = srcY - iSrcY;

    float wy[4];
    GetCubicCoeffs(fy, wy[0], wy[1], wy[2], wy[3]);

    // Clamp each y tap coordinate to [0, srcSize-1] (replicate border) -- shared by all columns.
    int sy[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) sy[k] = cuda::clamp(iSrcY + k - 1, 0, srcSize.y - 1);

    const int xBase = blockIdx.x * blockDim.x * COLS + threadIdx.x;

#pragma unroll
    for (int i = 0; i < COLS; ++i)
    {
        const int dstX = xBase + i * (int)blockDim.x;
        if (dstX >= dstSize.x)
            continue;

        const float srcX  = ((float)dstX + .5f) * scaleRatio.x - .5f;
        const int   iSrcX = (int)floor(srcX);
        const float fx    = srcX - iSrcX;

        float wx[4];
        GetCubicCoeffs(fx, wx[0], wx[1], wx[2], wx[3]);

        FT sum = FT{};

        // Clamp each source tap coordinate independently to [0, srcSize-1] (replicate border),
        // matching the behavior of OpenCV INTER_CUBIC and PyTorch bicubic interpolation.
#pragma unroll
        for (int cy = 0; cy < 4; cy++)
        {
#pragma unroll
            for (int cx = 0; cx < 4; cx++)
            {
                int sx = cuda::clamp(iSrcX + cx - 1, 0, srcSize.x - 1);
                sum += src[int3{sx, sy[cy], z}] * (wx[cx] * wy[cy]);
            }
        }

        // SaturateCast clamps to the valid output range (e.g. [0,255] for uint8).
        // No abs() -- cubic interpolation can produce slightly negative intermediate values;
        // the correct behavior is to clamp to 0, which SaturateCast already handles.
        dst[int3{dstX, dstY, z}] = cuda::SaturateCast<T>(sum);
    }
}

// Planar cubic resize: one thread owns an output pixel across ALL channel planes of a sample.
// The cubic coefficients and clamped tap coordinates depend only on (x, y), not on the channel,
// so they are computed once and reused for every plane -- avoiding the (channels - 1)x redundant
// coefficient math that results from launching the single-channel CubicResize over N*C flattened
// planes. The flattened single-channel view maps plane (n, c) to sample index n*channels + c.
template<class SrcWrapper, class DstWrapper>
__global__ void CubicResizePlanar(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio,
                                  int channels)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int n    = blockIdx.z;

    if (dstY < dstSize.y && dstX < dstSize.x)
    {
        float2 srcCoord{((float)dstX + .5f) * scaleRatio.x - .5f, ((float)dstY + .5f) * scaleRatio.y - .5f};
        int2   iSrcCoord{(int)floor(srcCoord.x), (int)floor(srcCoord.y)};

        float fx = srcCoord.x - iSrcCoord.x;
        float fy = srcCoord.y - iSrcCoord.y;

        float wx[4];
        float wy[4];
        GetCubicCoeffs(fx, wx[0], wx[1], wx[2], wx[3]);
        GetCubicCoeffs(fy, wy[0], wy[1], wy[2], wy[3]);

        // Clamp each source tap coordinate once (replicate border); reused across all planes.
        // Precompute the 16 separable tap weights once too -- they are channel-independent, so the
        // channel loop becomes a pure load + FMA over shared weights.
        int   sx[4];
        int   sy[4];
        float w[4][4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            sx[k] = cuda::clamp(iSrcCoord.x + k - 1, 0, srcSize.x - 1);
            sy[k] = cuda::clamp(iSrcCoord.y + k - 1, 0, srcSize.y - 1);
        }
#pragma unroll
        for (int cy = 0; cy < 4; ++cy)
#pragma unroll
            for (int cx = 0; cx < 4; ++cx) w[cy][cx] = wx[cx] * wy[cy];

        const int base = n * channels;
        for (int c = 0; c < channels; ++c)
        {
            const int plane = base + c;

            FT sum = FT{};
#pragma unroll
            for (int cy = 0; cy < 4; cy++)
            {
#pragma unroll
                for (int cx = 0; cx < 4; cx++)
                {
                    sum += src[int3{sx[cx], sy[cy], plane}] * w[cy][cx];
                }
            }
            dst[int3{dstX, dstY, plane}] = cuda::SaturateCast<T>(sum);
        }
    }
}

// Area ------------------------------------------------------------------------

// Each thread produces NIX<T> consecutive output pixels of a row and writes them as one vector
// store. The scalar one-pixel-per-thread kernel left the narrow-element AREA path (uint8 NIX=16,
// uchar3/uchar4 NIX=4) bound by tiny, poorly-coalesced stores and per-thread launch overhead;
// batching NIX pixels + a single WritePack fixes that. The per-pixel value
// `src[StaticCast<float>(coord)]` (the AREA box filter inside the interpolation wrap) is identical
// to the scalar kernel, so the result is bit-exact. Wide elements with NIX==1 (float3/float4)
// degrade to one pixel per thread -- same work as before.
template<class SrcWrapper, class DstWrapper>
__global__ void AreaResizeVec(SrcWrapper src, DstWrapper dst, int2 dstSize)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = blockIdx.y * blockDim.y + threadIdx.y;
    dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (dstCoord.y >= dstSize.y || dstCoord.x >= dstSize.x)
        return;

    T        *dstRow     = dst.ptr(dstCoord.z, dstCoord.y);
    const int validCount = cuda::min(NIX<T>, dstSize.x - dstCoord.x);

    T dstPack[NIX<T>];
#pragma unroll
    for (int p = 0; p < NIX<T>; ++p)
    {
        if (p >= validCount)
            break;
        dstPack[p] = src[cuda::StaticCast<float>(int3{dstCoord.x + p, dstCoord.y, dstCoord.z})];
    }

    if (validCount == NIX<T> && CheckRowAlign(dstRow)) // uniform across the warp
        WritePack(dstRow[dstCoord.x], dstPack);
    else
    {
        T *dstPtr = dstRow + dstCoord.x;
#pragma unroll
        for (int i = 0; i < NIX<T>; ++i)
            if (i < validCount)
                dstPtr[i] = dstPack[i];
    }
}

// Integer-ratio AREA downscale (the is_area_fast case: scale is an exact integer >= 1, e.g. the 2x
// CONTRACT configs). Each output pixel is the mean of an iscale.x * iscale.y source box. A dedicated
// kernel for this path -- dispatched at launch when the scale is integer -- replaces the
// InterpolationWrap's per-access AREA-mode runtime branch and index math with direct source-row-
// pointer accumulation (the wrap path was issue-bound on that ALU: ~56% compute / ALU-top at IPC
// 2.2). NIX columns/thread + WritePack keep stores coalesced (mirrors AreaResizeVec). The four source
// row pointers for the box are resolved per column from src.ptr(z, y); the integer box is always
// in-bounds for an exact ratio, but taps are min-clamped defensively.
//
// PRECISION TRADE (not bit-exact): the accumulation sums the box then normalizes once (sum * 1/area),
// which can differ from the wrap's per-tap weighting by <= 1 ULP-scale rounding -- the result stays
// within the +/-1 tolerance of the OpenCV AREA reference but is not byte-identical to the wrap-based
// kernel. Both the interleaved and planar(flattened) AREA paths use this same kernel for the integer
// case, so the planar==interleaved parity (EXPECT_EQ) is preserved.
template<class SrcWrapper, class DstWrapper>
__global__ void AreaResizeFastDirect(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, int2 iscale)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int z     = blockIdx.z;
    const int dstY  = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstX0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (dstY >= dstSize.y || dstX0 >= dstSize.x)
        return;

    const int   ymin       = dstY * iscale.y;
    const float invArea    = 1.f / (float)(iscale.x * iscale.y);
    T          *dstRow     = dst.ptr(z, dstY);
    const int   validCount = cuda::min(NIX<T>, dstSize.x - dstX0);

    T dstPack[NIX<T>];
#pragma unroll
    for (int p = 0; p < NIX<T>; ++p)
    {
        if (p >= validCount)
            break;

        const int xmin = (dstX0 + p) * iscale.x;

        FT acc = FT{};
        for (int yy = 0; yy < iscale.y; ++yy)
        {
            const T *srcRow = src.ptr(z, cuda::min(ymin + yy, srcSize.y - 1));
            for (int xx = 0; xx < iscale.x; ++xx)
                acc += cuda::StaticCast<float>(srcRow[cuda::min(xmin + xx, srcSize.x - 1)]);
        }
        dstPack[p] = cuda::SaturateCast<T>(acc * invArea);
    }

    if (validCount == NIX<T> && CheckRowAlign(dstRow))
        WritePack(dstRow[dstX0], dstPack);
    else
    {
        T *dstPtr = dstRow + dstX0;
#pragma unroll
        for (int i = 0; i < NIX<T>; ++i)
            if (i < validCount)
                dstPtr[i] = dstPack[i];
    }
}

// Exact-2x specialization of AreaResizeFastDirect: every output is the mean of a fully-in-bounds
// 2x2 box, so the runtime-bounded box loops and defensive tap clamps collapse to four unrolled
// taps, and the 2*NIX consecutive source elements each thread needs per row are staged as two
// vector packs instead of single-element loads (the general integer kernel is L1-transaction-bound
// on the byte rows: 33% A100 / 24% H100 BWUtil). Accumulation is the same raw sum then one
// * invArea (0.25f) normalize as AreaResizeFastDirect, so the result is bit-exact with it; the
// flattened planar view takes this path too.
template<class SrcWrapper, class DstWrapper>
__global__ void AreaResizeContract2x(SrcWrapper src, DstWrapper dst, int2 dstSize)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    const int z  = blockIdx.z;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (y >= dstSize.y || x0 >= dstSize.x)
        return;

    const T *row0 = src.ptr(z, 2 * y);
    const T *row1 = src.ptr(z, 2 * y + 1);

    const int sx0        = 2 * x0;
    const int validCount = cuda::min(NIX<T>, dstSize.x - x0);

    T win0[2 * NIX<T>], win1[2 * NIX<T>];
    if (validCount == NIX<T> && CheckRowAlign(const_cast<T *>(row0) + sx0)
        && CheckRowAlign(const_cast<T *>(row1) + sx0))
    {
#pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            reinterpret_cast<DPT<T> *>(win0)[p] = reinterpret_cast<const DPT<T> *>(row0 + sx0)[p];
            reinterpret_cast<DPT<T> *>(win1)[p] = reinterpret_cast<const DPT<T> *>(row1 + sx0)[p];
        }
    }
    else
    {
        // Source width is exactly 2*dstSize.x, so tail loads stay guarded per element.
#pragma unroll
        for (int c = 0; c < 2 * NIX<T>; ++c)
        {
            const int sx = sx0 + c;
            win0[c]      = (sx < 2 * dstSize.x) ? row0[sx] : T{};
            win1[c]      = (sx < 2 * dstSize.x) ? row1[sx] : T{};
        }
    }

    T dstPack[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        FT acc = FT{};
        acc += cuda::StaticCast<float>(win0[2 * i]);
        acc += cuda::StaticCast<float>(win0[2 * i + 1]);
        acc += cuda::StaticCast<float>(win1[2 * i]);
        acc += cuda::StaticCast<float>(win1[2 * i + 1]);
        dstPack[i] = cuda::SaturateCast<T>(acc * 0.25f);
    }

    T *dstRow = dst.ptr(z, y);
    if (validCount == NIX<T> && CheckRowAlign(dstRow))
        WritePack(dstRow[x0], dstPack);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int i = 0; i < NIX<T>; ++i)
            if (i < validCount)
                dstPtr[i] = dstPack[i];
    }
}

inline bool UseContract2xSingleOutputPack()
{
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));

    // A single-output pack has no cross-output load reuse, and its staged load burst stalls on
    // Turing. Ampere and newer architectures retain the profitable specialization.
    return sm >= 80;
}

// Fractional AREA downscale for the tensor path with NIX consecutive output columns per thread.
// Every box tap of a zoom-out is in bounds, so the InterpolationWrap's per-tap border handling and
// per-access bounds recomputation are replaced by hoisted row pointers and once-per-thread y box
// terms; the NIX results leave as one vector pack (the wrap-based kernel is issue/L1-bound on the
// anisotropic rows: 12.8-30% BWUtil). Term order replicates the wrap's AddFractionalArea exactly --
// interior rows (columns ascending, then left, then right edge), top edge row (columns, then
// top-left, then top-right corner), bottom edge row (columns, then bottom-right, then bottom-left
// corner), each tap * scale -- so the result is bit-exact with operator[] and with
// AreaResizePlanar's bounds-sharing path. Integer ratios and zoom-ins keep their existing kernels.
template<class SrcWrapper, class DstWrapper>
__global__ void AreaResizeFractional(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    using T  = typename DstWrapper::ValueType;
    using FT = nvcv::cuda::ConvertBaseTypeTo<float, T>;

    // Independent accumulation chains per thread: the serial one-output-at-a-time walk left the
    // kernel memory-latency bound (IPC 0.78, 39 warp cycles per instruction, DRAM 64%). Grouping
    // GN outputs and bounding the interior box width at compile time (dispatch guarantees
    // scale < 3, so at most kMaxC interior columns) turns each source row into an unrolled grid of
    // independent guarded loads and FMAs. Per-output term order is unchanged -- interior columns
    // ascending, then left, then right edge per row; top row with its corners; bottom row with its
    // corners -- so the result stays bit-exact with the wrap's AddFractionalArea.
    // Three-channel elements lose more to L1 pressure than they gain from extra chains.
    constexpr int GN    = (cuda::NumElements<T> == 3) ? 2 : 4;
    constexpr int kMaxC = 3;

    const int z  = blockIdx.z;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (y >= dstSize.y || x0 >= dstSize.x)
        return;

    const float fsy1 = y * scaleRatio.y;
    const float fsy2 = fsy1 + scaleRatio.y;
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Guard the edge taps against the float-rounding boundary: fsx2/fsy2 = dst*scale can exceed
    // the source extent by an ulp, putting sx2/sy2 one past the last element. The wrap-based
    // kernels read the constant border (zero) there, contributing exactly +0.0, so skipping the
    // tap is bit-identical -- and it is what keeps this kernel's reads in bounds.
    const bool  hasTop = (float)sy1 > fsy1;
    const bool  hasBot = (float)sy2 < fsy2 && sy2 < srcSize.y;
    const float wTop   = (float)sy1 - fsy1;
    const float wBot   = fsy2 - (float)sy2;

    T out[NIX<T>];

#pragma unroll
    for (int g = 0; g < NIX<T>; g += GN)
    {
        float fsx1[GN], fsx2[GN], scaleA[GN], wL[GN], wR[GN];
        int   sx1[GN], sx2[GN], nC[GN];
        bool  hasL[GN], hasR[GN];
#pragma unroll
        for (int j = 0; j < GN; ++j)
        {
            const int x = cuda::min(x0 + g + j, dstSize.x - 1);
            fsx1[j]     = x * scaleRatio.x;
            fsx2[j]     = fsx1[j] + scaleRatio.x;
            sx1[j]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[j]);
            sx2[j]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[j]);
            nC[j]       = sx2[j] - sx1[j];
            hasL[j]     = (float)sx1[j] > fsx1[j];
            hasR[j]     = (float)sx2[j] < fsx2[j] && sx2[j] < srcSize.x;
            wL[j]       = (float)sx1[j] - fsx1[j];
            wR[j]       = fsx2[j] - (float)sx2[j];
            scaleA[j]   = 1.f / (fminf(scaleRatio.x, srcSize.x - fsx1[j]) * fminf(scaleRatio.y, srcSize.y - fsy1));
        }

        FT acc[GN];
#pragma unroll
        for (int j = 0; j < GN; ++j) acc[j] = FT{};

        for (int cy = sy1; cy < sy2; ++cy)
        {
            const T *row = src.ptr(z, cy);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] += row[sx1[j] + c] * scaleA[j];
                }
            }
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
                if (hasL[j])
                    acc[j] += row[sx1[j] - 1] * (wL[j] * scaleA[j]);
                if (hasR[j])
                    acc[j] += row[sx2[j]] * (wR[j] * scaleA[j]);
            }
        }

        if (hasTop)
        {
            const T *row = src.ptr(z, sy1 - 1);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] += row[sx1[j] + c] * (wTop * scaleA[j]);
                }
                if (hasL[j])
                    acc[j] += row[sx1[j] - 1] * (wTop * wL[j] * scaleA[j]);
                if (hasR[j])
                    acc[j] += row[sx2[j]] * (wTop * wR[j] * scaleA[j]);
            }
        }

        if (hasBot)
        {
            const T *row = src.ptr(z, sy2);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] += row[sx1[j] + c] * (wBot * scaleA[j]);
                }
                if (hasR[j])
                    acc[j] += row[sx2[j]] * (wBot * wR[j] * scaleA[j]);
                if (hasL[j])
                    acc[j] += row[sx1[j] - 1] * (wBot * wL[j] * scaleA[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < GN; ++j) out[g + j] = cuda::SaturateCast<T>(acc[j]);
    }

    T        *dstRow     = dst.ptr(z, y);
    const int validCount = cuda::min(NIX<T>, dstSize.x - x0);

    if (validCount == NIX<T> && CheckRowAlign(dstRow)) // uniform across the warp
        WritePack(dstRow[x0], out);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int c = 0; c < NIX<T>; ++c)
            if (c < validCount)
                dstPtr[c] = out[c];
    }
}

// Planar area resize: one thread owns an output pixel across ALL channel planes of a sample. The area
// box geometry depends only on (x, y), not the channel, so it is computed once (via the interpolation
// wrap's reusable bounds) and reused for every plane -- avoiding the per-plane recomputation incurred
// by running the single-channel AreaResize over N*C flattened planes. Plane (n, c) is sample n*C + c.
template<class SrcWrapper, class DstWrapper>
__global__ void AreaResizePlanar(SrcWrapper src, DstWrapper dst, int2 dstSize, int channels)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= dstSize.x || y >= dstSize.y)
        return;

    const int base = n * channels;

    // Bounds depend only on (x, y); compute once for this sample and reuse across its planes.
    const auto bounds = src.computeAreaBounds(float3{(float)x, (float)y, (float)base});

    for (int c = 0; c < channels; ++c)
    {
        const int plane        = base + c;
        dst[int3{x, y, plane}] = src.interpolateWithBounds(float3{(float)x, (float)y, (float)plane}, bounds);
    }
}

// Host run resize functions ---------------------------------------------------

template<typename T>
void RunResizeInterp(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                     const nvcv::TensorDataStridedCuda &dstData, int2 srcSize, int2 dstSize, int batchSize,
                     const NVCVInterpolationType interpolation, int planarChannels = 1)
{
    float2 scaleRatio{(float)srcSize.x / dstSize.x, (float)srcSize.y / dstSize.y};

    auto srcTW = cuda::CreateTensorWrapNHW<const T, int32_t>(srcData);
    auto dstTW = cuda::CreateTensorWrapNHW<T, int32_t>(dstData);
    auto srcIW = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA, int32_t>(
        srcData, T{}, scaleRatio.x, scaleRatio.y);

    dim3 threads1(32, 4, 1);
    dim3 blocks1(util::DivUp(dstSize.x, threads1.x * NIX<T>), util::DivUp(dstSize.y, threads1.y), batchSize);

    dim3 threads2(128, 1, 1);
    dim3 blocks2(util::DivUp(dstSize.x, threads2.x), util::DivUp(dstSize.y, threads2.y), batchSize);

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        if (scaleRatio.x < 1)
            NearestResize<true><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        else
            NearestResize<false><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;

    case NVCV_INTERP_LINEAR:
        if constexpr (sizeof(cuda::BaseType<T>) == 1)
        {
            // Exact-2x upscale fast path for byte types; the flattened planar view takes it too.
            if (dstSize.x == 2 * srcSize.x && dstSize.y == 2 * srcSize.y)
            {
                dim3 e2Block(32, 4, 1);
                dim3 e2Grid(util::DivUp(dstSize.x, e2Block.x * NIX<T>), util::DivUp(dstSize.y, e2Block.y * 2),
                            batchSize);
                LinearResizeExpand2x<<<e2Grid, e2Block, 0, stream>>>(srcTW, dstTW, srcSize, dstSize);
                break;
            }
        }
        if (scaleRatio.x < 2)
            LinearResize<true><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        else
            LinearResize<false><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;

    case NVCV_INTERP_CUBIC:
    {
        // Byte types (any channel count) and scalar types (any base size) fit the register-staged
        // window; wide float3/float4 interleaved take the vertical-pair variant below.
        if constexpr (sizeof(cuda::BaseType<T>) == 1 || cuda::NumElements<T> == 1)
        {
            // Exact-2x upscale fast path (issue-bound in the general kernel). The flattened planar
            // view takes it too -- including float planes: per-plane phases and weights are
            // identical, so it supersedes CubicResizePlanar's per-pixel channel amortization for
            // this case while remaining bit-exact with it.
            if (dstSize.x == 2 * srcSize.x && dstSize.y == 2 * srcSize.y)
            {
                dim3 e2Block(32, 4, 1);
                dim3 e2Grid(util::DivUp(dstSize.x, e2Block.x * NIX<T>), util::DivUp(dstSize.y, e2Block.y * 2),
                            batchSize);
                CubicResizeExpand2x<<<e2Grid, e2Block, 0, stream>>>(srcTW, dstTW, srcSize, dstSize);
                break;
            }
            // Any other upscale: consecutive columns with the float-cached sliding window.
            if (dstSize.x > srcSize.x && dstSize.y > srcSize.y)
            {
                dim3 upBlock(32, 4, 1);
                dim3 upGrid(util::DivUp(dstSize.x, upBlock.x * NIX<T>), util::DivUp(dstSize.y, upBlock.y), batchSize);
                CubicResizeUpscale<<<upGrid, upBlock, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
                break;
            }
        }
        if (planarChannels > 1)
        {
            // grid.z spans samples (not N*C planes); each thread loops the channel planes.
            // A 2D (64x2) block gives the 4x4 cubic window better L1 reuse across neighbouring rows
            // than the default 1D (128x1) block, which matters most for the load-bound float planes.
            dim3 pcBlock(64, 2, 1);
            dim3 blocksP(util::DivUp(dstSize.x, pcBlock.x), util::DivUp(dstSize.y, pcBlock.y),
                         batchSize / planarChannels);
            CubicResizePlanar<<<blocksP, pcBlock, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio,
                                                               planarChannels);
            break;
        }
        if constexpr (sizeof(cuda::BaseType<T>) == 4 && cuda::NumElements < T >> 1)
        {
            // Exact-2x upscale for wide float elements: the vertical-pair register window replaces
            // the per-pixel coefficient math and the smem tile round-trip on this case.
            if (dstSize.x == 2 * srcSize.x && dstSize.y == 2 * srcSize.y)
            {
                dim3 wBlock(32, 4, 1);
                dim3 wGrid(util::DivUp(dstSize.x, wBlock.x), util::DivUp(dstSize.y, wBlock.y * 2), batchSize);
                CubicResizeExpand2xWide<<<wGrid, wBlock, 0, stream>>>(srcTW, dstTW, srcSize, dstSize);
                break;
            }
        }
        if constexpr (sizeof(cuda::BaseType<T>) == 4)
        {
            if (scaleRatio.x < 1.f && scaleRatio.y < 1.f && UseSharedMemoryCubicExpand())
            {
                // Float (float32/float3/float4) upscale: the 4x4 gather is L1/TEX-bound on reference SKUs
                // (DRAM has slack). Stage the small per-block source tile in shared memory and serve the
                // 16 taps/pixel from smem. tile dims are an upper bound on (block_extent*scale + cubic
                // support); generous slack keeps every block's taps in range. Bit-exact with CubicResize.
                dim3         scBlock(32, 8, 1);
                dim3         scGrid(util::DivUp(dstSize.x, scBlock.x), util::DivUp(dstSize.y, scBlock.y), batchSize);
                const int    tileW     = (int)ceilf(scBlock.x * scaleRatio.x) + 6;
                const int    tileH     = (int)ceilf(scBlock.y * scaleRatio.y) + 6;
                const size_t smemBytes = (size_t)tileW * tileH * sizeof(T);
                CubicResizeSharedExpand<<<scGrid, scBlock, smemBytes, stream>>>(srcTW, dstTW, srcSize, dstSize,
                                                                                scaleRatio, tileW, tileH);
                break;
            }
        }

        // Each thread processes CUBIC_COLS output columns so the row-invariant y-axis cubic
        // coefficients and clamps are computed once and reused (the kernel was issue-bound on
        // redundant per-pixel ALU work); columns are grid-strided by blockDim.x for coalescing.
        // Gated to 1-byte-base-type elements (uint8/uchar3/uchar4): a CI run on A100 showed the
        // multi-column path improves uint8 ~11% but regresses float32/float3 CUBIC by +17-44%
        // (memory/latency-bound there, hurt by the added register pressure), so float base types
        // keep the original one-column path. Scalar uint8 profiles as instruction-bound even at
        // COLS=4, so it gets a little more y-axis reuse while byte-vector types keep the proven
        // lower-register grouping. CUBIC_COLS=1 is byte-identical to the scalar kernel.
        constexpr int CUBIC_COLS = (sizeof(cuda::BaseType<T>) == 1) ? (cuda::NumElements<T> == 1 ? 8 : 4) : 1;
        dim3 blocksC(util::DivUp(dstSize.x, threads2.x * CUBIC_COLS), util::DivUp(dstSize.y, threads2.y), batchSize);
        CubicResize<CUBIC_COLS><<<blocksC, threads2, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        break;
    }

    case NVCV_INTERP_AREA:
    {
        // Exact integer-ratio downscale (e.g. the 2x CONTRACT configs) takes a dedicated direct-pointer
        // kernel that drops the InterpolationWrap's per-access AREA-mode runtime branch + index math
        // (precision trade, within +/-1 of the AREA reference). Used for BOTH interleaved and planar
        // (flattened) so their parity is preserved. All other AREA cases (non-integer downscale, any
        // upscale/zoom-in) keep the wrap-based kernels below.
        const int  isx      = (int)(scaleRatio.x + 0.5f);
        const int  isy      = (int)(scaleRatio.y + 0.5f);
        const bool areaFast = scaleRatio.x >= 1.f && scaleRatio.y >= 1.f && fabsf(scaleRatio.x - isx) < 1e-5f
                           && fabsf(scaleRatio.y - isy) < 1e-5f;
        if (areaFast)
        {
            const bool exact2x       = isx == 2 && isy == 2 && dstSize.x * 2 == srcSize.x && dstSize.y * 2 == srcSize.y;
            bool       useContract2x = exact2x;

            // Multi-output packs remain profitable on every supported GPU. Only unamortized
            // single-output packs need an architecture-specific decision; flattened planar float
            // has NIX=4 and never enters this branch.
            if constexpr (NIX<T> == 1)
                useContract2x = useContract2x && UseContract2xSingleOutputPack();

            if (useContract2x)
                AreaResizeContract2x<<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, dstSize);
            else
                AreaResizeFastDirect<<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, int2{isx, isy});
        }
        // Vectorized wrap path: each thread emits NIX<T> output pixels with one vector store. For
        // interleaved this wins on narrow elements and is a no-op (NIX=1) on float3/float4. For planar
        // (NCHW) on the flattened single-channel view, a 1-byte element (uint8, NIX=16) vectorizes
        // hugely, but a wide float plane (NIX=4) loses the per-output bounds-sharing that
        // AreaResizePlanar gets across channels and regresses CONTRACT, so wide-float planar keeps the
        // channel-amortized kernel.
        else
        {
            // Fractional zoom-out: every tap is in bounds, so the dedicated kernel replaces the
            // wrap-based paths (see AreaResizeFractional); zoom-ins keep the wrap kernels, and
            // wide-float planar planes keep AreaResizePlanar's cross-channel bounds amortization
            // (routing them here regressed float3/float4 planes 14-28% locally).
            // The kernel's compile-time interior-column bound (kMaxC) holds for x scales < 3;
            // larger fractional downscales keep the wrap path.
            if (scaleRatio.x >= 1.f && scaleRatio.y >= 1.f && scaleRatio.x < 3.f
                && (planarChannels == 1 || sizeof(cuda::BaseType<T>) == 1))
            {
                AreaResizeFractional<<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
                break;
            }
            if constexpr (NIX<T> < 8)
            {
                if (planarChannels > 1)
                {
                    dim3 blocksP(util::DivUp(dstSize.x, threads2.x), util::DivUp(dstSize.y, threads2.y),
                                 batchSize / planarChannels);
                    AreaResizePlanar<<<blocksP, threads2, 0, stream>>>(srcIW, dstTW, dstSize, planarChannels);
                    break;
                }
            }
            AreaResizeVec<<<blocks1, threads1, 0, stream>>>(srcIW, dstTW, dstSize);
        }
        break;
    }

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation");
    }
}

inline void RunResizeInterpType(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                const nvcv::TensorDataStridedCuda &dstData, int2 srcSize, int2 dstSize, int numChannels,
                                int batchSize, const NVCVInterpolationType interpolation, int planarChannels = 1)
{
    // The data type may contain the channels baked in or the number of channels is in the tensor shape

    // clang-format off

#define CVCUDA_RUN_RESIZE(BT, DT, T)                                             \
    ((srcData.dtype() == nvcv::TYPE_##BT && numChannels == cuda::NumElements<T>) \
     || (srcData.dtype() == nvcv::TYPE_##DT && numChannels == 1))                \
        RunResizeInterp<T>(stream, srcData, dstData, srcSize, dstSize, batchSize, interpolation, planarChannels);

    if CVCUDA_RUN_RESIZE(U8, U8, uchar1)
    else if CVCUDA_RUN_RESIZE(U8, 3U8, uchar3)
    else if CVCUDA_RUN_RESIZE(U8, 4U8, uchar4)
    else if CVCUDA_RUN_RESIZE(U16, U16, ushort)
    else if CVCUDA_RUN_RESIZE(U16, 3U16, ushort3)
    else if CVCUDA_RUN_RESIZE(U16, 4U16, ushort4)
    else if CVCUDA_RUN_RESIZE(S16, S16, short)
    else if CVCUDA_RUN_RESIZE(S16, 3S16, short3)
    else if CVCUDA_RUN_RESIZE(S16, 4S16, short4)
    else if CVCUDA_RUN_RESIZE(F32, F32, float)
    else if CVCUDA_RUN_RESIZE(F32, 3F32, float3)
    else if CVCUDA_RUN_RESIZE(F32, 4F32, float4)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }

#undef CVCUDA_RUN_RESIZE

    // clang-format on
}

} // anonymous namespace

namespace cvcuda::priv {

// Tensor operator -------------------------------------------------------------

void Resize::RunResize(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                       const nvcv::TensorDataStridedCuda &dstData, const NVCVInterpolationType interpolation) const
{
    if (srcData.dtype() != dstData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data type are different");
    }
    if (srcData.layout() != dstData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data layout are different");
    }
    const nvcv::TensorLayout layout   = srcData.layout();
    const bool               isPlanar = (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW);
    if (layout != nvcv::TENSOR_HWC && layout != nvcv::TENSOR_NHWC && !isPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output samples are different");
    }
    if (srcAccess->numChannels() != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output channels are different");
    }
    if (srcAccess->numChannels() > 4 || srcAccess->numChannels() < 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of channels");
    }

    constexpr int32_t kIntMax = cuda::TypeTraits<int32_t>::max;

    int64_t srcMaxStride = srcAccess->sampleStride() * srcAccess->numSamples();
    int64_t dstMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) > kIntMax || srcAccess->numSamples() > kIntMax
        || srcAccess->numCols() > kIntMax || srcAccess->numRows() > kIntMax || dstAccess->numCols() > kIntMax
        || dstAccess->numRows() > kIntMax)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input or output tensors are too large");
    }

    int  numChannels{(int)srcAccess->numChannels()};
    int  batchSize{(int)srcAccess->numSamples()};
    int2 srcSize{(int)srcAccess->numCols(), (int)srcAccess->numRows()};
    int2 dstSize{(int)dstAccess->numCols(), (int)dstAccess->numRows()};

    if (interpolation == NVCV_INTERP_LINEAR && (srcSize.x < 2 || srcSize.y < 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Linear interpolation requires source dimensions of at least 2x2");
    }

    if (isPlanar)
    {
        // View each of the N*C channel planes as a single-channel sample and reuse the
        // interleaved single-channel resize path. Channels are independent in resize, so this
        // produces identical results to resizing the equivalent NHWC single-channel data.
        // The flattened plane count becomes the kernel's grid z-dimension, so it must fit both a
        // 32-bit batch size and CUDA's 65535 grid-z limit; compute it in 64-bit to avoid overflow.
        constexpr int64_t kMaxGridZ   = 65535;
        const int64_t     planarBatch = static_cast<int64_t>(numChannels) * batchSize;
        if (planarBatch > kMaxGridZ)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar resize requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
        }

        auto srcView = PlanarAsSingleChannelView(srcData, *srcAccess);
        auto dstView = PlanarAsSingleChannelView(dstData, *dstAccess);
        RunResizeInterpType(stream, srcView, dstView, srcSize, dstSize, /*numChannels=*/1,
                            /*batchSize=*/static_cast<int>(planarBatch), interpolation,
                            /*planarChannels=*/static_cast<int>(numChannels));
        return;
    }

    RunResizeInterpType(stream, srcData, dstData, srcSize, dstSize, numChannels, batchSize, interpolation);
}

} // namespace cvcuda::priv
