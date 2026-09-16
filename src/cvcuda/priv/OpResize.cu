/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpResize.hpp"
#include "PlanarTensorView.hpp"

#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/Printer.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>

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
inline __device__ void NearestInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, int srcSizeX, int dstCoordX,
                                              int dstSizeX, float scaleRatioX)
{
    int iPrevCoordX;
    T   srcPack;

    // Hoist the int->float conversion: each column then costs one add against a literal.
    const float dstCoordXf = (float)dstCoordX + 0.5f;

    if (dstCoordX + NIX<T> - 1 < dstSizeX)
    {
        T dstPack[NIX<T>];
#pragma unroll
        for (int x = 0; x < NIX<T>; ++x)
        {
            iSrcCoord.x = floor((dstCoordXf + x) * scaleRatioX);
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
                iSrcCoord.x = floor((dstCoordXf + x) * scaleRatioX);
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

        float srcCoordY = ((float)dstCoord.y + 0.5f) * scaleRatio.y;
        int3  iSrcCoord{0, (int)floor(srcCoordY), dstCoord.z};

        iSrcCoord.y = cuda::min(iSrcCoord.y, srcSize.y - 1);

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        NearestInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcSize.x, dstCoord.x, dstSize.x, scaleRatio.x);
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
inline __device__ void LinearInterpolatePack(T *dstRow, SrcWrapper src, int3 iSrcCoord, int srcSizeX, int dstCoordX,
                                             int dstSizeX, float scaleRatioX, float2 w)
{
    float       sx;
    int         iPrevCoordX;
    LinearFT<T> srcPack[4];

    // Hoist the int->float conversion: each column then costs one add against a literal.
    const float dstCoordXf = (float)dstCoordX + 0.5f;

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
            sx          = (dstCoordXf + x) * scaleRatioX - 0.5f;
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
                sx          = (dstCoordXf + x) * scaleRatioX - 0.5f;
                iSrcCoord.x = floor(sx);

                w.x = ((iSrcCoord.x < 0) ? 0 : ((iSrcCoord.x > srcSizeX - 2) ? 1 : sx - iSrcCoord.x));

                iSrcCoord.x = cuda::max(0, cuda::min(iSrcCoord.x, srcSizeX - 2));

                dstRow[dstCoordX + x] = LinearSampleMaybe<INTERSECT, USE_ROW_PTR>(src, srcRow0, srcRow1, srcPack,
                                                                                  iPrevCoordX, iSrcCoord, x, w);
            }
        }
    }
}

// Threads per block for the tensor interpolation launches (threads1 in RunResizeInterp).
constexpr int kInterpThreadsPerBlock = 128;

// Requesting this many resident blocks caps the scalar-float LINEAR kernel at 32 registers on
// every supported toolkit. CUDA 12.2, which builds the shipped CUDA 12 packages, otherwise
// allocates 34 and loses a quarter of the SM's warps. The value is architecture-dependent
// because ptxas rejects a hint whose blocks * threads exceeds the target's threads-per-SM limit.
#if !defined(__CUDA_ARCH__)
constexpr int kLinearFloatBlocksPerSM = 16; // host pass; unused
#elif __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
constexpr int kLinearFloatBlocksPerSM = 16; // sm_80/sm_90: 2048 threads/SM
#elif __CUDA_ARCH__ < 860
constexpr int kLinearFloatBlocksPerSM = 8; // sm_75 and older: 1024 threads/SM
#else
constexpr int kLinearFloatBlocksPerSM = 12; // conservative: holds at 1536 threads/SM
#endif

// Only the scalar-float form. The byte and 16-bit forms need 38-48 registers and would spill.
template<typename T>
constexpr bool kPinLinearOccupancy = cuda::NumElements<T> == 1 && sizeof(cuda::BaseType<T>) == 4;

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__device__ __forceinline__ void LinearResizeBody(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize,
                                                 float2 scaleRatio)
{
    using T = typename DstWrapper::ValueType;

    int3 dstCoord;
    dstCoord.z = blockIdx.z;
    dstCoord.y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (dstCoord.y < dstSize.y)
    {
        dstCoord.x = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

        float srcCoordY = ((float)dstCoord.y + .5f) * scaleRatio.y - .5f;
        int3  iSrcCoord{0, (int)floor(srcCoordY), dstCoord.z};

        float2 w;

        w.y = ((iSrcCoord.y < 0) ? 0 : ((iSrcCoord.y > srcSize.y - 2) ? 1 : srcCoordY - iSrcCoord.y));

        iSrcCoord.y = cuda::max(0, cuda::min(iSrcCoord.y, srcSize.y - 2));

        T *dstRow = dst.ptr(dstCoord.z, dstCoord.y);

        LinearInterpolatePack<INTERSECT>(dstRow, src, iSrcCoord, srcSize.x, dstCoord.x, dstSize.x, scaleRatio.x, w);
    }
}

template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void LinearResize(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    LinearResizeBody<INTERSECT>(src, dst, srcSize, dstSize, scaleRatio);
}

// Same body as LinearResize; only the occupancy request differs.
template<bool INTERSECT, class SrcWrapper, class DstWrapper>
__global__ void __launch_bounds__(kInterpThreadsPerBlock, kLinearFloatBlocksPerSM)
    LinearResizePinned(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, float2 scaleRatio)
{
    LinearResizeBody<INTERSECT>(src, dst, srcSize, dstSize, scaleRatio);
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
    constexpr int preferredGN = (cuda::NumElements<T> == 3) ? 2 : 4;
    constexpr int GN          = NIX<T> < preferredGN ? NIX<T> : preferredGN;
    constexpr int kMaxC       = 3;
    static_assert(NIX<T> % GN == 0);

    const int z  = blockIdx.z;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (y >= dstSize.y || x0 >= dstSize.x)
        return;

    const float fsy1 = y * scaleRatio.y;
    const float fsy2 = fminf(fsy1 + scaleRatio.y, (float)srcSize.y);
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Clamp the continuous right/bottom bounds before rounding. At the final output coordinate,
    // float error can otherwise put sx2/sy2 one past the source; clamping removes only that
    // zero constant-border edge contribution and keeps the affected read in bounds.
    const bool  hasTop = (float)sy1 > fsy1;
    const bool  hasBot = (float)sy2 < fsy2;
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
            fsx2[j]     = fminf(fsx1[j] + scaleRatio.x, (float)srcSize.x);
            sx1[j]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[j]);
            sx2[j]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[j]);
            nC[j]       = sx2[j] - sx1[j];
            hasL[j]     = (float)sx1[j] > fsx1[j];
            hasR[j]     = (float)sx2[j] < fsx2[j];
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

    dim3 threads1(32, kInterpThreadsPerBlock / 32, 1);
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
        if constexpr (kPinLinearOccupancy<T>)
        {
            if (scaleRatio.x < 2)
                LinearResizePinned<true><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
            else
                LinearResizePinned<false><<<blocks1, threads1, 0, stream>>>(srcTW, dstTW, srcSize, dstSize, scaleRatio);
        }
        else if (scaleRatio.x < 2)
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

    // cudaGetLastError() is sticky until read: one check here covers every launch above.
    NVCV_CHECK_THROW(cudaGetLastError());
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
    // F16 instantiates real half kernels: the interpolation paths accumulate in FP32
    // (ConvertBaseTypeTo<float, T>) and round to half once via SaturateCast on store.
    else if CVCUDA_RUN_RESIZE(F16, F16, __half)
    else if CVCUDA_RUN_RESIZE(F16, 3F16, half3)
    else if CVCUDA_RUN_RESIZE(F16, 4F16, half4)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input data type");
    }

#undef CVCUDA_RUN_RESIZE

    // clang-format on
}

// VarShape kernels ------------------------------------------------------------

// Host-verified uniform batch scale for the var-shape fast paths: per-image sizes are only
// host-accessible through the batch handles (the exported imageList is device memory), so the
// operator classifies the batch once and the dispatch picks the matching specialized kernel.
// kGeneric covers mixed batches and every case without a fast path.
enum class VarShapeScale
{
    kGeneric,
    kExpand2x,          // every image pair is exactly 2x up on both axes
    kContract2x,        // every image pair is exactly 2x down on both axes
    kFractionalZoomOut, // every image pair zooms out with a non-integer ratio on both axes
};

// The interleaved (one plane, channels baked into T) and planar (one scalar plane per channel)
// var-shape paths compute the same per-pixel math. Both index grid-z by image; the planar kernels
// loop the channel planes per thread, which lets them hoist the plane-invariant coordinate work out
// of that loop. AREA shares one plane-aware __device__ body between the two paths (the interleaved
// kernel passes plane 0, and ptr(s, 0, y, x) == ptr(s, y, x) keeps its codegen unchanged); the other
// interpolations keep separate kernels because their fast paths differ beyond the plane mapping.

//******************** NN = Nearest Neighbor

// Interleaved var-shape nearest-neighbour. Each thread owns NN_NIX grid-strided output columns of
// one destination row and hoists the row-invariant terms -- the source scales (two FP divisions),
// the source row index sy, and the source row pointer -- computing them once and reusing them across
// the columns; the original kernel recomputed all of that per output pixel. Columns are grid-strided
// by blockDim.x so each store step stays coalesced. The per-pixel index math is byte-identical, so
// the result is bit-exact. The planar kernel (resize_NN_planar) is separate and unaffected.
template<int NN_NIX, typename T>
__global__ void resize_NN(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_y >= dstHeight)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    // Source row is row-invariant: compute sy and the row pointer once.
    const int sy     = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y), height - 1);
    const T  *srcRow = src.ptr(batch_idx, 0, sy, 0);

    const int xBase = blockIdx.x * blockDim.x * NN_NIX + threadIdx.x;

#pragma unroll
    for (int i = 0; i < NN_NIX; ++i)
    {
        const int dst_x = xBase + i * static_cast<int>(blockDim.x);
        if (dst_x >= dstWidth)
            continue;

        const int sx                         = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x), width - 1);
        *dst.ptr(batch_idx, 0, dst_y, dst_x) = srcRow[sx];
    }
} //resize_NN

//******************** Bilinear

// Interleaved var-shape bilinear. Each thread owns BILINEAR_NIX output columns of one destination
// row and hoists the row-invariant terms -- the source scales (two FP divisions), the y
// coordinate/weight, and the two source row pointers -- computing them once and reusing them across
// the columns. The original kernel ran one thread per output pixel and recomputed all of that per
// pixel, leaving it co-limited by L1/TEX and compute (~80% each, two divisions per pixel). Columns
// are grid-strided by blockDim.x so each store step stays coalesced. The per-pixel arithmetic is
// byte-identical to the scalar kernel, so the result is bit-exact. (The planar bilinear kernel has
// its own channel-amortized body, resize_bilinear_planar, and is unaffected.)
template<int BILINEAR_NIX, typename T>
__global__ void resize_bilinear(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_y >= dstHeight)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    // y coordinate + weight: row-invariant, computed once and reused across the thread's columns.
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

    fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
    sy = cuda::max(0, cuda::min(sy, height - 2));

    // Source row pointers: also row-invariant, hoisted out of the column loop.
    const T *aPtr = src.ptr(batch_idx, 0, sy, 0);     // upper row
    const T *bPtr = src.ptr(batch_idx, 0, sy + 1, 0); // lower row

    const int xBase = blockIdx.x * blockDim.x * BILINEAR_NIX + threadIdx.x;

#pragma unroll
    for (int i = 0; i < BILINEAR_NIX; ++i)
    {
        const int dst_x = xBase + i * static_cast<int>(blockDim.x);
        if (dst_x >= dstWidth)
            continue;

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

        fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
        sx = cuda::max(0, cuda::min(sx, width - 2));

        *dst.ptr(batch_idx, 0, dst_y, dst_x)
            = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                    + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
    }
} //resize_bilinear

// Interleaved var-shape bilinear for batches where every image is an exact-2x upscale, mirroring
// the tensor path's LinearResizeExpand2x: the weights are exactly 0.25f/0.75f per axis (the same
// fx/fy resize_bilinear derives), so each thread owns NIX output columns x 2 output rows,
// stages the shared clamped 3-row source window in registers, and writes two vector packs instead
// of NIX*2 single-byte stores (the general kernel is issue/L1-bound on those: SM 85%, IPC 3.0,
// DRAM 30% for U8). Border outputs replicate resize_bilinear's weight overrides (w=0/w=1), which
// zero exactly the taps whose window values differ from clamped-coordinate loads, and the blend
// uses resize_bilinear's exact expression tree, so the result is bit-exact. Dispatched only when
// the host verifies every image pair in the batch is exactly 2x on both axes.
template<typename T>
__global__ void resize_bilinear_expand2x(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    constexpr int KC = NIX<T> / 2; // source columns owned per thread
    constexpr int WC = KC + 2;     // staged window columns (taps k0-1 .. k0+KC)

    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    const int k0     = (blockIdx.x * blockDim.x + threadIdx.x) * KC; // first owned source column
    const int j      = blockIdx.y * blockDim.y + threadIdx.y;        // owned source row
    const int dst_x0 = k0 * 2;
    const int dst_y0 = j * 2;

    if (dst_x0 >= dstWidth || dst_y0 >= dstHeight)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    // Clamped source window shared by all 2*NIX outputs.
    T win[3][WC];
#pragma unroll
    for (int r = 0; r < 3; ++r)
    {
        const T *srcRow = src.ptr(batch_idx, cuda::clamp(j - 1 + r, 0, height - 1), 0);
#pragma unroll
        for (int c = 0; c < WC; ++c)
        {
            win[r][c] = srcRow[cuda::clamp(k0 - 1 + c, 0, width - 1)];
        }
    }

    // Even output row 2j: sy = j-1, fy = 0.75 (window rows 0,1); odd row 2j+1: sy = j, fy = 0.25
    // (rows 1,2) -- with resize_bilinear's border overrides.
    const float fyE = (j - 1 < 0) ? 0.f : 0.75f; // j-1 > height-2 cannot happen (j < height)
    const float fyO = (j > height - 2) ? 1.f : 0.25f;

    T out0[NIX<T>], out1[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        const bool  odd = (i & 1) != 0;
        const int   sx  = odd ? (k0 + i / 2) : (k0 + i / 2 - 1);
        const float fx  = (sx < 0) ? 0.f : ((sx > width - 2) ? 1.f : (odd ? 0.25f : 0.75f));
        const int   cb  = (i / 2) + (odd ? 1 : 0); // window column of the first x tap

        out0[i] = cuda::SaturateCast<T>((1.0f - fx) * (win[0][cb] * (1.0f - fyE) + win[1][cb] * fyE)
                                        + fx * (win[0][cb + 1] * (1.0f - fyE) + win[1][cb + 1] * fyE));
        out1[i] = cuda::SaturateCast<T>((1.0f - fx) * (win[1][cb] * (1.0f - fyO) + win[2][cb] * fyO)
                                        + fx * (win[1][cb + 1] * (1.0f - fyO) + win[2][cb + 1] * fyO));
    }

    const int validCount = cuda::min(NIX<T>, dstWidth - dst_x0);
    // dstHeight = 2*height is even, so the odd output row always exists.
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        T *dstRow              = dst.ptr(batch_idx, dst_y0 + p, 0);
        const T(&outp)[NIX<T>] = p == 0 ? out0 : out1;

        if (validCount == NIX<T> && CheckRowAlign(dstRow + dst_x0)) // uniform across the warp
            WritePack(dstRow[dst_x0], outp);
        else
        {
            T *dstPtr = dstRow + dst_x0;
#pragma unroll
            for (int c = 0; c < NIX<T>; ++c)
                if (c < validCount)
                    dstPtr[c] = outp[c];
        }
    }
}

// Interleaved var-shape bilinear for byte types with NIX consecutive output columns per
// thread: upscales revisit the same source cell for several adjacent outputs, so the four taps are
// cached as work-type floats (float(byte) conversion is exact) and refreshed only when the source
// column advances -- the grid-strided general kernel re-reads and re-converts them per output. The
// consecutive columns also allow one vector-pack store per row segment instead of NIX scalar
// stores. Coordinate math, border overrides, and the blend expression tree are identical to
// resize_bilinear, so the result is bit-exact at every scale; downscales simply never hit the
// cache-reuse fast case.
template<typename T>
__global__ void resize_bilinear_pack(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    using FT = cuda::ConvertBaseTypeTo<float, T>;

    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (dst_y >= dstHeight || x0 >= dstWidth)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

    fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
    sy = cuda::max(0, cuda::min(sy, height - 2));

    const T *aPtr = src.ptr(batch_idx, 0, sy, 0);     // upper row
    const T *bPtr = src.ptr(batch_idx, 0, sy + 1, 0); // lower row

    // Cached float taps of the current source cell [prevSx, prevSx+1].
    FT  a0{}, a1{}, b0{}, b1{};
    int prevSx = -2; // never matches or adjoins a valid first column

    T out[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        const int dst_x = x0 + i;

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

        fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
        sx = cuda::max(0, cuda::min(sx, width - 2));

        if (sx == prevSx + 1)
        {
            a0 = a1;
            b0 = b1;
            a1 = cuda::StaticCast<float>(aPtr[sx + 1]);
            b1 = cuda::StaticCast<float>(bPtr[sx + 1]);
        }
        else if (sx != prevSx)
        {
            a0 = cuda::StaticCast<float>(aPtr[sx]);
            a1 = cuda::StaticCast<float>(aPtr[sx + 1]);
            b0 = cuda::StaticCast<float>(bPtr[sx]);
            b1 = cuda::StaticCast<float>(bPtr[sx + 1]);
        }
        prevSx = sx;

        out[i] = cuda::SaturateCast<T>((1.0f - fx) * (a0 * (1.0f - fy) + b0 * fy) + fx * (a1 * (1.0f - fy) + b1 * fy));
    }

    T        *dstRow     = dst.ptr(batch_idx, 0, dst_y, 0);
    const int validCount = cuda::min(NIX<T>, dstWidth - x0);

    if (validCount == NIX<T> && CheckRowAlign(dstRow + x0))
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

//******************** Bicubic

// Interleaved var-shape bicubic. Each thread owns BICUBIC_NIX grid-strided output columns of one
// destination row and hoists the row-invariant terms -- the source scales (two FP divisions), the y
// coordinate/fraction, the four y cubic coefficients cY[4], and the four clamped source-row pointers
// -- computing them once and reusing them across the columns; the original kernel recomputed all of
// that per output pixel (16 taps each). Columns are grid-strided by blockDim.x so each store step
// stays coalesced. The per-pixel arithmetic and accumulation order are byte-identical to the scalar
// kernel, so the result is bit-exact. The planar kernel (resize_bicubic_planar) is separate and
// unaffected.
template<int BICUBIC_NIX, typename T>
__global__ void resize_bicubic(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_y >= dstHeight)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const float A = -0.75f;

    // y coordinate, y cubic coefficients, and the four clamped source-row pointers are all
    // row-invariant: compute them once and reuse across this thread's columns.
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
    fy -= sy;

    float cY[4];
    cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
    cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
    cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
    cY[3] = 1.f - cY[0] - cY[1] - cY[2];

    const T *rowPtr[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) rowPtr[row] = src.ptr(batch_idx, 0, cuda::clamp(sy + row - 1, 0, height - 1), 0);

    const int xBase = blockIdx.x * blockDim.x * BICUBIC_NIX + threadIdx.x;

#pragma unroll
    for (int i = 0; i < BICUBIC_NIX; ++i)
    {
        const int dst_x = xBase + i * static_cast<int>(blockDim.x);
        if (dst_x >= dstWidth)
            continue;

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;

        // Cubic polynomial coefficients -- coordinate clamping is deferred to the sampling loop.
        float cX[4];
        cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
        cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
        cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
        cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

        work_type accum = cuda::SetAll<work_type>(0);

        // Clamp each source tap coordinate independently to [0, size-1] (replicate border).
#pragma unroll
        for (int row = 0; row < 4; ++row)
        {
#pragma unroll
            for (int col = 0; col < 4; ++col)
            {
                int csx = cuda::clamp(sx + col - 1, 0, width - 1);
                accum += cY[row] * cX[col] * rowPtr[row][csx];
            }
        } //for row
        *dst.ptr(batch_idx, 0, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
    }
} //resize_bicubic

// Shared-memory-tiled var-shape bicubic for FLOAT elements. Per block (one image): if the image
// upscales (both scales < 1) the block's output tile maps to a SMALL source tile, so it is staged
// once in shared memory (replicate-border clamp baked in at load) and the 16 taps/pixel are served
// from smem instead of the L1/TEX-bound gather; CONTRACT/same-size blocks fall back to the direct
// gather (identical to resize_bicubic). Because every upscale block has scale < 1, the tile never
// exceeds blockDim + cubic support, so a single launch-time tile size (blockDim + 6) is valid for
// any per-image scale in the batch. Bit-exact with resize_bicubic: tile[tap-origin] == src[clamp(tap)]
// and the coefficient formula/accumulation order are identical.
template<typename T>
__global__ void resize_bicubic_smem(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                    int tileW, int tileH)
{
    extern __shared__ __align__(16) unsigned char smemRaw[];
    T                                            *tile = reinterpret_cast<T *>(smemRaw);

    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    const float A   = -0.75f;

    const int   b     = blockIdx.z;
    const int   width = src.width(b), height = src.height(b);
    const int   dstW = dst.width(b), dstH = dst.height(b);
    const float scale_x = static_cast<float>(width) / dstW;
    const float scale_y = static_cast<float>(height) / dstH;

    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Uniform across the block (depends only on the image): keeps the __syncthreads() below uniform.
    const bool expand = (scale_x < 1.f && scale_y < 1.f);

    auto coeffs = [&](float f, float c[4])
    {
        c[0] = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A;
        c[1] = ((A + 2) * f - (A + 3)) * f * f + 1;
        c[2] = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1;
        c[3] = 1.f - c[0] - c[1] - c[2];
    };

    if (expand)
    {
        const int ox0 = blockIdx.x * blockDim.x;
        const int oy0 = blockIdx.y * blockDim.y;
        const int tx0 = cuda::round<cuda::RoundMode::DOWN, int>((ox0 + 0.5f) * scale_x - 0.5f) - 1;
        const int ty0 = cuda::round<cuda::RoundMode::DOWN, int>((oy0 + 0.5f) * scale_y - 0.5f) - 1;

        const int nThreads = blockDim.x * blockDim.y;
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < tileW * tileH; idx += nThreads)
        {
            const int sx = cuda::clamp(tx0 + idx % tileW, 0, width - 1);
            const int sy = cuda::clamp(ty0 + idx / tileW, 0, height - 1);
            tile[idx]    = *src.ptr(b, 0, sy, sx);
        }
        __syncthreads();

        if (dst_x >= dstW || dst_y >= dstH)
            return;

        float fx = (dst_x + 0.5f) * scale_x - 0.5f;
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;
        float fy = (dst_y + 0.5f) * scale_y - 0.5f;
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;

        float cX[4], cY[4];
        coeffs(fx, cX);
        coeffs(fy, cY);

        work_type accum = cuda::SetAll<work_type>(0);
#pragma unroll
        for (int row = 0; row < 4; ++row)
        {
            const int jj = (sy + row - 1) - ty0;
#pragma unroll
            for (int col = 0; col < 4; ++col)
            {
                const int ii = (sx + col - 1) - tx0;
                accum += cY[row] * cX[col] * tile[jj * tileW + ii];
            }
        }
        *dst.ptr(b, 0, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
    }
    else
    {
        if (dst_x >= dstW || dst_y >= dstH)
            return;

        float fx = (dst_x + 0.5f) * scale_x - 0.5f;
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;
        float fy = (dst_y + 0.5f) * scale_y - 0.5f;
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;

        float cX[4], cY[4];
        coeffs(fx, cX);
        coeffs(fy, cY);

        const T *rowPtr[4];
#pragma unroll
        for (int row = 0; row < 4; ++row) rowPtr[row] = src.ptr(b, 0, cuda::clamp(sy + row - 1, 0, height - 1), 0);

        work_type accum = cuda::SetAll<work_type>(0);
#pragma unroll
        for (int row = 0; row < 4; ++row)
#pragma unroll
            for (int col = 0; col < 4; ++col)
            {
                const int csx = cuda::clamp(sx + col - 1, 0, width - 1);
                accum += cY[row] * cX[col] * rowPtr[row][csx];
            }
        *dst.ptr(b, 0, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
    }
}

//******************** Integrate area

template<typename T>
__device__ __forceinline__ void resizeAreaPlane(const cuda::ImageBatchVarShapeWrap<const T>                   src,
                                                const cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brd_src,
                                                cuda::ImageBatchVarShapeWrap<T> dst, int batch_idx, int plane, int x,
                                                int y)
{
    int dstWidth  = dst.width(batch_idx);
    int dstHeight = dst.height(batch_idx);

    if (x >= dstWidth || y >= dstHeight)
        return;
    int height = src.height(batch_idx), width = src.width(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    // Coordinate-space inverse scales kept in FP32: they only feed back into the
    // per-pixel float fy/fx computation below.  FP64 here was throughput-pinning
    // the kernel on consumer GPUs where FP64 runs at 1/64 of FP32.
    float inv_scale_x  = 1.f / scale_x;
    float inv_scale_y  = 1.f / scale_y;
    int   iscale_x     = cuda::SaturateCast<int>(scale_x);
    int   iscale_y     = cuda::SaturateCast<int>(scale_y);
    bool  is_area_fast = abs(scale_x - iscale_x) < DBL_EPSILON && abs(scale_y - iscale_y) < DBL_EPSILON;

    if (scale_x >= 1.0f && scale_y >= 1.0f) // zoom out
    {
        if (is_area_fast) // integer multiples
        {
            float scale = 1.f / (scale_x * scale_y);
            float fsx1  = x * scale_x;
            float fsx2  = fsx1 + scale_x;

            int sx1 = cuda::round<cuda::RoundMode::UP, int>(fsx1);
            int sx2 = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = cuda::round<cuda::RoundMode::UP, int>(fsy1);
            int sy2 = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

            using work_type = cuda::ConvertBaseTypeTo<float, T>;
            work_type out   = {0};

            // Integer downscale: the box [sx1,sx2) x [sy1,sy2) is fully in-bounds, so read the source
            // rows directly (pointer hoisted per row) instead of paying the per-tap border-wrap bounds
            // check. Bit-exact: same values, same per-tap scale, same accumulation order as brd_src[].
            for (int dy = sy1; dy < sy2; ++dy)
            {
                const T *srcRow = src.ptr(batch_idx, plane, dy, 0);

                for (int dx = sx1; dx < sx2; ++dx)
                {
                    out = out + srcRow[dx] * scale;
                }
            }
            *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(out);
            return;
        }

        float fsx1 = x * scale_x;
        float fsx2 = fsx1 + scale_x;

        int sx1 = cuda::round<cuda::RoundMode::UP, int>(fsx1);
        int sx2 = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);

        float fsy1 = y * scale_y;
        float fsy2 = fsy1 + scale_y;

        int sy1 = cuda::round<cuda::RoundMode::UP, int>(fsy1);
        int sy2 = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

        float scale
            = 1.f / (fminf(scale_x, src.width(batch_idx) - fsx1) * fminf(scale_y, src.height(batch_idx) - fsy1));

        using work_type = cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = {0};

        int4 srcCoord = {0, 0, plane, batch_idx};

        for (int dy = sy1; dy < sy2; ++dy)
        {
            srcCoord.y = dy;

            // Interior box columns [sx1,sx2) are in-bounds for a downscale, so read the row directly
            // (pointer hoisted once) instead of the per-tap border-wrap. Fractional edge taps below may
            // sit at sx1-1 / sx2 and keep the border-wrapped access. Bit-exact with the scalar path.
            const T *srcRow = src.ptr(batch_idx, plane, dy, 0);

            for (int dx = sx1; dx < sx2; ++dx)
            {
                out = out + srcRow[dx] * scale;
            }

            if (sx1 > fsx1)
            {
                srcCoord.x = sx1 - 1;
                out        = out + brd_src[srcCoord] * ((sx1 - fsx1) * scale);
            }

            if (sx2 < fsx2)
            {
                srcCoord.x = sx2;
                out        = out + brd_src[srcCoord] * ((fsx2 - sx2) * scale);
            }
        }

        if (sy1 > fsy1)
        {
            srcCoord.y = sy1 - 1;
            for (int dx = sx1; dx < sx2; ++dx)
            {
                srcCoord.x = dx;
                out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * scale);
            }
        }

        if (sy2 < fsy2)
        {
            srcCoord.y = sy2;
            for (int dx = sx1; dx < sx2; ++dx)
            {
                srcCoord.x = dx;
                out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * scale);
            }
        }

        if ((sy1 > fsy1) && (sx1 > fsx1))
        {
            srcCoord.y = (sy1 - 1);
            srcCoord.x = (sx1 - 1);
            out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * (sx1 - fsx1) * scale);
        }

        if ((sy1 > fsy1) && (sx2 < fsx2))
        {
            srcCoord.y = (sy1 - 1);
            srcCoord.x = sx2;
            out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * (fsx2 - sx2) * scale);
        }

        if ((sy2 < fsy2) && (sx2 < fsx2))
        {
            srcCoord.y = sy2;
            srcCoord.x = sx2;
            out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * (fsx2 - sx2) * scale);
        }

        if ((sy2 < fsy2) && (sx1 > fsx1))
        {
            srcCoord.y = sy2;
            srcCoord.x = sx1 - 1;
            out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * (sx1 - fsx1) * scale);
        }

        *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(out);
        return;
    }

    // zoom in, it is emulated using some variant of bilinear interpolation
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(y * scale_y);
    float fy = (y + 1) - (sy + 1) * inv_scale_y;
    fy       = fy <= 0 ? 0.f : fy - cuda::round<cuda::RoundMode::DOWN, int>(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = cuda::round<cuda::RoundMode::DOWN, int>(x * scale_x);
    float fx = (x + 1) - (sx + 1) * inv_scale_x;
    fx       = fx < 0 ? 0.f : fx - cuda::round<cuda::RoundMode::DOWN, int>(fx);

    if (sx < 0)
    {
        fx = 0, sx = 0;
    }

    if (sx >= src.width(batch_idx) - 1)
    {
        fx = 0, sx = src.width(batch_idx) - 2;
    }
    if (sy >= src.height(batch_idx) - 1)
    {
        sy = src.height(batch_idx) - 2;
    }

    float cbufx[2];
    cbufx[0] = 1.f - fx;
    cbufx[1] = fx;

    *dst.ptr(batch_idx, plane, y, x)
        = cuda::SaturateCast<T>((*src.ptr(batch_idx, plane, sy, sx) * cbufx[0] * cbufy[0]
                                 + *src.ptr(batch_idx, plane, sy + 1, sx) * cbufx[0] * cbufy[1]
                                 + *src.ptr(batch_idx, plane, sy, sx + 1) * cbufx[1] * cbufy[0]
                                 + *src.ptr(batch_idx, plane, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<typename T>
__global__ void resize_area_ocv_align(const cuda::ImageBatchVarShapeWrap<const T>                   src,
                                      const cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brd_src,
                                      cuda::ImageBatchVarShapeWrap<T>                               dst)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    resizeAreaPlane<T>(src, brd_src, dst, blockIdx.z, 0, x, y);
}

// Exact-2x AREA downscale body shared by the interleaved and planar kernels: every output is the
// mean of a fully-in-bounds 2x2 source box (sx1 = 2x, sy1 = 2y, scale = 1/(2*2) = 0.25f -- the same
// values resizeAreaPlane's integer branch derives), so the per-pixel coordinate rounding and the
// runtime-bounded box loops collapse to four unrolled taps. Each thread emits NIX consecutive
// outputs of one row as a vector pack instead of single-element stores (the general kernel is
// issue-bound on the U8 rows: SM 71%, IPC 2.8, DRAM 40% local). Accumulation order matches the
// general integer branch exactly (row-major taps, per-tap * scale), so the result is bit-exact.
template<typename T>
__device__ __forceinline__ void resizeAreaContract2xPlane(const cuda::ImageBatchVarShapeWrap<const T> src,
                                                          cuda::ImageBatchVarShapeWrap<T> dst, int batch_idx, int plane,
                                                          int x0, int y, int dstWidth)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const T *row0 = src.ptr(batch_idx, plane, 2 * y, 0);
    const T *row1 = src.ptr(batch_idx, plane, 2 * y + 1, 0);

    const int sx0        = 2 * x0;
    const int validCount = cuda::min(NIX<T>, dstWidth - x0);

    // Each thread consumes 2*NIX consecutive source elements per row; read them as two vector
    // packs per row instead of 4*NIX single-element loads -- the scalar-load version of this
    // body was L1/TEX-transaction-bound (99% L1 SOL, DRAM 70%, SM 29%). The staged values are
    // identical, so the accumulation below is unchanged.
    T win0[2 * NIX<T>], win1[2 * NIX<T>];
    if (validCount == NIX<T> && CheckRowAlign(row0 + sx0) && CheckRowAlign(row1 + sx0))
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
        // Source width is exactly 2*dstWidth, so tail loads stay guarded per element.
#pragma unroll
        for (int c = 0; c < 2 * NIX<T>; ++c)
        {
            const int sx = sx0 + c;
            win0[c]      = (sx < 2 * dstWidth) ? row0[sx] : T{};
            win1[c]      = (sx < 2 * dstWidth) ? row1[sx] : T{};
        }
    }

    T out[NIX<T>];
#pragma unroll
    for (int i = 0; i < NIX<T>; ++i)
    {
        work_type acc = {0};
        acc           = acc + win0[2 * i] * 0.25f;
        acc           = acc + win0[2 * i + 1] * 0.25f;
        acc           = acc + win1[2 * i] * 0.25f;
        acc           = acc + win1[2 * i + 1] * 0.25f;
        out[i]        = cuda::SaturateCast<T>(acc);
    }

    T *dstRow = dst.ptr(batch_idx, plane, y, 0);

    if (validCount == NIX<T> && CheckRowAlign(dstRow + x0))
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

template<typename T>
__global__ void resize_area_contract2x(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = blockIdx.z;
    const int x0        = (blockDim.x * blockIdx.x + threadIdx.x) * NIX<T>;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    resizeAreaContract2xPlane<T>(src, dst, batch_idx, 0, x0, y, dst.width(batch_idx));
}

template<typename T>
__global__ void resize_area_contract2x_planar(cuda::ImageBatchVarShapeWrap<const T> src,
                                              cuda::ImageBatchVarShapeWrap<T> dst, int channels)
{
    const int batch_idx = blockIdx.z;
    const int x0        = (blockDim.x * blockIdx.x + threadIdx.x) * NIX<T>;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        resizeAreaContract2xPlane<T>(src, dst, batch_idx, plane, x0, y, dst.width(batch_idx));
    }
}

// Interleaved var-shape fractional AREA downscale with NIX consecutive output columns per
// thread. For a zoom-out every box tap -- including the fractional edge/corner taps -- is in
// bounds (sx1-1 >= floor(fsx1) >= 0 and sx2 <= width-1, likewise in y), so the general body's
// per-tap border-wrap accesses are replaced by plain hoisted row-pointer reads (value-identical),
// the row-invariant y box terms are computed once per thread, and the NIX results leave as one
// vector pack instead of scalar stores (the general kernel is issue/L1-bound: SM 75%, IPC 2.8,
// L1 83%, DRAM 34% local). Per-output term order -- interior rows (cols ascending, then left,
// then right edge), top row, bottom row, then TL/TR/BR/BL corners, each * scale -- replicates
// resizeAreaPlane exactly, so the result is bit-exact. Integer ratios and zoom-ins keep their
// existing kernels.
template<typename T>
__device__ __forceinline__ void ResizeAreaFractionalPackBody(cuda::ImageBatchVarShapeWrap<const T> src,
                                                             cuda::ImageBatchVarShapeWrap<T>       dst)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    // Grouped independent accumulation chains (GN outputs at a time) with a compile-time interior
    // column bound (dispatch guarantees x scale < 3, so at most kMaxC interior columns): the
    // serial one-output walk was memory-latency bound. Per-output term order is unchanged
    // (interior columns ascending, then left, then right edge per row; top row; bottom row; then
    // TL/TR/BR/BL corners), so the result stays bit-exact with resizeAreaPlane.
    // Three-channel elements lose to L1 pressure with four chains: GN = 4 regressed the uchar3
    // row +14% on A100 against its reference baseline (locally it was +5%); GN = 2 matches the
    // tensor kernel's measured split.
    constexpr int preferredGN = (cuda::NumElements<T> == 3) ? 2 : 4;
    constexpr int GN          = NIX<T> < preferredGN ? NIX<T> : preferredGN;
    constexpr int kMaxC       = 3;
    static_assert(NIX<T> % GN == 0);

    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (y >= dstHeight || x0 >= dstWidth)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    const float fsy1 = y * scale_y;
    const float fsy2 = fminf(fsy1 + scale_y, (float)height);
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Clamp the continuous right/bottom bounds before rounding. At the final output coordinate,
    // float error can otherwise put sx2/sy2 one past the source; clamping removes only that
    // zero constant-border edge contribution and keeps the affected read in bounds.
    const bool  hasTop = (float)sy1 > fsy1;
    const bool  hasBot = (float)sy2 < fsy2;
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
            const int x = cuda::min(x0 + g + j, dstWidth - 1);
            fsx1[j]     = x * scale_x;
            fsx2[j]     = fminf(fsx1[j] + scale_x, (float)width);
            sx1[j]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[j]);
            sx2[j]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[j]);
            nC[j]       = sx2[j] - sx1[j];
            hasL[j]     = (float)sx1[j] > fsx1[j];
            hasR[j]     = (float)sx2[j] < fsx2[j];
            wL[j]       = (float)sx1[j] - fsx1[j];
            wR[j]       = fsx2[j] - (float)sx2[j];
            scaleA[j]   = 1.f / (fminf(scale_x, width - fsx1[j]) * fminf(scale_y, height - fsy1));
        }

        work_type acc[GN];
#pragma unroll
        for (int j = 0; j < GN; ++j) acc[j] = work_type{0};

        for (int dy = sy1; dy < sy2; ++dy)
        {
            const T *row = src.ptr(batch_idx, 0, dy, 0);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] = acc[j] + row[sx1[j] + c] * scaleA[j];
                }
            }
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
                if (hasL[j])
                    acc[j] = acc[j] + row[sx1[j] - 1] * (wL[j] * scaleA[j]);
                if (hasR[j])
                    acc[j] = acc[j] + row[sx2[j]] * (wR[j] * scaleA[j]);
            }
        }

        if (hasTop)
        {
            const T *row = src.ptr(batch_idx, 0, sy1 - 1, 0);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] = acc[j] + row[sx1[j] + c] * (wTop * scaleA[j]);
                }
            }
        }
        if (hasBot)
        {
            const T *row = src.ptr(batch_idx, 0, sy2, 0);
#pragma unroll
            for (int j = 0; j < GN; ++j)
            {
#pragma unroll
                for (int c = 0; c < kMaxC; ++c)
                {
                    if (c < nC[j])
                        acc[j] = acc[j] + row[sx1[j] + c] * (wBot * scaleA[j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < GN; ++j)
        {
            if (hasTop && hasL[j])
                acc[j] = acc[j] + src.ptr(batch_idx, 0, sy1 - 1, 0)[sx1[j] - 1] * (wTop * wL[j] * scaleA[j]);
            if (hasTop && hasR[j])
                acc[j] = acc[j] + src.ptr(batch_idx, 0, sy1 - 1, 0)[sx2[j]] * (wTop * wR[j] * scaleA[j]);
            if (hasBot && hasR[j])
                acc[j] = acc[j] + src.ptr(batch_idx, 0, sy2, 0)[sx2[j]] * (wBot * wR[j] * scaleA[j]);
            if (hasBot && hasL[j])
                acc[j] = acc[j] + src.ptr(batch_idx, 0, sy2, 0)[sx1[j] - 1] * (wBot * wL[j] * scaleA[j]);
        }

#pragma unroll
        for (int j = 0; j < GN; ++j) out[g + j] = cuda::SaturateCast<T>(acc[j]);
    }

    T        *dstRow     = dst.ptr(batch_idx, 0, y, 0);
    const int validCount = cuda::min(NIX<T>, dstWidth - x0);

    if (validCount == NIX<T> && CheckRowAlign(dstRow + x0))
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

// Threads per block for the var-shape launches (THREADS_PER_BLOCK in RunResizeVarShape); four
// resident blocks per SM is a 64-register cap.
constexpr int kVarShapeThreadsPerBlock = 256;
constexpr int kVarShapeAreaBlocksPerSM = 4;

// The 3-channel byte form sits right on that cap: CUDA 12.2 wants 68 registers and drops to three
// resident blocks, measured as ~5.6% on the RGB8 var-shape anisotropic AREA downscale on A100.
// No other element type is pinned: the 16-bit forms would lose a block they already have, and the
// wider forms carry far more live box state and never had four.
template<typename T>
constexpr bool kPinVarShapeAreaOccupancy = cuda::NumElements<T> == 3 && sizeof(cuda::BaseType<T>) == 1;

template<typename T>
__global__ void resize_area_fractional_pack(cuda::ImageBatchVarShapeWrap<const T> src,
                                            cuda::ImageBatchVarShapeWrap<T>       dst)
{
    ResizeAreaFractionalPackBody<T>(src, dst);
}

// Same body as resize_area_fractional_pack; only the occupancy request differs.
template<typename T>
__global__ void __launch_bounds__(kVarShapeThreadsPerBlock, kVarShapeAreaBlocksPerSM)
    resize_area_fractional_pack_pinned(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    ResizeAreaFractionalPackBody<T>(src, dst);
}

//******************** Planar (NCHW/CHW) variants
//
// Resize treats channels independently, so each plane of a planar image is resized exactly like a
// single-channel image. One thread owns an output pixel across ALL channel planes of an image
// (grid.z runs over numImages), computing the per-pixel coordinates/weights -- which depend only on
// (x, y), not the channel -- once and looping the planes. This avoids the (channels - 1)x redundant
// coordinate math of one thread per (image, plane), and keeps the per-plane result bit-exact with
// the interleaved path.

template<typename T>
__global__ void resize_NN_planar(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                 int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

        const float scale_x = static_cast<float>(width) / dstWidth;
        const float scale_y = static_cast<float>(height) / dstHeight;
        const int   sx      = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x), width - 1);
        const int   sy      = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y), height - 1);

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(batch_idx, plane, dst_y, dst_x) = *src.ptr(batch_idx, plane, sy, sx);
        }
    }
}

// Planar bilinear, vectorized: each thread owns PNIX grid-strided output columns of one row. The
// row-invariant y terms (scale, fy, sy) are hoisted once, and the per-column x terms (fx, sx) are
// precomputed once and reused across every channel plane -- the scalar kernel recomputed the source
// scales (two FP divisions) and x/y coordinates per output pixel. Each plane's two row pointers are
// resolved once. The per-pixel arithmetic is byte-identical, so the result is bit-exact.
template<int PNIX, typename T>
__global__ void resize_bilinear_planar(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                       int channels)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    if (dst_y >= dstHeight)
        return;

    const int   width   = src.width(batch_idx);
    const int   height  = src.height(batch_idx);
    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
    fy       = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
    sy       = cuda::max(0, cuda::min(sy, height - 2));

    const int xBase = blockIdx.x * blockDim.x * PNIX + threadIdx.x;

    // Per-column x terms, shared across all channel planes.
    int   dx[PNIX], sxA[PNIX];
    float fxA[PNIX];
#pragma unroll
    for (int i = 0; i < PNIX; ++i)
    {
        dx[i]    = xBase + i * static_cast<int>(blockDim.x);
        float fx = (float)((dx[i] + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fxA[i]   = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
        sxA[i]   = cuda::max(0, cuda::min(sx, width - 2));
    }

    for (int plane = 0; plane < channels; ++plane)
    {
        const T *aPtr = src.ptr(batch_idx, plane, sy, 0);     // upper row
        const T *bPtr = src.ptr(batch_idx, plane, sy + 1, 0); // lower row
#pragma unroll
        for (int i = 0; i < PNIX; ++i)
        {
            if (dx[i] >= dstWidth)
                continue;
            const int   sx = sxA[i];
            const float fx = fxA[i];
            *dst.ptr(batch_idx, plane, dst_y, dx[i])
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
}

// Channel-amortized planar bicubic: grid.z runs over images (not image*plane), so one thread owns an
// output pixel across ALL channel planes. The cubic coefficients and clamped tap coordinates depend
// only on (x, y), not the channel, so they are computed once and reused for every plane -- avoiding
// the (channels - 1)x redundant coefficient math of calling resizeBicubicPlane once per plane. The
// per-plane accumulation is identical, so output stays bit-exact with the interleaved path.
template<typename T>
__global__ void resize_bicubic_planar(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                      int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) & (dst_y < dstHeight))
    {
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

        const float scale_x = static_cast<float>(width) / dstWidth;
        const float scale_y = static_cast<float>(height) / dstHeight;

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;

        float cX[4];
        cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
        cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
        cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
        cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

        // Clamp each tap coordinate once (replicate border); reused across all planes.
        int csy[4];
        int csx[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            csy[k] = cuda::clamp(sy + k - 1, 0, height - 1);
            csx[k] = cuda::clamp(sx + k - 1, 0, width - 1);
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            work_type accum = cuda::SetAll<work_type>(0);
#pragma unroll
            for (int row = 0; row < 4; ++row)
            {
#pragma unroll
                for (int col = 0; col < 4; ++col)
                {
                    accum += cY[row] * cX[col] * *src.ptr(batch_idx, plane, csy[row], csx[col]);
                }
            }
            *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
        }
    }
}

// Planar var-shape fractional AREA downscale: one thread owns NIX consecutive output columns
// of one row and loops the channel planes. The x/y box geometry (bounds, edge weights, scale) is
// hoisted into per-column arrays outside the plane loop -- keeping the cross-channel amortization
// of the general planar kernel while adding the consecutive-column/pack-store and in-bounds
// row-pointer wins of resize_area_fractional_pack. Per-output term order matches resizeAreaPlane
// exactly, so the result is bit-exact with the general planar kernel and the interleaved path.
template<typename T>
__global__ void resize_area_fractional_pack_planar(cuda::ImageBatchVarShapeWrap<const T> src,
                                                   cuda::ImageBatchVarShapeWrap<T> dst, int channels)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const int batch_idx = blockIdx.z;
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * NIX<T>;

    if (y >= dstHeight || x0 >= dstWidth)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    const float fsy1 = y * scale_y;
    const float fsy2 = fminf(fsy1 + scale_y, (float)height);
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Per-column x box terms, shared by every channel plane. Wide elements (NIX == 4) hoist
    // them into arrays; byte planes (NIX == 16) recompute them inline per plane instead --
    // sixteen hoisted term sets spill to local memory and regressed the byte rows 1.6x.
    constexpr int  XN      = (NIX<T> <= 4) ? NIX<T> : 1;
    constexpr bool HOISTED = XN == NIX<T>;
    float          fsx1[XN], fsx2[XN], scale[XN];
    int            sx1[XN], sx2[XN];
    if constexpr (HOISTED)
    {
#pragma unroll
        for (int i = 0; i < XN; ++i)
        {
            const int x = cuda::min(x0 + i, dstWidth - 1);
            fsx1[i]     = x * scale_x;
            fsx2[i]     = fminf(fsx1[i] + scale_x, (float)width);
            sx1[i]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[i]);
            sx2[i]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[i]);
            scale[i]    = 1.f / (fminf(scale_x, width - fsx1[i]) * fminf(scale_y, height - fsy1));
        }
    }

    const int validCount = cuda::min(NIX<T>, dstWidth - x0);

    for (int plane = 0; plane < channels; ++plane)
    {
        T out[NIX<T>];
#pragma unroll
        for (int i = 0; i < NIX<T>; ++i)
        {
            if (i >= validCount)
                break;

            const int ti = HOISTED ? i : 0;
            if constexpr (!HOISTED)
            {
                const int x = cuda::min(x0 + i, dstWidth - 1);
                fsx1[0]     = x * scale_x;
                fsx2[0]     = fminf(fsx1[0] + scale_x, (float)width);
                sx1[0]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[0]);
                sx2[0]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[0]);
                scale[0]    = 1.f / (fminf(scale_x, width - fsx1[0]) * fminf(scale_y, height - fsy1));
            }

            work_type acc = {0};

            for (int dy = sy1; dy < sy2; ++dy)
            {
                const T *srcRow = src.ptr(batch_idx, plane, dy, 0);
                for (int dx = sx1[ti]; dx < sx2[ti]; ++dx)
                {
                    acc = acc + srcRow[dx] * scale[ti];
                }
                if (sx1[ti] > fsx1[ti])
                {
                    acc = acc + srcRow[sx1[ti] - 1] * ((sx1[ti] - fsx1[ti]) * scale[ti]);
                }
                if (sx2[ti] < fsx2[ti])
                {
                    acc = acc + srcRow[sx2[ti]] * ((fsx2[ti] - sx2[ti]) * scale[ti]);
                }
            }
            if (sy1 > fsy1)
            {
                const T *topRow = src.ptr(batch_idx, plane, sy1 - 1, 0);
                for (int dx = sx1[ti]; dx < sx2[ti]; ++dx)
                {
                    acc = acc + topRow[dx] * ((sy1 - fsy1) * scale[ti]);
                }
            }
            if (sy2 < fsy2)
            {
                const T *botRow = src.ptr(batch_idx, plane, sy2, 0);
                for (int dx = sx1[ti]; dx < sx2[ti]; ++dx)
                {
                    acc = acc + botRow[dx] * ((fsy2 - sy2) * scale[ti]);
                }
            }
            if ((sy1 > fsy1) && (sx1[ti] > fsx1[ti]))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy1 - 1, 0)[sx1[ti] - 1]
                          * ((sy1 - fsy1) * (sx1[ti] - fsx1[ti]) * scale[ti]);
            }
            if ((sy1 > fsy1) && (sx2[ti] < fsx2[ti]))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy1 - 1, 0)[sx2[ti]]
                          * ((sy1 - fsy1) * (fsx2[ti] - sx2[ti]) * scale[ti]);
            }
            if ((sy2 < fsy2) && (sx2[ti] < fsx2[ti]))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy2, 0)[sx2[ti]] * ((fsy2 - sy2) * (fsx2[ti] - sx2[ti]) * scale[ti]);
            }
            if ((sy2 < fsy2) && (sx1[ti] > fsx1[ti]))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy2, 0)[sx1[ti] - 1]
                          * ((fsy2 - sy2) * (sx1[ti] - fsx1[ti]) * scale[ti]);
            }

            out[i] = cuda::SaturateCast<T>(acc);
        }

        T *dstRow = dst.ptr(batch_idx, plane, y, 0);

        if (validCount == NIX<T> && CheckRowAlign(dstRow + x0))
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
}

// Channel-amortized planar area: one thread per output pixel per image (grid.z = numImages) loops the
// channel planes. The area geometry (scale factors, integer box bounds, fractional edge weights, and
// the zoom-in coordinates/weights) depends only on (x, y), not the channel, so it is computed once
// and reused for every plane; only the box accumulation reads the per-plane data. Per-plane results
// are identical to resizeAreaPlane, so output stays bit-exact with the interleaved path.
template<typename T>
__global__ void resize_area_ocv_align_planar(const cuda::ImageBatchVarShapeWrap<const T>                   src,
                                             const cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brd_src,
                                             cuda::ImageBatchVarShapeWrap<T> dst, int channels)
{
    const int x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if (x >= dstWidth || y >= dstHeight)
        return;

    const int height = src.height(batch_idx), width = src.width(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    const float inv_scale_x  = 1.f / scale_x;
    const float inv_scale_y  = 1.f / scale_y;
    const int   iscale_x     = cuda::SaturateCast<int>(scale_x);
    const int   iscale_y     = cuda::SaturateCast<int>(scale_y);
    const bool  is_area_fast = abs(scale_x - iscale_x) < DBL_EPSILON && abs(scale_y - iscale_y) < DBL_EPSILON;

    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    if (scale_x >= 1.0f && scale_y >= 1.0f) // zoom out
    {
        const float fsx1 = x * scale_x;
        const float fsx2 = fsx1 + scale_x;
        const int   sx1  = cuda::round<cuda::RoundMode::UP, int>(fsx1);
        const int   sx2  = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);
        const float fsy1 = y * scale_y;
        const float fsy2 = fsy1 + scale_y;
        const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
        const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

        if (is_area_fast) // integer multiples
        {
            const float scale = 1.f / (scale_x * scale_y);

            // Integer downscale: the box is fully in-bounds; read source rows directly (pointer hoisted
            // per row) instead of the per-tap border-wrap. Bit-exact and identical to the interleaved path.
            for (int plane = 0; plane < channels; ++plane)
            {
                work_type out = {0};
                for (int dy = sy1; dy < sy2; ++dy)
                {
                    const T *srcRow = src.ptr(batch_idx, plane, dy, 0);
                    for (int dx = sx1; dx < sx2; ++dx)
                    {
                        out = out + srcRow[dx] * scale;
                    }
                }
                *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(out);
            }
            return;
        }

        const float scale
            = 1.f / (fminf(scale_x, src.width(batch_idx) - fsx1) * fminf(scale_y, src.height(batch_idx) - fsy1));

        for (int plane = 0; plane < channels; ++plane)
        {
            work_type out      = {0};
            int4      srcCoord = {0, 0, plane, batch_idx};

            for (int dy = sy1; dy < sy2; ++dy)
            {
                srcCoord.y = dy;
                // Interior columns [sx1,sx2) are in-bounds for a downscale: read the row directly
                // (hoisted) instead of the per-tap border-wrap; fractional edges keep brd_src. Bit-exact.
                const T *srcRow = src.ptr(batch_idx, plane, dy, 0);
                for (int dx = sx1; dx < sx2; ++dx)
                {
                    out = out + srcRow[dx] * scale;
                }
                if (sx1 > fsx1)
                {
                    srcCoord.x = sx1 - 1;
                    out        = out + brd_src[srcCoord] * ((sx1 - fsx1) * scale);
                }
                if (sx2 < fsx2)
                {
                    srcCoord.x = sx2;
                    out        = out + brd_src[srcCoord] * ((fsx2 - sx2) * scale);
                }
            }
            if (sy1 > fsy1)
            {
                srcCoord.y = sy1 - 1;
                for (int dx = sx1; dx < sx2; ++dx)
                {
                    srcCoord.x = dx;
                    out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * scale);
                }
            }
            if (sy2 < fsy2)
            {
                srcCoord.y = sy2;
                for (int dx = sx1; dx < sx2; ++dx)
                {
                    srcCoord.x = dx;
                    out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * scale);
                }
            }
            if ((sy1 > fsy1) && (sx1 > fsx1))
            {
                srcCoord.y = (sy1 - 1);
                srcCoord.x = (sx1 - 1);
                out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * (sx1 - fsx1) * scale);
            }
            if ((sy1 > fsy1) && (sx2 < fsx2))
            {
                srcCoord.y = (sy1 - 1);
                srcCoord.x = sx2;
                out        = out + brd_src[srcCoord] * ((sy1 - fsy1) * (fsx2 - sx2) * scale);
            }
            if ((sy2 < fsy2) && (sx2 < fsx2))
            {
                srcCoord.y = sy2;
                srcCoord.x = sx2;
                out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * (fsx2 - sx2) * scale);
            }
            if ((sy2 < fsy2) && (sx1 > fsx1))
            {
                srcCoord.y = sy2;
                srcCoord.x = sx1 - 1;
                out        = out + brd_src[srcCoord] * ((fsy2 - sy2) * (sx1 - fsx1) * scale);
            }
            *dst.ptr(batch_idx, plane, y, x) = cuda::SaturateCast<T>(out);
        }
        return;
    }

    // zoom in, it is emulated using some variant of bilinear interpolation
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(y * scale_y);
    float fy = (y + 1) - (sy + 1) * inv_scale_y;
    fy       = fy <= 0 ? 0.f : fy - cuda::round<cuda::RoundMode::DOWN, int>(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = cuda::round<cuda::RoundMode::DOWN, int>(x * scale_x);
    float fx = (x + 1) - (sx + 1) * inv_scale_x;
    fx       = fx < 0 ? 0.f : fx - cuda::round<cuda::RoundMode::DOWN, int>(fx);

    if (sx < 0)
    {
        fx = 0, sx = 0;
    }
    if (sx >= src.width(batch_idx) - 1)
    {
        fx = 0, sx = src.width(batch_idx) - 2;
    }
    if (sy >= src.height(batch_idx) - 1)
    {
        sy = src.height(batch_idx) - 2;
    }

    float cbufx[2];
    cbufx[0] = 1.f - fx;
    cbufx[1] = fx;

    for (int plane = 0; plane < channels; ++plane)
    {
        *dst.ptr(batch_idx, plane, y, x)
            = cuda::SaturateCast<T>((*src.ptr(batch_idx, plane, sy, sx) * cbufx[0] * cbufy[0]
                                     + *src.ptr(batch_idx, plane, sy + 1, sx) * cbufx[0] * cbufy[1]
                                     + *src.ptr(batch_idx, plane, sy, sx + 1) * cbufx[1] * cbufy[0]
                                     + *src.ptr(batch_idx, plane, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
    }
}

// Host run resize var-shape functions ------------------------------------------

template<typename T>
void RunResizeVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &out, const int interpolation, cudaStream_t stream,
                       VarShapeScale batchScale)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    nvcv::Size2D outMaxSize = out.maxSize();

    //Performance degrades above 256 and below 16 (GMEM speed limited)
    const int THREADS_PER_BLOCK = kVarShapeThreadsPerBlock;
    int       BLOCK_WIDTH       = 8; //as in 32x4 or 32x8 or 8x32.

    // A warp row spans BLOCK_WIDTH * sizeof(T) bytes of the output. At width 8 the vectorized
    // interleaved types (uchar3/uchar4/float/float3/float4) already cover a 32-byte L1 sector, but a
    // 1-byte single-channel element (U8) covers only 8 bytes/row -- the bilinear/nearest kernels are
    // then L1/TEX-throughput bound (~85% L1 SOL) on those tiny, poorly-coalesced loads. Widen the
    // block to a full sector (32x8) for that case; the per-thread work is unchanged, so the result is
    // bit-exact. Wider types keep width 8 (widening only shrinks their grid), so their launch is
    // byte-identical to before.
    if (sizeof(T) == 1)
        BLOCK_WIDTH = 32;

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
    {
        // Each thread handles NN_NIX grid-strided columns, hoisting the scales and source row; the
        // wide float3/float4 elements (already near the bandwidth ridge) keep one column.
        constexpr int NN_NIX = (sizeof(T) <= 4) ? 4 : 1;
        const dim3    nnGrid(util::DivUp(outMaxSize.w, blockSize.x * NN_NIX), util::DivUp(outMaxSize.h, blockSize.y),
                             in.numImages());
        resize_NN<NN_NIX, T><<<nnGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        break;
    }

    case NVCV_INTERP_LINEAR:
    {
        // Exact-2x upscale fast path for byte types: dispatched only when the caller verified from
        // the host-side image handles that every image pair in the batch is exactly 2x on both axes
        // (the exported imageList is device memory, so it cannot be checked here).
        if constexpr (sizeof(cuda::BaseType<T>) == 1)
        {
            if (batchScale == VarShapeScale::kExpand2x)
            {
                const dim3 e2Block(32, 4, 1);
                const dim3 e2Grid(util::DivUp(outMaxSize.w, e2Block.x * NIX<T>),
                                  util::DivUp(outMaxSize.h, e2Block.y * 2), in.numImages());
                resize_bilinear_expand2x<T><<<e2Grid, e2Block, 0, stream>>>(src_ptr, dst_ptr);
                break;
            }
            // Generic byte path: consecutive columns with float-cached taps and pack stores (see
            // resize_bilinear_pack); bit-exact with resize_bilinear at every scale.
            const dim3 pkBlock(32, 4, 1);
            const dim3 pkGrid(util::DivUp(outMaxSize.w, pkBlock.x * NIX<T>), util::DivUp(outMaxSize.h, pkBlock.y),
                              in.numImages());
            resize_bilinear_pack<T><<<pkGrid, pkBlock, 0, stream>>>(src_ptr, dst_ptr);
            break;
        }
        // Each thread handles BILINEAR_NIX grid-strided columns, hoisting the row-invariant scales,
        // y-terms, and row pointers; shrink grid.x accordingly. Multi-column only helps elements up
        // to 4 bytes (U8/uchar3/uchar4/float), which were L1/compute co-limited; the wide float3/
        // float4 elements are already near the memory-bandwidth ridge (~89% BWUtil) and lose ~3%
        // from the added register pressure, so they keep one column (NIX=1, original behavior).
        constexpr int BILINEAR_NIX = (sizeof(T) <= 4) ? 4 : 1;
        const dim3    linGrid(util::DivUp(outMaxSize.w, blockSize.x * BILINEAR_NIX),
                              util::DivUp(outMaxSize.h, blockSize.y), in.numImages());
        resize_bilinear<BILINEAR_NIX, T><<<linGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        break;
    }

    case NVCV_INTERP_CUBIC:
    {
        // Wide-float (float3/float4) take the shared-memory-tiled kernel: upscale blocks serve the
        // 16-tap L1/TEX-bound gather from a staged source tile (per-block EXPAND gate, others fall
        // back to the direct gather identical to resize_bicubic NIX=1). Tile bound = block + cubic
        // support, valid for any upscale scale. Other dtypes keep the NIX grid-strided kernel.
        constexpr bool isWideFloat = (sizeof(T) > 4) && (sizeof(T) == sizeof(cuda::ConvertBaseTypeTo<float, T>));
        if constexpr (isWideFloat)
        {
            const dim3   scBlock(32, 8, 1);
            const dim3   scGrid(util::DivUp(outMaxSize.w, scBlock.x), util::DivUp(outMaxSize.h, scBlock.y),
                                in.numImages());
            const int    tileW = scBlock.x + 6, tileH = scBlock.y + 6;
            const size_t smemBytes = (size_t)tileW * tileH * sizeof(T);
            resize_bicubic_smem<T><<<scGrid, scBlock, smemBytes, stream>>>(src_ptr, dst_ptr, tileW, tileH);
        }
        else
        {
            // Each thread handles BICUBIC_NIX grid-strided columns, hoisting the scales, y cubic
            // coefficients, and the four source-row pointers (16 taps/pixel).
            constexpr int BICUBIC_NIX = (sizeof(T) <= 4) ? 4 : 1;
            const dim3    cubicGrid(util::DivUp(outMaxSize.w, blockSize.x * BICUBIC_NIX),
                                    util::DivUp(outMaxSize.h, blockSize.y), in.numImages());
            resize_bicubic<BICUBIC_NIX, T><<<cubicGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        break;
    }

    case NVCV_INTERP_AREA:
    {
        // Exact-2x downscale fast path, dispatched only when the caller verified every image pair
        // in the batch is exactly 2x on both axes.
        if (batchScale == VarShapeScale::kContract2x)
        {
            const dim3 c2Grid(util::DivUp(outMaxSize.w, blockSize.x * NIX<T>), util::DivUp(outMaxSize.h, blockSize.y),
                              in.numImages());
            resize_area_contract2x<T><<<c2Grid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
            break;
        }
        // Fractional zoom-out fast path (see resize_area_fractional_pack); every tap is in bounds,
        // so the border wrap is not needed.
        if (batchScale == VarShapeScale::kFractionalZoomOut)
        {
            const dim3 fpGrid(util::DivUp(outMaxSize.w, blockSize.x * NIX<T>), util::DivUp(outMaxSize.h, blockSize.y),
                              in.numImages());
            if constexpr (kPinVarShapeAreaOccupancy<T>)
                resize_area_fractional_pack_pinned<T><<<fpGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
            else
                resize_area_fractional_pack<T><<<fpGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
            break;
        }
        const dim3 gridSize(util::DivUp(outMaxSize.w, blockSize.x), util::DivUp(outMaxSize.h, blockSize.y),
                            in.numImages());

        cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brdSrc(in);
        resize_area_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, brdSrc, dst_ptr);
        break;
    }

    } //switch interpolation

    // cudaGetLastError() is sticky until read: one check here covers every launch above.
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
void RunResizeVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &out, const int channels,
                             const int interpolation, cudaStream_t stream, VarShapeScale batchScale)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    nvcv::Size2D outMaxSize = out.maxSize();

    const int THREADS_PER_BLOCK = kVarShapeThreadsPerBlock;

    // The planar kernels run one thread per output pixel and loop the channel planes, so grid.z spans
    // images. Size the block width so a warp row spans ~one 32-byte sector of a single (scalar) plane:
    // width = 32 / sizeof(element). The default 8-wide block underuses L1/TEX for 8-bit planes (only
    // 8 bytes/row), while wider-than-a-sector just shrinks the grid for wide elements; the interleaved
    // path's vectorized type already fills a sector at width 8.
    int planarWidth = 32 / static_cast<int>(sizeof(T));
    if (planarWidth < 1)
        planarWidth = 1;
    const dim3 planarBlock(planarWidth, THREADS_PER_BLOCK / planarWidth, 1);
    const dim3 planarGrid(util::DivUp(outMaxSize.w, planarBlock.x), util::DivUp(outMaxSize.h, planarBlock.y),
                          in.numImages());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        resize_NN_planar<T><<<planarGrid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
        break;

    case NVCV_INTERP_LINEAR:
    {
        // Vectorize only narrow (1-byte-base, i.e. uint8) planar elements: wide-float planar (NIX=1
        // here) keeps the scalar path, which avoids the CONTRACT regression seen when vectorizing it
        // (float3 CONTRACT +21%). PNIX=1 is the original one-column behaviour.
        constexpr int PNIX = (sizeof(T) == 1) ? 4 : 1;
        const dim3    pbGrid(util::DivUp(outMaxSize.w, planarBlock.x * PNIX), util::DivUp(outMaxSize.h, planarBlock.y),
                             in.numImages());
        resize_bilinear_planar<PNIX, T><<<pbGrid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
    }
    break;

    case NVCV_INTERP_CUBIC:
        resize_bicubic_planar<T><<<planarGrid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
        break;

    case NVCV_INTERP_AREA:
    {
        // Exact-2x downscale fast path (see resizeAreaContract2xPlane); the channel loop stays
        // per-thread like the general planar kernel.
        if (batchScale == VarShapeScale::kContract2x)
        {
            const dim3 c2Grid(util::DivUp(outMaxSize.w, planarBlock.x * NIX<T>),
                              util::DivUp(outMaxSize.h, planarBlock.y), in.numImages());
            resize_area_contract2x_planar<T><<<c2Grid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
            break;
        }
        // Fractional zoom-out fast path with cross-channel geometry amortization (see
        // resize_area_fractional_pack_planar).
        if (batchScale == VarShapeScale::kFractionalZoomOut)
        {
            const dim3 fpGrid(util::DivUp(outMaxSize.w, planarBlock.x * NIX<T>),
                              util::DivUp(outMaxSize.h, planarBlock.y), in.numImages());
            resize_area_fractional_pack_planar<T><<<fpGrid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
            break;
        }
        cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brdSrc(in);
        resize_area_ocv_align_planar<T><<<planarGrid, planarBlock, 0, stream>>>(src_ptr, brdSrc, dst_ptr, channels);
        break;
    }

    } //switch interpolation

    // cudaGetLastError() is sticky until read: one check here covers every launch above.
    NVCV_CHECK_THROW(cudaGetLastError());
}

// VarShape validation and dispatch --------------------------------------------

// Batch layout, mirroring the legacy helpers::GetLegacyDataFormat classification (including its
// batched/unbatched distinction, which is what made a numImages mismatch surface as a format
// mismatch) so the input-vs-output comparison rejects exactly what the legacy path rejected.
enum class VarShapeLayout
{
    kHWC,
    kNHWC,
    kCHW,
    kNCHW
};

// Both legacy classifiers opened with this check, so it stays at the head of each of their
// replacements: that keeps the message and the order in which it fires unchanged, including for any
// future caller that reaches one classifier without the other.
inline void RequireUniformPlaneDataType(nvcv::ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }
}

inline VarShapeLayout ClassifyVarShapeLayout(const nvcv::ImageBatchVarShapeDataStridedCuda &data)
{
    nvcv::ImageFormat fmt = data.uniqueFormat();

    RequireUniformPlaneDataType(fmt);

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar images must have one channel per plane");
        }
        return data.numImages() >= 2 ? VarShapeLayout::kNCHW : VarShapeLayout::kCHW;
    }
    return data.numImages() >= 2 ? VarShapeLayout::kNHWC : VarShapeLayout::kHWC;
}

// Element classification, mirroring the legacy helpers::GetLegacyDataType: the decision is on
// bits-per-channel plus data kind, so packed 2U8/3U8/4U8 all resolve to 8-bit unsigned exactly as
// the legacy kCV_8U row did. Non-uniform channel widths (e.g. NVCV_PACKING_X32_Y24b8) are rejected
// first, then every (kind, width) pair the legacy enum had no name for -- 32-/64-bit unsigned,
// 64-bit signed, and the complex/unspecified kinds -- is rejected here, before the operator's own
// supported-type gate is consulted.
struct VarShapeDataType
{
    nvcv::DataKind kind;
    int32_t        bits;
};

inline VarShapeDataType ClassifyVarShapeDataType(nvcv::ImageFormat fmt)
{
    RequireUniformPlaneDataType(fmt);

    nvcv::DataType dtype = fmt.planeDataType(0);
    auto           bpc   = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    const int32_t bits = bpc[0];

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::FLOAT:
        if (bits != 16 && bits != 32 && bits != 64)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ",
                                  bits);
        }
        break;

    case nvcv::DataKind::SIGNED:
        if (bits != 8 && bits != 16 && bits != 32)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ",
                                  bits);
        }
        break;

    case nvcv::DataKind::UNSIGNED:
        if (bits != 8 && bits != 16)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ",
                                  bits);
        }
        break;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Only floating-point, signed integer and unsigned integer data kinds are supported ");
    }

    return {dtype.dataKind(), bits};
}

// BT is the element's base type; the interleaved kernels bake the channel count into it. Planar
// batches are one scalar plane per channel, so the scalar specialization alone covers any channel
// count.
template<typename BT>
inline void RunResizeVarShapeChannels(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &outData, int channels,
                                      bool isPlanar, const NVCVInterpolationType interpolation, cudaStream_t stream,
                                      VarShapeScale batchScale)
{
    if (isPlanar)
    {
        RunResizeVarShapePlanar<BT>(inData, outData, channels, interpolation, stream, batchScale);
        return;
    }

    switch (channels)
    {
    case 1:
        RunResizeVarShape<BT>(inData, outData, interpolation, stream, batchScale);
        return;
    case 3:
        RunResizeVarShape<cuda::MakeType<BT, 3>>(inData, outData, interpolation, stream, batchScale);
        return;
    case 4:
        RunResizeVarShape<cuda::MakeType<BT, 4>>(inData, outData, interpolation, stream, batchScale);
        return;
    }

    // 2-channel interleaved has no kernel: the legacy dispatch table left that slot null and called
    // through it (its guarding assert() is compiled out in release builds). Reject it instead of
    // reproducing that undefined behavior.
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
}

} // anonymous namespace

namespace cvcuda::priv {

void Resize::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Resize::operator()[Tensor]");
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

    RunResize(stream, *inData, *outData, interpolation);
}

// VarShape operator -----------------------------------------------------------

void Resize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Resize::operator()[ImageBatchVarShape]");
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        for (const auto &img : in)
        {
            auto sz = img.size();
            if (sz.w < 2 || sz.h < 2)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Linear interpolation requires source dimensions of at least 2x2");
            }
        }
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

    // The uniform-scale fast paths need per-image sizes, which are only host-accessible through
    // the batch handles (the exported imageList is device memory). The fractional zoom-out check
    // replicates the kernel's float scale/is_area_fast computation so dispatch and per-image
    // branch agree exactly.
    const bool sameCount      = in.numImages() == out.numImages() && in.numImages() > 0;
    bool       allExpand2x    = sameCount;
    bool       allContract2x  = sameCount;
    bool       allFracZoomOut = sameCount;
    for (int32_t i = 0; i < in.numImages(); ++i)
    {
        if (!(allExpand2x || allContract2x || allFracZoomOut))
        {
            break;
        }
        auto srcSize  = in[i].size();
        auto dstSize  = out[i].size();
        allExpand2x   = allExpand2x && dstSize.w == 2 * srcSize.w && dstSize.h == 2 * srcSize.h;
        allContract2x = allContract2x && srcSize.w == 2 * dstSize.w && srcSize.h == 2 * dstSize.h;

        const float scaleX     = static_cast<float>(srcSize.w) / static_cast<float>(dstSize.w);
        const float scaleY     = static_cast<float>(srcSize.h) / static_cast<float>(dstSize.h);
        const bool  isAreaFast = std::abs(scaleX - static_cast<float>(static_cast<int>(scaleX))) < DBL_EPSILON
                             && std::abs(scaleY - static_cast<float>(static_cast<int>(scaleY))) < DBL_EPSILON;
        allFracZoomOut = allFracZoomOut && scaleX >= 1.f && scaleY >= 1.f && scaleX < 3.f && !isAreaFast;
    }

    VarShapeScale batchScale = VarShapeScale::kGeneric;
    if (allExpand2x)
        batchScale = VarShapeScale::kExpand2x;
    else if (allContract2x)
        batchScale = VarShapeScale::kContract2x;
    else if (allFracZoomOut)
        batchScale = VarShapeScale::kFractionalZoomOut;

    if (!inData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in input batch must all have the same format ");
    }
    if (!outData->uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in output batch must all have the same format ");
    }

    const VarShapeLayout inLayout  = ClassifyVarShapeLayout(*inData);
    const VarShapeLayout outLayout = ClassifyVarShapeLayout(*outData);

    if (inLayout != outLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }

    const bool isPlanar = (inLayout == VarShapeLayout::kNCHW || inLayout == VarShapeLayout::kCHW);

    const int channels = inData->uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format, and the
    // interleaved path likewise does not support 2 channels (matches the Normalize operator).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    // The planar kernels loop the channel planes per thread, so the launched grid.z dimension is
    // numImages, not numImages*channels. Enforce only the CUDA 65535 grid-z limit on numImages.
    // See TestOpResize, varshape_planar_grid_z_is_images_not_planes.
    if (isPlanar && inData->numImages() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar resize requires numImages <= 65535 (CUDA grid-z limit)");
    }

    const VarShapeDataType dataType = ClassifyVarShapeDataType(inData->uniqueFormat());

    const bool supportedType
        = (dataType.kind == nvcv::DataKind::UNSIGNED && (dataType.bits == 8 || dataType.bits == 16))
       || (dataType.kind == nvcv::DataKind::SIGNED && dataType.bits == 16)
       || (dataType.kind == nvcv::DataKind::FLOAT && (dataType.bits == 32 || dataType.bits == 16));

    if (!supportedType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC || interpolation == NVCV_INTERP_AREA))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation %d", (int)interpolation);
    }

    // supportedType above already narrowed dataType to exactly these five (kind, bits) pairs, so the
    // rows below are exhaustive and the last one is F16. F16 instantiates real half kernels: the
    // interpolation paths accumulate in FP32 (ConvertBaseTypeTo<float, T>) and round to half once via
    // SaturateCast on store, rather than aliasing the 16-bit integer kernels.

    // clang-format off
#define CVCUDA_RUN_RESIZE_VS(BT) \
    RunResizeVarShapeChannels<BT>(*inData, *outData, channels, isPlanar, interpolation, stream, batchScale);

    if (dataType.kind == nvcv::DataKind::UNSIGNED && dataType.bits == 8)  CVCUDA_RUN_RESIZE_VS(unsigned char)
    else if (dataType.kind == nvcv::DataKind::UNSIGNED)                   CVCUDA_RUN_RESIZE_VS(ushort)
    else if (dataType.kind == nvcv::DataKind::SIGNED)                     CVCUDA_RUN_RESIZE_VS(short)
    else if (dataType.bits == 32)                                         CVCUDA_RUN_RESIZE_VS(float)
    else                                                                  CVCUDA_RUN_RESIZE_VS(__half)

#undef CVCUDA_RUN_RESIZE_VS
    // clang-format on
}

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
