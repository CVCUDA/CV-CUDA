/* Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/Compat.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

namespace {

#define MAX_BUFFER_BYTES_VS 128 //multiple of 4 for word-aligned read, multiple of 16 for cacheline alignment (float4)
#define MAX_BUFFER_WORDS_VS (MAX_BUFFER_BYTES_VS / 4) //extra bytes for cache alignment

#define LEGACY_BICUBIC_MATH_VS //apparently the legacy code has an abs() that needs to be matched

// Replaced below 15 to 0 due to a reported regression
#define CACHE_MEMORY_ALIGNMENT_VS 0 //this is 'M' for _cacheAlignedBufferedReadVS

//legal values for CACHE_MEMORY_ALIGNMENT_VS are:
// 31: 256-bit alignment
// 15: 128-bit alignment <-- should be ideal for Ampere
//  7:  64-bit alignment
//  3:  32-bit alignment (word)
//  0:  disable buffering
template<typename T, size_t M>
inline __device__ T *_cacheAlignedBufferedReadVS(cuda::ImageBatchVarShapeWrap<const T> srcImage, int width,
                                                 uint *pReadBuffer, uint nReadBufferWordsMax, int nBatch, int nYPos,
                                                 int nXPosMin, int nXPosMax)
{
    const T *lineStartPtr = srcImage.ptr(nBatch, nYPos, 0); //do not access prior to this address
    const T *pixSrcPtr    = &lineStartPtr[nXPosMin];
    if (M == 0)
        return (T *)pixSrcPtr; //return GMEM pointer instead
    else
    {
        uint     *memSrcPtr       = (uint *)(((size_t)pixSrcPtr) & (~M)); //(M+1) byte alignment
        const T  *pixBeyondPtr    = &lineStartPtr[nXPosMax + 1];
        const int functionalWidth = ((size_t)pixBeyondPtr + M) & (~M) - ((size_t)lineStartPtr);
        const int nWordsToRead    = (((size_t)pixBeyondPtr + M) & (~M) - (size_t)memSrcPtr) / 4;

        if (((size_t)memSrcPtr < (size_t)lineStartPtr) || (width * sizeof(T) < functionalWidth)
            || (nWordsToRead > nReadBufferWordsMax))
            return (T *)pixSrcPtr; //return GMEM pointer instead if running off the image
        else
        {                                             //copy out source data, aligned based upon M (31, 15, 7, 3)
            const int skew = ((size_t)pixSrcPtr) & M; //byte offset for nXPosMin
            int       i    = 0;
            if (M >= 31) //256-bit align, 32 bytes at a time
                for (; i < nWordsToRead; i += 8)
                    *((double4_16a *)(&pReadBuffer[i])) = *((double4_16a *)(&memSrcPtr[i]));
            if (M == 15) //128-bit align, 16 bytes at a time
                for (; i < nWordsToRead; i += 4) *((float4 *)(&pReadBuffer[i])) = *((float4 *)(&memSrcPtr[i]));
            if (M == 7) //64-bit align, 8 bytes at a time
                for (; i < nWordsToRead; i += 2) *((float2 *)(&pReadBuffer[i])) = *((float2 *)(&memSrcPtr[i]));
            //32-bit align, 4 bytes at a time
            for (; i < nWordsToRead; ++i) pReadBuffer[i] = memSrcPtr[i];

            return (T *)(((size_t)pReadBuffer) + skew); //buffered pixel data
        }
    }
} //_cacheAlignedBufferedReadVS

// Local pack-write helpers for the exact-2x bilinear kernel (same pattern as OpResize.cu and
// random_resized_crop_common.cuh): NIX consecutive elements of T stored as one vector pack.
template<typename T>
using RVS_DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;

template<typename T>
constexpr int RVS_NIX = sizeof(RVS_DPT<T>) / sizeof(T);

template<typename T>
constexpr unsigned int RVS_MSK = (sizeof(RVS_DPT<T>) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(RVS_DPT<T>)) - 1;

template<typename T>
__device__ __forceinline__ void RVSWritePack(T &u, const T (&v)[RVS_NIX<T>])
{
    reinterpret_cast<RVS_DPT<T> &>(u) = reinterpret_cast<const RVS_DPT<T> &>(v);
}

template<typename T>
__device__ __forceinline__ bool RVSCheckRowAlign(T *row)
{
    return (static_cast<unsigned int>(reinterpret_cast<size_t>(row)) & RVS_MSK<T>) == 0;
}

// The interleaved (one plane, channels baked into T) and planar (one scalar plane per channel)
// var-shape paths share identical per-pixel math; the only difference is which (image, plane) a
// thread maps to. Each interpolation has its work factored into a plane-aware __device__ body that
// both kernels call: the interleaved kernel passes plane 0 and the planar kernel decodes the plane
// from grid-z. With __forceinline__ and ptr(s, 0, y, x) == ptr(s, y, x), the interleaved codegen is
// unchanged.

//******************** NN = Nearest Neighbor

// Interleaved var-shape nearest-neighbour. Each thread owns NN_NIX grid-strided output columns of
// one destination row and hoists the row-invariant terms -- the source scales (two FP divisions),
// the source row index sy, and the source row pointer -- computing them once and reusing them across
// the columns; the original kernel recomputed all of that per output pixel. Columns are grid-strided
// by blockDim.x so each store step stays coalesced. The per-pixel index math is byte-identical, so
// the result is bit-exact. The shared planar body (resize_NN_planar) is unaffected.
template<int NN_NIX, typename T>
__global__ void resize_NN(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = get_batch_idx();
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
    const int batch_idx = get_batch_idx();
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
// fx/fy resize_bilinear derives), so each thread owns RVS_NIX output columns x 2 output rows,
// stages the shared clamped 3-row source window in registers, and writes two vector packs instead
// of RVS_NIX*2 single-byte stores (the general kernel is issue/L1-bound on those: SM 85%, IPC 3.0,
// DRAM 30% for U8). Border outputs replicate resize_bilinear's weight overrides (w=0/w=1), which
// zero exactly the taps whose window values differ from clamped-coordinate loads, and the blend
// uses resize_bilinear's exact expression tree, so the result is bit-exact. Dispatched only when
// the host verifies every image pair in the batch is exactly 2x on both axes.
template<typename T>
__global__ void resize_bilinear_expand2x(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    constexpr int KC = RVS_NIX<T> / 2; // source columns owned per thread
    constexpr int WC = KC + 2;         // staged window columns (taps k0-1 .. k0+KC)

    const int batch_idx = get_batch_idx();
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

    // Clamped source window shared by all 2*RVS_NIX outputs.
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

    T out0[RVS_NIX<T>], out1[RVS_NIX<T>];
#pragma unroll
    for (int i = 0; i < RVS_NIX<T>; ++i)
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

    const int validCount = cuda::min(RVS_NIX<T>, dstWidth - dst_x0);
    // dstHeight = 2*height is even, so the odd output row always exists.
#pragma unroll
    for (int p = 0; p < 2; ++p)
    {
        T *dstRow                  = dst.ptr(batch_idx, dst_y0 + p, 0);
        const T(&outp)[RVS_NIX<T>] = p == 0 ? out0 : out1;

        if (validCount == RVS_NIX<T> && RVSCheckRowAlign(dstRow + dst_x0)) // uniform across the warp
            RVSWritePack(dstRow[dst_x0], outp);
        else
        {
            T *dstPtr = dstRow + dst_x0;
#pragma unroll
            for (int c = 0; c < RVS_NIX<T>; ++c)
                if (c < validCount)
                    dstPtr[c] = outp[c];
        }
    }
}

// Interleaved var-shape bilinear for byte types with RVS_NIX consecutive output columns per
// thread: upscales revisit the same source cell for several adjacent outputs, so the four taps are
// cached as work-type floats (float(byte) conversion is exact) and refreshed only when the source
// column advances -- the grid-strided general kernel re-reads and re-converts them per output. The
// consecutive columns also allow one vector-pack store per row segment instead of RVS_NIX scalar
// stores. Coordinate math, border overrides, and the blend expression tree are identical to
// resize_bilinear, so the result is bit-exact at every scale; downscales simply never hit the
// cache-reuse fast case.
template<typename T>
__global__ void resize_bilinear_pack(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    using FT = cuda::ConvertBaseTypeTo<float, T>;

    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * RVS_NIX<T>;

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

    T out[RVS_NIX<T>];
#pragma unroll
    for (int i = 0; i < RVS_NIX<T>; ++i)
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
    const int validCount = cuda::min(RVS_NIX<T>, dstWidth - x0);

    if (validCount == RVS_NIX<T> && RVSCheckRowAlign(dstRow + x0))
        RVSWritePack(dstRow[x0], out);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int c = 0; c < RVS_NIX<T>; ++c)
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
// kernel, so the result is bit-exact. The shared planar body (resize_bicubic_planar) is unaffected.
template<int BICUBIC_NIX, typename T>
__global__ void resize_bicubic(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = get_batch_idx();
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

    const int   b     = get_batch_idx();
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
    resizeAreaPlane<T>(src, brd_src, dst, get_batch_idx(), 0, x, y);
}

// Exact-2x AREA downscale body shared by the interleaved and planar kernels: every output is the
// mean of a fully-in-bounds 2x2 source box (sx1 = 2x, sy1 = 2y, scale = 1/(2*2) = 0.25f -- the same
// values resizeAreaPlane's integer branch derives), so the per-pixel coordinate rounding and the
// runtime-bounded box loops collapse to four unrolled taps. Each thread emits RVS_NIX consecutive
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
    const int validCount = cuda::min(RVS_NIX<T>, dstWidth - x0);

    // Each thread consumes 2*RVS_NIX consecutive source elements per row; read them as two vector
    // packs per row instead of 4*RVS_NIX single-element loads -- the scalar-load version of this
    // body was L1/TEX-transaction-bound (99% L1 SOL, DRAM 70%, SM 29%). The staged values are
    // identical, so the accumulation below is unchanged.
    T win0[2 * RVS_NIX<T>], win1[2 * RVS_NIX<T>];
    if (validCount == RVS_NIX<T> && RVSCheckRowAlign(row0 + sx0) && RVSCheckRowAlign(row1 + sx0))
    {
#pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            reinterpret_cast<RVS_DPT<T> *>(win0)[p] = reinterpret_cast<const RVS_DPT<T> *>(row0 + sx0)[p];
            reinterpret_cast<RVS_DPT<T> *>(win1)[p] = reinterpret_cast<const RVS_DPT<T> *>(row1 + sx0)[p];
        }
    }
    else
    {
        // Source width is exactly 2*dstWidth, so tail loads stay guarded per element.
#pragma unroll
        for (int c = 0; c < 2 * RVS_NIX<T>; ++c)
        {
            const int sx = sx0 + c;
            win0[c]      = (sx < 2 * dstWidth) ? row0[sx] : T{};
            win1[c]      = (sx < 2 * dstWidth) ? row1[sx] : T{};
        }
    }

    T out[RVS_NIX<T>];
#pragma unroll
    for (int i = 0; i < RVS_NIX<T>; ++i)
    {
        work_type acc = {0};
        acc           = acc + win0[2 * i] * 0.25f;
        acc           = acc + win0[2 * i + 1] * 0.25f;
        acc           = acc + win1[2 * i] * 0.25f;
        acc           = acc + win1[2 * i + 1] * 0.25f;
        out[i]        = cuda::SaturateCast<T>(acc);
    }

    T *dstRow = dst.ptr(batch_idx, plane, y, 0);

    if (validCount == RVS_NIX<T> && RVSCheckRowAlign(dstRow + x0))
        RVSWritePack(dstRow[x0], out);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int c = 0; c < RVS_NIX<T>; ++c)
            if (c < validCount)
                dstPtr[c] = out[c];
    }
}

template<typename T>
__global__ void resize_area_contract2x(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int batch_idx = get_batch_idx();
    const int x0        = (blockDim.x * blockIdx.x + threadIdx.x) * RVS_NIX<T>;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    resizeAreaContract2xPlane<T>(src, dst, batch_idx, 0, x0, y, dst.width(batch_idx));
}

template<typename T>
__global__ void resize_area_contract2x_planar(cuda::ImageBatchVarShapeWrap<const T> src,
                                              cuda::ImageBatchVarShapeWrap<T> dst, int channels)
{
    const int batch_idx = get_batch_idx();
    const int x0        = (blockDim.x * blockIdx.x + threadIdx.x) * RVS_NIX<T>;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;

    if (x0 >= dst.width(batch_idx) || y >= dst.height(batch_idx))
        return;

    for (int plane = 0; plane < channels; ++plane)
    {
        resizeAreaContract2xPlane<T>(src, dst, batch_idx, plane, x0, y, dst.width(batch_idx));
    }
}

// Interleaved var-shape fractional AREA downscale with RVS_NIX consecutive output columns per
// thread. For a zoom-out every box tap -- including the fractional edge/corner taps -- is in
// bounds (sx1-1 >= floor(fsx1) >= 0 and sx2 <= width-1, likewise in y), so the general body's
// per-tap border-wrap accesses are replaced by plain hoisted row-pointer reads (value-identical),
// the row-invariant y box terms are computed once per thread, and the RVS_NIX results leave as one
// vector pack instead of scalar stores (the general kernel is issue/L1-bound: SM 75%, IPC 2.8,
// L1 83%, DRAM 34% local). Per-output term order -- interior rows (cols ascending, then left,
// then right edge), top row, bottom row, then TL/TR/BR/BL corners, each * scale -- replicates
// resizeAreaPlane exactly, so the result is bit-exact. Integer ratios and zoom-ins keep their
// existing kernels.
template<typename T>
__global__ void resize_area_fractional_pack(cuda::ImageBatchVarShapeWrap<const T> src,
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
    constexpr int GN    = (cuda::NumElements<T> == 3) ? 2 : 4;
    constexpr int kMaxC = 3;

    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * RVS_NIX<T>;

    if (y >= dstHeight || x0 >= dstWidth)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    const float fsy1 = y * scale_y;
    const float fsy2 = fsy1 + scale_y;
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Skip edge taps that float rounding pushes one past the source extent: the border-wrapped
    // kernel reads the zero constant border there (contribution exactly +0.0), so skipping is
    // bit-identical and keeps the direct reads in bounds.
    const bool  hasTop = (float)sy1 > fsy1;
    const bool  hasBot = (float)sy2 < fsy2 && sy2 < height;
    const float wTop   = (float)sy1 - fsy1;
    const float wBot   = fsy2 - (float)sy2;

    T out[RVS_NIX<T>];

#pragma unroll
    for (int g = 0; g < RVS_NIX<T>; g += GN)
    {
        float fsx1[GN], fsx2[GN], scaleA[GN], wL[GN], wR[GN];
        int   sx1[GN], sx2[GN], nC[GN];
        bool  hasL[GN], hasR[GN];
#pragma unroll
        for (int j = 0; j < GN; ++j)
        {
            const int x = cuda::min(x0 + g + j, dstWidth - 1);
            fsx1[j]     = x * scale_x;
            fsx2[j]     = fsx1[j] + scale_x;
            sx1[j]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[j]);
            sx2[j]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[j]);
            nC[j]       = sx2[j] - sx1[j];
            hasL[j]     = (float)sx1[j] > fsx1[j];
            hasR[j]     = (float)sx2[j] < fsx2[j] && sx2[j] < width;
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
    const int validCount = cuda::min(RVS_NIX<T>, dstWidth - x0);

    if (validCount == RVS_NIX<T> && RVSCheckRowAlign(dstRow + x0))
        RVSWritePack(dstRow[x0], out);
    else
    {
        T *dstPtr = dstRow + x0;
#pragma unroll
        for (int c = 0; c < RVS_NIX<T>; ++c)
            if (c < validCount)
                dstPtr[c] = out[c];
    }
}

template<typename T>
void resize(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
            const int interpolation, cudaStream_t stream, ResizeVarShapeScale batchScale)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    Size2D outMaxSize = out.maxSize();

    const int THREADS_PER_BLOCK = 256; //Performance degrades above 256 and below 16 (GMEM speed limited)
    int       BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8 or 8x32.

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
    const dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    //quad permits aligned writes to output image, if image is multiple of 4.  kernels in resize_varshape are smart
    const int  out_quad_width = outMaxSize.w / 4;
    const dim3 quadGridSize(divUp(out_quad_width, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
    {
        // Each thread handles NN_NIX grid-strided columns, hoisting the scales and source row; the
        // wide float3/float4 elements (already near the bandwidth ridge) keep one column.
        constexpr int NN_NIX = (sizeof(T) <= 4) ? 4 : 1;
        const dim3 nnGrid(divUp(outMaxSize.w, blockSize.x * NN_NIX), divUp(outMaxSize.h, blockSize.y), in.numImages());
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
            if (batchScale == ResizeVarShapeScale::kExpand2x)
            {
                const dim3 e2Block(32, 4, 1);
                const dim3 e2Grid(divUp(outMaxSize.w, e2Block.x * RVS_NIX<T>), divUp(outMaxSize.h, e2Block.y * 2),
                                  in.numImages());
                resize_bilinear_expand2x<T><<<e2Grid, e2Block, 0, stream>>>(src_ptr, dst_ptr);
                break;
            }
            // Generic byte path: consecutive columns with float-cached taps and pack stores (see
            // resize_bilinear_pack); bit-exact with resize_bilinear at every scale.
            const dim3 pkBlock(32, 4, 1);
            const dim3 pkGrid(divUp(outMaxSize.w, pkBlock.x * RVS_NIX<T>), divUp(outMaxSize.h, pkBlock.y),
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
        const dim3    linGrid(divUp(outMaxSize.w, blockSize.x * BILINEAR_NIX), divUp(outMaxSize.h, blockSize.y),
                              in.numImages());
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
            const dim3   scGrid(divUp(outMaxSize.w, scBlock.x), divUp(outMaxSize.h, scBlock.y), in.numImages());
            const int    tileW = scBlock.x + 6, tileH = scBlock.y + 6;
            const size_t smemBytes = (size_t)tileW * tileH * sizeof(T);
            resize_bicubic_smem<T><<<scGrid, scBlock, smemBytes, stream>>>(src_ptr, dst_ptr, tileW, tileH);
        }
        else
        {
            // Each thread handles BICUBIC_NIX grid-strided columns, hoisting the scales, y cubic
            // coefficients, and the four source-row pointers (16 taps/pixel).
            constexpr int BICUBIC_NIX = (sizeof(T) <= 4) ? 4 : 1;
            const dim3    cubicGrid(divUp(outMaxSize.w, blockSize.x * BICUBIC_NIX), divUp(outMaxSize.h, blockSize.y),
                                    in.numImages());
            resize_bicubic<BICUBIC_NIX, T><<<cubicGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        break;
    }

    case NVCV_INTERP_AREA:
    {
        // Exact-2x downscale fast path, dispatched only when the caller verified every image pair
        // in the batch is exactly 2x on both axes.
        if (batchScale == ResizeVarShapeScale::kContract2x)
        {
            const dim3 c2Grid(divUp(outMaxSize.w, blockSize.x * RVS_NIX<T>), divUp(outMaxSize.h, blockSize.y),
                              in.numImages());
            resize_area_contract2x<T><<<c2Grid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
            break;
        }
        // Fractional zoom-out fast path (see resize_area_fractional_pack); every tap is in bounds,
        // so the border wrap is not needed.
        if (batchScale == ResizeVarShapeScale::kFractionalZoomOut)
        {
            const dim3 fpGrid(divUp(outMaxSize.w, blockSize.x * RVS_NIX<T>), divUp(outMaxSize.h, blockSize.y),
                              in.numImages());
            resize_area_fractional_pack<T><<<fpGrid, blockSize, 0, stream>>>(src_ptr, dst_ptr);
            break;
        }
        cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brdSrc(in);
        resize_area_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, brdSrc, dst_ptr);
        break;
    }

    } //switch interpolation
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
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
    const int batch_idx = get_batch_idx();

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

// Planar bilinear, vectorized: each thread owns NIX grid-strided output columns of one row. The
// row-invariant y terms (scale, fy, sy) are hoisted once, and the per-column x terms (fx, sx) are
// precomputed once and reused across every channel plane -- the scalar kernel recomputed the source
// scales (two FP divisions) and x/y coordinates per output pixel. Each plane's two row pointers are
// resolved once. The per-pixel arithmetic is byte-identical, so the result is bit-exact.
template<int NIX, typename T>
__global__ void resize_bilinear_planar(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                       int channels)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

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

    const int xBase = blockIdx.x * blockDim.x * NIX + threadIdx.x;

    // Per-column x terms, shared across all channel planes.
    int   dx[NIX], sxA[NIX];
    float fxA[NIX];
#pragma unroll
    for (int i = 0; i < NIX; ++i)
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
        for (int i = 0; i < NIX; ++i)
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
    const int batch_idx = get_batch_idx();

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

// Planar var-shape fractional AREA downscale: one thread owns RVS_NIX consecutive output columns
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

    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * RVS_NIX<T>;

    if (y >= dstHeight || x0 >= dstWidth)
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    const float fsy1 = y * scale_y;
    const float fsy2 = fsy1 + scale_y;
    const int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
    const int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

    // Per-column x box terms, shared by every channel plane. Wide elements (RVS_NIX == 4) hoist
    // them into arrays; byte planes (RVS_NIX == 16) recompute them inline per plane instead --
    // sixteen hoisted term sets spill to local memory and regressed the byte rows 1.6x.
    constexpr int  XN      = (RVS_NIX<T> <= 4) ? RVS_NIX<T> : 1;
    constexpr bool HOISTED = XN == RVS_NIX<T>;
    float          fsx1[XN], fsx2[XN], scale[XN];
    int            sx1[XN], sx2[XN];
    if constexpr (HOISTED)
    {
#pragma unroll
        for (int i = 0; i < XN; ++i)
        {
            const int x = cuda::min(x0 + i, dstWidth - 1);
            fsx1[i]     = x * scale_x;
            fsx2[i]     = fsx1[i] + scale_x;
            sx1[i]      = cuda::round<cuda::RoundMode::UP, int>(fsx1[i]);
            sx2[i]      = cuda::round<cuda::RoundMode::DOWN, int>(fsx2[i]);
            scale[i]    = 1.f / (fminf(scale_x, width - fsx1[i]) * fminf(scale_y, height - fsy1));
        }
    }

    const int validCount = cuda::min(RVS_NIX<T>, dstWidth - x0);

    for (int plane = 0; plane < channels; ++plane)
    {
        T out[RVS_NIX<T>];
#pragma unroll
        for (int i = 0; i < RVS_NIX<T>; ++i)
        {
            if (i >= validCount)
                break;

            const int ti = HOISTED ? i : 0;
            if constexpr (!HOISTED)
            {
                const int x = cuda::min(x0 + i, dstWidth - 1);
                fsx1[0]     = x * scale_x;
                fsx2[0]     = fsx1[0] + scale_x;
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
                if (sx2[ti] < fsx2[ti] && sx2[ti] < width)
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
            if (sy2 < fsy2 && sy2 < height)
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
            if ((sy1 > fsy1) && (sx2[ti] < fsx2[ti] && sx2[ti] < width))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy1 - 1, 0)[sx2[ti]]
                          * ((sy1 - fsy1) * (fsx2[ti] - sx2[ti]) * scale[ti]);
            }
            if ((sy2 < fsy2 && sy2 < height) && (sx2[ti] < fsx2[ti] && sx2[ti] < width))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy2, 0)[sx2[ti]] * ((fsy2 - sy2) * (fsx2[ti] - sx2[ti]) * scale[ti]);
            }
            if ((sy2 < fsy2 && sy2 < height) && (sx1[ti] > fsx1[ti]))
            {
                acc = acc
                    + src.ptr(batch_idx, plane, sy2, 0)[sx1[ti] - 1]
                          * ((fsy2 - sy2) * (sx1[ti] - fsx1[ti]) * scale[ti]);
            }

            out[i] = cuda::SaturateCast<T>(acc);
        }

        T *dstRow = dst.ptr(batch_idx, plane, y, 0);

        if (validCount == RVS_NIX<T> && RVSCheckRowAlign(dstRow + x0))
            RVSWritePack(dstRow[x0], out);
        else
        {
            T *dstPtr = dstRow + x0;
#pragma unroll
            for (int c = 0; c < RVS_NIX<T>; ++c)
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
    const int batch_idx = get_batch_idx();

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

template<typename T>
void resize_planar(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                   const int channels, const int interpolation, cudaStream_t stream, ResizeVarShapeScale batchScale)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    Size2D outMaxSize = out.maxSize();

    const int THREADS_PER_BLOCK = 256;

    // The planar kernels run one thread per output pixel and loop the channel planes, so grid.z spans
    // images. Size the block width so a warp row spans ~one 32-byte sector of a single (scalar) plane:
    // width = 32 / sizeof(element). The default 8-wide block underuses L1/TEX for 8-bit planes (only
    // 8 bytes/row), while wider-than-a-sector just shrinks the grid for wide elements; the interleaved
    // path's vectorized type already fills a sector at width 8.
    int planarWidth = 32 / static_cast<int>(sizeof(T));
    if (planarWidth < 1)
        planarWidth = 1;
    const dim3 planarBlock(planarWidth, THREADS_PER_BLOCK / planarWidth, 1);
    const dim3 planarGrid(divUp(outMaxSize.w, planarBlock.x), divUp(outMaxSize.h, planarBlock.y), in.numImages());

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
        const dim3    pbGrid(divUp(outMaxSize.w, planarBlock.x * PNIX), divUp(outMaxSize.h, planarBlock.y),
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
        if (batchScale == ResizeVarShapeScale::kContract2x)
        {
            const dim3 c2Grid(divUp(outMaxSize.w, planarBlock.x * RVS_NIX<T>), divUp(outMaxSize.h, planarBlock.y),
                              in.numImages());
            resize_area_contract2x_planar<T><<<c2Grid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
            break;
        }
        // Fractional zoom-out fast path with cross-channel geometry amortization (see
        // resize_area_fractional_pack_planar).
        if (batchScale == ResizeVarShapeScale::kFractionalZoomOut)
        {
            const dim3 fpGrid(divUp(outMaxSize.w, planarBlock.x * RVS_NIX<T>), divUp(outMaxSize.h, planarBlock.y),
                              in.numImages());
            resize_area_fractional_pack_planar<T><<<fpGrid, planarBlock, 0, stream>>>(src_ptr, dst_ptr, channels);
            break;
        }
        cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brdSrc(in);
        resize_area_ocv_align_planar<T><<<planarGrid, planarBlock, 0, stream>>>(src_ptr, brdSrc, dst_ptr, channels);
        break;
    }

    } //switch interpolation
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

} // namespace

ErrorCode ResizeVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream,
                                ResizeVarShapeScale batchScale)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in input batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in output batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inData.uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format, and the
    // interleaved path likewise does not support 2 channels (matches the Normalize operator).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // The planar path launches one grid-z slice per (image, plane), so numImages * channels must
    // fit CUDA's 65535 grid-z limit. Compute in 64-bit to avoid overflow before the comparison.
    if (isPlanar && static_cast<int64_t>(inData.numImages()) * channels > 65535)
    {
        LOG_ERROR("Planar resize requires numImages * channels <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC || interpolation == NVCV_INTERP_AREA))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: each channel is resized as a separate
        // single-channel plane, so one scalar specialization per dtype covers any channel count.
        typedef void (*planar_func_t)(const ImageBatchVarShapeDataStridedCuda &in,
                                      const ImageBatchVarShapeDataStridedCuda &out, const int channels,
                                      const int interpolation, cudaStream_t stream, ResizeVarShapeScale batchScale);

        static const planar_func_t planar_funcs[6] = {
            resize_planar<uchar>, 0 /*schar*/, resize_planar<ushort>,
            resize_planar<short>, 0 /*int*/,   resize_planar<float>,
        };

        const planar_func_t planar_func = planar_funcs[data_type];
        NVCV_ASSERT(planar_func != 0);
        planar_func(inData, outData, channels, interpolation, stream, batchScale);
        return ErrorCode::SUCCESS;
    }

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
                           const int interpolation, cudaStream_t stream, ResizeVarShapeScale batchScale);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,      resize<uchar3>,      resize<uchar4>},
        {0 /*resize<schar>*/,   0 /*resize<char2>*/, 0 /*resize<char3>*/, 0 /*resize<char4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,     resize<ushort3>,     resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,      resize<short3>,      resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,  0 /*resize<int3>*/,  0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,      resize<float3>,      resize<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];

    assert(func != 0);
    func(inData, outData, interpolation, stream, batchScale);
    return ErrorCode::SUCCESS;
} // namespace

} // namespace nvcv::legacy::cuda_op
