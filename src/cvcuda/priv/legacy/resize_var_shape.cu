/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/SaturateCast.hpp>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

namespace {

#define MAX_BUFFER_BYTES_VS 128 //multiple of 4 for word-aligned read, multiple of 16 for cacheline alignment (float4)
#define MAX_BUFFER_WORDS_VS (MAX_BUFFER_BYTES_VS / 4) //extra bytes for cache alignment

#define LEGACY_BICUBIC_MATH_VS //apparently the legacy code has an abs() that needs to be matched

#define CACHE_MEMORY_ALIGNMENT_VS 15 //this is 'M' for _cacheAlignedBufferedReadVS

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
                for (; i < nWordsToRead; i += 8) *((double4 *)(&pReadBuffer[i])) = *((double4 *)(&memSrcPtr[i]));
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

template<typename T>
inline void __device__ _alignedCudaMemcpyQuadVS(T *pDst, T *pSrc)
{
    //copy 4 T's, assuming 32-bit alignment for both pSrc and pDst
    uint *uPtrSrc = (uint *)pSrc;
    uint *uPtrDst = (uint *)pDst;

#pragma unroll
    for (int i = 0; i < sizeof(T); ++i) uPtrDst[i] = uPtrSrc[i];

} //_alignedCudaMemcpyQuadVS

//******************** NN = Nearest Neighbor

template<typename T>
__global__ void resize_NN(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);

    if ((dst_x < dstWidth) && (dst_y < dstHeight))
    { //generic copy pixel to pixel
        const int width  = src.width(batch_idx);
        const int height = src.height(batch_idx);

        const float scale_x               = static_cast<float>(width) / dstWidth;
        const float scale_y               = static_cast<float>(height) / dstHeight;
        const int   sx                    = cuda::min(__float2int_rd(dst_x * scale_x), width - 1);
        const int   sy                    = cuda::min(__float2int_rd(dst_y * scale_y), height - 1);
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
} //resize_NN

template<typename T>
__global__ void resize_NN_quad_combo(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const float MAX_BUFFERED_X_SCALE = 4.0f; //probably more efficient all the way up to 4.0

    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * 4; //quad
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstWidth4 = dstWidth & ~3;
    const int dstHeight = dst.height(batch_idx);

    //0 - bail if out-of-range
    if ((dst_x >= dstWidth) | (dst_y >= dstHeight))
        return;

    const int   width   = src.width(batch_idx);
    const int   height  = src.height(batch_idx);
    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;
    const int   sy      = cuda::min(__float2int_rd(dst_y * scale_y), height - 1);

    if (dstWidth != dstWidth4) //non-aligned case, up to 4 pixels
    {                          //do up to 4 pixels, unoptimized
        const int pixels = cuda::min(dstWidth - dst_x, 4);
        for (int i = 0; i < pixels; ++i)
        {
            const int sxi                         = cuda::min(__float2int_rd((dst_x + i) * scale_x), width - 1);
            *dst.ptr(batch_idx, dst_y, dst_x + i) = *src.ptr(batch_idx, sy, sxi);
        }
    }
    else //quad-case: memory is aligned, do 4 pixels
    {
        const int sx0 = cuda::min(__float2int_rd(dst_x * scale_x), width - 1);
        const int sx1 = cuda::min(__float2int_rd(dst_x * scale_x + scale_x), width - 1);
        const int sx2 = cuda::min(__float2int_rd((dst_x + 2) * scale_x), width - 1);
        const int sx3 = cuda::min(__float2int_rd((dst_x + 3) * scale_x), width - 1);

        //1 - optimized case if scale_x < some finite limit
        if ((scale_x <= MAX_BUFFERED_X_SCALE)) //local buffering is more efficient
        {
            uint readBuffer[MAX_BUFFER_WORDS_VS];

            //2 - copy out source data, 32-bit aligned aligned
            T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                src, width, &readBuffer[0], MAX_BUFFER_WORDS_VS, batch_idx, sy, sx0, sx3);

            //3 - NN sampling
            T gather[4] = {aPtr[0], aPtr[sx1 - sx0], aPtr[sx2 - sx0], aPtr[sx3 - sx0]};

            //4 - aligned write back out
            _alignedCudaMemcpyQuadVS<T>(dst.ptr(batch_idx, dst_y, dst_x), gather);
        }
        else //6 - standard sampling, no optimization
        {
            //sample all 4 points

            const T *aPtr = src.ptr(batch_idx, sy, sx0);

            T gather[4] = {aPtr[0], aPtr[sx1 - sx0], aPtr[sx2 - sx0], aPtr[sx3 - sx0]};

            _alignedCudaMemcpyQuadVS<T>(dst.ptr(batch_idx, dst_y, dst_x), gather);
        }
    }
} //resize_NN_quad_combo

//******************** Bilinear

template<typename T>
__global__ void resize_bilinear(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
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

        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int   sy = __float2int_rd(fy);
        fy -= sy;
        sy = cuda::max(0, cuda::min(sy, height - 2));

        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        { //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
            int   sx = __float2int_rd(fx);
            fx -= sx;
            fx *= ((sx >= 0) && (sx < width - 1));
            sx = cuda::max(0, cuda::min(sx, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
} //resize_bilinear

template<typename T>
__global__ void resize_bilinear_quad_combo(cuda::ImageBatchVarShapeWrap<const T> src,
                                           cuda::ImageBatchVarShapeWrap<T>       dst)
{
    const float MAX_BUFFERED_X_SCALE = 4.0f; //probably more efficient all the way up to 4.0

    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * 4; //quad
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstWidth4 = dstWidth & ~3;
    const int dstHeight = dst.height(batch_idx);

    //0 - if one pixel is out, they're all out
    if ((dst_x >= dstWidth) | (dst_y >= dstHeight))
        return;

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    //y coordinate math is the same for all points
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = __float2int_rd(fy);
    fy -= sy;
    sy = cuda::max(0, cuda::min(sy, height - 2));

    if (dstWidth != dstWidth4) //non-aligned case, up to 4 pixels
    {
        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        const int pixels = cuda::min(dstWidth - dst_x, 4);
        for (int i = 0; i < pixels; ++i)
        { //compute source data position and weight for [xi] components
            float fxi = (float)((dst_x + 0.5f + i) * scale_x - 0.5f);
            int   sxi = __float2int_rd(fxi);
            fxi -= sxi;
            fxi *= ((sxi >= 0) && (sxi < width - 1));
            sxi = cuda::max(0, cuda::min(sxi, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x + i)
                = cuda::SaturateCast<T>((1.0f - fxi) * (aPtr[sxi] * (1.0f - fy) + bPtr[sxi] * fy)
                                        + fxi * (aPtr[sxi + 1] * (1.0f - fy) + bPtr[sxi + 1] * fy));
        }
    }
    else //quad-aligned case, 4 pixels
    {
        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        //sx0
        float fx0 = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx0 = __float2int_rd(fx0);
        fx0 -= sx0;
        fx0 *= ((sx0 >= 0) && (sx0 < width - 1));
        sx0 = cuda::max(0, cuda::min(sx0, width - 2));

        //sx1
        float fx1 = (float)((dst_x + 1.5) * scale_x - 0.5f);
        int   sx1 = __float2int_rd(fx1);
        fx1 -= sx1;
        fx1 *= ((sx1 >= 0) && (sx1 < width - 1));
        sx1 = cuda::max(0, cuda::min(sx1, width - 2));

        //sx2
        float fx2 = (float)((dst_x + 2.5f) * scale_x - 0.5f);
        int   sx2 = __float2int_rd(fx2);
        fx2 -= sx2;
        fx2 *= ((sx2 >= 0) && (sx2 < width - 1));
        sx2 = cuda::max(0, cuda::min(sx2, width - 2));

        //sx3
        float fx3 = (float)((dst_x + 3.5f) * scale_x - 0.5f);
        int   sx3 = __float2int_rd(fx3);
        fx3 -= sx3;
        fx3 *= ((sx3 >= 0) && (sx3 < width - 1));
        sx3 = cuda::max(0, cuda::min(sx3, width - 2));

        uint readBuffer[MAX_BUFFER_WORDS_VS];

        T result[4];

        //1 - optimized case if scale_x < some finite limit
        if (scale_x <= MAX_BUFFERED_X_SCALE) //local buffering is more efficient
        {
            work_type accum[4];

            //2 - aligned load a-row and add partial product
            T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy, sx0, sx3 + 1);
            //const T * aPtr = src.ptr(batch_idx, sy,   sx0); //start of upper row

            accum[0] = (1.0f - fy) * (aPtr[sx0 - sx0] * (1.0f - fx0) + aPtr[sx0 - sx0 + 1] * fx0);
            accum[1] = (1.0f - fy) * (aPtr[sx1 - sx0] * (1.0f - fx1) + aPtr[sx1 - sx0 + 1] * fx1);
            accum[2] = (1.0f - fy) * (aPtr[sx2 - sx0] * (1.0f - fx2) + aPtr[sx2 - sx0 + 1] * fx2);
            accum[3] = (1.0f - fy) * (aPtr[sx3 - sx0] * (1.0f - fx3) + aPtr[sx3 - sx0 + 1] * fx3);

            //3 - aligned load b-row and add remaining partial product
            T *bPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy + 1, sx0, sx3 + 1);
            //const T * bPtr = src.ptr(batch_idx, sy+1, sx0); //start of lower row

            //$$$ only need to cast, not saturatecast
            result[0]
                = cuda::SaturateCast<T>(accum[0] + fy * (bPtr[sx0 - sx0] * (1.0f - fx0) + bPtr[sx0 - sx0 + 1] * fx0));
            result[1]
                = cuda::SaturateCast<T>(accum[1] + fy * (bPtr[sx1 - sx0] * (1.0f - fx1) + bPtr[sx1 - sx0 + 1] * fx1));
            result[2]
                = cuda::SaturateCast<T>(accum[2] + fy * (bPtr[sx2 - sx0] * (1.0f - fx2) + bPtr[sx2 - sx0 + 1] * fx2));
            result[3]
                = cuda::SaturateCast<T>(accum[3] + fy * (bPtr[sx3 - sx0] * (1.0f - fx3) + bPtr[sx3 - sx0 + 1] * fx3));
        }
        else //unbuffered
        {
            //row pointers
            const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
            const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

            //$$$ only need to cast, not saturatecast
            result[0] = cuda::SaturateCast<T>(aPtr[sx0] * (1.0f - fx0) * (1.0f - fy) + bPtr[sx0] * (1.0f - fx0) * fy
                                              + aPtr[sx0 + 1] * fx0 * (1.0f - fy) + bPtr[sx0 + 1] * fx0 * fy);

            result[1] = cuda::SaturateCast<T>(aPtr[sx1] * (1.0f - fx1) * (1.0f - fy) + bPtr[sx1] * (1.0f - fx1) * fy
                                              + aPtr[sx1 + 1] * fx1 * (1.0f - fy) + bPtr[sx1 + 1] * fx1 * fy);

            result[2] = cuda::SaturateCast<T>(aPtr[sx2] * (1.0f - fx2) * (1.0f - fy) + bPtr[sx2] * (1.0f - fx2) * fy
                                              + aPtr[sx2 + 1] * fx2 * (1.0f - fy) + bPtr[sx2 + 1] * fx2 * fy);

            result[3] = cuda::SaturateCast<T>(aPtr[sx3] * (1.0f - fx3) * (1.0f - fy) + bPtr[sx3] * (1.0f - fx3) * fy
                                              + aPtr[sx3 + 1] * fx3 * (1.0f - fy) + bPtr[sx3 + 1] * fx3 * fy);
        }

        //aligned write 4 pixels
        _alignedCudaMemcpyQuadVS<T>(dst.ptr(batch_idx, dst_y, dst_x), result);
    }
} //resize_bilinear_quad_combo

//******************** Bicubic

template<typename T>
__global__ void resize_bicubic(cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{ //optimized for aligned read
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

        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        uint readBuffer[MAX_BUFFER_WORDS_VS];

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int   sy = __float2int_rd(fy);
        fy -= sy;
        sy = cuda::max(1, cuda::min(sy, height - 3));

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        work_type accum = cuda::SetAll<work_type>(0);

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = __float2int_rd(fx);
        fx -= sx;
        fx *= ((sx >= 1) && (sx < width - 3));
        sx = cuda::max(1, cuda::min(sx, width - 3));

        float cX[4];
        cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
        cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
        cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
        cX[3] = 1.0f - cX[0] - cX[1] - cX[2];
#pragma unroll
        for (int row = 0; row < 4; ++row)
        {
            //1 - load each sub row from sx-1 to sx+3 inclusive, aligned
            //const T * aPtr = src.ptr(batch_idx, sy + row - 1, sx-1);
            T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy + row - 1, sx - 1, sx + 2);

            //2 - do a pixel's partial on this row
            accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
        } //for row
#ifndef LEGACY_BICUBIC_MATH_VS
        //correct math
        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
#else
        //abs() needed to match legacy operator.
        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(cuda::abs(accum));
#endif
    }
} //resize_bicubic

template<typename T>
__global__ void resize_bicubic_quad_combo(cuda::ImageBatchVarShapeWrap<const T> src,
                                          cuda::ImageBatchVarShapeWrap<T>       dst)
{                                            //optimized for aligned read and write, plus buffering
    const float MAX_BUFFERED_X_SCALE = 4.0f; //probably more efficient all the way up to 4.0

    const int dst_x     = (blockIdx.x * blockDim.x + threadIdx.x) * 4; //quad
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int dstWidth  = dst.width(batch_idx);
    const int dstWidth4 = dstWidth & ~3;
    const int dstHeight = dst.height(batch_idx);

    //0 - quad-aligned so if one pixel is out, they're all out
    if ((dst_x >= dstWidth) | (dst_y >= dstHeight))
        return;

    uint readBuffer[MAX_BUFFER_WORDS_VS];
    T    result[4];

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    const float scale_x = static_cast<float>(width) / dstWidth;
    const float scale_y = static_cast<float>(height) / dstHeight;

    //float space for weighted addition
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    //y coordinate
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int   sy = __float2int_rd(fy);
    fy -= sy;
    sy = cuda::max(1, cuda::min(sy, height - 3));

    const float A = -0.75f;

    float cY[4];
    cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
    cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
    cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
    cY[3] = 1.f - cY[0] - cY[1] - cY[2];

    if (dstWidth != dstWidth4) //non-aligned case, up to 4 pixels
    {
        uint readBuffer[MAX_BUFFER_WORDS_VS];

        const int pixels = cuda::min(dstWidth - dst_x, 4);
        for (int i = 0; i < pixels; ++i)
        {
            float fxi = (float)((dst_x + 0.5f + i) * scale_x - 0.5f);
            int   sxi = __float2int_rd(fxi);
            fxi -= sxi;
            fxi *= ((sxi >= 1) && (sxi < width - 3));
            sxi = cuda::max(1, cuda::min(sxi, width - 3));

            float cX[4];
            cX[0] = ((A * (fxi + 1.0f) - 5.0f * A) * (fxi + 1.0f) + 8.0f * A) * (fxi + 1.0f) - 4.0f * A;
            cX[1] = ((A + 2.0f) * fxi - (A + 3.0f)) * fxi * fxi + 1.0f;
            cX[2] = ((A + 2.0f) * (1.0f - fxi) - (A + 3.0f)) * (1.0f - fxi) * (1.0f - fxi) + 1.0f;
            cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

            work_type accum = cuda::SetAll<work_type>(0);
#pragma unroll
            for (int row = 0; row < 4; ++row)
            {
                //1 - load each sub row from sx-1 to sx+3 inclusive, aligned
                //const T * aPtr = src.ptr(batch_idx, sy + row - 1, sx-1);
                T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                    src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy + row - 1, sxi - 1, sxi + 2);

                //2 - do a pixel's partial on this row
                accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
            } //for row
#ifndef LEGACY_BICUBIC_MATH_VS
            //correct math
            *dst.ptr(batch_idx, dst_y, dst_x + i) = cuda::SaturateCast<T>(accum);
#else
            //abs() needed to match legacy operator.
            *dst.ptr(batch_idx, dst_y, dst_x + i) = cuda::SaturateCast<T>(cuda::abs(accum));
#endif
        } //for pixels
    }
    else //quad-aligned case, 4 pixels
    {
        //1 - optimized case if scale_x < some finite limit
        if (scale_x <= MAX_BUFFERED_X_SCALE) //local buffering
        {                                    //buffered read

            work_type accum[4];
            float     fx[4];
            int       sx[4];
            float     cX[4][4];

            //initialize data for each pixel position
#pragma unroll
            for (int pix = 0; pix < 4; ++pix)
            {
                accum[pix] = cuda::SetAll<work_type>(0);

                //1 - precalc sx's ahead of time to get range from sx0-1..sx3+2
                fx[pix] = (float)((dst_x + pix + 0.5f) * scale_x - 0.5f);
                sx[pix] = __float2int_rd(fx[pix]);
                fx[pix] -= sx[pix];
                fx[pix] *= ((sx[pix] >= 1) && (sx[pix] < width - 3));
                sx[pix] = cuda::max(1, cuda::min(sx[pix], width - 3));

                //2 - precalc cX[][] 2D array
                cX[pix][0]
                    = ((A * (fx[pix] + 1.0f) - 5.0f * A) * (fx[pix] + 1.0f) + 8.0f * A) * (fx[pix] + 1.0f) - 4.0f * A;
                cX[pix][1] = ((A + 2.0f) * fx[pix] - (A + 3.0f)) * fx[pix] * fx[pix] + 1.0f;
                cX[pix][2] = ((A + 2.0f) * (1.0f - fx[pix]) - (A + 3.0f)) * (1.0f - fx[pix]) * (1.0f - fx[pix]) + 1.0f;
                cX[pix][3] = 1.0f - cX[pix][0] - cX[pix][1] - cX[pix][2];
            }
            const int rowOffset = sx[0] - 1;

            //contribute each row into 4 pixels
#pragma unroll
            for (int row = 0; row < 4; ++row)
            {
                //1 - load each row from sx[0]-1 to sx[3]+3 inclusive, aligned
                T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                    src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy + row - 1, sx[0] - 1, sx[3] + 2);

//2 - do each pixel's partial on this row
#pragma unroll
                for (int pix = 0; pix < 4; ++pix)
                {
                    accum[pix]
                        += cY[row]
                         * (cX[row][0] * aPtr[sx[pix] - rowOffset - 1] + cX[row][1] * aPtr[sx[pix] - rowOffset + 0]
                            + cX[row][2] * aPtr[sx[pix] - rowOffset + 1] + cX[row][3] * aPtr[sx[pix] - rowOffset + 2]);
                }
            }

#pragma unroll
            for (int pix = 0; pix < 4; ++pix)
#ifndef LEGACY_BICUBIC_MATH_VS
                result[pix] = cuda::SaturateCast<T>(accum[pix]);
#else
                result[pix] = cuda::SaturateCast<T>(cuda::abs(accum[pix]));
#endif
        }
        else
        { //partially buffered read 4 pixels at a time across each bicubic: 16 coalesced reads instead of 64
#pragma unroll
            for (int pix = 0; pix < 4; ++pix)
            {
                work_type accum = cuda::SetAll<work_type>(0);

                float fx = (float)((dst_x + pix + 0.5f) * scale_x - 0.5f);
                int   sx = __float2int_rd(fx);
                fx -= sx;
                fx *= ((sx >= 1) && (sx < width - 3));
                sx = cuda::max(1, cuda::min(sx, width - 3));

                float cX[4];
                cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
                cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
                cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
                cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

#pragma unroll
                for (int row = 0; row < 4; ++row)
                {
                    //1 - load each sub row from sx[pix]-1 to sx[pix]+2 inclusive, aligned
                    //const T * aPtr = src.ptr(batch_idx, sy + row - 1, sx-1);
                    const T *aPtr = _cacheAlignedBufferedReadVS<T, CACHE_MEMORY_ALIGNMENT_VS>(
                        src, width, readBuffer, MAX_BUFFER_WORDS_VS, batch_idx, sy + row - 1, sx - 1, sx + 2);

                    //2 - do a pixel's partial on this row
                    accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
                } //for row
#ifndef LEGACY_BICUBIC_MATH_VS
                result[pix] = cuda::SaturateCast<T>(accum);
#else
                result[pix] = cuda::SaturateCast<T>(cuda::abs(accum));
#endif
            } //for pix
        }

        //aligned write 4 pixels
        _alignedCudaMemcpyQuadVS<T>(dst.ptr(batch_idx, dst_y, dst_x), result);
    }
} //resize_bicubic_quad_combo

//******************** Integrate area

template<typename T>
__global__ void resize_area_ocv_align(const cuda::ImageBatchVarShapeWrap<const T>                   src,
                                      const cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brd_src,
                                      cuda::ImageBatchVarShapeWrap<T>                               dst)
{
    const int x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int dstWidth  = dst.width(batch_idx);
    int dstHeight = dst.height(batch_idx);

    if (x >= dstWidth || y >= dstHeight)
        return;
    int height = src.height(batch_idx), width = src.width(batch_idx);

    float scale_x = static_cast<float>(width) / dstWidth;
    float scale_y = static_cast<float>(height) / dstHeight;

    double inv_scale_x  = 1. / scale_x;
    double inv_scale_y  = 1. / scale_y;
    int    iscale_x     = cuda::SaturateCast<int>(scale_x);
    int    iscale_y     = cuda::SaturateCast<int>(scale_y);
    bool   is_area_fast = abs(scale_x - iscale_x) < DBL_EPSILON && abs(scale_y - iscale_y) < DBL_EPSILON;

    if (scale_x >= 1.0f && scale_y >= 1.0f) // zoom out
    {
        if (is_area_fast) // integer multiples
        {
            float scale = 1.f / (scale_x * scale_y);
            float fsx1  = x * scale_x;
            float fsx2  = fsx1 + scale_x;

            int sx1 = __float2int_ru(fsx1);
            int sx2 = __float2int_rd(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = __float2int_ru(fsy1);
            int sy2 = __float2int_rd(fsy2);

            using work_type = cuda::ConvertBaseTypeTo<float, T>;
            work_type out   = {0};

            int3 srcCoord = {0, 0, batch_idx};

            for (int dy = sy1; dy < sy2; ++dy)
            {
                srcCoord.y = dy;

                for (int dx = sx1; dx < sx2; ++dx)
                {
                    srcCoord.x = dx;

                    out = out + brd_src[srcCoord] * scale;
                }
            }
            *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(out);
            return;
        }

        float fsx1 = x * scale_x;
        float fsx2 = fsx1 + scale_x;

        int sx1 = __float2int_ru(fsx1);
        int sx2 = __float2int_rd(fsx2);

        float fsy1 = y * scale_y;
        float fsy2 = fsy1 + scale_y;

        int sy1 = __float2int_ru(fsy1);
        int sy2 = __float2int_rd(fsy2);

        float scale
            = 1.f / (fminf(scale_x, src.width(batch_idx) - fsx1) * fminf(scale_y, src.height(batch_idx) - fsy1));

        using work_type = cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = {0};

        int3 srcCoord = {0, 0, batch_idx};

        for (int dy = sy1; dy < sy2; ++dy)
        {
            srcCoord.y = dy;

            for (int dx = sx1; dx < sx2; ++dx)
            {
                srcCoord.x = dx;

                out = out + brd_src[srcCoord] * scale;
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

        *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>(out);
        return;
    }

    // zoom in, it is emulated using some variant of bilinear interpolation
    int   sy = __float2int_rd(y * scale_y);
    float fy = (float)((y + 1) - (sy + 1) * inv_scale_y);
    fy       = fy <= 0 ? 0.f : fy - __float2int_rd(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = __float2int_rd(x * scale_x);
    float fx = (float)((x + 1) - (sx + 1) * inv_scale_x);
    fx       = fx < 0 ? 0.f : fx - __float2int_rd(fx);

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

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>((*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0]
                                                       + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
                                                       + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
                                                       + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<class Filter, typename T>
__global__ void resize_area_v2(const Filter src, cuda_op::Ptr2dVarShapeNHWC<T> dst)
{
    int       dst_x     = blockDim.x * blockIdx.x + threadIdx.x;
    int       dst_y     = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.cols[batch_idx] || dst_y >= dst.rows[batch_idx])
        return;

    *dst.ptr(batch_idx, dst_y, dst_x) = src(batch_idx, dst_y, dst_x);
}

template<typename T>
void resize(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
            const int interpolation, cudaStream_t stream)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    Size2D outMaxSize = out.maxSize();

    bool can_quad = true;
    //bool can_quad = false;  //<-- force single pixel per kernel mode, smaller register file

    const int THREADS_PER_BLOCK = 256; //Performance degrades above 256 and below 16 (GMEM speed limited)
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8 or 8x32.

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(outMaxSize.w, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    //quad permits aligned writes to output image, if image is multiple of 4.  kernels in resize_varshape are smart
    const int  out_quad_width = outMaxSize.w / 4;
    const dim3 quadGridSize(divUp(out_quad_width, blockSize.x), divUp(outMaxSize.h, blockSize.y), in.numImages());

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        if (can_quad)
        { //thread does 4 pixels horizontally for aligned read and write
            resize_NN_quad_combo<T><<<quadGridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        else
        { //generic single pixel per thread case
            resize_NN<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        break;
    case NVCV_INTERP_LINEAR:
        if (can_quad)
        { //thread does 4 pixels horizontally for aligned read and write
            resize_bilinear_quad_combo<T><<<quadGridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        else
        { //generic single pixel per thread case
            resize_bilinear<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        break;
    case NVCV_INTERP_CUBIC:
        if (can_quad)
        { //thread does 4 pixels horizontally for aligned read and write
            resize_bicubic_quad_combo<T><<<quadGridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        else
        { //generic single pixel per thread case
            resize_bicubic<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        }
        break;
    case NVCV_INTERP_AREA:
        cuda::BorderVarShapeWrap<const T, NVCV_BORDER_CONSTANT> brdSrc(in);

        resize_area_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, brdSrc, dst_ptr);

        break;
    } //switch interpolation
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

} // namespace

ErrorCode ResizeVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                const IImageBatchVarShapeDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in input batch must all have the same format ");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
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

    typedef void (*func_t)(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
                           const int interpolation, cudaStream_t stream);

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
    func(inData, outData, interpolation, stream);
    return ErrorCode::SUCCESS;
} // namespace

} // namespace nvcv::legacy::cuda_op
