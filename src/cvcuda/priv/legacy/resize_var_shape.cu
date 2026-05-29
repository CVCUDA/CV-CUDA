/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        const float scale_x = static_cast<float>(width) / dstWidth;
        const float scale_y = static_cast<float>(height) / dstHeight;
        const int   sx      = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x), width - 1);
        const int   sy      = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y), height - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
} //resize_NN

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
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        { //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

            fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
            sx = cuda::max(0, cuda::min(sx, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
} //resize_bilinear

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
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;
        const int syClamped = cuda::max(1, cuda::min(sy, height - 3));
        fy += static_cast<float>(sy - syClamped); // rebase fractional offset after clamp
        sy = syClamped;

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        work_type accum = cuda::SetAll<work_type>(0);

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
        fx -= sx;
        const int sxClamped = cuda::max(1, cuda::min(sx, width - 3));
        fx += static_cast<float>(sx - sxClamped);
        sx = sxClamped;

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

            int sx1 = cuda::round<cuda::RoundMode::UP, int>(fsx1);
            int sx2 = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);

            float fsy1 = y * scale_y;
            float fsy2 = fsy1 + scale_y;

            int sy1 = cuda::round<cuda::RoundMode::UP, int>(fsy1);
            int sy2 = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

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
    int   sy = cuda::round<cuda::RoundMode::DOWN, int>(y * scale_y);
    float fy = (float)((y + 1) - (sy + 1) * inv_scale_y);
    fy       = fy <= 0 ? 0.f : fy - cuda::round<cuda::RoundMode::DOWN, int>(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = cuda::round<cuda::RoundMode::DOWN, int>(x * scale_x);
    float fx = (float)((x + 1) - (sx + 1) * inv_scale_x);
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

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<T>((*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0]
                                                       + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
                                                       + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
                                                       + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<typename T>
void resize(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
            const int interpolation, cudaStream_t stream)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<T>       dst_ptr(out);

    Size2D outMaxSize = out.maxSize();

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
        resize_NN<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        break;

    case NVCV_INTERP_LINEAR:
        resize_bilinear<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
        break;

    case NVCV_INTERP_CUBIC:
        resize_bicubic<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);
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

ErrorCode ResizeVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                const ImageBatchVarShapeDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream)
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

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
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

    typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &in, const ImageBatchVarShapeDataStridedCuda &out,
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
