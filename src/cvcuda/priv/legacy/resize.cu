/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//$$$ replace these with the new (non-legacy) nvcv approach

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/cuda/MathWrappers.hpp>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

//private internal API

#define MAX_BUFFER_BYTES 128 //multiple of 4 for word-aligned read, multiple of 16 for cacheline alignment (float4)
#define MAX_BUFFER_WORDS (MAX_BUFFER_BYTES / 4) //extra bytes for cache alignment

#define LEGACY_BICUBIC_MATH //apparently the legacy code has an abs() that needs to be matched

// Replaced below 15 to 0 due to a reported regression
#define CACHE_MEMORY_ALIGNMENT 0 //this is 'M' for _cacheAlignedBufferedRead

//legal values for CACHE_MEMORY_ALIGNMENT are:
// 31: 256-bit alignment
// 15: 128-bit alignment <-- should be ideal for Ampere
//  7:  64-bit alignment
//  3:  32-bit alignment (word)
//  0:  disable buffering
template<size_t M, class SrcWrapper, typename ValueType = typename SrcWrapper::ValueType>
inline const __device__ ValueType *_cacheAlignedBufferedRead(SrcWrapper srcImage, int width, uint *pReadBuffer,
                                                             uint nReadBufferWordsMax, int nBatch, int nYPos,
                                                             int nXPosMin, int nXPosMax)
{
    const ValueType *lineStartPtr = srcImage.ptr(nBatch, nYPos); //do not access prior to this address
    const ValueType *pixSrcPtr    = &lineStartPtr[nXPosMin];
    if (M == 0)
        return pixSrcPtr; //return GMEM pointer instead
    else
    {
        uint            *memSrcPtr       = (uint *)(((size_t)pixSrcPtr) & (~M)); //(M+1) byte alignment
        const ValueType *pixBeyondPtr    = &lineStartPtr[nXPosMax + 1];
        const int        functionalWidth = ((size_t)pixBeyondPtr + M) & (~M) - ((size_t)lineStartPtr);
        const int        nWordsToRead    = (((size_t)pixBeyondPtr + M) & (~M) - (size_t)memSrcPtr) / 4;

        if (((size_t)memSrcPtr < (size_t)lineStartPtr) || (width * sizeof(ValueType) < functionalWidth)
            || (nWordsToRead > nReadBufferWordsMax))
            return pixSrcPtr; //return GMEM pointer instead if running off the image
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

            return (const ValueType *)(((size_t)pReadBuffer) + skew); //buffered pixel data
        }
    }
} //_cacheAlignedBufferedRead

//******************** NN = Nearest Neighbor

template<class SrcWrapper, class DstWrapper>
__global__ void resize_NN(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const float scale_x,
                          const float scale_y)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    { //generic copy pixel to pixel
        const int sx = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>(dst_x * scale_x), srcSize.x - 1);
        const int sy = cuda::min(cuda::round<cuda::RoundMode::DOWN, int>(dst_y * scale_y), srcSize.y - 1);
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
} //resize_NN

//******************** Bilinear

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_bilinear(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const float scale_x,
                                const float scale_y)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;
        sy = cuda::max(0, cuda::min(sy, height - 2));

        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        { //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
            fx -= sx;
            fx *= ((sx >= 0) && (sx < width - 1));
            sx = cuda::max(0, cuda::min(sx, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
} //resize_bilinear

//******************** Bicubic

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_bicubic(SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const float scale_x,
                               const float scale_y)
{ //optimized for aligned read
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) & (dst_y < out_height))
    {
        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        uint readBuffer[MAX_BUFFER_WORDS];

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
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
        int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);
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
            const T *aPtr = _cacheAlignedBufferedRead<CACHE_MEMORY_ALIGNMENT>(
                src, srcSize.x, readBuffer, MAX_BUFFER_WORDS, batch_idx, sy + row - 1, sx - 1, sx + 2);

            //2 - do a pixel's partial on this row
            accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
        } //for row
#ifndef LEGACY_BICUBIC_MATH
        //correct math
        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
#else
        //abs() needed to match legacy operator.
        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(cuda::abs(accum));
#endif
    }
} //resize_bicubic

template<class SrcWrapper, class DstWrapper>
__global__ void resize_area_ocv_align(SrcWrapper src, DstWrapper dst, int2 dstSize)
{
    const int x          = blockDim.x * blockIdx.x + threadIdx.x;
    const int y          = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dstSize.y, out_width = dstSize.x;

    if (x >= out_width || y >= out_height)
        return;

    const int3 coord{x, y, batch_idx};

    dst[coord] = src[cuda::StaticCast<float>(coord)];
}

template<typename T>
void resize(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
            NVCVInterpolationType interpolation, cudaStream_t stream)

{
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int batch_size = inAccess->numSamples();
    const int in_width   = inAccess->numCols();
    const int in_height  = inAccess->numRows();
    const int out_width  = outAccess->numCols();
    const int out_height = outAccess->numRows();

    float scale_x = ((float)in_width) / out_width;
    float scale_y = ((float)in_height) / out_height;

    int2 srcSize{in_width, in_height};
    int2 dstSize{out_width, out_height};

    auto src = cuda::CreateTensorWrapNHW<const T>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T>(outData);

    const int THREADS_PER_BLOCK = 256; //256?  64?
    const int BLOCK_WIDTH       = 8;   //as in 32x4 or 32x8.  16x8 and 16x16 are also viable

    const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    //Note: resize is fundamentally a gather memory operation, with a little bit of compute
    //      our goals are to (a) maximize throughput, and (b) minimize occupancy for the same performance

    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        resize_NN<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, scale_x, scale_y);
        break;

    case NVCV_INTERP_LINEAR:
        resize_bilinear<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, scale_x, scale_y);
        break;

    case NVCV_INTERP_CUBIC:
        resize_bicubic<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, scale_x, scale_y);
        break;

    case NVCV_INTERP_AREA:
    {
        auto src = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA>(inData, T{},
                                                                                                     scale_x, scale_y);
        auto dst = cuda::CreateTensorWrapNHW<T>(outData);

        resize_area_ocv_align<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
    }
    break;

    default:
        //$$$ need to throw or log an error here
        break;
    } //switch

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
} //resize

ErrorCode Resize::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                        const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef void (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                           const NVCVInterpolationType interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,       resize<uchar3>,       resize<uchar4>},
        {0 /*resize<schar>*/,  0 /*resize<schar2>*/, 0 /*resize<schar3>*/, 0 /*resize<schar4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,      resize<ushort3>,      resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,       resize<short3>,       resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,   0 /*resize<int3>*/,   0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,       resize<float3>,       resize<float4>}
    };

    //note: schar1,3,4 should all work...

    if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
        || interpolation == NVCV_INTERP_CUBIC || interpolation == NVCV_INTERP_AREA)
    {
        const func_t func = funcs[data_type][channels - 1];
        NVCV_ASSERT(func != 0);

        func(inData, outData, interpolation, stream);
    }
    else
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
    return SUCCESS;
} //Resize::infer

} // namespace nvcv::legacy::cuda_op
