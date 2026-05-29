/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include <cmath>
#include <random>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

#define BLOCK 32

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
                for (; i < nWordsToRead; i += 8)
                    *((double4_16a *)(&pReadBuffer[i])) = *((double4_16a *)(&memSrcPtr[i]));
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

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_linear_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                 const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);

        fy = ((sy < 0) ? 0 : ((sy > height - 2) ? 1 : fy - sy));
        sy = cuda::max(0, cuda::min(sy, height - 2));

        //row pointers
        const T *aPtr = src.ptr(batch_idx, sy, 0);     //start of upper row
        const T *bPtr = src.ptr(batch_idx, sy + 1, 0); //start of lower row

        { //compute source data position and weight for [x0] components
            float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
            int   sx = cuda::round<cuda::RoundMode::DOWN, int>(fx);

            fx = ((sx < 0) ? 0 : ((sx > width - 2) ? 1 : fx - sx));
            sx = cuda::max(0, cuda::min(sx, width - 2));

            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
                                        + fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
        }
    }
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void resize_nearest_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                  const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) && (dst_y < out_height))
    { //generic copy pixel to pixel
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * scale_x) + left, srcSize.x - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * scale_y) + top, srcSize.y - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
}

template<typename SrcWrapper, typename DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void resize_cubic_v1(const SrcWrapper src, DstWrapper dst, int2 srcSize, int2 dstSize, const int *top_,
                                const int *left_, const float *scale_x_, const float *scale_y_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

    if ((dst_x < out_width) & (dst_y < out_height))
    {
        const float scale_x = scale_x_[batch_idx];
        const float scale_y = scale_y_[batch_idx];
        const int   top     = top_[batch_idx];
        const int   left    = left_[batch_idx];

        //float space for weighted addition
        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        uint readBuffer[MAX_BUFFER_WORDS];

        //y coordinate
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f + top);
        int   sy = cuda::round<cuda::RoundMode::DOWN, int>(fy);
        fy -= sy;
        const int syClamped = cuda::max(1, cuda::min(sy, height - 3));
        fy += static_cast<float>(sy - syClamped);
        sy = syClamped;

        const float A = -0.75f;

        float cY[4];
        cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
        cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
        cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
        cY[3] = 1.f - cY[0] - cY[1] - cY[2];

        work_type accum = cuda::SetAll<work_type>(0);

        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f + left);
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
}

template<typename SrcWrapper, typename DstWrapper>
void resize(const SrcWrapper &src, const DstWrapper &dst, const NVCVInterpolationType interpolation,
            cudaStream_t stream, const int *top, const int *left, const float *scale_x, const float *scale_y,
            int2 srcSize, int2 dstSize, int batchSize)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(dstSize.x, blockSize.x), divUp(dstSize.y, blockSize.y), batchSize);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        resize_linear_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        resize_nearest_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        checkKernelErrors();
    }
    else
    {
        resize_cubic_v1<<<gridSize, blockSize, 0, stream>>>(src, dst, srcSize, dstSize, top, left, scale_x, scale_y);
        checkKernelErrors();
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T>
ErrorCode resize(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                 const NVCVInterpolationType interpolation, cudaStream_t stream, const int *top, const int *left,
                 const float *scale_x, const float *scale_y)
{
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 srcSize{inAccess->numCols(), inAccess->numRows()};
    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};

    int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

    if (std::max(srcMaxStride, dstMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateTensorWrapNHW<T, int32_t>(inData);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
        resize(src, dst, interpolation, stream, top, left, scale_x, scale_y, srcSize, dstSize, batchSize);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

RandomResizedCrop::RandomResizedCrop(DataShape max_input_shape, DataShape max_output_shape, const double min_scale,
                                     const double max_scale, const double min_ratio, const double max_ratio,
                                     int32_t maxBatchSize, uint32_t seed)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , min_scale_(min_scale)
    , max_scale_(max_scale)
    , min_ratio_(min_ratio)
    , max_ratio_(max_ratio)
    , m_maxBatchSize(maxBatchSize)
{
    if (min_scale_ > max_scale_ || min_ratio_ > max_ratio_)
    {
        LOG_ERROR("Invalid Parameter: scale and ratio should be of kind (min, max)");
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameter error!");
    }
    if (maxBatchSize > 0)
    {
        size_t bufferSize = (sizeof(int) * 2 + sizeof(float) * 2) * maxBatchSize;
        NVCV_CHECK_LOG(cudaMalloc(&m_gpuCropParams, bufferSize));
        m_cpuCropParams = malloc(bufferSize);
        if (!m_cpuCropParams)
        {
            LOG_ERROR("Memory allocation error of size: " << bufferSize);
            throw std::runtime_error("Memory allocation error!");
        }
    }
    if (seed == 0)
    {
        std::random_device rand_dev;
        generator_ = std::mt19937(rand_dev());
    }
    else
    {
        generator_ = std::mt19937(seed);
    }
}

RandomResizedCrop::~RandomResizedCrop()
{
    NVCV_CHECK_LOG(cudaFree(m_gpuCropParams));
    free(m_cpuCropParams);
}

size_t RandomResizedCrop::calBufferSize(int batch_size)
{
    // buffer size for batch oftop index, left index, scale y, scale x
    return (sizeof(int) * 2 + sizeof(float) * 2) * batch_size;
}

void RandomResizedCrop::getCropParams(int input_rows, int input_cols, int *top_indices, int *left_indices,
                                      int *crop_rows, int *crop_cols)
{
    int                                    rows          = input_rows;
    int                                    cols          = input_cols;
    double                                 area          = rows * cols;
    const double                           log_min_ratio = std::log(min_ratio_);
    const double                           log_max_ratio = std::log(max_ratio_);
    std::uniform_real_distribution<double> scale_dist(min_scale_, max_scale_);
    std::uniform_real_distribution<double> ratio_dist(log_min_ratio, log_max_ratio);
    bool                                   got_params = false;
    for (int i = 0; i < 10; ++i)
    {
        if (got_params)
            return;
        int    target_area  = area * scale_dist(generator_);
        double aspect_ratio = std::exp(ratio_dist(generator_));

        *crop_cols = int(std::round(std::sqrt(target_area * aspect_ratio)));
        *crop_rows = int(std::round(std::sqrt(target_area / aspect_ratio)));

        if (*crop_cols > 0 && *crop_cols <= cols && *crop_rows > 0 && *crop_rows <= rows)
        {
            std::uniform_int_distribution<int> row_uni(0, rows - *crop_rows);
            std::uniform_int_distribution<int> col_uni(0, cols - *crop_cols);
            *top_indices  = row_uni(generator_);
            *left_indices = col_uni(generator_);
            got_params    = true;
        }
    }
    // Fallback to central crop
    if (!got_params)
    {
        double in_ratio = double(cols) / double(rows);
        if (in_ratio < min_ratio_)
        {
            *crop_cols = cols;
            *crop_rows = int(std::round(*crop_cols / min_ratio_));
        }
        else if (in_ratio > max_ratio_)
        {
            *crop_rows = rows;
            *crop_cols = int(std::round(*crop_rows * max_ratio_));
        }
        else // whole image
        {
            *crop_cols = cols;
            *crop_rows = rows;
        }
        *top_indices  = (rows - *crop_rows) / 2;
        *left_indices = (cols - *crop_cols) / 2;
    }
}

ErrorCode RandomResizedCrop::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    DataType  in_data_type  = helpers::GetLegacyDataType(inData.dtype());
    DataType  out_data_type = helpers::GetLegacyDataType(outData.dtype());
    DataShape input_shape   = helpers::GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(in_data_type == kCV_8U || in_data_type == kCV_16U || in_data_type == kCV_16S || in_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (in_data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << in_data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(interpolation == NVCV_INTERP_LINEAR || interpolation == NVCV_INTERP_NEAREST
          || interpolation == NVCV_INTERP_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    int out_cols = outAccess->numCols();
    int out_rows = outAccess->numRows();

    int batch   = inAccess->numSamples();
    int in_cols = inAccess->numCols();
    int in_rows = inAccess->numRows();

    float *scale_y = (float *)(m_cpuCropParams);
    float *scale_x = (float *)((char *)scale_y + sizeof(float) * batch);
    int   *tops    = (int *)((char *)scale_x + sizeof(float) * batch);
    int   *lefts   = (int *)((char *)tops + sizeof(int) * batch);

    for (int i = 0; i < batch; ++i)
    {
        int top, left, crop_rows, crop_cols;
        getCropParams(in_rows, in_cols, &top, &left, &crop_rows, &crop_cols);
        scale_x[i] = ((float)crop_cols) / out_cols;
        scale_y[i] = ((float)crop_rows) / out_rows;
        tops[i]    = top;
        lefts[i]   = left;
    }

    float *scale_y_gpu = (float *)(m_gpuCropParams);
    float *scale_x_gpu = (float *)((char *)scale_y_gpu + sizeof(float) * batch);
    int   *tops_gpu    = (int *)((char *)scale_x_gpu + sizeof(float) * batch);
    int   *lefts_gpu   = (int *)((char *)tops_gpu + sizeof(int) * batch);

    size_t buffer_size = calBufferSize(batch);
    checkCudaErrors(
        cudaMemcpyAsync((void *)m_gpuCropParams, (void *)m_cpuCropParams, buffer_size, cudaMemcpyHostToDevice, stream));

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const NVCVInterpolationType interpolation, cudaStream_t stream, const int *top,
                                const int *left, const float *scale_x, const float *scale_y);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,      resize<uchar3>,      resize<uchar4>},
        {0 /*resize<schar>*/,   0 /*resize<char2>*/, 0 /*resize<char3>*/, 0 /*resize<char4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,     resize<ushort3>,     resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,      resize<short3>,      resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,  0 /*resize<int3>*/,  0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,      resize<float3>,      resize<float4>}
    };

    const func_t func = funcs[in_data_type][channels - 1];
    return func(inData, outData, interpolation, stream, tops_gpu, lefts_gpu, scale_x_gpu, scale_y_gpu);
}

} // namespace nvcv::legacy::cuda_op
