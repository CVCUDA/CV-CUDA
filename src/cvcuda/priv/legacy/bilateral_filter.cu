/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

static __device__ __forceinline__ float norm1(const float &a)
{
    return std::abs(a);
}

static __device__ __forceinline__ float norm1(const float2 &a)
{
    return cuda::abs(a.x) + cuda::abs(a.y);
}

static __device__ __forceinline__ float norm1(const float3 &a)
{
    return cuda::abs(a.x) + cuda::abs(a.y) + cuda::abs(a.z);
}

static __device__ __forceinline__ float norm1(const float4 &a)
{
    return cuda::abs(a.x) + cuda::abs(a.y) + cuda::abs(a.z) + cuda::abs(a.w);
}

template<typename T>
struct BilateralFilterPlanarTensorWrap
{
    const NVCVByte *srcBase;
    NVCVByte       *dstBase;
    int64_t         srcSampleStride;
    int64_t         srcChStride;
    int64_t         srcRowStride;
    int64_t         srcColStride;
    int64_t         dstSampleStride;
    int64_t         dstChStride;
    int64_t         dstRowStride;
    int64_t         dstColStride;

    __device__ __forceinline__ float read(int sample, int channel, int y, int x) const
    {
        const NVCVByte *ptr
            = srcBase + sample * srcSampleStride + channel * srcChStride + y * srcRowStride + x * srcColStride;
        return static_cast<float>(*reinterpret_cast<const T *>(ptr));
    }

    __device__ __forceinline__ void write(int sample, int channel, int y, int x, T value) const
    {
        NVCVByte *ptr
            = dstBase + sample * dstSampleStride + channel * dstChStride + y * dstRowStride + x * dstColStride;
        *reinterpret_cast<T *>(ptr) = value;
    }
};

template<NVCVBorderType B, bool FAST_INTERIOR>
__device__ __forceinline__ bool mapBorderCoordinate(int &x, int &y, int columns, int rows, bool windowInside)
{
    if constexpr (B == NVCV_BORDER_CONSTANT)
    {
        return !cuda::IsOutside(x, columns) && !cuda::IsOutside(y, rows);
    }
    else
    {
        if constexpr (FAST_INTERIOR)
        {
            if (windowInside || (!cuda::IsOutside(x, columns) && !cuda::IsOutside(y, rows)))
            {
                return true;
            }
        }
        x = cuda::GetIndexWithBorder<B>(x, columns);
        y = cuda::GetIndexWithBorder<B>(y, rows);
        return true;
    }
}

template<class SrcWrapper>
__device__ __forceinline__ typename SrcWrapper::ValueType readPackedPixel(const SrcWrapper &src, int3 coord, int rows,
                                                                          int columns, bool windowInside)
{
    if constexpr (SrcWrapper::kBorderType != NVCV_BORDER_CONSTANT)
    {
        if (windowInside || (!cuda::IsOutside(coord.x, columns) && !cuda::IsOutside(coord.y, rows)))
        {
            return src.tensorWrap()[coord];
        }
    }
    return src[coord];
}

template<typename T, NVCVBorderType B, int CHANNELS>
__device__ __forceinline__ void BilateralFilterPlanarTile(BilateralFilterPlanarTensorWrap<T> img, int batch_idx,
                                                          int colIdx, int rowIdx, int rows, int columns, int radius,
                                                          int squared_radius, float color_coefficient,
                                                          float space_coefficient)
{
    const int  x[4] = {colIdx, colIdx + 1, colIdx, colIdx + 1};
    const int  y[4] = {rowIdx, rowIdx, rowIdx + 1, rowIdx + 1};
    const bool windowInside
        = colIdx >= radius && rowIdx >= radius && colIdx + radius + 1 < columns && rowIdx + radius + 1 < rows;

    bool valid[4];
    valid[0] = colIdx < columns && rowIdx < rows;
    valid[1] = colIdx + 1 < columns && rowIdx < rows;
    valid[2] = colIdx < columns && rowIdx + 1 < rows;
    valid[3] = colIdx + 1 < columns && rowIdx + 1 < rows;

    if (!(valid[0] || valid[1] || valid[2] || valid[3]))
    {
        return;
    }

    float center[4][CHANNELS]    = {};
    float numerator[4][CHANNELS] = {};
    float denominator[4]         = {};

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (valid[p])
        {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                center[p][ch] = img.read(batch_idx, ch, y[p], x[p]);
            }
        }
    }

    for (int r = rowIdx - radius; r < rowIdx + radius + 2; r++)
    {
        for (int c = colIdx - radius; c < colIdx + radius + 2; c++)
        {
            const int dx0          = std::abs(c - colIdx);
            const int dy0          = cuda::abs(r - rowIdx);
            const int dx1          = std::abs(c - (colIdx + 1));
            const int dy1          = cuda::abs(r - (rowIdx + 1));
            const int squared_dis0 = dx0 * dx0 + dy0 * dy0;
            const int squared_dis1 = dx1 * dx1 + dy0 * dy0;
            const int squared_dis2 = dx0 * dx0 + dy1 * dy1;
            const int squared_dis3 = dx1 * dx1 + dy1 * dy1;

            if (!(squared_dis0 <= squared_radius || squared_dis1 <= squared_radius || squared_dis2 <= squared_radius
                  || squared_dis3 <= squared_radius))
            {
                continue;
            }

            int            srcX         = c;
            int            srcY         = r;
            constexpr bool fastInterior = !(std::is_same_v<T, float> && CHANNELS == 4);
            bool           inside       = mapBorderCoordinate<B, fastInterior>(srcX, srcY, columns, rows, windowInside);

            float curr[CHANNELS] = {};
            if (inside)
            {
#pragma unroll
                for (int ch = 0; ch < CHANNELS; ++ch)
                {
                    curr[ch] = img.read(batch_idx, ch, srcY, srcX);
                }
            }

            const int squared_dis[4] = {squared_dis0, squared_dis1, squared_dis2, squared_dis3};
#pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                if (valid[p] && squared_dis[p] <= squared_radius)
                {
                    float one_norm_size = 0.f;
#pragma unroll
                    for (int ch = 0; ch < CHANNELS; ++ch)
                    {
                        one_norm_size += cuda::abs(curr[ch] - center[p][ch]);
                    }

                    const float e_space = squared_dis[p] * space_coefficient;
                    const float e_color = one_norm_size * one_norm_size * color_coefficient;
                    const float weight  = cuda::exp(e_space + e_color);
                    denominator[p] += weight;
#pragma unroll
                    for (int ch = 0; ch < CHANNELS; ++ch)
                    {
                        numerator[p][ch] += weight * curr[ch];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (valid[p])
        {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                img.write(batch_idx, ch, y[p], x[p], cuda::SaturateCast<T>(numerator[p][ch] / denominator[p]));
            }
        }
    }
}

template<typename T, NVCVBorderType B, int CHANNELS>
__global__ void BilateralFilterPlanarKernel(BilateralFilterPlanarTensorWrap<T> img, int radius, float sigmaColor,
                                            float sigmaSpace, int rows, int columns)
{
    const int colIdx    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx    = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int batch_idx = blockIdx.z;

    const int   squared_radius    = radius * radius;
    const float space_coefficient = -1 / (2 * sigmaSpace * sigmaSpace);
    const float color_coefficient = -1 / (2 * sigmaColor * sigmaColor);

    BilateralFilterPlanarTile<T, B, CHANNELS>(img, batch_idx, colIdx, rowIdx, rows, columns, radius, squared_radius,
                                              color_coefficient, space_coefficient);
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void BilateralFilterKernel(SrcWrapper src, DstWrapper dst, const int radius, const float sigmaColor,
                                      const float sigmaSpace, const int rows, const int columns)
{
    const int colIdx    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx    = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int batch_idx = blockIdx.z;

    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    int3       coord0{colIdx, rowIdx, batch_idx};
    int3       coord1{colIdx + 1, rowIdx, batch_idx};
    int3       coord2{colIdx, rowIdx + 1, batch_idx};
    int3       coord3{colIdx + 1, rowIdx + 1, batch_idx};
    const bool windowInside = !std::is_same_v<T, float3> && colIdx >= radius && rowIdx >= radius
                           && colIdx + radius + 1 < columns && rowIdx + radius + 1 < rows;
    work_type center0 = cuda::StaticCast<float>(readPackedPixel(src, coord0, rows, columns, windowInside));
    work_type center1 = cuda::StaticCast<float>(readPackedPixel(src, coord1, rows, columns, windowInside));
    work_type center2 = cuda::StaticCast<float>(readPackedPixel(src, coord2, rows, columns, windowInside));
    work_type center3 = cuda::StaticCast<float>(readPackedPixel(src, coord3, rows, columns, windowInside));

    int       squared_radius    = radius * radius;
    float     space_coefficient = -1 / (2 * sigmaSpace * sigmaSpace);
    float     color_coefficient = -1 / (2 * sigmaColor * sigmaColor);
    work_type numerator0        = cuda::SetAll<work_type>(0);
    work_type numerator1        = cuda::SetAll<work_type>(0);
    work_type numerator2        = cuda::SetAll<work_type>(0);
    work_type numerator3        = cuda::SetAll<work_type>(0);
    float     denominator0      = 0;
    float     denominator1      = 0;
    float     denominator2      = 0;
    float     denominator3      = 0;

    for (int c = colIdx - radius; c < colIdx + radius + 2; c++)
    {
        for (int r = rowIdx - radius; r < rowIdx + radius + 2; r++)
        {
            int t0 = std::abs(c - colIdx), t1 = cuda::abs(r - rowIdx);
            int t2 = std::abs(c - (colIdx + 1)), t3 = cuda::abs(r - (rowIdx + 1));
            int squared_dis0 = t0 * t0 + t1 * t1;
            int squared_dis1 = t2 * t2 + t1 * t1;
            int squared_dis2 = t0 * t0 + t3 * t3;
            int squared_dis3 = t2 * t2 + t3 * t3;

            if (!(squared_dis0 <= squared_radius || squared_dis1 <= squared_radius || squared_dis2 <= squared_radius
                  || squared_dis3 <= squared_radius))
            {
                continue;
            }

            int3      coord{c, r, batch_idx};
            work_type curr = cuda::StaticCast<float>(readPackedPixel(src, coord, rows, columns, windowInside));

            if (squared_dis0 <= squared_radius)
            {
                float e_space       = squared_dis0 * space_coefficient;
                float one_norm_size = norm1(curr - center0);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator0 += weight;
                numerator0 += weight * curr;
            }

            if (squared_dis1 <= squared_radius)
            {
                float e_space       = squared_dis1 * space_coefficient;
                float one_norm_size = norm1(curr - center1);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator1 += weight;
                numerator1 = numerator1 + (weight * curr);
            }

            if (squared_dis2 <= squared_radius)
            {
                float e_space       = squared_dis2 * space_coefficient;
                float one_norm_size = norm1(curr - center2);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator2 += weight;
                numerator2 = numerator2 + (weight * curr);
            }

            if (squared_dis3 <= squared_radius)
            {
                float e_space       = squared_dis3 * space_coefficient;
                float one_norm_size = norm1(curr - center3);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator3 += weight;
                numerator3 = numerator3 + (weight * curr);
            }
        }
    }
    if (colIdx < columns && rowIdx < rows)
    {
        dst[coord0] = nvcv::cuda::SaturateCast<T>(numerator0 / denominator0);
    }
    if (colIdx + 1 < columns && rowIdx < rows)
    {
        dst[coord1] = nvcv::cuda::SaturateCast<T>(numerator1 / denominator1);
    }
    if (colIdx < columns && rowIdx + 1 < rows)
    {
        dst[coord2] = nvcv::cuda::SaturateCast<T>(numerator2 / denominator2);
    }
    if (colIdx + 1 < columns && rowIdx + 1 < rows)
    {
        dst[coord3] = nvcv::cuda::SaturateCast<T>(numerator3 / denominator3);
    }
}

template<typename T, NVCVBorderType B>
ErrorCode BilateralFilterPlanarCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                      const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                      const nvcv::TensorDataAccessStridedImagePlanar &outAccess, const int batch,
                                      int rows, int columns, int channels, int radius, float sigmaColor,
                                      float sigmaSpace, cudaStream_t stream)
{
    if (inAccess.sampleStride() * inAccess.numSamples() <= cuda::TypeTraits<int32_t>::max)
    {
        dim3 block(32, 2);
        dim3 grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

        BilateralFilterPlanarTensorWrap<T> img{
            reinterpret_cast<const NVCVByte *>(inData.basePtr()),
            reinterpret_cast<NVCVByte *>(outData.basePtr()),
            inAccess.sampleStride(),
            inAccess.chStride(),
            inAccess.rowStride(),
            inAccess.colStride(),
            outAccess.sampleStride(),
            outAccess.chStride(),
            outAccess.rowStride(),
            outAccess.colStride(),
        };

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif

        switch (channels)
        {
        case 1:
            BilateralFilterPlanarKernel<T, B, 1>
                <<<grid, block, 0, stream>>>(img, radius, sigmaColor, sigmaSpace, rows, columns);
            break;
        case 3:
            BilateralFilterPlanarKernel<T, B, 3>
                <<<grid, block, 0, stream>>>(img, radius, sigmaColor, sigmaSpace, rows, columns);
            break;
        case 4:
            BilateralFilterPlanarKernel<T, B, 4>
                <<<grid, block, 0, stream>>>(img, radius, sigmaColor, sigmaSpace, rows, columns);
            break;
        default:
            LOG_ERROR("Invalid planar channel number ch = " << channels);
            return ErrorCode::INVALID_DATA_SHAPE;
        }

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }
    else
    {
        LOG_ERROR("Input size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<typename T, NVCVBorderType B, typename StrideType>
void BilateralFilterCallerS(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, const int batch,
                            int rows, int columns, int radius, float sigmaColor, float sigmaSpace, float borderValue,
                            cudaStream_t stream)
{
    dim3 block(32, 2);
    dim3 grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

    auto src = cuda::CreateBorderWrapNHW<const T, B, StrideType>(inData, cuda::SetAll<T>(borderValue));
    auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(outData);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    BilateralFilterKernel<<<grid, block, 0, stream>>>(src, dst, radius, sigmaColor, sigmaSpace, rows, columns);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T, NVCVBorderType B>
ErrorCode BilateralFilterCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                const int batch, int rows, int columns, int radius, float sigmaColor, float sigmaSpace,
                                float borderValue, cudaStream_t stream)
{
    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    if (inAccess->sampleStride() * inAccess->numSamples() <= cuda::TypeTraits<int32_t>::max)
    {
        BilateralFilterCallerS<T, B, int32_t>(inData, outData, batch, rows, columns, radius, sigmaColor, sigmaSpace,
                                              borderValue, stream);
    }
    else
    {
        LOG_ERROR("Input size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode BilateralFilter::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, int d,
                                 float sigmaColor, float sigmaSpace, NVCVBorderType borderMode, cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (input_format != output_format)
    {
        LOG_ERROR("Input data format (" << input_format << ") and output data format (" << output_format
                                        << ") must be the same.");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(input_format == kNHWC || input_format == kHWC || input_format == kNCHW || input_format == kCHW))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kHWC, kNHWC, kCHW, or kNCHW");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar = input_format == kNCHW || input_format == kCHW;

    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE || borderMode == NVCV_BORDER_REFLECT
          || borderMode == NVCV_BORDER_WRAP || borderMode == NVCV_BORDER_REFLECT101))
    {
        LOG_ERROR("[Error] Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = GetLegacyDataType(outData.dtype());
    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("[Error] Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (sigmaColor <= 0)
    {
        sigmaColor = 1;
    }
    if (sigmaSpace <= 0)
    {
        sigmaSpace = 1;
    }

    int radius;
    if (d <= 0)
    {
        radius = std::roundf(sigmaSpace * 1.5f);
    }
    else
    {
        radius = d / 2;
    }
    if (radius < 1)
    {
        radius = 1;
    }
    assert(radius < 10000);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    cuda_op::DataShape inputShape  = GetLegacyDataShape(inAccess->infoShape());
    cuda_op::DataShape outputShape = GetLegacyDataShape(outAccess->infoShape());

    if (inputShape != outputShape)
    {
        LOG_ERROR("Input/output shape is different " << inputShape << "/" << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int columns  = inAccess->numCols();
    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (isPlanar && channels == 2)
    {
        LOG_ERROR("Planar BilateralFilter does not support 2-channel images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef ErrorCode (*bilateral_filter_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                            int batch, int rows, int columns, int radius, float sigmaColor,
                                            float sigmaSpace, float borderValue, cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts. The kCV_8S row is null because validation above rejects signed 8-bit input.
    static const bilateral_filter_t funcs[5][6][4] = {
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<uchar2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {nullptr, nullptr, nullptr, nullptr},
         {BilateralFilterCaller<ushort, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<ushort2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<ushort3, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<ushort4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterCaller<short, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<short2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<short3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<short4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterCaller<int, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<int2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<int3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<int4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterCaller<float, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<float2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<float3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<float4, NVCV_BORDER_CONSTANT>},
         },
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<uchar2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<uchar4, NVCV_BORDER_REPLICATE>},
         {nullptr, nullptr, nullptr, nullptr},
         {BilateralFilterCaller<ushort, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<ushort2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<ushort3, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<ushort4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterCaller<short, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<short2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<short3, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<short4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterCaller<int, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<int2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<int3, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<int4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterCaller<float, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<float2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<float3, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<float4, NVCV_BORDER_REPLICATE>},
         },
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_REFLECT>, BilateralFilterCaller<uchar2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<uchar4, NVCV_BORDER_REFLECT>},
         {nullptr, nullptr, nullptr, nullptr},
         {BilateralFilterCaller<ushort, NVCV_BORDER_REFLECT>, BilateralFilterCaller<ushort2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<ushort3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<ushort4, NVCV_BORDER_REFLECT>},
         {BilateralFilterCaller<short, NVCV_BORDER_REFLECT>, BilateralFilterCaller<short2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<short3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<short4, NVCV_BORDER_REFLECT>},
         {BilateralFilterCaller<int, NVCV_BORDER_REFLECT>, BilateralFilterCaller<int2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<int3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<int4, NVCV_BORDER_REFLECT>},
         {BilateralFilterCaller<float, NVCV_BORDER_REFLECT>, BilateralFilterCaller<float2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<float3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<float4, NVCV_BORDER_REFLECT>},
         },
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_WRAP>, BilateralFilterCaller<uchar2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_WRAP>, BilateralFilterCaller<uchar4, NVCV_BORDER_WRAP>},
         {nullptr, nullptr, nullptr, nullptr},
         {BilateralFilterCaller<ushort, NVCV_BORDER_WRAP>, BilateralFilterCaller<ushort2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<ushort3, NVCV_BORDER_WRAP>, BilateralFilterCaller<ushort4, NVCV_BORDER_WRAP>},
         {BilateralFilterCaller<short, NVCV_BORDER_WRAP>, BilateralFilterCaller<short2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<short3, NVCV_BORDER_WRAP>, BilateralFilterCaller<short4, NVCV_BORDER_WRAP>},
         {BilateralFilterCaller<int, NVCV_BORDER_WRAP>, BilateralFilterCaller<int2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<int3, NVCV_BORDER_WRAP>, BilateralFilterCaller<int4, NVCV_BORDER_WRAP>},
         {BilateralFilterCaller<float, NVCV_BORDER_WRAP>, BilateralFilterCaller<float2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<float3, NVCV_BORDER_WRAP>, BilateralFilterCaller<float4, NVCV_BORDER_WRAP>},
         },
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<uchar2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<uchar4, NVCV_BORDER_REFLECT101>},
         {nullptr, nullptr, nullptr, nullptr},
         {BilateralFilterCaller<ushort, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<ushort2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<ushort3, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<ushort4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterCaller<short, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<short2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<short3, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<short4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterCaller<int, NVCV_BORDER_REFLECT101>, BilateralFilterCaller<int2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<int3, NVCV_BORDER_REFLECT101>, BilateralFilterCaller<int4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterCaller<float, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<float2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<float3, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<float4, NVCV_BORDER_REFLECT101>},
         },
    };
    typedef ErrorCode (*bilateral_filter_planar_t)(
        const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
        const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
        const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int batch, int rows, int columns, int channels,
        int radius, float sigmaColor, float sigmaSpace, cudaStream_t stream);
    static const bilateral_filter_planar_t planarFuncs[5][6] = {
        {BilateralFilterPlanarCaller<uchar,   NVCV_BORDER_CONSTANT>, nullptr,
         BilateralFilterPlanarCaller<ushort,   NVCV_BORDER_CONSTANT>,
         BilateralFilterPlanarCaller<short,   NVCV_BORDER_CONSTANT>,
         BilateralFilterPlanarCaller<int,   NVCV_BORDER_CONSTANT>,
         BilateralFilterPlanarCaller<float,   NVCV_BORDER_CONSTANT>                                                          },
        {BilateralFilterPlanarCaller<uchar,  NVCV_BORDER_REPLICATE>, nullptr,
         BilateralFilterPlanarCaller<ushort,  NVCV_BORDER_REPLICATE>,
         BilateralFilterPlanarCaller<short,  NVCV_BORDER_REPLICATE>,
         BilateralFilterPlanarCaller<int,  NVCV_BORDER_REPLICATE>,
         BilateralFilterPlanarCaller<float,  NVCV_BORDER_REPLICATE>                                                          },
        {BilateralFilterPlanarCaller<uchar,    NVCV_BORDER_REFLECT>, nullptr,
         BilateralFilterPlanarCaller<ushort,    NVCV_BORDER_REFLECT>,
         BilateralFilterPlanarCaller<short,    NVCV_BORDER_REFLECT>, BilateralFilterPlanarCaller<int,    NVCV_BORDER_REFLECT>,
         BilateralFilterPlanarCaller<float,    NVCV_BORDER_REFLECT>                                                          },
        {BilateralFilterPlanarCaller<uchar,       NVCV_BORDER_WRAP>, nullptr,
         BilateralFilterPlanarCaller<ushort,       NVCV_BORDER_WRAP>, BilateralFilterPlanarCaller<short,       NVCV_BORDER_WRAP>,
         BilateralFilterPlanarCaller<int,       NVCV_BORDER_WRAP>, BilateralFilterPlanarCaller<float,       NVCV_BORDER_WRAP>},
        {BilateralFilterPlanarCaller<uchar, NVCV_BORDER_REFLECT101>, nullptr,
         BilateralFilterPlanarCaller<ushort, NVCV_BORDER_REFLECT101>,
         BilateralFilterPlanarCaller<short, NVCV_BORDER_REFLECT101>,
         BilateralFilterPlanarCaller<int, NVCV_BORDER_REFLECT101>,
         BilateralFilterPlanarCaller<float, NVCV_BORDER_REFLECT101>                                                          },
    };
    if (isPlanar)
    {
        return planarFuncs[borderMode][data_type](inData, outData, *inAccess, *outAccess, batch, rows, columns,
                                                  channels, radius, sigmaColor, sigmaSpace, stream);
    }
    return funcs[borderMode][data_type][channels - 1](inData, outData, batch, rows, columns, radius, sigmaColor,
                                                      sigmaSpace, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
