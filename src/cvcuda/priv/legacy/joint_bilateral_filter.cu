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

#include <cvcuda/cuda_tools/TypeTraits.hpp>

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
struct JointBilateralFilterPlanarTensorWrap
{
    const NVCVByte *srcBase;
    const NVCVByte *srcColorBase;
    NVCVByte       *dstBase;
    int64_t         srcSampleStride;
    int64_t         srcChStride;
    int64_t         srcRowStride;
    int64_t         srcColStride;
    int64_t         srcColorSampleStride;
    int64_t         srcColorChStride;
    int64_t         srcColorRowStride;
    int64_t         srcColorColStride;
    int64_t         dstSampleStride;
    int64_t         dstChStride;
    int64_t         dstRowStride;
    int64_t         dstColStride;

    __device__ __forceinline__ float readSrc(int sample, int channel, int y, int x) const
    {
        const NVCVByte *ptr
            = srcBase + sample * srcSampleStride + channel * srcChStride + y * srcRowStride + x * srcColStride;
        return static_cast<float>(*reinterpret_cast<const T *>(ptr));
    }

    __device__ __forceinline__ float readColor(int sample, int channel, int y, int x) const
    {
        const NVCVByte *ptr = srcColorBase + sample * srcColorSampleStride + channel * srcColorChStride
                            + y * srcColorRowStride + x * srcColorColStride;
        return static_cast<float>(*reinterpret_cast<const T *>(ptr));
    }

    __device__ __forceinline__ void write(int sample, int channel, int y, int x, T value) const
    {
        NVCVByte *ptr
            = dstBase + sample * dstSampleStride + channel * dstChStride + y * dstRowStride + x * dstColStride;
        *reinterpret_cast<T *>(ptr) = value;
    }
};

template<NVCVBorderType B>
__device__ __forceinline__ bool mapBorderCoordinate(int &x, int &y, int columns, int rows)
{
    if constexpr (B == NVCV_BORDER_CONSTANT)
    {
        return !cuda::IsOutside(x, columns) && !cuda::IsOutside(y, rows);
    }
    else
    {
        x = cuda::GetIndexWithBorder<B>(x, columns);
        y = cuda::GetIndexWithBorder<B>(y, rows);
        return true;
    }
}

template<typename T, NVCVBorderType B, int CHANNELS>
__device__ __forceinline__ void JointBilateralFilterPlanarTile(JointBilateralFilterPlanarTensorWrap<T> img,
                                                               int batch_idx, int colIdx, int rowIdx, int rows,
                                                               int columns, int radius, int squared_radius,
                                                               float color_coefficient, float space_coefficient)
{
    const int x[4] = {colIdx, colIdx + 1, colIdx, colIdx + 1};
    const int y[4] = {rowIdx, rowIdx, rowIdx + 1, rowIdx + 1};

    bool valid[4];
    valid[0] = colIdx < columns && rowIdx < rows;
    valid[1] = colIdx + 1 < columns && rowIdx < rows;
    valid[2] = colIdx < columns && rowIdx + 1 < rows;
    valid[3] = colIdx + 1 < columns && rowIdx + 1 < rows;

    if (!(valid[0] || valid[1] || valid[2] || valid[3]))
    {
        return;
    }

    float centerColor[4][CHANNELS] = {};
    float numerator[4][CHANNELS]   = {};
    float denominator[4]           = {};

#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        if (valid[p])
        {
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                centerColor[p][ch] = img.readColor(batch_idx, ch, y[p], x[p]);
            }
        }
    }

    for (int c = colIdx - radius; c < colIdx + radius + 2; c++)
    {
        for (int r = rowIdx - radius; r < rowIdx + radius + 2; r++)
        {
            const int dx0          = cuda::abs(c - colIdx);
            const int dy0          = cuda::abs(r - rowIdx);
            const int dx1          = cuda::abs(c - (colIdx + 1));
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

            int  srcX   = c;
            int  srcY   = r;
            bool inside = mapBorderCoordinate<B>(srcX, srcY, columns, rows);

            float curr[CHANNELS]      = {};
            float currColor[CHANNELS] = {};
            if (inside)
            {
#pragma unroll
                for (int ch = 0; ch < CHANNELS; ++ch)
                {
                    curr[ch]      = img.readSrc(batch_idx, ch, srcY, srcX);
                    currColor[ch] = img.readColor(batch_idx, ch, srcY, srcX);
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
                        one_norm_size += cuda::abs(currColor[ch] - centerColor[p][ch]);
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
            const float den = denominator[p] != 0.f ? denominator[p] : 1.f;
#pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                img.write(batch_idx, ch, y[p], x[p], cuda::SaturateCast<T>(numerator[p][ch] / den));
            }
        }
    }
}

template<typename T, NVCVBorderType B, int CHANNELS>
__global__ void JointBilateralFilterPlanarKernel(JointBilateralFilterPlanarTensorWrap<T> img, int radius,
                                                 float sigmaColor, float sigmaSpace, int rows, int columns)
{
    const int colIdx    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx    = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int batch_idx = blockIdx.z;

    const int   squared_radius    = radius * radius;
    const float space_coefficient = -1 / (2 * sigmaSpace * sigmaSpace);
    const float color_coefficient = -1 / (2 * sigmaColor * sigmaColor);

    JointBilateralFilterPlanarTile<T, B, CHANNELS>(img, batch_idx, colIdx, rowIdx, rows, columns, radius,
                                                   squared_radius, color_coefficient, space_coefficient);
}

template<typename SrcWrapper, typename DstWrapper>
__global__ void JointBilateralFilterKernel(SrcWrapper src, SrcWrapper srcColor, DstWrapper dst, const int radius,
                                           const float sigmaColor, const float sigmaSpace, const int rows,
                                           const int columns)
{
    const int colIdx    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx    = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int batch_idx = blockIdx.z;

    using T         = typename DstWrapper::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    int3      coord0{colIdx, rowIdx, batch_idx};
    int3      coord1{colIdx + 1, rowIdx, batch_idx};
    int3      coord2{colIdx, rowIdx + 1, batch_idx};
    int3      coord3{colIdx + 1, rowIdx + 1, batch_idx};
    work_type centerColor0 = cuda::StaticCast<float>(srcColor[coord0]);
    work_type centerColor1 = cuda::StaticCast<float>(srcColor[coord1]);
    work_type centerColor2 = cuda::StaticCast<float>(srcColor[coord2]);
    work_type centerColor3 = cuda::StaticCast<float>(srcColor[coord3]);

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
            const int t0 = c - colIdx, t1 = r - rowIdx;
            const int t2 = c - (colIdx + 1), t3 = r - (rowIdx + 1);
            int       squared_dis0 = t0 * t0 + t1 * t1;
            int       squared_dis1 = t2 * t2 + t1 * t1;
            int       squared_dis2 = t0 * t0 + t3 * t3;
            int       squared_dis3 = t2 * t2 + t3 * t3;

            if (!(squared_dis0 <= squared_radius || squared_dis1 <= squared_radius || squared_dis2 <= squared_radius
                  || squared_dis3 <= squared_radius))
            {
                continue;
            }

            int3      coord{c, r, batch_idx};
            work_type curr      = cuda::StaticCast<float>(src[coord]);
            work_type currColor = cuda::StaticCast<float>(srcColor[coord]);

            if (squared_dis0 <= squared_radius)
            {
                float e_space       = squared_dis0 * space_coefficient;
                float one_norm_size = norm1(currColor - centerColor0);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator0 += weight;
                numerator0 += weight * curr;
            }

            if (squared_dis1 <= squared_radius)
            {
                float e_space       = squared_dis1 * space_coefficient;
                float one_norm_size = norm1(currColor - centerColor1);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator1 += weight;
                numerator1 = numerator1 + (weight * curr);
            }

            if (squared_dis2 <= squared_radius)
            {
                float e_space       = squared_dis2 * space_coefficient;
                float one_norm_size = norm1(currColor - centerColor2);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator2 += weight;
                numerator2 = numerator2 + (weight * curr);
            }

            if (squared_dis3 <= squared_radius)
            {
                float e_space       = squared_dis3 * space_coefficient;
                float one_norm_size = norm1(currColor - centerColor3);
                float e_color       = one_norm_size * one_norm_size * color_coefficient;
                float weight        = cuda::exp(e_space + e_color);
                denominator3 += weight;
                numerator3 = numerator3 + (weight * curr);
            }
        }
    }
    denominator0 = (denominator0 != 0) ? denominator0 : 1.0f;
    denominator1 = (denominator1 != 0) ? denominator1 : 1.0f;
    denominator2 = (denominator2 != 0) ? denominator2 : 1.0f;
    denominator3 = (denominator3 != 0) ? denominator3 : 1.0f;
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
ErrorCode JointBilateralFilterCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
                                     const TensorDataStridedCuda &outData, const int batch, int rows, int columns,
                                     int radius, float sigmaColor, float sigmaSpace, float borderValue,
                                     cudaStream_t stream)
{
    using BT                    = cuda::BaseType<T>;
    constexpr int  numElements  = cuda::NumElements<T>;
    constexpr bool useWideBlock = !(sizeof(BT) == 4 && numElements > 1);
    dim3           block(useWideBlock ? 32 : 8, useWideBlock ? 2 : 8);
    dim3           grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    if (inAccess->sampleStride() * inAccess->numSamples() <= cuda::TypeTraits<int32_t>::max)
    {
        auto src      = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue));
        auto srcColor = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inColorData, cuda::SetAll<T>(borderValue));
        auto dst      = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        JointBilateralFilterKernel<<<grid, block, 0, stream>>>(src, srcColor, dst, radius, sigmaColor, sigmaSpace, rows,
                                                               columns);
    }
    else
    {
        LOG_ERROR("Input size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
    return ErrorCode::SUCCESS;
}

template<typename T, NVCVBorderType B>
ErrorCode JointBilateralFilterPlanarCaller(const TensorDataStridedCuda                    &inData,
                                           const TensorDataStridedCuda                    &inColorData,
                                           const TensorDataStridedCuda                    &outData,
                                           const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                           const nvcv::TensorDataAccessStridedImagePlanar &inColorAccess,
                                           const nvcv::TensorDataAccessStridedImagePlanar &outAccess, const int batch,
                                           int rows, int columns, int channels, int radius, float sigmaColor,
                                           float sigmaSpace, cudaStream_t stream)
{
    dim3 block(32, 2);
    dim3 grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

    JointBilateralFilterPlanarTensorWrap<T> img{reinterpret_cast<const NVCVByte *>(inData.basePtr()),
                                                reinterpret_cast<const NVCVByte *>(inColorData.basePtr()),
                                                reinterpret_cast<NVCVByte *>(outData.basePtr()),
                                                inAccess.sampleStride(),
                                                inAccess.chStride(),
                                                inAccess.rowStride(),
                                                inAccess.colStride(),
                                                inColorAccess.sampleStride(),
                                                inColorAccess.chStride(),
                                                inColorAccess.rowStride(),
                                                inColorAccess.colStride(),
                                                outAccess.sampleStride(),
                                                outAccess.chStride(),
                                                outAccess.rowStride(),
                                                outAccess.colStride()};

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    switch (channels)
    {
    case 1:
        JointBilateralFilterPlanarKernel<T, B, 1>
            <<<grid, block, 0, stream>>>(img, radius, sigmaColor, sigmaSpace, rows, columns);
        break;
    case 3:
        JointBilateralFilterPlanarKernel<T, B, 3>
            <<<grid, block, 0, stream>>>(img, radius, sigmaColor, sigmaSpace, rows, columns);
        break;
    case 4:
        JointBilateralFilterPlanarKernel<T, B, 4>
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
    return ErrorCode::SUCCESS;
}

ErrorCode JointBilateralFilter::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
                                      const TensorDataStridedCuda &outData, int d, float sigmaColor, float sigmaSpace,
                                      NVCVBorderType borderMode, cudaStream_t stream)
{
    cuda_op::DataFormat input_format      = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat inputColor_format = GetLegacyDataFormat(inColorData.layout());
    cuda_op::DataFormat output_format     = GetLegacyDataFormat(outData.layout());

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inColorData.dtype() != outData.dtype())
    {
        LOG_ERROR("InputColor and Output formats must be same input format ="
                  << inColorData.dtype() << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (input_format != output_format)
    {
        LOG_ERROR("Input data format (" << input_format << ") and output data format (" << output_format
                                        << ") must be the same.");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inputColor_format != output_format)
    {
        LOG_ERROR("InputColor data format (" << inputColor_format << ") and output data format (" << output_format
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
    auto inColorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inColorData);
    if (!inColorAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
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
        LOG_ERROR("Planar JointBilateralFilter does not support 2-channel images");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef ErrorCode (*joint_bilateral_filter_t)(
        const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
        const TensorDataStridedCuda &outData, int batch, int rows, int columns, int radius, float sigmaColor,
        float sigmaSpace, float borderValue, cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts. The kCV_8S row is null because validation above rejects signed 8-bit input.
    static const joint_bilateral_filter_t funcs[5][6][4] = {
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {nullptr, nullptr, nullptr, nullptr},
         {JointBilateralFilterCaller<ushort, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<ushort2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<ushort3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<ushort4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterCaller<short, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<short2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<short3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<short4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterCaller<int, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<int2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<int3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<int4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterCaller<float, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<float2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<float3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<float4, NVCV_BORDER_CONSTANT>},
         },
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<uchar2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_REPLICATE>},
         {nullptr, nullptr, nullptr, nullptr},
         {JointBilateralFilterCaller<ushort, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<ushort2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<ushort3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<ushort4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterCaller<short, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<short2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<short3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<short4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterCaller<int, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<int2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<int3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<int4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterCaller<float, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<float2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<float3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<float4, NVCV_BORDER_REPLICATE>},
         },
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<uchar2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_REFLECT>},
         {nullptr, nullptr, nullptr, nullptr},
         {JointBilateralFilterCaller<ushort, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<ushort2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<ushort3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<ushort4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterCaller<short, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<short2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<short3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<short4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterCaller<int, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<int2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<int3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<int4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterCaller<float, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<float2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<float3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<float4, NVCV_BORDER_REFLECT>},
         },
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<uchar2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_WRAP>},
         {nullptr, nullptr, nullptr, nullptr},
         {JointBilateralFilterCaller<ushort, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<ushort2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<ushort3, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<ushort4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterCaller<short, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<short2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<short3, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<short4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterCaller<int, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<int2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<int3, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<int4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterCaller<float, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<float2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<float3, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<float4, NVCV_BORDER_WRAP>},
         },
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<uchar2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_REFLECT101>},
         {nullptr, nullptr, nullptr, nullptr},
         {JointBilateralFilterCaller<ushort, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<ushort2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<ushort3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<ushort4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterCaller<short, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<short2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<short3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<short4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterCaller<int, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<int2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<int3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<int4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterCaller<float, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<float2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<float3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<float4, NVCV_BORDER_REFLECT101>},
         },
    };
    typedef ErrorCode (*joint_bilateral_filter_planar_t)(
        const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
        const TensorDataStridedCuda &outData, const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
        const nvcv::TensorDataAccessStridedImagePlanar &inColorAccess,
        const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int batch, int rows, int columns, int channels,
        int radius, float sigmaColor, float sigmaSpace, cudaStream_t stream);

    static const joint_bilateral_filter_planar_t planarFuncs[5][6] = {
        {JointBilateralFilterPlanarCaller<uchar,   NVCV_BORDER_CONSTANT>, nullptr,
         JointBilateralFilterPlanarCaller<ushort,   NVCV_BORDER_CONSTANT>,
         JointBilateralFilterPlanarCaller<short,   NVCV_BORDER_CONSTANT>,
         JointBilateralFilterPlanarCaller<int,   NVCV_BORDER_CONSTANT>,
         JointBilateralFilterPlanarCaller<float,   NVCV_BORDER_CONSTANT>},
        {JointBilateralFilterPlanarCaller<uchar,  NVCV_BORDER_REPLICATE>, nullptr,
         JointBilateralFilterPlanarCaller<ushort,  NVCV_BORDER_REPLICATE>,
         JointBilateralFilterPlanarCaller<short,  NVCV_BORDER_REPLICATE>,
         JointBilateralFilterPlanarCaller<int,  NVCV_BORDER_REPLICATE>,
         JointBilateralFilterPlanarCaller<float,  NVCV_BORDER_REPLICATE>},
        {JointBilateralFilterPlanarCaller<uchar,    NVCV_BORDER_REFLECT>, nullptr,
         JointBilateralFilterPlanarCaller<ushort,    NVCV_BORDER_REFLECT>,
         JointBilateralFilterPlanarCaller<short,    NVCV_BORDER_REFLECT>,
         JointBilateralFilterPlanarCaller<int,    NVCV_BORDER_REFLECT>,
         JointBilateralFilterPlanarCaller<float,    NVCV_BORDER_REFLECT>},
        {JointBilateralFilterPlanarCaller<uchar,       NVCV_BORDER_WRAP>, nullptr,
         JointBilateralFilterPlanarCaller<ushort,       NVCV_BORDER_WRAP>,
         JointBilateralFilterPlanarCaller<short,       NVCV_BORDER_WRAP>,
         JointBilateralFilterPlanarCaller<int,       NVCV_BORDER_WRAP>,
         JointBilateralFilterPlanarCaller<float,       NVCV_BORDER_WRAP>},
        {JointBilateralFilterPlanarCaller<uchar, NVCV_BORDER_REFLECT101>, nullptr,
         JointBilateralFilterPlanarCaller<ushort, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterPlanarCaller<short, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterPlanarCaller<int, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterPlanarCaller<float, NVCV_BORDER_REFLECT101>},
    };

    if (isPlanar)
    {
        return planarFuncs[borderMode][data_type](inData, inColorData, outData, *inAccess, *inColorAccess, *outAccess,
                                                  batch, rows, columns, channels, radius, sigmaColor, sigmaSpace,
                                                  stream);
    }

    return funcs[borderMode][data_type][channels - 1](inData, inColorData, outData, batch, rows, columns, radius,
                                                      sigmaColor, sigmaSpace, borderValue, stream);
}

} // namespace nvcv::legacy::cuda_op
