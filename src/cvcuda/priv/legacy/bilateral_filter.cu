/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template<typename SrcWrapper, typename DstWrapper>
__global__ void BilateralFilterKernel(SrcWrapper src, DstWrapper dst, const int radius, const float sigmaColor,
                                      const float sigmaSpace, const int rows, const int columns)
{
    const int colIdx    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx    = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int batch_idx = blockIdx.z;

    using T         = typename DstWrapper::ValueType;
    using BT        = cuda::BaseType<T>;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    int3      coord0{colIdx, rowIdx, batch_idx};
    int3      coord1{colIdx + 1, rowIdx, batch_idx};
    int3      coord2{colIdx, rowIdx + 1, batch_idx};
    int3      coord3{colIdx + 1, rowIdx + 1, batch_idx};
    work_type center0 = cuda::StaticCast<float>(src[coord0]);
    work_type center1 = cuda::StaticCast<float>(src[coord1]);
    work_type center2 = cuda::StaticCast<float>(src[coord2]);
    work_type center3 = cuda::StaticCast<float>(src[coord3]);

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
            work_type curr = cuda::StaticCast<float>(src[coord]);

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
        dst[coord0] = nvcv::cuda::SaturateCast<BT>(numerator0 / denominator0);
    }
    if (colIdx + 1 < columns && rowIdx < rows)
    {
        dst[coord1] = nvcv::cuda::SaturateCast<BT>(numerator1 / denominator1);
    }
    if (colIdx < columns && rowIdx + 1 < rows)
    {
        dst[coord2] = nvcv::cuda::SaturateCast<BT>(numerator2 / denominator2);
    }
    if (colIdx + 1 < columns && rowIdx + 1 < rows)
    {
        dst[coord3] = nvcv::cuda::SaturateCast<BT>(numerator3 / denominator3);
    }
}

template<typename T, NVCVBorderType B>
void BilateralFilterCaller(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData, const int batch,
                           int rows, int columns, int radius, float sigmaColor, float sigmaSpace, float borderValue,
                           cudaStream_t stream)
{
    dim3 block(8, 8);
    dim3 grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

    cuda::BorderWrapNHW<const T, B> src(inData, cuda::SetAll<T>(borderValue));
    cuda::Tensor3DWrap<T>           dst(outData);

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

ErrorCode BilateralFilter::infer(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData, int d,
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

    if ((input_format != kNHWC) && (input_format != kHWC))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kHWC or kNHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

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
    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int columns  = inAccess->numCols();
    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*bilateral_filter_t)(const ITensorDataStridedCuda &inData, const ITensorDataStridedCuda &outData,
                                       int batch, int rows, int columns, int radius, float sigmaColor, float sigmaSpace,
                                       float borderValue, cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts
    static const bilateral_filter_t funcs[5][6][4] = {
        {
         {BilateralFilterCaller<uchar, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<uchar2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<uchar3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterCaller<char, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<char2, NVCV_BORDER_CONSTANT>,
         BilateralFilterCaller<char3, NVCV_BORDER_CONSTANT>, BilateralFilterCaller<char4, NVCV_BORDER_CONSTANT>},
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
         {BilateralFilterCaller<char, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<char2, NVCV_BORDER_REPLICATE>,
         BilateralFilterCaller<char3, NVCV_BORDER_REPLICATE>, BilateralFilterCaller<char4, NVCV_BORDER_REPLICATE>},
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
         {BilateralFilterCaller<char, NVCV_BORDER_REFLECT>, BilateralFilterCaller<char2, NVCV_BORDER_REFLECT>,
         BilateralFilterCaller<char3, NVCV_BORDER_REFLECT>, BilateralFilterCaller<char4, NVCV_BORDER_REFLECT>},
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
         {BilateralFilterCaller<char, NVCV_BORDER_WRAP>, BilateralFilterCaller<char2, NVCV_BORDER_WRAP>,
         BilateralFilterCaller<char3, NVCV_BORDER_WRAP>, BilateralFilterCaller<char4, NVCV_BORDER_WRAP>},
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
         {BilateralFilterCaller<char, NVCV_BORDER_REFLECT101>, BilateralFilterCaller<char2, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<char3, NVCV_BORDER_REFLECT101>,
         BilateralFilterCaller<char4, NVCV_BORDER_REFLECT101>},
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
    funcs[borderMode][data_type][channels - 1](inData, outData, batch, rows, columns, radius, sigmaColor, sigmaSpace,
                                               borderValue, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
