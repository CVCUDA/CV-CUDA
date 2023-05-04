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
void JointBilateralFilterCaller(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
                                const TensorDataStridedCuda &outData, const int batch, int rows, int columns,
                                int radius, float sigmaColor, float sigmaSpace, float borderValue, cudaStream_t stream)
{
    dim3 block(8, 8);
    dim3 grid(divUp(columns, block.x * 2), divUp(rows, block.y * 2), batch);

    auto src      = cuda::CreateBorderWrapNHW<const T, B>(inData, cuda::SetAll<T>(borderValue));
    auto srcColor = cuda::CreateBorderWrapNHW<const T, B>(inColorData, cuda::SetAll<T>(borderValue));
    auto dst      = cuda::CreateTensorWrapNHW<T>(outData);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    JointBilateralFilterKernel<<<grid, block, 0, stream>>>(src, srcColor, dst, radius, sigmaColor, sigmaSpace, rows,
                                                           columns);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
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

    if ((input_format != kNHWC) && (input_format != kHWC))
    {
        LOG_ERROR("Invalid DataFormat both Input and Output must be kHWC or kNHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if ((inputColor_format != kNHWC) && (inputColor_format != kHWC))
    {
        LOG_ERROR("Invalid DataFormat both InputColor and Output must be kHWC or kNHWC");
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

    typedef void (*joint_bilateral_filter_t)(
        const TensorDataStridedCuda &inData, const TensorDataStridedCuda &inColorData,
        const TensorDataStridedCuda &outData, int batch, int rows, int columns, int radius, float sigmaColor,
        float sigmaSpace, float borderValue, cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts
    static const joint_bilateral_filter_t funcs[5][6][4] = {
        {
         {JointBilateralFilterCaller<uchar, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterCaller<char, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<char2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<char3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterCaller<char4, NVCV_BORDER_CONSTANT>},
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
         {JointBilateralFilterCaller<char, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<char2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<char3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterCaller<char4, NVCV_BORDER_REPLICATE>},
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
         {JointBilateralFilterCaller<char, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<char2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<char3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterCaller<char4, NVCV_BORDER_REFLECT>},
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
         {JointBilateralFilterCaller<char, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<char2, NVCV_BORDER_WRAP>,
         JointBilateralFilterCaller<char3, NVCV_BORDER_WRAP>, JointBilateralFilterCaller<char4, NVCV_BORDER_WRAP>},
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
         {JointBilateralFilterCaller<char, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<char2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<char3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterCaller<char4, NVCV_BORDER_REFLECT101>},
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
    funcs[borderMode][data_type][channels - 1](inData, inColorData, outData, batch, rows, columns, radius, sigmaColor,
                                               sigmaSpace, borderValue, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
