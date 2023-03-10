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

template<class SrcWrapper, class DstWrapper>
__global__ void BilateralFilterVarShapeKernel(const SrcWrapper src, DstWrapper dst,
                                              const cuda::Tensor1DWrap<int>   inDiameter,
                                              const cuda::Tensor1DWrap<float> inSigmaColor,
                                              const cuda::Tensor1DWrap<float> inSigmaSpace)
{
    using T             = typename DstWrapper::ValueType;
    const int batch_idx = get_batch_idx();
    const int rows      = dst.height(batch_idx);
    const int columns   = dst.width(batch_idx);

    // Preprocessing moved here because tensors are GPU resident
    float sigmaColor = inSigmaColor[batch_idx];
    if (sigmaColor <= 0)
    {
        sigmaColor = 1;
    }
    float sigmaSpace = inSigmaSpace[batch_idx];
    if (sigmaSpace <= 0)
    {
        sigmaSpace = 1;
    }

    int radius;
    int diameter = inDiameter[batch_idx];
    if (diameter <= 0)
    {
        radius = std::roundf(sigmaSpace * 1.5f);
    }
    else
    {
        radius = diameter / 2;
    }
    if (radius < 1)
    {
        radius = 1;
    }
    assert(radius < 10000);

    const int colIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int rowIdx = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    using work_type  = cuda::ConvertBaseTypeTo<float, T>;
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

            int3      srcCoord{c, r, batch_idx};
            work_type curr = cuda::StaticCast<float>(src[srcCoord]);

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
        *dst.ptr(coord0.z, coord0.y, coord0.x) = nvcv::cuda::SaturateCast<T>(numerator0 / denominator0);
    }
    if (colIdx + 1 < columns && rowIdx < rows)
    {
        *dst.ptr(coord1.z, coord1.y, coord1.x) = nvcv::cuda::SaturateCast<T>(numerator1 / denominator1);
    }
    if (colIdx < columns && rowIdx + 1 < rows)
    {
        *dst.ptr(coord2.z, coord2.y, coord2.x) = nvcv::cuda::SaturateCast<T>(numerator2 / denominator2);
    }
    if (colIdx + 1 < columns && rowIdx + 1 < rows)
    {
        *dst.ptr(coord3.z, coord3.y, coord3.x) = nvcv::cuda::SaturateCast<T>(numerator3 / denominator3);
    }
}

template<typename T, NVCVBorderType B>
void BilateralFilterVarShapeCaller(const IImageBatchVarShapeDataStridedCuda &inData,
                                   const IImageBatchVarShapeDataStridedCuda &outData, int batch,
                                   const cuda::Tensor1DWrap<int>   &inDiameter,
                                   const cuda::Tensor1DWrap<float> &inSigmaColor,
                                   const cuda::Tensor1DWrap<float> &inSigmaSpace, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    Size2D outMaxSize = outData.maxSize();
    dim3   block(8, 8);
    dim3   grid(divUp(outMaxSize.w, block.x * 2), divUp(outMaxSize.h, block.y * 2), batch);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    BilateralFilterVarShapeKernel<<<grid, block, 0, stream>>>(src, dst, inDiameter, inSigmaColor, inSigmaSpace);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

ErrorCode BilateralFilterVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                         const IImageBatchVarShapeDataStridedCuda &outData,
                                         const ITensorDataStridedCuda             &diameterData,
                                         const ITensorDataStridedCuda             &sigmaColorData,
                                         const ITensorDataStridedCuda &sigmaSpaceData, NVCVBorderType borderMode,
                                         cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData);
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.uniqueFormat() != outData.uniqueFormat())
    {
        LOG_ERROR("Input and Output formats must be same input format ="
                  << helpers::GetLegacyDataType(inData.uniqueFormat())
                  << " output format = " << helpers::GetLegacyDataType(outData.uniqueFormat()));
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REPLICATE || borderMode == NVCV_BORDER_REFLECT
          || borderMode == NVCV_BORDER_WRAP || borderMode == NVCV_BORDER_REFLECT101))
    {
        LOG_ERROR("[Error] Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType data_type = GetLegacyDataType(outData.uniqueFormat());
    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("[Error] Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType diameter_data_type = GetLegacyDataType(diameterData.dtype());
    if (diameter_data_type != kCV_32S)
    {
        LOG_ERROR("[Error] Invalid diameterData DataType " << diameter_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType sigmaColor_data_type = GetLegacyDataType(sigmaColorData.dtype());
    if (sigmaColor_data_type != kCV_32F)
    {
        LOG_ERROR("[Error] Invalid sigmaColorData DataType " << sigmaColor_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType sigmaSpace_data_type = GetLegacyDataType(sigmaSpaceData.dtype());
    if (sigmaSpace_data_type != kCV_32F)
    {
        LOG_ERROR("[Error] Invalid sigmaSpaceData DataType " << sigmaSpace_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (inData.numImages() != outData.numImages())
    {
        LOG_ERROR("Input and Output data must have the same number of images (" << inData.numImages()
                                                                                << " != " << outData.numImages());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch    = inData.numImages();
    int channels = inData.uniqueFormat().numChannels();
    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // Create Tensor wrappers for parameter arrays
    cuda::Tensor1DWrap<int>   inDiameter(diameterData);
    cuda::Tensor1DWrap<float> inSigmaColor(sigmaColorData);
    cuda::Tensor1DWrap<float> inSigmaSpace(sigmaSpaceData);

    typedef void (*bilateral_filter_var_shape_t)(
        const IImageBatchVarShapeDataStridedCuda &inData, const IImageBatchVarShapeDataStridedCuda &outData, int batch,
        const cuda::Tensor1DWrap<int> &inDiameter, const cuda::Tensor1DWrap<float> &inSigmaColor,
        const cuda::Tensor1DWrap<float> &inSigmaSpace, cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts
    static const bilateral_filter_var_shape_t funcs[5][6][4] = {
        {
         {BilateralFilterVarShapeCaller<uchar, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterVarShapeCaller<char, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<char2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<char3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<char4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterVarShapeCaller<ushort, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterVarShapeCaller<short, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<short2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<short3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<short4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterVarShapeCaller<int, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<int2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<int3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<int4, NVCV_BORDER_CONSTANT>},
         {BilateralFilterVarShapeCaller<float, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<float2, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<float3, NVCV_BORDER_CONSTANT>,
         BilateralFilterVarShapeCaller<float4, NVCV_BORDER_CONSTANT>},
         },
        {
         {BilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterVarShapeCaller<char, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<char2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<char3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<char4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterVarShapeCaller<short, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<short2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<short3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<short4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterVarShapeCaller<int, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<int2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<int3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<int4, NVCV_BORDER_REPLICATE>},
         {BilateralFilterVarShapeCaller<float, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<float2, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<float3, NVCV_BORDER_REPLICATE>,
         BilateralFilterVarShapeCaller<float4, NVCV_BORDER_REPLICATE>},
         },
        {
         {BilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REFLECT>},
         {BilateralFilterVarShapeCaller<char, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<char2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<char3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<char4, NVCV_BORDER_REFLECT>},
         {BilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REFLECT>},
         {BilateralFilterVarShapeCaller<short, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<short2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<short3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<short4, NVCV_BORDER_REFLECT>},
         {BilateralFilterVarShapeCaller<int, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<int2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<int3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<int4, NVCV_BORDER_REFLECT>},
         {BilateralFilterVarShapeCaller<float, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<float2, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<float3, NVCV_BORDER_REFLECT>,
         BilateralFilterVarShapeCaller<float4, NVCV_BORDER_REFLECT>},
         },
        {
         {BilateralFilterVarShapeCaller<uchar, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_WRAP>},
         {BilateralFilterVarShapeCaller<char, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<char2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<char3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<char4, NVCV_BORDER_WRAP>},
         {BilateralFilterVarShapeCaller<ushort, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_WRAP>},
         {BilateralFilterVarShapeCaller<short, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<short2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<short3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<short4, NVCV_BORDER_WRAP>},
         {BilateralFilterVarShapeCaller<int, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<int2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<int3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<int4, NVCV_BORDER_WRAP>},
         {BilateralFilterVarShapeCaller<float, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<float2, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<float3, NVCV_BORDER_WRAP>,
         BilateralFilterVarShapeCaller<float4, NVCV_BORDER_WRAP>},
         },
        {
         {BilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterVarShapeCaller<char, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<char2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<char3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<char4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterVarShapeCaller<short, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<short2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<short3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<short4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterVarShapeCaller<int, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<int2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<int3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<int4, NVCV_BORDER_REFLECT101>},
         {BilateralFilterVarShapeCaller<float, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<float2, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<float3, NVCV_BORDER_REFLECT101>,
         BilateralFilterVarShapeCaller<float4, NVCV_BORDER_REFLECT101>},
         },
    };

    funcs[borderMode][data_type][channels - 1](inData, outData, batch, inDiameter, inSigmaColor, inSigmaSpace, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
