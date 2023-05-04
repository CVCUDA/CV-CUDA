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
__global__ void JointBilateralFilterVarShapeKernel(const SrcWrapper src, const SrcWrapper srcColor, DstWrapper dst,
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

            int3      srcCoord{c, r, batch_idx};
            work_type curr      = cuda::StaticCast<float>(src[srcCoord]);
            work_type currColor = cuda::StaticCast<float>(srcColor[srcCoord]);

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
void JointBilateralFilterVarShapeCaller(const ImageBatchVarShapeDataStridedCuda &inData,
                                        const ImageBatchVarShapeDataStridedCuda &inColorData,
                                        const ImageBatchVarShapeDataStridedCuda &outData, int batch,
                                        const cuda::Tensor1DWrap<int>   &inDiameter,
                                        const cuda::Tensor1DWrap<float> &inSigmaColor,
                                        const cuda::Tensor1DWrap<float> &inSigmaSpace, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData);
    cuda::BorderVarShapeWrap<const T, B> srcColor(inColorData);
    cuda::ImageBatchVarShapeWrap<T>      dst(outData);

    Size2D outMaxSize = outData.maxSize();
    dim3   block(8, 8);
    dim3   grid(divUp(outMaxSize.w, block.x * 2), divUp(outMaxSize.h, block.y * 2), batch);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    JointBilateralFilterVarShapeKernel<<<grid, block, 0, stream>>>(src, srcColor, dst, inDiameter, inSigmaColor,
                                                                   inSigmaSpace);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

ErrorCode JointBilateralFilterVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                              const ImageBatchVarShapeDataStridedCuda &inColorData,
                                              const ImageBatchVarShapeDataStridedCuda &outData,
                                              const TensorDataStridedCuda             &diameterData,
                                              const TensorDataStridedCuda             &sigmaColorData,
                                              const TensorDataStridedCuda &sigmaSpaceData, NVCVBorderType borderMode,
                                              cudaStream_t stream)
{
    cuda_op::DataFormat input_format      = GetLegacyDataFormat(inData);
    cuda_op::DataFormat inputColor_format = GetLegacyDataFormat(inColorData);
    cuda_op::DataFormat output_format     = GetLegacyDataFormat(outData);

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!inColorData.uniqueFormat())
    {
        LOG_ERROR("Images in the inputColor varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.uniqueFormat() != outData.uniqueFormat())
    {
        LOG_ERROR("Input and Output formats must be same input format ="
                  << helpers::GetLegacyDataType(inData.uniqueFormat())
                  << " output format = " << helpers::GetLegacyDataType(outData.uniqueFormat()));
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inColorData.uniqueFormat() != outData.uniqueFormat())
    {
        LOG_ERROR("InputColor and Output formats must be same input format ="
                  << helpers::GetLegacyDataType(inColorData.uniqueFormat())
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

    typedef void (*joint_bilateral_filter_var_shape_t)(
        const ImageBatchVarShapeDataStridedCuda &inData, const ImageBatchVarShapeDataStridedCuda &inColorData,
        const ImageBatchVarShapeDataStridedCuda &outData, int batch, const cuda::Tensor1DWrap<int> &inDiameter,
        const cuda::Tensor1DWrap<float> &inSigmaColor, const cuda::Tensor1DWrap<float> &inSigmaSpace,
        cudaStream_t stream);

    // All templated functions instantiated here to remove one level of indirection that just hides the same lookup
    // table in 5 parts
    static const joint_bilateral_filter_var_shape_t funcs[5][6][4] = {
        {
         {JointBilateralFilterVarShapeCaller<uchar, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterVarShapeCaller<char, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<char2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<char3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<char4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterVarShapeCaller<ushort, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterVarShapeCaller<short, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<short2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<short3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<short4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterVarShapeCaller<int, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<int2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<int3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<int4, NVCV_BORDER_CONSTANT>},
         {JointBilateralFilterVarShapeCaller<float, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<float2, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<float3, NVCV_BORDER_CONSTANT>,
         JointBilateralFilterVarShapeCaller<float4, NVCV_BORDER_CONSTANT>},
         },
        {
         {JointBilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterVarShapeCaller<char, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<char2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<char3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<char4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterVarShapeCaller<short, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<short2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<short3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<short4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterVarShapeCaller<int, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<int2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<int3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<int4, NVCV_BORDER_REPLICATE>},
         {JointBilateralFilterVarShapeCaller<float, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<float2, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<float3, NVCV_BORDER_REPLICATE>,
         JointBilateralFilterVarShapeCaller<float4, NVCV_BORDER_REPLICATE>},
         },
        {
         {JointBilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterVarShapeCaller<char, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<char2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<char3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<char4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterVarShapeCaller<short, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<short2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<short3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<short4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterVarShapeCaller<int, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<int2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<int3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<int4, NVCV_BORDER_REFLECT>},
         {JointBilateralFilterVarShapeCaller<float, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<float2, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<float3, NVCV_BORDER_REFLECT>,
         JointBilateralFilterVarShapeCaller<float4, NVCV_BORDER_REFLECT>},
         },
        {
         {JointBilateralFilterVarShapeCaller<uchar, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterVarShapeCaller<char, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<char2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<char3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<char4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterVarShapeCaller<ushort, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterVarShapeCaller<short, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<short2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<short3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<short4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterVarShapeCaller<int, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<int2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<int3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<int4, NVCV_BORDER_WRAP>},
         {JointBilateralFilterVarShapeCaller<float, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<float2, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<float3, NVCV_BORDER_WRAP>,
         JointBilateralFilterVarShapeCaller<float4, NVCV_BORDER_WRAP>},
         },
        {
         {JointBilateralFilterVarShapeCaller<uchar, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<uchar2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<uchar3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<uchar4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterVarShapeCaller<char, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<char2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<char3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<char4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterVarShapeCaller<ushort, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<ushort2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<ushort3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<ushort4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterVarShapeCaller<short, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<short2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<short3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<short4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterVarShapeCaller<int, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<int2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<int3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<int4, NVCV_BORDER_REFLECT101>},
         {JointBilateralFilterVarShapeCaller<float, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<float2, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<float3, NVCV_BORDER_REFLECT101>,
         JointBilateralFilterVarShapeCaller<float4, NVCV_BORDER_REFLECT101>},
         },
    };

    funcs[borderMode][data_type][channels - 1](inData, inColorData, outData, batch, inDiameter, inSigmaColor,
                                               inSigmaSpace, stream);
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
