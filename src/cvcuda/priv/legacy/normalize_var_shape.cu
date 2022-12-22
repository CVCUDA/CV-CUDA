/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/OpNormalize.h> // for CVCUDA_NORMALIZE_SCALE_IS_STDDEV, etc.
#include <nvcv/cuda/MathWrappers.hpp>

namespace nvcv::legacy::cuda_op {

namespace {

#define BLOCK 32

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normKernel(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<out_T> dst,
                           const scale_type *scale, const base_type *base, float global_scale, float global_shift)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T out = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x)
        = cuda::SaturateCast<cuda::BaseType<out_T>>((out - *base) * *scale * global_scale + global_shift);
}

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevKernel(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<out_T> dst,
                                    const scale_type *scale, const base_type *base, float global_scale,
                                    float global_shift, float epsilon)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    scale_type s   = *scale;
    scale_type x   = s * s + epsilon;
    scale_type mul = 1.0f / cuda::sqrt(x);

    T out = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x)
        = cuda::SaturateCast<cuda::BaseType<out_T>>((out - *base) * mul * global_scale + global_shift);
}

template<typename T, typename out_T, typename base_type, typename scale_type>
void normWrap(const IImageBatchVarShapeDataStridedCuda &in, const base_type *base, const scale_type *scale,
              const IImageBatchVarShapeDataStridedCuda &out, float global_scale, float shift, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                              block(BLOCK, BLOCK / 4, 1);
    dim3                              grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda_op::Ptr2dVarShapeNHWC<T>     src_ptr(in);
    cuda_op::Ptr2dVarShapeNHWC<out_T> dst_ptr(out);

    normKernel<T, out_T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift);
    checkKernelErrors();
}

template<typename T, typename out_T, typename base_type, typename scale_type>
void normInvStdDevWrap(const IImageBatchVarShapeDataStridedCuda &in, const base_type *base, const scale_type *scale,
                       const IImageBatchVarShapeDataStridedCuda &out, float global_scale, float shift, float epsilon,
                       cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);

    cuda_op::Ptr2dVarShapeNHWC<T>     src_ptr(in);
    cuda_op::Ptr2dVarShapeNHWC<out_T> dst_ptr(out);
    normInvStdDevKernel<T, out_T>
        <<<grid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift, epsilon);
    checkKernelErrors();
}

template<typename T, typename out_T>
void norm(const IImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
          const TensorDataAccessStridedImagePlanar &scale, const IImageBatchVarShapeDataStridedCuda &out,
          float global_scale, float shift, cudaStream_t stream)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    if (base.numChannels() != 1 && scale.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else if (base.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else if (scale.numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
}

template<typename T, typename out_T>
void normInvStdDev(const IImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
                   const TensorDataAccessStridedImagePlanar &scale, const IImageBatchVarShapeDataStridedCuda &out,
                   float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    if (base.numChannels() != 1 && scale.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else if (base.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else if (scale.numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
}

} // namespace

ErrorCode NormalizeVarShape::infer(const nvcv::IImageBatchVarShapeDataStridedCuda &inData,
                                   const nvcv::ITensorDataStridedCuda             &baseData,
                                   const nvcv::ITensorDataStridedCuda             &scaleData,
                                   const nvcv::IImageBatchVarShapeDataStridedCuda &outData, const float global_scale,
                                   const float shift, const float epsilon, const uint32_t flags, cudaStream_t stream)
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
        LOG_ERROR("Invliad DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());

    if (!(out_data_type == kCV_8U || out_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid Output DataType " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    int channels = inData.uniqueFormat().numChannels();
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto baseAccess = TensorDataAccessStridedImagePlanar::Create(baseData);
    if (!baseAccess)
    {
        LOG_ERROR("Invalid DataFormat(base) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto scaleAccess = TensorDataAccessStridedImagePlanar::Create(scaleData);
    if (!scaleAccess)
    {
        LOG_ERROR("Invalid DataFormat(scale) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    typedef void (*normalize_t)(
        const IImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
        const TensorDataAccessStridedImagePlanar &scale, const IImageBatchVarShapeDataStridedCuda &out,
        float global_scale, float shift, cudaStream_t stream);

    typedef void (*normalizeInvStdDev_t)(
        const IImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
        const TensorDataAccessStridedImagePlanar &scale, const IImageBatchVarShapeDataStridedCuda &out,
        float global_scale, float shift, float epsilon, cudaStream_t stream);

    int out_type_code = out_data_type == kCV_8U ? 0 : 1;

    static const normalize_t funcs_normalize[6][2][4] = {
        {    {norm<uchar, uchar>, norm<uchar2, uchar2>, norm<uchar3, uchar3>, norm<uchar4, uchar4>},
         {norm<uchar, float>, norm<uchar2, float2>, norm<uchar3, float3>, norm<uchar4, float4>}    },
        {       {norm<schar, uchar>, norm<char2, uchar2>, norm<char3, uchar3>, norm<char4, uchar4>},
         {norm<schar, float>, norm<char2, float2>, norm<char3, float3>, norm<char4, float4>}       },
        {{norm<ushort, uchar>, norm<ushort2, uchar2>, norm<ushort3, uchar3>, norm<ushort4, uchar4>},
         {norm<ushort, float>, norm<ushort2, float2>, norm<ushort3, float3>, norm<ushort4, float4>}},
        {    {norm<short, uchar>, norm<short2, uchar2>, norm<short3, uchar3>, norm<short4, uchar4>},
         {norm<short, float>, norm<short2, float2>, norm<short3, float3>, norm<short4, float4>}    },
        {            {norm<int, uchar>, norm<int2, uchar2>, norm<int3, uchar3>, norm<int4, uchar4>},
         {norm<int, float>, norm<int2, float2>, norm<int3, float3>, norm<int4, float4>}            },
        {    {norm<float, uchar>, norm<float2, uchar2>, norm<float3, uchar3>, norm<float4, uchar4>},
         {norm<float, float>, norm<float2, float2>, norm<float3, float3>, norm<float4, float4>}    },
    };

    static const normalizeInvStdDev_t funcs_normalize_stddev[6][2][4] = {
        { {normInvStdDev<uchar, uchar>, normInvStdDev<uchar2, uchar2>, normInvStdDev<uchar3, uchar3>,
 normInvStdDev<uchar4, uchar4>},
         {normInvStdDev<uchar, float>, normInvStdDev<uchar2, float2>, normInvStdDev<uchar3, float3>,
         normInvStdDev<uchar4, float4>} },
        {  {normInvStdDev<schar, uchar>, normInvStdDev<char2, uchar2>, normInvStdDev<char3, uchar3>,
  normInvStdDev<char4, uchar4>},
         {normInvStdDev<schar, float>, normInvStdDev<char2, float2>, normInvStdDev<char3, float3>,
         normInvStdDev<char4, float4>}  },
        {{normInvStdDev<ushort, uchar>, normInvStdDev<ushort2, uchar2>, normInvStdDev<ushort3, uchar3>,
normInvStdDev<ushort4, uchar4>},
         {normInvStdDev<ushort, float>, normInvStdDev<ushort2, float2>, normInvStdDev<ushort3, float3>,
         normInvStdDev<ushort4, float4>}},
        { {normInvStdDev<short, uchar>, normInvStdDev<short2, uchar2>, normInvStdDev<short3, uchar3>,
 normInvStdDev<short4, uchar4>},
         {normInvStdDev<short, float>, normInvStdDev<short2, float2>, normInvStdDev<short3, float3>,
         normInvStdDev<short4, float4>} },
        {   {normInvStdDev<int, uchar>, normInvStdDev<int2, uchar2>, normInvStdDev<int3, uchar3>,
   normInvStdDev<int4, uchar4>},
         {normInvStdDev<int, float>, normInvStdDev<int2, float2>, normInvStdDev<int3, float3>,
         normInvStdDev<int4, float4>}   },
        { {normInvStdDev<float, uchar>, normInvStdDev<float2, uchar2>, normInvStdDev<float3, uchar3>,
 normInvStdDev<float4, uchar4>},
         {normInvStdDev<float, float>, normInvStdDev<float2, float2>, normInvStdDev<float3, float3>,
         normInvStdDev<float4, float4>} },
    };

    if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        funcs_normalize_stddev[data_type][out_type_code][channels - 1](inData, *baseAccess, *scaleAccess, outData,
                                                                       global_scale, shift, epsilon, stream);
    }
    else
    {
        funcs_normalize[data_type][out_type_code][channels - 1](inData, *baseAccess, *scaleAccess, outData,
                                                                global_scale, shift, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
