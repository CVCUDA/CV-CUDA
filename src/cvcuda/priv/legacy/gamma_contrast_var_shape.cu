/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#define BLOCK 32

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

namespace nvcv::legacy::cuda_op {

static __device__ __forceinline__ float simple_clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

static __device__ __forceinline__ float2 simple_clamp(float2 v, float a, float b)
{
    return {simple_clamp(v.x, a, b), simple_clamp(v.y, a, b)};
}

static __device__ __forceinline__ float3 simple_clamp(float3 v, float a, float b)
{
    return {simple_clamp(v.x, a, b), simple_clamp(v.y, a, b), simple_clamp(v.z, a, b)};
}

static __device__ __forceinline__ float4 simple_clamp(float4 v, float a, float b)
{
    return {simple_clamp(v.x, a, b), simple_clamp(v.y, a, b), simple_clamp(v.z, a, b), simple_clamp(v.w, a, b)};
}

static __device__ __forceinline__ float simple_powf(float a, float b)
{
    return powf(a, b);
}

static __device__ __forceinline__ float2 simple_powf(float2 a, float2 b)
{
    return make_float2(powf(a.x, b.x), powf(a.y, b.y));
}

static __device__ __forceinline__ float3 simple_powf(float3 a, float3 b)
{
    return make_float3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z));
}

static __device__ __forceinline__ float4 simple_powf(float4 a, float4 b)
{
    return make_float4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w));
}

__global__ void copyGammaValues(float *gammaArray, const cuda::Tensor1DWrap<float> gamma, const int numImages,
                                const int channelCount)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    for (int i = 0; i < channelCount; i++)
    {
        gammaArray[index * channelCount + i] = *gamma.ptr(index);
    }
}

// apply 255*((x/255)**gamma) on each pixel
template<typename D, typename gamma_type>
__global__ void gamma_contrast_kernel(const Ptr2dVarShapeNHWC<D> src, Ptr2dVarShapeNHWC<D> dst,
                                      const cuda::Tensor1DWrap<gamma_type> gamma_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    gamma_type gamma = *gamma_.ptr(batch_idx);
    gamma_type tmp   = (*src.ptr(batch_idx, dst_y, dst_x) + 0.0f) / 255.0f;

    D out                             = nvcv::cuda::SaturateCast<cuda::BaseType<D>>(simple_powf(tmp, gamma) * 255.0f);
    *dst.ptr(batch_idx, dst_y, dst_x) = out;
}

// apply (x**gamma) on each pixel
template<typename D, typename gamma_type>
__global__ void gamma_contrast_float_kernel(const Ptr2dVarShapeNHWC<D> src, Ptr2dVarShapeNHWC<D> dst,
                                            const cuda::Tensor1DWrap<gamma_type> gamma_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    gamma_type gamma = *gamma_.ptr(batch_idx);

    D out = nvcv::cuda::SaturateCast<cuda::BaseType<D>>(simple_powf(*src.ptr(batch_idx, dst_y, dst_x), gamma));
    *dst.ptr(batch_idx, dst_y, dst_x) = simple_clamp(out, 0.0, 1.0);
}

template<typename T>
void gamma_contrast(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
                    float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                          block(BLOCK, BLOCK / 4, 1);
    dim3                          grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(in);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T>
void gamma_contrast_float(const IImageBatchVarShapeDataStridedCuda &in, const IImageBatchVarShapeDataStridedCuda &out,
                          float *gammaValues, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                          block(BLOCK, BLOCK / 4, 1);
    dim3                          grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(in);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(out);

    using gamma_type = cuda::ConvertBaseTypeTo<float, T>;
    cuda::Tensor1DWrap<gamma_type> gamma(gammaValues);
    gamma_contrast_float_kernel<T, gamma_type><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, gamma);
    checkKernelErrors();
}

GammaContrastVarShape::GammaContrastVarShape(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
    : CudaBaseOp()
    , m_maxBatchSize(maxVarShapeBatchSize)
    , m_maxChannelCount(maxVarShapeChannelCount)
{
    if (m_maxBatchSize > 0 && m_maxChannelCount > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&m_gammaArray, m_maxBatchSize * m_maxChannelCount * sizeof(float)));
    }
}

GammaContrastVarShape::~GammaContrastVarShape()
{
    NVCV_CHECK_LOG(cudaFree(m_gammaArray));
}

ErrorCode GammaContrastVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                       const IImageBatchVarShapeDataStridedCuda &outData,
                                       const ITensorDataStridedCuda &gammas, cudaStream_t stream)
{
    if (m_maxBatchSize <= 0 || inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Invalid maximum batch size");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxChannelCount <= 0 || inData.uniqueFormat().numChannels() > m_maxChannelCount)
    {
        LOG_ERROR("Invalid maximum channel count");
        return ErrorCode::INVALID_PARAMETER;
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

    auto gammasAccess = nvcv::TensorDataAccessStrided::Create(gammas);
    NVCV_ASSERT(gammasAccess);

    int numElements = 1;
    for (int i = 0; i < gammas.rank(); i++)
    {
        numElements *= gammas.shape(i);
    }

    if (inData.numImages() * channels == numElements)
    {
        // Copy the data device to device
        checkCudaErrors(cudaMemcpyAsync(m_gammaArray, gammasAccess->sampleData(0),
                                        sizeof(float) * inData.numImages() * channels, cudaMemcpyDeviceToDevice,
                                        stream));
    }
    else
    {
        cuda::Tensor1DWrap<float> gammaTensorWrap(gammas);
        copyGammaValues<<<1, inData.numImages(), 0, stream>>>(m_gammaArray, gammaTensorWrap, inData.numImages(),
                                                              channels);
        checkKernelErrors();
    }

    typedef void (*func_t)(const nvcv::IImageBatchVarShapeDataStridedCuda &in,
                           const nvcv::IImageBatchVarShapeDataStridedCuda &out, float *gammas, cudaStream_t stream);

    static const func_t funcs[5][4] = {
        {      gamma_contrast<uchar>,      gamma_contrast<uchar2>,      gamma_contrast<uchar3>,gamma_contrast<uchar4>                                                                                               },
        {0 /*gamma_contrast<schar>*/, 0 /*gamma_contrast<char2>*/, 0 /*gamma_contrast<char3>*/,
         0 /*gamma_contrast<char4>*/                                                                                   },
        {     gamma_contrast<ushort>,     gamma_contrast<ushort2>,     gamma_contrast<ushort3>, gamma_contrast<ushort4>},
        {      gamma_contrast<short>,      gamma_contrast<short2>,      gamma_contrast<short3>,  gamma_contrast<short4>},
        {        gamma_contrast<int>,        gamma_contrast<int2>,        gamma_contrast<int3>,    gamma_contrast<int4>},
    };

    static const func_t funcs_float[4] = {gamma_contrast_float<float>, gamma_contrast_float<float2>,
                                          gamma_contrast_float<float3>, gamma_contrast_float<float4>};

    if (data_type == kCV_32F)
    {
        const func_t func = funcs_float[channels - 1];
        func(inData, outData, m_gammaArray, stream);
    }
    else
    {
        const func_t func = funcs[data_type][channels - 1];
        func(inData, outData, m_gammaArray, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
