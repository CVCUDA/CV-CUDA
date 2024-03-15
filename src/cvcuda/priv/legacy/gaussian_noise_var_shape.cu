/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gaussian_noise_util.cuh"

#include <curand_kernel.h>
using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

using namespace nvcv::cuda;

#define BLOCK 512

template<typename T>
__global__ void gaussian_noise_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                      curandState *state, Tensor1DWrap<float> mu, Tensor1DWrap<float> sigma)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = dst.height(batch_idx) * dst.width(batch_idx);
    curandState localState = state[id];
    while (offset < total_size)
    {
        int   dst_x                       = offset % dst.width(batch_idx);
        int   dst_y                       = offset / dst.width(batch_idx);
        float rand                        = curand_normal(&localState);
        float delta                       = mu[batch_idx] + rand * sigma[batch_idx];
        *dst.ptr(batch_idx, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T>
__global__ void gaussian_noise_per_channel_kernel(const ImageBatchVarShapeWrapNHWC<T> src,
                                                  ImageBatchVarShapeWrapNHWC<T> dst, curandState *state,
                                                  Tensor1DWrap<float> mu, Tensor1DWrap<float> sigma)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = dst.height(batch_idx) * dst.width(batch_idx);
    int         channel    = src.numChannels();
    curandState localState = state[id];
    while (offset < total_size)
    {
        int dst_x = offset % dst.width(batch_idx);
        int dst_y = offset / dst.width(batch_idx);
        for (int ch = 0; ch < channel; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = mu[batch_idx] + rand * sigma[batch_idx];
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x, ch) + delta);
        }
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T>
__global__ void gaussian_noise_float_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                            curandState *state, Tensor1DWrap<float> mu, Tensor1DWrap<float> sigma)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = dst.height(batch_idx) * dst.width(batch_idx);
    curandState localState = state[id];
    while (offset < total_size)
    {
        int   dst_x                       = offset % dst.width(batch_idx);
        int   dst_y                       = offset / dst.width(batch_idx);
        float rand                        = curand_normal(&localState);
        float delta                       = mu[batch_idx] + rand * sigma[batch_idx];
        T     out                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        *dst.ptr(batch_idx, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T>
__global__ void gaussian_noise_float_per_channel_kernel(const ImageBatchVarShapeWrapNHWC<T> src,
                                                        ImageBatchVarShapeWrapNHWC<T> dst, curandState *state,
                                                        Tensor1DWrap<float> mu, Tensor1DWrap<float> sigma)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = dst.height(batch_idx) * dst.width(batch_idx);
    int         channel    = src.numChannels();
    curandState localState = state[id];
    while (offset < total_size)
    {
        int dst_x = offset % dst.width(batch_idx);
        int dst_y = offset / dst.width(batch_idx);
        for (int ch = 0; ch < channel; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = mu[batch_idx] + rand * sigma[batch_idx];
            T     out                             = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x, ch) + delta);
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = clamp(StaticCast<float>(out), 0.f, 1.f);
        }
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T>
void gaussian_noise(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                    const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                    cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch = d_in.numImages();
    gaussian_noise_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma);
    checkKernelErrors();
}

template<typename T>
void gaussian_noise_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                curandState *m_states, const nvcv::TensorDataStridedCuda &_mu,
                                const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrapNHWC<T> src_ptr(d_in, channels);
    ImageBatchVarShapeWrapNHWC<T> dst_ptr(d_out, channels);
    Tensor1DWrap<float>           mu(_mu);
    Tensor1DWrap<float>           sigma(_sigma);

    int batch = d_in.numImages();
    gaussian_noise_per_channel_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma);
    checkKernelErrors();
}

template<typename T>
void gaussian_noise_float(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                          const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                          cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch = d_in.numImages();
    gaussian_noise_float_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma);
    checkKernelErrors();
}

template<typename T>
void gaussian_noise_float_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                      curandState *m_states, const nvcv::TensorDataStridedCuda &_mu,
                                      const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrapNHWC<T> src_ptr(d_in, channels);
    ImageBatchVarShapeWrapNHWC<T> dst_ptr(d_out, channels);
    Tensor1DWrap<float>           mu(_mu);
    Tensor1DWrap<float>           sigma(_sigma);

    int batch = d_in.numImages();
    gaussian_noise_float_per_channel_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma);
    checkKernelErrors();
}

namespace nvcv::legacy::cuda_op {

GaussianNoiseVarShape::GaussianNoiseVarShape(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_states(nullptr)
    , m_seed(0)
    , m_maxBatchSize(maxBatchSize)
    , m_setupDone(false)
{
    if (maxBatchSize < 0)
    {
        LOG_ERROR("Invalid num of max batch size " << maxBatchSize);
        throw std::runtime_error("Parameter error!");
    }
    cudaError_t err = cudaMalloc((void **)&m_states, sizeof(curandState) * BLOCK * maxBatchSize);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(curandState) * BLOCK * maxBatchSize);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

GaussianNoiseVarShape::~GaussianNoiseVarShape()
{
    cudaError_t err = cudaFree(m_states);
    if (err != cudaSuccess)
        LOG_ERROR("CUDA memory free error, possible memory leak!");
}

ErrorCode GaussianNoiseVarShape::infer(const ImageBatchVarShapeDataStridedCuda &inData,
                                       const ImageBatchVarShapeDataStridedCuda &outData,
                                       const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma,
                                       bool per_channel, unsigned long long seed, cudaStream_t stream)
{
    DataFormat in_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat out_format = helpers::GetLegacyDataFormat(outData);
    if (!(in_format == kNHWC || in_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << in_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(out_format == kNHWC || out_format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << out_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType in_data_type = helpers::GetLegacyDataType(inData.uniqueFormat());
    if (!(in_data_type == kCV_8U || in_data_type == kCV_16U || in_data_type == kCV_16S || in_data_type == kCV_32S
          || in_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());
    if (in_data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << in_data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType mu_data_type = GetLegacyDataType(mu.dtype());
    if (mu_data_type != kCV_32F)
    {
        LOG_ERROR("Invalid mu DataType " << mu_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int mu_dim = mu.layout().rank();
    if (mu_dim != 1)
    {
        LOG_ERROR("Invalid mu Dim " << mu_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType sigma_data_type = GetLegacyDataType(sigma.dtype());
    if (sigma_data_type != kCV_32F)
    {
        LOG_ERROR("Invalid sigma DataType " << sigma_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int sigma_dim = sigma.layout().rank();
    if (sigma_dim != 1)
    {
        LOG_ERROR("Invalid sigma Dim " << sigma_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!m_setupDone || m_seed != seed)
    {
        m_seed = seed;
        setup_gaussian_rand_kernel<<<m_maxBatchSize, BLOCK, 0, stream>>>(m_states, m_seed);
        m_setupDone = true;
    }

    if (per_channel)
    {
        typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &d_in,
                               const ImageBatchVarShapeDataStridedCuda &d_out, int channels, curandState *m_states,
                               const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma,
                               cudaStream_t stream);

        static const func_t funcs[5] = {
            gaussian_noise_per_channel<uchar>, 0, gaussian_noise_per_channel<ushort>, gaussian_noise_per_channel<short>,
            gaussian_noise_per_channel<int>,
        };

        static const func_t float_funcs[1] = {
            gaussian_noise_float_per_channel<float>,
        };

        if (in_data_type == kCV_32F)
        {
            const func_t func = float_funcs[0];
            assert(func != 0);
            func(inData, outData, channels, m_states, mu, sigma, stream);
        }
        else
        {
            const func_t func = funcs[in_data_type];
            assert(func != 0);
            func(inData, outData, channels, m_states, mu, sigma, stream);
        }
    }
    else
    {
        typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &d_in,
                               const ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                               const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma,
                               cudaStream_t stream);

        static const func_t funcs[5][4] = {
            {      gaussian_noise<uchar>,      gaussian_noise<uchar2>,      gaussian_noise<uchar3>,gaussian_noise<uchar4>                                                                                                   },
            {0 /*gaussian_noise<schar>*/, 0 /*gaussian_noise<char2>*/, 0 /*gaussian_noise<char3>*/,
             0 /*gaussian_noise<char4>*/                                                                                   },
            {     gaussian_noise<ushort>,     gaussian_noise<ushort2>,     gaussian_noise<ushort3>, gaussian_noise<ushort4>},
            {      gaussian_noise<short>,      gaussian_noise<short2>,      gaussian_noise<short3>,  gaussian_noise<short4>},
            {        gaussian_noise<int>,        gaussian_noise<int2>,        gaussian_noise<int3>,    gaussian_noise<int4>},
        };

        static const func_t float_funcs[4] = {gaussian_noise_float<float>, gaussian_noise_float<float2>,
                                              gaussian_noise_float<float3>, gaussian_noise_float<float4>};

        if (in_data_type == kCV_32F)
        {
            const func_t func = float_funcs[channels - 1];
            assert(func != 0);
            func(inData, outData, m_states, mu, sigma, stream);
        }
        else
        {
            const func_t func = funcs[in_data_type][channels - 1];
            assert(func != 0);
            func(inData, outData, m_states, mu, sigma, stream);
        }
    }
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
