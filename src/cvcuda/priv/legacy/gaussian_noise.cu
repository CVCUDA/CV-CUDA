/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_kernel(const Tensor3DWrap<T, StrideType> src, Tensor3DWrap<T, StrideType> dst,
                                      curandState *state, Tensor1DWrap<float, int32_t> mu,
                                      Tensor1DWrap<float, int32_t> sigma, int rows, int cols)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    curandState localState = state[id];
    while (offset < total_size)
    {
        int   dst_x                       = offset % cols;
        int   dst_y                       = offset / cols;
        float rand                        = curand_normal(&localState);
        float delta                       = mu[batch_idx] + rand * sigma[batch_idx];
        *dst.ptr(batch_idx, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T, typename StrideType>
__global__ void gaussian_noise_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                  Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                  Tensor1DWrap<float, int32_t> mu, Tensor1DWrap<float, int32_t> sigma,
                                                  int rows, int cols, int channel)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    curandState localState = state[id];
    while (offset < total_size)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_kernel(const Tensor3DWrap<T, StrideType> src, Tensor3DWrap<T, StrideType> dst,
                                            curandState *state, Tensor1DWrap<float, int32_t> mu,
                                            Tensor1DWrap<float, int32_t> sigma, int rows, int cols)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    curandState localState = state[id];
    while (offset < total_size)
    {
        int   dst_x                       = offset % cols;
        int   dst_y                       = offset / cols;
        float rand                        = curand_normal(&localState);
        float delta                       = mu[batch_idx] + rand * sigma[batch_idx];
        T     out                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        *dst.ptr(batch_idx, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        offset += blockDim.x;
    }
    state[id] = localState;
}

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                        Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                        Tensor1DWrap<float, int32_t> mu,
                                                        Tensor1DWrap<float, int32_t> sigma, int rows, int cols,
                                                        int channel)
{
    int         offset     = threadIdx.x;
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    curandState localState = state[id];
    while (offset < total_size)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<typename T, typename StrideType = int32_t>
void gaussian_noise(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out, int batch,
                    int rows, int cols, curandState *m_states, const nvcv::TensorDataStridedCuda &_mu,
                    const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    gaussian_noise_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma, rows, cols);
    checkKernelErrors();
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_per_channel(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out,
                                int batch, int channels, int rows, int cols, curandState *m_states,
                                const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                                cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHWC<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHWC<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    gaussian_noise_per_channel_kernel<T>
        <<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma, rows, cols, channels);
    checkKernelErrors();
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out, int batch,
                          int rows, int cols, curandState *m_states, const nvcv::TensorDataStridedCuda &_mu,
                          const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    gaussian_noise_float_kernel<T><<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma, rows, cols);
    checkKernelErrors();
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float_per_channel(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out,
                                      int batch, int channels, int rows, int cols, curandState *m_states,
                                      const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                                      cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHWC<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHWC<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    gaussian_noise_float_per_channel_kernel<T>
        <<<batch, BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, mu, sigma, rows, cols, channels);
    checkKernelErrors();
}

namespace nvcv::legacy::cuda_op {

GaussianNoise::GaussianNoise(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_states(nullptr)
    , m_seed(0)
    , m_maxBatchSize(maxBatchSize)
    , m_setupDone(false)
{
    if (maxBatchSize < 0)
    {
        LOG_ERROR("Invalid num of max batch size " << maxBatchSize);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameter error!");
    }
    cudaError_t err = cudaMalloc((void **)&m_states, sizeof(curandState) * BLOCK * maxBatchSize);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(curandState) * BLOCK * maxBatchSize);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

GaussianNoise::~GaussianNoise()
{
    cudaError_t err = cudaFree(m_states);
    if (err != cudaSuccess)
        LOG_ERROR("CUDA memory free error, possible memory leak!");
}

ErrorCode GaussianNoise::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                               const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma, bool per_channel,
                               unsigned long long seed, cudaStream_t stream)
{
    DataFormat in_format  = GetLegacyDataFormat(inData.layout());
    DataFormat out_format = GetLegacyDataFormat(outData.layout());
    if (!(in_format == kNHWC || in_format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << in_format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(out_format == kNHWC || out_format == kHWC))
    {
        LOG_ERROR("Invalid output DataFormat " << out_format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    int channels = inAccess->numChannels();
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto inMaxStride = inAccess->sampleStride() * inAccess->numSamples();
    if (inMaxStride > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (outMaxStride > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Output size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType in_data_type = GetLegacyDataType(inData.dtype());
    if (!(in_data_type == kCV_8U || in_data_type == kCV_16U || in_data_type == kCV_16S || in_data_type == kCV_32S
          || in_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << in_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType out_data_type = GetLegacyDataType(outData.dtype());
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
        typedef void (*func_t)(const TensorDataStridedCuda &d_in, const TensorDataStridedCuda &d_out, int batch,
                               int channels, int rows, int cols, curandState *m_states, const TensorDataStridedCuda &mu,
                               const TensorDataStridedCuda &sigma, cudaStream_t stream);

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
            func(inData, outData, inAccess->numSamples(), channels, inAccess->numRows(), inAccess->numCols(), m_states,
                 mu, sigma, stream);
        }
        else
        {
            const func_t func = funcs[in_data_type];
            assert(func != 0);
            func(inData, outData, inAccess->numSamples(), channels, inAccess->numRows(), inAccess->numCols(), m_states,
                 mu, sigma, stream);
        }
    }
    else
    {
        typedef void (*func_t)(const TensorDataStridedCuda &d_in, const TensorDataStridedCuda &d_out, int batch,
                               int rows, int cols, curandState *m_states, const TensorDataStridedCuda &mu,
                               const TensorDataStridedCuda &sigma, cudaStream_t stream);

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
            func(inData, outData, inAccess->numSamples(), inAccess->numRows(), inAccess->numCols(), m_states, mu, sigma,
                 stream);
        }
        else
        {
            const func_t func = funcs[in_data_type][channels - 1];
            assert(func != 0);
            func(inData, outData, inAccess->numSamples(), inAccess->numRows(), inAccess->numCols(), m_states, mu, sigma,
                 stream);
        }
    }
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
