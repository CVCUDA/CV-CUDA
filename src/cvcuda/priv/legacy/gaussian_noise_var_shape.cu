/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../SafeSize.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "gaussian_noise_util.cuh"

#include <curand_kernel.h>
using namespace nvcv::legacy::helpers;

using namespace nvcv::legacy::cuda_op;

using namespace nvcv::cuda;

#define BLOCK 512

static int gaussian_noise_var_shape_segments(const nvcv::ImageBatchVarShapeDataStridedCuda &d_out)
{
    nvcv::Size2D maxSize = d_out.maxSize();
    return gaussian_noise_segments(maxSize.w * maxSize.h, BLOCK);
}

template<typename T>
__global__ void gaussian_noise_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                      curandState *state, curandState *nextState, Tensor1DWrap<float> mu,
                                      Tensor1DWrap<float> sigma)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dst_x0                       = offset % width;
        int    dst_y0                       = offset / width;
        int    offset1                      = offset + blockDim.x;
        int    dst_x1                       = offset1 % width;
        int    dst_y1                       = offset1 / width;
        float2 rand                         = gaussian_noise_normal2(localState);
        float  delta0                       = batchMu + rand.x * batchSigma;
        float  delta1                       = batchMu + rand.y * batchSigma;
        *dst.ptr(batch_idx, dst_y0, dst_x0) = SaturateCast<T>(*src.ptr(batch_idx, dst_y0, dst_x0) + delta0);
        *dst.ptr(batch_idx, dst_y1, dst_x1) = SaturateCast<T>(*src.ptr(batch_idx, dst_y1, dst_x1) + delta1);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dst_x                       = offset % width;
        int   dst_y                       = offset / width;
        float rand                        = curand_normal(&localState);
        float delta                       = batchMu + rand * batchSigma;
        *dst.ptr(batch_idx, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_per_channel_kernel(const ImageBatchVarShapeWrapNHWC<T> src,
                                                  ImageBatchVarShapeWrapNHWC<T> dst, curandState *state,
                                                  curandState *nextState, Tensor1DWrap<float> mu,
                                                  Tensor1DWrap<float> sigma)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    int         channel    = src.numChannels();
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channel, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % width;
        int dst_y0  = offset / width;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % width;
        int dst_y1  = offset1 / width;
        gaussian_noise_store_channel_pair<false, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                    channel, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % width;
        int dst_y = offset / width;
        for (int ch = 0; ch < channel; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = batchMu + rand * batchSigma;
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x, ch) + delta);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_float_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                            curandState *state, curandState *nextState, Tensor1DWrap<float> mu,
                                            Tensor1DWrap<float> sigma)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dst_x0                       = offset % width;
        int    dst_y0                       = offset / width;
        int    offset1                      = offset + blockDim.x;
        int    dst_x1                       = offset1 % width;
        int    dst_y1                       = offset1 / width;
        float2 rand                         = gaussian_noise_normal2(localState);
        float  delta0                       = batchMu + rand.x * batchSigma;
        float  delta1                       = batchMu + rand.y * batchSigma;
        T      out0                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y0, dst_x0) + delta0);
        T      out1                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y1, dst_x1) + delta1);
        *dst.ptr(batch_idx, dst_y0, dst_x0) = clamp(StaticCast<float>(out0), 0.f, 1.f);
        *dst.ptr(batch_idx, dst_y1, dst_x1) = clamp(StaticCast<float>(out1), 0.f, 1.f);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dst_x                       = offset % width;
        int   dst_y                       = offset / width;
        float rand                        = curand_normal(&localState);
        float delta                       = batchMu + rand * batchSigma;
        T     out                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        *dst.ptr(batch_idx, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_float_per_channel_kernel(const ImageBatchVarShapeWrapNHWC<T> src,
                                                        ImageBatchVarShapeWrapNHWC<T> dst, curandState *state,
                                                        curandState *nextState, Tensor1DWrap<float> mu,
                                                        Tensor1DWrap<float> sigma)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    int         channel    = src.numChannels();
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channel, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % width;
        int dst_y0  = offset / width;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % width;
        int dst_y1  = offset1 / width;
        gaussian_noise_store_channel_pair<false, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                    channel, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % width;
        int dst_y = offset / width;
        for (int ch = 0; ch < channel; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = batchMu + rand * batchSigma;
            T     out                             = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x, ch) + delta);
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = clamp(StaticCast<float>(out), 0.f, 1.f);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_planar_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                             curandState *state, curandState *nextState, Tensor1DWrap<float> mu,
                                             Tensor1DWrap<float> sigma, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dst_x0  = offset % width;
        int    dst_y0  = offset / width;
        int    offset1 = offset + blockDim.x;
        int    dst_x1  = offset1 % width;
        int    dst_y1  = offset1 / width;
        float2 rand    = gaussian_noise_normal2(localState);
        float  delta0  = batchMu + rand.x * batchSigma;
        float  delta1  = batchMu + rand.y * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            *dst.ptr(batch_idx, ch, dst_y0, dst_x0) = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y0, dst_x0) + delta0);
            *dst.ptr(batch_idx, ch, dst_y1, dst_x1) = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y1, dst_x1) + delta1);
        }
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dst_x = offset % width;
        int   dst_y = offset / width;
        float rand  = curand_normal(&localState);
        float delta = batchMu + rand * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y, dst_x) + delta);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_planar_per_channel_kernel(const ImageBatchVarShapeWrap<T> src,
                                                         ImageBatchVarShapeWrap<T> dst, curandState *state,
                                                         curandState *nextState, Tensor1DWrap<float> mu,
                                                         Tensor1DWrap<float> sigma, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channels, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % width;
        int dst_y0  = offset / width;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % width;
        int dst_y1  = offset1 / width;
        gaussian_noise_store_channel_pair<true, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                   channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % width;
        int dst_y = offset / width;
        for (int ch = 0; ch < channels; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = batchMu + rand * batchSigma;
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y, dst_x) + delta);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_float_planar_kernel(const ImageBatchVarShapeWrap<T> src, ImageBatchVarShapeWrap<T> dst,
                                                   curandState *state, curandState *nextState, Tensor1DWrap<float> mu,
                                                   Tensor1DWrap<float> sigma, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dst_x0  = offset % width;
        int    dst_y0  = offset / width;
        int    offset1 = offset + blockDim.x;
        int    dst_x1  = offset1 % width;
        int    dst_y1  = offset1 / width;
        float2 rand    = gaussian_noise_normal2(localState);
        float  delta0  = batchMu + rand.x * batchSigma;
        float  delta1  = batchMu + rand.y * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            T out0                                  = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y0, dst_x0) + delta0);
            T out1                                  = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y1, dst_x1) + delta1);
            *dst.ptr(batch_idx, ch, dst_y0, dst_x0) = clamp(StaticCast<float>(out0), 0.f, 1.f);
            *dst.ptr(batch_idx, ch, dst_y1, dst_x1) = clamp(StaticCast<float>(out1), 0.f, 1.f);
        }
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dst_x = offset % width;
        int   dst_y = offset / width;
        float rand  = curand_normal(&localState);
        float delta = batchMu + rand * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            T out                                 = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y, dst_x) + delta);
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
__global__ void gaussian_noise_float_planar_per_channel_kernel(const ImageBatchVarShapeWrap<T> src,
                                                               ImageBatchVarShapeWrap<T> dst, curandState *state,
                                                               curandState *nextState, Tensor1DWrap<float> mu,
                                                               Tensor1DWrap<float> sigma, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batch_idx);
    int         total_size = dst.height(batch_idx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channels, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % width;
        int dst_y0  = offset / width;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % width;
        int dst_y1  = offset1 / width;
        gaussian_noise_store_channel_pair<true, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                   channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % width;
        int dst_y = offset / width;
        for (int ch = 0; ch < channels; ch++)
        {
            float rand                            = curand_normal(&localState);
            float delta                           = batchMu + rand * batchSigma;
            T     out                             = SaturateCast<T>(*src.ptr(batch_idx, ch, dst_y, dst_x) + delta);
            *dst.ptr(batch_idx, ch, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T>
void gaussian_noise(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                    curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                    const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                curandState *m_states, curandState *m_nextStates,
                                const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                                cudaStream_t stream)
{
    ImageBatchVarShapeWrapNHWC<T> src_ptr(d_in, channels);
    ImageBatchVarShapeWrapNHWC<T> dst_ptr(d_out, channels);
    Tensor1DWrap<float>           mu(_mu);
    Tensor1DWrap<float>           sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_per_channel_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_float(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                          curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                          const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_float_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                      const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                      curandState *m_states, curandState *m_nextStates,
                                      const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                                      cudaStream_t stream)
{
    ImageBatchVarShapeWrapNHWC<T> src_ptr(d_in, channels);
    ImageBatchVarShapeWrapNHWC<T> dst_ptr(d_out, channels);
    Tensor1DWrap<float>           mu(_mu);
    Tensor1DWrap<float>           sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_per_channel_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_planar(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels, curandState *m_states,
                           curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                           const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_planar_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_planar_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                       const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                       curandState *m_states, curandState *m_nextStates,
                                       const nvcv::TensorDataStridedCuda &_mu,
                                       const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_planar_per_channel_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_float_planar(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                 const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                 curandState *m_states, curandState *m_nextStates,
                                 const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                                 cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_planar_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void gaussian_noise_float_planar_per_channel(const nvcv::ImageBatchVarShapeDataStridedCuda &d_in,
                                             const nvcv::ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                             curandState *m_states, curandState *m_nextStates,
                                             const nvcv::TensorDataStridedCuda &_mu,
                                             const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    ImageBatchVarShapeWrap<T> src_ptr(d_in);
    ImageBatchVarShapeWrap<T> dst_ptr(d_out);
    Tensor1DWrap<float>       mu(_mu);
    Tensor1DWrap<float>       sigma(_sigma);

    int batch    = d_in.numImages();
    int segments = gaussian_noise_var_shape_segments(d_out);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_planar_per_channel_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

namespace nvcv::legacy::cuda_op {

GaussianNoiseVarShape::GaussianNoiseVarShape(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , m_states(nullptr)
    , m_nextStates(nullptr)
    , m_seed(0)
    , m_maxBatchSize(maxBatchSize)
    , m_setupDone(false)
{
    if (maxBatchSize < 0)
    {
        LOG_ERROR("Invalid num of max batch size " << maxBatchSize);
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }
    const size_t stateBytes
        = cvcuda::priv::CheckedMulMany({sizeof(curandState), static_cast<size_t>(BLOCK),
                                        cvcuda::priv::CheckedNonNegativeToSize(maxBatchSize, "maxBatchSize")},
                                       "GaussianNoiseVarShape curand state allocation size overflow");
    const size_t allocationBytes
        = cvcuda::priv::CheckedMulMany({stateBytes, 2}, "GaussianNoiseVarShape curand state allocation size overflow");
    cudaError_t err = cudaMalloc((void **)&m_states, allocationBytes);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << allocationBytes);
        throw LegacyCudaAllocationError("CUDA memory allocation error!");
    }
    m_nextStates = maxBatchSize > 0 ? m_states + static_cast<size_t>(BLOCK) * maxBatchSize : m_states;
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
    if (!(in_format == kNHWC || in_format == kHWC || in_format == kNCHW || in_format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << in_format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    if (!(out_format == kNHWC || out_format == kHWC || out_format == kNCHW || out_format == kCHW))
    {
        LOG_ERROR("Invalid output DataFormat " << out_format
                                               << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    const bool isPlanar    = in_format == kNCHW || in_format == kCHW;
    const bool isOutPlanar = out_format == kNCHW || out_format == kCHW;
    if (isPlanar != isOutPlanar)
    {
        LOG_ERROR("Input and output must both be interleaved or both be planar");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (isPlanar && outData.uniqueFormat().numChannels() != channels)
    {
        LOG_ERROR("Planar input and output must have matching channel counts");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (isPlanar && static_cast<int64_t>(inData.numImages()) * channels > 65535)
    {
        LOG_ERROR("Planar GaussianNoise requires numImages * channels <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (inData.numImages() > m_maxBatchSize)
    {
        LOG_ERROR("Input batch exceeds maxBatchSize");
        return ErrorCode::INVALID_PARAMETER;
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
                               curandState *m_nextStates, const TensorDataStridedCuda &mu,
                               const TensorDataStridedCuda &sigma, cudaStream_t stream);

        static const func_t funcs[5] = {
            gaussian_noise_per_channel<uchar>, 0, gaussian_noise_per_channel<ushort>, gaussian_noise_per_channel<short>,
            gaussian_noise_per_channel<int>,
        };

        static const func_t float_funcs[1] = {
            gaussian_noise_float_per_channel<float>,
        };
        static const func_t planar_funcs[5] = {
            gaussian_noise_planar_per_channel<uchar>,  0,
            gaussian_noise_planar_per_channel<ushort>, gaussian_noise_planar_per_channel<short>,
            gaussian_noise_planar_per_channel<int>,
        };
        static const func_t planar_float_funcs[1] = {
            gaussian_noise_float_planar_per_channel<float>,
        };

        if (in_data_type == kCV_32F)
        {
            const func_t func = isPlanar ? planar_float_funcs[0] : float_funcs[0];
            assert(func != 0);
            func(inData, outData, channels, m_states, m_nextStates, mu, sigma, stream);
        }
        else
        {
            const func_t func = isPlanar ? planar_funcs[in_data_type] : funcs[in_data_type];
            assert(func != 0);
            func(inData, outData, channels, m_states, m_nextStates, mu, sigma, stream);
        }
    }
    else
    {
        typedef void (*func_t)(const ImageBatchVarShapeDataStridedCuda &d_in,
                               const ImageBatchVarShapeDataStridedCuda &d_out, curandState *m_states,
                               curandState *m_nextStates, const TensorDataStridedCuda &mu,
                               const TensorDataStridedCuda &sigma, cudaStream_t stream);
        typedef void (*planar_func_t)(const ImageBatchVarShapeDataStridedCuda &d_in,
                                      const ImageBatchVarShapeDataStridedCuda &d_out, int channels,
                                      curandState *m_states, curandState *m_nextStates, const TensorDataStridedCuda &mu,
                                      const TensorDataStridedCuda &sigma, cudaStream_t stream);

        static const func_t funcs[5][4] = {
            {      gaussian_noise<uchar>,      gaussian_noise<uchar2>,      gaussian_noise<uchar3>,gaussian_noise<uchar4>                                                                                                   },
            {0 /*gaussian_noise<schar>*/, 0 /*gaussian_noise<char2>*/, 0 /*gaussian_noise<char3>*/,
             0 /*gaussian_noise<char4>*/                                                                                   },
            {     gaussian_noise<ushort>,     gaussian_noise<ushort2>,     gaussian_noise<ushort3>, gaussian_noise<ushort4>},
            {      gaussian_noise<short>,      gaussian_noise<short2>,      gaussian_noise<short3>,  gaussian_noise<short4>},
            {        gaussian_noise<int>,        gaussian_noise<int2>,        gaussian_noise<int3>,    gaussian_noise<int4>},
        };

        static const func_t        float_funcs[4]  = {gaussian_noise_float<float>, gaussian_noise_float<float2>,
                                                      gaussian_noise_float<float3>, gaussian_noise_float<float4>};
        static const planar_func_t planar_funcs[5] = {
            gaussian_noise_planar<uchar>, 0, gaussian_noise_planar<ushort>, gaussian_noise_planar<short>,
            gaussian_noise_planar<int>,
        };
        static const planar_func_t planar_float_funcs[1] = {
            gaussian_noise_float_planar<float>,
        };

        if (in_data_type == kCV_32F)
        {
            if (isPlanar)
            {
                const planar_func_t func = planar_float_funcs[0];
                assert(func != 0);
                func(inData, outData, channels, m_states, m_nextStates, mu, sigma, stream);
            }
            else
            {
                const func_t func = float_funcs[channels - 1];
                assert(func != 0);
                func(inData, outData, m_states, m_nextStates, mu, sigma, stream);
            }
        }
        else
        {
            if (isPlanar)
            {
                const planar_func_t func = planar_funcs[in_data_type];
                assert(func != 0);
                func(inData, outData, channels, m_states, m_nextStates, mu, sigma, stream);
            }
            else
            {
                const func_t func = funcs[in_data_type][channels - 1];
                assert(func != 0);
                func(inData, outData, m_states, m_nextStates, mu, sigma, stream);
            }
        }
    }
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
