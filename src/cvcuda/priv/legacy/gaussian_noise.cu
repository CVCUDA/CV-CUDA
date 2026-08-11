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

template<typename T, typename StrideType>
__global__ void gaussian_noise_kernel(const Tensor3DWrap<T, StrideType> src, Tensor3DWrap<T, StrideType> dst,
                                      curandState *state, curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                      Tensor1DWrap<float, int32_t> sigma, int rows, int cols)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int    dst_x0                       = offset % cols;
        int    dst_y0                       = offset / cols;
        int    offset1                      = offset + blockDim.x;
        int    dst_x1                       = offset1 % cols;
        int    dst_y1                       = offset1 / cols;
        float2 rand                         = gaussian_noise_normal2(localState);
        float  delta0                       = batchMu + rand.x * batchSigma;
        float  delta1                       = batchMu + rand.y * batchSigma;
        *dst.ptr(batch_idx, dst_y0, dst_x0) = SaturateCast<T>(*src.ptr(batch_idx, dst_y0, dst_x0) + delta0);
        *dst.ptr(batch_idx, dst_y1, dst_x1) = SaturateCast<T>(*src.ptr(batch_idx, dst_y1, dst_x1) + delta1);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dst_x                       = offset % cols;
        int   dst_y                       = offset / cols;
        float rand                        = curand_normal(&localState);
        float delta                       = batchMu + rand * batchSigma;
        *dst.ptr(batch_idx, dst_y, dst_x) = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T, typename StrideType>
__global__ void gaussian_noise_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                  Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                  curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                                  Tensor1DWrap<float, int32_t> sigma, int rows, int cols, int channel)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channel, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % cols;
        int dst_y0  = offset / cols;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % cols;
        int dst_y1  = offset1 / cols;
        gaussian_noise_store_channel_pair<false, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                    channel, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_kernel(const Tensor3DWrap<T, StrideType> src, Tensor3DWrap<T, StrideType> dst,
                                            curandState *state, curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                            Tensor1DWrap<float, int32_t> sigma, int rows, int cols)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int    dst_x0                       = offset % cols;
        int    dst_y0                       = offset / cols;
        int    offset1                      = offset + blockDim.x;
        int    dst_x1                       = offset1 % cols;
        int    dst_y1                       = offset1 / cols;
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
        int   dst_x                       = offset % cols;
        int   dst_y                       = offset / cols;
        float rand                        = curand_normal(&localState);
        float delta                       = batchMu + rand * batchSigma;
        T     out                         = SaturateCast<T>(*src.ptr(batch_idx, dst_y, dst_x) + delta);
        *dst.ptr(batch_idx, dst_y, dst_x) = clamp(StaticCast<float>(out), 0.f, 1.f);
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, total_size, segmentEnd, localState);
}

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                        Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                        curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                                        Tensor1DWrap<float, int32_t> sigma, int rows, int cols,
                                                        int channel)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batch_idx];
    const float batchSigma = sigma[batch_idx];
    curandState localState;
    if (!gaussian_noise_segment_state(state, id, total_size, channel, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dst_x0  = offset % cols;
        int dst_y0  = offset / cols;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % cols;
        int dst_y1  = offset1 / cols;
        gaussian_noise_store_channel_pair<false, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                    channel, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_planar_kernel(const Tensor4DWrap<T, StrideType> src, Tensor4DWrap<T, StrideType> dst,
                                             curandState *state, curandState *nextState,
                                             Tensor1DWrap<float, int32_t> mu, Tensor1DWrap<float, int32_t> sigma,
                                             int rows, int cols, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int    dst_x0  = offset % cols;
        int    dst_y0  = offset / cols;
        int    offset1 = offset + blockDim.x;
        int    dst_x1  = offset1 % cols;
        int    dst_y1  = offset1 / cols;
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
        int   dst_x = offset % cols;
        int   dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_planar_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                         Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                         curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                                         Tensor1DWrap<float, int32_t> sigma, int rows, int cols,
                                                         int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int dst_x0  = offset % cols;
        int dst_y0  = offset / cols;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % cols;
        int dst_y1  = offset1 / cols;
        gaussian_noise_store_channel_pair<true, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                   channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_planar_kernel(const Tensor4DWrap<T, StrideType> src,
                                                   Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                   curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                                   Tensor1DWrap<float, int32_t> sigma, int rows, int cols, int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int    dst_x0  = offset % cols;
        int    dst_y0  = offset / cols;
        int    offset1 = offset + blockDim.x;
        int    dst_x1  = offset1 % cols;
        int    dst_y1  = offset1 / cols;
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
        int   dst_x = offset % cols;
        int   dst_y = offset / cols;
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

template<typename T, typename StrideType>
__global__ void gaussian_noise_float_planar_per_channel_kernel(const Tensor4DWrap<T, StrideType> src,
                                                               Tensor4DWrap<T, StrideType> dst, curandState *state,
                                                               curandState *nextState, Tensor1DWrap<float, int32_t> mu,
                                                               Tensor1DWrap<float, int32_t> sigma, int rows, int cols,
                                                               int channels)
{
    int         batch_idx  = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         total_size = rows * cols;
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
        int dst_x0  = offset % cols;
        int dst_y0  = offset / cols;
        int offset1 = offset + blockDim.x;
        int dst_x1  = offset1 % cols;
        int dst_y1  = offset1 / cols;
        gaussian_noise_store_channel_pair<true, T>(src, dst, localState, batch_idx, dst_y0, dst_x0, dst_y1, dst_x1,
                                                   channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dst_x = offset % cols;
        int dst_y = offset / cols;
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

template<bool Clip, typename T>
struct GaussianNoiseScalarOutput;

template<bool Clip>
struct GaussianNoiseScalarOutput<Clip, uchar>
{
    static __device__ uchar Convert(uchar value, float delta)
    {
        // Torchvision converts the generated noise to int16 before adding it to
        // uint8 input. The conversion truncates toward zero; the final uint8
        // conversion either clamps or wraps depending on clip.
        int noise = static_cast<int>(static_cast<short>(__float2int_rz(delta)));
        int sum   = static_cast<int>(value) + noise;
        if constexpr (Clip)
        {
            sum = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
        }
        return static_cast<uchar>(sum);
    }
};

template<bool Clip>
struct GaussianNoiseScalarOutput<Clip, float>
{
    __device__ static float Convert(float value, float delta)
    {
        float out = value + delta;
        if constexpr (Clip)
        {
            out = clamp(out, 0.f, 1.f);
        }
        return out;
    }
};

template<bool Planar, bool Clip, typename T, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void gaussian_noise_scalar_store_channel(const SrcWrap &src, const DstWrap &dst,
                                                                    int batchIdx, int y, int x, int ch, float delta)
{
    if constexpr (Planar)
    {
        *dst.ptr(batchIdx, ch, y, x) = GaussianNoiseScalarOutput<Clip, T>::Convert(*src.ptr(batchIdx, ch, y, x), delta);
    }
    else
    {
        *dst.ptr(batchIdx, y, x, ch) = GaussianNoiseScalarOutput<Clip, T>::Convert(*src.ptr(batchIdx, y, x, ch), delta);
    }
}

template<bool Planar, bool Clip, typename T, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void gaussian_noise_scalar_store_channel_pair(const SrcWrap &src, const DstWrap &dst,
                                                                         curandState &localState, int batchIdx, int y0,
                                                                         int x0, int y1, int x1, int channels, float mu,
                                                                         float sigma)
{
    int ch = 0;
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y0, x0, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y1, x1, 0, mu + rand.y * sigma);
        ch = 1;
    }
    else
    {
        ch = 0;
    }
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y1, x1, ch, mu + rand.x * sigma);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y1, x1, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float rand = curand_normal(&localState);
        gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y1, x1, ch, mu + rand * sigma);
    }
}

template<bool Planar, bool PerChannel, bool Clip, typename T, typename SrcWrap, typename DstWrap>
__global__ void gaussian_noise_scalar_kernel(const SrcWrap src, DstWrap dst, curandState *state, curandState *nextState,
                                             float mu, float sigma, int rows, int cols, int channels)
{
    int           batchIdx   = blockIdx.x;
    int           id         = threadIdx.x + blockIdx.x * blockDim.x;
    int           totalSize  = rows * cols;
    int           offset     = 0;
    int           segmentEnd = 0;
    curandState   localState;
    constexpr int kSharedNoiseRngs = 1;
    int           rngsPerPixel     = PerChannel ? channels : kSharedNoiseRngs;
    if (!gaussian_noise_segment_state(state, id, totalSize, rngsPerPixel, offset, segmentEnd, localState))
    {
        return;
    }

    while (offset + blockDim.x < segmentEnd)
    {
        int x0      = offset % cols;
        int y0      = offset / cols;
        int offset1 = offset + blockDim.x;
        int x1      = offset1 % cols;
        int y1      = offset1 / cols;
        if constexpr (PerChannel)
        {
            gaussian_noise_scalar_store_channel_pair<Planar, Clip, T>(src, dst, localState, batchIdx, y0, x0, y1, x1,
                                                                      channels, mu, sigma);
        }
        else
        {
            float2 rand   = gaussian_noise_normal2(localState);
            float  delta0 = mu + rand.x * sigma;
            float  delta1 = mu + rand.y * sigma;
            for (int ch = 0; ch < channels; ++ch)
            {
                gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y0, x0, ch, delta0);
                gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y1, x1, ch, delta1);
            }
        }
        offset += 2 * blockDim.x;
    }

    while (offset < segmentEnd)
    {
        int x = offset % cols;
        int y = offset / cols;
        if constexpr (PerChannel)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                float rand = curand_normal(&localState);
                gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y, x, ch, mu + rand * sigma);
            }
        }
        else
        {
            float rand  = curand_normal(&localState);
            float delta = mu + rand * sigma;
            for (int ch = 0; ch < channels; ++ch)
            {
                gaussian_noise_scalar_store_channel<Planar, Clip, T>(src, dst, batchIdx, y, x, ch, delta);
            }
        }
        offset += blockDim.x;
    }
    gaussian_noise_store_segment_state(state, nextState, id, totalSize, segmentEnd, localState);
}

template<bool Planar, bool PerChannel, bool Clip, typename T, typename StrideType = int32_t>
void launch_gaussian_noise_scalar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  int batch, int channels, int rows, int cols, curandState *states,
                                  curandState *nextStates, float mu, float sigma, cudaStream_t stream)
{
    int totalSize = rows * cols;
    int segments  = gaussian_noise_segments(totalSize, BLOCK);
    gaussian_noise_copy_segment_states(nextStates, states, batch, BLOCK, segments, stream);

    if constexpr (Planar)
    {
        auto src = CreateTensorWrapNCHW<T, StrideType>(inData);
        auto dst = CreateTensorWrapNCHW<T, StrideType>(outData);
        gaussian_noise_scalar_kernel<Planar, PerChannel, Clip, T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
            src, dst, states, nextStates, mu, sigma, rows, cols, channels);
    }
    else
    {
        auto src = CreateTensorWrapNHWC<T, StrideType>(inData);
        auto dst = CreateTensorWrapNHWC<T, StrideType>(outData);
        gaussian_noise_scalar_kernel<Planar, PerChannel, Clip, T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
            src, dst, states, nextStates, mu, sigma, rows, cols, channels);
    }

    checkKernelErrors();
    gaussian_noise_copy_segment_states(states, nextStates, batch, BLOCK, segments, stream);
}

template<typename T>
void dispatch_gaussian_noise_scalar(const nvcv::TensorDataStridedCuda &inData,
                                    const nvcv::TensorDataStridedCuda &outData, int batch, int channels, int rows,
                                    int cols, curandState *states, curandState *nextStates, float mu, float sigma,
                                    bool planar, bool perChannel, bool clip, cudaStream_t stream)
{
#define CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(PLANAR, PER_CHANNEL, CLIP)                                               \
    launch_gaussian_noise_scalar<PLANAR, PER_CHANNEL, CLIP, T>(inData, outData, batch, channels, rows, cols, states, \
                                                               nextStates, mu, sigma, stream)

    if (planar)
    {
        if (perChannel)
        {
            clip ? CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(true, true, true)
                 : CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(true, true, false);
        }
        else
        {
            clip ? CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(true, false, true)
                 : CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(true, false, false);
        }
    }
    else if (perChannel)
    {
        clip ? CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(false, true, true)
             : CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(false, true, false);
    }
    else
    {
        clip ? CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(false, false, true)
             : CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH(false, false, false);
    }

#undef CVCUDA_GAUSSIAN_NOISE_SCALAR_LAUNCH
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out, int batch,
                    int rows, int cols, curandState *m_states, curandState *m_nextStates,
                    const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                    cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_per_channel(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out,
                                int batch, int channels, int rows, int cols, curandState *m_states,
                                curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                                const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHWC<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHWC<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_per_channel_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out, int batch,
                          int rows, int cols, curandState *m_states, curandState *m_nextStates,
                          const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                          cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_kernel<T>
        <<<dim3(batch, segments), BLOCK, 0, stream>>>(src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float_per_channel(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out,
                                      int batch, int channels, int rows, int cols, curandState *m_states,
                                      curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                                      const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNHWC<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNHWC<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_per_channel_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_planar(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out, int batch,
                           int channels, int rows, int cols, curandState *m_states, curandState *m_nextStates,
                           const nvcv::TensorDataStridedCuda &_mu, const nvcv::TensorDataStridedCuda &_sigma,
                           cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNCHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNCHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_planar_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_planar_per_channel(const nvcv::TensorDataStridedCuda &d_in,
                                       const nvcv::TensorDataStridedCuda &d_out, int batch, int channels, int rows,
                                       int cols, curandState *m_states, curandState *m_nextStates,
                                       const nvcv::TensorDataStridedCuda &_mu,
                                       const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNCHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNCHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_planar_per_channel_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float_planar(const nvcv::TensorDataStridedCuda &d_in, const nvcv::TensorDataStridedCuda &d_out,
                                 int batch, int channels, int rows, int cols, curandState *m_states,
                                 curandState *m_nextStates, const nvcv::TensorDataStridedCuda &_mu,
                                 const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNCHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNCHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_planar_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

template<typename T, typename StrideType = int32_t>
void gaussian_noise_float_planar_per_channel(const nvcv::TensorDataStridedCuda &d_in,
                                             const nvcv::TensorDataStridedCuda &d_out, int batch, int channels,
                                             int rows, int cols, curandState *m_states, curandState *m_nextStates,
                                             const nvcv::TensorDataStridedCuda &_mu,
                                             const nvcv::TensorDataStridedCuda &_sigma, cudaStream_t stream)
{
    auto                         src_ptr = CreateTensorWrapNCHW<T, StrideType>(d_in);
    auto                         dst_ptr = CreateTensorWrapNCHW<T, StrideType>(d_out);
    Tensor1DWrap<float, int32_t> mu(_mu);
    Tensor1DWrap<float, int32_t> sigma(_sigma);

    int total_size = rows * cols;
    int segments   = gaussian_noise_segments(total_size, BLOCK);

    gaussian_noise_copy_segment_states(m_nextStates, m_states, batch, BLOCK, segments, stream);
    gaussian_noise_float_planar_per_channel_kernel<T><<<dim3(batch, segments), BLOCK, 0, stream>>>(
        src_ptr, dst_ptr, m_states, m_nextStates, mu, sigma, rows, cols, channels);
    checkKernelErrors();
    gaussian_noise_copy_segment_states(m_states, m_nextStates, batch, BLOCK, segments, stream);
}

namespace nvcv::legacy::cuda_op {

GaussianNoise::GaussianNoise(DataShape max_input_shape, DataShape max_output_shape, int maxBatchSize)
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
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Parameter error!");
    }
    const size_t stateBytes
        = cvcuda::priv::CheckedMulMany({sizeof(curandState), static_cast<size_t>(BLOCK),
                                        cvcuda::priv::CheckedNonNegativeToSize(maxBatchSize, "maxBatchSize")},
                                       "GaussianNoise curand state allocation size overflow");
    const size_t allocationBytes
        = cvcuda::priv::CheckedMulMany({stateBytes, 2}, "GaussianNoise curand state allocation size overflow");
    cudaError_t err = cudaMalloc((void **)&m_states, allocationBytes);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << allocationBytes);
        throw LegacyCudaAllocationError("CUDA memory allocation error!");
    }
    m_nextStates = maxBatchSize > 0 ? m_states + static_cast<size_t>(BLOCK) * maxBatchSize : m_states;
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

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    int channels = inAccess->numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int batch = inAccess->numSamples();
    if (batch > m_maxBatchSize)
    {
        LOG_ERROR("Input batch exceeds maxBatchSize");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inMaxStride = inAccess->sampleStride() * batch;
    if (inMaxStride > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);
    if (isPlanar
        && (outAccess->numSamples() != batch || outAccess->numChannels() != channels
            || outAccess->numRows() != inAccess->numRows() || outAccess->numCols() != inAccess->numCols()))
    {
        LOG_ERROR("Planar input and output must have matching sample, channel, height, and width");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (isPlanar && batch > 1
        && (inAccess->sampleStride() != channels * inAccess->chStride()
            || outAccess->sampleStride() != channels * outAccess->chStride()))
    {
        LOG_ERROR("Planar GaussianNoise of a batched tensor requires tightly packed channel planes");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (isPlanar && static_cast<int64_t>(batch) * channels > 65535)
    {
        LOG_ERROR("Planar GaussianNoise requires numSamples * channels <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_PARAMETER;
    }

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
                               int channels, int rows, int cols, curandState *m_states, curandState *m_nextStates,
                               const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma,
                               cudaStream_t stream);

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
            func(inData, outData, batch, channels, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates, mu,
                 sigma, stream);
        }
        else
        {
            const func_t func = isPlanar ? planar_funcs[in_data_type] : funcs[in_data_type];
            assert(func != 0);
            func(inData, outData, batch, channels, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates, mu,
                 sigma, stream);
        }
    }
    else
    {
        typedef void (*func_t)(const TensorDataStridedCuda &d_in, const TensorDataStridedCuda &d_out, int batch,
                               int rows, int cols, curandState *m_states, curandState *m_nextStates,
                               const TensorDataStridedCuda &mu, const TensorDataStridedCuda &sigma,
                               cudaStream_t stream);
        typedef void (*planar_func_t)(const TensorDataStridedCuda &d_in, const TensorDataStridedCuda &d_out, int batch,
                                      int channels, int rows, int cols, curandState *m_states,
                                      curandState *m_nextStates, const TensorDataStridedCuda &mu,
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
                func(inData, outData, batch, channels, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates,
                     mu, sigma, stream);
            }
            else
            {
                const func_t func = float_funcs[channels - 1];
                assert(func != 0);
                func(inData, outData, batch, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates, mu,
                     sigma, stream);
            }
        }
        else
        {
            if (isPlanar)
            {
                const planar_func_t func = planar_funcs[in_data_type];
                assert(func != 0);
                func(inData, outData, batch, channels, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates,
                     mu, sigma, stream);
            }
            else
            {
                const func_t func = funcs[in_data_type][channels - 1];
                assert(func != 0);
                func(inData, outData, batch, inAccess->numRows(), inAccess->numCols(), m_states, m_nextStates, mu,
                     sigma, stream);
            }
        }
    }
    return SUCCESS;
}

ErrorCode GaussianNoise::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData, float mu,
                               float sigma, bool per_channel, unsigned long long seed, bool reseed, bool clip,
                               cudaStream_t stream)
{
    DataFormat inFormat  = GetLegacyDataFormat(inData.layout());
    DataFormat outFormat = GetLegacyDataFormat(outData.layout());
    if (!(inFormat == kNHWC || inFormat == kHWC || inFormat == kNCHW || inFormat == kCHW) || inFormat != outFormat)
    {
        LOG_ERROR("Input and output must have the same NHWC, HWC, NCHW, or CHW layout");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess  = TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess)
    {
        LOG_ERROR("Input and output must be image tensors");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = inFormat == kNCHW || inFormat == kCHW;
    const int  channels = inAccess->numChannels();
    const int  batch    = inAccess->numSamples();
    if (channels < 1 || channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (batch > m_maxBatchSize)
    {
        LOG_ERROR("Input batch exceeds maxBatchSize");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (outAccess->numSamples() != batch || outAccess->numChannels() != channels
        || outAccess->numRows() != inAccess->numRows() || outAccess->numCols() != inAccess->numCols())
    {
        LOG_ERROR("Input and output must have identical shape");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (isPlanar && batch > 1
        && (inAccess->sampleStride() != channels * inAccess->chStride()
            || outAccess->sampleStride() != channels * outAccess->chStride()))
    {
        LOG_ERROR("Planar GaussianNoise of a batched tensor requires tightly packed channel planes");
        return ErrorCode::INVALID_PARAMETER;
    }
    if (isPlanar && static_cast<int64_t>(batch) * channels > 65535)
    {
        LOG_ERROR("Planar GaussianNoise requires numSamples * channels <= 65535");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inMaxStride  = inAccess->sampleStride() * batch;
    auto outMaxStride = outAccess->sampleStride() * batch;
    if (inMaxStride > cuda::TypeTraits<int32_t>::max || outMaxStride > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output tensor is too large for 32-bit indexing");
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType inType  = GetLegacyDataType(inData.dtype());
    DataType outType = GetLegacyDataType(outData.dtype());
    if (!(inType == kCV_8U || inType == kCV_32F) || inType != outType)
    {
        LOG_ERROR("Scalar GaussianNoise supports matching U8 or F32 input and output");
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (sigma < 0.f)
    {
        LOG_ERROR("sigma must be non-negative");
        return ErrorCode::INVALID_PARAMETER;
    }

    if (reseed || !m_setupDone)
    {
        m_seed = seed;
        setup_gaussian_rand_kernel<<<m_maxBatchSize, BLOCK, 0, stream>>>(m_states, m_seed);
        m_setupDone = true;
    }

    if (inType == kCV_32F)
    {
        dispatch_gaussian_noise_scalar<float>(inData, outData, batch, channels, inAccess->numRows(),
                                              inAccess->numCols(), m_states, m_nextStates, mu, sigma, isPlanar,
                                              per_channel, clip, stream);
    }
    else
    {
        dispatch_gaussian_noise_scalar<uchar>(inData, outData, batch, channels, inAccess->numRows(),
                                              inAccess->numCols(), m_states, m_nextStates, mu, sigma, isPlanar,
                                              per_channel, clip, stream);
    }
    return SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
