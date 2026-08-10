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

#ifndef GAUSSIAN_NOISE_UTIL_CUH
#define GAUSSIAN_NOISE_UTIL_CUH

#include "CvCudaUtils.cuh"

#include <curand_kernel.h>

__global__ void setup_gaussian_rand_kernel(curandState *state, unsigned long long seed);

// Tuned pixels-per-RNG-segment: balances per-thread RNG state/register pressure
// with enough segmented work to keep target GPUs occupied. Retune if the RNG
// state layout or the per-pixel RNG consumption changes.
constexpr int kGaussianNoiseSegmentPixels = 128;

inline int gaussian_noise_segments(int totalSize, int blockSize)
{
    int segments
        = (totalSize + blockSize * kGaussianNoiseSegmentPixels - 1) / (blockSize * kGaussianNoiseSegmentPixels);
    return segments > 0 ? segments : 1;
}

__device__ __forceinline__ bool gaussian_noise_segment_range(int totalSize, int &segmentStart, int &offset,
                                                             int &segmentEnd)
{
    segmentStart = blockIdx.y * kGaussianNoiseSegmentPixels;
    offset       = threadIdx.x + segmentStart * blockDim.x;
    if (offset >= totalSize)
    {
        return false;
    }

    segmentEnd = threadIdx.x + (segmentStart + kGaussianNoiseSegmentPixels) * blockDim.x;
    segmentEnd = segmentEnd < totalSize ? segmentEnd : totalSize;
    return true;
}

__device__ __forceinline__ void gaussian_noise_advance_normal_state(unsigned long long normalCount,
                                                                    curandState       &localState)
{
    if (normalCount == 0)
    {
        return;
    }

    // curand's Box-Muller path caches one normal sample; consume it before
    // skipahead so localState.boxmuller_flag / EXTRA_FLAG_NORMAL advancement
    // stays deterministic. Re-check this if curand_normal internals change.
    if (localState.boxmuller_flag == EXTRA_FLAG_NORMAL)
    {
        (void)curand_normal(&localState);
        if (--normalCount == 0)
        {
            return;
        }
    }

    if ((normalCount & 1) == 0)
    {
        skipahead(normalCount, &localState);
    }
    else
    {
        if (normalCount > 1)
        {
            skipahead(normalCount - 1, &localState);
        }
        (void)curand_normal(&localState);
    }
}

__device__ __forceinline__ unsigned long long gaussian_noise_thread_pixels(int totalSize)
{
    return threadIdx.x < totalSize ? 1ull + static_cast<unsigned long long>(totalSize - 1 - threadIdx.x) / blockDim.x
                                   : 0ull;
}

__device__ __forceinline__ bool gaussian_noise_segment_state(curandState *state, int id, int totalSize,
                                                             int rngsPerPixel, int &offset, int &segmentEnd,
                                                             curandState &localState)
{
    int segmentStart;
    if (!gaussian_noise_segment_range(totalSize, segmentStart, offset, segmentEnd))
    {
        return false;
    }

    // Segmented launches share state[id] across gridDim.y, so each segment
    // advances a localState by its deterministic rngsPerPixel offset.
    localState = state[id];
    if (segmentStart > 0)
    {
        gaussian_noise_advance_normal_state(static_cast<unsigned long long>(segmentStart) * rngsPerPixel, localState);
    }
    return true;
}

__device__ __forceinline__ void gaussian_noise_store_segment_state(curandState *state, curandState *nextState, int id,
                                                                   int totalSize, int segmentEnd,
                                                                   curandState localState)
{
    if (gridDim.y == 1)
    {
        state[id] = localState;
    }
    else if (segmentEnd == totalSize)
    {
        // Exactly one segment per RNG thread reaches the end of its range.
        // Publish to a separate buffer so all segments keep reading the same
        // immutable starting state.
        nextState[id] = localState;
    }
}

inline void gaussian_noise_copy_segment_states(curandState *dst, const curandState *src, int batch, int blockSize,
                                               int segments, cudaStream_t stream)
{
    if (segments <= 1)
    {
        return;
    }

    const size_t stateBytes = static_cast<size_t>(batch) * blockSize * sizeof(curandState);
    nvcv::legacy::cuda_op::__checkCudaErrors(cudaMemcpyAsync(dst, src, stateBytes, cudaMemcpyDeviceToDevice, stream),
                                             __FILE__, __LINE__);
}

__device__ __forceinline__ float2 gaussian_noise_normal2(curandState &localState)
{
    if (localState.boxmuller_flag == EXTRA_FLAG_NORMAL)
    {
        return {curand_normal(&localState), curand_normal(&localState)};
    }
    return curand_normal2(&localState);
}

template<typename T>
__device__ __forceinline__ T gaussian_noise_saturate(T value, float delta)
{
    return nvcv::cuda::SaturateCast<T>(value + delta);
}

template<>
__device__ __forceinline__ float gaussian_noise_saturate<float>(float value, float delta)
{
    return nvcv::cuda::clamp(value + delta, 0.f, 1.f);
}

template<bool Planar, typename T, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void gaussian_noise_store_channel(const SrcWrap &src, const DstWrap &dst, int batchIdx,
                                                             int y, int x, int ch, float delta)
{
    if constexpr (Planar)
    {
        *dst.ptr(batchIdx, ch, y, x) = gaussian_noise_saturate<T>(*src.ptr(batchIdx, ch, y, x), delta);
    }
    else
    {
        *dst.ptr(batchIdx, y, x, ch) = gaussian_noise_saturate<T>(*src.ptr(batchIdx, y, x, ch), delta);
    }
}

template<bool Planar, typename T, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void gaussian_noise_store_channel_pair(const SrcWrap &src, const DstWrap &dst,
                                                                  curandState &localState, int batchIdx, int y0, int x0,
                                                                  int y1, int x1, int channels, float mu, float sigma)
{
    int ch = 0;
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y0, x0, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y1, x1, 0, mu + rand.y * sigma);
        ch = 1;
    }
    else
    {
        ch = 0;
    }
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = gaussian_noise_normal2(localState);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y1, x1, ch, mu + rand.x * sigma);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y1, x1, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float rand = curand_normal(&localState);
        gaussian_noise_store_channel<Planar, T>(src, dst, batchIdx, y1, x1, ch, mu + rand * sigma);
    }
}

#endif // GAUSSIAN_NOISE_UTIL_CUH
