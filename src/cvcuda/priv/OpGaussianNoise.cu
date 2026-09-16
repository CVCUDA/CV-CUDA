/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpGaussianNoise.hpp"
#include "SafeSize.hpp"

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cuda = nvcv::cuda;

namespace {

using uchar  = unsigned char;
using ushort = unsigned short;

// Threads per block for the seeding kernel and every noise kernel.  It also sets how many
// curandState entries one sample owns, so the seeding grid and the noise grid stay in step.
constexpr int kBlockSize = 512;

// Tuned pixels-per-RNG-segment: balances per-thread RNG state/register pressure
// with enough segmented work to keep target GPUs occupied. Retune if the RNG
// state layout or the per-pixel RNG consumption changes.
constexpr int kSegmentPixels = 128;

__global__ void SetupGaussianRandKernel(curandState *state, unsigned long long seed)
{
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

inline int GaussianNoiseSegments(int totalSize, int blockSize)
{
    int segments = nvcv::util::DivUp(totalSize, blockSize * kSegmentPixels);
    return segments > 0 ? segments : 1;
}

__device__ __forceinline__ bool GaussianNoiseSegmentRange(int totalSize, int &segmentStart, int &offset,
                                                          int &segmentEnd)
{
    segmentStart = blockIdx.y * kSegmentPixels;
    offset       = threadIdx.x + segmentStart * blockDim.x;
    if (offset >= totalSize)
    {
        return false;
    }

    segmentEnd = threadIdx.x + (segmentStart + kSegmentPixels) * blockDim.x;
    segmentEnd = segmentEnd < totalSize ? segmentEnd : totalSize;
    return true;
}

__device__ __forceinline__ void GaussianNoiseAdvanceNormalState(unsigned long long normalCount, curandState &localState)
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

__device__ __forceinline__ bool GaussianNoiseSegmentState(curandState *state, int id, int totalSize, int rngsPerPixel,
                                                          int &offset, int &segmentEnd, curandState &localState)
{
    int segmentStart;
    if (!GaussianNoiseSegmentRange(totalSize, segmentStart, offset, segmentEnd))
    {
        return false;
    }

    // Segmented launches share state[id] across gridDim.y, so each segment
    // advances a localState by its deterministic rngsPerPixel offset.
    localState = state[id];
    if (segmentStart > 0)
    {
        GaussianNoiseAdvanceNormalState(static_cast<unsigned long long>(segmentStart) * rngsPerPixel, localState);
    }
    return true;
}

__device__ __forceinline__ void GaussianNoiseStoreSegmentState(curandState *state, curandState *nextState, int id,
                                                               int totalSize, int segmentEnd, curandState localState)
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

inline void GaussianNoiseCopySegmentStates(curandState *dst, const curandState *src, int batch, int segments,
                                           cudaStream_t stream)
{
    if (segments <= 1)
    {
        return;
    }

    const size_t stateBytes = static_cast<size_t>(batch) * kBlockSize * sizeof(curandState);
    NVCV_CHECK_THROW(cudaMemcpyAsync(dst, src, stateBytes, cudaMemcpyDeviceToDevice, stream));
}

// Every noise launch is bracketed the same way: seed the staging half from the live states, run the
// kernel, then adopt whatever the last segment published.  Both copies are no-ops for a single
// segment, which is the common case.
template<class Launch>
inline void LaunchSegmented(int batch, int segments, curandState *states, curandState *nextStates, cudaStream_t stream,
                            Launch &&launch)
{
    GaussianNoiseCopySegmentStates(nextStates, states, batch, segments, stream);
    launch();
    NVCV_CHECK_THROW(cudaGetLastError());
    GaussianNoiseCopySegmentStates(states, nextStates, batch, segments, stream);
}

__device__ __forceinline__ float2 GaussianNoiseNormal2(curandState &localState)
{
    if (localState.boxmuller_flag == EXTRA_FLAG_NORMAL)
    {
        return {curand_normal(&localState), curand_normal(&localState)};
    }
    return curand_normal2(&localState);
}

// Floating-point output is clipped to the unit range; integer output only saturates.  The legacy
// layer expressed this as separate `_float` kernels dispatched on the same predicate, and admitted
// F16 by routing it to that same family: the sum is taken in float and only the store narrows.
template<class T>
inline constexpr bool kClipToUnitRange
    = std::is_same_v<cuda::BaseType<T>, float> || cuda::detail::IsHalfV<cuda::BaseType<T>>;

template<typename T>
__device__ __forceinline__ T GaussianNoiseSaturate(T value, float delta)
{
    if constexpr (kClipToUnitRange<T>)
    {
        // `value + delta` widens half to float, so the clip and the add both happen in float and
        // the SaturateCast is the single narrowing step (an identity for float).
        return cuda::SaturateCast<T>(cuda::clamp(value + delta, 0.f, 1.f));
    }
    else
    {
        return cuda::SaturateCast<T>(value + delta);
    }
}

// The clip step of the legacy `_float` kernel family: the noisy value has already been saturated
// into T, so clipping goes back through float and narrows again.  The outer SaturateCast is an
// identity for float; for the F16 instantiations it converts the clamped value back to half, and
// because `out` is already half-rounded and 0 and 1 are exact in half it adds no rounding step.
// Kept separate from GaussianNoiseSaturate above, which clips before its single narrowing: the two
// are value-identical but not instruction-identical, and each mirrors the legacy site it came from.
template<typename T>
__device__ __forceinline__ T GaussianNoiseClipToUnitRange(T out)
{
    return cuda::SaturateCast<T>(cuda::clamp(cuda::StaticCast<float>(out), 0.f, 1.f));
}

// How a noisy value becomes a stored one.  The per-sample paths saturate (clipping first for the
// float family); the scalar path has its own per-dtype rounding contract, GaussianNoiseScalarOutput
// below.  Expressing both as a policy lets the two families share one channel-pair helper, so the
// cuRAND draw order -- the operator's bit-exactness contract -- is written down exactly once.
template<typename T>
struct GaussianNoiseSaturateOutput
{
    static __device__ __forceinline__ T Convert(T value, float delta)
    {
        return GaussianNoiseSaturate<T>(value, delta);
    }
};

template<bool Planar, class Out, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void GaussianNoiseStoreChannel(const SrcWrap &src, const DstWrap &dst, int batchIdx, int y,
                                                          int x, int ch, float delta)
{
    if constexpr (Planar)
    {
        *dst.ptr(batchIdx, ch, y, x) = Out::Convert(*src.ptr(batchIdx, ch, y, x), delta);
    }
    else
    {
        *dst.ptr(batchIdx, y, x, ch) = Out::Convert(*src.ptr(batchIdx, y, x, ch), delta);
    }
}

// The channel-pair draw order, shared by both families: two pixels' worth of channels consumed from
// paired normal draws, with the odd-channel case splicing one pair across the two pixels.
template<bool Planar, class Out, typename SrcWrap, typename DstWrap>
__device__ __forceinline__ void GaussianNoiseStoreChannelPair(const SrcWrap &src, const DstWrap &dst,
                                                              curandState &localState, int batchIdx, int y0, int x0,
                                                              int y1, int x1, int channels, float mu, float sigma)
{
    int ch = 0;
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = GaussianNoiseNormal2(localState);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y0, x0, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float2 rand = GaussianNoiseNormal2(localState);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y0, x0, ch, mu + rand.x * sigma);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y1, x1, 0, mu + rand.y * sigma);
        ch = 1;
    }
    else
    {
        ch = 0;
    }
    for (; ch + 1 < channels; ch += 2)
    {
        float2 rand = GaussianNoiseNormal2(localState);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y1, x1, ch, mu + rand.x * sigma);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y1, x1, ch + 1, mu + rand.y * sigma);
    }
    if (ch < channels)
    {
        float rand = curand_normal(&localState);
        GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y1, x1, ch, mu + rand * sigma);
    }
}

template<class Wrap>
struct IsVarShapeWrap : std::false_type
{
};

template<class T>
struct IsVarShapeWrap<cuda::ImageBatchVarShapeWrap<T>> : std::true_type
{
};

template<class T>
struct IsVarShapeWrap<cuda::ImageBatchVarShapeWrapNHWC<T>> : std::true_type
{
};

// A var-shape batch carries its geometry per image, a tensor carries it once for the whole batch;
// that difference is all that separated the two kernel families.
template<class DstWrap>
__device__ __forceinline__ void GaussianNoiseExtent(const DstWrap &dst, int batchIdx, int rows, int cols, int &width,
                                                    int &totalSize)
{
    if constexpr (IsVarShapeWrap<DstWrap>::value)
    {
        width     = dst.width(batchIdx);
        totalSize = dst.height(batchIdx) * width;
    }
    else
    {
        width     = cols;
        totalSize = rows * cols;
    }
}

// One noise value per pixel, interleaved.  T is the whole pixel (e.g. uchar3), so the channels are
// loaded and stored as one vector.
template<class T, class SrcWrap, class DstWrap, class MuWrap>
__global__ void GaussianNoiseSharedKernel(const SrcWrap src, DstWrap dst, curandState *state, curandState *nextState,
                                          MuWrap mu, MuWrap sigma, int rows, int cols)
{
    int batchIdx = blockIdx.x;
    int id       = threadIdx.x + blockIdx.x * blockDim.x;
    int width, totalSize;
    GaussianNoiseExtent(dst, batchIdx, rows, cols, width, totalSize);
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batchIdx];
    const float batchSigma = sigma[batchIdx];
    curandState localState;
    if (!GaussianNoiseSegmentState(state, id, totalSize, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dstX0   = offset % width;
        int    dstY0   = offset / width;
        int    offset1 = offset + blockDim.x;
        int    dstX1   = offset1 % width;
        int    dstY1   = offset1 / width;
        float2 rand    = GaussianNoiseNormal2(localState);
        float  delta0  = batchMu + rand.x * batchSigma;
        float  delta1  = batchMu + rand.y * batchSigma;
        if constexpr (kClipToUnitRange<T>)
        {
            T out0                           = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY0, dstX0) + delta0);
            T out1                           = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY1, dstX1) + delta1);
            *dst.ptr(batchIdx, dstY0, dstX0) = GaussianNoiseClipToUnitRange(out0);
            *dst.ptr(batchIdx, dstY1, dstX1) = GaussianNoiseClipToUnitRange(out1);
        }
        else
        {
            *dst.ptr(batchIdx, dstY0, dstX0) = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY0, dstX0) + delta0);
            *dst.ptr(batchIdx, dstY1, dstX1) = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY1, dstX1) + delta1);
        }
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dstX  = offset % width;
        int   dstY  = offset / width;
        float rand  = curand_normal(&localState);
        float delta = batchMu + rand * batchSigma;
        if constexpr (kClipToUnitRange<T>)
        {
            T out                          = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY, dstX) + delta);
            *dst.ptr(batchIdx, dstY, dstX) = GaussianNoiseClipToUnitRange(out);
        }
        else
        {
            *dst.ptr(batchIdx, dstY, dstX) = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY, dstX) + delta);
        }
        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
}

// One noise value per channel, interleaved.  T is a single channel and the wrap is NHWC.
template<class T, class SrcWrap, class DstWrap, class MuWrap>
__global__ void GaussianNoisePerChannelKernel(const SrcWrap src, DstWrap dst, curandState *state,
                                              curandState *nextState, MuWrap mu, MuWrap sigma, int rows, int cols,
                                              int channels)
{
    int batchIdx = blockIdx.x;
    int id       = threadIdx.x + blockIdx.x * blockDim.x;
    int width, totalSize;
    GaussianNoiseExtent(dst, batchIdx, rows, cols, width, totalSize);
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batchIdx];
    const float batchSigma = sigma[batchIdx];
    curandState localState;
    if (!GaussianNoiseSegmentState(state, id, totalSize, channels, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dstX0   = offset % width;
        int dstY0   = offset / width;
        int offset1 = offset + blockDim.x;
        int dstX1   = offset1 % width;
        int dstY1   = offset1 / width;
        GaussianNoiseStoreChannelPair<false, GaussianNoiseSaturateOutput<T>>(
            src, dst, localState, batchIdx, dstY0, dstX0, dstY1, dstX1, channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dstX = offset % width;
        int dstY = offset / width;
        for (int ch = 0; ch < channels; ch++)
        {
            float rand  = curand_normal(&localState);
            float delta = batchMu + rand * batchSigma;
            if constexpr (kClipToUnitRange<T>)
            {
                T out                              = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY, dstX, ch) + delta);
                *dst.ptr(batchIdx, dstY, dstX, ch) = GaussianNoiseClipToUnitRange(out);
            }
            else
            {
                *dst.ptr(batchIdx, dstY, dstX, ch) = cuda::SaturateCast<T>(*src.ptr(batchIdx, dstY, dstX, ch) + delta);
            }
        }
        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
}

// One noise value per pixel, shared by every plane.
template<class T, class SrcWrap, class DstWrap, class MuWrap>
__global__ void GaussianNoisePlanarSharedKernel(const SrcWrap src, DstWrap dst, curandState *state,
                                                curandState *nextState, MuWrap mu, MuWrap sigma, int rows, int cols,
                                                int channels)
{
    int batchIdx = blockIdx.x;
    int id       = threadIdx.x + blockIdx.x * blockDim.x;
    int width, totalSize;
    GaussianNoiseExtent(dst, batchIdx, rows, cols, width, totalSize);
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batchIdx];
    const float batchSigma = sigma[batchIdx];
    curandState localState;
    if (!GaussianNoiseSegmentState(state, id, totalSize, 1, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int    dstX0   = offset % width;
        int    dstY0   = offset / width;
        int    offset1 = offset + blockDim.x;
        int    dstX1   = offset1 % width;
        int    dstY1   = offset1 / width;
        float2 rand    = GaussianNoiseNormal2(localState);
        float  delta0  = batchMu + rand.x * batchSigma;
        float  delta1  = batchMu + rand.y * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            if constexpr (kClipToUnitRange<T>)
            {
                T out0 = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY0, dstX0) + delta0);
                T out1 = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY1, dstX1) + delta1);
                *dst.ptr(batchIdx, ch, dstY0, dstX0) = GaussianNoiseClipToUnitRange(out0);
                *dst.ptr(batchIdx, ch, dstY1, dstX1) = GaussianNoiseClipToUnitRange(out1);
            }
            else
            {
                *dst.ptr(batchIdx, ch, dstY0, dstX0)
                    = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY0, dstX0) + delta0);
                *dst.ptr(batchIdx, ch, dstY1, dstX1)
                    = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY1, dstX1) + delta1);
            }
        }
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dstX  = offset % width;
        int   dstY  = offset / width;
        float rand  = curand_normal(&localState);
        float delta = batchMu + rand * batchSigma;
        for (int ch = 0; ch < channels; ch++)
        {
            if constexpr (kClipToUnitRange<T>)
            {
                T out                              = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY, dstX) + delta);
                *dst.ptr(batchIdx, ch, dstY, dstX) = GaussianNoiseClipToUnitRange(out);
            }
            else
            {
                *dst.ptr(batchIdx, ch, dstY, dstX) = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY, dstX) + delta);
            }
        }
        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
}

// One noise value per plane and pixel.
template<class T, class SrcWrap, class DstWrap, class MuWrap>
__global__ void GaussianNoisePlanarPerChannelKernel(const SrcWrap src, DstWrap dst, curandState *state,
                                                    curandState *nextState, MuWrap mu, MuWrap sigma, int rows, int cols,
                                                    int channels)
{
    int batchIdx = blockIdx.x;
    int id       = threadIdx.x + blockIdx.x * blockDim.x;
    int width, totalSize;
    GaussianNoiseExtent(dst, batchIdx, rows, cols, width, totalSize);
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batchIdx];
    const float batchSigma = sigma[batchIdx];
    curandState localState;
    if (!GaussianNoiseSegmentState(state, id, totalSize, channels, offset, segmentEnd, localState))
    {
        return;
    }
    while (offset + blockDim.x < segmentEnd)
    {
        int dstX0   = offset % width;
        int dstY0   = offset / width;
        int offset1 = offset + blockDim.x;
        int dstX1   = offset1 % width;
        int dstY1   = offset1 / width;
        GaussianNoiseStoreChannelPair<true, GaussianNoiseSaturateOutput<T>>(
            src, dst, localState, batchIdx, dstY0, dstX0, dstY1, dstX1, channels, batchMu, batchSigma);
        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int dstX = offset % width;
        int dstY = offset / width;
        for (int ch = 0; ch < channels; ch++)
        {
            float rand  = curand_normal(&localState);
            float delta = batchMu + rand * batchSigma;
            if constexpr (kClipToUnitRange<T>)
            {
                T out                              = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY, dstX) + delta);
                *dst.ptr(batchIdx, ch, dstY, dstX) = GaussianNoiseClipToUnitRange(out);
            }
            else
            {
                *dst.ptr(batchIdx, ch, dstY, dstX) = cuda::SaturateCast<T>(*src.ptr(batchIdx, ch, dstY, dstX) + delta);
            }
        }
        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
}

// Var-shape planar 3-channel specialisation of the kernel above.  With the channel count known at
// compile time the three plane descriptors are read once instead of on every element access, the
// common tightly packed case addresses each plane with the linear pixel index, and the pixel loop
// unrolls four deep.  The RNG stream is unchanged: three paired draws cover two pixels x three
// channels in exactly the order GaussianNoiseStoreChannelPair emits them for channels == 3
// (pixel0 ch0/ch1, pixel0 ch2 + pixel1 ch0, pixel1 ch1/ch2).
template<class T, class MuWrap>
__global__ void GaussianNoisePlanarPerChannel3Kernel(const cuda::ImageBatchVarShapeWrap<T> src,
                                                     cuda::ImageBatchVarShapeWrap<T> dst, curandState *state,
                                                     curandState *nextState, MuWrap mu, MuWrap sigma)
{
    int         batchIdx   = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    const int   width      = dst.width(batchIdx);
    int         totalSize  = dst.height(batchIdx) * width;
    int         offset     = 0;
    int         segmentEnd = 0;
    const float batchMu    = mu[batchIdx];
    const float batchSigma = sigma[batchIdx];
    curandState localState;
    if (!GaussianNoiseSegmentState(state, id, totalSize, 3, offset, segmentEnd, localState))
    {
        // This path adopts the staging half by flipping the two pointers rather than copying, so
        // a thread with no work has to carry its own state over itself.
        if (gridDim.y > 1 && blockIdx.y == 0)
        {
            nextState[id] = state[id];
        }
        return;
    }

    const auto srcPlane0 = src.plane(batchIdx, 0);
    const auto srcPlane1 = src.plane(batchIdx, 1);
    const auto srcPlane2 = src.plane(batchIdx, 2);
    const auto dstPlane0 = dst.plane(batchIdx, 0);
    const auto dstPlane1 = dst.plane(batchIdx, 1);
    const auto dstPlane2 = dst.plane(batchIdx, 2);

    const T *src0 = reinterpret_cast<const T *>(srcPlane0.basePtr);
    const T *src1 = reinterpret_cast<const T *>(srcPlane1.basePtr);
    const T *src2 = reinterpret_cast<const T *>(srcPlane2.basePtr);
    T       *dst0 = reinterpret_cast<T *>(dstPlane0.basePtr);
    T       *dst1 = reinterpret_cast<T *>(dstPlane1.basePtr);
    T       *dst2 = reinterpret_cast<T *>(dstPlane2.basePtr);

    const int packedRowStride = width * static_cast<int>(sizeof(T));
    if (srcPlane0.rowStride == packedRowStride && srcPlane1.rowStride == packedRowStride
        && srcPlane2.rowStride == packedRowStride && dstPlane0.rowStride == packedRowStride
        && dstPlane1.rowStride == packedRowStride && dstPlane2.rowStride == packedRowStride)
    {
        while (offset + 3 * blockDim.x < segmentEnd)
        {
            const int offset1 = offset + blockDim.x;
            const int offset2 = offset + 2 * blockDim.x;
            const int offset3 = offset + 3 * blockDim.x;

            float2 rand01 = GaussianNoiseNormal2(localState);
            float2 rand20 = GaussianNoiseNormal2(localState);
            float2 rand12 = GaussianNoiseNormal2(localState);

            dst0[offset] = GaussianNoiseSaturate<T>(src0[offset], batchMu + rand01.x * batchSigma);
            dst1[offset] = GaussianNoiseSaturate<T>(src1[offset], batchMu + rand01.y * batchSigma);
            dst2[offset] = GaussianNoiseSaturate<T>(src2[offset], batchMu + rand20.x * batchSigma);

            dst0[offset1] = GaussianNoiseSaturate<T>(src0[offset1], batchMu + rand20.y * batchSigma);
            dst1[offset1] = GaussianNoiseSaturate<T>(src1[offset1], batchMu + rand12.x * batchSigma);
            dst2[offset1] = GaussianNoiseSaturate<T>(src2[offset1], batchMu + rand12.y * batchSigma);

            rand01 = GaussianNoiseNormal2(localState);
            rand20 = GaussianNoiseNormal2(localState);
            rand12 = GaussianNoiseNormal2(localState);

            dst0[offset2] = GaussianNoiseSaturate<T>(src0[offset2], batchMu + rand01.x * batchSigma);
            dst1[offset2] = GaussianNoiseSaturate<T>(src1[offset2], batchMu + rand01.y * batchSigma);
            dst2[offset2] = GaussianNoiseSaturate<T>(src2[offset2], batchMu + rand20.x * batchSigma);

            dst0[offset3] = GaussianNoiseSaturate<T>(src0[offset3], batchMu + rand20.y * batchSigma);
            dst1[offset3] = GaussianNoiseSaturate<T>(src1[offset3], batchMu + rand12.x * batchSigma);
            dst2[offset3] = GaussianNoiseSaturate<T>(src2[offset3], batchMu + rand12.y * batchSigma);

            offset += 4 * blockDim.x;
        }
        while (offset + blockDim.x < segmentEnd)
        {
            const int offset1 = offset + blockDim.x;
            float2    rand01  = GaussianNoiseNormal2(localState);
            float2    rand20  = GaussianNoiseNormal2(localState);
            float2    rand12  = GaussianNoiseNormal2(localState);

            dst0[offset] = GaussianNoiseSaturate<T>(src0[offset], batchMu + rand01.x * batchSigma);
            dst1[offset] = GaussianNoiseSaturate<T>(src1[offset], batchMu + rand01.y * batchSigma);
            dst2[offset] = GaussianNoiseSaturate<T>(src2[offset], batchMu + rand20.x * batchSigma);

            dst0[offset1] = GaussianNoiseSaturate<T>(src0[offset1], batchMu + rand20.y * batchSigma);
            dst1[offset1] = GaussianNoiseSaturate<T>(src1[offset1], batchMu + rand12.x * batchSigma);
            dst2[offset1] = GaussianNoiseSaturate<T>(src2[offset1], batchMu + rand12.y * batchSigma);

            offset += 2 * blockDim.x;
        }
        while (offset < segmentEnd)
        {
            float rand0 = curand_normal(&localState);
            float rand1 = curand_normal(&localState);
            float rand2 = curand_normal(&localState);

            dst0[offset] = GaussianNoiseSaturate<T>(src0[offset], batchMu + rand0 * batchSigma);
            dst1[offset] = GaussianNoiseSaturate<T>(src1[offset], batchMu + rand1 * batchSigma);
            dst2[offset] = GaussianNoiseSaturate<T>(src2[offset], batchMu + rand2 * batchSigma);

            offset += blockDim.x;
        }
        GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
        return;
    }

    const int srcStride0 = srcPlane0.rowStride / static_cast<int>(sizeof(T));
    const int srcStride1 = srcPlane1.rowStride / static_cast<int>(sizeof(T));
    const int srcStride2 = srcPlane2.rowStride / static_cast<int>(sizeof(T));
    const int dstStride0 = dstPlane0.rowStride / static_cast<int>(sizeof(T));
    const int dstStride1 = dstPlane1.rowStride / static_cast<int>(sizeof(T));
    const int dstStride2 = dstPlane2.rowStride / static_cast<int>(sizeof(T));

    while (offset + blockDim.x < segmentEnd)
    {
        int    dstX0   = offset % width;
        int    dstY0   = offset / width;
        int    offset1 = offset + blockDim.x;
        int    dstX1   = offset1 % width;
        int    dstY1   = offset1 / width;
        float2 rand01  = GaussianNoiseNormal2(localState);
        float2 rand20  = GaussianNoiseNormal2(localState);
        float2 rand12  = GaussianNoiseNormal2(localState);

        dst0[dstY0 * dstStride0 + dstX0]
            = GaussianNoiseSaturate<T>(src0[dstY0 * srcStride0 + dstX0], batchMu + rand01.x * batchSigma);
        dst1[dstY0 * dstStride1 + dstX0]
            = GaussianNoiseSaturate<T>(src1[dstY0 * srcStride1 + dstX0], batchMu + rand01.y * batchSigma);
        dst2[dstY0 * dstStride2 + dstX0]
            = GaussianNoiseSaturate<T>(src2[dstY0 * srcStride2 + dstX0], batchMu + rand20.x * batchSigma);

        dst0[dstY1 * dstStride0 + dstX1]
            = GaussianNoiseSaturate<T>(src0[dstY1 * srcStride0 + dstX1], batchMu + rand20.y * batchSigma);
        dst1[dstY1 * dstStride1 + dstX1]
            = GaussianNoiseSaturate<T>(src1[dstY1 * srcStride1 + dstX1], batchMu + rand12.x * batchSigma);
        dst2[dstY1 * dstStride2 + dstX1]
            = GaussianNoiseSaturate<T>(src2[dstY1 * srcStride2 + dstX1], batchMu + rand12.y * batchSigma);

        offset += 2 * blockDim.x;
    }
    while (offset < segmentEnd)
    {
        int   dstX  = offset % width;
        int   dstY  = offset / width;
        float rand0 = curand_normal(&localState);
        float rand1 = curand_normal(&localState);
        float rand2 = curand_normal(&localState);

        dst0[dstY * dstStride0 + dstX]
            = GaussianNoiseSaturate<T>(src0[dstY * srcStride0 + dstX], batchMu + rand0 * batchSigma);
        dst1[dstY * dstStride1 + dstX]
            = GaussianNoiseSaturate<T>(src1[dstY * srcStride1 + dstX], batchMu + rand1 * batchSigma);
        dst2[dstY * dstStride2 + dstX]
            = GaussianNoiseSaturate<T>(src2[dstY * srcStride2 + dstX], batchMu + rand2 * batchSigma);

        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
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
            out = cuda::clamp(out, 0.f, 1.f);
        }
        return out;
    }
};

template<bool Clip>
struct GaussianNoiseScalarOutput<Clip, __half>
{
    // F16 mirrors the F32 policy: the sum stays in float, clip clamps to the same [0, 1] range,
    // and only the store narrows to half (one round-to-nearest step; no-clip keeps F32's
    // unbounded behavior, overflowing to +/-inf rather than saturating).
    static __device__ __half Convert(__half value, float delta)
    {
        float out = value + delta;
        if constexpr (Clip)
        {
            out = cuda::clamp(out, 0.f, 1.f);
        }
        return cuda::SaturateCast<__half>(out);
    }
};

template<bool Planar, bool PerChannel, bool Clip, typename T, typename SrcWrap, typename DstWrap>
__global__ void GaussianNoiseScalarKernel(const SrcWrap src, DstWrap dst, curandState *state, curandState *nextState,
                                          float mu, float sigma, int rows, int cols, int channels)
{
    using Out = GaussianNoiseScalarOutput<Clip, T>;

    int         batchIdx   = blockIdx.x;
    int         id         = threadIdx.x + blockIdx.x * blockDim.x;
    int         totalSize  = rows * cols;
    int         offset     = 0;
    int         segmentEnd = 0;
    curandState localState;
    int         rngsPerPixel = PerChannel ? channels : 1;
    if (!GaussianNoiseSegmentState(state, id, totalSize, rngsPerPixel, offset, segmentEnd, localState))
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
            GaussianNoiseStoreChannelPair<Planar, Out>(src, dst, localState, batchIdx, y0, x0, y1, x1, channels, mu,
                                                       sigma);
        }
        else
        {
            float2 rand   = GaussianNoiseNormal2(localState);
            float  delta0 = mu + rand.x * sigma;
            float  delta1 = mu + rand.y * sigma;
            for (int ch = 0; ch < channels; ++ch)
            {
                GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y0, x0, ch, delta0);
                GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y1, x1, ch, delta1);
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
                GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y, x, ch, mu + rand * sigma);
            }
        }
        else
        {
            float rand  = curand_normal(&localState);
            float delta = mu + rand * sigma;
            for (int ch = 0; ch < channels; ++ch)
            {
                GaussianNoiseStoreChannel<Planar, Out>(src, dst, batchIdx, y, x, ch, delta);
            }
        }
        offset += blockDim.x;
    }
    GaussianNoiseStoreSegmentState(state, nextState, id, totalSize, segmentEnd, localState);
}

// --------------------------------------------------------------------------------------------
// Validation and dispatch
// --------------------------------------------------------------------------------------------

enum class NoiseLayout
{
    kNHWC,
    kHWC,
    kNCHW,
    kCHW
};

// The legacy operator classified on bits-per-channel plus data kind, never on an exact DataType, so
// packed types such as TYPE_3U8 land on the same branch as TYPE_U8.  Reproduced exactly: only
// (UNSIGNED,8), (UNSIGNED,16), (SIGNED,16), (SIGNED,32), (FLOAT,16) and (FLOAT,32) are nameable, so
// every other pair -- 8-bit signed, 64-bit float, 32-/64-bit unsigned, 64-bit signed, complex --
// is rejected with the same status before the operator's own list is consulted.
enum class NoiseDataType
{
    U8,
    U16,
    S16,
    S32,
    F16,
    F32
};

inline bool IsPlanarLayout(NoiseLayout layout)
{
    return layout == NoiseLayout::kNCHW || layout == NoiseLayout::kCHW;
}

inline NoiseLayout GetNoiseLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return NoiseLayout::kNCHW;
    }
    if (layout == nvcv::TENSOR_CHW)
    {
        return NoiseLayout::kCHW;
    }
    if (layout == nvcv::TENSOR_NHWC)
    {
        return NoiseLayout::kNHWC;
    }
    if (layout == nvcv::TENSOR_HWC)
    {
        return NoiseLayout::kHWC;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline void ThrowIfPlaneDataTypesDiffer(const nvcv::ImageFormat &fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }
}

inline NoiseLayout GetNoiseLayout(const nvcv::ImageBatchVarShapeDataStridedCuda &batch)
{
    nvcv::ImageFormat fmt = batch.uniqueFormat();
    if (!fmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    ThrowIfPlaneDataTypesDiffer(fmt);

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar images must have one channel per plane");
        }
        return NoiseLayout::kNCHW;
    }
    return NoiseLayout::kNHWC;
}

inline NoiseDataType GetNoiseDataType(const nvcv::DataType &dtype)
{
    auto bpc = dtype.bitsPerChannel();
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::UNSIGNED:
        if (bpc[0] == 8)
        {
            return NoiseDataType::U8;
        }
        if (bpc[0] == 16)
        {
            return NoiseDataType::U16;
        }
        break;

    case nvcv::DataKind::SIGNED:
        if (bpc[0] == 16)
        {
            return NoiseDataType::S16;
        }
        if (bpc[0] == 32)
        {
            return NoiseDataType::S32;
        }
        break;

    case nvcv::DataKind::FLOAT:
        if (bpc[0] == 16)
        {
            return NoiseDataType::F16;
        }
        if (bpc[0] == 32)
        {
            return NoiseDataType::F32;
        }
        break;

    default:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
}

inline NoiseDataType GetNoiseDataType(const nvcv::ImageFormat &fmt)
{
    ThrowIfPlaneDataTypesDiffer(fmt);
    return GetNoiseDataType(fmt.planeDataType(0));
}

// mu and sigma are per-sample rank-1 F32 tensors on both per-sample overloads.  Kept in one place
// so the four rejections keep the same messages and the same relative order in each.
inline void ValidateMuSigma(const nvcv::TensorDataStridedCuda &muData, const nvcv::TensorDataStridedCuda &sigmaData)
{
    if (GetNoiseDataType(muData.dtype()) != NoiseDataType::F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mu DataType");
    }
    if (muData.layout().rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid mu Dim %d", muData.layout().rank());
    }

    if (GetNoiseDataType(sigmaData.dtype()) != NoiseDataType::F32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid sigma DataType");
    }
    if (sigmaData.layout().rank() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid sigma Dim %d", sigmaData.layout().rank());
    }
}

template<class T>
struct TypeTag
{
    using type = T;
};

template<class F>
inline void DispatchNoiseBaseType(NoiseDataType type, F &&f)
{
    switch (type)
    {
    case NoiseDataType::U8:
        f(TypeTag<uchar>{});
        break;
    case NoiseDataType::U16:
        f(TypeTag<ushort>{});
        break;
    case NoiseDataType::S16:
        f(TypeTag<short>{});
        break;
    case NoiseDataType::S32:
        f(TypeTag<int>{});
        break;
    case NoiseDataType::F16:
        f(TypeTag<__half>{});
        break;
    case NoiseDataType::F32:
        f(TypeTag<float>{});
        break;
    }
}

// The interleaved shared-noise path is the one fast path that widens its access: it types the whole
// pixel, so a 3-channel uchar image moves as uchar3 rather than three uchar loads and stores.
template<class F>
inline void DispatchNoisePixelType(NoiseDataType type, int channels, F &&f)
{
    DispatchNoiseBaseType(type,
                          [&](auto tag)
                          {
                              using B = typename decltype(tag)::type;
                              switch (channels)
                              {
                              case 1:
                                  f(TypeTag<B>{});
                                  break;
                              case 2:
                                  f(TypeTag<cuda::MakeType<B, 2>>{});
                                  break;
                              case 3:
                                  f(TypeTag<cuda::MakeType<B, 3>>{});
                                  break;
                              case 4:
                                  f(TypeTag<cuda::MakeType<B, 4>>{});
                                  break;
                              default:
                                  NVCV_ASSERT(false);
                                  break;
                              }
                          });
}

// The device-state buffers are held as raw bytes so <curand_kernel.h> stays out of the header.
inline curandState *AsStates(std::byte *raw)
{
    return reinterpret_cast<curandState *>(raw);
}

// --------------------------------------------------------------------------------------------
// Tensor launches (per-sample mu/sigma)
// --------------------------------------------------------------------------------------------

struct TensorLaunchParams
{
    int          batch;
    int          channels;
    int          rows;
    int          cols;
    curandState *states;
    curandState *nextStates;
    cudaStream_t stream;
};

inline void RunGaussianNoiseTensor(const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData,
                                   const nvcv::TensorDataStridedCuda &muData,
                                   const nvcv::TensorDataStridedCuda &sigmaData, NoiseDataType type, bool isPlanar,
                                   bool perChannel, const TensorLaunchParams &p)
{
    cuda::Tensor1DWrap<float, int32_t> mu(muData);
    cuda::Tensor1DWrap<float, int32_t> sigma(sigmaData);

    const int  segments = GaussianNoiseSegments(p.rows * p.cols, kBlockSize);
    const dim3 grid(p.batch, segments);

    auto launchSegmented = [&](auto &&launch)
    {
        LaunchSegmented(p.batch, segments, p.states, p.nextStates, p.stream, launch);
    };

    if (isPlanar)
    {
        DispatchNoiseBaseType(
            type,
            [&](auto tag)
            {
                using T  = typename decltype(tag)::type;
                auto src = cuda::CreateTensorWrapNCHW<T, int32_t>(inData);
                auto dst = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);
                launchSegmented(
                    [&]
                    {
                        if (perChannel)
                        {
                            GaussianNoisePlanarPerChannelKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                src, dst, p.states, p.nextStates, mu, sigma, p.rows, p.cols, p.channels);
                        }
                        else
                        {
                            GaussianNoisePlanarSharedKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                src, dst, p.states, p.nextStates, mu, sigma, p.rows, p.cols, p.channels);
                        }
                    });
            });
    }
    else if (perChannel)
    {
        DispatchNoiseBaseType(type,
                              [&](auto tag)
                              {
                                  using T  = typename decltype(tag)::type;
                                  auto src = cuda::CreateTensorWrapNHWC<T, int32_t>(inData);
                                  auto dst = cuda::CreateTensorWrapNHWC<T, int32_t>(outData);
                                  launchSegmented(
                                      [&]
                                      {
                                          GaussianNoisePerChannelKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                              src, dst, p.states, p.nextStates, mu, sigma, p.rows, p.cols, p.channels);
                                      });
                              });
    }
    else
    {
        DispatchNoisePixelType(type, p.channels,
                               [&](auto tag)
                               {
                                   using T  = typename decltype(tag)::type;
                                   auto src = cuda::CreateTensorWrapNHW<T, int32_t>(inData);
                                   auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);
                                   launchSegmented(
                                       [&]
                                       {
                                           GaussianNoiseSharedKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                               src, dst, p.states, p.nextStates, mu, sigma, p.rows, p.cols);
                                       });
                               });
    }
}

// --------------------------------------------------------------------------------------------
// VarShape launches (per-sample mu/sigma)
// --------------------------------------------------------------------------------------------

struct VarShapeLaunchParams
{
    int                                      batch;
    int                                      channels;
    // The C3 fast path below adopts the staging half by flipping the two pointers, so this path
    // needs the owning object rather than a snapshot of the two addresses.
    cvcuda::priv::GaussianNoiseDeviceStates *deviceStates;
    cudaStream_t                             stream;
};

inline void RunGaussianNoiseVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                     const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                     const nvcv::TensorDataStridedCuda             &muData,
                                     const nvcv::TensorDataStridedCuda &sigmaData, NoiseDataType type, bool isPlanar,
                                     bool perChannel, const VarShapeLaunchParams &p)
{
    cuda::Tensor1DWrap<float> mu(muData);
    cuda::Tensor1DWrap<float> sigma(sigmaData);

    curandState *states     = AsStates(p.deviceStates->states);
    curandState *nextStates = AsStates(p.deviceStates->nextStates);

    // The segment count is sized from the largest image so every image in the batch keeps the same
    // grid, exactly as the legacy var-shape launcher did.
    const nvcv::Size2D maxSize  = outData.maxSize();
    const int          segments = GaussianNoiseSegments(maxSize.w * maxSize.h, kBlockSize);
    const dim3         grid(p.batch, segments);

    auto launchSegmented = [&](auto &&launch)
    {
        LaunchSegmented(p.batch, segments, states, nextStates, p.stream, launch);
    };

    // The kernels are shared with the tensor paths, where the batch geometry is a launch argument.
    // A var-shape wrap carries it per image instead, so GaussianNoiseExtent reads it off the wrap
    // and the trailing rows/cols arguments below are unused -- hence the literal zeros.
    if (isPlanar)
    {
        // Planar F32 with three planes and one noise value per plane is the shape the var-shape
        // benchmark drives, and it gets a dedicated kernel: plane descriptors hoisted out of the
        // element loop, and the two state copies replaced by a pointer flip, which the kernel
        // compensates for by carrying the idle threads' states itself.
        if (perChannel && type == NoiseDataType::F32 && p.channels == 3)
        {
            cuda::ImageBatchVarShapeWrap<float> src(inData);
            cuda::ImageBatchVarShapeWrap<float> dst(outData);
            GaussianNoisePlanarPerChannel3Kernel<float>
                <<<grid, kBlockSize, 0, p.stream>>>(src, dst, states, nextStates, mu, sigma);
            NVCV_CHECK_THROW(cudaGetLastError());
            if (segments > 1)
            {
                p.deviceStates->Swap();
            }
            return;
        }

        DispatchNoiseBaseType(type,
                              [&](auto tag)
                              {
                                  using T = typename decltype(tag)::type;
                                  cuda::ImageBatchVarShapeWrap<T> src(inData);
                                  cuda::ImageBatchVarShapeWrap<T> dst(outData);
                                  launchSegmented(
                                      [&]
                                      {
                                          if (perChannel)
                                          {
                                              GaussianNoisePlanarPerChannelKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                                  src, dst, states, nextStates, mu, sigma, 0, 0, p.channels);
                                          }
                                          else
                                          {
                                              GaussianNoisePlanarSharedKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                                  src, dst, states, nextStates, mu, sigma, 0, 0, p.channels);
                                          }
                                      });
                              });
    }
    else if (perChannel)
    {
        DispatchNoiseBaseType(type,
                              [&](auto tag)
                              {
                                  using T = typename decltype(tag)::type;
                                  cuda::ImageBatchVarShapeWrapNHWC<T> src(inData, p.channels);
                                  cuda::ImageBatchVarShapeWrapNHWC<T> dst(outData, p.channels);
                                  launchSegmented(
                                      [&]
                                      {
                                          GaussianNoisePerChannelKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                              src, dst, states, nextStates, mu, sigma, 0, 0, p.channels);
                                      });
                              });
    }
    else
    {
        DispatchNoisePixelType(type, p.channels,
                               [&](auto tag)
                               {
                                   using T = typename decltype(tag)::type;
                                   cuda::ImageBatchVarShapeWrap<T> src(inData);
                                   cuda::ImageBatchVarShapeWrap<T> dst(outData);
                                   launchSegmented(
                                       [&] {
                                           GaussianNoiseSharedKernel<T><<<grid, kBlockSize, 0, p.stream>>>(
                                               src, dst, states, nextStates, mu, sigma, 0, 0);
                                       });
                               });
    }
}

// --------------------------------------------------------------------------------------------
// Tensor launches (scalar mu/sigma)
// --------------------------------------------------------------------------------------------

template<bool Planar, bool PerChannel, bool Clip, typename T>
inline void LaunchGaussianNoiseScalar(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData, float mu, float sigma,
                                      const TensorLaunchParams &p)
{
    const int  segments = GaussianNoiseSegments(p.rows * p.cols, kBlockSize);
    const dim3 grid(p.batch, segments);

    auto launch = [&](auto src, auto dst)
    {
        LaunchSegmented(p.batch, segments, p.states, p.nextStates, p.stream,
                        [&]
                        {
                            GaussianNoiseScalarKernel<Planar, PerChannel, Clip, T><<<grid, kBlockSize, 0, p.stream>>>(
                                src, dst, p.states, p.nextStates, mu, sigma, p.rows, p.cols, p.channels);
                        });
    };

    if constexpr (Planar)
    {
        launch(cuda::CreateTensorWrapNCHW<T, int32_t>(inData), cuda::CreateTensorWrapNCHW<T, int32_t>(outData));
    }
    else
    {
        launch(cuda::CreateTensorWrapNHWC<T, int32_t>(inData), cuda::CreateTensorWrapNHWC<T, int32_t>(outData));
    }
}

template<typename T>
inline void DispatchGaussianNoiseScalar(const nvcv::TensorDataStridedCuda &inData,
                                        const nvcv::TensorDataStridedCuda &outData, float mu, float sigma, bool planar,
                                        bool perChannel, bool clip, const TensorLaunchParams &p)
{
    if (planar)
    {
        if (perChannel)
        {
            clip ? LaunchGaussianNoiseScalar<true, true, true, T>(inData, outData, mu, sigma, p)
                 : LaunchGaussianNoiseScalar<true, true, false, T>(inData, outData, mu, sigma, p);
        }
        else
        {
            clip ? LaunchGaussianNoiseScalar<true, false, true, T>(inData, outData, mu, sigma, p)
                 : LaunchGaussianNoiseScalar<true, false, false, T>(inData, outData, mu, sigma, p);
        }
    }
    else if (perChannel)
    {
        clip ? LaunchGaussianNoiseScalar<false, true, true, T>(inData, outData, mu, sigma, p)
             : LaunchGaussianNoiseScalar<false, true, false, T>(inData, outData, mu, sigma, p);
    }
    else
    {
        clip ? LaunchGaussianNoiseScalar<false, false, true, T>(inData, outData, mu, sigma, p)
             : LaunchGaussianNoiseScalar<false, false, false, T>(inData, outData, mu, sigma, p);
    }
}

// Seeding is deferred to the first submission and repeated only when the caller asks for a new
// stream of numbers; otherwise the generator continues from the state the previous call left
// behind, which is what makes back-to-back calls produce different noise.
inline void EnsureSeeded(cvcuda::priv::GaussianNoiseDeviceStates &st, unsigned long long seed, bool reseed,
                         cudaStream_t stream)
{
    if (!reseed && st.setupDone)
    {
        return;
    }

    st.seed = seed;
    SetupGaussianRandKernel<<<st.maxBatchSize, kBlockSize, 0, stream>>>(AsStates(st.states), st.seed);
    st.setupDone = true;
}

} // namespace

namespace cvcuda::priv {

GaussianNoiseDeviceStates::GaussianNoiseDeviceStates(int maxBatchSize)
    : maxBatchSize(maxBatchSize)
{
    const size_t stateBytes = CheckedMulMany(
        {sizeof(curandState), static_cast<size_t>(kBlockSize), CheckedNonNegativeToSize(maxBatchSize, "maxBatchSize")},
        "GaussianNoise curand state allocation size overflow");
    const size_t allocationBytes
        = CheckedMulMany({stateBytes, 2}, "GaussianNoise curand state allocation size overflow");

    void *raw = nullptr;
    NVCV_CHECK_THROW(cudaMalloc(&raw, allocationBytes));

    allocation = static_cast<std::byte *>(raw);
    states     = allocation;
    nextStates = maxBatchSize > 0 ? states + stateBytes : states;
}

GaussianNoiseDeviceStates::~GaussianNoiseDeviceStates()
{
    cudaFree(allocation);
    allocation = nullptr;
    states     = nullptr;
    nextStates = nullptr;
}

GaussianNoise::GaussianNoise(int maxBatchSize)
    // The cuRAND state is single-device by design: it is allocated once and indexed by thread id
    // with no device-switching logic. PerDeviceResource creates one instance per CUDA device for
    // transparent multi-GPU support. The tensor and var-shape paths keep separate generators, as
    // the two legacy operator classes did.
    : m_tensorStates([maxBatchSize](int) { return std::make_unique<GaussianNoiseDeviceStates>(maxBatchSize); })
    , m_varShapeStates([maxBatchSize](int) { return std::make_unique<GaussianNoiseDeviceStates>(maxBatchSize); })
{
    if (maxBatchSize < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "maxBatchSize must be >= 0");
    }
}

void GaussianNoise::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                               const nvcv::Tensor &mu, const nvcv::Tensor &sigma, bool per_channel,
                               unsigned long long seed) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GaussianNoise::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto muData = mu.exportData<nvcv::TensorDataStridedCuda>();
    if (muData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "mu must be cuda-accessible, pitch-linear tensor");
    }

    auto sigmaData = sigma.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigma must be cuda-accessible, pitch-linear tensor");
    }

    const NoiseLayout inLayout  = GetNoiseLayout(inData->layout());
    const NoiseLayout outLayout = GetNoiseLayout(outData->layout());

    const bool isPlanar    = IsPlanarLayout(inLayout);
    const bool isOutPlanar = IsPlanarLayout(outLayout);
    if (isPlanar != isOutPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved or both be planar");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);
    const int channels = inAccess->numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    GaussianNoiseDeviceStates &states = m_tensorStates.get();

    const int batch = inAccess->numSamples();
    if (batch > states.maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    if (inAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    NVCV_ASSERT(outAccess);
    if (isPlanar
        && (outAccess->numSamples() != batch || outAccess->numChannels() != channels
            || outAccess->numRows() != inAccess->numRows() || outAccess->numCols() != inAccess->numCols()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar input and output must have matching sample, channel, height, and width");
    }
    if (isPlanar && batch > 1
        && (inAccess->sampleStride() != channels * inAccess->chStride()
            || outAccess->sampleStride() != channels * outAccess->chStride()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar GaussianNoise of a batched tensor requires tightly packed channel planes");
    }
    if (isPlanar && static_cast<int64_t>(batch) * channels > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar GaussianNoise requires numSamples * channels <= 65535 (CUDA grid-z limit)");
    }

    if (outAccess->sampleStride() * outAccess->numSamples() > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }

    const NoiseDataType inType = GetNoiseDataType(inData->dtype());
    if (inType != GetNoiseDataType(outData->dtype()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    ValidateMuSigma(*muData, *sigmaData);

    EnsureSeeded(states, seed, states.seed != seed, stream);

    const TensorLaunchParams params{
        batch, channels, inAccess->numRows(), inAccess->numCols(), AsStates(states.states), AsStates(states.nextStates),
        stream};
    RunGaussianNoiseTensor(*inData, *outData, *muData, *sigmaData, inType, isPlanar, per_channel, params);
}

void GaussianNoise::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, float mu,
                               float sigma, bool per_channel, unsigned long long seed, bool reseed, bool clip) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GaussianNoise::operator()[Tensor scalar]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    const NoiseLayout inLayout  = GetNoiseLayout(inData->layout());
    const NoiseLayout outLayout = GetNoiseLayout(outData->layout());
    if (inLayout != outLayout)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same NHWC, HWC, NCHW, or CHW layout");
    }

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    if (!inAccess || !outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must be image tensors");
    }

    const bool isPlanar = IsPlanarLayout(inLayout);
    const int  channels = inAccess->numChannels();
    const int  batch    = inAccess->numSamples();
    if (channels < 1 || channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    GaussianNoiseDeviceStates &states = m_tensorStates.get();

    if (batch > states.maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }
    if (outAccess->numSamples() != batch || outAccess->numChannels() != channels
        || outAccess->numRows() != inAccess->numRows() || outAccess->numCols() != inAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have identical shape");
    }
    if (isPlanar && batch > 1
        && (inAccess->sampleStride() != channels * inAccess->chStride()
            || outAccess->sampleStride() != channels * outAccess->chStride()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar GaussianNoise of a batched tensor requires tightly packed channel planes");
    }
    if (isPlanar && static_cast<int64_t>(batch) * channels > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar GaussianNoise requires numSamples * channels <= 65535");
    }

    if (inAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max
        || outAccess->sampleStride() * batch > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output tensor is too large for 32-bit indexing");
    }

    const NoiseDataType inType  = GetNoiseDataType(inData->dtype());
    const NoiseDataType outType = GetNoiseDataType(outData->dtype());
    if (!(inType == NoiseDataType::U8 || inType == NoiseDataType::F32 || inType == NoiseDataType::F16)
        || inType != outType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Scalar GaussianNoise supports matching U8, F16, or F32 input and output");
    }
    if (sigma < 0.f)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "sigma must be non-negative");
    }

    EnsureSeeded(states, seed, reseed, stream);

    const TensorLaunchParams params{
        batch, channels, inAccess->numRows(), inAccess->numCols(), AsStates(states.states), AsStates(states.nextStates),
        stream};

    if (inType == NoiseDataType::F32)
    {
        DispatchGaussianNoiseScalar<float>(*inData, *outData, mu, sigma, isPlanar, per_channel, clip, params);
    }
    else if (inType == NoiseDataType::F16)
    {
        DispatchGaussianNoiseScalar<__half>(*inData, *outData, mu, sigma, isPlanar, per_channel, clip, params);
    }
    else
    {
        DispatchGaussianNoiseScalar<uchar>(*inData, *outData, mu, sigma, isPlanar, per_channel, clip, params);
    }
}

void GaussianNoise::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                               const nvcv::ImageBatchVarShape &out, const nvcv::Tensor &mu, const nvcv::Tensor &sigma,
                               bool per_channel, unsigned long long seed) const
{
    CVCUDA_NVTX_RANGE("cvcuda::GaussianNoise::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto muData = mu.exportData<nvcv::TensorDataStridedCuda>();
    if (muData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "mu must be cuda-accessible, pitch-linear tensor");
    }

    auto sigmaData = sigma.exportData<nvcv::TensorDataStridedCuda>();
    if (sigmaData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigma must be cuda-accessible, pitch-linear tensor");
    }

    const NoiseLayout inLayout  = GetNoiseLayout(*inData);
    const NoiseLayout outLayout = GetNoiseLayout(*outData);

    const bool isPlanar    = IsPlanarLayout(inLayout);
    const bool isOutPlanar = IsPlanarLayout(outLayout);
    if (isPlanar != isOutPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must both be interleaved or both be planar");
    }

    const int channels = inData->uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }
    if (isPlanar && outData->uniqueFormat().numChannels() != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar input and output must have matching channel counts");
    }
    if (isPlanar && static_cast<int64_t>(inData->numImages()) * channels > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar GaussianNoise requires numImages * channels <= 65535 (CUDA grid-z limit)");
    }

    GaussianNoiseDeviceStates &states = m_varShapeStates.get();

    if (inData->numImages() > states.maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input batch exceeds maxBatchSize");
    }

    const NoiseDataType inType = GetNoiseDataType(inData->uniqueFormat());
    if (inType != GetNoiseDataType(outData->uniqueFormat()))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "DataType of input and output must be equal");
    }

    ValidateMuSigma(*muData, *sigmaData);

    EnsureSeeded(states, seed, states.seed != seed, stream);

    const VarShapeLaunchParams params{inData->numImages(), channels, &states, stream};
    RunGaussianNoiseVarShape(*inData, *outData, *muData, *sigmaData, inType, isPlanar, per_channel, params);
}

} // namespace cvcuda::priv
