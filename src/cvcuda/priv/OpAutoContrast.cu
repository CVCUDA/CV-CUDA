/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Nvtx.hpp"
#include "OpAutoContrast.hpp"
#include "PerDeviceResource.hpp"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <math_constants.h>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

namespace cuda = nvcv::cuda;

namespace {

constexpr int WARP_WIDTH = 32;

// Reduce each warp with shuffles, write one per-warp pair to shared memory, and
// have the first warp fold those pairs. nThreads must be a multiple of 32 (the
// reduction launches use 128 or 256 threads). partialStride is the number of
// CTA partials for one (sample, channel).
template<int MAXC>
inline __device__ void ReduceBlockToPartials(float *sm, int nThreads, int tid, int nChan, const float (&vMin)[MAXC],
                                             const float (&vMax)[MAXC], float *loBase, float *hiBase, int partialStride)
{
    const int lane     = tid & (WARP_WIDTH - 1);
    const int warp     = tid / WARP_WIDTH;
    const int numWarps = nThreads / WARP_WIDTH;
    float    *sMin     = sm;
    float    *sMax     = sm + nChan * numWarps;
#pragma unroll
    for (int c = 0; c < MAXC; ++c)
    {
        if (c < nChan)
        {
            float warpMin = vMin[c];
            float warpMax = vMax[c];
#pragma unroll
            for (int offset = WARP_WIDTH / 2; offset > 0; offset >>= 1)
            {
                warpMin = fminf(warpMin, __shfl_down_sync(0xffffffffu, warpMin, offset));
                warpMax = fmaxf(warpMax, __shfl_down_sync(0xffffffffu, warpMax, offset));
            }
            if (lane == 0)
            {
                sMin[c * numWarps + warp] = warpMin;
                sMax[c * numWarps + warp] = warpMax;
            }
        }
    }
    __syncthreads();

    if (warp == 0)
    {
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
        {
            if (c < nChan)
            {
                float blockMin = lane < numWarps ? sMin[c * numWarps + lane] : CUDART_INF_F;
                float blockMax = lane < numWarps ? sMax[c * numWarps + lane] : -CUDART_INF_F;
#pragma unroll
                for (int offset = WARP_WIDTH / 2; offset > 0; offset >>= 1)
                {
                    blockMin = fminf(blockMin, __shfl_down_sync(0xffffffffu, blockMin, offset));
                    blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffffu, blockMax, offset));
                }
                if (lane == 0)
                {
                    const int partialOffset = c * partialStride;
                    loBase[partialOffset]   = blockMin;
                    hiBase[partialOffset]   = blockMax;
                }
            }
        }
    }
    __syncthreads(); // required when a caller reuses the reduction shared memory
}

// The VarShape planar-F32 kernel carries enough wrapper state that the shuffle
// fold crosses a register-allocation boundary on SM80/SM90. Retaining the
// shared-memory tree for that specialization preserves full CTA residency.
template<int MAXC>
inline __device__ void ReduceBlockToPartialsShared(float *sm, int nThreads, int tid, int nChan,
                                                   const float (&vMin)[MAXC], const float (&vMax)[MAXC], float *loBase,
                                                   float *hiBase, int partialStride)
{
    float *sMin = sm;
    float *sMax = sm + nChan * nThreads;
#pragma unroll
    for (int c = 0; c < MAXC; ++c)
    {
        if (c < nChan)
        {
            sMin[c * nThreads + tid] = vMin[c];
            sMax[c * nThreads + tid] = vMax[c];
        }
    }
    __syncthreads();

    for (int stride = nThreads >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int c = 0; c < nChan; ++c)
            {
                sMin[c * nThreads + tid] = fminf(sMin[c * nThreads + tid], sMin[c * nThreads + tid + stride]);
                sMax[c * nThreads + tid] = fmaxf(sMax[c * nThreads + tid], sMax[c * nThreads + tid + stride]);
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        for (int c = 0; c < nChan; ++c)
        {
            const int partialOffset = c * partialStride;
            loBase[partialOffset]   = sMin[c * nThreads];
            hiBase[partialOffset]   = sMax[c * nThreads];
        }
    }
    __syncthreads(); // required when a caller reuses the reduction shared memory
}

// One block deterministically folds all CTA partials for one (sample, channel).
// The grid has exactly numSamples*numChannels blocks, so there is no global
// fan-in, atomic retry, or output aliasing in this pass.
__global__ void FinalizeMinMax(const float *partialLo, const float *partialHi, float *lo, float *hi, size_t numPartials)
{
    extern __shared__ float sm[];
    const int               tid  = threadIdx.x;
    const size_t            base = static_cast<size_t>(blockIdx.x) * numPartials;
    float                   vMin[1]{CUDART_INF_F};
    float                   vMax[1]{-CUDART_INF_F};

    for (size_t i = static_cast<size_t>(tid); i < numPartials; i += blockDim.x)
    {
        vMin[0] = fminf(vMin[0], partialLo[base + i]);
        vMax[0] = fmaxf(vMax[0], partialHi[base + i]);
    }
    ReduceBlockToPartials<1>(sm, blockDim.x, tid, 1, vMin, vMax, &lo[blockIdx.x], &hi[blockIdx.x], 1);
}

static __device__ __noinline__ float RemapWideFloat(float in, float lo, float hi)
{
    const double wideLo = static_cast<double>(lo);
    return static_cast<float>((static_cast<double>(in) - wideLo) / (static_cast<double>(hi) - wideLo));
}

// AutoContrast remap, shared by every layout/container. A flat channel passes
// through unchanged. Integer results are clamped and truncated to match Pillow
// and torchvision; floating-point results retain the scaled value.
template<typename BT>
inline __device__ BT RemapPixel(BT in, float lo, float hi, float bound)
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        if (!isfinite(in))
        {
            return in;
        }
    }
    if (hi == lo)
    {
        return in;
    }
    float range = hi - lo;
    float val;
    if (isfinite(range))
    {
        val = (static_cast<float>(in) - lo) * bound / range;
    }
    else
    {
        if constexpr (std::is_floating_point_v<BT>)
        {
            val = RemapWideFloat(static_cast<float>(in), lo, hi);
        }
        else
        {
            // Unreachable for supported integer types: their extrema and ranges
            // are exactly representable in float.
            const float halfLo = lo * 0.5f;
            val                = (static_cast<float>(in) * 0.5f - halfLo) * bound / (hi * 0.5f - halfLo);
        }
    }
    val = fminf(fmaxf(val, 0.f), bound);
    return static_cast<BT>(val);
}

// ------------------------------- Reduce kernels ----------------------------

constexpr int    REDUCE_X_STEPS                   = 8;
constexpr int    BLOCK_X                          = 32;
constexpr int    BLOCK_Y                          = 4;
constexpr int    FINAL_THREADS                    = 256;
constexpr int    MAX_GRID_Y                       = 65535;
constexpr int    MAX_HEIGHT                       = BLOCK_Y * MAX_GRID_Y;
constexpr size_t MAX_PARTIALS_PER_SAMPLE          = 4096;
constexpr size_t MAX_REDUCTION_WORKSPACE_BYTES    = 6 * 1024 * 1024;
constexpr size_t MAX_REDUCTION_WORKSPACE_ONE_SIDE = MAX_REDUCTION_WORKSPACE_BYTES / (2 * sizeof(float));

inline int ComputeGridX(int width, int xSteps, int numPlanes = 1)
{
    const int64_t grid = cvcuda::priv::detail::AutoContrastGridX(width, BLOCK_X, xSteps, numPlanes);
    if (grid > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "CUDA grid width exceeds %d",
                              cuda::TypeTraits<int32_t>::max);
    }
    return static_cast<int>(grid);
}

template<typename BT>
inline __device__ void AccumulateExtrema(float value, float &lo, float &hi)
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        if (!isfinite(value))
        {
            return;
        }
    }
    lo = fminf(lo, value);
    hi = fmaxf(hi, value);
}

// Exact-grid overload. Its launch grid is the logical image-tile grid, so this
// kernel deliberately retains the comparator's direct blockIdx mapping and
// argument list. In particular, no bounded-grid iterator state reaches this
// specialization.
template<bool IsPlanar, typename T, class SrcWrapper>
__global__ void ReduceTensor(SrcWrapper src, float *partialLo, float *partialHi, int2 size, int numChannels,
                             int numPlanes)
{
    extern __shared__ float sm[];
    const int               x0          = blockIdx.x * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
    const int               y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int               s           = blockIdx.z;
    const int               tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int               nThreads    = blockDim.x * blockDim.y;
    const int               numPartials = gridDim.x * gridDim.y;
    const int               partial     = blockIdx.y * gridDim.x + blockIdx.x;
    const int               base        = s * numChannels * numPartials + partial;

    if constexpr (!IsPlanar)
    {
        constexpr int C = cuda::NumElements<T>;
        float         vMin[C], vMax[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            vMin[c] = CUDART_INF_F;
            vMax[c] = -CUDART_INF_F;
        }
#pragma unroll
        for (int i = 0; i < REDUCE_X_STEPS; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < size.x && y < size.y)
            {
                T pix = src[int3{x, y, s}];
#pragma unroll
                for (int c = 0; c < C; ++c)
                {
                    float v = static_cast<float>(cuda::GetElement(pix, c));
                    AccumulateExtrema<cuda::BaseType<T>>(v, vMin[c], vMax[c]);
                }
            }
        }
        ReduceBlockToPartials<C>(sm, nThreads, tid, C, vMin, vMax, &partialLo[base], &partialHi[base], numPartials);
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            float vMin[1], vMax[1];
            vMin[0] = CUDART_INF_F;
            vMax[0] = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < REDUCE_X_STEPS; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < size.x && y < size.y)
                {
                    float v = static_cast<float>(src[int4{x, y, p, s}]);
                    AccumulateExtrema<cuda::BaseType<T>>(v, vMin[0], vMax[0]);
                }
            }
            const int planeBase = base + p * numPartials;
            ReduceBlockToPartials<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase], &partialHi[planeBase],
                                     numPartials);
        }
    }
}

// Bounded-grid overload. A physical CTA deterministically owns every logical
// tile congruent to blockIdx in both grid dimensions.
template<bool IsPlanar, typename T, class SrcWrapper>
__global__ void ReduceTensor(SrcWrapper src, float *partialLo, float *partialHi, int2 size, int numChannels,
                             int numPlanes, int logicalGridX, int logicalGridY)
{
    extern __shared__ float sm[];
    const int               s           = blockIdx.z;
    const int               tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int               nThreads    = blockDim.x * blockDim.y;
    const int               numPartials = gridDim.x * gridDim.y;
    const int               partial     = blockIdx.y * gridDim.x + blockIdx.x;
    const int               base        = s * numChannels * numPartials + partial;

    if constexpr (!IsPlanar)
    {
        constexpr int C = cuda::NumElements<T>;
        float         vMin[C], vMax[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            vMin[c] = CUDART_INF_F;
            vMax[c] = -CUDART_INF_F;
        }
        for (int tileY = blockIdx.y; tileY < logicalGridY; tileY += gridDim.y)
        {
            const int y = tileY * blockDim.y + threadIdx.y;
            for (int tileX = blockIdx.x; tileX < logicalGridX; tileX += gridDim.x)
            {
                const int x0 = tileX * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
#pragma unroll
                for (int i = 0; i < REDUCE_X_STEPS; ++i)
                {
                    const int x = x0 + i * blockDim.x;
                    if (x < size.x && y < size.y)
                    {
                        T pix = src[int3{x, y, s}];
#pragma unroll
                        for (int c = 0; c < C; ++c)
                        {
                            float v = static_cast<float>(cuda::GetElement(pix, c));
                            AccumulateExtrema<cuda::BaseType<T>>(v, vMin[c], vMax[c]);
                        }
                    }
                }
            }
        }
        ReduceBlockToPartials<C>(sm, nThreads, tid, C, vMin, vMax, &partialLo[base], &partialHi[base], numPartials);
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            float vMin[1], vMax[1];
            vMin[0] = CUDART_INF_F;
            vMax[0] = -CUDART_INF_F;
            for (int tileY = blockIdx.y; tileY < logicalGridY; tileY += gridDim.y)
            {
                const int y = tileY * blockDim.y + threadIdx.y;
                for (int tileX = blockIdx.x; tileX < logicalGridX; tileX += gridDim.x)
                {
                    const int x0 = tileX * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
#pragma unroll
                    for (int i = 0; i < REDUCE_X_STEPS; ++i)
                    {
                        const int x = x0 + i * blockDim.x;
                        if (x < size.x && y < size.y)
                        {
                            float v = static_cast<float>(src[int4{x, y, p, s}]);
                            AccumulateExtrema<cuda::BaseType<T>>(v, vMin[0], vMax[0]);
                        }
                    }
                }
            }
            const int planeBase = base + p * numPartials;
            ReduceBlockToPartials<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase], &partialHi[planeBase],
                                     numPartials);
        }
    }
}

// Exact-grid overload matching the comparator's direct blockIdx mapping and
// argument list.
template<bool IsPlanar, typename T, class SrcWrapper>
__global__ void ReduceVarShape(SrcWrapper src, float *partialLo, float *partialHi, int numChannels, int numPlanes)
{
    extern __shared__ float sm[];
    const int               x0          = blockIdx.x * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
    const int               y           = blockIdx.y * blockDim.y + threadIdx.y;
    const int               s           = blockIdx.z;
    const int               tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int               nThreads    = blockDim.x * blockDim.y;
    const int               width       = src.width(s);
    const int               height      = src.height(s);
    const int               numPartials = gridDim.x * gridDim.y;
    const int               partial     = blockIdx.y * gridDim.x + blockIdx.x;
    const int               base        = s * numChannels * numPartials + partial;

    if constexpr (!IsPlanar)
    {
        constexpr int C = cuda::NumElements<T>;
        float         vMin[C], vMax[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            vMin[c] = CUDART_INF_F;
            vMax[c] = -CUDART_INF_F;
        }
#pragma unroll
        for (int i = 0; i < REDUCE_X_STEPS; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < width && y < height)
            {
                T pix = src[int3{x, y, s}];
#pragma unroll
                for (int c = 0; c < C; ++c)
                {
                    float v = static_cast<float>(cuda::GetElement(pix, c));
                    AccumulateExtrema<cuda::BaseType<T>>(v, vMin[c], vMax[c]);
                }
            }
        }
        ReduceBlockToPartials<C>(sm, nThreads, tid, C, vMin, vMax, &partialLo[base], &partialHi[base], numPartials);
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            float vMin[1], vMax[1];
            vMin[0] = CUDART_INF_F;
            vMax[0] = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < REDUCE_X_STEPS; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < width && y < height)
                {
                    float v = static_cast<float>(src[int4{x, y, p, s}]);
                    AccumulateExtrema<cuda::BaseType<T>>(v, vMin[0], vMax[0]);
                }
            }
            const int planeBase = base + p * numPartials;
            if constexpr (std::is_same_v<cuda::BaseType<T>, float>)
            {
                ReduceBlockToPartialsShared<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase],
                                               &partialHi[planeBase], numPartials);
            }
            else
            {
                ReduceBlockToPartials<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase], &partialHi[planeBase],
                                         numPartials);
            }
        }
    }
}

// Bounded-grid overload using deterministic 2-D grid-stride ownership.
template<bool IsPlanar, typename T, class SrcWrapper>
__global__ void ReduceVarShape(SrcWrapper src, float *partialLo, float *partialHi, int numChannels, int numPlanes,
                               int logicalGridX, int logicalGridY)
{
    extern __shared__ float sm[];
    const int               s           = blockIdx.z;
    const int               tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int               nThreads    = blockDim.x * blockDim.y;
    const int               width       = src.width(s);
    const int               height      = src.height(s);
    const int               numPartials = gridDim.x * gridDim.y;
    const int               partial     = blockIdx.y * gridDim.x + blockIdx.x;
    const int               base        = s * numChannels * numPartials + partial;

    if constexpr (!IsPlanar)
    {
        constexpr int C = cuda::NumElements<T>;
        float         vMin[C], vMax[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            vMin[c] = CUDART_INF_F;
            vMax[c] = -CUDART_INF_F;
        }
        for (int tileY = blockIdx.y; tileY < logicalGridY; tileY += gridDim.y)
        {
            const int y = tileY * blockDim.y + threadIdx.y;
            for (int tileX = blockIdx.x; tileX < logicalGridX; tileX += gridDim.x)
            {
                const int x0 = tileX * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
#pragma unroll
                for (int i = 0; i < REDUCE_X_STEPS; ++i)
                {
                    const int x = x0 + i * blockDim.x;
                    if (x < width && y < height)
                    {
                        T pix = src[int3{x, y, s}];
#pragma unroll
                        for (int c = 0; c < C; ++c)
                        {
                            float v = static_cast<float>(cuda::GetElement(pix, c));
                            AccumulateExtrema<cuda::BaseType<T>>(v, vMin[c], vMax[c]);
                        }
                    }
                }
            }
        }
        ReduceBlockToPartials<C>(sm, nThreads, tid, C, vMin, vMax, &partialLo[base], &partialHi[base], numPartials);
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            float vMin[1], vMax[1];
            vMin[0] = CUDART_INF_F;
            vMax[0] = -CUDART_INF_F;
            for (int tileY = blockIdx.y; tileY < logicalGridY; tileY += gridDim.y)
            {
                const int y = tileY * blockDim.y + threadIdx.y;
                for (int tileX = blockIdx.x; tileX < logicalGridX; tileX += gridDim.x)
                {
                    const int x0 = tileX * blockDim.x * REDUCE_X_STEPS + threadIdx.x;
#pragma unroll
                    for (int i = 0; i < REDUCE_X_STEPS; ++i)
                    {
                        const int x = x0 + i * blockDim.x;
                        if (x < width && y < height)
                        {
                            float v = static_cast<float>(src[int4{x, y, p, s}]);
                            AccumulateExtrema<cuda::BaseType<T>>(v, vMin[0], vMax[0]);
                        }
                    }
                }
            }
            const int planeBase = base + p * numPartials;
            if constexpr (std::is_same_v<cuda::BaseType<T>, float>)
            {
                ReduceBlockToPartialsShared<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase],
                                               &partialHi[planeBase], numPartials);
            }
            else
            {
                ReduceBlockToPartials<1>(sm, nThreads, tid, 1, vMin, vMax, &partialLo[planeBase], &partialHi[planeBase],
                                         numPartials);
            }
        }
    }
}

// ------------------------------- Apply kernels -----------------------------

constexpr int APPLY_X_STEPS = 4;

template<bool IsPlanar, typename T, class SrcWrapper, class DstWrapper>
__global__ void ApplyTensor(SrcWrapper src, DstWrapper dst, const float *lo, const float *hi, int2 size,
                            int numChannels, int numPlanes, float bound)
{
    const int x0           = blockIdx.x * blockDim.x * APPLY_X_STEPS + threadIdx.x;
    const int y            = blockIdx.y * blockDim.y + threadIdx.y;
    const int s            = blockIdx.z;
    const int sampleOffset = s * numChannels;
    if (x0 >= size.x || y >= size.y)
    {
        return;
    }

    if constexpr (!IsPlanar)
    {
        using BT        = cuda::BaseType<T>;
        constexpr int C = cuda::NumElements<T>;
        float         minValue[C], maxValue[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            minValue[c] = lo[sampleOffset + c];
            maxValue[c] = hi[sampleOffset + c];
        }
#pragma unroll
        for (int i = 0; i < APPLY_X_STEPS; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < size.x)
            {
                T pix = src[int3{x, y, s}];
                T out{};
#pragma unroll
                for (int c = 0; c < C; ++c)
                {
                    cuda::GetElement(out, c)
                        = RemapPixel<BT>(cuda::GetElement(pix, c), minValue[c], maxValue[c], bound);
                }
                dst[int3{x, y, s}] = out;
            }
        }
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            const float minValue = lo[sampleOffset + p];
            const float maxValue = hi[sampleOffset + p];
#pragma unroll
            for (int i = 0; i < APPLY_X_STEPS; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < size.x)
                {
                    T in                  = src[int4{x, y, p, s}];
                    dst[int4{x, y, p, s}] = RemapPixel<T>(in, minValue, maxValue, bound);
                }
            }
        }
    }
}

template<bool IsPlanar, typename T, class SrcWrapper, class DstWrapper>
__global__ void ApplyVarShape(SrcWrapper src, DstWrapper dst, const float *lo, const float *hi, int numChannels,
                              int numPlanes, float bound)
{
    const int x0           = blockIdx.x * blockDim.x * APPLY_X_STEPS + threadIdx.x;
    const int y            = blockIdx.y * blockDim.y + threadIdx.y;
    const int s            = blockIdx.z;
    const int width        = dst.width(s);
    const int height       = dst.height(s);
    const int sampleOffset = s * numChannels;
    if (x0 >= width || y >= height)
    {
        return;
    }

    if constexpr (!IsPlanar)
    {
        using BT        = cuda::BaseType<T>;
        constexpr int C = cuda::NumElements<T>;
        float         minValue[C], maxValue[C];
#pragma unroll
        for (int c = 0; c < C; ++c)
        {
            minValue[c] = lo[sampleOffset + c];
            maxValue[c] = hi[sampleOffset + c];
        }
#pragma unroll
        for (int i = 0; i < APPLY_X_STEPS; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < width)
            {
                T pix = src[int3{x, y, s}];
                T out{};
#pragma unroll
                for (int c = 0; c < C; ++c)
                {
                    cuda::GetElement(out, c)
                        = RemapPixel<BT>(cuda::GetElement(pix, c), minValue[c], maxValue[c], bound);
                }
                dst[int3{x, y, s}] = out;
            }
        }
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            const float minValue = lo[sampleOffset + p];
            const float maxValue = hi[sampleOffset + p];
#pragma unroll
            for (int i = 0; i < APPLY_X_STEPS; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < width)
                {
                    T in                  = src[int4{x, y, p, s}];
                    dst[int4{x, y, p, s}] = RemapPixel<T>(in, minValue, maxValue, bound);
                }
            }
        }
    }
}

// ------------------------------- Launchers ---------------------------------

template<typename BT>
inline float DtypeBound()
{
    return std::is_floating_point_v<BT> ? 1.0f : static_cast<float>(cuda::TypeTraits<BT>::max);
}

struct ReductionWorkspace
{
    float *partialLo;
    float *partialHi;
    float *lo;
    float *hi;
    size_t numPartials;
    size_t numExtrema;
};

class WorkspaceReleaseGuard
{
public:
    WorkspaceReleaseGuard(cvcuda::priv::AutoContrastWorkspace &workspace, cudaStream_t stream)
        : m_workspace(workspace)
        , m_stream(stream)
    {
    }

    ~WorkspaceReleaseGuard()
    {
        if (m_active)
        {
            m_workspace.releaseNoThrow(m_stream);
        }
    }

    void finish()
    {
        m_workspace.release(m_stream);
        m_active = false;
    }

private:
    cvcuda::priv::AutoContrastWorkspace &m_workspace;
    cudaStream_t                         m_stream;
    bool                                 m_active = true;
};

struct ReductionGrid
{
    dim3 launch;
    int  logicalX;
    int  logicalY;

    bool isExact() const
    {
        return launch.x == static_cast<unsigned int>(logicalX) && launch.y == static_cast<unsigned int>(logicalY);
    }
};

inline size_t CheckedWorkspaceMul(size_t a, size_t b)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AutoContrast reduction workspace size overflows size_t");
    }
    return a * b;
}

inline size_t CheckedWorkspaceAdd(size_t a, size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AutoContrast reduction workspace size overflows size_t");
    }
    return a + b;
}

inline ReductionGrid GetReductionGrid(int width, int height, int numSamples, int numChannels)
{
    const int    logicalX        = ComputeGridX(width, REDUCE_X_STEPS);
    const int    logicalY        = cvcuda::priv::detail::AutoContrastDivUp(height, BLOCK_Y);
    const size_t logicalPartials = CheckedWorkspaceMul(logicalX, logicalY);
    const size_t numExtrema      = CheckedWorkspaceMul(static_cast<size_t>(numSamples), numChannels);

    // The workspace owns two equal sides: lo partials plus final lo, then the
    // corresponding hi values. Bound both the persistent allocation and the
    // amount of final-pass work, while every logical tile is still visited by
    // a physical CTA through the grid-stride loop in Reduce*.
    if (numExtrema == 0 || numExtrema > MAX_REDUCTION_WORKSPACE_ONE_SIDE / 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW,
                              "AutoContrast extrema count exceeds the bounded reduction workspace");
    }
    const size_t maxPartialsByWorkspace = MAX_REDUCTION_WORKSPACE_ONE_SIDE / numExtrema - 1;
    const size_t partialLimit = std::min(logicalPartials, std::min(MAX_PARTIALS_PER_SAMPLE, maxPartialsByWorkspace));
    const size_t launchX      = std::min(static_cast<size_t>(logicalX), partialLimit);
    const size_t launchY      = std::min(static_cast<size_t>(logicalY), partialLimit / launchX);

    NVCV_ASSERT(launchX > 0 && launchY > 0);
    return {
        {static_cast<unsigned int>(launchX), static_cast<unsigned int>(launchY), static_cast<unsigned int>(numSamples)},
        logicalX,
        logicalY
    };
}

inline ReductionWorkspace GetReductionWorkspace(cvcuda::priv::AutoContrastWorkspace &ws, cudaStream_t stream,
                                                int numSamples, int numChannels, dim3 reduceGrid)
{
    const size_t numPartials  = CheckedWorkspaceMul(reduceGrid.x, reduceGrid.y);
    const size_t numExtrema   = CheckedWorkspaceMul(static_cast<size_t>(numSamples), numChannels);
    const size_t partialCount = CheckedWorkspaceMul(numExtrema, numPartials);
    const size_t pairedCount  = CheckedWorkspaceAdd(partialCount, numExtrema);
    float       *buf          = ws.acquire(pairedCount, stream);

    float *partialLo = buf;
    float *partialHi = partialLo + partialCount;
    float *lo        = partialHi + partialCount;
    float *hi        = lo + numExtrema;
    return {partialLo, partialHi, lo, hi, numPartials, numExtrema};
}

template<bool IsPlanar, class Access>
inline bool TensorAddressingFitsInt32(const Access &access)
{
    constexpr int64_t limit = cuda::TypeTraits<int32_t>::max;
    int64_t           maxOffset{};
    auto              accumulate = [&maxOffset](int64_t stride, int64_t extent)
    {
        if (stride < 0 || stride > limit)
        {
            return false;
        }
        const int64_t index = extent > 0 ? extent - 1 : 0;
        if (index > 0 && stride > (limit - maxOffset) / index)
        {
            return false;
        }
        maxOffset += stride * index;
        return true;
    };

    return accumulate(access.sampleStride(), access.numSamples())
        && (!IsPlanar || accumulate(access.planeStride(), access.numPlanes()))
        && accumulate(access.rowStride(), access.numRows()) && accumulate(access.colStride(), access.numCols());
}

template<bool IsPlanar, typename T>
void RunTensor(cudaStream_t stream, cvcuda::priv::AutoContrastWorkspace &ws, const nvcv::TensorDataStridedCuda &srcData,
               const nvcv::TensorDataStridedCuda &dstData, int numSamples, int numChannels, int numPlanes)
{
    using BT          = cuda::BaseType<T>;
    const float bound = DtypeBound<BT>();

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});

    if (!TensorAddressingFitsInt32<IsPlanar>(*srcAccess) || !TensorAddressingFitsInt32<IsPlanar>(*dstAccess))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW,
                              "Input or output dynamic stride or maximum byte offset exceeds %d",
                              cuda::TypeTraits<int32_t>::max);
    }
    using StrideType = int32_t;

    dim3          block(BLOCK_X, BLOCK_Y, 1);
    ReductionGrid reduceGrid = GetReductionGrid(size.x, size.y, numSamples, numChannels);
    dim3      applyGrid(ComputeGridX(size.x, APPLY_X_STEPS), cvcuda::priv::detail::AutoContrastDivUp(size.y, BLOCK_Y),
                        numSamples);
    const int nThreads               = block.x * block.y;
    const int numWarps               = nThreads / WARP_WIDTH;
    const size_t          reduceChan = IsPlanar ? 1 : cuda::NumElements<T>;
    const size_t          smem       = 2 * reduceChan * numWarps * sizeof(float);
    const size_t          finalSmem  = 2 * (FINAL_THREADS / WARP_WIDTH) * sizeof(float);
    auto                  reduction  = GetReductionWorkspace(ws, stream, numSamples, numChannels, reduceGrid.launch);
    WorkspaceReleaseGuard releaseGuard(ws, stream);

    if constexpr (!IsPlanar)
    {
        auto src = cuda::CreateTensorWrapNHW<const T, StrideType>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(dstData);
        if (reduceGrid.isExact())
        {
            ReduceTensor<false, T><<<reduceGrid.launch, block, smem, stream>>>(
                src, reduction.partialLo, reduction.partialHi, size, numChannels, 1);
        }
        else
        {
            ReduceTensor<false, T>
                <<<reduceGrid.launch, block, smem, stream>>>(src, reduction.partialLo, reduction.partialHi, size,
                                                             numChannels, 1, reduceGrid.logicalX, reduceGrid.logicalY);
        }
        FinalizeMinMax<<<static_cast<unsigned int>(reduction.numExtrema), FINAL_THREADS, finalSmem, stream>>>(
            reduction.partialLo, reduction.partialHi, reduction.lo, reduction.hi, reduction.numPartials);
        ApplyTensor<false, T>
            <<<applyGrid, block, 0, stream>>>(src, dst, reduction.lo, reduction.hi, size, numChannels, 1, bound);
    }
    else
    {
        auto src = cuda::Tensor4DWrap<const T, StrideType>(
            srcData.basePtr(), static_cast<StrideType>(srcAccess->sampleStride()),
            static_cast<StrideType>(srcAccess->planeStride()), static_cast<StrideType>(srcAccess->rowStride()));
        auto dst = cuda::Tensor4DWrap<T, StrideType>(
            dstData.basePtr(), static_cast<StrideType>(dstAccess->sampleStride()),
            static_cast<StrideType>(dstAccess->planeStride()), static_cast<StrideType>(dstAccess->rowStride()));
        if (reduceGrid.isExact())
        {
            ReduceTensor<true, T><<<reduceGrid.launch, block, smem, stream>>>(
                src, reduction.partialLo, reduction.partialHi, size, numChannels, numPlanes);
        }
        else
        {
            ReduceTensor<true, T><<<reduceGrid.launch, block, smem, stream>>>(
                src, reduction.partialLo, reduction.partialHi, size, numChannels, numPlanes, reduceGrid.logicalX,
                reduceGrid.logicalY);
        }
        FinalizeMinMax<<<static_cast<unsigned int>(reduction.numExtrema), FINAL_THREADS, finalSmem, stream>>>(
            reduction.partialLo, reduction.partialHi, reduction.lo, reduction.hi, reduction.numPartials);
        ApplyTensor<true, T><<<applyGrid, block, 0, stream>>>(src, dst, reduction.lo, reduction.hi, size, numChannels,
                                                              numPlanes, bound);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    releaseGuard.finish();
}

template<bool IsPlanar, typename T>
void RunVarShapeBatch(cudaStream_t stream, cvcuda::priv::AutoContrastWorkspace &ws,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, int numSamples, int numChannels,
                      int numPlanes)
{
    using BT          = cuda::BaseType<T>;
    const float bound = DtypeBound<BT>();

    int3          maxSize{dstData.maxSize().w, dstData.maxSize().h, numSamples};
    dim3          block(BLOCK_X, BLOCK_Y, 1);
    ReductionGrid reduceGrid = GetReductionGrid(maxSize.x, maxSize.y, numSamples, numChannels);
    dim3 applyGrid(ComputeGridX(maxSize.x, APPLY_X_STEPS), cvcuda::priv::detail::AutoContrastDivUp(maxSize.y, BLOCK_Y),
                   numSamples);
    const int             nThreads      = block.x * block.y;
    const int             numWarps      = nThreads / WARP_WIDTH;
    constexpr bool        useSharedTree = IsPlanar && std::is_same_v<BT, float>;
    const size_t          reduceChan    = IsPlanar ? 1 : cuda::NumElements<T>;
    const size_t          reduceSlots   = useSharedTree ? nThreads : numWarps;
    const size_t          smem          = 2 * reduceChan * reduceSlots * sizeof(float);
    const size_t          finalSmem     = 2 * (FINAL_THREADS / WARP_WIDTH) * sizeof(float);
    auto                  reduction     = GetReductionWorkspace(ws, stream, numSamples, numChannels, reduceGrid.launch);
    WorkspaceReleaseGuard releaseGuard(ws, stream);

    cuda::ImageBatchVarShapeWrap<const T> src(srcData);
    cuda::ImageBatchVarShapeWrap<T>       dst(dstData);

    if (reduceGrid.isExact())
    {
        ReduceVarShape<IsPlanar, T><<<reduceGrid.launch, block, smem, stream>>>(
            src, reduction.partialLo, reduction.partialHi, numChannels, numPlanes);
    }
    else
    {
        ReduceVarShape<IsPlanar, T>
            <<<reduceGrid.launch, block, smem, stream>>>(src, reduction.partialLo, reduction.partialHi, numChannels,
                                                         numPlanes, reduceGrid.logicalX, reduceGrid.logicalY);
    }
    FinalizeMinMax<<<static_cast<unsigned int>(reduction.numExtrema), FINAL_THREADS, finalSmem, stream>>>(
        reduction.partialLo, reduction.partialHi, reduction.lo, reduction.hi, reduction.numPartials);
    ApplyVarShape<IsPlanar, T>
        <<<applyGrid, block, 0, stream>>>(src, dst, reduction.lo, reduction.hi, numChannels, numPlanes, bound);
    NVCV_CHECK_THROW(cudaGetLastError());
    releaseGuard.finish();
}

// Pick the supported base type (u8 / u16 / f32 only) from a tensor/format dtype.
template<typename Cb>
inline void DispatchBaseType(nvcv::DataType dtype, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

#define CVCUDA_AC_BASE(DYN, STAT)                                                                 \
    ((dtype == nvcv::TYPE_4##DYN) || (dtype == nvcv::TYPE_3##DYN) || (dtype == nvcv::TYPE_2##DYN) \
     || (dtype == nvcv::TYPE_##DYN)) cb(STAT{});

    // clang-format off
    if      CVCUDA_AC_BASE(U8, uchar)
    else if CVCUDA_AC_BASE(U16, ushort)
    else if CVCUDA_AC_BASE(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input/output data type (only U8, U16, F32 are supported)");
    }
        // clang-format on

#undef CVCUDA_AC_BASE
}

// Build the vectorized value type from (base type, channels, planar-ness) and dispatch the callback.
template<typename Cb>
inline void DispatchType(nvcv::DataType dtype, int numInterleavedChannels, int numPlanes, const Cb &cb)
{
    DispatchBaseType(dtype,
                     [&](auto dummyBase)
                     {
                         using BaseT = decltype(dummyBase);
                         if (numInterleavedChannels == 1)
                         {
                             if (numPlanes == 1)
                             {
                                 cb(BaseT{}, std::integral_constant<bool, false>{});
                             }
                             else
                             {
                                 cb(BaseT{}, std::integral_constant<bool, true>{});
                             }
                         }
                         else if (numInterleavedChannels == 3)
                         {
                             cb(cuda::MakeType<BaseT, 3>{}, std::integral_constant<bool, false>{});
                         }
                         else if (numInterleavedChannels == 4)
                         {
                             cb(cuda::MakeType<BaseT, 4>{}, std::integral_constant<bool, false>{});
                         }
                         else
                         {
                             throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                   "Invalid number of channels (only 1, 3, 4 are supported)");
                         }
                     });
}

// ------------------------------- Validation --------------------------------

inline bool ValidateSrcDstTensors(int &numSamples, int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                  const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData)
{
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }
    if (!(srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC
          || srcData->layout() == nvcv::TENSOR_CHW || srcData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }
    if (srcData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor data type must be scalar; use the C dimension for image channels");
    }
    if (srcData->shape() != dstData->shape())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching tensor shapes");
    }

    const auto &shape  = srcData->shape();
    const auto &layout = srcData->layout();
    auto        extent = [&shape, &layout](char label, int64_t implicit)
    {
        const int index = layout.find(label);
        return index >= 0 ? shape[index] : implicit;
    };
    const int64_t rawSamples  = extent('N', 1);
    const int64_t rawRows     = extent('H', 1);
    const int64_t rawCols     = extent('W', 1);
    const int64_t rawChannels = extent('C', 1);
    if (rawSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size %ld exceeds the maximum supported (65535)", rawSamples);
    }
    if (rawRows > MAX_HEIGHT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image height %ld exceeds the maximum supported (%d)", rawRows, MAX_HEIGHT);
    }
    if (rawCols > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Image width %ld exceeds signed 32-bit kernel addressing",
                              rawCols);
    }
    if (rawChannels != 1 && rawChannels != 3 && rawChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %ld (only 1, 3, 4)",
                              rawChannels);
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    numSamples = srcAccess->numSamples();
    if (numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }
    // Each sample is mapped to grid.z in the reduce/apply kernels, so the batch size must fit
    // the CUDA grid.z limit.
    if (numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size %d exceeds the maximum supported (65535)", numSamples);
    }

    int numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel images are not supported");
    }
    if (numChannels != 1 && numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d (only 1, 3, 4)",
                              numChannels);
    }

    numPlanes = srcAccess->numPlanes();
    if (numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }
    if (srcAccess->numRows() > MAX_HEIGHT)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image height %ld exceeds the maximum supported (%d)", srcAccess->numRows(), MAX_HEIGHT);
    }

    const int64_t elementStride     = srcData->dtype().strideBytes();
    const int64_t expectedColStride = elementStride * (srcAccess->infoLayout().isChannelLast() ? numChannels : 1);
    if (srcAccess->colStride() != expectedColStride || dstAccess->colStride() != expectedColStride
        || (srcAccess->infoLayout().isChannelLast()
            && (srcAccess->chStride() != elementStride || dstAccess->chStride() != elementStride)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor pixels must have packed width and channel strides");
    }

    dtype                  = srcData->dtype();
    numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? numChannels : 1;
    return numSamples == 0 || srcAccess->numCols() == 0 || srcAccess->numRows() == 0;
}

inline void ValidateImagePlanes(const nvcv::Image &image, const nvcv::ImageFormat &format)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    if (!data || data->numPlanes() != format.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image plane descriptors must match the image format");
    }

    constexpr int64_t limit = cuda::TypeTraits<int32_t>::max;
    for (int p = 0; p < data->numPlanes(); ++p)
    {
        const nvcv::ImagePlaneStrided &plane       = data->plane(p);
        const nvcv::Size2D             expected    = format.planeSize(image.size(), p);
        const int64_t                  pixelStride = format.planePixelStrideBytes(p);
        if (plane.width != expected.w || plane.height != expected.h)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image plane descriptors must match the image format and size");
        }
        if (plane.rowStride < 0
            || (static_cast<int64_t>(plane.height - 1) * plane.rowStride
                    + static_cast<int64_t>(plane.width - 1) * pixelStride
                > limit))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW,
                                  "Input or output image-plane maximum byte offset exceeds %d",
                                  cuda::TypeTraits<int32_t>::max);
        }
    }
}

inline auto ValidateSrcDstVarShape(int &numSamples, int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                   cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                   const nvcv::ImageBatchVarShape &dst)
{
    using MaybeVarShape = nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda>;
    std::tuple<MaybeVarShape, MaybeVarShape> srcDstData{
        src.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream),
        dst.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream)};
    auto &[srcData, dstData] = srcDstData;

    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }
    if (numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size %d exceeds the maximum supported (65535)", numSamples);
    }
    if (numSamples == 0)
    {
        return srcDstData;
    }

    const auto &srcFormat = srcData->uniqueFormat();
    const auto &dstFormat = dstData->uniqueFormat();
    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }
    if (srcFormat != dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    int numChannels = srcFormat.numChannels();
    if (numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel images are not supported");
    }
    if (numChannels != 1 && numChannels != 3 && numChannels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d (only 1, 3, 4)",
                              numChannels);
    }

    nvcv::ExtraChannelInfo extraChannels{};
    srcFormat.extraChannelInfo(&extraChannels);
    if (extraChannels.numChannels != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image formats with extra channels are not supported");
    }
    if (srcFormat.chromaSubsampling() != nvcv::ChromaSubsampling::CSS_444)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Chroma-subsampled image formats are not supported");
    }

    numPlanes = srcFormat.numPlanes();
    dtype     = srcFormat.planeDataType(0);
    if (numPlanes == 1)
    {
        if (dtype.numChannels() != numChannels || srcFormat.planePixelStrideBytes(0) != dtype.strideBytes())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Single-plane images must have one packed element per pixel");
        }
    }
    else if (numPlanes != numChannels || dtype.numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar images must have one scalar, full-resolution plane per channel");
    }
    for (int p = 1; p < numPlanes; ++p)
    {
        if (dtype != srcFormat.planeDataType(p) || srcFormat.planeDataType(p).numChannels() != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All image planes must have the same scalar data type");
        }
    }

    for (int i = 0; i < numSamples; ++i)
    {
        const nvcv::Size2D size = src[i].size();
        if (size != dst[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output must have matching width and height");
        }
        if (size.w <= 0 || size.h <= 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image width and height must be greater than zero");
        }
        if (size.h > MAX_HEIGHT)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Image height %d exceeds the maximum supported (%d)", size.h, MAX_HEIGHT);
        }
        ValidateImagePlanes(src[i], srcFormat);
        ValidateImagePlanes(dst[i], dstFormat);
        const nvcv::Size2D plane0Size = srcFormat.planeSize(size, 0);
        for (int p = 1; p < numPlanes; ++p)
        {
            if (srcFormat.planeSize(size, p) != plane0Size)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "All image planes must have matching width and height");
            }
        }
    }

    numInterleavedChannels = dtype.numChannels();
    return srcDstData;
}

inline void RejectStreamCapture(cudaStream_t stream)
{
    // Captured kernels would retain the shared workspace pointer but replay outside the host-side
    // submission lock. Reject before validation can enqueue work, rather than silently aliasing it.
    cudaStreamCaptureStatus captureStatus;
    NVCV_CHECK_THROW(cudaStreamIsCapturing(stream, &captureStatus));
    if (captureStatus != cudaStreamCaptureStatusNone)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_OPERATION,
                              "AutoContrast does not support CUDA stream capture");
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// --------------------------------- Workspace -------------------------------

AutoContrastWorkspace::AutoContrastWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_ready, cudaEventDisableTiming));
}

AutoContrastWorkspace::~AutoContrastWorkspace()
{
    if (m_pending)
    {
        NVCV_CHECK_LOG(cudaEventSynchronize(m_ready));
    }
    if (m_data != nullptr)
    {
        NVCV_CHECK_LOG(cudaFree(m_data));
    }
    if (m_ready != nullptr)
    {
        NVCV_CHECK_LOG(cudaEventDestroy(m_ready));
    }
}

[[nodiscard]] std::unique_lock<std::mutex> AutoContrastWorkspace::lock()
{
    return std::unique_lock<std::mutex>{m_mutex};
}

float *AutoContrastWorkspace::acquire(size_t count, cudaStream_t stream)
{
    if (count > std::numeric_limits<size_t>::max() / 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AutoContrast reduction workspace size overflows size_t");
    }
    const size_t need = 2 * count;
    if (need > std::numeric_limits<size_t>::max() / sizeof(float))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AutoContrast reduction workspace size overflows size_t");
    }

    if (m_pending)
    {
        if (need > m_capacity)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(m_ready));
            m_pending = false;
        }
        else
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, m_ready));
        }
    }
    if (need > m_capacity)
    {
        if (m_data != nullptr)
        {
            NVCV_CHECK_THROW(cudaFree(m_data));
            m_data     = nullptr;
            m_capacity = 0;
        }
        void *ptr = nullptr;
        if (cudaMalloc(&ptr, need * sizeof(float)) != cudaSuccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "AutoContrast: failed to allocate workspace");
        }
        m_data     = static_cast<float *>(ptr);
        m_capacity = need;
    }
    return m_data;
}

void AutoContrastWorkspace::release(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(m_ready, stream));
    m_pending = true;
}

void AutoContrastWorkspace::releaseNoThrow(cudaStream_t stream) noexcept
{
    cudaError_t err = cudaEventRecord(m_ready, stream);
    if (err == cudaSuccess)
    {
        m_pending = true;
    }
    NVCV_CHECK_LOG(err);
}

// --------------------------------- Operator --------------------------------

AutoContrast::AutoContrast()
    : m_workspace([](int) { return std::make_unique<AutoContrastWorkspace>(); })
{
}

void AutoContrast::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AutoContrast::operator()[Tensor]");
    RejectStreamCapture(stream);

    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = in.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = out.exportData<nvcv::TensorDataStridedCuda>();
    const bool isEmpty = ValidateSrcDstTensors(numSamples, numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    const int numChannels = numInterleavedChannels > 1 ? numInterleavedChannels : numPlanes;

    DispatchType(dtype, numInterleavedChannels, numPlanes,
                 [&](auto dummyVal, auto isPlanar)
                 {
                     using T             = decltype(dummyVal);
                     constexpr bool Plnr = decltype(isPlanar)::value;
                     if (!isEmpty)
                     {
                         auto &workspace = m_workspace.get();
                         auto  lock      = workspace.lock();
                         RunTensor<Plnr, T>(stream, workspace, *srcData, *dstData, numSamples, numChannels, numPlanes);
                     }
                 });
}

void AutoContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                              const nvcv::ImageBatchVarShape &out) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AutoContrast::operator()[ImageBatchVarShape]");
    RejectStreamCapture(stream);

    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto srcDstData = ValidateSrcDstVarShape(numSamples, numInterleavedChannels, numPlanes, dtype, stream, in, out);
    if (numSamples == 0)
    {
        return;
    }

    const int numChannels = numInterleavedChannels > 1 ? numInterleavedChannels : numPlanes;

    DispatchType(dtype, numInterleavedChannels, numPlanes,
                 [&](auto dummyVal, auto isPlanar)
                 {
                     using T                  = decltype(dummyVal);
                     constexpr bool Plnr      = decltype(isPlanar)::value;
                     auto &[srcData, dstData] = srcDstData;
                     auto &workspace          = m_workspace.get();
                     auto  lock               = workspace.lock();
                     RunVarShapeBatch<Plnr, T>(stream, workspace, *srcData, *dstData, numSamples, numChannels,
                                               numPlanes);
                 });
}

} // namespace cvcuda::priv
