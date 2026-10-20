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

#include "PhotometricBound.cuh"
#include "SameShapeCommon.cuh"

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

// S16 extrema make the divisor an integer in [1, 65535] and the rounded numerator
// finite in [0, 65535 * 32767]. The ordinary division's underflow/overflow slow
// path is unnecessary, including for zero numerators. Keep its RN correction.
inline __device__ float DivideS16Range(float numerator, float range)
{
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(range));
    const float error     = __fmaf_rn(-range, reciprocal, 1.f);
    reciprocal            = __fmaf_rn(reciprocal, error, reciprocal);
    const float quotient  = __fmaf_rn(numerator, reciprocal, 0.f);
    const float remainder = __fmaf_rn(-range, quotient, numerator);
    return __fmaf_rn(reciprocal, remainder, quotient);
}

// AutoContrast remap, shared by every layout/container. A flat channel passes
// through unchanged. Integer results are clamped and truncated to match Pillow
// and torchvision; floating-point results retain the scaled value.
template<typename BT>
inline __device__ BT RemapPixel(BT in, float lo, float hi, float bound)
{
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        // Widening to float is exact for both float and half, and there is no isfinite(__half).
        if (!isfinite(static_cast<float>(in)))
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
    if constexpr (std::is_same_v<BT, short>)
    {
        val = DivideS16Range((static_cast<float>(in) - lo) * bound, range);
    }
    else if (isfinite(range))
    {
        val = (static_cast<float>(in) - lo) * bound / range;
    }
    else
    {
        // Unreachable for F16: half extrema widen to float exactly and their range is finite.
        if constexpr (cuda::detail::IsFloatingPointV<BT>)
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
    // F16 sources can carry non-finite values too; value is already widened to float.
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
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

// Element-stream apply. ApplyTensor below is ALU-bound; warp-contiguous vectors carry a rotating channel phase.

constexpr int kStreamBlock    = 256;
constexpr int kStreamVecBytes = 16;

using StreamVec = uint4;

struct StreamShape
{
    int32_t segElems;
    int32_t numSegments;
    int32_t segStride;
    int32_t sampleStride;
    int32_t planeStride;
};

template<typename BT>
inline constexpr int kStreamElemsPerVec = kStreamVecBytes / static_cast<int>(sizeof(BT));

// Selects keep indices literal; a runtime index would spill.
template<int C>
inline __device__ void RotateExtrema(int r, const float (&loV)[C], const float (&hiV)[C], float (&rl)[C],
                                     float (&rh)[C], int (&rt)[C])
{
#pragma unroll
    for (int j = 0; j < C; ++j)
    {
        const int c = (r + j) < C ? (r + j) : (r + j - C);
        float     l = loV[0];
        float     h = hiV[0];
        int       t = 0;
#pragma unroll
        for (int m = 1; m < C; ++m)
        {
            const bool sel = (c == m);
            l              = sel ? loV[m] : l;
            h              = sel ? hiV[m] : h;
            t              = sel ? (m << 8) : t;
        }
        rl[j] = l;
        rh[j] = h;
        rt[j] = t;
    }
}

// RemapPixel with the flat and non-finite tests hoisted out; both are uniform per (sample, channel).
template<typename BT>
inline __device__ BT RemapNormal(BT in, float lo, float hi, float bound)
{
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        // IsFloatingPointV, not std::is_floating_point_v: the latter excludes __half and drops this for F16.
        if (!isfinite(static_cast<float>(in)))
        {
            return in;
        }
    }
    float val;
    if constexpr (std::is_same_v<BT, short>)
    {
        val = DivideS16Range((static_cast<float>(in) - lo) * bound, hi - lo);
    }
    else
    {
        val = (static_cast<float>(in) - lo) * bound / (hi - lo);
    }
    return static_cast<BT>(fminf(fmaxf(val, 0.f), bound));
}

// `Skew` advances by a compile-time constant, so one RotateExtrema serves the whole thread.
template<typename BT, int C, bool UseLut, bool AllNormal, int Skew>
inline __device__ void RemapVector(BT *b, const float (&rl)[C], const float (&rh)[C], const int (&rt)[C], float bound,
                                   const unsigned char *lut)
{
    constexpr int E = kStreamElemsPerVec<BT>;
#pragma unroll
    for (int i = 0; i < E; ++i)
    {
        const int j = (Skew + i) % C;
        if constexpr (UseLut)
        {
            b[i] = static_cast<BT>(lut[rt[j] + static_cast<int>(b[i])]);
        }
        else if constexpr (AllNormal)
        {
            b[i] = RemapNormal<BT>(b[i], rl[j], rh[j], bound);
        }
        else
        {
            b[i] = RemapPixel<BT>(b[i], rl[j], rh[j], bound);
        }
    }
}

template<int K, int VPT, typename BT, int C, bool UseLut, bool AllNormal, int Skew>
inline __device__ void RemapVectors(StreamVec *v, const float (&rl)[C], const float (&rh)[C], const int (&rt)[C],
                                    float bound, const unsigned char *lut)
{
    if constexpr (K < VPT)
    {
        RemapVector<BT, C, UseLut, AllNormal, (K * Skew) % C>(reinterpret_cast<BT *>(&v[K]), rl, rh, rt, bound, lut);
        RemapVectors<K + 1, VPT, BT, C, UseLut, AllNormal, Skew>(v, rl, rh, rt, bound, lut);
    }
}

template<bool IsPlanar, typename BT, int C, int VPT, bool UseLut>
__global__ __launch_bounds__(kStreamBlock) void ApplyStream(const BT *__restrict__ src, BT *__restrict__ dst,
                                                            const float *__restrict__ lo, const float *__restrict__ hi,
                                                            StreamShape shape, int numChannels, float bound)
{
    constexpr int E    = kStreamElemsPerVec<BT>;
    constexpr int kTab = IsPlanar ? 1 : C;

    __shared__ unsigned char lut[UseLut ? kTab * 256 : 1];

    const int sample = blockIdx.z;
    const int plane  = IsPlanar ? static_cast<int>(blockIdx.y) / shape.numSegments : 0;
    const int seg    = IsPlanar ? static_cast<int>(blockIdx.y) % shape.numSegments : static_cast<int>(blockIdx.y);

    const int sampleOffset = sample * numChannels;
    float     loV[C], hiV[C];
#pragma unroll
    for (int c = 0; c < C; ++c)
    {
        const int idx = IsPlanar ? (sampleOffset + plane) : (sampleOffset + c);
        loV[c]        = lo[idx];
        hiV[c]        = hi[idx];
    }

    bool allNormal = true;
#pragma unroll
    for (int c = 0; c < C; ++c)
    {
        allNormal = allNormal && (hiV[c] != loV[c]) && isfinite(hiV[c] - loV[c]);
    }

    if constexpr (UseLut)
    {
        // Built from the same RemapPixel call, and a 1-byte type has exactly 256 inputs.
#pragma unroll
        for (int c = 0; c < kTab; ++c)
        {
            for (int v = threadIdx.x; v < 256; v += kStreamBlock)
            {
                lut[(c << 8) + v]
                    = static_cast<unsigned char>(RemapPixel<BT>(static_cast<BT>(v), loV[c], hiV[c], bound));
            }
        }
        __syncthreads();
    }

    const int32_t base        = sample * shape.sampleStride + plane * shape.planeStride + seg * shape.segStride;
    const BT *__restrict__ sp = src + base;
    BT *__restrict__ dp       = dst + base;

    const int32_t tile     = static_cast<int32_t>(blockIdx.x) * (VPT * kStreamBlock * E);
    const int32_t segElems = shape.segElems;

    const int32_t oBase = tile + static_cast<int32_t>(threadIdx.x) * E;
    // Span test hoisted out of the unrolled body; per-vector guards serialise the loads.
    if (oBase + (VPT - 1) * kStreamBlock * E + E <= segElems)
    {
        StreamVec v[VPT];
#pragma unroll
        for (int k = 0; k < VPT; ++k)
        {
            v[k] = *reinterpret_cast<const StreamVec *>(sp + oBase + k * kStreamBlock * E);
        }
        const int r0 = C == 1 ? 0 : static_cast<int>(oBase % C);
        float     rl[C], rh[C];
        int       rt[C];
        RotateExtrema<C>(r0, loV, hiV, rl, rh, rt);
        constexpr int kSkew = (kStreamBlock * E) % C;
        if (allNormal)
        {
            RemapVectors<0, VPT, BT, C, UseLut, true, kSkew>(v, rl, rh, rt, bound, lut);
        }
        else
        {
            RemapVectors<0, VPT, BT, C, UseLut, false, kSkew>(v, rl, rh, rt, bound, lut);
        }
#pragma unroll
        for (int k = 0; k < VPT; ++k)
        {
            *reinterpret_cast<StreamVec *>(dp + oBase + k * kStreamBlock * E) = v[k];
        }
        return;
    }

#pragma unroll
    for (int k = 0; k < VPT; ++k)
    {
        const int32_t o = oBase + k * kStreamBlock * E;
        if (o >= segElems)
        {
            break;
        }
        const int r = C == 1 ? 0 : static_cast<int>(o % C);
        float     rl[C], rh[C];
        int       rt[C];
        RotateExtrema<C>(r, loV, hiV, rl, rh, rt);

        if (o + E <= segElems)
        {
            StreamVec v = *reinterpret_cast<const StreamVec *>(sp + o);
            BT       *b = reinterpret_cast<BT *>(&v);
            RemapVector<BT, C, UseLut, false, 0>(b, rl, rh, rt, bound, lut);
            *reinterpret_cast<StreamVec *>(dp + o) = v;
        }
        else
        {
            for (int32_t e = o; e < segElems; ++e)
            {
                const int j  = C == 1 ? 0 : static_cast<int>(e - o) % C;
                float     l  = rl[0];
                float     h  = rh[0];
                int       tb = rt[0];
#pragma unroll
                for (int m = 1; m < C; ++m)
                {
                    const bool sel = (j == m);
                    l              = sel ? rl[m] : l;
                    h              = sel ? rh[m] : h;
                    tb             = sel ? rt[m] : tb;
                }
                if constexpr (UseLut)
                {
                    dp[e] = static_cast<BT>(lut[tb + static_cast<int>(sp[e])]);
                }
                else
                {
                    dp[e] = RemapPixel<BT>(sp[e], l, h, bound);
                }
            }
        }
    }
}

// Packed RGBF32 segments with no tail use a lean application kernel. Keeping the
// general tail path out of this instantiation avoids extending the live ranges of
// the per-channel extrema through the hot vector body.
constexpr int kF32C3FullBlock = 320;
constexpr int kF32C3FullVPT   = 2;

__global__ __launch_bounds__(kF32C3FullBlock) void ApplyF32C3FullStream(const float *__restrict__ src,
                                                                        float *__restrict__ dst,
                                                                        const float *__restrict__ lo,
                                                                        const float *__restrict__ hi, StreamShape shape)
{
    constexpr int E       = kStreamElemsPerVec<float>;
    constexpr int perTile = (kF32C3FullVPT + 1) * kF32C3FullBlock * E;

    const int sample = blockIdx.z;
    const int seg    = blockIdx.y;

    const int   sampleOffset = sample * 3;
    const int   lane         = static_cast<int>(threadIdx.x) & (WARP_WIDTH - 1);
    const float laneLo       = lane < 3 ? lo[sampleOffset + lane] : 0.f;
    const float laneHi       = lane < 3 ? hi[sampleOffset + lane] : 0.f;
    float       loV[3], hiV[3];
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        loV[c] = __shfl_sync(0xffffffffu, laneLo, c);
        hiV[c] = __shfl_sync(0xffffffffu, laneHi, c);
    }

    bool allNormal = true;
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        allNormal = allNormal && (hiV[c] != loV[c]) && isfinite(hiV[c] - loV[c]);
    }

    const int32_t base = sample * shape.sampleStride + seg * shape.segStride;
    const int32_t tile = static_cast<int32_t>(blockIdx.x) * perTile;
    const int32_t o    = tile + static_cast<int32_t>(threadIdx.x) * E;

    const int r0 = static_cast<int>(o % 3);
    float     rl[3], rh[3];
    int       rt[3];
    RotateExtrema<3>(r0, loV, hiV, rl, rh, rt);

    StreamVec v[kF32C3FullVPT];
#pragma unroll
    for (int k = 0; k < kF32C3FullVPT; ++k)
    {
        v[k] = *reinterpret_cast<const StreamVec *>(src + base + o + k * kF32C3FullBlock * E);
    }
    if (allNormal)
    {
        RemapVectors<0, kF32C3FullVPT, float, 3, false, true, (kF32C3FullBlock * E) % 3>(v, rl, rh, rt, 1.f, nullptr);
    }
    else
    {
        RemapVectors<0, kF32C3FullVPT, float, 3, false, false, (kF32C3FullBlock * E) % 3>(v, rl, rh, rt, 1.f, nullptr);
    }
#pragma unroll
    for (int k = 0; k < kF32C3FullVPT; ++k)
    {
        *reinterpret_cast<StreamVec *>(dst + base + o + k * kF32C3FullBlock * E) = v[k];
    }

    constexpr int extraOffset = kF32C3FullVPT * kF32C3FullBlock * E;
    StreamVec     extra       = *reinterpret_cast<const StreamVec *>(src + base + o + extraOffset);
    if (allNormal)
    {
        RemapVector<float, 3, false, true, extraOffset % 3>(reinterpret_cast<float *>(&extra), rl, rh, rt, 1.f,
                                                            nullptr);
    }
    else
    {
        RemapVector<float, 3, false, false, extraOffset % 3>(reinterpret_cast<float *>(&extra), rl, rh, rt, 1.f,
                                                             nullptr);
    }
    *reinterpret_cast<StreamVec *>(dst + base + o + extraOffset) = extra;
}

// Element-stream reduce. Only the order values reach a lane changes, and fminf/fmaxf are associative.

template<bool IsPlanar, typename BT, int C, int VPT>
__global__ __launch_bounds__(kStreamBlock) void ReduceStream(const BT *__restrict__ src, float *partialLo,
                                                             float *partialHi, StreamShape shape, int numChannels,
                                                             int numPartials)
{
    extern __shared__ float sm[];
    constexpr int           E = kStreamElemsPerVec<BT>;

    const int sample = blockIdx.z;
    const int plane  = IsPlanar ? static_cast<int>(blockIdx.y) / shape.numSegments : 0;
    const int seg    = IsPlanar ? static_cast<int>(blockIdx.y) % shape.numSegments : static_cast<int>(blockIdx.y);

    const int32_t base        = sample * shape.sampleStride + plane * shape.planeStride + seg * shape.segStride;
    const BT *__restrict__ sp = src + base;

    // One accumulator per rotation slot; a vector spanning whole pixels has zero phase and needs one.
    constexpr int SLOTS = (E % C == 0) ? 1 : VPT;
    float         aMin[SLOTS][C], aMax[SLOTS][C];
#pragma unroll
    for (int k = 0; k < SLOTS; ++k)
    {
#pragma unroll
        for (int j = 0; j < C; ++j)
        {
            aMin[k][j] = CUDART_INF_F;
            aMax[k][j] = -CUDART_INF_F;
        }
    }

    const int32_t segElems = shape.segElems;
    const int32_t perGrid  = static_cast<int32_t>(gridDim.x) * (VPT * kStreamBlock * E);
    for (int32_t tile = static_cast<int32_t>(blockIdx.x) * (VPT * kStreamBlock * E); tile < segElems; tile += perGrid)
    {
        const int32_t oBase = tile + static_cast<int32_t>(threadIdx.x) * E;
        if (oBase + (VPT - 1) * kStreamBlock * E + E <= segElems)
        {
            StreamVec v[VPT];
#pragma unroll
            for (int k = 0; k < VPT; ++k)
            {
                v[k] = *reinterpret_cast<const StreamVec *>(sp + oBase + k * kStreamBlock * E);
            }
#pragma unroll
            for (int k = 0; k < VPT; ++k)
            {
                const BT *b = reinterpret_cast<const BT *>(&v[k]);
#pragma unroll
                for (int i = 0; i < E; ++i)
                {
                    AccumulateExtrema<BT>(static_cast<float>(b[i]), aMin[SLOTS == 1 ? 0 : k][i % C],
                                          aMax[SLOTS == 1 ? 0 : k][i % C]);
                }
            }
        }
        else
        {
#pragma unroll
            for (int k = 0; k < VPT; ++k)
            {
                const int32_t o = oBase + k * kStreamBlock * E;
                if (o >= segElems)
                {
                    break;
                }
                if (o + E <= segElems)
                {
                    StreamVec v = *reinterpret_cast<const StreamVec *>(sp + o);
                    const BT *b = reinterpret_cast<const BT *>(&v);
#pragma unroll
                    for (int i = 0; i < E; ++i)
                    {
                        AccumulateExtrema<BT>(static_cast<float>(b[i]), aMin[SLOTS == 1 ? 0 : k][i % C],
                                              aMax[SLOTS == 1 ? 0 : k][i % C]);
                    }
                }
                else
                {
                    for (int32_t e = o; e < segElems; ++e)
                    {
                        const int   rel = C == 1 ? 0 : static_cast<int>(e - o) % C;
                        const float val = static_cast<float>(sp[e]);
#pragma unroll
                        for (int m = 0; m < C; ++m)
                        {
                            if (rel == m)
                            {
                                AccumulateExtrema<BT>(val, aMin[SLOTS == 1 ? 0 : k][m], aMax[SLOTS == 1 ? 0 : k][m]);
                            }
                        }
                    }
                }
            }
        }
    }

    float vMin[C], vMax[C];
#pragma unroll
    for (int c = 0; c < C; ++c)
    {
        vMin[c] = CUDART_INF_F;
        vMax[c] = -CUDART_INF_F;
    }
#pragma unroll
    for (int k = 0; k < SLOTS; ++k)
    {
        const int32_t o = static_cast<int32_t>(blockIdx.x) * (VPT * kStreamBlock * E)
                        + (k * kStreamBlock + static_cast<int32_t>(threadIdx.x)) * E;
        const int r = (C == 1 || SLOTS == 1) ? 0 : static_cast<int>(o % C);
#pragma unroll
        for (int j = 0; j < C; ++j)
        {
            const int c = (r + j) < C ? (r + j) : (r + j - C);
#pragma unroll
            for (int m = 0; m < C; ++m)
            {
                if (c == m)
                {
                    vMin[m] = fminf(vMin[m], aMin[k][j]);
                    vMax[m] = fmaxf(vMax[m], aMax[k][j]);
                }
            }
        }
    }

    const int partial   = seg * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int extremaId = sample * numChannels * numPartials + (IsPlanar ? plane * numPartials : 0) + partial;
    ReduceBlockToPartials<C>(sm, kStreamBlock, static_cast<int>(threadIdx.x), C, vMin, vMax, &partialLo[extremaId],
                             &partialHi[extremaId], numPartials);
}

// A full RGBF32 tile gives every thread four values from each channel. Accumulate
// those three channels directly instead of retaining one rotated accumulator set
// per vector; the value order within each channel stays identical to ReduceStream.
__global__ __launch_bounds__(kStreamBlock) void ReduceF32C3FullStream(const float *__restrict__ src, float *partialLo,
                                                                      float *partialHi, StreamShape shape,
                                                                      int numChannels, int numPartials)
{
    extern __shared__ float sm[];
    constexpr int           E       = kStreamElemsPerVec<float>;
    constexpr int           perTile = 3 * kStreamBlock * E;

    const int sample = blockIdx.z;
    const int seg    = blockIdx.y;

    const int32_t base = sample * shape.sampleStride + seg * shape.segStride;
    const int32_t o    = static_cast<int32_t>(blockIdx.x) * perTile + static_cast<int32_t>(threadIdx.x) * E;

    StreamVec v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        v[k] = *reinterpret_cast<const StreamVec *>(src + base + o + k * kStreamBlock * E);
    }

    const float *b0 = reinterpret_cast<const float *>(&v[0]);
    const float *b1 = reinterpret_cast<const float *>(&v[1]);
    const float *b2 = reinterpret_cast<const float *>(&v[2]);

    float qMin[3]{CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
    float qMax[3]{-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};

    AccumulateExtrema<float>(b0[0], qMin[0], qMax[0]);
    AccumulateExtrema<float>(b0[3], qMin[0], qMax[0]);
    AccumulateExtrema<float>(b1[2], qMin[0], qMax[0]);
    AccumulateExtrema<float>(b2[1], qMin[0], qMax[0]);

    AccumulateExtrema<float>(b0[1], qMin[1], qMax[1]);
    AccumulateExtrema<float>(b1[0], qMin[1], qMax[1]);
    AccumulateExtrema<float>(b1[3], qMin[1], qMax[1]);
    AccumulateExtrema<float>(b2[2], qMin[1], qMax[1]);

    AccumulateExtrema<float>(b0[2], qMin[2], qMax[2]);
    AccumulateExtrema<float>(b1[1], qMin[2], qMax[2]);
    AccumulateExtrema<float>(b2[0], qMin[2], qMax[2]);
    AccumulateExtrema<float>(b2[3], qMin[2], qMax[2]);

    const int r = static_cast<int>(threadIdx.x % 3);
    float     vMin[3], vMax[3];
    vMin[0] = r == 0 ? qMin[0] : (r == 1 ? qMin[2] : qMin[1]);
    vMin[1] = r == 0 ? qMin[1] : (r == 1 ? qMin[0] : qMin[2]);
    vMin[2] = r == 0 ? qMin[2] : (r == 1 ? qMin[1] : qMin[0]);
    vMax[0] = r == 0 ? qMax[0] : (r == 1 ? qMax[2] : qMax[1]);
    vMax[1] = r == 0 ? qMax[1] : (r == 1 ? qMax[0] : qMax[2]);
    vMax[2] = r == 0 ? qMax[2] : (r == 1 ? qMax[1] : qMax[0]);

    const int partial   = seg * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int extremaId = sample * numChannels * numPartials + partial;
    ReduceBlockToPartials<3>(sm, kStreamBlock, static_cast<int>(threadIdx.x), 3, vMin, vMax, &partialLo[extremaId],
                             &partialHi[extremaId], numPartials);
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

// Declining leaves the wrapper-indexed ApplyTensor path in charge, so every shape stays supported.
template<bool IsPlanar, typename BT, class Access>
inline bool MakeStreamShape(const Access &srcAccess, const Access &dstAccess, const void *srcPtr, const void *dstPtr,
                            int numSamples, int numPlanes, StreamShape &shape)
{
    constexpr int64_t elemSize  = sizeof(BT);
    const int64_t     rowStride = srcAccess.rowStride();
    if (rowStride != dstAccess.rowStride() || srcAccess.sampleStride() != dstAccess.sampleStride())
    {
        return false;
    }
    if (rowStride % elemSize != 0 || srcAccess.sampleStride() % elemSize != 0)
    {
        return false;
    }
    auto aligned = [](int64_t v)
    {
        return v % kStreamVecBytes == 0;
    };
    if (!aligned(reinterpret_cast<uintptr_t>(srcPtr)) || !aligned(reinterpret_cast<uintptr_t>(dstPtr))
        || !aligned(srcAccess.sampleStride()))
    {
        return false;
    }

    const int64_t rowElems   = IsPlanar ? srcAccess.numCols() : srcAccess.numCols() * srcAccess.numChannels();
    const bool    rowsPacked = (rowStride == rowElems * elemSize);

    int64_t planeStride = 0;
    if constexpr (IsPlanar)
    {
        if (srcAccess.planeStride() != dstAccess.planeStride() || srcAccess.planeStride() % elemSize != 0
            || !aligned(srcAccess.planeStride()))
        {
            return false;
        }
        planeStride = srcAccess.planeStride() / elemSize;
    }

    int64_t segElems, numSegments, segStride;
    if (rowsPacked)
    {
        segElems    = rowElems * srcAccess.numRows();
        numSegments = 1;
        segStride   = 0;
    }
    else
    {
        if (!aligned(rowStride))
        {
            return false;
        }
        segElems    = rowElems;
        numSegments = srcAccess.numRows();
        segStride   = rowStride / elemSize;
    }

    const int64_t gridY = numSegments * (IsPlanar ? numPlanes : 1);
    if (segElems <= 0 || gridY > 65535 || numSamples > 65535)
    {
        return false;
    }
    constexpr int64_t i32max = cuda::TypeTraits<int32_t>::max;
    if (segElems > i32max || segStride > i32max || planeStride > i32max || srcAccess.sampleStride() / elemSize > i32max)
    {
        return false;
    }

    shape.segElems     = static_cast<int32_t>(segElems);
    shape.numSegments  = static_cast<int32_t>(numSegments);
    shape.segStride    = static_cast<int32_t>(segStride);
    shape.sampleStride = static_cast<int32_t>(srcAccess.sampleStride() / elemSize);
    shape.planeStride  = static_cast<int32_t>(planeStride);
    return true;
}

// One vector wins where the carrier spans whole pixels; C3 prefers three to reuse the rotation.
template<typename BT, int C>
inline constexpr int kStreamVPT = (kStreamElemsPerVec<BT> % C == 0) ? 4 : 3;

template<bool IsPlanar, typename BT, int C>
inline bool LaunchApplyStream(cudaStream_t stream, const BT *src, BT *dst, const float *lo, const float *hi,
                              const StreamShape &shape, int numSamples, int numChannels, int numPlanes, float bound)
{
    constexpr int     E        = kStreamElemsPerVec<BT>;
    constexpr int     VPT      = kStreamVPT<BT, C>;
    constexpr bool    UseLut   = std::is_same_v<BT, unsigned char>;
    constexpr int64_t fullTile = static_cast<int64_t>(kF32C3FullVPT + 1) * kF32C3FullBlock * kStreamElemsPerVec<float>;

    if constexpr (!IsPlanar && std::is_same_v<BT, float> && C == 3)
    {
        if (shape.segElems % fullTile == 0)
        {
            const int64_t gridX = shape.segElems / fullTile;
            if (gridX <= 0 || gridX > cuda::TypeTraits<int32_t>::max)
            {
                return false;
            }
            dim3 grid(static_cast<unsigned int>(gridX), static_cast<unsigned int>(shape.numSegments),
                      static_cast<unsigned int>(numSamples));
            ApplyF32C3FullStream<<<grid, kF32C3FullBlock, 0, stream>>>(src, dst, lo, hi, shape);
            return true;
        }
    }

    const int64_t perTile = static_cast<int64_t>(VPT) * kStreamBlock * E;
    const int64_t gridX   = (shape.segElems + perTile - 1) / perTile;
    if (gridX <= 0 || gridX > cuda::TypeTraits<int32_t>::max)
    {
        return false;
    }
    dim3 grid(static_cast<unsigned int>(gridX),
              static_cast<unsigned int>(shape.numSegments * (IsPlanar ? numPlanes : 1)),
              static_cast<unsigned int>(numSamples));
    ApplyStream<IsPlanar, BT, C, VPT, UseLut>
        <<<grid, kStreamBlock, 0, stream>>>(src, dst, lo, hi, shape, numChannels, bound);
    return true;
}

// Bounded so the partial count stays inside the workspace budget; the grid-stride loop covers rest.
inline int StreamReduceGridX(int64_t segElems, int64_t perTile, int64_t numSegments, int64_t numExtrema)
{
    constexpr int64_t kWorkspaceOneSide = static_cast<int64_t>(MAX_REDUCTION_WORKSPACE_ONE_SIDE);
    if (numExtrema <= 0 || numExtrema > kWorkspaceOneSide / 2)
    {
        return 0;
    }
    const int64_t byWorkspace = kWorkspaceOneSide / numExtrema - 1;
    const int64_t maxPartials = std::min<int64_t>(MAX_PARTIALS_PER_SAMPLE, byWorkspace);
    if (maxPartials < numSegments)
    {
        return 0; // one partial per segment is the floor this layout can express
    }
    const int64_t need  = (segElems + perTile - 1) / perTile;
    const int64_t byCap = maxPartials / numSegments;
    const int64_t gridX = std::min<int64_t>(need, byCap);
    return gridX > 0 && gridX <= cuda::TypeTraits<int32_t>::max ? static_cast<int>(gridX) : 0;
}

template<bool IsPlanar, typename BT, int C>
inline void LaunchReduceStream(cudaStream_t stream, const BT *src, float *partialLo, float *partialHi,
                               const StreamShape &shape, int numSamples, int numChannels, int numPlanes, int gridX,
                               int numPartials)
{
    constexpr int VPT   = kStreamVPT<BT, C>;
    constexpr int E     = kStreamElemsPerVec<BT>;
    const int64_t gridY = static_cast<int64_t>(shape.numSegments) * (IsPlanar ? numPlanes : 1);
    NVCV_ASSERT(gridY <= 65535 && gridX > 0);
    const size_t smem = 2 * static_cast<size_t>(C) * (kStreamBlock / WARP_WIDTH) * sizeof(float);
    dim3         grid(static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY),
                      static_cast<unsigned int>(numSamples));

    if constexpr (!IsPlanar && std::is_same_v<BT, float> && C == 3)
    {
        constexpr int64_t perTile = static_cast<int64_t>(3) * kStreamBlock * E;
        if (shape.segElems % perTile == 0 && gridX == shape.segElems / perTile)
        {
            ReduceF32C3FullStream<<<grid, kStreamBlock, smem, stream>>>(src, partialLo, partialHi, shape, numChannels,
                                                                        numPartials);
            return;
        }
    }

    ReduceStream<IsPlanar, BT, C, VPT>
        <<<grid, kStreamBlock, smem, stream>>>(src, partialLo, partialHi, shape, numChannels, numPartials);
}

template<bool IsPlanar, typename T>
void RunTensor(cudaStream_t stream, cvcuda::priv::AutoContrastWorkspace &ws, const nvcv::TensorDataStridedCuda &srcData,
               const nvcv::TensorDataStridedCuda &dstData, int numSamples, int numChannels, int numPlanes)
{
    using BT          = cuda::BaseType<T>;
    const float bound = cvcuda::priv::PhotometricUpperBound<BT, float>();

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

    constexpr int     C       = IsPlanar ? 1 : cuda::NumElements<T>;
    constexpr int     E       = kStreamElemsPerVec<BT>;
    constexpr int     VPT     = kStreamVPT<BT, C>;
    constexpr int64_t perTile = static_cast<int64_t>(VPT) * kStreamBlock * E;
    StreamShape       shape{};
    const bool canStream   = MakeStreamShape<IsPlanar, BT>(*srcAccess, *dstAccess, srcData.basePtr(), dstData.basePtr(),
                                                         numSamples, numPlanes, shape);
    int        streamGridX = 0;
    if (canStream && static_cast<int64_t>(shape.numSegments) * (IsPlanar ? numPlanes : 1) <= 65535)
    {
        streamGridX = StreamReduceGridX(shape.segElems, perTile, shape.numSegments,
                                        static_cast<int64_t>(numSamples) * numChannels);
    }
    const bool useStreamReduce = canStream && streamGridX > 0;

    dim3          block(BLOCK_X, BLOCK_Y, 1);
    ReductionGrid reduceGrid = GetReductionGrid(size.x, size.y, numSamples, numChannels);
    dim3      applyGrid(ComputeGridX(size.x, APPLY_X_STEPS), cvcuda::priv::detail::AutoContrastDivUp(size.y, BLOCK_Y),
                        numSamples);
    const int nThreads               = block.x * block.y;
    const int numWarps               = nThreads / WARP_WIDTH;
    const size_t          reduceChan = IsPlanar ? 1 : cuda::NumElements<T>;
    const size_t          smem       = 2 * reduceChan * numWarps * sizeof(float);
    const size_t          finalSmem  = 2 * (FINAL_THREADS / WARP_WIDTH) * sizeof(float);
    const dim3            wsGrid     = useStreamReduce ? dim3(static_cast<unsigned int>(streamGridX),
                                                              static_cast<unsigned int>(shape.numSegments), 1)
                                                       : reduceGrid.launch;
    auto                  reduction  = GetReductionWorkspace(ws, stream, numSamples, numChannels, wsGrid);
    WorkspaceReleaseGuard releaseGuard(ws, stream);

    if constexpr (!IsPlanar)
    {
        auto src = cuda::CreateTensorWrapNHW<const T, StrideType>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(dstData);
        if (useStreamReduce)
        {
            LaunchReduceStream<false, BT, C>(stream, reinterpret_cast<const BT *>(srcData.basePtr()),
                                             reduction.partialLo, reduction.partialHi, shape, numSamples, numChannels,
                                             numPlanes, streamGridX, static_cast<int>(reduction.numPartials));
        }
        else
        {
            if (reduceGrid.isExact())
            {
                ReduceTensor<false, T><<<reduceGrid.launch, block, smem, stream>>>(
                    src, reduction.partialLo, reduction.partialHi, size, numChannels, 1);
            }
            else
            {
                ReduceTensor<false, T><<<reduceGrid.launch, block, smem, stream>>>(
                    src, reduction.partialLo, reduction.partialHi, size, numChannels, 1, reduceGrid.logicalX,
                    reduceGrid.logicalY);
            }
        }
        FinalizeMinMax<<<static_cast<unsigned int>(reduction.numExtrema), FINAL_THREADS, finalSmem, stream>>>(
            reduction.partialLo, reduction.partialHi, reduction.lo, reduction.hi, reduction.numPartials);
        if (!canStream
            || !LaunchApplyStream<false, BT, C>(stream, reinterpret_cast<const BT *>(srcData.basePtr()),
                                                reinterpret_cast<BT *>(dstData.basePtr()), reduction.lo, reduction.hi,
                                                shape, numSamples, numChannels, numPlanes, bound))
        {
            ApplyTensor<false, T>
                <<<applyGrid, block, 0, stream>>>(src, dst, reduction.lo, reduction.hi, size, numChannels, 1, bound);
        }
    }
    else
    {
        auto src = cuda::Tensor4DWrap<const T, StrideType>(
            srcData.basePtr(), static_cast<StrideType>(srcAccess->sampleStride()),
            static_cast<StrideType>(srcAccess->planeStride()), static_cast<StrideType>(srcAccess->rowStride()));
        auto dst = cuda::Tensor4DWrap<T, StrideType>(
            dstData.basePtr(), static_cast<StrideType>(dstAccess->sampleStride()),
            static_cast<StrideType>(dstAccess->planeStride()), static_cast<StrideType>(dstAccess->rowStride()));
        if (useStreamReduce)
        {
            LaunchReduceStream<true, BT, 1>(stream, reinterpret_cast<const BT *>(srcData.basePtr()),
                                            reduction.partialLo, reduction.partialHi, shape, numSamples, numChannels,
                                            numPlanes, streamGridX, static_cast<int>(reduction.numPartials));
        }
        else if (reduceGrid.isExact())
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
        if (!canStream
            || !LaunchApplyStream<true, BT, 1>(stream, reinterpret_cast<const BT *>(srcData.basePtr()),
                                               reinterpret_cast<BT *>(dstData.basePtr()), reduction.lo, reduction.hi,
                                               shape, numSamples, numChannels, numPlanes, bound))
        {
            ApplyTensor<true, T><<<applyGrid, block, 0, stream>>>(src, dst, reduction.lo, reduction.hi, size,
                                                                  numChannels, numPlanes, bound);
        }
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
    const float bound = cvcuda::priv::PhotometricUpperBound<BT, float>();

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

// Pick the supported base type (u8 / u16 / f16 / f32 only) from a tensor/format dtype.
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
    else if CVCUDA_AC_BASE(S16, short)
    else if CVCUDA_AC_BASE(F16, __half)
    else if CVCUDA_AC_BASE(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input/output data type (only U8, U16, S16, F16, F32 are supported)");
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
    const auto info = cvcuda::priv::same_shape::ValidateSrcDstTensors(srcData, dstData);

    // The shared validator compares 32-bit extents; the full-shape compare also rejects a pair
    // that differs only above that range.
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
    // Each sample is mapped to grid.z in the reduce/apply kernels.
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

    // Stricter than the shared column check: the kernels also assume packed channels, and a
    // one-column image must still be packed.
    const int64_t elementStride     = info.srcDtype.strideBytes();
    const int64_t expectedColStride = elementStride * info.numInterleavedChannels;
    if (srcAccess->colStride() != expectedColStride || dstAccess->colStride() != expectedColStride
        || (srcAccess->infoLayout().isChannelLast()
            && (srcAccess->chStride() != elementStride || dstAccess->chStride() != elementStride)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor pixels must have packed width and channel strides");
    }

    numSamples             = info.numSamples;
    numPlanes              = info.numPlanes;
    dtype                  = info.srcDtype;
    numInterleavedChannels = info.numInterleavedChannels;
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
