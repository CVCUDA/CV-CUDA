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
#include "OpAdjustContrast.hpp"

#include "AdjustColorCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/ColorSpec.hpp>
#include <nvcv/DataLayout.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

namespace cuda   = nvcv::cuda;
namespace util   = nvcv::util;
namespace adjust = cvcuda::priv::adjust;

namespace {

inline void ValidateFactor(double factor)
{
    if (factor < 0.0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Contrast factor must be non-negative");
    }
}

constexpr int     kBlockX                = 32;
constexpr int     kBlockY                = 4;
constexpr int     kApplyXSteps           = 4;
constexpr int     kMeanBlock             = 256; // fixed legacy reduction width; must remain a power of two
constexpr int     kF32MeanThreads        = 64;
constexpr int64_t kF32MinParallelPixels  = 8192;
constexpr int     kMeanValuesPerThread   = 32;
constexpr size_t  kMaxMeanWorkspaceBytes = 8 * 1024 * 1024;
static_assert(kMeanBlock % kF32MeanThreads == 0);
static_assert(sizeof(float) == sizeof(uint32_t));

// Deterministic shared-memory tree reduction of a per-thread partial into means[blockIdx.z] = sum /
// count. One block per image (grid = {1, 1, numImages}); the fixed reduction order keeps results
// run-to-run deterministic (no atomics).
template<int BLOCK>
inline __device__ void WriteMean(float local, int64_t count, float *means)
{
    __shared__ float smem[BLOCK];
    const int        t = threadIdx.x;
    smem[t]            = local;
    __syncthreads();
#pragma unroll
    for (int s = BLOCK / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            smem[t] += smem[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        means[blockIdx.z] = smem[0] / static_cast<float>(count);
    }
}

// Accumulate the grayscale value of one pixel position (x, y) of image z into `local`, dispatching
// on interleaved vs planar layout. Planar reads the channels from separate planes (numPlanes is 1
// or 3); interleaved reads the vector pixel directly.
template<bool IsPlanar, typename BT, class SrcWrapper>
inline __device__ float PixelGray(const SrcWrapper &src, int x, int y, int z, int numPlanes, int3 rgbIndices)
{
    if constexpr (!IsPlanar)
    {
        const auto pixel = src[int3{x, y, z}];
        if constexpr (cuda::NumElements<std::remove_const_t<typename SrcWrapper::ValueType>> == 1)
        {
            return static_cast<float>(cuda::GetElement(pixel, 0));
        }
        else
        {
            return adjust::GrayFromRGB<BT>(cuda::GetElement(pixel, rgbIndices.x), cuda::GetElement(pixel, rgbIndices.y),
                                           cuda::GetElement(pixel, rgbIndices.z));
        }
    }
    else
    {
        // Planar wrappers carry a 1-element vector element type (uchar1 / float1); extract the base
        // component before combining channels from separate planes.
        if (numPlanes == 1)
        {
            return static_cast<float>(cuda::GetElement(src[int4{x, y, 0, z}], 0));
        }
        return adjust::GrayFromRGB<BT>(cuda::GetElement(src[int4{x, y, rgbIndices.x, z}], 0),
                                       cuda::GetElement(src[int4{x, y, rgbIndices.y, z}], 0),
                                       cuda::GetElement(src[int4{x, y, rgbIndices.z, z}], 0));
    }
}

// Grayscale-mean reduction, tensor variant (uniform per-image size).
template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanTensor(SrcWrapper src, int2 size, int numPlanes, int3 rgbIndices, float *means)
{
    using BT        = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    const int     z = blockIdx.z;
    const int64_t n = static_cast<int64_t>(size.x) * size.y;

    int64_t   idx   = threadIdx.x;
    int       x     = static_cast<int>(idx % size.x);
    int       y     = static_cast<int>(idx / size.x);
    const int xStep = blockDim.x % size.x;
    const int yStep = blockDim.x / size.x;

    float local = 0.0f;
    for (; idx < n; idx += blockDim.x)
    {
        local += PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices);

        x += xStep;
        y += yStep;
        if (x >= size.x)
        {
            x -= size.x;
            ++y;
        }
    }
    WriteMean<BLOCK>(local, n, means);
}

// Grayscale-mean reduction, var-shape variant (per-image size from the wrapper).
template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanVarShape(SrcWrapper src, int numPlanes, int3 rgbIndices, float *means)
{
    using BT        = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    const int     z = blockIdx.z;
    const int     w = src.width(z);
    const int     h = src.height(z);
    const int64_t n = static_cast<int64_t>(w) * h;

    int64_t   idx   = threadIdx.x;
    int       x     = static_cast<int>(idx % w);
    int       y     = static_cast<int>(idx / w);
    const int xStep = blockDim.x % w;
    const int yStep = blockDim.x / w;

    float local = 0.0f;
    for (; idx < n; idx += blockDim.x)
    {
        local += PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices);

        x += xStep;
        y += yStep;
        if (x >= w)
        {
            x -= w;
            ++y;
        }
    }
    WriteMean<BLOCK>(local, n, means);
}

// Split the fixed F32 reduction lanes across two-warp CTAs without changing their arithmetic.
// Lane t still visits t, t + 256, ... in order; the final kernel below replays the original tree.
template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanF32Tensor(SrcWrapper src, int2 size, int numPlanes, int3 rgbIndices, float *laneSums)
{
    using BT = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    static_assert(std::is_same_v<BT, float>);

    const int     z    = blockIdx.z;
    const int     lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n    = static_cast<int64_t>(size.x) * size.y;

    int64_t   idx   = lane;
    int       x     = static_cast<int>(idx % size.x);
    int       y     = static_cast<int>(idx / size.x);
    const int xStep = BLOCK % size.x;
    const int yStep = BLOCK / size.x;

    float local = 0.0f;
    for (; idx < n; idx += BLOCK)
    {
        local += PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices);

        x += xStep;
        y += yStep;
        if (x >= size.x)
        {
            x -= size.x;
            ++y;
        }
    }
    laneSums[static_cast<size_t>(z) * BLOCK + lane] = local;
}

template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanF32VarShape(SrcWrapper src, int numPlanes, int3 rgbIndices, float *laneSums)
{
    using BT = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    static_assert(std::is_same_v<BT, float>);

    const int     z    = blockIdx.z;
    const int     lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int     w    = src.width(z);
    const int     h    = src.height(z);
    const int64_t n    = static_cast<int64_t>(w) * h;

#if __CUDA_ARCH__ == 900
    // Direct planar addressing keeps independent RGB loads in flight on sm_90; its wrapper
    // subscript codegen serializes each triplet. Other architectures retain their faster wrapper path.
    NVCVImagePlaneStrided redPlane{};
    NVCVImagePlaneStrided greenPlane{};
    NVCVImagePlaneStrided bluePlane{};
    if constexpr (IsPlanar)
    {
        redPlane   = src.plane(z, rgbIndices.x);
        greenPlane = src.plane(z, rgbIndices.y);
        bluePlane  = src.plane(z, rgbIndices.z);
    }
#endif

    int64_t   idx   = lane;
    int       x     = static_cast<int>(idx % w);
    int       y     = static_cast<int>(idx / w);
    const int xStep = BLOCK % w;
    const int yStep = BLOCK / w;

    float local = 0.0f;
    for (; idx < n; idx += BLOCK)
    {
#if __CUDA_ARCH__ == 900
        if constexpr (IsPlanar)
        {
            const int redOffset   = y * redPlane.rowStride + x * sizeof(BT);
            const int greenOffset = y * greenPlane.rowStride + x * sizeof(BT);
            const int blueOffset  = y * bluePlane.rowStride + x * sizeof(BT);
            const BT  red         = *reinterpret_cast<const BT *>(redPlane.basePtr + redOffset);
            const BT  green       = *reinterpret_cast<const BT *>(greenPlane.basePtr + greenOffset);
            const BT  blue        = *reinterpret_cast<const BT *>(bluePlane.basePtr + blueOffset);
            local += adjust::GrayFromRGB<BT>(red, green, blue);
        }
        else
        {
            local += PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices);
        }
#else
        local += PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices);
#endif

        x += xStep;
        y += yStep;
        if (x >= w)
        {
            x -= w;
            ++y;
        }
    }
    laneSums[static_cast<size_t>(z) * BLOCK + lane] = local;
}

template<int BLOCK>
__global__ void FinalizeMeanF32Tensor(const float *laneSums, int64_t count, float *means)
{
    const int   z     = blockIdx.z;
    const float local = laneSums[static_cast<size_t>(z) * BLOCK + threadIdx.x];
    WriteMean<BLOCK>(local, count, means);
}

template<int BLOCK, class SrcWrapper>
__global__ void FinalizeMeanF32VarShape(SrcWrapper src, const float *laneSums, float *means)
{
    const int     z     = blockIdx.z;
    const float   local = laneSums[static_cast<size_t>(z) * BLOCK + threadIdx.x];
    const int64_t count = static_cast<int64_t>(src.width(z)) * src.height(z);
    WriteMean<BLOCK>(local, count, means);
}

// Parallel U8 first pass. A thread always owns the same lane as in ContrastMean*, and each logical
// chunk begins on a multiple of BLOCK. Consequently, the final pass can reconstruct the exact 256
// per-lane values used by the original reduction tree.
template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanU8Tensor(SrcWrapper src, int2 size, int numPlanes, int3 rgbIndices, uint32_t *partials)
{
    using BT = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    static_assert(std::is_same_v<BT, unsigned char>);

    constexpr int64_t chunkPixels = static_cast<int64_t>(BLOCK) * kMeanValuesPerThread;
    const int         z           = blockIdx.z;
    const int64_t     n           = static_cast<int64_t>(size.x) * size.y;
    const int         xStep       = BLOCK % size.x;
    const int         yStep       = BLOCK / size.x;
    uint32_t          local       = 0;

    for (int64_t base = static_cast<int64_t>(blockIdx.x) * chunkPixels + threadIdx.x; base < n;
         base += static_cast<int64_t>(gridDim.x) * chunkPixels)
    {
        int64_t idx = base;
        int     x   = static_cast<int>(idx % size.x);
        int     y   = static_cast<int>(idx / size.x);
#pragma unroll
        for (int i = 0; i < kMeanValuesPerThread; ++i)
        {
            if (idx < n)
            {
                local += static_cast<uint32_t>(PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices));
            }
            idx += BLOCK;
            x += xStep;
            y += yStep;
            if (x >= size.x)
            {
                x -= size.x;
                ++y;
            }
        }
    }

    const size_t partial = (static_cast<size_t>(z) * gridDim.x + blockIdx.x) * BLOCK + threadIdx.x;
    partials[partial]    = local;
}

template<int BLOCK, bool IsPlanar, class SrcWrapper>
__global__ void ContrastMeanU8VarShape(SrcWrapper src, int numPlanes, int3 rgbIndices, uint32_t *partials)
{
    using BT = cuda::BaseType<std::remove_const_t<typename SrcWrapper::ValueType>>;
    static_assert(std::is_same_v<BT, unsigned char>);

    constexpr int64_t chunkPixels = static_cast<int64_t>(BLOCK) * kMeanValuesPerThread;
    const int         z           = blockIdx.z;
    const int         w           = src.width(z);
    const int         h           = src.height(z);
    const int64_t     n           = static_cast<int64_t>(w) * h;
    const int         xStep       = BLOCK % w;
    const int         yStep       = BLOCK / w;
    uint32_t          local       = 0;

    for (int64_t base = static_cast<int64_t>(blockIdx.x) * chunkPixels + threadIdx.x; base < n;
         base += static_cast<int64_t>(gridDim.x) * chunkPixels)
    {
        int64_t idx = base;
        int     x   = static_cast<int>(idx % w);
        int     y   = static_cast<int>(idx / w);
#pragma unroll
        for (int i = 0; i < kMeanValuesPerThread; ++i)
        {
            if (idx < n)
            {
                local += static_cast<uint32_t>(PixelGray<IsPlanar, BT>(src, x, y, z, numPlanes, rgbIndices));
            }
            idx += BLOCK;
            x += xStep;
            y += yStep;
            if (x >= w)
            {
                x -= w;
                ++y;
            }
        }
    }

    const size_t partial = (static_cast<size_t>(z) * gridDim.x + blockIdx.x) * BLOCK + threadIdx.x;
    partials[partial]    = local;
}

template<int BLOCK>
__global__ void FinalizeMeanU8Tensor(const uint32_t *partials, int numPartials, int64_t count, float *means)
{
    const int z = blockIdx.z;
    uint32_t  local{};
    for (int p = 0; p < numPartials; ++p)
    {
        const size_t offset = (static_cast<size_t>(z) * numPartials + p) * BLOCK + threadIdx.x;
        local += partials[offset];
    }
    WriteMean<BLOCK>(static_cast<float>(local), count, means);
}

template<int BLOCK, class SrcWrapper>
__global__ void FinalizeMeanU8VarShape(SrcWrapper src, const uint32_t *partials, int numPartials, float *means)
{
    const int z = blockIdx.z;
    uint32_t  local{};
    for (int p = 0; p < numPartials; ++p)
    {
        const size_t offset = (static_cast<size_t>(z) * numPartials + p) * BLOCK + threadIdx.x;
        local += partials[offset];
    }
    const int64_t count = static_cast<int64_t>(src.width(z)) * src.height(z);
    WriteMean<BLOCK>(static_cast<float>(local), count, means);
}

// out = clamp(factor * in + (1 - factor) * mean, 0, bound) per channel component.
template<typename T>
inline __device__ T AdjustContrastPixel(T pixel, float factor, float mean)
{
    using BT                         = cuda::BaseType<T>;
    static constexpr int numChannels = cuda::NumElements<T>;
    const float          bound       = adjust::Bound<BT>();

    T out{};
#pragma unroll
    for (int c = 0; c < numChannels; ++c)
    {
        cuda::GetElement(out, c) = adjust::Blend<BT>(cuda::GetElement(pixel, c), factor, mean, bound);
    }
    return out;
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper>
__global__ void ApplyTensor(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, float factor, const float *means)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    const int x0 = blockIdx.x * blockDim.x * kApplyXSteps + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int z  = blockIdx.z;
    if (y >= size.y)
    {
        return;
    }

    const float mean = means[z];
    if constexpr (!IsPlanar)
    {
#pragma unroll
        for (int i = 0; i < kApplyXSteps; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < size.x)
            {
                const int3 coord{x, y, z};
                dst[coord] = AdjustContrastPixel<DstT>(src[coord], factor, mean);
            }
        }
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
#pragma unroll
            for (int i = 0; i < kApplyXSteps; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < size.x)
                {
                    const int4 coord{x, y, p, z};
                    dst[coord] = AdjustContrastPixel<DstT>(src[coord], factor, mean);
                }
            }
        }
    }
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper>
__global__ void ApplyVarShape(SrcWrapper src, DstWrapper dst, int numPlanes, float factor, const float *means)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    const int z      = blockIdx.z;
    const int width  = dst.width(z);
    const int height = dst.height(z);
    const int x0     = blockIdx.x * blockDim.x * kApplyXSteps + threadIdx.x;
    const int y      = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
    {
        return;
    }

    const float mean = means[z];
    if constexpr (!IsPlanar)
    {
#pragma unroll
        for (int i = 0; i < kApplyXSteps; ++i)
        {
            const int x = x0 + i * blockDim.x;
            if (x < width)
            {
                const int3 coord{x, y, z};
                dst[coord] = AdjustContrastPixel<DstT>(src[coord], factor, mean);
            }
        }
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
#pragma unroll
            for (int i = 0; i < kApplyXSteps; ++i)
            {
                const int x = x0 + i * blockDim.x;
                if (x < width)
                {
                    const int4 coord{x, y, p, z};
                    dst[coord] = AdjustContrastPixel<DstT>(src[coord], factor, mean);
                }
            }
        }
    }
}

struct ReductionGrid
{
    dim3 launch;
    bool parallel;
};

struct ReductionWorkspace
{
    uint32_t *partials;
    float    *means;
    size_t    numPartials;
};

inline size_t CheckedWorkspaceMul(size_t a, size_t b)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AdjustContrast reduction workspace size overflows size_t");
    }
    return a * b;
}

inline size_t CheckedWorkspaceAdd(size_t a, size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AdjustContrast reduction workspace size overflows size_t");
    }
    return a + b;
}

inline ReductionGrid GetReductionGrid(int64_t pixels, int numSamples, bool isU8)
{
    const dim3        fallback{1, 1, static_cast<unsigned int>(numSamples)};
    constexpr int64_t chunkPixels = static_cast<int64_t>(kMeanBlock) * kMeanValuesPerThread;

    const size_t meansBytes = CheckedWorkspaceMul(static_cast<size_t>(numSamples), sizeof(float));

    if (!isU8)
    {
        const size_t partialBytes = CheckedWorkspaceMul(
            CheckedWorkspaceMul(static_cast<size_t>(numSamples), static_cast<size_t>(kMeanBlock)), sizeof(float));
        if (pixels <= kF32MinParallelPixels || CheckedWorkspaceAdd(partialBytes, meansBytes) > kMaxMeanWorkspaceBytes)
        {
            return {fallback, false};
        }
        return {{{kMeanBlock / kF32MeanThreads, 1, static_cast<unsigned int>(numSamples)}}, true};
    }

    // The final pass casts each complete legacy lane sum to float once. Keep that cast exact so the
    // unchanged 256-lane float tree produces the same bits as the original one-CTA reduction.
    const int64_t maxTermsPerLane = (static_cast<int64_t>(1) << 24) / 255;
    if (pixels <= chunkPixels || util::DivUp(pixels, static_cast<int64_t>(kMeanBlock)) > maxTermsPerLane)
    {
        return {fallback, false};
    }

    const size_t onePartialBytes = CheckedWorkspaceMul(
        CheckedWorkspaceMul(static_cast<size_t>(numSamples), static_cast<size_t>(kMeanBlock)), sizeof(uint32_t));
    if (meansBytes >= kMaxMeanWorkspaceBytes)
    {
        return {fallback, false};
    }

    const size_t maxPartials     = (kMaxMeanWorkspaceBytes - meansBytes) / onePartialBytes;
    const size_t logicalPartials = static_cast<size_t>(util::DivUp(pixels, static_cast<int64_t>(chunkPixels)));
    const size_t numPartials     = std::min(logicalPartials, maxPartials);
    // One partial exposes no more CTAs than the fallback and adds a finalization launch.
    if (numPartials <= 1)
    {
        return {fallback, false};
    }

    return {{{static_cast<unsigned int>(numPartials), 1, static_cast<unsigned int>(numSamples)}}, true};
}

inline dim3 GetApplyGrid(int width, int height, int numSamples)
{
    const int64_t gridX = util::DivUp(static_cast<int64_t>(width), static_cast<int64_t>(kBlockX) * kApplyXSteps);
    const int64_t gridY = util::DivUp(static_cast<int64_t>(height), static_cast<int64_t>(kBlockY));
    if (gridX > cuda::TypeTraits<int32_t>::max || gridY > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "AdjustContrast apply grid exceeds the CUDA launch limits");
    }
    return {static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY), static_cast<unsigned int>(numSamples)};
}

template<class Workspace>
inline ReductionWorkspace GetReductionWorkspace(Workspace &ws, cudaStream_t stream, int numSamples, ReductionGrid grid,
                                                bool isU8)
{
    const size_t numPartials = grid.parallel ? grid.launch.x : 0;
    const size_t partialsPerSample
        = grid.parallel ? (isU8 ? CheckedWorkspaceMul(numPartials, kMeanBlock) : kMeanBlock) : 0;
    const size_t partialCount = CheckedWorkspaceMul(static_cast<size_t>(numSamples), partialsPerSample);
    const size_t partialBytes = CheckedWorkspaceMul(partialCount, sizeof(uint32_t));
    const size_t meansBytes   = CheckedWorkspaceMul(static_cast<size_t>(numSamples), sizeof(float));
    auto        *buf = static_cast<unsigned char *>(ws.acquire(CheckedWorkspaceAdd(partialBytes, meansBytes), stream));

    return {reinterpret_cast<uint32_t *>(buf), reinterpret_cast<float *>(buf + partialBytes), numPartials};
}

template<class Workspace>
class WorkspaceReleaseGuard
{
public:
    WorkspaceReleaseGuard(Workspace &workspace, cudaStream_t stream)
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
    Workspace   &m_workspace;
    cudaStream_t m_stream;
    bool         m_active = true;
};

// cudaMallocAsync/freeAsync become graph allocation nodes during stream capture. Keep that existing
// behavior for captured submissions; ordinary submissions use the retained AutoContrast-style
// workspace below.
class CapturedWorkspace
{
public:
    void *acquire(size_t sizeBytes, cudaStream_t stream)
    {
        NVCV_CHECK_THROW(cudaMallocAsync(&m_data, sizeBytes, stream));
        return m_data;
    }

    void release(cudaStream_t stream)
    {
        NVCV_CHECK_THROW(cudaFreeAsync(m_data, stream));
        m_data = nullptr;
    }

    void releaseNoThrow(cudaStream_t stream) noexcept
    {
        if (m_data != nullptr)
        {
            NVCV_CHECK_LOG(cudaFreeAsync(m_data, stream));
            m_data = nullptr;
        }
    }

private:
    void *m_data = nullptr;
};

template<bool IsPlanar, typename ValueT, class Workspace>
inline void RunTensor(cudaStream_t stream, Workspace &ws, const nvcv::TensorDataStridedCuda &srcData,
                      const nvcv::TensorDataStridedCuda &dstData, float factor, int numPlanes, int3 rgbIndices)
{
    auto      srcAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto      dstAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    int2      size       = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
    const int numSamples = srcAccess->numSamples();

    if (numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }

    int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
    int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
    using StrideType = int32_t;

    using BT                                    = cuda::BaseType<ValueT>;
    constexpr bool                   isU8       = std::is_same_v<BT, unsigned char>;
    const int64_t                    pixels     = static_cast<int64_t>(size.x) * size.y;
    auto                             reduceGrid = GetReductionGrid(pixels, numSamples, isU8);
    auto                             reduction  = GetReductionWorkspace(ws, stream, numSamples, reduceGrid, isU8);
    WorkspaceReleaseGuard<Workspace> releaseGuard(ws, stream);

    dim3 block(kBlockX, kBlockY, 1);
    dim3 meanGrid(1, 1, numSamples);
    dim3 applyGrid = GetApplyGrid(size.x, size.y, numSamples);

    if constexpr (!IsPlanar)
    {
        auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
        auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
        if constexpr (isU8)
        {
            if (reduceGrid.parallel)
            {
                ContrastMeanU8Tensor<kMeanBlock, IsPlanar>
                    <<<reduceGrid.launch, kMeanBlock, 0, stream>>>(src, size, 1, rgbIndices, reduction.partials);
                FinalizeMeanU8Tensor<kMeanBlock><<<meanGrid, kMeanBlock, 0, stream>>>(
                    reduction.partials, reduction.numPartials, pixels, reduction.means);
            }
            else
            {
                ContrastMeanTensor<kMeanBlock, IsPlanar>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(src, size, 1, rgbIndices, reduction.means);
            }
        }
        else
        {
            if (reduceGrid.parallel)
            {
                auto *laneSums = reinterpret_cast<float *>(reduction.partials);
                ContrastMeanF32Tensor<kMeanBlock, IsPlanar>
                    <<<reduceGrid.launch, kF32MeanThreads, 0, stream>>>(src, size, 1, rgbIndices, laneSums);
                FinalizeMeanF32Tensor<kMeanBlock>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(laneSums, pixels, reduction.means);
            }
            else
            {
                ContrastMeanTensor<kMeanBlock, IsPlanar>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(src, size, 1, rgbIndices, reduction.means);
            }
        }
        ApplyTensor<IsPlanar><<<applyGrid, block, 0, stream>>>(src, dst, size, 1, factor, reduction.means);
    }
    else
    {
        auto src = cuda::Tensor4DWrap<const ValueT, StrideType>(
            srcData.basePtr(), static_cast<int>(srcAccess->sampleStride()), static_cast<int>(srcAccess->planeStride()),
            static_cast<int>(srcAccess->rowStride()));
        auto dst = cuda::Tensor4DWrap<ValueT, StrideType>(
            dstData.basePtr(), static_cast<int>(dstAccess->sampleStride()), static_cast<int>(dstAccess->planeStride()),
            static_cast<int>(dstAccess->rowStride()));
        if constexpr (isU8)
        {
            if (reduceGrid.parallel)
            {
                ContrastMeanU8Tensor<kMeanBlock, IsPlanar><<<reduceGrid.launch, kMeanBlock, 0, stream>>>(
                    src, size, numPlanes, rgbIndices, reduction.partials);
                FinalizeMeanU8Tensor<kMeanBlock><<<meanGrid, kMeanBlock, 0, stream>>>(
                    reduction.partials, reduction.numPartials, pixels, reduction.means);
            }
            else
            {
                ContrastMeanTensor<kMeanBlock, IsPlanar>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(src, size, numPlanes, rgbIndices, reduction.means);
            }
        }
        else
        {
            if (reduceGrid.parallel)
            {
                auto *laneSums = reinterpret_cast<float *>(reduction.partials);
                ContrastMeanF32Tensor<kMeanBlock, IsPlanar>
                    <<<reduceGrid.launch, kF32MeanThreads, 0, stream>>>(src, size, numPlanes, rgbIndices, laneSums);
                FinalizeMeanF32Tensor<kMeanBlock>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(laneSums, pixels, reduction.means);
            }
            else
            {
                ContrastMeanTensor<kMeanBlock, IsPlanar>
                    <<<meanGrid, kMeanBlock, 0, stream>>>(src, size, numPlanes, rgbIndices, reduction.means);
            }
        }
        ApplyTensor<IsPlanar><<<applyGrid, block, 0, stream>>>(src, dst, size, numPlanes, factor, reduction.means);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    releaseGuard.finish();
}

template<bool IsPlanar, typename ValueT, class Workspace>
inline void RunVarShapeBatch(cudaStream_t stream, Workspace &ws, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, float factor, int numPlanes,
                             int3 rgbIndices, int64_t maxPixels)
{
    const int numImages = dstData.numImages();
    if (numImages > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }

    using BT                              = cuda::BaseType<ValueT>;
    constexpr bool                   isU8 = std::is_same_v<BT, unsigned char>;
    int3                             maxSize{dstData.maxSize().w, dstData.maxSize().h, numImages};
    auto                             reduceGrid = GetReductionGrid(maxPixels, numImages, isU8);
    auto                             reduction  = GetReductionWorkspace(ws, stream, numImages, reduceGrid, isU8);
    WorkspaceReleaseGuard<Workspace> releaseGuard(ws, stream);

    dim3 block(kBlockX, kBlockY, 1);
    dim3 meanGrid(1, 1, numImages);
    dim3 applyGrid = GetApplyGrid(maxSize.x, maxSize.y, numImages);

    cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
    cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);

    if constexpr (isU8)
    {
        if (reduceGrid.parallel)
        {
            ContrastMeanU8VarShape<kMeanBlock, IsPlanar>
                <<<reduceGrid.launch, kMeanBlock, 0, stream>>>(src, numPlanes, rgbIndices, reduction.partials);
            FinalizeMeanU8VarShape<kMeanBlock>
                <<<meanGrid, kMeanBlock, 0, stream>>>(src, reduction.partials, reduction.numPartials, reduction.means);
        }
        else
        {
            ContrastMeanVarShape<kMeanBlock, IsPlanar>
                <<<meanGrid, kMeanBlock, 0, stream>>>(src, numPlanes, rgbIndices, reduction.means);
        }
    }
    else
    {
        if (reduceGrid.parallel)
        {
            auto *laneSums = reinterpret_cast<float *>(reduction.partials);
            ContrastMeanF32VarShape<kMeanBlock, IsPlanar>
                <<<reduceGrid.launch, kF32MeanThreads, 0, stream>>>(src, numPlanes, rgbIndices, laneSums);
            FinalizeMeanF32VarShape<kMeanBlock><<<meanGrid, kMeanBlock, 0, stream>>>(src, laneSums, reduction.means);
        }
        else
        {
            ContrastMeanVarShape<kMeanBlock, IsPlanar>
                <<<meanGrid, kMeanBlock, 0, stream>>>(src, numPlanes, rgbIndices, reduction.means);
        }
    }
    ApplyVarShape<IsPlanar><<<applyGrid, block, 0, stream>>>(src, dst, numPlanes, factor, reduction.means);
    NVCV_CHECK_THROW(cudaGetLastError());
    releaseGuard.finish();
}

// Dispatch over base data type (u8 / f32) and channel count (1 / 3) ----------------------

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar = unsigned char;

#define NVCV_ADJUST_CONTRAST_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE) \
    ((dType == nvcv::TYPE_3##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_ADJUST_CONTRAST_RUN_TYPED(U8, uchar)
    else if NVCV_ADJUST_CONTRAST_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: AdjustContrast supports 8-bit unsigned and 32-bit float");
    }
        // clang-format on

#undef NVCV_ADJUST_CONTRAST_RUN_TYPED
}

template<typename Cb>
inline void RunChannelSwitch(int numChannels, int numPlanes, nvcv::DataType dType, const Cb &cb)
{
    RunTypeSwitch(dType,
                  [&numChannels, &numPlanes, &cb](auto dummyVal)
                  {
                      using ValBase = decltype(dummyVal);
                      // clang-format off
            if (numChannels == 1)
            {
                using Val = cuda::MakeType<ValBase, 1>;
                if (numPlanes == 1)
                {
                    cb(Val{}, std::integral_constant<bool, false>{});
                }
                else if (numPlanes == 3)
                {
                    cb(Val{}, std::integral_constant<bool, true>{});
                }
                else
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "Invalid number of channels: AdjustContrast supports 1 or 3 channels");
                }
            }
            else if (numChannels == 3)
            {
                cb(cuda::MakeType<ValBase, 3>{}, std::integral_constant<bool, false>{});
            }
            else
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid number of channels: AdjustContrast supports 1 or 3 channels");
            }
                      // clang-format on
                  });
}

// Validation ------------------------------------------------------------------------------

inline bool ValidateSrcDstTensors(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
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

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    if (srcAccess->numSamples() != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    int numChannels = srcAccess->numChannels();
    if (numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    numPlanes = srcAccess->numPlanes();
    if (numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: AdjustContrast supports 1 or 3 channels");
    }
    if (numPlanes != 1 && numPlanes != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of planes: AdjustContrast supports 1 or 3 planes");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
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
    return srcAccess->numSamples() == 0 || srcAccess->numCols() == 0 || srcAccess->numRows() == 0;
}

inline auto ValidateSrcDstVarBatch(int &numInterleavedChannels, int &numPlanes, int3 &rgbIndices, nvcv::DataType &dtype,
                                   int64_t &maxPixels, cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                   const nvcv::ImageBatchVarShape &dst)
{
    using maybeVarShape = nvcv::Optional<nvcv::ImageBatchVarShapeDataStridedCuda>;
    std::tuple<maybeVarShape, maybeVarShape> srcDstData{
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

    int numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
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

    const int numChannels = srcFormat.numChannels();
    if (numChannels != 1 && numChannels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: AdjustContrast supports 1 or 3 channels");
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

    dtype = srcFormat.planeDataType(0);
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
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dtype != srcFormat.planeDataType(i) || srcFormat.planeDataType(i).numChannels() != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All planes in the input image must have the same data type");
        }
    }

    numInterleavedChannels = dtype.numChannels();

    rgbIndices = int3{0, 1, 2};
    if (numChannels == 3)
    {
        if (srcFormat.colorModel() != nvcv::ColorModel::RGB && srcFormat.colorModel() != nvcv::ColorModel::UNDEFINED)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Three-channel images must use the RGB color model");
        }

        const auto swizzleChannels = nvcv::GetChannels(srcFormat.swizzle());
        auto       channelIndex    = [](nvcv::Channel channel)
        {
            const int index = static_cast<int>(channel) - static_cast<int>(nvcv::Channel::X);
            if (index < 0 || index >= 3)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Three-channel images must use an RGB channel permutation");
            }
            return index;
        };
        rgbIndices = int3{channelIndex(swizzleChannels[0]), channelIndex(swizzleChannels[1]),
                          channelIndex(swizzleChannels[2])};
        if (rgbIndices.x == rgbIndices.y || rgbIndices.x == rgbIndices.z || rgbIndices.y == rgbIndices.z)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Three-channel images must use an RGB channel permutation");
        }
    }

    for (int i = 0; i < numSamples; i++)
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
        maxPixels                     = std::max(maxPixels, static_cast<int64_t>(size.w) * size.h);
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

    return srcDstData;
}

inline bool IsStreamCapturing(cudaStream_t stream)
{
    cudaStreamCaptureStatus captureStatus;
    NVCV_CHECK_THROW(cudaStreamIsCapturing(stream, &captureStatus));
    return captureStatus != cudaStreamCaptureStatusNone;
}

template<class Callback>
inline void RunWithWorkspace(cudaStream_t stream, bool isCapturing,
                             cvcuda::priv::PerDeviceResource<cvcuda::priv::AdjustContrastWorkspace> &resources,
                             const Callback                                                         &cb)
{
    if (isCapturing)
    {
        CapturedWorkspace workspace;
        cb(workspace);
    }
    else
    {
        auto &workspace = resources.get();
        auto  lock      = workspace.lock();
        cb(workspace);
    }
}

} // anonymous namespace

namespace cvcuda::priv {

// --------------------------------- Workspace -------------------------------

AdjustContrastWorkspace::AdjustContrastWorkspace()
{
    NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_ready, cudaEventDisableTiming));
}

AdjustContrastWorkspace::~AdjustContrastWorkspace()
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

[[nodiscard]] std::unique_lock<std::mutex> AdjustContrastWorkspace::lock()
{
    return std::unique_lock<std::mutex>{m_mutex};
}

void *AdjustContrastWorkspace::acquire(size_t sizeBytes, cudaStream_t stream)
{
    if (m_pending)
    {
        if (sizeBytes > m_capacity)
        {
            NVCV_CHECK_THROW(cudaEventSynchronize(m_ready));
            m_pending = false;
        }
        else
        {
            NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, m_ready));
        }
    }
    if (sizeBytes > m_capacity)
    {
        if (m_data != nullptr)
        {
            NVCV_CHECK_THROW(cudaFree(m_data));
            m_data     = nullptr;
            m_capacity = 0;
        }
        if (cudaMalloc(&m_data, sizeBytes) != cudaSuccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY,
                                  "AdjustContrast: failed to allocate reduction workspace");
        }
        m_capacity = sizeBytes;
    }
    return m_data;
}

void AdjustContrastWorkspace::release(cudaStream_t stream)
{
    NVCV_CHECK_THROW(cudaEventRecord(m_ready, stream));
    m_pending = true;
}

void AdjustContrastWorkspace::releaseNoThrow(cudaStream_t stream) noexcept
{
    cudaError_t err = cudaEventRecord(m_ready, stream);
    if (err == cudaSuccess)
    {
        m_pending = true;
    }
    NVCV_CHECK_LOG(err);
}

// --------------------------------- Operator --------------------------------

AdjustContrast::AdjustContrast()
    : m_workspace([](int) { return std::make_unique<AdjustContrastWorkspace>(); })
{
}

// Tensor input variant
void AdjustContrast::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                double contrastFactor) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdjustContrast::operator()[Tensor]");
    ValidateFactor(contrastFactor);
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    const bool     isEmpty = ValidateSrcDstTensors(numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    if (isEmpty)
    {
        RunChannelSwitch(numInterleavedChannels, numPlanes, dtype, [](auto, auto) {});
        return;
    }

    const float factor = static_cast<float>(contrastFactor);
    const int3  rgbIndices{0, 1, 2};
    const bool  isCapturing = IsStreamCapturing(stream);

    RunChannelSwitch(
        numInterleavedChannels, numPlanes, dtype,
        [this, &stream, &srcData, &dstData, factor, numPlanes, rgbIndices, isCapturing](auto dummyVal, auto isPlanar)
        {
            using ValueT   = decltype(dummyVal);
            using IsPlanar = decltype(isPlanar);
            RunWithWorkspace(stream, isCapturing, m_workspace,
                             [&](auto &workspace) {
                                 RunTensor<IsPlanar::value, ValueT>(stream, workspace, *srcData, *dstData, factor,
                                                                    numPlanes, rgbIndices);
                             });
        });
}

// VarShape input variant
void AdjustContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                const nvcv::ImageBatchVarShape &dst, double contrastFactor) const
{
    CVCUDA_NVTX_RANGE("cvcuda::AdjustContrast::operator()[ImageBatchVarShape]");
    ValidateFactor(contrastFactor);
    int            numInterleavedChannels;
    int            numPlanes;
    int3           rgbIndices;
    nvcv::DataType dtype;
    int64_t        maxPixels{};
    auto           srcDstData
        = ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, rgbIndices, dtype, maxPixels, stream, src, dst);
    if (src.numImages() == 0)
    {
        return;
    }

    const float factor      = static_cast<float>(contrastFactor);
    const bool  isCapturing = IsStreamCapturing(stream);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [this, &stream, &srcDstData, factor, numPlanes, rgbIndices, maxPixels, isCapturing](auto dummyVal,
                                                                                                         auto isPlanar)
                     {
                         using ValueT   = decltype(dummyVal);
                         using IsPlanar = decltype(isPlanar);
                         RunWithWorkspace(stream, isCapturing, m_workspace,
                                          [&](auto &workspace)
                                          {
                                              RunVarShapeBatch<IsPlanar::value, ValueT>(
                                                  stream, workspace, *std::get<0>(srcDstData), *std::get<1>(srcDstData),
                                                  factor, numPlanes, rgbIndices, maxPixels);
                                          });
                     });
}

} // namespace cvcuda::priv
