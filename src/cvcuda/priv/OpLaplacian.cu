/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpLaplacian.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include "FilterCommon.cuh"

#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <tuple>
#include <type_traits>

namespace cvcuda::priv {

namespace {

namespace cuda   = nvcv::cuda;
namespace util   = nvcv::util;
namespace filter = cvcuda::priv::filter;

constexpr int kBlockWidth  = 16;
constexpr int kBlockHeight = 16;

// The planar tiled kernels stage a row of NIX-wide output per thread, so the block is one warp wide.
constexpr int kPlanarBlockWidth = 32;
constexpr int kPlanarNIX        = 4;

constexpr int2 kStencilSize{3, 3};
constexpr int2 kStencilAnchor{1, 1};

inline int CurrentDeviceSMOrZero()
{
    int sm = 0;
    return cvcuda::priv::GetCurrentDeviceSM(sm) == cudaSuccess ? sm : 0;
}

// Keep RGB tiling scoped to measured architectures. The sparse stencil improves sm_120,
// where the previous generic correlation tile had regressed.
template<typename T>
inline bool TiledFilterPaysHere(int sm)
{
    if constexpr (std::is_same_v<T, uchar3>)
    {
        return sm == 80 || sm == 90 || sm == 120;
    }
    else if constexpr (std::is_same_v<cuda::BaseType<T>, __half> && cuda::NumElements<T> == 3)
    {
        return sm == 120;
    }
    else
    {
        return true;
    }
}

// The two 3x3 Laplacian stencils the operator exposes: ksize 1 is the 5-point form, ksize 3 the
// corner-weighted 9-point form. Spelled as a function so host dispatch and device code read the same
// definition instead of keeping a separate __device__ copy in sync.
// clang-format off
inline __host__ __device__ float LaplacianStencil(bool fivePoint, int idx)
{
    constexpr float kFivePoint[9] = {0.0f,  1.0f, 0.0f,
                                     1.0f, -4.0f, 1.0f,
                                     0.0f,  1.0f, 0.0f};
    constexpr float kNinePoint[9] = {2.0f,  0.0f, 2.0f,
                                     0.0f, -8.0f, 0.0f,
                                     2.0f,  0.0f, 2.0f};
    return fivePoint ? kFivePoint[idx] : kNinePoint[idx];
}

// clang-format on

// Scaling the whole stencil once on the host is the same product the taps would form individually;
// a scale of exactly 1 leaves every coefficient bit-identical.
inline cuda::math::Vector<float, 9> MakeScaledStencil(int ksize, float scale)
{
    cuda::math::Vector<float, 9> stencil;
    for (int i = 0; i < 9; ++i)
    {
        stencil[i] = LaplacianStencil(ksize == 1, i) * scale;
    }
    return stencil;
}

// ---------------------------------------------------------------------------------------------
// Parameter sources
// ---------------------------------------------------------------------------------------------

// A tensor submit carries one (ksize, scale) pair for the whole batch, so ksize can be bound at
// compile time and the stencil choice folds away. A var-shape submit carries one pair per image.
template<int KSize>
struct FixedLaplacianParams
{
    float scale;

    inline __device__ bool FivePoint(int) const
    {
        return KSize == 1;
    }

    inline __device__ float Scale(int) const
    {
        return scale;
    }
};

// A var-shape submit carries one (ksize, scale) pair per image. The same source answers for the
// specialized kernels (FivePoint/Scale) and for the generic correlation kernels (Size/Anchor/At/
// Linear), which is the same stencil scaled the same way.
struct VarShapeLaplacianParams
{
    cuda::Tensor1DWrap<int, int32_t>   ksize;
    cuda::Tensor1DWrap<float, int32_t> scale;

    inline __device__ bool FivePoint(int b) const
    {
        const int ksizeVal = ksize[b];
        NVCV_CUDA_ASSERT(ksizeVal == 1 || ksizeVal == 3, "E Wrong ksize = %d, expected: 1 or 3", ksizeVal);
        return ksizeVal == 1;
    }

    inline __device__ float Scale(int b) const
    {
        return scale[b];
    }

    inline __device__ int2 Size(int) const
    {
        return kStencilSize;
    }

    inline __device__ int2 Anchor(int, int2) const
    {
        return kStencilAnchor;
    }

    inline __device__ float At(int b, int i, int j) const
    {
        return Linear(b, i * kStencilSize.x + j, kStencilSize);
    }

    inline __device__ float Linear(int b, int idx, int2) const
    {
        return LaplacianStencil(FivePoint(b), idx) * Scale(b);
    }
};

// ---------------------------------------------------------------------------------------------
// Tap arithmetic
// ---------------------------------------------------------------------------------------------

// Keep scalar planar and compound interleaved kernels on the same explicitly rounded instruction
// sequence. The value may be float- or half-based: taps accumulate in the float work type R, and
// the per-element widen is exact for half and a no-op for float, so float instantiations are
// bit-identical to the value-type-only form.
template<typename T, typename R = cuda::ConvertBaseTypeTo<float, T>>
inline __device__ R LaplacianFloatMulRN(T value, float coefficient)
{
    static_assert(std::is_same_v<cuda::BaseType<R>, float>);

    R result;
#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        cuda::GetElement(result, c) = __fmul_rn(static_cast<float>(cuda::GetElement(value, c)), coefficient);
    }
    return result;
}

template<typename T, typename R>
inline __device__ R LaplacianFloatFmaRN(T value, float coefficient, R result)
{
    static_assert(std::is_same_v<cuda::BaseType<R>, float>);
    static_assert(cuda::NumElements<T> == cuda::NumElements<R>);

#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        cuda::GetElement(result, c)
            = __fmaf_rn(static_cast<float>(cuda::GetElement(value, c)), coefficient, cuda::GetElement(result, c));
    }
    return result;
}

// `ExplicitRounding` picks between the spelled-out mul/fma sequence and the plain expression the
// compiler contracts into the same instructions. Both are kept because integer inputs never need the
// explicit widen and reading `value * k` there is clearer.
template<bool ExplicitRounding, class Work, typename T>
inline __device__ Work LaplacianFirstTap(T value, float coefficient)
{
    if constexpr (ExplicitRounding)
    {
        return LaplacianFloatMulRN<T, Work>(value, coefficient);
    }
    else
    {
        return value * coefficient;
    }
}

template<bool ExplicitRounding, class Work, typename T>
inline __device__ Work LaplacianNextTap(T value, float coefficient, Work result)
{
    if constexpr (ExplicitRounding)
    {
        return LaplacianFloatFmaRN<T, Work>(value, coefficient, result);
    }
    else
    {
        return result + value * coefficient;
    }
}

// ---------------------------------------------------------------------------------------------
// Specialized kernels
// ---------------------------------------------------------------------------------------------

// Both stencils are sparse -- five non-zero taps out of nine -- so the dtypes that dominate the
// benchmark surface skip the generic correlation loop and address the five taps directly. `at(dx,
// dy)` is the only thing that differs between reading global memory and reading a staged tile, so
// the tap order lives here once.
template<class Work, bool ExplicitRounding, class Fetch>
inline __device__ Work LaplacianTaps(bool fivePoint, float scale, const Fetch &at)
{
    Work res;
    if (fivePoint)
    {
        const float axisK   = scale;
        const float centerK = -4.0f * scale;

        res = LaplacianFirstTap<ExplicitRounding, Work>(at(0, -1), axisK);
        res = LaplacianNextTap<ExplicitRounding>(at(-1, 0), axisK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(0, 0), centerK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(1, 0), axisK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(0, 1), axisK, res);
    }
    else
    {
        const float cornerK = 2.0f * scale;
        const float centerK = -8.0f * scale;

        res = LaplacianFirstTap<ExplicitRounding, Work>(at(-1, -1), cornerK);
        res = LaplacianNextTap<ExplicitRounding>(at(1, -1), cornerK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(0, 0), centerK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(-1, 1), cornerK, res);
        res = LaplacianNextTap<ExplicitRounding>(at(1, 1), cornerK, res);
    }
    return res;
}

template<bool ExplicitRounding, int OutputsPerThread, class SrcWrapper, class Dst, class Params>
__global__ void Laplacian2D(SrcWrapper src, Dst dst, Params params)
{
    using T         = typename Dst::ValueType;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    constexpr int kOutputsPerThread = OutputsPerThread;
    const int     y                 = blockIdx.y * blockDim.y + threadIdx.y;
    const int     batch_idx         = blockIdx.z;
    if (y >= dst.Height(batch_idx, 0))
        return;
    const int   width     = dst.Width(batch_idx, 0);
    const bool  fivePoint = params.FivePoint(batch_idx);
    const float scale     = params.Scale(batch_idx);
#pragma unroll
    for (int output = 0; output < kOutputsPerThread; ++output)
    {
        const unsigned int outputX = blockIdx.x * blockDim.x * kOutputsPerThread + threadIdx.x + output * blockDim.x;
        if (outputX >= static_cast<unsigned int>(width))
            continue;
        const int       x   = static_cast<int>(outputX);
        const work_type res = LaplacianTaps<work_type, ExplicitRounding>(
            fivePoint, scale,
            [&](int dx, int dy)
            {
                const int3 coord{x + dx, y + dy, batch_idx};
                // A constant border's `operator[]` has to
                // return a reference, and out of bounds the
                // only address it can name is the wrap's own
                // border member, which puts the whole wrapper
                // on the stack and demotes the tap loads to
                // generic ones. The by-value accessor reads
                // the same value without an address.
                if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                {
                    return src.value(coord);
                }
                else
                {
                    return src[coord];
                }
            });

        *dst.Ptr(batch_idx, 0, y, x) = cuda::SaturateCast<T>(res);
    }
}

// Five taps out of a 3x3 neighbourhood is enough reuse to pay for a shared-memory tile once the
// image is single-channel, which is what every planar plane is.
template<class T, bool ExplicitRounding, class Tile>
inline __device__ T LaplacianTileOutput(const Tile *tile, int tileW, int tileY, int tileX, bool fivePoint, float scale)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    const work_type res = LaplacianTaps<work_type, ExplicitRounding>(
        fivePoint, scale,
        [&](int dx, int dy)
        {
            const auto value = tile[(tileY + dy) * tileW + tileX + dx];
            if constexpr (std::is_same_v<T, uchar3> && std::is_same_v<Tile, uchar4>)
                return make_uchar3(value.x, value.y, value.z);
            else
                return value;
        });

    return cuda::SaturateCast<T>(res);
}

template<class Tile, class T>
inline __device__ Tile LaplacianStagePixel(T value)
{
    if constexpr (std::is_same_v<T, uchar3>)
        return make_uchar4(value.x, value.y, value.z, 0);
    else
        return value;
}

// Returning here never bypasses the caller's end-of-plane barrier.
template<int NIX, bool ExplicitRounding, bool VectorStore, bool CacheCenterRow, bool Planar, class Tile, class Dst>
inline __device__ void WriteLaplacianTileRow(const Tile *tile, int tileW, Dst dst, int batch_idx, int plane,
                                             int outBaseX, int outBaseY, int width, int height, bool fivePoint,
                                             float scale)
{
    using T               = typename Dst::ValueType;
    constexpr int kRadius = 1;
    const int     y       = outBaseY + threadIdx.y;
    if constexpr (!Planar)
        height = dst.Height(batch_idx, 0);
    if (y >= height)
        return;

    if constexpr (!Planar)
        width = dst.Width(batch_idx, 0);
    const int tileY = threadIdx.y + kRadius;

    if constexpr (VectorStore)
    {
        static_assert(NIX == 4 && (std::is_same_v<T, filter::uchar> || std::is_same_v<T, float>));
        const int x0 = outBaseX + threadIdx.x * NIX;
        if (x0 >= width)
            return;

        const int valid = width - x0 < NIX ? width - x0 : NIX;
        T         outputs[NIX];

        [[maybe_unused]] float centerRow[NIX + 2];
        if constexpr (CacheCenterRow)
        {
            if (fivePoint)
            {
#pragma unroll
                for (int j = 0; j < NIX + 2; ++j)
                    centerRow[j] = static_cast<float>(tile[tileY * tileW + threadIdx.x * NIX + j]);
            }
        }

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            if (i < valid)
                if constexpr (CacheCenterRow)
                {
                    const float result = LaplacianTaps<float, ExplicitRounding>(
                        fivePoint, scale,
                        [&](int dx, int dy) -> float
                        {
                            if (fivePoint && dy == 0)
                                return centerRow[i + kRadius + dx];
                            return static_cast<float>(
                                tile[(tileY + dy) * tileW + threadIdx.x * NIX + i + kRadius + dx]);
                        });
                    outputs[i] = cuda::SaturateCast<T>(result);
                }
                else
                    outputs[i] = LaplacianTileOutput<T, ExplicitRounding>(
                        tile, tileW, tileY, threadIdx.x * NIX + i + kRadius, fivePoint, scale);
        }

        T *dstPtr = dst.Ptr(batch_idx, plane, y, x0);
        if (valid == NIX && (reinterpret_cast<uintptr_t>(dstPtr) & (sizeof(T) * NIX - 1)) == 0)
        {
            if constexpr (std::is_same_v<T, filter::uchar>)
                *reinterpret_cast<uchar4 *>(dstPtr) = make_uchar4(outputs[0], outputs[1], outputs[2], outputs[3]);
            else
                *reinterpret_cast<float4 *>(dstPtr) = make_float4(outputs[0], outputs[1], outputs[2], outputs[3]);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i)
            {
                if (i < valid)
                    dstPtr[i] = outputs[i];
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = outBaseX + threadIdx.x + i * blockDim.x;
            if (x < width)
                *dst.Ptr(batch_idx, plane, y, x) = LaplacianTileOutput<T, ExplicitRounding>(
                    tile, tileW, tileY, threadIdx.x + i * blockDim.x + kRadius, fivePoint, scale);
        }
    }
}

// Tensor planes are flattened into the batch; planar VarShape reuses the same tile per plane.
// Keep the measured load/store gates explicit while sharing staging, taps and tail handling.
template<int NIX, bool ExplicitRounding, bool VectorStore, bool Planar, class SrcWrapper, class Dst, class Params>
inline __device__ void LaplacianTiled(SrcWrapper src, Dst dst, Params params, int channels)
{
    using T                  = typename Dst::ValueType;
    using Tile               = std::conditional_t<!Planar && std::is_same_v<T, __half>, float,
                                    std::conditional_t<std::is_same_v<T, uchar3>, uchar4, T>>;
    constexpr int  kRadius   = 1;
    constexpr int  outputW   = kPlanarBlockWidth * NIX;
    constexpr bool kVarShape = std::is_same_v<Params, VarShapeLaplacianParams>;
#if defined(__CUDA_ARCH__)
    constexpr int kSM = __CUDA_ARCH__ / 10;
#else
    constexpr int kSM = 0;
#endif
    constexpr bool kFloatVectorStage
        = kSM == 120
       && std::is_same_v<T, float> && (Planar ? SrcWrapper::kBorderType != NVCV_BORDER_CONSTANT : kVarShape);
    constexpr bool kHalf3Stage
        = kSM == 120 && !Planar && std::is_same_v<cuda::BaseType<T>, __half> && cuda::NumElements<T> == 3;
    constexpr bool kInteriorLoads
        = kSM == 120
       && (Planar ? (std::is_same_v<T, filter::uchar> || std::is_same_v<T, float>)&&SrcWrapper::kBorderType
                        != NVCV_BORDER_CONSTANT
                  : kVarShape || std::is_same_v<T, filter::uchar> || std::is_same_v<T, __half>);
    constexpr bool kVectorStage
        = kSM == 120
       && (Planar
               ? kInteriorLoads
               : kFloatVectorStage || kHalf3Stage
                     || std::is_same_v<
                         T,
                         filter::
                             uchar> || std::is_same_v<T, __half> || ((std::is_same_v<T, uchar3> || std::is_same_v<T, uchar4>)&&SrcWrapper::kBorderType != NVCV_BORDER_CONSTANT));
    // Row loads regress RGBA Tensor on sm80; packed RGB VarShape favors flattened traversal.
    constexpr bool kRowLoads = !Planar && kSM != 80
                            && (std::is_same_v<T, uchar4> || (kSM == 120 && std::is_same_v<T, uchar3> && !kVarShape));
    constexpr bool kLateParams     = kSM == 120 && !Planar && kVarShape && std::is_same_v<T, filter::uchar>;
    constexpr bool kCacheCenterRow = kSM == 120 && !Planar && !kVarShape && std::is_same_v<T, filter::uchar>;

    const int outBaseX  = blockIdx.x * outputW;
    const int outBaseY  = blockIdx.y * blockDim.y;
    const int batch_idx = blockIdx.z;
    int       dstWidth = 0, dstHeight = 0;
    if constexpr (Planar)
    {
        dstWidth  = dst.Width(batch_idx, 0);
        dstHeight = dst.Height(batch_idx, 0);
        if (outBaseX >= dstWidth || outBaseY >= dstHeight)
            return;
    }
    else if (outBaseX >= dst.Width(batch_idx, 0) || outBaseY >= dst.Height(batch_idx, 0))
        return;

    bool  fivePoint;
    float scale;
    if constexpr (!kLateParams)
    {
        fivePoint = params.FivePoint(batch_idx);
        scale     = params.Scale(batch_idx);
    }
    constexpr int     tileDataW = outputW + 2 * kRadius;
    constexpr int     tileW     = kFloatVectorStage ? outputW + 4 : tileDataW;
    const int         tileH     = blockDim.y + 2 * kRadius;
    const int         tileElems = tileDataW * tileH;
    const int         tid       = threadIdx.y * blockDim.x + threadIdx.x;
    extern __shared__ __align__(16) unsigned char tileRaw[];
    // Align the first interior packed group without changing the shared allocation geometry.
    Tile *tile = reinterpret_cast<Tile *>(tileRaw + (kHalf3Stage ? 2 : kFloatVectorStage ? 3 * sizeof(float) : 0));

    for (int plane = 0; plane < (Planar ? channels : 1); ++plane)
    {
        auto makeCoord = [&](int x, int y)
        {
            if constexpr (Planar)
                return int4{x, y, plane, batch_idx};
            else
                return int3{x, y, batch_idx};
        };
        int width = dst.Width(batch_idx, 0), height = dst.Height(batch_idx, 0);
        if constexpr (Planar)
        {
            width  = src.imageBatchWrap().width(batch_idx, plane);
            height = src.imageBatchWrap().height(batch_idx, plane);
        }
        const bool interior = kInteriorLoads && outBaseX >= kRadius && outBaseY >= kRadius
                           && outBaseX <= width - outputW - kRadius
                           && outBaseY <= height - static_cast<int>(blockDim.y) - kRadius;
        // Keep wrapper member lookup dependent for older CUDA compilers.
        auto readTile = [&](const auto &input, auto coord) -> T
        {
            if constexpr (kInteriorLoads)
            {
                if (interior)
                {
                    if constexpr (kVarShape)
                        return input.imageBatchWrap()[coord];
                    else
                        return input.tensorWrap()[coord];
                }
            }
            return input[coord];
        };
        bool vectorStaged = false;
        if constexpr (kVectorStage)
        {
            static_assert(NIX == 4);
            bool stageInterior = interior;
            if constexpr (!Planar
                          && ((kHalf3Stage && !kVarShape)
                              || std::is_same_v<T, uchar3> || (std::is_same_v<T, uchar4> && !kVarShape)))
            {
                stageInterior = outBaseX >= kRadius && outBaseY >= kRadius
                             && outBaseX <= dst.Width(batch_idx, 0) - outputW - kRadius
                             && outBaseY <= dst.Height(batch_idx, 0) - static_cast<int>(blockDim.y) - kRadius;
            }
            if (stageInterior)
            {
                for (int loadY = threadIdx.y; loadY < tileH; loadY += blockDim.y)
                {
                    const int loadX = threadIdx.x * NIX + kRadius;
                    auto      coord = makeCoord(outBaseX + threadIdx.x * NIX, outBaseY - kRadius + loadY);
                    const T  *ptr;
                    if constexpr (kVarShape)
                        ptr = &src.imageBatchWrap()[coord];
                    else
                        ptr = &src.tensorWrap()[coord];
                    constexpr int kLoadAlignment = kFloatVectorStage || std::is_same_v<T, uchar4> ? alignof(float4)
                                                 : std::is_same_v<T, __half>                      ? alignof(uint2)
                                                                                                  : alignof(uint3);
                    if ((reinterpret_cast<uintptr_t>(ptr) & (kLoadAlignment - 1)) == 0)
                    {
                        if constexpr (kFloatVectorStage)
                        {
                            Tile *sharedPtr                        = tile + loadY * tileW + loadX;
                            *reinterpret_cast<float4 *>(sharedPtr) = *reinterpret_cast<const float4 *>(ptr);
                        }
                        else if constexpr (kHalf3Stage)
                        {
                            Tile        *sharedPtr   = tile + loadY * tileW + loadX;
                            const uint3 *words       = reinterpret_cast<const uint3 *>(ptr);
                            uint3       *sharedWords = reinterpret_cast<uint3 *>(sharedPtr);
                            sharedWords[0]           = words[0];
                            sharedWords[1]           = words[1];
                        }
                        else if constexpr (std::is_same_v<T, __half>)
                        {
                            const uint2 value           = *reinterpret_cast<const uint2 *>(ptr);
                            tile[loadY * tileW + loadX] = __ushort_as_half(static_cast<unsigned short>(value.x));
                            tile[loadY * tileW + loadX + 1]
                                = __ushort_as_half(static_cast<unsigned short>(value.x >> 16));
                            tile[loadY * tileW + loadX + 2] = __ushort_as_half(static_cast<unsigned short>(value.y));
                            tile[loadY * tileW + loadX + 3]
                                = __ushort_as_half(static_cast<unsigned short>(value.y >> 16));
                        }
                        else if constexpr (std::is_same_v<T, filter::uchar>)
                        {
                            const uchar4 value              = *reinterpret_cast<const uchar4 *>(ptr);
                            tile[loadY * tileW + loadX]     = value.x;
                            tile[loadY * tileW + loadX + 1] = value.y;
                            tile[loadY * tileW + loadX + 2] = value.z;
                            tile[loadY * tileW + loadX + 3] = value.w;
                        }
                        else if constexpr (std::is_same_v<T, uchar4>)
                        {
                            const uint4 value = *reinterpret_cast<const uint4 *>(ptr);
                            tile[loadY * tileW + loadX]
                                = make_uchar4(value.x, value.x >> 8, value.x >> 16, value.x >> 24);
                            tile[loadY * tileW + loadX + 1]
                                = make_uchar4(value.y, value.y >> 8, value.y >> 16, value.y >> 24);
                            tile[loadY * tileW + loadX + 2]
                                = make_uchar4(value.z, value.z >> 8, value.z >> 16, value.z >> 24);
                            tile[loadY * tileW + loadX + 3]
                                = make_uchar4(value.w, value.w >> 8, value.w >> 16, value.w >> 24);
                        }
                        else
                        {
                            static_assert(std::is_same_v<T, uchar3>);
                            const uint3 value               = *reinterpret_cast<const uint3 *>(ptr);
                            tile[loadY * tileW + loadX]     = make_uchar4(value.x, value.x >> 8, value.x >> 16, 0);
                            tile[loadY * tileW + loadX + 1] = make_uchar4(value.x >> 24, value.y, value.y >> 8, 0);
                            tile[loadY * tileW + loadX + 2] = make_uchar4(value.y >> 16, value.y >> 24, value.z, 0);
                            tile[loadY * tileW + loadX + 3]
                                = make_uchar4(value.z >> 8, value.z >> 16, value.z >> 24, 0);
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < NIX; ++i)
                            tile[loadY * tileW + loadX + i] = LaplacianStagePixel<Tile>(ptr[i]);
                    }
                    if (threadIdx.x == 0)
                    {
                        tile[loadY * tileW]                 = LaplacianStagePixel<Tile>(ptr[-1]);
                        tile[loadY * tileW + tileDataW - 1] = LaplacianStagePixel<Tile>(ptr[outputW]);
                    }
                }
                vectorStaged = true;
            }
        }
        if (!vectorStaged)
        {
            if constexpr (kRowLoads)
            {
                for (int loadY = threadIdx.y; loadY < tileH; loadY += blockDim.y)
                {
                    for (int loadX = threadIdx.x; loadX < tileW; loadX += blockDim.x)
                    {
                        auto coord                  = makeCoord(outBaseX - kRadius + loadX, outBaseY - kRadius + loadY);
                        tile[loadY * tileW + loadX] = LaplacianStagePixel<Tile>(readTile(src, coord));
                    }
                }
            }
            else
            {
                for (int idx = tid; idx < tileElems; idx += blockDim.x * blockDim.y)
                {
                    const int loadX             = idx % tileDataW;
                    const int loadY             = idx / tileDataW;
                    auto      coord             = makeCoord(outBaseX - kRadius + loadX, outBaseY - kRadius + loadY);
                    tile[loadY * tileW + loadX] = LaplacianStagePixel<Tile>(readTile(src, coord));
                }
            }
        }
        __syncthreads();
        if constexpr (kLateParams)
        {
            fivePoint = params.FivePoint(batch_idx);
            scale     = params.Scale(batch_idx);
        }
        WriteLaplacianTileRow<NIX, ExplicitRounding, VectorStore, kCacheCenterRow, Planar>(
            tile, tileW, dst, batch_idx, plane, outBaseX, outBaseY, dstWidth, dstHeight, fivePoint, scale);
        if constexpr (Planar)
            __syncthreads();
    }
}

template<int NIX, bool ExplicitRounding, bool VectorStore, class SrcWrapper, class Dst, class Params>
__global__ void Laplacian2DTensorPlanarTiled(SrcWrapper src, Dst dst, Params params)
{
    LaplacianTiled<NIX, ExplicitRounding, VectorStore, false>(src, dst, params, 1);
}

template<int NIX, bool ExplicitRounding, bool VectorStore, class SrcWrapper, class Dst, class Params>
__global__ void Laplacian2DVarShapePlanarTiled(SrcWrapper src, Dst dst, Params params, int channels)
{
    LaplacianTiled<NIX, ExplicitRounding, VectorStore, true>(src, dst, params, channels);
}

// ---------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------

template<typename T, NVCVBorderType B>
auto MakeLaplacianTensorViews(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              float borderValue)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);
    filter::RequireInt32Addressable(*inAccess, *outAccess);

    const int2 size{outAccess->numCols(), outAccess->numRows()};
    return std::make_tuple(cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, cuda::SetAll<T>(borderValue)),
                           filter::MakeUniformDst(cuda::CreateTensorWrapNHW<T, int32_t>(outData), size),
                           make_int3(size.x, size.y, outAccess->numSamples()));
}

template<bool ExplicitRounding, class Src, class Dst, class Params>
void LaunchLaplacian(Src src, Dst dst, Params params, int3 size, cudaStream_t stream)
{
    using T                  = typename Dst::ValueType;
    constexpr auto B         = Src::kBorderType;
    constexpr bool kVarShape = std::is_same_v<Params, VarShapeLaplacianParams>;
    constexpr bool kScalar = std::is_same_v<T, filter::uchar> || std::is_same_v<T, float> || std::is_same_v<T, __half>;
    constexpr bool kHalf3  = std::is_same_v<cuda::BaseType<T>, __half> && cuda::NumElements<T> == 3;
    constexpr bool kTiledPixel = kScalar || std::is_same_v<T, uchar3> || std::is_same_v<T, uchar4> || kHalf3;
    const bool     warpRows    = B != NVCV_BORDER_CONSTANT
                       && (std::is_same_v<T, float3> || (kVarShape && std::is_same_v<T, float4>))
                       && CurrentDeviceSMOrZero() == 120;
    const int     launchWidth  = B == NVCV_BORDER_CONSTANT || warpRows ? 32 : kBlockWidth;
    const int     launchHeight = B == NVCV_BORDER_CONSTANT || warpRows ? 8 : kBlockHeight;
    constexpr int kOutputs     = sizeof(T) <= 8 ? 2 : 1;
    dim3          block(launchWidth, launchHeight);
    dim3          grid(util::DivUp(size.x, launchWidth * kOutputs), util::DivUp(size.y, launchHeight), size.z);

    // Tensor and VarShape retain their separately measured compound-pixel policies.
    if constexpr (kTiledPixel && B != NVCV_BORDER_CONSTANT)
    {
        if (kScalar || (kVarShape ? CurrentDeviceSMOrZero() == 120 : TiledFilterPaysHere<T>(CurrentDeviceSMOrZero())))
        {
            constexpr int    kTiledOutputs = 4;
            constexpr size_t kSharedBytes  = (kPlanarBlockWidth * kTiledOutputs + 2) * (kBlockHeight + 2)
                                              * (std::is_same_v<T, uchar3>   ? sizeof(uchar4)
                                                 : std::is_same_v<T, __half> ? sizeof(float)
                                                                             : sizeof(T))
                                          + (kHalf3 ? 2 : 0);
            static_assert(kSharedBytes <= filter::kMaxSharedBytes);
            size_t sharedBytes = kSharedBytes;
            if constexpr (kVarShape && std::is_same_v<T, float>)
            {
                // Compatible devices can execute the compiled sm120 layout, including its padding.
                sharedBytes += (2 * (kBlockHeight + 2) + 3) * sizeof(float);
            }
            dim3 tiledBlock(kPlanarBlockWidth, kBlockHeight);
            dim3 tiledGrid(util::DivUp(size.x, kPlanarBlockWidth * kTiledOutputs), util::DivUp(size.y, kBlockHeight),
                           size.z);
            Laplacian2DTensorPlanarTiled<kTiledOutputs, ExplicitRounding, sizeof(T) == 1>
                <<<tiledGrid, tiledBlock, sharedBytes, stream>>>(src, dst, params);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }

    // Reference measurements favor one output per thread for these Ampere VarShape byte paths.
    if constexpr (kVarShape && B != NVCV_BORDER_CONSTANT && (std::is_same_v<T, uchar3> || std::is_same_v<T, uchar4>))
    {
        if (CurrentDeviceSMOrZero() == 80)
        {
            dim3 singleGrid(util::DivUp(size.x, launchWidth), grid.y, grid.z);
            Laplacian2D<ExplicitRounding, 1><<<singleGrid, block, 0, stream>>>(src, dst, params);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }
    if constexpr (B == NVCV_BORDER_CONSTANT && std::is_same_v<T, uchar3>)
    {
        if (CurrentDeviceSMOrZero() == 120)
        {
            constexpr int kConstantOutputs = 4;
            dim3          constantGrid(util::DivUp(size.x, launchWidth * kConstantOutputs), grid.y, grid.z);
            Laplacian2D<ExplicitRounding, kConstantOutputs><<<constantGrid, block, 0, stream>>>(src, dst, params);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }
    }
    // Avoid instantiating a fallback for types that always returned from the tiled launch.
    if constexpr (B == NVCV_BORDER_CONSTANT || !(kScalar || (!kVarShape && std::is_same_v<T, uchar4>)))
    {
        Laplacian2D<ExplicitRounding, kOutputs><<<grid, block, 0, stream>>>(src, dst, params);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename T, NVCVBorderType B, bool ExplicitRounding, int KSize>
void LaplacianTensorDirect(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           float scale, float borderValue, cudaStream_t stream)
{
    auto [src, dst, size] = MakeLaplacianTensorViews<T, B>(inData, outData, borderValue);
    LaunchLaplacian<ExplicitRounding>(src, dst, FixedLaplacianParams<KSize>{scale}, size, stream);
}

template<typename T, NVCVBorderType B, int KSize>
void LaplacianTensorPlanarTiled(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                float scale, float borderValue, cudaStream_t stream)
{
    auto [src, dst, size] = MakeLaplacianTensorViews<T, B>(inData, outData, borderValue);
    dim3         block(kPlanarBlockWidth, kBlockHeight);
    dim3         grid(util::DivUp(size.x, kPlanarBlockWidth * kPlanarNIX), util::DivUp(size.y, kBlockHeight), size.z);
    const size_t sharedBytes = static_cast<size_t>(kPlanarBlockWidth * kPlanarNIX + 2) * (block.y + 2)
                             * (std::is_same_v<T, __half> ? sizeof(float) : sizeof(T));
    // Planar half retains its distinct rounding; only byte pixels use the packed output store.
    Laplacian2DTensorPlanarTiled<kPlanarNIX, std::is_same_v<T, float>, sizeof(T) == 1>
        <<<grid, block, sharedBytes, stream>>>(src, dst, FixedLaplacianParams<KSize>{scale});
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void LaplacianTensorGeneric(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            const cuda::math::Vector<float, 9> &stencil, float borderValue, cudaStream_t stream)
{
    auto [src, dst, size] = MakeLaplacianTensorViews<T, B>(inData, outData, borderValue);
    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(size.x, kBlockWidth), util::DivUp(size.y, kBlockHeight), size.z);
    filter::Filter2D<<<grid, block, 0, stream>>>(src, dst,
                                                 filter::MakeUniformCoefs(stencil, kStencilSize, kStencilAnchor));
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B, bool ExplicitRounding>
void LaplacianVarShapeDirect(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                             const VarShapeLaplacianParams &params, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));
    auto                                 dst = filter::MakeVarShapeDst(cuda::ImageBatchVarShapeWrap<T>(outData));
    LaunchLaplacian<ExplicitRounding>(src, dst, params,
                                      make_int3(inData.maxSize().w, inData.maxSize().h, outData.numImages()), stream);
}

template<typename T, NVCVBorderType B>
void LaplacianVarShapeGeneric(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                              const VarShapeLaplacianParams &coefs, float borderValue, cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto dst = filter::MakeVarShapeDst(cuda::ImageBatchVarShapeWrap<T>(outData));

    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(inData.maxSize().w, kBlockWidth), util::DivUp(inData.maxSize().h, kBlockHeight),
              outData.numImages());

    filter::Filter2D<<<grid, block, 0, stream>>>(src, dst, coefs);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void LaplacianVarShapePlanarGeneric(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                    const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                    const VarShapeLaplacianParams &coefs, int channels, float borderValue,
                                    cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto dst = filter::MakeVarShapePlanarDst(cuda::ImageBatchVarShapeWrap<T>(outData), channels);

    dim3 block(kBlockWidth, kBlockHeight);
    dim3 grid(util::DivUp(outData.maxSize().w, kBlockWidth), util::DivUp(outData.maxSize().h, kBlockHeight),
              outData.numImages());

    filter::Filter2DPlanar<<<grid, block, 0, stream>>>(src, dst, coefs, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVBorderType B>
void LaplacianVarShapePlanarTiled(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                  const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                                  const VarShapeLaplacianParams &params, int channels, float borderValue,
                                  cudaStream_t stream)
{
    cuda::BorderVarShapeWrap<const T, B> src(inData, cuda::SetAll<T>(borderValue));

    auto dst = filter::MakeVarShapePlanarDst(cuda::ImageBatchVarShapeWrap<T>(outData), channels);

    // float planes need the shorter block: their tile is four bytes per element, so a 16-row block
    // would spend more of the SM's shared memory than the extra occupancy buys back.
    dim3 block(kPlanarBlockWidth, std::is_same_v<T, float> ? 8 : kBlockHeight);
    dim3 grid(util::DivUp(outData.maxSize().w, kPlanarBlockWidth * kPlanarNIX),
              util::DivUp(outData.maxSize().h, static_cast<int>(block.y)), outData.numImages());

    size_t sharedBytes = static_cast<size_t>(kPlanarBlockWidth * kPlanarNIX + 2) * (block.y + 2) * sizeof(T);
    if constexpr (std::is_same_v<T, float> && B != NVCV_BORDER_CONSTANT)
    {
        // Reserve the compiled sm120 layout even when it runs on a compatible device.
        sharedBytes += (2 * (block.y + 2) + 3) * sizeof(T);
    }

    constexpr bool kVectorStore      = std::is_same_v<T, filter::uchar> || std::is_same_v<T, float>;
    constexpr bool kExplicitRounding = std::is_same_v<T, float>;

    Laplacian2DVarShapePlanarTiled<kPlanarNIX, kExplicitRounding, kVectorStore>
        <<<grid, block, sharedBytes, stream>>>(src, dst, params, channels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// ClassifySupportedType admits {U8, U16, F32, F16} and every caller below peels U16 off first, so
// the shared six-type switches would instantiate U16, S16 and S32 kernels nothing can reach.
template<class Cb>
inline void RunLaplacianBaseTypeSwitch(filter::ElemType type, const Cb &cb)
{
    switch (type)
    {
    case filter::ElemType::kU8:
        return cb(filter::uchar{});
    case filter::ElemType::kF32:
        return cb(float{});
    case filter::ElemType::kF16:
        return cb(__half{});
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type");
    }
}

template<class Base, class Cb>
inline void RunLaplacianChannelSwitch(int channels, const Cb &cb)
{
    if (channels == 1)
        cb(cuda::MakeType<Base, 0>{});
    else if (channels == 3)
        cb(cuda::MakeType<Base, 3>{});
    else if (channels == 4)
        cb(cuda::MakeType<Base, 4>{});
    else
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels (%d): 1, 3 or 4 are supported", channels);
}

template<class Cb>
inline void RunLaplacianTypeSwitch(filter::ElemType type, int channels, const Cb &cb)
{
    RunLaplacianBaseTypeSwitch(type,
                               [channels, &cb](auto base) { RunLaplacianChannelSwitch<decltype(base)>(channels, cb); });
}

inline filter::ElemType ClassifySupportedType(nvcv::DataType dtype)
{
    const filter::ElemType elemType = filter::ClassifyDataType(dtype);
    if (!(elemType == filter::ElemType::kU8 || elemType == filter::ElemType::kU16 || elemType == filter::ElemType::kF32
          || elemType == filter::ElemType::kF16))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type");
    }
    return elemType;
}

// Bind ksize to a template parameter so the stencil choice folds out of the tensor kernels.
template<class Cb>
inline void RunKSizeSwitch(int ksize, const Cb &cb)
{
    if (ksize == 1)
    {
        cb(std::integral_constant<int, 1>{});
    }
    else
    {
        cb(std::integral_constant<int, 3>{});
    }
}

} // namespace

void Laplacian::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const int ksize,
                           const float scale, const NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Laplacian::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    if (!(ksize == 1 || ksize == 3))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid ksize %d, expected 1 or 3", ksize);
    }

    RequireSameDtype(*inData, *outData);

    filter::ValidateTensorLayout(inData->layout());
    filter::ValidateTensorLayout(outData->layout());
    RequireSameLayout(*inData, *outData);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = ClassifySupportedType(inData->dtype());

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    RequireImageAccess(inAccess, "Input");
    RequireImageAccess(outAccess, "Output");
    RequireSameImageShape(*inAccess, *outAccess);

    const int channels = inAccess->numChannels();
    filter::ValidateChannelCount(channels);

    constexpr float borderValue = 0.f;

    auto planarViews = PlanarSingleChannelViews(*inData, *outData);

    const nvcv::TensorDataStridedCuda &src = planarViews ? planarViews->first : *inData;
    const nvcv::TensorDataStridedCuda &dst = planarViews ? planarViews->second : *outData;

    // Only the 16-bit unsigned arm still runs the generic correlation; it is the one dtype with no
    // benchmarked specialization, so it keeps the host-scaled stencil the generic kernel expects.
    if (elemType == filter::ElemType::kU16)
    {
        const cuda::math::Vector<float, 9> stencil = MakeScaledStencil(ksize, scale);

        RunLaplacianChannelSwitch<filter::ushort>(
            planarViews ? 1 : channels,
            [&](auto value)
            {
                using T = decltype(value);
                filter::DispatchBorderMode(
                    borderMode, [&](auto border)
                    { LaplacianTensorGeneric<T, decltype(border)::value>(src, dst, stencil, borderValue, stream); });
            });
        return;
    }

    if (planarViews)
    {
        RunLaplacianBaseTypeSwitch(
            elemType,
            [&](auto base)
            {
                using T = decltype(base);
                filter::DispatchBorderMode(
                    borderMode,
                    [&](auto border)
                    {
                        RunKSizeSwitch(
                            ksize,
                            [&](auto kSizeTag)
                            {
                                LaplacianTensorPlanarTiled<T, decltype(border)::value, decltype(kSizeTag)::value>(
                                    src, dst, scale, borderValue, stream);
                            });
                    });
            });
        return;
    }

    RunLaplacianTypeSwitch(elemType, channels,
                           [&](auto value)
                           {
                               using T = decltype(value);
                               // Integer taps need no explicit widen; the float and half arms keep the spelled-out
                               // mul/fma sequence so the rounding does not depend on compiler contraction.
                               constexpr bool kExplicitRounding = !std::is_same_v<cuda::BaseType<T>, filter::uchar>;
                               filter::DispatchBorderMode(
                                   borderMode,
                                   [&](auto border)
                                   {
                                       RunKSizeSwitch(
                                           ksize,
                                           [&](auto kSizeTag)
                                           {
                                               LaplacianTensorDirect<T, decltype(border)::value, kExplicitRounding,
                                                                     decltype(kSizeTag)::value>(src, dst, scale,
                                                                                                borderValue, stream);
                                           });
                                   });
                           });
}

void Laplacian::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                           const nvcv::Tensor &ksize, const nvcv::Tensor &scale, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Laplacian::operator()[ImageBatchVarShape]");
    const int numImages = in.numImages();
    if (numImages != out.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images");
    }
    RequireSameVarShapeImageSizes(in, out);

    auto inData    = RequireCudaVarShape(in, stream, "Input");
    auto outData   = RequireCudaVarShape(out, stream, "Output");
    auto ksizeData = RequireCudaTensor(ksize, "Kernel aperture size");
    auto scaleData = RequireCudaTensor(scale, "Kernel scale");

    // Both are read through Tensor1DWrap, which reinterprets rather than converts, so the dtype has
    // to be checked here as well as the rank and length.
    RequirePerImageParam(*ksizeData, nvcv::TYPE_S32, numImages, "Kernel aperture size");
    RequirePerImageParam(*scaleData, nvcv::TYPE_F32, numImages, "Kernel scale");

    RequireUniqueFormat(*inData, "input");
    RequireUniqueFormat(*outData, "output");

    const filter::BatchLayout inLayout  = filter::ClassifyBatchLayout(*inData);
    const filter::BatchLayout outLayout = filter::ClassifyBatchLayout(*outData);
    filter::RequireSameBatchLayout(inLayout, outLayout);

    filter::ValidateBorderMode(borderMode);

    const filter::ElemType elemType = ClassifySupportedType(inData->uniqueFormat().planeDataType(0));

    const int channels = inData->uniqueFormat().numChannels();
    filter::ValidateChannelCount(channels);

    constexpr float borderValue = 0.f;

    const VarShapeLaplacianParams params{cuda::Tensor1DWrap<int, int32_t>(*ksizeData),
                                         cuda::Tensor1DWrap<float, int32_t>(*scaleData)};

    if (inLayout.planar)
    {
        filter::RequirePlanarGridZ(outData->numImages());

        if (elemType == filter::ElemType::kU16)
        {
            filter::DispatchBorderMode(borderMode,
                                       [&](auto border)
                                       {
                                           LaplacianVarShapePlanarGeneric<filter::ushort, decltype(border)::value>(
                                               *inData, *outData, params, channels, borderValue, stream);
                                       });
            return;
        }

        RunLaplacianBaseTypeSwitch(elemType,
                                   [&](auto base)
                                   {
                                       using T = decltype(base);
                                       filter::DispatchBorderMode(
                                           borderMode,
                                           [&](auto border)
                                           {
                                               LaplacianVarShapePlanarTiled<T, decltype(border)::value>(
                                                   *inData, *outData, params, channels, borderValue, stream);
                                           });
                                   });
        return;
    }

    if (elemType == filter::ElemType::kU16)
    {
        RunLaplacianChannelSwitch<filter::ushort>(
            channels,
            [&](auto value)
            {
                filter::DispatchBorderMode(borderMode,
                                           [&](auto border)
                                           {
                                               LaplacianVarShapeGeneric<decltype(value), decltype(border)::value>(
                                                   *inData, *outData, params, borderValue, stream);
                                           });
            });
        return;
    }

    RunLaplacianTypeSwitch(elemType, channels,
                           [&](auto value)
                           {
                               using T                          = decltype(value);
                               constexpr bool kExplicitRounding = !std::is_same_v<cuda::BaseType<T>, filter::uchar>;
                               filter::DispatchBorderMode(
                                   borderMode,
                                   [&](auto border)
                                   {
                                       LaplacianVarShapeDirect<T, decltype(border)::value, kExplicitRounding>(
                                           *inData, *outData, params, borderValue, stream);
                                   });
                           });
}

} // namespace cvcuda::priv
