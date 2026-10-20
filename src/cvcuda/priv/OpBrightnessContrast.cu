/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "BrightnessContrastPolicy.hpp"
#include "CudaDeviceUtils.hpp"
#include "Nvtx.hpp"
#include "OpBrightnessContrast.hpp"

#include "AdjustColorCommon.cuh"
#include "ElementStreamCommon.cuh"
#include "PhotometricBound.cuh"
#include "SameShapeCommon.cuh"

#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <tuple>
#include <type_traits>

namespace adjust      = cvcuda::priv::adjust;
namespace cuda        = nvcv::cuda;
namespace util        = nvcv::util;
namespace same_shape  = cvcuda::priv::same_shape;
namespace elem_stream = cvcuda::priv::elem_stream;

namespace {

template<typename T, typename Ret>
constexpr __host__ __device__ Ret HalfRange()
{
    // Use T{1} to avoid int-width UB when T is a 64-bit integer (1<<62 in int is UB).
    if constexpr (std::is_integral_v<T>)
        return static_cast<Ret>(T{1} << (8 * sizeof(T) - std::is_signed<T>::value - 1));
    else
        return Ret(0.5);
}

template<typename T>
inline constexpr bool RequiresDouble = (std::is_integral_v<T> && sizeof(T) >= 4) || std::is_same_v<T, double>;

template<typename SrcBT, typename DstBT>
using GetArgType = std::conditional_t<RequiresDouble<SrcBT> || RequiresDouble<DstBT>, double, float>;

template<typename ArgT>
using ArgWrapper = cuda::Tensor1DWrap<const ArgT, int32_t>;

template<typename BT, bool IsScalar>
struct SampleArgs;

template<typename BT>
struct SampleArgs<BT, false>
{
    BT brightness;
    BT contrast;
    BT brightnessShift;
    BT contrastCenter;
};

template<typename BT>
struct SampleArgs<BT, true>
{
    BT   brightness;
    BT   contrast;
    BT   brightnessShift;
    BT   contrastCenter;
    bool clamp;
    bool truncate;
};

template<typename BT, bool IsScalar>
struct BatchArgsWrap;

template<typename BT>
struct BatchArgsWrap<BT, false>
{
    int                  brightnessLen, contrastLen, brightnessShiftLen, contrastCenterLen;
    const ArgWrapper<BT> brightness;
    const ArgWrapper<BT> contrast;
    const ArgWrapper<BT> brightnessShift;
    const ArgWrapper<BT> contrastCenter;
};

template<typename BT>
struct BatchArgsWrap<BT, true>
{
    BT   brightness;
    BT   contrast;
    BT   brightnessShift;
    BT   contrastCenter;
    bool clamp;
    bool truncate;
};

template<typename BT>
inline __device__ BT GetArg(const ArgWrapper<BT> &tensorArg, int argLen, int sampleIdx, BT defaultVal)
{
    if (argLen == 0)
    {
        return defaultVal;
    }
    else if (argLen == 1)
    {
        return tensorArg[0];
    }
    else
    {
        return tensorArg[sampleIdx];
    }
}

template<typename SrcBT, typename BT, bool IsScalar>
inline __device__ SampleArgs<BT, IsScalar> GetBrightnessContrastArg(const BatchArgsWrap<BT, IsScalar> &args,
                                                                    int                                sampleIdx)
{
    if constexpr (IsScalar)
    {
        return {args.brightness, args.contrast, args.brightnessShift, args.contrastCenter, args.clamp, args.truncate};
    }
    else
    {
        return {GetArg(args.brightness, args.brightnessLen, sampleIdx, BT{1}),
                GetArg(args.contrast, args.contrastLen, sampleIdx, BT{1}),
                GetArg(args.brightnessShift, args.brightnessShiftLen, sampleIdx, BT{0}),
                GetArg(args.contrastCenter, args.contrastCenterLen, sampleIdx, HalfRange<SrcBT, BT>())};
    }
}

template<typename DstT, typename T>
inline __device__ T ClampToImageRange(T value, bool clamp)
{
    if (!clamp)
    {
        return value;
    }

    using BT                         = cuda::BaseType<T>;
    using DstBT                      = cuda::BaseType<DstT>;
    static constexpr int numElements = cuda::NumElements<T>;
    // Upper clamp bound keyed on the destination dtype, in the (float/double) intermediate type.
    const BT             bound = cvcuda::priv::PhotometricUpperBound<DstBT, BT>();
#pragma unroll
    for (int c = 0; c < numElements; ++c)
    {
        auto &v = cuda::GetElement(value, c);
        v       = v < BT{0} ? BT{0} : (v > bound ? bound : v);
    }
    return value;
}

// How the clamp step is decided. Runtime reads arg.clamp, which only scalar arguments carry; the
// element stream instead lifts that bit to a template argument so the branch leaves the inner loop.
// ClampToImageRange(v, false) is the identity, so Never simply omits the step.
enum class BCClampMode
{
    Runtime,
    Always,
    Never
};

// Blackwell conversions share the scarce FP64 pipe with the affine math. Integer
// normalization and ties-to-even rounding preserve native conversion bits without that contention.
__device__ __forceinline__ double BrightnessContrastToDouble(long long v)
{
#if __CUDA_ARCH__ >= 1000
    const unsigned long long mag
        = v < 0 ? 0ULL - static_cast<unsigned long long>(v) : static_cast<unsigned long long>(v);
    if (!mag)
        return 0.;
    const int          top = 63 - __clzll(mag);
    unsigned long long sig;
    if (top <= 52)
        sig = mag << (52 - top);
    else
    {
        const unsigned distance = top - 52;
        sig                     = (mag + ((1ULL << (distance - 1)) - 1) + ((mag >> distance) & 1)) >> distance;
    }
    const unsigned long long bits
        = (v < 0 ? 0x8000000000000000ULL : 0) | ((static_cast<unsigned long long>(1022 + top) << 52) + sig);
    return __longlong_as_double(bits);
#else
    return static_cast<double>(v);
#endif
}

__device__ __forceinline__ long long BrightnessContrastToS64(double v)
{
    unsigned long long bits;
    // An opaque bitcast keeps NVCC from lowering the magnitude mask to FP64 abs.
    asm("mov.b64 %0, %1;" : "=l"(bits) : "d"(v));
    const unsigned long long magnitude = bits & 0x7fffffffffffffffULL;
    if (magnitude >= 0x43e0000000000000ULL)
    {
        if (magnitude > 0x7ff0000000000000ULL)
            return __double2ll_rn(v);
        return bits >> 63 ? (-0x7fffffffffffffffLL - 1) : 0x7fffffffffffffffLL;
    }
#if __CUDA_ARCH__ >= 1000
    const unsigned e = magnitude >> 52;
    if (e < 1022)
        return 0;
    const unsigned long long sig = (bits & 0xfffffffffffffULL) | 0x10000000000000ULL;
    unsigned long long       result;
    if (e >= 1075)
        result = sig << (e - 1075);
    else
    {
        const unsigned d = 1075 - e;
        result           = (sig + (1ULL << (d - 1)) - 1 + ((sig >> d) & 1)) >> d;
    }
    return bits >> 63 ? -static_cast<long long>(result) : static_cast<long long>(result);
#else
    return __double2ll_rn(v);
#endif
}

// Per-pixel affine, factored out so every path shares byte-identical math.
template<BCClampMode kClampMode, typename DstT, typename SrcT, typename ArgT, bool IsScalar>
inline __device__ DstT ApplyBrightnessContrast(SrcT v, const SampleArgs<ArgT, IsScalar> &arg)
{
    using IntermediateT = decltype(std::declval<ArgT>() * std::declval<SrcT>());
    using BI            = cuda::BaseType<IntermediateT>;
    IntermediateT pixel;
    if constexpr (std::is_same_v<cuda::BaseType<SrcT>, long long>)
    {
#pragma unroll
        for (int c = 0; c < cuda::NumElements<SrcT>; ++c)
            cuda::GetElement(pixel, c) = BrightnessContrastToDouble(cuda::GetElement(v, c));
    }
    else
    {
        pixel = cuda::StaticCast<BI>(v);
    }
    pixel = arg.brightnessShift + arg.brightness * (arg.contrastCenter + arg.contrast * (pixel - arg.contrastCenter));
    if constexpr (kClampMode == BCClampMode::Runtime)
    {
        if constexpr (IsScalar)
        {
            pixel = ClampToImageRange<DstT>(pixel, arg.clamp);
        }
    }
    else if constexpr (kClampMode == BCClampMode::Always)
    {
        pixel = ClampToImageRange<DstT>(pixel, true);
    }
    if constexpr (IsScalar)
    {
        pixel = adjust::RoundForOutput<DstT>(pixel, arg.truncate);
    }
    if constexpr (std::is_same_v<cuda::BaseType<DstT>, long long>)
    {
        DstT out;
#pragma unroll
        for (int c = 0; c < cuda::NumElements<DstT>; ++c)
        {
            cuda::GetElement(out, c) = BrightnessContrastToS64(cuda::GetElement(pixel, c));
        }
        return out;
    }
    else
    {
        return cuda::SaturateCast<DstT>(pixel);
    }
}

template<typename SrcT, typename DstT, typename ArgT, bool IsScalar>
inline __device__ void BrightnessContrastPixel(const SrcT *src, DstT *dst, const SampleArgs<ArgT, IsScalar> &arg)
{
    SrcT pixel;
    if constexpr (alignof(SrcT) > alignof(cuda::BaseType<SrcT>))
    {
        pixel = reinterpret_cast<uintptr_t>(src) % alignof(SrcT) == 0 ? *src : same_shape::LoadUnalignedPixel(src);
    }
    else
    {
        pixel = *src;
    }
    const DstT out = ApplyBrightnessContrast<BCClampMode::Runtime, DstT>(pixel, arg);
    if constexpr (alignof(DstT) > alignof(cuda::BaseType<DstT>))
    {
        if (reinterpret_cast<uintptr_t>(dst) % alignof(DstT) != 0)
        {
            same_shape::StoreUnalignedPixel(dst, out);
            return;
        }
    }
    *dst = out;
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper, typename ArgT, bool IsScalar>
inline __device__ void DoBrightnessContrast(SrcWrapper src, DstWrapper dst, const SampleArgs<ArgT, IsScalar> arg,
                                            const int2 size, const int p)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    using IntermediateT              = decltype(std::declval<ArgT>() * std::declval<SrcT>());
    using SrcBT                      = cuda::BaseType<SrcT>;
    using DstBT                      = cuda::BaseType<DstT>;
    using BI                         = cuda::BaseType<IntermediateT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    // if planar then no interleaved channels
    static_assert(!IsPlanar || numChannels == 1);
    static_assert(std::is_same_v<BI, GetArgType<SrcBT, DstBT>>);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    if constexpr (IsPlanar)
    {
        auto coord = same_shape::GetCoordForLayout<IsPlanar>(nhwCoord, p);
        dst[coord] = ApplyBrightnessContrast<BCClampMode::Runtime, DstT>(src[coord], arg);
    }
    else
    {
        BrightnessContrastPixel(src.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x),
                                dst.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x), arg);
    }
}

// Element stream. Arguments are per sample, so every base element takes the same map.

// Element count is set by the wider base type, so that side fills exactly one 128-bit access.
template<typename SrcBT, typename DstBT>
constexpr int BC_STREAM_N
    = elem_stream::kCarrierBytes / (sizeof(SrcBT) > sizeof(DstBT) ? sizeof(SrcBT) : sizeof(DstBT));

template<bool kClamp>
inline constexpr BCClampMode kBCStaticClamp = kClamp ? BCClampMode::Always : BCClampMode::Never;

constexpr int kBCStreamBlockW = 256;
// Cache setup is measurable on short nonrepeating streams.
constexpr int kBCStreamCacheMinElems = 1 << 20;

// More groups amortize the per-sample argument loads; 4-byte elements saturate earlier.
//
// Capped at 4 rather than 8 for 1-byte elements. Every group's 16-byte load is live across the
// whole load loop before the first store, so the payload alone costs `Groups * 4` registers; at 8
// that is 32 registers of data and ptxas settles the kernel at 64, which caps a 256-thread block
// at 4 blocks per SM (50% theoretical occupancy on sm_80/sm_90). At 4 it is 48 registers and 5
// blocks (62.5%), with no spills. The trade is memory-level parallelism: 4 16-byte requests per
// thread still leaves far more bytes in flight per SM than is needed to cover DRAM latency, and
// the extra resident warp slots give some of that depth back. `__launch_bounds__` cannot
// substitute for this -- the liveness is real, so budgeting below 64 at Groups == 8 only spills.
template<int MaxElemBytes>
constexpr int kBCStreamGroups = MaxElemBytes == 1 ? 4 : (MaxElemBytes == 2 ? 4 : 2);

// A segment is the whole sample when rows and planes are packed, one row otherwise.
template<bool kClamp, bool kEnableCache, int kBlockW = kBCStreamBlockW, bool kFullTiles = false, typename SrcBT,
         typename DstBT, typename ArgT, bool IsScalar>
__global__ void BrightnessContrastStream(const SrcBT *__restrict__ src, DstBT *__restrict__ dst, int srcSampleStride,
                                         int dstSampleStride, int srcSegStride, int dstSegStride, int segElems,
                                         BatchArgsWrap<ArgT, IsScalar> batchArgs)
{
    constexpr int N = BC_STREAM_N<SrcBT, DstBT>;
    constexpr int G = kBCStreamGroups<elem_stream::kCarrierBytes / N>;
    using SrcPack   = typename elem_stream::PackType<N * sizeof(SrcBT)>::type;
    using DstPack   = typename elem_stream::PackType<N * sizeof(DstBT)>::type;

    const int z = blockIdx.z;
#if __CUDA_ARCH__ >= 1000
    constexpr bool kCacheRepeated
        = kEnableCache && std::is_same_v<SrcBT, long long> && std::is_same_v<DstBT, long long>;
    constexpr bool kReuseDouble = kEnableCache && std::is_same_v<SrcBT, double> && std::is_same_v<DstBT, double>;
#else
    constexpr bool kCacheRepeated = false;
    constexpr bool kReuseDouble   = false;
#endif
    SampleArgs<ArgT, IsScalar> arg;
    if constexpr (!kCacheRepeated)
        arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);

    const SrcBT *sp = src + z * srcSampleStride + static_cast<int>(blockIdx.y) * srcSegStride;
    DstBT       *dp = dst + z * dstSampleStride + static_cast<int>(blockIdx.y) * dstSegStride;

    const int e0 = (blockIdx.x * kBlockW * G + threadIdx.x) * N;

    alignas(SrcPack) SrcBT in[G][N];
    bool                   full[G];
#pragma unroll
    for (int g = 0; g < G; ++g)
    {
        const int e = e0 + g * (kBlockW * N);
        // DECLARED HARDENING (not a refactor): written as `e <= segElems - N` rather than
        // `e + N <= segElems`. The grid covers ceil(segElems / (N * kBlockW * G)) blocks, so
        // the largest `e` any thread forms is that ceiling times N * kBlockW * G, minus N.
        // That product is a power of two no greater than 32768 and 2^31 is a multiple of it, so for
        // the largest segElems MakeElemPlan admits (INT32_MAX) the largest `e` is exactly
        // 2^31 - N -- in range -- while `e + N` is exactly 2^31, which is signed overflow. Under
        // wrapping the test reads as true and the thread issues its wide load past the end of the
        // last segment. Subtracting on the segment side instead cannot overflow (segElems >= 0 and
        // N <= 16) and is otherwise the same predicate. It needs a dense ~2 GiB single-sample
        // tensor to reach, which fits on a large device.
        full[g] = kFullTiles || e <= segElems - N;
        if (full[g])
        {
            *reinterpret_cast<SrcPack *>(in[g]) = *reinterpret_cast<const SrcPack *>(sp + e);
        }
    }

    if constexpr (kReuseDouble)
    {
        static_assert(G == 2 && N == 2);
        constexpr unsigned mask = 0xffffffffu;
        unsigned long long raw[G][N]{};
#pragma unroll
        for (int g = 0; g < G; ++g)
        {
            if (full[g])
            {
#pragma unroll
                for (int k = 0; k < N; ++k) asm("mov.b64 %0, %1;" : "=l"(raw[g][k]) : "d"(in[g][k]));
            }
        }
        const auto key0  = __shfl_sync(mask, raw[0][0], 0);
        const auto key1  = __shfl_sync(mask, raw[0][1], 0);
        bool       match = full[G - 1];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int k = 0; k < N; ++k) match &= raw[g][k] == key0 || raw[g][k] == key1;

        // Raw-bit membership preserves signed zeros. Uniform admission avoids mixed-hit divergence.
        if (__all_sync(mask, match))
        {
            const int lane   = threadIdx.x & 31;
            double    result = 0;
            if (lane < 2)
                result = ApplyBrightnessContrast<kBCStaticClamp<kClamp>, DstBT>(
                    __longlong_as_double(lane == 0 ? key0 : key1), arg);
            const double out0 = __shfl_sync(mask, result, 0);
            const double out1 = __shfl_sync(mask, result, 1);
#pragma unroll
            for (int g = 0; g < G; ++g)
            {
                alignas(DstPack) DstBT out[N];
#pragma unroll
                for (int k = 0; k < N; ++k) out[k] = raw[g][k] == key0 ? out0 : out1;
                const int e                          = e0 + g * (kBlockW * N);
                *reinterpret_cast<DstPack *>(dp + e) = *reinterpret_cast<const DstPack *>(out);
            }
            return;
        }
    }

    if constexpr (kCacheRepeated)
    {
        // Repeated image values share the same exact affine result within one sample.
        // Require three matching warps to amortize cache population and synchronization.
        if (kFullTiles || segElems >= 3 * 32 * N)
        {
            constexpr unsigned mask           = 0xffffffffu;
            const int          lane           = threadIdx.x & 31;
            const int          base           = e0 - threadIdx.x * N;
            const int          candidateIndex = base + lane;
            const SrcBT        candidate      = (kFullTiles || candidateIndex < segElems) ? sp[candidateIndex] : 0;
            const SrcBT        key0           = __shfl_sync(mask, candidate, 0);
            const unsigned     different
                = __ballot_sync(mask, (kFullTiles || candidateIndex < segElems) && candidate != key0);
            const SrcBT key1 = __shfl_sync(mask, candidate, different ? __ffs(different) - 1 : 0);

            bool allMatch = true;
            bool hasData  = false;
#pragma unroll
            for (int g = 0; g < G; ++g)
            {
                const int e = e0 + g * (kBlockW * N);
                if (full[g])
                {
                    hasData = true;
#pragma unroll
                    for (int k = 0; k < N; ++k) allMatch &= in[g][k] == key0 || in[g][k] == key1;
                }
                else if (e < segElems)
                {
                    hasData  = true;
                    in[g][0] = sp[e];
                    allMatch &= in[g][0] == key0 || in[g][0] == key1;
                }
            }
            const unsigned valid     = __ballot_sync(mask, hasData);
            const bool     warpMatch = __all_sync(mask, allMatch) && valid != 0;
            // This barrier also completes all source reads before any in-place store.
            const int      matchingWarps = __syncthreads_count(lane == 0 && warpMatch);
            if (matchingWarps >= 3)
            {
                __shared__ DstBT cached[2];
                if (threadIdx.x < 2)
                {
                    const auto cacheArg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
                    cached[threadIdx.x] = ApplyBrightnessContrast<kBCStaticClamp<kClamp>, DstBT>(
                        threadIdx.x == 0 ? key0 : key1, cacheArg);
                }
                __syncthreads();
                if (warpMatch)
                {
#pragma unroll
                    for (int g = 0; g < G; ++g)
                    {
                        const int e = e0 + g * (kBlockW * N);
                        if (full[g])
                        {
                            alignas(DstPack) DstBT out[N];
#pragma unroll
                            for (int k = 0; k < N; ++k) out[k] = cached[in[g][k] == key0 ? 0 : 1];
                            *reinterpret_cast<DstPack *>(dp + e) = *reinterpret_cast<const DstPack *>(out);
                        }
                        else if (e < segElems)
                        {
                            dp[e] = cached[in[g][0] == key0 ? 0 : 1];
                        }
                    }
                    return;
                }
            }
        }
        arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
    }

#pragma unroll
    for (int g = 0; g < G; ++g)
    {
        const int e = e0 + g * (kBlockW * N);
        if (full[g])
        {
            alignas(DstPack) DstBT out[N];
#pragma unroll
            for (int k = 0; k < N; ++k)
            {
                out[k] = ApplyBrightnessContrast<kBCStaticClamp<kClamp>, DstBT>(in[g][k], arg);
            }
            *reinterpret_cast<DstPack *>(dp + e) = *reinterpret_cast<const DstPack *>(out);
        }
        else
        {
            // At most one partial group per segment, so no separate tail kernel is needed.
            for (int i = e; i < segElems; ++i)
            {
                dp[i] = ApplyBrightnessContrast<kBCStaticClamp<kClamp>, DstBT>(sp[i], arg);
            }
        }
    }
}

// Vector pack type for T: uint3 (12B) for 3-element T, else uint4 (16B). BC_NIX = pixels/thread.
template<typename T>
using BC_DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;
template<typename T>
constexpr int BC_NIX = sizeof(BC_DPT<T>) / sizeof(T);
template<typename T>
using BC_BATCH_DPT = std::conditional_t<std::is_same_v<cuda::BaseType<T>, unsigned char> && cuda::NumElements<T> == 1,
                                        uint2, BC_DPT<T>>;
template<typename PackT>
constexpr uintptr_t BC_PACK_MSK = (sizeof(PackT) == sizeof(uint3) ? sizeof(uint) : sizeof(PackT)) - 1;
template<typename T>
constexpr uintptr_t BC_MSK = BC_PACK_MSK<BC_DPT<T>>;

// Vectorized interleaved (NHWC) brightness/contrast. Keep this path separate from planar batching so
// that the existing one-plane kernel does not pay for runtime plane selection.
template<class SrcWrapper, class DstWrapper, typename ArgT, bool IsScalar>
__global__ void BrightnessContrastVec(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT, IsScalar> batchArgs,
                                      int2 size)
{
    using SrcT             = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT             = typename DstWrapper::ValueType;
    using SrcBT            = cuda::BaseType<SrcT>;
    static constexpr int N = BC_NIX<SrcT>;

    const int z = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= size.y)
        return;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (x0 >= size.x)
        return;

    const auto  arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
    const SrcT *sp  = src.ptr(z, y, x0);
    DstT       *dp  = dst.ptr(z, y, x0);

    if (x0 + N - 1 < size.x && (reinterpret_cast<uintptr_t>(sp) & BC_MSK<SrcT>) == 0
        && (reinterpret_cast<uintptr_t>(dp) & BC_MSK<DstT>) == 0)
    {
        alignas(BC_DPT<SrcT>) SrcT in[N];
        *reinterpret_cast<BC_DPT<SrcT> *>(in) = *reinterpret_cast<const BC_DPT<SrcT> *>(sp);
        alignas(BC_DPT<DstT>) DstT out[N];
#pragma unroll
        for (int i = 0; i < N; ++i) out[i] = ApplyBrightnessContrast<BCClampMode::Runtime, DstT>(in[i], arg);
        *reinterpret_cast<BC_DPT<DstT> *>(dp) = *reinterpret_cast<const BC_DPT<DstT> *>(out);
    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (x0 + i < size.x)
                BrightnessContrastPixel(sp + i, dp + i, arg);
    }
}

// Batch adjacent planar or var-shape elements while reusing wrapper lookups and per-image arguments.
template<bool IsPlanar, bool IsVarShape, typename PackT, class SrcWrapper, class DstWrapper, typename ArgT,
         bool IsScalar>
__global__ void BrightnessContrastBatchVec(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT, IsScalar> batchArgs,
                                           int2 tensorSize, int numPlanes)
{
    using SrcT             = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT             = typename DstWrapper::ValueType;
    using SrcBT            = cuda::BaseType<SrcT>;
    static constexpr int N = sizeof(PackT) / sizeof(SrcT);

    const int z = blockIdx.z;
    int2      size{tensorSize};
    if constexpr (IsVarShape)
    {
        size = {dst.width(z), dst.height(z)};
    }
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= size.y)
        return;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (x0 >= size.x)
        return;

    const auto arg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);
    assert(IsPlanar || numPlanes == 1);
    for (int p = 0; p < numPlanes; ++p)
    {
        const SrcT *sp;
        DstT       *dp;
        if constexpr (IsPlanar)
        {
            auto coord = same_shape::GetCoordForLayout<IsPlanar>(int3{x0, y, z}, p);
            sp         = &src[coord];
            dp         = &dst[coord];
        }
        else
        {
            sp = src.ptr(z, y, x0);
            dp = dst.ptr(z, y, x0);
        }

        if (x0 + N - 1 < size.x && (reinterpret_cast<uintptr_t>(sp) & BC_PACK_MSK<PackT>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & BC_PACK_MSK<PackT>) == 0)
        {
            alignas(PackT) SrcT in[N];
            *reinterpret_cast<PackT *>(in) = *reinterpret_cast<const PackT *>(sp);
            alignas(PackT) DstT out[N];
#pragma unroll
            for (int i = 0; i < N; ++i) out[i] = ApplyBrightnessContrast<BCClampMode::Runtime, DstT>(in[i], arg);
            *reinterpret_cast<PackT *>(dp) = *reinterpret_cast<const PackT *>(out);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < N; ++i)
                if (x0 + i < size.x)
                    BrightnessContrastPixel(sp + i, dp + i, arg);
        }
    }
}

// BrightnessContrast kernel --------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename ArgT, bool IsScalar>
__global__ void BrightnessContrast(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT, IsScalar> batchArgs, int2 size,
                                   int numPlanes)
{
    using SrcBT    = cuda::BaseType<typename SrcWrapper::ValueType>;
    int  z         = blockIdx.z;
    auto sampleArg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);

    same_shape::ForEachPlane<isPlanar>(numPlanes,
                                       [=](int p) { DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, p); });
}

// VarBatch variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename ArgT, bool IsScalar>
__global__ void BrightnessContrast(SrcWrapper src, DstWrapper dst, BatchArgsWrap<ArgT, IsScalar> batchArgs,
                                   int numPlanes)
{
    using SrcBT = cuda::BaseType<typename SrcWrapper::ValueType>;
    int  z      = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};
    auto sampleArg = GetBrightnessContrastArg<SrcBT>(batchArgs, z);

    same_shape::ForEachPlane<isPlanar>(numPlanes,
                                       [=](int p) { DoBrightnessContrast<isPlanar>(src, dst, sampleArg, size, p); });
}

template<bool isPlanar, typename SrcValueT, typename DstValueT, typename ArgT, bool IsScalar>
inline bool TryRunBrightnessContrastStream(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                                           const nvcv::TensorDataStridedCuda              &dstData,
                                           const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                           const nvcv::TensorDataAccessStridedImagePlanar &dstAccess,
                                           const BatchArgsWrap<ArgT, IsScalar>            &batchArgs)
{
    using SrcBT = cuda::BaseType<SrcValueT>;
    using DstBT = cuda::BaseType<DstValueT>;

    constexpr int kN        = BC_STREAM_N<SrcBT, DstBT>;
    constexpr int kSrcPackB = kN * static_cast<int>(sizeof(SrcBT));
    constexpr int kDstPackB = kN * static_cast<int>(sizeof(DstBT));

    // Planned per side, because the two sides' element widths differ and so their byte runs do.
    // Sample collapse is deliberately not available here: the arguments are per sample and the
    // kernel reads them at grid.z, so folding samples into one run would lose the index.
    const bool collapseRows = elem_stream::RowsPacked<isPlanar, SrcValueT>(srcAccess)
                           && elem_stream::RowsPacked<isPlanar, DstValueT>(dstAccess);

    elem_stream::ElemPlan srcPlan, dstPlan;
    if (!elem_stream::MakeElemPlan<isPlanar, SrcValueT>(srcAccess, collapseRows, srcPlan)
        || !elem_stream::MakeElemPlan<isPlanar, DstValueT>(dstAccess, collapseRows, dstPlan))
    {
        return false;
    }
    // Both sides must cut the stream identically so a thread's group covers the same elements.
    if (srcPlan.runElems != dstPlan.runElems || srcPlan.numRegions != dstPlan.numRegions)
    {
        return false;
    }

    const int numSamples = static_cast<int>(srcAccess.numSamples());
    if (numSamples > elem_stream::kMaxGridYZ || srcPlan.numRegions > elem_stream::kMaxGridYZ)
    {
        return false;
    }

    const auto sideAligned = [](const void *base, const elem_stream::ElemPlan &plan, int elemSize, int packBytes)
    {
        return elem_stream::StridesAligned(base, static_cast<int64_t>(plan.sampleStride) * elemSize,
                                           static_cast<int64_t>(plan.regionStride) * elemSize, packBytes);
    };
    if (!sideAligned(srcData.basePtr(), srcPlan, sizeof(SrcBT), kSrcPackB)
        || !sideAligned(dstData.basePtr(), dstPlan, sizeof(DstBT), kDstPackB))
    {
        return false;
    }

    constexpr int kG     = kBCStreamGroups<elem_stream::kCarrierBytes / kN>;
    auto          launch = [&](auto clampTag)
    {
        auto launchCache = [&](auto cacheTag)
        {
            auto launchBlock = [&](auto blockTag, auto fullTilesTag)
            {
                constexpr int kBlockW = decltype(blockTag)::value;
                dim3          block(kBlockW, 1, 1);
                dim3          grid(util::DivUp(srcPlan.runElems, kN * kBlockW * kG),
                                   static_cast<unsigned int>(srcPlan.numRegions), static_cast<unsigned int>(numSamples));
                BrightnessContrastStream<decltype(clampTag)::value, decltype(cacheTag)::value, kBlockW,
                                         decltype(fullTilesTag)::value><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const SrcBT *>(srcData.basePtr()), reinterpret_cast<DstBT *>(dstData.basePtr()),
                    srcPlan.sampleStride, dstPlan.sampleStride, srcPlan.regionStride, dstPlan.regionStride,
                    srcPlan.runElems, batchArgs);
            };
            if constexpr (decltype(cacheTag)::value
                          && std::is_same_v<SrcBT, long long> && std::is_same_v<DstBT, long long>)
            {
                int sm = 0;
                NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
                if (sm >= 100)
                {
                    constexpr int kCacheBlockW = 128;
                    if (srcPlan.runElems % (kN * kCacheBlockW * kG) == 0)
                        launchBlock(std::integral_constant<int, kCacheBlockW>{}, std::true_type{});
                    else
                        launchBlock(std::integral_constant<int, kCacheBlockW>{}, std::false_type{});
                    return;
                }
            }
            launchBlock(std::integral_constant<int, kBCStreamBlockW>{}, std::false_type{});
        };
        if constexpr ((std::is_same_v<SrcBT, long long> && std::is_same_v<DstBT, long long>)
                      || (std::is_same_v<SrcBT, double> && std::is_same_v<DstBT, double>))
        {
            if (srcPlan.runElems >= kBCStreamCacheMinElems)
            {
                launchCache(std::true_type{});
                return;
            }
        }
        launchCache(std::false_type{});
    };
    if constexpr (IsScalar)
    {
        if (batchArgs.clamp)
        {
            launch(std::true_type{});
        }
        else
        {
            launch(std::false_type{});
        }
    }
    else
    {
        launch(std::false_type{});
    }
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

// Run BrightnessContrast kernel ----------------------------------------------------------

template<bool isPlanar, typename SrcValueT, typename DstValueT, typename ArgT, bool IsScalar, class SrcData,
         class DstData>
inline void RunBrightnessContrast(cudaStream_t stream, const SrcData &srcData, const DstData &dstData,
                                  const BatchArgsWrap<ArgT, IsScalar> &batchArgs)
{
    using SrcBT                        = cuda::BaseType<SrcValueT>;
    static constexpr int  kNix         = BC_NIX<SrcValueT>;
    static constexpr int  kBatchNix    = sizeof(BC_BATCH_DPT<SrcValueT>) / sizeof(SrcValueT);
    static constexpr int  kNumElements = cuda::NumElements<SrcValueT>;
    static constexpr bool kUseBatchedPath
        = kNix > 1 && std::is_same_v<SrcBT, unsigned char> && (kNumElements == 1 || kNumElements == 3);
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);

        // DECLARED HARDENING (not a refactor): the grid half of this preamble is new here. Every
        // tensor path below -- the stream, the vectorized kernels and the scalar kernels alike --
        // puts numSamples in grid.z, which CUDA caps at 65535, so a batch above that used to
        // surface as a raw cudaErrorInvalidConfiguration from cudaGetLastError. The other five
        // operators in this family reject the same input as ERROR_INVALID_ARGUMENT; this makes
        // BrightnessContrast agree instead of reporting an opaque CUDA error for an API-level
        // mistake. `prep.scalarGrid` is evaluated at each row-per-grid.y launch and not here, for
        // the reason TensorLaunchPrep documents.
        //
        // The int32 span half is not new, but it is not the same test either: it was
        // `max(sampleStride * numSamples) > INT32_MAX` and is now
        // same_shape::CheckTensorStrideLimit, which changed shape in both directions -- tighter
        // wherever the span inside one sample exceeds that product (a rank-3 HWC/CHW tensor, whose
        // old test was vacuous), looser wherever it does not, by up to a whole sample stride. See
        // that function for both directions and for why the looser one is sound.
        const auto prep = same_shape::MakeTensorLaunchPrep(*srcAccess, *dstAccess);
        const int2 size = prep.size;

        using StrideType = int32_t;

        if (TryRunBrightnessContrastStream<isPlanar, SrcValueT, DstValueT>(stream, srcData, dstData, *srcAccess,
                                                                           *dstAccess, batchArgs))
        {
            return;
        }

        if constexpr (!isPlanar)
        {
            auto src = cuda::CreateTensorWrapNHW<const SrcValueT, StrideType>(srcData);
            auto dst = cuda::CreateTensorWrapNHW<DstValueT, StrideType>(dstData);
            if constexpr (std::is_same_v<SrcValueT, DstValueT>)
            {
                // Vectorized interleaved path (BC_NIX pixels/thread). Same-type only so the load/store
                // pack widths match; mixed in/out types keep the scalar kernel.
                // When kNix==0 the pack type is narrower than one pixel (e.g. long long3/double3);
                // vectorization is inapplicable so we always fall back to the scalar kernel.
                if constexpr (kNix > 0)
                {
                    int sm = 0;
                    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
                    constexpr bool kIsUnsignedByte = std::is_same_v<SrcBT, unsigned char>;
                    auto           policy
                        = cvcuda::priv::BrightnessContrastTensorKernelPolicyForSM(sm, kIsUnsignedByte, kNumElements);
                    if (policy == cvcuda::priv::BrightnessContrastTensorKernelPolicy::kScalar)
                    {
                        BrightnessContrast<isPlanar>
                            <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, batchArgs, size, 1);
                    }
                    else
                    {
                        same_shape::CheckRowGridLimit(size.y, static_cast<int>(block.y));
                        dim3 gridV(util::DivUp(size.x, block.x * kNix), util::DivUp(size.y, block.y),
                                   srcAccess->numSamples());
                        BrightnessContrastVec<<<gridV, block, 0, stream>>>(src, dst, batchArgs, size);
                    }
                }
                else
                {
                    BrightnessContrast<isPlanar>
                        <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, batchArgs, size, 1);
                }
            }
            else
            {
                BrightnessContrast<isPlanar>
                    <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, batchArgs, size, 1);
            }
        }
        else
        {
            auto [src, dst] = same_shape::MakePlanarWrapPair<SrcValueT, DstValueT, StrideType>(srcData, *srcAccess,
                                                                                               dstData, *dstAccess);
            int numPlanes   = srcAccess->numPlanes();
            if constexpr (std::is_same_v<SrcValueT, DstValueT> && kUseBatchedPath)
            {
                same_shape::CheckRowGridLimit(size.y, static_cast<int>(block.y));
                dim3 gridV(util::DivUp(size.x, block.x * kBatchNix), util::DivUp(size.y, block.y),
                           srcAccess->numSamples());
                BrightnessContrastBatchVec<true, false, BC_BATCH_DPT<SrcValueT>>
                    <<<gridV, block, 0, stream>>>(src, dst, batchArgs, size, numPlanes);
            }
            else
            {
                BrightnessContrast<isPlanar>
                    <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, batchArgs, size, numPlanes);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);

        // DECLARED HARDENING (not a refactor), the var-shape half of the tensor-side grid check
        // above: every var-shape kernel here puts the image index in grid.z, so a batch above the
        // 65535 cap used to surface as a raw cudaErrorInvalidConfiguration while the other five
        // operators in this family rejected the same input as ERROR_INVALID_ARGUMENT. It has to be
        // called here rather than inherited from same_shape::ValidateSrcDstVarBatch because
        // BrightnessContrast validates its own batch (it accepts a dst dtype differing from src).
        const auto prep = same_shape::MakeVarShapeLaunchPrep(dstData);

        cuda::ImageBatchVarShapeWrap<const SrcValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<DstValueT>       dst(dstData);

        int            numPlanes = dstData.uniqueFormat().numPlanes();
        constexpr bool kPackedRGBA
            = std::is_same_v<SrcValueT,
                             DstValueT> && (std::is_same_v<SrcValueT, uchar4> || std::is_same_v<SrcValueT, half4>);
        if constexpr (!isPlanar && kPackedRGBA)
        {
            // Amortize per-image arguments without adding packed-tail overhead to narrow images.
            if (prep.maxSize.x >= static_cast<int>(block.x) * kBatchNix)
            {
                same_shape::CheckRowGridLimit(prep.maxSize.y, static_cast<int>(block.y));
                dim3 gridV(util::DivUp(prep.maxSize.x, block.x * kBatchNix), util::DivUp(prep.maxSize.y, block.y),
                           prep.gridZ);
                BrightnessContrastBatchVec<false, true, BC_BATCH_DPT<SrcValueT>>
                    <<<gridV, block, 0, stream>>>(src, dst, batchArgs, int2{}, numPlanes);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }
        if constexpr (std::is_same_v<SrcValueT, DstValueT> && kUseBatchedPath)
        {
            same_shape::CheckRowGridLimit(prep.maxSize.y, static_cast<int>(block.y));
            dim3 gridV(util::DivUp(prep.maxSize.x, block.x * kBatchNix), util::DivUp(prep.maxSize.y, block.y),
                       prep.gridZ);
            BrightnessContrastBatchVec<isPlanar, true, BC_BATCH_DPT<SrcValueT>>
                <<<gridV, block, 0, stream>>>(src, dst, batchArgs, int2{}, numPlanes);
        }
        else
        {
            BrightnessContrast<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, batchArgs, numPlanes);
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar = unsigned char;
    using llong = long long;

#define NVCV_BRIGHTNESS_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                    \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_BRIGHTNESS_RUN_TYPED(U8, uchar)
    else if NVCV_BRIGHTNESS_RUN_TYPED(S16, short)
    else if NVCV_BRIGHTNESS_RUN_TYPED(U16, ushort)
    else if NVCV_BRIGHTNESS_RUN_TYPED(S32, int)
    else if NVCV_BRIGHTNESS_RUN_TYPED(S64, llong)
    else if NVCV_BRIGHTNESS_RUN_TYPED(F32, float)
    else if NVCV_BRIGHTNESS_RUN_TYPED(F16, __half)
    else if NVCV_BRIGHTNESS_RUN_TYPED(F64, double)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input/output data types");
    }
        // clang-format on

#undef NVCV_BRIGHTNESS_RUN_TYPED
}

template<typename Cb>
inline void RunTypeSwitch(int numChannels, int numPlanes, nvcv::DataType srcType, nvcv::DataType dstType,
                          nvcv::DataType argType, const Cb &cb)
{
#define NVCV_BRIGHTNESS_RUN_CHANNELS(NUM_CHANNELS)                             \
    (numChannels == NUM_CHANNELS)                                              \
    {                                                                          \
        using SrcVal = cuda::MakeType<SrcValBase, NUM_CHANNELS>;               \
        using DstVal = cuda::MakeType<DstValBase, NUM_CHANNELS>;               \
        cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, false>{}); \
    }

    RunTypeSwitch(
        srcType,
        [&dstType, &argType, &numChannels, &numPlanes, &cb](auto dummySrcVal)
        {
            using SrcValBase = decltype(dummySrcVal);
            RunTypeSwitch(
                dstType,
                [&argType, &numChannels, &numPlanes, &cb](auto dummyDstVal)
                {
                    using DstValBase = decltype(dummyDstVal);
                    using ArgT       = GetArgType<SrcValBase, DstValBase>;
                    static_assert(std::is_same_v<ArgT, float> || std::is_same_v<ArgT, double>);
                    auto requiredArgDtype = std::is_same_v<ArgT, float> ? nvcv::TYPE_F32 : nvcv::TYPE_F64;
                    if (argType != NVCV_DATA_TYPE_NONE && argType != requiredArgDtype)
                    {
                        if (requiredArgDtype == nvcv::TYPE_F32)
                        {
                            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                  "The brightness/contrast/brightness shift/contrast center arguments "
                                                  "are expected to have float32 type.");
                        }
                        else
                        {
                            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                                  "When the input or output is a 32/64-bit integer or float64, the "
                                                  "brightness/contrast/brightness shift/contrast center arguments "
                                                  "must have double (float64) type.");
                        }
                    }
                    // clang-format off
                    if (numChannels == 1)
                    {
                        using SrcVal = cuda::MakeType<SrcValBase, 1>;
                        using DstVal = cuda::MakeType<DstValBase, 1>;
                        if (numPlanes == 1)
                        {
                            cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, false>{});
                        }
                        else
                        {
                            cb(SrcVal{}, DstVal{}, ArgT{}, std::integral_constant<bool, true>{});
                        }
                    }
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (2)
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (3)
                    else if NVCV_BRIGHTNESS_RUN_CHANNELS (4)
                    else
                    {
                        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                            "Invalid number of channels");
                    }
                    // clang-format on
                });
        });
#undef NVCV_BRIGHTNESS_RUN_CHANNELS

} // namespace

// Argument validation helpers ----------------------------------------------------

// This operator's tensor precondition check is same_shape::ValidateSrcDstTensors plus the only two
// things BrightnessContrast genuinely does differently: it may convert between dtypes, so
// src.dtype() == dst.dtype() is not required, and it caps the C axis at 4, which the rest of the
// family does not (they admit a planar tensor of any plane count at one channel per plane).
//
// It used to be a hand-maintained second copy of that function, which is why the two rejections
// that shared validator documents -- a channel-carrying dtype and a padded column stride -- each
// had to be written and fixed twice.
//
// Two deliberate consequences of calling the shared code instead of restating it:
//
//  - The channel cap is applied after the shared validator, not in the middle of it, so a tensor
//    that violates both the cap and a later shared rule now reports the shared rule first. Only
//    which message comes first changes: every one of those arms is ERROR_INVALID_ARGUMENT, and no
//    test asserts the text.
//  - The old copy's arm throwing "Images in plannar format are expected to have one channel per
//    plane" for numInterleavedChannels > 1 && numPlanes > 1 is dropped rather than carried over,
//    because on the tensor path it cannot fire: numInterleavedChannels > 1 requires a channel-last
//    layout, and TensorShapeInfoImagePlanar reports exactly one plane for every channel-last
//    layout. validateSrcDstVarBatch below keeps its own copy of that text, where the condition is
//    reachable -- an image format's numChannels is the sum over its planes.
inline same_shape::SrcDstTensorInfo ValidateSrcDstTensors(const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                                          const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData)
{
    auto info = same_shape::ValidateSrcDstTensors(srcData, dstData, /*requireSameDtype=*/false);
    if (info.numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", info.numChannels);
    }
    return info;
}

inline int GetArgNumSamples(const nvcv::Optional<nvcv::TensorDataStridedCuda> &argData)
{
    if (!argData)
    {
        return 0;
    }
    if (argData->rank() == 0)
    {
        return 1;
    }
    return argData->shape()[0];
}

template<typename ArgT>
inline BatchArgsWrap<ArgT, false> GetBatchArgsWrap(nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessData,
                                                   nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastData,
                                                   nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessShiftData,
                                                   nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastCenterData)
{
    int brightnessLen      = GetArgNumSamples(brightnessData);
    int contrastLen        = GetArgNumSamples(contrastData);
    int brightnessShiftLen = GetArgNumSamples(brightnessShiftData);
    int constrastCenterLen = GetArgNumSamples(contrastCenterData);
    return {brightnessLen,
            contrastLen,
            brightnessShiftLen,
            constrastCenterLen,
            brightnessLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*brightnessData),
            contrastLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*contrastData),
            brightnessShiftLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*brightnessShiftData),
            constrastCenterLen == 0 ? ArgWrapper<ArgT>{} : ArgWrapper<ArgT>(*contrastCenterData)};
}

template<typename ArgT>
inline BatchArgsWrap<ArgT, true> GetScalarBatchArgsWrap(double brightness, double contrast, double brightnessShift,
                                                        double contrastCenter, bool clamp, bool truncate)
{
    return {static_cast<ArgT>(brightness),
            static_cast<ArgT>(contrast),
            static_cast<ArgT>(brightnessShift),
            static_cast<ArgT>(contrastCenter),
            clamp,
            truncate};
}

inline auto validateSrcDstVarBatch(int &numSamples, int &numInterleavedChannels, int &numPlanes,
                                   nvcv::DataType &srcDtype, nvcv::DataType &dstDtype, cudaStream_t stream,
                                   const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst)
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

    numSamples = srcData->numImages();
    if (numSamples != dstData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    const auto &srcFormat = srcData->uniqueFormat();
    const auto &dstFormat = dstData->uniqueFormat();

    if (!srcFormat || !dstFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images in a batch must have the same format");
    }

    int numChannels = srcFormat.numChannels();
    if (numChannels != dstFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }
    if (numChannels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", numChannels);
    }

    numPlanes = srcFormat.numPlanes();
    if (numPlanes != dstFormat.numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    srcDtype = srcFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (srcDtype != srcFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Ale planes in the input image must have the same data type");
        }
    }

    dstDtype = dstFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dstDtype != dstFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Ale planes in the output image must have the same data type");
        }
    }

    if (numPlanes > 1 && numPlanes != srcFormat.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in plannar format are expected to have one channel per plane");
    }

    // 1. In case of varbatch images, the number of interleaved channels is always baked
    // in the vectorized data type.
    // 2. The srcFormat.numChannels() is a sum of channels number for all planes. Thus,
    // if numPlanes == 1 -> srcDtype.numChannels() == srcFormat.numChannels(), otheriwse by
    // the check above enforcing numPlanes == srcFormat.numChannels() ->
    // srcDtype.numChannels() == 1.
    numInterleavedChannels = srcDtype.numChannels();

    for (int i = 0; i < numSamples; i++)
    {
        if (src[i].size() != dst[i].size())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output must have matching width and height");
        }
    }

    return srcDstData;
}

inline void ValidateTensorArgs(nvcv::DataType &argDType, nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &brightnessShiftData,
                               nvcv::Optional<nvcv::TensorDataStridedCuda> &contrastCenterData, int numSamples,
                               const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                               const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter)
{
    auto validateArgData
        = [&argDType, &numSamples](const std::string &argName, nvcv::Optional<nvcv::TensorDataStridedCuda> &argData,
                                   const nvcv::Tensor &argTensor)
    {
        if (argTensor)
        {
            argData = argTensor.exportData<nvcv::TensorDataStridedCuda>();
            if (!argData)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must be cuda-accessible, pitch-linear tensor", argName.c_str());
            }
            if (argData->rank() > 1)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must be a scalar or 1D tensor", argName.c_str());
            }
            else if (argData->rank() == 1)
            {
                int argNumSamples = argData->shape()[0];
                if (argNumSamples != 0 && argNumSamples != 1 && argNumSamples != numSamples)
                {
                    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                          "If the %s argument is specified, it must be a scalar or 1D tensor whose "
                                          "length must match the number of input images",
                                          argName.c_str());
                }
            }
            if (argData->dtype() != nvcv::TYPE_F32 && argData->dtype() != nvcv::TYPE_F64)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "The %s argument must have float type or double for int32 input types",
                                      argName.c_str());
            }
            if (argDType == NVCV_DATA_TYPE_NONE)
            {
                argDType = argData->dtype();
            }
            else if (argDType != argData->dtype())
            {
                throw nvcv::Exception(
                    nvcv::Status::ERROR_INVALID_ARGUMENT,
                    "The brightness/contrast/brightness shift and contrast center arguments must be of the same type");
            }
        }
    };

    validateArgData("brightness", brightnessData, brightness);
    validateArgData("contrast", contrastData, contrast);
    validateArgData("brightness shift", brightnessShiftData, brightnessShift);
    validateArgData("contrast center", contrastCenterData, contrastCenter);
}

} // anonymous namespace

namespace cvcuda::priv {

// Constructor -----------------------------------------------------------------

BrightnessContrast::BrightnessContrast() {}

// Operator --------------------------------------------------------------------

// Tensor input variant
void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                    const nvcv::Tensor &brightness, const nvcv::Tensor &contrast,
                                    const nvcv::Tensor &brightnessShift, const nvcv::Tensor &contrastCenter) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[Tensor]");
    auto                                        srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto                                        dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto                                        info    = ValidateSrcDstTensors(srcData, dstData);
    nvcv::DataType                              argDType;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessShiftData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastCenterData;
    ValidateTensorArgs(argDType, brightnessData, contrastData, brightnessShiftData, contrastCenterData, info.numSamples,
                       brightness, contrast, brightnessShift, contrastCenter);
    RunTypeSwitch(info.numInterleavedChannels, info.numPlanes, info.srcDtype, info.dstDtype, argDType,
                  [&brightnessData, &contrastData, &brightnessShiftData, &contrastCenterData, &stream, &srcData,
                   &dstData](auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args      = GetBatchArgsWrap<ArgT>(brightnessData, contrastData, brightnessShiftData,
                                                         contrastCenterData);
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

// VarShape input variant
void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                    const nvcv::ImageBatchVarShape &dst, const nvcv::Tensor &brightness,
                                    const nvcv::Tensor &contrast, const nvcv::Tensor &brightnessShift,
                                    const nvcv::Tensor &contrastCenter) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[ImageBatchVarShape]");
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcDstData
        = validateSrcDstVarBatch(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, stream, src, dst);
    nvcv::DataType                              argDType;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> brightnessShiftData;
    nvcv::Optional<nvcv::TensorDataStridedCuda> contrastCenterData;
    ValidateTensorArgs(argDType, brightnessData, contrastData, brightnessShiftData, contrastCenterData, numSamples,
                       brightness, contrast, brightnessShift, contrastCenter);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, argDType,
                  [&brightnessData, &contrastData, &brightnessShiftData, &contrastCenterData, &stream, &srcDstData](
                      auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args      = GetBatchArgsWrap<ArgT>(brightnessData, contrastData, brightnessShiftData,
                                                         contrastCenterData);
                      auto &[srcData, dstData] = srcDstData;
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                    double brightness, double contrast, double brightnessShift, double contrastCenter,
                                    bool clamp, NVCVRoundMode roundMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[Tensor,Scalar]");
    adjust::ValidateRoundMode(roundMode);
    const bool truncate = roundMode == NVCV_ROUND_TRUNCATE;
    auto       srcData  = src.exportData<nvcv::TensorDataStridedCuda>();
    auto       dstData  = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto       info     = ValidateSrcDstTensors(srcData, dstData);
    RunTypeSwitch(info.numInterleavedChannels, info.numPlanes, info.srcDtype, info.dstDtype, nvcv::DataType{},
                  [&, brightness, contrast, brightnessShift, contrastCenter, clamp, truncate](
                      auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args = GetScalarBatchArgsWrap<ArgT>(brightness, contrast, brightnessShift, contrastCenter,
                                                               clamp, truncate);
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

void BrightnessContrast::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                    const nvcv::ImageBatchVarShape &dst, double brightness, double contrast,
                                    double brightnessShift, double contrastCenter, bool clamp,
                                    NVCVRoundMode roundMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::BrightnessContrast::operator()[ImageBatchVarShape,Scalar]");
    adjust::ValidateRoundMode(roundMode);
    const bool     truncate = roundMode == NVCV_ROUND_TRUNCATE;
    int            numSamples;
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype;
    auto           srcDstData
        = validateSrcDstVarBatch(numSamples, numInterleavedChannels, numPlanes, srcDtype, dstDtype, stream, src, dst);
    RunTypeSwitch(numInterleavedChannels, numPlanes, srcDtype, dstDtype, nvcv::DataType{},
                  [&, brightness, contrast, brightnessShift, contrastCenter, clamp, truncate](
                      auto dummySrcVal, auto dummyDstVal, auto dummyArg, auto isPlanar)
                  {
                      using InT      = decltype(dummySrcVal);
                      using OutT     = decltype(dummyDstVal);
                      using ArgT     = decltype(dummyArg);
                      using IsPlanar = decltype(isPlanar);
                      auto args = GetScalarBatchArgsWrap<ArgT>(brightness, contrast, brightnessShift, contrastCenter,
                                                               clamp, truncate);
                      auto &[srcData, dstData] = srcDstData;
                      RunBrightnessContrast<IsPlanar::value, InT, OutT>(stream, *srcData, *dstData, args);
                  });
}

} // namespace cvcuda::priv
