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
#include "OpSolarize.hpp"

#include "ElementStreamCommon.cuh"
#include "PhotometricBound.cuh"
#include "SameShapeCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace cuda        = nvcv::cuda;
namespace elem_stream = cvcuda::priv::elem_stream;
namespace util        = nvcv::util;
namespace same_shape  = cvcuda::priv::same_shape;

namespace {

using cvcuda::priv::PhotometricUpperBound;

// Threshold comparison hoisted out of the FP64 pipe. The gold semantics are
// `static_cast<double>(v) >= threshold`; per element that runs on the (1/64-rate on consumer SM)
// double pipe and makes Solarize compute-bound (NCU: ~87% SM, ~26% DRAM, decoupled-math stalls).
// For the discrete supported value types this is *exactly* equivalent to comparing v against a
// precomputed value-domain boundary in the integer / float32 domain, so the result stays bit-identical
// to the double-domain gold reference while the per-element FP64 disappears:
//   - integer v:  (double)v >= threshold  <=>  (long long)v >= ceil(threshold)
//   - float32 v:  (double)v >= threshold  <=>  v >= vmin, vmin = smallest float >= threshold
// The element stream and scalar half kernels resolve the boundary on the host. Other paths
// compute it in the thread before comparing the pixel components.
// __half takes the float branch: every half value is exactly representable as float, so the
// float32 boundary equivalence above holds for it too.
// The integer branch would instead compare the value truncated to long long — wrong semantics.
template<typename BT>
struct SolarizeThreshold
{
    long long vmin_i = 0;
    float     vmin_f = 0.0f;

    // Host-callable too: the element stream resolves the boundary once, off the FP64 pipe.
    inline __host__ __device__ explicit SolarizeThreshold(double threshold)
    {
        if constexpr (cuda::detail::IsFloatingPointV<BT>)
        {
            float f = static_cast<float>(threshold);
            if (static_cast<double>(f) < threshold) // round up to the smallest float >= threshold
            {
                f = ::nextafterf(f, 3.402823466e38f);
            }
            vmin_f = f;
        }
        else if (!(threshold == threshold)) // NaN, spelled so it compiles on host and device
        {
            // (v >= NaN) is always false; use the max sentinel so no integer value inverts,
            // and avoid the undefined behavior of casting a NaN to long long.
            vmin_i = 9223372036854775807LL;
        }
        else
        {
            const double c = ::ceil(threshold);
            vmin_i         = c >= 9.2233720368547758e18
                               ? 9223372036854775807LL
                               : (c <= -9.2233720368547758e18 ? (-9223372036854775807LL - 1) : static_cast<long long>(c));
        }
    }

    inline __device__ bool ge(BT v) const
    {
        if constexpr (cuda::detail::IsFloatingPointV<BT>)
        {
            // __half widens to float losslessly, so the compare runs in float precision for F16
            // exactly as it does for float32 (the cast is an identity for float v).
            return static_cast<float>(v) >= vmin_f;
        }
        else
        {
            return static_cast<long long>(v) >= vmin_i;
        }
    }
};

template<>
struct SolarizeThreshold<double>
{
    unsigned long long key;

    inline __host__ __device__ static unsigned long long Bits(double v)
    {
#ifdef __CUDA_ARCH__
        unsigned long long bits;
        // Prevent NVCC from rematerializing the integer masks and zero tests as FP64 operations.
        asm("mov.b64 %0, %1;" : "=l"(bits) : "d"(v));
        return bits;
#else
        unsigned long long bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
#endif
    }

    inline __host__ __device__ static unsigned long long Ordered(unsigned long long bits)
    {
        // IEEE ordering reverses negative encodings; both zeros must compare equal.
        if ((bits & 0x7fffffffffffffffULL) == 0)
            bits = 0;
        return bits & 0x8000000000000000ULL ? ~bits : bits ^ 0x8000000000000000ULL;
    }

    inline __host__ __device__ explicit SolarizeThreshold(double threshold)
    {
        const auto bits = Bits(threshold);
        key             = (bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL ? ~0ULL : Ordered(bits);
    }

    inline __device__ bool ge(double v) const
    {
        const auto bits = Bits(v);
        return (bits & 0x7fffffffffffffffULL) <= 0x7ff0000000000000ULL && Ordered(bits) >= key;
    }
};

// out = (in >= threshold) ? (bound - in) : in, per channel component (threshold hoisted, see above).
template<typename BT>
inline __device__ BT SolarizeApply(BT v, BT bound, const SolarizeThreshold<BT> &th)
{
#if __CUDA_ARCH__ >= 1000
    if constexpr (std::is_same_v<BT, double>)
    {
        // For thresholds in [0.5, 1], selected values below one share exponent -1.
        // Their difference from one is an exact integer remainder times 2^-53.
        if (th.key - 0xbfe0000000000000ULL <= 0x0010000000000000ULL)
        {
            const auto bits = SolarizeThreshold<double>::Bits(v);
            if (bits < (th.key ^ 0x8000000000000000ULL) || bits > 0x7ff0000000000000ULL)
                return v;
            if (bits > 0x3ff0000000000000ULL)
                return bound - v;
            const unsigned long long remainder = 0x3ff0000000000000ULL - bits;
            if (!remainder)
                return 0.;
            const unsigned leading = __clzll(remainder);
            return __longlong_as_double((static_cast<unsigned long long>(1032 - leading) << 52)
                                        + (remainder << (leading - 11)));
        }
    }
#endif
    return th.ge(v) ? static_cast<BT>(bound - v) : v;
}

// Scalar per-pixel path (multi-channel interleaved + planar fallback). For multi-channel INTEGER
// pixels the FP64 threshold compares dominate (uchar3/uchar4 NHWC are ~36% BWUtil, FP64-bound), so
// hoist the boundary once per pixel (one ceil amortized over 3-4 integer compares = net win). For
// 1-channel integers (hoist would add a ceil with nothing to amortize) and float keep the original
// double compare. Half pixels receive the exact float-domain boundary from the host, avoiding
// per-pixel FP64 setup.
template<typename T, typename Threshold>
inline __device__ T SolarizeElem(T pixel, Threshold threshold)
{
    using BT                         = cuda::BaseType<T>;
    static constexpr int numChannels = cuda::NumElements<T>;
    const BT             bound       = PhotometricUpperBound<BT>();

    T out{};
    if constexpr (cuda::detail::IsHalfV<
                      BT> || std::is_same_v<BT, double> || (numChannels > 1 && !std::is_floating_point_v<BT>))
    {
        const SolarizeThreshold<BT> th(threshold);
#pragma unroll
        for (int c = 0; c < numChannels; ++c)
        {
            cuda::GetElement(out, c) = SolarizeApply<BT>(cuda::GetElement(pixel, c), bound, th);
        }
    }
    else
    {
#pragma unroll
        for (int c = 0; c < numChannels; ++c)
        {
            const BT v               = cuda::GetElement(pixel, c);
            cuda::GetElement(out, c) = (static_cast<double>(v) >= threshold) ? static_cast<BT>(bound - v) : v;
        }
    }
    return out;
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper, class Threshold>
inline __device__ void DoSolarize(SrcWrapper src, DstWrapper dst, const int2 size, const int p, Threshold threshold)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    static_assert(!IsPlanar || numChannels == 1);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    auto coord = same_shape::GetCoordForLayout<IsPlanar>(nhwCoord, p);
    if constexpr (!IsPlanar && alignof(SrcT) > alignof(cuda::BaseType<SrcT>))
    {
        const SrcT *srcPtr = src.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        DstT       *dstPtr = dst.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        if (reinterpret_cast<uintptr_t>(srcPtr) % alignof(SrcT) != 0
            || reinterpret_cast<uintptr_t>(dstPtr) % alignof(DstT) != 0)
        {
            same_shape::StoreUnalignedPixel(dstPtr,
                                            SolarizeElem<DstT>(same_shape::LoadUnalignedPixel(srcPtr), threshold));
            return;
        }
    }
    dst[coord] = SolarizeElem<DstT>(src[coord], threshold);
}

// Solarize kernel ------------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, class Threshold>
__global__ void Solarize(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, Threshold threshold)
{
    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoSolarize<isPlanar>(src, dst, size, p, threshold); });
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, class Threshold>
__global__ void Solarize(SrcWrapper src, DstWrapper dst, int numPlanes, Threshold threshold)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoSolarize<isPlanar>(src, dst, size, p, threshold); });
}

// Vectorized planar / 1-channel element map ----------------------------------------------
//
// The scalar kernels above move one element/thread/plane, leaving the small-dtype planar and
// 1-channel-interleaved paths memory-latency bound (long-scoreboard stalls, low BWUtil).
// same_shape::PlanarVec4Kernel fixes that (wide loads, NGROUP groups per thread, one grid.z block per
// (sample, plane)); this is the only Solarize-specific part of it, and it is bit-identical to
// SolarizeElem per element (same (in>=threshold)?(bound-in):in).
//
// The threshold boundary is derived from the raw double here, inside the device constructor the shared
// kernel calls once per thread, rather than being passed in ready-made: that is where the FP64 ceil ran
// before, and moving it to the host would change what the kernel computes even though the value is the
// same.
// Double uses its own 128-bit stream carrier and exact ordered-encoding comparison.
template<typename BT>
struct SolarizeVec4Op
{
    static_assert(!std::is_same_v<BT, double>, "F64 Solarize uses its double2 stream carrier");

    using Param = double;
    BT                    bound;
    SolarizeThreshold<BT> th;

    inline __device__ explicit SolarizeVec4Op(Param threshold)
        : bound(PhotometricUpperBound<BT>())
        , th(threshold)
    {
    }

    inline __device__ BT operator()(BT v) const
    {
        return SolarizeApply<BT>(v, bound, th);
    }
};

// Element stream. Deeper than the other operators; measured in context, not in a standalone stream.
constexpr int kSolarizeStreamGrps = 4;

// Allow-listed by base type, not width: the integer carrier needs (max - v) == ~v, false for __half.
template<typename BT, typename Enable = void>
struct SolarizeStreamCarrier : elem_stream::DecliningCarrier
{
};

// (bound - v) == ~v, so the select collapses to an XOR with the __vcmpgeu4/2 compare mask.
template<typename BT>
struct SolarizeStreamCarrier<BT,
                             std::enable_if_t<std::is_same_v<BT, unsigned char> || std::is_same_v<BT, unsigned short>>>
{
    static constexpr bool kSupported = true;
    using type                       = uint4;

    struct Param
    {
        uint32_t boundary; // threshold replicated over the lanes of a 32-bit word
        uint32_t enable;   // 0 when no representable value reaches the threshold, else all ones
    };

    static Param MakeParam(double threshold)
    {
        const SolarizeThreshold<BT> th(threshold);
        constexpr long long         maxV = static_cast<long long>(cuda::TypeTraits<BT>::max);

        Param p{};
        if (th.vmin_i > maxV)
        {
            // Nothing inverts, and no boundary expresses that for an unsigned compare; suppress.
            p.boundary = 0u;
            p.enable   = 0u;
        }
        else
        {
            const uint32_t v = static_cast<uint32_t>(th.vmin_i < 0 ? 0 : th.vmin_i);
            p.boundary       = elem_stream::BroadcastToWord(static_cast<BT>(v));
            p.enable         = 0xFFFFFFFFu;
        }
        return p;
    }

    inline static __device__ uint32_t ApplyWord(uint32_t w, Param p)
    {
        if constexpr (sizeof(BT) == 1)
        {
            return w ^ (__vcmpgeu4(w, p.boundary) & p.enable);
        }
        else
        {
            return w ^ (__vcmpgeu2(w, p.boundary) & p.enable);
        }
    }

    inline static __device__ type Apply(type v, Param p)
    {
        return type{ApplyWord(v.x, p), ApplyWord(v.y, p), ApplyWord(v.z, p), ApplyWord(v.w, p)};
    }

    // The one place in this family's unification where the device code is not identical to what the
    // hand-written loop emitted. The narrow of p.boundary is not the cause -- the open-coded loop
    // hoisted it too, and the kernel has a single tail call site. What changed is that the loop
    // body is now a lambda over two locals instead of a loop reading Param directly, and nvcc
    // folds the resulting per-element selects: sm_80 PTX for
    // StreamKernel<4, SolarizeStreamCarrier<unsigned char>> goes and 44 -> 40, cvt 14 -> 10, and for
    // <unsigned short> and 39 -> 38, cvt 7 -> 3. Same values, strictly less work. Bisected to this
    // body alone: restoring the open-coded loop restores byte-identical PTX, while reverting only
    // the `enable` hoist below leaves the delta in place, so the hoist is not what causes it.
    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param p)
    {
        const BT       t      = elem_stream::NarrowFromWord<BT>(p.boundary);
        const uint32_t enable = p.enable;
        elem_stream::TailElementwise<BT>(
            s, d, bytes, [enable, t](BT v) { return (enable != 0u && v >= t) ? static_cast<BT>(~v) : v; });
    }
};

template<typename BT>
struct SolarizeStreamCarrier<BT, std::enable_if_t<std::is_same_v<BT, float>>>
{
    static constexpr bool kSupported = true;
    using type                       = float4;

    struct Param
    {
        float vmin;
    };

    static Param MakeParam(double threshold)
    {
        const SolarizeThreshold<BT> th(threshold);
        return Param{th.vmin_f};
    }

    inline static __device__ BT One(BT v, Param p)
    {
        const BT kBound = PhotometricUpperBound<BT>();
        return (v >= p.vmin) ? static_cast<BT>(kBound - v) : v;
    }

    inline static __device__ type Apply(type v, Param p)
    {
        return type{One(v.x, p), One(v.y, p), One(v.z, p), One(v.w, p)};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param p)
    {
        elem_stream::TailElementwise<BT>(s, d, bytes, [p](BT v) { return One(v, p); });
    }
};

template<>
struct SolarizeStreamCarrier<double>
{
    static constexpr bool kSupported = true;
    using type                       = double2;
    using Param                      = SolarizeThreshold<double>;

    static Param MakeParam(double threshold)
    {
        return Param{threshold};
    }

    inline static __device__ double One(double v, Param p)
    {
        return SolarizeApply(v, 1., p);
    }

    inline static __device__ type Apply(type v, Param p)
    {
        return type{One(v.x, p), One(v.y, p)};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param p)
    {
        elem_stream::TailElementwise<double>(s, d, bytes, [p](double v) { return One(v, p); });
    }
};

// Run Solarize kernel --------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunSolarize(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, double threshold)
{
    dim3       block(32, 4, 1);
    const auto scalarThreshold = [threshold]
    {
        using BT = cuda::BaseType<ValueT>;
        if constexpr (cuda::detail::IsHalfV<BT>)
            return SolarizeThreshold<BT>(threshold);
        else
            return threshold;
    }();
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto       srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto       dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        const auto prep      = same_shape::MakeTensorLaunchPrep(*srcAccess, *dstAccess);
        const int2 size      = prep.size;

        using StrideType = int32_t;

        if (elem_stream::TryRunCarrier<kSolarizeStreamGrps, SolarizeStreamCarrier, isPlanar, ValueT>(
                stream, srcData, dstData, *srcAccess, *dstAccess, threshold))
        {
            return;
        }

        using BT = cuda::BaseType<ValueT>;
        if constexpr (!isPlanar)
        {
            bool launchedVec = false;
            // 1-channel interleaved (e.g. U16) is byte-identical to a single plane and latency-bound;
            // route it through the vectorized planar kernel (C == 1). Multi-channel interleaved
            // (uchar3/uchar4/float3/float4) already moves 3-4 B/thread at the ridge -> keep scalar.
            // F16 is excluded: the wide path's lane type is cuda::MakeType<BT, 4>, so a 2-byte BT
            // gives ushort4, whose lanes would feed raw half bit patterns into SolarizeApply as integers.
            // F64 has a double2 stream carrier above; an unaligned stream uses the scalar fallback.
            if constexpr (cuda::NumElements<ValueT> == 1 && !cuda::detail::IsHalfV<BT> && !std::is_same_v<BT, double>)
            {
                constexpr int NGROUP = 4;
                launchedVec          = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, SolarizeVec4Op<BT>>(
                    stream, false, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block,
                    threshold);
            }
            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Solarize<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, 1, scalarThreshold);
            }
        }
        else
        {
            const int numPlanes   = srcAccess->numPlanes();
            bool      launchedVec = false;
            if constexpr (sizeof(BT) == 1 || sizeof(BT) == 4)
            {
                constexpr int NGROUP = sizeof(BT) == 1 ? 4 : 2;
                launchedVec          = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, SolarizeVec4Op<BT>>(
                    stream, true, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block, threshold);
            }
            if (!launchedVec)
            {
                auto [src, dst] = same_shape::MakePlanarWrapPair<ValueT, ValueT, StrideType>(srcData, *srcAccess,
                                                                                             dstData, *dstAccess);
                Solarize<isPlanar>
                    <<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, numPlanes, scalarThreshold);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        const auto prep = same_shape::MakeVarShapeLaunchPrep(dstData);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        // Vectorized planar var-shape path for 1-byte planes (uchar4 wide load, gated per row by the
        // kernel's own alignment test). float var-shape stays scalar because it is already at the
        // bandwidth ridge.
        //
        // u16 var-shape also stays scalar, but not because the wide load is unsafe: the shared
        // kernel's per-row test is `alignof(Vec4)`, which is 8 for ushort4, and Invert's u16
        // var-shape path already runs on it. It stays scalar only for want of a measurement.
        using BT = cuda::BaseType<ValueT>;
        if constexpr (isPlanar && sizeof(BT) == 1)
        {
            constexpr int NGROUP = 4;
            if (same_shape::TryRunVec4VarShapeFor<NGROUP, BT, SolarizeVec4Op<BT>>(stream, srcData, dstData, numPlanes,
                                                                                  prep.maxSize, block, threshold))
            {
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Solarize<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, numPlanes, scalarThreshold);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

} // anonymous namespace

namespace cvcuda::priv {

Solarize::Solarize() {}

// Tensor input variant
void Solarize::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, double threshold) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Solarize::operator()[Tensor]");
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto info    = same_shape::ValidateSrcDstTensors(srcData, dstData);

    same_shape::DispatchTypeWithF64AndChannels(info.numInterleavedChannels, info.numPlanes, info.srcDtype, "Solarize",
                                               [&stream, &srcData, &dstData, threshold](auto dummyVal, auto isPlanar)
                                               {
                                                   using ValueT   = decltype(dummyVal);
                                                   using IsPlanar = decltype(isPlanar);
                                                   RunSolarize<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                        threshold);
                                               });
}

// VarShape input variant
void Solarize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                          double threshold) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Solarize::operator()[ImageBatchVarShape]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto srcDstData = same_shape::ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    same_shape::DispatchTypeWithF64AndChannels(numInterleavedChannels, numPlanes, dtype, "Solarize",
                                               [&stream, &srcDstData, threshold](auto dummyVal, auto isPlanar)
                                               {
                                                   using ValueT             = decltype(dummyVal);
                                                   using IsPlanar           = decltype(isPlanar);
                                                   auto &[srcData, dstData] = srcDstData;
                                                   RunSolarize<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                        threshold);
                                               });
}

} // namespace cvcuda::priv
