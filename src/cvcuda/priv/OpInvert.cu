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

#include "CudaDeviceUtils.hpp"
#include "InvertPolicy.hpp"
#include "Nvtx.hpp"
#include "OpInvert.hpp"

#include "ElementStreamCommon.cuh"
#include "PhotometricBound.cuh"
#include "SameShapeCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
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

#include <type_traits>

namespace cuda        = nvcv::cuda;
namespace elem_stream = cvcuda::priv::elem_stream;
namespace util        = nvcv::util;
namespace same_shape  = cvcuda::priv::same_shape;

namespace {

using cvcuda::priv::PhotometricUpperBound;

inline bool UsePackedU8C3VarShapeKernelOnCurrentDevice()
{
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
    return cvcuda::priv::UsePackedU8C3VarShapeKernelForDevice(sm);
}

template<typename BT>
inline __device__ BT InvertValue(BT value)
{
    const BT bound = PhotometricUpperBound<BT>();
    if constexpr (std::is_integral_v<BT> && std::is_signed_v<BT>)
    {
        // max - a negative input is above max; saturate before signed subtraction can overflow.
        if (value < 0)
        {
            return bound;
        }
    }
    return cuda::SaturateCast<BT>(bound - value);
}

inline __device__ double InvertDoubleStreamValue(double value)
{
#if __CUDA_ARCH__ >= 1000
    unsigned long long bits;
    asm("mov.b64 %0, %1;" : "=l"(bits) : "d"(value));
    if (bits - 0x3fe0000000000000ULL <= 0x0010000000000000ULL)
    {
        // On [0.5, 1], the exact difference is this integer remainder times 2^-53.
        const unsigned long long remainder = 0x3ff0000000000000ULL - bits;
        if (!remainder)
            return 0.;
        const unsigned leading = __clzll(remainder);
        return __longlong_as_double((static_cast<unsigned long long>(1032 - leading) << 52)
                                    + (remainder << (leading - 11)));
    }
#endif
    return InvertValue(value);
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper>
inline __device__ void DoInvert(SrcWrapper src, DstWrapper dst, const int2 size, const int p)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    // if planar then no interleaved channels
    static_assert(!IsPlanar || numChannels == 1);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    auto coord = same_shape::GetCoordForLayout<IsPlanar>(nhwCoord, p);

    const auto invertPixel = [](SrcT pixel)
    {
        DstT out;
#pragma unroll
        for (int c = 0; c < numChannels; ++c)
        {
            cuda::GetElement(out, c) = InvertValue(cuda::GetElement(pixel, c));
        }
        return out;
    };

    if constexpr (!IsPlanar && alignof(SrcT) > alignof(cuda::BaseType<SrcT>))
    {
        const SrcT *srcPtr = src.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        DstT       *dstPtr = dst.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        if ((reinterpret_cast<uintptr_t>(srcPtr) | reinterpret_cast<uintptr_t>(dstPtr)) % alignof(SrcT) != 0)
        {
            same_shape::StoreUnalignedPixel(dstPtr, invertPixel(same_shape::LoadUnalignedPixel(srcPtr)));
            return;
        }
    }
    dst[coord] = invertPixel(src[coord]);
}

// Invert kernel --------------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Invert(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes)
{
    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoInvert<isPlanar>(src, dst, size, p); });
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Invert(SrcWrapper src, DstWrapper dst, int numPlanes)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoInvert<isPlanar>(src, dst, size, p); });
}

// Vectorized planar element map ------------------------------------------------------------
//
// The scalar planar kernel above moves one element per thread per plane, so the 1-byte planar paths
// are memory-latency bound (NCU: ~82% long-scoreboard stalls, ~58-70% BWUtil) -- each warp has only a
// single outstanding load. same_shape::PlanarVec4Kernel fixes that (wide loads, NGROUP groups per
// thread, one grid.z block per (sample, plane)); this is the only Invert-specific part of it, and it
// is bit-identical to DoInvert per element (same saturating InvertValue).
template<typename BT>
struct InvertVec4Op
{
    using Param = same_shape::NoParam;

    inline __device__ explicit InvertVec4Op(Param) {}

    inline __device__ BT operator()(BT v) const
    {
        return InvertValue(v);
    }

    // Four independent U8 complements are exactly one packed word complement. Keeping the value
    // packed avoids lane extraction without changing the scalar fallback used by ragged rows.
    template<typename T = BT, std::enable_if_t<std::is_same_v<T, unsigned char>, int> = 0>
    inline __device__ uint32_t ApplyWord(uint32_t v) const
    {
        return ~v;
    }
};

// Four RGB8 pixels occupy exactly three aligned 32-bit words. Packing at that boundary avoids the
// scalar uchar3 path's excessive sectors while preserving byte-exact 255-v arithmetic.
template<int NGROUP>
__global__ void InvertU8C3VarShapePackedKernel(cuda::ImageBatchVarShapeWrap<const uchar3> src,
                                               cuda::ImageBatchVarShapeWrap<uchar3>       dst)
{
    const int g0    = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int width = dst.width(batch);

    if (g0 * 4 >= width || dst_y >= dst.height(batch))
    {
        return;
    }

    const uchar3 *srow = src.ptr(batch, dst_y, 0);
    uchar3       *drow = dst.ptr(batch, dst_y, 0);
    const bool    wide = reinterpret_cast<uintptr_t>(srow) % alignof(uint32_t) == 0
                   && reinterpret_cast<uintptr_t>(drow) % alignof(uint32_t) == 0;

    // Bound live packed words to one pixel group so register residency stays proportional to one
    // transaction rather than the compile-time batching factor.
#pragma unroll 1
    for (int i = 0; i < NGROUP; ++i)
    {
        const int cx = (g0 + i * blockDim.x) * 4;
        if (wide && cx + 4 <= width)
        {
            const uint32_t *packed = reinterpret_cast<const uint32_t *>(srow + cx);
            uint32_t       *out    = reinterpret_cast<uint32_t *>(drow + cx);
            out[0]                 = ~packed[0];
            out[1]                 = ~packed[1];
            out[2]                 = ~packed[2];
        }
        else if (cx < width)
        {
            for (int x = cx; x < min(cx + 4, width); ++x)
            {
                const uchar3 value = srow[x];
                drow[x] = uchar3{static_cast<unsigned char>(255 - value.x), static_cast<unsigned char>(255 - value.y),
                                 static_cast<unsigned char>(255 - value.z)};
            }
        }
    }
}

// Dense flat kernel -------------------------------------------------------------------------
//
// A fully dense U8 tensor (packed channels/columns/rows/planes/samples in both src and dst) is one
// contiguous byte range, so layout and channel count stop mattering: the scalar interleaved
// kernels' narrow 3-4 byte per-thread accesses (measured 65% BWUtil on uchar3 NHWC Tensor at
// locked clocks) collapse to flat 8-byte words of bitwise NOT, since 255 - v == ~v per unsigned
// lane — bit-identical to DoInvert. One word per thread won a measured geometry sweep on the
// uchar3 NHWC Tensor row (uint4 x4/thread 86.8% BWUtil, uint4 x1 87.5%, uint2 x1 87.7%,
// uint32 x1 84.9%): maximizing resident threads beats per-thread batching for pure streaming.
// The sub-word tail is finished by global thread 0. U16 and F32 dense tensors stay on the per-element kernels:
// they were measured at the 87.8-88.6% BWUtil ridge already, and this flat shape benched 1-2%
// slower there (beyond noise), so the dense path is gated to 1-byte base types.

__global__ void InvertDenseNotKernel(const uint2 *__restrict__ src, uint2 *__restrict__ dst, int64_t numWords,
                                     const unsigned char *srcTail, unsigned char *dstTail, int tailBytes)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numWords)
    {
        const uint2 v = src[i];
        dst[i]        = uint2{~v.x, ~v.y};
    }
    if (i == 0)
    {
        for (int t = 0; t < tailBytes; ++t)
        {
            dstTail[t] = static_cast<unsigned char>(~srcTail[t]);
        }
    }
}

// Returns the tensor's contiguous byte count when every axis is packed (the whole tensor is one
// byte block), or -1 when it is padded. Channel/column packing is implied by comparing rowStride
// against the packed row size computed from ValueT (interleaved) or its base type (planar
// single-channel rows). Rank-3 HWC/CHW tensors report sampleStride() == 0; their single sample is
// dense whenever rows/planes are packed.
template<bool isPlanar, typename ValueT>
inline int64_t DenseTensorBytes(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    using BT                     = cuda::BaseType<ValueT>;
    const int64_t packedRowBytes = access.numCols() * (isPlanar ? sizeof(BT) : sizeof(ValueT));
    if (access.rowStride() != packedRowBytes)
    {
        return -1;
    }
    const int64_t planeBytes = access.numRows() * packedRowBytes;
    int64_t       sampleBytes;
    if constexpr (isPlanar)
    {
        if (access.planeStride() != planeBytes)
        {
            return -1;
        }
        sampleBytes = access.numPlanes() * planeBytes;
    }
    else
    {
        sampleBytes = planeBytes;
    }
    if (access.sampleStride() != sampleBytes && !(access.sampleStride() == 0 && access.numSamples() == 1))
    {
        return -1;
    }
    return access.numSamples() * sampleBytes;
}

template<bool isPlanar, typename ValueT>
inline bool TryRunInvertDense(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                              const nvcv::TensorDataStridedCuda              &dstData,
                              const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                              const nvcv::TensorDataAccessStridedImagePlanar &dstAccess)
{
    using BT = cuda::BaseType<ValueT>;

    // Measured dtype gate: only 1-byte base types beat their per-element kernels here (see the
    // dense-kernel comment above).
    if constexpr (sizeof(BT) != 1)
    {
        return false;
    }
    else
    {
        constexpr int kWordBytes = sizeof(uint2);
        const int64_t totalBytes = DenseTensorBytes<isPlanar, ValueT>(srcAccess);
        if (reinterpret_cast<uintptr_t>(srcData.basePtr()) % kWordBytes != 0
            || reinterpret_cast<uintptr_t>(dstData.basePtr()) % kWordBytes != 0 || totalBytes < 0
            || DenseTensorBytes<isPlanar, ValueT>(dstAccess) != totalBytes)
        {
            return false;
        }
        const int64_t numWords  = totalBytes / kWordBytes;
        const int     tailBytes = static_cast<int>(totalBytes - numWords * kWordBytes);

        constexpr int kBlock  = 512;
        const int64_t threads = std::max<int64_t>(numWords, 1);
        dim3          grid(static_cast<unsigned int>(util::DivUp(threads, static_cast<int64_t>(kBlock))));

        const auto *src = reinterpret_cast<const uint2 *>(srcData.basePtr());
        auto       *dst = reinterpret_cast<uint2 *>(dstData.basePtr());
        InvertDenseNotKernel<<<grid, kBlock, 0, stream>>>(
            src, dst, numWords, reinterpret_cast<const unsigned char *>(srcData.basePtr()) + numWords * kWordBytes,
            reinterpret_cast<unsigned char *>(dstData.basePtr()) + numWords * kWordBytes, tailBytes);
        NVCV_CHECK_THROW(cudaGetLastError());
        return true;
    }
}

// Element stream. Allow-listed by base type, not width: F16 must not be complemented as integers.
constexpr int kInvertStreamGrps = 2;

template<typename BT, typename Enable = void>
struct InvertStreamCarrier : elem_stream::DecliningCarrier
{
};

template<typename BT>
struct InvertStreamCarrier<BT,
                           std::enable_if_t<std::is_same_v<BT, unsigned char> || std::is_same_v<BT, unsigned short>>>
{
    static constexpr bool kSupported = true;
    using type                       = uint4;
    using Param                      = elem_stream::NoParam;

    static Param MakeParam()
    {
        return {};
    }

    inline static __device__ type Apply(type v, Param)
    {
        return type{~v.x, ~v.y, ~v.z, ~v.w};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param)
    {
        for (int b = 0; b < bytes; ++b)
        {
            d[b] = static_cast<unsigned char>(~s[b]);
        }
    }
};

template<typename BT>
struct InvertStreamCarrier<BT, std::enable_if_t<std::is_same_v<BT, short>>>
{
    static constexpr bool kSupported = true;
    using type                       = uint4;
    using Param                      = elem_stream::NoParam;

    static Param MakeParam()
    {
        return {};
    }

    inline static __device__ uint32_t ApplyWord(uint32_t v)
    {
        // Expand each lane's sign bit without a cross-lane borrow. Negative lanes
        // saturate to 32767; nonnegative lanes retain their low-15-bit complement.
        const uint32_t sign = v & 0x80008000u;
        const uint32_t mask = sign - (sign >> 15);
        return (~v | mask) & 0x7fff7fffu;
    }

    inline static __device__ type Apply(type v, Param)
    {
        return type{ApplyWord(v.x), ApplyWord(v.y), ApplyWord(v.z), ApplyWord(v.w)};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param)
    {
        elem_stream::TailElementwise<BT>(s, d, bytes, [](BT v) { return InvertValue(v); });
    }
};

template<typename BT>
struct InvertStreamCarrier<BT, std::enable_if_t<std::is_same_v<BT, float>>>
{
    static constexpr bool kSupported = true;
    using type                       = float4;
    using Param                      = elem_stream::NoParam;

    static Param MakeParam()
    {
        return {};
    }

    inline static __device__ type Apply(type v, Param)
    {
        const BT bound = PhotometricUpperBound<BT>();
        return type{cuda::SaturateCast<BT>(bound - v.x), cuda::SaturateCast<BT>(bound - v.y),
                    cuda::SaturateCast<BT>(bound - v.z), cuda::SaturateCast<BT>(bound - v.w)};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param)
    {
        const BT bound = PhotometricUpperBound<BT>();
        elem_stream::TailElementwise<BT>(s, d, bytes, [bound](BT v) { return cuda::SaturateCast<BT>(bound - v); });
    }
};

template<typename BT>
struct InvertStreamCarrier<BT, std::enable_if_t<std::is_same_v<BT, double>>>
{
    static constexpr bool kSupported = true;
    using type                       = double2;
    using Param                      = elem_stream::NoParam;

    static Param MakeParam()
    {
        return {};
    }

    inline static __device__ type Apply(type v, Param)
    {
        return type{InvertDoubleStreamValue(v.x), InvertDoubleStreamValue(v.y)};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param)
    {
        elem_stream::TailElementwise<BT>(s, d, bytes, [](BT v) { return InvertDoubleStreamValue(v); });
    }
};

// Run Invert kernel ----------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunInvert(cudaStream_t stream, const SrcData &srcData, const DstData &dstData)
{
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto       srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto       dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        const auto prep      = same_shape::MakeTensorLaunchPrep(*srcAccess, *dstAccess);
        const int2 size      = prep.size;

        using StrideType = int32_t;

        // 4-byte planar is the one arm the stream does not improve; it defers to below.
        constexpr bool kStreamFirst = !(isPlanar && sizeof(cuda::BaseType<ValueT>) == 4);
        if constexpr (kStreamFirst)
        {
            if (elem_stream::TryRunCarrier<kInvertStreamGrps, InvertStreamCarrier, isPlanar, ValueT>(
                    stream, srcData, dstData, *srcAccess, *dstAccess))
            {
                return;
            }
        }

        // Fully dense src+dst collapse to one contiguous byte range, which the flat 8-byte-word
        // kernel above walks. Two ways dense work reaches it:
        //   (a) elem_stream::IsCarrierAligned declined, because a base pointer or an uncollapsed
        //       sample stride is a multiple of 8 but not of 16, which an 8-byte word does not care
        //       about. A single-sample tensor is the common case: MakePlan's sample collapse
        //       requires numSamples > 1, so its sampleStride stays at the (unused) sample size and
        //       is tested for 16-byte alignment even though nothing indexes with it.
        //   (b) MakePlan's final `runBytes <= INT32_MAX` test declined. That is reachable even
        //       though MakeTensorLaunchPrep ran first, because CheckTensorStrideLimit bounds the
        //       largest offset a wrap can form (totalBytes - colStride), not the byte count: a
        //       dense single-sample tensor of exactly 2^31 bytes has a span bound of
        //       2^31 - colStride <= INT32_MAX, passes the validator, and then fails MakePlan.
        // MakePlan's remaining exits cannot leave dense work here -- an oversized batch is already
        // thrown by MakeTensorLaunchPrep above, and an unpacked row or plane stride makes
        // DenseTensorBytes return -1 too. Falls through to the layout-specific kernels otherwise.
        if (TryRunInvertDense<isPlanar, ValueT>(stream, srcData, dstData, *srcAccess, *dstAccess))
        {
            return;
        }

        if constexpr (!isPlanar)
        {
            using BT         = cuda::BaseType<ValueT>;
            bool launchedVec = false;

            // 1-channel interleaved (e.g. U16) is byte-identical to a single plane and moves only
            // sizeof(BT) bytes/thread, so it is latency-bound (NCU: ~18 cyc long-scoreboard, ~77% BW)
            // like the planar paths. Reuse the vectorized planar kernel (C == 1). The multi-channel
            // interleaved cases (uchar3/uchar4/float3/float4) already move 3-4 bytes/thread and sit at
            // the bandwidth ridge, so they keep the scalar kernel untouched.
            // F16 is excluded: the wide path's lane type is cuda::MakeType<BT, 4>, so a 2-byte BT
            // gives ushort4, whose lanes would feed raw half bit patterns into `bound - in` as integers.
            // Falling back to the scalar kernel keeps real half arithmetic; a dedicated half
            // wide-load path is a future optimization.
            if constexpr (cuda::NumElements<ValueT> == 1 && !cuda::detail::IsHalfV<BT>)
            {
                constexpr int NGROUP = 4;
                launchedVec          = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, InvertVec4Op<BT>>(
                    stream, false, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block,
                    same_shape::NoParam{});
            }

            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Invert<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, 1);
            }
        }
        else
        {
            using BT              = cuda::BaseType<ValueT>;
            const int numPlanes   = srcAccess->numPlanes();
            bool      launchedVec = false;

            // Vectorized planar path for 1-byte / 4-byte planes: map each (sample, plane) to grid.z and
            // move 4 columns/thread with a wide load (batched NGROUP loads -> high memory-level
            // parallelism), eliminating the scalar planar kernel's long-scoreboard stalls. Requires
            // sizeof(Vec4)-aligned base + strides; falls back to the scalar kernel otherwise (the
            // kernel's per-thread tail handles a width that is not a multiple of 4).
            if constexpr (sizeof(BT) == 1 || sizeof(BT) == 4)
            {
                constexpr int NGROUP = sizeof(BT) == 1 ? 4 : 2;
                launchedVec          = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, InvertVec4Op<BT>>(
                    stream, true, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block,
                    same_shape::NoParam{});
            }

            if (!launchedVec && !kStreamFirst
                && elem_stream::TryRunCarrier<kInvertStreamGrps, InvertStreamCarrier, isPlanar, ValueT>(
                    stream, srcData, dstData, *srcAccess, *dstAccess))
            {
                return;
            }

            if (!launchedVec)
            {
                auto [src, dst] = same_shape::MakePlanarWrapPair<ValueT, ValueT, StrideType>(srcData, *srcAccess,
                                                                                             dstData, *dstAccess);
                Invert<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, numPlanes);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        const auto prep = same_shape::MakeVarShapeLaunchPrep(dstData);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        if constexpr (!isPlanar && std::is_same_v<ValueT, uchar3>)
        {
            if (UsePackedU8C3VarShapeKernelOnCurrentDevice())
            {
                constexpr int NGROUP = 4;
                // Rows map onto grid.y here too, and no var-shape path below divides rows any more
                // coarsely, so there is nothing to decline to: bound it by throwing.
                same_shape::CheckRowGridLimit(prep.maxSize.y, static_cast<int>(block.y));
                dim3 vgrid(util::DivUp(util::DivUp(prep.maxSize.x, 4), static_cast<int>(block.x) * NGROUP),
                           util::DivUp(prep.maxSize.y, block.y), prep.gridZ);
                cuda::ImageBatchVarShapeWrap<const uchar3> srcV(srcData);
                cuda::ImageBatchVarShapeWrap<uchar3>       dstV(dstData);
                InvertU8C3VarShapePackedKernel<NGROUP><<<vgrid, block, 0, stream>>>(srcV, dstV);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        // Preserve the established 1-byte planar kernel; it needs cross-group ILP to stay at ridge.
        // Collapsing this path into the register-bounded U16 kernel above regresses the reference
        // A100/H100 planar RGB8 workload, so the two var-shape wide paths stay distinct.
        using BT = cuda::BaseType<ValueT>;
        if constexpr (isPlanar && sizeof(BT) == 1)
        {
            constexpr int NGROUP = 4;
            if (same_shape::TryRunVec4VarShapeFor<NGROUP, BT, InvertVec4Op<BT>>(
                    stream, srcData, dstData, numPlanes, prep.maxSize, block, same_shape::NoParam{}))
            {
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        // The U16 interleaved path keeps one vector group live at a time to cap register residency,
        // which is what `Serial` selects in the shared kernel. F16 is excluded like in the tensor
        // path: the ushort4 lanes would treat half bit patterns as integers, so it falls back to
        // the scalar kernel's real half arithmetic.
        if constexpr (!isPlanar && cuda::NumElements<ValueT> == 1 && sizeof(BT) == 2 && !cuda::detail::IsHalfV<BT>)
        {
            constexpr int NGROUP = 4;
            // Of the two caps inside TryRunVec4VarShape only the row one can decline here:
            // numPlanes is 1 and CheckVarShapeGridLimit above has already rejected an image count
            // over 65535, so grid.z is in range, but a batch whose tallest image needs more than
            // 65535 block-rows falls through to the scalar kernel, which rejects it.
            if (same_shape::TryRunVec4VarShapeFor<NGROUP, BT, InvertVec4Op<BT>, /*Serial=*/true>(
                    stream, srcData, dstData, 1, prep.maxSize, block, same_shape::NoParam{}))
            {
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Invert<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, numPlanes);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

} // anonymous namespace

namespace cvcuda::priv {

Invert::Invert() {}

// Tensor input variant
void Invert::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Invert::operator()[Tensor]");
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto info    = same_shape::ValidateSrcDstTensors(srcData, dstData);

    same_shape::DispatchWideTypeAndChannels(info.numInterleavedChannels, info.numPlanes, info.srcDtype, "Invert",
                                            [&stream, &srcData, &dstData](auto dummyVal, auto isPlanar)
                                            {
                                                using ValueT   = decltype(dummyVal);
                                                using IsPlanar = decltype(isPlanar);
                                                RunInvert<IsPlanar::value, ValueT>(stream, *srcData, *dstData);
                                            });
}

// VarShape input variant
void Invert::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                        const nvcv::ImageBatchVarShape &dst) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Invert::operator()[ImageBatchVarShape]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto srcDstData = same_shape::ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    same_shape::DispatchWideTypeAndChannels(numInterleavedChannels, numPlanes, dtype, "Invert",
                                            [&stream, &srcDstData](auto dummyVal, auto isPlanar)
                                            {
                                                using ValueT             = decltype(dummyVal);
                                                using IsPlanar           = decltype(isPlanar);
                                                auto &[srcData, dstData] = srcDstData;
                                                RunInvert<IsPlanar::value, ValueT>(stream, *srcData, *dstData);
                                            });
}

} // namespace cvcuda::priv
