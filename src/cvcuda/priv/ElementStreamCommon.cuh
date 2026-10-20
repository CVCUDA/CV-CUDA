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

#ifndef CVCUDA_PRIV_ELEMENT_STREAM_COMMON_CUH
#define CVCUDA_PRIV_ELEMENT_STREAM_COMMON_CUH

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

// Moves a pointwise operator's bytes in fixed-width carriers, not pixels: access width is independent of dtype and
// layout. A carrier declares the type it loads (`type`), optionally a different type it stores (`dst_type`, when the
// operator's destination element is a different width), an `Apply` and a `Tail`. Everything else -- describing the
// layout as byte runs, choosing the block shape, and the launch -- belongs here.
namespace cvcuda::priv::elem_stream {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

inline constexpr int kCarrierBytes = 16; // vector width; carriers may contain fewer bytes or several vectors
inline constexpr int kBlockThreads = 512;
inline constexpr int kMaxGridYZ    = 65535; // grid.y and grid.z are both 16-bit

struct NoParam
{
};

// Whole-period carriers keep byte-to-state indices compile-time constant without a runtime phase.
template<int NVec>
struct WideVec
{
    uint4 v[NVec];
};

template<int NVec>
using CarrierVec = std::conditional_t<NVec == 1, uint4, WideVec<NVec>>;

// All threads call MakeState before the framework barrier, allowing cooperative shared-memory fills.
template<class Carrier, class = void>
struct CarrierTraits
{
    using State                        = NoParam;
    static constexpr bool kStateful    = false;
    static constexpr int  kSharedBytes = 0;
    static constexpr bool kSplitPlanes = false;
};

template<class Carrier>
struct CarrierTraits<Carrier, std::void_t<typename Carrier::State>>
{
    using State                        = typename Carrier::State;
    static constexpr bool kStateful    = true;
    static constexpr int  kSharedBytes = Carrier::kSharedBytes;
    static constexpr bool kSplitPlanes = Carrier::kSplitPlanes;
};

inline __device__ int FlatThreadId()
{
    return static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
}

inline __device__ int FlatBlockThreads()
{
    return static_cast<int>(blockDim.x * blockDim.y);
}

// Per-device atomics prevent host-thread races; relaxed ordering suffices because no other state is published.
inline int MultiProcessorCount()
{
    constexpr int           kMaxDevices = 32;
    constexpr int           kFallback   = 32;
    static std::atomic<int> cache[kMaxDevices]{};

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices)
    {
        return kFallback;
    }
    int cached = cache[dev].load(std::memory_order_relaxed);
    if (cached == 0)
    {
        int sms = 0;
        cached  = (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && sms > 0)
                    ? sms
                    : kFallback;
        cache[dev].store(cached, std::memory_order_relaxed);
    }
    return cached;
}

// Zero means unknown, so failed queries retain the general path instead of selecting an architecture fast path.
inline int DeviceComputeCapability()
{
    constexpr int           kMaxDevices = 32;
    static std::atomic<int> cache[kMaxDevices]{};

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices)
    {
        return 0;
    }
    int cached = cache[dev].load(std::memory_order_relaxed);
    if (cached == 0)
    {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess
            && cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && major > 0)
        {
            cached = major * 10 + minor;
            cache[dev].store(cached, std::memory_order_relaxed);
        }
    }
    return cached;
}

inline int64_t DeviceBlockBudget(int blocksPerSM, int64_t regions)
{
    return std::max<int64_t>(static_cast<int64_t>(MultiProcessorCount()) * blocksPerSM / std::max<int64_t>(regions, 1),
                             1);
}

// Where a carrier sits in the stream. A carrier that declares `kPositionAware = true` receives this as a third
// argument to Apply and Tail; that is what lets an operator whose result depends on an element's channel index --
// per-channel normalization parameters, a channel permutation -- run on a carrier that is not a whole number of
// pixels wide. Carriers that do not declare it keep the two-argument form.
struct Position
{
    int32_t carrier; // index of this carrier inside its run
    int32_t region;  // index of the run inside its sample
    int32_t sample;
};

// Primary template for an operator's carrier: declines every base type. MakeParam must exist because
// TryRun forms its argument before kSupported is tested.
struct DecliningCarrier
{
    static constexpr bool kSupported = false;
    using type                       = uint4;
    using Param                      = NoParam;

    template<typename... Args>
    static Param MakeParam(Args...)
    {
        return {};
    }
};

// Carrier traits are detected rather than required, so a carrier only spells out what it needs.
template<class Carrier, class = void>
struct DstCarrierOf
{
    using type = typename Carrier::type;
};

template<class Carrier>
struct DstCarrierOf<Carrier, std::void_t<typename Carrier::dst_type>>
{
    using type = typename Carrier::dst_type;
};

template<class Carrier>
using DstCarrier = typename DstCarrierOf<Carrier>::type;

template<class Carrier, class = void>
struct IsPositionAware : std::false_type
{
};

template<class Carrier>
struct IsPositionAware<Carrier, std::void_t<decltype(Carrier::kPositionAware)>>
    : std::bool_constant<Carrier::kPositionAware>
{
};

// A carrier that declares a `Context` gets `MakeContext(param, position)` called once per thread, before any load,
// and receives the result in place of the Param. That is where a parameter set read from device memory belongs:
// resolving it inside Apply would reload it for every carrier the thread owns, because a parameter pointer is not
// known not to alias the destination. A carrier that does not declare one receives its Param unchanged.
template<class Carrier, class = void>
struct ContextOf
{
    using type = typename Carrier::Param;
};

template<class Carrier>
struct ContextOf<Carrier, std::void_t<typename Carrier::Context>>
{
    using type = typename Carrier::Context;
};

template<class Carrier>
using CarrierContext = typename ContextOf<Carrier>::type;

// Exact, so the low lane recovers the original value.
template<typename BT>
inline uint32_t BroadcastToWord(BT v)
{
    static_assert(sizeof(BT) == 1 || sizeof(BT) == 2);
    return sizeof(BT) == 1 ? static_cast<uint32_t>(v) * 0x01010101u : static_cast<uint32_t>(v) * 0x00010001u;
}

// Named inverse of BroadcastToWord, so a carrier that recovers the scalar from the broadcast word
// depends on that exactness by name instead of silently.
template<typename BT>
inline __device__ BT NarrowFromWord(uint32_t w)
{
    static_assert(sizeof(BT) == 1 || sizeof(BT) == 2);
    return static_cast<BT>(w);
}

// Shared scaffold for an element-typed Carrier::Tail. A trailing partial element is dropped: a run
// whose length is not a whole number of elements cannot be mapped elementwise, and every caller's
// run length is a multiple of sizeof(BT) by construction.
template<typename BT, class F>
inline __device__ void TailElementwise(const unsigned char *s, unsigned char *d, int bytes, F f)
{
    const BT *se = reinterpret_cast<const BT *>(s);
    BT       *de = reinterpret_cast<BT *>(d);
    for (int e = 0; e < bytes / static_cast<int>(sizeof(BT)); ++e)
    {
        de[e] = f(se[e]);
    }
}

// `numRegions` runs of `runBytes`, region r of sample n at `n * sampleStride + r * regionStride`.
// One tensor's byte runs: `numRegions` runs of `runBytes`, region r of sample n at
// `n * sampleStride + r * regionStride`. Planned one side at a time, because an operator whose src
// and dst element widths differ moves different byte counts for the same element and so cannot
// describe both sides with a single run length.
struct SidePlan
{
    int64_t regionStride;
    int64_t sampleStride;
    int64_t runBytes;
    int64_t numRegions; // 64-bit: the caller bounds it before its own narrowing cast
    int64_t regionsPerPlane;
};

// `numRegions` runs of `runBytes`, region r of sample n at `n * sampleStride + r * regionStride`.
// Channel index of the first element of carrier `carrier`, for a carrier holding `Elems` elements of a
// `Channels`-channel interleaved image whose runs start on a pixel boundary. Folds to zero whenever a carrier spans a
// whole number of pixels, which is every channel count except 3.
template<int Elems, int Channels>
inline __device__ int ChannelPhase(int32_t carrier)
{
    if constexpr (Elems % Channels == 0)
    {
        return 0;
    }
    else
    {
        // Reduced before multiplying: `carrier * Elems` overflows int32 on a long run.
        return static_cast<int>(((carrier % Channels) * (Elems % Channels)) % Channels);
    }
}

// `out[k] = in[(k + phase) % C]`. Both loops are fully unrolled, so every index is a compile-time constant and the
// array stays in registers; `phase` only selects among them. A runtime index would spill it to local memory.
template<int C, typename T>
inline __device__ void RotateLanes(const T (&in)[C], int phase, T (&out)[C])
{
#pragma unroll
    for (int k = 0; k < C; ++k)
    {
        T v = in[k];
#pragma unroll
        for (int j = 1; j < C; ++j)
        {
            v = (phase == j) ? in[(k + j) % C] : v;
        }
        out[k] = v;
    }
}

// `numRegions` runs of `runBytes` source bytes (`dstRunBytes` destination bytes), region r of sample n at
// `n * sampleStride + r * regionStride`. The two run lengths differ only when the operator's destination element is a
// different width from its source element; they always hold the same number of elements.
struct Plan
{
    int64_t srcRegionStride, dstRegionStride;
    int64_t srcSampleStride, dstSampleStride;
    int64_t runBytes, dstRunBytes;
    int     numRegions;
    int     numSamples;
    int     regionsPerPlane;
};

template<bool IsPlanar, typename ValueT>
inline int64_t RowBytes(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    using BT = cuda::BaseType<ValueT>;
    return access.numCols() * static_cast<int64_t>(IsPlanar ? sizeof(BT) : sizeof(ValueT));
}

// True when rows tile the sample with no gap, so a whole plane collapses to one run.
template<bool IsPlanar, typename ValueT>
inline bool RowsPacked(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return access.rowStride() == RowBytes<IsPlanar, ValueT>(access);
}

// Runtime-width core of MakeSidePlan: the element width and the layout are arguments, so an
// operator whose channel count is only known at run time describes the same bytes without a
// template instantiation per count. False when planes do not tile the sample uniformly, which is
// what makes a layout inexpressible as byte runs at all. `collapseRows` is the caller's decision
// rather than this side's: every side that has to agree on a run length must cut the stream the
// same way, so a caller passes it only when all of them are packed.
inline bool MakeSidePlanBytes(const nvcv::TensorDataAccessStridedImagePlanar &access, int64_t elemBytes,
                              bool collapseRows, bool isPlanar, SidePlan &out, bool split = false,
                              bool splitRuns = false)
{
    const int64_t rowBytes  = access.numCols() * elemBytes;
    const int64_t rowStride = access.rowStride();
    const int64_t numRows   = access.numRows();
    const int64_t numPlanes = isPlanar ? access.numPlanes() : 1;

    if (isPlanar && splitRuns)
    {
        out.sampleStride    = access.sampleStride();
        out.runBytes        = numRows * rowBytes;
        out.numRegions      = numPlanes;
        out.regionStride    = access.planeStride();
        out.regionsPerPlane = 1;
        return true;
    }
    if (isPlanar && access.planeStride() != numRows * rowStride)
    {
        return false;
    }

    out.sampleStride = access.sampleStride();

    if (collapseRows)
    {
        out.runBytes        = numPlanes * numRows * rowBytes;
        out.numRegions      = 1;
        out.regionStride    = 0;
        out.regionsPerPlane = 1;
    }
    else
    {
        // numRegions drives grid.y, capped at 65535 too, and is left in 64 bits for the caller to
        // bound before its narrowing cast: `numRows * numPlanes` is a 64-bit product of two int32
        // counts, so truncating first can land back inside the 65535 window and let a plan through
        // that describes fewer regions than the tensor has. That needs a planar tensor of at least
        // 2^32 rows-times-planes -- ~137 GiB of src plus dst at the 16-byte-aligned minimum row
        // stride this path also requires -- so it is out of reach of a single 80 GiB device today,
        // but it is a truncation, not a range check, and it costs nothing to do in 64 bits.
        out.runBytes        = rowBytes;
        out.numRegions      = numRows * numPlanes;
        out.regionStride    = rowStride;
        out.regionsPerPlane = split ? numRows : numRows * numPlanes;
    }
    return true;
}

template<bool IsPlanar, typename ValueT>
inline bool MakeSidePlan(const nvcv::TensorDataAccessStridedImagePlanar &access, bool collapseRows, SidePlan &out,
                         bool split = false, bool splitRuns = false)
{
    return MakeSidePlanBytes(access, IsPlanar ? sizeof(cuda::BaseType<ValueT>) : sizeof(ValueT), collapseRows, IsPlanar,
                             out, split, splitRuns);
}

// False when the layout is not describable as byte runs, or exceeds the kernel's 32-bit offsets. The element widths
// are arguments rather than template parameters because an operator whose channel count is only known at run time
// would otherwise instantiate the whole stream once per channel count to describe the same bytes.
inline bool MakePlanBytes(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int64_t srcElemBytes,
                          int64_t dstElemBytes, bool isPlanar, Plan &out, bool splitPlanes = false,
                          bool keepSamples = false)
{
    // A row is only a byte run when consecutive columns are adjacent. Every nvcv-allocated tensor
    // lays them out that way; an externally wrapped buffer need not, and its row would otherwise be
    // read as if it were packed.
    if (srcAccess.colStride() != srcElemBytes || dstAccess.colStride() != dstElemBytes)
    {
        return false;
    }

    // Rows collapse only when both sides are packed. One packed side alone would cut the two
    // streams into different run lengths, and the kernel walks them at a single index. With
    // asymmetric element widths each side packs against its own row length.
    const int64_t srcRowBytes  = srcAccess.numCols() * srcElemBytes;
    const int64_t dstRowBytes  = dstAccess.numCols() * dstElemBytes;
    const bool    rowsPacked   = srcAccess.rowStride() == srcRowBytes && dstAccess.rowStride() == dstRowBytes;
    const bool    split        = splitPlanes && isPlanar;
    const bool    splitRuns    = split && rowsPacked;
    const bool    collapseRows = rowsPacked && !split;

    SidePlan s, d;
    if (!MakeSidePlanBytes(srcAccess, srcElemBytes, collapseRows, isPlanar, s, split, splitRuns)
        || !MakeSidePlanBytes(dstAccess, dstElemBytes, collapseRows, isPlanar, d, split, splitRuns))
    {
        return false;
    }

    if (s.numRegions > kMaxGridYZ)
    {
        return false;
    }

    // Different byte widths must still describe the same elements and plane partition.
    if (s.runBytes / srcElemBytes != d.runBytes / dstElemBytes || s.numRegions != d.numRegions
        || s.regionsPerPlane != d.regionsPerPlane)
    {
        return false;
    }
    out.runBytes        = s.runBytes;
    out.dstRunBytes     = d.runBytes;
    out.numRegions      = static_cast<int>(s.numRegions);
    out.srcRegionStride = s.regionStride;
    out.dstRegionStride = d.regionStride;
    out.srcSampleStride = s.sampleStride;
    out.dstSampleStride = d.sampleStride;
    out.regionsPerPlane = static_cast<int>(s.regionsPerPlane);

    // Kept in 64 bits until the very end of this function: it is what grid.z will be, CUDA caps
    // grid.z at 65535, and `out.numSamples` is an int, so both the range test and the narrowing
    // cast have to happen after the collapse block below may have driven this down to 1.
    int64_t numSamples = srcAccess.numSamples();

    // Collapsing packed samples drops the per-sample offset, keeping a ragged row on the wide path.
    if (!keepSamples && out.numRegions == 1 && numSamples > 1 && out.srcSampleStride == out.runBytes
        && out.dstSampleStride == out.dstRunBytes && out.runBytes * numSamples <= cuda::TypeTraits<int32_t>::max
        && out.dstRunBytes * numSamples <= cuda::TypeTraits<int32_t>::max)
    {
        out.runBytes *= numSamples;
        out.dstRunBytes *= numSamples;
        numSamples          = 1;
        out.srcSampleStride = 0;
        out.dstSampleStride = 0;
    }

    // numSamples is grid.z, which CUDA caps at 65535, so it is bounded in 64 bits before the
    // narrowing cast below -- neither an oversized nor a truncated batch can reach the launch.
    // The test sits after the collapse deliberately: a fully dense batch of more than 65535
    // samples collapses to numSamples == 1 and launches a legal grid, and testing before the
    // collapse would decline that plan and push legal work onto a slower path.
    //
    // Current callers reject oversized batches before entry; retain this guard for callers without that check.
    if (numSamples > kMaxGridYZ)
    {
        return false;
    }
    out.numSamples = static_cast<int>(numSamples);

    // The two grid dimensions are bounded above, each before its own narrowing cast; what is left
    // is the kernel's 32-bit run offsets.
    return out.runBytes <= cuda::TypeTraits<int32_t>::max && out.dstRunBytes <= cuda::TypeTraits<int32_t>::max;
}

template<bool IsPlanar, typename SrcValueT, typename DstValueT = SrcValueT>
inline bool MakePlan(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                     const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, Plan &out, bool splitPlanes = false,
                     bool keepSamples = false)
{
    return MakePlanBytes(srcAccess, dstAccess, IsPlanar ? sizeof(cuda::BaseType<SrcValueT>) : sizeof(SrcValueT),
                         IsPlanar ? sizeof(cuda::BaseType<DstValueT>) : sizeof(DstValueT), IsPlanar, out, splitPlanes,
                         keepSamples);
}

// `numRegions` runs of `runElems`, for a kernel that indexes in elements rather than bytes -- which
// is what an operator whose src and dst element widths differ has to do, since the two sides then
// move different byte counts for the same element index.
struct ElemPlan
{
    int runElems;
    int numRegions;
    int regionStride;
    int sampleStride;
};

// False for everything MakeSidePlan declines, plus anything past the kernel's 32-bit element
// offsets. Strides are whole numbers of elements: nvcv rejects any other tensor at construction.
template<bool IsPlanar, typename ValueT>
inline bool MakeElemPlan(const nvcv::TensorDataAccessStridedImagePlanar &access, bool collapseRows, ElemPlan &out)
{
    using BT                = cuda::BaseType<ValueT>;
    constexpr int64_t kElem = sizeof(BT);

    SidePlan p;
    if (!MakeSidePlan<IsPlanar, ValueT>(access, collapseRows, p))
    {
        return false;
    }

    // Both are bounded here, before the narrowing cast, rather than after it: truncating first can
    // land a too-large value back inside the valid window and describe fewer elements or regions
    // than the tensor has.
    //
    // The runElems bound is reachable, and same_shape::MakeTensorLaunchPrep does not subsume it:
    // CheckTensorStrideLimit bounds the largest offset a wrap can form, which for a dense
    // single-sample tensor is totalBytes - colStride, not totalBytes. A dense NHWC U8 tensor of
    // exactly 2^31 bytes (say {1, 32768, 65536, 1}) therefore has a span bound of INT32_MAX,
    // passes that check, and arrives here with runElems == 2^31. Declining is correct: the
    // narrower kernels serve it with int32 offsets that do fit. This is the same edge OpInvert.cu's
    // dense-path comment describes for MakePlan's runBytes test. The numRegions bound is the latent
    // one -- reaching it needs numRows * numPlanes above 2^31.
    constexpr int64_t kMaxInt  = cuda::TypeTraits<int32_t>::max;
    const int64_t     runElems = p.runBytes / kElem;
    if (runElems > kMaxInt || p.numRegions > kMaxInt)
    {
        return false;
    }

    out.runElems     = static_cast<int>(runElems);
    out.numRegions   = static_cast<int>(p.numRegions);
    out.regionStride = static_cast<int>(p.regionStride / kElem);
    out.sampleStride = static_cast<int>(p.sampleStride / kElem);
    return true;
}

// Every run start a thread addresses sits on a `packBytes` boundary.
inline bool StridesAligned(const void *base, int64_t sampleStride, int64_t regionStride, int64_t packBytes)
{
    return reinterpret_cast<uintptr_t>(base) % packBytes == 0 && sampleStride % packBytes == 0
        && regionStride % packBytes == 0;
}

inline bool IsPlanAligned(const void *srcBase, const void *dstBase, const Plan &plan, int srcAlign, int dstAlign)
{
    const auto s = reinterpret_cast<uintptr_t>(srcBase);
    const auto d = reinterpret_cast<uintptr_t>(dstBase);
    return s % srcAlign == 0 && d % dstAlign == 0 && plan.srcSampleStride % srcAlign == 0
        && plan.dstSampleStride % dstAlign == 0 && plan.srcRegionStride % srcAlign == 0
        && plan.dstRegionStride % dstAlign == 0;
}

inline bool IsCarrierAligned(const void *srcBase, const void *dstBase, const Plan &plan)
{
    return IsPlanAligned(srcBase, dstBase, plan, kCarrierBytes, kCarrierBytes);
}

// Byte width to an unsigned pack type of exactly that size.
template<int Bytes>
struct PackType
{
    using type = cuda::MakeType<unsigned int, Bytes / 4>;
};

template<>
struct PackType<1>
{
    using type = unsigned char;
};

template<>
struct PackType<2>
{
    using type = unsigned short;
};

template<>
struct PackType<4>
{
    using type = unsigned int;
};

// Widest width whose idle slots stay under an eighth of the run, so a short row still fills a block.
template<int Groups>
inline int ChooseBlockWidth(int32_t runCarriers)
{
    const auto wastedSlots = [&](int w)
    {
        const int64_t perBlock = static_cast<int64_t>(w) * Groups;
        return util::DivUp<int64_t>(runCarriers, perBlock) * perBlock - runCarriers;
    };
    for (int w = kBlockThreads; w >= 32; w /= 2)
    {
        if (wastedSlots(w) * 8 <= runCarriers)
        {
            return w;
        }
        if (w == 32 || wastedSlots(w / 2) > wastedSlots(w))
        {
            return w;
        }
    }
    return 32;
}

template<typename Carrier>
inline __device__ CarrierContext<Carrier> MakeCarrierContext(const typename Carrier::Param &param, Position pos)
{
    if constexpr (std::is_same_v<CarrierContext<Carrier>, typename Carrier::Param>)
    {
        return param;
    }
    else
    {
        return Carrier::MakeContext(param, pos);
    }
}

template<typename Carrier, typename SrcVec>
inline __device__ DstCarrier<Carrier> ApplyCarrier(SrcVec v, const CarrierContext<Carrier> &ctx, Position pos)
{
    if constexpr (IsPositionAware<Carrier>::value)
    {
        return Carrier::Apply(v, ctx, pos);
    }
    else
    {
        return Carrier::Apply(v, ctx);
    }
}

template<typename Carrier>
inline __device__ void TailCarrier(const unsigned char *s, unsigned char *d, int srcBytes,
                                   const CarrierContext<Carrier> &ctx, Position pos)
{
    if constexpr (IsPositionAware<Carrier>::value)
    {
        Carrier::Tail(s, d, srcBytes, ctx, pos);
    }
    else
    {
        Carrier::Tail(s, d, srcBytes, ctx);
    }
}

// Every byte of a run reaches Apply or Tail once, in carrier groups from the run's start.
template<int Groups, typename Carrier>
__global__ void StreamKernel(const unsigned char *__restrict__ srcBase, unsigned char *__restrict__ dstBase,
                             int32_t runCarriers, int32_t runTailBytes, int32_t numRegions, int32_t regionsPerPlane,
                             int64_t srcRegionStride, int64_t srcSampleStride, int64_t dstRegionStride,
                             int64_t dstSampleStride, typename Carrier::Param param)
{
    using SrcVec = typename Carrier::type;
    using DstVec = DstCarrier<Carrier>;
    using Traits = CarrierTraits<Carrier>;

    const int32_t region = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);

    // Stateful blocks cover one valid region with uniform (sample, plane); all threads must reach the barrier.
    typename Traits::State state{};
    if constexpr (Traits::kStateful)
    {
        __shared__ __align__(16) unsigned char shared[Traits::kSharedBytes > 0 ? Traits::kSharedBytes : 1];
        state = Carrier::MakeState(param, static_cast<int>(blockIdx.z),
                                   regionsPerPlane > 0 ? region / regionsPerPlane : 0, shared);
        __syncthreads();
    }

    if (region >= numRegions)
    {
        return;
    }

    const int64_t sOff
        = static_cast<int64_t>(blockIdx.z) * srcSampleStride + static_cast<int64_t>(region) * srcRegionStride;
    const int64_t dOff
        = static_cast<int64_t>(blockIdx.z) * dstSampleStride + static_cast<int64_t>(region) * dstRegionStride;

    const SrcVec *s = reinterpret_cast<const SrcVec *>(srcBase + sOff);
    DstVec       *d = reinterpret_cast<DstVec *>(dstBase + dOff);

    const int32_t sweep = static_cast<int32_t>(gridDim.x * (blockDim.x * Groups));
    const int32_t first = static_cast<int32_t>(blockIdx.x * (blockDim.x * Groups) + threadIdx.x);

    const Position                threadPos{first, region, static_cast<int32_t>(blockIdx.z)};
    const CarrierContext<Carrier> ctx = MakeCarrierContext<Carrier>(param, threadPos);

    for (int32_t i0 = first; !Traits::kStateful || i0 < runCarriers; i0 += sweep)
    {
        // Every load issues before any store, so a warp keeps `Groups` requests outstanding.
        SrcVec v[Groups];
        bool   live[Groups];
#pragma unroll
        for (int g = 0; g < Groups; ++g)
        {
            const int32_t i = i0 + g * static_cast<int32_t>(blockDim.x);
            live[g]         = i < runCarriers;
            if (live[g])
            {
                v[g] = s[i];
            }
        }
#pragma unroll
        for (int g = 0; g < Groups; ++g)
        {
            const int32_t i = i0 + g * static_cast<int32_t>(blockDim.x);
            if (live[g])
            {
                if constexpr (Traits::kStateful)
                {
                    d[i] = Carrier::Apply(v[g], param, state);
                }
                else
                {
                    d[i] = ApplyCarrier<Carrier>(v[g], ctx, Position{i, region, static_cast<int32_t>(blockIdx.z)});
                }
            }
        }

        if constexpr (!Traits::kStateful)
        {
            break;
        }
    }

    if (runTailBytes > 0 && blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int64_t sTail = static_cast<int64_t>(runCarriers) * static_cast<int64_t>(sizeof(SrcVec));
        const int64_t dTail = static_cast<int64_t>(runCarriers) * static_cast<int64_t>(sizeof(DstVec));
        if constexpr (Traits::kStateful)
        {
            Carrier::Tail(srcBase + sOff + sTail, dstBase + dOff + dTail, runTailBytes, param, state);
        }
        else
        {
            // Rebuilt at the tail's own position rather than reusing `ctx`: the tail sits at carrier
            // `runCarriers` while `ctx` was built at this thread's first carrier, so a carrier whose
            // `MakeContext` reads the position would otherwise resolve the wrong phase or parameter
            // slice. No carrier does today, and for a position-independent one this folds away.
            const Position                tailPos{runCarriers, region, static_cast<int32_t>(blockIdx.z)};
            const CarrierContext<Carrier> tailCtx = MakeCarrierContext<Carrier>(param, tailPos);
            TailCarrier<Carrier>(srcBase + sOff + sTail, dstBase + dOff + dTail, runTailBytes, tailCtx, tailPos);
        }
    }
}

// Launches a plan the caller has already built and alignment-checked.
template<int Groups, typename Carrier>
inline void LaunchPlan(cudaStream_t stream, const void *srcBase, void *dstBase, const Plan &plan,
                       typename Carrier::Param param, int maxBlocksX = 0)
{
    using Traits                  = CarrierTraits<Carrier>;
    constexpr int32_t kSrcCarrier = static_cast<int32_t>(sizeof(typename Carrier::type));

    const int32_t runCarriers = static_cast<int32_t>(plan.runBytes / kSrcCarrier);
    const int32_t runTail     = static_cast<int32_t>(plan.runBytes - static_cast<int64_t>(runCarriers) * kSrcCarrier);

    const int blockW  = Traits::kStateful ? 256 : ChooseBlockWidth<Groups>(runCarriers);
    const int blockH  = Traits::kStateful ? 1 : std::min<int>(kBlockThreads / blockW, std::max(plan.numRegions, 1));
    int64_t   blocksX = std::max<int64_t>(util::DivUp<int64_t>(runCarriers, static_cast<int64_t>(blockW) * Groups), 1);
    if constexpr (Traits::kStateful)
    {
        if (maxBlocksX > 0)
        {
            blocksX = std::min<int64_t>(blocksX, maxBlocksX);
        }
    }

    dim3 block(static_cast<unsigned int>(blockW), static_cast<unsigned int>(blockH), 1);
    dim3 grid(static_cast<unsigned int>(blocksX), static_cast<unsigned int>(util::DivUp(plan.numRegions, blockH)),
              static_cast<unsigned int>(plan.numSamples));

    StreamKernel<Groups, Carrier><<<grid, block, 0, stream>>>(
        reinterpret_cast<const unsigned char *>(srcBase), reinterpret_cast<unsigned char *>(dstBase), runCarriers,
        runTail, plan.numRegions, plan.regionsPerPlane, plan.srcRegionStride, plan.srcSampleStride,
        plan.dstRegionStride, plan.dstSampleStride, param);
    NVCV_CHECK_THROW(cudaGetLastError());
}

// False without launching when the carrier, layout or alignment gate declines.
template<int Groups, typename Carrier>
inline bool TryRunBytes(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                        const nvcv::TensorDataStridedCuda              &dstData,
                        const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                        const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int64_t srcElemBytes,
                        int64_t dstElemBytes, bool isPlanar, typename Carrier::Param param, int maxBlocksX = 0)
{
    if constexpr (!Carrier::kSupported)
    {
        return false;
    }
    else
    {
        using Traits = CarrierTraits<Carrier>;
        Plan plan;
        if (!MakePlanBytes(srcAccess, dstAccess, srcElemBytes, dstElemBytes, isPlanar, plan, Traits::kSplitPlanes,
                           Traits::kStateful))
        {
            return false;
        }

        if (!IsPlanAligned(srcData.basePtr(), dstData.basePtr(), plan,
                           static_cast<int>(alignof(typename Carrier::type)),
                           static_cast<int>(alignof(DstCarrier<Carrier>))))
        {
            return false;
        }

        LaunchPlan<Groups, Carrier>(stream, srcData.basePtr(), dstData.basePtr(), plan, param, maxBlocksX);
        return true;
    }
}

template<int Groups, typename Carrier, bool IsPlanar, typename ValueT, typename DstValueT = ValueT>
inline bool TryRun(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                   const nvcv::TensorDataStridedCuda              &dstData,
                   const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                   const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, typename Carrier::Param param,
                   int maxBlocksX = 0)
{
    return TryRunBytes<Groups, Carrier>(
        stream, srcData, dstData, srcAccess, dstAccess, IsPlanar ? sizeof(cuda::BaseType<ValueT>) : sizeof(ValueT),
        IsPlanar ? sizeof(cuda::BaseType<DstValueT>) : sizeof(DstValueT), IsPlanar, param, maxBlocksX);
}

// Boilerplate an operator would otherwise repeat: resolve the carrier for the base type and build
// its Param from the operator's own arguments. Host-side only; the launch is still TryRun's.
template<int Groups, template<class, class> class CarrierTpl, bool IsPlanar, typename ValueT, typename... Args>
inline bool TryRunCarrier(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                          const nvcv::TensorDataStridedCuda              &dstData,
                          const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, Args &&...makeParamArgs)
{
    using Carrier = CarrierTpl<cuda::BaseType<ValueT>, void>;
    return TryRun<Groups, Carrier, IsPlanar, ValueT>(stream, srcData, dstData, srcAccess, dstAccess,
                                                     Carrier::MakeParam(std::forward<Args>(makeParamArgs)...));
}

__host__ __device__ inline constexpr int LutTableIndex(int kTables, int i, int byte)
{
    return (((kTables == 1) ? 0 : (i % kTables)) << 8) | byte;
}

// Exhaustive byte tables are bit-exact only when Entry uses the same expression as the per-element path.
template<class Table>
struct LutCarrier
{
    static constexpr int  kTables      = Table::kTables;
    static constexpr int  kVecs        = (kCarrierBytes % kTables == 0) ? 1 : kTables;
    static constexpr int  kBytes       = kVecs * kCarrierBytes;
    static constexpr bool kSupported   = true;
    static constexpr int  kSharedBytes = kTables * 256;
    static constexpr bool kSplitPlanes = true;

    using type  = CarrierVec<kVecs>;
    using Param = typename Table::Param;

    struct State
    {
        const unsigned char *lut;
    };

    inline static __device__ State MakeState(const Param &p, int sample, int plane, void *shared)
    {
        unsigned char *lut     = static_cast<unsigned char *>(shared);
        const int      threads = FlatBlockThreads();
        for (int t = FlatThreadId(); t < kTables * 256; t += threads)
        {
            lut[t] = Table::Entry(p, sample, plane, t >> 8, t & 0xFF);
        }
        return State{lut};
    }

    inline static __device__ type Apply(type v, const Param &, const State &st)
    {
        unsigned char b[kBytes];
        memcpy(b, &v, kBytes);
#pragma unroll
        for (int i = 0; i < kBytes; ++i)
        {
            b[i] = st.lut[LutTableIndex(kTables, i, b[i])];
        }
        memcpy(&v, b, kBytes);
        return v;
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, const Param &,
                                       const State &st)
    {
        for (int i = 0; i < bytes; ++i)
        {
            d[i] = st.lut[LutTableIndex(kTables, i, s[i])];
        }
    }
};

// Group bodies, tails and per-pixel fallbacks must share Op::Channels to preserve the expression tree.

inline constexpr int kGroupBlockW = 256;

// Cap launch bounds per device pass: unsatisfiable bounds fail ptxas under -Werror all-warnings.
// Per-architecture caps preserve occupancy on wider targets; validate with sm_XX SASS, not compute_XX-only PTX.
// The host pass emits no launch and uses the smallest cap.
__host__ __device__ inline constexpr int MaxThreadsPerSM()
{
#if defined(__CUDA_ARCH__)
    return (__CUDA_ARCH__ < 800)   ? 1024
         : (__CUDA_ARCH__ == 800)  ? 2048
         : (__CUDA_ARCH__ < 900)   ? 1536
         : (__CUDA_ARCH__ == 900)  ? 2048
         : (__CUDA_ARCH__ == 1000) ? 2048
         : (__CUDA_ARCH__ == 1200) ? 1536
                                   : 1024; // Unknown targets use the smallest known cap to avoid over-requesting.
#else
    return 1024;
#endif
}

__host__ __device__ inline constexpr int ResidentBlocks(int threads, int requested)
{
    const int cap = MaxThreadsPerSM() / threads;
    return (cap <= 0) ? 1 : (requested < cap ? requested : cap);
}

// Callers bound tensor byte extents and image-plane offsets by INT32_MAX, permitting cheaper 32-bit addressing.
struct GroupPlan
{
    int32_t segPixels;
    int32_t srcSegStride;
    int32_t dstSegStride;
    int32_t srcSampleStride;
    int32_t dstSampleStride;
    int32_t srcPlaneStride;
    int32_t dstPlaneStride;
};

// OR-ing every plane pointer preserves exactly their common power-of-two alignment.
struct GroupBatchInfo
{
    bool      dense     = true;
    uintptr_t ptrOr     = 0;
    int64_t   maxPixels = 0;
};

template<int GroupBytes, typename BT, bool kPlanar, class Op>
inline __device__ void PixelGroupBody(const BT *s0, const BT *s1, const BT *s2, BT *d0, BT *d1, BT *d2,
                                      uint32_t segPixels, uint32_t g, const typename Op::Param &param)
{
    constexpr int kPix = GroupBytes / static_cast<int>(sizeof(BT));
    static_assert(GroupBytes == 4 || GroupBytes == 8 || GroupBytes == 16,
                  "A pixel group is one 32-, 64- or 128-bit value");
    using Vec = typename PackType<GroupBytes>::type;

    const uint32_t p0 = g * kPix;
    if (p0 >= segPixels)
    {
        return;
    }

    // Keep indices compile-time constant to avoid local-memory spills.
    const BT *sp[3] = {s0, s1, s2};
    BT       *dp[3] = {d0, d1, d2};

    if (segPixels - p0 >= kPix)
    {
        BT e[3 * kPix];
#pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            const BT *base = kPlanar ? (sp[j] + p0) : (s0 + p0 * 3 + j * kPix);
            const Vec v    = *reinterpret_cast<const Vec *>(base);
            const BT *vb   = reinterpret_cast<const BT *>(&v);
#pragma unroll
            for (int l = 0; l < kPix; ++l)
            {
                e[j * kPix + l] = vb[l];
            }
        }

#pragma unroll
        for (int k = 0; k < kPix; ++k)
        {
            const int a = kPlanar ? (0 * kPix + k) : (3 * k + 0);
            const int b = kPlanar ? (1 * kPix + k) : (3 * k + 1);
            const int c = kPlanar ? (2 * kPix + k) : (3 * k + 2);
            Op::Channels(e[a], e[b], e[c], param, e[a], e[b], e[c]);
        }

#pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            Vec v;
            BT *vb = reinterpret_cast<BT *>(&v);
#pragma unroll
            for (int l = 0; l < kPix; ++l)
            {
                vb[l] = e[j * kPix + l];
            }
            BT *base                       = kPlanar ? (dp[j] + p0) : (d0 + p0 * 3 + j * kPix);
            *reinterpret_cast<Vec *>(base) = v;
        }
    }
    else
    {
        for (uint32_t p = p0; p < segPixels; ++p)
        {
            if constexpr (kPlanar)
            {
                Op::Channels(s0[p], s1[p], s2[p], param, d0[p], d1[p], d2[p]);
            }
            else
            {
                const BT *s = s0 + p * 3;
                BT       *d = d0 + p * 3;
                Op::Channels(s[0], s[1], s[2], param, d[0], d[1], d[2]);
            }
        }
    }
}

template<int GroupBytes, typename BT, bool kPlanar, class Op>
__global__ __launch_bounds__(kGroupBlockW, ResidentBlocks(kGroupBlockW, Op::kMinBlocksPerMP)) void PixelGroupKernel(
    const BT *__restrict__ srcBase, BT *__restrict__ dstBase, GroupPlan plan, uint32_t numGroups,
    typename Op::Param param)
{
    static_assert(Op::kMinBlocksPerMP >= 1, "An occupancy floor of at least one block is required");
    static_assert(kGroupBlockW * ResidentBlocks(kGroupBlockW, Op::kMinBlocksPerMP) <= MaxThreadsPerSM(),
                  "__launch_bounds__ must be satisfiable on the narrowest target architecture");

    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGroups)
    {
        return;
    }

    const int32_t srcOff = static_cast<int32_t>(blockIdx.z) * plan.srcSampleStride
                         + static_cast<int32_t>(blockIdx.y) * plan.srcSegStride;
    const int32_t dstOff = static_cast<int32_t>(blockIdx.z) * plan.dstSampleStride
                         + static_cast<int32_t>(blockIdx.y) * plan.dstSegStride;

    const auto sAt = [&](int32_t extra)
    {
        return reinterpret_cast<const BT *>(reinterpret_cast<const char *>(srcBase) + srcOff + extra);
    };
    const auto dAt = [&](int32_t extra)
    {
        return reinterpret_cast<BT *>(reinterpret_cast<char *>(dstBase) + dstOff + extra);
    };

    PixelGroupBody<GroupBytes, BT, kPlanar, Op>(sAt(0), sAt(plan.srcPlaneStride), sAt(2 * plan.srcPlaneStride), dAt(0),
                                                dAt(plan.dstPlaneStride), dAt(2 * plan.dstPlaneStride), plan.segPixels,
                                                g, param);
}

// The host must verify dense, aligned planes before treating each image as one contiguous run.
template<int GroupBytes, typename BT, bool kPlanar, class Op>
__global__
    __launch_bounds__(kGroupBlockW, ResidentBlocks(kGroupBlockW, Op::kMinBlocksPerMP)) void PixelGroupVarShapeKernel(
        cuda::ImageBatchVarShapeWrap<const BT> src, cuda::ImageBatchVarShapeWrap<BT> dst, typename Op::Param param)
{
    static_assert(Op::kMinBlocksPerMP >= 1, "An occupancy floor of at least one block is required");
    static_assert(kGroupBlockW * ResidentBlocks(kGroupBlockW, Op::kMinBlocksPerMP) <= MaxThreadsPerSM(),
                  "__launch_bounds__ must be satisfiable on the narrowest target architecture");

    const int      z         = blockIdx.z;
    const uint32_t segPixels = static_cast<uint32_t>(dst.width(z, 0)) * dst.height(z, 0);
    const uint32_t g         = blockIdx.x * blockDim.x + threadIdx.x;

    const BT *s0 = src.ptr(z, 0, 0, 0);
    BT       *d0 = dst.ptr(z, 0, 0, 0);
    if constexpr (kPlanar)
    {
        PixelGroupBody<GroupBytes, BT, true, Op>(s0, src.ptr(z, 1, 0, 0), src.ptr(z, 2, 0, 0), d0, dst.ptr(z, 1, 0, 0),
                                                 dst.ptr(z, 2, 0, 0), segPixels, g, param);
    }
    else
    {
        PixelGroupBody<GroupBytes, BT, false, Op>(s0, s0, s0, d0, d0, d0, segPixels, g, param);
    }
}

inline bool MakeGroupSide(const nvcv::TensorDataStridedCuda              &data,
                          const nvcv::TensorDataAccessStridedImagePlanar &access, bool isPlanar, int64_t elemSize,
                          int64_t align, int64_t &segPixels, int32_t &numSegments, int32_t &segStride,
                          int32_t &sampleStride, int32_t &planeStride)
{
    if (reinterpret_cast<uintptr_t>(data.basePtr()) % align != 0)
    {
        return false;
    }
    if (access.colStride() != (isPlanar ? elemSize : 3 * elemSize))
    {
        return false;
    }
    if (!isPlanar && access.chStride() != elemSize)
    {
        return false;
    }

    sampleStride = static_cast<int32_t>(access.sampleStride());
    planeStride  = isPlanar ? static_cast<int32_t>(access.planeStride()) : 0;
    if (sampleStride % align != 0 || planeStride % align != 0)
    {
        return false;
    }

    const int64_t rowStride  = access.rowStride();
    const int64_t denseBytes = access.numCols() * access.colStride();
    if (rowStride == denseBytes)
    {
        segPixels   = access.numRows() * access.numCols();
        numSegments = 1;
        segStride   = 0;
    }
    else
    {
        if (rowStride % align != 0 || access.numRows() > kMaxGridYZ)
        {
            return false;
        }
        segPixels   = access.numCols();
        numSegments = static_cast<int>(access.numRows());
        segStride   = static_cast<int32_t>(rowStride);
    }
    return true;
}

template<int GroupBytes, typename BT, bool kPlanar, class Op>
inline bool TryRunPixelGroup(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                             const nvcv::TensorDataStridedCuda              &dstData,
                             const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                             const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, typename Op::Param param)
{
    constexpr int64_t elemSize = sizeof(BT);
    constexpr int     kPix     = GroupBytes / static_cast<int>(sizeof(BT));

    GroupPlan plan{};
    int64_t   srcSegPixels, dstSegPixels;
    int32_t   srcSegs, dstSegs;
    if (!MakeGroupSide(srcData, srcAccess, kPlanar, elemSize, GroupBytes, srcSegPixels, srcSegs, plan.srcSegStride,
                       plan.srcSampleStride, plan.srcPlaneStride)
        || !MakeGroupSide(dstData, dstAccess, kPlanar, elemSize, GroupBytes, dstSegPixels, dstSegs, plan.dstSegStride,
                          plan.dstSampleStride, plan.dstPlaneStride))
    {
        return false;
    }
    if (srcSegPixels != dstSegPixels || srcSegs != dstSegs)
    {
        return false;
    }
    // The caller's INT32_MAX byte-extent bound also bounds pixels: each consumes 3 * elemSize bytes.
    plan.segPixels = static_cast<int32_t>(srcSegPixels);

    const int64_t numGroups = util::DivUp<int64_t>(plan.segPixels, kPix);
    const int64_t blocksX   = util::DivUp<int64_t>(numGroups, kGroupBlockW);
    if (blocksX > cuda::TypeTraits<int32_t>::max)
    {
        return false;
    }

    const dim3 block(kGroupBlockW, 1, 1);
    const dim3 grid(static_cast<unsigned int>(blocksX), static_cast<unsigned int>(srcSegs),
                    static_cast<unsigned int>(srcAccess.numSamples()));

    PixelGroupKernel<GroupBytes, BT, kPlanar, Op><<<grid, block, 0, stream>>>(
        reinterpret_cast<const BT *>(srcData.basePtr()), reinterpret_cast<BT *>(dstData.basePtr()), plan,
        static_cast<uint32_t>(numGroups), param);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<int GroupBytes, typename BT, bool kPlanar, class Op>
inline bool TryRunPixelGroupVarShape(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                     const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, const GroupBatchInfo &info,
                                     typename Op::Param param)
{
    constexpr int kPix = GroupBytes / static_cast<int>(sizeof(BT));

    if (!info.dense || info.ptrOr % GroupBytes != 0)
    {
        return false;
    }

    const int64_t numGroups = util::DivUp<int64_t>(info.maxPixels, kPix);
    const int64_t blocksX   = util::DivUp<int64_t>(numGroups, kGroupBlockW);
    if (blocksX > cuda::TypeTraits<int32_t>::max)
    {
        return false;
    }

    const dim3 block(kGroupBlockW, 1, 1);
    const dim3 grid(static_cast<unsigned int>(blocksX), 1, static_cast<unsigned int>(dstData.numImages()));

    cuda::ImageBatchVarShapeWrap<const BT> src(srcData);
    cuda::ImageBatchVarShapeWrap<BT>       dst(dstData);
    PixelGroupVarShapeKernel<GroupBytes, BT, kPlanar, Op><<<grid, block, 0, stream>>>(src, dst, param);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Shifted carriers.
//
// The stream above needs the source and the destination to share a 16-byte phase. A movement --
// a crop, a border, a flip -- breaks that: the source bytes a destination
// carrier needs start at an arbitrary offset, may run backwards, and may be a byte permutation of
// what the source holds. The primitives below keep the *store* naturally aligned, which is
// unconditional, and repair the source in registers: a window of consecutive source words is read
// with aligned 128-bit loads plus a funnel shift, then the destination words are built from that
// window with PRMT.
//
// A movement supplies its window width and address, source-region mapping, Apply, and TailBytes.
// Permutation carriers additionally describe the gather stages and phases; a row copy does not.
// ---------------------------------------------------------------------------------------------

inline constexpr int kEdgeThreads = 4; // threads per run end for the head and tail bytes
inline constexpr int kEdgeBytes   = kCarrierBytes / kEdgeThreads;

// Returns bytes [shift, shift+16) of the 32-byte window (lo, hi). The word selection is two levels
// of predicated moves rather than an indexed array: an index the compiler cannot fold would push
// the whole window to local memory. The leftover sub-word offset is applied with PRMT, whose
// selector may be a runtime register.
__device__ __forceinline__ uint4 ShiftWindow(uint4 lo, uint4 hi, uint32_t shift)
{
    const bool     drop2 = (shift & 8u) != 0;
    const uint32_t d0    = drop2 ? lo.z : lo.x;
    const uint32_t d1    = drop2 ? lo.w : lo.y;
    const uint32_t d2    = drop2 ? hi.x : lo.z;
    const uint32_t d3    = drop2 ? hi.y : lo.w;
    const uint32_t d4    = drop2 ? hi.z : hi.x;
    const uint32_t d5    = drop2 ? hi.w : hi.y;

    const bool     drop1 = (shift & 4u) != 0;
    const uint32_t e0    = drop1 ? d1 : d0;
    const uint32_t e1    = drop1 ? d2 : d1;
    const uint32_t e2    = drop1 ? d3 : d2;
    const uint32_t e3    = drop1 ? d4 : d3;
    const uint32_t e4    = drop1 ? d5 : d4;

    const uint32_t sel = 0x3210u + (shift & 3u) * 0x1111u;

    uint4 out;
    out.x = __byte_perm(e0, e1, sel);
    out.y = __byte_perm(e1, e2, sel);
    out.z = __byte_perm(e2, e3, sel);
    out.w = __byte_perm(e3, e4, sel);
    return out;
}

// Reads `Words` consecutive 32-bit words from an arbitrarily aligned address. The widened aligned
// window can reach past what the movement needs, so it is taken only while it stays inside
// [lo, hi); the handful of carriers at a buffer's edges fall back to a byte-wise read of exactly
// the bytes they own.
template<int Words>
__device__ __forceinline__ void LoadWindow(const unsigned char *p, const unsigned char *lo, const unsigned char *hi,
                                           uint32_t (&out)[Words])
{
    static_assert(Words % 4 == 0, "a window is a whole number of 16-byte carriers");
    constexpr int kVecs = Words / 4;

    const uint32_t       shift = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) & (kCarrierBytes - 1));
    const unsigned char *base  = p - shift;

    if (base >= lo && base + (kVecs + 1) * kCarrierBytes <= hi)
    {
        uint4 v[kVecs + 1];
#pragma unroll
        for (int i = 0; i <= kVecs; ++i)
        {
            v[i] = *reinterpret_cast<const uint4 *>(base + kCarrierBytes * i);
        }
#pragma unroll
        for (int i = 0; i < kVecs; ++i)
        {
            const uint4 r  = (shift == 0) ? v[i] : ShiftWindow(v[i], v[i + 1], shift);
            out[4 * i + 0] = r.x;
            out[4 * i + 1] = r.y;
            out[4 * i + 2] = r.z;
            out[4 * i + 3] = r.w;
        }
        return;
    }

    // A movement's window may be anchored outside the buffer even though every byte it goes on to
    // name is inside it -- a reversal near a row's end is the case that reaches furthest back. The
    // bytes outside are never selected, so reading them as zero keeps the result while keeping the
    // access legal.
#pragma unroll
    for (int i = 0; i < Words; ++i)
    {
        uint32_t w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b)
        {
            const unsigned char *q = p + 4 * i + b;
            w |= (q >= lo && q < hi ? static_cast<uint32_t>(*q) : 0u) << (8 * b);
        }
        out[i] = w;
    }
}

// Selects one of `Phases` values with a runtime phase while keeping every array index an
// immediate: an index the compiler cannot fold costs the whole array a trip through local memory.
template<int Phases>
__device__ __forceinline__ uint32_t PickPhase(const uint32_t (&v)[Phases], int phase)
{
    uint32_t r = v[0];
#pragma unroll
    for (int i = 1; i < Phases; ++i)
    {
        r = (phase == i) ? v[i] : r;
    }
    return r;
}

// PRMT selectors for one destination carrier, one per (destination word, stage, phase).
// Reversal and broadcast geometry fixes the selectors; the source anchor remains a launch parameter.
template<int Stages, int Phases>
struct GatherSelectors
{
    uint32_t sel[4][Stages][Phases];
    uint32_t keep[4][Phases]; // clears the destination lanes the movement fills with zero
};

// Exact window geometry for reversal and broadcast byte maps. Deriving the window width,
// per-word gather start and PRMT depth here avoids restating the network in each map.
template<class Map>
__host__ __device__ constexpr int GatherShift()
{
    int m = 0;
    for (int p = 0; p < Map::kPhases; ++p)
    {
        for (int j = 0; j < kCarrierBytes; ++j)
        {
            m = Map::RawIndex(p, j) < m ? Map::RawIndex(p, j) : m;
        }
    }
    return -m;
}

template<class Map>
__host__ __device__ constexpr int GatherWindowWords()
{
    int m = 0;
    for (int p = 0; p < Map::kPhases; ++p)
    {
        for (int j = 0; j < kCarrierBytes; ++j)
        {
            m = Map::RawIndex(p, j) > m ? Map::RawIndex(p, j) : m;
        }
    }
    const int maxWord = (GatherShift<Map>() + m) / 4;
    return (maxWord + 4) / 4 * 4;
}

template<class Map>
__host__ __device__ constexpr int GatherWordBase(int w)
{
    int m = 1 << 30;
    for (int p = 0; p < Map::kPhases; ++p)
    {
        for (int k = 0; k < 4; ++k)
        {
            const int q = (GatherShift<Map>() + Map::RawIndex(p, 4 * w + k)) / 4;
            m           = q < m ? q : m;
        }
    }
    return m;
}

template<class Map>
__host__ __device__ constexpr int GatherStages()
{
    int stages = 1;
    for (int w = 0; w < 4; ++w)
    {
        const int base = GatherWordBase<Map>(w);
        for (int p = 0; p < Map::kPhases; ++p)
        {
            for (int k = 0; k < 4; ++k)
            {
                const int q = (GatherShift<Map>() + Map::RawIndex(p, 4 * w + k)) / 4;
                stages      = q - base > stages ? q - base : stages;
            }
        }
    }
    return stages;
}

// Builds every selector a movement needs from the one thing that distinguishes it: the byte map
// `map(phase, j)` giving the window byte destination byte `j` of a carrier at that phase reads, or
// a negative value for a byte the movement fills with zero. This is the single place the PRMT
// network is expressed; a movement only states its map.
//
// Stage 0 folds window words WordBase(w) and WordBase(w)+1, and stage s > 0 folds in word
// WordBase(w)+s+1, so a destination byte fed by a later word holds a placeholder until its stage
// overwrites it and a byte already resolved passes through as slot `k` of the accumulator.
template<class Carrier, class MapFn>
inline GatherSelectors<Carrier::kStages, Carrier::kPhases> BuildGatherSelectors(const MapFn &map)
{
    GatherSelectors<Carrier::kStages, Carrier::kPhases> g{};

    for (int phase = 0; phase < Carrier::kPhases; ++phase)
    {
        for (int w = 0; w < 4; ++w)
        {
            const int base = Carrier::WordBase(w);
            uint32_t  keep = 0;
            for (int k = 0; k < 4; ++k)
            {
                const int b = map(phase, 4 * w + k);
                const int q = b >> 2;
                const int r = b & 3;

                uint32_t nib0 = 0;
                if (b >= 0)
                {
                    assert(q >= base && q <= base + Carrier::kStages);
                    keep |= 0xFFu << (8 * k);
                    nib0 = q == base ? static_cast<uint32_t>(r) : (q == base + 1 ? static_cast<uint32_t>(4 + r) : 0u);
                }
                g.sel[w][0][phase] |= nib0 << (4 * k);

                for (int s = 1; s < Carrier::kStages; ++s)
                {
                    const uint32_t nib
                        = (b >= 0 && q == base + s + 1) ? static_cast<uint32_t>(4 + r) : static_cast<uint32_t>(k);
                    g.sel[w][s][phase] |= nib << (4 * k);
                }
            }
            g.keep[w][phase] = keep;
        }
    }
    return g;
}

// Builds one destination word from the window in `kStages` PRMTs. `WordBase` is the window word
// the chain starts at, which the geometry fixes, so every index stays an immediate and the window
// stays in registers.
template<class Carrier, int W, int Words>
__device__ __forceinline__ uint32_t GatherWord(const uint32_t (&win)[Words], int phase,
                                               const GatherSelectors<Carrier::kStages, Carrier::kPhases> &g)
{
    constexpr int kBase = Carrier::WordBase(W);
    static_assert(kBase >= 0 && kBase < Words, "the gather chain starts outside the source window");

    // A chain whose depth runs past the window's last word is one whose destination word had
    // nothing left to fold in; the selector is then a pass-through, so clamping the operand keeps
    // the result while keeping the index an immediate.
    constexpr int kOp0 = kBase + 1 < Words ? kBase + 1 : Words - 1;

    uint32_t acc = __byte_perm(win[kBase], win[kOp0], PickPhase<Carrier::kPhases>(g.sel[W][0], phase));
#pragma unroll
    for (int s = 1; s < Carrier::kStages; ++s)
    {
        const int op = kBase + s + 1 < Words ? kBase + s + 1 : Words - 1;
        acc          = __byte_perm(acc, win[op], PickPhase<Carrier::kPhases>(g.sel[W][s], phase));
    }
    return acc & PickPhase<Carrier::kPhases>(g.keep[W], phase);
}

// The whole 16-byte carrier. The permutation repeats every `kPhases` carriers, so a carrier's
// phase is its index modulo that; a movement whose map does not repeat declares one phase.
template<class Carrier, int Words>
__device__ __forceinline__ uint4 GatherCarrier(const uint32_t (&win)[Words], int32_t u,
                                               const GatherSelectors<Carrier::kStages, Carrier::kPhases> &g)
{
    const int phase = Carrier::kPhases == 1 ? 0 : static_cast<int>(u % Carrier::kPhases);

    uint4 out;
    out.x = GatherWord<Carrier, 0>(win, phase, g);
    out.y = GatherWord<Carrier, 1>(win, phase, g);
    out.z = GatherWord<Carrier, 2>(win, phase, g);
    out.w = GatherWord<Carrier, 3>(win, phase, g);
    return out;
}

// `numRegions` runs of `runBytes` bytes, built from a byte geometry the movement names rather than
// from the image row. Regions and then samples are folded into the run whenever the next one
// starts exactly where the previous ends, so a packed full-width movement degenerates into a
// single flat run. A movement whose source-region map is not the identity -- a mirror, a border --
// must not fold, because folding drops the per-region source offset.
inline bool MakeRunPlan(int64_t runBytes, int64_t numRegions, int64_t numSamples, int64_t srcRegionStride,
                        int64_t dstRegionStride, int64_t srcSampleStride, int64_t dstSampleStride, bool allowFold,
                        Plan &out)
{
    if (runBytes <= 0 || numRegions <= 0 || numSamples <= 0)
    {
        return false;
    }

    if (allowFold && numRegions > 1 && srcRegionStride == runBytes && dstRegionStride == runBytes
        && runBytes * numRegions <= cuda::TypeTraits<int32_t>::max)
    {
        runBytes *= numRegions;
        numRegions      = 1;
        srcRegionStride = 0;
        dstRegionStride = 0;
    }
    if (allowFold && numRegions == 1 && numSamples > 1 && srcSampleStride == runBytes && dstSampleStride == runBytes
        && runBytes * numSamples <= cuda::TypeTraits<int32_t>::max)
    {
        runBytes *= numSamples;
        numSamples      = 1;
        srcSampleStride = 0;
        dstSampleStride = 0;
    }

    // numRegions is grid.y and numSamples grid.z, both 16-bit: bound the 64-bit counts before the
    // narrowing casts, as MakePlan does -- truncating first can land back inside the valid window
    // and describe fewer regions or samples than the movement has.
    if (numRegions > kMaxGridYZ || numSamples > kMaxGridYZ || runBytes > cuda::TypeTraits<int32_t>::max)
    {
        return false;
    }

    out.runBytes        = runBytes;
    out.dstRunBytes     = runBytes;
    out.numRegions      = static_cast<int>(numRegions);
    out.regionsPerPlane = static_cast<int>(numRegions);
    out.numSamples      = static_cast<int>(numSamples);
    out.srcRegionStride = numRegions > 1 ? srcRegionStride : 0;
    out.dstRegionStride = numRegions > 1 ? dstRegionStride : 0;
    out.srcSampleStride = numSamples > 1 ? srcSampleStride : 0;
    out.dstSampleStride = numSamples > 1 ? dstSampleStride : 0;
    return true;
}

// The shifted path only stores through the carrier type, so only the destination strides have to
// be carrier multiples; the source keeps whatever byte phase the movement gives it, and the
// destination run's own phase becomes the head the kernel copies byte-wise. Aligned strides are
// what makes that head the same in every region and sample.
inline bool IsDstCarrierAligned(const Plan &plan)
{
    return plan.dstRegionStride % kCarrierBytes == 0 && plan.dstSampleStride % kCarrierBytes == 0;
}

// One past the last byte a movement may read: the end of the final row of the final sample.
// Deriving it from the extents rather than from the sample stride keeps it exact for a view that
// starts at a plane offset inside a larger buffer, where the stride would overshoot the allocation.
inline const unsigned char *SourceEnd(const unsigned char *base, const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return base + (access.numSamples() - 1) * access.sampleStride() + (access.numRows() - 1) * access.rowStride()
         + access.numCols() * access.colStride();
}

// Destination region r reads source region r.
struct IdentityRegions
{
    using Param = NoParam;

    static __device__ int32_t SrcRegion(int32_t region, const Param &)
    {
        return region;
    }
};

// The movement that only shifts: destination byte j of a run is source byte j of the mapped source
// run, so the window is one carrier and no permutation runs at all. `Region` supplies the
// destination-to-source region map -- the identity for a crop, a border fold or a mirror for
// others -- and the parameter that map needs.
template<class Region>
struct ShiftCarrier
{
    using Param = typename Region::Param;

    // The map does not repeat, so the run's destination may start at any byte and the kernel copies
    // the leading bytes to the first carrier boundary.
    static constexpr bool kAlignedRun  = false;
    static constexpr int  kWindowWords = 4;

    static __device__ int32_t SrcRegion(int32_t region, const Param &p)
    {
        return Region::SrcRegion(region, p);
    }

    static __device__ int64_t WindowByte(int32_t dstByte, int32_t, int32_t, const Param &)
    {
        return dstByte;
    }

    static __device__ uint4 Apply(const uint32_t (&w)[kWindowWords], int32_t, const Param &)
    {
        return uint4{w[0], w[1], w[2], w[3]};
    }

    __device__ static void TailBytes(const unsigned char *s, unsigned char *d, int32_t first, int32_t last, int32_t,
                                     const Param &)
    {
        for (int32_t j = first; j < last; ++j)
        {
            d[j] = s[j];
        }
    }
};

__host__ __device__ constexpr int GcdInt(int a, int b)
{
    return b == 0 ? a : GcdInt(b, a % b);
}

// Pixels reversed inside the run: destination byte at run offset o is source byte
// anchor - o + 2 * (o mod P). A horizontal flip and a reflected border strip are that same map and
// differ only in the anchor, which is a launch-time value.
template<int P>
struct ReverseMap
{
    static constexpr int kPhases = P / GcdInt(P, kCarrierBytes);
    static constexpr int kStep   = -1;

    struct Data
    {
        int64_t anchor;
    };

    __host__ __device__ static constexpr int RawIndex(int phase, int j)
    {
        return 2 * ((kCarrierBytes * phase + j) % P) - j;
    }

    __host__ __device__ static constexpr int Index(int phase, int j, const Data &)
    {
        return RawIndex(phase, j) + GatherShift<ReverseMap>();
    }

    static int64_t WindowAnchor(const Data &d)
    {
        return d.anchor - GatherShift<ReverseMap>();
    }

    static __device__ int64_t TailByte(int32_t j, int32_t, const Data &d)
    {
        return d.anchor - j + 2 * (j % P);
    }
};

// One pixel repeated across the run: a replicated border strip.
template<int P>
struct BroadcastMap
{
    static constexpr int kPhases = P / GcdInt(P, kCarrierBytes);
    static constexpr int kStep   = 0;

    struct Data
    {
        int64_t src; // byte offset of the repeated pixel in the source run
    };

    __host__ __device__ static constexpr int RawIndex(int phase, int j)
    {
        return (kCarrierBytes * phase + j) % P;
    }

    __host__ __device__ static constexpr int Index(int phase, int j, const Data &)
    {
        return RawIndex(phase, j);
    }

    static int64_t WindowAnchor(const Data &d)
    {
        return d.src;
    }

    static __device__ int64_t TailByte(int32_t j, int32_t, const Data &d)
    {
        return d.src + (j % P);
    }
};

// A movement whose destination carrier is a byte permutation of a window of source bytes. `Map`
// fixes the geometry and the byte map, `Region` the destination-to-source region map. The carrier
// index is the permutation's phase, so the run's destination must start on a carrier boundary.
template<class Map, class Region>
struct PermuteCarrier
{
    // The carrier index carries the permutation's phase, so the run's destination must start on a
    // carrier boundary for that index to name the right phase.
    static constexpr bool kAlignedRun  = true;
    static constexpr int  kPhases      = Map::kPhases;
    static constexpr int  kWindowWords = GatherWindowWords<Map>();
    static constexpr int  kStages      = GatherStages<Map>();

    __host__ __device__ static constexpr int WordBase(int w)
    {
        return GatherWordBase<Map>(w);
    }

    struct Param
    {
        typename Region::Param            region;
        typename Map::Data                map;
        GatherSelectors<kStages, kPhases> gather;
        int64_t                           anchor;
    };

    static __device__ int32_t SrcRegion(int32_t region, const Param &p)
    {
        return Region::SrcRegion(region, p.region);
    }

    static __device__ int64_t WindowByte(int32_t dstByte, int32_t, int32_t, const Param &p)
    {
        return p.anchor + Map::kStep * static_cast<int64_t>(dstByte);
    }

    static __device__ uint4 Apply(const uint32_t (&win)[kWindowWords], int32_t u, const Param &p)
    {
        return GatherCarrier<PermuteCarrier>(win, u, p.gather);
    }

    __device__ static void TailBytes(const unsigned char *s, unsigned char *d, int32_t first, int32_t last,
                                     int32_t runBytes, const Param &p)
    {
        for (int32_t j = first; j < last; ++j)
        {
            const int64_t b = Map::TailByte(j, runBytes, p.map);
            d[j]            = b < 0 ? static_cast<unsigned char>(0) : s[b];
        }
    }
};

template<class Map, class Region>
inline typename PermuteCarrier<Map, Region>::Param MakePermuteParam(const typename Map::Data     &data,
                                                                    const typename Region::Param &region)
{
    using C = PermuteCarrier<Map, Region>;

    typename C::Param p{};
    p.region = region;
    p.map    = data;
    p.anchor = Map::WindowAnchor(data);
    p.gather = BuildGatherSelectors<C>([&](int phase, int j) { return Map::Index(phase, j, data); });
    return p;
}

// Every byte of a run reaches Apply or TailBytes exactly once. `headBytes` is what it takes to
// reach the run's first 16-byte boundary and `bodyCarriers` the whole carriers after it; both ends
// are spread over kEdgeThreads threads so a short edge does not become the long pole of its warp.
template<int Groups, typename Carrier>
__global__ void ShiftedStreamKernel(const unsigned char *__restrict__ srcBase, unsigned char *__restrict__ dstBase,
                                    const unsigned char *srcLo, const unsigned char *srcHi, int32_t headBytes,
                                    int32_t bodyCarriers, int32_t runBytes, int32_t numRegions, int64_t srcRegionStride,
                                    int64_t srcSampleStride, int64_t dstRegionStride, int64_t dstSampleStride,
                                    typename Carrier::Param param)
{
    const int32_t region = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (region >= numRegions)
    {
        return;
    }

    const int64_t sOff = static_cast<int64_t>(blockIdx.z) * srcSampleStride
                       + static_cast<int64_t>(Carrier::SrcRegion(region, param)) * srcRegionStride;
    const int64_t dOff
        = static_cast<int64_t>(blockIdx.z) * dstSampleStride + static_cast<int64_t>(region) * dstRegionStride;

    const unsigned char *s = srcBase + sOff;
    unsigned char       *d = dstBase + dOff;

    const int32_t i0 = static_cast<int32_t>(blockIdx.x * (blockDim.x * Groups) + threadIdx.x);

    // Every window is loaded before the first store, so a warp keeps `Groups` requests outstanding.
    uint32_t win[Groups][Carrier::kWindowWords];
    int32_t  idx[Groups];
    bool     live[Groups];
#pragma unroll
    for (int g = 0; g < Groups; ++g)
    {
        idx[g]  = i0 + g * static_cast<int32_t>(blockDim.x);
        live[g] = idx[g] < bodyCarriers;
        if (live[g])
        {
            const int32_t dstByte = headBytes + idx[g] * kCarrierBytes;
            LoadWindow<Carrier::kWindowWords>(s + Carrier::WindowByte(dstByte, idx[g], runBytes, param), srcLo, srcHi,
                                              win[g]);
        }
    }
#pragma unroll
    for (int g = 0; g < Groups; ++g)
    {
        if (live[g])
        {
            *reinterpret_cast<uint4 *>(d + headBytes + idx[g] * kCarrierBytes) = Carrier::Apply(win[g], idx[g], param);
        }
    }

    if (blockIdx.x == 0 && threadIdx.x < 2 * kEdgeThreads)
    {
        const int32_t lane    = static_cast<int32_t>(threadIdx.x);
        const int32_t bodyEnd = headBytes + bodyCarriers * kCarrierBytes;
        const int32_t first   = lane < kEdgeThreads ? lane * kEdgeBytes : bodyEnd + (lane - kEdgeThreads) * kEdgeBytes;
        const int32_t last    = ::min(first + kEdgeBytes, lane < kEdgeThreads ? headBytes : runBytes);
        if (first < last)
        {
            Carrier::TailBytes(s, d, first, last, runBytes, param);
        }
    }
}

// False without launching when the destination alignment gate declines.
template<int Groups, typename Carrier>
inline bool TryRunShifted(cudaStream_t stream, const unsigned char *srcBase, unsigned char *dstBase,
                          const unsigned char *srcLo, const unsigned char *srcHi, const Plan &plan,
                          typename Carrier::Param param)
{
    if (!IsDstCarrierAligned(plan))
    {
        return false;
    }

    const int32_t runBytes     = static_cast<int32_t>(plan.runBytes);
    const int32_t headBytes    = static_cast<int32_t>(std::min<int64_t>(
        static_cast<int64_t>((0u - reinterpret_cast<uintptr_t>(dstBase)) & (kCarrierBytes - 1)), plan.runBytes));
    const int32_t bodyCarriers = (runBytes - headBytes) / kCarrierBytes;

    if (Carrier::kAlignedRun && headBytes != 0)
    {
        return false;
    }

    // A run with no whole carrier at all -- a strip a few bytes wide -- is entirely head and tail,
    // so it only needs the edge threads; the width heuristic would hand it the widest block.
    const int blockW
        = bodyCarriers > 0 ? std::max(ChooseBlockWidth<Groups>(bodyCarriers), 2 * kEdgeThreads) : 2 * kEdgeThreads;
    const int blockH = std::min<int>(kBlockThreads / blockW, std::max(plan.numRegions, 1));

    dim3 block(static_cast<unsigned int>(blockW), static_cast<unsigned int>(blockH), 1);
    dim3 grid(static_cast<unsigned int>(
                  std::max<int64_t>(util::DivUp<int64_t>(bodyCarriers, static_cast<int64_t>(blockW) * Groups), 1)),
              static_cast<unsigned int>(util::DivUp(plan.numRegions, blockH)),
              static_cast<unsigned int>(plan.numSamples));

    ShiftedStreamKernel<Groups, Carrier><<<grid, block, 0, stream>>>(
        srcBase, dstBase, srcLo, srcHi, headBytes, bodyCarriers, runBytes, plan.numRegions, plan.srcRegionStride,
        plan.srcSampleStride, plan.dstRegionStride, plan.dstSampleStride, param);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

} // namespace cvcuda::priv::elem_stream

#endif // CVCUDA_PRIV_ELEMENT_STREAM_COMMON_CUH
