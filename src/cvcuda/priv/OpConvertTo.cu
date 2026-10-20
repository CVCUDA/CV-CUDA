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

#include "CudaDeviceUtils.hpp"
#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpConvertTo.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include "ElementStreamCommon.cuh"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cuda        = nvcv::cuda;
namespace elem_stream = cvcuda::priv::elem_stream;
namespace util        = nvcv::util;

namespace {

using uchar  = unsigned char;
using schar  = signed char;
using ushort = unsigned short;

// True when a kWrap=true kernel instantiation differs from kWrap=false.
// Float destinations do not clip so wrap is inert; S32 is rejected at the API layer
// because float32 arithmetic cannot represent the 2^31 wrap period exactly.
template<typename T>
inline constexpr bool kWrapEligibleDst = std::is_integral_v<T> && sizeof(T) <= 2;

// Input must already be rounded to an integral value (caller's responsibility).
// Only valid for 8- and 16-bit T: float32 cannot represent the 2^31 wrap period exactly
// (S32 is rejected at the API layer before this function is instantiated with T=int32_t).
template<typename T, typename W>
__device__ __forceinline__ W WrapToRange(W work)
{
    static_assert(std::is_integral_v<T> && sizeof(T) <= 2, "WrapToRange is only valid for 8- and 16-bit integer types");
    // Route through uint64_t so max+1 is an exact integer (a power of two) before casting
    // to W, avoiding reliance on FP rounding direction for the conversion.
    constexpr W kMaxPlusOne = static_cast<W>(static_cast<uint64_t>(cuda::TypeTraits<T>::max) + 1ull);

    if constexpr (std::is_unsigned_v<T>)
    {
        if (work < W{0})
            return W{0};
        return work - kMaxPlusOne * cuda::round<cuda::RoundMode::ZERO>(work / kMaxPlusOne);
    }
    else
    {
        constexpr W kSpan = W{2} * kMaxPlusOne;
        W           w     = work - kSpan * cuda::round<cuda::RoundMode::ZERO>(work / kSpan);
        if (w >= kMaxPlusOne)
            w -= kSpan;
        else if (w < -kMaxPlusOne)
            w += kSpan;
        return w;
    }
}

template<typename SRC_TYPE, typename DST_TYPE, typename S, bool kWrap = false>
struct Convertor
{
    S    alpha;
    S    beta;
    bool truncate; // round toward zero instead of to-nearest (only affects integer outputs)

    __device__ __forceinline__ DST_TYPE operator()(SRC_TYPE src) const
    {
        auto work = alpha * src + beta;
        if constexpr (std::is_integral_v<cuda::BaseType<DST_TYPE>>)
        {
            if (truncate)
                work = cuda::round<cuda::RoundMode::ZERO>(work);
            if constexpr (kWrap)
            {
                if (!truncate)
                    work = cuda::round<cuda::RoundMode::NEAREST>(work);
                using WB   = cuda::BaseType<decltype(work)>;
                auto *lane = reinterpret_cast<WB *>(&work);
#pragma unroll
                for (int c = 0; c < cuda::NumElements<DST_TYPE>; ++c)
                    lane[c] = WrapToRange<cuda::BaseType<DST_TYPE>, WB>(lane[c]);
            }
        }
        return cuda::SaturateCast<DST_TYPE>(work);
    }
};

// Each thread processes NIX columns strided by the total x-thread count, so consecutive threads stay
// coalesced while NIX independent loads are in flight at once. NIX is a measured constant, and NIX>=2
// only covers every column if the grid is sized to DivUp(size.x, NIX) x-threads.
template<int NIX, class SrcWrapper, class DstWrapper, class UnOp>
__global__ void ConvertFormat(SrcWrapper src, DstWrapper dst, UnOp op, int2 size)
{
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (src_y >= size.y)
        return;

    const int x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    using SrcPx = std::remove_cv_t<std::remove_reference_t<decltype(*src.ptr(batch_idx, src_y, x0))>>;
    SrcPx v[NIX];
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int x = x0 + i * stride;
        if (x < size.x)
            v[i] = *src.ptr(batch_idx, src_y, x);
    }
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int x = x0 + i * stride;
        if (x < size.x)
            *dst.ptr(batch_idx, src_y, x) = op(v[i]);
    }
}

inline int CurrentDeviceSMOrZero()
{
    int sm = 0;
    return cvcuda::priv::GetCurrentDeviceSM(sm) == cudaSuccess ? sm : 0;
}

constexpr int kPlanarBulkThreads = 512;
constexpr int kPlanarBulkChunks  = 2;
constexpr int kPlanarBulkPixels  = kPlanarBulkThreads * kPlanarBulkChunks * 4;

__global__ void __launch_bounds__(kPlanarBulkThreads)
    ConvertPlanarU8F32Bulk(const uchar4 *source, float4 *destination, Convertor<uchar4, float4, float> op)
{
    const int64_t chunkBase = static_cast<int64_t>(blockIdx.x) * kPlanarBulkThreads * kPlanarBulkChunks;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ __align__(128) float4 tile[kPlanarBulkThreads * kPlanarBulkChunks];
#    pragma unroll
    for (int chunk = 0; chunk < kPlanarBulkChunks; ++chunk)
    {
        const int lane = chunk * kPlanarBulkThreads + threadIdx.x;
        tile[lane]     = op(source[chunkBase + lane]);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const auto tileAddress = static_cast<unsigned int>(__cvta_generic_to_shared(tile));
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
            "cp.async.bulk.commit_group;\n"
            "cp.async.bulk.wait_group.read 0;\n"
            :
            : "l"(destination + chunkBase), "r"(tileAddress), "n"(kPlanarBulkPixels * sizeof(float))
            : "memory");
    }
    __syncthreads();
#else
#    pragma unroll
    for (int chunk = 0; chunk < kPlanarBulkChunks; ++chunk)
    {
        const int lane                = chunk * kPlanarBulkThreads + threadIdx.x;
        destination[chunkBase + lane] = op(source[chunkBase + lane]);
    }
#endif
}

inline bool TryConvertPlanarU8F32Bulk(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData, double alpha, double beta,
                                      cudaStream_t stream)
{
    if (CurrentDeviceSMOrZero() != 90)
        return false;

    auto in  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!in || !out || inData.dtype().numChannels() != 1 || outData.dtype().numChannels() != 1 || in->numCols() <= 0
        || in->numRows() <= 0 || in->numSamples() <= 0)
        return false;

    constexpr int64_t kOutputBytes = sizeof(float);
    const int64_t     planePixels  = static_cast<int64_t>(in->numCols()) * in->numRows();
    if (planePixels > cuda::TypeTraits<int32_t>::max / kOutputBytes / in->numSamples())
        return false;
    if (in->colStride() != 1 || out->colStride() != kOutputBytes || in->rowStride() != in->numCols()
        || out->rowStride() != static_cast<int64_t>(in->numCols()) * kOutputBytes || in->sampleStride() != planePixels
        || out->sampleStride() != planePixels * kOutputBytes)
        return false;

    const int64_t pixels = planePixels * in->numSamples();
    if (pixels % kPlanarBulkPixels != 0)
        return false;
    const auto source      = reinterpret_cast<std::uintptr_t>(inData.basePtr());
    const auto destination = reinterpret_cast<std::uintptr_t>(outData.basePtr());
    if (source % alignof(uchar4) != 0 || destination % alignof(float4) != 0)
        return false;
    if (source <= destination ? destination - source < static_cast<std::uintptr_t>(pixels)
                              : source - destination < static_cast<std::uintptr_t>(pixels * kOutputBytes))
        return false;

    Convertor<uchar4, float4, float> op{cuda::SaturateCast<float>(alpha), cuda::SaturateCast<float>(beta), false};
    ConvertPlanarU8F32Bulk<<<static_cast<int>(pixels / kPlanarBulkPixels), kPlanarBulkThreads, 0, stream>>>(
        reinterpret_cast<const uchar4 *>(inData.basePtr()), reinterpret_cast<float4 *>(outData.basePtr()), op);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

// COLS_PER_ELEM contiguous columns make up one MakeType<DT, NC> element: 1 for the interleaved path
// (one element is one pixel of NC channels), NC for the wide single-channel path (one element is NC
// contiguous columns of a 1-channel image). The caller guarantees cols % COLS_PER_ELEM == 0.
template<typename DT_SOURCE, typename DT_DEST, int NC, int NIX, int COLS_PER_ELEM = 1, bool kWrap = false>
void ConvertToScaleCNImpl(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                          const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 size       = {inAccess->numCols() / COLS_PER_ELEM, inAccess->numRows()};
    const int  batch_size = inAccess->numSamples();

    dim3 block(32, 8);
    dim3 grid(util::DivUp(util::DivUp(size.x, NIX), static_cast<int>(block.x)),
              util::DivUp(size.y, static_cast<int>(block.y)), batch_size);

    using DT_AB         = decltype(float() * DT_SOURCE() * DT_DEST());
    using SRC_DATA_TYPE = cuda::MakeType<DT_SOURCE, NC>;
    using DST_DATA_TYPE = cuda::MakeType<DT_DEST, NC>;

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB, kWrap> op;

    op.alpha    = cuda::SaturateCast<DT_AB>(alpha);
    op.beta     = cuda::SaturateCast<DT_AB>(beta);
    op.truncate = truncate;

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    auto src = cuda::CreateTensorWrapNHW<SRC_DATA_TYPE, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<DST_DATA_TYPE, int32_t>(outData);

    ConvertFormat<NIX><<<grid, block, 0, stream>>>(src, dst, op, size);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename DT_SOURCE, typename DT_DEST, int NC, bool kWrap = false>
void ConvertToScaleCN(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                      const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    // A100 and H100 profiles favor the legacy one-pixel policy for these wide float32 paths, while
    // SM86 regresses beyond paired noise. Keep NIX=4 on unmeasured architectures rather than
    // extrapolating the reference result. The short2 path uses NIX=1 only on SM80; NIX=4 is 5% faster
    // on SM90 and remains at ridge on SM86.
    constexpr bool kWideFloat32
        = std::is_same_v<
              DT_DEST,
              float> && ((NC == 3 && (std::is_same_v<DT_SOURCE, uint8_t> || std::is_same_v<DT_SOURCE, float>)) || (NC == 4 && std::is_same_v<DT_SOURCE, float>));
    if constexpr (kWideFloat32)
    {
        int sm = CurrentDeviceSMOrZero();
        if (sm == 80 || sm == 90 || (sm == 120 && NC == 3 && std::is_same_v<DT_SOURCE, uint8_t>))
        {
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1, 1, kWrap>(inData, outData, alpha, beta, truncate, stream);
            return;
        }
    }
    else if constexpr (std::is_same_v<DT_SOURCE, int16_t> && std::is_same_v<DT_DEST, float> && NC == 2)
    {
        if (CurrentDeviceSMOrZero() == 80)
        {
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1, 1, kWrap>(inData, outData, alpha, beta, truncate, stream);
            return;
        }
    }
    ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 4, 1, kWrap>(inData, outData, alpha, beta, truncate, stream);
}

// True when a single-channel convert can use VEC-wide contiguous vectorization: cols divisible by VEC,
// and the in/out base pointers + row/sample strides aligned to the VEC-wide vector type.
template<typename DT_SOURCE, typename DT_DEST, int VEC>
bool WideSingleChannelEligible(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData)
{
    auto in  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!in || !out)
        return false;
    if (in->numCols() % VEC != 0)
        return false;
    const std::uintptr_t aS = alignof(cuda::MakeType<DT_SOURCE, VEC>);
    const std::uintptr_t aD = alignof(cuda::MakeType<DT_DEST, VEC>);
    auto                 ok = [](std::uintptr_t base, int64_t rowS, int64_t smpS, std::uintptr_t a)
    {
        return ((base | static_cast<std::uintptr_t>(rowS) | static_cast<std::uintptr_t>(smpS)) & (a - 1)) == 0;
    };
    return ok(reinterpret_cast<std::uintptr_t>(inData.basePtr()), in->rowStride(), in->sampleStride(), aS)
        && ok(reinterpret_cast<std::uintptr_t>(outData.basePtr()), out->rowStride(), out->sampleStride(), aD);
}

// ---------------------------------------------------------------------------------------------------
// Element stream.
//
// ConvertTo applies the same alpha/beta to every element regardless of which channel it belongs to, so the whole
// tensor is one element stream and the pixel structure is irrelevant. That matters because a per-pixel access is as
// narrow as the pixel: a uchar3 has alignment 1 and a float3 alignment 4, so a three-channel pixel decomposes into
// three separate narrow accesses, and even a single-channel convert moves at most sizeof(element) bytes per access
// because cuda::MakeType tops out at four components. Profiling the resulting kernels found them issue-bound on
// access count rather than on bandwidth (uint8 to uint8: Compute SM 85.0% and Mem Pipes Busy 85.0% against DRAM
// 35.9%).
//
// The carrier moves a whole group as one native vector for the load and the store, and views the same registers as
// MakeType vectors for the conversion, which is what the arithmetic and the saturating cast are already written
// against. Viewing them as an array of scalars instead made the compiler extract every element with shifts and
// masks, which cost more than the wider access saved. The per-element expression is unchanged, so the result is
// bit-identical for every round mode.
// ---------------------------------------------------------------------------------------------------

// A group of EPG contiguous elements, viewed either as the native vector that moves it or as the MakeType vectors
// the conversion is written against. MakeType has no more than four components, so a longer group is viewed as
// several of them; a single-element group is the plain scalar, which is MakeType's zero-component form.
template<typename T, int EPG>
union FlatPack
{
    static constexpr int kLanes = EPG < 4 ? EPG : 4;
    static constexpr int kSubs  = EPG / kLanes;

    using WideT   = typename elem_stream::PackType<EPG *static_cast<int>(sizeof(T))>::type;
    using SubType = cuda::MakeType<T, kLanes == 1 ? 0 : kLanes>;

    WideT   wide;
    SubType v[kSubs];
};

// A group narrower than either element cannot carry one, so float64 on either side has no 4-byte arm.
template<int GroupBytes, typename DT_SOURCE, typename DT_DEST>
inline constexpr bool kFlatPackValid = sizeof(DT_SOURCE) <= GroupBytes && sizeof(DT_DEST) <= GroupBytes;

// A group is the run of contiguous elements one thread converts with one load and one store. Its length is set by
// the wider of the two element types, so that side fills exactly one GroupBytes access: a thread owns a contiguous
// run, so letting the wider side span more than one access would spread a warp's lanes over GroupBytes*32 bytes per
// access and waste most of every transaction. The narrower side then uses a proportionally shorter, still fully
// coalesced access. All the types have power-of-two sizes, so the divisions are exact.
template<int GroupBytes, typename DT_SOURCE, typename DT_DEST>
struct FlatPackPolicy
{
    static constexpr int kMaxSize
        = static_cast<int>(sizeof(DT_SOURCE) > sizeof(DT_DEST) ? sizeof(DT_SOURCE) : sizeof(DT_DEST));
    static constexpr int kWide = GroupBytes / kMaxSize;
    // A 2-byte-wide destination converts through cvt.rni.sat.u16.f32, whose pipe throughput -- not the access width
    // -- is the limit (48% conversion-pipe utilization, MIO-throttle the dominant stall, DRAM 55%), and widening
    // past the four-component group measured 2.4% slower there.
    static constexpr int kElems = (kMaxSize == 2 && kWide > 4) ? 4 : kWide;
    // Four source accesses in flight cover the load latency once a group carries at least 8 bytes; narrower source
    // groups move too little per access, so they carry twice as many.
    static constexpr int kGroups = (kElems * static_cast<int>(sizeof(DT_SOURCE)) < 8) ? 8 : 4;
};

template<typename DT_SOURCE, typename DT_DEST, int GroupBytes, bool kWrap, typename Enable = void>
struct ConvertStreamCarrier : elem_stream::DecliningCarrier
{
};

template<typename DT_SOURCE, typename DT_DEST, int GroupBytes, bool kWrap>
struct ConvertStreamCarrier<DT_SOURCE, DT_DEST, GroupBytes, kWrap,
                            std::enable_if_t<kFlatPackValid<GroupBytes, DT_SOURCE, DT_DEST>>>
{
    using Policy                = FlatPackPolicy<GroupBytes, DT_SOURCE, DT_DEST>;
    static constexpr int kElems = Policy::kElems;

    using SrcPack = FlatPack<DT_SOURCE, kElems>;
    using DstPack = FlatPack<DT_DEST, kElems>;
    using DT_AB   = decltype(float() * DT_SOURCE() * DT_DEST());

    static constexpr bool kSupported = true;
    using type                       = typename SrcPack::WideT;
    using dst_type                   = typename DstPack::WideT;

    struct Param
    {
        DT_AB alpha, beta;
        bool  truncate;
    };

    static Param MakeParam(double alpha, double beta, bool truncate)
    {
        return Param{cuda::SaturateCast<DT_AB>(alpha), cuda::SaturateCast<DT_AB>(beta), truncate};
    }

    inline static __device__ dst_type Apply(type v, Param p)
    {
        const Convertor<typename SrcPack::SubType, typename DstPack::SubType, DT_AB, kWrap> vecOp{p.alpha, p.beta,
                                                                                                  p.truncate};

        SrcPack in;
        in.wide = v;
        DstPack out;
#pragma unroll
        for (int q = 0; q < SrcPack::kSubs; ++q)
        {
            out.v[q] = vecOp(in.v[q]);
        }
        return out.wide;
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int srcBytes, Param p)
    {
        const Convertor<DT_SOURCE, DT_DEST, DT_AB, kWrap> elemOp{p.alpha, p.beta, p.truncate};

        const DT_SOURCE *se = reinterpret_cast<const DT_SOURCE *>(s);
        DT_DEST         *de = reinterpret_cast<DT_DEST *>(d);
        for (int e = 0; e < srcBytes / static_cast<int>(sizeof(DT_SOURCE)); ++e)
        {
            de[e] = elemOp(se[e]);
        }
    }
};

// Runs the convert as one element stream at the widest group the buffers' alignment allows. `numChannels` only sets
// how many bytes a column occupies, so the same instantiation serves every channel count.
template<typename DT_SOURCE, typename DT_DEST, int GroupBytes, bool kWrap>
bool TryConvertStream(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                      const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                      const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int numChannels, const double alpha,
                      const double beta, bool truncate, cudaStream_t stream)
{
    using Carrier = ConvertStreamCarrier<DT_SOURCE, DT_DEST, GroupBytes, kWrap>;
    if constexpr (!Carrier::kSupported)
    {
        return false;
    }
    else
    {
        return elem_stream::TryRunBytes<Carrier::Policy::kGroups, Carrier>(
            stream, inData, outData, inAccess, outAccess, numChannels * static_cast<int64_t>(sizeof(DT_SOURCE)),
            numChannels * static_cast<int64_t>(sizeof(DT_DEST)), false, Carrier::MakeParam(alpha, beta, truncate));
    }
}

// The stream only pays when a group carries more elements than a pixel does: an RGBA float32 pixel is already a
// 16-byte access, and streaming it measured 1.5% slower. Falls through to the per-pixel path when the layout is not
// a byte run or the buffers are not group-aligned.
template<typename DT_SOURCE, typename DT_DEST, bool kWrap>
bool TryConvertToStream(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        int numChannels, const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!inAccess || !outAccess)
    {
        return false;
    }

    // A widening convert sizes the group by the destination, so the source side is left carrying only
    // `GroupBytes / sizeof(DT_DEST)` elements -- a 4-byte load for uint8 into float32. That is too
    // little per access to cover the load latency, and it measured slower than the per-pixel path on
    // an A100: 0.98x for uint8 to float32 and 0.97x for the two-channel int16 to float32.
    if constexpr (sizeof(DT_DEST) > sizeof(DT_SOURCE))
    {
        return false;
    }
    // A 2-byte destination is limited by conversion-pipe throughput rather than access width -- the
    // same effect that already caps `FlatPackPolicy::kElems` at four there -- so a wider access buys
    // nothing. Measured across every such configuration on an A100: no gain anywhere (best 1.01x) and
    // down to 0.98x for float32 to float16. One-byte destinations, where the wins are (1.06x for
    // RGB8), and 4-byte ones keep the stream.
    else if constexpr (sizeof(DT_DEST) == 2)
    {
        return false;
    }

    if (FlatPackPolicy<16, DT_SOURCE, DT_DEST>::kElems > numChannels
        && TryConvertStream<DT_SOURCE, DT_DEST, 16, kWrap>(inData, outData, *inAccess, *outAccess, numChannels, alpha,
                                                           beta, truncate, stream))
    {
        return true;
    }

    if constexpr (kFlatPackValid<4, DT_SOURCE, DT_DEST>)
    {
        if (FlatPackPolicy<4, DT_SOURCE, DT_DEST>::kElems > numChannels
            && TryConvertStream<DT_SOURCE, DT_DEST, 4, kWrap>(inData, outData, *inAccess, *outAccess, numChannels,
                                                              alpha, beta, truncate, stream))
        {
            return true;
        }
    }
    return false;
}

template<typename DT_SOURCE, typename DT_DEST, bool kWrap = false>
void ConvertToScale(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    int numChannels, const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    if (TryConvertToStream<DT_SOURCE, DT_DEST, kWrap>(inData, outData, numChannels, alpha, beta, truncate, stream))
    {
        return;
    }

    // Measured on the reference SKUs, not a free parameter: kVec contiguous columns are viewed as one
    // MakeType<DT, kVec> element so the existing vector Convertor cuts the per-element instruction
    // count on the otherwise issue-bound single-channel path.
    constexpr int kVec = 4;

    switch (numChannels)
    {
    case 1:
        // Wide contiguous vectorization (issue-bound fix); falls back to the scalar single-channel path
        // when cols isn't kVec-divisible or the buffers aren't vector-aligned.
        if (WideSingleChannelEligible<DT_SOURCE, DT_DEST, kVec>(inData, outData))
        {
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, kVec, 4, kVec, kWrap>(inData, outData, alpha, beta, truncate,
                                                                           stream);
            return;
        }
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 1, kWrap>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 2:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 2, kWrap>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 3:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 3, kWrap>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 4:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 4, kWrap>(inData, outData, alpha, beta, truncate, stream);
        return;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown number of channels");
    }
}

// The element type is semantic here, not a width carrier: alpha * x + beta is evaluated in a work type
// derived from both ends, and the saturation bounds come from the destination type. So the dispatch has
// to name both types, exactly as the legacy 8x8 function-pointer table did.
using ElemType = cvcuda::priv::legacy_format::DataTypeCode;
using cvcuda::priv::legacy_format::ClassifyDataType;

// Calling this once per axis reproduces the legacy funcs[8][8] cross product: that table had no null
// cells, so every (source, destination) pair is a live instantiation and there are no holes to skip.
template<typename Cb>
inline void WithElemType(ElemType type, const Cb &cb)
{
    switch (type)
    {
    case ElemType::kU8:
        cb(uchar{});
        return;
    case ElemType::kS8:
        cb(schar{});
        return;
    case ElemType::kU16:
        cb(ushort{});
        return;
    case ElemType::kS16:
        cb(short{});
        return;
    case ElemType::kS32:
        cb(int{});
        return;
    case ElemType::kF32:
        cb(float{});
        return;
    case ElemType::kF64:
        cb(double{});
        return;
    case ElemType::kF16:
        cb(__half{});
        return;
    }
}

// F16 pairs instantiate real __half kernels whose work type resolves to float (double only
// when paired with F64) via the DT_AB deduction.  kWrap is a compile-time template parameter
// (a runtime branch measured a 0.8–2.4% regression); kWrap=true is only instantiated for
// U8/S8/U16/S16 via kWrapEligibleDst — other types always use kWrap=false.
inline void ConvertToScaleDispatch(ElemType inType, ElemType outType, const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData, int numChannels, const double alpha,
                                   const double beta, bool truncate, bool wrap, cudaStream_t stream)
{
    WithElemType(inType,
                 [&](auto srcVal)
                 {
                     WithElemType(outType,
                                  [&](auto dstVal)
                                  {
                                      using Src = decltype(srcVal);
                                      using Dst = decltype(dstVal);
                                      if constexpr (kWrapEligibleDst<Dst>)
                                      {
                                          if (wrap)
                                          {
                                              ConvertToScale<Src, Dst, true>(inData, outData, numChannels, alpha, beta,
                                                                             truncate, stream);
                                              return;
                                          }
                                      }
                                      ConvertToScale<Src, Dst, false>(inData, outData, numChannels, alpha, beta,
                                                                      truncate, stream);
                                  });
                 });
}

// GetLegacyDataFormat(layout) admitted exactly these four layouts and threw on anything else, before
// any other validation ran; the interleaved/planar split is all the caller needs from it.
inline bool IsPlanarLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
        return true;
    if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC)
        return false;
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline void RunConvertTo(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                         const nvcv::TensorDataStridedCuda &outData, const double alpha, const double beta,
                         NVCVRoundMode roundMode)
{
    if (roundMode != NVCV_ROUND_NEAREST && roundMode != NVCV_ROUND_TRUNCATE && roundMode != NVCV_ROUND_NEAREST_WRAP
        && roundMode != NVCV_ROUND_TRUNCATE_WRAP)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid round mode %d",
                              static_cast<int>(roundMode));
    }

    const bool truncate = (roundMode == NVCV_ROUND_TRUNCATE || roundMode == NVCV_ROUND_TRUNCATE_WRAP);
    const bool wrap     = (roundMode == NVCV_ROUND_NEAREST_WRAP || roundMode == NVCV_ROUND_TRUNCATE_WRAP);

    // Validation order matches the legacy path exactly: both layouts, then both data types, then the
    // interleaved/planar agreement, then the shape and channel-count checks. Every legacy ErrorCode
    // (INVALID_DATA_FORMAT / INVALID_DATA_SHAPE / INVALID_DATA_TYPE / INVALID_PARAMETER) mapped to
    // NVCV_ERROR_INVALID_ARGUMENT through TranslateError, and the helper conversions that could throw
    // ahead of them threw that same status, so every rejection keeps its observable status.
    const bool inPlanar  = IsPlanarLayout(inData.layout());
    const bool outPlanar = IsPlanarLayout(outData.layout());

    const ElemType inType  = ClassifyDataType(inData.dtype());
    const ElemType outType = ClassifyDataType(outData.dtype());

    if (wrap && outType == ElemType::kS32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Wrap round modes are not supported for 32-bit integer destinations: the float32 working "
                              "arithmetic cannot represent the 2^31 wrap period without quantization error. "
                              "Use an 8- or 16-bit integer destination, or a saturating round mode.");
    }

    // Input and output must both be interleaved or both planar; the operator does not transpose.
    if (inPlanar != outPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat, must be kHWC/kNHWC or kCHW/kNCHW with matching layouts");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (outAccess->numRows() != rows || outAccess->numCols() != cols || outAccess->numSamples() != batch
        || outAccess->numChannels() != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "input/output shape is different (N = %d, H = %d, W = %d, C = %d)/(N = %d, H = %d, W = "
                              "%d, C = %d)",
                              batch, rows, cols, channels, static_cast<int>(outAccess->numSamples()),
                              static_cast<int>(outAccess->numRows()), static_cast<int>(outAccess->numCols()),
                              static_cast<int>(outAccess->numChannels()));
    }

    if (channels > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (inPlanar)
    {
        // The higher-level PlanarSingleChannelViews wrapper is deliberately not used: it rejects
        // 2-channel planar, which this operator accepts, and orders its checks differently.
        //
        // Each channel plane is processed as an independent single-channel image flattened into the
        // N*C sample dimension. That flattened count becomes the kernel's grid z-dimension, capped at
        // CUDA's 65535 limit; compute it in 64-bit to avoid overflow.
        constexpr int64_t kMaxGridZ   = 65535;
        const int64_t     planarBatch = static_cast<int64_t>(channels) * batch;
        if (planarBatch > kMaxGridZ)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar ConvertTo requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
        }

        // PlanarAsSingleChannelView rejects the batched non-tightly-packed case its uniform flatten
        // cannot represent, with the same status the legacy pair of checks raised.
        auto inView  = cvcuda::priv::PlanarAsSingleChannelView(inData, *inAccess);
        auto outView = cvcuda::priv::PlanarAsSingleChannelView(outData, *outAccess);
        if (inData.layout() == nvcv::TENSOR_NCHW && outData.layout() == nvcv::TENSOR_NCHW && inType == ElemType::kU8
            && outType == ElemType::kF32 && TryConvertPlanarU8F32Bulk(inView, outView, alpha, beta, stream))
        {
            return;
        }
        ConvertToScaleDispatch(inType, outType, inView, outView, 1, alpha, beta, truncate, wrap, stream);
        return;
    }

    ConvertToScaleDispatch(inType, outType, inData, outData, channels, alpha, beta, truncate, wrap, stream);
}

} // namespace

namespace cvcuda::priv {

void ConvertTo::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double alpha,
                           const double beta, NVCVRoundMode roundMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ConvertTo::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RunConvertTo(stream, *inData, *outData, alpha, beta, roundMode);
}

} // namespace cvcuda::priv
