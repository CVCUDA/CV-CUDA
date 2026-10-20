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
#include "OpPosterize.hpp"

#include "ElementStreamCommon.cuh"
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

struct PosterizeFloatParam
{
    float levels;
    float reciprocal;
};

template<typename BT>
using PosterizeParamT = std::conditional_t<cuda::detail::IsFloatingPointV<BT>, PosterizeFloatParam, BT>;

template<typename BT>
inline __device__ BT PosterizeValue(BT v, PosterizeParamT<BT> param)
{
    if constexpr (std::is_same_v<BT, double>)
    {
        const uint32_t high      = static_cast<uint32_t>(__double2hiint(v));
        const uint32_t low       = static_cast<uint32_t>(__double2loint(v));
        const uint32_t magnitude = high & 0x7fffffffu;
        if (magnitude > 0x7ff00000u || (magnitude == 0x7ff00000u && low != 0))
        {
            return v + v;
        }

        // Quantized results have at most eight significant bits, all in the high word.
        // Clearing the fractional significand bits implements floor(v * 2^bits) / 2^bits
        // without routing conversions through the low-throughput FP64 pipeline.
        const int bits   = (__float_as_uint(param.levels) >> 23) - 127;
        uint32_t  result = 0;
        if (high < 0x3ff00000u)
        {
            const int exponent        = high >> 20;
            const int minimumExponent = 1023 - bits;
            if (exponent >= minimumExponent)
            {
                result = high & (0xffffffffu << (minimumExponent + 20 - exponent));
            }
        }
        else if (high <= 0x7ff00000u)
        {
            result = bits == 0 ? 0 : 0x3ff00000u - (1u << (21 - bits));
        }
        if (magnitude == 0 && low == 0)
        {
            result = high;
        }
        return __hiloint2double(result, 0);
    }
    else if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        const float value = static_cast<float>(v);
        float       q     = ::floorf(value * param.levels);
        q                 = q < 0.f ? 0.f : (q > param.levels - 1.f ? param.levels - 1.f : q);
        return cuda::SaturateCast<BT>(q * param.reciprocal);
    }
    else
    {
        return static_cast<BT>(v & param);
    }
}

template<typename T, typename Param>
inline __device__ T PosterizeElem(T pixel, Param param)
{
    static constexpr int numChannels = cuda::NumElements<T>;
    T                    out{};
#pragma unroll
    for (int c = 0; c < numChannels; ++c)
    {
        cuda::GetElement(out, c) = PosterizeValue<cuda::BaseType<T>>(cuda::GetElement(pixel, c), param);
    }
    return out;
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper, typename BT>
inline __device__ void DoPosterize(SrcWrapper src, DstWrapper dst, const int2 size, const int p, BT mask)
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
    if constexpr (!IsPlanar && alignof(SrcT) > alignof(cuda::BaseType<SrcT>))
    {
        const auto *srcPixel = src.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        auto       *dstPixel = dst.ptr(nhwCoord.z, nhwCoord.y, nhwCoord.x);
        if ((reinterpret_cast<uintptr_t>(srcPixel) | reinterpret_cast<uintptr_t>(dstPixel)) % alignof(SrcT) != 0)
        {
            same_shape::StoreUnalignedPixel(dstPixel,
                                            PosterizeElem<DstT>(same_shape::LoadUnalignedPixel(srcPixel), mask));
            return;
        }
    }
    auto coord = same_shape::GetCoordForLayout<IsPlanar>(nhwCoord, p);
    dst[coord] = PosterizeElem<DstT>(src[coord], mask);
}

// Posterize kernel -----------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename BT>
__global__ void Posterize(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, BT mask)
{
    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoPosterize<isPlanar>(src, dst, size, p, mask); });
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper, typename BT>
__global__ void Posterize(SrcWrapper src, DstWrapper dst, int numPlanes, BT mask)
{
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    same_shape::ForEachPlane<isPlanar>(numPlanes, [=](int p) { DoPosterize<isPlanar>(src, dst, size, p, mask); });
}

// Floating-point levels and their reciprocal are powers of two, so both are exact.
// The integer path uses a wide intermediate so the shift
// never hits the type-width UB (bits == 0 -> mask 0; bits == W -> all ones).
template<typename BT>
inline PosterizeParamT<BT> PosterizeParam(int bits)
{
    if constexpr (cuda::detail::IsFloatingPointV<BT>)
    {
        const float levels = static_cast<float>(1u << bits);
        return {levels, 1.f / levels};
    }
    else
    {
        constexpr int W = static_cast<int>(sizeof(BT) * 8);
        uint32_t      m;
        if (bits <= 0)
        {
            m = 0u;
        }
        else if (bits >= W)
        {
            m = ~0u;
        }
        else
        {
            m = ~((1u << (W - bits)) - 1u);
        }
        return static_cast<BT>(m);
    }
}

// Vectorized planar / 1-channel element map ----------------------------------------------
//
// Wider loads keep multiple requests outstanding on planar and single-channel paths;
// using the scalar value operation also keeps partial vector groups exact.
template<typename BT>
struct PosterizeVec4Op
{
    using Param = PosterizeParamT<BT>;

    Param mask;

    inline __device__ explicit PosterizeVec4Op(Param m)
        : mask(m)
    {
    }

    inline __device__ BT operator()(BT v) const
    {
        return PosterizeValue<BT>(v, mask);
    }
};

// Every element uses the same parameter, so carriers may cross pixel and channel boundaries.
constexpr int kPosterizeStreamGrps = 2;

template<typename BT, typename Enable = void>
struct PosterizeStreamCarrier : elem_stream::DecliningCarrier
{
};

template<typename BT>
struct PosterizeStreamCarrier<BT,
                              std::enable_if_t<std::is_same_v<BT, unsigned char> || std::is_same_v<BT, unsigned short>>>
{
    static constexpr bool kSupported = true;
    using type                       = uint4;

    struct Param
    {
        uint32_t maskWord;
    };

    static Param MakeParam(BT mask)
    {
        return Param{elem_stream::BroadcastToWord(mask)};
    }

    inline static __device__ type Apply(type v, Param p)
    {
        return type{v.x & p.maskWord, v.y & p.maskWord, v.z & p.maskWord, v.w & p.maskWord};
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param p)
    {
        elem_stream::TailElementwise<BT>(
            s, d, bytes, [p](BT v) { return static_cast<BT>(v & elem_stream::NarrowFromWord<BT>(p.maskWord)); });
    }
};

template<typename BT>
struct PosterizeStreamCarrier<BT, std::enable_if_t<std::is_same_v<BT, float> || std::is_same_v<BT, double>>>
{
    static constexpr bool kSupported = true;
    // Native vectors retain wide memory instructions; aligned arrays are scalarized by nvcc.
    using type  = uint4;
    using Param = PosterizeFloatParam;

    static Param MakeParam(Param param)
    {
        return param;
    }

    inline static __device__ type Apply(type v, Param param)
    {
        if constexpr (std::is_same_v<BT, double>)
        {
#pragma unroll
            for (int i = 0; i < 4; i += 2)
            {
                const double value         = __hiloint2double(cuda::GetElement(v, i + 1), cuda::GetElement(v, i));
                const double result        = PosterizeValue<double>(value, param);
                cuda::GetElement(v, i)     = __double2loint(result);
                cuda::GetElement(v, i + 1) = __double2hiint(result);
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const uint32_t word    = cuda::GetElement(v, i);
                cuda::GetElement(v, i) = __float_as_uint(PosterizeValue<float>(__uint_as_float(word), param));
            }
        }
        return v;
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param param)
    {
        elem_stream::TailElementwise<BT>(s, d, bytes, [param](BT v) { return PosterizeValue<BT>(v, param); });
    }
};

template<typename BT>
struct PosterizeStreamCarrier<BT, std::enable_if_t<cuda::detail::IsHalfV<BT>>>
{
    static constexpr bool kSupported = true;
    using type                       = uint4;

    struct Param
    {
        __half2             levels;
        __half2             reciprocal;
        __half2             upper;
        PosterizeFloatParam scalar;
    };

    static Param MakeParam(PosterizeFloatParam param)
    {
        return {__float2half2_rn(param.levels), __float2half2_rn(param.reciprocal),
                __float2half2_rn(param.levels - 1.f), param};
    }

    inline static __device__ type Apply(type v, Param param)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const uint32_t word = cuda::GetElement(v, i);
            const __half2  pair{
                __half2_raw{static_cast<uint16_t>(word), static_cast<uint16_t>(word >> 16)}
            };
            const __half2  zero         = __float2half2_rn(0.f);
            const uint32_t zeroSigns    = __heq2_mask(pair, zero) & word & 0x80008000u;
            __half2        q            = h2floor(__hmul2(pair, param.levels));
            q                           = __hmin2_nan(__hmax2_nan(q, zero), param.upper);
            const __half2_raw quantized = __hmul2(q, param.reciprocal);
            // min/max choose positive zero; retain an input negative zero as the scalar path does.
            cuda::GetElement(v, i) = quantized.x | (static_cast<uint32_t>(quantized.y) << 16) | zeroSigns;
        }
        return v;
    }

    inline static __device__ void Tail(const unsigned char *s, unsigned char *d, int bytes, Param param)
    {
        elem_stream::TailElementwise<BT>(s, d, bytes, [param](BT v) { return PosterizeValue<BT>(v, param.scalar); });
    }
};

// Posterize is cheap enough that independently loading the var-shape plane descriptor in every lane
// dominates the useful work. One x-warp owns a row, broadcasts that descriptor, and keeps four word
// loads in flight per lane; wrapped images retain a scalar path whenever a row is not word-aligned.
template<int NGroups>
__global__ void PosterizePlanarVarShapePacked(cuda::ImageBatchVarShapeWrap<const unsigned char> src,
                                              cuda::ImageBatchVarShapeWrap<unsigned char> dst, int numPlanes,
                                              uint32_t maskWord)
{
    const int              g0        = blockIdx.x * blockDim.x * NGroups + threadIdx.x;
    const int              y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int              nc        = blockIdx.z;
    const int              batch     = nc / numPlanes;
    const int              channel   = nc % numPlanes;
    constexpr unsigned int kFullWarp = 0xffffffffu;
    const bool             leader    = threadIdx.x == 0;
    NVCVImagePlaneStrided  srcPlane{};
    NVCVImagePlaneStrided  dstPlane{};
    if (leader)
    {
        srcPlane = src.plane(batch, channel);
        dstPlane = dst.plane(batch, channel);
    }

    const int       width     = __shfl_sync(kFullWarp, dstPlane.width, 0);
    const int       height    = __shfl_sync(kFullWarp, dstPlane.height, 0);
    const int       srcStride = __shfl_sync(kFullWarp, srcPlane.rowStride, 0);
    const int       dstStride = __shfl_sync(kFullWarp, dstPlane.rowStride, 0);
    const uintptr_t srcBase
        = __shfl_sync(kFullWarp, leader ? reinterpret_cast<uintptr_t>(srcPlane.basePtr) : uintptr_t{0}, 0);
    const uintptr_t dstBase
        = __shfl_sync(kFullWarp, leader ? reinterpret_cast<uintptr_t>(dstPlane.basePtr) : uintptr_t{0}, 0);

    if (g0 * 4 >= width || y >= height)
    {
        return;
    }

    const unsigned char *srow = reinterpret_cast<const unsigned char *>(srcBase + static_cast<int64_t>(y) * srcStride);
    unsigned char       *drow = reinterpret_cast<unsigned char *>(dstBase + static_cast<int64_t>(y) * dstStride);
    const bool           wordAligned = (reinterpret_cast<uintptr_t>(srow) | reinterpret_cast<uintptr_t>(drow)) % 4 == 0;

    if (!wordAligned)
    {
#pragma unroll
        for (int i = 0; i < NGroups; ++i)
        {
            const int begin = (g0 + i * blockDim.x) * 4;
            const int end   = min(begin + 4, width);
            for (int x = begin; x < end; ++x)
            {
                drow[x] = static_cast<unsigned char>(srow[x] & maskWord);
            }
        }
        return;
    }

    int      begin[NGroups];
    bool     full[NGroups];
    uint32_t value[NGroups];
#pragma unroll
    for (int i = 0; i < NGroups; ++i)
    {
        begin[i] = (g0 + i * blockDim.x) * 4;
        full[i]  = begin[i] + 4 <= width;
        if (full[i])
        {
            value[i] = *reinterpret_cast<const uint32_t *>(srow + begin[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < NGroups; ++i)
    {
        if (full[i])
        {
            *reinterpret_cast<uint32_t *>(drow + begin[i]) = value[i] & maskWord;
        }
        else if (begin[i] < width)
        {
            for (int x = begin[i]; x < width; ++x)
            {
                drow[x] = static_cast<unsigned char>(srow[x] & maskWord);
            }
        }
    }
}

// Run Posterize kernel -------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunPosterize(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, int bits)
{
    using BT                   = cuda::BaseType<ValueT>;
    constexpr bool kIsIntegral = std::is_integral_v<BT>;
    // Integers admit one level per value bit. Floating point has no bit layout to keep, so the
    // range is the reference's: torchvision posterizes a float image into at most 2^8 levels.
    constexpr int  kMaxBits = kIsIntegral ? static_cast<int>(sizeof(BT) * 8) : 8;
    if (bits < 0 || bits > kMaxBits)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Posterize 'bits' must be in [0, %d] for this data type", kMaxBits);
    }
    const auto mask = PosterizeParam<BT>(bits);

    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto       srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto       dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        const auto prep      = same_shape::MakeTensorLaunchPrep(*srcAccess, *dstAccess);
        const int2 size      = prep.size;

        using StrideType = int32_t;

        if (elem_stream::TryRunCarrier<kPosterizeStreamGrps, PosterizeStreamCarrier, isPlanar, ValueT>(
                stream, srcData, dstData, *srcAccess, *dstAccess, mask))
        {
            return;
        }

        if constexpr (!isPlanar)
        {
            bool launchedVec = false;
            // A single interleaved channel is byte-identical to a plane; use the same
            // vector fallback when the wider element stream cannot describe its strides.
            if constexpr (cuda::NumElements<ValueT> == 1)
            {
                constexpr int NGROUP = 4;
                launchedVec          = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, PosterizeVec4Op<BT>>(
                    stream, false, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block, mask);
            }
            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Posterize<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, 1, mask);
            }
        }
        else
        {
            const int     numPlanes   = srcAccess->numPlanes();
            constexpr int NGROUP      = 4;
            const bool    launchedVec = same_shape::TryRunVec4TensorFor<NGROUP, BT, StrideType, PosterizeVec4Op<BT>>(
                stream, true, srcData.basePtr(), dstData.basePtr(), *srcAccess, *dstAccess, size, block, mask);
            if (!launchedVec)
            {
                auto [src, dst] = same_shape::MakePlanarWrapPair<ValueT, ValueT, StrideType>(srcData, *srcAccess,
                                                                                             dstData, *dstAccess);
                Posterize<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, size, numPlanes, mask);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        const auto prep = same_shape::MakeVarShapeLaunchPrep(dstData);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        // U8 benefits from broadcasting its plane descriptor, while U16 and F16
        // keep one vector group live at a time to bound register residency.
        if constexpr (cuda::NumElements<ValueT> == 1 && sizeof(BT) <= 2
                      && (isPlanar || std::is_same_v<BT, unsigned short> || cuda::detail::IsHalfV<BT>))
        {
            if constexpr (std::is_same_v<BT, unsigned char>)
            {
                constexpr int NGroups = 4;
                const dim3    packedBlock(32, 16, 1);
                if (same_shape::TryRunVec4VarShape<NGroups, BT>(
                        srcData, dstData, numPlanes, prep.maxSize, packedBlock,
                        [&](auto srcV, auto dstV, int planes, dim3 grid)
                        {
                            PosterizePlanarVarShapePacked<NGroups><<<grid, packedBlock, 0, stream>>>(
                                srcV, dstV, planes, elem_stream::BroadcastToWord(mask));
                        }))
                {
                    NVCV_CHECK_THROW(cudaGetLastError());
                    return;
                }
            }
            else
            {
                static_assert(std::is_same_v<BT, unsigned short> || cuda::detail::IsHalfV<BT>);
                constexpr int NGroups = 4;
                if (same_shape::TryRunVec4VarShapeFor<NGroups, BT, PosterizeVec4Op<BT>, /*Serial=*/true>(
                        stream, srcData, dstData, numPlanes, prep.maxSize, block, mask))
                {
                    NVCV_CHECK_THROW(cudaGetLastError());
                    return;
                }
            }
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Posterize<isPlanar><<<prep.scalarGrid(block), block, 0, stream>>>(src, dst, numPlanes, mask);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

} // anonymous namespace

namespace cvcuda::priv {

Posterize::Posterize() {}

// Tensor input variant
void Posterize::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, int32_t bits) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Posterize::operator()[Tensor]");
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto info    = same_shape::ValidateSrcDstTensors(srcData, dstData);

    same_shape::DispatchTypeWithF64AndChannels(info.numInterleavedChannels, info.numPlanes, info.srcDtype, "Posterize",
                                               [&stream, &srcData, &dstData, bits](auto dummyVal, auto isPlanar)
                                               {
                                                   using ValueT   = decltype(dummyVal);
                                                   using IsPlanar = decltype(isPlanar);
                                                   RunPosterize<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                         bits);
                                               });
}

// VarShape input variant
void Posterize::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                           const nvcv::ImageBatchVarShape &dst, int32_t bits) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Posterize::operator()[ImageBatchVarShape]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto srcDstData = same_shape::ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    same_shape::DispatchTypeWithF64AndChannels(numInterleavedChannels, numPlanes, dtype, "Posterize",
                                               [&stream, &srcDstData, bits](auto dummyVal, auto isPlanar)
                                               {
                                                   using ValueT             = decltype(dummyVal);
                                                   using IsPlanar           = decltype(isPlanar);
                                                   auto &[srcData, dstData] = srcDstData;
                                                   RunPosterize<IsPlanar::value, ValueT>(stream, *srcData, *dstData,
                                                                                         bits);
                                               });
}

} // namespace cvcuda::priv
