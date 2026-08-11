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

#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// Photometric-negative bound per base type: dtype max for unsigned integers, 1.0 for float.
template<typename BT>
inline __host__ __device__ BT InvertBound()
{
    if constexpr (std::is_floating_point_v<BT>)
    {
        return BT(1);
    }
    else
    {
        return cuda::TypeTraits<BT>::max;
    }
}

// Threshold comparison hoisted out of the FP64 pipe. The gold semantics are
// `static_cast<double>(v) >= threshold`; per element that runs on the (1/64-rate on consumer SM)
// double pipe and makes Solarize compute-bound (NCU: ~87% SM, ~26% DRAM, decoupled-math stalls).
// For the discrete supported value types this is *exactly* equivalent to comparing v against a
// precomputed value-domain boundary in the integer / float32 domain, so the result stays bit-identical
// to the double-domain gold reference while the per-element FP64 disappears:
//   - integer v:  (double)v >= threshold  <=>  (long long)v >= ceil(threshold)
//   - float32 v:  (double)v >= threshold  <=>  v >= vmin, vmin = smallest float >= threshold
// The boundary is computed once per thread (one FP64 op, amortized) instead of once per element.
template<typename BT>
struct SolarizeThreshold
{
    long long vmin_i = 0;
    float     vmin_f = 0.0f;

    inline __device__ explicit SolarizeThreshold(double threshold)
    {
        if constexpr (std::is_floating_point_v<BT>)
        {
            float f = static_cast<float>(threshold);
            if (static_cast<double>(f) < threshold) // round up to the smallest float >= threshold
            {
                f = nextafterf(f, 3.402823466e38f);
            }
            vmin_f = f;
        }
        else if (isnan(threshold))
        {
            // (v >= NaN) is always false; use the max sentinel so no integer value inverts,
            // and avoid the undefined behavior of casting a NaN to long long.
            vmin_i = 9223372036854775807LL;
        }
        else
        {
            const double c = ceil(threshold);
            vmin_i         = c >= 9.2233720368547758e18
                               ? 9223372036854775807LL
                               : (c <= -9.2233720368547758e18 ? (-9223372036854775807LL - 1) : static_cast<long long>(c));
        }
    }

    inline __device__ bool ge(BT v) const
    {
        if constexpr (std::is_floating_point_v<BT>)
        {
            return v >= vmin_f;
        }
        else
        {
            return static_cast<long long>(v) >= vmin_i;
        }
    }
};

// out = (in >= threshold) ? (bound - in) : in, per channel component (threshold hoisted, see above).
template<typename BT>
inline __device__ BT SolarizeApply(BT v, BT bound, const SolarizeThreshold<BT> &th)
{
    return th.ge(v) ? static_cast<BT>(bound - v) : v;
}

// Scalar per-pixel path (multi-channel interleaved + planar fallback). For multi-channel INTEGER
// pixels the FP64 threshold compares dominate (uchar3/uchar4 NHWC are ~36% BWUtil, FP64-bound), so
// hoist the boundary once per pixel (one ceil amortized over 3-4 integer compares = net win). For
// 1-channel (hoist would add a ceil with nothing to amortize -> regression) and float (already
// bandwidth-bound at the ridge) keep the original double compare. Bit-identical to the gold either way.
template<typename T>
inline __device__ T SolarizeElem(T pixel, double threshold)
{
    using BT                         = cuda::BaseType<T>;
    static constexpr int numChannels = cuda::NumElements<T>;
    const BT             bound       = InvertBound<BT>();

    T out{};
    if constexpr (numChannels > 1 && !std::is_floating_point_v<BT>)
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

template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> GetCoordForLayout(int3 nhwCoord, int p)
{
    if constexpr (!IsPlanar)
    {
        return nhwCoord;
    }
    else
    {
        return {nhwCoord.x, nhwCoord.y, p, nhwCoord.z};
    }
}

template<bool IsPlanar, class SrcWrapper, class DstWrapper>
inline __device__ void DoSolarize(SrcWrapper src, DstWrapper dst, const int2 size, const int p, double threshold)
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
    auto coord = GetCoordForLayout<IsPlanar>(nhwCoord, p);
    dst[coord] = SolarizeElem<DstT>(src[coord], threshold);
}

// Solarize kernel ------------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Solarize(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes, double threshold)
{
    assert(isPlanar || numPlanes == 1);
    if constexpr (!isPlanar)
    {
        DoSolarize<isPlanar>(src, dst, size, 0, threshold);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoSolarize<isPlanar>(src, dst, size, p, threshold);
        }
    }
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Solarize(SrcWrapper src, DstWrapper dst, int numPlanes, double threshold)
{
    assert(isPlanar || numPlanes == 1);
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    if constexpr (!isPlanar)
    {
        DoSolarize<isPlanar>(src, dst, size, 0, threshold);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoSolarize<isPlanar>(src, dst, size, p, threshold);
        }
    }
}

// Vectorized planar / 1-channel kernels --------------------------------------------------
//
// The scalar kernels above move one element/thread/plane, leaving the small-dtype planar and
// 1-channel-interleaved paths memory-latency bound (long-scoreboard stalls, low BWUtil). These map
// each (sample, plane) to grid.z and have each thread issue NGROUP wide vector loads (uchar4 /
// ushort4 / float4) before compute, raising memory-level parallelism. Per element bit-identical to
// SolarizeElem (same (in>=threshold)?(bound-in):in). Modeled on legacy/normalize_planar.cuh; caller
// guards sizeof(Vec4)-aligned base+strides with a scalar fallback; per-thread tail handles width%4.
template<typename T, int Size = sizeof(T)>
struct SolarizeVec4Type;

template<typename T>
struct SolarizeVec4Type<T, 1>
{
    using type = uchar4;
};

template<typename T>
struct SolarizeVec4Type<T, 2>
{
    using type = ushort4;
};

template<typename T>
struct SolarizeVec4Type<T, 4>
{
    using type = float4;
};

template<int NGROUP, typename BT>
__global__ void SolarizePlanarVec4Kernel(cuda::Tensor4DWrap<const BT, int32_t> src, cuda::Tensor4DWrap<BT, int32_t> dst,
                                         int4 inout_size, double threshold)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;
    const int width   = inout_size.w;

    if (g0 * 4 >= width || src_y >= inout_size.z)
    {
        return;
    }

    using Vec4                        = typename SolarizeVec4Type<BT>::type;
    const BT                    bound = InvertBound<BT>();
    const SolarizeThreshold<BT> th(threshold);

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
        {
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx[i]));
        }
    }

#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            Vec4 out4;
            out4.x                                                           = SolarizeApply<BT>(in4[i].x, bound, th);
            out4.y                                                           = SolarizeApply<BT>(in4[i].y, bound, th);
            out4.z                                                           = SolarizeApply<BT>(in4[i].z, bound, th);
            out4.w                                                           = SolarizeApply<BT>(in4[i].w, bound, th);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, src_y, x) = SolarizeApply<BT>(*src.ptr(batch, channel, src_y, x), bound, th);
            }
        }
    }
}

template<int NGROUP, typename BT>
__global__ void SolarizePlanarVarShapeVec4Kernel(cuda::ImageBatchVarShapeWrap<const BT> src,
                                                 cuda::ImageBatchVarShapeWrap<BT> dst, int num_channels,
                                                 double threshold)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / num_channels;
    const int channel = nc % num_channels;
    const int width   = dst.width(batch, channel);

    if (g0 * 4 >= width || dst_y >= dst.height(batch, channel))
    {
        return;
    }

    using Vec4                        = uchar4; // 1-byte planes only (caller guards sizeof(BT) == 1)
    const BT                    bound = InvertBound<BT>();
    const SolarizeThreshold<BT> th(threshold);

    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
        {
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, dst_y, cx[i]));
        }
    }

#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            Vec4 out4;
            out4.x                                                           = SolarizeApply<BT>(in4[i].x, bound, th);
            out4.y                                                           = SolarizeApply<BT>(in4[i].y, bound, th);
            out4.z                                                           = SolarizeApply<BT>(in4[i].z, bound, th);
            out4.w                                                           = SolarizeApply<BT>(in4[i].w, bound, th);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, dst_y, x) = SolarizeApply<BT>(*src.ptr(batch, channel, dst_y, x), bound, th);
            }
        }
    }
}

// Run Solarize kernel --------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunSolarize(cudaStream_t stream, const SrcData &srcData, const DstData &dstData, double threshold)
{
    dim3 block(32, 4, 1);
    if constexpr (std::is_same_v<SrcData, nvcv::TensorDataStridedCuda>)
    {
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
        int2 size      = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});

        // Each sample maps to one grid.z block (planar channels are looped inside the kernel, so
        // grid.z is the sample count, not N*C); CUDA caps grid.z at 65535.
        if (srcAccess->numSamples() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        dim3 grid(util::DivUp(size.x, block.x), util::DivUp(size.y, block.y), srcAccess->numSamples());

        int64_t inMaxStride  = srcAccess->sampleStride() * srcAccess->numSamples();
        int64_t outMaxStride = dstAccess->sampleStride() * dstAccess->numSamples();
        if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }
        using StrideType = int32_t;

        using BT = cuda::BaseType<ValueT>;
        if constexpr (!isPlanar)
        {
            bool launchedVec = false;
            // 1-channel interleaved (e.g. U16) is byte-identical to a single plane and latency-bound;
            // route it through the vectorized planar kernel (C == 1). Multi-channel interleaved
            // (uchar3/uchar4/float3/float4) already moves 3-4 B/thread at the ridge -> keep scalar.
            if constexpr (cuda::NumElements<ValueT> == 1)
            {
                using Vec4            = typename SolarizeVec4Type<BT>::type;
                constexpr int NGROUP  = 4;
                const int64_t sStride = srcAccess->sampleStride(), rStride = srcAccess->rowStride();
                const int64_t dsStride = dstAccess->sampleStride(), drStride = dstAccess->rowStride();
                const bool    aligned = reinterpret_cast<uintptr_t>(srcData.basePtr()) % sizeof(Vec4) == 0
                                  && reinterpret_cast<uintptr_t>(dstData.basePtr()) % sizeof(Vec4) == 0
                                  && sStride % sizeof(Vec4) == 0 && rStride % sizeof(Vec4) == 0
                                  && dsStride % sizeof(Vec4) == 0 && drStride % sizeof(Vec4) == 0;
                if (aligned)
                {
                    auto srcV = cuda::Tensor4DWrap<const BT, StrideType>(srcData.basePtr(), static_cast<int>(sStride),
                                                                         static_cast<int>(sStride),
                                                                         static_cast<int>(rStride));
                    auto dstV
                        = cuda::Tensor4DWrap<BT, StrideType>(dstData.basePtr(), static_cast<int>(dsStride),
                                                             static_cast<int>(dsStride), static_cast<int>(drStride));
                    dim3 vgrid(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGROUP),
                               util::DivUp(size.y, block.y), srcAccess->numSamples());
                    SolarizePlanarVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(
                        srcV, dstV, int4{static_cast<int>(srcAccess->numSamples()), 1, size.y, size.x}, threshold);
                    launchedVec = true;
                }
            }
            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Solarize<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, 1, threshold);
            }
        }
        else
        {
            const int numPlanes   = srcAccess->numPlanes();
            const int numSamples  = static_cast<int>(srcAccess->numSamples());
            bool      launchedVec = false;
            if constexpr (sizeof(BT) == 1 || sizeof(BT) == 4)
            {
                using Vec4            = typename SolarizeVec4Type<BT>::type;
                constexpr int NGROUP  = sizeof(BT) == 1 ? 4 : 2;
                const int64_t planes  = static_cast<int64_t>(numSamples) * numPlanes;
                const int64_t sStride = srcAccess->sampleStride(), pStride = srcAccess->planeStride(),
                              rStride  = srcAccess->rowStride();
                const int64_t dsStride = dstAccess->sampleStride(), dpStride = dstAccess->planeStride(),
                              drStride = dstAccess->rowStride();
                const bool aligned
                    = planes <= 65535 && reinterpret_cast<uintptr_t>(srcData.basePtr()) % sizeof(Vec4) == 0
                   && reinterpret_cast<uintptr_t>(dstData.basePtr()) % sizeof(Vec4) == 0 && sStride % sizeof(Vec4) == 0
                   && pStride % sizeof(Vec4) == 0 && rStride % sizeof(Vec4) == 0 && dsStride % sizeof(Vec4) == 0
                   && dpStride % sizeof(Vec4) == 0 && drStride % sizeof(Vec4) == 0;
                if (aligned)
                {
                    auto srcV = cuda::Tensor4DWrap<const BT, StrideType>(srcData.basePtr(), static_cast<int>(sStride),
                                                                         static_cast<int>(pStride),
                                                                         static_cast<int>(rStride));
                    auto dstV
                        = cuda::Tensor4DWrap<BT, StrideType>(dstData.basePtr(), static_cast<int>(dsStride),
                                                             static_cast<int>(dpStride), static_cast<int>(drStride));
                    dim3 vgrid(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGROUP),
                               util::DivUp(size.y, block.y), static_cast<unsigned int>(planes));
                    SolarizePlanarVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(
                        srcV, dstV, int4{numSamples, numPlanes, size.y, size.x}, threshold);
                    launchedVec = true;
                }
            }
            if (!launchedVec)
            {
                auto src = cuda::Tensor4DWrap<const ValueT, StrideType>(
                    srcData.basePtr(), static_cast<int>(srcAccess->sampleStride()),
                    static_cast<int>(srcAccess->planeStride()), static_cast<int>(srcAccess->rowStride()));
                auto dst = cuda::Tensor4DWrap<ValueT, StrideType>(
                    dstData.basePtr(), static_cast<int>(dstAccess->sampleStride()),
                    static_cast<int>(dstAccess->planeStride()), static_cast<int>(dstAccess->rowStride()));
                Solarize<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, numPlanes, threshold);
            }
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        static_assert(std::is_same_v<SrcData, nvcv::ImageBatchVarShapeDataStridedCuda>);
        // One grid.z block per image; CUDA caps grid.z at 65535.
        if (dstData.numImages() > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Batch size exceeds the CUDA grid.z limit of 65535");
        }
        int3 dstMaxSize{dstData.maxSize().w, dstData.maxSize().h, dstData.numImages()};
        dim3 grid(util::DivUp(dstMaxSize.x, block.x), util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);

        const int numPlanes = dstData.uniqueFormat().numPlanes();

        // Vectorized planar var-shape path for 1-byte planes (uchar4 wide load; NVCV row pitch is
        // >= 4-byte aligned). float/uint16 var-shape stay scalar (already at ridge / no safe wide align).
        using BT = cuda::BaseType<ValueT>;
        if constexpr (isPlanar && sizeof(BT) == 1)
        {
            constexpr int NGROUP = 4;
            const int64_t planes = static_cast<int64_t>(dstData.numImages()) * numPlanes;
            if (planes <= 65535)
            {
                cuda::ImageBatchVarShapeWrap<const BT> srcV(srcData);
                cuda::ImageBatchVarShapeWrap<BT>       dstV(dstData);
                dim3 vgrid(util::DivUp(util::DivUp(dstMaxSize.x, 4), static_cast<int>(block.x) * NGROUP),
                           util::DivUp(dstMaxSize.y, block.y), static_cast<unsigned int>(planes));
                SolarizePlanarVarShapeVec4Kernel<NGROUP, BT>
                    <<<vgrid, block, 0, stream>>>(srcV, dstV, numPlanes, threshold);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Solarize<isPlanar><<<grid, block, 0, stream>>>(src, dst, numPlanes, threshold);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

// Dispatch over base data type (u8 / u16 / f32) and channel count (1 / 3 / 4) -------------

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

#define NVCV_SOLARIZE_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                      \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_SOLARIZE_RUN_TYPED(U8, uchar)
    else if NVCV_SOLARIZE_RUN_TYPED(U16, ushort)
    else if NVCV_SOLARIZE_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: Solarize supports 8-bit unsigned, 16-bit unsigned and 32-bit float");
    }
        // clang-format on

#undef NVCV_SOLARIZE_RUN_TYPED
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
                else
                {
                    cb(Val{}, std::integral_constant<bool, true>{});
                }
            }
            else if (numChannels == 3)
            {
                cb(cuda::MakeType<ValBase, 3>{}, std::integral_constant<bool, false>{});
            }
            else if (numChannels == 4)
            {
                cb(cuda::MakeType<ValBase, 4>{}, std::integral_constant<bool, false>{});
            }
            else
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Invalid number of channels: Solarize supports 1, 3 or 4 channels");
            }
                      // clang-format on
                  });
}

// Validation ------------------------------------------------------------------------------

inline void ValidateSrcDstTensors(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
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
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    dtype                  = srcData->dtype();
    numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? numChannels : 1;
}

inline auto ValidateSrcDstVarBatch(int &numInterleavedChannels, int &numPlanes, nvcv::DataType &dtype,
                                   cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
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
    numPlanes       = srcFormat.numPlanes();
    if (numPlanes > 1 && numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    dtype = srcFormat.planeDataType(0);
    for (int i = 1; i < numPlanes; ++i)
    {
        if (dtype != srcFormat.planeDataType(i))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All planes in the input image must have the same data type");
        }
    }

    numInterleavedChannels = dtype.numChannels();

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

} // anonymous namespace

namespace cvcuda::priv {

Solarize::Solarize() {}

// Tensor input variant
void Solarize::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, double threshold) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Solarize::operator()[Tensor]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcData, &dstData, threshold](auto dummyVal, auto isPlanar)
                     {
                         using ValueT   = decltype(dummyVal);
                         using IsPlanar = decltype(isPlanar);
                         RunSolarize<IsPlanar::value, ValueT>(stream, *srcData, *dstData, threshold);
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
    auto           srcDstData = ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcDstData, threshold](auto dummyVal, auto isPlanar)
                     {
                         using ValueT             = decltype(dummyVal);
                         using IsPlanar           = decltype(isPlanar);
                         auto &[srcData, dstData] = srcDstData;
                         RunSolarize<IsPlanar::value, ValueT>(stream, *srcData, *dstData, threshold);
                     });
}

} // namespace cvcuda::priv
