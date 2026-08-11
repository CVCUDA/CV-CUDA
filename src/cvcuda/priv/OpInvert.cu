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

#include "InvertPolicy.hpp"
#include "Nvtx.hpp"
#include "OpInvert.hpp"

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

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

inline bool UsePackedU8C3VarShapeKernelOnCurrentDevice()
{
    int device = 0;
    NVCV_CHECK_THROW(cudaGetDevice(&device));

    static thread_local int  cachedDevice    = -1;
    static thread_local bool cachedUsePacked = true;
    if (cachedDevice != device)
    {
        cudaDeviceProp properties{};
        NVCV_CHECK_THROW(cudaGetDeviceProperties(&properties, device));

        const int sm    = properties.major * 10 + properties.minor;
        cachedUsePacked = cvcuda::priv::UsePackedU8C3VarShapeKernelForDevice(sm, properties.name);
        cachedDevice    = device;
    }

    return cachedUsePacked;
}

// Photometric-negative bound per base type: dtype max for unsigned integers, 1.0 for float.
// Matches torchvision.transforms.v2.functional.invert / OpenCV cv::bitwise_not (unsigned).
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
inline __device__ void DoInvert(SrcWrapper src, DstWrapper dst, const int2 size, const int p)
{
    using SrcT                       = std::remove_const_t<typename SrcWrapper::ValueType>;
    using DstT                       = typename DstWrapper::ValueType;
    using BT                         = cuda::BaseType<SrcT>;
    static constexpr int numChannels = cuda::NumElements<SrcT>;
    static_assert(numChannels == cuda::NumElements<DstT>);
    // if planar then no interleaved channels
    static_assert(!IsPlanar || numChannels == 1);

    int3 nhwCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);
    if (nhwCoord.x >= size.x || nhwCoord.y >= size.y)
    {
        return;
    }
    auto coord = GetCoordForLayout<IsPlanar>(nhwCoord, p);

    // out = bound - in, per element. The subtraction is exact for the supported types, so the
    // result is bit-exact with the equivalent interleaved computation (Golden rule / COV-BITEXACT).
    dst[coord] = cuda::SaturateCast<DstT>(InvertBound<BT>() - src[coord]);
}

// Invert kernel --------------------------------------------------------------------------

// Tensor variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Invert(SrcWrapper src, DstWrapper dst, int2 size, int numPlanes)
{
    assert(isPlanar || numPlanes == 1);
    if constexpr (!isPlanar)
    {
        DoInvert<isPlanar>(src, dst, size, 0);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoInvert<isPlanar>(src, dst, size, p);
        }
    }
}

// VarShape variant
template<bool isPlanar, class SrcWrapper, class DstWrapper>
__global__ void Invert(SrcWrapper src, DstWrapper dst, int numPlanes)
{
    assert(isPlanar || numPlanes == 1);
    int  z = blockIdx.z;
    int2 size{dst.width(z), dst.height(z)};

    if constexpr (!isPlanar)
    {
        DoInvert<isPlanar>(src, dst, size, 0);
    }
    else
    {
        for (int p = 0; p < numPlanes; p++)
        {
            DoInvert<isPlanar>(src, dst, size, p);
        }
    }
}

// Vectorized planar kernel -----------------------------------------------------------------
//
// The scalar planar kernel above moves one element per thread per plane, so the 1-byte planar paths
// are memory-latency bound (NCU: ~82% long-scoreboard stalls, ~58-70% BWUtil) — each warp has only a
// single outstanding load. This kernel maps each (sample, plane) to grid.z (no per-plane loop) and
// has each thread own NGROUP independent 4-column groups, issuing all NGROUP wide vector loads
// (uchar4 / float4) before any compute so many memory requests are outstanding at once. Output is
// bit-identical to DoInvert per element (same SaturateCast(bound - in)). Modeled on
// legacy/normalize_planar.cuh. Caller guards sizeof(Vec4)-aligned strides and falls back to the
// scalar kernel otherwise; a width not a multiple of 4 is handled by the per-thread scalar tail.
template<typename T, int Size = sizeof(T)>
struct InvertVec4Type;

template<typename T>
struct InvertVec4Type<T, 1>
{
    using type = uchar4;
};

template<typename T>
struct InvertVec4Type<T, 2>
{
    using type = ushort4;
};

template<typename T>
struct InvertVec4Type<T, 4>
{
    using type = float4;
};

template<int NGROUP, typename BT>
__global__ void InvertPlanarVec4Kernel(cuda::Tensor4DWrap<const BT, int32_t> src, cuda::Tensor4DWrap<BT, int32_t> dst,
                                       int4 inout_size)
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

    using Vec4     = typename InvertVec4Type<BT>::type;
    const BT bound = InvertBound<BT>();

    // Loads first: issue all NGROUP wide loads before any compute (raises outstanding requests).
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
            out4.x                                                           = cuda::SaturateCast<BT>(bound - in4[i].x);
            out4.y                                                           = cuda::SaturateCast<BT>(bound - in4[i].y);
            out4.z                                                           = cuda::SaturateCast<BT>(bound - in4[i].z);
            out4.w                                                           = cuda::SaturateCast<BT>(bound - in4[i].w);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, src_y, x) = cuda::SaturateCast<BT>(bound - *src.ptr(batch, channel, src_y, x));
            }
        }
    }
}

// Keep the existing planar kernel's cross-group instruction-level parallelism. Collapsing this path
// into the register-bounded U16 kernel below regresses the reference A100/H100 planar RGB8 workload.
template<int NGROUP, typename BT>
__global__ void InvertPlanarVarShapeVec4Kernel(cuda::ImageBatchVarShapeWrap<const BT> src,
                                               cuda::ImageBatchVarShapeWrap<BT> dst, int numChannels)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / numChannels;
    const int channel = nc % numChannels;
    const int width   = dst.width(batch, channel);

    if (g0 * 4 >= width || dst_y >= dst.height(batch, channel))
    {
        return;
    }

    using Vec4     = uchar4; // 1-byte planes only (caller guards sizeof(BT) == 1)
    const BT bound = InvertBound<BT>();

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
            out4.x                                                           = cuda::SaturateCast<BT>(bound - in4[i].x);
            out4.y                                                           = cuda::SaturateCast<BT>(bound - in4[i].y);
            out4.z                                                           = cuda::SaturateCast<BT>(bound - in4[i].z);
            out4.w                                                           = cuda::SaturateCast<BT>(bound - in4[i].w);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, dst_y, x) = cuda::SaturateCast<BT>(bound - *src.ptr(batch, channel, dst_y, x));
            }
        }
    }
}

// Vectorized U16 var-shape kernel: resolve the row pointers once per thread so each group does not
// repeat the image-list lookup. Rows without Vec4 alignment and partial groups use the scalar fallback.
template<int NGROUP, typename BT>
__global__ void InvertU16VarShapeVec4Kernel(cuda::ImageBatchVarShapeWrap<const BT> src,
                                            cuda::ImageBatchVarShapeWrap<BT> dst, int numChannels)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / numChannels;
    const int channel = nc % numChannels;
    const int width   = dst.width(batch, channel);

    if (g0 * 4 >= width || dst_y >= dst.height(batch, channel))
    {
        return;
    }

    using Vec4       = typename InvertVec4Type<BT>::type;
    const BT   bound = InvertBound<BT>();
    const BT  *srow  = src.ptr(batch, channel, dst_y, 0);
    BT        *drow  = dst.ptr(batch, channel, dst_y, 0);
    const bool wide  = reinterpret_cast<uintptr_t>(srow) % alignof(Vec4) == 0
                   && reinterpret_cast<uintptr_t>(drow) % alignof(Vec4) == 0;

    // Keep one vector live at a time. Unrolling all groups raises the active register footprint
    // above the optimization campaign's automatic memory allowance without improving coalescing.
#pragma unroll 1
    for (int i = 0; i < NGROUP; ++i)
    {
        const int cx = (g0 + i * blockDim.x) * 4;
        if (wide && cx + 4 <= width)
        {
            const Vec4 in4 = *reinterpret_cast<const Vec4 *>(srow + cx);
            Vec4       out4;
            out4.x                               = cuda::SaturateCast<BT>(bound - in4.x);
            out4.y                               = cuda::SaturateCast<BT>(bound - in4.y);
            out4.z                               = cuda::SaturateCast<BT>(bound - in4.z);
            out4.w                               = cuda::SaturateCast<BT>(bound - in4.w);
            *reinterpret_cast<Vec4 *>(drow + cx) = out4;
        }
        else if (cx < width)
        {
            for (int x = cx; x < min(cx + 4, width); ++x)
            {
                drow[x] = cuda::SaturateCast<BT>(bound - srow[x]);
            }
        }
    }
}

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

// Run Invert kernel ----------------------------------------------------------------------

template<bool isPlanar, typename ValueT, class SrcData, class DstData>
inline void RunInvert(cudaStream_t stream, const SrcData &srcData, const DstData &dstData)
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

        // Fully dense src+dst collapse to one contiguous byte range; the flat 16-byte-word kernels
        // beat every per-element path (see the dense-kernel block above). Falls through to the
        // layout-specific kernels for padded or externally-wrapped strides.
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
            if constexpr (cuda::NumElements<ValueT> == 1)
            {
                using Vec4            = typename InvertVec4Type<BT>::type;
                constexpr int NGROUP  = 4;
                const int64_t sStride = srcAccess->sampleStride(), rStride = srcAccess->rowStride();
                const int64_t dsStride = dstAccess->sampleStride(), drStride = dstAccess->rowStride();
                const bool    aligned = reinterpret_cast<uintptr_t>(srcData.basePtr()) % sizeof(Vec4) == 0
                                  && reinterpret_cast<uintptr_t>(dstData.basePtr()) % sizeof(Vec4) == 0
                                  && sStride % sizeof(Vec4) == 0 && rStride % sizeof(Vec4) == 0
                                  && dsStride % sizeof(Vec4) == 0 && drStride % sizeof(Vec4) == 0;
                if (aligned)
                {
                    // C == 1, so planeStride is never indexed (channel is always 0); pass sampleStride.
                    auto srcV = cuda::Tensor4DWrap<const BT, StrideType>(srcData.basePtr(), static_cast<int>(sStride),
                                                                         static_cast<int>(sStride),
                                                                         static_cast<int>(rStride));
                    auto dstV
                        = cuda::Tensor4DWrap<BT, StrideType>(dstData.basePtr(), static_cast<int>(dsStride),
                                                             static_cast<int>(dsStride), static_cast<int>(drStride));
                    dim3 vgrid(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGROUP),
                               util::DivUp(size.y, block.y), srcAccess->numSamples());
                    InvertPlanarVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(
                        srcV, dstV, int4{static_cast<int>(srcAccess->numSamples()), 1, size.y, size.x});
                    launchedVec = true;
                }
            }

            if (!launchedVec)
            {
                auto src = cuda::CreateTensorWrapNHW<const ValueT, StrideType>(srcData);
                auto dst = cuda::CreateTensorWrapNHW<ValueT, StrideType>(dstData);
                Invert<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, 1);
            }
        }
        else
        {
            using BT              = cuda::BaseType<ValueT>;
            const int numPlanes   = srcAccess->numPlanes();
            const int numSamples  = static_cast<int>(srcAccess->numSamples());
            bool      launchedVec = false;

            // Vectorized planar path for 1-byte / 4-byte planes: map each (sample, plane) to grid.z and
            // move 4 columns/thread with a wide load (batched NGROUP loads -> high memory-level
            // parallelism), eliminating the scalar planar kernel's long-scoreboard stalls. Requires
            // sizeof(Vec4)-aligned base + strides; falls back to the scalar kernel otherwise (the
            // kernel's per-thread tail handles a width that is not a multiple of 4).
            if constexpr (sizeof(BT) == 1 || sizeof(BT) == 4)
            {
                using Vec4            = typename InvertVec4Type<BT>::type;
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
                    InvertPlanarVec4Kernel<NGROUP, BT>
                        <<<vgrid, block, 0, stream>>>(srcV, dstV, int4{numSamples, numPlanes, size.y, size.x});
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
                Invert<isPlanar><<<grid, block, 0, stream>>>(src, dst, size, numPlanes);
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

        if constexpr (!isPlanar && std::is_same_v<ValueT, uchar3>)
        {
            if (UsePackedU8C3VarShapeKernelOnCurrentDevice())
            {
                constexpr int NGROUP = 4;
                dim3          vgrid(util::DivUp(util::DivUp(dstMaxSize.x, 4), static_cast<int>(block.x) * NGROUP),
                                    util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);
                cuda::ImageBatchVarShapeWrap<const uchar3> srcV(srcData);
                cuda::ImageBatchVarShapeWrap<uchar3>       dstV(dstData);
                InvertU8C3VarShapePackedKernel<NGROUP><<<vgrid, block, 0, stream>>>(srcV, dstV);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        // Preserve the established 1-byte planar kernel; it needs cross-group ILP to stay at ridge.
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
                InvertPlanarVarShapeVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(srcV, dstV, numPlanes);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        // The U16 interleaved path keeps one vector group live at a time to cap register residency.
        if constexpr (!isPlanar && cuda::NumElements<ValueT> == 1 && sizeof(BT) == 2)
        {
            constexpr int                          NGROUP = 4;
            cuda::ImageBatchVarShapeWrap<const BT> srcV(srcData);
            cuda::ImageBatchVarShapeWrap<BT>       dstV(dstData);
            dim3 vgrid(util::DivUp(util::DivUp(dstMaxSize.x, 4), static_cast<int>(block.x) * NGROUP),
                       util::DivUp(dstMaxSize.y, block.y), dstMaxSize.z);
            InvertU16VarShapeVec4Kernel<NGROUP, BT><<<vgrid, block, 0, stream>>>(srcV, dstV, 1);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        }

        cuda::ImageBatchVarShapeWrap<const ValueT> src(srcData);
        cuda::ImageBatchVarShapeWrap<ValueT>       dst(dstData);
        Invert<isPlanar><<<grid, block, 0, stream>>>(src, dst, numPlanes);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
}

// Dispatch over base data type (u8 / u16 / f32) and channel count (1 / 3 / 4) -------------

template<typename Cb>
inline void RunTypeSwitch(nvcv::DataType dType, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

#define NVCV_INVERT_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                        \
    ((dType == nvcv::TYPE_4##DYN_BASE_TYPE) || (dType == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dType == nvcv::TYPE_2##DYN_BASE_TYPE) || (dType == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

    // clang-format off
    if NVCV_INVERT_RUN_TYPED(U8, uchar)
    else if NVCV_INVERT_RUN_TYPED(U16, ushort)
    else if NVCV_INVERT_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: Invert supports 8-bit unsigned, 16-bit unsigned and 32-bit float");
    }
        // clang-format on

#undef NVCV_INVERT_RUN_TYPED
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
                                      "Invalid number of channels: Invert supports 1, 3 or 4 channels");
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

Invert::Invert() {}

// Tensor input variant
void Invert::operator()(cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Invert::operator()[Tensor]");
    int            numInterleavedChannels;
    int            numPlanes;
    nvcv::DataType dtype;
    auto           srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto           dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ValidateSrcDstTensors(numInterleavedChannels, numPlanes, dtype, srcData, dstData);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
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
    auto           srcDstData = ValidateSrcDstVarBatch(numInterleavedChannels, numPlanes, dtype, stream, src, dst);

    RunChannelSwitch(numInterleavedChannels, numPlanes, dtype,
                     [&stream, &srcDstData](auto dummyVal, auto isPlanar)
                     {
                         using ValueT             = decltype(dummyVal);
                         using IsPlanar           = decltype(isPlanar);
                         auto &[srcData, dstData] = srcDstData;
                         RunInvert<IsPlanar::value, ValueT>(stream, *srcData, *dstData);
                     });
}

} // namespace cvcuda::priv
