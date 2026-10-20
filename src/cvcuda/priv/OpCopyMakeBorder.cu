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

#include "CopyMakeBorderPolicy.hpp"
#include "CudaDeviceUtils.hpp"
#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpCopyMakeBorder.hpp"
#include "TensorAccessValidation.hpp"

#include "ElementStreamCommon.cuh"
#include "SameShapeCommon.cuh"

#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;
namespace es   = cvcuda::priv::elem_stream;

namespace {

// Independent 16-byte transfers issued per thread.
constexpr int kCmbCarriersPerThread = 4;

using uchar = unsigned char;

constexpr int kBlock = 32;

// ---------------------------------------------------------------------------------------------
// Classification helpers.
//
// These reproduce nvcv::legacy::helpers::GetLegacyDataType / GetLegacyDataFormat exactly, so the
// admitted input set, the rejection order and the rejection messages are unchanged. The dtype gate
// classifies on bits-per-channel plus data kind -- not on an exact nvcv::DataType -- so kU8 also
// covers the packed TYPE_2U8/3U8/4U8 types the interleaved 2/3/4-channel paths rely on.
// ---------------------------------------------------------------------------------------------

// Values mirror nvcv::legacy::cuda_op::DataType so the dispatch tables below keep their row order.
enum CvType
{
    kU8  = 0,
    kS8  = 1,
    kU16 = 2,
    kS16 = 3,
    kS32 = 4,
    kF32 = 5,
    kF64 = 6,
    kF16 = 7,
    // Past the legacy enum's range; appended so rows 0-7 keep their order.
    kS64 = 8,
};

inline CvType ToCvType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bpc == 64)
            return kF64;
        if (bpc == 32)
            return kF32;
        if (bpc == 16)
            return kF16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);

    case nvcv::DataKind::SIGNED:
        if (bpc == 8)
            return kS8;
        if (bpc == 16)
            return kS16;
        if (bpc == 32)
            return kS32;
        if (bpc == 64)
            return kS64;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);

    case nvcv::DataKind::UNSIGNED:
        if (bpc == 8)
            return kU8;
        if (bpc == 16)
            return kU16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline CvType ToCvType(nvcv::DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ToCvType(bpc[0], dtype.dataKind());
}

inline CvType ToCvType(nvcv::ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return ToCvType(fmt.planeDataType(0));
}

inline bool IsSupportedCvType(CvType type)
{
    // Every class below has a row in the dispatch tables. Two-channel interleaved is still a hole
    // in most rows, but that is caught by the channel gate rather than here.
    return type == kU8 || type == kS8 || type == kU16 || type == kS16 || type == kS32 || type == kS64 || type == kF32
        || type == kF64 || type == kF16;
}

enum CvFormat
{
    kNCHW,
    kNHWC,
    kCHW,
    kHWC,
};

inline bool IsPlanar(CvFormat format)
{
    return format == kNCHW || format == kCHW;
}

inline bool SameLayoutFamily(CvFormat lhs, CvFormat rhs)
{
    return IsPlanar(lhs) == IsPlanar(rhs);
}

inline const char *FormatName(CvFormat format)
{
    switch (format)
    {
    case kNCHW:
        return "NCHW";
    case kNHWC:
        return "NHWC";
    case kCHW:
        return "CHW";
    case kHWC:
        return "HWC";
    }
    return "unknown";
}

inline CvFormat ToCvFormat(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return kNCHW;
    }
    else if (layout == nvcv::TENSOR_CHW)
    {
        return kCHW;
    }
    else if (layout == nvcv::TENSOR_NHWC)
    {
        return kNHWC;
    }
    else if (layout == nvcv::TENSOR_HWC)
    {
        return kHWC;
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
}

inline CvFormat ToCvFormat(const nvcv::TensorDataStridedCuda &container)
{
    return ToCvFormat(container.layout());
}

inline CvFormat ToCvFormat(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    nvcv::ImageFormat fmt = cvcuda::priv::legacy_format::RequireUniformVarShapeFormat(imgBatch);

    if (fmt.numPlanes() >= 2)
    {
        return imgBatch.numImages() >= 2 ? kNCHW : kCHW;
    }

    return imgBatch.numImages() >= 2 ? kNHWC : kHWC;
}

inline nvcv::Size2D MaxImageSize(const nvcv::TensorDataStridedCuda &tensor)
{
    if (auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensor))
    {
        return access->size();
    }

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Get size failed.");
}

inline nvcv::Size2D MaxImageSize(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    return imgBatch.maxSize();
}

inline bool IsSupportedBorderType(NVCVBorderType borderType)
{
    return borderType == NVCV_BORDER_CONSTANT || borderType == NVCV_BORDER_REPLICATE
        || borderType == NVCV_BORDER_REFLECT || borderType == NVCV_BORDER_REFLECT101 || borderType == NVCV_BORDER_WRAP;
}

using cvcuda::priv::UsePackedRGB8Reflect101ForSM;
using cvcuda::priv::UsePlanarRGBF32Reflect101Vector;

inline int CurrentDeviceSMOrZero()
{
    int sm = 0;
    return cvcuda::priv::GetCurrentDeviceSM(sm) == cudaSuccess ? sm : 0;
}

inline int CurrentDeviceReflect101L2CacheSizeOrZero()
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 0;

    static thread_local int cachedDevice = -1;
    static thread_local int cachedBytes  = 0;
    if (cachedDevice != device)
    {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device) != cudaSuccess)
            return 0;
        cachedDevice = device;
        cachedBytes  = bytes;
    }
    return cachedBytes;
}

// ---------------------------------------------------------------------------------------------
// Tensor path
// ---------------------------------------------------------------------------------------------

inline nvcv::TensorDataStridedCuda TensorPlanarChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                           const nvcv::TensorDataAccessStridedImagePlanar &access,
                                                           int                                             plane)
{
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr()) + plane * access.chStride();
    buf.strides[0] = access.sampleStride();
    buf.strides[1] = access.rowStride();
    buf.strides[2] = access.colStride();
    buf.strides[3] = access.colStride();

    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{access.numSamples(), access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(), buf
    };
}

template<class SrcWrapper, class DstWrapper>
__global__ void CopyMakeBorderTensorKernel(SrcWrapper src, DstWrapper dst, int2 dstSize, int left, int top)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    int3 srcCoord = {dstCoord.x - left, dstCoord.y - top, dstCoord.z};

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        using T = typename DstWrapper::ValueType;
        if constexpr (sizeof(T) == 32)
        {
            const T *srcPtr = src.ptr(srcCoord.z, srcCoord.y, srcCoord.x);
            T       *dstPtr = dst.ptr(dstCoord.z, dstCoord.y, dstCoord.x);
            if ((reinterpret_cast<uintptr_t>(srcPtr) | reinterpret_cast<uintptr_t>(dstPtr)) % alignof(T) != 0)
            {
                T value;
                if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                    value = srcPtr ? cvcuda::priv::same_shape::LoadUnalignedPixel(srcPtr) : src.value(srcCoord);
                else
                    value = cvcuda::priv::same_shape::LoadUnalignedPixel(srcPtr);
                cvcuda::priv::same_shape::StoreUnalignedPixel(dstPtr, value);
                return;
            }
        }
        if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
            dst[dstCoord] = src.value(srcCoord);
        else
            dst[dstCoord] = src[srcCoord];
    }
}

// Adjacent threads copy adjacent 16-byte halves of a wide pixel, so each warp
// fills every sector instead of issuing two strided loads for 32-byte pixels.
template<class SrcWrapper, class DstWrapper>
__global__ void CopyMakeBorderTensorWide(SrcWrapper src, DstWrapper dst, int2 dstSize, int left, int top,
                                         uint4 borderLow, uint4 borderHigh)
{
    const unsigned int word = blockIdx.x * blockDim.x + threadIdx.x;
    const int          x    = word >> 1;
    const int          y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int          z    = blockIdx.z;
    if (x >= dstSize.x || y >= dstSize.y)
        return;

    const int   half = word & 1;
    const auto *sp   = src.ptr(z, y - top, x - left);
    uint4       value;
    if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
    {
        value = sp ? reinterpret_cast<const uint4 *>(sp)[half] : (half ? borderHigh : borderLow);
    }
    else
    {
        value = reinterpret_cast<const uint4 *>(sp)[half];
    }
    reinterpret_cast<uint4 *>(&dst[int3{x, y, z}])[half] = value;
}

// Vector pack type for T: uint3 (12B) for 3-element T, else uint4 (16B). NIX = elements/thread.
template<typename T>
using TensorPackType = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;
template<typename T>
constexpr int kTensorNIX = sizeof(TensorPackType<T>) / sizeof(T);
template<typename T>
constexpr uintptr_t kTensorMask = alignof(TensorPackType<T>) - 1;

// copy_make_border is a pure copy: the only per-pixel cost is the BorderWrap's bounds test + index
// clamp, which dominates (compute-bound) even though the interior is just a straight copy. This kernel
// processes kTensorNIX consecutive output x-elements per thread and takes a fast path for the INTERIOR
// (the [left,left+srcW) x [top,top+srcH) region, i.e. the bulk): a single aligned vector load+store
// straight from the raw source, skipping the BorderWrap entirely. Threads touching the border (or an
// unaligned/partial span) fall back to the per-element BorderWrap path. Bit-exact (a copy is exact;
// border pixels go through the same BorderWrap as before).
template<class SrcWrapper, class SrcRawWrapper, class DstWrapper, typename T>
__global__ void CopyMakeBorderTensorKernelVec(SrcWrapper src, SrcRawWrapper srcRaw, DstWrapper dst, int2 dstSize,
                                              int2 srcSize, int left, int top)
{
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int z    = blockIdx.z;
    if (dstY >= dstSize.y)
        return;
    const int dstX0 = (blockIdx.x * blockDim.x + threadIdx.x) * kTensorNIX<T>;
    if (dstX0 >= dstSize.x)
        return;

    const int  srcY         = dstY - top;
    const int  srcX0        = dstX0 - left;
    const bool full         = (dstX0 + kTensorNIX<T> - 1 < dstSize.x);
    const bool interiorRow  = (srcY >= 0 && srcY < srcSize.y);
    const bool interiorSpan = (srcX0 >= 0 && srcX0 + kTensorNIX<T> - 1 < srcSize.x);

    if (full && interiorRow && interiorSpan)
    {
        const T *sp = &srcRaw[int3{srcX0, srcY, z}];
        T       *dp = &dst[int3{dstX0, dstY, z}];
        if ((reinterpret_cast<uintptr_t>(sp) & kTensorMask<T>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & kTensorMask<T>) == 0)
            *reinterpret_cast<TensorPackType<T> *>(dp) = *reinterpret_cast<const TensorPackType<T> *>(sp);
        else
#pragma unroll
            for (int i = 0; i < kTensorNIX<T>; ++i) dp[i] = sp[i];
    }
    else
    {
#pragma unroll
        for (int i = 0; i < kTensorNIX<T>; ++i)
        {
            const int dstX = dstX0 + i;
            if (dstX < dstSize.x)
            {
                const int3 srcCoord{dstX - left, srcY, z};
                if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                    dst[int3{dstX, dstY, z}] = src.value(srcCoord);
                else
                    dst[int3{dstX, dstY, z}] = src[srcCoord];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Byte path.
//
// CopyMakeBorder moves bytes and computes nothing, so the destination only ever needs to be
// described in bytes and every thread can carry the same 16-byte unit whatever the dtype and the
// channel count are. Each output row splits into at most three spans that each map to a contiguous
// run of one source row, and each of those is one of the movements the shared carrier already
// provides:
//
//   * the middle span is a straight copy once the row index has gone through the border map, and a
//     WRAP side strip is a straight copy of the opposite edge -- the shift carrier;
//   * a REFLECT / REFLECT101 side strip is the same run with the pixel order reversed -- the same
//     reversal map a horizontal flip uses, with the strip's own anchor;
//   * a REPLICATE side strip repeats one pixel -- the broadcast map.
//
// The middle span's destination is offset by left * pixelBytes, an arbitrary byte phase, which is
// what the shifted carrier repairs. CONSTANT is not a movement -- it reads nothing -- so it keeps
// the per-pixel kernel.
// ---------------------------------------------------------------------------------------------

// Destination row r reads the source row the border map names.
template<NVCVBorderType B>
struct BorderRows
{
    struct Param
    {
        int32_t top;
        int32_t srcH;
    };

    inline static __device__ int32_t SrcRegion(int32_t region, const Param &p)
    {
        return cuda::GetIndexWithBorder<B>(region - p.top, p.srcH);
    }
};

// A side strip maps to a contiguous run of one source row only while it stays inside one fold of
// the source; past that the reflect and wrap indices stop being affine.
template<NVCVBorderType B>
constexpr bool CmbStripIsAffine(int width, int srcW)
{
    if constexpr (B == NVCV_BORDER_REPLICATE)
    {
        return true;
    }
    else if constexpr (B == NVCV_BORDER_REFLECT101)
    {
        return width <= srcW - 1;
    }
    else
    {
        return width <= srcW;
    }
}

template<int P, NVCVBorderType B>
void LaunchCmbStrip(cudaStream_t stream, const unsigned char *srcLo, const unsigned char *srcHi, unsigned char *dstBase,
                    const es::Plan &plan, const typename BorderRows<B>::Param &rows, bool isLeft, int32_t width,
                    int32_t left, int32_t srcW)
{
    using Rows = BorderRows<B>;

    const int64_t dstColOff = isLeft ? 0 : static_cast<int64_t>(left + srcW) * P;

    if constexpr (B == NVCV_BORDER_WRAP)
    {
        const int64_t srcColOff = isLeft ? static_cast<int64_t>(srcW - width) * P : 0;
        es::TryRunShifted<kCmbCarriersPerThread, es::ShiftCarrier<Rows>>(stream, srcLo + srcColOff, dstBase + dstColOff,
                                                                         srcLo, srcHi, plan, rows);
    }
    else if constexpr (B == NVCV_BORDER_REPLICATE)
    {
        using Map = es::BroadcastMap<P>;

        const typename Map::Data data{isLeft ? int64_t{0} : static_cast<int64_t>(srcW - 1) * P};
        es::TryRunShifted<kCmbCarriersPerThread, es::PermuteCarrier<Map, Rows>>(
            stream, srcLo, dstBase + dstColOff, srcLo, srcHi, plan, es::MakePermuteParam<Map, Rows>(data, rows));
    }
    else
    {
        using Map = es::ReverseMap<P>;

        // Source column of destination column x is bias - x. REFLECT repeats the edge pixel and
        // REFLECT101 does not, which shifts the reflection point by one.
        constexpr int32_t kEdge = (B == NVCV_BORDER_REFLECT101) ? 0 : 1;
        const int64_t     bias  = isLeft ? int64_t{left - kEdge} : int64_t{2 * srcW - 2 + kEdge + left};

        const typename Map::Data data{bias * P - dstColOff};
        es::TryRunShifted<kCmbCarriersPerThread, es::PermuteCarrier<Map, Rows>>(
            stream, srcLo, dstBase + dstColOff, srcLo, srcHi, plan, es::MakePermuteParam<Map, Rows>(data, rows));
    }
}

template<int P, NVCVBorderType B>
bool TryCopyMakeBorderCarrierStream(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                    const nvcv::TensorDataStridedCuda              &outData,
                                    const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                    const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int left, int top)
{
    if constexpr (B == NVCV_BORDER_CONSTANT
                  || ((P == 12 || P == 16) && (B == NVCV_BORDER_REFLECT || B == NVCV_BORDER_REFLECT101)))
    {
        // Interleaved 32-bit RGB/RGBA reflection is faster on the existing tensor kernel.
        // Planar channels use P == sizeof(element), so they retain the carrier path.
        return false;
    }
    else
    {
        // Columns must be packed for a span to be a byte run holding whole pixels.
        if (inAccess.colStride() != P || outAccess.colStride() != P)
        {
            return false;
        }

        const int32_t srcW  = static_cast<int32_t>(inAccess.numCols());
        const int32_t srcH  = static_cast<int32_t>(inAccess.numRows());
        const int32_t dstW  = static_cast<int32_t>(outAccess.numCols());
        const int32_t dstH  = static_cast<int32_t>(outAccess.numRows());
        const int32_t right = dstW - left - srcW;

        const int64_t samples   = outAccess.numSamples();
        const int64_t srcRow    = inAccess.rowStride();
        const int64_t dstRow    = outAccess.rowStride();
        const int64_t srcSample = inAccess.sampleStride();
        const int64_t dstSample = outAccess.sampleStride();

        auto *dstBase = reinterpret_cast<unsigned char *>(outData.basePtr());

        // Every span stores through the carrier type, so the destination strides must be carrier
        // multiples, and a strip carries the permutation's phase in its carrier index, so its own
        // origin must be carrier-aligned too. All of this is decided before anything launches: a
        // span that declined halfway would leave the output partly written.
        if (dstRow % es::kCarrierBytes != 0 || dstSample % es::kCarrierBytes != 0)
        {
            return false;
        }
        if ((left > 0 || right > 0) && reinterpret_cast<uintptr_t>(dstBase) % es::kCarrierBytes != 0)
        {
            return false;
        }
        if (right > 0 && (static_cast<int64_t>(left + srcW) * P) % es::kCarrierBytes != 0)
        {
            return false;
        }
        if ((left > 0 && !CmbStripIsAffine<B>(left, srcW)) || (right > 0 && !CmbStripIsAffine<B>(right, srcW)))
        {
            return false;
        }

        es::Plan middlePlan, leftPlan, rightPlan;
        if (!es::MakeRunPlan(static_cast<int64_t>(srcW) * P, dstH, samples, srcRow, dstRow, srcSample, dstSample, false,
                             middlePlan))
        {
            return false;
        }
        if (left > 0
            && !es::MakeRunPlan(static_cast<int64_t>(left) * P, dstH, samples, srcRow, dstRow, srcSample, dstSample,
                                false, leftPlan))
        {
            return false;
        }
        if (right > 0
            && !es::MakeRunPlan(static_cast<int64_t>(right) * P, dstH, samples, srcRow, dstRow, srcSample, dstSample,
                                false, rightPlan))
        {
            return false;
        }

        const auto          *srcLo = reinterpret_cast<const unsigned char *>(inData.basePtr());
        const unsigned char *srcHi = es::SourceEnd(srcLo, inAccess);

        using Rows = BorderRows<B>;

        const typename Rows::Param rows{top, srcH};

        es::TryRunShifted<kCmbCarriersPerThread, es::ShiftCarrier<Rows>>(
            stream, srcLo, dstBase + static_cast<int64_t>(left) * P, srcLo, srcHi, middlePlan, rows);

        if (left > 0)
        {
            LaunchCmbStrip<P, B>(stream, srcLo, srcHi, dstBase, leftPlan, rows, true, left, left, srcW);
        }
        if (right > 0)
        {
            LaunchCmbStrip<P, B>(stream, srcLo, srcHi, dstBase, rightPlan, rows, false, right, left, srcW);
        }
        return true;
    }
}

template<typename Carrier>
__device__ __forceinline__ void CopyMakeBorderReflect101RowSpan(const unsigned char *src, unsigned char *dst,
                                                                const unsigned char *srcLo, const unsigned char *srcHi,
                                                                int firstCarrier, int carriers,
                                                                const typename Carrier::Param &param)
{
    if (firstCarrier >= carriers)
        return;
    uint32_t  window[Carrier::kWindowWords];
    const int dstByte = firstCarrier * es::kCarrierBytes;
    es::LoadWindow<Carrier::kWindowWords>(
        src + Carrier::WindowByte(dstByte, firstCarrier, carriers * es::kCarrierBytes, param), srcLo, srcHi, window);
    reinterpret_cast<uint4 *>(dst)[firstCarrier] = Carrier::Apply(window, firstCarrier, param);
}

template<int PixelBytes>
__global__ void CopyMakeBorderReflect101RowCarrier(
    const unsigned char *__restrict__ srcBase, unsigned char *__restrict__ dstBase, const unsigned char *srcHi,
    es::Plan plan, int srcHeight, int leftBytes, int middleEndBytes, int top, int leftBlocks, int middleBlocks,
    typename es::PermuteCarrier<es::ReverseMap<PixelBytes>, BorderRows<NVCV_BORDER_REFLECT101>>::Param leftParam,
    typename es::PermuteCarrier<es::ReverseMap<PixelBytes>, BorderRows<NVCV_BORDER_REFLECT101>>::Param rightParam)
{
    using Reverse = es::PermuteCarrier<es::ReverseMap<PixelBytes>, BorderRows<NVCV_BORDER_REFLECT101>>;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= plan.numRegions)
        return;
    const int   srcRow = cuda::GetIndexWithBorder<NVCV_BORDER_REFLECT101>(row - top, srcHeight);
    const auto *src    = srcBase + static_cast<int64_t>(blockIdx.z) * plan.srcSampleStride
                    + static_cast<int64_t>(srcRow) * plan.srcRegionStride;
    auto *dst = dstBase + static_cast<int64_t>(blockIdx.z) * plan.dstSampleStride
              + static_cast<int64_t>(row) * plan.dstRegionStride;
    if (blockIdx.x < leftBlocks)
    {
        const int firstCarrier = blockIdx.x * blockDim.x + threadIdx.x;
        CopyMakeBorderReflect101RowSpan<Reverse>(src, dst, srcBase, srcHi, firstCarrier, leftBytes / es::kCarrierBytes,
                                                 leftParam);
    }
    else if (blockIdx.x < leftBlocks + middleBlocks)
    {
        const int firstCarrier = (blockIdx.x - leftBlocks) * blockDim.x + threadIdx.x;
        CopyMakeBorderReflect101RowSpan<es::ShiftCarrier<es::IdentityRegions>>(
            src, dst + leftBytes, srcBase, srcHi, firstCarrier, (middleEndBytes - leftBytes) / es::kCarrierBytes, {});
    }
    else
    {
        const int firstCarrier = (blockIdx.x - leftBlocks - middleBlocks) * blockDim.x + threadIdx.x;
        CopyMakeBorderReflect101RowSpan<Reverse>(src, dst + middleEndBytes, srcBase, srcHi, firstCarrier,
                                                 (plan.runBytes - middleEndBytes) / es::kCarrierBytes, rightParam);
    }
}

template<int PixelBytes>
bool TryCopyMakeBorderReflect101RowCarrier(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                           const nvcv::TensorDataStridedCuda              &outData,
                                           const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                           const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int left, int top)
{
    const int cols    = inAccess.numCols();
    const int outCols = outAccess.numCols();
    if (!cvcuda::priv::CanUseReflect101RowCarrier(PixelBytes, cols, outCols, left) || inAccess.colStride() != PixelBytes
        || outAccess.colStride() != PixelBytes
        || reinterpret_cast<uintptr_t>(outData.basePtr()) % es::kCarrierBytes != 0)
        return false;
    es::Plan plan;
    if (!es::MakeRunPlan(static_cast<int64_t>(outCols) * PixelBytes, outAccess.numRows(), outAccess.numSamples(),
                         inAccess.rowStride(), outAccess.rowStride(), inAccess.sampleStride(), outAccess.sampleStride(),
                         false, plan)
        || !es::IsDstCarrierAligned(plan))
        return false;
    using Rows = BorderRows<NVCV_BORDER_REFLECT101>;
    using Map  = es::ReverseMap<PixelBytes>;
    const typename Rows::Param rows{top, inAccess.numRows()};
    const auto  leftParam        = es::MakePermuteParam<Map, Rows>({static_cast<int64_t>(left) * PixelBytes}, rows);
    const auto  rightParam       = es::MakePermuteParam<Map, Rows>({static_cast<int64_t>(cols - 2) * PixelBytes}, rows);
    const int   blockWidth       = 32;
    const int   blockHeight      = 8;
    const int   carriersPerBlock = blockWidth;
    const int   leftBlocks       = util::DivUp(left * PixelBytes / es::kCarrierBytes, carriersPerBlock);
    const int   middleBlocks     = util::DivUp(cols * PixelBytes / es::kCarrierBytes, carriersPerBlock);
    const int   rightBlocks = util::DivUp((outCols - left - cols) * PixelBytes / es::kCarrierBytes, carriersPerBlock);
    const dim3  block(blockWidth, blockHeight);
    const dim3  grid(leftBlocks + middleBlocks + rightBlocks, util::DivUp(plan.numRegions, blockHeight),
                     plan.numSamples);
    const auto *src = reinterpret_cast<const unsigned char *>(inData.basePtr());
    auto       *dst = reinterpret_cast<unsigned char *>(outData.basePtr());
    CopyMakeBorderReflect101RowCarrier<PixelBytes><<<grid, block, 0, stream>>>(
        src, dst, es::SourceEnd(src, inAccess), plan, inAccess.numRows(), left * PixelBytes, (left + cols) * PixelBytes,
        top, leftBlocks, middleBlocks, leftParam, rightParam);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<int PixelBytes>
__global__ void CopyMakeBorderReflect101GeneralRowCarrier(
    const unsigned char *__restrict__ srcBase, unsigned char *__restrict__ dstBase, const unsigned char *srcHi,
    es::Plan plan, int srcWidth, int srcHeight, int left, int top,
    typename es::PermuteCarrier<es::ReverseMap<PixelBytes>, es::IdentityRegions>::Param reverseParam)
{
    using Reverse            = es::PermuteCarrier<es::ReverseMap<PixelBytes>, es::IdentityRegions>;
    const int     row        = blockIdx.y * blockDim.y + threadIdx.y;
    const int     carrier    = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t byteOffset = static_cast<int64_t>(carrier) * es::kCarrierBytes;
    if (row >= plan.numRegions || byteOffset >= plan.runBytes)
        return;
    const int dstByte = static_cast<int>(byteOffset);

    const int   srcRow = cuda::GetIndexWithBorder<NVCV_BORDER_REFLECT101>(row - top, srcHeight);
    const auto *src    = srcBase + static_cast<int64_t>(blockIdx.z) * plan.srcSampleStride
                    + static_cast<int64_t>(srcRow) * plan.srcRegionStride;
    auto *dst = dstBase + static_cast<int64_t>(blockIdx.z) * plan.dstSampleStride
              + static_cast<int64_t>(row) * plan.dstRegionStride;
    const int period   = 2 * (srcWidth - 1);
    int       position = (dstByte / PixelBytes - left) % period;
    if (position < 0)
        position += period;
    const int channel = dstByte % PixelBytes;
    const int advance = (channel + es::kCarrierBytes - 1) / PixelBytes;
    if (plan.runBytes - dstByte >= es::kCarrierBytes && position + advance < srcWidth)
    {
        uint32_t window[4];
        es::LoadWindow<4>(src + position * PixelBytes + channel, srcBase, srcHi, window);
        reinterpret_cast<uint4 *>(dst)[carrier] = uint4{window[0], window[1], window[2], window[3]};
    }
    else if (plan.runBytes - dstByte >= es::kCarrierBytes && position >= srcWidth - 1 && position + advance <= period)
    {
        uint32_t  window[Reverse::kWindowWords];
        const int firstByte
            = (period - position) * PixelBytes - channel - es::GatherShift<es::ReverseMap<PixelBytes>>();
        es::LoadWindow<Reverse::kWindowWords>(src + firstByte, srcBase, srcHi, window);
        reinterpret_cast<uint4 *>(dst)[carrier] = Reverse::Apply(window, carrier, reverseParam);
    }
    else
    {
#pragma unroll
        for (int offset = 0; offset < es::kCarrierBytes; ++offset)
        {
            const int outputByte = dstByte + offset;
            if (outputByte < plan.runBytes)
            {
                const int srcCol
                    = cuda::GetIndexWithBorder<NVCV_BORDER_REFLECT101>(outputByte / PixelBytes - left, srcWidth);
                dst[outputByte] = src[srcCol * PixelBytes + outputByte % PixelBytes];
            }
        }
    }
}

template<int PixelBytes>
bool TryCopyMakeBorderReflect101GeneralRowCarrier(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                                  const nvcv::TensorDataStridedCuda              &outData,
                                                  const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                                  const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int left,
                                                  int top)
{
    static_assert(PixelBytes == 3 || PixelBytes == 4);
    if (inAccess.numCols() <= 1 || outAccess.numCols() <= inAccess.numCols() || inAccess.colStride() != PixelBytes
        || outAccess.colStride() != PixelBytes
        || reinterpret_cast<uintptr_t>(outData.basePtr()) % es::kCarrierBytes != 0)
        return false;
    es::Plan plan;
    if (!es::MakeRunPlan(static_cast<int64_t>(outAccess.numCols()) * PixelBytes, outAccess.numRows(),
                         outAccess.numSamples(), inAccess.rowStride(), outAccess.rowStride(), inAccess.sampleStride(),
                         outAccess.sampleStride(), false, plan)
        || !es::IsDstCarrierAligned(plan))
        return false;
    const auto  reverseParam = es::MakePermuteParam<es::ReverseMap<PixelBytes>, es::IdentityRegions>({0}, {});
    const dim3  block(32, 8);
    const dim3  grid(util::DivUp(plan.runBytes, static_cast<int64_t>(block.x * es::kCarrierBytes)),
                     util::DivUp(plan.numRegions, static_cast<int>(block.y)), plan.numSamples);
    const auto *src = reinterpret_cast<const unsigned char *>(inData.basePtr());
    auto       *dst = reinterpret_cast<unsigned char *>(outData.basePtr());
    CopyMakeBorderReflect101GeneralRowCarrier<PixelBytes><<<grid, block, 0, stream>>>(
        src, dst, es::SourceEnd(src, inAccess), plan, inAccess.numCols(), inAccess.numRows(), left, top, reverseParam);
    NVCV_CHECK_THROW(cudaGetLastError());
    return true;
}

template<typename T, NVCVBorderType B, bool UseCarrierStream = true, bool PreferReflect101RowCarrier = false>
struct CopyMakeBorderTensorDispatcher
{
    static void call(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                     const T &borderValue, const int left, const int top, cudaStream_t stream)
    {
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);

        int2 dstSize{outAccess->numCols(), outAccess->numRows()};

        constexpr int kBlockDimX = kBlock;
        constexpr int kBlockDimY = kBlock / 4;

        dim3 blockSize(kBlockDimX, kBlockDimY, 1);

        int64_t srcMaxStride = inAccess->sampleStride() * inAccess->numSamples();
        int64_t dstMaxStride = outAccess->sampleStride() * outAccess->numSamples();

        if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input or output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<int32_t>::max);
        }

        // Taken after the extent check so a tensor the per-pixel kernel rejects still reports that,
        // rather than silently succeeding on the byte path.
        // Wide reflection carriers regress on reference GPUs; retain their existing pixel kernels.
        if constexpr (B == NVCV_BORDER_REFLECT101 && (std::is_same_v<T, uchar3> || PreferReflect101RowCarrier))
        {
            bool useRowCarrier = PreferReflect101RowCarrier;
            if constexpr (std::is_same_v<T, uchar3>)
            {
                if (outAccess->numCols() > inAccess->numCols())
                {
                    const int64_t pixelsPerSample = static_cast<int64_t>(inAccess->numRows()) * inAccess->numCols()
                                                  + static_cast<int64_t>(outAccess->numRows()) * outAccess->numCols();
                    useRowCarrier = cvcuda::priv::UseRGB8Reflect101TensorRowCarrier(
                        pixelsPerSample, inAccess->numSamples(), CurrentDeviceReflect101L2CacheSizeOrZero());
                }
            }
            if (useRowCarrier
                && (TryCopyMakeBorderReflect101RowCarrier<static_cast<int>(sizeof(T))>(stream, inData, outData,
                                                                                       *inAccess, *outAccess, left, top)
                    || TryCopyMakeBorderReflect101GeneralRowCarrier<static_cast<int>(sizeof(T))>(
                        stream, inData, outData, *inAccess, *outAccess, left, top)))
                return;
        }
        constexpr bool kSlowReflection
            = (B == NVCV_BORDER_REFLECT101
               && (std::is_same_v<T, half4> || (sizeof(T) == 8 && cuda::NumElements<T> == 1)))
           || (sizeof(cuda::BaseType<T>) == 8 && cuda::NumElements<T> == 3
               && (B == NVCV_BORDER_REFLECT || B == NVCV_BORDER_REFLECT101));
        if constexpr (!kSlowReflection && UseCarrierStream)
        {
            if (TryCopyMakeBorderCarrierStream<static_cast<int>(sizeof(T)), B>(stream, inData, outData, *inAccess,
                                                                               *outAccess, left, top))
            {
                return;
            }
        }

        // BorderWrap for the border pixels; a raw TensorWrap for the interior vector fast path.
        auto src = cuda::CreateBorderWrapNHW<const T, B, int32_t>(inData, borderValue);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        // This kernel splits exactly one 32-byte pixel; narrower pixels use the existing packs.
        if constexpr (sizeof(T) == 2 * sizeof(uint4))
        {
            const auto aligned
                = [](const nvcv::TensorDataStridedCuda &data, const nvcv::TensorDataAccessStridedImagePlanar &access)
            {
                return ((reinterpret_cast<uintptr_t>(data.basePtr()) | access.rowStride() | access.sampleStride())
                        & (alignof(uint4) - 1))
                    == 0;
            };
            if (aligned(inData, *inAccess) && aligned(outData, *outAccess))
            {
                uint4 low, high;
                std::memcpy(&low, &borderValue, sizeof(low));
                std::memcpy(&high, reinterpret_cast<const unsigned char *>(&borderValue) + sizeof(low), sizeof(high));
                dim3 gridSize(util::DivUp(dstSize.x, kBlockDimX / 2), util::DivUp(dstSize.y, kBlockDimY),
                              outAccess->numSamples());
                CopyMakeBorderTensorWide<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, left, top, low, high);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        // The uint3 (12B, 3-element interleaved) vector pack and the REPLICATE border both regress
        // on A100 (and give no win on H100), so keep the scalar copy for those; the vectorized
        // interior fast path is used only where it measurably helps (1-/4-element interleaved and
        // planar single-channel, which dispatch through the single-channel path).
        //
        // Every grid divisor below is a power of two (kBlockDimX/kBlockDimY are 32/8, and
        // kTensorNIX is 16/sizeof(T) on this branch), so the legacy ceil((float)a/b) and DivUp's
        // (a+b-1)/b agree exactly and no grid dimension changes.
        // A wide pixel has kTensorNIX == 0; if its 16-byte alignment check failed above,
        // retain the scalar copy rather than dividing by zero in the vector grid.
        if constexpr (cuda::NumElements<T> == 3 || B == NVCV_BORDER_REPLICATE || kTensorNIX<T> == 0)
        {
            dim3 gridSize(util::DivUp(dstSize.x, kBlockDimX), util::DivUp(dstSize.y, kBlockDimY),
                          outAccess->numSamples());
            CopyMakeBorderTensorKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, left, top);
        }
        else
        {
            int2 srcSize{inAccess->numCols(), inAccess->numRows()};
            dim3 gridSize(util::DivUp(dstSize.x, kBlockDimX * kTensorNIX<T>), util::DivUp(dstSize.y, kBlockDimY),
                          outAccess->numSamples());
            auto srcRaw = cuda::CreateTensorWrapNHW<const T, int32_t>(inData);

            CopyMakeBorderTensorKernelVec<decltype(src), decltype(srcRaw), decltype(dst), T>
                <<<gridSize, blockSize, 0, stream>>>(src, srcRaw, dst, dstSize, srcSize, left, top);
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<typename T> // uchar3 float3 uchar float
void RunCopyMakeBorderTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                             const int top, const int left, const NVCVBorderType borderType, const float4 &borderValue,
                             cudaStream_t stream)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

    typedef void (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           const T &borderValue, const int left, const int top, cudaStream_t stream);

    static const func_t funcs[] = {CopyMakeBorderTensorDispatcher<T, NVCV_BORDER_CONSTANT>::call,
                                   CopyMakeBorderTensorDispatcher<T, NVCV_BORDER_REPLICATE>::call,
                                   CopyMakeBorderTensorDispatcher<T, NVCV_BORDER_REFLECT>::call,
                                   CopyMakeBorderTensorDispatcher<T, NVCV_BORDER_WRAP>::call,
                                   CopyMakeBorderTensorDispatcher<T, NVCV_BORDER_REFLECT101>::call};

    funcs[borderType](inData, outData, bvalue, left, top, stream);
}

inline void RunCopyMakeBorder(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              const int top, const int left, const NVCVBorderType borderType, const float4 &borderValue,
                              cudaStream_t stream)
{
    CvFormat inputFormat  = ToCvFormat(inData.layout());
    CvFormat outputFormat = ToCvFormat(outData.layout());

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%s) and output (%s)", FormatName(inputFormat),
                              FormatName(outputFormat));
    }

    const bool isPlanar = IsPlanar(inputFormat);

    if (inData.dtype() != outData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType between input and output");
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    CvType dataType = ToCvType(inData.dtype());

    const int channels = inAccess->numChannels();

    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (!isPlanar && channels == 2 && dataType != kU8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid channel number %d for DataType %d, 2 channels only supported for 8bit Unsigned",
                              channels, (int)dataType);
    }

    if (!IsSupportedCvType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", (int)dataType);
    }

    if (!IsSupportedBorderType(borderType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid borderType %d", (int)borderType);
    }

    const int rows    = inAccess->numRows();
    const int cols    = inAccess->numCols();
    const int outRows = outAccess->numRows();
    const int outCols = outAccess->numCols();

    if (!(top >= 0 && outRows >= top + rows && left >= 0 && outCols >= left + cols))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid border %d %d %d %d, top >= 0 && bottom >= 0 && left >= 0 && right >= 0, in "
                              "resolution: %dx%d, out resolution: %dx%d",
                              top, outRows - top - rows, left, outCols - left - cols, rows, cols, outRows, outCols);
    }

    typedef void (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           const int top, const int left, const NVCVBorderType borderType, const float4 &borderValue,
                           cudaStream_t stream);

    // The constant border value is converted to the element type, so the F16 row instantiates the
    // real half kernels instead of aliasing the 16-bit integer ones.
    // clang-format off
    static const func_t funcs[9][4] = {
        {RunCopyMakeBorderTensor<uchar1>      , RunCopyMakeBorderTensor<uchar2>       , RunCopyMakeBorderTensor<uchar3>      , RunCopyMakeBorderTensor<uchar4>      },
        {RunCopyMakeBorderTensor<char1>       , 0 /*RunCopyMakeBorderTensor<char2>*/  , RunCopyMakeBorderTensor<char3>       , RunCopyMakeBorderTensor<char4>       },
        {RunCopyMakeBorderTensor<ushort1>     , 0 /*RunCopyMakeBorderTensor<ushort2>*/, RunCopyMakeBorderTensor<ushort3>     , RunCopyMakeBorderTensor<ushort4>     },
        {RunCopyMakeBorderTensor<short1>      , 0 /*RunCopyMakeBorderTensor<short2>*/, RunCopyMakeBorderTensor<short3>      , RunCopyMakeBorderTensor<short4>      },
        {RunCopyMakeBorderTensor<int1>        , 0 /*RunCopyMakeBorderTensor<int2>*/   , RunCopyMakeBorderTensor<int3>        , RunCopyMakeBorderTensor<int4>        },
        {RunCopyMakeBorderTensor<float1>      , 0 /*RunCopyMakeBorderTensor<float2>*/, RunCopyMakeBorderTensor<float3>      , RunCopyMakeBorderTensor<float4>      },
        {RunCopyMakeBorderTensor<double1>     , 0 /*RunCopyMakeBorderTensor<double2>*/, RunCopyMakeBorderTensor<double3>    , RunCopyMakeBorderTensor<double4_16a> },
        {RunCopyMakeBorderTensor<half1>       , 0 /*RunCopyMakeBorderTensor<half2>*/ , RunCopyMakeBorderTensor<half3>       , RunCopyMakeBorderTensor<half4>       },
        {RunCopyMakeBorderTensor<longlong1>   , 0 /*RunCopyMakeBorderTensor<longlong2>*/, RunCopyMakeBorderTensor<longlong3>, RunCopyMakeBorderTensor<longlong4_16a>}
    };
    // clang-format on

    if (isPlanar)
    {
        const func_t planarFunc = funcs[dataType][0];
        NVCV_ASSERT(planarFunc != 0);

        bool useVector     = false;
        bool useRowCarrier = false;
        if (dataType == kF32 && channels == 3 && borderType == NVCV_BORDER_REFLECT101 && outCols > cols)
        {
            const int64_t planeElements = static_cast<int64_t>(rows) * cols + static_cast<int64_t>(outRows) * outCols;
            useRowCarrier               = UsePlanarRGBF32Reflect101Vector(planeElements, inAccess->numSamples(),
                                                                          CurrentDeviceReflect101L2CacheSizeOrZero());
            useVector = useRowCarrier && cvcuda::priv::UseLegacyPlanarReflect101VectorForSM(CurrentDeviceSMOrZero());
        }

        for (int c = 0; c < channels; ++c)
        {
            auto planeIn  = TensorPlanarChannelView(inData, *inAccess, c);
            auto planeOut = TensorPlanarChannelView(outData, *outAccess, c);

            if (useVector)
            {
                CopyMakeBorderTensorDispatcher<float1, NVCV_BORDER_REFLECT101, false>::call(
                    planeIn, planeOut, float1{cuda::GetElement(borderValue, c)}, left, top, stream);
            }
            else if (useRowCarrier)
            {
                CopyMakeBorderTensorDispatcher<float1, NVCV_BORDER_REFLECT101, true, true>::call(
                    planeIn, planeOut, float1{cuda::GetElement(borderValue, c)}, left, top, stream);
            }
            else
            {
                planarFunc(planeIn, planeOut, top, left, borderType,
                           cuda::SetAll<float4>(cuda::GetElement(borderValue, c)), stream);
            }
        }
        return;
    }

    const func_t func = funcs[dataType][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, top, left, borderType, borderValue, stream);
}

// ---------------------------------------------------------------------------------------------
// VarShape path
// ---------------------------------------------------------------------------------------------

template<typename T>
constexpr int kVarShapeNIX = sizeof(T) == 1 ? 8
                           : std::is_same_v<cuda::BaseType<T>, float> && sizeof(T) == 4
                               ? 2
                               : (cuda::NumElements<T> == 3 ? sizeof(uint3) : sizeof(uint4)) / sizeof(T);

template<typename T, int NIX>
using VarShapePackType
    = std::conditional_t<sizeof(T) * NIX == sizeof(uint), uint,
                         std::conditional_t<sizeof(T) * NIX == sizeof(uint2), uint2,
                                            std::conditional_t<sizeof(T) * NIX == sizeof(uint3), uint3, uint4>>>;

template<typename T, int NIX>
constexpr uintptr_t kVarShapeMask = alignof(VarShapePackType<T, NIX>) - 1;

template<class SrcWrapper, class DstWrapper>
__global__ void CopyMakeBorderVarShapeKernel(const SrcWrapper src, DstWrapper dst,
                                             const cuda::Tensor3DWrap<int, int32_t> left_,
                                             const cuda::Tensor3DWrap<int, int32_t> top_, int out_height, int out_width)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    int       left    = *left_.ptr(0, 0, batch_idx);
    int       top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    if (x < out_width && y < out_height)
    {
        int3 srcCoord = {x_shift, y_shift, batch_idx};
        if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
            *dst.ptr(batch_idx, y, x) = src.value(srcCoord);
        else
            *dst.ptr(batch_idx, y, x) = src[srcCoord];
    }
}

template<bool PLANAR, bool FIXED_OUTPUT, class SrcWrapper, class SrcRawWrapper, class DstWrapper, typename T,
         int NIX = kVarShapeNIX<T>>
__global__ void CopyMakeBorderVarShapeKernelVec(const SrcWrapper src, const SrcRawWrapper srcRaw, DstWrapper dst,
                                                const cuda::Tensor3DWrap<int, int32_t> left_,
                                                const cuda::Tensor3DWrap<int, int32_t> top_, int fixedHeight,
                                                int fixedWidth, int plane)
{
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0        = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    const int batch_idx = blockIdx.z;

    int outHeight = fixedHeight;
    int outWidth  = fixedWidth;
    if constexpr (!FIXED_OUTPUT)
    {
        if constexpr (PLANAR)
        {
            outHeight = dst.height(batch_idx, plane);
            outWidth  = dst.width(batch_idx, plane);
        }
        else
        {
            outHeight = dst.height(batch_idx);
            outWidth  = dst.width(batch_idx);
        }
    }

    if (x0 >= outWidth || y >= outHeight)
        return;

    const int left = *left_.ptr(0, 0, batch_idx);
    const int top  = *top_.ptr(0, 0, batch_idx);
    const int sx0  = x0 - left;
    const int sy   = y - top;

    int srcHeight;
    int srcWidth;
    if constexpr (PLANAR)
    {
        srcHeight = srcRaw.height(batch_idx, plane);
        srcWidth  = srcRaw.width(batch_idx, plane);
    }
    else
    {
        srcHeight = srcRaw.height(batch_idx);
        srcWidth  = srcRaw.width(batch_idx);
    }

    const bool full         = x0 + NIX - 1 < outWidth;
    const bool interiorRow  = sy >= 0 && sy < srcHeight;
    const bool interiorSpan = sx0 >= 0 && sx0 + NIX - 1 < srcWidth;

    if constexpr (!PLANAR && SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT && std::is_same_v<T, uchar3> && NIX == 4)
    {
        if (full && (!interiorRow || sx0 <= -NIX || sx0 >= srcWidth))
        {
            T *dp = dst.ptr(batch_idx, y, x0);
            if ((reinterpret_cast<uintptr_t>(dp) & kVarShapeMask<T, NIX>) == 0)
            {
                const uint3                    value = cuda::StaticCast<uint>(src.value(int3{sx0, sy, batch_idx}));
                const uint                     rgb   = value.x | (value.y << 8) | (value.z << 16);
                const VarShapePackType<T, NIX> packed{rgb | (value.x << 24), (rgb >> 8) | (rgb << 16),
                                                      (rgb >> 16) | (rgb << 8)};
                *reinterpret_cast<VarShapePackType<T, NIX> *>(dp) = packed;
                return;
            }
        }
    }

    if (full && interiorRow && interiorSpan)
    {
        const T *sp;
        T       *dp;
        if constexpr (PLANAR)
        {
            sp = srcRaw.ptr(batch_idx, plane, sy, sx0);
            dp = dst.ptr(batch_idx, plane, y, x0);
        }
        else
        {
            sp = srcRaw.ptr(batch_idx, sy, sx0);
            dp = dst.ptr(batch_idx, y, x0);
        }

        if ((reinterpret_cast<uintptr_t>(sp) & kVarShapeMask<T, NIX>) == 0
            && (reinterpret_cast<uintptr_t>(dp) & kVarShapeMask<T, NIX>) == 0)
        {
            *reinterpret_cast<VarShapePackType<T, NIX> *>(dp) = *reinterpret_cast<const VarShapePackType<T, NIX> *>(sp);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX; ++i) dp[i] = sp[i];
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = x0 + i;
            if (x < outWidth)
            {
                if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                {
                    if constexpr (PLANAR)
                        *dst.ptr(batch_idx, plane, y, x) = src.value(int4{x - left, sy, plane, batch_idx});
                    else
                        *dst.ptr(batch_idx, y, x) = src.value(int3{x - left, sy, batch_idx});
                }
                else
                {
                    if constexpr (PLANAR)
                        *dst.ptr(batch_idx, plane, y, x) = src[int4{x - left, sy, plane, batch_idx}];
                    else
                        *dst.ptr(batch_idx, y, x) = src[int3{x - left, sy, batch_idx}];
                }
            }
        }
    }
}

template<class SrcWrapper, class DstWrapper>
__global__ void CopyMakeBorderVarShapeKernel(const SrcWrapper src, DstWrapper dst,
                                             const cuda::Tensor3DWrap<int, int32_t> left_,
                                             const cuda::Tensor3DWrap<int, int32_t> top_)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int left    = *left_.ptr(0, 0, batch_idx);
    const int top     = *top_.ptr(0, 0, batch_idx);
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);

    if (x < out_width && y < out_height)
    {
        int3 srcCoord = {x_shift, y_shift, batch_idx};
        if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
            *dst.ptr(batch_idx, y, x) = src.value(srcCoord);
        else
            *dst.ptr(batch_idx, y, x) = src[srcCoord];
    }
}

// Every grid divisor below is a power of two -- kBlock is 32, BLOCK_Y is 2/4/8, and NIX is
// 1/2/4/8 -- so the legacy ceil((float)a/b) and DivUp's (a+b-1)/b agree exactly and no grid
// dimension changes.
template<NVCVBorderType B, typename T, bool FORCE_SCALAR = false>
struct CopyMakeBorderVarShapeDispatcher
{
    static constexpr int BLOCK_Y
        = FORCE_SCALAR                                ? kBlock / 4
        : B == NVCV_BORDER_CONSTANT && sizeof(T) == 1 ? 2
        : B == NVCV_BORDER_REFLECT101 && std::is_same_v<cuda::BaseType<T>, uchar> && cuda::NumElements<T> >= 3 ? 2
        : B == NVCV_BORDER_REFLECT101 || B == NVCV_BORDER_REPLICATE                                            ? 4
                                                                                                               : 8;

    static constexpr int NIX
        = sizeof(T) == 1 && (B == NVCV_BORDER_CONSTANT || B == NVCV_BORDER_REPLICATE) ? 4 : kVarShapeNIX<T>;

    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &src, cuda::Tensor3DWrap<T> dst,
                     const T &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, cudaStream_t stream)
    {
        dim3 blockSize(kBlock, BLOCK_Y, 1);

        cuda::BorderVarShapeWrap<const T, B> brdSrc(src, borderValue);
        if constexpr (NIX > 1 && !FORCE_SCALAR)
        {
            dim3 gridSize(util::DivUp(max_width, kBlock * NIX), util::DivUp(max_height, BLOCK_Y), src.numImages());
            cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
            CopyMakeBorderVarShapeKernelVec<false, true, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, max_height, max_width, 0);
        }
        else
        {
            dim3 gridSize(util::DivUp(max_width, kBlock), util::DivUp(max_height, BLOCK_Y), src.numImages());
            CopyMakeBorderVarShapeKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top, max_height,
                                                                             max_width);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &src, cuda::ImageBatchVarShapeWrap<T> dst,
                     const T &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, cudaStream_t stream)
    {
        constexpr int blockY
            = !FORCE_SCALAR && B == NVCV_BORDER_CONSTANT && sizeof(T) == 1 && std::is_same_v<cuda::BaseType<T>, uchar>
                ? 4
                : BLOCK_Y;
        dim3 blockSize(kBlock, blockY, 1);

        cuda::BorderVarShapeWrap<const T, B> brdSrc(src, borderValue);
        if constexpr (NIX > 1 && !FORCE_SCALAR)
        {
            dim3 gridSize(util::DivUp(max_width, kBlock * NIX), util::DivUp(max_height, blockY), src.numImages());
            cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);
            CopyMakeBorderVarShapeKernelVec<false, false, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, 0, 0, 0);
        }
        else
        {
            dim3 gridSize(util::DivUp(max_width, kBlock), util::DivUp(max_height, blockY), src.numImages());
            CopyMakeBorderVarShapeKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<NVCVBorderType B, typename T>
struct CopyMakeBorderVarShapePlanarDispatcher
{
    static constexpr int kBlockDimY = kBlock / 4;
    static constexpr int NIX        = sizeof(T) == 1 ? 4 : kVarShapeNIX<T>;

    // Every planar element type reaching this dispatcher (uchar/ushort/short/float/__half) yields
    // NIX >= 2, so only the vectorized kernel is ever launched; there is no scalar planar fallback.
    static_assert(NIX > 1);

    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &src, cuda::Tensor4DWrap<T> dst,
                     const float4 &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, int channels,
                     cudaStream_t stream)
    {
        dim3 blockSize(kBlock, kBlockDimY, 1);
        dim3 gridSize(util::DivUp(max_width, kBlock * NIX), util::DivUp(max_height, kBlockDimY), src.numImages());

        cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);

        for (int c = 0; c < channels; ++c)
        {
            const T bVal = static_cast<T>(cuda::GetElement(borderValue, c));

            cuda::BorderVarShapeWrap<const T, B> brdSrc(src, bVal);
            CopyMakeBorderVarShapeKernelVec<true, true, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, max_height, max_width, c);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }

    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &src, cuda::ImageBatchVarShapeWrap<T> dst,
                     const float4 &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                     const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width, int channels,
                     cudaStream_t stream)
    {
        dim3 blockSize(kBlock, kBlockDimY, 1);
        dim3 gridSize(util::DivUp(max_width, kBlock * NIX), util::DivUp(max_height, kBlockDimY), src.numImages());

        cuda::ImageBatchVarShapeWrap<const T> srcRaw(src);

        for (int c = 0; c < channels; ++c)
        {
            const T bVal = static_cast<T>(cuda::GetElement(borderValue, c));

            cuda::BorderVarShapeWrap<const T, B> brdSrc(src, bVal);
            CopyMakeBorderVarShapeKernelVec<true, false, decltype(brdSrc), decltype(srcRaw), decltype(dst), T, NIX>
                <<<gridSize, blockSize, 0, stream>>>(brdSrc, srcRaw, dst, left, top, 0, 0, c);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<typename T, int cn, typename OutType> // uchar3 float3 uchar float
void RunCopyMakeBorderVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                               const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                               const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    typedef cuda::MakeType<T, cn> src_type;
    const src_type                brdVal = cuda::DropCast<cn>(cuda::StaticCast<T>(value));

    cuda::Tensor3DWrap<int, int32_t> topVec(top);
    cuda::Tensor3DWrap<int, int32_t> leftVec(left);

    auto outSize = MaxImageSize(outData);

    using out_type =
        typename std::conditional<std::is_same<OutType, nvcv::TensorDataStridedCuda>::value,
                                  cuda::Tensor3DWrap<src_type>, cuda::ImageBatchVarShapeWrap<src_type>>::type;

    out_type dstWrap(outData);

    if constexpr (std::is_same_v<src_type, uchar3>)
    {
        if (borderType == NVCV_BORDER_REFLECT101 && !UsePackedRGB8Reflect101ForSM(CurrentDeviceSMOrZero()))
        {
            CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_REFLECT101, src_type, true>::call(
                inData, dstWrap, brdVal, leftVec, topVec, outSize.h, outSize.w, stream);
            return;
        }
    }

    typedef void (*func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &src, out_type dst,
                           const src_type &borderValue, const cuda::Tensor3DWrap<int, int32_t> &left,
                           const cuda::Tensor3DWrap<int, int32_t> &top, int max_height, int max_width,
                           cudaStream_t stream);

    static const func_t funcs[] = {CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_CONSTANT, src_type>::call,
                                   CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_REPLICATE, src_type>::call,
                                   CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_REFLECT, src_type>::call,
                                   CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_WRAP, src_type>::call,
                                   CopyMakeBorderVarShapeDispatcher<NVCV_BORDER_REFLECT101, src_type>::call};

    funcs[borderType](inData, dstWrap, brdVal, leftVec, topVec, outSize.h, outSize.w, stream);
}

template<typename T, typename OutType>
void RunCopyMakeBorderVarShapePlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData, const OutType &outData,
                                     const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                                     const NVCVBorderType borderType, const float4 value, cudaStream_t stream,
                                     int channels)
{
    cuda::Tensor3DWrap<int, int32_t> topVec(top);
    cuda::Tensor3DWrap<int, int32_t> leftVec(left);

    auto outSize = MaxImageSize(outData);

    using out_type = std::conditional_t<std::is_same_v<OutType, nvcv::TensorDataStridedCuda>, cuda::Tensor4DWrap<T>,
                                        cuda::ImageBatchVarShapeWrap<T>>;

    // A planar (N)CHW tensor is addressed as an NCHW 4D wrap; the var-shape batch wraps directly.
    auto dstWrap = [&]() -> out_type
    {
        if constexpr (std::is_same_v<OutType, nvcv::TensorDataStridedCuda>)
        {
            return cuda::CreateTensorWrapNCHW<T>(outData);
        }
        else
        {
            return out_type(outData);
        }
    }();

    typedef void (*func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &src, out_type dst, const float4 &borderValue,
                           const cuda::Tensor3DWrap<int, int32_t> &left, const cuda::Tensor3DWrap<int, int32_t> &top,
                           int max_height, int max_width, int channels, cudaStream_t stream);

    static const func_t funcs[] = {CopyMakeBorderVarShapePlanarDispatcher<NVCV_BORDER_CONSTANT, T>::call,
                                   CopyMakeBorderVarShapePlanarDispatcher<NVCV_BORDER_REPLICATE, T>::call,
                                   CopyMakeBorderVarShapePlanarDispatcher<NVCV_BORDER_REFLECT, T>::call,
                                   CopyMakeBorderVarShapePlanarDispatcher<NVCV_BORDER_WRAP, T>::call,
                                   CopyMakeBorderVarShapePlanarDispatcher<NVCV_BORDER_REFLECT101, T>::call};

    funcs[borderType](inData, dstWrap, value, leftVec, topVec, outSize.h, outSize.w, channels, stream);
}

template<class OutType>
inline CvType OutputCvType(const OutType &dataOut)
{
    if constexpr (std::is_same_v<OutType, nvcv::TensorDataStridedCuda>)
    {
        return ToCvType(dataOut.dtype());
    }
    else
    {
        return ToCvType(dataOut.uniqueFormat());
    }
}

template<class OutType>
void RunCopyMakeBorderVarShapeInfer(const nvcv::ImageBatchVarShapeDataStridedCuda &dataIn, const OutType &dataOut,
                                    const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                                    const NVCVBorderType borderType, const float4 value, cudaStream_t stream)
{
    CvFormat inputFormat  = ToCvFormat(dataIn);
    CvFormat outputFormat = ToCvFormat(dataOut);

    // The var-shape -> var-shape overload keeps the legacy strict format equality; the
    // var-shape -> tensor overload only requires the same layout family, since the two
    // containers classify their batch dimension differently.
    if constexpr (std::is_same_v<OutType, nvcv::ImageBatchVarShapeDataStridedCuda>)
    {
        if (inputFormat != outputFormat)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid DataFormat between input (%s) and output (%s)", FormatName(inputFormat),
                                  FormatName(outputFormat));
        }
    }
    else
    {
        if (!SameLayoutFamily(inputFormat, outputFormat))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Invalid DataFormat between input (%s) and output (%s)", FormatName(inputFormat),
                                  FormatName(outputFormat));
        }
    }

    const bool isPlanar = IsPlanar(inputFormat);

    int channels = dataIn.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    CvType dataType    = ToCvType(dataIn.uniqueFormat());
    CvType outDataType = OutputCvType(dataOut);

    if (dataType != outDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DataType of input and output must be equal, but got %d and %d", (int)dataType,
                              (int)outDataType);
    }

    if (!isPlanar && channels == 2 && dataType != kU8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid channel number %d for DataType %d, 2 channels only supported for 8bit Unsigned",
                              channels, (int)dataType);
    }

    if (!IsSupportedCvType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", (int)dataType);
    }

    if (!IsSupportedBorderType(borderType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid borderType %d", (int)borderType);
    }

    CvType   leftDataType = ToCvType(left.dtype());
    CvFormat leftFormat   = ToCvFormat(left.layout());
    if (!(leftFormat == kNHWC || leftFormat == kHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Left DataFormat %s",
                              FormatName(leftFormat));
    }
    if (leftDataType != kS32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Left DataType %d", (int)leftDataType);
    }

    CvType   topDataType = ToCvType(top.dtype());
    CvFormat topFormat   = ToCvFormat(top.layout());
    if (!(topFormat == kNHWC || topFormat == kHWC))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Top DataFormat %s", FormatName(topFormat));
    }
    if (topDataType != kS32)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Top DataType %d", (int)topDataType);
    }

    typedef void (*func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &dIn, const OutType &dOut,
                           const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                           const NVCVBorderType borderType, const float4 value, cudaStream_t stream);

    // The constant border value is converted to the element type, so the F16 row instantiates the
    // real half kernels instead of aliasing the 16-bit integer ones.
    // clang-format off
    static const func_t funcs[9][4] = {
        {RunCopyMakeBorderVarShape<uchar, 1, OutType>,        RunCopyMakeBorderVarShape<uchar, 2, OutType>,        RunCopyMakeBorderVarShape<uchar, 3, OutType>,        RunCopyMakeBorderVarShape<uchar, 4, OutType>       },
        {RunCopyMakeBorderVarShape<signed char, 1, OutType>,        0 /*RunCopyMakeBorderVarShape<schar , 2>*/, RunCopyMakeBorderVarShape<signed char, 3, OutType>,        RunCopyMakeBorderVarShape<signed char, 4, OutType>       },
        {RunCopyMakeBorderVarShape<ushort, 1, OutType>,       0 /*RunCopyMakeBorderVarShape<ushort, 2>*/, RunCopyMakeBorderVarShape<ushort, 3, OutType>,       RunCopyMakeBorderVarShape<ushort, 4, OutType>      },
        {RunCopyMakeBorderVarShape<short, 1, OutType>,        0 /*RunCopyMakeBorderVarShape<short , 2>*/, RunCopyMakeBorderVarShape<short, 3, OutType>,        RunCopyMakeBorderVarShape<short, 4, OutType>       },
        {RunCopyMakeBorderVarShape<int, 1, OutType>,          0 /*RunCopyMakeBorderVarShape<int   , 2>*/, RunCopyMakeBorderVarShape<int, 3, OutType>,          RunCopyMakeBorderVarShape<int, 4, OutType>         },
        {RunCopyMakeBorderVarShape<float, 1, OutType>,        0 /*RunCopyMakeBorderVarShape<float , 2>*/, RunCopyMakeBorderVarShape<float, 3, OutType>,        RunCopyMakeBorderVarShape<float, 4, OutType>       },
        {RunCopyMakeBorderVarShape<double, 1, OutType>,       0 /*RunCopyMakeBorderVarShape<double, 2>*/, RunCopyMakeBorderVarShape<double, 3, OutType>,       RunCopyMakeBorderVarShape<double, 4, OutType>      },
        {RunCopyMakeBorderVarShape<__half, 1, OutType>,       0 /*RunCopyMakeBorderVarShape<__half, 2>*/, RunCopyMakeBorderVarShape<__half, 3, OutType>,       RunCopyMakeBorderVarShape<__half, 4, OutType>      },
        {RunCopyMakeBorderVarShape<long long, 1, OutType>,    0 /*RunCopyMakeBorderVarShape<long long, 2>*/, RunCopyMakeBorderVarShape<long long, 3, OutType>, RunCopyMakeBorderVarShape<long long, 4, OutType>   }
    };
    // clang-format on

    if (isPlanar)
    {
        typedef void (*planar_func_t)(const nvcv::ImageBatchVarShapeDataStridedCuda &dIn, const OutType &dOut,
                                      const nvcv::TensorDataStridedCuda &top, const nvcv::TensorDataStridedCuda &left,
                                      const NVCVBorderType borderType, const float4 value, cudaStream_t stream,
                                      int channels);

        // The per-channel constant border value is converted to the element type, so the F16 slot
        // instantiates the real half kernel instead of aliasing the 16-bit integer one.
        static const planar_func_t planarFuncs[9] = {
            RunCopyMakeBorderVarShapePlanar<uchar, OutType>,
            RunCopyMakeBorderVarShapePlanar<signed char, OutType>,
            RunCopyMakeBorderVarShapePlanar<ushort, OutType>,
            RunCopyMakeBorderVarShapePlanar<short, OutType>,
            RunCopyMakeBorderVarShapePlanar<int, OutType>,
            RunCopyMakeBorderVarShapePlanar<float, OutType>,
            RunCopyMakeBorderVarShapePlanar<double, OutType> /*64F*/,
            RunCopyMakeBorderVarShapePlanar<__half, OutType> /*16F*/,
            RunCopyMakeBorderVarShapePlanar<long long, OutType> /*64S*/,
        };

        const planar_func_t planarFunc = planarFuncs[dataType];
        NVCV_ASSERT(planarFunc != 0);
        planarFunc(dataIn, dataOut, top, left, borderType, value, stream, channels);
        return;
    }

    const func_t func = funcs[dataType][channels - 1];
    NVCV_ASSERT(func != 0);

    func(dataIn, dataOut, top, left, borderType, value, stream);
}

} // namespace

namespace cvcuda::priv {

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const int top,
                                const int left, const NVCVBorderType borderMode, const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CopyMakeBorder::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    RunCopyMakeBorder(*inData, *outData, top, left, borderMode, borderValue, stream);
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::Tensor &out,
                                const nvcv::Tensor &top, const nvcv::Tensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CopyMakeBorder::operator()[ImageBatch->Tensor]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData  = RequireCudaTensor(out, "Output");
    auto topData  = RequireCudaTensor(top, "Top");
    auto leftData = RequireCudaTensor(left, "Left");

    RunCopyMakeBorderVarShapeInfer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream);
}

void CopyMakeBorder::operator()(cudaStream_t stream, const nvcv::ImageBatch &in, const nvcv::ImageBatch &out,
                                const nvcv::Tensor &top, const nvcv::Tensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CopyMakeBorder::operator()[ImageBatch]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto topData  = RequireCudaTensor(top, "Top");
    auto leftData = RequireCudaTensor(left, "Left");

    RunCopyMakeBorderVarShapeInfer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream);
}

} // namespace cvcuda::priv
