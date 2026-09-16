/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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
#include "OpCvtColor.hpp"

#include "LabColorConversion.cuh"
#include "PhotometricBound.cuh"

#include <cvcuda/cuda_tools/Compat.hpp>
#include <cvcuda/cuda_tools/HalfTypes.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

// NOTE: Below are the "standard" (NTSC and ITU Rec.601) RGB to luma conversion
// coefficients. More accurate coefficents, given as comments on the right, are
// found at http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html and
// https://www.imagemagick.org/include/api/pixel.php.
static constexpr float R2YF = 0.299f; // 0.298839
static constexpr float G2YF = 0.587f; // 0.586811
static constexpr float B2YF = 0.114f; // 0.114350

static constexpr int gray_shift = 15;
static constexpr int yuv_shift  = 14;
static constexpr int RY15       = 9798;  // == R2YF*32768 + 0.5
static constexpr int GY15       = 19235; // == G2YF*32768 + 0.5
static constexpr int BY15       = 3735;  // == B2YF*32768 + 0.5

static constexpr int R2Y  = 4899;  // == R2YF*16384
static constexpr int G2Y  = 9617;  // == G2YF*16384
static constexpr int B2Y  = 1868;  // == B2YF*16384
static constexpr int R2VI = 14369; // == R2VF*16384
static constexpr int B2UI = 8061;  // == B2UF*16384

static constexpr float B2UF = 0.492f; // 0.492111: U = (B - Y) * B2UF + 0.5
static constexpr float R2VF = 0.877f; // 0.877283: V = (R - Y) * R2VF + 0.5

static constexpr int U2BI = 33292;
static constexpr int U2GI = -6472;
static constexpr int V2GI = -9519;
static constexpr int V2RI = 18678;

static constexpr float U2BF = 2.032f;
static constexpr float U2GF = -0.395f;
static constexpr float V2GF = -0.581f;
static constexpr float V2RF = 1.140f;

// Coefficients for YUV420sp to RGB conversion
static constexpr int ITUR_BT_601_CY    = 1220542;
static constexpr int ITUR_BT_601_CUB   = 2116026;
static constexpr int ITUR_BT_601_CUG   = -409993;
static constexpr int ITUR_BT_601_CVG   = -852492;
static constexpr int ITUR_BT_601_CVR   = 1673527;
static constexpr int ITUR_BT_601_SHIFT = 20;
// Coefficients for RGB to YUV420p conversion
static constexpr int ITUR_BT_601_CRY = 269484;
static constexpr int ITUR_BT_601_CGY = 528482;
static constexpr int ITUR_BT_601_CBY = 102760;
static constexpr int ITUR_BT_601_CRU = -155188;
static constexpr int ITUR_BT_601_CGU = -305135;
static constexpr int ITUR_BT_601_CBU = 460324;
static constexpr int ITUR_BT_601_CGV = -385875;
static constexpr int ITUR_BT_601_CBV = -74448;

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

#define BLOCK 32

#define DEVICE_INLINE __device__ __forceinline__
#define GLOBAL_BOUNDS __global__ __launch_bounds__(Policy::BlockSize)

namespace {

using uchar = unsigned char;

// Opaque alpha for an image with base type BT: the dtype maximum for integers, 1 for the float
// family (including __half, whose maximum is not a constant expression).
template<typename BT>
inline __host__ __device__ BT Alpha()
{
    return cvcuda::priv::PhotometricUpperBound<BT>();
}

// The planar kernels map the batch onto grid dimension z, which CUDA caps at 65535.
inline void RequirePlanarBatchFitsGridZ(int numSamples)
{
    if (numSamples > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor requires numImages <= 65535 (CUDA grid-z limit)");
    }
}

inline void RequireUniformPlaneDataType(nvcv::ImageFormat fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }
}

// Element classification: bits-per-channel plus data kind, the exact axis the legacy dispatch
// switched on. Reproducing it as an enum (rather than as direct nvcv::DataType comparisons)
// keeps the admitted set identical -- notably `k8U` also covers the packed 2U8/3U8/4U8 dtypes,
// and every (kind, width) pair with no name here is rejected, as it was before.
enum class ElemType
{
    k8U,
    k8S,
    k16U,
    k16S,
    k16F,
    k32S,
    k32F,
    k64F
};

enum class ImageLayout
{
    kNHWC,
    kHWC,
    kNCHW,
    kCHW
};

struct ImageShape
{
    int N; // batch
    int C; // channel
    int H; // height
    int W; // width

    bool operator==(const ImageShape &s) const
    {
        return s.N == N && s.H == H && s.W == W && s.C == C;
    }

    bool operator!=(const ImageShape &s) const
    {
        return !(*this == s);
    }
};

inline ElemType FloatElemType(int32_t bpc)
{
    if (bpc == 64)
        return ElemType::k64F;
    if (bpc == 32)
        return ElemType::k32F;
    if (bpc == 16)
        return ElemType::k16F;

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bits-per-channel (%d) for float data type",
                          bpc);
}

inline ElemType SignedElemType(int32_t bpc)
{
    if (bpc == 8)
        return ElemType::k8S;
    if (bpc == 16)
        return ElemType::k16S;
    if (bpc == 32)
        return ElemType::k32S;

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bits-per-channel (%d) for signed data type",
                          bpc);
}

inline ElemType UnsignedElemType(int32_t bpc)
{
    if (bpc == 8)
        return ElemType::k8U;
    if (bpc == 16)
        return ElemType::k16U;

    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bits-per-channel (%d) for unsigned data type",
                          bpc);
}

inline ElemType ClassifyElemType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        return FloatElemType(bpc);

    case nvcv::DataKind::SIGNED:
        return SignedElemType(bpc);

    case nvcv::DataKind::UNSIGNED:
        return UnsignedElemType(bpc);

    case nvcv::DataKind::COMPLEX:
        break;

    case nvcv::DataKind::UNSPECIFIED:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported");
}

inline ElemType ClassifyElemType(nvcv::DataType dtype)
{
    auto bpc = dtype.bitsPerChannel();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        // A mixed-width packing (NVCV_PACKING_X32_Y24b8 is a real one) was rejected before the
        // element type was classified at all; keep the rejection at that same point.
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ClassifyElemType(bpc[0], dtype.dataKind());
}

inline ElemType ClassifyElemType(nvcv::ImageFormat fmt)
{
    RequireUniformPlaneDataType(fmt);

    return ClassifyElemType(fmt.planeDataType(0));
}

inline ImageLayout GetImageLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return ImageLayout::kNCHW;
    }
    else if (layout == nvcv::TENSOR_CHW)
    {
        return ImageLayout::kCHW;
    }
    else if (layout == nvcv::TENSOR_NHWC)
    {
        return ImageLayout::kNHWC;
    }
    else if (layout == nvcv::TENSOR_HWC)
    {
        return ImageLayout::kHWC;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }
}

inline ImageLayout GetImageLayout(const nvcv::ImageBatchVarShapeDataStridedCuda &imgBatch)
{
    nvcv::ImageFormat fmt = imgBatch.uniqueFormat();
    if (!fmt)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All images must have the same format");
    }

    RequireUniformPlaneDataType(fmt);

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar images must have one channel per plane");
        }

        return imgBatch.numImages() >= 2 ? ImageLayout::kNCHW : ImageLayout::kCHW;
    }
    else
    {
        return imgBatch.numImages() >= 2 ? ImageLayout::kNHWC : ImageLayout::kHWC;
    }
}

inline bool IsPlanar(ImageLayout layout)
{
    return layout == ImageLayout::kNCHW || layout == ImageLayout::kCHW;
}

inline ImageShape GetImageShape(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return ImageShape{static_cast<int>(access.numSamples()), access.numChannels(), access.numRows(), access.numCols()};
}

// The kernels are instantiated on int32_t strides whenever the whole batch is addressable with
// them, so the wider (and slower) int64_t instantiation is reserved for batches that need it.
inline bool NeedsWideStrides(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                             const nvcv::TensorDataAccessStridedImagePlanar &outAccess)
{
    return std::max(inAccess.sampleStride() * inAccess.numSamples(), outAccess.sampleStride() * outAccess.numSamples())
         > cuda::TypeTraits<int32_t>::max;
}

// The Tensor and the ImageBatchVarShape paths were two legacy translation units with independent
// kernel families that happen to share several names; they stay in sibling namespaces so each
// kernel keeps the identity (and therefore the generated code) it had before.
namespace tensor {

template<typename T, typename StrideT>
using TensorWrap3D = nvcv::cuda::Tensor3DWrap<T, StrideT>;

template<typename T, typename StrideT>
using TensorWrap4D = nvcv::cuda::Tensor4DWrap<T, StrideT>;

template<int BlockWidth_, int BlockHeight_, int RowsPerThread_>
struct CvtKernelPolicy
{
    static_assert(BlockWidth_ % 32 == 0);
    static constexpr int BlockWidth      = BlockWidth_;
    static constexpr int BlockHeight     = BlockHeight_;
    static constexpr int BlockSize       = BlockWidth * BlockHeight;
    static constexpr int RowsPerThread   = RowsPerThread_;
    static constexpr int TileWidth       = BlockWidth;
    static constexpr int TileHeight      = BlockHeight * RowsPerThread;
    static constexpr int ThreadRowStride = BlockHeight;
};

template<typename Policy, int N_IN, int N_OUT, typename EltT, typename LoadOpT, typename ConvOpT, typename StoreOpT>
DEVICE_INLINE void color_conversion_common(LoadOpT load_op, ConvOpT conv_op, StoreOpT store_op, int2 size)
{
    const int x         = blockIdx.x * Policy::TileWidth + threadIdx.x;
    const int y0        = blockIdx.y * Policy::TileHeight + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (x >= size.x)
    {
        return;
    }

    // Branchless efficient path for inner blocks.
    if (y0 + Policy::TileHeight <= size.y)
    {
        EltT r_in[Policy::RowsPerThread][N_IN];
        EltT r_out[Policy::RowsPerThread][N_OUT];

#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++)
        {
            const int y = y0 + Policy::ThreadRowStride * i;
            load_op(r_in[i], batch_idx, x, y);
        }
#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++) conv_op(r_in[i], r_out[i]);
#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++)
        {
            const int y = y0 + Policy::ThreadRowStride * i;
            store_op(r_out[i], batch_idx, x, y);
        }
    }
    else
    {
        int y = y0;
        for (int i = 0; i < Policy::RowsPerThread && y < size.y; i++)
        {
            EltT r_in[N_IN];
            EltT r_out[N_OUT];

            load_op(r_in, batch_idx, x, y);
            conv_op(r_in, r_out);
            store_op(r_out, batch_idx, x, y);

            y += Policy::ThreadRowStride;
        }
    }
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load3_nhwc(const TensorWrap3D<const SrcT, StrideT> &src, EltT &C0, EltT &C1, EltT &C2, int batch_idx,
                              int x, int y)
{
    SrcT vec = *src.ptr(batch_idx, y, x);
    C0       = vec.x;
    C1       = vec.y;
    C2       = vec.z;
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store3_nhwc(const TensorWrap3D<DstT, StrideT> &dst, EltT C0, EltT C1, EltT C2, int batch_idx, int x,
                               int y)
{
    DstT vec;
    vec.x                     = C0;
    vec.y                     = C1;
    vec.z                     = C2;
    *dst.ptr(batch_idx, y, x) = vec;
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load_bgra_nhwc(const TensorWrap3D<const SrcT, StrideT> &src, EltT &B, EltT &G, EltT &R, EltT &A,
                                  int batch_idx, int x, int y, int bidx)
{
    SrcT vec = *src.ptr(batch_idx, y, x);
    B        = bidx == 0 ? vec.x : vec.z;
    G        = vec.y;
    R        = bidx == 0 ? vec.z : vec.x;
    if constexpr (nvcv::cuda::NumComponents<SrcT> == 4)
    {
        A = vec.w;
    }
    else
    {
        A = Alpha<EltT>();
    }
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store_bgra_nhwc(const TensorWrap3D<DstT, StrideT> &dst, EltT B, EltT G, EltT R, EltT A,
                                   int batch_idx, int x, int y, int bidx)
{
    DstT vec;
    vec.x = bidx == 0 ? B : R;
    vec.y = G;
    vec.z = bidx == 0 ? R : B;
    if constexpr (nvcv::cuda::NumComponents<DstT> == 4)
    {
        vec.w = A;
    }
    *dst.ptr(batch_idx, y, x) = vec;
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load3_nchw(const TensorWrap4D<const SrcT, StrideT> &src, EltT &C0, EltT &C1, EltT &C2, int batch_idx,
                              int x, int y)
{
    C0 = *src.ptr(batch_idx, 0, y, x);
    C1 = *src.ptr(batch_idx, 1, y, x);
    C2 = *src.ptr(batch_idx, 2, y, x);
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store3_nchw(const TensorWrap4D<DstT, StrideT> &dst, EltT C0, EltT C1, EltT C2, int batch_idx, int x,
                               int y)
{
    *dst.ptr(batch_idx, 0, y, x) = C0;
    *dst.ptr(batch_idx, 1, y, x) = C1;
    *dst.ptr(batch_idx, 2, y, x) = C2;
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load_bgra_nchw(const TensorWrap4D<const SrcT, StrideT> &src, EltT &B, EltT &G, EltT &R, EltT &A,
                                  int batch_idx, int x, int y, int bidx, int srcChannels)
{
    B = *src.ptr(batch_idx, bidx, y, x);
    G = *src.ptr(batch_idx, 1, y, x);
    R = *src.ptr(batch_idx, bidx ^ 2, y, x);
    A = srcChannels == 4 ? *src.ptr(batch_idx, 3, y, x) : Alpha<EltT>();
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store_bgra_nchw(const TensorWrap4D<DstT, StrideT> &dst, EltT B, EltT G, EltT R, EltT A,
                                   int batch_idx, int x, int y, int bidx, int dstChannels)
{
    *dst.ptr(batch_idx, bidx, y, x)     = B;
    *dst.ptr(batch_idx, 1, y, x)        = G;
    *dst.ptr(batch_idx, bidx ^ 2, y, x) = R;
    if (dstChannels == 4)
    {
        *dst.ptr(batch_idx, 3, y, x) = A;
    }
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void rgb_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 4, 4, EltT>(
        [&src, bidx] __device__(EltT(&r_in)[4], int batch_idx, int x, int y)
        { load_bgra_nhwc(src, r_in[0], r_in[1], r_in[2], r_in[3], batch_idx, x, y, bidx); },
        [] __device__(const EltT(&r_in)[4], EltT(&r_out)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_out[i] = r_in[i];
        },
        [&dst] __device__(const EltT(&r_out)[4], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_out[0], r_out[1], r_out[2], r_out[3], batch_idx, x, y, 0); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void rgb_to_bgr_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, int srcChannels, int dstChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 4, 4, EltT>(
        [&src, bidx, srcChannels] __device__(EltT(&r_in)[4], int batch_idx, int x, int y)
        { load_bgra_nchw(src, r_in[0], r_in[1], r_in[2], r_in[3], batch_idx, x, y, bidx, srcChannels); },
        [] __device__(const EltT(&r_in)[4], EltT(&r_out)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_out[i] = r_in[i];
        },
        [&dst, dstChannels] __device__(const EltT(&r_out)[4], int batch_idx, int x, int y)
        { store_bgra_nchw(dst, r_out[0], r_out[1], r_out[2], r_out[3], batch_idx, x, y, 0, dstChannels); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void gray_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                    int2 dstSize)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 1, 4, EltT>(
        [&src] __device__(EltT(&r_gray)[1], int batch_idx, int x, int y) { r_gray[0] = *src.ptr(batch_idx, y, x); },
        [] __device__(const EltT(&r_gray)[1], EltT(&r_BGRA)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_BGRA[i] = r_gray[0];
        },
        [&dst] __device__(const EltT(&r_BGRA)[4], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGRA[0], r_BGRA[1], r_BGRA[2], r_BGRA[3], batch_idx, x, y, 0); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void gray_to_bgr_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                    int2 dstSize, int dstChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 1, 4, EltT>(
        [&src] __device__(EltT(&r_gray)[1], int batch_idx, int x, int y) { r_gray[0] = *src.ptr(batch_idx, 0, y, x); },
        [] __device__(const EltT(&r_gray)[1], EltT(&r_BGRA)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_BGRA[i] = r_gray[0];
        },
        [&dst, dstChannels] __device__(const EltT(&r_BGRA)[4], int batch_idx, int x, int y)
        { store_bgra_nchw(dst, r_BGRA[0], r_BGRA[1], r_BGRA[2], r_BGRA[3], batch_idx, x, y, 0, dstChannels); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_gray_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                    int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 1, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_gray)[1])
        {
            if constexpr (std::is_integral_v<EltT>)
                r_gray[0]
                    = (EltT)CV_DESCALE((int)r_BGR[0] * BY15 + (int)r_BGR[1] * GY15 + (int)r_BGR[2] * RY15, gray_shift);
            else
                r_gray[0] = (EltT)(r_BGR[0] * B2YF + r_BGR[1] * G2YF + r_BGR[2] * R2YF);
        },
        [&dst] __device__(const EltT(&r_gray)[1], int batch_idx, int x, int y)
        { *dst.ptr(batch_idx, y, x) = r_gray[0]; },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_gray_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                    int2 dstSize, int bidx, int srcChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 1, EltT>(
        [&src, bidx, srcChannels] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nchw(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx, srcChannels);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_gray)[1])
        {
            if constexpr (std::is_integral_v<EltT>)
                r_gray[0]
                    = (EltT)CV_DESCALE((int)r_BGR[0] * BY15 + (int)r_BGR[1] * GY15 + (int)r_BGR[2] * RY15, gray_shift);
            else
                r_gray[0] = (EltT)(r_BGR[0] * B2YF + r_BGR[1] * G2YF + r_BGR[2] * R2YF);
        },
        [&dst] __device__(const EltT(&r_gray)[1], int batch_idx, int x, int y)
        { *dst.ptr(batch_idx, 0, y, x) = r_gray[0]; },
        dstSize);
}

template<typename T>
DEVICE_INLINE void bgr_to_yuv_int(T B_, T G_, T R_, T &Y_, T &Cb_, T &Cr_)
{
    constexpr int C0 = R2Y, C1 = G2Y, C2 = B2Y, C3 = R2VI, C4 = B2UI;
    constexpr int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) << yuv_shift;

    const int B = B_, G = G_, R = R_;

    const int Y  = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    const int Cr = CV_DESCALE((R - Y) * C3 + delta, yuv_shift);
    const int Cb = CV_DESCALE((B - Y) * C4 + delta, yuv_shift);

    Y_  = cuda::SaturateCast<T>(Y);
    Cb_ = cuda::SaturateCast<T>(Cb);
    Cr_ = cuda::SaturateCast<T>(Cr);
}

DEVICE_INLINE void bgr_to_yuv_float(float B, float G, float R, float &Y, float &Cb, float &Cr)
{
    constexpr float C0 = R2YF, C1 = G2YF, C2 = B2YF, C3 = R2VF, C4 = B2UF;
    constexpr float delta = 0.5f;

    Y  = R * C0 + G * C1 + B * C2;
    Cr = (R - Y) * C3 + delta;
    Cb = (B - Y) * C4 + delta;
}

// __half overload: the float helper's reference outputs cannot bind to __half, so compute in
// float (the __half inputs widen losslessly) and round once on the outputs.
DEVICE_INLINE void bgr_to_yuv_half(__half B, __half G, __half R, __half &Y_, __half &Cb_, __half &Cr_)
{
    float Y, Cb, Cr;
    bgr_to_yuv_float(B, G, R, Y, Cb, Cr);
    Y_  = cuda::SaturateCast<__half>(Y);
    Cb_ = cuda::SaturateCast<__half>(Cb);
    Cr_ = cuda::SaturateCast<__half>(Cr);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_yuv_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_YCbCr)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_yuv_int(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
            else if constexpr (std::is_same_v<EltT, __half>)
                bgr_to_yuv_half(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
            else
                bgr_to_yuv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
        },
        [&dst] __device__(const EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { store3_nhwc(dst, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_yuv_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, int srcChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx, srcChannels] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nchw(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx, srcChannels);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_YCbCr)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_yuv_int(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
            else if constexpr (std::is_same_v<EltT, __half>)
                bgr_to_yuv_half(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
            else
                bgr_to_yuv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
        },
        [&dst] __device__(const EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { store3_nchw(dst, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        dstSize);
}

template<typename T>
DEVICE_INLINE void yuv_to_bgr_int(T Y_, T Cb_, T Cr_, T &B_, T &G_, T &R_)
{
    constexpr int C0 = V2RI, C1 = V2GI, C2 = U2GI, C3 = U2BI;
    constexpr int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));

    const int Y = Y_, Cb = Cb_, Cr = Cr_;
    const int B = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    const int G = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    const int R = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    B_ = cuda::SaturateCast<T>(B);
    G_ = cuda::SaturateCast<T>(G);
    R_ = cuda::SaturateCast<T>(R);
}

DEVICE_INLINE void yuv_to_bgr_flt(float Y, float Cb, float Cr, float &B, float &G, float &R)
{
    constexpr float C0 = V2RF, C1 = V2GF, C2 = U2GF, C3 = U2BF;
    constexpr float delta = 0.5f;

    B = Y + (Cb - delta) * C3;
    G = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
    R = Y + (Cr - delta) * C0;
}

// __half overload: compute in float and round once on the outputs (see bgr_to_yuv_half).
DEVICE_INLINE void yuv_to_bgr_half(__half Y, __half Cb, __half Cr, __half &B_, __half &G_, __half &R_)
{
    float B, G, R;
    yuv_to_bgr_flt(Y, Cb, Cr, B, G, R);
    B_ = cuda::SaturateCast<__half>(B);
    G_ = cuda::SaturateCast<__half>(G);
    R_ = cuda::SaturateCast<__half>(R);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void yuv_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { load3_nhwc(src, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        [] __device__(const EltT(&r_YCbCr)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                yuv_to_bgr_int(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            else if constexpr (std::is_same_v<EltT, __half>)
                yuv_to_bgr_half(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            else
                yuv_to_bgr_flt(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>(), batch_idx, x, y, bidx); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void yuv_to_bgr_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, int dstChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { load3_nchw(src, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        [] __device__(const EltT(&r_YCbCr)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                yuv_to_bgr_int(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            else if constexpr (std::is_same_v<EltT, __half>)
                yuv_to_bgr_half(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            else
                yuv_to_bgr_flt(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
        },
        [&dst, bidx, dstChannels] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nchw(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>(), batch_idx, x, y, bidx, dstChannels); },
        dstSize);
}

DEVICE_INLINE void bgr_to_hsv_uchar(uchar b8, uchar g8, uchar r8, uchar &h8, uchar &s8, uchar &v8, bool isFullRange)
{
    const int hrange    = isFullRange ? 256 : 180;
    const int hsv_shift = 12;

    const int b = (int)b8;
    const int g = (int)g8;
    const int r = (int)r8;

    const int vmin = cuda::min(b, cuda::min(g, r));
    const int v    = cuda::max(b, cuda::max(g, r));

    const int diff = v - vmin;
    const int vr   = v == r ? -1 : 0;
    const int vg   = v == g ? -1 : 0;

    const int hdiv_table = diff == 0 ? 0 : cuda::SaturateCast<int>((hrange << hsv_shift) / (6.f * diff));
    const int sdiv_table = v == 0 ? 0 : cuda::SaturateCast<int>((255 << hsv_shift) / (float)v);

    const int s = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    int       h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));

    h = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += h < 0 ? hrange : 0;

    h8 = cuda::SaturateCast<uint8_t>(h);
    s8 = (uint8_t)s;
    v8 = (uint8_t)v;
}

DEVICE_INLINE void bgr_to_hsv_float(float b, float g, float r, float &h, float &s, float &v)
{
    float vmin = cuda::min(r, cuda::min(g, b));
    v          = cuda::max(r, cuda::max(g, b));
    float diff = v - vmin;
    s          = diff / (fabs(v) + FLT_EPSILON);
    diff       = 60.f / (diff + FLT_EPSILON);

    // clang-format off
    if      (v == r) h = (g - b) * diff;
    else if (v == g) h = (b - r) * diff + 120.f;
    else             h = (r - g) * diff + 240.f;

    if (h < 0.f) h += 360.f;
    // clang-format on
}

// __half overload: compute in float and round once on the outputs (see bgr_to_yuv_half). Hue
// keeps the float path's [0, 360) range; the FULL-range flag only affects 8-bit inputs.
DEVICE_INLINE void bgr_to_hsv_half(__half b, __half g, __half r, __half &h_, __half &s_, __half &v_)
{
    float h, s, v;
    bgr_to_hsv_float(b, g, r, h, s, v);
    h_ = cuda::SaturateCast<__half>(h);
    s_ = cuda::SaturateCast<__half>(s);
    v_ = cuda::SaturateCast<__half>(v);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_hsv_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [isFullRange] __device__(const EltT(&r_BGR)[3], EltT(&r_HSV)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_hsv_uchar(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2], isFullRange);
            else if constexpr (std::is_same_v<EltT, __half>)
                bgr_to_hsv_half(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2]);
            else
                bgr_to_hsv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2]);
        },
        [&dst] __device__(const EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { store3_nhwc(dst, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_hsv_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange, int srcChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx, srcChannels] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nchw(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx, srcChannels);
        },
        [isFullRange] __device__(const EltT(&r_BGR)[3], EltT(&r_HSV)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_hsv_uchar(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2], isFullRange);
            else if constexpr (std::is_same_v<EltT, __half>)
                bgr_to_hsv_half(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2]);
            else
                bgr_to_hsv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2]);
        },
        [&dst] __device__(const EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { store3_nchw(dst, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        dstSize);
}

template<bool RGB2Lab, bool BGR, bool SRGB, typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void rgb_lab_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                int2 dstSize)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_in)[3], int batch_idx, int x, int y)
        { load3_nhwc(src, r_in[0], r_in[1], r_in[2], batch_idx, x, y); },
        [] __device__(const EltT(&r_in)[3], EltT(&r_out)[3])
        {
            if constexpr (RGB2Lab)
            {
                cvcuda::priv::lab::RGBToLab<SRGB, false>(r_in[BGR ? 2 : 0], r_in[1], r_in[BGR ? 0 : 2], r_out[0],
                                                         r_out[1], r_out[2]);
            }
            else
            {
                cvcuda::priv::lab::LabToRGB<SRGB, false>(r_in[0], r_in[1], r_in[2], r_out[BGR ? 2 : 0], r_out[1],
                                                         r_out[BGR ? 0 : 2]);
            }
        },
        [&dst] __device__(const EltT(&r_out)[3], int batch_idx, int x, int y)
        { store3_nhwc(dst, r_out[0], r_out[1], r_out[2], batch_idx, x, y); },
        dstSize);
}

template<bool RGB2Lab, bool BGR, bool SRGB, typename Policy, typename T, typename StrideT>
GLOBAL_BOUNDS void rgb_lab_nchw(const TensorWrap4D<const T, StrideT> src, const TensorWrap4D<T, StrideT> dst,
                                int2 dstSize)
{
    color_conversion_common<Policy, 3, 3, T>(
        [&src] __device__(T(&r_in)[3], int batch_idx, int x, int y)
        { load3_nchw(src, r_in[0], r_in[1], r_in[2], batch_idx, x, y); },
        [] __device__(const T(&r_in)[3], T(&r_out)[3])
        {
            if constexpr (RGB2Lab)
            {
                cvcuda::priv::lab::RGBToLab<SRGB, false>(r_in[BGR ? 2 : 0], r_in[1], r_in[BGR ? 0 : 2], r_out[0],
                                                         r_out[1], r_out[2]);
            }
            else
            {
                cvcuda::priv::lab::LabToRGB<SRGB, false>(r_in[0], r_in[1], r_in[2], r_out[BGR ? 2 : 0], r_out[1],
                                                         r_out[BGR ? 0 : 2]);
            }
        },
        [&dst] __device__(const T(&r_out)[3], int batch_idx, int x, int y)
        { store3_nchw(dst, r_out[0], r_out[1], r_out[2], batch_idx, x, y); },
        dstSize);
}

template<typename T>
DEVICE_INLINE T select4_reg(const T (&tab)[4], int idx)
{
    // Random access in a register array of size 4, with 6 instructions.
    // The compiler was generating 10 instructions for tab[idx].
    T out;
    out = idx == 1 ? tab[1] : tab[0];
    out = idx == 2 ? tab[2] : out;
    out = idx == 3 ? tab[3] : out;
    return out;
}

DEVICE_INLINE void hsv_to_bgr_float(float h, float s, float v, float &b, float &g, float &r)
{
    if (s == 0)
        b = g = r = v;
    else
    {
        h += 6 * (h < 0);
        int idx = static_cast<int>(h); // Sector index.
        h -= idx;                      // Fractional part of h.
        idx = (idx % 6) << 2;          // Shift index for sector LUT.

        // clang-format off
        const float tab[4] {v,
                            v * (1 - s),
                            v * (1 - s * h),
                            v * (1 - s * (1 - h))};
        // clang-format on

        constexpr int32_t idx_lutb = 0x00200311;
        constexpr int32_t idx_lutg = 0x00112003;
        constexpr int32_t idx_lutr = 0x00031120;

        b = select4_reg(tab, (idx_lutb >> idx) & 0xf);
        g = select4_reg(tab, (idx_lutg >> idx) & 0xf);
        r = select4_reg(tab, (idx_lutr >> idx) & 0xf);
    }
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void hsv_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { load3_nhwc(src, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        [isFullRange] __device__(const EltT(&r_HSV)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_same_v<EltT, uchar>)
            {
                const float     scaleH  = isFullRange ? (6.0f / 256.0f) : (6.0f / 180.0f);
                constexpr float scaleSV = 1.0f / 255.0f;

                float Bf, Gf, Rf;

                hsv_to_bgr_float((float)r_HSV[0] * scaleH, r_HSV[1] * scaleSV, r_HSV[2] * scaleSV, Bf, Gf, Rf);

                r_BGR[0] = cuda::SaturateCast<uchar>(Bf * 255.0f);
                r_BGR[1] = cuda::SaturateCast<uchar>(Gf * 255.0f);
                r_BGR[2] = cuda::SaturateCast<uchar>(Rf * 255.0f);
            }
            else if constexpr (std::is_same_v<EltT, __half>)
            {
                // hsv_to_bgr_float's reference outputs cannot bind to __half: compute in float
                // (H scaling matches the float path) and round once on the stores.
                constexpr float scaleH = 6.0f / 360.0f;

                float B, G, R;

                hsv_to_bgr_float(r_HSV[0] * scaleH, r_HSV[1], r_HSV[2], B, G, R);

                r_BGR[0] = cuda::SaturateCast<EltT>(B);
                r_BGR[1] = cuda::SaturateCast<EltT>(G);
                r_BGR[2] = cuda::SaturateCast<EltT>(R);
            }
            else
            {
                constexpr float scaleH = 6.0f / 360.0f;

                hsv_to_bgr_float(r_HSV[0] * scaleH, r_HSV[1], r_HSV[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            }
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>(), batch_idx, x, y, bidx); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void hsv_to_bgr_nchw(const TensorWrap4D<const SrcT, StrideT> src, const TensorWrap4D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange, int dstChannels)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { load3_nchw(src, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        [isFullRange] __device__(const EltT(&r_HSV)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_same_v<EltT, uchar>)
            {
                const float     scaleH  = isFullRange ? (6.0f / 256.0f) : (6.0f / 180.0f);
                constexpr float scaleSV = 1.0f / 255.0f;

                float Bf, Gf, Rf;

                hsv_to_bgr_float((float)r_HSV[0] * scaleH, r_HSV[1] * scaleSV, r_HSV[2] * scaleSV, Bf, Gf, Rf);

                r_BGR[0] = cuda::SaturateCast<uchar>(Bf * 255.0f);
                r_BGR[1] = cuda::SaturateCast<uchar>(Gf * 255.0f);
                r_BGR[2] = cuda::SaturateCast<uchar>(Rf * 255.0f);
            }
            else if constexpr (std::is_same_v<EltT, __half>)
            {
                // hsv_to_bgr_float's reference outputs cannot bind to __half: compute in float
                // (H scaling matches the float path) and round once on the stores.
                constexpr float scaleH = 6.0f / 360.0f;

                float B, G, R;

                hsv_to_bgr_float(r_HSV[0] * scaleH, r_HSV[1], r_HSV[2], B, G, R);

                r_BGR[0] = cuda::SaturateCast<EltT>(B);
                r_BGR[1] = cuda::SaturateCast<EltT>(G);
                r_BGR[2] = cuda::SaturateCast<EltT>(R);
            }
            else
            {
                constexpr float scaleH = 6.0f / 360.0f;

                hsv_to_bgr_float(r_HSV[0] * scaleH, r_HSV[1], r_HSV[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            }
        },
        [&dst, bidx, dstChannels] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nchw(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>(), batch_idx, x, y, bidx, dstChannels); },
        dstSize);
}

template<bool IsSemiPlanar, typename EltT, typename StrideT>
DEVICE_INLINE void load_yuv420(const nvcv::cuda::Tensor4DWrap<const EltT, StrideT> &src, EltT &Y, EltT &U, EltT &V,
                               int2 size, int batch_idx, int x, int y, int uidx)
{
    if constexpr (IsSemiPlanar)
    {
        // U and V are subsampled at half the full resolution (in both x and y), combined (i.e., interleaved), and
        // arranged as full rows after the full resolution Y data. Example memory layout for 4 x 4 image (NV12):
        //   Y_00 Y_01 Y_02 Y_03
        //   Y_10 Y_11 Y_12 Y_13
        //   Y_20 Y_21 Y_22 Y_23
        //   Y_30 Y_31 Y_32 Y_33
        //   U_00 V_00 U_02 V_02
        //   U_20 V_20 U_22 V_22
        // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
        // and Y_11. Each full U-V row represents 2 rows of Y values. Some layouts (e.g., NV21) swap the location
        // of the U and V values in each U-V pair (indicated by the uidx parameter).

        const int uv_y = size.y + y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
        const int uv_x = (x & ~1);       // Convert x to even # (set lowest bit to 0).

        Y = *src.ptr(batch_idx, y, x);                    // Y (luma) is at full resolution.
        U = *src.ptr(batch_idx, uv_y, uv_x + uidx);       // Some formats swap the U and V elements (as indicated
        V = *src.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)); //   by the uidx parameter).
    }
    else
    {
        // U and V are subsampled at half the full resolution (in both x and y) and arranged as non-interleaved planes
        // (i.e., planar format). Each subsampled U and V "plane" is arranged as full rows after the full resolution Y
        // data--so two consecutive subsampled U or V rows are combined into one row spanning the same width as the Y
        // plane. Example memory layout for 4 x 4 image (e.g. I420):
        //   Y_00 Y_01 Y_02 Y_03
        //   Y_10 Y_11 Y_12 Y_13
        //   Y_20 Y_21 Y_22 Y_23
        //   Y_30 Y_31 Y_32 Y_33
        //   U_00 U_02 U_20 U_22
        //   V_00 V_02 V_20 V_22
        // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
        // and Y_11. Each full U and V row represents 4 rows of Y values. Some layouts (e.g., YV12) swap the location
        // of the U and V planes (indicated by the uidx parameter).

        const int by = size.y + y / 4; // Base row coordinate for U and V: subsampled plane is 1/4 the height.
        const int h4 = size.y / 4;     // Height (# of rows) of each subsampled U and V plane.

        // Compute x position that combines two subsampled rows into one.
        const int uv_x = (x / 2) + ((size.x / 2) & -((y / 2) & 1)); // Second half of row for odd y coordinates.

        Y = *src.ptr(batch_idx, y, x);                       // Y (luma) is at full resolution.
        U = *src.ptr(batch_idx, by + h4 * uidx, uv_x);       // Some formats swap the U and V "planes" (as indicated
        V = *src.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x); //   by the uidx parameter).
    }
}

template<bool IsSemiPlanar, typename EltT, typename StrideT>
DEVICE_INLINE void store_yuv420(const TensorWrap4D<EltT, StrideT> &dst, EltT Y, EltT U, EltT V, int2 size,
                                int batch_idx, int x, int y, int uidx)
{
    if constexpr (IsSemiPlanar)
    {
        // See YUV420 semi-planar layout commments in load_yuv420 above.
        *dst.ptr(batch_idx, y, x) = Y; // Y (luma) is at full resolution.
        if (y % 2 == 0 && x % 2 == 0)
        {
            const int uv_y = size.y + y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
            const int uv_x = (x & ~1);       // Convert x to even # (set lowest bit to 0).

            *dst.ptr(batch_idx, uv_y, uv_x + uidx)       = U; // Some formats swap the U and V elements (as indicated
            *dst.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)) = V; //   by the uidx parameter).
        }
    }
    else
    {
        // See YUV420 planar layout commments in load_yuv420 above.
        *dst.ptr(batch_idx, y, x, 0) = Y; // Y (luma) is at full resolution.
        if (y % 2 == 0 && x % 2 == 0)
        {
            const int by = size.y + y / 4; // Base row coordinate for U and V: subsampled plane is 1/4 the height.
            const int h4 = size.y / 4;     // Height (# of rows) of each subsampled U and V plane.

            // Compute x position that combines two subsampled rows into one.
            const int uv_x = (x / 2) + ((size.x / 2) & -((y / 2) & 1)); // Second half of row for odd y coordinates.

            *dst.ptr(batch_idx, by + h4 * uidx, uv_x)       = U; // Some formats swap the U and V "planes" (as indicated
            *dst.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x) = V; //   by the uidx parameter).
        }
    }
}

DEVICE_INLINE void bgr_to_yuv42xxp(const uchar &b, const uchar &g, const uchar &r, uchar &Y, uchar &U, uchar &V)
{
    const int shifted16 = (16 << ITUR_BT_601_SHIFT);
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    int       yy        = ITUR_BT_601_CRY * r + ITUR_BT_601_CGY * g + ITUR_BT_601_CBY * b + halfShift + shifted16;

    Y = cuda::SaturateCast<uchar>(yy >> ITUR_BT_601_SHIFT);

    const int shifted128 = (128 << ITUR_BT_601_SHIFT);
    int       uu         = ITUR_BT_601_CRU * r + ITUR_BT_601_CGU * g + ITUR_BT_601_CBU * b + halfShift + shifted128;
    int       vv         = ITUR_BT_601_CBU * r + ITUR_BT_601_CGV * g + ITUR_BT_601_CBV * b + halfShift + shifted128;

    U = cuda::SaturateCast<uchar>(uu >> ITUR_BT_601_SHIFT);
    V = cuda::SaturateCast<uchar>(vv >> ITUR_BT_601_SHIFT);
}

DEVICE_INLINE void yuv42xxp_to_bgr(const int &Y, const int &U, const int &V, uchar &b, uchar &g, uchar &r)
{
    //R = 1.164(Y - 16) + 1.596(V - 128)
    //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    //B = 1.164(Y - 16)                  + 2.018(U - 128)

    //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
    //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
    //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
    const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG,
              C4           = ITUR_BT_601_CUB;
    const int yuv4xx_shift = ITUR_BT_601_SHIFT;

    int yy = cuda::max(0, Y - 16) * C0;
    int uu = U - 128;
    int vv = V - 128;

    r = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C1 * vv), yuv4xx_shift));
    g = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C2 * vv + C3 * uu), yuv4xx_shift));
    b = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C4 * uu), yuv4xx_shift));
}

template<bool IsSemiPlanar, typename Policy, typename SrcT, typename EltT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_yuv420_char_nhwc(const TensorWrap3D<const SrcT, StrideT> src,
                                           const TensorWrap4D<EltT, StrideT> dst, int2 size, int bidx, int uidx)
{
    static_assert(std::is_same_v<nvcv::cuda::BaseType<SrcT>, EltT>);
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_YUV)[3])
        { bgr_to_yuv42xxp(r_BGR[0], r_BGR[1], r_BGR[2], r_YUV[0], r_YUV[1], r_YUV[2]); },
        [&dst, uidx, size] __device__(const EltT(&r_YUV)[3], int batch_idx, int x, int y)
        { store_yuv420<IsSemiPlanar>(dst, r_YUV[0], r_YUV[1], r_YUV[2], size, batch_idx, x, y, uidx); },
        size);
}

template<bool IsSemiPlanar, typename Policy, typename EltT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void yuv420_to_bgr_char_nhwc(const TensorWrap4D<const EltT, StrideT> src,
                                           const TensorWrap3D<DstT, StrideT> dst, int2 size, int bidx, int uidx)
{
    static_assert(std::is_same_v<nvcv::cuda::BaseType<DstT>, EltT>);
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, uidx, size] __device__(EltT(&r_YUV)[3], int batch_idx, int x, int y)
        { load_yuv420<IsSemiPlanar>(src, r_YUV[0], r_YUV[1], r_YUV[2], size, batch_idx, x, y, uidx); },
        [] __device__(const EltT(&r_YUV)[3], EltT(&r_BGR)[3])
        {
            yuv42xxp_to_bgr(static_cast<int>(r_YUV[0]), static_cast<int>(r_YUV[1]), static_cast<int>(r_YUV[2]),
                            r_BGR[0], r_BGR[1], r_BGR[2]);
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>(), batch_idx, x, y, bidx); },
        size);
}

// YUV 422 interleaved formats (e.g., YUYV, YVYU, and UYVY) group 2 pixels into groups of 4 elements. Each group of two
// pixels has two distinct luma (Y) values, one for each pixel. The chromaticity values (U and V) are subsampled by a
// factor of two so that there is only one U and one V value for each group of 2 pixels. Example memory layout for
// 4 x 4 image (UYVY format):
//   U_00 Y_00 V_00 Y_01 U_02 Y_02 V_02 Y_03
//   U_10 Y_10 V_10 Y_11 U_12 Y_12 V_12 Y_13
//   U_20 Y_20 V_20 Y_21 U_22 Y_22 V_22 Y_23
//   U_30 Y_30 V_30 Y_31 U_32 Y_32 V_32 Y_33
// Each U and V value corresponds to two Y values--e.g. U_00 and V_00 correspond to Y_00 and Y_10 while U_12 and V_12
// correspond to Y_12 and Y_13. Thus, a given Y value, Y_rc = Y(r,c) (where r is the row, or y coordinate, and c is the
// column, or x coordinate), corresponds to U(r,c') and V(r,c') where c' is the even column coordinate <= c -- that is,
// c' = 2 * floor(c/2) = (c & ~1). Some layouts swap the positions of the chromaticity and luma values (e.g., YUYV)
// (indicated by the yidx parameter) and / or swap the the positions of the U and V chromaticity valus (e.g., YVYU)
// (indicated by the uidx parameter).
// The data layout is treated as a single channel tensor, so each group of 4 values corresponds to two pixels. As such,
// the tensor width is twice the actual pixel width. Thus, it's easiest to process 4 consecutive values (2 pixels) per
// thread.
template<class SrcWrapper, class DstWrapper>
__global__ void yuv422_to_bgr_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int dcn, int bidx, int yidx,
                                        int uidx)
{
    using T = typename DstWrapper::ValueType;

    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dstSize.y)
        return;

    int dst_x = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // Process 2 destination pixels/thread.
    if (dst_x >= dstSize.x)
        return;

    const int batch_idx = blockIdx.z;

    const int src_x = 2 * dst_x;    // Process 4 source elements/thread (i.e., 2 destination pixels).
    const int uv_x  = (src_x & ~3); // Compute "even" x coordinate for U and V (set lowest two bits to 0).

    const T Y0 = *src.ptr(batch_idx, dst_y, src_x + yidx);
    const T Y1 = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
    const T U  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + uidx);
    const T V  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + (uidx ^ 2));

    T r{0}, g{0}, b{0};

    yuv42xxp_to_bgr(int(Y0), int(U), int(V), b, g, r);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = Alpha<T>();

    dst_x++; // Move to next output pixel.
    yuv42xxp_to_bgr(int(Y1), int(U), int(V), b, g, r);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = Alpha<T>();
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void yuv422_to_gray_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int yidx)
{
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dstSize.y)
        return;

    int dst_x = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // Process 2 destination pixels/thread.
    if (dst_x >= dstSize.x)
        return;

    const int batch_idx = blockIdx.z;

    const int src_x = 2 * dst_x; // Process 4 source elements/thread.

    *dst.ptr(batch_idx, dst_y, dst_x++) = *src.ptr(batch_idx, dst_y, src_x + yidx);
    *dst.ptr(batch_idx, dst_y, dst_x)   = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
}

template<typename SrcT, typename DstT>
inline void Launch_BGR_to_RGB(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              ImageShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);
    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    rgb_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_BGR_to_RGB_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                     int srcChannels, int dstChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);
    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNCHW<const T>(inData);
    auto dstWrap = cuda::CreateTensorWrapNCHW<T>(outData);
    rgb_to_bgr_nchw<Policy>
        <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, srcChannels, dstChannels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_RGB(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       NVCVColorConversionCode code, cudaStream_t stream)
{
    int sch  = (code == NVCV_COLOR_BGRA2BGR || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int dch  = (code == NVCV_COLOR_BGR2BGRA || code == NVCV_COLOR_BGR2RGBA || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int bidx = (code != NVCV_COLOR_BGRA2BGR && code != NVCV_COLOR_BGR2BGRA) ? 2 : 0;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != sch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d -- expecting %d",
                              inputShape.C, sch);
    }
    if (outputShape.C != dch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d -- expecting %d",
                              outputShape.C, dch);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mismatched input / output DataTypes");
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Shape mismatch -- output tensor shape doesn't match input tensor shape");
    }
    if (isPlanar)
    {
#define CVCUDA_BGR2RGB_PLANAR_CASE(T) \
    return Launch_BGR_to_RGB_Planar<T>(inData, outData, inputShape, bidx, sch, dch, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
        case ElemType::k8S:
            CVCUDA_BGR2RGB_PLANAR_CASE(uchar);
        // F16 runs real __half kernels: pure channel swaps stay bit-identical to the former
        // 16-bit integer route, and an added alpha channel becomes 1.0 (the floating-point
        // opaque value) instead of a reinterpreted integer max.
        case ElemType::k16F:
            CVCUDA_BGR2RGB_PLANAR_CASE(__half);
        case ElemType::k16U:
        case ElemType::k16S:
            CVCUDA_BGR2RGB_PLANAR_CASE(ushort);
        case ElemType::k32S:
            CVCUDA_BGR2RGB_PLANAR_CASE(int);
        case ElemType::k32F:
            CVCUDA_BGR2RGB_PLANAR_CASE(float);
        case ElemType::k64F:
            CVCUDA_BGR2RGB_PLANAR_CASE(double);
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
#undef CVCUDA_BGR2RGB_PLANAR_CASE
    }

#define CVCUDA_BGR2RGB_IF(SCH, DCH, SRC_T, DST_T) \
    if (sch == SCH && dch == DCH)                 \
    return Launch_BGR_to_RGB<SRC_T, DST_T>(inData, outData, inputShape, bidx, stream)

#define CVCUDA_BGR2RGB_CASE(T3, T4)                                                                                    \
    CVCUDA_BGR2RGB_IF(3, 3, T3, T3);                                                                                   \
    else CVCUDA_BGR2RGB_IF(3, 4, T3, T4);                                                                              \
    else CVCUDA_BGR2RGB_IF(4, 3, T4, T3);                                                                              \
    else CVCUDA_BGR2RGB_IF(4, 4, T4, T4);                                                                              \
    else throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input / output channel numbers %d / %d", \
                               sch, dch)

    switch (inDataType)
    {
    case ElemType::k8U:
    case ElemType::k8S:
        CVCUDA_BGR2RGB_CASE(uchar3, uchar4);
    case ElemType::k16F: // Real __half kernels; alpha (when added) becomes 1.0, swaps stay bit-exact.
        CVCUDA_BGR2RGB_CASE(half3, half4);
    case ElemType::k16U:
    case ElemType::k16S:
        CVCUDA_BGR2RGB_CASE(ushort3, ushort4);
    case ElemType::k32S:
        CVCUDA_BGR2RGB_CASE(int3, int4);
    case ElemType::k32F:
        CVCUDA_BGR2RGB_CASE(float3, float4);
    case ElemType::k64F:
        CVCUDA_BGR2RGB_CASE(double3, double4_16a);
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
#undef CVCUDA_BGR2RGB_CASE
#undef CVCUDA_BGR2RGB_IF
}

template<typename SrcT, typename DstT>
inline void Launch_GRAY_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                               ImageShape shape, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 8>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    gray_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_GRAY_to_BGR_Planar(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int dstChannels,
                                      cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 8>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNCHW<const T>(inData);
    auto dstWrap = cuda::CreateTensorWrapNCHW<T>(outData);
    gray_to_bgr_nchw<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, dstChannels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void GRAY_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        NVCVColorConversionCode code, cudaStream_t stream)
{
    int dch = (code == NVCV_COLOR_GRAY2BGRA) ? 4 : 3;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d -- expecting 1",
                              inputShape.C);
    }
    if (outputShape.C != dch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d -- expecting %d",
                              outputShape.C, dch);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mismatched input / output DataTypes");
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Shape mismatch -- output tensor shape doesn't match input tensor shape");
    }
    if (isPlanar)
    {
#define CVCUDA_GRAY2BGR_PLANAR_CASE(T) return Launch_GRAY_to_BGR_Planar<T>(inData, outData, inputShape, dch, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
        case ElemType::k8S:
            CVCUDA_GRAY2BGR_PLANAR_CASE(uchar);
        // F16 runs real __half kernels; the broadcast (including the 4th channel, which copies
        // the gray value like every other dtype) stays bit-identical to the former integer route.
        case ElemType::k16F:
            CVCUDA_GRAY2BGR_PLANAR_CASE(__half);
        case ElemType::k16U:
        case ElemType::k16S:
            CVCUDA_GRAY2BGR_PLANAR_CASE(ushort);
        case ElemType::k32S:
            CVCUDA_GRAY2BGR_PLANAR_CASE(int);
        case ElemType::k32F:
            CVCUDA_GRAY2BGR_PLANAR_CASE(float);
        case ElemType::k64F:
            CVCUDA_GRAY2BGR_PLANAR_CASE(double);
        }
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
#undef CVCUDA_GRAY2BGR_PLANAR_CASE
    }

#define CVCUDA_GRAY2BGR_IF(DCH, SRC_T, DST_T) \
    if (dch == DCH)                           \
    return Launch_GRAY_to_BGR<SRC_T, DST_T>(inData, outData, inputShape, stream)

#define CVCUDA_GRAY2BGR_CASE(T, T3, T4) \
    CVCUDA_GRAY2BGR_IF(3, T, T3);       \
    else CVCUDA_GRAY2BGR_IF(4, T, T4);  \
    else throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", dch)

    switch (inDataType)
    {
    case ElemType::k8U:
    case ElemType::k8S:
        CVCUDA_GRAY2BGR_CASE(uchar, uchar3, uchar4);
    case ElemType::k16F: // Real __half kernels; the gray broadcast stays bit-exact.
        CVCUDA_GRAY2BGR_CASE(__half, half3, half4);
    case ElemType::k16U:
    case ElemType::k16S:
        CVCUDA_GRAY2BGR_CASE(ushort, ushort3, ushort4);
    case ElemType::k32S:
        CVCUDA_GRAY2BGR_CASE(int, int3, int4);
    case ElemType::k32F:
        CVCUDA_GRAY2BGR_CASE(float, float3, float4);
    case ElemType::k64F:
        CVCUDA_GRAY2BGR_CASE(double, double3, double4_16a);
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
#undef CVCUDA_GRAY2BGR_CASE
#undef CVCUDA_GRAY2BGR_IF
}

template<typename SrcT, typename DstT>
inline void Launch_BGR_to_GRAY(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                               ImageShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    bgr_to_gray_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_BGR_to_GRAY_Planar(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                      int srcChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNCHW<const T>(inData);
    auto dstWrap = cuda::CreateTensorWrapNCHW<T>(outData);
    bgr_to_gray_nchw<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, srcChannels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_GRAY(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_RGB2GRAY) ? 2 : 0;
    int sch  = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_BGRA2GRAY) ? 4 : 3;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != sch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d -- expecting %d",
                              inputShape.C, sch);
    }
    if (outputShape.C != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d -- expecting 1",
                              outputShape.C);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Mismatched input / output DataTypes");
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Shape mismatch -- output tensor shape doesn't match input tensor shape");
    }

    if (isPlanar)
    {
#define CVCUDA_BGR2GRAY_PLANAR_CASE(T) \
    return Launch_BGR_to_GRAY_Planar<T>(inData, outData, inputShape, bidx, sch, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
            CVCUDA_BGR2GRAY_PLANAR_CASE(uchar);
        case ElemType::k16U:
            CVCUDA_BGR2GRAY_PLANAR_CASE(ushort);
        case ElemType::k16F: // Luma accumulates in float via the mixed __half operators, rounds on store.
            CVCUDA_BGR2GRAY_PLANAR_CASE(__half);
        case ElemType::k32F:
            CVCUDA_BGR2GRAY_PLANAR_CASE(float);
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
        }
#undef CVCUDA_BGR2GRAY_PLANAR_CASE
    }

#define CVCUDA_BGR2GRAY_IF(SCH, SRC_T, DST_T) \
    if (sch == SCH)                           \
    return Launch_BGR_to_GRAY<SRC_T, DST_T>(inData, outData, inputShape, bidx, stream)

#define CVCUDA_BGR2GRAY_CASE(T, T3, T4) \
    CVCUDA_BGR2GRAY_IF(3, T3, T);       \
    else CVCUDA_BGR2GRAY_IF(4, T4, T);  \
    else throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", sch)

    switch (inDataType)
    {
    case ElemType::k8U:
        CVCUDA_BGR2GRAY_CASE(uchar, uchar3, uchar4);
    case ElemType::k16U:
        CVCUDA_BGR2GRAY_CASE(ushort, ushort3, ushort4);
    case ElemType::k16F: // Luma accumulates in float via the mixed __half operators, rounds on store.
        CVCUDA_BGR2GRAY_CASE(__half, half3, half4);
    case ElemType::k32F:
        CVCUDA_BGR2GRAY_CASE(float, float3, float4);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_BGR2GRAY_CASE
#undef CVCUDA_BGR2GRAY_IF
}

template<typename SrcT, typename DstT>
inline void Launch_BGR_to_YUV(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              ImageShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    bgr_to_yuv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_BGR_to_YUV_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                     int srcChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNCHW<const T>(inData);
    auto dstWrap = cuda::CreateTensorWrapNCHW<T>(outData);
    bgr_to_yuv_nchw<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, srcChannels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_YUV(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_BGR2YUV ? 0 : 2;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }
    if (outputShape != inputShape)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape different than output shape");
    }

    if (isPlanar)
    {
#define CVCUDA_BGR2YUV_PLANAR_CASE(T) \
    return Launch_BGR_to_YUV_Planar<T>(inData, outData, inputShape, bidx, inputShape.C, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
            CVCUDA_BGR2YUV_PLANAR_CASE(uchar);
        case ElemType::k16U:
            CVCUDA_BGR2YUV_PLANAR_CASE(ushort);
        case ElemType::k16F: // Follows the float path (bgr_to_yuv_half computes in float).
            CVCUDA_BGR2YUV_PLANAR_CASE(__half);
        case ElemType::k32F:
            CVCUDA_BGR2YUV_PLANAR_CASE(float);
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
        }
#undef CVCUDA_BGR2YUV_PLANAR_CASE
    }

#define CVCUDA_BGR2YUV_CASE(T3) return Launch_BGR_to_YUV<T3, T3>(inData, outData, inputShape, bidx, stream)
    switch (inDataType)
    {
    case ElemType::k8U:
        CVCUDA_BGR2YUV_CASE(uchar3);
    case ElemType::k16U:
        CVCUDA_BGR2YUV_CASE(ushort3);
    case ElemType::k16F: // Follows the float path (bgr_to_yuv_half computes in float).
        CVCUDA_BGR2YUV_CASE(half3);
    case ElemType::k32F:
        CVCUDA_BGR2YUV_CASE(float3);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_BGR2YUV_CASE
}

template<typename SrcT, typename DstT>
inline void Launch_YUV_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              ImageShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    yuv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_YUV_to_BGR_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                     int dstChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNCHW<const T>(inData);
    auto dstWrap = cuda::CreateTensorWrapNCHW<T>(outData);
    yuv_to_bgr_nchw<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, dstChannels);
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void YUV_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_YUV2BGR ? 0 : 2;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }
    if (outputShape != inputShape)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape different than output shape");
    }

    if (isPlanar)
    {
#define CVCUDA_YUV2BGR_PLANAR_CASE(T) \
    return Launch_YUV_to_BGR_Planar<T>(inData, outData, inputShape, bidx, outputShape.C, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
            CVCUDA_YUV2BGR_PLANAR_CASE(uchar);
        case ElemType::k16U:
            CVCUDA_YUV2BGR_PLANAR_CASE(ushort);
        case ElemType::k16F: // Follows the float path (yuv_to_bgr_half computes in float).
            CVCUDA_YUV2BGR_PLANAR_CASE(__half);
        case ElemType::k32F:
            CVCUDA_YUV2BGR_PLANAR_CASE(float);
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
        }
#undef CVCUDA_YUV2BGR_PLANAR_CASE
    }

#define CVCUDA_YUV2BGR_CASE(T3) return Launch_YUV_to_BGR<T3, T3>(inData, outData, inputShape, bidx, stream)
    switch (inDataType)
    {
    case ElemType::k8U:
        CVCUDA_YUV2BGR_CASE(uchar3);
    case ElemType::k16U:
        CVCUDA_YUV2BGR_CASE(ushort3);
    case ElemType::k16F: // Follows the float path (yuv_to_bgr_half computes in float).
        CVCUDA_YUV2BGR_CASE(half3);
    case ElemType::k32F:
        CVCUDA_YUV2BGR_CASE(float3);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_YUV2BGR_CASE
}

template<bool RGB2Lab, bool BGR, bool SRGB, typename T3>
inline void Launch_RGB_Lab(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                           ImageShape shape, bool strides_64b, cudaStream_t stream)
{
    constexpr int kRowsPerThread = RGB2Lab && std::is_same_v<nvcv::cuda::BaseType<T3>, __half>
                                     ? 8
                                     : (RGB2Lab && std::is_same_v<nvcv::cuda::BaseType<T3>, unsigned char> ? 2 : 4);
    using Policy                 = CvtKernelPolicy<32, 4, kRowsPerThread>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);
    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const T3, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<T3, int64_t>(outData);
        rgb_lab_nhwc<RGB2Lab, BGR, SRGB, Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const T3, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<T3, int32_t>(outData);
        rgb_lab_nhwc<RGB2Lab, BGR, SRGB, Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<bool RGB2Lab, bool BGR, bool SRGB, typename T>
inline void Launch_RGB_Lab_Planar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  ImageShape shape, bool strides_64b, cudaStream_t stream)
{
    constexpr int kRowsPerThread = RGB2Lab && (std::is_same_v<T, __half> || std::is_same_v<T, unsigned char>) ? 8 : 4;
    using Policy                 = CvtKernelPolicy<32, 4, kRowsPerThread>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);
    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int64_t>(outData);
        rgb_lab_nchw<RGB2Lab, BGR, SRGB, Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);
        rgb_lab_nchw<RGB2Lab, BGR, SRGB, Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T3>
inline void Dispatch_RGB_Lab(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                             ImageShape shape, bool strides_64b, NVCVColorConversionCode code, cudaStream_t stream)
{
#define CVCUDA_RGB_LAB_CASE(CODE, RGB2LAB, BGR, SRGB) \
    case CODE:                                        \
        return Launch_RGB_Lab<RGB2LAB, BGR, SRGB, T3>(inData, outData, shape, strides_64b, stream)

    switch (code)
    {
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_BGR2Lab, true, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_RGB2Lab, true, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2BGR, false, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2RGB, false, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LBGR2Lab, true, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LRGB2Lab, true, false, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LBGR, false, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LRGB, false, false, false);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Lab conversion code");
    }

#undef CVCUDA_RGB_LAB_CASE
}

template<typename T>
inline void Dispatch_RGB_Lab_Planar(const nvcv::TensorDataStridedCuda &inData,
                                    const nvcv::TensorDataStridedCuda &outData, ImageShape shape, bool strides_64b,
                                    NVCVColorConversionCode code, cudaStream_t stream)
{
#define CVCUDA_RGB_LAB_CASE(CODE, RGB2LAB, BGR, SRGB) \
    case CODE:                                        \
        return Launch_RGB_Lab_Planar<RGB2LAB, BGR, SRGB, T>(inData, outData, shape, strides_64b, stream)

    switch (code)
    {
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_BGR2Lab, true, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_RGB2Lab, true, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2BGR, false, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2RGB, false, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LBGR2Lab, true, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LRGB2Lab, true, false, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LBGR, false, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LRGB, false, false, false);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Lab conversion code");
    }

#undef CVCUDA_RGB_LAB_CASE
}

inline void RGB_Lab(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    NVCVColorConversionCode code, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    ElemType   inDataType  = ClassifyElemType(inData.dtype());
    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape inputShape  = GetImageShape(*inAccess);
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 3 || outputShape.C != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "RGB/Lab conversion requires 3-channel input and output");
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }
    if (outputShape != inputShape)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape different than output shape");
    }

    const bool isPlanar    = IsPlanar(GetImageLayout(inData.layout()));
    const bool strides_64b = NeedsWideStrides(*inAccess, *outAccess);

    switch (inDataType)
    {
    case ElemType::k8U:
        if (isPlanar)
            return Dispatch_RGB_Lab_Planar<uchar>(inData, outData, inputShape, strides_64b, code, stream);
        return Dispatch_RGB_Lab<uchar3>(inData, outData, inputShape, strides_64b, code, stream);
    case ElemType::k16F:
        if (isPlanar)
            return Dispatch_RGB_Lab_Planar<__half>(inData, outData, inputShape, strides_64b, code, stream);
        return Dispatch_RGB_Lab<half3>(inData, outData, inputShape, strides_64b, code, stream);
    case ElemType::k32F:
        if (isPlanar)
            return Dispatch_RGB_Lab_Planar<float>(inData, outData, inputShape, strides_64b, code, stream);
        return Dispatch_RGB_Lab<float3>(inData, outData, inputShape, strides_64b, code, stream);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
}

template<typename SrcT, typename DstT>
inline void Launch_BGR_to_HSV(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              ImageShape shape, int bidx, bool isFullRange, bool strides_64b, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        bgr_to_hsv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        bgr_to_hsv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_BGR_to_HSV_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                     bool isFullRange, bool strides_64b, int srcChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int64_t>(outData);
        bgr_to_hsv_nchw<Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange, srcChannels);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);
        bgr_to_hsv_nchw<Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange, srcChannels);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_HSV(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       NVCVColorConversionCode code, cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
    int  bidx        = (code == NVCV_COLOR_BGR2HSV || code == NVCV_COLOR_BGR2HSV_FULL) ? 0 : 2;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }
    if (outputShape != inputShape)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape different than output shape");
    }

    const bool strides_64b = NeedsWideStrides(*inAccess, *outAccess);

#define CVCUDA_BGR2HSV_CASE(T3) \
    return Launch_BGR_to_HSV<T3, T3>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream)

    if (isPlanar)
    {
#define CVCUDA_BGR2HSV_PLANAR_CASE(T)                                                                             \
    return Launch_BGR_to_HSV_Planar<T>(inData, outData, inputShape, bidx, isFullRange, strides_64b, inputShape.C, \
                                       stream)
        switch (inDataType)
        {
        case ElemType::k8U:
            CVCUDA_BGR2HSV_PLANAR_CASE(uchar);
        case ElemType::k16F: // Follows the float path (hue in [0, 360); FULL only affects 8-bit).
            CVCUDA_BGR2HSV_PLANAR_CASE(__half);
        case ElemType::k32F:
            CVCUDA_BGR2HSV_PLANAR_CASE(float);
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
        }
#undef CVCUDA_BGR2HSV_PLANAR_CASE
    }

    switch (inDataType)
    {
    case ElemType::k8U:
        CVCUDA_BGR2HSV_CASE(uchar3);
    case ElemType::k16F: // Follows the float path (hue in [0, 360); FULL only affects 8-bit).
        CVCUDA_BGR2HSV_CASE(half3);
    case ElemType::k32F:
        CVCUDA_BGR2HSV_CASE(float3);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_BGR2HSV_CASE
}

template<typename SrcT, typename DstT>
inline void Launch_HSV_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              ImageShape shape, int bidx, bool isFullRange, bool strides_64b, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        hsv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        hsv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void Launch_HSV_to_BGR_Planar(const nvcv::TensorDataStridedCuda &inData,
                                     const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx,
                                     bool isFullRange, bool strides_64b, int dstChannels, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    RequirePlanarBatchFitsGridZ(shape.N);

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int64_t>(outData);
        hsv_to_bgr_nchw<Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange, dstChannels);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNCHW<const T, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNCHW<T, int32_t>(outData);
        hsv_to_bgr_nchw<Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange, dstChannels);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void HSV_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                       NVCVColorConversionCode code, cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL);
    int  bidx        = (code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL) ? 0 : 2;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const bool isPlanar = IsPlanar(GetImageLayout(inData.layout()));

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (outputShape.C != 3 && outputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", outputShape.C);
    }
    if (inputShape.C != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (outDataType != inDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape");
    }

    const int  dcn         = outputShape.C;
    const bool strides_64b = NeedsWideStrides(*inAccess, *outAccess);

#define CVCUDA_HSV2BGR_CASE(T3, T4)                                                                            \
    if (dcn == 3)                                                                                              \
        return Launch_HSV_to_BGR<T3, T3>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream); \
    else                                                                                                       \
        return Launch_HSV_to_BGR<T3, T4>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream)

    if (isPlanar)
    {
#define CVCUDA_HSV2BGR_PLANAR_CASE(T) \
    return Launch_HSV_to_BGR_Planar<T>(inData, outData, inputShape, bidx, isFullRange, strides_64b, dcn, stream)
        switch (inDataType)
        {
        case ElemType::k8U:
            CVCUDA_HSV2BGR_PLANAR_CASE(uchar);
        case ElemType::k16F: // Follows the float path; an added alpha channel becomes 1.0.
            CVCUDA_HSV2BGR_PLANAR_CASE(__half);
        case ElemType::k32F:
            CVCUDA_HSV2BGR_PLANAR_CASE(float);
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
        }
#undef CVCUDA_HSV2BGR_PLANAR_CASE
    }

    switch (inDataType)
    {
    case ElemType::k8U:
        CVCUDA_HSV2BGR_CASE(uchar3, uchar4);
    case ElemType::k16F: // Follows the float path; an added alpha channel becomes 1.0.
        CVCUDA_HSV2BGR_CASE(half3, half4);
    case ElemType::k32F:
        CVCUDA_HSV2BGR_CASE(float3, float4);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_HSV2BGR_CASE
}

template<bool IsSemiPlanar, typename SrcT, typename DstT>
inline void Launch_YUV420xp_to_BGR(const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData, ImageShape shape, int bidx, int uidx,
                                   bool strides_64b, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(shape.W, Policy::TileWidth), util::DivUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        // YUV420 input: 4D tensor with scalar type.
        auto srcWrap = cuda::CreateTensorWrapNHWC<const SrcT, int64_t>(inData);
        // BGR output: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        yuv420_to_bgr_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, uidx);
    }
    else
    {
        // YUV420 input: 4D tensor with scalar type.
        auto srcWrap = cuda::CreateTensorWrapNHWC<const SrcT, int32_t>(inData);
        // BGR output: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        yuv420_to_bgr_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, uidx);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void YUV420xp_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData.layout())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support subsampled YUV420 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2BGR_NV21
           || code == NVCV_COLOR_YUV2BGRA_NV21 || code == NVCV_COLOR_YUV2BGR_YV12 || code == NVCV_COLOR_YUV2BGRA_YV12
           || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2RGB_NV12
           || code == NVCV_COLOR_YUV2RGBA_NV12 || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV
           || code == NVCV_COLOR_YUV2RGB_IYUV || code == NVCV_COLOR_YUV2RGBA_IYUV)
            ? 0
            : 1;

    // clang-format off
    bool p420 = (code == NVCV_COLOR_YUV2BGR_YV12 || code == NVCV_COLOR_YUV2BGRA_YV12 ||
                 code == NVCV_COLOR_YUV2RGB_YV12 || code == NVCV_COLOR_YUV2RGBA_YV12 ||
                 code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV ||
                 code == NVCV_COLOR_YUV2RGB_IYUV || code == NVCV_COLOR_YUV2RGBA_IYUV);
    // clang-format on

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if ((code != NVCV_COLOR_YUV2GRAY_420 || outputShape.C != 1) && outputShape.C != 3 && outputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", outputShape.C);
    }
    if (inputShape.C != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (inputShape.H % 3 != 0 || inputShape.W % 2 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape");
    }
    if (inDataType != ElemType::k8U || outDataType != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int rgb_width  = inputShape.W;
    int rgb_height = inputShape.H * 2 / 3;

    if (outputShape.H != rgb_height || outputShape.W != rgb_width || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape");
    }
    if (p420 && rgb_height % 4 != 0) // YUV 420 planar formats need 4 rows of Y for every full row of U or V.
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input shape: to convert from YUV 420 planar formats, the output "
                              "tensor height must be a multiple of 4; height = %d",
                              rgb_height);
    }

    const int  dcn         = outputShape.C;
    const bool strides_64b = NeedsWideStrides(*inAccess, *outAccess);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_420:
    {
        int dpitch = static_cast<int>(outAccess->rowStride());
        int spitch = static_cast<int>(inAccess->rowStride());

        for (int i = 0; i < inputShape.N; i++)
        {
            const void *srcPtr = inData.basePtr() + (size_t)i * inAccess->sampleStride();

            void *dstPtr = outData.basePtr() + (size_t)i * outAccess->sampleStride();

            NVCV_CHECK_THROW(cudaMemcpy2DAsync(dstPtr, dpitch, srcPtr, spitch, rgb_width, rgb_height,
                                               cudaMemcpyDeviceToDevice, stream));
        }
    }
    break;
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2BGRA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV21:
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2RGBA_NV21:
        if (dcn == 3)
            return Launch_YUV420xp_to_BGR<true, uchar, uchar3>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                               stream);
        else
            return Launch_YUV420xp_to_BGR<true, uchar, uchar4>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                               stream);
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
        if (dcn == 3)
            return Launch_YUV420xp_to_BGR<false, uchar, uchar3>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                                stream);
        else
            return Launch_YUV420xp_to_BGR<false, uchar, uchar4>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                                stream);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

template<bool IsSemiPlanar, typename SrcT, typename DstT>
inline void Launch_BGR_to_YUV420xp(const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData, ImageShape inputShape, int bidx,
                                   int uidx, bool strides_64b, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    int2 srcSize{inputShape.W, inputShape.H};

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(util::DivUp(inputShape.W, Policy::TileWidth), util::DivUp(inputShape.H, Policy::TileHeight),
                  inputShape.N);

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData); // RGB input: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHWC<DstT, int64_t>(outData);     // YUV420 output: 4D scalar tensor.

        bgr_to_yuv420_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, srcSize, bidx, uidx);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData); // RGB input: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHWC<DstT, int32_t>(outData);     // YUV420 output: 4D scalar tensor.

        bgr_to_yuv420_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, srcSize, bidx, uidx);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_YUV420xp(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData.layout())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support subsampled YUV420 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_NV21
           || code == NVCV_COLOR_BGRA2YUV_NV21 || code == NVCV_COLOR_BGR2YUV_YV12 || code == NVCV_COLOR_BGRA2YUV_YV12
           || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_RGB2YUV_NV12
           || code == NVCV_COLOR_RGBA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV
           || code == NVCV_COLOR_RGB2YUV_IYUV || code == NVCV_COLOR_RGBA2YUV_IYUV)
            ? 0
            : 1;

    // clang-format off
    bool p420 = (code == NVCV_COLOR_BGR2YUV_YV12 || code == NVCV_COLOR_BGRA2YUV_YV12 ||
                 code == NVCV_COLOR_RGB2YUV_YV12 || code == NVCV_COLOR_RGBA2YUV_YV12 ||
                 code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV ||
                 code == NVCV_COLOR_RGB2YUV_IYUV || code == NVCV_COLOR_RGBA2YUV_IYUV);
    // clang-format on

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.C != 3 && inputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", inputShape.C);
    }
    if (inputShape.H % 2 != 0 || inputShape.W % 2 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input shape");
    }
    if (p420 && inputShape.H % 4 != 0) // YUV 420 planar formats need 4 rows of Y for every full row of U or V.
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input shape: to convert to YUV 420 planar formats, the input "
                              "tensor height must be a multiple of 4; height = %d",
                              inputShape.H);
    }
    if (inDataType != ElemType::k8U || outDataType != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int yuv420_width  = inputShape.W;
    int yuv420_height = inputShape.H / 2 * 3;

    if (outputShape.H != yuv420_height || outputShape.W != yuv420_width || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape");
    }

    const bool strides_64b = NeedsWideStrides(*inAccess, *outAccess);

    switch (code)
    {
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV21:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV21:
        if (inputShape.C == 3)
            return Launch_BGR_to_YUV420xp<true, uchar3, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                               stream);
        else
            return Launch_BGR_to_YUV420xp<true, uchar4, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                               stream);
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
        if (inputShape.C == 3)
            return Launch_BGR_to_YUV420xp<false, uchar3, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                                stream);
        else
            return Launch_BGR_to_YUV420xp<false, uchar4, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                                stream);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

inline void YUV422_to_BGR(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                          NVCVColorConversionCode code, cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData.layout())))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support packed YUV422 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU
           || code == NVCV_COLOR_YUV2BGRA_YVYU || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY)
            ? 0
            : 2;

    int yidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU || code == NVCV_COLOR_YUV2BGRA_YVYU
           || code == NVCV_COLOR_YUV2RGB_YVYU || code == NVCV_COLOR_YUV2RGBA_YVYU || code == NVCV_COLOR_YUV2GRAY_YUY2)
            ? 0
            : 1;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY
           || code == NVCV_COLOR_YUV2RGB_UYVY || code == NVCV_COLOR_YUV2RGBA_UYVY)
            ? 0
            : 2;

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    ElemType   inDataType = ClassifyElemType(inData.dtype());
    ImageShape inputShape = GetImageShape(*inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    ElemType   outDataType = ClassifyElemType(outData.dtype());
    ImageShape outputShape = GetImageShape(*outAccess);

    if (inputShape.W % 4 != 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input shape -- width must be a multiple of 4");
    }

    if ((code != NVCV_COLOR_YUV2GRAY_UYVY && code != NVCV_COLOR_YUV2GRAY_YUY2 || outputShape.C != 1)
        && outputShape.C != 3 && outputShape.C != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid output channel number %d -- RGB output must have 3 or 4 channels and "
                              "grayscale output must have 1 channel",
                              outputShape.C);
    }
    if (inputShape.C != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid input channel number %d -- input must have 1 channel", inputShape.C);
    }
    if (inDataType != ElemType::k8U || outDataType != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input / output DataType");
    }
    if (outputShape.H != inputShape.H || 2 * outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output shape");
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(inputShape.W / 4, static_cast<int>(blockSize.x)),
                  util::DivUp(inputShape.H, static_cast<int>(blockSize.y)), inputShape.N);

    int2 dstSize{outputShape.W, outputShape.H};
    int  dcn = outputShape.C;

    auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t>(outData);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_YUY2:
    case NVCV_COLOR_YUV2GRAY_UYVY:
    {
        yuv422_to_gray_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, yidx);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    case NVCV_COLOR_YUV2BGR_YUY2:
    case NVCV_COLOR_YUV2BGR_YVYU:
    case NVCV_COLOR_YUV2BGRA_YUY2:
    case NVCV_COLOR_YUV2BGRA_YVYU:
    case NVCV_COLOR_YUV2RGB_YUY2:
    case NVCV_COLOR_YUV2RGB_YVYU:
    case NVCV_COLOR_YUV2RGBA_YUY2:
    case NVCV_COLOR_YUV2RGBA_YVYU:
    case NVCV_COLOR_YUV2RGB_UYVY:
    case NVCV_COLOR_YUV2BGR_UYVY:
    case NVCV_COLOR_YUV2RGBA_UYVY:
    case NVCV_COLOR_YUV2BGRA_UYVY:
    {
        yuv422_to_bgr_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, dcn, bidx, yidx, uidx);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

inline void Infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                  NVCVColorConversionCode code, cudaStream_t stream)
{
    ImageLayout input_format  = GetImageLayout(inData.layout());
    ImageLayout output_format = GetImageLayout(outData.layout());

    if (input_format != output_format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input and output tensors");
    }

    using func_t = void (*)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream);

    // The set of supported NVCVColorConversionCode values is defined by this table: a null entry is
    // a code the operator rejects. Indices and comments are kept verbatim so the supported set can
    // be diffed against the public NVCVColorConversionCode enum by eye.
    static const func_t funcs[] = {
        BGR_to_RGB, // CV_BGR2BGRA    =0
        BGR_to_RGB, // CV_BGRA2BGR    =1
        BGR_to_RGB, // CV_BGR2RGBA    =2
        BGR_to_RGB, // CV_RGBA2BGR    =3
        BGR_to_RGB, // CV_BGR2RGB     =4
        BGR_to_RGB, // CV_BGRA2RGBA   =5

        BGR_to_GRAY, // CV_BGR2GRAY    =6
        BGR_to_GRAY, // CV_RGB2GRAY    =7
        GRAY_to_BGR, // CV_GRAY2BGR    =8
        GRAY_to_BGR, // CV_GRAY2BGRA, CV_GRAY2RGBA   =9
        BGR_to_GRAY, // CV_BGRA2GRAY   =10
        BGR_to_GRAY, // CV_RGBA2GRAY   =11

        nullptr, //BGR_to_BGR565,          // CV_BGR2BGR565  =12
        nullptr, //RGB_to_BGR565,          // CV_RGB2BGR565  =13
        nullptr, //BGR565_to_BGR,          // CV_BGR5652BGR  =14
        nullptr, //BGR565_to_RGB,          // CV_BGR5652RGB  =15
        nullptr, //BGRA_to_BGR565,         // CV_BGRA2BGR565 =16
        nullptr, //RGBA_to_BGR565,         // CV_RGBA2BGR565 =17
        nullptr, //BGR565_to_BGRA,         // CV_BGR5652BGRA =18
        nullptr, //BGR565_to_RGBA,         // CV_BGR5652RGBA =19

        nullptr, //GRAY_to_BGR565,         // CV_GRAY2BGR565 =20
        nullptr, //BGR565_to_GRAY,         // CV_BGR5652GRAY =21

        nullptr, //BGR_to_BGR555,          // CV_BGR2BGR555  =22
        nullptr, //RGB_to_BGR555,          // CV_RGB2BGR555  =23
        nullptr, //BGR555_to_BGR,          // CV_BGR5552BGR  =24
        nullptr, //BGR555_to_RGB,          // CV_BGR5552RGB  =25
        nullptr, //BGRA_to_BGR555,         // CV_BGRA2BGR555 =26
        nullptr, //RGBA_to_BGR555,         // CV_RGBA2BGR555 =27
        nullptr, //BGR555_to_BGRA,         // CV_BGR5552BGRA =28
        nullptr, //BGR555_to_RGBA,         // CV_BGR5552RGBA =29

        nullptr, //GRAY_to_BGR555,         // CV_GRAY2BGR555 =30
        nullptr, //BGR555_to_GRAY,         // CV_BGR5552GRAY =31

        nullptr, //BGR_to_XYZ,             // CV_BGR2XYZ     =32
        nullptr, //RGB_to_XYZ,             // CV_RGB2XYZ     =33
        nullptr, //XYZ_to_BGR,             // CV_XYZ2BGR     =34
        nullptr, //XYZ_to_RGB,             // CV_XYZ2RGB     =35

        nullptr, //BGR_to_YCrCb,           // CV_BGR2YCrCb   =36
        nullptr, //RGB_to_YCrCb,           // CV_RGB2YCrCb   =37
        nullptr, //YCrCb_to_BGR,           // CV_YCrCb2BGR   =38
        nullptr, //YCrCb_to_RGB,           // CV_YCrCb2RGB   =39

        BGR_to_HSV, //BGR_to_HSV,             // CV_BGR2HSV     =40
        BGR_to_HSV, //RGB_to_HSV,             // CV_RGB2HSV     =41

        nullptr, //                =42
        nullptr, //                =43

        RGB_Lab, //BGR_to_Lab,             // CV_BGR2Lab     =44
        RGB_Lab, //RGB_to_Lab,             // CV_RGB2Lab     =45

        nullptr, //bayerBG_to_BGR,         // CV_BayerBG2BGR =46
        nullptr, //bayeRGB_to_BGR,         // CV_BayeRGB2BGR =47
        nullptr, //bayerRG_to_BGR,         // CV_BayerRG2BGR =48
        nullptr, //bayerGR_to_BGR,         // CV_BayerGR2BGR =49

        nullptr, //BGR_to_Luv,             // CV_BGR2Luv     =50
        nullptr, //RGB_to_Luv,             // CV_RGB2Luv     =51

        nullptr, //BGR_to_HLS,             // CV_BGR2HLS     =52
        nullptr, //RGB_to_HLS,             // CV_RGB2HLS     =53

        HSV_to_BGR, // CV_HSV2BGR     =54
        HSV_to_BGR, // CV_HSV2RGB     =55

        RGB_Lab, //Lab_to_BGR,             // CV_Lab2BGR     =56
        RGB_Lab, //Lab_to_RGB,             // CV_Lab2RGB     =57
        nullptr, //Luv_to_BGR,             // CV_Luv2BGR     =58
        nullptr, //Luv_to_RGB,             // CV_Luv2RGB     =59

        nullptr, //HLS_to_BGR,             // CV_HLS2BGR     =60
        nullptr, //HLS_to_RGB,             // CV_HLS2RGB     =61

        nullptr, // CV_BayerBG2BGR_VNG =62
        nullptr, // CV_BayeRGB2BGR_VNG =63
        nullptr, // CV_BayerRG2BGR_VNG =64
        nullptr, // CV_BayerGR2BGR_VNG =65

        BGR_to_HSV, //BGR_to_HSV_FULL,        // CV_BGR2HSV_FULL = 66
        BGR_to_HSV, //RGB_to_HSV_FULL,        // CV_RGB2HSV_FULL = 67
        nullptr,    //BGR_to_HLS_FULL,        // CV_BGR2HLS_FULL = 68
        nullptr,    //RGB_to_HLS_FULL,        // CV_RGB2HLS_FULL = 69

        HSV_to_BGR, // CV_HSV2BGR_FULL = 70
        HSV_to_BGR, // CV_HSV2RGB_FULL = 71
        nullptr,    //HLS_to_BGR_FULL,        // CV_HLS2BGR_FULL = 72
        nullptr,    //HLS_to_RGB_FULL,        // CV_HLS2RGB_FULL = 73

        RGB_Lab, //LBGR_to_Lab,            // CV_LBGR2Lab     = 74
        RGB_Lab, //LRGB_to_Lab,            // CV_LRGB2Lab     = 75
        nullptr, //LBGR_to_Luv,            // CV_LBGR2Luv     = 76
        nullptr, //LRGB_to_Luv,            // CV_LRGB2Luv     = 77

        RGB_Lab, //Lab_to_LBGR,            // CV_Lab2LBGR     = 78
        RGB_Lab, //Lab_to_LRGB,            // CV_Lab2LRGB     = 79
        nullptr, //Luv_to_LBGR,            // CV_Luv2LBGR     = 80
        nullptr, //Luv_to_LRGB,            // CV_Luv2LRGB     = 81

        BGR_to_YUV, // CV_BGR2YUV      = 82
        BGR_to_YUV, // CV_RGB2YUV      = 83
        YUV_to_BGR, // CV_YUV2BGR      = 84
        YUV_to_BGR, // CV_YUV2RGB      = 85

        nullptr, //bayerBG_to_gray,        // CV_BayerBG2GRAY = 86
        nullptr, //bayeRGB_to_GRAY,        // CV_BayeRGB2GRAY = 87
        nullptr, //bayerRG_to_gray,        // CV_BayerRG2GRAY = 88
        nullptr, //bayerGR_to_gray,        // CV_BayerGR2GRAY = 89

        //! YUV 4:2:0 family to RGB
        YUV420xp_to_BGR, // CV_YUV2RGB_NV12 = 90,
        YUV420xp_to_BGR, // CV_YUV2BGR_NV12 = 91,
        YUV420xp_to_BGR, // CV_YUV2RGB_NV21 = 92, CV_YUV420sp2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_NV21 = 93, CV_YUV420sp2BGR

        YUV420xp_to_BGR, // CV_YUV2RGBA_NV12 = 94,
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV12 = 95,
        YUV420xp_to_BGR, // CV_YUV2RGBA_NV21 = 96, CV_YUV420sp2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV21 = 97, CV_YUV420sp2BGRA

        YUV420xp_to_BGR, // CV_YUV2RGB_YV12 = 98, CV_YUV420p2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_YV12 = 99, CV_YUV420p2BGR
        YUV420xp_to_BGR, // CV_YUV2RGB_IYUV = 100, CV_YUV2RGB_I420
        YUV420xp_to_BGR, // CV_YUV2BGR_IYUV = 101, CV_YUV2BGR_I420

        YUV420xp_to_BGR, // CV_YUV2RGBA_YV12 = 102, CV_YUV420p2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_YV12 = 103, CV_YUV420p2BGRA
        YUV420xp_to_BGR, // CV_YUV2RGBA_IYUV = 104, CV_YUV2RGBA_I420
        YUV420xp_to_BGR, // CV_YUV2BGRA_IYUV = 105, CV_YUV2BGRA_I420

        YUV420xp_to_BGR, // CV_YUV2GRAY_420 = 106,
        // CV_YUV2GRAY_NV21,
        // CV_YUV2GRAY_NV12,
        // CV_YUV2GRAY_YV12,
        // CV_YUV2GRAY_IYUV,
        // CV_YUV2GRAY_I420,
        // CV_YUV420sp2GRAY,
        // CV_YUV420p2GRAY ,

        //! YUV 4:2:2 family to RGB
        YUV422_to_BGR, // CV_YUV2RGB_UYVY = 107, CV_YUV2RGB_Y422, CV_YUV2RGB_UYNV
        YUV422_to_BGR, // CV_YUV2BGR_UYVY = 108, CV_YUV2BGR_Y422, CV_YUV2BGR_UYNV
        nullptr,       // CV_YUV2RGB_VYUY = 109,
        nullptr,       // CV_YUV2BGR_VYUY = 110,

        YUV422_to_BGR, // CV_YUV2RGBA_UYVY = 111, CV_YUV2RGBA_Y422, CV_YUV2RGBA_UYNV
        YUV422_to_BGR, // CV_YUV2BGRA_UYVY = 112, CV_YUV2BGRA_Y422, CV_YUV2BGRA_UYNV
        nullptr,       // CV_YUV2RGBA_VYUY = 113,
        nullptr,       // CV_YUV2BGRA_VYUY = 114,

        YUV422_to_BGR, // CV_YUV2RGB_YUY2 = 115, CV_YUV2RGB_YUYV, CV_YUV2RGB_YUNV
        YUV422_to_BGR, // CV_YUV2BGR_YUY2 = 116, CV_YUV2BGR_YUYV, CV_YUV2BGR_YUNV
        YUV422_to_BGR, // CV_YUV2RGB_YVYU = 117,
        YUV422_to_BGR, // CV_YUV2BGR_YVYU = 118,

        YUV422_to_BGR, // CV_YUV2RGBA_YUY2 = 119, CV_YUV2RGBA_YUYV, CV_YUV2RGBA_YUNV
        YUV422_to_BGR, // CV_YUV2BGRA_YUY2 = 120, CV_YUV2BGRA_YUYV, CV_YUV2BGRA_YUNV
        YUV422_to_BGR, // CV_YUV2RGBA_YVYU = 121,
        YUV422_to_BGR, // CV_YUV2BGRA_YVYU = 122,

        YUV422_to_BGR, // CV_YUV2GRAY_UYVY = 123, CV_YUV2GRAY_Y422, CV_YUV2GRAY_UYNV
        YUV422_to_BGR, // CV_YUV2GRAY_YUY2 = 124, CV_YUV2GRAY_YVYU, CV_YUV2GRAY_YUYV, CV_YUV2GRAY_YUNV

        //! alpha premultiplication
        nullptr, //RGBA_to_mBGRA,         // CV_RGBA2mRGBA = 125,
        nullptr, // CV_mRGBA2RGBA = 126,

        //! RGB to YUV 4:2:0 family (three plane YUV)
        BGR_to_YUV420xp, // CV_RGB2YUV_I420  = 127, CV_RGB2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGR2YUV_I420  = 128, CV_BGR2YUV_IYUV

        BGR_to_YUV420xp, // CV_RGBA2YUV_I420 = 129, CV_RGBA2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGRA2YUV_I420 = 130, CV_BGRA2YUV_IYUV
        BGR_to_YUV420xp, // CV_RGB2YUV_YV12  = 131,
        BGR_to_YUV420xp, // CV_BGR2YUV_YV12  = 132,
        BGR_to_YUV420xp, // CV_RGBA2YUV_YV12 = 133,
        BGR_to_YUV420xp, // CV_BGRA2YUV_YV12 = 134,

        //! Edge-Aware Demosaicing
        nullptr, // CV_BayerBG2BGR_EA  = 135,
        nullptr, // CV_BayerGB2BGR_EA  = 136,
        nullptr, // CV_BayerRG2BGR_EA  = 137,
        nullptr, // CV_BayerGR2BGR_EA  = 138,

        nullptr, // OpenCV COLORCVT_MAX = 139

        //! RGB to YUV 4:2:0 family (two plane YUV, not in OpenCV)
        BGR_to_YUV420xp, // CV_RGB2YUV_NV12 = 140,
        BGR_to_YUV420xp, // CV_BGR2YUV_NV12 = 141,
        BGR_to_YUV420xp, // CV_RGB2YUV_NV21 = 142, CV_RGB2YUV420sp
        BGR_to_YUV420xp, // CV_BGR2YUV_NV21 = 143, CV_BGR2YUV420sp

        BGR_to_YUV420xp, // CV_RGBA2YUV_NV12 = 144,
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV12 = 145,
        BGR_to_YUV420xp, // CV_RGBA2YUV_NV21 = 146, CV_RGBA2YUV420sp
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV21 = 147, CV_BGRA2YUV420sp

        nullptr, // CV_COLORCVT_MAX  = 148
    };

    if (code < 0 || static_cast<size_t>(code) >= sizeof(funcs) / sizeof(funcs[0]))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid convert color code: %d", (int)code);
    }

    func_t func = funcs[code];

    if (func == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid convert color code: %d", (int)code);
    }

    func(inData, outData, code, stream);
}

} // namespace tensor

namespace varshape {

// Only used in assert() — omit from release builds to avoid NVCC warning #177-D
// ("declared but never referenced" on device functions absent from non-debug paths).
#ifndef NDEBUG
inline __device__ bool checkShapeFromYUV420(int rows, int cols, NVCVColorConversionCode code)
{
    int valid_row = 1, valid_col = 1;
    switch (code)
    {
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2BGRA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV21:
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2RGBA_NV21:
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
    case NVCV_COLOR_YUV2GRAY_420:
        valid_row = 3;
        valid_col = 2;
        break;
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV21:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
        valid_row = 2;
        valid_col = 2;
        break;
    default:
        return false;
    }
    if (rows % valid_row != 0 || cols % valid_col != 0)
    {
        return false;
    }
    return true;
}
#endif // NDEBUG

template<class SrcWrapper, typename T>
__device__ __forceinline__ void load_bgra_chw(SrcWrapper src, T &B, T &G, T &R, T &A, int batch_idx, int x, int y,
                                              int bidx, int srcChannels)
{
    B = *src.ptr(batch_idx, bidx, y, x);
    G = *src.ptr(batch_idx, 1, y, x);
    R = *src.ptr(batch_idx, bidx ^ 2, y, x);
    A = srcChannels == 4 ? *src.ptr(batch_idx, 3, y, x) : Alpha<T>();
}

template<class DstWrapper, typename T>
__device__ __forceinline__ void store_bgra_chw(DstWrapper dst, T B, T G, T R, T A, int batch_idx, int x, int y,
                                               int bidx, int dstChannels)
{
    *dst.ptr(batch_idx, bidx, y, x)     = B;
    *dst.ptr(batch_idx, 1, y, x)        = G;
    *dst.ptr(batch_idx, bidx ^ 2, y, x) = R;
    if (dstChannels == 4)
    {
        *dst.ptr(batch_idx, 3, y, x) = A;
    }
}

template<class T>
__global__ void rgb_to_bgr_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    T g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = r;

    if (dst.numChannels() == 4)
    {
        T al = src.numChannels() == 4 ? *src.ptr(batch_idx, dst_y, dst_x, 3) : Alpha<T>();
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = al;
    }
}

template<class T>
__global__ void rgb_to_bgr_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                               int srcChannels, int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T b, g, r, a;
    load_bgra_chw(src, b, g, r, a, batch_idx, dst_x, dst_y, bidx, srcChannels);
    store_bgra_chw(dst, b, g, r, a, batch_idx, dst_x, dst_y, 0, dstChannels);
}

template<class T>
__global__ void gray_to_bgr_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T g = *src.ptr(batch_idx, dst_y, dst_x, 0);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = g;
    if (dst.numChannels() == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = g;
    }
}

template<class T>
__global__ void gray_to_bgr_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T g = *src.ptr(batch_idx, 0, dst_y, dst_x);

    *dst.ptr(batch_idx, 0, dst_y, dst_x) = g;
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = g;
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = g;
    if (dstChannels == 4)
    {
        *dst.ptr(batch_idx, 3, dst_y, dst_x) = g;
    }
}

template<int NIX, class T>
__global__ void bgr_to_gray_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                      int bidx)
{
    int       dst_x0    = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    const int dst_width = dst.width(batch_idx);
    if (dst_x0 >= dst_width || dst_y >= dst.height(batch_idx))
        return;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        int dst_x = dst_x0 + i;
        if (dst_x >= dst_width)
            break;

        int b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
        int g = *src.ptr(batch_idx, dst_y, dst_x, 1);
        int r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

        T gray                               = (T)CV_DESCALE(b * BY15 + g * GY15 + r * RY15, gray_shift);
        *dst.ptr(batch_idx, dst_y, dst_x, 0) = gray;
    }
}

template<class T>
__global__ void bgr_to_gray_char_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                     int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T b, g, r;
    load_bgra_chw(src, b, g, r, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    T gray                               = (T)CV_DESCALE(b * BY15 + g * GY15 + r * RY15, gray_shift);
    *dst.ptr(batch_idx, 0, dst_y, dst_x) = gray;
}

template<class T>
__global__ void bgr_to_gray_float_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                       int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    T g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    T gray                               = (T)(b * B2YF + g * G2YF + r * R2YF);
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = gray;
}

template<class T>
__global__ void bgr_to_gray_float_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst,
                                      int bidx, int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T b, g, r;
    load_bgra_chw(src, b, g, r, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    T gray                               = (T)(b * B2YF + g * G2YF + r * R2YF);
    *dst.ptr(batch_idx, 0, dst_y, dst_x) = gray;
}

template<class T>
__global__ void bgr_to_yuv_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                     int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    int B = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int G = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int R = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    int C0 = R2Y, C1 = G2Y, C2 = B2Y, C3 = R2VI, C4 = B2UI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) * (1 << yuv_shift);
    int Y     = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    int Cr    = CV_DESCALE((R - Y) * C3 + delta, yuv_shift);
    int Cb    = CV_DESCALE((B - Y) * C4 + delta, yuv_shift);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = cuda::SaturateCast<T>(Cb);
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = cuda::SaturateCast<T>(Cr);
}

template<class T>
__global__ void bgr_to_yuv_char_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                    int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T B, G, R;
    load_bgra_chw(src, B, G, R, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    int C0 = R2Y, C1 = G2Y, C2 = B2Y, C3 = R2VI, C4 = B2UI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) * (1 << yuv_shift);
    int Y     = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    int Cr    = CV_DESCALE((R - Y) * C3 + delta, yuv_shift);
    int Cb    = CV_DESCALE((B - Y) * C4 + delta, yuv_shift);

    *dst.ptr(batch_idx, 0, dst_y, dst_x) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = cuda::SaturateCast<T>(Cb);
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = cuda::SaturateCast<T>(Cr);
}

template<class T>
__global__ void bgr_to_yuv_float_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                      int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    // Coefficients and accumulation stay in float regardless of T: identical to the previous
    // all-T code for T=float, and the compute-in-float-round-once path for T=__half (whose
    // native arithmetic would also round the coefficients). SaturateCast is a float identity.
    float B = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    float G = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float R = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    float C0 = R2YF, C1 = G2YF, C2 = B2YF, C3 = R2VF, C4 = B2UF;
    float delta                          = 0.5f;
    float Y                              = R * C0 + G * C1 + B * C2;
    float Cr                             = (R - Y) * C3 + delta;
    float Cb                             = (B - Y) * C4 + delta;
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = cuda::SaturateCast<T>(Cb);
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = cuda::SaturateCast<T>(Cr);
}

template<class T>
__global__ void bgr_to_yuv_float_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                     int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T b, g, r;
    load_bgra_chw(src, b, g, r, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    // Float working type: see bgr_to_yuv_float_nhwc.
    float B = b, G = g, R = r;

    float C0 = R2YF, C1 = G2YF, C2 = B2YF, C3 = R2VF, C4 = B2UF;
    float delta = 0.5f;
    float Y     = R * C0 + G * C1 + B * C2;
    float Cr    = (R - Y) * C3 + delta;
    float Cb    = (B - Y) * C4 + delta;

    *dst.ptr(batch_idx, 0, dst_y, dst_x) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = cuda::SaturateCast<T>(Cb);
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = cuda::SaturateCast<T>(Cr);
}

template<class T>
__global__ void yuv_to_bgr_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                     int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T Y  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    T Cb = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T Cr = *src.ptr(batch_idx, dst_y, dst_x, 2);

    int C0 = V2RI, C1 = V2GI, C2 = U2GI, C3 = U2BI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));
    int b     = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    int g     = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    int r     = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<T>(b);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<T>(g);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<T>(r);
}

template<class T>
__global__ void yuv_to_bgr_char_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                    int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T Y  = *src.ptr(batch_idx, 0, dst_y, dst_x);
    T Cb = *src.ptr(batch_idx, 1, dst_y, dst_x);
    T Cr = *src.ptr(batch_idx, 2, dst_y, dst_x);

    int C0 = V2RI, C1 = V2GI, C2 = U2GI, C3 = U2BI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));
    int b     = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    int g     = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    int r     = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    store_bgra_chw(dst, cuda::SaturateCast<T>(b), cuda::SaturateCast<T>(g), cuda::SaturateCast<T>(r),
                   cuda::TypeTraits<T>::max, batch_idx, dst_x, dst_y, bidx, dstChannels);
}

template<class T>
__global__ void yuv_to_bgr_float_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                      int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    // Float working type: see bgr_to_yuv_float_nhwc.
    float Y  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    float Cb = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float Cr = *src.ptr(batch_idx, dst_y, dst_x, 2);

    float C0 = V2RF, C1 = V2GF, C2 = U2GF, C3 = U2BF;
    float delta = 0.5f;
    float b     = Y + (Cb - delta) * C3;
    float g     = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
    float r     = Y + (Cr - delta) * C0;

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<T>(b);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<T>(g);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<T>(r);
}

template<class T>
__global__ void yuv_to_bgr_float_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                     int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    // Float working type: see bgr_to_yuv_float_nhwc.
    float Y  = *src.ptr(batch_idx, 0, dst_y, dst_x);
    float Cb = *src.ptr(batch_idx, 1, dst_y, dst_x);
    float Cr = *src.ptr(batch_idx, 2, dst_y, dst_x);

    float C0 = V2RF, C1 = V2GF, C2 = U2GF, C3 = U2BF;
    float delta = 0.5f;
    float b     = Y + (Cb - delta) * C3;
    float g     = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
    float r     = Y + (Cr - delta) * C0;

    store_bgra_chw(dst, cuda::SaturateCast<T>(b), cuda::SaturateCast<T>(g), cuda::SaturateCast<T>(r), Alpha<T>(),
                   batch_idx, dst_x, dst_y, bidx, dstChannels);
}

template<class T>
__global__ void bgr_to_hsv_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                     int bidx, bool isFullRange)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    int       b         = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int       g         = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int       r         = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);
    int       hrange    = isFullRange ? 256 : 180;
    int       hr        = hrange;
    const int hsv_shift = 12;
    int       h, s, v = b;
    int       vmin = b;
    int       vr, vg;

    v    = cuda::max(v, g);
    v    = cuda::max(v, r);
    vmin = min(vmin, g);
    vmin = min(vmin, r);

    uint8_t diff = cuda::SaturateCast<uint8_t>(v - vmin);
    vr           = v == r ? -1 : 0;
    vg           = v == g ? -1 : 0;

    int hdiv_table = diff == 0 ? 0 : cuda::SaturateCast<int>((hrange << hsv_shift) / (6. * diff));
    int sdiv_table = v == 0 ? 0 : cuda::SaturateCast<int>((255 << hsv_shift) / (1. * v));
    s              = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h              = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h              = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += h < 0 ? hr : 0;

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<uint8_t>(h);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = (uint8_t)s;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = (uint8_t)v;
}

template<class T>
__global__ void bgr_to_hsv_char_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                    bool isFullRange, int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T b8, g8, r8;
    load_bgra_chw(src, b8, g8, r8, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    int       b         = b8;
    int       g         = g8;
    int       r         = r8;
    int       hrange    = isFullRange ? 256 : 180;
    int       hr        = hrange;
    const int hsv_shift = 12;
    int       h, s, v = b;
    int       vmin = b;
    int       vr, vg;

    v    = cuda::max(v, g);
    v    = cuda::max(v, r);
    vmin = min(vmin, g);
    vmin = min(vmin, r);

    uint8_t diff = cuda::SaturateCast<uint8_t>(v - vmin);
    vr           = v == r ? -1 : 0;
    vg           = v == g ? -1 : 0;

    int hdiv_table = diff == 0 ? 0 : cuda::SaturateCast<int>((hrange << hsv_shift) / (6. * diff));
    int sdiv_table = v == 0 ? 0 : cuda::SaturateCast<int>((255 << hsv_shift) / (1. * v));
    s              = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h              = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h              = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += h < 0 ? hr : 0;

    *dst.ptr(batch_idx, 0, dst_y, dst_x) = cuda::SaturateCast<uint8_t>(h);
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = (uint8_t)s;
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = (uint8_t)v;
}

template<class T>
__global__ void bgr_to_hsv_float_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                      int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    float b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    float g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);
    float h, s, v;
    float hrange = 360.0;
    float hscale = hrange * (1.f / 360.f);

    float vmin, diff;

    v = vmin = r;
    if (v < g)
        v = g;
    if (v < b)
        v = b;
    if (vmin > g)
        vmin = g;
    if (vmin > b)
        vmin = b;

    diff = v - vmin;
    s    = diff / (float)(fabs(v) + FLT_EPSILON);
    diff = (float)(60. / (diff + FLT_EPSILON));
    if (v == r)
        h = (g - b) * diff;
    else if (v == g)
        h = (b - r) * diff + 120.f;
    else
        h = (r - g) * diff + 240.f;

    if (h < 0)
        h += 360.f;

    // SaturateCast is a float identity; it rounds the float result once for T=__half.
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<T>(h * hscale);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = cuda::SaturateCast<T>(s);
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = cuda::SaturateCast<T>(v);
}

template<class T>
__global__ void bgr_to_hsv_float_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                     int srcChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    T unusedA;
    T bT, gT, rT;
    load_bgra_chw(src, bT, gT, rT, unusedA, batch_idx, dst_x, dst_y, bidx, srcChannels);

    // Widen to float before any arithmetic so that T=__half matches the interleaved kernel's
    // compute-in-float behavior, e.g. in the (g - b) hue differences (a no-op for T=float).
    float b = bT, g = gT, r = rT;

    float h, s, v;
    float hrange = 360.0;
    float hscale = hrange * (1.f / 360.f);

    float vmin, diff;

    v = vmin = r;
    if (v < g)
        v = g;
    if (v < b)
        v = b;
    if (vmin > g)
        vmin = g;
    if (vmin > b)
        vmin = b;

    diff = v - vmin;
    s    = diff / (float)(fabs(v) + FLT_EPSILON);
    diff = (float)(60. / (diff + FLT_EPSILON));
    if (v == r)
        h = (g - b) * diff;
    else if (v == g)
        h = (b - r) * diff + 120.f;
    else
        h = (r - g) * diff + 240.f;

    if (h < 0)
        h += 360.f;

    // SaturateCast is a float identity; it rounds the float result once for T=__half.
    *dst.ptr(batch_idx, 0, dst_y, dst_x) = cuda::SaturateCast<T>(h * hscale);
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = cuda::SaturateCast<T>(s);
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = cuda::SaturateCast<T>(v);
}

inline __device__ void HSV2RGB_native_var_shape(float h, float s, float v, float &b, float &g, float &r)
{
    if (s == 0)
        b = g = r = v;
    else
    {
        h += 6 * (h < 0);              // Add 6 if h < 0.
        int idx = static_cast<int>(h); // Sector index.
        h -= idx;                      // Fractional part of h.
        idx %= 6;                      // Make sure index is in valid range.

        const float p = v * (1 - s);
        const float q = v * (1 - s * h);
        const float t = v * (1 - s * (1 - h));
        switch (idx)
        {
        case 0:
            b = p;
            g = t;
            r = v;
            break;
        case 1:
            b = p;
            g = v;
            r = q;
            break;
        case 2:
            b = t;
            g = v;
            r = p;
            break;
        case 3:
            b = v;
            g = q;
            r = p;
            break;
        case 4:
            b = v;
            g = p;
            r = t;
            break;
        default:
            b = q;
            g = p;
            r = v;
            break;
        }
    }
}

template<int NIX, class T>
__global__ void hsv_to_bgr_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                     int bidx, bool isFullRange)
{
    int       dst_x0    = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    const int dst_width = dst.width(batch_idx);
    if (dst_x0 >= dst_width || dst_y >= dst.height(batch_idx))
        return;

    const float     scaleH  = 6.f / (isFullRange ? 256 : 180);
    constexpr float scaleSV = 1.0f / 255.0f;
    constexpr T     alpha   = cuda::TypeTraits<T>::max;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        int dst_x = dst_x0 + i;
        if (dst_x >= dst_width)
            break;

        float h = *src.ptr(batch_idx, dst_y, dst_x, 0) * scaleH;
        float s = *src.ptr(batch_idx, dst_y, dst_x, 1) * scaleSV;
        float v = *src.ptr(batch_idx, dst_y, dst_x, 2) * scaleSV;

        float b, g, r;
        HSV2RGB_native_var_shape(h, s, v, b, g, r);

        *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<uchar>(b * 255.0f);
        *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<uchar>(g * 255.0f);
        *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<uchar>(r * 255.0f);
        if (dst.numChannels() == 4)
            *dst.ptr(batch_idx, dst_y, dst_x, 3) = alpha;
    }
}

template<class T>
__global__ void hsv_to_bgr_char_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                    bool isFullRange, int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    const float     scaleH  = 6.f / (isFullRange ? 256 : 180);
    constexpr float scaleSV = 1.0f / 255.0f;
    constexpr T     alpha   = cuda::TypeTraits<T>::max;

    float h = *src.ptr(batch_idx, 0, dst_y, dst_x) * scaleH;
    float s = *src.ptr(batch_idx, 1, dst_y, dst_x) * scaleSV;
    float v = *src.ptr(batch_idx, 2, dst_y, dst_x) * scaleSV;

    float b, g, r;
    HSV2RGB_native_var_shape(h, s, v, b, g, r);

    store_bgra_chw(dst, cuda::SaturateCast<uchar>(b * 255.0f), cuda::SaturateCast<uchar>(g * 255.0f),
                   cuda::SaturateCast<uchar>(r * 255.0f), alpha, batch_idx, dst_x, dst_y, bidx, dstChannels);
}

template<class T>
__global__ void hsv_to_bgr_float_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                      int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    constexpr float scaleH = 6.0f / 360.0f;
    constexpr float alpha  = 1.0f;

    float h = *src.ptr(batch_idx, dst_y, dst_x, 0) * scaleH;
    float s = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float v = *src.ptr(batch_idx, dst_y, dst_x, 2);

    float b, g, r;
    HSV2RGB_native_var_shape(h, s, v, b, g, r);

    // SaturateCast is a float identity; it rounds the float result once for T=__half.
    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<T>(b);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<T>(g);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<T>(r);
    if (dst.numChannels() == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = cuda::SaturateCast<T>(alpha);
}

template<class T>
__global__ void hsv_to_bgr_float_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst, int bidx,
                                     int dstChannels)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    constexpr float scaleH = 6.0f / 360.0f;
    constexpr float alpha  = 1.0f;

    float h = *src.ptr(batch_idx, 0, dst_y, dst_x) * scaleH;
    float s = *src.ptr(batch_idx, 1, dst_y, dst_x);
    float v = *src.ptr(batch_idx, 2, dst_y, dst_x);

    float b, g, r;
    HSV2RGB_native_var_shape(h, s, v, b, g, r);

    store_bgra_chw(dst, static_cast<T>(b), static_cast<T>(g), static_cast<T>(r), static_cast<T>(alpha), batch_idx,
                   dst_x, dst_y, bidx, dstChannels);
}

__device__ __forceinline__ void yuv42xxp_to_bgr_kernel(const int &Y, const int &U, const int &V, uchar &r, uchar &g,
                                                       uchar &b)
{
    //R = 1.164(Y - 16) + 1.596(V - 128)
    //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    //B = 1.164(Y - 16)                  + 2.018(U - 128)

    //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
    //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
    //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
    const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG,
              C4           = ITUR_BT_601_CUB;
    const int yuv4xx_shift = ITUR_BT_601_SHIFT;

    int yy = cuda::max(0, Y - 16) * C0;
    int uu = U - 128;
    int vv = V - 128;

    r = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C1 * vv), yuv4xx_shift));
    g = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C2 * vv + C3 * uu), yuv4xx_shift));
    b = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C4 * uu), yuv4xx_shift));
}

__device__ __forceinline__ void bgr_to_yuv42xxp_kernel(const uchar &r, const uchar &g, const uchar &b, uchar &Y,
                                                       uchar &U, uchar &V)
{
    const int shifted16 = (16 << ITUR_BT_601_SHIFT);
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    int       yy        = ITUR_BT_601_CRY * r + ITUR_BT_601_CGY * g + ITUR_BT_601_CBY * b + halfShift + shifted16;

    Y = cuda::SaturateCast<uchar>(yy >> ITUR_BT_601_SHIFT);

    const int shifted128 = (128 << ITUR_BT_601_SHIFT);
    int       uu         = ITUR_BT_601_CRU * r + ITUR_BT_601_CGU * g + ITUR_BT_601_CBU * b + halfShift + shifted128;
    int       vv         = ITUR_BT_601_CBU * r + ITUR_BT_601_CGV * g + ITUR_BT_601_CBV * b + halfShift + shifted128;

    U = cuda::SaturateCast<uchar>(uu >> ITUR_BT_601_SHIFT);
    V = cuda::SaturateCast<uchar>(vv >> ITUR_BT_601_SHIFT);
}

template<class T>
__global__ void bgr_to_yuv420sp_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                          cuda::ImageBatchVarShapeWrapNHWC<T> dst, int bidx, int uidx,
                                          NVCVColorConversionCode code)
{
    int       src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    int       src_cols  = src.width(batch_idx);
    int       src_rows  = src.height(batch_idx);
    if (src_x >= src_cols || src_y >= src_rows)
        return;

    assert(checkShapeFromYUV420(dst.height(batch_idx), dst.width(batch_idx), code));

    uchar b = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx));
    uchar g = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, 1));
    uchar r = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx ^ 2));
    // Ignore alpha channel if input is RGBA.

    uchar Y{0}, U{0}, V{0};
    bgr_to_yuv42xxp_kernel(r, g, b, Y, U, V);

    // U and V are subsampled at half the full resolution (both in x and y), combined (i.e., interleaved), and arranged
    // as full rows after the full resolution Y data. Example memory layout for 4 x 4 image (NV12):
    //   Y_00 Y_01 Y_02 Y_03
    //   Y_10 Y_11 Y_12 Y_13
    //   Y_20 Y_21 Y_22 Y_23
    //   Y_30 Y_31 Y_32 Y_33
    //   U_00 V_00 U_02 V_02
    //   U_20 V_20 U_22 V_22
    // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
    // and Y_11. Each full U-V row represents 2 rows of Y values. Some layouts (e.g., NV21) swap the location
    // of the U and V values in each U-V pair.

    *dst.ptr(batch_idx, src_y, src_x) = Y;
    if (src_y % 2 == 0 && src_x % 2 == 0)
    {
        const int uv_y = src_rows + src_y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
        const int uv_x = (src_x & ~1);         // Convert x to even # (set lowest bit to 0).

        *dst.ptr(batch_idx, uv_y, uv_x + uidx)       = U; // Some formats swap the U and V elements (as indicated
        *dst.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)) = V; //   by the uidx parameter).
    }
}

template<class T>
__global__ void bgr_to_yuv420p_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                         cuda::ImageBatchVarShapeWrapNHWC<T> dst, int bidx, int uidx,
                                         NVCVColorConversionCode code)
{
    int       src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    int       src_cols  = src.width(batch_idx);
    int       src_rows  = src.height(batch_idx);
    if (src_x >= src_cols || src_y >= src_rows)
        return;

    assert(checkShapeFromYUV420(dst.height(batch_idx), dst.width(batch_idx), code));

    uchar b = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx));
    uchar g = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, 1));
    uchar r = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx ^ 2));
    // Ignore alpha channel if input is RGBA.

    uchar Y{0}, U{0}, V{0};
    bgr_to_yuv42xxp_kernel(r, g, b, Y, U, V);

    // U and V are sampled at half the full resolution (in both x and y) and arranged as non-interleaved planes
    // (i.e., planar format). Each subsampled U and V "plane" is arranged as full rows after the full resolution Y
    // data--so two consecutive subsampled U or V rows are combined into one row spanning the same width as the Y
    // plane. Example memory layout for 4 x 4 image (e.g. I420):
    //   Y_00 Y_01 Y_02 Y_03
    //   Y_10 Y_11 Y_12 Y_13
    //   Y_20 Y_21 Y_22 Y_23
    //   Y_30 Y_31 Y_32 Y_33
    //   U_00 U_02 U_20 U_22
    //   V_00 V_02 V_20 V_22
    // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
    // and Y_11. Each full U and V row represents 4 rows of Y values. Some layouts (e.g., YV12) swap the location
    // of the U and V planes.

    *dst.ptr(batch_idx, src_y, src_x) = Y;
    if (src_y % 2 == 0 && src_x % 2 == 0)
    {
        const int by = src_rows + src_y / 4; // Base row index for U and V: subsampled plane is 1/4 the height.
        const int h4 = src_rows / 4;         // Height (# of rows) of each subsampled U and V plane.

        // Compute x position that combines two subsampled rows into one.
        const int uv_x = (src_x / 2) + ((src_rows / 2) & -((src_y / 2) & 1));

        *dst.ptr(batch_idx, by + h4 * uidx, uv_x)       = U; // Some formats swap the U and V "planes" (as indicated
        *dst.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x) = V; //   by the uidx parameter).
    }
}

template<class T>
__global__ void yuv420sp_to_bgr_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                          cuda::ImageBatchVarShapeWrapNHWC<T> dst, int bidx, int uidx,
                                          NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    int       dst_cols  = dst.width(batch_idx);
    int       dst_rows  = dst.height(batch_idx);
    if (dst_x >= dst_cols || dst_y >= dst_rows)
        return;

    assert(checkShapeFromYUV420(src.height(batch_idx), src.width(batch_idx), code));

    // See layout commments in bgr_to_yuv420sp_char_nhwc.
    const int uv_y = dst_rows + dst_y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
    const int uv_x = (dst_x & ~1);         // Convert x to even # (set lowest bit to 0).

    T Y = *src.ptr(batch_idx, dst_y, dst_x);
    T U = *src.ptr(batch_idx, uv_y, uv_x + uidx);       // Some formats swap the U and V elements (as indicated
    T V = *src.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)); //   by the uidx parameter).

    uchar r{0}, g{0}, b{0}, a{0xff};
    yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dst.numChannels() == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

template<class T>
__global__ void yuv420p_to_bgr_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                         cuda::ImageBatchVarShapeWrapNHWC<T> dst, int bidx, int uidx,
                                         NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    int       dst_cols  = dst.width(batch_idx);
    int       dst_rows  = dst.height(batch_idx);
    if (dst_x >= dst_cols || dst_y >= dst_rows)
        return;

    assert(checkShapeFromYUV420(src.height(batch_idx), src.width(batch_idx), code));

    // See layout commments in bgr_to_yuv420p_char_nhwc.
    const int by = dst_rows + dst_y / 4; // Base row index for U and V: subsampled plane is 1/4 the height.
    const int h4 = dst_rows / 4;         // Height (# of rows) of each subsampled U and V plane.

    // Compute x position that combines two subsampled rows into one.
    const int uv_x = (dst_x / 2) + ((dst_cols / 2) & -((dst_y / 2) & 1));

    T Y = *src.ptr(batch_idx, dst_y, dst_x);
    T U = *src.ptr(batch_idx, by + h4 * uidx, uv_x);       // Some formats swap the U and V "planes" (as indicated
    T V = *src.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x); //   by the uidx parameter).

    uchar r{0}, g{0}, b{0}, a{0xff};
    yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dst.numChannels() == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

// YUV 422 interleaved formats (e.g., YUYV, YVYU, and UYVY) group 2 pixels into groups of 4 elements. Each group of two
// pixels has two distinct luma (Y) values, one for each pixel. The chromaticity values (U and V) are subsampled by a
// factor of two so that there is only one U and one V value for each group of 2 pixels. Example memory layout for
// 4 x 4 image (UYVY format):
//   U_00 Y_00 V_00 Y_01 U_02 Y_02 V_02 Y_03
//   U_10 Y_10 V_10 Y_11 U_12 Y_12 V_12 Y_13
//   U_20 Y_20 V_20 Y_21 U_22 Y_22 V_22 Y_23
//   U_30 Y_30 V_30 Y_31 U_32 Y_32 V_32 Y_33
// Each U and V value corresponds to two Y values--e.g. U_00 and V_00 correspond to Y_00 and Y_10 while U_12 and V_12
// correspond to Y_12 and Y_13. Thus, a given Y value, Y_rc = Y(r,c) (where r is the row, or y coordinate, and c is the
// column, or x coordinate), corresponds to U(r,c') and V(r,c') where c' is the even column coordinate <= c -- that is,
// c' = 2 * floor(c/2) = (c & ~1). Some layouts swap the positions of the chromaticity and luma values (e.g., YUYV)
// (indicated by the yidx parameter) and / or swap the the positions of the U and V chromaticity valus (e.g., YVYU)
// (indicated by the uidx parameter).
// The data layout is treated as a single channel tensor, so each group of 4 values corresponds to two pixels. As such,
// the tensor width is twice the actual pixel width. Thus, it's easiest to process 4 consecutive values (2 pixels) per
// thread.
template<class T>
__global__ void yuv422_to_bgr_char_nhwc(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                        int dcn, int bidx, int yidx, int uidx)
{
    const int batch_idx = blockIdx.z;

    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dst.height(batch_idx))
        return;

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_x >= dst.width(batch_idx))
        return;

    int src_x = 2 * dst_x;    // Process 4 source elements/thread (i.e., 2 destination pixels).
    int uv_x  = (src_x & ~3); // Compute "even" x coordinate for U and V (set lowest two bits to 0).

    T Y0 = *src.ptr(batch_idx, dst_y, src_x + yidx);
    T Y1 = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
    T U  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + uidx);
    T V  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + (uidx ^ 2));

    uchar r{0}, g{0}, b{0}, a{0xff};

    yuv42xxp_to_bgr_kernel(int(Y0), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;

    dst_x++; // Move to next output pixel.
    yuv42xxp_to_bgr_kernel(int(Y1), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
}

template<class T>
__global__ void yuv420_to_gray_char_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                         cuda::ImageBatchVarShapeWrapNHWC<T> dst, NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    assert(checkShapeFromYUV420(src.height(batch_idx), src.width(batch_idx), code));

    T Y                                  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = Y;
}

// See layout comment before yuv422_to_bgr_char_nhwc.
template<class T>
__global__ void yuv422_to_gray_char_nhwc(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst,
                                         int yidx)
{
    const int batch_idx = blockIdx.z;

    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dst.height(batch_idx))
        return;

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_x >= dst.width(batch_idx))
        return;

    int src_x = 2 * dst_x; // Process 4 source elements/thread (i.e., 2 destination pixels).

    *dst.ptr(batch_idx, dst_y, dst_x++) = *src.ptr(batch_idx, dst_y, src_x + yidx);
    *dst.ptr(batch_idx, dst_y, dst_x)   = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
}

inline void BGR_to_RGB(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                       cudaStream_t stream)
{
    int sch  = (code == NVCV_COLOR_BGRA2BGR || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int dch  = (code == NVCV_COLOR_BGR2BGRA || code == NVCV_COLOR_BGR2RGBA || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int bidx = (code == NVCV_COLOR_BGR2RGB || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA
                || code == NVCV_COLOR_BGR2RGBA)
                 ? 2
                 : 0;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != sch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d expecting: %d",
                              channels, sch);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != dch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d expecting: %d",
                              dcn, dch);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

#define CVCUDA_RUN_BGR2RGB(T)                                                                        \
    do                                                                                               \
    {                                                                                                \
        if (isPlanar)                                                                                \
        {                                                                                            \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                         \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                        \
            rgb_to_bgr_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, sch, dch); \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, sch);                                \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dch);                               \
            rgb_to_bgr_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);          \
        }                                                                                            \
        NVCV_CHECK_THROW(cudaGetLastError());                                                        \
    }                                                                                                \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    case ElemType::k8S:
    {
        CVCUDA_RUN_BGR2RGB(unsigned char);
    }
    break;
    // F16 runs real __half kernels: pure channel swaps stay bit-identical to the former 16-bit
    // integer route, and an added alpha channel becomes 1.0 (the floating-point opaque value).
    case ElemType::k16F:
    {
        CVCUDA_RUN_BGR2RGB(__half);
    }
    break;
    case ElemType::k16U:
    case ElemType::k16S:
    {
        CVCUDA_RUN_BGR2RGB(uint16_t);
    }
    break;
    case ElemType::k32S:
    {
        CVCUDA_RUN_BGR2RGB(int32_t);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_BGR2RGB(float);
    }
    break;
    case ElemType::k64F:
    {
        CVCUDA_RUN_BGR2RGB(double);
    }
    break;
    }
#undef CVCUDA_RUN_BGR2RGB
}

inline void GRAY_to_BGR(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                        cudaStream_t stream)
{
    int dch = (code == NVCV_COLOR_GRAY2BGRA) ? 4 : 3;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d expecting: 1",
                              channels);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != dch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d expecting: %d",
                              dcn, dch);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

#define CVCUDA_RUN_GRAY2BGR(T)                                                             \
    do                                                                                     \
    {                                                                                      \
        if (isPlanar)                                                                      \
        {                                                                                  \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                               \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                              \
            gray_to_bgr_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch); \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                 \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dch);                     \
            gray_to_bgr_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr);     \
        }                                                                                  \
        NVCV_CHECK_THROW(cudaGetLastError());                                              \
    }                                                                                      \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    case ElemType::k8S:
    {
        CVCUDA_RUN_GRAY2BGR(unsigned char);
    }
    break;
    case ElemType::k16F: // Real __half kernels; the gray broadcast stays bit-exact.
    {
        CVCUDA_RUN_GRAY2BGR(__half);
    }
    break;
    case ElemType::k16U:
    case ElemType::k16S:
    {
        CVCUDA_RUN_GRAY2BGR(uint16_t);
    }
    break;
    case ElemType::k32S:
    {
        CVCUDA_RUN_GRAY2BGR(int32_t);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_GRAY2BGR(float);
    }
    break;
    case ElemType::k64F:
    {
        CVCUDA_RUN_GRAY2BGR(double);
    }
    break;
    }
#undef CVCUDA_RUN_GRAY2BGR
}

inline void BGR_to_GRAY(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                        cudaStream_t stream)
{
    int bidx = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_RGB2GRAY) ? 2 : 0;
    int sch  = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_BGRA2GRAY) ? 4 : 3;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != sch)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d expecting: %d",
                              channels, sch);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d expecting: 1",
                              dcn);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);
    dim3 gridSize4(util::DivUp(max_width, static_cast<int>(blockSize.x * 4)), gridSize.y, gridSize.z);

#define CVCUDA_RUN_BGR2GRAY_CHAR(T, NIX)                                                              \
    do                                                                                                \
    {                                                                                                 \
        if (isPlanar)                                                                                 \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                          \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                         \
            bgr_to_gray_char_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, sch); \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                            \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                \
            bgr_to_gray_char_nhwc<NIX, T>                                                             \
                <<<NIX == 4 ? gridSize4 : gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);  \
        }                                                                                             \
        NVCV_CHECK_THROW(cudaGetLastError());                                                         \
    }                                                                                                 \
    while (0)

#define CVCUDA_RUN_BGR2GRAY_FLOAT(T)                                                                   \
    do                                                                                                 \
    {                                                                                                  \
        if (isPlanar)                                                                                  \
        {                                                                                              \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                           \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                          \
            bgr_to_gray_float_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, sch); \
        }                                                                                              \
        else                                                                                           \
        {                                                                                              \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                             \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                 \
            bgr_to_gray_float_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);     \
        }                                                                                              \
        NVCV_CHECK_THROW(cudaGetLastError());                                                          \
    }                                                                                                  \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    {
        CVCUDA_RUN_BGR2GRAY_CHAR(unsigned char, 4);
    }
    break;
    case ElemType::k16U:
    {
        CVCUDA_RUN_BGR2GRAY_CHAR(unsigned short, 1);
    }
    break;
    case ElemType::k16F: // Luma accumulates in float via the mixed __half operators, rounds on store.
    {
        CVCUDA_RUN_BGR2GRAY_FLOAT(__half);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_BGR2GRAY_FLOAT(float);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_RUN_BGR2GRAY_FLOAT
#undef CVCUDA_RUN_BGR2GRAY_CHAR
}

inline void BGR_to_YUV(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                       cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_BGR2YUV ? 0 : 2;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d expecting: 3",
                              dcn);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

#define CVCUDA_RUN_BGR2YUV_CHAR(T)                                                                        \
    do                                                                                                    \
    {                                                                                                     \
        if (isPlanar)                                                                                     \
        {                                                                                                 \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                              \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                             \
            bgr_to_yuv_char_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, channels); \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                                \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                    \
            bgr_to_yuv_char_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);          \
        }                                                                                                 \
        NVCV_CHECK_THROW(cudaGetLastError());                                                             \
    }                                                                                                     \
    while (0)

#define CVCUDA_RUN_BGR2YUV_FLOAT(T)                                                                        \
    do                                                                                                     \
    {                                                                                                      \
        if (isPlanar)                                                                                      \
        {                                                                                                  \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                               \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                              \
            bgr_to_yuv_float_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, channels); \
        }                                                                                                  \
        else                                                                                               \
        {                                                                                                  \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                                 \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                     \
            bgr_to_yuv_float_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);          \
        }                                                                                                  \
        NVCV_CHECK_THROW(cudaGetLastError());                                                              \
    }                                                                                                      \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    {
        CVCUDA_RUN_BGR2YUV_CHAR(unsigned char);
    }
    break;
    case ElemType::k16U:
    {
        CVCUDA_RUN_BGR2YUV_CHAR(unsigned short);
    }
    break;
    case ElemType::k16F: // Follows the float path (the kernels compute in float, round on store).
    {
        CVCUDA_RUN_BGR2YUV_FLOAT(__half);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_BGR2YUV_FLOAT(float);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_RUN_BGR2YUV_FLOAT
#undef CVCUDA_RUN_BGR2YUV_CHAR
}

inline void YUV_to_BGR(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                       cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_YUV2BGR ? 0 : 2;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid output channel number %d different than input channel %d", dcn, channels);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

#define CVCUDA_RUN_YUV2BGR_CHAR(T)                                                                   \
    do                                                                                               \
    {                                                                                                \
        if (isPlanar)                                                                                \
        {                                                                                            \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                         \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                        \
            yuv_to_bgr_char_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, dcn); \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                           \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                               \
            yuv_to_bgr_char_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);     \
        }                                                                                            \
        NVCV_CHECK_THROW(cudaGetLastError());                                                        \
    }                                                                                                \
    while (0)

#define CVCUDA_RUN_YUV2BGR_FLOAT(T)                                                                   \
    do                                                                                                \
    {                                                                                                 \
        if (isPlanar)                                                                                 \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                          \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                         \
            yuv_to_bgr_float_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, dcn); \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                            \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                \
            yuv_to_bgr_float_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);     \
        }                                                                                             \
        NVCV_CHECK_THROW(cudaGetLastError());                                                         \
    }                                                                                                 \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    {
        CVCUDA_RUN_YUV2BGR_CHAR(unsigned char);
    }
    break;
    case ElemType::k16U:
    {
        CVCUDA_RUN_YUV2BGR_CHAR(unsigned short);
    }
    break;
    case ElemType::k16F: // Follows the float path (the kernels compute in float, round on store).
    {
        CVCUDA_RUN_YUV2BGR_FLOAT(__half);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_YUV2BGR_FLOAT(float);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_RUN_YUV2BGR_FLOAT
#undef CVCUDA_RUN_YUV2BGR_CHAR
}

template<bool RGB2Lab, bool BGR, bool SRGB, class T>
__global__ void rgb_lab_nhwc(cuda::ImageBatchVarShapeWrapNHWC<T> src, cuda::ImageBatchVarShapeWrapNHWC<T> dst)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    const T in0 = *src.ptr(batch_idx, dst_y, dst_x, 0);
    const T in1 = *src.ptr(batch_idx, dst_y, dst_x, 1);
    const T in2 = *src.ptr(batch_idx, dst_y, dst_x, 2);

    T out0, out1, out2;
    if constexpr (RGB2Lab)
    {
        cvcuda::priv::lab::RGBToLab<SRGB, true>(BGR ? in2 : in0, in1, BGR ? in0 : in2, out0, out1, out2);
    }
    else
    {
        cvcuda::priv::lab::LabToRGB<SRGB, true>(in0, in1, in2, BGR ? out2 : out0, out1, BGR ? out0 : out2);
    }

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = out0;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = out1;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = out2;
}

template<bool RGB2Lab, bool BGR, bool SRGB, class T>
__global__ void rgb_lab_chw(cuda::ImageBatchVarShapeWrap<T> src, cuda::ImageBatchVarShapeWrap<T> dst)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    if (dst_x >= dst.width(batch_idx, 0) || dst_y >= dst.height(batch_idx, 0))
        return;

    const T in0 = *src.ptr(batch_idx, 0, dst_y, dst_x);
    const T in1 = *src.ptr(batch_idx, 1, dst_y, dst_x);
    const T in2 = *src.ptr(batch_idx, 2, dst_y, dst_x);

    T out0, out1, out2;
    if constexpr (RGB2Lab)
    {
        cvcuda::priv::lab::RGBToLab<SRGB, true>(BGR ? in2 : in0, in1, BGR ? in0 : in2, out0, out1, out2);
    }
    else
    {
        cvcuda::priv::lab::LabToRGB<SRGB, true>(in0, in1, in2, BGR ? out2 : out0, out1, BGR ? out0 : out2);
    }

    *dst.ptr(batch_idx, 0, dst_y, dst_x) = out0;
    *dst.ptr(batch_idx, 1, dst_y, dst_x) = out1;
    *dst.ptr(batch_idx, 2, dst_y, dst_x) = out2;
}

template<bool RGB2Lab, bool BGR, bool SRGB, class T>
inline void Launch_RGB_Lab(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                           const nvcv::ImageBatchVarShapeDataStridedCuda &outData, bool isPlanar, dim3 gridSize,
                           dim3 blockSize, cudaStream_t stream)
{
    constexpr int channels = 3;
    if (isPlanar)
    {
        cuda::ImageBatchVarShapeWrap<T> src(inData);
        cuda::ImageBatchVarShapeWrap<T> dst(outData);
        rgb_lab_chw<RGB2Lab, BGR, SRGB><<<gridSize, blockSize, 0, stream>>>(src, dst);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<T> src(inData, channels);
        cuda::ImageBatchVarShapeWrapNHWC<T> dst(outData, channels);
        rgb_lab_nhwc<RGB2Lab, BGR, SRGB><<<gridSize, blockSize, 0, stream>>>(src, dst);
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<class T>
inline void Dispatch_RGB_Lab(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                             const nvcv::ImageBatchVarShapeDataStridedCuda &outData, bool isPlanar, dim3 gridSize,
                             dim3 blockSize, NVCVColorConversionCode code, cudaStream_t stream)
{
#define CVCUDA_RGB_LAB_CASE(CODE, RGB2LAB, BGR, SRGB) \
    case CODE:                                        \
        return Launch_RGB_Lab<RGB2LAB, BGR, SRGB, T>(inData, outData, isPlanar, gridSize, blockSize, stream)

    switch (code)
    {
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_BGR2Lab, true, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_RGB2Lab, true, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2BGR, false, true, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2RGB, false, false, true);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LBGR2Lab, true, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_LRGB2Lab, true, false, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LBGR, false, true, false);
        CVCUDA_RGB_LAB_CASE(NVCV_COLOR_Lab2LRGB, false, false, false);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Lab conversion code");
    }

#undef CVCUDA_RGB_LAB_CASE
}

inline void RGB_Lab(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                    const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                    cudaStream_t stream)
{
    constexpr int channels       = 3;
    const bool    fastF16Reverse = code == NVCV_COLOR_Lab2BGR || code == NVCV_COLOR_Lab2RGB;
    const bool    isPlanar       = IsPlanar(GetImageLayout(inData));

    const int srcChannels = inData.uniqueFormat().numChannels();
    const int dstChannels = outData.uniqueFormat().numChannels();
    if (srcChannels != channels || dstChannels != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "RGB/Lab conversion requires 3-channel input and output");
    }

    ElemType dataType    = ClassifyElemType(inData.uniqueFormat());
    ElemType outDataType = ClassifyElemType(outData.uniqueFormat());
    if (dataType != outDataType)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    const int batchSize = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batchSize);
    }

    const int maxWidth    = inData.maxSize().w;
    const int maxHeight   = inData.maxSize().h;
    const int blockHeight = dataType == ElemType::k16F && fastF16Reverse ? BLOCK / 8 : BLOCK / 4;
    dim3      blockSize(BLOCK, blockHeight, 1);
    dim3      gridSize(util::DivUp(maxWidth, static_cast<int>(blockSize.x)),
                       util::DivUp(maxHeight, static_cast<int>(blockSize.y)), batchSize);

    switch (dataType)
    {
    case ElemType::k8U:
        return Dispatch_RGB_Lab<unsigned char>(inData, outData, isPlanar, gridSize, blockSize, code, stream);
    case ElemType::k16F:
        return Dispatch_RGB_Lab<__half>(inData, outData, isPlanar, gridSize, blockSize, code, stream);
    case ElemType::k32F:
        return Dispatch_RGB_Lab<float>(inData, outData, isPlanar, gridSize, blockSize, code, stream);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
}

inline void BGR_to_HSV(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                       cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
    int  bidx        = (code == NVCV_COLOR_BGR2HSV || code == NVCV_COLOR_BGR2HSV_FULL) ? 0 : 2;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != channels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid output channel number %d different than input channel %d", dcn, channels);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

#define CVCUDA_RUN_BGR2HSV_CHAR(T)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (isPlanar)                                                                                                  \
        {                                                                                                              \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                                           \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                                          \
            bgr_to_hsv_char_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, isFullRange, channels); \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                                             \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                                 \
            bgr_to_hsv_char_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, isFullRange);          \
        }                                                                                                              \
        NVCV_CHECK_THROW(cudaGetLastError());                                                                          \
    }                                                                                                                  \
    while (0)

#define CVCUDA_RUN_BGR2HSV_FLOAT(T)                                                                        \
    do                                                                                                     \
    {                                                                                                      \
        if (isPlanar)                                                                                      \
        {                                                                                                  \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                               \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                              \
            bgr_to_hsv_float_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, channels); \
        }                                                                                                  \
        else                                                                                               \
        {                                                                                                  \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                                 \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                     \
            bgr_to_hsv_float_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);          \
        }                                                                                                  \
        NVCV_CHECK_THROW(cudaGetLastError());                                                              \
    }                                                                                                      \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    {
        CVCUDA_RUN_BGR2HSV_CHAR(unsigned char);
    }
    break;
    case ElemType::k16F: // Follows the float path (hue in [0, 360); FULL only affects 8-bit).
    {
        CVCUDA_RUN_BGR2HSV_FLOAT(__half);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_BGR2HSV_FLOAT(float);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_RUN_BGR2HSV_FLOAT
#undef CVCUDA_RUN_BGR2HSV_CHAR
}

inline void HSV_to_BGR(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                       const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                       cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL);
    int  bidx        = (code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL) ? 0 : 2;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());
    bool     isPlanar      = IsPlanar(GetImageLayout(inData));

    if (channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }

    if (data_type != out_data_type)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3 && dcn != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", dcn);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();
    if (isPlanar)
    {
        RequirePlanarBatchFitsGridZ(batch_size);
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);
    dim3 gridSize2(util::DivUp(max_width, static_cast<int>(blockSize.x * 2)), gridSize.y, gridSize.z);

#define CVCUDA_RUN_HSV2BGR_CHAR(T)                                                                                \
    do                                                                                                            \
    {                                                                                                             \
        if (isPlanar)                                                                                             \
        {                                                                                                         \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                                      \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                                     \
            hsv_to_bgr_char_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, isFullRange, dcn); \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                                        \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                            \
            hsv_to_bgr_char_nhwc<2, T><<<gridSize2, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, isFullRange); \
        }                                                                                                         \
        NVCV_CHECK_THROW(cudaGetLastError());                                                                     \
    }                                                                                                             \
    while (0)

#define CVCUDA_RUN_HSV2BGR_FLOAT(T)                                                                   \
    do                                                                                                \
    {                                                                                                 \
        if (isPlanar)                                                                                 \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrap<T> src_ptr(inData);                                          \
            cuda::ImageBatchVarShapeWrap<T> dst_ptr(outData);                                         \
            hsv_to_bgr_float_chw<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, dcn); \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            cuda::ImageBatchVarShapeWrapNHWC<T> src_ptr(inData, channels);                            \
            cuda::ImageBatchVarShapeWrapNHWC<T> dst_ptr(outData, dcn);                                \
            hsv_to_bgr_float_nhwc<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);     \
        }                                                                                             \
        NVCV_CHECK_THROW(cudaGetLastError());                                                         \
    }                                                                                                 \
    while (0)

    switch (data_type)
    {
    case ElemType::k8U:
    {
        CVCUDA_RUN_HSV2BGR_CHAR(unsigned char);
    }
    break;
    case ElemType::k16F: // Follows the float path; an added alpha channel becomes 1.0.
    {
        CVCUDA_RUN_HSV2BGR_FLOAT(__half);
    }
    break;
    case ElemType::k32F:
    {
        CVCUDA_RUN_HSV2BGR_FLOAT(float);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported DataType");
    }
#undef CVCUDA_RUN_HSV2BGR_FLOAT
#undef CVCUDA_RUN_HSV2BGR_CHAR
}

inline void YUV420xp_to_BGR(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData)) || IsPlanar(GetImageLayout(outData)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support subsampled YUV420 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2BGR_NV21
           || code == NVCV_COLOR_YUV2BGRA_NV21 || code == NVCV_COLOR_YUV2BGR_YV12 || code == NVCV_COLOR_YUV2BGRA_YV12
           || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2RGB_NV12
           || code == NVCV_COLOR_YUV2RGBA_NV12 || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV
           || code == NVCV_COLOR_YUV2RGB_IYUV || code == NVCV_COLOR_YUV2RGBA_IYUV)
            ? 0
            : 1;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());

    if (channels != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }
    if (data_type != ElemType::k8U || out_data_type != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if ((code != NVCV_COLOR_YUV2GRAY_420 || dcn != 1) && dcn != 3 && dcn != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", dcn);
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height * 2 / 3, static_cast<int>(blockSize.y)), batch_size);

    cuda::ImageBatchVarShapeWrapNHWC<unsigned char> src_ptr(inData, channels);
    cuda::ImageBatchVarShapeWrapNHWC<unsigned char> dst_ptr(outData, dcn);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_420:
    {
        yuv420_to_gray_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, code);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2BGRA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV21:
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2RGBA_NV21:
    {
        yuv420sp_to_bgr_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, uidx, code);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
    {
        yuv420p_to_bgr_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, uidx, code);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

inline void YUV422_to_BGR(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                          const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                          cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData)) || IsPlanar(GetImageLayout(outData)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support packed YUV422 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU
           || code == NVCV_COLOR_YUV2BGRA_YVYU || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY)
            ? 0
            : 2;

    int yidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU || code == NVCV_COLOR_YUV2BGRA_YVYU
           || code == NVCV_COLOR_YUV2RGB_YVYU || code == NVCV_COLOR_YUV2RGBA_YVYU || code == NVCV_COLOR_YUV2GRAY_YUY2)
            ? 0
            : 1;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY
           || code == NVCV_COLOR_YUV2RGB_UYVY || code == NVCV_COLOR_YUV2RGBA_UYVY)
            ? 0
            : 2;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());

    if (channels != 3)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }
    if (data_type != ElemType::k8U || out_data_type != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    int dcn = outData.uniqueFormat().numChannels();

    if ((code != NVCV_COLOR_YUV2GRAY_UYVY && code != NVCV_COLOR_YUV2GRAY_YUY2 || dcn != 1) && dcn != 3 && dcn != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output channel number %d", dcn);
    }

    auto inList = inData.imageList();

    for (int i = 0; i < inData.numImages(); i++)
    {
        if (inList[i].numPlanes != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input batch images must all be a single plane of data");
        }

        NVCVImagePlaneStrided plane = inList[i].planes[0];

        if (plane.width % 2 != 0)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input batch images must all have a width that is a multiple of 2");
        }
        if (plane.rowStride < plane.width * 2)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Insufficient input batch image stride");
        }
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(max_width / 2, static_cast<int>(blockSize.x)),
                  util::DivUp(max_height, static_cast<int>(blockSize.y)), batch_size);

    cuda::ImageBatchVarShapeWrap<uint8_t>     src_ptr(inData);
    cuda::ImageBatchVarShapeWrapNHWC<uint8_t> dst_ptr(outData, dcn);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_YUY2:
    case NVCV_COLOR_YUV2GRAY_UYVY:
    {
        yuv422_to_gray_char_nhwc<uint8_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, yidx);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    case NVCV_COLOR_YUV2BGR_YUY2:
    case NVCV_COLOR_YUV2BGR_YVYU:
    case NVCV_COLOR_YUV2BGRA_YUY2:
    case NVCV_COLOR_YUV2BGRA_YVYU:
    case NVCV_COLOR_YUV2RGB_YUY2:
    case NVCV_COLOR_YUV2RGB_YVYU:
    case NVCV_COLOR_YUV2RGBA_YUY2:
    case NVCV_COLOR_YUV2RGBA_YVYU:
    case NVCV_COLOR_YUV2RGB_UYVY:
    case NVCV_COLOR_YUV2BGR_UYVY:
    case NVCV_COLOR_YUV2RGBA_UYVY:
    case NVCV_COLOR_YUV2BGRA_UYVY:
    {
        yuv422_to_bgr_char_nhwc<uint8_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dcn, bidx, yidx, uidx);
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

// The planar and semi-planar writers take the same arguments over the same grid; only the kernel
// differs, so it is selected at compile time.
template<bool IsSemiPlanar>
inline void bgr_to_yuv420_launcher(cuda::ImageBatchVarShapeWrapNHWC<uchar> src_ptr,
                                   cuda::ImageBatchVarShapeWrapNHWC<uchar> dst_ptr, ImageShape inputShape, int bidx,
                                   int uidx, NVCVColorConversionCode code, cudaStream_t stream)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(util::DivUp(inputShape.W, static_cast<int>(blockSize.x)),
                  util::DivUp(inputShape.H, static_cast<int>(blockSize.y)), inputShape.N);

    if constexpr (IsSemiPlanar)
    {
        bgr_to_yuv420sp_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, uidx, code);
    }
    else
    {
        bgr_to_yuv420p_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, uidx, code);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

inline void BGR_to_YUV420xp(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    if (IsPlanar(GetImageLayout(inData)) || IsPlanar(GetImageLayout(outData)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar CvtColor does not support subsampled YUV420 conversion codes");
    }

    int bidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_NV21
           || code == NVCV_COLOR_BGRA2YUV_NV21 || code == NVCV_COLOR_BGR2YUV_YV12 || code == NVCV_COLOR_BGRA2YUV_YV12
           || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_RGB2YUV_NV12
           || code == NVCV_COLOR_RGBA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV
           || code == NVCV_COLOR_RGB2YUV_IYUV || code == NVCV_COLOR_RGBA2YUV_IYUV)
            ? 0
            : 1;

    int      channels      = inData.uniqueFormat().numChannels();
    ElemType data_type     = ClassifyElemType(inData.uniqueFormat());
    ElemType out_data_type = ClassifyElemType(outData.uniqueFormat());

    if (channels != 3 && channels != 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input channel number %d", channels);
    }
    if (data_type != ElemType::k8U || out_data_type != ElemType::k8U)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported input/output DataType");
    }

    // The YUV420 writers address the destination as one interleaved single-channel plane, so the
    // wrapper is built with one channel whatever the output format reports.
    int dcn = 1;

    // BGR input
    cuda::ImageBatchVarShapeWrapNHWC<unsigned char> src_ptr(inData, channels);
    // YUV420xp output
    cuda::ImageBatchVarShapeWrapNHWC<unsigned char> dst_ptr(outData, dcn);

    ImageShape maxInputShape{inData.numImages(), channels, inData.maxSize().h, inData.maxSize().w};

    switch (code)
    {
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV21:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV21:
    {
        bgr_to_yuv420_launcher<true>(src_ptr, dst_ptr, maxInputShape, bidx, uidx, code, stream);
    }
    break;
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
    {
        bgr_to_yuv420_launcher<false>(src_ptr, dst_ptr, maxInputShape, bidx, uidx, code, stream);
    }
    break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported conversion code %d", (int)code);
    }
}

inline void Infer(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                  const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                  cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the input batch must all have the same format");
    }
    if (!outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the output batch must all have the same format");
    }

    ImageLayout input_format  = GetImageLayout(inData);
    ImageLayout output_format = GetImageLayout(outData);
    if (input_format != output_format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input and output batches");
    }

    using func_t = void (*)(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream);

    // The set of supported NVCVColorConversionCode values is defined by this table: a null entry is
    // a code the operator rejects. Indices and comments are kept verbatim so the supported set can
    // be diffed against the public NVCVColorConversionCode enum by eye.
    static const func_t funcs[] = {
        BGR_to_RGB, // CV_BGR2BGRA    =0
        BGR_to_RGB, // CV_BGRA2BGR    =1
        BGR_to_RGB, // CV_BGR2RGBA    =2
        BGR_to_RGB, // CV_RGBA2BGR    =3
        BGR_to_RGB, // CV_BGR2RGB     =4
        BGR_to_RGB, // CV_BGRA2RGBA   =5

        BGR_to_GRAY, // CV_BGR2GRAY    =6
        BGR_to_GRAY, // CV_RGB2GRAY    =7
        GRAY_to_BGR, // CV_GRAY2BGR    =8
        GRAY_to_BGR, // CV_GRAY2BGRA, CV_GRAY2RGBA   =9
        BGR_to_GRAY, // CV_BGRA2GRAY   =10
        BGR_to_GRAY, // CV_RGBA2GRAY   =11

        nullptr, //BGR_to_BGR565,          // CV_BGR2BGR565  =12
        nullptr, //RGB_to_BGR565,          // CV_RGB2BGR565  =13
        nullptr, //BGR565_to_BGR,          // CV_BGR5652BGR  =14
        nullptr, //BGR565_to_RGB,          // CV_BGR5652RGB  =15
        nullptr, //BGRA_to_BGR565,         // CV_BGRA2BGR565 =16
        nullptr, //RGBA_to_BGR565,         // CV_RGBA2BGR565 =17
        nullptr, //BGR565_to_BGRA,         // CV_BGR5652BGRA =18
        nullptr, //BGR565_to_RGBA,         // CV_BGR5652RGBA =19

        nullptr, //GRAY_to_BGR565,         // CV_GRAY2BGR565 =20
        nullptr, //BGR565_to_GRAY,         // CV_BGR5652GRAY =21

        nullptr, //BGR_to_BGR555,          // CV_BGR2BGR555  =22
        nullptr, //RGB_to_BGR555,          // CV_RGB2BGR555  =23
        nullptr, //BGR555_to_BGR,          // CV_BGR5552BGR  =24
        nullptr, //BGR555_to_RGB,          // CV_BGR5552RGB  =25
        nullptr, //BGRA_to_BGR555,         // CV_BGRA2BGR555 =26
        nullptr, //RGBA_to_BGR555,         // CV_RGBA2BGR555 =27
        nullptr, //BGR555_to_BGRA,         // CV_BGR5552BGRA =28
        nullptr, //BGR555_to_RGBA,         // CV_BGR5552RGBA =29

        nullptr, //GRAY_to_BGR555,         // CV_GRAY2BGR555 =30
        nullptr, //BGR555_to_GRAY,         // CV_BGR5552GRAY =31

        nullptr, //BGR_to_XYZ,             // CV_BGR2XYZ     =32
        nullptr, //RGB_to_XYZ,             // CV_RGB2XYZ     =33
        nullptr, //XYZ_to_BGR,             // CV_XYZ2BGR     =34
        nullptr, //XYZ_to_RGB,             // CV_XYZ2RGB     =35

        nullptr, //BGR_to_YCrCb,           // CV_BGR2YCrCb   =36
        nullptr, //RGB_to_YCrCb,           // CV_RGB2YCrCb   =37
        nullptr, //YCrCb_to_BGR,           // CV_YCrCb2BGR   =38
        nullptr, //YCrCb_to_RGB,           // CV_YCrCb2RGB   =39

        BGR_to_HSV, //BGR_to_HSV,             // CV_BGR2HSV     =40
        BGR_to_HSV, //RGB_to_HSV,             // CV_RGB2HSV     =41

        nullptr, //                =42
        nullptr, //                =43

        RGB_Lab, //BGR_to_Lab,             // CV_BGR2Lab     =44
        RGB_Lab, //RGB_to_Lab,             // CV_RGB2Lab     =45

        nullptr, //bayerBG_to_BGR,         // CV_BayerBG2BGR =46
        nullptr, //bayeRGB_to_BGR,         // CV_BayeRGB2BGR =47
        nullptr, //bayerRG_to_BGR,         // CV_BayerRG2BGR =48
        nullptr, //bayerGR_to_BGR,         // CV_BayerGR2BGR =49

        nullptr, //BGR_to_Luv,             // CV_BGR2Luv     =50
        nullptr, //RGB_to_Luv,             // CV_RGB2Luv     =51

        nullptr, //BGR_to_HLS,             // CV_BGR2HLS     =52
        nullptr, //RGB_to_HLS,             // CV_RGB2HLS     =53

        HSV_to_BGR, // CV_HSV2BGR     =54
        HSV_to_BGR, // CV_HSV2RGB     =55

        RGB_Lab, //Lab_to_BGR,             // CV_Lab2BGR     =56
        RGB_Lab, //Lab_to_RGB,             // CV_Lab2RGB     =57
        nullptr, //Luv_to_BGR,             // CV_Luv2BGR     =58
        nullptr, //Luv_to_RGB,             // CV_Luv2RGB     =59

        nullptr, //HLS_to_BGR,             // CV_HLS2BGR     =60
        nullptr, //HLS_to_RGB,             // CV_HLS2RGB     =61

        nullptr, // CV_BayerBG2BGR_VNG =62
        nullptr, // CV_BayeRGB2BGR_VNG =63
        nullptr, // CV_BayerRG2BGR_VNG =64
        nullptr, // CV_BayerGR2BGR_VNG =65

        BGR_to_HSV, //BGR_to_HSV_FULL,        // CV_BGR2HSV_FULL = 66
        BGR_to_HSV, //RGB_to_HSV_FULL,        // CV_RGB2HSV_FULL = 67
        nullptr,    //BGR_to_HLS_FULL,        // CV_BGR2HLS_FULL = 68
        nullptr,    //RGB_to_HLS_FULL,        // CV_RGB2HLS_FULL = 69

        HSV_to_BGR, // CV_HSV2BGR_FULL = 70
        HSV_to_BGR, // CV_HSV2RGB_FULL = 71
        nullptr,    //HLS_to_BGR_FULL,        // CV_HLS2BGR_FULL = 72
        nullptr,    //HLS_to_RGB_FULL,        // CV_HLS2RGB_FULL = 73

        RGB_Lab, //LBGR_to_Lab,            // CV_LBGR2Lab     = 74
        RGB_Lab, //LRGB_to_Lab,            // CV_LRGB2Lab     = 75
        nullptr, //LBGR_to_Luv,            // CV_LBGR2Luv     = 76
        nullptr, //LRGB_to_Luv,            // CV_LRGB2Luv     = 77

        RGB_Lab, //Lab_to_LBGR,            // CV_Lab2LBGR     = 78
        RGB_Lab, //Lab_to_LRGB,            // CV_Lab2LRGB     = 79
        nullptr, //Luv_to_LBGR,            // CV_Luv2LBGR     = 80
        nullptr, //Luv_to_LRGB,            // CV_Luv2LRGB     = 81

        BGR_to_YUV, // CV_BGR2YUV      = 82
        BGR_to_YUV, // CV_RGB2YUV      = 83
        YUV_to_BGR, // CV_YUV2BGR      = 84
        YUV_to_BGR, // CV_YUV2RGB      = 85

        nullptr, //bayerBG_to_gray,        // CV_BayerBG2GRAY = 86
        nullptr, //bayeRGB_to_GRAY,        // CV_BayeRGB2GRAY = 87
        nullptr, //bayerRG_to_gray,        // CV_BayerRG2GRAY = 88
        nullptr, //bayerGR_to_gray,        // CV_BayerGR2GRAY = 89

        //! YUV 4:2:0 family to RGB
        YUV420xp_to_BGR, // CV_YUV2RGB_NV12 = 90,
        YUV420xp_to_BGR, // CV_YUV2BGR_NV12 = 91,
        YUV420xp_to_BGR, // CV_YUV2RGB_NV21 = 92, CV_YUV420sp2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_NV21 = 93, CV_YUV420sp2BGR

        YUV420xp_to_BGR, // CV_YUV2RGBA_NV12 = 94,
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV12 = 95,
        YUV420xp_to_BGR, // CV_YUV2RGBA_NV21 = 96, CV_YUV420sp2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV21 = 97, CV_YUV420sp2BGRA

        YUV420xp_to_BGR, // CV_YUV2RGB_YV12 = 98, CV_YUV420p2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_YV12 = 99, CV_YUV420p2BGR
        YUV420xp_to_BGR, // CV_YUV2RGB_IYUV = 100, CV_YUV2RGB_I420
        YUV420xp_to_BGR, // CV_YUV2BGR_IYUV = 101, CV_YUV2BGR_I420

        YUV420xp_to_BGR, // CV_YUV2RGBA_YV12 = 102, CV_YUV420p2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_YV12 = 103, CV_YUV420p2BGRA
        YUV420xp_to_BGR, // CV_YUV2RGBA_IYUV = 104, CV_YUV2RGBA_I420
        YUV420xp_to_BGR, // CV_YUV2BGRA_IYUV = 105, CV_YUV2BGRA_I420

        YUV420xp_to_BGR, // CV_YUV2GRAY_420 = 106,
        // CV_YUV2GRAY_NV21,
        // CV_YUV2GRAY_NV12,
        // CV_YUV2GRAY_YV12,
        // CV_YUV2GRAY_IYUV,
        // CV_YUV2GRAY_I420,
        // CV_YUV420sp2GRAY,
        // CV_YUV420p2GRAY ,

        //! YUV 4:2:2 family to RGB
        YUV422_to_BGR, // CV_YUV2RGB_UYVY = 107, CV_YUV2RGB_Y422, CV_YUV2RGB_UYNV
        YUV422_to_BGR, // CV_YUV2BGR_UYVY = 108, CV_YUV2BGR_Y422, CV_YUV2BGR_UYNV
        nullptr,       // CV_YUV2RGB_VYUY = 109,
        nullptr,       // CV_YUV2BGR_VYUY = 110,

        YUV422_to_BGR, // CV_YUV2RGBA_UYVY = 111, CV_YUV2RGBA_Y422, CV_YUV2RGBA_UYNV
        YUV422_to_BGR, // CV_YUV2BGRA_UYVY = 112, CV_YUV2BGRA_Y422, CV_YUV2BGRA_UYNV
        nullptr,       // CV_YUV2RGBA_VYUY = 113,
        nullptr,       // CV_YUV2BGRA_VYUY = 114,

        YUV422_to_BGR, // CV_YUV2RGB_YUY2 = 115, CV_YUV2RGB_YUYV, CV_YUV2RGB_YUNV
        YUV422_to_BGR, // CV_YUV2BGR_YUY2 = 116, CV_YUV2BGR_YUYV, CV_YUV2BGR_YUNV
        YUV422_to_BGR, // CV_YUV2RGB_YVYU = 117,
        YUV422_to_BGR, // CV_YUV2BGR_YVYU = 118,

        YUV422_to_BGR, // CV_YUV2RGBA_YUY2 = 119, CV_YUV2RGBA_YUYV, CV_YUV2RGBA_YUNV
        YUV422_to_BGR, // CV_YUV2BGRA_YUY2 = 120, CV_YUV2BGRA_YUYV, CV_YUV2BGRA_YUNV
        YUV422_to_BGR, // CV_YUV2RGBA_YVYU = 121,
        YUV422_to_BGR, // CV_YUV2BGRA_YVYU = 122,

        YUV422_to_BGR, // CV_YUV2GRAY_UYVY = 123, CV_YUV2GRAY_Y422, CV_YUV2GRAY_UYNV
        YUV422_to_BGR, // CV_YUV2GRAY_YUY2 = 124, CV_YUV2GRAY_YVYU, CV_YUV2GRAY_YUYV, CV_YUV2GRAY_YUNV

        //! alpha premultiplication
        nullptr, //RGBA_to_mBGRA,         // CV_RGBA2mRGBA = 125,
        nullptr, // CV_mRGBA2RGBA = 126,

        //! RGB to YUV 4:2:0 family (three plane YUV)
        BGR_to_YUV420xp, // CV_RGB2YUV_I420  = 127, CV_RGB2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGR2YUV_I420  = 128, CV_BGR2YUV_IYUV

        BGR_to_YUV420xp, // CV_RGBA2YUV_I420 = 129, CV_RGBA2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGRA2YUV_I420 = 130, CV_BGRA2YUV_IYUV
        BGR_to_YUV420xp, // CV_RGB2YUV_YV12  = 131,
        BGR_to_YUV420xp, // CV_BGR2YUV_YV12  = 132,
        BGR_to_YUV420xp, // CV_RGBA2YUV_YV12 = 133,
        BGR_to_YUV420xp, // CV_BGRA2YUV_YV12 = 134,

        //! Edge-Aware Demosaicing
        nullptr, // CV_BayerBG2BGR_EA  = 135,
        nullptr, // CV_BayerGB2BGR_EA  = 136,
        nullptr, // CV_BayerRG2BGR_EA  = 137,
        nullptr, // CV_BayerGR2BGR_EA  = 138,

        nullptr, // OpenCV COLORCVT_MAX = 139

        //! RGB to YUV 4:2:0 family (two plane YUV, not in OpenCV)
        BGR_to_YUV420xp, // CV_RGB2YUV_NV12 = 140,
        BGR_to_YUV420xp, // CV_BGR2YUV_NV12 = 141,
        BGR_to_YUV420xp, // CV_RGB2YUV_NV21 = 142, CV_RGB2YUV420sp
        BGR_to_YUV420xp, // CV_BGR2YUV_NV21 = 143, CV_BGR2YUV420sp

        BGR_to_YUV420xp, // CV_RGBA2YUV_NV12 = 144,
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV12 = 145,
        BGR_to_YUV420xp, // CV_RGBA2YUV_NV21 = 146, CV_RGBA2YUV420sp
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV21 = 147, CV_BGRA2YUV420sp

        nullptr, // CV_COLORCVT_MAX  = 148
    };

    if (code < 0 || static_cast<size_t>(code) >= sizeof(funcs) / sizeof(funcs[0]))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid convert color code: %d", (int)code);
    }

    func_t func = funcs[code];

    if (func == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid convert color code: %d", (int)code);
    }

    func(inData, outData, code, stream);
}

} // namespace varshape

} // namespace

namespace cvcuda::priv {

void CvtColor::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                          NVCVColorConversionCode code) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CvtColor::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    tensor::Infer(*inData, *outData, code, stream);
}

void CvtColor::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                          NVCVColorConversionCode code) const
{
    CVCUDA_NVTX_RANGE("cvcuda::CvtColor::operator()[ImageBatchVarShape]");
    auto inData = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto outData = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    varshape::Infer(*inData, *outData, code, stream);
}

} // namespace cvcuda::priv
