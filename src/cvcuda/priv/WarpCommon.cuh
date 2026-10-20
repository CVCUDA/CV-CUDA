/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

/**
 * @file WarpCommon.cuh
 *
 * @brief Kernels, dispatch and validation shared by the WarpAffine and WarpPerspective operators.
 *
 * WarpAffine and WarpPerspective differ only in how a destination pixel is mapped back to a source
 * coordinate (Transform::calcCoord) and in how the caller-supplied matrix is inverted. Everything
 * downstream of that -- sampling, border handling, the CUBIC fast path, the layout/dtype/channel
 * validation, and the launch geometry -- is identical, so it lives here once and both operators
 * instantiate it on their own transform type.
 */

#ifndef CVCUDA_PRIV_WARP_COMMON_CUH
#define CVCUDA_PRIV_WARP_COMMON_CUH

#include "WarpTransformMatrix.hpp"

#include "PixelLayoutCommon.cuh"
#include "WarpSampleCommon.cuh"

#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/HalfTypes.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/InterpolationVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/InterpolationWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataLayout.hpp>
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
#include <type_traits>
#include <utility>

namespace cvcuda::priv::warp {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

using uchar  = unsigned char;
using ushort = unsigned short;

constexpr int kBlock = 32;

// ---------------------------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------------------------

struct WarpAffineTransform
{
    // The restructured CUBIC sampler pays off for affine maps (the scatter-heavy case); perspective
    // maps keep the wrap path, whose per-tap loads coalesce well and carry no extra registers.
    static constexpr bool kFastCubicSampler = true;

    // PerspectiveTransform is already passed by value, so only affine's specializations were
    // measured to benefit from staging the coefficients in shared memory.
    static constexpr bool kStageCoeffsInSharedMem = true;

    // LINEAR perspective is the only path measured to benefit from amortizing that staging across
    // two coalesced X outputs.
    static constexpr bool kVarShapeLinearX2 = false;

    // The multiply-add structure is written out rather than left to the compiler's contraction
    // choice: at more than one output per thread the compiler may hoist the y-dependent half out
    // of the x loop and contract the remainder differently, which moves the last bit of the
    // coordinate and breaks parity between the single-channel planar and interleaved kernels.
    static __device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y)
    {
        const auto  fx   = static_cast<float>(x);
        const auto  fy   = static_cast<float>(y);
        const float xcoo = __fmaf_rn(fx, c_warpMat[0], fy * c_warpMat[1]) + c_warpMat[2];
        const float ycoo = __fmaf_rn(fx, c_warpMat[3], fy * c_warpMat[4]) + c_warpMat[5];

        return make_float2(xcoo, ycoo);
    }

    float xform[kNumCoeffs]; // NOSONAR: CUDA kernels consume this fixed-size transform storage.
};

struct PerspectiveTransform
{
    static constexpr bool kFastCubicSampler       = false;
    static constexpr bool kStageCoeffsInSharedMem = false;
    static constexpr bool kVarShapeLinearX2       = true;

    explicit PerspectiveTransform(const float *transMatrix)
    {
        std::copy_n(transMatrix, kNumCoeffs, xform);
    }

    // Pinned for the same reason as WarpAffineTransform::calcCoord.
    static __device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y)
    {
        const auto  fx    = static_cast<float>(x);
        const auto  fy    = static_cast<float>(y);
        const float coeff = 1.0f / (__fmaf_rn(fx, c_warpMat[6], fy * c_warpMat[7]) + c_warpMat[8]);

        const float xcoo = coeff * (__fmaf_rn(fx, c_warpMat[0], fy * c_warpMat[1]) + c_warpMat[2]);
        const float ycoo = coeff * (__fmaf_rn(fx, c_warpMat[3], fy * c_warpMat[4]) + c_warpMat[5]);

        return make_float2(xcoo, ycoo);
    }

    float xform[kNumCoeffs]; // NOSONAR: CUDA kernels consume this fixed-size transform storage.
};

// Evaluate each row independently: stepping homogeneous coordinates changes rounding and can
// amplify cancellation near a finite perspective horizon. calcCoord pins the FMA order for all
// layouts and X/Y thread fanouts.
template<int NIY, class Transform, class Body>
__device__ __forceinline__ void WarpRowTile(const float *matrix, int x, int y0, int height, Body &&body)
{
#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int y = y0 + i;
        if (y >= height)
        {
            break;
        }
        body(y, Transform::calcCoord(matrix, x, y));
    }
}

// The CUBIC fast sampler and the dtype/layout classification live in
// WarpSampleCommon.cuh, together with the rationale for restructuring them.

// Whether a warp kernel samples through the restructured CUBIC sampler above rather than the
// InterpolationWrap CUBIC path. Both are bit-exact; which one is faster is a property of the map
// (see WarpAffineTransform::kFastCubicSampler) crossed with the pixel type.
// The pixel type a wrapper yields, with the wrapper's const stripped.
template<class SrcWrapper>
using SrcPixelType = std::remove_cv_t<typename SrcWrapper::ValueType>;

// Affine float3 additionally benefits from restructuring its scalarized tap loads. Keep the
// shared sampler gate unchanged for Rotate. Coverage mode uses InterpolationWrap to apply its blend.
template<class SrcWrapper, class Transform>
constexpr bool                                              kUseFastCubic
    = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC &&Transform::kFastCubicSampler
   && (kCubicFastSampler<SrcPixelType<SrcWrapper>> || std::is_same_v<SrcPixelType<SrcWrapper>, float3>)&&SrcWrapper::
              BorderWrapper::kBorderType
          != NVCV_BORDER_CONSTANT_COVERAGE;

// Scalar F32 affine NEAREST and constant-border perspective LINEAR use single-row Tensor
// scheduling after measured regressions. VarShape retains its independently qualified scheduling.
constexpr int kWarpRowTileHeight = 4;

template<class Transform, typename T, NVCVInterpolationType I, NVCVBorderType B>
constexpr int kTensorRowsPerThread
    = ((std::is_same_v<Transform, WarpAffineTransform> && I == NVCV_INTERP_NEAREST && cuda::NumElements<T> == 1
        && !std::is_same_v<cuda::BaseType<T>, float>)
       || (std::is_same_v<Transform, PerspectiveTransform> && I == NVCV_INTERP_LINEAR && B != NVCV_BORDER_CONSTANT))
        ? kWarpRowTileHeight
        : 1;

template<class Transform, NVCVInterpolationType I>
constexpr int kVarShapeRowsPerThread
    = std::is_same_v<Transform, PerspectiveTransform> &&I == NVCV_INTERP_LINEAR ? kWarpRowTileHeight : 1;

// Everything the launch path needs from validation, in the order the legacy infer() computed it.
struct WarpParams
{
    ElemKind   elem;
    int        channels;
    int        interpolation;
    LayoutKind layout;
};

// The tensor validation both operators ran, in the legacy order: input layout, output layout,
// layout agreement, dtype classification, channels, dtype admission, interpolation, border mode.
// The legacy "Invalid input DataFormat" branch is not reproduced -- the layout conversion in front
// of it already admits exactly the four layouts it listed, so it was unreachable.
inline WarpParams ValidateTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                 const int32_t flags, const NVCVBorderType borderMode)
{
    const LayoutKind inputFormat  = ClassifyLayout(inData.layout());
    const LayoutKind outputFormat = ClassifyLayout(outData.layout());

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be an image-like tensor");
    }

    // Warp resamples, so rows and columns may differ, but the destination wrapper is instantiated
    // from the input's element type and channel count and grid.z comes from the output: a differing
    // dtype, channel count or sample count addresses the destination wrongly.
    if (inData.dtype() != outData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same DataType");
    }
    if (inAccess->numSamples() != outAccess->numSamples()
        || inAccess->infoShape().numChannels() != outAccess->infoShape().numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of samples and channels");
    }

    const ElemKind elem          = ClassifyElem(inData.dtype());
    const int      channels      = inAccess->infoShape().numChannels();
    const int      interpolation = flags & NVCV_INTERP_MAX;

    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (!IsAdmittedElem(elem))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(elem));
    }

    ValidateInterpolation(interpolation);
    ValidateBorderMode(borderMode, /*allowCoverage=*/true);

    return {elem, channels, interpolation, inputFormat};
}

// The var-shape validation common to both operators. The two checks that precede it -- the
// max-batch-size guard and the unique-format guard -- are ordered differently by WarpAffineVarShape
// and WarpPerspectiveVarShape, so each operator runs its own pair before calling this.
inline WarpParams ValidateVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                   const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const int32_t flags,
                                   const NVCVBorderType borderMode)
{
    const LayoutKind inputFormat  = ClassifyLayout(inData);
    const LayoutKind outputFormat = ClassifyLayout(outData);

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    // grid.z is sized from the output batch while the destination wrappers are instantiated from
    // the input's format, so the two batches must agree on both count and format.
    if (inData.numImages() != outData.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images, got %d and %d", inData.numImages(),
                              outData.numImages());
    }
    if (inData.uniqueFormat() != outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same format");
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    const int      interpolation = flags & NVCV_INTERP_MAX;
    const ElemKind elem          = ClassifyElem(inData.uniqueFormat());

    if (!IsAdmittedElem(elem))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(elem));
    }

    ValidateInterpolation(interpolation);
    // The coverage mode needs an interpolation wrap that can report which taps fell inside, and
    // only the tensor wraps have one. Accepting it here would fall through to the generic
    // index-remapping border path, which has no case for it and reads out of bounds.
    ValidateBorderMode(borderMode, /*allowCoverage=*/false);

    return {elem, channels, interpolation, inputFormat};
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

template<typename T>
struct TypeTag
{
    using type = T;
};

template<NVCVBorderType B>
using BorderTag = std::integral_constant<NVCVBorderType, B>;

template<NVCVInterpolationType I>
using InterpTag = std::integral_constant<NVCVInterpolationType, I>;

// Planar dispatch indexes by dtype only: each channel plane is warped as an independent
// single-channel image, so one 1-component specialization per dtype covers any channel count.
// This is the legacy planar_funcs[8] table (and the tensor path's funcs[dtype][0] column), with
// only the five admitted rows populated. Warp samples through the interpolation wrap, which
// accumulates the taps in float and rounds to the element type once on store, so the F16 row
// resolves to real half kernels instead of aliasing the 16-bit integer ones.
template<class F>
inline void DispatchVec1Type(ElemKind elem, F &&fn)
{
    switch (elem)
    {
    case ElemKind::kU8:
        return fn(TypeTag<uchar1>{});
    case ElemKind::kU16:
        return fn(TypeTag<ushort1>{});
    case ElemKind::kS16:
        return fn(TypeTag<short1>{});
    case ElemKind::kF32:
        return fn(TypeTag<float1>{});
    case ElemKind::kF16:
        return fn(TypeTag<half1>{});
    default:
        break;
    }

    NVCV_ASSERT(false);
}

// The legacy 8x4 dtype x channel-count function-pointer table (rows in legacy DataType order,
// columns 1/2/3/4 channels). Only the kCV_8U/16U/16S/32F/16F rows and the 1/3/4-channel columns
// were ever populated; every other cell was null and is unreachable behind the dtype and channel
// gates that run first. The populated cells are exactly the dtype row above crossed with the three
// channel counts, so cuda::MakeType re-derives them rather than spelling all fifteen out.
template<class F>
inline void DispatchPixelType(ElemKind elem, int channels, F &&fn)
{
    DispatchVec1Type(elem,
                     [&](auto tag)
                     {
                         using BT = cuda::BaseType<typename decltype(tag)::type>;

                         switch (channels)
                         {
                         case 1:
                             return fn(TypeTag<cuda::MakeType<BT, 1>>{});
                         case 3:
                             return fn(TypeTag<cuda::MakeType<BT, 3>>{});
                         case 4:
                             return fn(TypeTag<cuda::MakeType<BT, 4>>{});
                         default:
                             break;
                         }

                         NVCV_ASSERT(false);
                     });
}

// The legacy 3x5 interpolation x border function-pointer table, resolved once here into
// compile-time tags instead of once per launch path: the four callers below differ only in which
// dispatcher they name, so spelling the fifteen cells out in each of them made a mis-wired cell
// (a border case naming its neighbour's specialization) invisible on review.
// AllowCoverage is false by default for containers whose wraps have no coverage specialization;
// supported tensor paths opt in explicitly.
template<bool AllowCoverage = false, class F>
inline void DispatchBorderMode(int borderMode, F &&fn)
{
    switch (borderMode)
    {
    case NVCV_BORDER_CONSTANT:
        return fn(BorderTag<NVCV_BORDER_CONSTANT>{});
    case NVCV_BORDER_REPLICATE:
        return fn(BorderTag<NVCV_BORDER_REPLICATE>{});
    case NVCV_BORDER_REFLECT:
        return fn(BorderTag<NVCV_BORDER_REFLECT>{});
    case NVCV_BORDER_WRAP:
        return fn(BorderTag<NVCV_BORDER_WRAP>{});
    case NVCV_BORDER_REFLECT101:
        return fn(BorderTag<NVCV_BORDER_REFLECT101>{});
    case NVCV_BORDER_CONSTANT_COVERAGE:
        if constexpr (AllowCoverage)
        {
            return fn(BorderTag<NVCV_BORDER_CONSTANT_COVERAGE>{});
        }
        break;
    default:
        break;
    }

    NVCV_ASSERT(false);
}

template<bool AllowCoverage = false, class F>
inline void DispatchInterpBorder(int interpolation, int borderMode, F &&fn)
{
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        return DispatchBorderMode<AllowCoverage>(borderMode, [&](auto b) { fn(InterpTag<NVCV_INTERP_NEAREST>{}, b); });
    case NVCV_INTERP_LINEAR:
        return DispatchBorderMode<AllowCoverage>(borderMode, [&](auto b) { fn(InterpTag<NVCV_INTERP_LINEAR>{}, b); });
    case NVCV_INTERP_CUBIC:
        return DispatchBorderMode<AllowCoverage>(borderMode, [&](auto b) { fn(InterpTag<NVCV_INTERP_CUBIC>{}, b); });
    default:
        break;
    }

    NVCV_ASSERT(false);
}

// The warp kernels index their tensors with int32 strides, so a batch whose sample stride times
// batch size overflows int32 has to be rejected before any launch. Both tensor dispatchers run
// this check at the same point in their prologue.
inline void CheckTensorFitsInt32(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                                 const nvcv::TensorDataAccessStridedImagePlanar &outAccess, int batchSize)
{
    const int64_t srcMaxStride = inAccess.sampleStride() * batchSize;
    const int64_t dstMaxStride = outAccess.sampleStride() * batchSize;

    if (std::max(srcMaxStride, dstMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }
}

// A 32x8 block, one grid layer per sample; xUnroll and yUnroll count outputs per thread.
inline std::pair<dim3, dim3> WarpLaunchConfig(int dstWidth, int dstHeight, int batchSize, int xUnroll, int yUnroll = 1)
{
    const dim3 block(kBlock, kBlock / 4);
    const dim3 grid(util::DivUp(dstWidth, static_cast<int>(block.x) * xUnroll),
                    util::DivUp(dstHeight, static_cast<int>(block.y) * yUnroll), batchSize);
    return {block, grid};
}

// X outputs per thread for the tensor warp kernel. One thread emitting several outputs amortizes
// the per-block prologue -- the coefficient staging and its barrier, the block index arithmetic
// and the bounds setup. The outputs are strided by blockDim.x so every store stays one coalesced
// transaction per warp. CUBIC is the exception and gets one output per thread: four outputs each
// reading a 4x4 neighbourhood is where the register budget runs out, and the A100 reference run
// regressed warpaffine half3 CUBIC by 10.2% (5736 -> 6321 us). Its fused CUBIC/u8 planar path
// emits all planes together and deliberately remains one output per thread; the other planar
// cases dispatch one plane at a time through this policy.
template<NVCVInterpolationType I>
constexpr int TensorWarpXUnroll()
{
    return I == NVCV_INTERP_CUBIC ? 1 : 4;
}

// ---------------------------------------------------------------------------------------------
// Tensor path
// ---------------------------------------------------------------------------------------------

namespace tensor {

template<class BorderWrapper, bool Inside>
struct UniformConstantBorder : BorderWrapper
{
    explicit __host__ __device__ UniformConstantBorder(const BorderWrapper &border)
        : BorderWrapper(border)
    {
    }

    // All four LINEAR taps have the same validity; preserve the interpolation arithmetic.
    inline __host__ __device__ auto value(int3 c) const
    {
        if constexpr (Inside)
            return *this->tensorWrap().ptr(c.z, c.y, c.x);
        else
            return this->borderValue();
    }
};

template<class Transform, class SrcWrapper, class DstWrapper>
__device__ __forceinline__ void WarpOutput(const SrcWrapper &src, DstWrapper &dst, const float *xform, int3 dstCoord,
                                           int2 srcSize)
{
    const float2 coord = Transform::calcCoord(xform, dstCoord.x, dstCoord.y);

    if constexpr (kUseFastCubic<SrcWrapper, Transform>)
    {
        dst[dstCoord] = CubicSampleTensor(src, dstCoord.z, coord, srcSize);
    }
    else if constexpr (SrcWrapper::kInterpolationType == NVCV_INTERP_LINEAR
                       && SrcWrapper::BorderWrapper::kBorderType == NVCV_BORDER_CONSTANT
                       && std::is_same_v<Transform, WarpAffineTransform>)
    {
        const float3 c{coord.x, coord.y, static_cast<float>(dstCoord.z)};
        if (coord.x >= 0 && coord.y >= 0 && coord.x < srcSize.x - 1 && coord.y < srcSize.y - 1)
        {
            using Border = UniformConstantBorder<typename SrcWrapper::BorderWrapper, true>;
            cuda::InterpolationWrap<Border, NVCV_INTERP_LINEAR, true> interior{Border{src.borderWrap()}};
            dst[dstCoord] = interior[c];
        }
        else
        {
            const int x = cuda::GetIndexForInterpolation<NVCV_INTERP_LINEAR, 1, int>(coord.x);
            const int y = cuda::GetIndexForInterpolation<NVCV_INTERP_LINEAR, 1, int>(coord.y);
            if (isfinite(coord.x) && isfinite(coord.y) && (x < -1 || y < -1 || x >= srcSize.x || y >= srcSize.y))
            {
                using Border = UniformConstantBorder<typename SrcWrapper::BorderWrapper, false>;
                cuda::InterpolationWrap<Border, NVCV_INTERP_LINEAR, true> exterior{Border{src.borderWrap()}};
                dst[dstCoord] = exterior[c];
            }
            else
            {
                dst[dstCoord] = src[c];
            }
        }
    }
    else
    {
        dst[dstCoord] = src[float3{coord.x, coord.y, static_cast<float>(dstCoord.z)}];
    }
}

template<bool UseSharedTransform, class Transform>
__device__ __forceinline__ const float *StageWarpTransform(const Transform &transform)
{
    if constexpr (UseSharedTransform)
    {
        extern __shared__ float coeff[];

        const int lid = threadIdx.y * blockDim.x + threadIdx.x;
        if (lid < kNumCoeffs)
        {
            coeff[lid] = transform.xform[lid];
        }

        __syncthreads();
        return coeff;
    }
    else
    {
        return transform.xform;
    }
}

// `NIX` is the number of X outputs one thread emits, strided by blockDim.x rather than packed next
// to each other, so each of the NIX stores is still one coalesced transaction per warp.
// `calcCoord` is still evaluated per output rather than stepped incrementally, so the source
// coordinate of a given destination pixel is the same expression at every NIX and the result is
// bit-exact.
template<bool UseSharedTransform, int NIX, int NIY, class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize, Transform transform)
{
    if constexpr (NIY > 1)
    {
        const int    dstX0 = static_cast<int>(blockIdx.x * blockDim.x * NIX + threadIdx.x);
        const int    y0    = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) * NIY;
        const int    batch = static_cast<int>(blockIdx.z);
        const float *xform = StageWarpTransform<UseSharedTransform>(transform);

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int x = dstX0 + i * static_cast<int>(blockDim.x);
            if (x < dstSize.x)
            {
                WarpRowTile<NIY, Transform>(
                    xform, x, y0, dstSize.y,
                    [&](int y, float2 coord) {
                        dst[int3{x, y, batch}] = src[float3{coord.x, coord.y, static_cast<float>(batch)}];
                    });
            }
        }
    }
    else if constexpr (NIX == 1)
    {
        // One output per thread keeps the vector index form rather than sharing the unrolled
        // path's scalar one: ptxas recognizes `blockDim * blockIdx + threadIdx` as block-uniform
        // and reads CTAID through uniform registers (S2UR), where the scalar spelling falls back
        // to per-thread S2R and costs two registers. Measured on sm_90 that took the uchar1 CUBIC
        // kernels (REPLICATE, REFLECT, WRAP) from REG 32 to 34 and the SM from 8 resident blocks
        // to 6. The unrolled path below cannot use this form: its x origin strides by
        // blockDim.x * NIX.
        const int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

        const float *xform = StageWarpTransform<UseSharedTransform>(transform);

        if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
        {
            WarpOutput<Transform>(src, dst, xform, dstCoord, srcSize);
        }
    }
    else
    {
        const int dstX0 = static_cast<int>(blockIdx.x * blockDim.x * NIX + threadIdx.x);
        const int dstY  = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int batch = static_cast<int>(blockIdx.z);

        const float *xform = StageWarpTransform<UseSharedTransform>(transform);

        if (dstY >= dstSize.y)
        {
            return;
        }

#pragma unroll
        for (int i = 0; i < NIX; ++i)
        {
            const int3 dstCoord{dstX0 + i * static_cast<int>(blockDim.x), dstY, batch};

            if (dstCoord.x < dstSize.x)
            {
                WarpOutput<Transform>(src, dst, xform, dstCoord, srcSize);
            }
        }
    }
}

// Fused planar (NCHW/CHW) CUBIC warp: one launch covers every channel plane, sharing the
// transformed coordinate, cubic weights, and border resolution across planes instead of
// relaunching the single-channel kernel per plane. src/dst wrap plane 0; the other planes are
// addressed by the plane byte stride.
template<bool UseSharedTransform, class Transform, typename BT, NVCVBorderType B, int NP, class SrcWrapper,
         class DstWrapper>
__global__ void warp_planar_fused(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize, int64_t srcPlaneStride,
                                  int64_t dstPlaneStride, float4 borderValue, Transform transform)
{
    const int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const float *xform = StageWarpTransform<UseSharedTransform>(transform);

    if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        const float2 coord = Transform::calcCoord(xform, dstCoord.x, dstCoord.y);

        CubicWarpPlanes<BT, B, NP, int32_t>(
            [&](int p, int yy) {
                return reinterpret_cast<const BT *>(reinterpret_cast<const char *>(src.ptr(dstCoord.z, yy))
                                                    + p * srcPlaneStride);
            },
            [&](int p, BT v)
            {
                *(reinterpret_cast<BT *>(reinterpret_cast<char *>(dst.ptr(dstCoord.z, dstCoord.y)) + p * dstPlaneStride)
                  + dstCoord.x)
                    = v;
            },
            borderValue, srcSize, coord);
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpDispatcher
{
    static void call(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                     Transform transform, const float4 &borderValue, cudaStream_t stream)
    {
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);

        const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
        const int  batchSize{static_cast<int>(outAccess->numSamples())};

        constexpr int kNIX = TensorWarpXUnroll<I>();
        constexpr int kNIY = kTensorRowsPerThread<Transform, T, I, B>;

        const auto [block, grid] = WarpLaunchConfig(dstSize.x, dstSize.y, batchSize, kNIX, kNIY);

        auto bVal = WrapBorderValue<T, B>(borderValue);

        CheckTensorFitsInt32(*inAccess, *outAccess, batchSize);

        auto src = cuda::CreateInterpolationWrapNHW<const T, B, I, int32_t, true>(inData, bVal);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

        const int2 srcSize{inAccess->numCols(), inAccess->numRows()};

        constexpr bool useSharedTransform
            = Transform::kStageCoeffsInSharedMem && (cuda::NumElements<T> == 3 || I == NVCV_INTERP_CUBIC);
        constexpr int smemSize = useSharedTransform ? kNumCoeffs * sizeof(float) : 0;
        warp<useSharedTransform, kNIX, kNIY, Transform>
            <<<grid, block, smemSize, stream>>>(src, dst, dstSize, srcSize, transform);

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<class Transform, typename T>
inline void warp_caller(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        Transform transform, const int interpolation, const int borderMode, const float4 &borderValue,
                        cudaStream_t stream)
{
    DispatchInterpBorder<true>(interpolation, borderMode,
                               [&](auto interp, auto border)
                               {
                                   WarpDispatcher<Transform, T, decltype(border)::value, decltype(interp)::value>::call(
                                       inData, outData, transform, borderValue, stream);
                               });
}

// Build a single-channel (N, H, W, 1) NHWC view of channel plane `plane` of a packed planar
// (NCHW/CHW) tensor. Warp samples each channel at the same transformed coordinate, so plane (n, c)
// is an independent single-channel image: viewing one plane across all N samples (sample stride
// unchanged, base offset by plane*chStride) lets the existing single-channel warp kernel handle it,
// bit-exact with the equivalent NHWC single-channel warp. One such view is processed per plane.
inline nvcv::TensorDataStridedCuda PlanarChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                     const nvcv::TensorDataAccessStridedImagePlanar &access, int plane)
{
    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr()) + plane * access.chStride();
    buf.strides[0] = access.sampleStride(); // N
    buf.strides[1] = access.rowStride();    // H
    buf.strides[2] = access.colStride();    // W
    buf.strides[3] = access.colStride();    // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{access.numSamples(), access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(), buf
    };
}

template<class Transform, typename BT, NVCVBorderType B>
struct WarpPlanarFusedDispatcher
{
    static void call(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                     Transform transform, const float4 &borderValue, int numPlanes, cudaStream_t stream)
    {
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
        const int2 srcSize{inAccess->numCols(), inAccess->numRows()};
        const int  batchSize{static_cast<int>(outAccess->numSamples())};

        const auto [block, grid] = WarpLaunchConfig(dstSize.x, dstSize.y, batchSize, 1);

        CheckTensorFitsInt32(*inAccess, *outAccess, batchSize);

        auto planeIn  = PlanarChannelView(inData, *inAccess, 0);
        auto planeOut = PlanarChannelView(outData, *outAccess, 0);

        auto src = cuda::CreateTensorWrapNHW<const BT, int32_t>(planeIn);
        auto dst = cuda::CreateTensorWrapNHW<BT, int32_t>(planeOut);

        constexpr bool useSharedTransform = Transform::kStageCoeffsInSharedMem;
        constexpr int  smemSize           = useSharedTransform ? kNumCoeffs * sizeof(float) : 0;

        switch (numPlanes)
        {
        case 1:
            warp_planar_fused<useSharedTransform, Transform, BT, B, 1><<<grid, block, smemSize, stream>>>(
                src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            break;
        case 3:
            if constexpr (!std::is_same_v<Transform, WarpAffineTransform> || !std::is_same_v<BT, float>)
            {
                warp_planar_fused<useSharedTransform, Transform, BT, B, 3><<<grid, block, smemSize, stream>>>(
                    src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            }
            break;
        case 4:
            warp_planar_fused<useSharedTransform, Transform, BT, B, 4><<<grid, block, smemSize, stream>>>(
                src, dst, dstSize, srcSize, inAccess->chStride(), outAccess->chStride(), borderValue, transform);
            break;
        default:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid planar channel number %d", numPlanes);
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<class Transform, typename BT>
inline void warpPlanarFusedCubic(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                 Transform transform, const int borderMode, const float4 &borderValue, int numPlanes,
                                 cudaStream_t stream)
{
    DispatchBorderMode(borderMode,
                       [&](auto border)
                       {
                           WarpPlanarFusedDispatcher<Transform, BT, decltype(border)::value>::call(
                               inData, outData, transform, borderValue, numPlanes, stream);
                       });
}

} // namespace tensor

// Run one tensor warp. `transform` already holds the destination-to-source map.
template<class Transform>
inline void RunWarpTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                          Transform transform, const WarpParams &params, const int borderMode,
                          const float4 &borderValue, cudaStream_t stream)
{
    if (IsPlanar(params.layout))
    {
        // Every plane samples the same coordinate, so fusing avoids repeating shared work.
        // Coverage mode uses the per-plane InterpolationWrap path to apply its coverage blend.
        // F32 C3 affine fusion regressed on Orin; keep its established per-plane path.
        if (params.interpolation == NVCV_INTERP_CUBIC && (params.elem == ElemKind::kU8 || params.elem == ElemKind::kF32)
            && !(std::is_same_v<Transform, WarpAffineTransform> && params.elem == ElemKind::kF32
                 && params.channels == 3)
            && borderMode != NVCV_BORDER_CONSTANT_COVERAGE)
        {
            if (params.elem == ElemKind::kU8)
            {
                tensor::warpPlanarFusedCubic<Transform, uchar>(inData, outData, transform, borderMode, borderValue,
                                                               params.channels, stream);
            }
            else
            {
                tensor::warpPlanarFusedCubic<Transform, float>(inData, outData, transform, borderMode, borderValue,
                                                               params.channels, stream);
            }
            return;
        }

        // Warp each channel plane as an independent single-channel image (same transform,
        // per-channel border value), reusing the interleaved single-channel kernel (the legacy
        // table's column 0). This is bit-exact with warping the equivalent NHWC single-channel
        // data. The planar layout carries a per-channel border value, so each plane is given its
        // own.
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
        NVCV_ASSERT(inAccess);

        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
        NVCV_ASSERT(outAccess);

        DispatchVec1Type(params.elem,
                         [&](auto tag)
                         {
                             using T = typename decltype(tag)::type;
                             for (int c = 0; c < params.channels; ++c)
                             {
                                 auto planeIn  = tensor::PlanarChannelView(inData, *inAccess, c);
                                 auto planeOut = tensor::PlanarChannelView(outData, *outAccess, c);
                                 tensor::warp_caller<Transform, T>(planeIn, planeOut, transform, params.interpolation,
                                                                   borderMode, PlanarBorderValue(borderValue, c),
                                                                   stream);
                             }
                         });
        return;
    }

    DispatchPixelType(params.elem, params.channels,
                      [&](auto tag)
                      {
                          using T = typename decltype(tag)::type;
                          tensor::warp_caller<Transform, T>(inData, outData, transform, params.interpolation,
                                                            borderMode, borderValue, stream);
                      });
}

// ---------------------------------------------------------------------------------------------
// Var-shape path
// ---------------------------------------------------------------------------------------------

namespace varshape {

template<class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs)
{
    int3      dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < kNumCoeffs)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    if (dstCoord.x < dst.width(dstCoord.z) && dstCoord.y < dst.height(dstCoord.z))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        if constexpr (kUseFastCubic<SrcWrapper, Transform>)
        {
            dst[dstCoord] = CubicSampleVarShape(src, dstCoord.z, coord);
        }
        else
        {
            dst[dstCoord] = src[float3{coord.x, coord.y, static_cast<float>(dstCoord.z)}];
        }
    }
}

// LINEAR Perspective benefits from amortizing per-block coefficient staging across two coalesced X outputs. Keep a
// separate kernel so WarpAffine and the other interpolation modes retain their qualified code generation.
template<int NIX, int NIY, class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp_x2(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs)
{
    const int dstX0 = blockIdx.x * blockDim.x * NIX + threadIdx.x;
    const int dstY0 = (blockIdx.y * blockDim.y + threadIdx.y) * NIY;
    const int batch = blockIdx.z;
    const int lid   = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < kNumCoeffs)
    {
        coeff[lid] = *coeffs.ptr(batch, lid);
    }

    __syncthreads();

    const int dstWidth  = dst.width(batch);
    const int dstHeight = dst.height(batch);

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dstX = dstX0 + i * static_cast<int>(blockDim.x);
        if (dstX < dstWidth)
        {
            WarpRowTile<NIY, Transform>(
                coeff, dstX, dstY0, dstHeight,
                [&](int y, float2 coord) {
                    dst[int3{dstX, y, batch}] = src[float3{coord.x, coord.y, static_cast<float>(batch)}];
                });
        }
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpDispatcher
{
    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                     const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                     const cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, cudaStream_t stream)
    {
        nvcv::Size2D outMaxSize = outData.maxSize();

        constexpr bool useX2 = Transform::kVarShapeLinearX2 && I == NVCV_INTERP_LINEAR;
        constexpr int  kNIX  = useX2 ? 2 : 1;
        constexpr int  kNIY  = useX2 ? kVarShapeRowsPerThread<Transform, I> : 1;

        const auto [block, grid] = WarpLaunchConfig(outMaxSize.w, outMaxSize.h, outData.numImages(), kNIX, kNIY);

        auto bVal = WrapBorderValue<T, B>(borderValue);

        cuda::InterpolationVarShapeWrap<const T, B, I, true> src(inData, bVal);
        cuda::ImageBatchVarShapeWrap<T>                      dst(outData);

        size_t smem_size = kNumCoeffs * sizeof(float);

        if constexpr (useX2)
        {
            warp_x2<kNIX, kNIY, Transform><<<grid, block, smem_size, stream>>>(src, dst, transform);
        }
        else
        {
            warp<Transform><<<grid, block, smem_size, stream>>>(src, dst, transform);
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<class Transform, typename T>
inline void warp_caller(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                        const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                        cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation, const int borderMode,
                        const float4 &borderValue, cudaStream_t stream)
{
    DispatchInterpBorder</*AllowCoverage=*/false>(
        interpolation, borderMode,
        [&](auto interp, auto border)
        {
            WarpDispatcher<Transform, T, decltype(border)::value, decltype(interp)::value>::call(
                inData, outData, transform, borderValue, stream);
        });
}

// Fused planar var-shape CUBIC warp: one launch covers every channel plane, sharing the
// transformed coordinate, cubic weights, and border resolution across planes instead of
// relaunching the single-plane kernel per plane.
template<class Transform, typename BT, NVCVBorderType B, int NP, class SrcWrapper, class DstWrapper>
__global__ void warp_planar_fused(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs,
                                  float4 borderValue)
{
    const int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int  lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < kNumCoeffs)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    const int z = dstCoord.z;
    if (dstCoord.x < dst.width(z) && dstCoord.y < dst.height(z))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        CubicWarpPlanes<BT, B, NP, int>(
            [&](int p, int yy) { return reinterpret_cast<const BT *>(src.ptr(z, p, yy, 0)); },
            [&](int p, BT v) { *reinterpret_cast<BT *>(dst.ptr(z, p, dstCoord.y, dstCoord.x)) = v; }, borderValue,
            int2{src.width(z), src.height(z)}, coord);
    }
}

// Planar (NCHW/CHW) var-shape warp. Warp samples each channel at the same transformed coordinate,
// so each plane is an independent single-channel image: this kernel warps one plane (selected by
// the plane-indexed (sample, plane, y, x) coordinate), reusing the interpolation/border wrap. One
// launch per plane lets each plane carry its own constant-border component. Bit-exact with the
// interleaved single-channel warp.
template<int NIY, class Transform, class SrcWrapper, class DstWrapper>
__global__ void warp_planar(SrcWrapper src, DstWrapper dst, const cuda::Tensor2DWrap<float, int32_t> coeffs, int plane)
{
    int3      dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);
    const int lid      = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float coeff[];

    if (lid < kNumCoeffs)
    {
        coeff[lid] = *coeffs.ptr(dstCoord.z, lid);
    }

    __syncthreads();

    const int batch = dstCoord.z;

    if constexpr (NIY > 1)
    {
        if (dstCoord.x >= dst.width(batch))
        {
            return;
        }

        const int y0 = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) * NIY;

        WarpRowTile<NIY, Transform>(
            coeff, dstCoord.x, y0, dst.height(batch),
            [&](int y, float2 coord)
            {
                dst[int4{dstCoord.x, y, plane, batch}]
                    = src[float4{coord.x, coord.y, static_cast<float>(plane), static_cast<float>(batch)}];
            });

        return;
    }
    else if (dstCoord.x < dst.width(batch) && dstCoord.y < dst.height(batch))
    {
        const float2 coord = Transform::calcCoord(coeff, dstCoord.x, dstCoord.y);

        if constexpr (kUseFastCubic<SrcWrapper, Transform>)
        {
            dst[int4{dstCoord.x, dstCoord.y, plane, batch}] = CubicSampleVarShapePlane(src, batch, plane, coord);
        }
        else
        {
            dst[int4{dstCoord.x, dstCoord.y, plane, batch}]
                = src[float4{coord.x, coord.y, static_cast<float>(plane), static_cast<float>(batch)}];
        }
    }
}

template<class Transform, typename T, NVCVBorderType B, NVCVInterpolationType I>
struct WarpPlanarDispatcher
{
    static void call(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                     const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                     const cuda::Tensor2DWrap<float, int32_t> transform, const float4 &borderValue, int channels,
                     cudaStream_t stream)
    {
        nvcv::Size2D outMaxSize = outData.maxSize();

        constexpr int kNIY = kVarShapeRowsPerThread<Transform, I>;

        const auto [block, grid] = WarpLaunchConfig(outMaxSize.w, outMaxSize.h, outData.numImages(), 1, kNIY);

        const size_t smem_size = kNumCoeffs * sizeof(float);

        cuda::ImageBatchVarShapeWrap<T> dst(outData);

        // Reuse coordinates, weights and border resolution across byte and float planes.
        if constexpr (I == NVCV_INTERP_CUBIC
                      && (sizeof(cuda::BaseType<T>) == 1 || std::is_same_v<cuda::BaseType<T>, float>))
        {
            cuda::ImageBatchVarShapeWrap<const T> src(inData);

            // VarShape is classified as planar only when there are multiple planes.
            switch (channels)
            {
            case 3:
                warp_planar_fused<Transform, cuda::BaseType<T>, B, 3>
                    <<<grid, block, smem_size, stream>>>(src, dst, transform, borderValue);
                break;
            case 4:
                warp_planar_fused<Transform, cuda::BaseType<T>, B, 4>
                    <<<grid, block, smem_size, stream>>>(src, dst, transform, borderValue);
                break;
            default:
                break;
            }
        }
        else
        {
            for (int c = 0; c < channels; ++c)
            {
                auto bVal = WrapBorderValue<T, B>(PlanarBorderValue(borderValue, c));

                cuda::InterpolationVarShapeWrap<const T, B, I, true> src(inData, bVal);

                warp_planar<kNIY, Transform><<<grid, block, smem_size, stream>>>(src, dst, transform, c);
            }
        }

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

template<class Transform, typename T>
inline void warp_planar_caller(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                               cuda::Tensor2DWrap<float, int32_t> transform, const int interpolation,
                               const int borderMode, int channels, const float4 &borderValue, cudaStream_t stream)
{
    DispatchInterpBorder</*AllowCoverage=*/false>(
        interpolation, borderMode,
        [&](auto interp, auto border)
        {
            WarpPlanarDispatcher<Transform, T, decltype(border)::value, decltype(interp)::value>::call(
                inData, outData, transform, borderValue, channels, stream);
        });
}

} // namespace varshape

// Run one var-shape warp. `transform` wraps the per-image coefficient block the operator staged.
template<class Transform>
inline void RunWarpVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData,
                            cuda::Tensor2DWrap<float, int32_t> transform, const WarpParams &params,
                            const int borderMode, const float4 &borderValue, cudaStream_t stream)
{
    if (IsPlanar(params.layout))
    {
        // Planar dispatch indexes by dtype only: each channel is warped as a single-channel plane
        // (per-channel border value handled inside the planar caller).
        DispatchVec1Type(params.elem,
                         [&](auto tag)
                         {
                             using T = typename decltype(tag)::type;
                             varshape::warp_planar_caller<Transform, T>(inData, outData, transform,
                                                                        params.interpolation, borderMode,
                                                                        params.channels, borderValue, stream);
                         });
        return;
    }

    DispatchPixelType(params.elem, params.channels,
                      [&](auto tag)
                      {
                          using T = typename decltype(tag)::type;
                          varshape::warp_caller<Transform, T>(inData, outData, transform, params.interpolation,
                                                              borderMode, borderValue, stream);
                      });
}

} // namespace cvcuda::priv::warp

#endif // CVCUDA_PRIV_WARP_COMMON_CUH
