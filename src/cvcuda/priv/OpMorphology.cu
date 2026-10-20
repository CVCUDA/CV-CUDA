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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpMorphology.hpp"
#include "PlanarTensorView.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"
#include "SameShapeCommon.cuh"

#include <cuda_fp16.h>
#include <cvcuda/Types.h>
#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/BorderWrap.hpp>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageData.hpp>
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
#include <optional>
#include <type_traits>
#include <utility>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using cvcuda::priv::same_shape::GetCoordForLayout;

using uchar  = unsigned char;
using ushort = unsigned short;

// The widest planar var-shape launch carries one (image, plane) pair per CUDA grid-z entry, so the
// whole planar path is admitted on that bound -- including the 8-bit kernels, which fold the planes
// into a per-thread loop and would fit a looser one.
constexpr int64_t kMaxGridZ = 65535;

using cvcuda::priv::legacy_format::ClassifyDataFormat;
using cvcuda::priv::legacy_format::ClassifyDataType;
using cvcuda::priv::legacy_format::DataFormatCode;
using cvcuda::priv::legacy_format::DataTypeCode;
using cvcuda::priv::legacy_format::IsPlanar;

// The exact set the operator admits: (unsigned, 8), (unsigned, 16), (float, 16) and (float, 32).
inline bool IsSupportedDataType(DataTypeCode code)
{
    return code == DataTypeCode::kU8 || code == DataTypeCode::kU16 || code == DataTypeCode::kF32
        || code == DataTypeCode::kF16;
}

// -------------------------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------------------------

enum class MorphKind
{
    kErode,
    kDilate,
};

// Identity element of the running min/max, and the value a constant border contributes: dilate
// starts from the lowest representable value, erode from the highest. cuda::Lowest is unavailable
// for __half (not a literal type, so TypeTraits has no constexpr members for it), so half uses the
// function-form limits instead.
template<MorphKind K, typename BT>
inline __host__ __device__ BT MorphIdentityValue()
{
    if constexpr (std::is_same_v<BT, __half>)
    {
        return (K == MorphKind::kDilate) ? cuda::HalfLowest() : cuda::HalfMax();
    }
    else
    {
        // OpMorphology.dilate_f32_all_negative_neighborhood
        return (K == MorphKind::kDilate) ? cuda::Lowest<BT> : cuda::TypeTraits<BT>::max;
    }
}

template<MorphKind K, class T>
__device__ __forceinline__ T MorphCombine(T current, T next)
{
    if constexpr (K == MorphKind::kDilate)
    {
        return cuda::max(current, next);
    }
    else
    {
        return cuda::min(current, next);
    }
}

// The window served by the unrolled reads and selected by the specialized dispatch.
inline __host__ __device__ bool IsCentered3x3(int2 kernelSize, int2 anchor)
{
    return kernelSize.x == 3 && kernelSize.y == 3 && anchor.x == 1 && anchor.y == 1;
}

__device__ __forceinline__ bool IsMorphInterior(int2 kernelSize, int2 anchor, int x, int y, int2 size)
{
    // Use bitwise & (no short-circuit) so the compiler accumulates all four
    // conditions into a single predicate register with chained ISETPs, matching
    // the pre-migration dilate<>/erode<> pattern.  Short-circuit && causes the
    // compiler to emit two intermediate divergent BRA instructions, which in
    // turn forces a nested BSSY B2 scope (+2 BSSY, +2 BSYNC, +3 BREAK) and a
    // WARPSYNC after the outer BSYNC — an extra ~8 instructions on sm_80.
    return (x >= anchor.x) & (y >= anchor.y) & (x + kernelSize.x - anchor.x <= size.x)
         & (y + kernelSize.y - anchor.y <= size.y);
}

// The general case: every tap goes through the border wrapper, which resolves out-of-image
// coordinates per the border mode baked into its type.
template<MorphKind K, bool IsPlanar, class PT, class SrcWrapper>
__device__ __forceinline__ PT MorphBorderWindow(const SrcWrapper &src, int batch, int channel, int y, int x,
                                                int2 kernelSize, int2 anchor, PT res)
{
    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            const auto coord = GetCoordForLayout<IsPlanar>(int3{x - anchor.x + j, y - anchor.y + i, batch}, channel);
            if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                res = MorphCombine<K>(res, src.value(coord));
            else
                res = MorphCombine<K>(res, src[coord]);
        }
    }
    return res;
}

// One output element, folding the whole mask window into `res`.
//
// Three read paths, in the order the legacy kernels tried them: a fully unrolled 9-tap bounds-free
// read for a centered 3x3 mask away from the image edge, then -- only where the caller asked for it
// -- a bounds-free read of an arbitrary window that happens to lie entirely inside, and finally the
// border-wrapped general case. The first two skip the per-tap border arithmetic; they are chosen so
// that every tap they take is in range, so all three agree on the result.
template<MorphKind K, bool IsPlanar, bool UseGenericInterior, class PT, class SrcWrapper, class RawSrcWrapper>
__device__ __forceinline__ PT MorphWindow(const SrcWrapper &src, const RawSrcWrapper &rawSrc, int batch, int channel,
                                          int y, int x, int2 size, int2 kernelSize, int2 anchor, PT res)
{
    if (IsCentered3x3(kernelSize, anchor) && x > 0 && y > 0 && x + 1 < size.x && y + 1 < size.y)
    {
#pragma unroll
        for (int i = -1; i <= 1; ++i)
        {
#pragma unroll
            for (int j = -1; j <= 1; ++j)
            {
                res = MorphCombine<K>(res, rawSrc[GetCoordForLayout<IsPlanar>(int3{x + j, y + i, batch}, channel)]);
            }
        }
    }
    // Single-exit structure: no early returns keep the compiler from inserting extra
    // BSSY/BSYNC/BREAK warp-convergence barriers.  The real guard against those
    // barriers is IsMorphInterior's use of bitwise & instead of short-circuit &&
    // (see the comment there); this if-else form is kept for clarity.
    else if constexpr (UseGenericInterior)
    {
        if (IsMorphInterior(kernelSize, anchor, x, y, size))
        {
            const int srcY0 = y - anchor.y;
            const int srcX0 = x - anchor.x;

            for (int i = 0; i < kernelSize.y; ++i)
            {
                for (int j = 0; j < kernelSize.x; ++j)
                {
                    res = MorphCombine<K>(
                        res, rawSrc[GetCoordForLayout<IsPlanar>(int3{srcX0 + j, srcY0 + i, batch}, channel)]);
                }
            }
        }
        else
        {
            res = MorphBorderWindow<K, IsPlanar>(src, batch, channel, y, x, kernelSize, anchor, res);
        }
    }
    else
    {
        res = MorphBorderWindow<K, IsPlanar>(src, batch, channel, y, x, kernelSize, anchor, res);
    }
    return res;
}

// Where a kernel reads its output extent and its mask geometry. The tensor path has one of each for
// the whole batch, resolved on the host; the var-shape path reads the extent from the destination
// batch and the mask/anchor pair from the caller's per-image tensors.
struct UniformMorphParams
{
    int2 dstSize;
    int2 kernelSize;
    int2 anchor;

    template<bool IsPlanar, class DstWrapper>
    __device__ __forceinline__ int2 extent(const DstWrapper &, int, int) const
    {
        return dstSize;
    }

    __device__ __forceinline__ int2 mask(int) const
    {
        return kernelSize;
    }

    __device__ __forceinline__ int2 anchorAt(int) const
    {
        return anchor;
    }
};

struct VarShapeMorphParams
{
    cuda::Tensor1DWrap<int2> kernelSizes;
    cuda::Tensor1DWrap<int2> anchors;

    template<bool IsPlanar, class DstWrapper>
    __device__ __forceinline__ int2 extent(const DstWrapper &dst, int batch, int channel) const
    {
        if constexpr (IsPlanar)
        {
            return int2{dst.width(batch, channel), dst.height(batch, channel)};
        }
        else
        {
            return int2{dst.width(batch), dst.height(batch)};
        }
    }

    __device__ __forceinline__ int2 mask(int batch) const
    {
        return kernelSizes[batch];
    }

    __device__ __forceinline__ int2 anchorAt(int batch) const
    {
        return anchors[batch];
    }
};

// One thread per output element. `channels` is read only on the planar instantiations, where a
// grid-z entry is an (image, plane) pair rather than an image index and has to be split back.
template<MorphKind K, bool IsPlanar, bool UseGenericInterior, class Params, class SrcWrapper, class RawSrcWrapper,
         class DstWrapper, class PT = typename DstWrapper::ValueType>
__global__ void MorphKernel(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst, Params params,
                            int channels, cuda::BaseType<PT> identity)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int channel = 0;
    int batch   = blockIdx.z;
    if constexpr (IsPlanar)
    {
        channel = blockIdx.z % channels;
        batch   = blockIdx.z / channels;
    }

    const int2 size = params.template extent<IsPlanar>(dst, batch, channel);

    if (x >= size.x || y >= size.y)
        return;

    PT res = MorphWindow<K, IsPlanar, UseGenericInterior>(src, rawSrc, batch, channel, y, x, size, params.mask(batch),
                                                          params.anchorAt(batch), cuda::SetAll<PT>(identity));

    dst[GetCoordForLayout<IsPlanar>(int3{x, y, batch}, channel)] = cuda::SaturateCast<cuda::BaseType<PT>>(res);
}

// 8-bit planar variant that walks the channels inside one thread instead of spreading them over
// grid-z: at one byte per element the per-thread setup dominates, so amortizing it over the planes
// beats the extra parallelism.
template<MorphKind K, bool UseGenericInterior, class SrcWrapper, class RawSrcWrapper, class DstWrapper,
         class PT = typename DstWrapper::ValueType>
__global__ void MorphPlanarChannelsKernel(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst,
                                          VarShapeMorphParams params, int channels, cuda::BaseType<PT> identity)
{
    const int x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    const int2 kernelSize = params.mask(batch);
    const int2 anchor     = params.anchorAt(batch);

    for (int channel = 0; channel < channels; ++channel)
    {
        const int2 size = params.extent<true>(dst, batch, channel);

        if (x >= size.x || y >= size.y)
            continue;

        PT res = MorphWindow<K, true, UseGenericInterior>(src, rawSrc, batch, channel, y, x, size, kernelSize, anchor,
                                                          cuda::SetAll<PT>(identity));

        dst[GetCoordForLayout<true>(int3{x, y, batch}, channel)] = cuda::SaturateCast<cuda::BaseType<PT>>(res);
    }
}

template<MorphKind K, class T>
__device__ __forceinline__ T Morph3(T a, T b, T c)
{
    // Seed each row so NaNs cannot enter a floating accumulator and the finite
    // identity still bounds all-NaN or non-winning-infinity neighborhoods.
    if constexpr (!std::is_integral_v<T>)
        a = MorphCombine<K>(MorphIdentityValue<K, T>(), a);
    return MorphCombine<K>(MorphCombine<K>(a, b), c);
}

// Horizontal pass of a 3x3 window over four adjacent columns, from one vector load plus the two
// neighbours that fall outside the quad.
template<MorphKind K, class T>
__device__ __forceinline__ cuda::MakeType<T, 4> RowMorph3x3X4(const T *row, T left, T right)
{
    const auto center = *reinterpret_cast<const cuda::MakeType<T, 4> *>(row);

    return {Morph3<K>(left, center.x, center.y), Morph3<K>(center.x, center.y, center.z),
            Morph3<K>(center.y, center.z, center.w), Morph3<K>(center.z, center.w, right)};
}

// Vector loads and stores require alignment appropriate to the four-element type.
// Wrapped input and output rows can have independent pitches and base alignment.
template<class T>
__device__ __forceinline__ bool IsMorphQuadRowAligned(const T *srcRow, int srcStride, const T *dstRow, int dstStride)
{
    constexpr std::uintptr_t mask = alignof(cuda::MakeType<T, 4>) - 1;

    return ((reinterpret_cast<std::uintptr_t>(srcRow) | static_cast<std::uintptr_t>(srcStride)
             | reinterpret_cast<std::uintptr_t>(dstRow) | static_cast<std::uintptr_t>(dstStride))
            & mask)
        == 0;
}

template<class Params, class Wrapper>
__device__ __forceinline__ int MorphRowStride(const Wrapper &wrap, int batch, int channel)
{
    if constexpr (std::is_same_v<Params, UniformMorphParams>)
        return wrap.strides()[1];
    else
        return wrap.rowStride(batch, channel);
}

template<MorphKind K, bool IsPlanar, class RawSrcWrapper>
__device__ __forceinline__ auto Morph3x3X4(const RawSrcWrapper &rawSrc, int batch, int channel, int y, int x)
{
    using T      = std::remove_const_t<typename RawSrcWrapper::ValueType>;
    const T *top = &rawSrc[GetCoordForLayout<IsPlanar>(int3{x, y - 1, batch}, channel)];
    const T *mid = &rawSrc[GetCoordForLayout<IsPlanar>(int3{x, y, batch}, channel)];
    const T *bot = &rawSrc[GetCoordForLayout<IsPlanar>(int3{x, y + 1, batch}, channel)];

    const auto topRow = RowMorph3x3X4<K>(top, top[-1], top[4]);
    const auto midRow = RowMorph3x3X4<K>(mid, mid[-1], mid[4]);
    const auto botRow = RowMorph3x3X4<K>(bot, bot[-1], bot[4]);

    return MorphCombine<K>(MorphCombine<K>(topRow, midRow), botRow);
}

// Tail of the quad path: the columns the vectorized read cannot serve, one at a time.
template<MorphKind K, bool IsPlanar, class SrcWrapper, class DstWrapper>
__device__ __forceinline__ void MorphScalarGenericX4(const SrcWrapper &src, const DstWrapper &dst, int batch,
                                                     int channel, int y, int x, int width, int2 kernelSize, int2 anchor)
{
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int dstX = x + k;
        if (dstX >= width)
        {
            return;
        }

        dst[GetCoordForLayout<IsPlanar>(int3{dstX, y, batch}, channel)] = MorphBorderWindow<K, IsPlanar>(
            src, batch, channel, y, dstX, kernelSize, anchor, MorphIdentityValue<K, typename DstWrapper::ValueType>());
    }
}

// One thread per four output columns, walking each selected plane. Only the centered
// 3x3 window on aligned rows can be served by the vectorized read; everything else falls back to
// the per-column path above, which is why the quad shape is decided per row rather than per launch.
template<MorphKind K, bool IsPlanar, int Channels, class Params, class SrcWrapper, class RawSrcWrapper,
         class DstWrapper>
__global__ void MorphScalar3x3X4Kernel(const SrcWrapper src, const RawSrcWrapper rawSrc, DstWrapper dst, Params params)
{
    using T         = typename DstWrapper::ValueType;
    const int x     = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    const int2 kernelSize = params.mask(batch);
    const int2 anchor     = params.anchorAt(batch);

    const bool centered3x3 = IsCentered3x3(kernelSize, anchor);

#pragma unroll
    for (int channel = 0; channel < Channels; ++channel)
    {
        const int2 size = params.template extent<IsPlanar>(dst, batch, channel);

        if (x >= size.x || y >= size.y)
        {
            continue;
        }

        const auto row     = GetCoordForLayout<IsPlanar>(int3{0, y, batch}, channel);
        const T   *srcRow  = &rawSrc[row];
        T         *dstRow  = &dst[row];
        const bool aligned = IsMorphQuadRowAligned(srcRow, MorphRowStride<Params>(rawSrc, batch, channel), dstRow,
                                                   MorphRowStride<Params>(dst, batch, channel));

        if (centered3x3 && aligned && x > 0 && y > 0 && x + 4 < size.x && y + 1 < size.y)
        {
            *reinterpret_cast<cuda::MakeType<T, 4> *>(&dst[GetCoordForLayout<IsPlanar>(int3{x, y, batch}, channel)])
                = Morph3x3X4<K, IsPlanar>(rawSrc, batch, channel, y, x);
        }
        else
        {
            MorphScalarGenericX4<K, IsPlanar>(src, dst, batch, channel, y, x, size.x, kernelSize, anchor);
        }
    }
}

// Resolves the -1 sentinels in the caller's mask and anchor tensors, in place. The public header
// documents masks/anchors as [in, out] for exactly this reason: the masks are per image and live in
// device memory, so the host cannot resolve them and hand the kernels a private copy.
__global__ void ResolveMaskSentinels(cuda::Tensor1DWrap<int2> masks, cuda::Tensor1DWrap<int2> anchors, int numImages)
{
    int1 coord;
    coord.x = cuda::StaticCast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (coord.x >= numImages)
        return;

    int2 maskSize = masks[coord];
    int2 anchor   = anchors[coord];

    if (maskSize.x == -1 || maskSize.y == -1)
        maskSize.x = maskSize.y = 3;
    if (anchor.x < 0)
        anchor.x = maskSize.x / 2;
    if (anchor.y < 0)
        anchor.y = maskSize.y / 2;

    masks[coord]   = maskSize;
    anchors[coord] = anchor;
}

// -------------------------------------------------------------------------------------------
// Dispatch
// -------------------------------------------------------------------------------------------

template<NVCVBorderType B>
using BorderTag = std::integral_constant<NVCVBorderType, B>;

template<MorphKind K>
using MorphKindTag = std::integral_constant<MorphKind, K>;

// The five border modes the operator accepts; validated by the caller, so the last one is the
// default arm rather than a rejection.
template<class Cb>
void DispatchByBorder(NVCVBorderType borderMode, const Cb &cb)
{
    switch (borderMode)
    {
    case NVCV_BORDER_CONSTANT:
        return cb(BorderTag<NVCV_BORDER_CONSTANT>{});
    case NVCV_BORDER_REPLICATE:
        return cb(BorderTag<NVCV_BORDER_REPLICATE>{});
    case NVCV_BORDER_REFLECT:
        return cb(BorderTag<NVCV_BORDER_REFLECT>{});
    case NVCV_BORDER_WRAP:
        return cb(BorderTag<NVCV_BORDER_WRAP>{});
    default: // NVCV_BORDER_REFLECT101
        return cb(BorderTag<NVCV_BORDER_REFLECT101>{});
    }
}

template<class Cb>
void DispatchByMorphKind(NVCVMorphologyType morphType, const Cb &cb)
{
    if (morphType == NVCVMorphologyType::NVCV_DILATE)
    {
        return cb(MorphKindTag<MorphKind::kDilate>{});
    }
    return cb(MorphKindTag<MorphKind::kErode>{});
}

// Calls cb with a value-initialized element of the type that (code, channels) names. This replaces
// the legacy 8x4 function-pointer table, whose only populated rows were the four dtypes
// IsSupportedDataType admits and whose only populated columns were 1, 3 and 4 channels -- the two
// gates the caller applies first.
//
// `PlanesOnly` restricts the instantiation to the scalar element types, for the planar paths that
// process one channel plane at a time. It cannot be expressed by passing channels == 1: `channels`
// is a runtime value, so every arm of the channel switch is compiled either way, and the planar
// kernels would be instantiated for the eight vector element types they can never receive.
template<bool PlanesOnly = false, class Cb>
void DispatchByElementType(DataTypeCode code, int channels, const Cb &cb)
{
    auto byChannels = [&](auto scalar)
    {
        using BT = decltype(scalar);

        if constexpr (PlanesOnly)
        {
            return cb(BT{});
        }
        else
        {
            switch (channels)
            {
            case 1:
                return cb(BT{});
            case 3:
                return cb(cuda::MakeType<BT, 3>{});
            default:
                return cb(cuda::MakeType<BT, 4>{});
            }
        }
    };

    switch (code)
    {
    case DataTypeCode::kU8:
        return byChannels(uchar{});
    case DataTypeCode::kU16:
        return byChannels(ushort{});
    case DataTypeCode::kF32:
        return byChannels(float{});
    default: // DataTypeCode::kF16
        return byChannels(__half{});
    }
}

// The operator's three runtime configuration axes, turned into template arguments in one place.
template<bool PlanesOnly = false, class Cb>
void DispatchMorphConfig(NVCVMorphologyType morphType, NVCVBorderType borderMode, DataTypeCode code, int channels,
                         const Cb &cb)
{
    DispatchByMorphKind(morphType,
                        [&](auto kind)
                        {
                            DispatchByBorder(borderMode,
                                             [&](auto border) {
                                                 DispatchByElementType<PlanesOnly>(
                                                     code, channels, [&](auto element) { cb(kind, border, element); });
                                             });
                        });
}

// The generic-interior read is a compile-time kernel variant selected from a runtime flag.
template<class Cb>
void DispatchByInterior(bool useGenericInterior, const Cb &cb)
{
    if (useGenericInterior)
    {
        return cb(std::true_type{});
    }
    return cb(std::false_type{});
}

// -------------------------------------------------------------------------------------------
// Tensor path
// -------------------------------------------------------------------------------------------

template<MorphKind K, class D, NVCVBorderType B>
void MorphTensorRun(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    UniformMorphParams params, int numSamples, cudaStream_t stream)
{
    using BT = cuda::BaseType<D>;

    const BT identity = MorphIdentityValue<K, BT>();

    auto src    = cuda::CreateBorderWrapNHW<const D, B, int32_t>(inData, cuda::SetAll<D>(identity));
    auto rawSrc = cuda::CreateTensorWrapNHW<const D, int32_t>(inData);
    auto dst    = cuda::CreateTensorWrapNHW<D, int32_t>(outData);

    const dim3 block(16, 16);
    const dim3 grid(util::DivUp(params.dstSize.x, static_cast<int>(block.x)),
                    util::DivUp(params.dstSize.y, static_cast<int>(block.y)), numSamples);

    // A centered 3x3 mask needs no generic-interior branch.
    DispatchByInterior(
        !IsCentered3x3(params.kernelSize, params.anchor),
        [&](auto useGenericInterior)
        {
            if constexpr ((std::is_same_v<D, uchar> || std::is_same_v<D, float>)&&B == NVCV_BORDER_REPLICATE
                          && !decltype(useGenericInterior)::value)
            {
                const dim3 quadGrid(util::DivUp(params.dstSize.x, static_cast<int>(block.x) * 4), grid.y, numSamples);
                MorphScalar3x3X4Kernel<K, false, 1><<<quadGrid, block, 0, stream>>>(src, rawSrc, dst, params);
            }
            else
            {
                MorphKernel<K, false, decltype(useGenericInterior)::value>
                    <<<grid, block, 0, stream>>>(src, rawSrc, dst, params, 1, identity);
            }
        });
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Bulk device-to-device copy standing in for a mask that selects exactly one element, or for an
// explicitly requested no-op.
inline void MorphTensorCopy(const nvcv::TensorDataAccessStridedImagePlanar &inAccess,
                            const nvcv::TensorDataAccessStridedImagePlanar &outAccess, cudaStream_t stream)
{
    for (int32_t i = 0; i < inAccess.numSamples(); ++i)
    {
        nvcv::Byte *inSampData  = inAccess.sampleData(i);
        nvcv::Byte *outSampData = outAccess.sampleData(i);

        for (int32_t p = 0; p < inAccess.numPlanes(); ++p)
        {
            NVCV_CHECK_THROW(cudaMemcpy2DAsync(outAccess.planeData(p, outSampData), outAccess.rowStride(),
                                               inAccess.planeData(p, inSampData), inAccess.rowStride(),
                                               inAccess.numCols() * inAccess.colStride(), inAccess.numRows(),
                                               cudaMemcpyDeviceToDevice, stream));
        }
    }
}

inline void MorphTensorSubmit(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                              NVCVMorphologyType morphType, nvcv::Size2D maskSize, int2 anchor, bool noop,
                              NVCVBorderType borderMode, cudaStream_t stream)
{
    const DataFormatCode inputFormat  = ClassifyDataFormat(inData.layout());
    const DataFormatCode outputFormat = ClassifyDataFormat(outData.layout());
    const DataTypeCode   dataType     = ClassifyDataType(inData.dtype());

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    if (!IsSupportedDataType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(dataType));
    }

    // DispatchMorphConfig picks the element type and channel vector width from the input, and the
    // output is then wrapped with that same type, so a differing output would be written with the
    // wrong element width -- or, if smaller, past its storage.
    if (inData.dtype() != outData.dtype() || inAccess->numSamples() != outAccess->numSamples()
        || inAccess->numChannels() != outAccess->numChannels() || inAccess->numRows() != outAccess->numRows()
        || inAccess->numCols() != outAccess->numCols())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same shape and DataType");
    }

    const int channels = inAccess->numChannels();
    if (!(channels == 1 || channels == 3 || channels == 4))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    cvcuda::priv::ValidateBorderMode(borderMode);

    if (!(morphType == NVCVMorphologyType::NVCV_ERODE || morphType == NVCVMorphologyType::NVCV_DILATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid morph_type %d",
                              static_cast<int>(morphType));
    }

    // Erode and dilate treat the channels independently, so a planar tensor runs as a flattened
    // single-channel (N*C, H, W, 1) view of itself through the interleaved path.
    if (IsPlanar(inputFormat))
    {
        auto views = cvcuda::priv::PlanarSingleChannelViews(inData, outData);
        NVCV_ASSERT(views);
        return MorphTensorSubmit(views->first, views->second, morphType, maskSize, anchor, noop, borderMode, stream);
    }

    int2 kernelSize{maskSize.w, maskSize.h};
    if (kernelSize.x == -1 || kernelSize.y == -1)
    {
        kernelSize.x = 3;
        kernelSize.y = 3;
    }

    int2 kernelAnchor = anchor;
    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x = kernelSize.x >> 1;
    }
    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y = kernelSize.y >> 1;
    }
    NVCV_ASSERT(0 <= kernelAnchor.x && kernelAnchor.x < kernelSize.x);
    NVCV_ASSERT(0 <= kernelAnchor.y && kernelAnchor.y < kernelSize.y);

    if (noop || kernelSize.x * kernelSize.y == 1)
    {
        MorphTensorCopy(*inAccess, *outAccess, stream);
        return;
    }

    const int numSamples = outAccess->numSamples();

    // The kernel wrappers are instantiated with 32-bit strides, which is what keeps the addressing
    // arithmetic in the kernel 32-bit; a tensor that does not fit is rejected rather than widened.
    // Only the launched paths are affected, so this sits after the copy shortcut above.
    const int64_t outMaxStride = outAccess->sampleStride() * numSamples;
    const int64_t inMaxStride  = inAccess->sampleStride() * numSamples;
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    const UniformMorphParams params{
        int2{outAccess->numCols(), outAccess->numRows()},
        kernelSize, kernelAnchor
    };

    DispatchMorphConfig(morphType, borderMode, dataType, channels,
                        [&](auto kind, auto border, auto element)
                        {
                            MorphTensorRun<decltype(kind)::value, decltype(element), decltype(border)::value>(
                                inData, outData, params, numSamples, stream);
                        });
}

// -------------------------------------------------------------------------------------------
// Var-shape path
// -------------------------------------------------------------------------------------------

template<MorphKind K, class D, NVCVBorderType B>
void MorphVarShapeRun(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                      const nvcv::ImageBatchVarShapeDataStridedCuda &outData, VarShapeMorphParams params,
                      bool enableGenericInterior, cudaStream_t stream)
{
    using BT = cuda::BaseType<D>;

    const BT identity = MorphIdentityValue<K, BT>();

    const nvcv::Size2D outMaxSize = outData.maxSize();

    const dim3 block(16, 16);
    const dim3 grid(util::DivUp(outMaxSize.w, static_cast<int>(block.x)),
                    util::DivUp(outMaxSize.h, static_cast<int>(block.y)), outData.numImages());

    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(identity));
    cuda::ImageBatchVarShapeWrap<const D> rawSrc(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

    DispatchByInterior(enableGenericInterior || (B == NVCV_BORDER_CONSTANT && std::is_same_v<D, uchar3>),
                       [&](auto useGenericInterior)
                       {
                           MorphKernel<K, false, decltype(useGenericInterior)::value>
                               <<<grid, block, 0, stream>>>(src, rawSrc, dst, params, 1, identity);
                       });
    NVCV_CHECK_THROW(cudaGetLastError());
}

template<MorphKind K, class D, NVCVBorderType B>
void MorphVarShapeRunPlanar(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::ImageBatchVarShapeDataStridedCuda &outData, VarShapeMorphParams params,
                            int channels, bool enableGenericInterior, cudaStream_t stream)
{
    using BT = cuda::BaseType<D>;

    const BT identity = MorphIdentityValue<K, BT>();

    const nvcv::Size2D outMaxSize = outData.maxSize();
    const int          numImages  = outData.numImages();

    const dim3 block(16, 16);
    const int  gridX = util::DivUp(outMaxSize.w, static_cast<int>(block.x));
    const int  gridY = util::DivUp(outMaxSize.h, static_cast<int>(block.y));

    cuda::BorderVarShapeWrap<const D, B>  src(inData, cuda::SetAll<D>(identity));
    cuda::ImageBatchVarShapeWrap<const D> rawSrc(inData);
    cuda::ImageBatchVarShapeWrap<D>       dst(outData);

    if constexpr (std::is_same_v<D, uchar>)
    {
        // The packed path is validated for this VarShape channel/border combination.
        if constexpr (B == NVCV_BORDER_REPLICATE)
        {
            if (!enableGenericInterior && channels == 3)
            {
                const dim3 quadGrid(util::DivUp(outMaxSize.w, static_cast<int>(block.x) * 4), gridY, numImages);

                MorphScalar3x3X4Kernel<K, true, 3><<<quadGrid, block, 0, stream>>>(src, rawSrc, dst, params);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        const dim3 imageGrid(gridX, gridY, numImages);
        DispatchByInterior(enableGenericInterior,
                           [&](auto useGenericInterior)
                           {
                               MorphPlanarChannelsKernel<K, decltype(useGenericInterior)::value>
                                   <<<imageGrid, block, 0, stream>>>(src, rawSrc, dst, params, channels, identity);
                           });
    }
    else
    {
        const dim3 planeGrid(gridX, gridY, channels * numImages);
        DispatchByInterior(enableGenericInterior,
                           [&](auto useGenericInterior)
                           {
                               MorphKernel<K, true, decltype(useGenericInterior)::value>
                                   <<<planeGrid, block, 0, stream>>>(src, rawSrc, dst, params, channels, identity);
                           });
    }
    NVCV_CHECK_THROW(cudaGetLastError());
}

// Bulk device-to-device copy for an explicitly requested no-op, plane by plane so that images with
// different row strides are each copied with their own.
inline void MorphVarShapeCopy(const nvcv::ImageBatchVarShape &inBatch, const nvcv::ImageBatchVarShape &outBatch,
                              const nvcv::ImageFormat &format, cudaStream_t stream)
{
    for (auto init = inBatch.begin(), outit = outBatch.begin(); init != inBatch.end() && outit != outBatch.end();
         ++init, ++outit)
    {
        auto inImgData  = (*init).exportData<nvcv::ImageDataStridedCuda>();
        auto outImgData = (*outit).exportData<nvcv::ImageDataStridedCuda>();

        for (int32_t p = 0; p < inImgData->numPlanes(); ++p)
        {
            const nvcv::ImagePlaneStrided &inPlane  = inImgData->plane(p);
            const nvcv::ImagePlaneStrided &outPlane = outImgData->plane(p);
            const size_t rowBytes = static_cast<size_t>(inPlane.width) * format.planePixelStrideBytes(p);

            NVCV_CHECK_THROW(cudaMemcpy2DAsync(outPlane.basePtr, outPlane.rowStride, inPlane.basePtr, inPlane.rowStride,
                                               rowBytes, inPlane.height, cudaMemcpyDeviceToDevice, stream));
        }
    }
}

// `enableGenericInterior` is a performance gate, not a correctness one: it selects the kernel
// variant that tries a bounds-free read for any mask window that lies wholly inside the image. The
// tensor path decides this from the mask geometry, but here the masks are per image and live in
// device memory, so the caller can only pass a proxy -- inherited from the legacy operator, which
// used `iteration > 1`. When the mask really is a centered 3x3 the extra branch never fires,
// because the unrolled 9-tap read has already returned.
inline void MorphVarShapeSubmit(const nvcv::ImageBatchVarShape &inBatch, const nvcv::ImageBatchVarShape &outBatch,
                                NVCVMorphologyType morphType, const nvcv::TensorDataStridedCuda &masks,
                                const nvcv::TensorDataStridedCuda &anchors, bool noop, NVCVBorderType borderMode,
                                bool enableGenericInterior, cudaStream_t stream)
{
    auto inData  = cvcuda::priv::RequireCudaVarShape(inBatch, stream, "Input");
    auto outData = cvcuda::priv::RequireCudaVarShape(outBatch, stream, "Output");

    const DataFormatCode    inputFormat  = ClassifyDataFormat(*inData);
    const DataFormatCode    outputFormat = ClassifyDataFormat(*outData);
    const nvcv::ImageFormat format       = inData->uniqueFormat();
    const DataTypeCode      dataType     = ClassifyDataType(format);

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    if (!IsSupportedDataType(dataType))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(dataType));
    }

    // ClassifyDataFormat only compares planar-ness and batched-ness. The kernels write through the
    // output batch while their types come from the input's unique format, so both must match.
    if (format != outData->uniqueFormat() || inData->numImages() != outData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same format and number of images");
    }

    cvcuda::priv::ValidateBorderMode(borderMode);

    if (!(morphType == NVCVMorphologyType::NVCV_ERODE || morphType == NVCVMorphologyType::NVCV_DILATE))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid morph_type %d",
                              static_cast<int>(morphType));
    }

    const int  channels = format.numChannels();
    const bool isPlanar = IsPlanar(inputFormat);

    if (!(channels == 1 || channels == 3 || channels == 4))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (isPlanar && static_cast<int64_t>(inData->numImages()) * channels > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar Morphology requires numImages * channels <= 65535 (CUDA grid-z limit)");
    }

    if (noop)
    {
        MorphVarShapeCopy(inBatch, outBatch, format, stream);
        return;
    }

    const dim3 block(32);
    const dim3 grid(util::DivUp(inData->numImages(), static_cast<int>(block.x)));

    // Both tensors are wrapped as int2 and written back through by ResolveMaskSentinels, so a
    // different element type or a shorter length would have it read and write past their storage.
    cvcuda::priv::RequirePerImageParam(masks, nvcv::TYPE_2S32, inData->numImages(), "masks");
    cvcuda::priv::RequirePerImageParam(anchors, nvcv::TYPE_2S32, inData->numImages(), "anchors");

    VarShapeMorphParams params{cuda::Tensor1DWrap<int2>{masks}, cuda::Tensor1DWrap<int2>{anchors}};

    ResolveMaskSentinels<<<grid, block, 0, stream>>>(params.kernelSizes, params.anchors, inData->numImages());
    NVCV_CHECK_THROW(cudaGetLastError());

    // The layout decides which kernel family runs and which element types exist, so it is resolved
    // above the dispatch tree rather than inside it.
    if (isPlanar)
    {
        DispatchMorphConfig<true>(
            morphType, borderMode, dataType, channels,
            [&](auto kind, auto border, auto element)
            {
                MorphVarShapeRunPlanar<decltype(kind)::value, decltype(element), decltype(border)::value>(
                    *inData, *outData, params, channels, enableGenericInterior, stream);
            });
    }
    else
    {
        DispatchMorphConfig(morphType, borderMode, dataType, channels,
                            [&](auto kind, auto border, auto element)
                            {
                                MorphVarShapeRun<decltype(kind)::value, decltype(element), decltype(border)::value>(
                                    *inData, *outData, params, enableGenericInterior, stream);
                            });
    }
}

// -------------------------------------------------------------------------------------------
// Multi-pass drivers
// -------------------------------------------------------------------------------------------

inline std::pair<NVCVMorphologyType, NVCVMorphologyType> SplitMorphologyPasses(NVCVMorphologyType morphType)
{
    if (morphType == NVCVMorphologyType::NVCV_OPEN)
    {
        return {NVCVMorphologyType::NVCV_ERODE, NVCVMorphologyType::NVCV_DILATE};
    }

    return {NVCVMorphologyType::NVCV_DILATE, NVCVMorphologyType::NVCV_ERODE};
}

inline void RunTensorDilateErode(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                                 const nvcv::TensorDataStridedCuda &outData, nvcv::OptionalTensorConstRef workspace,
                                 NVCVMorphologyType morphType, nvcv::Size2D maskSize, int2 anchor, int32_t iteration,
                                 NVCVBorderType borderMode)
{
    // iteration == 0 is always a no-op copy regardless of whether a workspace is provided.
    // Without this early return the multi-pass driver writes to workspace (not outData) and the
    // loop body never executes, leaving outData untouched.
    if (iteration == 0)
    {
        MorphTensorSubmit(inData, outData, morphType, maskSize, anchor, true, borderMode, stream);
        return;
    }

    if (workspace == nullptr && iteration > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for iterations > 1");
    }

    if (workspace == nullptr)
    {
        // iteration == 0 is handled above; the only remaining case here is iteration == 1.
        MorphTensorSubmit(inData, outData, morphType, maskSize, anchor, false, borderMode, stream);
        return;
    }

    auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
    NVCV_ASSERT(workspaceData);

    const nvcv::TensorDataStridedCuda *iterIn  = &inData;
    const nvcv::TensorDataStridedCuda *iterOut = (iteration % 2 == 1) ? &outData : &(*workspaceData);
    MorphTensorSubmit(*iterIn, *iterOut, morphType, maskSize, anchor, false, borderMode, stream);

    // The first pass's destination becomes the next pass's source; the parity picks the buffer that
    // makes the last pass land in outData.
    iterIn  = iterOut;
    iterOut = (iteration % 2 == 0) ? &outData : &(*workspaceData);

    for (int i = 1; i < iteration; ++i)
    {
        MorphTensorSubmit(*iterIn, *iterOut, morphType, maskSize, anchor, false, borderMode, stream);
        std::swap(iterIn, iterOut);
    }
}

inline void RunTensorOpenClose(cudaStream_t stream, const nvcv::TensorDataStridedCuda &inData,
                               const nvcv::TensorDataStridedCuda &outData, nvcv::OptionalTensorConstRef workspace,
                               NVCVMorphologyType morphType, nvcv::Size2D maskSize, int2 anchor, int32_t iteration,
                               NVCVBorderType borderMode)
{
    if (workspace == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Workspace must be provided for NVCV_CLOSE or NVCV_OPEN");
    }

    auto [first, second] = SplitMorphologyPasses(morphType);

    auto workspaceData = workspace->get().exportData<nvcv::TensorDataStridedCuda>();
    NVCV_ASSERT(workspaceData);

    MorphTensorSubmit(inData, *workspaceData, first, maskSize, anchor, iteration == 0, borderMode, stream);
    MorphTensorSubmit(*workspaceData, outData, second, maskSize, anchor, iteration == 0, borderMode, stream);

    for (int i = 1; i < iteration; ++i)
    {
        MorphTensorSubmit(outData, *workspaceData, first, maskSize, anchor, false, borderMode, stream);
        MorphTensorSubmit(*workspaceData, outData, second, maskSize, anchor, false, borderMode, stream);
    }
}

inline void RunVarShapeDilateErode(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape          &out,
                                   nvcv::OptionalImageBatchVarShapeConstRef workspace, NVCVMorphologyType morphType,
                                   const nvcv::TensorDataStridedCuda &masksData,
                                   const nvcv::TensorDataStridedCuda &anchorsData, int32_t iteration,
                                   NVCVBorderType borderMode)
{
    // iteration == 0 is always a no-op copy regardless of whether a workspace is provided.
    // Without this early return the multi-pass driver writes to workspace (not out) and the
    // loop body never executes, leaving out untouched.
    if (iteration == 0)
    {
        MorphVarShapeSubmit(in, out, morphType, masksData, anchorsData, true, borderMode, false, stream);
        return;
    }

    if (workspace == nullptr && iteration > 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for iterations > 1");
    }

    if (workspace == nullptr)
    {
        // iteration == 0 is handled above; the only remaining case here is iteration == 1.
        MorphVarShapeSubmit(in, out, morphType, masksData, anchorsData, false, borderMode, false, stream);
        return;
    }

    NVCV_ASSERT(workspace);

    const nvcv::ImageBatchVarShape *iterIn  = &in;
    const nvcv::ImageBatchVarShape *iterOut = (iteration % 2 == 1) ? &out : &workspace->get();

    MorphVarShapeSubmit(*iterIn, *iterOut, morphType, masksData, anchorsData, false, borderMode, iteration > 1, stream);

    // The first pass's destination becomes the next pass's source; the parity picks the buffer that
    // makes the last pass land in out.
    iterIn  = iterOut;
    iterOut = (iteration % 2 == 0) ? &out : &workspace->get();

    for (int i = 1; i < iteration; ++i)
    {
        // This loop only runs for iteration >= 2, which fixes both flags to what the expressions
        // above would evaluate to.
        MorphVarShapeSubmit(*iterIn, *iterOut, morphType, masksData, anchorsData, false, borderMode, true, stream);
        std::swap(iterIn, iterOut);
    }
}

inline void RunVarShapeOpenClose(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                 const nvcv::ImageBatchVarShape          &out,
                                 nvcv::OptionalImageBatchVarShapeConstRef workspace, NVCVMorphologyType morphType,
                                 const nvcv::TensorDataStridedCuda &masksData,
                                 const nvcv::TensorDataStridedCuda &anchorsData, int32_t iteration,
                                 NVCVBorderType borderMode)
{
    if (workspace == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Workspace must be provided for NVCV_CLOSE");
    }

    auto [first, second] = SplitMorphologyPasses(morphType);

    MorphVarShapeSubmit(in, *workspace, first, masksData, anchorsData, iteration == 0, borderMode, false, stream);
    MorphVarShapeSubmit(*workspace, out, second, masksData, anchorsData, iteration == 0, borderMode, false, stream);

    for (int i = 1; i < iteration; ++i)
    {
        // This loop only runs for iteration >= 2, so neither pass can be a no-op.
        MorphVarShapeSubmit(out, *workspace, first, masksData, anchorsData, false, borderMode, false, stream);
        MorphVarShapeSubmit(*workspace, out, second, masksData, anchorsData, false, borderMode, false, stream);
    }
}

} // namespace

namespace cvcuda::priv {

Morphology::Morphology() = default;

void Morphology::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                            nvcv::OptionalTensorConstRef workspace, NVCVMorphologyType morph_type,
                            nvcv::Size2D mask_size, int2 anchor, int32_t iteration,
                            const NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Morphology::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");
    if (iteration < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Iteration must be >= 0");
    }

    switch (morph_type)
    {
    case NVCVMorphologyType::NVCV_DILATE:
    case NVCVMorphologyType::NVCV_ERODE:
        RunTensorDilateErode(stream, *inData, *outData, workspace, morph_type, mask_size, anchor, iteration,
                             borderMode);
        break;
    case NVCVMorphologyType::NVCV_OPEN:
    case NVCVMorphologyType::NVCV_CLOSE:
        RunTensorOpenClose(stream, *inData, *outData, workspace, morph_type, mask_size, anchor, iteration, borderMode);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Wrong morph_type");
    }
}

void Morphology::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                            const nvcv::ImageBatchVarShape &out, nvcv::OptionalImageBatchVarShapeConstRef workspace,
                            NVCVMorphologyType morph_type, const nvcv::Tensor &masks, const nvcv::Tensor &anchors,
                            int32_t iteration, NVCVBorderType borderMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Morphology::operator()[ImageBatchVarShape]");
    auto masksData = masks.exportData<nvcv::TensorDataStridedCuda>();
    if (masksData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "masksData must be a tensor");
    }

    auto anchorsData = anchors.exportData<nvcv::TensorDataStridedCuda>();
    if (anchorsData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "anchors must be a tensor");
    }
    if (iteration < 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Iteration must be >= 0");
    }

    switch (morph_type)
    {
    case NVCVMorphologyType::NVCV_DILATE:
    case NVCVMorphologyType::NVCV_ERODE:
        RunVarShapeDilateErode(stream, in, out, workspace, morph_type, *masksData, *anchorsData, iteration, borderMode);
        break;

    case NVCVMorphologyType::NVCV_CLOSE:
    case NVCVMorphologyType::NVCV_OPEN:
        RunVarShapeOpenClose(stream, in, out, workspace, morph_type, *masksData, *anchorsData, iteration, borderMode);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Wrong morph_type");
    }
}

} // namespace cvcuda::priv
