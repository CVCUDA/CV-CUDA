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

#ifndef CVCUDA_PRIV_SAME_SHAPE_COMMON_CUH
#define CVCUDA_PRIV_SAME_SHAPE_COMMON_CUH

#include "ElementStreamCommon.cuh"

#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/Optional.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

// Shared preamble for operators whose destination geometry is identical to the source's: the
// host-side precondition checks and the runtime-to-compile-time channel dispatch that each such
// operator previously repeated verbatim. Nothing here computes pixel values.
//
// Membership is decided by that contract, not by the shape of the kernel. #47 calls these the
// "pointwise clone family", but that label is loose in both directions: AdjustSharpness is a 3x3
// convolution plus a blend and qualifies, while BrightnessContrast is genuinely pointwise and does
// not, because it accepts a dst dtype differing from src and so needs its own validator.
namespace cvcuda::priv::same_shape {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

// Wrapped rows need only scalar alignment. Volatile lane accesses prevent the
// compiler from combining this fallback into an aligned vector transaction.
template<typename T>
inline __device__ T LoadUnalignedPixel(const T *ptr)
{
    using BT = cuda::BaseType<T>;
    T value;
#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        if constexpr (cuda::detail::IsHalfV<BT>)
        {
            cuda::GetElement(value, c) = __ushort_as_half(reinterpret_cast<const volatile unsigned short *>(ptr)[c]);
        }
        else
        {
            cuda::GetElement(value, c) = reinterpret_cast<const volatile BT *>(ptr)[c];
        }
    }
    return value;
}

template<typename T>
inline __device__ void StoreUnalignedPixel(T *ptr, T value)
{
    using BT = cuda::BaseType<T>;
#pragma unroll
    for (int c = 0; c < cuda::NumElements<T>; ++c)
    {
        if constexpr (cuda::detail::IsHalfV<BT>)
        {
            reinterpret_cast<volatile unsigned short *>(ptr)[c] = __half_as_ushort(cuda::GetElement(value, c));
        }
        else
        {
            reinterpret_cast<volatile BT *>(ptr)[c] = cuda::GetElement(value, c);
        }
    }
}

// The plane loop every scalar kernel in this family wraps its per-element body in: an interleaved
// sample is one plane and is visited once with p == 0, a planar sample is visited once per plane.
// The `numPlanes == 1` assert is the interleaved contract -- an interleaved kernel ignores
// numPlanes entirely, so a caller passing anything else would silently drop planes.
//
// Callers capture by value (`[=]`), which the per-element bodies want anyway since they take their
// wrappers by value. It is not only style: capturing by reference changed the PTX of the
// interleaved BrightnessContrast instantiations at -O3 -DNDEBUG (nvcc stopped hoisting the
// blockIdx.z read out of the per-argument branches), while by value the only codegen difference
// against the hand-written loops is a reduction in generic address casts.
template<bool IsPlanar, class F>
inline __device__ void ForEachPlane(int numPlanes, F &&f)
{
    assert(IsPlanar || numPlanes == 1);
    if constexpr (!IsPlanar)
    {
        f(0);
    }
    else
    {
        for (int p = 0; p < numPlanes; ++p)
        {
            f(p);
        }
    }
}

// Address a sample in either layout from one kernel body. Interleaved samples are already indexed
// by (x, y, sample); planar samples splice the plane index in to give (x, y, plane, sample).
template<bool IsPlanar>
inline __device__ std::conditional_t<IsPlanar, int4, int3> GetCoordForLayout(int3 nhwCoord, int p)
{
    if constexpr (!IsPlanar)
    {
        assert(p == 0);
        return nhwCoord;
    }
    else
    {
        return {nhwCoord.x, nhwCoord.y, p, nhwCoord.z};
    }
}

// Resolve a runtime channel count to a compile-time vector type and planar flag, then hand both to
// `cb`. `ValBase` is the base type the caller's own type switch already resolved; splitting the
// channel arm out this way keeps that switch in the operator's translation unit, where ADL can
// still find it. `opName` names the operator in the error message only.
template<typename ValBase, typename Cb>
inline void DispatchChannels(int numChannels, int numPlanes, const char *opName, const Cb &cb)
{
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
    else if (numChannels == 3 || numChannels == 4)
    {
        // DECLARED HARDENING (not a refactor): the multi-channel arms are interleaved-only, so a
        // multi-plane caller would have every plane but the first silently dropped -- the kernels
        // pass `isPlanar == false` straight to ForEachPlane, which visits plane 0 alone (and
        // asserts in a debug build). No caller reaches this today: a tensor's numInterleavedChannels
        // is forced to 1 unless the layout is channel-last, and channel-last implies numPlanes ==
        // 1; a var-shape batch takes it from planeDataType(0), which is single-channel for every
        // multi-plane predefined format. It is a custom multi-plane format whose first plane
        // carried 3 or 4 channels that would land here, so state the invariant rather than trust
        // it. BrightnessContrast's own validator already rejects the same combination.
        if (numPlanes != 1)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "%s does not support %d interleaved channels across %d planes; planar images are "
                                  "expected to have one channel per plane",
                                  opName, numChannels, numPlanes);
        }
        if (numChannels == 3)
        {
            cb(cuda::MakeType<ValBase, 3>{}, std::integral_constant<bool, false>{});
        }
        else
        {
            cb(cuda::MakeType<ValBase, 4>{}, std::integral_constant<bool, false>{});
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid number of channels: %s supports 1, 3 or 4 channels", opName);
    }
}

// Resolve a runtime dtype to the compile-time base type the operator's kernels are templated on.
// The family agrees on a handful of dtype sets, so there is one of these per set rather than one
// per operator. `opName` is substituted into the message so the text each operator throws is the same
// character sequence it threw when the switch lived in its own translation unit.
//
// The 2/3/4-channel spellings of a base type select the same base type, because the channel count
// is resolved separately by DispatchChannels. Those spellings reach here from the var-shape path,
// where the channel count comes from the dtype itself (an interleaved RGB8 image reports
// planeDataType(0) == TYPE_3U8). A tensor cannot: ValidateSrcDstTensors takes the channel count
// from the C axis and now rejects a multi-channel tensor dtype outright, because the two were not
// tied together and disagreed silently.
#define NVCV_SAME_SHAPE_RUN_TYPED(DYN_BASE_TYPE, STATIC_BASE_TYPE)                    \
    ((dtype == nvcv::TYPE_4##DYN_BASE_TYPE) || (dtype == nvcv::TYPE_3##DYN_BASE_TYPE) \
     || (dtype == nvcv::TYPE_2##DYN_BASE_TYPE) || (dtype == nvcv::TYPE_##DYN_BASE_TYPE)) cb(STATIC_BASE_TYPE{});

// u8 / u16 / f16 / f32.
template<typename Cb>
inline void DispatchBaseType(nvcv::DataType dtype, const char *opName, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

    // clang-format off
    if NVCV_SAME_SHAPE_RUN_TYPED(U8, uchar)
    else if NVCV_SAME_SHAPE_RUN_TYPED(U16, ushort)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F16, __half)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F32, float)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: %s supports 8-bit unsigned, 16-bit unsigned, 16-bit float and "
                              "32-bit float", opName);
    }
    // clang-format on
}

template<typename Cb>
inline void DispatchBaseTypeWithF64(nvcv::DataType dtype, const char *opName, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;

    // clang-format off
    if NVCV_SAME_SHAPE_RUN_TYPED(U8, uchar)
    else if NVCV_SAME_SHAPE_RUN_TYPED(U16, ushort)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F16, __half)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F32, float)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F64, double)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: %s supports 8-bit unsigned, 16-bit unsigned, 16-bit float, "
                              "32-bit float and 64-bit float", opName);
    }
    // clang-format on
}

template<typename Cb>
inline void DispatchWideBaseType(nvcv::DataType dtype, const char *opName, const Cb &cb)
{
    using uchar  = unsigned char;
    using ushort = unsigned short;
    using llong  = long long;

    // clang-format off
    if NVCV_SAME_SHAPE_RUN_TYPED(U8, uchar)
    else if NVCV_SAME_SHAPE_RUN_TYPED(U16, ushort)
    else if NVCV_SAME_SHAPE_RUN_TYPED(S16, short)
    else if NVCV_SAME_SHAPE_RUN_TYPED(S32, int)
    else if NVCV_SAME_SHAPE_RUN_TYPED(S64, llong)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F16, __half)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F32, float)
    else if NVCV_SAME_SHAPE_RUN_TYPED(F64, double)
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid data type: %s supports unsigned integers (U8, U16), signed integers "
                              "(S16, S32, S64), and floats (F16, F32, F64)", opName);
    }
    // clang-format on
}

#undef NVCV_SAME_SHAPE_RUN_TYPED

// The two dispatches an operator entry point actually wants: dtype first, then channel count, so
// an unsupported dtype is still reported before an unsupported channel count.
template<typename Cb>
inline void DispatchTypeAndChannels(int numChannels, int numPlanes, nvcv::DataType dtype, const char *opName,
                                    const Cb &cb)
{
    DispatchBaseType(dtype, opName,
                     [&](auto dummyVal) { DispatchChannels<decltype(dummyVal)>(numChannels, numPlanes, opName, cb); });
}

template<typename Cb>
inline void DispatchTypeWithF64AndChannels(int numChannels, int numPlanes, nvcv::DataType dtype, const char *opName,
                                           const Cb &cb)
{
    DispatchBaseTypeWithF64(dtype, opName,
                            [&](auto dummyVal)
                            { DispatchChannels<decltype(dummyVal)>(numChannels, numPlanes, opName, cb); });
}

template<typename Cb>
inline void DispatchWideTypeAndChannels(int numChannels, int numPlanes, nvcv::DataType dtype, const char *opName,
                                        const Cb &cb)
{
    DispatchWideBaseType(dtype, opName,
                         [&](auto dummyVal)
                         { DispatchChannels<decltype(dummyVal)>(numChannels, numPlanes, opName, cb); });
}

// Validation ------------------------------------------------------------------------------

// BUG FIX (not a refactor): reject a tensor whose columns are spaced further apart than one pixel.
//
// Nothing downstream of this validator can express such a spacing. elem_stream::MakePlan takes a
// row to be `numCols * sizeof(ValueT)` contiguous bytes, and both CreateTensorWrapNHW and
// Tensor4DWrap index x by sizeof(ValueT); so a padded column stride is not merely unsupported, it
// is silently mis-processed -- the operator maps the pad bytes and leaves the real columns
// untouched. AdjustSharpness's element path already declines this shape (TryRunElemTensor tests
// colStride() against kChannels * kElemSize) but then falls through to a path that does not, and
// no other operator tested it at all.
//
// The shape is reachable: nvcv::priv::ValidateTensorBufferStrided skips any axis whose extent is 1
// when it pins the packed axes, so an NHWC U8 tensor with shape {1, 4, 4, 1} never has strides[3]
// pinned to the element size, and its width stride is then only required to equal
// strides[3] * 1. Strides {64, 16, 4, 4} pass that validator and construct through
// nvcvTensorWrapDataConstruct or a hand-built __cuda_array_interface__ dict.
//
// What it is *not* is a plain `t[..., 0:1]` view of an interleaved image: such a view keeps the
// channel stride at the element size (numpy/cupy give {..., 4, 1}), and because the validator then
// requires the width stride to equal that channel stride, nvcv rejects the view outright rather
// than admitting it here. The reachable form is the one where the two trailing strides agree.
//
// Guarded on numCols() > 1 for the same reason the nvcv validator skips extent-1 axes: with a
// single column the stride is never added to any address, and a tensor that works today (a
// single-column view of a wider image) must keep working. There is deliberately no companion
// chStride() test: for an interleaved tensor with more than one channel nvcv has already pinned
// the channel stride to the element size, and with exactly one channel the channel stride *is*
// the column stride this check reads.
inline void CheckPackedColumns(const nvcv::TensorDataAccessStridedImagePlanar &access, int numInterleavedChannels,
                               nvcv::DataType dtype, const char *what)
{
    if (access.numCols() <= 1)
    {
        return;
    }

    const int64_t pixelStride = static_cast<int64_t>(numInterleavedChannels) * dtype.strideBytes();
    if (access.colStride() != pixelStride)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "%s column stride must be %ld bytes (one pixel); padded columns are not supported, "
                              "but it is %ld",
                              what, static_cast<long>(pixelStride), static_cast<long>(access.colStride()));
    }
}

// What a validated src/dst tensor pair yields. Returned as a struct rather than written through
// out-params because BrightnessContrast needs five of these values and a five-out-param signature
// is unreadable at the call site.
struct SrcDstTensorInfo
{
    int numSamples;
    int numChannels;            // the C axis, whatever the layout puts there
    int numInterleavedChannels; // equal to numChannels only for a channel-last layout, else 1
    int numPlanes;

    nvcv::DataType srcDtype;
    nvcv::DataType dstDtype; // equal to srcDtype unless the caller passed requireSameDtype == false
};

// `requireSameDtype` is the single axis on which this family's members disagree: the in-place
// callers demand src.dtype() == dst.dtype(), while BrightnessContrast may convert between dtypes and
// must not. Everything else below -- the null, layout, scalar-dtype, sample, channel, plane, size
// and column-stride rules -- is common, and replaced hand-maintained per-operator copies that had
// to be written, and fixed, separately.
//
// Channel and plane bounds stay at the call site because they differ per operator: some callers
// add their own (BrightnessContrast, AdjustContrast, AutoContrast), while the rest reach
// DispatchChannels, which admits a planar tensor of any plane count at one channel per plane.
// Folding a bound in here would silently change the callers that do not want one.
inline SrcDstTensorInfo ValidateSrcDstTensors(const nvcv::Optional<nvcv::TensorDataStridedCuda> &srcData,
                                              const nvcv::Optional<nvcv::TensorDataStridedCuda> &dstData,
                                              bool requireSameDtype = true)
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
    // Layout is checked before TensorDataAccessStridedImagePlanar::Create, which fails outright for
    // a layout outside the allow-list below.
    if (srcData->layout() != dstData->layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same layout");
    }
    if (!(srcData->layout() == nvcv::TENSOR_HWC || srcData->layout() == nvcv::TENSOR_NHWC
          || srcData->layout() == nvcv::TENSOR_CHW || srcData->layout() == nvcv::TENSOR_NCHW))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must have (N)HWC or (N)CHW layout");
    }
    if (requireSameDtype && srcData->dtype() != dstData->dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output must have the same data type");
    }
    // BUG FIX (not a refactor): reject a tensor whose *dtype* carries the channels.
    //
    // Everything downstream of this validator takes the channel count from the tensor's C axis:
    // TensorShapeInfoImage::numChannels() reads shape[idxChannel()] and nothing else, while
    // DispatchBaseType deliberately maps TYPE_2/3/4_x onto the same base type as TYPE_x. Nothing
    // ties the two together, so a tensor spelled {1, H, W, 1} with dtype TYPE_3U8 dispatches as
    // uchar1: elem_stream::MakePlan then takes a row to be numCols * 1 bytes when the real row is
    // numCols * 3, and the operator transforms the first third of every row and silently leaves
    // the rest. Tensor::CalcRequirements has no dtype-vs-C-axis consistency check, so that tensor
    // allocates (colStride == dtype.strideBytes() == 3, which CheckPackedColumns below then
    // accepts because numInterleavedChannels is 1) and reaches here.
    //
    // Both sides are tested, not just src: with requireSameDtype the dst test is implied by the
    // src one, and without it (BrightnessContrast) the dst dtype is an independent input.
    //
    // The dtype arms this guards are still load-bearing on the var-shape path, where
    // ValidateSrcDstVarBatch takes the channel count from the dtype itself and an interleaved RGB8
    // image legitimately reports planeDataType(0) == TYPE_3U8 -- hence a tensor-only rejection.
    //
    // No image-format-derived tensor is affected: Tensor::CalcRequirements builds the tensor dtype
    // from the format's first *channel* packing (swizzle X000, all but the first bits cleared), so
    // it is always single-channel.
    if (srcData->dtype().numChannels() != 1 || dstData->dtype().numChannels() != 1)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Tensor data type must be scalar; use the C dimension for image channels");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    NVCV_ASSERT(srcAccess && dstAccess);

    SrcDstTensorInfo info{};

    info.numSamples = srcAccess->numSamples();
    if (info.numSamples != dstAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of samples");
    }

    info.numChannels = srcAccess->numChannels();
    if (info.numChannels != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    info.numPlanes = srcAccess->numPlanes();
    if (info.numPlanes != dstAccess->numPlanes())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of planes");
    }
    if (info.numPlanes > 1 && info.numChannels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "2-channel planar images are not supported");
    }

    if (srcAccess->numCols() != dstAccess->numCols() || srcAccess->numRows() != dstAccess->numRows())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have matching width and height");
    }

    info.srcDtype = srcData->dtype();
    info.dstDtype = dstData->dtype();
    // A planar layout addresses one channel at a time, so its pixel is one element wide however
    // many planes there are. numInterleavedChannels > 1 therefore implies a channel-last layout,
    // which TensorShapeInfoImagePlanar always reports as exactly one plane.
    info.numInterleavedChannels = srcAccess->infoLayout().isChannelLast() ? info.numChannels : 1;

    CheckPackedColumns(*srcAccess, info.numInterleavedChannels, info.srcDtype, "Input");
    CheckPackedColumns(*dstAccess, info.numInterleavedChannels, info.dstDtype, "Output");

    return info;
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

// Launch limits ---------------------------------------------------------------------------

// Every kernel in this family maps one sample (or one plane of one sample) to grid.z, which CUDA
// caps at kMaxGridYZ, and wraps the tensors with a 32-bit stride type, which caps the addressable
// span at INT32_MAX. Both are launch preconditions rather than API contract violations, so they are
// checked once up front instead of being re-derived by each candidate path.
//
// The two halves are written out separately so the two distinct failure modes -- an oversized batch
// (ERROR_INVALID_ARGUMENT) and an oversized addressable span (ERROR_OVERFLOW) -- and the order they
// are reported in stay explicit. No operator takes one half alone today; BrightnessContrast, which
// used to take only the stride half, was given the grid one as declared hardening.
inline void CheckTensorGridLimit(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess)
{
    if (srcAccess.numSamples() > elem_stream::kMaxGridYZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }
}

// Largest byte offset a wrap over this tensor can form, over the four axes the wraps index.
//
// An index i on an axis contributes i * stride, so the largest term an axis contributes is
// (extent - 1) * stride, not extent * stride: an axis of extent 1 is only ever indexed at 0 and
// contributes nothing. That distinction is not cosmetic here, because
// nvcv::priv::ValidateTensorBufferStrided skips any axis of extent 1 when it pins the packed and
// the monotonic strides, so such an axis' stride is completely unconstrained -- counting it at
// full stride rejects, say, an NCHW {1, 1, H, W} tensor carrying a huge unused C stride, which the
// wraps never add to any address.
//
// Summing the four terms is the exact largest offset over those axes. It is still a lower bound on
// the largest offset overall, because an interleaved tensor's channel axis is not one of the four
// (its span is covered by the column term only up to a packed pixel); under-approximating is the
// safe direction, since it can only fail to reject, never reject a tensor that fits.
//
// Every term has to be here because the axis that carries the size is layout-dependent. Reading
// only the sample term (which is what this check did before) is vacuous for a rank-3 HWC or CHW
// tensor: TensorDataAccessStridedImagePlanar::sampleStride() returns 0 when the layout has no
// sample axis and numSamples() reports 1, so the product is 0 at any tensor size, and an HWC
// tensor whose rows span more than INT32_MAX then wrapped its int32 offsets silently.
inline int64_t TensorSpanBound(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t sampleSpan = access.sampleStride() * (access.numSamples() - 1);
    const int64_t planeSpan  = access.planeStride() * (access.numPlanes() - 1);
    const int64_t rowSpan    = access.rowStride() * (access.numRows() - 1);
    const int64_t colSpan    = access.colStride() * (access.numCols() - 1);
    return sampleSpan + planeSpan + rowSpan + colSpan;
}

// This bound changed shape in *both* directions when the five operators adopted it. It is not a
// monotone tightening of the `sampleStride() * numSamples()` test it replaced, and which way it
// moves depends on the tensor:
//
//  - It tightens wherever the span inside one sample exceeds that old product. The extreme case is
//    a rank-3 HWC or CHW tensor, whose sampleStride() is defined as 0, making the old product
//    identically 0 at any size (see TensorSpanBound): an HWC tensor whose rows alone span more than
//    INT32_MAX was accepted and then wrapped its int32 offsets. That is the bug fix, and
//    TestOpInvert.cpp's `tensor_hwc_row_span_exceeds_int32` pins it.
//  - It relaxes wherever that span is below the old product, because the sample term is now
//    sampleStride * (numSamples - 1) and not * numSamples. The margin is one column stride for a
//    dense batch and up to a whole sample stride otherwise -- for a single-sample tensor the sample
//    stride drops out entirely, however large it is. A tensor overshooting INT32_MAX by less than
//    that margin is now accepted where it used to be rejected, and TestOpInvert.cpp's
//    `tensor_unused_extent1_stride_not_rejected` pins that direction.
//
// The relaxation is sound for every int32 consumer downstream, not merely believed to be:
// cuda::TensorWrapT::doGetOffset accumulates `index * stride` in StrideT over exactly the axes
// this function sums, all terms non-negative, so the final offset is also the largest partial sum
// -- bounding the total bounds every intermediate. The element width of the access at that offset
// is added afterwards, in 64-bit pointer arithmetic, so it does not need to fit. What the bound
// deliberately does not cover is an interleaved tensor's channel axis, which is not one of the
// four; CheckPackedColumns pins the column stride to exactly one packed pixel, so the wraps'
// `sizeof(ValueT)` column step is the colStride term and the only uncounted offset is the trailing
// channels of the very last pixel, again resolved in 64-bit pointer arithmetic.
inline void CheckTensorStrideLimit(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                   const nvcv::TensorDataAccessStridedImagePlanar &dstAccess)
{
    if (std::max(TensorSpanBound(srcAccess), TensorSpanBound(dstAccess)) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_OVERFLOW, "Input or output size exceeds %d. Tensor is too large.",
                              cuda::TypeTraits<int32_t>::max);
    }
}

// Order matters: callers of the combined form report an oversized batch as ERROR_INVALID_ARGUMENT
// before they report an oversized span as ERROR_OVERFLOW.
inline void CheckTensorGridAndStrideLimits(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                           const nvcv::TensorDataAccessStridedImagePlanar &dstAccess)
{
    CheckTensorGridLimit(srcAccess);
    CheckTensorStrideLimit(srcAccess, dstAccess);
}

// Var-shape counterpart: one grid.z block per image. There is no stride companion here because the
// var-shape wrappers carry per-image strides rather than one batch-wide span.
inline void CheckVarShapeGridLimit(const nvcv::ImageBatchVarShapeDataStridedCuda &dstData)
{
    if (dstData.numImages() > elem_stream::kMaxGridYZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Batch size exceeds the CUDA grid.z limit of 65535");
    }
}

// BUG FIX (not a refactor): grid.y is 16-bit exactly as grid.z is, and every kernel in this family
// except the element stream maps image rows straight onto grid.y with no in-kernel row loop, so a
// tall enough image built an illegal launch configuration and surfaced as a raw
// cudaErrorInvalidConfiguration. With the family's 32-row-per-block-row shapes the ceiling is
// 65535 * blockRows rows, e.g. 262140 for a block of 4 rows.
//
// Unlike the batch bound this must not be checked up front, because the height a tensor may have
// depends on which path serves it: elem_stream::MakePlan folds a packed tensor's rows into one run
// and is then bounded by numRegions == 1, not by height, so an NHWC U8 {1, 300000, 16, 3} tensor
// with packed rows launches a legal grid today and has to keep doing so. The bounded paths
// therefore decline (FitsRowGridLimit), leaving the work to a path that may still fold the rows
// away, and only the scalar fallback -- which has nowhere left to fall -- throws.
inline bool FitsRowGridLimit(int numRows, int blockRows)
{
    return util::DivUp(numRows, blockRows) <= elem_stream::kMaxGridYZ;
}

inline void CheckRowGridLimit(int numRows, int blockRows)
{
    if (!FitsRowGridLimit(numRows, blockRows))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Image height exceeds the CUDA grid.y limit of 65535 blocks");
    }
}

// The launch grid of this family's row-per-grid.y kernels, with that bound enforced. Call it at
// the launch, not at the top of the operator: see FitsRowGridLimit for why the check has to be
// late. `gridZ` is not re-checked here -- CheckTensorGridLimit / CheckVarShapeGridLimit own it.
inline dim3 MakeCheckedGrid(int2 size, dim3 block, unsigned int gridZ)
{
    CheckRowGridLimit(size.y, static_cast<int>(block.y));
    return dim3(static_cast<unsigned int>(util::DivUp(size.x, static_cast<int>(block.x))),
                static_cast<unsigned int>(util::DivUp(size.y, static_cast<int>(block.y))), gridZ);
}

// The whole var-shape launch preamble, which was byte-identical in all five operators: bound the
// image count against grid.z, then carry the batch's bounding size and that grid.z for the paths
// below to size their own grids from. `scalarGrid` is a member rather than a stored dim3 because
// MakeCheckedGrid throws on a too-tall batch and the wide paths tried first may still serve one;
// it must not be evaluated until the scalar kernel is the chosen path (see FitsRowGridLimit).
struct VarShapeLaunchPrep
{
    int2         maxSize; // bounding {width, height} over the batch
    unsigned int gridZ;   // one block per image

    dim3 scalarGrid(dim3 block) const
    {
        return MakeCheckedGrid(maxSize, block, gridZ);
    }
};

inline VarShapeLaunchPrep MakeVarShapeLaunchPrep(const nvcv::ImageBatchVarShapeDataStridedCuda &dstData)
{
    CheckVarShapeGridLimit(dstData);
    return {
        int2{dstData.maxSize().w, dstData.maxSize().h},
        static_cast<unsigned int>(dstData.numImages())
    };
}

// Tensor counterpart of VarShapeLaunchPrep, and the same preamble every tensor path in this family
// opened with: bound the batch against grid.z and the addressable span against int32, then carry
// the image size and that grid.z for the paths below to size their own grids from.
//
// `scalarGrid` is a member rather than a stored dim3 for the same reason as there: MakeCheckedGrid
// throws on a too-tall tensor and the wide or streaming paths tried first may still serve one, so
// it must not be evaluated until the row-per-grid.y kernel is the chosen path (see
// FitsRowGridLimit). Having one object own that rule replaces five copies of a comment saying so.
struct TensorLaunchPrep
{
    int2         size;  // {width, height} of one sample
    unsigned int gridZ; // one block per sample

    dim3 scalarGrid(dim3 block) const
    {
        return MakeCheckedGrid(size, block, gridZ);
    }
};

inline TensorLaunchPrep MakeTensorLaunchPrep(const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                             const nvcv::TensorDataAccessStridedImagePlanar &dstAccess)
{
    CheckTensorGridAndStrideLimits(srcAccess, dstAccess);
    return {cuda::StaticCast<int>(long2{srcAccess.numCols(), srcAccess.numRows()}),
            static_cast<unsigned int>(srcAccess.numSamples())};
}

// Tensor wrap pairs -----------------------------------------------------------------------

// The planar (sample, plane, row, column) wrap pair, which was spelled out character-for-character
// at every planar tensor launch in the family. Src and dst value types are separate parameters
// because BrightnessContrast alone may convert between dtypes.
template<typename SrcValueT, typename DstValueT, typename StrideT>
struct PlanarWrapPair
{
    cuda::Tensor4DWrap<const SrcValueT, StrideT> src;
    cuda::Tensor4DWrap<DstValueT, StrideT>       dst;
};

// The int32 narrowing here is what CheckTensorStrideLimit guards, and callers reach this only
// after MakeTensorLaunchPrep has run it. That check bounds each axis by (extent - 1) * stride, so
// what it proves is that every offset the wraps *form* fits in int32; the stride of an axis of
// extent 1 is narrowed here without being bounded, which is harmless precisely because such an
// axis is only ever indexed at 0 and its narrowed value is never added to an address.
template<typename SrcValueT, typename DstValueT, typename StrideT>
inline PlanarWrapPair<SrcValueT, DstValueT, StrideT> MakePlanarWrapPair(
    const nvcv::TensorDataStridedCuda &srcData, const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
    const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataAccessStridedImagePlanar &dstAccess)
{
    return {cuda::Tensor4DWrap<const SrcValueT, StrideT>(srcData.basePtr(), static_cast<int>(srcAccess.sampleStride()),
                                                         static_cast<int>(srcAccess.planeStride()),
                                                         static_cast<int>(srcAccess.rowStride())),
            cuda::Tensor4DWrap<DstValueT, StrideT>(dstData.basePtr(), static_cast<int>(dstAccess.sampleStride()),
                                                   static_cast<int>(dstAccess.planeStride()),
                                                   static_cast<int>(dstAccess.rowStride()))};
}

// Vec4 wide-load tensor paths -------------------------------------------------------------

// A Vec4 kernel loads four consecutive base-type elements at once, so every address it forms --
// both base pointers and each stride it may add -- has to be a whole number of Vec4s. Interleaved
// callers have no plane axis and pass their sample stride for the plane stride, which is exactly
// what they index with, so the extra conjunct is implied by the sample-stride one.
template<typename BT>
inline bool IsVec4Aligned(const void *srcBase, const void *dstBase, int64_t srcSampleStride, int64_t srcPlaneStride,
                          int64_t srcRowStride, int64_t dstSampleStride, int64_t dstPlaneStride, int64_t dstRowStride)
{
    using Vec4 = cuda::MakeType<BT, 4>;
    return reinterpret_cast<uintptr_t>(srcBase) % sizeof(Vec4) == 0
        && reinterpret_cast<uintptr_t>(dstBase) % sizeof(Vec4) == 0 && srcSampleStride % sizeof(Vec4) == 0
        && srcPlaneStride % sizeof(Vec4) == 0 && srcRowStride % sizeof(Vec4) == 0 && dstSampleStride % sizeof(Vec4) == 0
        && dstPlaneStride % sizeof(Vec4) == 0 && dstRowStride % sizeof(Vec4) == 0;
}

template<typename BT, typename StrideT>
struct Vec4TensorLaunch
{
    cuda::Tensor4DWrap<const BT, StrideT> src;
    cuda::Tensor4DWrap<BT, StrideT>       dst;
    dim3                                  grid;
};

// The wrap pair and grid the Vec4 kernels launch with. `NGroups` is a template argument and never
// derived here on purpose: the family does not agree on it (1-byte planar uses 4 while 4-byte
// planar uses 2), so deriving it would silently change some callers' launch shape.
template<int NGroups, typename BT, typename StrideT>
inline Vec4TensorLaunch<BT, StrideT> MakeVec4TensorLaunch(const void *srcBase, void *dstBase, int64_t srcSampleStride,
                                                          int64_t srcPlaneStride, int64_t srcRowStride,
                                                          int64_t dstSampleStride, int64_t dstPlaneStride,
                                                          int64_t dstRowStride, int2 size, dim3 block,
                                                          unsigned int gridZ)
{
    return {cuda::Tensor4DWrap<const BT, StrideT>(srcBase, static_cast<int>(srcSampleStride),
                                                  static_cast<int>(srcPlaneStride), static_cast<int>(srcRowStride)),
            cuda::Tensor4DWrap<BT, StrideT>(dstBase, static_cast<int>(dstSampleStride),
                                            static_cast<int>(dstPlaneStride), static_cast<int>(dstRowStride)),
            dim3(util::DivUp(util::DivUp(size.x, 4), static_cast<int>(block.x) * NGroups), util::DivUp(size.y, block.y),
                 gridZ)};
}

// Marker parameter for an Op whose whole state is derived on the device from BT alone (Invert's
// photometric bound). It is the kernel's trailing parameter, and parameters are laid out in
// declaration order, so an empty one cannot move the parameters ahead of it.
//
// One definition, spelled from both namespaces: elem_stream owns it because that header is the
// lower layer (byte runs, no nvcv image or tensor-wrap types), so this header may depend on it
// while the reverse would invert the layering.
using NoParam = elem_stream::NoParam;

// The wide-load tensor kernel of this family. Each (sample, plane) maps to grid.z -- no per-plane
// loop -- and each thread owns NGROUP independent 4-column groups, issuing every wide load before
// any compute so a warp has several memory requests outstanding; that is what removes the scalar
// kernels' long-scoreboard stalls. `inout_size` is {samples, planes, height, width}; a width that
// is not a multiple of 4 falls to the per-thread scalar tail, so the caller only has to guarantee
// sizeof(Vec4)-aligned bases and strides.
//
// `Op` is the per-element map and is constructed on the device from `Op::Param`. Constructing it
// inside the kernel rather than passing a ready-made functor is deliberate: an operator whose
// state costs real work to derive (Solarize's threshold boundary, one FP64 ceil) keeps deriving it
// once per thread, in the same place and at the same time it did before this was shared.
template<int NGROUP, typename BT, class Op>
__global__ void PlanarVec4Kernel(cuda::Tensor4DWrap<const BT, int32_t> src, cuda::Tensor4DWrap<BT, int32_t> dst,
                                 int4 inout_size, typename Op::Param param)
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

    using Vec4 = cuda::MakeType<BT, 4>;
    const Op op(param);

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
            out4.x                                                           = op(in4[i].x);
            out4.y                                                           = op(in4[i].y);
            out4.z                                                           = op(in4[i].z);
            out4.w                                                           = op(in4[i].w);
            *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch, channel, src_y, x) = op(*src.ptr(batch, channel, src_y, x));
            }
        }
    }
}

// Host side of the same path: decide whether the wide kernel may run and, if so, launch it.
// `launchFn` receives the wrap pair, the {samples, planes, height, width} shape and the grid, and
// does nothing but the launch -- the operator-specific kernel argument stays at the call site.
//
// The dtype/channel predicate that decides whether an arm is eligible at all stays at the call
// site too, and deliberately: the family does not agree on it (Invert and Solarize keep __half on
// the scalar kernel, Posterize does not, and NGroups differs by dtype), so folding it in here
// would silently move operators between paths.
template<int NGroups, typename BT, typename StrideT, class LaunchFn>
inline bool TryRunVec4Tensor(bool isPlanar, const void *srcBase, void *dstBase,
                             const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                             const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int2 size, dim3 block,
                             const LaunchFn &launchFn)
{
    const int     numSamples = static_cast<int>(srcAccess.numSamples());
    const int     numPlanes  = isPlanar ? srcAccess.numPlanes() : 1;
    const int64_t gridZ      = static_cast<int64_t>(numSamples) * numPlanes;

    const int64_t sStride = srcAccess.sampleStride(), rStride = srcAccess.rowStride();
    const int64_t dsStride = dstAccess.sampleStride(), drStride = dstAccess.rowStride();
    // An interleaved caller here is C == 1, so the kernel's channel index is always 0 and the plane
    // stride is never added; passing the sample stride keeps the alignment test on what is indexed.
    const int64_t pStride  = isPlanar ? srcAccess.planeStride() : sStride;
    const int64_t dpStride = isPlanar ? dstAccess.planeStride() : dsStride;

    // Each (sample, plane) is its own grid.z block here, so the cap applies to the product. For an
    // interleaved caller the product is numSamples, which CheckTensorGridLimit has already rejected
    // above that cap, so this restates a decision rather than adding one.
    if (gridZ > elem_stream::kMaxGridYZ)
    {
        return false;
    }
    // Rows map straight onto grid.y here as well; declining leaves a taller tensor to a path that
    // can fold its rows away, and to CheckRowGridLimit at the scalar fallback if none can.
    if (!FitsRowGridLimit(size.y, static_cast<int>(block.y)))
    {
        return false;
    }
    if (!IsVec4Aligned<BT>(srcBase, dstBase, sStride, pStride, rStride, dsStride, dpStride, drStride))
    {
        return false;
    }

    auto launch
        = MakeVec4TensorLaunch<NGroups, BT, StrideT>(srcBase, dstBase, sStride, pStride, rStride, dsStride, dpStride,
                                                     drStride, size, block, static_cast<unsigned int>(gridZ));
    launchFn(launch.src, launch.dst, int4{numSamples, numPlanes, size.y, size.x}, launch.grid);
    return true;
}

// The whole launch for the common case where the only operator-specific part is `Op`. Kept separate
// from TryRunVec4Tensor rather than folded into it because the eligibility predicate that decides
// whether this arm runs at all still belongs at the call site (see above), and an operator that
// needs a different kernel on the same host gate can still call TryRunVec4Tensor directly.
template<int NGroups, typename BT, typename StrideT, class Op>
inline bool TryRunVec4TensorFor(cudaStream_t stream, bool isPlanar, const void *srcBase, void *dstBase,
                                const nvcv::TensorDataAccessStridedImagePlanar &srcAccess,
                                const nvcv::TensorDataAccessStridedImagePlanar &dstAccess, int2 size, dim3 block,
                                typename Op::Param param)
{
    return TryRunVec4Tensor<NGroups, BT, StrideT>(
        isPlanar, srcBase, dstBase, srcAccess, dstAccess, size, block,
        [&](auto vecSrc, auto vecDst, int4 shape, dim3 vecGrid)
        { PlanarVec4Kernel<NGroups, BT, Op><<<vecGrid, block, 0, stream>>>(vecSrc, vecDst, shape, param); });
}

// Vec4 wide-load var-shape paths ----------------------------------------------------------

// The shared scalar group body. The wrapper below is the outlined entry point; the kernel selects between them.
template<typename BT, class Op>
__device__ __forceinline__ void RunVarShapeVec4ScalarGroupInline(const BT *src, BT *dst, int begin, int width,
                                                                 const Op op)
{
    const int end = min(begin + 4, width);
    for (int x = begin; x < end; ++x)
    {
        dst[x] = op(src[x]);
    }
}

template<typename BT, class Op>
__device__ __noinline__ void RunVarShapeVec4ScalarGroup(const BT *src, BT *dst, int begin, int width, const Op op)
{
    RunVarShapeVec4ScalarGroupInline(src, dst, begin, width, op);
}

template<class Op, class = void>
struct HasWordApply : std::false_type
{
};

// A byte operation can opt into transforming the four loaded lanes while they remain one word.
template<class Op>
struct HasWordApply<Op, std::void_t<decltype(std::declval<const Op &>().ApplyWord(std::declval<uint32_t>()))>>
    : std::true_type
{
};

// Var-shape counterpart of PlanarVec4Kernel, on the same `Op` contract. Each (image, plane) maps to
// grid.z and each thread owns NGROUP independent 4-column groups, all loads issued before any
// compute. `numPlanes` is what grid.z is split by; an interleaved caller passes 1.
//
// Restricted by static_assert to 1- and 2-byte planes, and every wide access is gated on a per-row
// alignment test. NVCV gives no alignment guarantee for a wrapped image:
// ImageWrapData::doValidateData requires only a non-null plane base pointer and
// rowStride >= width * pixelStride, so a batch built with nvcv::ImageWrapData
// can legitimately present an odd base pointer and an odd row stride, and an unguarded 4-byte load
// on it faults with a sticky misaligned-address error. Only the allocating constructor aligns.
// The tensor kernel needs no such test because its host gate (IsVec4Aligned) has already proved
// every base and stride it will add is a whole number of Vec4s. The row test generalises to a
// 2-byte base type on its own: `alignof(Vec4)` is 8 for ushort4, which is exactly what the wide
// load needs; it is spelled in terms of Vec4 rather than a hard-coded 4.
//
// `Serial` selects the group loop's shape and is a real codegen choice, not a style one. The
// default issues every wide load before any compute, so a warp keeps NGROUP requests outstanding.
// `Serial` instead keeps one vector live at a time, which the U16 var-shape path measured as the
// better trade: unrolling all groups there raised the active register footprint above the
// optimization campaign's automatic memory allowance without improving coalescing.
// `IsolateScalarFallback` outlines all groups of an unaligned row and the final group of each
// ragged-width row from the non-serial kernel. For allocator-aligned Invert U8 planar VarShape,
// this reduces the hot kernel's register footprint without changing its vector/scalar dispatch.
// Performance of unaligned (`wide == false`) rows, where every group is outlined, has not been
// measured separately. Combining it with `Serial` is intentionally unsupported because that
// combination is also unmeasured and has no caller. Isolation is limited to the measured SM80/SM90
// paths and the SM100 target; other architectures retain the inline body. In particular, paired
// SM120 measurements showed that isolation regresses its aligned path.
template<int NGROUP, typename BT, class Op, bool Serial = false, bool IsolateScalarFallback = false>
__global__ void VarShapeVec4Kernel(cuda::ImageBatchVarShapeWrap<const BT> src, cuda::ImageBatchVarShapeWrap<BT> dst,
                                   int numPlanes, typename Op::Param param)
{
    static_assert(sizeof(BT) <= 2, "VarShapeVec4Kernel assumes a 1- or 2-byte base type");
    static_assert(!Serial || !IsolateScalarFallback,
                  "IsolateScalarFallback is supported only by the non-serial VarShapeVec4Kernel");

#if defined(__CUDA_ARCH__)
    constexpr bool kIsolateScalarFallback
        = IsolateScalarFallback && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000);
#else
    constexpr bool kIsolateScalarFallback = IsolateScalarFallback;
#endif

    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / numPlanes;
    const int channel = nc % numPlanes;
    const int width   = dst.width(batch, channel);

    if (g0 * 4 >= width || dst_y >= dst.height(batch, channel))
    {
        return;
    }

    constexpr bool kWordApply = HasWordApply<Op>::value;
    static_assert(!kWordApply || sizeof(BT) == 1, "ApplyWord must map four one-byte elements");
    using Vec4 = std::conditional_t<kWordApply, uint32_t, cuda::MakeType<BT, 4>>;
    const Op op(param);

    // BUG FIX (not a refactor): `wide` is new for the 1-byte *planar* instantiations. The three
    // kernels this one replaces there -- InvertPlanarVarShapeVec4Kernel,
    // PosterizePlanarVarShapeVec4Kernel and SolarizePlanarVarShapeVec4Kernel -- dereferenced
    // `src.ptr(...)` as a uchar4 with no alignment test at all, so a planar U8 batch whose plane
    // base or row stride is odd (which nvcv::ImageWrapData accepts, see above) faulted with a
    // sticky misaligned-address error instead of producing a result. The interleaved 2-byte
    // instantiation already carried this test; generalising it is what fixes the planar ones.
    //
    // Resolved once per thread, as Invert's packed-RGB8 var-shape kernel also does: one image-list
    // lookup for the whole row, and one alignment test the whole warp agrees on for an
    // allocator-produced batch, where `wide` is uniformly true and the scalar arm is unreachable.
    const BT  *srow = src.ptr(batch, channel, dst_y, 0);
    BT        *drow = dst.ptr(batch, channel, dst_y, 0);
    const bool wide = reinterpret_cast<uintptr_t>(srow) % alignof(Vec4) == 0
                   && reinterpret_cast<uintptr_t>(drow) % alignof(Vec4) == 0;

    if constexpr (Serial)
    {
#pragma unroll 1
        for (int i = 0; i < NGROUP; ++i)
        {
            const int cx = (g0 + i * blockDim.x) * 4;
            if (wide && cx + 4 <= width)
            {
                const Vec4 in4 = *reinterpret_cast<const Vec4 *>(srow + cx);
                Vec4       out4;
                if constexpr (kWordApply)
                {
                    out4 = op.ApplyWord(in4);
                }
                else
                {
                    out4.x = op(in4.x);
                    out4.y = op(in4.y);
                    out4.z = op(in4.z);
                    out4.w = op(in4.w);
                }
                *reinterpret_cast<Vec4 *>(drow + cx) = out4;
            }
            // Bounded by the group rather than by the row: when `wide` is false a full group also
            // lands here, and walking to `width` would make every group redo the whole row's tail.
            else if (cx < width)
            {
                RunVarShapeVec4ScalarGroupInline(srow, drow, cx, width, op);
            }
        }
    }
    else
    {
        int  cx[NGROUP];
        bool full[NGROUP];
        Vec4 in4[NGROUP];
#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            cx[i]   = (g0 + i * blockDim.x) * 4;
            full[i] = wide && cx[i] + 4 <= width;
            if (full[i])
            {
                in4[i] = *reinterpret_cast<const Vec4 *>(srow + cx[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            if (full[i])
            {
                Vec4 out4;
                if constexpr (kWordApply)
                {
                    out4 = op.ApplyWord(in4[i]);
                }
                else
                {
                    out4.x = op(in4[i].x);
                    out4.y = op(in4[i].y);
                    out4.z = op(in4[i].z);
                    out4.w = op(in4[i].w);
                }
                *reinterpret_cast<Vec4 *>(drow + cx[i]) = out4;
            }
            // Bounded by the group rather than by the row: when `wide` is false a full group also
            // lands here, and walking to `width` would make every group redo the whole row's tail.
            else if (cx[i] < width)
            {
                if constexpr (kIsolateScalarFallback)
                {
                    RunVarShapeVec4ScalarGroup(srow, drow, cx[i], width, op);
                }
                else
                {
                    RunVarShapeVec4ScalarGroupInline(srow, drow, cx[i], width, op);
                }
            }
        }
    }
}

// Host side of the var-shape wide path, mirroring TryRunVec4Tensor: decide whether the wide kernel
// may run and, if so, hand the wrap pair and grid to `launchFn`. As there, the dtype/channel
// eligibility predicate stays at the call site.
template<int NGroups, typename BT, class LaunchFn>
inline bool TryRunVec4VarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                               const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, int numPlanes, int2 maxSize,
                               dim3 block, const LaunchFn &launchFn)
{
    // Each (image, plane) is its own grid.z block, so the cap applies to the product; the caller's
    // CheckVarShapeGridLimit only bounds the image count.
    const int64_t planes = static_cast<int64_t>(dstData.numImages()) * numPlanes;
    if (planes > elem_stream::kMaxGridYZ)
    {
        return false;
    }
    if (!FitsRowGridLimit(maxSize.y, static_cast<int>(block.y)))
    {
        return false;
    }

    cuda::ImageBatchVarShapeWrap<const BT> srcV(srcData);
    cuda::ImageBatchVarShapeWrap<BT>       dstV(dstData);
    dim3 vgrid(util::DivUp(util::DivUp(maxSize.x, 4), static_cast<int>(block.x) * NGroups),
               util::DivUp(maxSize.y, block.y), static_cast<unsigned int>(planes));
    launchFn(srcV, dstV, numPlanes, vgrid);
    return true;
}

// Var-shape counterpart of TryRunVec4TensorFor: the whole launch when the only operator-specific
// part is `Op`. Split from TryRunVec4VarShape for the same reason -- the eligibility predicate
// stays at the call site, and an operator needing a different kernel on the same gate can still
// call TryRunVec4VarShape directly.
template<int NGroups, typename BT, class Op, bool Serial = false, bool IsolateScalarFallback = false>
inline bool TryRunVec4VarShapeFor(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &srcData,
                                  const nvcv::ImageBatchVarShapeDataStridedCuda &dstData, int numPlanes, int2 maxSize,
                                  dim3 block, typename Op::Param param)
{
    return TryRunVec4VarShape<NGroups, BT>(srcData, dstData, numPlanes, maxSize, block,
                                           [&](auto srcV, auto dstV, int planes, dim3 vgrid)
                                           {
                                               VarShapeVec4Kernel<NGroups, BT, Op, Serial, IsolateScalarFallback>
                                                   <<<vgrid, block, 0, stream>>>(srcV, dstV, planes, param);
                                           });
}

} // namespace cvcuda::priv::same_shape

#endif // CVCUDA_PRIV_SAME_SHAPE_COMMON_CUH
