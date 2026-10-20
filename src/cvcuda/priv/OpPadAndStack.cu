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

#include "Nvtx.hpp"
#include "OpPadAndStack.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"

#include <cuda_fp16.h>
#include <cvcuda/cuda_tools/BorderVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <cstdint>
#include <cstring>
#include <optional>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

// int32 output strides keep the destination address arithmetic in 32-bit registers; the size guard
// in InterleavedLauncher::Launch is the precondition that makes the narrowing safe.
using StrideType = int32_t;

// 8-bit signed is deliberately absent: it is rejected by the dtype gate before dispatch is reached.
enum class ElementKind
{
    U8,
    U16,
    S16,
    S32,
    F16,
    F32
};

// Dispatch keys on data kind plus bits-per-channel, never on the exact dtype, so a packed
// multi-channel type classifies like its scalar sibling -- TYPE_3U8 and TYPE_U8 are both "8-bit
// unsigned". That is what makes the channel count an independent dispatch axis.
struct ElementClass
{
    nvcv::DataKind kind;
    int            bits;
};

// Two-stage dtype rejection, and the stages are not interchangeable. An unclassifiable (kind, width)
// pair must be rejected here, ahead of the border-mode check; the narrower set this operator
// actually accepts is checked later, after it. Merging the stages would change which error a
// doubly-invalid argument reports.
inline void ValidateClassifiable(const ElementClass &elem, const char *what)
{
    bool classifiable = false;
    switch (elem.kind)
    {
    case nvcv::DataKind::FLOAT:
        classifiable = (elem.bits == 16 || elem.bits == 32 || elem.bits == 64);
        break;
    case nvcv::DataKind::SIGNED:
        classifiable = (elem.bits == 8 || elem.bits == 16 || elem.bits == 32);
        break;
    case nvcv::DataKind::UNSIGNED:
        classifiable = (elem.bits == 8 || elem.bits == 16);
        break;
    default:
        break;
    }

    if (!classifiable)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Unsupported %s data kind or bit depth (%d bits per channel)", what, elem.bits);
    }
}

inline ElementClass ClassifyElement(const nvcv::DataType &dtype, const char *what)
{
    const auto bitsPerChannel = dtype.bitsPerChannel();
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bitsPerChannel[i] != bitsPerChannel[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "All channels of the %s must have the same bit-depth", what);
        }
    }

    ElementClass elem{dtype.dataKind(), bitsPerChannel[0]};
    ValidateClassifiable(elem, what);
    return elem;
}

inline std::optional<ElementKind> ToElementKind(const ElementClass &elem)
{
    switch (elem.kind)
    {
    case nvcv::DataKind::UNSIGNED:
        if (elem.bits == 8)
        {
            return ElementKind::U8;
        }
        if (elem.bits == 16)
        {
            return ElementKind::U16;
        }
        break;
    case nvcv::DataKind::SIGNED:
        if (elem.bits == 16)
        {
            return ElementKind::S16;
        }
        if (elem.bits == 32)
        {
            return ElementKind::S32;
        }
        break;
    case nvcv::DataKind::FLOAT:
        if (elem.bits == 16)
        {
            return ElementKind::F16;
        }
        if (elem.bits == 32)
        {
            return ElementKind::F32;
        }
        break;
    default:
        break;
    }
    return std::nullopt;
}

inline bool IsPlanarLayout(const nvcv::TensorLayout &layout, const char *what)
{
    if (layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
    {
        return true;
    }
    if (layout == nvcv::TENSOR_NHWC || layout == nvcv::TENSOR_HWC)
    {
        return false;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s tensor layout not supported", what);
}

// Only the planar/interleaved split matters here; unlike the legacy DataFormat enum this replaces,
// the batched/unbatched distinction is never consulted.
inline bool IsPlanarBatch(const nvcv::ImageFormat &fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    if (fmt.numPlanes() < 2)
    {
        return false;
    }

    if (fmt.numPlanes() != fmt.numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
    }
    return true;
}

// The dtype is classified before the layout is checked; that order is observable when both are
// invalid.
inline void ValidatePaddingTensor(const nvcv::TensorDataStridedCuda &data, const char *what, int requiredSamples)
{
    const ElementClass elem = ClassifyElement(data.dtype(), what);
    if (IsPlanarLayout(data.layout(), what))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s DataFormat", what);
    }
    if (!(elem.kind == nvcv::DataKind::SIGNED && elem.bits == 32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s DataType", what);
    }
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(data);
    if (!access)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s tensor", what);
    }
    // The kernels read this as topVec.ptr(0, 0, sample), so the per-sample values run along the
    // width axis - callers pass a 1 x N row - and it is numCols that has to cover the batch.
    if (access->numCols() < requiredSamples)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "%s must provide one value per output sample (%d)",
                              what, requiredSamples);
    }
}

// util::DivUp replaces the legacy float-based divUp (ceil((float)a / b)). They agree for every
// destination extent below 2^24; above it the float form can round an exact-looking quotient down
// and under-launch, leaving the last columns unwritten.
inline dim3 PadAndStackGrid(const dim3 &block, int2 dstSize, int numSamples)
{
    return dim3(util::DivUp(dstSize.x, static_cast<int>(block.x)), util::DivUp(dstSize.y, static_cast<int>(block.y)),
                numSamples);
}

template<typename T>
constexpr int kPadNIX = cuda::NumElements<T> == 1 && sizeof(T) <= sizeof(uint16_t) ? sizeof(uint2) / sizeof(T) : 1;

template<int NIX, class SrcWrapper, typename T>
__device__ void PadAndStackRow(SrcWrapper src, T *dstRow, int dstWidth, int sy, int sample, int plane, int left)
{
    using Pack        = uint2;
    const auto &raw   = src.imageBatchWrap();
    const int   width = raw.width(sample, plane);
    const bool  inRow = sy >= 0 && sy < raw.height(sample, plane);
    const auto *row   = inRow ? raw.ptr(sample, plane, sy, 0) : nullptr;
    uintptr_t   align = reinterpret_cast<uintptr_t>(dstRow);
    if (inRow)
        align |= reinterpret_cast<uintptr_t>(row) - static_cast<uintptr_t>(left) * sizeof(T);
    const int blockX = blockIdx.x * blockDim.x * NIX;
    bool      packed = (align & (alignof(Pack) - 1)) == 0;
    if constexpr (SrcWrapper::kBorderType != NVCV_BORDER_CONSTANT)
    {
        const int tileWidth = min(int(blockDim.x) * NIX, dstWidth - blockX);
        packed              = packed && inRow && blockX - left >= 0 && blockX - left <= width - tileWidth;
    }
    // Edge and unaligned rows keep neighboring scalar pixels in neighboring lanes.
    const int step = packed ? 1 : blockDim.x;
    const int x    = blockX + threadIdx.x * (packed ? NIX : 1);
    const int sx   = x - left;
    if (x >= dstWidth)
        return;
    if (packed && NIX <= dstWidth - x)
    {
        if (inRow && sx >= 0 && sx <= width - NIX)
        {
            *reinterpret_cast<Pack *>(dstRow + x) = *reinterpret_cast<const Pack *>(row + sx);
            return;
        }
        if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
        {
            if (!inRow || sx <= -NIX || sx >= width)
            {
                const auto   border = src.borderValue();
                unsigned int bits   = 0;
                std::memcpy(&bits, &border, sizeof(T));
                if constexpr (sizeof(T) == 1)
                    bits *= 0x01010101u;
                *reinterpret_cast<Pack *>(dstRow + x) = cuda::SetAll<Pack>(bits);
                return;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NIX; ++i)
        if (i * step < dstWidth - x)
        {
            const int srcX = sx + i * step;
            if (inRow && srcX >= 0 && srcX < width)
                dstRow[x + i * step] = row[srcX];
            else
            {
                const int4 in{srcX, sy, plane, sample};
                if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
                    dstRow[x + i * step] = src.value(in);
                else
                    dstRow[x + i * step] = src[in];
            }
        }
}

template<int NIX, class SrcWrapper, class DstWrapper, class VecWrapper>
__global__ void PadAndStackKernel(SrcWrapper src, DstWrapper dst, VecWrapper topVec, VecWrapper leftVec, int2 dstSize)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const int top  = *topVec.ptr(0, 0, dstCoord.z);
    const int left = *leftVec.ptr(0, 0, dstCoord.z);

    int3 srcCoord = {dstCoord.x - left, dstCoord.y - top, dstCoord.z};
    if constexpr (NIX > 1)
    {
        if (dstCoord.y < dstSize.y)
            PadAndStackRow<NIX>(src, dst.ptr(dstCoord.z, dstCoord.y, 0), dstSize.x, dstCoord.y - top, dstCoord.z, 0,
                                left);
    }
    else if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT)
            dst[dstCoord] = src.value(srcCoord);
        else
            dst[dstCoord] = src[srcCoord];
    }
}

template<int NIX, class SrcWrapper, typename T, class VecWrapper>
__global__ void PadAndStackPlanarKernel(SrcWrapper src, cuda::Tensor4DWrap<T> dst, VecWrapper topVec,
                                        VecWrapper leftVec, int2 dstSize, int channels)
{
    int3 dstCoord = cuda::StaticCast<int>(blockDim * blockIdx + threadIdx);

    const int top  = *topVec.ptr(0, 0, dstCoord.z);
    const int left = *leftVec.ptr(0, 0, dstCoord.z);

    if constexpr (NIX > 1)
    {
        if (dstCoord.y < dstSize.y)
            for (int plane = 0; plane < channels; ++plane)
                PadAndStackRow<NIX>(src, dst.ptr(dstCoord.z, plane, dstCoord.y, 0), dstSize.x, dstCoord.y - top,
                                    dstCoord.z, plane, left);
    }
    else if (dstCoord.x < dstSize.x && dstCoord.y < dstSize.y)
    {
        for (int plane = 0; plane < channels; ++plane)
        {
            const int4 srcCoord{dstCoord.x - left, dstCoord.y - top, plane, dstCoord.z};
            if constexpr (SrcWrapper::kBorderType == NVCV_BORDER_CONSTANT && cuda::NumElements<T> == 1)
                *dst.ptr(dstCoord.z, plane, dstCoord.y, dstCoord.x) = src.value(srcCoord);
            else
                *dst.ptr(dstCoord.z, plane, dstCoord.y, dstCoord.x) = src[srcCoord];
        }
    }
}

struct PadAndStackArgs
{
    cudaStream_t                                    stream;
    const nvcv::ImageBatchVarShapeDataStridedCuda  &inData;
    const nvcv::TensorDataStridedCuda              &outData;
    const nvcv::TensorDataAccessStridedImagePlanar &outAccess;
    NVCVBorderType                                  borderMode;
    float                                           borderValue;
    cuda::Tensor3DWrap<const int, int64_t>          topVec;
    cuda::Tensor3DWrap<const int, int64_t>          leftVec;
    int2                                            dstSize;
};

struct InterleavedLauncher
{
    template<typename T, NVCVBorderType B>
    static void Launch(const PadAndStackArgs &args)
    {
        cuda::BorderVarShapeWrap<const T, B> src(args.inData, cuda::SetAll<T>(args.borderValue));

        constexpr int NIX = sizeof(T) == sizeof(uint16_t) && B == NVCV_BORDER_CONSTANT ? 1 : kPadNIX<T>;
        constexpr int BX  = NIX > 1 ? 128 : 32;
        dim3          block(BX, 256 / BX);
        if constexpr (NIX > 1)
            if (util::DivUp(args.dstSize.y, static_cast<int>(block.y)) > 65535)
                block = dim3(32, 8);
        dim3 grid
            = PadAndStackGrid(block, {util::DivUp(args.dstSize.x, NIX), args.dstSize.y}, args.outAccess.numSamples());

        if (args.outAccess.sampleStride() * args.outAccess.numSamples() > cuda::TypeTraits<StrideType>::max)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output size exceeds %d. Tensor is too large.",
                                  cuda::TypeTraits<StrideType>::max);
        }

        auto dst = cuda::CreateTensorWrapNHW<T, StrideType>(args.outData);

        PadAndStackKernel<NIX><<<grid, block, 0, args.stream>>>(src, dst, args.topVec, args.leftVec, args.dstSize);

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

struct PlanarLauncher
{
    template<typename T, NVCVBorderType B>
    static void Launch(const PadAndStackArgs &args)
    {
        cuda::BorderVarShapeWrap<const T, B> src(args.inData, static_cast<T>(args.borderValue));

        // A single thread writes every channel plane of one (x, y), so the output is addressed as a
        // genuine 4D NCHW tensor rather than through the NHW wrap the interleaved path uses.
        auto dst = cuda::CreateTensorWrapNCHW<T>(args.outData);

        constexpr int NIX = B == NVCV_BORDER_CONSTANT ? 1 : kPadNIX<T>;
        constexpr int BX  = NIX > 1 ? 128 : 32;
        dim3          block(BX, (sizeof(T) == 1 ? 128 : 256) / BX);
        if constexpr (NIX > 1)
            if (util::DivUp(args.dstSize.y, static_cast<int>(block.y)) > 65535)
                block = dim3(32, sizeof(T) == 1 ? 4 : 8);
        dim3 grid
            = PadAndStackGrid(block, {util::DivUp(args.dstSize.x, NIX), args.dstSize.y}, args.outAccess.numSamples());

        PadAndStackPlanarKernel<NIX><<<grid, block, 0, args.stream>>>(src, dst, args.topVec, args.leftVec, args.dstSize,
                                                                      args.outAccess.numChannels());

        NVCV_CHECK_THROW(cudaGetLastError());
    }
};

// Exhaustive and deliberately without a default: an added NVCVBorderType should be a compile error
// here, not a runtime one. PadAndStackInto has already rejected any value outside the enum.
template<class Launcher, typename T>
inline void DispatchBorderMode(const PadAndStackArgs &args)
{
    switch (args.borderMode)
    {
    case NVCV_BORDER_CONSTANT:
        Launcher::template Launch<T, NVCV_BORDER_CONSTANT>(args);
        break;
    case NVCV_BORDER_REPLICATE:
        Launcher::template Launch<T, NVCV_BORDER_REPLICATE>(args);
        break;
    case NVCV_BORDER_REFLECT:
        Launcher::template Launch<T, NVCV_BORDER_REFLECT>(args);
        break;
    case NVCV_BORDER_WRAP:
        Launcher::template Launch<T, NVCV_BORDER_WRAP>(args);
        break;
    case NVCV_BORDER_REFLECT101:
        Launcher::template Launch<T, NVCV_BORDER_REFLECT101>(args);
        break;
    case NVCV_BORDER_CONSTANT_COVERAGE:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "NVCV_BORDER_CONSTANT_COVERAGE is not supported by PadAndStack");
    }
}

// The planar kernel walks the channel axis itself, so the scalar element type is its whole key; the
// interleaved kernel carries the channel count in its element type's component count.
// kAllowTwoChannel is false for every element class but 8-bit unsigned, keeping the two-channel
// kernels uninstantiated for element classes whose two-channel form validation already rejects.
template<typename BaseT, bool kAllowTwoChannel>
inline void RunElementRow(bool planar, int channels, const PadAndStackArgs &args)
{
    if (planar)
    {
        DispatchBorderMode<PlanarLauncher, BaseT>(args);
        return;
    }
    if (channels == 1)
    {
        DispatchBorderMode<InterleavedLauncher, cuda::MakeType<BaseT, 1>>(args);
        return;
    }
    if (channels == 3)
    {
        DispatchBorderMode<InterleavedLauncher, cuda::MakeType<BaseT, 3>>(args);
        return;
    }
    if (channels == 4)
    {
        DispatchBorderMode<InterleavedLauncher, cuda::MakeType<BaseT, 4>>(args);
        return;
    }
    if constexpr (kAllowTwoChannel)
    {
        if (channels == 2)
        {
            DispatchBorderMode<InterleavedLauncher, cuda::MakeType<BaseT, 2>>(args);
            return;
        }
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
}

// Exhaustive over ElementKind and deliberately without a default, so a new element class is a
// compile error here rather than a runtime one.
inline void RunPadAndStack(ElementKind kind, bool planar, int channels, const PadAndStackArgs &args)
{
    switch (kind)
    {
    case ElementKind::U8:
        RunElementRow<unsigned char, true>(planar, channels, args);
        return;
    case ElementKind::U16:
        RunElementRow<unsigned short, false>(planar, channels, args);
        return;
    case ElementKind::S16:
        RunElementRow<short, false>(planar, channels, args);
        return;
    case ElementKind::S32:
        RunElementRow<int, false>(planar, channels, args);
        return;
    // F16 gets real half kernels rather than aliasing the 16-bit integer ones: a CONSTANT
    // borderValue is converted float -> pixel type, which must round to half bits instead of
    // truncating to an integer that is then reinterpreted as half.
    case ElementKind::F16:
        RunElementRow<__half, false>(planar, channels, args);
        return;
    case ElementKind::F32:
        RunElementRow<float, false>(planar, channels, args);
        return;
    }
}

inline void PadAndStackInto(cudaStream_t stream, const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                            const nvcv::TensorDataStridedCuda &outData, const nvcv::TensorDataStridedCuda &top,
                            const nvcv::TensorDataStridedCuda &left, const NVCVBorderType borderMode,
                            const float borderValue)
{
    const nvcv::ImageFormat inFormat = inData.uniqueFormat();
    if (!inFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in the input varshape must all have the same format");
    }

    const bool         inPlanar  = IsPlanarBatch(inFormat);
    const ElementClass inElem    = ClassifyElement(inFormat.planeDataType(0), "input");
    const bool         outPlanar = IsPlanarLayout(outData.layout(), "Output");
    const ElementClass outElem   = ClassifyElement(outData.dtype(), "output");

    if (inPlanar != outPlanar)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input and output: one is planar, the other interleaved");
    }

    cvcuda::priv::ValidateBorderMode(borderMode);

    const std::optional<ElementKind> outKind = ToElementKind(outElem);
    if (!outKind)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output DataType");
    }
    const std::optional<ElementKind> inKind = ToElementKind(inElem);
    if (!inKind)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input DataType");
    }

    const auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid output DataFormat");
    }

    ValidatePaddingTensor(left, "Left", outAccess->numSamples());
    ValidatePaddingTensor(top, "Top", outAccess->numSamples());

    const int channels      = outAccess->numChannels();
    const int inputChannels = inFormat.numChannels();
    if (channels != inputChannels)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output channel counts must match, but got %d and %d", inputChannels, channels);
    }
    if (channels > 4 || (outPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }
    if (*outKind != *inKind)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output DataTypes must match");
    }
    // Two interleaved channels are dispatchable for 8-bit unsigned only. The test is on kind plus
    // width, not on dtype identity: a two-channel 8U tensor has dtype TYPE_2U8, so comparing
    // against TYPE_U8 would reject exactly the case the legacy table did support.
    if (!outPlanar && channels == 2 && *outKind != ElementKind::U8)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid channel number %d for this DataType, 2 channels only supported for 8bit "
                              "Unsigned",
                              channels);
    }

    // The planar kernel takes one grid-z entry per sample. Checked here rather than in the launcher
    // because it depends only on the sample count, not on the element type or border mode the
    // launcher is templated over (cf. OpRemap.cu).
    if (outPlanar && outAccess->numSamples() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar batch size %d exceeds CUDA grid z limit 65535.", outAccess->numSamples());
    }

    // grid.z is the output sample count, and each thread indexes the input batch and both padding
    // tensors at that sample. All three must therefore cover every output sample.
    if (outAccess->numSamples() != inData.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output sample count %d must equal the input image count %d", outAccess->numSamples(),
                              inData.numImages());
    }
    const PadAndStackArgs args{
        stream,
        inData,
        outData,
        *outAccess,
        borderMode,
        borderValue,
        cuda::CreateTensorWrapNHW<const int>(top),
        cuda::CreateTensorWrapNHW<const int>(left),
        int2{outAccess->numCols(), outAccess->numRows()}
    };

    RunPadAndStack(*outKind, outPlanar, channels, args);
}

} // namespace

namespace cvcuda::priv {

void PadAndStack::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::Tensor &out,
                             const nvcv::Tensor &top, const nvcv::Tensor &left, const NVCVBorderType borderMode,
                             const float borderValue) const
{
    CVCUDA_NVTX_RANGE("cvcuda::PadAndStack::operator()[ImageBatchVarShape]");
    auto inData = RequireCudaVarShape(in, stream, "Input");

    auto outData  = RequireCudaTensor(out, "Output");
    auto topData  = RequireCudaTensor(top, "Top");
    auto leftData = RequireCudaTensor(left, "Left");

    PadAndStackInto(stream, *inData, *outData, *topData, *leftData, borderMode, borderValue);
}

} // namespace cvcuda::priv
