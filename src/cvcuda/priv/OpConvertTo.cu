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
#include "Nvtx.hpp"
#include "OpConvertTo.hpp"
#include "PlanarTensorView.hpp"

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

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using uchar  = unsigned char;
using schar  = signed char;
using ushort = unsigned short;

template<typename SRC_TYPE, typename DST_TYPE, typename S>
struct Convertor
{
    S    alpha;
    S    beta;
    bool truncate; // round toward zero instead of to-nearest (only affects integer outputs)

    __device__ __forceinline__ DST_TYPE operator()(SRC_TYPE src) const
    {
        auto work = alpha * src + beta; // scalar S or a MakeType<S, NC> vector, matching DST_TYPE's channels
        // SaturateCast rounds float->integer to-nearest, so for truncation pre-round toward zero
        // first. Compiled out for floating-point outputs, where the rounding mode has no effect.
        if constexpr (std::is_integral_v<cuda::BaseType<DST_TYPE>>)
        {
            if (truncate)
                work = cuda::round<cuda::RoundMode::ZERO>(work);
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

// COLS_PER_ELEM contiguous columns make up one MakeType<DT, NC> element: 1 for the interleaved path
// (one element is one pixel of NC channels), NC for the wide single-channel path (one element is NC
// contiguous columns of a 1-channel image). The caller guarantees cols % COLS_PER_ELEM == 0.
template<typename DT_SOURCE, typename DT_DEST, int NC, int NIX, int COLS_PER_ELEM = 1>
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

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB> op;

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

template<typename DT_SOURCE, typename DT_DEST, int NC>
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
        if (sm == 80 || sm == 90)
        {
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1>(inData, outData, alpha, beta, truncate, stream);
            return;
        }
    }
    else if constexpr (std::is_same_v<DT_SOURCE, int16_t> && std::is_same_v<DT_DEST, float> && NC == 2)
    {
        if (CurrentDeviceSMOrZero() == 80)
        {
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1>(inData, outData, alpha, beta, truncate, stream);
            return;
        }
    }
    ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 4>(inData, outData, alpha, beta, truncate, stream);
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

template<typename DT_SOURCE, typename DT_DEST>
void ConvertToScale(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    int numChannels, const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
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
            ConvertToScaleCNImpl<DT_SOURCE, DT_DEST, kVec, 4, kVec>(inData, outData, alpha, beta, truncate, stream);
            return;
        }
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 1>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 2:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 2>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 3:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 3>(inData, outData, alpha, beta, truncate, stream);
        return;

    case 4:
        ConvertToScaleCN<DT_SOURCE, DT_DEST, 4>(inData, outData, alpha, beta, truncate, stream);
        return;

    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unknown number of channels");
    }
}

// The element type is semantic here, not a width carrier: alpha * x + beta is evaluated in a work type
// derived from both ends, and the saturation bounds come from the destination type. So the dispatch has
// to name both types, exactly as the legacy 8x8 function-pointer table did.
enum class ElemType
{
    kU8,
    kS8,
    kU16,
    kS16,
    kS32,
    kF32,
    kF64,
    kF16
};

// Reproduce the legacy admitted set exactly. The legacy path reached its dispatch index through
// GetLegacyDataType, which classifies on bits-per-channel plus data kind and rejects every
// (kind, width) pair its enum has no name for -- so 32-/64-bit unsigned and 64-bit signed were
// rejected there even though the operator's own type list looked permissive. A gate phrased as
// "not unsigned-32 and a whole number of bytes" would be wider than legacy.
inline ElemType ClassifyElemType(const nvcv::DataType &dtype)
{
    const auto bpc         = dtype.bitsPerChannel();
    const int  numChannels = dtype.numChannels();

    // GetLegacyDataType rejected a mixed-width packing before classifying it; NVCV_PACKING_X32_Y24b8
    // is a real one, so this is reachable rather than defensive.
    for (int i = 1; i < numChannels; ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    switch (dtype.dataKind())
    {
    case nvcv::DataKind::FLOAT:
        if (bpc[0] == 64)
            return ElemType::kF64;
        if (bpc[0] == 32)
            return ElemType::kF32;
        if (bpc[0] == 16)
            return ElemType::kF16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc[0]);

    case nvcv::DataKind::SIGNED:
        if (bpc[0] == 8)
            return ElemType::kS8;
        if (bpc[0] == 16)
            return ElemType::kS16;
        if (bpc[0] == 32)
            return ElemType::kS32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc[0]);

    case nvcv::DataKind::UNSIGNED:
        if (bpc[0] == 8)
            return ElemType::kU8;
        if (bpc[0] == 16)
            return ElemType::kU16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ",
                              bpc[0]);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

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

// ConvertTo is arithmetic (alpha * x + beta), so the F16 pairs instantiate real __half kernels; their
// work type resolves to float (double only when paired with F64) via the DT_AB deduction.
inline void ConvertToScaleDispatch(ElemType inType, ElemType outType, const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &outData, int numChannels, const double alpha,
                                   const double beta, bool truncate, cudaStream_t stream)
{
    WithElemType(inType,
                 [&](auto srcVal)
                 {
                     WithElemType(outType,
                                  [&](auto dstVal) {
                                      ConvertToScale<decltype(srcVal), decltype(dstVal)>(inData, outData, numChannels,
                                                                                         alpha, beta, truncate, stream);
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
    const bool truncate = (roundMode == NVCV_ROUND_TRUNCATE);

    // Validation order matches the legacy path exactly: both layouts, then both data types, then the
    // interleaved/planar agreement, then the shape and channel-count checks. Every legacy ErrorCode
    // (INVALID_DATA_FORMAT / INVALID_DATA_SHAPE / INVALID_DATA_TYPE / INVALID_PARAMETER) mapped to
    // NVCV_ERROR_INVALID_ARGUMENT through TranslateError, and the helper conversions that could throw
    // ahead of them threw that same status, so every rejection keeps its observable status.
    const bool inPlanar  = IsPlanarLayout(inData.layout());
    const bool outPlanar = IsPlanarLayout(outData.layout());

    const ElemType inType  = ClassifyElemType(inData.dtype());
    const ElemType outType = ClassifyElemType(outData.dtype());

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
        ConvertToScaleDispatch(inType, outType, inView, outView, 1, alpha, beta, truncate, stream);
        return;
    }

    ConvertToScaleDispatch(inType, outType, inData, outData, channels, alpha, beta, truncate, stream);
}

} // namespace

namespace cvcuda::priv {

void ConvertTo::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double alpha,
                           const double beta, NVCVRoundMode roundMode) const
{
    CVCUDA_NVTX_RANGE("cvcuda::ConvertTo::operator()[Tensor]");
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

    RunConvertTo(stream, *inData, *outData, alpha, beta, roundMode);
}

} // namespace cvcuda::priv
