/* Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "../CudaDeviceUtils.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

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
        if constexpr (std::is_integral_v<nvcv::cuda::BaseType<DST_TYPE>>)
        {
            if (truncate)
                work = nvcv::cuda::round<nvcv::cuda::RoundMode::ZERO>(work);
        }
        return nvcv::cuda::SaturateCast<DST_TYPE>(work);
    }
};

// Element-wise convert is latency-bound (1 pixel/thread leaves too few memory requests in flight). Each
// thread processes NIX columns strided by the total x-thread count: consecutive threads read consecutive
// columns (coalesced), and the NIX per-thread accesses are issued as a batch so NIX independent loads are
// in flight at once (memory-level parallelism), hiding load latency. NIX>=2 needs the grid sized to
// divUp(size.x, NIX) x-threads.
template<int NIX, class SrcWrapper, class DstWrapper, class UnOp>
__global__ void convertFormat(SrcWrapper src, DstWrapper dst, UnOp op, int2 size)
{
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
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

template<typename DT_SOURCE, typename DT_DEST, int NC, int NIX>
ErrorCode convertToScaleCNImpl(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                               const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int2 size       = {inAccess->numCols(), inAccess->numRows()};
    const int  batch_size = inAccess->numSamples();

    dim3 block(32, 8);
    dim3 grid(divUp(divUp(size.x, NIX), static_cast<int>(block.x)), divUp(size.y, block.y), batch_size);

    using DT_AB         = decltype(float() * DT_SOURCE() * DT_DEST()); //pick correct scalar
    using SRC_DATA_TYPE = nvcv::cuda::MakeType<DT_SOURCE, NC>;
    using DST_DATA_TYPE = nvcv::cuda::MakeType<DT_DEST, NC>;

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB> op;

    op.alpha    = nvcv::cuda::SaturateCast<DT_AB>(alpha);
    op.beta     = nvcv::cuda::SaturateCast<DT_AB>(beta);
    op.truncate = truncate;

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= nvcv::cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHW<SRC_DATA_TYPE, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHW<DST_DATA_TYPE, int32_t>(outData);

        convertFormat<NIX><<<grid, block, 0, stream>>>(src, dst, op, size);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<typename DT_SOURCE, typename DT_DEST, int NC>
ErrorCode convertToScaleCN(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
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
            return convertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1>(inData, outData, alpha, beta, truncate, stream);
    }
    else if constexpr (std::is_same_v<DT_SOURCE, int16_t> && std::is_same_v<DT_DEST, float> && NC == 2)
    {
        if (CurrentDeviceSMOrZero() == 80)
            return convertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 1>(inData, outData, alpha, beta, truncate, stream);
    }
    return convertToScaleCNImpl<DT_SOURCE, DT_DEST, NC, 4>(inData, outData, alpha, beta, truncate, stream);
}

// Build a single-channel (N*C, H, W, 1) NHWC view of a packed planar (NCHW/CHW) tensor.
// ConvertTo applies the same scalar alpha/beta to every element regardless of channel, so each
// (sample, channel) plane is just a single-channel image; viewing the planes as N*C flat samples
// lets the existing interleaved single-channel kernel process planar data unchanged and bit-exact.
// The view places plane (n, c) at byte offset n*sampleStride + c*chStride, which equals a uniform
// (n*C + c)*chStride stride only when the channel planes are tightly packed across samples
// (sampleStride == numChannels * chStride); always true within a single sample. See the reference
// PlanarAsSingleChannelView in OpResize.cu (.agents/guidance/PLANAR_GUIDELINES.md).
inline nvcv::TensorDataStridedCuda PlanarAsSingleChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t numSamples  = access.numSamples();
    const int64_t numChannels = access.numChannels();
    const int64_t numRows     = access.numRows();
    const int64_t numCols     = access.numCols();

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access.chStride();  // N*C flattened planes
    buf.strides[1] = access.rowStride(); // H
    buf.strides[2] = access.colStride(); // W
    buf.strides[3] = access.colStride(); // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{numSamples * numChannels, numRows, numCols, 1}, "NHWC"},
        data.dtype(), buf
    };
}

// Single-channel ConvertTo is issue-bound on the reference SKUs (A100 ncu: Issue Slots 82%, DRAM only
// 46% — 1 element/thread pays the full per-thread overhead, index math + convert, for a single element).
// VEC contiguous columns are coalesced and reinterpretable as one MakeType<DT,VEC> "pixel", so reusing
// the existing vector Convertor over them cuts the per-element instruction count ~VEC× (relieving the
// issue bound) -- the inverse of NIX-strided MLP, which only helped the latency-bound dev GPU. Selected
// only when cols % VEC == 0 and the buffers are vector-aligned (caller-checked); otherwise the scalar
// single-channel path runs.
template<typename DT_SOURCE, typename DT_DEST, int VEC>
ErrorCode convertToScaleWideSingleChannel(const nvcv::TensorDataStridedCuda &inData,
                                          const nvcv::TensorDataStridedCuda &outData, const double alpha,
                                          const double beta, bool truncate, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    // Each "pixel" is VEC contiguous columns; cols is divisible by VEC (caller-checked).
    const int2 size       = {inAccess->numCols() / VEC, inAccess->numRows()};
    const int  batch_size = inAccess->numSamples();

    constexpr int kNIX = 4;
    dim3          block(32, 8);
    dim3          grid(divUp(divUp(size.x, kNIX), static_cast<int>(block.x)), divUp(size.y, block.y), batch_size);

    using DT_AB         = decltype(float() * DT_SOURCE() * DT_DEST());
    using SRC_DATA_TYPE = nvcv::cuda::MakeType<DT_SOURCE, VEC>;
    using DST_DATA_TYPE = nvcv::cuda::MakeType<DT_DEST, VEC>;

    Convertor<SRC_DATA_TYPE, DST_DATA_TYPE, DT_AB> op;
    op.alpha    = nvcv::cuda::SaturateCast<DT_AB>(alpha);
    op.beta     = nvcv::cuda::SaturateCast<DT_AB>(beta);
    op.truncate = truncate;

    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    if (std::max(outMaxStride, inMaxStride) <= nvcv::cuda::TypeTraits<int32_t>::max)
    {
        auto src = nvcv::cuda::CreateTensorWrapNHW<SRC_DATA_TYPE, int32_t>(inData);
        auto dst = nvcv::cuda::CreateTensorWrapNHW<DST_DATA_TYPE, int32_t>(outData);
        convertFormat<kNIX><<<grid, block, 0, stream>>>(src, dst, op, size);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << nvcv::cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

// True when a single-channel convert can use VEC-wide contiguous vectorization: cols divisible by VEC,
// and the in/out base pointers + row/sample strides aligned to the VEC-wide vector type.
template<typename DT_SOURCE, typename DT_DEST, int VEC>
static bool wideSingleChannelEligible(const nvcv::TensorDataStridedCuda &inData,
                                      const nvcv::TensorDataStridedCuda &outData)
{
    auto in  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto out = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!in || !out)
        return false;
    if (in->numCols() % VEC != 0)
        return false;
    const std::uintptr_t aS = alignof(nvcv::cuda::MakeType<DT_SOURCE, VEC>);
    const std::uintptr_t aD = alignof(nvcv::cuda::MakeType<DT_DEST, VEC>);
    auto                 ok = [](std::uintptr_t base, int64_t rowS, int64_t smpS, std::uintptr_t a)
    {
        return ((base | static_cast<std::uintptr_t>(rowS) | static_cast<std::uintptr_t>(smpS)) & (a - 1)) == 0;
    };
    return ok(reinterpret_cast<std::uintptr_t>(inData.basePtr()), in->rowStride(), in->sampleStride(), aS)
        && ok(reinterpret_cast<std::uintptr_t>(outData.basePtr()), out->rowStride(), out->sampleStride(), aD);
}

template<typename DT_SOURCE, typename DT_DEST> // <uchar, float> <float double>
ErrorCode convertToScale(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                         int numChannels, const double alpha, const double beta, bool truncate, cudaStream_t stream)
{
    switch (numChannels)
    {
    case 1:
        // Wide contiguous vectorization (issue-bound fix); falls back to the scalar single-channel path
        // when cols isn't VEC-divisible or the buffers aren't vector-aligned.
        if (wideSingleChannelEligible<DT_SOURCE, DT_DEST, 4>(inData, outData))
            return convertToScaleWideSingleChannel<DT_SOURCE, DT_DEST, 4>(inData, outData, alpha, beta, truncate,
                                                                          stream);
        return convertToScaleCN<DT_SOURCE, DT_DEST, 1>(inData, outData, alpha, beta, truncate, stream);

    case 2:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 2>(inData, outData, alpha, beta, truncate, stream);

    case 3:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 3>(inData, outData, alpha, beta, truncate, stream);

    case 4:
        return convertToScaleCN<DT_SOURCE, DT_DEST, 4>(inData, outData, alpha, beta, truncate, stream);

    default:
        LOG_ERROR("Unknown number of channels");
        return ErrorCode::INVALID_PARAMETER;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nvcv::legacy::cuda_op {

ErrorCode ConvertTo::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                           const double alpha, const double beta, NVCVRoundMode roundMode, cudaStream_t stream)
{
    const bool truncate = (roundMode == NVCV_ROUND_TRUNCATE);

    cuda_op::DataFormat input_format    = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format   = GetLegacyDataFormat(outData.layout());
    cuda_op::DataType   input_datatype  = GetLegacyDataType(inData.dtype());
    cuda_op::DataType   output_datatype = GetLegacyDataType(outData.dtype());

    const bool inPlanar  = (input_format == kNCHW || input_format == kCHW);
    const bool outPlanar = (output_format == kNCHW || output_format == kCHW);
    const bool inOk      = (input_format == kNHWC || input_format == kHWC) || inPlanar;
    const bool outOk     = (output_format == kNHWC || output_format == kHWC) || outPlanar;

    // Input and output must both be interleaved or both planar; the operator does not transpose.
    if (!(inOk && outOk && inPlanar == outPlanar))
    {
        LOG_ERROR("Invalid DataFormat, must be kHWC/kNHWC or kCHW/kNCHW with matching layouts");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataShape inputShape  = helpers::GetLegacyDataShape(inAccess->infoShape());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("input/output shape is different " << inputShape << "/" << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(input_datatype == kCV_8U || input_datatype == kCV_8S || input_datatype == kCV_16U || input_datatype == kCV_16S
          || input_datatype == kCV_32S || input_datatype == kCV_32F || input_datatype == kCV_64F))
    {
        LOG_ERROR("Invalid DataType " << input_datatype);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(output_datatype == kCV_8U || output_datatype == kCV_8S || output_datatype == kCV_16U
          || output_datatype == kCV_16S || output_datatype == kCV_32S || output_datatype == kCV_32F
          || output_datatype == kCV_64F))
    {
        LOG_ERROR("Invalid Converted DataType " << output_datatype);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                int numChannels, const double alpha, const double beta, bool truncate,
                                cudaStream_t stream);

    // clang-format off
    static const func_t funcs[7][7] = {
        { convertToScale<uchar, uchar>,  convertToScale<uchar, schar>,  convertToScale<uchar, ushort>,  convertToScale<uchar, short>,  convertToScale<uchar, int>,       convertToScale<uchar, float>,    convertToScale<uchar, double>   },
        { convertToScale<schar, uchar>,  convertToScale<schar, schar>,  convertToScale<schar, ushort>,  convertToScale<schar, short>,  convertToScale<schar, int>,       convertToScale<schar, float>,    convertToScale<schar, double>   },
        { convertToScale<ushort, uchar>, convertToScale<ushort, schar>, convertToScale<ushort, ushort>, convertToScale<ushort, short>, convertToScale<ushort, int>,      convertToScale<ushort, float>,   convertToScale<ushort, double>  },
        { convertToScale<short, uchar>,  convertToScale<short, schar>,  convertToScale<short, ushort>,  convertToScale<short, short>,  convertToScale<short, int>,       convertToScale<short, float>,    convertToScale<short, double>   },
        { convertToScale<int, uchar>,    convertToScale<int, schar>,    convertToScale<int, ushort>,    convertToScale<int, short>,    convertToScale<int, int>,         convertToScale<int, float>,      convertToScale<int, double>     },
        { convertToScale<float, uchar>,  convertToScale<float, schar>,  convertToScale<float, ushort>,  convertToScale<float, short>,  convertToScale<float, int>,       convertToScale<float, float>,    convertToScale<float, double>   },
        { convertToScale<double, uchar>, convertToScale<double, schar>, convertToScale<double, ushort>, convertToScale<double, short>, convertToScale<double, int>,      convertToScale<double, float>,   convertToScale<double, double>  }
    };

    // clang-format on
    const func_t func = funcs[input_datatype][output_datatype];

    if (inPlanar)
    {
        // Each channel plane is processed as an independent single-channel image flattened into the
        // N*C sample dimension. That flattened count becomes the kernel's grid z-dimension, capped at
        // CUDA's 65535 limit; compute it in 64-bit to avoid overflow.
        constexpr int64_t kMaxGridZ   = 65535;
        const int64_t     planarBatch = static_cast<int64_t>(channels) * batch;
        if (planarBatch > kMaxGridZ)
        {
            LOG_ERROR("Planar ConvertTo requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        // The flattened view assumes channel planes are tightly packed across samples.
        if (batch > 1 && inAccess->sampleStride() != channels * inAccess->chStride())
        {
            LOG_ERROR("Planar ConvertTo of a batched tensor requires tightly packed input channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (batch > 1 && outAccess->sampleStride() != channels * outAccess->chStride())
        {
            LOG_ERROR("Planar ConvertTo of a batched tensor requires tightly packed output channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto inView  = PlanarAsSingleChannelView(inData, *inAccess);
        auto outView = PlanarAsSingleChannelView(outData, *outAccess);
        return func(inView, outView, 1, alpha, beta, truncate, stream);
    }

    return func(inData, outData, channels, alpha, beta, truncate, stream);
}

} // namespace nvcv::legacy::cuda_op
