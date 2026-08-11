/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "../PlanarTensorView.hpp"
#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <cstdint>

namespace nvcv::legacy::cuda_op {

using namespace nvcv::legacy::helpers;

// Flip is a pure pixel remap, so it is memory/latency-bound: one element per thread leaves too few
// memory requests in flight (ncu on the single-channel uchar kernel: DRAM 45%, "latency issues").
// Each thread instead processes NIX columns strided by the total x-thread count, so NIX independent
// loads/stores are issued together (memory-level parallelism) while consecutive threads still touch
// consecutive columns (coalesced). NIX>1 needs the grid sized to divUp(width, NIX) x-threads. The
// remap is unchanged, so every output is bit-exact with the one-element-per-thread version.
constexpr int kFlipNIX       = 4;
constexpr int kFlipScalarNIX = 1;

inline bool CurrentDeviceIsSM86()
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;

    return prop.major == 8 && prop.minor == 6;
}

template<int NIX, typename SrcWrapper, typename DstWrapper>
__global__ void flipHorizontal(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= dstSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t width  = dstSize.w;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < width)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, dst_y, (width - 1 - dst_x));
        }
    }
}

template<int NIX, typename SrcWrapper, typename DstWrapper>
__global__ void flipVertical(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= dstSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t width  = dstSize.w;
    const int32_t src_y  = dstSize.h - 1 - dst_y;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < width)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, src_y, dst_x);
        }
    }
}

template<int NIX, typename SrcWrapper, typename DstWrapper>
__global__ void flipHorizontalVertical(SrcWrapper src, DstWrapper dst, Size2D dstSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= dstSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t width  = dstSize.w;
    const int32_t src_y  = dstSize.h - 1 - dst_y;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < width)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, src_y, (width - 1 - dst_x));
        }
    }
}

template<int NIX, typename SrcWrap, typename DstWrap>
void runFlipKernel(SrcWrap src, DstWrap dst, Size2D dstSize, int numSamples, int32_t flipCode, cudaStream_t stream)
{
    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(divUp(dstSize.w, NIX), static_cast<int>(blockSize.x)), divUp(dstSize.h, blockSize.y),
                  numSamples);

    if (flipCode > 0)
    {
        flipHorizontal<NIX><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
    else if (flipCode == 0)
    {
        flipVertical<NIX><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
    else
    {
        flipHorizontalVertical<NIX><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize);
        checkKernelErrors();
    }
}

template<typename T, int NIX>
ErrorCode flipImpl(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
                   cudaStream_t stream)
{
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(output);
    NVCV_ASSERT(outAccess);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(input);
    NVCV_ASSERT(inAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto src = cuda::CreateTensorWrapNHW<const T, int32_t>(input);
        auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(output);

        runFlipKernel<NIX>(src, dst, dstSize, outAccess->numSamples(), flipCode, stream);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif // CUDA_DEBUG_LOG

    return ErrorCode::SUCCESS;
}

template<typename T>
ErrorCode flip(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
               cudaStream_t stream)
{
    return flipImpl<T, kFlipNIX>(input, output, flipCode, stream);
}

template<typename T>
ErrorCode flipScalar(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
                     cudaStream_t stream)
{
    return flipImpl<T, kFlipScalarNIX>(input, output, flipCode, stream);
}

// Wide single-channel flip: a run of VEC contiguous columns is loaded/stored as one MakeType<T,VEC>
// vector. A single-channel uchar plane moves only 1 byte/thread, so even with NIX MLP the kernel pays
// the per-element index/branch overhead VEC times more than it must; coalescing VEC columns into one
// wide access cuts that overhead ~VEC x and raises bytes/thread. Vertical flip keeps the column order,
// so the source vector is the same column block in row (H-1-y) -- a direct wide copy. Horizontal flip
// reverses the columns, so a destination block [j*VEC .. j*VEC+VEC-1] maps to the mirrored source block
// (W/VEC-1-j) with its VEC lanes reversed in registers; this is exact only when the column count is a
// multiple of VEC (caller-checked) so the block boundaries line up. The result is bit-identical to the
// scalar single-channel flip.
template<int VEC, typename WideT>
__device__ __forceinline__ WideT flipLanes(WideT v)
{
    WideT out;
#pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        cuda::GetElement(out, k) = cuda::GetElement(v, VEC - 1 - k);
    }
    return out;
}

template<int NIX, int VEC, typename SrcWrapper, typename DstWrapper>
__global__ void flipWideHorizontal(SrcWrapper src, DstWrapper dst, Size2D vecSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= vecSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t wvec   = vecSize.w;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < wvec)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = flipLanes<VEC>(*src.ptr(batch_idx, dst_y, (wvec - 1 - dst_x)));
        }
    }
}

template<int NIX, int VEC, typename SrcWrapper, typename DstWrapper>
__global__ void flipWideVertical(SrcWrapper src, DstWrapper dst, Size2D vecSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= vecSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t wvec   = vecSize.w;
    const int32_t src_y  = vecSize.h - 1 - dst_y;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < wvec)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, src_y, dst_x);
        }
    }
}

template<int NIX, int VEC, typename SrcWrapper, typename DstWrapper>
__global__ void flipWideHorizontalVertical(SrcWrapper src, DstWrapper dst, Size2D vecSize)
{
    const int32_t dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int32_t batch_idx = get_batch_idx();
    if (dst_y >= vecSize.h)
        return;

    const int32_t x0     = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = gridDim.x * blockDim.x;
    const int32_t wvec   = vecSize.w;
    const int32_t src_y  = vecSize.h - 1 - dst_y;

#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int32_t dst_x = x0 + i * stride;
        if (dst_x < wvec)
        {
            *dst.ptr(batch_idx, dst_y, dst_x) = flipLanes<VEC>(*src.ptr(batch_idx, src_y, (wvec - 1 - dst_x)));
        }
    }
}

// True when the single-channel flip can use VEC-wide contiguous vectorization: cols divisible by VEC,
// and both base pointers + row/sample strides aligned to the VEC-wide vector type.
template<typename T, int VEC>
static bool wideSingleChannelEligible(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output)
{
    auto in  = TensorDataAccessStridedImagePlanar::Create(input);
    auto out = TensorDataAccessStridedImagePlanar::Create(output);
    if (!in || !out)
        return false;
    if (in->numCols() % VEC != 0)
        return false;
    const std::uintptr_t a  = alignof(cuda::MakeType<T, VEC>);
    auto                 ok = [a](std::uintptr_t base, int64_t rowS, int64_t smpS)
    {
        return ((base | static_cast<std::uintptr_t>(rowS) | static_cast<std::uintptr_t>(smpS)) & (a - 1)) == 0;
    };
    return ok(reinterpret_cast<std::uintptr_t>(input.basePtr()), in->rowStride(), in->sampleStride())
        && ok(reinterpret_cast<std::uintptr_t>(output.basePtr()), out->rowStride(), out->sampleStride());
}

template<typename T, int VEC>
ErrorCode flipWideSingleChannel(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output,
                                const int32_t flipCode, cudaStream_t stream)
{
    using WideT = cuda::MakeType<T, VEC>;

    auto inAccess  = TensorDataAccessStridedImagePlanar::Create(input);
    auto outAccess = TensorDataAccessStridedImagePlanar::Create(output);
    NVCV_ASSERT(inAccess);
    NVCV_ASSERT(outAccess);

    // Each "pixel" spans VEC contiguous columns; numCols is divisible by VEC (caller-checked).
    Size2D  vecSize{outAccess->numCols() / VEC, outAccess->numRows()};
    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto src = cuda::CreateTensorWrapNHW<const WideT, int32_t>(input);
    auto dst = cuda::CreateTensorWrapNHW<WideT, int32_t>(output);

    dim3 blockSize(32, 8, 1);
    dim3 gridSize(divUp(divUp(vecSize.w, kFlipNIX), static_cast<int>(blockSize.x)), divUp(vecSize.h, blockSize.y),
                  outAccess->numSamples());

    if (flipCode > 0)
    {
        flipWideHorizontal<kFlipNIX, VEC><<<gridSize, blockSize, 0, stream>>>(src, dst, vecSize);
    }
    else if (flipCode == 0)
    {
        flipWideVertical<kFlipNIX, VEC><<<gridSize, blockSize, 0, stream>>>(src, dst, vecSize);
    }
    else
    {
        flipWideHorizontalVertical<kFlipNIX, VEC><<<gridSize, blockSize, 0, stream>>>(src, dst, vecSize);
    }
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

// Single-channel flip: use VEC-wide contiguous vectorization when the columns are VEC-divisible and the
// buffers are vector-aligned, otherwise fall back to the scalar single-channel kernel. Used by the
// interleaved C==1 path and by the planar path (each NCHW plane is a single-channel image).
template<typename T>
ErrorCode flipSingleChannel(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output,
                            const int32_t flipCode, cudaStream_t stream)
{
    if (wideSingleChannelEligible<T, 4>(input, output))
        return flipWideSingleChannel<T, 4>(input, output, flipCode, stream);
    return flip<T>(input, output, flipCode, stream);
}

ErrorCode Flip::infer(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output, const int32_t flipCode,
                      cudaStream_t stream)
{
    if (input.dtype() != output.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << input.dtype() << ") and output (" << output.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataFormat inputFormat  = GetLegacyDataFormat(input.layout());
    DataFormat outputFormat = GetLegacyDataFormat(output.layout());
    if (inputFormat != outputFormat)
    {
        LOG_ERROR("Invalid DataFormat between input (" << inputFormat << ") and output (" << outputFormat << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = inputFormat;
    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    cuda_op::DataType dataType = GetLegacyDataType(input.dtype());
    if (!(dataType == kCV_8U || dataType == kCV_16U || dataType == kCV_32S || dataType == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << dataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    auto inputWrapper = TensorDataAccessStridedImagePlanar::Create(input);
    NVCV_ASSERT(inputWrapper);

    cuda_op::DataShape inputShape = GetLegacyDataShape(inputWrapper->infoShape());
    // Planar 2-channel layout is rejected (no defined 2-plane format); the interleaved path likewise
    // does not support 2 channels (matches the Normalize and Resize operators).
    if (inputShape.C > 4 || inputShape.C == 2)
    {
        LOG_ERROR("Invalid channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // using flip_t = void(const TensorDataStridedCuda & input,
    //                     const TensorDataStridedCuda & output,
    //                     const int32_t flipCode, cudaStream_t stream);
    typedef ErrorCode (*flip_t)(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output,
                                const int32_t flipCode, cudaStream_t stream);

    // Column 0 (single channel) routes through flipSingleChannel, which picks the wide-vectorized
    // kernel when eligible and otherwise the scalar single-channel kernel. The planar path also uses
    // column 0 (each NCHW plane is a single-channel image).
    static const flip_t funcs[6][4] = {
        {  flipSingleChannel<uchar>, 0,       flip<uchar3>,  flip<uchar4>},
        {                         0, 0,                  0,             0},
        { flipSingleChannel<ushort>, 0,      flip<ushort3>, flip<ushort4>},
        {                         0, 0,                  0,             0},
        {flipSingleChannel<int32_t>, 0,         flip<int3>,    flip<int4>},
        {  flipSingleChannel<float>, 0, flipScalar<float3>,  flip<float4>}
    };

    const int32_t channels = inputShape.C;

    if (isPlanar)
    {
        // View each of the N*C channel planes as a single-channel sample and reuse the interleaved
        // single-channel flip kernel (funcs column 0). Channels are independent in flip, so this is
        // bit-exact with flipping the equivalent NHWC single-channel data. The flattened plane count
        // becomes the kernel's grid.z, capped at CUDA's 65535 limit; compute it in 64-bit to avoid
        // overflow. The flattened view is only valid when the planes are tightly packed across samples.
        auto outputWrapper = TensorDataAccessStridedImagePlanar::Create(output);
        NVCV_ASSERT(outputWrapper);

        const int64_t numSamples = inputWrapper->numSamples();
        if (numSamples > 1
            && (inputWrapper->sampleStride() != static_cast<int64_t>(channels) * inputWrapper->chStride()
                || outputWrapper->sampleStride() != static_cast<int64_t>(channels) * outputWrapper->chStride()))
        {
            LOG_ERROR("Planar flip of a batched tensor requires tightly packed channel planes");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (numSamples * channels > 65535)
        {
            LOG_ERROR("Planar flip requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
            return ErrorCode::INVALID_DATA_SHAPE;
        }

        auto srcView = cvcuda::priv::PlanarAsSingleChannelView(input, *inputWrapper);
        auto dstView = cvcuda::priv::PlanarAsSingleChannelView(output, *outputWrapper);
        // GA102/SM86 regresses on vectorized planar tensor float3; keep other SKUs on the wider path.
        if (dataType == kCV_32F && channels == 3 && CurrentDeviceIsSM86())
        {
            return flipScalar<float>(srcView, dstView, flipCode, stream);
        }
        return funcs[dataType][0](srcView, dstView, flipCode, stream);
    }

    return funcs[dataType][channels - 1](input, output, flipCode, stream);
}

} // namespace nvcv::legacy::cuda_op
