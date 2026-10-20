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
#include "OpRotate.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"
#include "WarpSampleCommon.cuh"

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
#include <memory>
#include <type_traits>

using namespace cvcuda::priv::warp; // NOLINT(google-build-using-namespace)

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using uchar  = unsigned char;
using ushort = unsigned short;

constexpr int kBlock = 32;

// Affine coefficients per rotated image: the 2x3 warp matrix ComputeWarpAffine writes and the
// rotate kernels read back. Sizes the device scratch and strides the var-shape kernels' per-image
// indexing into it, so the two must agree.
constexpr int kAffineCoeffs = 6;

constexpr double kPi = 3.1415926535897932384626433832795;

// The CUBIC fast sampler and the dtype/layout classification live in
// WarpSampleCommon.cuh, together with the rationale for restructuring them.

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

__global__ void ComputeWarpAffine(const double angle, const double xShift, const double yShift, float *aCoeffs)
{
    // Trig in FP64 for accuracy (one-shot, 1 thread; perf-irrelevant).
    // Stored as FP32 — the per-pixel rotate kernel reads these in FP32 to avoid
    // the 1/64-rate FP64 path on consumer GPUs.
    aCoeffs[0] = static_cast<float>(cos(angle * kPi / 180));
    aCoeffs[1] = static_cast<float>(sin(angle * kPi / 180));
    aCoeffs[2] = static_cast<float>(xShift);
    aCoeffs[3] = static_cast<float>(-sin(angle * kPi / 180));
    aCoeffs[4] = static_cast<float>(cos(angle * kPi / 180));
    aCoeffs[5] = static_cast<float>(yShift);
}

__global__ void ComputeWarpAffineVarShape(const int numImages, const cuda::Tensor1DWrap<double> angleDeg,
                                          const cuda::Tensor2DWrap<double> shift, float *d_aCoeffs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    // Trig in FP64 for accuracy (one-shot, 1 thread per image; perf-irrelevant).
    // Stored as FP32 — the per-pixel rotate kernel reads these in FP32 to avoid
    // the 1/64-rate FP64 path on consumer GPUs.
    float *aCoeffs = (float *)((char *)d_aCoeffs + (sizeof(float) * kAffineCoeffs) * index);

    double angle  = angleDeg[index];
    double xShift = *shift.ptr(index, 0);
    double yShift = *shift.ptr(index, 1);

    aCoeffs[0] = static_cast<float>(cos(angle * kPi / 180));
    aCoeffs[1] = static_cast<float>(sin(angle * kPi / 180));
    aCoeffs[2] = static_cast<float>(xShift);
    aCoeffs[3] = static_cast<float>(-sin(angle * kPi / 180));
    aCoeffs[4] = static_cast<float>(cos(angle * kPi / 180));
    aCoeffs[5] = static_cast<float>(yShift);
}

// Number of output rows each thread emits (Y-tiling factor), chosen per interpolation type and element
// size. The rotate kernel is latency/issue-bound on consumer Ampere (one interpolated gather per pixel
// through L1/TEX, low SOL). For small-element NEAREST/LINEAR the per-pixel working set is tiny, so a wide
// Y-tile amortizes the per-thread invariants (six affine coefficients, the column term, the source-bounds
// load) and overlaps several independent per-row gathers to hide latency -- up to ~2x on uint8 NEAREST,
// ~1.5x on uint8 LINEAR. Two cases must stay at one row per thread (the original one-pixel-per-thread
// mapping) to avoid regressing: CUBIC touches a 4x4 neighborhood per pixel and tiling it thrashes L1
// (measured up to ~5x slower on float4); and wide (>2-byte component) elements such as float move enough
// bytes per pixel that tiling also costs more than it saves (measured ~1.25x slower on float NEAREST).
// Y-tiling (rather than X-tiling) keeps the X dimension one thread per column so warp stores stay
// coalesced.
template<typename T, NVCVInterpolationType I>
constexpr int kRotateRowsPerThread = (I == NVCV_INTERP_CUBIC || sizeof(cuda::BaseType<T>) > 2) ? 1 : 8;

// Bit-exactness: src_x/src_y are recomputed per pixel with the same FP multiply form and operand order as
// the original scalar kernel (the column term is the same left operand of the same add), not accumulated
// incrementally, so the rounding is identical to one-pixel-per-thread. With NIY == 1 the body collapses
// to exactly the original mapping.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void RotateTensorKernel(SrcWrapper src, DstWrapper dst, int2 dstSize, const float *d_aCoeffs)
{
    const int dst_x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y0 = (blockIdx.y * blockDim.y + threadIdx.y) * NIY;
    const int dst_z  = blockIdx.z * blockDim.z + threadIdx.z;

    if (dst_x >= dstSize.x)
    {
        return;
    }

    const float c0 = d_aCoeffs[0];
    const float c1 = d_aCoeffs[1];
    const float c2 = d_aCoeffs[2];
    const float c3 = d_aCoeffs[3];
    const float c4 = d_aCoeffs[4];
    const float c5 = d_aCoeffs[5];

    const float dst_x_shift = static_cast<float>(dst_x) - c2;
    const float src_x_col   = dst_x_shift * c0;
    const float src_y_col   = dst_x_shift * (-c3);

    const long2 srcSize{src.borderWrap().tensorShape()[1], src.borderWrap().tensorShape()[0]};

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstSize.y)
        {
            break;
        }

        const float  dst_y_shift = static_cast<float>(dst_y) - c5;
        const float3 srcCoord{src_x_col + dst_y_shift * (-c1), src_y_col + dst_y_shift * c4, static_cast<float>(dst_z)};

        if (srcCoord.x > -0.5 && srcCoord.x < srcSize.x && srcCoord.y > -0.5 && srcCoord.y < srcSize.y)
        {
            const int3 dstCoord{dst_x, dst_y, dst_z};

            using SrcValueT = std::remove_cv_t<typename SrcWrapper::ValueType>;
            constexpr bool kFastCubic
                = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC && kCubicFastSampler<SrcValueT>;
            if constexpr (kFastCubic)
            {
                dst[dstCoord] = CubicSampleTensor<SrcWrapper, true>(
                    src, dst_z, float2{srcCoord.x, srcCoord.y},
                    int2{static_cast<int>(srcSize.x), static_cast<int>(srcSize.y)});
            }
            else if constexpr (SrcWrapper::kInterpolationType == NVCV_INTERP_LINEAR
                               && std::is_same_v<SrcValueT, uchar3>)
            {
                const int ix = cuda::GetIndexForInterpolation<NVCV_INTERP_LINEAR, 1, int>(srcCoord.x);
                const int iy = cuda::GetIndexForInterpolation<NVCV_INTERP_LINEAR, 1, int>(srcCoord.y);
                // Keep the shared loader's entire aligned window within each source row.
                if (ix >= 3 && ix + 3 + kCubicRowMarginPx<SrcValueT> < srcSize.x && iy >= 0 && iy + 1 < srcSize.y)
                {
                    const auto &srcTensor = src.borderWrap().tensorWrap();
                    const int   srcSample = static_cast<int>(srcCoord.z);
                    SrcValueT   top[4], bottom[4];
                    LoadTapRow(srcTensor.ptr(srcSample, iy, ix), top);
                    LoadTapRow(srcTensor.ptr(srcSample, iy + 1, ix), bottom);
                    auto out = cuda::SetAll<float3>(0);
                    out += top[0] * (static_cast<float>(ix + 1) - srcCoord.x)
                         * (static_cast<float>(iy + 1) - srcCoord.y);
                    out += top[1] * (srcCoord.x - static_cast<float>(ix)) * (static_cast<float>(iy + 1) - srcCoord.y);
                    out += bottom[0] * (static_cast<float>(ix + 1) - srcCoord.x)
                         * (srcCoord.y - static_cast<float>(iy));
                    out += bottom[1] * (srcCoord.x - static_cast<float>(ix)) * (srcCoord.y - static_cast<float>(iy));
                    dst[dstCoord] = cuda::SaturateCast<SrcValueT>(out);
                }
                else
                {
                    dst[dstCoord] = src[srcCoord];
                }
            }
            else
            {
                dst[dstCoord] = src[srcCoord];
            }
        }
    }
}

// Fused planar CUBIC rotate: one thread maps its output pixel once and gathers all NP channel
// planes, sharing the coordinate transform, cubic weights, and border resolution that the
// flattened per-plane launch recomputes NP times. Wraps address the flattened (N*C, H, W) view;
// plane p of sample z sits at flattened sample z*NP + p.
template<int NP, class SrcWrapper, class DstWrapper>
__global__ void RotatePlanarFusedTensorKernel(SrcWrapper src, DstWrapper dst, int2 dstSize, int2 srcSize,
                                              const float *d_aCoeffs)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int dst_z = blockIdx.z;

    if (dst_x >= dstSize.x || dst_y >= dstSize.y)
    {
        return;
    }

    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  dst_y_shift = static_cast<float>(dst_y) - d_aCoeffs[5];
    const float2 coord{dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]),
                       dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]};

    if (coord.x > -0.5f && coord.x < static_cast<float>(srcSize.x) && coord.y > -0.5f
        && coord.y < static_cast<float>(srcSize.y))
    {
        using BT = std::remove_cv_t<typename SrcWrapper::ValueType>;
        CubicWarpPlanes<BT, NVCV_BORDER_REPLICATE, NP, int32_t>(
            [&](int p, int yy) { return src.ptr(dst_z * NP + p, yy); },
            [&](int p, BT v) { *(dst.ptr(dst_z * NP + p, dst_y) + dst_x) = v; }, float4{0.f, 0.f, 0.f, 0.f}, srcSize,
            coord);
    }
}

// Bit-exact: src_x/src_y use the same FP multiply form and operand order as the original scalar kernel
// (the column term is the same left operand of the same add). With NIY == 1 the body collapses to the
// original one-pixel-per-thread mapping.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void RotateVarShapeKernel(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_)
{
    const int dst_x     = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y0    = (blockDim.y * blockIdx.y + threadIdx.y) * NIY;
    const int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    if (dst_x >= dstWidth)
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * kAffineCoeffs) * batch_idx);
    const float  c0          = d_aCoeffs[0];
    const float  c1          = d_aCoeffs[1];
    const float  c3          = d_aCoeffs[3];
    const float  c4          = d_aCoeffs[4];
    const float  c5          = d_aCoeffs[5];
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  src_x_col   = dst_x_shift * c0;
    const float  src_y_col   = dst_x_shift * (-c3);

    const int width  = src.borderWrap().imageBatchWrap().width(batch_idx);
    const int height = src.borderWrap().imageBatchWrap().height(batch_idx);

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstHeight)
            break;

        const float dst_y_shift = static_cast<float>(dst_y) - c5;
        const float src_x       = src_x_col + dst_y_shift * (-c1);
        const float src_y       = src_y_col + dst_y_shift * c4;

        if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
        {
            const int3 dstCoord{dst_x, dst_y, batch_idx};

            using SrcValueT = std::remove_cv_t<typename SrcWrapper::ValueType>;
            constexpr bool kFastCubic
                = SrcWrapper::kInterpolationType == NVCV_INTERP_CUBIC && kCubicFastSampler<SrcValueT>;
            if constexpr (kFastCubic)
            {
                dst[dstCoord] = CubicSampleVarShape<SrcWrapper, true>(src, batch_idx, float2{src_x, src_y});
            }
            else
            {
                const float3 srcCoord{src_x, src_y, static_cast<float>(batch_idx)};
                dst[dstCoord] = src[srcCoord];
            }
        }
    }
}

// Planar (NCHW/CHW) rotate. Rotate maps each output pixel to a source pixel with the same affine
// coefficients regardless of the channel, so the (src_x, src_y) mapping is computed once per output
// pixel and reused across every channel plane (grid.z runs over images, the kernel loops the planes).
// This avoids the per-plane redundant coordinate math of one thread per (image, plane). Each plane is
// sampled through the same InterpolationVarShapeWrap the interleaved kernel uses -- its operator[]
// takes a 4D {x, y, plane, sample} coordinate -- so each plane's result is bit-exact with the
// interleaved single-channel rotate and the interleaved codegen is left untouched.
template<int NIY, class SrcWrapper, class DstWrapper>
__global__ void RotatePlanarVarShapeKernel(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_, int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y0    = (blockIdx.y * blockDim.y + threadIdx.y) * NIY;
    const int batch_idx = blockIdx.z;

    const int dstWidth  = dst.width(batch_idx);
    const int dstHeight = dst.height(batch_idx);
    if (dst_x >= dstWidth)
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * kAffineCoeffs) * batch_idx);
    const float  c0          = d_aCoeffs[0];
    const float  c1          = d_aCoeffs[1];
    const float  c3          = d_aCoeffs[3];
    const float  c4          = d_aCoeffs[4];
    const float  c5          = d_aCoeffs[5];
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  src_x_col   = dst_x_shift * c0;
    const float  src_y_col   = dst_x_shift * (-c3);

    const int width  = src.borderWrap().imageBatchWrap().width(batch_idx);
    const int height = src.borderWrap().imageBatchWrap().height(batch_idx);

#pragma unroll
    for (int i = 0; i < NIY; ++i)
    {
        const int dst_y = dst_y0 + i;
        if (dst_y >= dstHeight)
            break;

        const float dst_y_shift = static_cast<float>(dst_y) - c5;
        const float src_x       = src_x_col + dst_y_shift * (-c1);
        const float src_y       = src_y_col + dst_y_shift * c4;

        if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
        {
            for (int plane = 0; plane < channels; ++plane)
            {
                const float4 srcCoord{src_x, src_y, static_cast<float>(plane), static_cast<float>(batch_idx)};
                *dst.ptr(batch_idx, plane, dst_y, dst_x) = src[srcCoord];
            }
        }
    }
}

// Fused planar CUBIC rotate: shares the coordinate transform, cubic weights, and border
// resolution across all NP channel planes; the generic kernel above re-resolves them per plane
// through the interpolation wrap.
template<int NP, class SrcWrapper, class DstWrapper>
__global__ void RotatePlanarFusedVarShapeKernel(SrcWrapper src, DstWrapper dst, const float *d_aCoeffs_)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    const float *d_aCoeffs   = (const float *)((char *)d_aCoeffs_ + (sizeof(float) * kAffineCoeffs) * batch_idx);
    const float  dst_x_shift = static_cast<float>(dst_x) - d_aCoeffs[2];
    const float  dst_y_shift = static_cast<float>(dst_y) - d_aCoeffs[5];
    const float2 coord{dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]),
                       dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]};

    const int width  = src.width(batch_idx);
    const int height = src.height(batch_idx);

    if (coord.x > -0.5f && coord.x < static_cast<float>(width) && coord.y > -0.5f
        && coord.y < static_cast<float>(height))
    {
        using BT = std::remove_cv_t<typename SrcWrapper::ValueType>;
        CubicWarpPlanes<BT, NVCV_BORDER_REPLICATE, NP, int32_t>(
            [&](int p, int yy) { return src.ptr(batch_idx, p, yy, 0); },
            [&](int p, BT v) { *dst.ptr(batch_idx, p, dst_y, dst_x) = v; }, float4{0.f, 0.f, 0.f, 0.f},
            int2{width, height}, coord);
    }
}

// ---------------------------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------------------------

template<NVCVInterpolationType I>
using InterpTag = std::integral_constant<NVCVInterpolationType, I>;

// Runs fn with the interpolation mode as a compile-time tag, or returns false when the mode has no
// kernel. The rejection is reported by the caller rather than here: the tensor and var-shape paths
// carry different messages for it, and the message text is part of the observable surface.
template<class F>
[[nodiscard]] inline bool DispatchInterp(NVCVInterpolationType interpolation, F &&fn)
{
    switch (interpolation)
    {
    case NVCV_INTERP_NEAREST:
        fn(InterpTag<NVCV_INTERP_NEAREST>{});
        return true;
    case NVCV_INTERP_LINEAR:
        fn(InterpTag<NVCV_INTERP_LINEAR>{});
        return true;
    case NVCV_INTERP_CUBIC:
        fn(InterpTag<NVCV_INTERP_CUBIC>{});
        return true;
    default:
        return false;
    }
}

template<typename T, NVCVInterpolationType I>
inline void RunRotateTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            float *d_aCoeffs, const double angleDeg, const double2 shift, cudaStream_t stream)
{
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int  batchSize{static_cast<int>(outAccess->numSamples())};

    ComputeWarpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);

    constexpr int kNIY = kRotateRowsPerThread<T, I>;

    dim3 blockSize(kBlock, kBlock / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(dstSize.y / (blockY*kNIY)).
    dim3 gridSize(util::DivUp(dstSize.x, blockSize.x), util::DivUp(dstSize.y, static_cast<int>(blockSize.y) * kNIY),
                  batchSize);

    int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    auto src = cuda::CreateInterpolationWrapNHW<const T, NVCV_BORDER_REPLICATE, I, int32_t>(inData);
    auto dst = cuda::CreateTensorWrapNHW<T, int32_t>(outData);

    RotateTensorKernel<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, d_aCoeffs);

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void RunRotateTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                            float *d_aCoeffs, const double angleDeg, const double2 shift,
                            const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    const bool dispatched = DispatchInterp(
        interpolation, [&](auto interp)
        { RunRotateTensor<T, decltype(interp)::value>(inData, outData, d_aCoeffs, angleDeg, shift, stream); });

    if (!dispatched)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid rotate interpolation %d",
                              static_cast<int>(interpolation));
    }
}

// Build a single-channel (N*C, H, W, 1) NHWC view of a packed planar (NCHW/CHW) tensor. Rotate maps
// each output pixel to a source pixel with the same per-sample affine coefficients regardless of the
// channel, so plane (n, c) is just a single-channel image; viewing the N*C planes as flat samples lets
// the existing interleaved single-channel rotate kernel handle planar data unchanged (bit-exact with
// the equivalent NHWC single-channel rotate). Plane (n, c) is placed at sample index n*C + c, byte
// offset (n*C + c)*chStride, which matches the real layout only when the channel planes are tightly
// packed across samples (sampleStride == C*chStride); the caller guards that. A single sample
// (N == 1) always satisfies it. The scalar angle/shift apply to every plane, so the single set of
// d_aCoeffs computed by ComputeWarpAffine drives all N*C flattened samples.
inline nvcv::TensorDataStridedCuda PlanarAsSingleChannelView(const nvcv::TensorDataStridedCuda              &data,
                                                             const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    const int64_t numSamples  = access.numSamples();
    const int64_t numChannels = access.numChannels();

    nvcv::TensorDataStridedCuda::Buffer buf;
    buf.basePtr    = reinterpret_cast<NVCVByte *>(data.basePtr());
    buf.strides[0] = access.chStride();  // N*C flattened planes
    buf.strides[1] = access.rowStride(); // H
    buf.strides[2] = access.colStride(); // W
    buf.strides[3] = access.colStride(); // C == 1
    return nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{numSamples * numChannels, access.numRows(), access.numCols(), 1}, "NHWC"},
        data.dtype(),
        buf
    };
}

template<typename BT>
inline void RunRotatePlanarFusedCubicTensor(const nvcv::TensorDataStridedCuda &inView,
                                            const nvcv::TensorDataStridedCuda &outView, int numPlanes, int numSamples,
                                            float *d_aCoeffs, const double angleDeg, const double2 shift,
                                            cudaStream_t stream)
{
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outView);
    NVCV_ASSERT(outAccess);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inView);
    NVCV_ASSERT(inAccess);

    const int2 dstSize{outAccess->numCols(), outAccess->numRows()};
    const int2 srcSize{inAccess->numCols(), inAccess->numRows()};

    const int64_t inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    const int64_t outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", cuda::TypeTraits<int32_t>::max);
    }

    ComputeWarpAffine<<<1, 1, 0, stream>>>(angleDeg, shift.x, shift.y, d_aCoeffs);

    auto src = cuda::CreateTensorWrapNHW<const BT, int32_t>(inView);
    auto dst = cuda::CreateTensorWrapNHW<BT, int32_t>(outView);

    dim3 blockSize(kBlock, kBlock / 4, 1);
    dim3 gridSize(util::DivUp(dstSize.x, blockSize.x), util::DivUp(dstSize.y, blockSize.y), numSamples);

    switch (numPlanes)
    {
    case 1:
        RotatePlanarFusedTensorKernel<1><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    case 3:
        RotatePlanarFusedTensorKernel<3><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    case 4:
        RotatePlanarFusedTensorKernel<4><<<gridSize, blockSize, 0, stream>>>(src, dst, dstSize, srcSize, d_aCoeffs);
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid planar channel number %d", numPlanes);
    }

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T, NVCVInterpolationType I>
inline void RunRotateVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs, cudaStream_t stream)
{
    nvcv::Size2D outMaxSize = out.maxSize();

    constexpr int kNIY = kRotateRowsPerThread<T, I>;

    dim3 blockSize(kBlock, kBlock / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(maxH / (blockY*kNIY)).
    dim3 gridSize(util::DivUp(outMaxSize.w, blockSize.x),
                  util::DivUp(outMaxSize.h, static_cast<int>(blockSize.y) * kNIY), in.numImages());

    cuda::InterpolationVarShapeWrap<const T, NVCV_BORDER_REPLICATE, I> src(in);
    cuda::ImageBatchVarShapeWrap<T>                                    dst(out);

    RotateVarShapeKernel<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void RunRotateVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                              const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
                              const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    NVCV_ASSERT(in.numImages() == out.numImages());

    const bool dispatched = DispatchInterp(
        interpolation, [&](auto interp) { RunRotateVarShape<T, decltype(interp)::value>(in, out, d_aCoeffs, stream); });

    if (!dispatched)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation type");
    }
}

template<typename T, NVCVInterpolationType I>
inline void RunRotatePlanarVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                    const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
                                    const int channels, cudaStream_t stream)
{
    nvcv::Size2D outMaxSize = out.maxSize();

    constexpr int kNIY = kRotateRowsPerThread<T, I>;

    dim3 blockSize(kBlock, kBlock / 4, 1);
    // Each thread emits kNIY output rows, so the grid covers ceil(maxH / (blockY*kNIY)).
    dim3 gridSize(util::DivUp(outMaxSize.w, blockSize.x),
                  util::DivUp(outMaxSize.h, static_cast<int>(blockSize.y) * kNIY), in.numImages());

    cuda::ImageBatchVarShapeWrap<T> dst(out);

    if constexpr (I == NVCV_INTERP_CUBIC)
    {
        cuda::ImageBatchVarShapeWrap<const T> src(in);

        switch (channels)
        {
        case 1:
            RotatePlanarFusedVarShapeKernel<1><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        case 3:
            RotatePlanarFusedVarShapeKernel<3><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        case 4:
            RotatePlanarFusedVarShapeKernel<4><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs);
            NVCV_CHECK_THROW(cudaGetLastError());
            return;
        default:
            break;
        }
    }

    cuda::InterpolationVarShapeWrap<const T, NVCV_BORDER_REPLICATE, I> src(in);

    RotatePlanarVarShapeKernel<kNIY><<<gridSize, blockSize, 0, stream>>>(src, dst, d_aCoeffs, channels);

    NVCV_CHECK_THROW(cudaGetLastError());
}

template<typename T>
inline void RunRotatePlanarVarShape(const nvcv::ImageBatchVarShapeDataStridedCuda &in,
                                    const nvcv::ImageBatchVarShapeDataStridedCuda &out, float *d_aCoeffs,
                                    const int channels, const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    const bool dispatched = DispatchInterp(
        interpolation, [&](auto interp)
        { RunRotatePlanarVarShape<T, decltype(interp)::value>(in, out, d_aCoeffs, channels, stream); });

    if (!dispatched)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid interpolation type");
    }
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

template<typename T>
struct TypeTag
{
    using type = T;
};

// The five admitted element types. Also the planar dispatch on its own: each channel plane is
// rotated as a separate single-channel image, so one scalar specialization per dtype covers any
// channel count. LINEAR/CUBIC sample through the interpolation wrap, which accumulates in float and
// rounds back to the element type, so F16 instantiates real half kernels rather than aliasing the
// 16-bit integer ones. IsAdmittedElem gates this, so the other ElemKind values are unreachable.
template<class F>
inline void DispatchBaseType(ElemKind elem, F &&fn)
{
    switch (elem)
    {
    case ElemKind::kU8:
        return fn(TypeTag<uchar>{});
    case ElemKind::kU16:
        return fn(TypeTag<ushort>{});
    case ElemKind::kS16:
        return fn(TypeTag<short>{});
    case ElemKind::kF32:
        return fn(TypeTag<float>{});
    case ElemKind::kF16:
        return fn(TypeTag<__half>{});
    default:
        break;
    }

    NVCV_ASSERT(false);
}

// Interleaved dispatch: the pixel type is the element type widened to the channel count. Only 1, 3
// and 4 channels have kernels, and the channel gate that runs first rejects everything else.
// MakeType<BT, 0> is the scalar BT itself; MakeType<BT, 1> would be the one-element vector (uchar1
// rather than uchar), a distinct type that would instantiate a different kernel.
template<class F>
inline void DispatchPixelType(ElemKind elem, int channels, F &&fn)
{
    DispatchBaseType(elem,
                     [&](auto baseTag)
                     {
                         using BT = typename decltype(baseTag)::type;

                         switch (channels)
                         {
                         case 1:
                             return fn(TypeTag<cuda::MakeType<BT, 0>>{});
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

} // namespace

namespace cvcuda::priv {

Rotate::AffineCoeffs::AffineCoeffs(int numBlocks)
{
    if (numBlocks > 0)
    {
        NVCV_CHECK_THROW(cudaMalloc(&aCoeffs, sizeof(float) * kAffineCoeffs * numBlocks));
    }
}

Rotate::AffineCoeffs::~AffineCoeffs()
{
    NVCV_CHECK_LOG(cudaFree(aCoeffs));
}

Rotate::Rotate(const int maxVarShapeBatchSize)
    : m_maxVarShapeBatchSize(maxVarShapeBatchSize)
    , m_tensorCoeffs([](int) { return std::make_unique<AffineCoeffs>(1); })
    , m_varShapeCoeffs([maxVarShapeBatchSize](int) { return std::make_unique<AffineCoeffs>(maxVarShapeBatchSize); })
{
}

void Rotate::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out, const double angleDeg,
                        const double2 shift, const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Rotate::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    float *d_aCoeffs = m_tensorCoeffs.get().aCoeffs;

    // ClassifyLayout admits exactly the four layouts the legacy format check listed, so the
    // legacy "Invalid input DataFormat" branch behind it was unreachable and is not reproduced.
    LayoutKind inputFormat  = ClassifyLayout(inData->layout());
    LayoutKind outputFormat = ClassifyLayout(outData->layout());

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);

    const ElemKind elem     = ClassifyElem(inData->dtype());
    const int      channels = inAccess->infoShape().numChannels();

    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    if (!IsAdmittedElem(elem))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(elem));
    }

    cvcuda::priv::ValidateInterpolation(interpolation);

    if (IsPlanar(inputFormat))
    {
        // View each of the N*C channel planes as a single-channel sample and reuse the interleaved
        // single-channel rotate kernel. Channels are independent in rotate, so
        // this is bit-exact with rotating the equivalent NHWC single-channel data. The flattened plane
        // count becomes the kernel's grid.z, capped at CUDA's 65535 limit; compute it in 64-bit to
        // avoid overflow. The flattened view is only valid when the planes are tightly packed across
        // samples.
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        NVCV_ASSERT(outAccess);

        const int64_t numSamples = inAccess->numSamples();
        if (numSamples > 1
            && (inAccess->sampleStride() != static_cast<int64_t>(channels) * inAccess->chStride()
                || outAccess->sampleStride() != static_cast<int64_t>(channels) * outAccess->chStride()))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar rotate of a batched tensor requires tightly packed channel planes");
        }
        if (numSamples * channels > 65535)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Planar rotate requires numSamples * numChannels <= 65535 (CUDA grid-z limit)");
        }

        auto srcView = PlanarAsSingleChannelView(*inData, *inAccess);
        auto dstView = PlanarAsSingleChannelView(*outData, *outAccess);

        if (interpolation == NVCV_INTERP_CUBIC)
        {
            DispatchBaseType(elem,
                             [&](auto tag)
                             {
                                 using BT = typename decltype(tag)::type;
                                 RunRotatePlanarFusedCubicTensor<BT>(srcView, dstView, channels,
                                                                     static_cast<int>(numSamples), d_aCoeffs, angleDeg,
                                                                     shift, stream);
                             });
            return;
        }

        DispatchBaseType(elem,
                         [&](auto tag)
                         {
                             using BT = typename decltype(tag)::type;
                             RunRotateTensor<BT>(srcView, dstView, d_aCoeffs, angleDeg, shift, interpolation, stream);
                         });
        return;
    }

    DispatchPixelType(elem, channels,
                      [&](auto tag)
                      {
                          using T = typename decltype(tag)::type;
                          RunRotateTensor<T>(*inData, *outData, d_aCoeffs, angleDeg, shift, interpolation, stream);
                      });
}

void Rotate::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in, const nvcv::ImageBatchVarShape &out,
                        const nvcv::Tensor &angleDeg, const nvcv::Tensor &shift,
                        const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::Rotate::operator()[ImageBatchVarShape]");
    auto inData = RequireCudaVarShape(in, stream, "Input");

    auto outData = RequireCudaVarShape(out, stream, "Output");

    auto angleDegData = angleDeg.exportData<nvcv::TensorDataStridedCuda>();
    if (angleDegData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "angleDeg must be a tensor");
    }

    auto shiftData = shift.exportData<nvcv::TensorDataStridedCuda>();
    if (shiftData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "shift must be a tensor");
    }

    float *d_aCoeffs = m_varShapeCoeffs.get().aCoeffs;

    RequireUniqueFormat(*inData, "input");

    RequireUniqueFormat(*outData, "output");

    if (m_maxVarShapeBatchSize <= 0)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Operator rotate var shape is not initialized properly, maxVarShapeBatchSize: %d",
                              m_maxVarShapeBatchSize);
    }

    if (m_maxVarShapeBatchSize < inData->numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of images, it should not exceed %d",
                              m_maxVarShapeBatchSize);
    }

    LayoutKind inputFormat  = ClassifyLayout(*inData);
    LayoutKind outputFormat = ClassifyLayout(*outData);

    if (inputFormat != outputFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inputFormat),
                              static_cast<int>(outputFormat));
    }

    const bool isPlanar = IsPlanar(inputFormat);

    const int channels = inData->uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format, and it matches
    // the Resize and Normalize operators. Interleaved 2-channel is rejected too -- the legacy table
    // had no 2-channel column, so it dereferenced a null function pointer for it.
    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    // The planar path launches grid.z over images and loops the channel planes inside the kernel, so
    // numImages must fit CUDA's 65535 grid-z limit. Compute in 64-bit to avoid overflow.
    if (isPlanar && static_cast<int64_t>(outData->numImages()) > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar rotate requires numImages <= 65535 (CUDA grid-z limit)");
    }

    const ElemKind elem      = ClassifyElem(inData->uniqueFormat());
    const ElemKind angleElem = ClassifyElem(angleDegData->dtype());
    const ElemKind shiftElem = ClassifyElem(shiftData->dtype());

    if (!IsAdmittedElem(elem))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(elem));
    }

    if (angleElem != ElemKind::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid angleDeg DataType %d",
                              static_cast<int>(elem));
    }

    if (shiftElem != ElemKind::kF64)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid shift DataType %d",
                              static_cast<int>(elem));
    }

    cvcuda::priv::ValidateInterpolation(interpolation);

    cuda::Tensor1DWrap<double> angleDegPtr(*angleDegData);
    cuda::Tensor2DWrap<double> shiftPtr(*shiftData);

    ComputeWarpAffineVarShape<<<1, inData->numImages(), 0, stream>>>(inData->numImages(), angleDegPtr, shiftPtr,
                                                                     d_aCoeffs);
    NVCV_CHECK_THROW(cudaGetLastError());

    if (isPlanar)
    {
        DispatchBaseType(elem,
                         [&](auto tag)
                         {
                             using BT = typename decltype(tag)::type;
                             RunRotatePlanarVarShape<BT>(*inData, *outData, d_aCoeffs, channels, interpolation, stream);
                         });
        return;
    }

    DispatchPixelType(elem, channels,
                      [&](auto tag)
                      {
                          using T = typename decltype(tag)::type;
                          RunRotateVarShape<T>(*inData, *outData, d_aCoeffs, interpolation, stream);
                      });
}

} // namespace cvcuda::priv
