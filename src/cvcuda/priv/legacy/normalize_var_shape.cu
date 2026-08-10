/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "normalize_planar.cuh" // for PlanarBroadcastIndex, ApplyPlanarNormalize

#include <cvcuda/OpNormalize.h> // for CVCUDA_NORMALIZE_SCALE_IS_STDDEV, etc.
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp> // for TypeTraits

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nvcv::legacy::cuda_op {

namespace {

#define BLOCK 32

// ILP depth used by the single-channel 1-byte inverse-std-dev var-shape wide-load path: each thread
// issues this many independent uchar4 loads before computing, raising the number of outstanding
// memory requests per resident thread (memory-level parallelism) at fixed occupancy. Matches the
// tensor-path knob; NGROUP=4 measured the best long-scoreboard-stall reduction without register
// pressure cutting occupancy.
static constexpr int kNormalizeILPNGroup = 4;

// ILP depth used by the single-channel F32 inverse-std-dev var-shape wide-load path. The scalar F32
// var-shape kernel moves one float (4 bytes) per thread, leaving the single-channel F32 path
// latency-bound (too few bytes in flight per thread to hide load latency). Here each thread owns
// NGROUP independent float4 groups (4*NGROUP consecutive columns) and issues all NGROUP loads before
// any compute, raising the number of outstanding memory requests per resident thread (memory-level
// parallelism) at fixed occupancy. NGROUP=4 matches the single-channel U8 var-shape value; a float4
// group is 16 bytes of live register state, so if registers balloon and cut occupancy this is the
// knob to drop to 2.
static constexpr int kNormalizeF32ILPNGroup = 4;

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normKernel(const cuda::ImageBatchVarShapeWrap<const T> src, cuda::ImageBatchVarShapeWrap<out_T> dst,
                           const scale_type *scale, const base_type *base, float global_scale, float global_shift)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    T out                             = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<out_T>((out - *base) * *scale * global_scale + global_shift);
}

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                    cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                    const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.width(batch_idx) || dst_y >= dst.height(batch_idx))
        return;

    scale_type s   = *scale;
    scale_type x   = s * s + epsilon;
    scale_type mul = 1.0f / cuda::sqrt(x);

    T out                             = *src.ptr(batch_idx, dst_y, dst_x);
    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<out_T>((out - *base) * mul * global_scale + global_shift);
}

// Vectorized 1-byte single-channel interleaved (NHWC) var-shape inverse-std-dev kernel: one thread
// owns NGROUP uchar4 groups (4*NGROUP pixels) of an image row, each moved with a single uchar4
// load/store, so a warp transfers full 128-byte lines. base/scale are scalar for single-channel
// input, so the inverse-std-dev multiplier is computed once per thread. The NGROUP uchar4 loads are
// issued up front (loads-first ILP) so several memory requests overlap per resident thread, raising
// memory-level parallelism at fixed occupancy on the latency-bound 1-byte path. Bit-identical to
// normInvStdDevKernel per element; columns past the last full uchar4 group stay scalar. Relies on
// NVCV's image row-pitch alignment (>= 4 bytes).
template<int NGROUP, typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevVec4Kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                        cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                        const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int g0        = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int width = dst.width(batch_idx);
    const int cx0   = g0 * 4;
    if (cx0 >= width || dst_y >= dst.height(batch_idx))
        return;

    const scale_type s   = *scale;
    const scale_type mul = 1.0f / cuda::sqrt(s * s + epsilon);
    const base_type  b   = *base;

    auto apply = [&](uchar raw) -> out_T
    { return cuda::SaturateCast<out_T>((raw - b) * mul * global_scale + global_shift); };

    // Loads first: issue all NGROUP uchar4 loads (for full in-range groups) before any compute.
    int    cx[NGROUP];
    bool   full[NGROUP];
    uchar4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
            in4[i] = *reinterpret_cast<const uchar4 *>(src.ptr(batch_idx, dst_y, cx[i]));
    }

    // Compute then store. Groups that are not a full uchar4 (the trailing columns) fall back to scalar.
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            uchar4 out4;
            out4.x                                                = apply(in4[i].x);
            out4.y                                                = apply(in4[i].y);
            out4.z                                                = apply(in4[i].z);
            out4.w                                                = apply(in4[i].w);
            *reinterpret_cast<uchar4 *>(dst.ptr(batch_idx, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch_idx, dst_y, x) = apply(*src.ptr(batch_idx, dst_y, x));
            }
        }
    }
}

// Vectorized single-channel F32 interleaved (NHWC) var-shape inverse-std-dev kernel: one thread owns
// NGROUP independent float4 groups (4*NGROUP columns) of an image row, each moved with a single 128-bit
// float4 load/store. base/scale are scalar for single-channel input, so the inverse-std-dev multiplier
// is computed once per thread. All NGROUP float4 loads are issued up front (loads-first ILP) so several
// memory requests overlap per resident thread, raising memory-level parallelism at fixed occupancy on
// the latency-bound single-channel F32 path. The per-element math matches normInvStdDevKernel (same FMA
// expression; F32 output needs no SaturateCast since the value is already float), so output is
// bit-for-bit unchanged. Relies on NVCV's image row-pitch alignment (>= 16 bytes, caller-guarded);
// columns past the last full float4 group stay scalar.
template<int NGROUP, typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevF32Vec4Kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                           cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                           const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int g0        = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int width = dst.width(batch_idx);
    const int cx0   = g0 * 4;
    if (cx0 >= width || dst_y >= dst.height(batch_idx))
        return;

    const scale_type s   = *scale;
    const scale_type mul = 1.0f / cuda::sqrt(s * s + epsilon);
    const base_type  b   = *base;

    auto apply = [&](float raw) -> out_T { return (raw - b) * mul * global_scale + global_shift; };

    // Loads first: issue all NGROUP float4 loads (for full in-range groups) before any compute.
    int    cx[NGROUP];
    bool   full[NGROUP];
    float4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
            in4[i] = *reinterpret_cast<const float4 *>(src.ptr(batch_idx, dst_y, cx[i]));
    }

    // Compute then store. Groups that are not a full float4 (the trailing columns) fall back to scalar.
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        if (full[i])
        {
            float4 out4;
            out4.x                                                       = apply(in4[i].x);
            out4.y                                                       = apply(in4[i].y);
            out4.z                                                       = apply(in4[i].z);
            out4.w                                                       = apply(in4[i].w);
            *reinterpret_cast<float4 *>(dst.ptr(batch_idx, dst_y, cx[i])) = out4;
        }
        else if (cx[i] < width)
        {
            for (int x = cx[i]; x < width; ++x)
            {
                *dst.ptr(batch_idx, dst_y, x) = apply(*src.ptr(batch_idx, dst_y, x));
            }
        }
    }
}

// Inverse-std-dev interleaved (NHWC) var-shape kernel processing NIX pixels per thread. base/scale
// are per-channel constants for the whole batch (single pointer deref), so the multiplier
// mul = 1 / sqrt(scale^2 + eps) is identical for every pixel; the scalar normInvStdDevKernel
// recomputes it per pixel, which makes the multi-channel paths (uchar3 / uchar4) issue-bound on the
// redundant sqrt rather than memory-bound. Here each thread computes mul once and applies it to NIX
// pixels strided by blockDim.x so each warp iteration stays coalesced. Bit-identical to
// normInvStdDevKernel (same per-element arithmetic; only the multiplier hoists out of the loop).
template<int NIX, typename T, typename out_T, typename base_type, typename scale_type>
__global__ void normInvStdDevHoistKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                         cuda::ImageBatchVarShapeWrap<out_T> dst, const scale_type *scale,
                                         const base_type *base, float global_scale, float global_shift, float epsilon)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int width = dst.width(batch_idx);
    if (dst_y >= dst.height(batch_idx))
        return;

    const base_type  b   = *base;
    const scale_type s   = *scale;
    const scale_type x   = s * s + epsilon;
    const scale_type mul = 1.0f / cuda::sqrt(x);

    const int x0 = blockIdx.x * blockDim.x * NIX + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int dst_x = x0 + i * blockDim.x;
        if (dst_x < width)
        {
            *dst.ptr(batch_idx, dst_y, dst_x)
                = cuda::SaturateCast<out_T>((*src.ptr(batch_idx, dst_y, dst_x) - b) * mul * global_scale + global_shift);
        }
    }
}

// Planar (NCHW / CHW) var-shape kernel: src/dst are per-image planar wraps indexed (n, c, y, x);
// base/scale are 4D scalar wraps broadcast per their logical (N, C, H, W) shape. is_stddev selects
// the inverse-stddev formula; the per-element math is shared with the tensor path.
template<typename T, typename out_T>
__global__ void normPlanarKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                 cuda::ImageBatchVarShapeWrap<out_T> dst,
                                 const cuda::Tensor4DWrap<float, int32_t> base,
                                 const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels, int4 base_size,
                                 int4 scale_size, float global_scale, float global_shift, float epsilon,
                                 bool is_stddev)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    if (dst_x >= dst.width(batch_idx, channel) || dst_y >= dst.height(batch_idx, channel))
        return;

    const int4 b = PlanarBroadcastIndex(base_size, batch_idx, channel, dst_y, dst_x);
    const int4 s = PlanarBroadcastIndex(scale_size, batch_idx, channel, dst_y, dst_x);

    *dst.ptr(batch_idx, channel, dst_y, dst_x)
        = ApplyPlanarNormalize<out_T>(*src.ptr(batch_idx, channel, dst_y, dst_x), *base.ptr(b.x, b.y, b.z, b.w),
                                      *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev, epsilon);
}

// Vectorized 1-byte planar var-shape kernel: one thread per four columns of plane (batch, channel),
// via the shared uchar4 body. Bit-identical to normPlanarKernel. Relies on NVCV's image row-pitch
// alignment (>= 4 bytes) so the uchar4 access is aligned; the existing planar var-shape tests
// (RGB8p / RGBA8p) validate correctness, including the W%4 scalar tail.
template<typename T, typename out_T>
__global__ void normPlanarVec4Kernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                     cuda::ImageBatchVarShapeWrap<out_T>      dst,
                                     const cuda::Tensor4DWrap<float, int32_t> base,
                                     const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels, int4 base_size,
                                     int4 scale_size, float global_scale, float global_shift, float epsilon,
                                     bool is_stddev)
{
    const int cx        = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    const int width = dst.width(batch_idx, channel);
    if (cx >= width || dst_y >= dst.height(batch_idx, channel))
        return;

    // The shared body reads only inout_size.w (the per-image width) for the vector/tail split.
    const int4 inout_size = {0, 0, 0, width};
    NormalizePlanarVec4<T>(src, dst, base, scale, batch_idx, channel, dst_y, cx, inout_size, base_size, scale_size,
                           global_scale, global_shift, is_stddev, epsilon);
}

// Vectorized 1-byte planar var-shape kernel for spatially-broadcast base/scale (the common
// per-channel case), processing NGROUP uchar4 groups (NGROUP*4 columns) per thread. normPlanarVec4Kernel
// above moves only 4 bytes per thread (one channel), so the fixed per-thread cost -- the per-element
// multiplier and, more importantly, the var-shape ImageBatchVarShapeWrap::ptr metadata lookup -- is
// amortized over 3x less data than the interleaved path, leaving the kernel co-limited by compute
// (~76% memory and ~76% compute SOL). Here each thread resolves base/scale and mul once and the plane
// row base pointer once (a single var-shape lookup; plane columns are contiguous), then strides across
// NGROUP coalesced uchar4 groups. Output is bit-identical to the broadcast path of NormalizePlanarVec4.
template<int NGROUP, typename T, typename out_T>
__global__ void normPlanarVec4HoistKernel(const cuda::ImageBatchVarShapeWrap<const T> src,
                                          cuda::ImageBatchVarShapeWrap<out_T>      dst,
                                          const cuda::Tensor4DWrap<float, int32_t> base,
                                          const cuda::Tensor4DWrap<float, int32_t> scale, int num_channels,
                                          int4 base_size, int4 scale_size, float global_scale, float global_shift,
                                          float epsilon, bool is_stddev)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc        = blockIdx.z;
    const int batch_idx = nc / num_channels;
    const int channel   = nc % num_channels;

    const int width = dst.width(batch_idx, channel);
    if (dst_y >= dst.height(batch_idx, channel))
        return;

    // base/scale are broadcast across H and W here (caller-guarded), so read them once per thread.
    const int4  b      = PlanarBroadcastIndex(base_size, batch_idx, channel, 0, 0);
    const int4  s      = PlanarBroadcastIndex(scale_size, batch_idx, channel, 0, 0);
    const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
    const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
    const float mul    = is_stddev ? (1.0f / nvcv::cuda::sqrt(scaleV * scaleV + epsilon)) : scaleV;

    // One var-shape lookup per thread; plane columns are contiguous, so groups index off the row base.
    const T *const rowSrc = src.ptr(batch_idx, channel, dst_y, 0);
    out_T *const   rowDst = dst.ptr(batch_idx, channel, dst_y, 0);

    const int g0 = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        const int cx = (g0 + i * blockDim.x) * 4;
        if (cx + 4 <= width)
        {
            const uchar4 in4 = *reinterpret_cast<const uchar4 *>(rowSrc + cx);
            uchar4       out4;
            out4.x = cuda::SaturateCast<out_T>((static_cast<float>(in4.x) - baseV) * mul * global_scale + global_shift);
            out4.y = cuda::SaturateCast<out_T>((static_cast<float>(in4.y) - baseV) * mul * global_scale + global_shift);
            out4.z = cuda::SaturateCast<out_T>((static_cast<float>(in4.z) - baseV) * mul * global_scale + global_shift);
            out4.w = cuda::SaturateCast<out_T>((static_cast<float>(in4.w) - baseV) * mul * global_scale + global_shift);
            *reinterpret_cast<uchar4 *>(rowDst + cx) = out4;
        }
        else if (cx < width)
        {
            for (int x = cx; x < width; ++x)
            {
                rowDst[x] = cuda::SaturateCast<out_T>((static_cast<float>(rowSrc[x]) - baseV) * mul * global_scale
                                                      + global_shift);
            }
        }
    }
}

template<typename T, typename out_T, typename base_type, typename scale_type>
void normWrap(const ImageBatchVarShapeDataStridedCuda &in, const base_type *base, const scale_type *scale,
              const ImageBatchVarShapeDataStridedCuda &out, float global_scale, float shift, cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3                                  block(BLOCK, BLOCK / 4, 1);
    dim3                                  grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);
    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<out_T>   dst_ptr(out);

    normKernel<T, out_T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift);
    checkKernelErrors();
}

template<typename T, typename out_T, typename base_type, typename scale_type>
void normInvStdDevWrap(const ImageBatchVarShapeDataStridedCuda &in, const base_type *base, const scale_type *scale,
                       const ImageBatchVarShapeDataStridedCuda &out, float global_scale, float shift, float epsilon,
                       cudaStream_t stream)
{
    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), batch);

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<out_T>   dst_ptr(out);

    // Single-channel 1-byte input goes through the vectorized (4 pixels/thread) path; sizeof(T) == 1
    // implies one channel here, so base/scale are scalar. NVCV row-pitch alignment makes the uchar4
    // access aligned; the W%4 tail is handled scalar.
    if constexpr (sizeof(T) == 1 && sizeof(out_T) == 1)
    {
        constexpr int NGROUP = kNormalizeILPNGroup;
        // Each thread covers NGROUP uchar4 groups (4*NGROUP columns) with loads-first ILP.
        dim3 vgrid(divUp(divUp(max_width, 4), static_cast<int>(block.x) * NGROUP),
                   divUp(max_height, static_cast<int>(block.y)), batch);
        normInvStdDevVec4Kernel<NGROUP, T, out_T>
            <<<vgrid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift, epsilon);
        checkKernelErrors();
        return;
    }

    // Single-channel F32 input goes through the vectorized (4*NGROUP columns/thread, 128-bit float4
    // transfers, loads-first ILP) path; T == float implies one channel here, so base/scale are scalar.
    // NVCV image row pitch is aligned to >= 256 bytes and float4 column offsets are multiples of 16
    // bytes, so the float4 access is aligned; the W%4 tail is handled scalar. Output is bit-identical to
    // normInvStdDevKernel (same float expression, no SaturateCast needed for float out).
    if constexpr (std::is_same_v<T, float> && std::is_same_v<out_T, float>)
    {
        constexpr int NGROUP = kNormalizeF32ILPNGroup;
        // Each thread covers NGROUP float4 groups (4*NGROUP columns) with loads-first ILP.
        dim3 vgrid(divUp(divUp(max_width, 4), static_cast<int>(block.x) * NGROUP),
                   divUp(max_height, static_cast<int>(block.y)), batch);
        normInvStdDevF32Vec4Kernel<NGROUP, T, out_T>
            <<<vgrid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift, epsilon);
        checkKernelErrors();
        return;
    }

    // Multi-channel 1-byte interleaved input (uchar3 / uchar4) is issue-bound on the redundant
    // per-pixel sqrt; hoist the multiplier and process NIX pixels per thread. float3 keeps the scalar
    // kernel since it is already memory-bound.
    if constexpr (sizeof(cuda::BaseType<T>) == 1)
    {
        constexpr int NIX = 4;
        dim3          hgrid(divUp(max_width, static_cast<int>(block.x) * NIX), divUp(max_height, static_cast<int>(block.y)),
                            batch);
        normInvStdDevHoistKernel<NIX, T, out_T>
            <<<hgrid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift, epsilon);
        checkKernelErrors();
        return;
    }

    normInvStdDevKernel<T, out_T>
        <<<grid, block, 0, stream>>>(src_ptr, dst_ptr, scale, base, global_scale, shift, epsilon);
    checkKernelErrors();
}

// Launch path for the planar varshape variants; isStdDev/epsilon are forwarded to the unified
// kernel (epsilon is unused when isStdDev is false).
template<typename T, typename out_T>
ErrorCode normPlanarImpl(const ImageBatchVarShapeDataStridedCuda &in, const TensorDataStridedCuda &baseData,
                         const TensorDataStridedCuda &scaleData, const ImageBatchVarShapeDataStridedCuda &out,
                         int num_channels, float global_scale, float shift, bool isStdDev, float epsilon,
                         cudaStream_t stream)
{
    auto baseAccess = TensorDataAccessStridedImagePlanar::Create(baseData);
    NVCV_ASSERT(baseAccess);

    auto scaleAccess = TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(scaleAccess);

    auto baseMaxStride  = baseAccess->sampleStride() * baseAccess->numSamples();
    auto scaleMaxStride = scaleAccess->sampleStride() * scaleAccess->numSamples();
    if (std::max(baseMaxStride, scaleMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Base or scale size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    int max_width  = in.maxSize().w;
    int max_height = in.maxSize().h;
    int batch      = in.numImages();

    int4 base_size = {static_cast<int>(baseAccess->numSamples()), static_cast<int>(baseAccess->numChannels()),
                      static_cast<int>(baseAccess->numRows()), static_cast<int>(baseAccess->numCols())};
    int4 scale_size = {static_cast<int>(scaleAccess->numSamples()), static_cast<int>(scaleAccess->numChannels()),
                       static_cast<int>(scaleAccess->numRows()), static_cast<int>(scaleAccess->numCols())};

    const uint64_t planes = static_cast<uint64_t>(batch) * num_channels;
    if (planes > 65535u)
    {
        LOG_ERROR("Planar normalize launch exceeds CUDA grid.z limit: N*C=" << planes);
        return ErrorCode::INVALID_PARAMETER;
    }

    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), static_cast<unsigned int>(planes));

    cuda::ImageBatchVarShapeWrap<const T> src_ptr(in);
    cuda::ImageBatchVarShapeWrap<out_T>   dst_ptr(out);
    auto                                  baseWrap  = cuda::CreateTensorWrapNCHW<float, int32_t>(baseData);
    auto                                  scaleWrap = cuda::CreateTensorWrapNCHW<float, int32_t>(scaleData);

    // 1-byte planes go through the vectorized (4 columns/thread) body; NVCV row-pitch alignment
    // (>= 4 bytes) guarantees the uchar4 access is aligned, and the W%4 tail is handled scalar.
    if constexpr (sizeof(T) == 1 && sizeof(out_T) == 1)
    {
        dim3 vblock(BLOCK, BLOCK / 4, 1);

        // When base/scale are broadcast across H and W (the common per-channel case) each thread can
        // resolve them and the plane row pointer once, then process several uchar4 groups -- amortizing
        // the var-shape lookup over more data so the kernel reaches memory-bound throughput.
        const bool spatialBroadcast
            = base_size.z == 1 && base_size.w == 1 && scale_size.z == 1 && scale_size.w == 1;
        if (spatialBroadcast)
        {
            constexpr int NGROUP = 4;
            dim3          hgrid(divUp(divUp(max_width, 4), static_cast<int>(vblock.x) * NGROUP),
                                divUp(max_height, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
            normPlanarVec4HoistKernel<NGROUP, T, out_T><<<hgrid, vblock, 0, stream>>>(
                src_ptr, dst_ptr, baseWrap, scaleWrap, num_channels, base_size, scale_size, global_scale, shift,
                epsilon, isStdDev);
            checkKernelErrors();
            return ErrorCode::SUCCESS;
        }

        dim3 vgrid(divUp(divUp(max_width, 4), static_cast<int>(vblock.x)), divUp(max_height, static_cast<int>(vblock.y)),
                   static_cast<unsigned int>(planes));
        normPlanarVec4Kernel<T, out_T><<<vgrid, vblock, 0, stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap,
                                                                     num_channels, base_size, scale_size, global_scale,
                                                                     shift, epsilon, isStdDev);
        checkKernelErrors();
        return ErrorCode::SUCCESS;
    }

    // F32 planes (float in, float out) go through the same vectorized body but with a 128-bit float4
    // load/store per thread (4 columns). Plain float4 vectorization only -- no loads-first ILP here,
    // since ILP regressed the 1-byte planar var-shape path. The shared body checks the per-row 16-byte
    // alignment of the float4 access and falls back to a scalar group for any unaligned plane, so
    // user-wrapped buffers with a sub-16-byte row pitch stay correct; the W%4 tail is handled scalar.
    // Bit-identical to normPlanarKernel.
    if constexpr (std::is_same_v<T, float> && std::is_same_v<out_T, float>)
    {
        dim3 vblock(BLOCK, BLOCK / 4, 1);
        dim3 vgrid(divUp(divUp(max_width, 4), static_cast<int>(vblock.x)),
                   divUp(max_height, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
        normPlanarVec4Kernel<T, out_T><<<vgrid, vblock, 0, stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap,
                                                                     num_channels, base_size, scale_size, global_scale,
                                                                     shift, epsilon, isStdDev);
        checkKernelErrors();
        return ErrorCode::SUCCESS;
    }

    normPlanarKernel<T, out_T><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, baseWrap, scaleWrap, num_channels,
                                                           base_size, scale_size, global_scale, shift, epsilon,
                                                           isStdDev);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

template<typename T, typename out_T>
ErrorCode normPlanar(const ImageBatchVarShapeDataStridedCuda &in, const TensorDataStridedCuda &baseData,
                     const TensorDataStridedCuda &scaleData, const ImageBatchVarShapeDataStridedCuda &out,
                     int num_channels, float global_scale, float shift, cudaStream_t stream)
{
    return normPlanarImpl<T, out_T>(in, baseData, scaleData, out, num_channels, global_scale, shift, false, 0.f,
                                    stream);
}

template<typename T, typename out_T>
ErrorCode normPlanarInvStdDev(const ImageBatchVarShapeDataStridedCuda &in, const TensorDataStridedCuda &baseData,
                              const TensorDataStridedCuda &scaleData, const ImageBatchVarShapeDataStridedCuda &out,
                              int num_channels, float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    return normPlanarImpl<T, out_T>(in, baseData, scaleData, out, num_channels, global_scale, shift, true, epsilon,
                                    stream);
}

template<typename T, typename out_T>
void norm(const ImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
          const TensorDataAccessStridedImagePlanar &scale, const ImageBatchVarShapeDataStridedCuda &out,
          float global_scale, float shift, cudaStream_t stream)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    if (base.numChannels() != 1 && scale.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else if (base.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else if (scale.numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                           reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift, stream);
    }
}

template<typename T, typename out_T>
void normInvStdDev(const ImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
                   const TensorDataAccessStridedImagePlanar &scale, const ImageBatchVarShapeDataStridedCuda &out,
                   float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    if (base.numChannels() != 1 && scale.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else if (base.numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else if (scale.numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normInvStdDevWrap<T, out_T>(in, reinterpret_cast<const base_type *>(base.sampleData(0)),
                                    reinterpret_cast<const scale_type *>(scale.sampleData(0)), out, global_scale, shift,
                                    epsilon, stream);
    }
}

} // namespace

ErrorCode NormalizeVarShape::infer(const nvcv::ImageBatchVarShapeDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda             &baseData,
                                   const nvcv::TensorDataStridedCuda             &scaleData,
                                   const nvcv::ImageBatchVarShapeDataStridedCuda &outData, const float global_scale,
                                   const float shift, const float epsilon, const uint32_t flags, cudaStream_t stream)
{
    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType out_data_type = helpers::GetLegacyDataType(outData.uniqueFormat());

    if (!(out_data_type == kCV_8U || out_data_type == kCV_32F))
    {
        LOG_ERROR("Invalid Output DataType " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    int channels = inData.uniqueFormat().numChannels();
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto baseAccess = TensorDataAccessStridedImagePlanar::Create(baseData);
    if (!baseAccess)
    {
        LOG_ERROR("Invalid DataFormat(base) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto scaleAccess = TensorDataAccessStridedImagePlanar::Create(scaleData);
    if (!scaleAccess)
    {
        LOG_ERROR("Invalid DataFormat(scale) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (isPlanar)
    {
        auto isParamCompatible = [](const TensorDataStridedCuda &data,
                                    const TensorDataAccessStridedImagePlanar &access)
        {
            DataFormat param_format = helpers::GetLegacyDataFormat(data);
            if (param_format == kNCHW || param_format == kCHW)
            {
                return true;
            }
            if (param_format == kNHWC || param_format == kHWC)
            {
                return access.numChannels() == 1;
            }
            return false;
        };

        if (!isParamCompatible(baseData, *baseAccess))
        {
            LOG_ERROR("base DataFormat " << helpers::GetLegacyDataFormat(baseData)
                                         << " is not compatible with planar input; use NCHW/CHW (or a scalar layout)");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
        if (!isParamCompatible(scaleData, *scaleAccess))
        {
            LOG_ERROR("scale DataFormat " << helpers::GetLegacyDataFormat(scaleData)
                                          << " is not compatible with planar input; use NCHW/CHW (or a scalar layout)");
            return ErrorCode::INVALID_DATA_FORMAT;
        }

        auto isParamShapeCompatible = [channels](const TensorDataAccessStridedImagePlanar &access)
        {
            return (access.numSamples() == 1) && (access.numChannels() == 1 || access.numChannels() == channels)
                && access.numRows() == 1 && access.numCols() == 1;
        };

        if (!isParamShapeCompatible(*baseAccess))
        {
            LOG_ERROR("Invalid planar base shape; expected scalar [1,1,1,1] or per-channel [1,C,1,1]");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (!isParamShapeCompatible(*scaleAccess))
        {
            LOG_ERROR("Invalid planar scale shape; expected scalar [1,1,1,1] or per-channel [1,C,1,1]");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
    }

    typedef void (*normalize_t)(
        const ImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
        const TensorDataAccessStridedImagePlanar &scale, const ImageBatchVarShapeDataStridedCuda &out,
        float global_scale, float shift, cudaStream_t stream);

    typedef void (*normalizeInvStdDev_t)(
        const ImageBatchVarShapeDataStridedCuda &in, const TensorDataAccessStridedImagePlanar &base,
        const TensorDataAccessStridedImagePlanar &scale, const ImageBatchVarShapeDataStridedCuda &out,
        float global_scale, float shift, float epsilon, cudaStream_t stream);

    int out_type_code = out_data_type == kCV_8U ? 0 : 1;

    static const normalize_t funcs_normalize[6][2][4] = {
        {    {norm<uchar, uchar>, norm<uchar2, uchar2>, norm<uchar3, uchar3>, norm<uchar4, uchar4>},
         {norm<uchar, float>, norm<uchar2, float2>, norm<uchar3, float3>, norm<uchar4, float4>}    },
        {       {norm<schar, uchar>, norm<char2, uchar2>, norm<char3, uchar3>, norm<char4, uchar4>},
         {norm<schar, float>, norm<char2, float2>, norm<char3, float3>, norm<char4, float4>}       },
        {{norm<ushort, uchar>, norm<ushort2, uchar2>, norm<ushort3, uchar3>, norm<ushort4, uchar4>},
         {norm<ushort, float>, norm<ushort2, float2>, norm<ushort3, float3>, norm<ushort4, float4>}},
        {    {norm<short, uchar>, norm<short2, uchar2>, norm<short3, uchar3>, norm<short4, uchar4>},
         {norm<short, float>, norm<short2, float2>, norm<short3, float3>, norm<short4, float4>}    },
        {            {norm<int, uchar>, norm<int2, uchar2>, norm<int3, uchar3>, norm<int4, uchar4>},
         {norm<int, float>, norm<int2, float2>, norm<int3, float3>, norm<int4, float4>}            },
        {    {norm<float, uchar>, norm<float2, uchar2>, norm<float3, uchar3>, norm<float4, uchar4>},
         {norm<float, float>, norm<float2, float2>, norm<float3, float3>, norm<float4, float4>}    },
    };

    static const normalizeInvStdDev_t funcs_normalize_stddev[6][2][4] = {
        { {normInvStdDev<uchar, uchar>, normInvStdDev<uchar2, uchar2>, normInvStdDev<uchar3, uchar3>,
 normInvStdDev<uchar4, uchar4>},
         {normInvStdDev<uchar, float>, normInvStdDev<uchar2, float2>, normInvStdDev<uchar3, float3>,
         normInvStdDev<uchar4, float4>} },
        {  {normInvStdDev<schar, uchar>, normInvStdDev<char2, uchar2>, normInvStdDev<char3, uchar3>,
  normInvStdDev<char4, uchar4>},
         {normInvStdDev<schar, float>, normInvStdDev<char2, float2>, normInvStdDev<char3, float3>,
         normInvStdDev<char4, float4>}  },
        {{normInvStdDev<ushort, uchar>, normInvStdDev<ushort2, uchar2>, normInvStdDev<ushort3, uchar3>,
normInvStdDev<ushort4, uchar4>},
         {normInvStdDev<ushort, float>, normInvStdDev<ushort2, float2>, normInvStdDev<ushort3, float3>,
         normInvStdDev<ushort4, float4>}},
        { {normInvStdDev<short, uchar>, normInvStdDev<short2, uchar2>, normInvStdDev<short3, uchar3>,
 normInvStdDev<short4, uchar4>},
         {normInvStdDev<short, float>, normInvStdDev<short2, float2>, normInvStdDev<short3, float3>,
         normInvStdDev<short4, float4>} },
        {   {normInvStdDev<int, uchar>, normInvStdDev<int2, uchar2>, normInvStdDev<int3, uchar3>,
   normInvStdDev<int4, uchar4>},
         {normInvStdDev<int, float>, normInvStdDev<int2, float2>, normInvStdDev<int3, float3>,
         normInvStdDev<int4, float4>}   },
        { {normInvStdDev<float, uchar>, normInvStdDev<float2, uchar2>, normInvStdDev<float3, uchar3>,
 normInvStdDev<float4, uchar4>},
         {normInvStdDev<float, float>, normInvStdDev<float2, float2>, normInvStdDev<float3, float3>,
         normInvStdDev<float4, float4>} },
    };

    if (isPlanar)
    {
        typedef ErrorCode (*normalizePlanar_t)(const ImageBatchVarShapeDataStridedCuda &in,
                                               const TensorDataStridedCuda &baseData,
                                               const TensorDataStridedCuda &scaleData,
                                               const ImageBatchVarShapeDataStridedCuda &out, int num_channels,
                                               float global_scale, float shift, cudaStream_t stream);

        typedef ErrorCode (*normalizePlanarInvStdDev_t)(const ImageBatchVarShapeDataStridedCuda &in,
                                                        const TensorDataStridedCuda &baseData,
                                                        const TensorDataStridedCuda &scaleData,
                                                        const ImageBatchVarShapeDataStridedCuda &out, int num_channels,
                                                        float global_scale, float shift, float epsilon,
                                                        cudaStream_t stream);

        static const normalizePlanar_t funcs_planar[6][2] = {
            { normPlanar<uchar, uchar>,  normPlanar<uchar, float>},
            { normPlanar<schar, uchar>,  normPlanar<schar, float>},
            {normPlanar<ushort, uchar>, normPlanar<ushort, float>},
            { normPlanar<short, uchar>,  normPlanar<short, float>},
            {   normPlanar<int, uchar>,    normPlanar<int, float>},
            { normPlanar<float, uchar>,  normPlanar<float, float>},
        };

        static const normalizePlanarInvStdDev_t funcs_planar_stddev[6][2] = {
            { normPlanarInvStdDev<uchar, uchar>,  normPlanarInvStdDev<uchar, float>},
            { normPlanarInvStdDev<schar, uchar>,  normPlanarInvStdDev<schar, float>},
            {normPlanarInvStdDev<ushort, uchar>, normPlanarInvStdDev<ushort, float>},
            { normPlanarInvStdDev<short, uchar>,  normPlanarInvStdDev<short, float>},
            {   normPlanarInvStdDev<int, uchar>,    normPlanarInvStdDev<int, float>},
            { normPlanarInvStdDev<float, uchar>,  normPlanarInvStdDev<float, float>},
        };

        if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
        {
            return funcs_planar_stddev[data_type][out_type_code](inData, baseData, scaleData, outData, channels,
                                                                 global_scale, shift, epsilon, stream);
        }
        else
        {
            return funcs_planar[data_type][out_type_code](inData, baseData, scaleData, outData, channels, global_scale,
                                                          shift, stream);
        }
    }

    if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        funcs_normalize_stddev[data_type][out_type_code][channels - 1](inData, *baseAccess, *scaleAccess, outData,
                                                                       global_scale, shift, epsilon, stream);
    }
    else
    {
        funcs_normalize[data_type][out_type_code][channels - 1](inData, *baseAccess, *scaleAccess, outData,
                                                                global_scale, shift, stream);
    }

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
