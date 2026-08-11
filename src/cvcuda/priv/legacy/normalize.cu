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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"
#include "normalize_planar.cuh" // for PlanarBroadcastIndex, ApplyPlanarNormalize

#include <cvcuda/OpNormalize.h>             // for CVCUDA_NORMALIZE_SCALE_IS_STDDEV, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp> // for TypeTraits

#include <cstdint>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;
namespace cuda = nvcv::cuda;

// Planar (NCHW / CHW) kernel: src/dst are 4D scalar wraps indexed (n, c, y, x); base/scale are 4D
// scalar wraps with per-axis broadcasting determined by their logical (N, C, H, W) shape. is_stddev
// selects the inverse-stddev formula; the per-element math is shared with the var-shape path.
// base/scale are template wrapper types (not a fixed Tensor4DWrap<float>) so the same kernel serves
// both the tensor path (Tensor4DWrap<float>) and the by-value path (a register-backed per-channel
// wrap); both only need ptr(n, c, y, x) -> const float*. The tensor path deduces the wrap types from
// its Tensor4DWrap<float> arguments, so it stays bit-identical.
template<typename T, typename BaseWrap, typename ScaleWrap>
__global__ void normalizePlanarKernel(const nvcv::cuda::Tensor4DWrap<T, int32_t> src, const BaseWrap base,
                                      const ScaleWrap scale, nvcv::cuda::Tensor4DWrap<T, int32_t> dst, int4 inout_size,
                                      int4 base_size, int4 scale_size, float global_scale, float global_shift,
                                      float epsilon, bool is_stddev)
{
    const int src_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;

    if (src_x >= inout_size.w || src_y >= inout_size.z)
        return;

    const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, src_x);
    const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, src_x);

    *dst.ptr(batch, channel, src_y, src_x)
        = ApplyPlanarNormalize<T>(*src.ptr(batch, channel, src_y, src_x), *base.ptr(b.x, b.y, b.z, b.w),
                                  *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev, epsilon);
}

// ILP depth for the 1-byte planar path: each thread owns NGROUP independent uchar4 groups (4*NGROUP
// columns) of plane (batch, channel) and issues all NGROUP loads before any compute, raising the
// number of outstanding memory requests per resident thread (memory-level parallelism) at fixed
// occupancy. A plane is single-channel from each thread's view, so the latency-bound 1-byte planar
// path benefits like the single-channel U8 interleaved path (rather than regressing like the
// multi-channel interleaved hoist). NGROUP=2 is a smaller depth than the single-channel interleaved
// knob: the broadcast fast path here carries more live per-group state, so depth 2 overlaps two
// outstanding loads per thread while keeping registers (hence occupancy) from ballooning.
static constexpr int kNormalizePlanarILPNGroup = 2;

// ILP depth for the F32 planar path: each thread owns NGROUP independent float4 groups (4*NGROUP
// columns) of plane (batch, channel) and issues all NGROUP float4 loads before any compute, raising
// memory-level parallelism at fixed occupancy on the latency-bound single-channel F32 planar path.
// A float4 group is 16 bytes of live register state (vs a uchar4 group's 4 bytes), so depth 2 is used
// (rather than 4) to overlap two outstanding loads per thread while keeping registers from ballooning
// and cutting occupancy.
static constexpr int kNormalizePlanarF32ILPNGroup = 2;

// Vectorized 1-byte planar kernel: each thread owns NGROUP independent char4/uchar4 groups (4*NGROUP
// columns) of plane (batch, channel), strided by blockDim.x so each warp iteration stays coalesced.
// All NGROUP loads are issued into a local array before any compute, so several memory requests
// overlap per resident thread (loads-first ILP). The per-element math reuses ApplyPlanarNormalize
// (with the broadcast fast path hoisting base/mul once when base/scale are broadcast across the four
// columns, the common per-channel case), so output is bit-identical to normalizePlanarKernel.
// Requires 4-byte-aligned plane/row/sample strides (caller-guarded); columns past the last full
// uchar4 group fall back to scalar.
template<int NGROUP, typename T, typename BaseWrap, typename ScaleWrap>
__global__ void normalizePlanarVec4Kernel(const nvcv::cuda::Tensor4DWrap<T, int32_t> src, const BaseWrap base,
                                          const ScaleWrap scale, nvcv::cuda::Tensor4DWrap<T, int32_t> dst,
                                          int4 inout_size, int4 base_size, int4 scale_size, float global_scale,
                                          float global_shift, float epsilon, bool is_stddev)
{
    const int g0      = blockIdx.x * blockDim.x * NGROUP + threadIdx.x;
    const int src_y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc      = blockIdx.z;
    const int batch   = nc / inout_size.y;
    const int channel = nc % inout_size.y;

    const int cx0 = g0 * 4;
    if (cx0 >= inout_size.w || src_y >= inout_size.z)
        return;

    const int width = inout_size.w;

    using Vec4 = typename PlanarVec4Type<T>::type;

    // Loads first: issue all NGROUP vector loads (for full in-range groups) into a local array before
    // any compute, then compute, then store. The vector width is char4/uchar4 (32-bit) for 1-byte planes and
    // float4 (128-bit) for F32 planes; both move 4 columns/group.
    int  cx[NGROUP];
    bool full[NGROUP];
    Vec4 in4[NGROUP];
#pragma unroll
    for (int i = 0; i < NGROUP; ++i)
    {
        cx[i]   = (g0 + i * blockDim.x) * 4;
        full[i] = cx[i] + 4 <= width;
        if (full[i])
            in4[i] = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx[i]));
    }

    if (base_size.w == 1 && scale_size.w == 1)
    {
        // base/scale are broadcast across the columns (the common per-channel case): resolve base and
        // the inverse-std-dev multiplier once and reuse for every column. Bit-identical to the
        // per-element path (same inputs).
        const int4  b      = PlanarBroadcastIndex(base_size, batch, channel, src_y, 0);
        const int4  s      = PlanarBroadcastIndex(scale_size, batch, channel, src_y, 0);
        const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
        const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
        const float mul    = is_stddev ? (1.0f / nvcv::cuda::sqrt(scaleV * scaleV + epsilon)) : scaleV;
        auto        apply  = [&](nvcv::cuda::BaseType<Vec4> raw) -> T
        {
            return nvcv::cuda::SaturateCast<T>((static_cast<float>(raw) - baseV) * mul * global_scale + global_shift);
        };

#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            if (full[i])
            {
                Vec4 out4;
                out4.x                                                           = apply(in4[i].x);
                out4.y                                                           = apply(in4[i].y);
                out4.z                                                           = apply(in4[i].z);
                out4.w                                                           = apply(in4[i].w);
                *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
            }
            else if (cx[i] < width)
            {
                for (int x = cx[i]; x < width; ++x)
                {
                    *dst.ptr(batch, channel, src_y, x) = apply(*src.ptr(batch, channel, src_y, x));
                }
            }
        }
    }
    else
    {
        // Per-column base/scale: index each column independently (matches normalizePlanarKernel).
        auto applyAt = [&](nvcv::cuda::BaseType<Vec4> raw, int x) -> T
        {
            const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, x);
            const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, x);
            return ApplyPlanarNormalize<T>(static_cast<T>(raw), *base.ptr(b.x, b.y, b.z, b.w),
                                           *scale.ptr(s.x, s.y, s.z, s.w), global_scale, global_shift, is_stddev,
                                           epsilon);
        };

#pragma unroll
        for (int i = 0; i < NGROUP; ++i)
        {
            if (full[i])
            {
                const nvcv::cuda::BaseType<Vec4> raw[4] = {in4[i].x, in4[i].y, in4[i].z, in4[i].w};
                Vec4                             out4;
                out4.x                                                           = applyAt(raw[0], cx[i] + 0);
                out4.y                                                           = applyAt(raw[1], cx[i] + 1);
                out4.z                                                           = applyAt(raw[2], cx[i] + 2);
                out4.w                                                           = applyAt(raw[3], cx[i] + 3);
                *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx[i])) = out4;
            }
            else if (cx[i] < width)
            {
                for (int x = cx[i]; x < width; ++x)
                {
                    *dst.ptr(batch, channel, src_y, x) = applyAt(*src.ptr(batch, channel, src_y, x), x);
                }
            }
        }
    }
}

// Launch path for the planar variants; isStdDev/epsilon are forwarded to the unified kernel
// (epsilon is unused when isStdDev is false).
template<typename T>
ErrorCode normalizePlanarImpl(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &baseData,
                              const nvcv::TensorDataStridedCuda &scaleData, const nvcv::TensorDataStridedCuda &outData,
                              float global_scale, float shift, bool isStdDev, float epsilon, cudaStream_t stream)
{
    auto inAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess   = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(inAccess && outAccess && baseAccess && scaleAccess);

    auto inMaxStride    = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride   = outAccess->sampleStride() * outAccess->numSamples();
    auto baseMaxStride  = baseAccess->sampleStride() * baseAccess->numSamples();
    auto scaleMaxStride = scaleAccess->sampleStride() * scaleAccess->numSamples();
    if (std::max(std::max(inMaxStride, outMaxStride), std::max(baseMaxStride, scaleMaxStride))
        > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input, output, base, or scale size exceeds " << cuda::TypeTraits<int32_t>::max
                                                                << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto srcWrap   = nvcv::cuda::CreateTensorWrapNCHW<T, int32_t>(inData);
    auto dstWrap   = nvcv::cuda::CreateTensorWrapNCHW<T, int32_t>(outData);
    auto baseWrap  = nvcv::cuda::CreateTensorWrapNCHW<float, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNCHW<float, int32_t>(scaleData);

    int4 inout_size = {static_cast<int>(inAccess->numSamples()), static_cast<int>(inAccess->numChannels()),
                       static_cast<int>(inAccess->numRows()), static_cast<int>(inAccess->numCols())};
    int4 base_size  = {static_cast<int>(baseAccess->numSamples()), static_cast<int>(baseAccess->numChannels()),
                       static_cast<int>(baseAccess->numRows()), static_cast<int>(baseAccess->numCols())};
    int4 scale_size = {static_cast<int>(scaleAccess->numSamples()), static_cast<int>(scaleAccess->numChannels()),
                       static_cast<int>(scaleAccess->numRows()), static_cast<int>(scaleAccess->numCols())};

    const uint64_t planes = static_cast<uint64_t>(inout_size.x) * inout_size.y;
    if (planes > 65535u)
    {
        LOG_ERROR("Planar normalize launch exceeds CUDA grid.z limit: N*C=" << planes);
        return ErrorCode::INVALID_PARAMETER;
    }

    // Vectorized fast path for the latency-bound 1-byte planar case (the scalar kernel moves 1
    // byte/thread). Each thread covers NGROUP uchar4 groups (4*NGROUP columns) with loads-first ILP;
    // bit-identical output. Requires 4-byte-aligned plane/row/sample strides, else the scalar kernel
    // runs.
    if constexpr (sizeof(T) == 1)
    {
        const bool aligned = inAccess->rowStride() % 4 == 0 && inAccess->chStride() % 4 == 0
                          && inAccess->sampleStride() % 4 == 0 && outAccess->rowStride() % 4 == 0
                          && outAccess->chStride() % 4 == 0 && outAccess->sampleStride() % 4 == 0;
        if (aligned)
        {
            constexpr int NGROUP = kNormalizePlanarILPNGroup;
            dim3          vblock(32, 8);
            dim3          vgrid(divUp(divUp(inout_size.w, 4), static_cast<int>(vblock.x) * NGROUP),
                                divUp(inout_size.z, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
            normalizePlanarVec4Kernel<NGROUP, T><<<vgrid, vblock, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap,
                                                                               inout_size, base_size, scale_size,
                                                                               global_scale, shift, epsilon, isStdDev);
            checkKernelErrors();
            return ErrorCode::SUCCESS;
        }
    }

    // Vectorized fast path for the latency-bound F32 planar case (the scalar kernel moves 4
    // bytes/thread). Each thread covers NGROUP float4 groups (4*NGROUP columns, 128-bit transfers)
    // with loads-first ILP; bit-identical output (no SaturateCast needed for float out). Requires
    // 16-byte-aligned plane/row/sample strides, else the scalar kernel runs.
    if constexpr (std::is_same_v<T, float>)
    {
        const bool aligned = inAccess->rowStride() % 16 == 0 && inAccess->chStride() % 16 == 0
                          && inAccess->sampleStride() % 16 == 0 && outAccess->rowStride() % 16 == 0
                          && outAccess->chStride() % 16 == 0 && outAccess->sampleStride() % 16 == 0;
        if (aligned)
        {
            constexpr int NGROUP = kNormalizePlanarF32ILPNGroup;
            dim3          vblock(32, 8);
            dim3          vgrid(divUp(divUp(inout_size.w, 4), static_cast<int>(vblock.x) * NGROUP),
                                divUp(inout_size.z, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
            normalizePlanarVec4Kernel<NGROUP, T><<<vgrid, vblock, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap,
                                                                               inout_size, base_size, scale_size,
                                                                               global_scale, shift, epsilon, isStdDev);
            checkKernelErrors();
            return ErrorCode::SUCCESS;
        }
    }

    dim3 block(32, 8);
    dim3 grid(divUp(inout_size.w, block.x), divUp(inout_size.z, block.y), static_cast<unsigned int>(planes));

    normalizePlanarKernel<T><<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                         scale_size, global_scale, shift, epsilon, isStdDev);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

template<typename T>
ErrorCode normalizePlanar(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &baseData,
                          const nvcv::TensorDataStridedCuda &scaleData, const nvcv::TensorDataStridedCuda &outData,
                          float global_scale, float shift, cudaStream_t stream)
{
    return normalizePlanarImpl<T>(inData, baseData, scaleData, outData, global_scale, shift, false, 0.f, stream);
}

template<typename T>
ErrorCode normalizePlanarInvStdDev(const nvcv::TensorDataStridedCuda &inData,
                                   const nvcv::TensorDataStridedCuda &baseData,
                                   const nvcv::TensorDataStridedCuda &scaleData,
                                   const nvcv::TensorDataStridedCuda &outData, float global_scale, float shift,
                                   float epsilon, cudaStream_t stream)
{
    return normalizePlanarImpl<T>(inData, baseData, scaleData, outData, global_scale, shift, true, epsilon, stream);
}

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeKernel(const input_type src, const base_type base, const scale_type scale, input_type dst,
                                int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                float global_shift)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_x         = base_size.x == 1 ? 0 : src_x;
    const int base_y         = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx = base_size.z == 1 ? 0 : batch_idx;

    const int scale_x         = scale_size.x == 1 ? 0 : src_x;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;

    *dst.ptr(batch_idx, src_y, src_x) = nvcv::cuda::SaturateCast<input_value_type>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x))
            * (*scale.ptr(scale_batch_idx, scale_y, scale_x)) * global_scale
        + global_shift);
}

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeInvStdDevKernel(const input_type src, const base_type base, const scale_type scale,
                                         input_type dst, int2 inout_size, int3 base_size, int3 scale_size,
                                         float global_scale, float global_shift, float epsilon)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_x         = base_size.x == 1 ? 0 : src_x;
    const int base_y         = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx = base_size.z == 1 ? 0 : batch_idx;

    const int scale_x         = scale_size.x == 1 ? 0 : src_x;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;
    using scale_value_type = typename scale_type::ValueType;

    scale_value_type s   = *scale.ptr(scale_batch_idx, scale_y, scale_x);
    scale_value_type x   = s * s + epsilon;
    scale_value_type mul = 1.0f / nvcv::cuda::sqrt(x);

    *dst.ptr(batch_idx, src_y, src_x) = nvcv::cuda::SaturateCast<input_value_type>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale
        + global_shift);
}

// ILP depth used by the single-channel 8-bit inverse-std-dev wide-load path: each thread issues this
// many independent uint4 loads before computing, raising the number of outstanding memory requests
// per resident thread (memory-level parallelism) at fixed occupancy. NGROUP=4 measured the best
// long-scoreboard-stall reduction on the single-channel uint4 path without register pressure growing
// enough to cut occupancy.
static constexpr int kNormalizeILPNGroup = 4;

// Inverse-std-dev normalize that processes NIX pixels per thread, used when base and scale are
// broadcast across the spatial extent (per-channel or scalar params, the common case). The scalar
// normalizeInvStdDevKernel recomputes the inverse-std-dev multiplier mul = 1 / sqrt(scale^2 + eps)
// for every pixel; for multi-channel interleaved input (uchar3 / uchar4) that is several sqrt +
// reciprocal per pixel, all redundant because scale is identical for every pixel of a sample. That
// makes those paths issue-bound (~83% issue-slot utilization on Ampere) rather than memory-bound.
// Here each thread reads base/scale and computes mul once, then applies it to NIX pixels, cutting the
// redundant SFU/issue work NIX-fold. Pixels are strided by blockDim.x so each warp iteration stays
// coalesced. Output is bit-identical to normalizeInvStdDevKernel (same per-element arithmetic; only
// the multiplier hoists out of the per-pixel loop).
template<int NIX, typename input_type, typename base_type, typename scale_type>
__global__ void normalizeInvStdDevHoistKernel(const input_type src, const base_type base, const scale_type scale,
                                              input_type dst, int2 inout_size, int3 base_size, int3 scale_size,
                                              float global_scale, float global_shift, float epsilon)
{
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_y >= inout_size.y)
        return;

    const int base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    using input_value_type = typename input_type::ValueType;
    using base_value_type  = typename base_type::ValueType;
    using scale_value_type = typename scale_type::ValueType;

    const base_value_type  b   = *base.ptr(base_batch_idx, 0, 0);
    const scale_value_type s   = *scale.ptr(scale_batch_idx, 0, 0);
    const scale_value_type x   = s * s + epsilon;
    const scale_value_type mul = 1.0f / nvcv::cuda::sqrt(x);

    const int x0 = blockIdx.x * blockDim.x * NIX + threadIdx.x;
#pragma unroll
    for (int i = 0; i < NIX; ++i)
    {
        const int src_x = x0 + i * blockDim.x;
        if (src_x < inout_size.x)
        {
            *dst.ptr(batch_idx, src_y, src_x) = nvcv::cuda::SaturateCast<input_value_type>(
                (*src.ptr(batch_idx, src_y, src_x) - b) * mul * global_scale + global_shift);
        }
    }
}

// Vectorized single-channel 8-bit inverse-std-dev normalize. The scalar kernel above processes one
// byte per thread, which leaves the 1-byte path latency-bound (~44% memory SOL, 0.72 issued
// warps/scheduler): too few bytes in flight per thread to hide load latency. Here each thread owns
// four consecutive columns and moves them with a single 32-bit (uchar4) load/store, so a warp
// transfers a full 128-byte line and address/compute overhead is amortized 4x. The per-element math
// is identical to normalizeInvStdDevKernel (same expression, same SaturateCast), so output is
// bit-for-bit unchanged. Requires 4-byte-aligned row/sample strides (caller-guarded); a width that
// is not a multiple of 4 falls back to scalar for the tail columns. base/scale are read per element,
// so all broadcast shapes stay correct.
template<typename SrcWrapper, typename BaseWrapper, typename ScaleWrapper, typename DstWrapper>
__global__ void normalizeInvStdDevU8Vec4Kernel(SrcWrapper src, BaseWrapper base, ScaleWrapper scale, DstWrapper dst,
                                               int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                               float global_shift, float epsilon)
{
    const int cx        = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (cx >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_y          = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    auto normOne = [&](unsigned char raw, int x) -> unsigned char
    {
        const int   base_x  = base_size.x == 1 ? 0 : x;
        const int   scale_x = scale_size.x == 1 ? 0 : x;
        const float s       = *scale.ptr(scale_batch_idx, scale_y, scale_x);
        const float mul     = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
        return nvcv::cuda::SaturateCast<unsigned char>(
            (static_cast<float>(raw) - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale + global_shift);
    };

    if (cx + 4 <= inout_size.x)
    {
        const uchar4 in4 = *reinterpret_cast<const uchar4 *>(src.ptr(batch_idx, src_y, cx));
        uchar4       out4;
        if (base_size.x == 1 && scale_size.x == 1)
        {
            // base/scale are broadcast across these four columns (the common per-channel case), so
            // the expensive inverse-std-dev multiplier and the base are identical for all four:
            // compute them once and reuse. Bit-identical to the per-element path (same inputs).
            const float baseV = *base.ptr(base_batch_idx, base_y, 0);
            const float s     = *scale.ptr(scale_batch_idx, scale_y, 0);
            const float mul   = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
            auto        apply = [&](unsigned char raw) -> unsigned char
            {
                return nvcv::cuda::SaturateCast<unsigned char>((static_cast<float>(raw) - baseV) * mul * global_scale
                                                               + global_shift);
            };
            out4.x = apply(in4.x);
            out4.y = apply(in4.y);
            out4.z = apply(in4.z);
            out4.w = apply(in4.w);
        }
        else
        {
            out4.x = normOne(in4.x, cx + 0);
            out4.y = normOne(in4.y, cx + 1);
            out4.z = normOne(in4.z, cx + 2);
            out4.w = normOne(in4.w, cx + 3);
        }
        *reinterpret_cast<uchar4 *>(dst.ptr(batch_idx, src_y, cx)) = out4;
    }
    else
    {
        for (int x = cx; x < inout_size.x; ++x)
        {
            *dst.ptr(batch_idx, src_y, x) = normOne(*src.ptr(batch_idx, src_y, x), x);
        }
    }
}

// Vectorized single-channel 8-bit inverse-std-dev kernel: each thread owns NGROUP independent uint4 groups
// (16*NGROUP consecutive columns). All NGROUP loads are issued into a local array before any compute,
// so NGROUP load requests are outstanding per thread at once. A single wide uint4 request raises
// bytes-per-request but not the number of in-flight requests; issuing several independent loads first
// raises memory-level parallelism (more outstanding requests per resident thread) at fixed occupancy,
// which is the real lever on the latency-bound (not bandwidth-bound) single-channel 8-bit path. The
// per-element math, broadcast fast-path, and SaturateCast are identical to the uint4 kernel, so output
// is bit-for-bit unchanged. Requires 16-byte-aligned row/sample strides (caller-guarded); columns past
// the last full 16*NGROUP block fall back to scalar.
template<int NGROUP, typename SrcWrapper, typename BaseWrapper, typename ScaleWrapper, typename DstWrapper>
__global__ void normalizeInvStdDevU8VecILPKernel(SrcWrapper src, BaseWrapper base, ScaleWrapper scale, DstWrapper dst,
                                                 int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                                 float global_shift, float epsilon)
{
    const int base_cx   = (blockIdx.x * blockDim.x + threadIdx.x) * (16 * NGROUP);
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (base_cx >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_y          = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    auto normOne = [&](unsigned char raw, int x) -> unsigned char
    {
        const int   base_x  = base_size.x == 1 ? 0 : x;
        const int   scale_x = scale_size.x == 1 ? 0 : x;
        const float s       = *scale.ptr(scale_batch_idx, scale_y, scale_x);
        const float mul     = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
        return nvcv::cuda::SaturateCast<unsigned char>(
            (static_cast<float>(raw) - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale + global_shift);
    };

    if (base_cx + 16 * NGROUP <= inout_size.x)
    {
        // Issue all NGROUP loads first so the requests overlap, then compute, then store.
        uint4 v[NGROUP];
#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            v[g] = *reinterpret_cast<const uint4 *>(src.ptr(batch_idx, src_y, base_cx + g * 16));
        }

        unsigned char o[NGROUP][16];
        if (base_size.x == 1 && scale_size.x == 1)
        {
            const float baseV = *base.ptr(base_batch_idx, base_y, 0);
            const float s     = *scale.ptr(scale_batch_idx, scale_y, 0);
            const float mul   = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
            auto        apply = [&](unsigned char raw) -> unsigned char
            {
                return nvcv::cuda::SaturateCast<unsigned char>((static_cast<float>(raw) - baseV) * mul * global_scale
                                                               + global_shift);
            };
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const unsigned char *b = reinterpret_cast<const unsigned char *>(&v[g]);
#pragma unroll
                for (int i = 0; i < 16; ++i)
                {
                    o[g][i] = apply(b[i]);
                }
            }
        }
        else
        {
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const unsigned char *b = reinterpret_cast<const unsigned char *>(&v[g]);
#pragma unroll
                for (int i = 0; i < 16; ++i)
                {
                    o[g][i] = normOne(b[i], base_cx + g * 16 + i);
                }
            }
        }

#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            *reinterpret_cast<uint4 *>(dst.ptr(batch_idx, src_y, base_cx + g * 16))
                = *reinterpret_cast<const uint4 *>(o[g]);
        }
    }
    else
    {
        for (int x = base_cx; x < inout_size.x; ++x)
        {
            *dst.ptr(batch_idx, src_y, x) = normOne(*src.ptr(batch_idx, src_y, x), x);
        }
    }
}

// ILP depth used by the single-channel F32 inverse-std-dev wide-load path. The scalar F32 kernel
// moves one float (4 bytes) per thread, leaving the single-channel F32 path latency-bound (too few
// bytes in flight per thread to hide load latency). Here each thread owns NGROUP independent float4
// groups (4*NGROUP consecutive columns) and issues all NGROUP loads before any compute, raising the
// number of outstanding memory requests per resident thread (memory-level parallelism) at fixed
// occupancy. NGROUP=4 matches the proven U8 value; the float4 path carries 4 floats (16 bytes) of
// live state per group rather than 16 bytes of packed bytes, so registers are watched.
static constexpr int kNormalizeF32ILPNGroup = 4;

// Vectorized single-channel F32 inverse-std-dev normalize: each thread owns NGROUP independent float4
// groups (4*NGROUP consecutive columns). All NGROUP float4 loads are issued into a local array before
// any compute, so NGROUP load requests are outstanding per thread at once -- raising memory-level
// parallelism (more outstanding requests per resident thread) at fixed occupancy, the lever on the
// latency-bound single-channel F32 path. When base/scale are broadcast across the four columns (the
// common per-channel case) the base and the inverse-std-dev multiplier are computed once per thread
// and reused. The per-element math is identical to normalizeInvStdDevKernel (same expression; F32
// output needs no SaturateCast since the value is already float), so output is bit-for-bit unchanged.
// Requires 16-byte-aligned row/sample strides (caller-guarded); columns past the last full 4*NGROUP
// block fall back to scalar.
template<int NGROUP, typename SrcWrapper, typename BaseWrapper, typename ScaleWrapper, typename DstWrapper>
__global__ void normalizeInvStdDevF32VecILPKernel(SrcWrapper src, BaseWrapper base, ScaleWrapper scale, DstWrapper dst,
                                                  int2 inout_size, int3 base_size, int3 scale_size, float global_scale,
                                                  float global_shift, float epsilon)
{
    const int base_cx   = (blockIdx.x * blockDim.x + threadIdx.x) * (4 * NGROUP);
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (base_cx >= inout_size.x || src_y >= inout_size.y)
        return;

    const int base_y          = base_size.y == 1 ? 0 : src_y;
    const int base_batch_idx  = base_size.z == 1 ? 0 : batch_idx;
    const int scale_y         = scale_size.y == 1 ? 0 : src_y;
    const int scale_batch_idx = scale_size.z == 1 ? 0 : batch_idx;

    auto normOne = [&](float raw, int x) -> float
    {
        const int   base_x  = base_size.x == 1 ? 0 : x;
        const int   scale_x = scale_size.x == 1 ? 0 : x;
        const float s       = *scale.ptr(scale_batch_idx, scale_y, scale_x);
        const float mul     = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
        return (raw - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale + global_shift;
    };

    if (base_cx + 4 * NGROUP <= inout_size.x)
    {
        // Issue all NGROUP float4 loads first so the requests overlap, then compute, then store.
        float4 v[NGROUP];
#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            v[g] = *reinterpret_cast<const float4 *>(src.ptr(batch_idx, src_y, base_cx + g * 4));
        }

        float4 o[NGROUP];
        if (base_size.x == 1 && scale_size.x == 1)
        {
            const float baseV = *base.ptr(base_batch_idx, base_y, 0);
            const float s     = *scale.ptr(scale_batch_idx, scale_y, 0);
            const float mul   = 1.0f / nvcv::cuda::sqrt(s * s + epsilon);
            auto        apply = [&](float raw) -> float
            {
                return (raw - baseV) * mul * global_scale + global_shift;
            };
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                o[g].x = apply(v[g].x);
                o[g].y = apply(v[g].y);
                o[g].z = apply(v[g].z);
                o[g].w = apply(v[g].w);
            }
        }
        else
        {
#pragma unroll
            for (int g = 0; g < NGROUP; ++g)
            {
                const int cx = base_cx + g * 4;
                o[g].x       = normOne(v[g].x, cx + 0);
                o[g].y       = normOne(v[g].y, cx + 1);
                o[g].z       = normOne(v[g].z, cx + 2);
                o[g].w       = normOne(v[g].w, cx + 3);
            }
        }

#pragma unroll
        for (int g = 0; g < NGROUP; ++g)
        {
            *reinterpret_cast<float4 *>(dst.ptr(batch_idx, src_y, base_cx + g * 4)) = o[g];
        }
    }
    else
    {
        for (int x = base_cx; x < inout_size.x; ++x)
        {
            *dst.ptr(batch_idx, src_y, x) = normOne(*src.ptr(batch_idx, src_y, x), x);
        }
    }
}

template<typename base_type, typename scale_type, typename WrapInput, typename WrapOutput>
void normalizeWrap(WrapInput srcWrap, WrapOutput dstWrap, DataShape input_shape,
                   const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                   float global_scale, float shift, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    auto baseWrap  = nvcv::cuda::CreateTensorWrapNHW<base_type, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNHW<scale_type, int32_t>(scaleData);

    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    NVCV_ASSERT(baseAccess);

    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(scaleAccess);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {static_cast<int>(baseAccess->numCols()), static_cast<int>(baseAccess->numRows()),
                       static_cast<int>(baseAccess->numSamples())};
    int3 scale_size = {static_cast<int>(scaleAccess->numCols()), static_cast<int>(scaleAccess->numRows()),
                       static_cast<int>(scaleAccess->numSamples())};

    normalizeKernel<<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                scale_size, global_scale, shift);
    checkKernelErrors();
}

template<typename base_type, typename scale_type, typename WrapInput, typename WrapOutput>
void normalizeInvStdDevWrap(WrapInput srcWrap, WrapOutput dstWrap, DataShape input_shape,
                            const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                            float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    auto baseWrap  = nvcv::cuda::CreateTensorWrapNHW<base_type, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNHW<scale_type, int32_t>(scaleData);

    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    NVCV_ASSERT(baseAccess);

    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(scaleAccess);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {static_cast<int>(baseAccess->numCols()), static_cast<int>(baseAccess->numRows()),
                       static_cast<int>(baseAccess->numSamples())};
    int3 scale_size = {static_cast<int>(scaleAccess->numCols()), static_cast<int>(scaleAccess->numRows()),
                       static_cast<int>(scaleAccess->numSamples())};

    // Multi-byte-element (float / float3 / ...) inputs are already memory-bound here, so recomputing
    // the multiplier per pixel costs nothing; only the 1-byte interleaved paths (uchar3 / uchar4) are
    // issue-bound on the redundant per-pixel sqrt. When base and scale are spatially broadcast (the
    // common per-channel case) hoist the multiplier and process NIX pixels per thread.
    using pixel_type = typename WrapInput::ValueType;
    if constexpr (sizeof(typename nvcv::cuda::BaseType<pixel_type>) == 1)
    {
        const bool spatialBroadcast = base_size.x == 1 && base_size.y == 1 && scale_size.x == 1 && scale_size.y == 1;
        if (spatialBroadcast)
        {
            constexpr int NIX = 4;
            dim3          hgrid(divUp(input_shape.W, block.x * NIX), divUp(input_shape.H, block.y), input_shape.N);
            normalizeInvStdDevHoistKernel<NIX><<<hgrid, block, 0, stream>>>(
                srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size, global_scale, shift, epsilon);
            checkKernelErrors();
            return;
        }
    }

    normalizeInvStdDevKernel<<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                         scale_size, global_scale, shift, epsilon);
    checkKernelErrors();
}

template<typename input_wrapper, typename output_wrapper>
void callNormalizeWrap(const input_wrapper &input, const DataShape &inputShape,
                       const nvcv::TensorDataStridedCuda &baseData, const nvcv::TensorDataStridedCuda &scaleData,
                       const output_wrapper &output, float global_scale, float shift, cudaStream_t stream)
{
    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    NVCV_ASSERT(baseAccess);

    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(scaleAccess);

    using input_type = typename input_wrapper::ValueType;
    using work_type  = nvcv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseAccess->numChannels() != 1 && scaleAccess->numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normalizeWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else if (baseAccess->numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normalizeWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else if (scaleAccess->numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normalizeWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normalizeWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
}

template<typename input_type>
ErrorCode normalize(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &baseData,
                    const nvcv::TensorDataStridedCuda &scaleData, const nvcv::TensorDataStridedCuda &outData,
                    float global_scale, float shift, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataShape inputShape = GetLegacyDataShape(inAccess->infoShape());

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto srcWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(inData);
        auto dstWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(outData);
        callNormalizeWrap(srcWrap, inputShape, baseData, scaleData, dstWrap, global_scale, shift, stream);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<typename input_wrapper, typename output_wrapper>
void callNormalizeInvStdDevWrap(const input_wrapper &input, const DataShape &inputShape,
                                const nvcv::TensorDataStridedCuda &baseData,
                                const nvcv::TensorDataStridedCuda &scaleData, const output_wrapper &output,
                                float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    NVCV_ASSERT(baseAccess);

    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(scaleAccess);

    using input_type = typename input_wrapper::ValueType;
    using work_type  = nvcv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseAccess->numChannels() != 1 && scaleAccess->numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normalizeInvStdDevWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else if (baseAccess->numChannels() != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normalizeInvStdDevWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else if (scaleAccess->numChannels() != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normalizeInvStdDevWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normalizeInvStdDevWrap<base_type, scale_type>(input, output, inputShape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
}

template<typename input_type>
ErrorCode normalizeInvStdDev(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &baseData,
                             const nvcv::TensorDataStridedCuda &scaleData, const nvcv::TensorDataStridedCuda &outData,
                             float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataShape inputShape = GetLegacyDataShape(inAccess->infoShape());

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= cuda::TypeTraits<int32_t>::max)
    {
        auto srcWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(inData);
        auto dstWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(outData);
        callNormalizeInvStdDevWrap(srcWrap, inputShape, baseData, scaleData, dstWrap, global_scale, shift, epsilon,
                                   stream);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

// Launcher for the vectorized single-channel 8-bit inverse-std-dev path. Mirrors normalizeInvStdDev
// (single channel => base/scale resolve to scalar float wraps) but launches the uchar4 kernel with
// one thread per four columns. Defers to the scalar specialization for tensors too large for 32-bit
// indexing, so behavior is preserved on that path.
inline ErrorCode normalizeInvStdDevU8Vec(const nvcv::TensorDataStridedCuda &inData,
                                         const nvcv::TensorDataStridedCuda &baseData,
                                         const nvcv::TensorDataStridedCuda &scaleData,
                                         const nvcv::TensorDataStridedCuda &outData, float global_scale, float shift,
                                         float epsilon, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(inAccess && outAccess);

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        return normalizeInvStdDev<uchar>(inData, baseData, scaleData, outData, global_scale, shift, epsilon, stream);
    }

    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    auto srcWrap   = nvcv::cuda::CreateTensorWrapNHW<uchar, int32_t>(inData);
    auto dstWrap   = nvcv::cuda::CreateTensorWrapNHW<uchar, int32_t>(outData);
    auto baseWrap  = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(scaleData);

    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(baseAccess && scaleAccess);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {static_cast<int>(baseAccess->numCols()), static_cast<int>(baseAccess->numRows()),
                       static_cast<int>(baseAccess->numSamples())};
    int3 scale_size = {static_cast<int>(scaleAccess->numCols()), static_cast<int>(scaleAccess->numRows()),
                       static_cast<int>(scaleAccess->numSamples())};

    dim3 block(32, 8);
    dim3 grid(divUp(divUp(input_shape.W, 4), static_cast<int>(block.x)),
              divUp(input_shape.H, static_cast<int>(block.y)), input_shape.N);

    normalizeInvStdDevU8Vec4Kernel<<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size,
                                                               base_size, scale_size, global_scale, shift, epsilon);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

// Launcher for the ILP (NGROUP independent uint4 groups/thread) single-channel 8-bit inverse-std-dev
// path. Mirrors normalizeInvStdDevU8Vec but each thread covers 16*NGROUP columns and the grid shrinks
// accordingly. Defers to the scalar specialization for tensors too large for 32-bit indexing.
template<int NGROUP>
inline ErrorCode normalizeInvStdDevU8VecILP(const nvcv::TensorDataStridedCuda &inData,
                                            const nvcv::TensorDataStridedCuda &baseData,
                                            const nvcv::TensorDataStridedCuda &scaleData,
                                            const nvcv::TensorDataStridedCuda &outData, float global_scale, float shift,
                                            float epsilon, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(inAccess && outAccess);

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        return normalizeInvStdDev<uchar>(inData, baseData, scaleData, outData, global_scale, shift, epsilon, stream);
    }

    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    auto srcWrap   = nvcv::cuda::CreateTensorWrapNHW<uchar, int32_t>(inData);
    auto dstWrap   = nvcv::cuda::CreateTensorWrapNHW<uchar, int32_t>(outData);
    auto baseWrap  = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(scaleData);

    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(baseAccess && scaleAccess);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {static_cast<int>(baseAccess->numCols()), static_cast<int>(baseAccess->numRows()),
                       static_cast<int>(baseAccess->numSamples())};
    int3 scale_size = {static_cast<int>(scaleAccess->numCols()), static_cast<int>(scaleAccess->numRows()),
                       static_cast<int>(scaleAccess->numSamples())};

    dim3 block(32, 8);
    dim3 grid(divUp(divUp(input_shape.W, 16 * NGROUP), static_cast<int>(block.x)),
              divUp(input_shape.H, static_cast<int>(block.y)), input_shape.N);

    normalizeInvStdDevU8VecILPKernel<NGROUP><<<grid, block, 0, stream>>>(
        srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size, global_scale, shift, epsilon);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

// Launcher for the ILP (NGROUP independent float4 groups/thread) single-channel F32 inverse-std-dev
// path. Mirrors normalizeInvStdDevU8VecILP but for float input/output: each thread covers 4*NGROUP
// columns and the grid shrinks accordingly. Defers to the scalar specialization for tensors too large
// for 32-bit indexing.
template<int NGROUP>
inline ErrorCode normalizeInvStdDevF32VecILP(const nvcv::TensorDataStridedCuda &inData,
                                             const nvcv::TensorDataStridedCuda &baseData,
                                             const nvcv::TensorDataStridedCuda &scaleData,
                                             const nvcv::TensorDataStridedCuda &outData, float global_scale,
                                             float shift, float epsilon, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(inAccess && outAccess);

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        return normalizeInvStdDev<float>(inData, baseData, scaleData, outData, global_scale, shift, epsilon, stream);
    }

    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    auto srcWrap   = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(inData);
    auto dstWrap   = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(outData);
    auto baseWrap  = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(baseData);
    auto scaleWrap = nvcv::cuda::CreateTensorWrapNHW<float, int32_t>(scaleData);

    auto baseAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(baseData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(scaleData);
    NVCV_ASSERT(baseAccess && scaleAccess);

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {static_cast<int>(baseAccess->numCols()), static_cast<int>(baseAccess->numRows()),
                       static_cast<int>(baseAccess->numSamples())};
    int3 scale_size = {static_cast<int>(scaleAccess->numCols()), static_cast<int>(scaleAccess->numRows()),
                       static_cast<int>(scaleAccess->numSamples())};

    dim3 block(32, 8);
    dim3 grid(divUp(divUp(input_shape.W, 4 * NGROUP), static_cast<int>(block.x)),
              divUp(input_shape.H, static_cast<int>(block.y)), input_shape.N);

    normalizeInvStdDevF32VecILPKernel<NGROUP><<<grid, block, 0, stream>>>(
        srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size, global_scale, shift, epsilon);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Tensor-free (by-value scalar / per-channel parameter) interleaved path.
//
// base/scale are supplied as host constants packed into a float4 and passed by value, so the caller
// avoids allocating and uploading per-channel parameter tensors (no device alloc, no H2D copy).
// ConstantValueWrap adapts a register-resident value to the same wrap interface (.ptr()/ValueType)
// the interleaved kernels already consume, so the existing normalizeKernel / normalizeInvStdDevKernel
// run unchanged with base_size/scale_size of {1,1,1} -- producing output bit-identical to the tensor
// path for the same values. Interleaved (NHWC/HWC) only; the planar (NCHW/CHW) by-value path is below.
// The 1-byte multiplier-hoist fast path is shared with the tensor dispatch. The specialized
// single-channel U8/F32 vectorized ILP kernels are not wired into the by-value dispatch yet.
// ---------------------------------------------------------------------------------------------------
template<typename T>
struct ConstantValueWrap
{
    using ValueType = T;

    T value;

    inline const __host__ __device__ T *ptr(int, int, int) const
    {
        return &value;
    }
};

// Build a work value (float / float3 / float4) from the leading lanes of a float4 on the host. For a
// scalar work type the identity of lane 0 is returned; NumElements is constexpr so the loop is fixed.
template<typename VecT>
inline VecT MakeWorkVec(float4 v)
{
    VecT out{};
    for (int i = 0; i < nvcv::cuda::NumElements<VecT>; ++i)
    {
        nvcv::cuda::GetElement(out, i) = nvcv::cuda::GetElement(v, i);
    }
    return out;
}

template<typename base_type, typename scale_type, typename WrapInput, typename WrapOutput>
void normalizeScalarWrap(WrapInput srcWrap, WrapOutput dstWrap, DataShape input_shape, base_type baseVal,
                         scale_type scaleVal, float global_scale, float shift, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    ConstantValueWrap<base_type>  baseWrap{baseVal};
    ConstantValueWrap<scale_type> scaleWrap{scaleVal};

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {1, 1, 1};
    int3 scale_size = {1, 1, 1};

    normalizeKernel<<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                scale_size, global_scale, shift);
    checkKernelErrors();
}

template<typename base_type, typename scale_type, typename WrapInput, typename WrapOutput>
void normalizeScalarInvStdDevWrap(WrapInput srcWrap, WrapOutput dstWrap, DataShape input_shape, base_type baseVal,
                                  scale_type scaleVal, float global_scale, float shift, float epsilon,
                                  cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    ConstantValueWrap<base_type>  baseWrap{baseVal};
    ConstantValueWrap<scale_type> scaleWrap{scaleVal};

    int2 inout_size = {input_shape.W, input_shape.H};
    int3 base_size  = {1, 1, 1};
    int3 scale_size = {1, 1, 1};

    // Match the tensor path's multi-channel 8-bit fast path. By-value parameters are spatially
    // broadcast by definition, so the inverse-std-dev multiplier can be hoisted and reused across
    // four pixels per thread instead of recomputed for every pixel.
    using pixel_type = typename WrapInput::ValueType;
    if constexpr (sizeof(typename nvcv::cuda::BaseType<pixel_type>) == 1)
    {
        constexpr int NIX = 4;
        dim3          hgrid(divUp(input_shape.W, block.x * NIX), divUp(input_shape.H, block.y), input_shape.N);
        normalizeInvStdDevHoistKernel<NIX><<<hgrid, block, 0, stream>>>(
            srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size, global_scale, shift, epsilon);
        checkKernelErrors();
        return;
    }

    normalizeInvStdDevKernel<<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size,
                                                         scale_size, global_scale, shift, epsilon);
    checkKernelErrors();
}

// Selects per-channel (work_type: float3/float4) vs scalar-broadcast (float) base/scale wraps from the
// provided counts, mirroring callNormalizeWrap's numChannels()-based branch. count == 1 broadcasts one
// value to every channel; count == channels supplies per-channel values.
template<typename input_wrapper, typename output_wrapper>
void callNormalizeScalarWrap(const input_wrapper &input, const DataShape &inputShape, float4 base, float4 scale,
                             int baseCount, int scaleCount, const output_wrapper &output, float global_scale,
                             float shift, cudaStream_t stream)
{
    using input_type = typename input_wrapper::ValueType;
    using work_type  = nvcv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseCount != 1 && scaleCount != 1)
    {
        normalizeScalarWrap<work_type, work_type>(input, output, inputShape, MakeWorkVec<work_type>(base),
                                                  MakeWorkVec<work_type>(scale), global_scale, shift, stream);
    }
    else if (baseCount != 1)
    {
        normalizeScalarWrap<work_type, float>(input, output, inputShape, MakeWorkVec<work_type>(base), scale.x,
                                              global_scale, shift, stream);
    }
    else if (scaleCount != 1)
    {
        normalizeScalarWrap<float, work_type>(input, output, inputShape, base.x, MakeWorkVec<work_type>(scale),
                                              global_scale, shift, stream);
    }
    else
    {
        normalizeScalarWrap<float, float>(input, output, inputShape, base.x, scale.x, global_scale, shift, stream);
    }
}

template<typename input_wrapper, typename output_wrapper>
void callNormalizeScalarInvStdDevWrap(const input_wrapper &input, const DataShape &inputShape, float4 base,
                                      float4 scale, int baseCount, int scaleCount, const output_wrapper &output,
                                      float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    using input_type = typename input_wrapper::ValueType;
    using work_type  = nvcv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseCount != 1 && scaleCount != 1)
    {
        normalizeScalarInvStdDevWrap<work_type, work_type>(input, output, inputShape, MakeWorkVec<work_type>(base),
                                                           MakeWorkVec<work_type>(scale), global_scale, shift, epsilon,
                                                           stream);
    }
    else if (baseCount != 1)
    {
        normalizeScalarInvStdDevWrap<work_type, float>(input, output, inputShape, MakeWorkVec<work_type>(base), scale.x,
                                                       global_scale, shift, epsilon, stream);
    }
    else if (scaleCount != 1)
    {
        normalizeScalarInvStdDevWrap<float, work_type>(input, output, inputShape, base.x, MakeWorkVec<work_type>(scale),
                                                       global_scale, shift, epsilon, stream);
    }
    else
    {
        normalizeScalarInvStdDevWrap<float, float>(input, output, inputShape, base.x, scale.x, global_scale, shift,
                                                   epsilon, stream);
    }
}

template<typename input_type>
ErrorCode normalizeScalar(const nvcv::TensorDataStridedCuda &inData, float4 base, float4 scale, int baseCount,
                          int scaleCount, const nvcv::TensorDataStridedCuda &outData, float global_scale, float shift,
                          cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataShape inputShape = GetLegacyDataShape(inAccess->infoShape());

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto srcWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(inData);
    auto dstWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(outData);
    callNormalizeScalarWrap(srcWrap, inputShape, base, scale, baseCount, scaleCount, dstWrap, global_scale, shift,
                            stream);
    return ErrorCode::SUCCESS;
}

template<typename input_type>
ErrorCode normalizeScalarInvStdDev(const nvcv::TensorDataStridedCuda &inData, float4 base, float4 scale, int baseCount,
                                   int scaleCount, const nvcv::TensorDataStridedCuda &outData, float global_scale,
                                   float shift, float epsilon, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    DataShape inputShape = GetLegacyDataShape(inAccess->infoShape());

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto srcWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(inData);
    auto dstWrap = nvcv::cuda::CreateTensorWrapNHW<input_type, int32_t>(outData);
    callNormalizeScalarInvStdDevWrap(srcWrap, inputShape, base, scale, baseCount, scaleCount, dstWrap, global_scale,
                                     shift, epsilon, stream);
    return ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------------------------------
// Tensor-free (by-value scalar / per-channel parameter) planar path.
//
// The planar kernels read one scalar base/scale per (batch, channel), where the channel is a grid
// dimension -- unlike the interleaved kernels, whose base/scale vector lanes ARE the channels. So the
// by-value planar wrap must be channel-indexed (return &data[channel]) rather than a single broadcast
// value. Launched with param extents {N=1, C=count, H=1, W=1}, PlanarBroadcastIndex maps channel -> 0
// for a broadcast scalar (count == 1) and channel -> channel for per-channel values (count == C), so
// output is bit-identical to the tensor planar path fed a {1, C, 1, 1} float parameter tensor. Reuses
// the base normalizePlanarKernel (like the interleaved by-value path reuses normalizeKernel) and the
// vectorized planar fast path used by the tensor path.
// ---------------------------------------------------------------------------------------------------
struct PlanarConstParamWrap
{
    using ValueType = float;

    float data[4];

    inline const __host__ __device__ float *ptr(int, int channel, int, int) const
    {
        return &data[channel];
    }
};

template<typename input_type>
ErrorCode normalizeScalarPlanarImpl(const nvcv::TensorDataStridedCuda &inData, float4 base, float4 scale, int baseCount,
                                    int scaleCount, const nvcv::TensorDataStridedCuda &outData, float global_scale,
                                    float shift, bool isStdDev, float epsilon, cudaStream_t stream)
{
    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(inAccess && outAccess);

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) > cuda::TypeTraits<int32_t>::max)
    {
        LOG_ERROR("Input or output size exceeds " << cuda::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return ErrorCode::INVALID_PARAMETER;
    }

    auto srcWrap = nvcv::cuda::CreateTensorWrapNCHW<input_type, int32_t>(inData);
    auto dstWrap = nvcv::cuda::CreateTensorWrapNCHW<input_type, int32_t>(outData);

    PlanarConstParamWrap baseWrap{
        {base.x, base.y, base.z, base.w}
    };
    PlanarConstParamWrap scaleWrap{
        {scale.x, scale.y, scale.z, scale.w}
    };

    int4 inout_size = {static_cast<int>(inAccess->numSamples()), static_cast<int>(inAccess->numChannels()),
                       static_cast<int>(inAccess->numRows()), static_cast<int>(inAccess->numCols())};
    int4 base_size  = {1, baseCount, 1, 1};
    int4 scale_size = {1, scaleCount, 1, 1};

    const uint64_t planes = static_cast<uint64_t>(inout_size.x) * inout_size.y;
    if (planes > 65535u)
    {
        LOG_ERROR("Planar normalize launch exceeds CUDA grid.z limit: N*C=" << planes);
        return ErrorCode::INVALID_PARAMETER;
    }

    // Match the tensor path's vectorized planar dispatch. The kernels accept generic parameter
    // wrappers, so by-value constants retain the same wide load/store and loads-first ILP behavior.
    if constexpr (sizeof(input_type) == 1)
    {
        const bool aligned = inAccess->rowStride() % 4 == 0 && inAccess->chStride() % 4 == 0
                          && inAccess->sampleStride() % 4 == 0 && outAccess->rowStride() % 4 == 0
                          && outAccess->chStride() % 4 == 0 && outAccess->sampleStride() % 4 == 0;
        if (aligned)
        {
            constexpr int NGROUP = kNormalizePlanarILPNGroup;
            dim3          vblock(32, 8);
            dim3          vgrid(divUp(divUp(inout_size.w, 4), static_cast<int>(vblock.x) * NGROUP),
                                divUp(inout_size.z, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
            normalizePlanarVec4Kernel<NGROUP, input_type>
                <<<vgrid, vblock, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size,
                                               global_scale, shift, epsilon, isStdDev);
            checkKernelErrors();
            return ErrorCode::SUCCESS;
        }
    }

    if constexpr (std::is_same_v<input_type, float>)
    {
        const bool aligned = inAccess->rowStride() % 16 == 0 && inAccess->chStride() % 16 == 0
                          && inAccess->sampleStride() % 16 == 0 && outAccess->rowStride() % 16 == 0
                          && outAccess->chStride() % 16 == 0 && outAccess->sampleStride() % 16 == 0;
        if (aligned)
        {
            constexpr int NGROUP = kNormalizePlanarF32ILPNGroup;
            dim3          vblock(32, 8);
            dim3          vgrid(divUp(divUp(inout_size.w, 4), static_cast<int>(vblock.x) * NGROUP),
                                divUp(inout_size.z, static_cast<int>(vblock.y)), static_cast<unsigned int>(planes));
            normalizePlanarVec4Kernel<NGROUP, input_type>
                <<<vgrid, vblock, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size, base_size, scale_size,
                                               global_scale, shift, epsilon, isStdDev);
            checkKernelErrors();
            return ErrorCode::SUCCESS;
        }
    }

    dim3 block(32, 8);
    dim3 grid(divUp(inout_size.w, block.x), divUp(inout_size.z, block.y), static_cast<unsigned int>(planes));
    normalizePlanarKernel<input_type><<<grid, block, 0, stream>>>(srcWrap, baseWrap, scaleWrap, dstWrap, inout_size,
                                                                  base_size, scale_size, global_scale, shift, epsilon,
                                                                  isStdDev);
    checkKernelErrors();
    return ErrorCode::SUCCESS;
}

template<typename input_type>
ErrorCode normalizeScalarPlanar(const nvcv::TensorDataStridedCuda &inData, float4 base, float4 scale, int baseCount,
                                int scaleCount, const nvcv::TensorDataStridedCuda &outData, float global_scale,
                                float shift, cudaStream_t stream)
{
    return normalizeScalarPlanarImpl<input_type>(inData, base, scale, baseCount, scaleCount, outData, global_scale,
                                                 shift, false, 0.f, stream);
}

template<typename input_type>
ErrorCode normalizeScalarPlanarInvStdDev(const nvcv::TensorDataStridedCuda &inData, float4 base, float4 scale,
                                         int baseCount, int scaleCount, const nvcv::TensorDataStridedCuda &outData,
                                         float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    return normalizeScalarPlanarImpl<input_type>(inData, base, scale, baseCount, scaleCount, outData, global_scale,
                                                 shift, true, epsilon, stream);
}

namespace nvcv::legacy::cuda_op {

bool Normalize::checkParamShape(DataShape input_shape, DataShape param_shape)
{
    return (param_shape.N == input_shape.N || param_shape.N == 1)
        && (param_shape.C == input_shape.C || param_shape.C == 1)
        && (param_shape.H == input_shape.H || param_shape.H == 1)
        && (param_shape.W == input_shape.W || param_shape.W == 1);
}

ErrorCode Normalize::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &baseData,
                           const nvcv::TensorDataStridedCuda &scaleData, const nvcv::TensorDataStridedCuda &outData,
                           const float global_scale, const float shift, const float epsilon, const uint32_t flags,
                           cudaStream_t stream)
{
    DataFormat format        = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    if (isPlanar)
    {
        DataFormat base_format  = helpers::GetLegacyDataFormat(baseData);
        DataFormat scale_format = helpers::GetLegacyDataFormat(scaleData);
        // base/scale must use a planar layout matching the input. NHWC/HWC are still allowed when
        // the parameter tensor's channel axis is 1 (i.e. truly scalar), which both layouts express
        // identically.
        auto       isParamCompatible = [&](DataFormat f, const nvcv::TensorDataStridedCuda &data)
        {
            if (f == kNCHW || f == kCHW)
                return true;
            if (f == kNHWC || f == kHWC)
            {
                auto access = TensorDataAccessStridedImagePlanar::Create(data);
                return access && access->numChannels() == 1;
            }
            return false;
        };
        if (!isParamCompatible(base_format, baseData))
        {
            LOG_ERROR("base DataFormat " << base_format
                                         << " is not compatible with planar input; use NCHW/CHW (or a scalar layout)");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
        if (!isParamCompatible(scale_format, scaleData))
        {
            LOG_ERROR("scale DataFormat " << scale_format
                                          << " is not compatible with planar input; use NCHW/CHW (or a scalar layout)");
            return ErrorCode::INVALID_DATA_FORMAT;
        }
    }

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        LOG_ERROR("Invalid DataFormat(in) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
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

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        LOG_ERROR("Invalid DataFormat(out) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType  data_type         = GetLegacyDataType(inData.dtype());
    DataShape input_shape       = GetLegacyDataShape(inAccess->infoShape());
    DataShape base_param_shape  = GetLegacyDataShape(baseAccess->infoShape());
    DataShape scale_param_shape = GetLegacyDataShape(scaleAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!checkParamShape(input_shape, base_param_shape))
    {
        LOG_ERROR("Invalid base shape " << base_param_shape << " for input shape " << input_shape
                                        << "; each dimension must either match the input or be 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!checkParamShape(input_shape, scale_param_shape))
    {
        LOG_ERROR("Invalid scale shape " << scale_param_shape << " for input shape " << input_shape
                                         << "; each dimension must either match the input or be 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef ErrorCode (*normalize_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &baseData,
                                     const TensorDataStridedCuda &scaleData, const TensorDataStridedCuda &outData,
                                     float global_scale, float shift, cudaStream_t stream);

    typedef ErrorCode (*normalizeInvStdDev_t)(
        const TensorDataStridedCuda &inData, const TensorDataStridedCuda &baseData,
        const TensorDataStridedCuda &scaleData, const TensorDataStridedCuda &outData, float global_scale, float shift,
        float epsilon, cudaStream_t stream);

    static const normalize_t funcs_normalize[6][4] = {
        { normalize<uchar>,  0 /*normalize<uchar2>*/,  normalize<uchar3>,  normalize<uchar4>},
        { normalize<schar>,   0 /*normalize<char2>*/,   normalize<char3>,   normalize<char4>},
        {normalize<ushort>, 0 /*normalize<ushort2>*/, normalize<ushort3>, normalize<ushort4>},
        { normalize<short>,  0 /*normalize<short2>*/,  normalize<short3>,  normalize<short4>},
        {   normalize<int>,    0 /*normalize<int2>*/,    normalize<int3>,    normalize<int4>},
        { normalize<float>,  0 /*normalize<float2>*/,  normalize<float3>,  normalize<float4>}
    };

    static const normalizeInvStdDev_t funcs_normalize_stddev[6][4] = {
        { normalizeInvStdDev<uchar>,  0 /*normalizeInvStdDev<uchar2>*/,  normalizeInvStdDev<uchar3>,
         normalizeInvStdDev<uchar4>                                                                                          },
        { normalizeInvStdDev<schar>,   0 /*normalizeInvStdDev<char2>*/,   normalizeInvStdDev<char3>,
         normalizeInvStdDev<char4>                                                                                           },
        {normalizeInvStdDev<ushort>, 0 /*normalizeInvStdDev<ushort2>*/, normalizeInvStdDev<ushort3>,
         normalizeInvStdDev<ushort4>                                                                                         },
        { normalizeInvStdDev<short>,  0 /*normalizeInvStdDev<short2>*/,  normalizeInvStdDev<short3>,
         normalizeInvStdDev<short4>                                                                                          },
        {   normalizeInvStdDev<int>,    0 /*normalizeInvStdDev<int2>*/,    normalizeInvStdDev<int3>, normalizeInvStdDev<int4>},
        { normalizeInvStdDev<float>,  0 /*normalizeInvStdDev<float2>*/,  normalizeInvStdDev<float3>,
         normalizeInvStdDev<float4>                                                                                          }
    };

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: the kernel treats each channel as a separate
        // plane of scalars, so the channel count (1/3/4) reuses one specialization per dtype.
        static const normalize_t funcs_planar[6] = {
            normalizePlanar<uchar>, normalizePlanar<schar>, normalizePlanar<ushort>,
            normalizePlanar<short>, normalizePlanar<int>,   normalizePlanar<float>,
        };
        static const normalizeInvStdDev_t funcs_planar_stddev[6] = {
            normalizePlanarInvStdDev<uchar>, normalizePlanarInvStdDev<schar>, normalizePlanarInvStdDev<ushort>,
            normalizePlanarInvStdDev<short>, normalizePlanarInvStdDev<int>,   normalizePlanarInvStdDev<float>,
        };

        if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
        {
            return funcs_planar_stddev[data_type](inData, baseData, scaleData, outData, global_scale, shift, epsilon,
                                                  stream);
        }
        return funcs_planar[data_type](inData, baseData, scaleData, outData, global_scale, shift, stream);
    }

    if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        // Vectorized fast path for the latency-bound single-channel 8-bit case (the scalar kernel
        // moves 1 byte/thread). Prefer the 16-wide uint4 path (16 columns/thread, 128-bit transfers)
        // when row/sample strides are 16-byte aligned to maximize bytes-in-flight per request; else
        // fall back to the 4-wide uchar4 path (4-byte aligned); else the scalar kernel runs. Output is
        // bit-identical across all three.
        if (data_type == kCV_8U && channels == 1)
        {
            // 16-byte-aligned single-channel U8 stddev: ILP wide-load kernel (kNormalizeILPNGroup
            // independent uint4 groups per thread → more outstanding memory requests / MLP). Falls back
            // to the 4-wide uchar4 path (4-byte aligned), else scalar. Output is bit-identical across all.
            if (inAccess->rowStride() % 16 == 0 && inAccess->sampleStride() % 16 == 0
                && outAccess->rowStride() % 16 == 0 && outAccess->sampleStride() % 16 == 0)
            {
                return normalizeInvStdDevU8VecILP<kNormalizeILPNGroup>(inData, baseData, scaleData, outData,
                                                                       global_scale, shift, epsilon, stream);
            }
            if (inAccess->rowStride() % 4 == 0 && inAccess->sampleStride() % 4 == 0 && outAccess->rowStride() % 4 == 0
                && outAccess->sampleStride() % 4 == 0)
            {
                return normalizeInvStdDevU8Vec(inData, baseData, scaleData, outData, global_scale, shift, epsilon,
                                               stream);
            }
        }
        // Vectorized fast path for the latency-bound single-channel F32 case (the scalar kernel moves
        // 4 bytes/thread). Each thread owns kNormalizeF32ILPNGroup independent float4 groups (128-bit
        // transfers, loads-first) when row/sample strides are 16-byte aligned, else the scalar kernel
        // runs. Output is bit-identical (same float expression, no SaturateCast needed for float out).
        if (data_type == kCV_32F && channels == 1)
        {
            if (inAccess->rowStride() % 16 == 0 && inAccess->sampleStride() % 16 == 0
                && outAccess->rowStride() % 16 == 0 && outAccess->sampleStride() % 16 == 0)
            {
                return normalizeInvStdDevF32VecILP<kNormalizeF32ILPNGroup>(inData, baseData, scaleData, outData,
                                                                           global_scale, shift, epsilon, stream);
            }
        }
        return funcs_normalize_stddev[data_type][channels - 1](inData, baseData, scaleData, outData, global_scale,
                                                               shift, epsilon, stream);
    }
    else
    {
        return funcs_normalize[data_type][channels - 1](inData, baseData, scaleData, outData, global_scale, shift,
                                                        stream);
    }
}

ErrorCode Normalize::infer(const nvcv::TensorDataStridedCuda &inData, const float4 base, const float4 scale,
                           const int baseCount, const int scaleCount, const nvcv::TensorDataStridedCuda &outData,
                           const float global_scale, const float shift, const float epsilon, const uint32_t flags,
                           cudaStream_t stream)
{
    DataFormat format        = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("By-value (scalar/per-channel) normalize supports NHWC/HWC/NCHW/CHW; got " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    if (!inAccess)
    {
        LOG_ERROR("Invalid DataFormat(in) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        LOG_ERROR("Invalid DataFormat(out) " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType  data_type   = GetLegacyDataType(inData.dtype());
    DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());
    int       channels    = input_shape.C;

    if (channels > 4 || channels == 2)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    // The public contract (OpNormalize.h Input/Output dependency) requires the output dtype and
    // extents to equal the input's; the kernels bounds-check against the input extents only, so a
    // mismatched output would be written out of bounds (or with the wrong element type) through its
    // wrap. Reject the mismatch up front.
    DataType out_data_type = GetLegacyDataType(outData.dtype());
    if (data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << data_type << " and " << out_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataShape output_shape = GetLegacyDataShape(outAccess->infoShape());
    if (input_shape != output_shape)
    {
        LOG_ERROR("Shape of input and output must be equal, but got " << input_shape << " and " << output_shape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // Each by-value parameter must be a single scalar (broadcast to all channels) or one value per
    // channel; a float4 holds at most 4 lanes so per-channel is bounded by the 1/3/4 channel support.
    if (!(baseCount == 1 || baseCount == channels))
    {
        LOG_ERROR("Invalid base value count " << baseCount << " for " << channels
                                              << "-channel input; must be 1 (broadcast) or the channel count");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (!(scaleCount == 1 || scaleCount == channels))
    {
        LOG_ERROR("Invalid scale value count " << scaleCount << " for " << channels
                                               << "-channel input; must be 1 (broadcast) or the channel count");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const uint64_t launchPlanes = static_cast<uint64_t>(input_shape.N) * static_cast<uint64_t>(isPlanar ? channels : 1);
    if (launchPlanes > 65535u)
    {
        LOG_ERROR("By-value normalize launch exceeds CUDA grid.z limit: " << (isPlanar ? "N*C=" : "N=")
                                                                          << launchPlanes);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    typedef ErrorCode (*normalize_scalar_t)(const TensorDataStridedCuda &inData, float4 base, float4 scale,
                                            int baseCount, int scaleCount, const TensorDataStridedCuda &outData,
                                            float global_scale, float shift, cudaStream_t stream);

    typedef ErrorCode (*normalize_scalar_stddev_t)(const TensorDataStridedCuda &inData, float4 base, float4 scale,
                                                   int baseCount, int scaleCount, const TensorDataStridedCuda &outData,
                                                   float global_scale, float shift, float epsilon, cudaStream_t stream);

    if (isPlanar)
    {
        // Planar dispatch indexes by dtype only: the kernel treats each channel as a separate plane of
        // scalars, so the channel count (1/3/4) reuses one specialization per dtype (mirrors funcs_planar
        // in the tensor path).
        static const normalize_scalar_t funcs_normalize_scalar_planar[6] = {
            normalizeScalarPlanar<uchar>, normalizeScalarPlanar<schar>, normalizeScalarPlanar<ushort>,
            normalizeScalarPlanar<short>, normalizeScalarPlanar<int>,   normalizeScalarPlanar<float>,
        };
        static const normalize_scalar_stddev_t funcs_normalize_scalar_planar_stddev[6] = {
            normalizeScalarPlanarInvStdDev<uchar>,  normalizeScalarPlanarInvStdDev<schar>,
            normalizeScalarPlanarInvStdDev<ushort>, normalizeScalarPlanarInvStdDev<short>,
            normalizeScalarPlanarInvStdDev<int>,    normalizeScalarPlanarInvStdDev<float>,
        };

        if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
        {
            return funcs_normalize_scalar_planar_stddev[data_type](inData, base, scale, baseCount, scaleCount, outData,
                                                                   global_scale, shift, epsilon, stream);
        }
        return funcs_normalize_scalar_planar[data_type](inData, base, scale, baseCount, scaleCount, outData,
                                                        global_scale, shift, stream);
    }

    // Same [dtype][channels-1] layout as funcs_normalize; the channels==2 slot is unused (rejected above).
    static const normalize_scalar_t funcs_normalize_scalar[6][4] = {
        { normalizeScalar<uchar>, 0,  normalizeScalar<uchar3>,  normalizeScalar<uchar4>},
        { normalizeScalar<schar>, 0,   normalizeScalar<char3>,   normalizeScalar<char4>},
        {normalizeScalar<ushort>, 0, normalizeScalar<ushort3>, normalizeScalar<ushort4>},
        { normalizeScalar<short>, 0,  normalizeScalar<short3>,  normalizeScalar<short4>},
        {   normalizeScalar<int>, 0,    normalizeScalar<int3>,    normalizeScalar<int4>},
        { normalizeScalar<float>, 0,  normalizeScalar<float3>,  normalizeScalar<float4>}
    };

    static const normalize_scalar_stddev_t funcs_normalize_scalar_stddev[6][4] = {
        { normalizeScalarInvStdDev<uchar>, 0,  normalizeScalarInvStdDev<uchar3>,  normalizeScalarInvStdDev<uchar4>},
        { normalizeScalarInvStdDev<schar>, 0,   normalizeScalarInvStdDev<char3>,   normalizeScalarInvStdDev<char4>},
        {normalizeScalarInvStdDev<ushort>, 0, normalizeScalarInvStdDev<ushort3>, normalizeScalarInvStdDev<ushort4>},
        { normalizeScalarInvStdDev<short>, 0,  normalizeScalarInvStdDev<short3>,  normalizeScalarInvStdDev<short4>},
        {   normalizeScalarInvStdDev<int>, 0,    normalizeScalarInvStdDev<int3>,    normalizeScalarInvStdDev<int4>},
        { normalizeScalarInvStdDev<float>, 0,  normalizeScalarInvStdDev<float3>,  normalizeScalarInvStdDev<float4>}
    };

    if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
    {
        return funcs_normalize_scalar_stddev[data_type][channels - 1](inData, base, scale, baseCount, scaleCount,
                                                                      outData, global_scale, shift, epsilon, stream);
    }
    return funcs_normalize_scalar[data_type][channels - 1](inData, base, scale, baseCount, scaleCount, outData,
                                                           global_scale, shift, stream);
}

} // namespace nvcv::legacy::cuda_op
