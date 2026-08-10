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

#ifndef CVCUDA_PRIV_LEGACY_NORMALIZE_PLANAR_CUH
#define CVCUDA_PRIV_LEGACY_NORMALIZE_PLANAR_CUH

#include <cvcuda/cuda_tools/MathWrappers.hpp> // for sqrt
#include <cvcuda/cuda_tools/SaturateCast.hpp> // for SaturateCast

#include <type_traits>

// Shared device helpers for the planar (NCHW / CHW) Normalize kernels. Both the tensor
// (normalize.cu) and image-batch var-shape (normalize_var_shape.cu) paths index base/scale the
// same way and apply the same per-element formula, so the logic lives here once.
namespace nvcv::legacy::cuda_op {

// Per-axis broadcast read index for a parameter with logical (N, C, H, W) extents in param_size:
// use 0 wherever the extent is 1 (broadcast over that axis), otherwise the data index.
__device__ __forceinline__ int4 PlanarBroadcastIndex(int4 param_size, int batch, int channel, int y, int x)
{
    return int4{param_size.x == 1 ? 0 : batch, param_size.y == 1 ? 0 : channel, param_size.z == 1 ? 0 : y,
                param_size.w == 1 ? 0 : x};
}

// Per-element planar normalize: out = saturate((src - base) * mul * global_scale + global_shift),
// where mul is the raw scale, or 1 / sqrt(scale^2 + epsilon) when is_stddev (the
// CVCUDA_NORMALIZE_SCALE_IS_STDDEV flag). base/scale are always float; OutT is the output type.
template<typename OutT, typename SrcT>
__device__ __forceinline__ OutT ApplyPlanarNormalize(SrcT src_val, float base_val, float scale_val, float global_scale,
                                                     float global_shift, bool is_stddev, float epsilon)
{
    const float mul = is_stddev ? (1.0f / nvcv::cuda::sqrt(scale_val * scale_val + epsilon)) : scale_val;
    return nvcv::cuda::SaturateCast<OutT>((static_cast<float>(src_val) - base_val) * mul * global_scale + global_shift);
}

// Width of one vectorized planar group: always 4 columns/thread, but the load/store width depends on
// the element size -- a char4/uchar4 (4 bytes) for 1-byte planes, a float4 (16 bytes) for 4-byte float
// planes. PlanarVec4Type selects the 4-wide vector type while preserving signed byte semantics.
template<typename T, int Size = sizeof(T)>
struct PlanarVec4Type;

template<typename T>
struct PlanarVec4Type<T, 1>
{
    using type = std::conditional_t<std::is_signed_v<T>, char4, uchar4>;
};

template<typename T>
struct PlanarVec4Type<T, 4>
{
    using type = float4;
};

// Vectorized (4 columns/thread) planar normalize body shared by the tensor and var-shape kernels and
// across element widths. The scalar planar kernels move one element per thread, leaving the planar
// paths latency-bound (1-byte ~30-36% BWUtil, F32 single-channel similarly). Here one thread owns four
// consecutive columns of plane (batch, channel) and moves them with a single 4-wide vector load/store
// (char4/uchar4 for 1-byte planes -> 32-bit transfer; float4 for F32 planes -> 128-bit transfer), so a warp
// transfers a full coalesced line. When base/scale are broadcast across those columns (base_size.w ==
// 1 && scale_size.w == 1, the common per-channel case) the base and the inverse-std-dev multiplier are
// computed once. Output is bit-identical to ApplyPlanarNormalize per element.
//
// Caller contract: T is a 1-byte or 4-byte float type, both src and dst plane/row/sample strides are
// sizeof(vec4)-byte aligned, and `cx` is a multiple of 4. A width that is not a multiple of 4 is
// handled scalar for the tail. SrcWrap/DstWrap/ParamWrap expose ptr(batch, channel, y, x); src/dst are
// T, params are float.
template<typename T, typename SrcWrap, typename DstWrap, typename ParamWrap>
__device__ __forceinline__ void NormalizePlanarVec4(const SrcWrap &src, const DstWrap &dst, const ParamWrap &base,
                                                    const ParamWrap &scale, int batch, int channel, int src_y, int cx,
                                                    int4 inout_size, int4 base_size, int4 scale_size,
                                                    float global_scale, float global_shift, bool is_stddev,
                                                    float epsilon)
{
    using Vec4 = typename PlanarVec4Type<T>::type;

    // The 1-byte vector load needs only 4-byte alignment, which every NVCV row pitch satisfies. The
    // float4 load needs 16-byte alignment; default-allocated planes satisfy it (texture-pitch row
    // alignment, texture-aligned base), but a user-wrapped buffer could pick a row pitch that is a
    // multiple of 4 yet not 16. Guard the wide path on the actual row-base alignment and fall back to
    // the scalar tail loop otherwise, so the kernel stays correct for any aligned-or-not input.
    const bool vecAligned = sizeof(T) == 1
                         || (reinterpret_cast<uintptr_t>(src.ptr(batch, channel, src_y, cx)) % sizeof(Vec4) == 0
                             && reinterpret_cast<uintptr_t>(dst.ptr(batch, channel, src_y, cx)) % sizeof(Vec4) == 0);

    if (cx + 4 <= inout_size.w && vecAligned)
    {
        const Vec4 in4 = *reinterpret_cast<const Vec4 *>(src.ptr(batch, channel, src_y, cx));
        Vec4       out4;
        if (base_size.w == 1 && scale_size.w == 1)
        {
            const int4  b      = PlanarBroadcastIndex(base_size, batch, channel, src_y, 0);
            const int4  s      = PlanarBroadcastIndex(scale_size, batch, channel, src_y, 0);
            const float baseV  = *base.ptr(b.x, b.y, b.z, b.w);
            const float scaleV = *scale.ptr(s.x, s.y, s.z, s.w);
            const float mul    = is_stddev ? (1.0f / nvcv::cuda::sqrt(scaleV * scaleV + epsilon)) : scaleV;

            out4.x
                = nvcv::cuda::SaturateCast<T>((static_cast<float>(in4.x) - baseV) * mul * global_scale + global_shift);
            out4.y
                = nvcv::cuda::SaturateCast<T>((static_cast<float>(in4.y) - baseV) * mul * global_scale + global_shift);
            out4.z
                = nvcv::cuda::SaturateCast<T>((static_cast<float>(in4.z) - baseV) * mul * global_scale + global_shift);
            out4.w
                = nvcv::cuda::SaturateCast<T>((static_cast<float>(in4.w) - baseV) * mul * global_scale + global_shift);
        }
        else
        {
            const T raw[4]
                = {static_cast<T>(in4.x), static_cast<T>(in4.y), static_cast<T>(in4.z), static_cast<T>(in4.w)};
            T res[4];
            for (int i = 0; i < 4; ++i)
            {
                const int  x = cx + i;
                const int4 b = PlanarBroadcastIndex(base_size, batch, channel, src_y, x);
                const int4 s = PlanarBroadcastIndex(scale_size, batch, channel, src_y, x);
                res[i] = ApplyPlanarNormalize<T>(raw[i], *base.ptr(b.x, b.y, b.z, b.w), *scale.ptr(s.x, s.y, s.z, s.w),
                                                 global_scale, global_shift, is_stddev, epsilon);
            }
            out4.x = res[0];
            out4.y = res[1];
            out4.z = res[2];
            out4.w = res[3];
        }
        *reinterpret_cast<Vec4 *>(dst.ptr(batch, channel, src_y, cx)) = out4;
    }
    else
    {
        // Scalar path for this thread's group: either the trailing partial group (cx+4 > width) or an
        // unaligned wide access. Bound to this group's 4 columns (or fewer at the row end) so threads
        // do not overlap; for a partial group cx+4 already exceeds width so the bound clamps to width.
        const int xEnd = cx + 4 < inout_size.w ? cx + 4 : inout_size.w;
        for (int x = cx; x < xEnd; ++x)
        {
            const int4 b                       = PlanarBroadcastIndex(base_size, batch, channel, src_y, x);
            const int4 s                       = PlanarBroadcastIndex(scale_size, batch, channel, src_y, x);
            *dst.ptr(batch, channel, src_y, x) = ApplyPlanarNormalize<T>(
                *src.ptr(batch, channel, src_y, x), *base.ptr(b.x, b.y, b.z, b.w), *scale.ptr(s.x, s.y, s.z, s.w),
                global_scale, global_shift, is_stddev, epsilon);
        }
    }
}

} // namespace nvcv::legacy::cuda_op

#endif // CVCUDA_PRIV_LEGACY_NORMALIZE_PLANAR_CUH
