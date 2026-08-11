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
#include "pillow_resize.h"

#include "CvCudaUtils.cuh"

#include <nvcv/Rect.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define BLOCK           32
#define SHARE_MEM_LIMIT 4096

namespace nvcv::legacy::cuda_op {

template<class Filter>
__global__ void _precomputeCoeffs(int in_size, int in0, work_type scale, work_type filterscale, work_type support,
                                  int out_size, int k_size, Filter filterp, int *bounds_out, work_type *kk_out,
                                  bool normalize_coeff, bool use_share_mem)
{
    const int xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    PillowPrecomputeCoeffs<Filter>(xx, local_id, x_offset, in_size, in0, scale, filterscale, support, out_size, k_size,
                                   filterp, bounds_out, kk_out, normalize_coeff, use_share_mem, precision_bits);
}

// SrcPtr2d / DstPtr2d are Ptr2dNHWC (interleaved) or Ptr2dNCHW (planar). Both expose the same
// ptr(b, y, x, c) / rows / cols / ch interface, so one kernel handles both layouts. For planar the
// grid is launched with one z-slice per sample (not per N*C flattened plane), and the per-output-
// pixel bounds / weight-pointer setup is computed once and reused across all `ch` planes -- amortizing
// the work the flattened single-channel view would otherwise repeat once per channel.
//
// NC is the compile-time channel count for the interleaved (NHWC) path: the channels of one output
// pixel are contiguous in memory, so the whole pixel is loaded/stored as one `MakeType<T, NC>` vector
// and accumulated in a `MakeType<work_type, NC>` register. This collapses the per-channel scalar loop
// (NC separate 1-byte loads + NC full ptr() index computations per tap) into a single wide, coalesced
// access per tap -- the resize passes are issue-bound on that per-channel address arithmetic, not on
// bandwidth. NC == 0 selects the original scalar channel loop, used for the planar (NCHW) layout where
// the channels live in separate planes and are not contiguous.
template<int NC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void horizontal_pass(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x    = threadIdx.x;
    const int x_offset   = blockIdx.x * blockDim.x;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_h), x_offset, h_ksize, out_width,
                                           blockDim.x, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int xmin = h_bounds[dst_x * 2];
        int xmax = h_bounds[dst_x * 2 + 1];

        work_type *h_k = &h_k_tmp[local_x * h_ksize];

        if constexpr (NC > 0)
        {
            // Interleaved: load the contiguous NC-channel pixel as one vector; advance one pixel
            // (NC elements) per horizontal tap.
            using vec_t   = cuda::MakeType<T, NC>;
            using accum_t = cuda::MakeType<work_type, NC>;

            const std::byte *p    = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, 0));
            accum_t          h_ss = cuda::SetAll<accum_t>(work_type{0});
            for (int x = 0; x < xmax; ++x)
            {
                h_ss = h_ss + h_k[x] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
                p += sizeof(vec_t);
            }

            vec_t *out = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
            if (round_up)
            {
                // Signed-integer path keeps std::round (half away from zero) per channel for bit
                // exactness with the scalar path; cuda::round has no half-away mode.
                work_type *acc = reinterpret_cast<work_type *>(&h_ss);
#pragma unroll
                for (int c = 0; c < NC; ++c) reinterpret_cast<T *>(out)[c] = cuda::SaturateCast<T>(std::round(acc[c]));
            }
            else
            {
                *out = cuda::SaturateCast<vec_t>(h_ss);
            }
        }
        else
        {
            for (int c = 0; c < src.ch; ++c)
            {
                work_type h_ss = 0.0;
                for (int x = 0; x < xmax; ++x)
                {
                    h_ss = h_ss + *src.ptr(batch_idx, dst_y, x + xmin, c) * h_k[x];
                }

                if (round_up)
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(h_ss));
                else
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(h_ss);
            }
        }
    }
}

// Interleaved horizontal pass computing two adjacent output pixels per thread. Neighboring tap
// windows overlap by roughly k minus the scale step, so the shared segment is loaded once and
// accumulated into both outputs. Three unpredicated segment loops (prefix / shared / suffix) keep
// the fma count and the ascending per-output tap order identical to horizontal_pass (bit-exact);
// only the load count drops. The launch site doubles the block's output span for the staged
// coefficients.
template<int NC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void horizontal_pass_paired(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                       int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                       work_type init_buffer, bool round_up, bool use_share_mem)
{
    static_assert(NC > 0, "paired horizontal pass is the interleaved channel-vectorized path");
    const int pair0      = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x    = threadIdx.x;
    const int x_offset   = blockIdx.x * blockDim.x * 2;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_hq[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_hq), x_offset, h_ksize,
                                           out_width, blockDim.x * 2, use_share_mem);

    if (pair0 < out_width && dst_y < out_height)
    {
        using vec_t   = cuda::MakeType<T, NC>;
        using accum_t = cuda::MakeType<work_type, NC>;

        const bool has1  = pair0 + 1 < out_width;
        const int  xmin0 = h_bounds[pair0 * 2];
        const int  xmax0 = h_bounds[pair0 * 2 + 1];
        const int  end0  = xmin0 + xmax0;
        // Without a second output, park its window at end0 so the shared and suffix loops are
        // empty and k1 (one slot past this image's coefficients) is never read.
        const int  xmin1 = has1 ? h_bounds[(pair0 + 1) * 2] : end0;
        const int  xmax1 = has1 ? h_bounds[(pair0 + 1) * 2 + 1] : 0;

        work_type *k0 = &h_k_tmp[(local_x * 2) * h_ksize];
        work_type *k1 = &h_k_tmp[(local_x * 2 + 1) * h_ksize];

        const int end1 = xmin1 + xmax1;

        accum_t s0 = cuda::SetAll<accum_t>(work_type{0});
        accum_t s1 = cuda::SetAll<accum_t>(work_type{0});

        const std::byte *base   = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, 0, 0));
        const int        preEnd = min(xmin1, end0);
        for (int x = xmin0; x < preEnd; ++x)
        {
            s0 = s0
               + k0[x - xmin0]
                     * cuda::StaticCast<work_type>(
                         *reinterpret_cast<const vec_t *>(base + static_cast<size_t>(x) * sizeof(vec_t)));
        }
        for (int x = xmin1; x < end0; ++x)
        {
            const accum_t px = cuda::StaticCast<work_type>(
                *reinterpret_cast<const vec_t *>(base + static_cast<size_t>(x) * sizeof(vec_t)));
            s0 = s0 + k0[x - xmin0] * px;
            s1 = s1 + k1[x - xmin1] * px;
        }
        const int sufBegin = max(xmin1, end0);
        for (int x = sufBegin; x < end1; ++x)
        {
            s1 = s1
               + k1[x - xmin1]
                     * cuda::StaticCast<work_type>(
                         *reinterpret_cast<const vec_t *>(base + static_cast<size_t>(x) * sizeof(vec_t)));
        }

        vec_t *out0 = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, pair0, 0));
        if (round_up)
        {
            work_type *acc0 = reinterpret_cast<work_type *>(&s0);
#pragma unroll
            for (int c = 0; c < NC; ++c) reinterpret_cast<T *>(out0)[c] = cuda::SaturateCast<T>(std::round(acc0[c]));
        }
        else
        {
            *out0 = cuda::SaturateCast<vec_t>(s0);
        }
        if (has1)
        {
            vec_t *out1 = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, pair0 + 1, 0));
            if (round_up)
            {
                work_type *acc1 = reinterpret_cast<work_type *>(&s1);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    reinterpret_cast<T *>(out1)[c] = cuda::SaturateCast<T>(std::round(acc1[c]));
            }
            else
            {
                *out1 = cuda::SaturateCast<vec_t>(s1);
            }
        }
    }
}

template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void horizontal_pass_planar_channels(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp,
                                                int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds,
                                                work_type *v_kk, work_type init_buffer, bool round_up,
                                                bool use_share_mem)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x    = threadIdx.x;
    const int x_offset   = blockIdx.x * blockDim.x;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_hp[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_hp), x_offset, h_ksize,
                                           out_width, blockDim.x, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int        xmin = h_bounds[dst_x * 2];
        int        xmax = h_bounds[dst_x * 2 + 1];
        work_type *h_k  = &h_k_tmp[local_x * h_ksize];
        work_type  h_ss[PC];

#pragma unroll
        for (int c = 0; c < PC; ++c)
        {
            h_ss[c] = 0.0;
        }

        // Hoist the per-plane row pointers: the tap loop then advances plain pointers instead of
        // re-deriving the full (b, y, x, c) offset for every tap of every plane. Three byte planes
        // only: at four planes the extra pointer registers measured 3.5-6.6% slower on multi-tap
        // rows, and float planes regressed contract cubic 6-16%.
        if constexpr (PC == 3 && sizeof(T) == 1)
        {
            const T *row[PC];
#pragma unroll
            for (int c = 0; c < PC; ++c) row[c] = src.ptr(batch_idx, dst_y, xmin, c);

            for (int x = 0; x < xmax; ++x)
            {
                const work_type k = h_k[x];
#pragma unroll
                for (int c = 0; c < PC; ++c)
                {
                    h_ss[c] = h_ss[c] + row[c][x] * k;
                }
            }
        }
        else
        {
            for (int x = 0; x < xmax; ++x)
            {
                const work_type k = h_k[x];
#pragma unroll
                for (int c = 0; c < PC; ++c)
                {
                    h_ss[c] = h_ss[c] + *src.ptr(batch_idx, dst_y, x + xmin, c) * k;
                }
            }
        }

#pragma unroll
        for (int c = 0; c < PC; ++c)
        {
            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(h_ss[c]));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(h_ss[c]);
        }
    }
}

template<int NC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                              int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                              work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_v[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_v), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        if constexpr (NC > 0)
        {
            // Interleaved: load the contiguous NC-channel pixel as one vector; advance one row
            // (rowStride bytes) per vertical tap.
            using vec_t   = cuda::MakeType<T, NC>;
            using accum_t = cuda::MakeType<work_type, NC>;

            const std::byte *p  = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x, 0));
            accum_t          ss = cuda::SetAll<accum_t>(init_buffer);
            for (int y = 0; y < ymax; ++y)
            {
                ss = ss + v_k[y] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
                p += src.rowStride;
            }

            vec_t *out = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
            if (round_up)
            {
                work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
                for (int c = 0; c < NC; ++c) reinterpret_cast<T *>(out)[c] = cuda::SaturateCast<T>(std::round(acc[c]));
            }
            else
            {
                *out = cuda::SaturateCast<vec_t>(ss);
            }
        }
        else
        {
            for (int c = 0; c < src.ch; ++c)
            {
                work_type ss = init_buffer;
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + *src.ptr(batch_idx, y + ymin, dst_x, c) * v_k[y];
                }

                if (round_up)
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(ss));
                else
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(ss);
            }
        }
    }
}

// Interleaved (NHWC) vertical pass vectorized along the flattened row. The vertical weights depend
// only on dst_y, so all cols*ch scalars of an output row share v_k: each thread accumulates VEC
// consecutive scalars with one MakeType<T, VEC> load per tap instead of one per-pixel channel
// vector, cutting the load/store and address-arithmetic issue count these passes are bound on.
// Per-element arithmetic (taps, order, cast, saturate) is identical to vertical_pass. The launch
// site gates on vector alignment of both buffers; the row tail falls back to a scalar loop.
template<int VEC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass_vec(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                  int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                  work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int elem0      = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y;
    const int batch_idx  = get_batch_idx();
    const int out_height = dst.rows;
    const int row_elems  = dst.cols * dst.ch;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_vv[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_vv), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (elem0 < row_elems && dst_y < out_height)
    {
        const int ymin = v_bounds[dst_y * 2];
        const int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        const std::byte *src_row
            = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, 0, 0)) + elem0 * sizeof(T);
        T *dst_row = dst.ptr(batch_idx, dst_y, 0, 0) + elem0;

        if (elem0 + VEC <= row_elems)
        {
            using vec_t   = cuda::MakeType<T, VEC>;
            using accum_t = cuda::MakeType<work_type, VEC>;

            accum_t ss = cuda::SetAll<accum_t>(init_buffer);
            for (int y = 0; y < ymax; ++y)
            {
                ss = ss + v_k[y] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(src_row));
                src_row += src.rowStride;
            }
            if (round_up)
            {
                work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
                for (int i = 0; i < VEC; ++i) dst_row[i] = cuda::SaturateCast<T>(std::round(acc[i]));
            }
            else
            {
                *reinterpret_cast<vec_t *>(dst_row) = cuda::SaturateCast<vec_t>(ss);
            }
        }
        else
        {
            for (int i = 0; elem0 + i < row_elems; ++i)
            {
                const std::byte *p  = src_row + i * sizeof(T);
                work_type        ss = init_buffer;
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + *reinterpret_cast<const T *>(p) * v_k[y];
                    p += src.rowStride;
                }
                dst_row[i] = round_up ? cuda::SaturateCast<T>(std::round(ss)) : cuda::SaturateCast<T>(ss);
            }
        }
    }
}

// Fused single-pass resize for the interleaved (NHWC) FLOAT path. Mathematically identical, op-for-op,
// to running horizontal_pass then vertical_pass through a FLOAT intermediate: for each contributing
// source row the inner horizontal sum reproduces exactly what the float intermediate would store
// (SaturateCast<float> is the identity, so no value is lost between passes), and accumulating
// v_k[y] * inner then yields the same float result as the separable path. The intermediate stays in a
// register, eliminating its full-frame DRAM round-trip -- the dominant cost at high resolution, where
// both passes sit at the DRAM ridge. FLOAT ONLY: an integer intermediate is quantized between passes
// (Pillow semantics), so fusing it would change results; that path keeps the separable kernels.
template<int NC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void fused_pass(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize, int v_ksize,
                           int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk, work_type init_buffer,
                           bool round_up, bool use_share_mem)
{
    static_assert(NC > 0, "fused_pass is the interleaved channel-vectorized path");
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    const int out_width  = dst.cols;
    const int out_height = dst.rows;

    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int xmin = h_bounds[dst_x * 2];
    const int xmax = h_bounds[dst_x * 2 + 1];
    const int ymin = v_bounds[dst_y * 2];
    const int ymax = v_bounds[dst_y * 2 + 1];

    const work_type *h_k = &h_kk[dst_x * h_ksize];
    const work_type *v_k = &v_kk[dst_y * v_ksize];

    using vec_t   = cuda::MakeType<T, NC>;
    using accum_t = cuda::MakeType<work_type, NC>;

    accum_t ss = cuda::SetAll<accum_t>(init_buffer);
    for (int y = 0; y < ymax; ++y)
    {
        // inner == the intermediate value at (ymin + y, dst_x): identical taps, identical order.
        const std::byte *p     = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin + y, xmin, 0));
        accum_t          inner = cuda::SetAll<accum_t>(work_type{0});
        for (int x = 0; x < xmax; ++x)
        {
            inner = inner + h_k[x] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
            p += sizeof(vec_t);
        }
        ss = ss + v_k[y] * inner;
    }

    vec_t *out = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
    if (round_up)
    {
        work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
        for (int c = 0; c < NC; ++c) reinterpret_cast<T *>(out)[c] = cuda::SaturateCast<T>(std::round(acc[c]));
    }
    else
    {
        *out = cuda::SaturateCast<vec_t>(ss);
    }
}

// Fused single-pass resize for the planar (NCHW) FLOAT path with a compile-time channel count.
// Channels live in separate planes, so each plane's horizontal window is walked from a hoisted
// per-plane row pointer; taps and order match horizontal_pass_planar_channels followed by the
// planar vertical pass through the float intermediate (identity between passes -- bit-exact), with
// the intermediate kept in registers instead of round-tripping DRAM. FLOAT ONLY for the same
// quantization reason as fused_pass; the byte rows are issue-bound and measured slower fused. The
// per-output bounds/weight setup is shared across all PC planes, like the separable planar kernels.
template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void fused_pass_planar(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                  int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                  work_type init_buffer, bool round_up, bool use_share_mem)
{
    static_assert(PC > 0, "fused planar pass needs a compile-time plane count");
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    const int out_width  = dst.cols;
    const int out_height = dst.rows;

    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int xmin = h_bounds[dst_x * 2];
    const int xmax = h_bounds[dst_x * 2 + 1];
    const int ymin = v_bounds[dst_y * 2];
    const int ymax = v_bounds[dst_y * 2 + 1];

    const work_type *h_k = &h_kk[dst_x * h_ksize];
    const work_type *v_k = &v_kk[dst_y * v_ksize];

    work_type ss[PC];
#pragma unroll
    for (int c = 0; c < PC; ++c) ss[c] = init_buffer;

    for (int y = 0; y < ymax; ++y)
    {
        const T *row[PC];
#pragma unroll
        for (int c = 0; c < PC; ++c) row[c] = src.ptr(batch_idx, ymin + y, xmin, c);

        work_type inner[PC];
#pragma unroll
        for (int c = 0; c < PC; ++c) inner[c] = work_type{0};
        for (int x = 0; x < xmax; ++x)
        {
            const work_type k = h_k[x];
#pragma unroll
            for (int c = 0; c < PC; ++c) inner[c] = inner[c] + row[c][x] * k;
        }
        const work_type vk = v_k[y];
#pragma unroll
        for (int c = 0; c < PC; ++c) ss[c] = ss[c] + vk * inner[c];
    }

#pragma unroll
    for (int c = 0; c < PC; ++c)
    {
        if (round_up)
            *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(ss[c]));
        else
            *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(ss[c]);
    }
}

// Planar (NCHW) vertical pass, vectorized across the X (column) dimension. Channels live in separate
// planes so the channel loop stays, but within a plane consecutive output columns are contiguous in
// memory AND share the same vertical resampling weights v_k (they only depend on dst_y). Each thread
// therefore computes VEC consecutive output columns: one MakeType<T, VEC> coalesced load per tap feeds
// VEC accumulators, collapsing VEC byte-loads + VEC index computations per tap into one wide access.
// The launch site only selects this kernel when every relevant base pointer / stride is aligned to
// sizeof(MakeType<T, VEC>) (vector loads require natural alignment); otherwise the scalar pass runs.
// The partial tail block (out_width not a multiple of VEC) falls back to a per-column scalar loop.
template<int VEC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass_planar_vec(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                         int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                         work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x0     = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_vp[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_vp), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (dst_x0 < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        using vec_t   = cuda::MakeType<T, VEC>;
        using accum_t = cuda::MakeType<work_type, VEC>;

        const bool full = (dst_x0 + VEC <= out_width);
        for (int c = 0; c < src.ch; ++c)
        {
            if (full)
            {
                const std::byte *p  = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x0, c));
                accum_t          ss = cuda::SetAll<accum_t>(init_buffer);
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + v_k[y] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
                    p += src.rowStride;
                }

                vec_t *out = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, dst_y, dst_x0, c));
                if (round_up)
                {
                    work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
                    for (int i = 0; i < VEC; ++i)
                        reinterpret_cast<T *>(out)[i] = cuda::SaturateCast<T>(std::round(acc[i]));
                }
                else
                {
                    *out = cuda::SaturateCast<vec_t>(ss);
                }
            }
            else
            {
                for (int i = 0; i < VEC && dst_x0 + i < out_width; ++i)
                {
                    const int dst_x = dst_x0 + i;
                    work_type ss    = init_buffer;
                    for (int y = 0; y < ymax; ++y)
                    {
                        ss = ss + *src.ptr(batch_idx, y + ymin, dst_x, c) * v_k[y];
                    }

                    if (round_up)
                        *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(ss));
                    else
                        *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(ss);
                }
            }
        }
    }
}

// Planar vertical pass computing two adjacent output rows per thread (VEC columns each). Adjacent
// output rows' vertical tap windows overlap by roughly k minus the scale step, so the shared source
// rows are loaded -- and converted to work_type -- once, then accumulated into both outputs (three
// segment loops like horizontal_pass_paired). Per-output tap order and FMA sequence are identical
// to vertical_pass_planar_vec (bit-exact); only the load/convert count drops. The launch site
// doubles the per-block output-row span for the coefficient indexing.
template<int VEC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass_planar_vec_paired(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp,
                                                int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds,
                                                work_type *v_kk, work_type init_buffer, bool round_up,
                                                bool use_share_mem)
{
    const int dst_x0     = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int row0       = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y * 2;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_vpq[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_vpq), y_offset, v_ksize,
                                           out_height, blockDim.y * 2, use_share_mem);

    if (dst_x0 < out_width && row0 < out_height)
    {
        const bool has1  = row0 + 1 < out_height;
        const int  ymin0 = v_bounds[row0 * 2];
        const int  ymax0 = v_bounds[row0 * 2 + 1];
        const int  end0  = ymin0 + ymax0;
        // Without a second output row, park its window at end0 so the shared and suffix loops are
        // empty and k1 (one slot past this image's coefficients) is never read.
        const int  ymin1 = has1 ? v_bounds[(row0 + 1) * 2] : end0;
        const int  ymax1 = has1 ? v_bounds[(row0 + 1) * 2 + 1] : 0;
        const int  end1  = ymin1 + ymax1;

        work_type *k0 = &v_k_tmp[(local_y * 2) * v_ksize];
        work_type *k1 = &v_k_tmp[(local_y * 2 + 1) * v_ksize];

        const int preEnd   = min(ymin1, end0);
        const int sufBegin = max(ymin1, end0);

        using vec_t   = cuda::MakeType<T, VEC>;
        using accum_t = cuda::MakeType<work_type, VEC>;

        const bool full = (dst_x0 + VEC <= out_width);
        for (int c = 0; c < src.ch; ++c)
        {
            if (full)
            {
                accum_t s0 = cuda::SetAll<accum_t>(init_buffer);
                accum_t s1 = cuda::SetAll<accum_t>(init_buffer);

                const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin0, dst_x0, c));
                for (int y = ymin0; y < preEnd; ++y)
                {
                    s0 = s0 + k0[y - ymin0] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
                    p += src.rowStride;
                }
                for (int y = preEnd; y < end0; ++y)
                {
                    const accum_t px = cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(p));
                    s0               = s0 + k0[y - ymin0] * px;
                    s1               = s1 + k1[y - ymin1] * px;
                    p += src.rowStride;
                }
                const std::byte *ps = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, sufBegin, dst_x0, c));
                for (int y = sufBegin; y < end1; ++y)
                {
                    s1 = s1 + k1[y - ymin1] * cuda::StaticCast<work_type>(*reinterpret_cast<const vec_t *>(ps));
                    ps += src.rowStride;
                }

                vec_t *out0 = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, row0, dst_x0, c));
                if (round_up)
                {
                    work_type *acc = reinterpret_cast<work_type *>(&s0);
#pragma unroll
                    for (int i = 0; i < VEC; ++i)
                        reinterpret_cast<T *>(out0)[i] = cuda::SaturateCast<T>(std::round(acc[i]));
                }
                else
                {
                    *out0 = cuda::SaturateCast<vec_t>(s0);
                }
                if (has1)
                {
                    vec_t *out1 = reinterpret_cast<vec_t *>(dst.ptr(batch_idx, row0 + 1, dst_x0, c));
                    if (round_up)
                    {
                        work_type *acc = reinterpret_cast<work_type *>(&s1);
#pragma unroll
                        for (int i = 0; i < VEC; ++i)
                            reinterpret_cast<T *>(out1)[i] = cuda::SaturateCast<T>(std::round(acc[i]));
                    }
                    else
                    {
                        *out1 = cuda::SaturateCast<vec_t>(s1);
                    }
                }
            }
            else
            {
                // Column tail: two independent scalar walks, identical to vertical_pass_planar_vec's
                // tail loop for each output row.
                for (int i = 0; i < VEC && dst_x0 + i < out_width; ++i)
                {
                    const int dst_x = dst_x0 + i;
                    work_type ss    = init_buffer;
                    for (int y = 0; y < ymax0; ++y)
                    {
                        ss = ss + *src.ptr(batch_idx, y + ymin0, dst_x, c) * k0[y];
                    }
                    if (round_up)
                        *dst.ptr(batch_idx, row0, dst_x, c) = cuda::SaturateCast<T>(std::round(ss));
                    else
                        *dst.ptr(batch_idx, row0, dst_x, c) = cuda::SaturateCast<T>(ss);

                    if (has1)
                    {
                        work_type st = init_buffer;
                        for (int y = 0; y < ymax1; ++y)
                        {
                            st = st + *src.ptr(batch_idx, y + ymin1, dst_x, c) * k1[y];
                        }
                        if (round_up)
                            *dst.ptr(batch_idx, row0 + 1, dst_x, c) = cuda::SaturateCast<T>(std::round(st));
                        else
                            *dst.ptr(batch_idx, row0 + 1, dst_x, c) = cuda::SaturateCast<T>(st);
                    }
                }
            }
        }
    }
}

template<typename Filter, typename elem_type>
void pillow_resize_v2(const TensorDataAccessStridedImagePlanar &inData,
                      const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace, bool normalize_coeff,
                      work_type init_buffer, bool round_up, bool planar, cudaStream_t stream)
{
    cuda_op::DataShape   input_shape = GetLegacyDataShape(inData.infoShape());
    Ptr2dNHWC<elem_type> src_ptr(inData);
    Ptr2dNHWC<elem_type> dst_ptr(outData);
    NVCVRectI            roi = {0, 0, src_ptr.cols, src_ptr.rows};
    Filter               filterp;
    work_type            h_scale = 0, v_scale = 0;
    work_type            h_filterscale = 0, v_filterscale = 0;
    h_filterscale = h_scale = static_cast<work_type>(roi.width) / dst_ptr.cols;
    v_filterscale = v_scale = static_cast<work_type>(roi.height) / dst_ptr.rows;

    int out_width  = dst_ptr.cols;
    int out_height = dst_ptr.rows;

    if (h_filterscale < 1.0)
    {
        h_filterscale = 1.0;
    }
    if (v_filterscale < 1.0)
    {
        v_filterscale = 1.0;
    }

    // Determine support size (length of resampling filter).
    work_type h_support = filterp.support() * h_filterscale;
    work_type v_support = filterp.support() * v_filterscale;

    // Maximum number of coeffs.
    int h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
    int v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;

    work_type     *h_kk     = (work_type *)((char *)gpu_workspace);
    work_type     *v_kk     = (work_type *)((char *)h_kk + dst_ptr.cols * h_k_size * sizeof(work_type));
    int           *h_bounds = (int *)((char *)v_kk + dst_ptr.rows * v_k_size * sizeof(work_type));
    int           *v_bounds = (int *)((char *)h_bounds + dst_ptr.cols * 2 * sizeof(int));
    // The intermediate buffer is read/written with up-to-16-byte vector loads/stores (MakeType<T,4>),
    // which require natural alignment. The coefficient/bounds regions above are work_type/int sized, so
    // the raw offset is only 4-byte aligned -- round d_h_data up to 16 bytes. getWorkspaceRequirements
    // reserves an extra 16 bytes for this padding.
    std::uintptr_t d_h_raw  = reinterpret_cast<std::uintptr_t>((char *)v_bounds + dst_ptr.rows * 2 * sizeof(int));
    elem_type     *d_h_data = reinterpret_cast<elem_type *>((d_h_raw + 15) & ~std::uintptr_t(15));

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSizeH(divUp(out_width, blockSize.x), divUp(input_shape.H, blockSize.y), input_shape.N);
    dim3 gridSizeV(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), input_shape.N);

    dim3 coef_block(BLOCK * 2, 1, 1);
    dim3 h_coef_grid(divUp(dst_ptr.cols, coef_block.x), 1, 1);
    dim3 v_coef_grid(divUp(dst_ptr.rows, coef_block.x), 1, 1);

    size_t h_sm_size = coef_block.x * (h_k_size * sizeof(work_type));
    size_t v_sm_size = coef_block.x * (v_k_size * sizeof(work_type));

    size_t hv_sm_size1     = h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2     = v_k_size * sizeof(work_type) * blockSize.y;
    bool   h_use_share_mem = h_sm_size <= SHARE_MEM_LIMIT;
    if (!h_use_share_mem)
    {
        h_sm_size = 0;
    }
    bool v_use_share_mem = v_sm_size <= SHARE_MEM_LIMIT;
    if (!v_use_share_mem)
    {
        v_sm_size = 0;
    }
    bool hv_use_share_mem = (hv_sm_size1 <= SHARE_MEM_LIMIT) && (hv_sm_size2 <= SHARE_MEM_LIMIT);
    if (!hv_use_share_mem)
    {
        hv_sm_size1 = 0;
        hv_sm_size2 = 0;
    }
    // compute horizental coef
    _precomputeCoeffs<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        src_ptr.cols, roi.x, h_scale, h_filterscale, h_support, dst_ptr.cols, h_k_size, filterp, h_bounds, h_kk,
        normalize_coeff, h_use_share_mem);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    // compute vertical coef
    _precomputeCoeffs<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        src_ptr.rows, roi.y, v_scale, v_filterscale, v_support, dst_ptr.rows, v_k_size, filterp, v_bounds, v_kk,
        normalize_coeff, v_use_share_mem);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    // Horizontal then vertical separable pass. The intermediate buffer holds the horizontally-resized
    // image (out_width columns, source rows) in the SAME layout as the input so the channel loop and
    // grid (z = N) match. Interleaved uses Ptr2dNHWC; planar uses Ptr2dNCHW, which keeps one z-slice
    // per sample and amortizes the per-pixel bounds/weight setup across the C planes.
    auto launch = [&](auto nc_const, auto src_p, auto inter_p, auto dst_p)
    {
        constexpr int  NC = decltype(nc_const)::value;
        // The fused kernel removes the intermediate write/read, but recomputes each horizontal sum for
        // every vertical tap. Upscales win across the measured GPUs; small-kernel downscales win only on
        // the architecture families selected by PillowResizeSupportsFusedDownscale. FLOAT only: an
        // integer intermediate is quantized between passes (Pillow semantics), and the byte rows are
        // issue-bound, not DRAM-bound -- the fused recompute measured 25-41% slower for uchar3.
        constexpr bool can_fuse     = (NC > 0 && std::is_same_v<elem_type, float>);
        const bool     is_upscale   = h_scale < work_type(1) && v_scale < work_type(1);
        const bool     small_kernel = h_k_size <= kPillowResizeFuseMaxKSize && v_k_size <= kPillowResizeFuseMaxKSize;
        const bool     fuse_pass    = can_fuse && filterp.support() <= work_type(1)
                            && (is_upscale || (small_kernel && PillowResizeSupportsFusedDownscale()));
        if (fuse_pass)
        {
            if constexpr (can_fuse) // guard instantiation: fused_pass static_asserts NC > 0
            {
                (void)inter_p; // the intermediate buffer is unused on the fused path
                fused_pass<NC, elem_type, Filter>
                    <<<gridSizeV, blockSize, 0, stream>>>(src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                          h_kk, v_bounds, v_kk, init_buffer, round_up, false);
                checkKernelErrors();
            }
        }
        else
        {
            // Pairing wins where neighboring tap windows are wide and overlap: three-channel byte
            // pixels on downscale. Float pixels and upscale windows measured slower paired locally;
            // single-channel and four-channel rows regressed on the A100/H100 regen burnins.
            bool paired = false;
            if constexpr (NC == 3 && sizeof(elem_type) == 1)
            {
                if (h_scale > work_type(1))
                {
                    const size_t paired_sm  = 2 * hv_sm_size1;
                    const bool   paired_use = hv_use_share_mem && paired_sm <= SHARE_MEM_LIMIT;
                    dim3         gridSizeHP(divUp(out_width, 2 * static_cast<int>(blockSize.x)),
                                            divUp(input_shape.H, blockSize.y), input_shape.N);
                    horizontal_pass_paired<NC, elem_type, Filter>
                        <<<gridSizeHP, blockSize, paired_use ? paired_sm : 0, stream>>>(
                            src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
                            init_buffer, round_up, paired_use);
                    paired = true;
                }
            }
            if (!paired)
            {
                horizontal_pass<NC, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
            checkKernelErrors();
            constexpr int    VEC    = 4;
            constexpr size_t amask  = alignof(cuda::MakeType<elem_type, VEC>) - 1;
            auto             flatOk = [&](const auto &p)
            {
                return ((reinterpret_cast<std::uintptr_t>(p.data) | static_cast<std::uintptr_t>(p.imgStride)
                         | static_cast<std::uintptr_t>(p.rowStride))
                        & amask)
                    == 0;
            };
            // Four-channel pixels already move as one vector per tap in vertical_pass; the flat
            // mapping only reshuffles threads there and measured slower on A100 (expand uchar4).
            if (NC != 4 && flatOk(inter_p) && flatOk(dst_p))
            {
                const int row_elems = out_width * input_shape.C;
                dim3      gridSizeVV(divUp(divUp(row_elems, VEC), static_cast<int>(blockSize.x)),
                                     divUp(out_height, blockSize.y), input_shape.N);
                // Threads of a warp share dst_y here, so the per-tap weight loads are L1
                // broadcasts; staging them through shared memory only adds the block barrier.
                vertical_pass_vec<VEC, elem_type, Filter>
                    <<<gridSizeVV, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            }
            else
            {
                vertical_pass<NC, elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
                    inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
            checkKernelErrors();
        }
    };

    if (planar)
    {
        // Planar (NCHW): channels are in separate planes. The horizontal pass uses the scalar channel
        // loop (NC == 0); the vertical pass is x-vectorized (consecutive columns share v_k) when the
        // intermediate and output buffers are vector-aligned, else scalar.
        Ptr2dNCHW<elem_type> src_p(inData);
        Ptr2dNCHW<elem_type> inter_p(input_shape.N, input_shape.H, out_width, input_shape.C, (elem_type *)d_h_data);
        Ptr2dNCHW<elem_type> dst_p(outData);

        // Fused planar resize (FLOAT only): drops the intermediate round-trip and reproduces the
        // separable result bit-exactly. Upscale fusion is gated to four channels and moderate ratios:
        // on the reference GPUs three-channel expands measured 0.95-1.00x fused (A100) and extreme
        // upscales (scale < 0.4, e.g. 4.5x anisotropic) regressed to 0.80x -- the intermediate
        // shrinks relative to the output, so the recompute and the lost vertical vectorization
        // outweigh the saved traffic. Downscale fusion wins on both reference GPUs for 3 and 4
        // channels (1.03-1.47x) and keeps the architecture gate.
        if constexpr (std::is_same_v<elem_type, float>)
        {
            const bool is_upscale   = h_scale < work_type(1) && v_scale < work_type(1);
            const bool moderate_up  = h_scale >= work_type(0.4) && v_scale >= work_type(0.4);
            const bool small_kernel = h_k_size <= kPillowResizeFuseMaxKSize && v_k_size <= kPillowResizeFuseMaxKSize;
            const bool fuse_planar  = filterp.support() <= work_type(1)
                                  && ((is_upscale && moderate_up && input_shape.C == 4)
                                      || (!is_upscale && small_kernel && PillowResizeSupportsFusedDownscale()
                                          && (input_shape.C == 3 || input_shape.C == 4)));
            if (fuse_planar)
            {
                if (input_shape.C == 3)
                {
                    fused_pass_planar<3, elem_type, Filter>
                        <<<gridSizeV, blockSize, 0, stream>>>(src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                              h_kk, v_bounds, v_kk, init_buffer, round_up, false);
                }
                else
                {
                    fused_pass_planar<4, elem_type, Filter>
                        <<<gridSizeV, blockSize, 0, stream>>>(src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                              h_kk, v_bounds, v_kk, init_buffer, round_up, false);
                }
                checkKernelErrors();
                return;
            }
        }

        switch (input_shape.C)
        {
        case 3:
            horizontal_pass_planar_channels<3, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer, round_up,
                hv_use_share_mem);
            break;
        case 4:
            if constexpr (!std::is_same_v<elem_type, float>)
            {
                horizontal_pass_planar_channels<4, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
            else
            {
                horizontal_pass<0, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
            break;
        default:
            horizontal_pass<0, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer, round_up,
                hv_use_share_mem);
            break;
        }
        checkKernelErrors();

        constexpr int VEC       = 4;
        const size_t  alignMask = VEC * sizeof(elem_type) - 1;
        auto          aligned   = [&](const Ptr2dNCHW<elem_type> &p)
        {
            return ((reinterpret_cast<std::uintptr_t>(p.data) | static_cast<size_t>(p.imgStride)
                     | static_cast<size_t>(p.chStride) | static_cast<size_t>(p.rowStride))
                    & alignMask)
                == 0;
        };
        if (aligned(inter_p) && aligned(dst_p))
        {
            // Threads of a warp share the output row (or row pair), so per-tap weight loads are L1
            // broadcasts; skip the shared-memory staging and its block barrier. Row pairing only
            // pays where adjacent output rows actually share source taps AND the shared load also
            // shares a byte-to-float convert: three-plane byte linear upscale. Downscale windows are
            // disjoint, float rows have no convert to share, and four byte planes regressed 6% on
            // A100 (the extra plane iteration doubles the paired accumulator pressure).
            if (sizeof(elem_type) == 1 && input_shape.C == 3 && v_scale < work_type(1)
                && filterp.support() <= work_type(1))
            {
                dim3 gridSizeVP(divUp(divUp(out_width, VEC), static_cast<int>(blockSize.x)),
                                divUp(out_height, 2 * static_cast<int>(blockSize.y)), input_shape.N);
                vertical_pass_planar_vec_paired<VEC, elem_type, Filter>
                    <<<gridSizeVP, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            }
            else
            {
                dim3 gridSizeVP(divUp(divUp(out_width, VEC), static_cast<int>(blockSize.x)),
                                divUp(out_height, blockSize.y), input_shape.N);
                vertical_pass_planar_vec<VEC, elem_type, Filter>
                    <<<gridSizeVP, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            }
        }
        else
        {
            vertical_pass<0, elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
                inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer, round_up,
                hv_use_share_mem);
        }
        checkKernelErrors();
    }
    else
    {
        // Interleaved (NHWC): vectorize over the contiguous channels at the known channel count.
        Ptr2dNHWC<elem_type> src_p(inData);
        Ptr2dNHWC<elem_type> inter_p(input_shape.N, input_shape.H, out_width, input_shape.C, (elem_type *)d_h_data);
        Ptr2dNHWC<elem_type> dst_p(outData);

        // The vectorized pixel load/store requires the buffers be aligned to alignof(MakeType<T,NC>).
        // The intermediate is 16-byte aligned above, but the user input/output tensors may have
        // arbitrary row strides -- gate on all three and fall back to the scalar loop (NC=0) otherwise.
        auto try_vec = [&](auto nc_const)
        {
            constexpr int    NC = decltype(nc_const)::value;
            constexpr size_t a  = alignof(cuda::MakeType<elem_type, NC>);
            auto             ok = [](const Ptr2dNHWC<elem_type> &p)
            {
                return ((reinterpret_cast<std::uintptr_t>(p.data) | static_cast<std::uintptr_t>(p.imgStride)
                         | static_cast<std::uintptr_t>(p.rowStride))
                        & (a - 1))
                    == 0;
            };
            if (ok(src_p) && ok(inter_p) && ok(dst_p))
                launch(nc_const, src_p, inter_p, dst_p);
            else
                launch(std::integral_constant<int, 0>{}, src_p, inter_p, dst_p);
        };
        switch (input_shape.C)
        {
        case 1:
            try_vec(std::integral_constant<int, 1>{});
            break;
        case 2:
            try_vec(std::integral_constant<int, 2>{});
            break;
        case 3:
            try_vec(std::integral_constant<int, 3>{});
            break;
        case 4:
            try_vec(std::integral_constant<int, 4>{});
            break;
        default:
            launch(std::integral_constant<int, 0>{}, src_p, inter_p, dst_p);
            break;
        }
    }
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename Filter>
void pillow_resize_filter(const TensorDataAccessStridedImagePlanar &inData,
                          const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                          NVCVInterpolationType interpolation, bool planar, cudaStream_t stream)
{
    cuda_op::DataType data_type = GetLegacyDataType(inData.dtype());
    switch (data_type)
    {
    case kCV_8U:
        pillow_resize_v2<Filter, unsigned char>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    case kCV_8S:
        pillow_resize_v2<Filter, signed char>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case kCV_16U:
        pillow_resize_v2<Filter, std::uint16_t>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    case kCV_16S:
        pillow_resize_v2<Filter, std::int16_t>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case kCV_32S:
        pillow_resize_v2<Filter, int>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case kCV_32F:
        pillow_resize_v2<Filter, float>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    default:
        break;
    }
}

WorkspaceRequirements PillowResize::getWorkspaceRequirements(DataShape max_input_shape, DataShape max_output_shape,
                                                             DataType max_data_type)
{
    int    max_support = 1; //3
    size_t size
        = std::ceil(
              max_output_shape.H
                  * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
                     + 2 * sizeof(int))
              + max_output_shape.W
                    * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                       + 2 * sizeof(int)))
        + static_cast<size_t>(max_input_shape.N) * max_input_shape.C * max_input_shape.H * max_output_shape.W
              * DataSize(max_data_type)
        + 16; // padding to 16-byte-align the intermediate buffer (vector loads/stores)
    WorkspaceRequirements req{};
    req.cudaMem = {size, 256};
    return req;
}

ErrorCode PillowResize::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                              const NVCVInterpolationType interpolation, cudaStream_t stream, const Workspace &ws)
{
    DataFormat format        = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (ws.cudaMem.ready != nullptr)
        checkCudaErrors(cudaStreamWaitEvent(stream, ws.cudaMem.ready));

    void *gpu_workspace = ws.cudaMem.data;

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
    const bool planar = (format == kNCHW || format == kCHW);

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

    if (channels > 4)
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

    // The kernels compute addresses as 32-bit products of sample/row strides (Ptr2dNHWC/Ptr2dNCHW),
    // so any tensor -- input, output, or the horizontally-resized intermediate -- whose byte extent
    // exceeds INT32_MAX overflows the addressing. Reject those instead of corrupting memory.
    {
        constexpr int64_t kMaxByteExtent = std::numeric_limits<int32_t>::max();
        const int64_t     interExtent    = static_cast<int64_t>(input_shape.N) * input_shape.C * input_shape.H
                                  * outAccess->numCols() * DataSize(data_type);
        const int64_t maxExtent = std::max(
            {inAccess->numSamples() * inAccess->sampleStride(), outAccess->numSamples() * outAccess->sampleStride(),
             inAccess->numRows() * inAccess->rowStride(), outAccess->numRows() * outAccess->rowStride(), interExtent});
        if (maxExtent > kMaxByteExtent)
        {
            LOG_ERROR("Tensor byte extent " << maxExtent << " exceeds the 32-bit addressing limit " << kMaxByteExtent
                                            << "; split the batch into smaller submissions");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
    }

    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
        pillow_resize_filter<BilinearFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, planar, stream);
        break;
    case NVCV_INTERP_CUBIC:
        pillow_resize_filter<BicubicFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, planar, stream);
        break;
    case NVCV_INTERP_LANCZOS:
        pillow_resize_filter<LanczosFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, planar, stream);
        break;
    case NVCV_INTERP_BOX:
        pillow_resize_filter<BoxFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, planar, stream);
        break;
    case NVCV_INTERP_HAMMING:
        pillow_resize_filter<HammingFilter>(*inAccess, *outAccess, gpu_workspace, interpolation, planar, stream);
        break;
    default:
        LOG_ERROR("Unsupported interpolation method " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
        break;
    }

    if (ws.cudaMem.ready != nullptr)
        checkCudaErrors(cudaEventRecord(ws.cudaMem.ready, stream));

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
