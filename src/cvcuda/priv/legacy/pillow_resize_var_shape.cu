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

#include <nvcv/ImageBatch.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

#define BLOCK           32
#define SHARE_MEM_LIMIT 4096

namespace nvcv::legacy::cuda_op {

static constexpr unsigned int precision_bits_var_shape = 32 - 8 - 2;

namespace {

template<class Filter>
__global__ void _precomputeCoeffsVarShape(int *in_size_batch, int *in0_batch, work_type *scale_batch,
                                          work_type *filterscale_batch, work_type *support_batch, int *out_size_batch,
                                          int *k_size_batch, Filter filterp, int *bounds_out_batch,
                                          int *bound_out_offset, work_type *kk_out_batch, int *kk_out_offset,
                                          bool normalize_coeff, bool use_share_mem)
{
    const int xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx   = get_batch_idx();
    int        in_size     = in_size_batch[batch_idx];
    int        in0         = in0_batch[batch_idx];
    work_type  scale       = scale_batch[batch_idx];
    work_type  filterscale = filterscale_batch[batch_idx];
    work_type  support     = support_batch[batch_idx];
    int        out_size    = out_size_batch[batch_idx];
    int        k_size      = k_size_batch[batch_idx];
    int       *bounds_out  = bounds_out_batch + bound_out_offset[batch_idx];
    work_type *kk_out      = kk_out_batch + kk_out_offset[batch_idx];

    PillowPrecomputeCoeffs<Filter>(xx, local_id, x_offset, in_size, in0, scale, filterscale, support, out_size, k_size,
                                   filterp, bounds_out, kk_out, normalize_coeff, use_share_mem,
                                   precision_bits_var_shape);
}

template<int NC, class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape(const Ptr2dVarShapeNHWC<T1> src, Ptr2dNHWC<T2> dst, Filter &filterp,
                                          int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch,
                                          int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                          int *v_bounds_batch, int *v_bounds_offset, work_type *v_kk_batch,
                                          int *v_kk_offset, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = get_batch_idx();
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);
    int in_height = src.at_rows(batch_idx);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_h), x_offset, h_ksize, out_width,
                                           blockDim.x, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int xmin = h_bounds[dst_x * 2];
        int xmax = h_bounds[dst_x * 2 + 1];

        work_type *h_k = &h_k_tmp[local_x * h_ksize];

        // The precompute step clamps xmin and the tap count to the source width, so this support
        // stays in one row. The max-sized intermediate still needs the per-image height guard, but
        // avoiding quotient/remainder work for every tap is material inside the hot loop.

        // Interleaved channel-vectorization (NC > 0): load the contiguous NC-channel input pixel as one
        // MakeType<T1,NC> vector and write the NC-channel float intermediate as one vector. The per-image
        // input alignment is checked once (uniform across this block's image, blockIdx.z == image) and the
        // scalar path runs when it is not vector-aligned. The intermediate is 16-byte aligned by the host.
        bool did_vec = false;
        if constexpr (NC > 0)
        {
            using vin_t               = cuda::MakeType<T1, NC>;
            using vout_t              = cuda::MakeType<T2, NC>;
            using accum_t             = cuda::MakeType<work_type, NC>;
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0));
            const std::uintptr_t step
                = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0)) - base; // row stride in bytes
            if (((base | step) & (alignof(vin_t) - 1)) == 0)
            {
                accum_t h_ss = cuda::SetAll<accum_t>(work_type{0});
                if (dst_y < in_height)
                {
                    const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, 0));
                    for (int x = 0; x < xmax; ++x)
                    {
                        h_ss = h_ss + h_k[x] * cuda::StaticCast<work_type>(*reinterpret_cast<const vin_t *>(p));
                        p += sizeof(vin_t);
                    }
                }
                vout_t *out = reinterpret_cast<vout_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
                if (round_up)
                {
                    work_type *acc = reinterpret_cast<work_type *>(&h_ss);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        reinterpret_cast<T2 *>(out)[c] = cuda::SaturateCast<T2>(std::round(acc[c]));
                }
                else
                {
                    *out = cuda::SaturateCast<vout_t>(h_ss);
                }
                did_vec = true;
            }
        }
        if (!did_vec)
        {
            for (int c = 0; c < src.nch; ++c)
            {
                work_type h_ss = 0.0;
                if (dst_y < in_height)
                {
                    const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, c));
                    for (int x = 0; x < xmax; ++x)
                    {
                        h_ss = h_ss + *reinterpret_cast<const T1 *>(p) * h_k[x];
                        p += src.nch * sizeof(T1);
                    }
                }
                if (round_up)
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(std::round(h_ss));
                else
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(h_ss);
            }
        }
    }
}

// Interleaved var-shape horizontal pass computing two adjacent output pixels per thread (see
// horizontal_pass_paired): the shared tap-window segment is loaded once and accumulated into both
// outputs with fma count and ascending per-output tap order identical to the per-pixel kernel.
// The per-image input alignment is checked once (uniform across this block's image); unaligned
// images take a per-output scalar channel loop.
template<int NC, class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape_paired(const Ptr2dVarShapeNHWC<T1> src, Ptr2dNHWC<T2> dst, Filter &filterp,
                                                 int *h_ksize_batch, int *out_cols_batch, int *h_bounds_batch,
                                                 int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                                 work_type init_buffer, bool round_up, bool use_share_mem)
{
    static_assert(NC > 0, "paired var-shape horizontal pass is the interleaved channel-vectorized path");
    const int pair0    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x * 2;

    const int  batch_idx = get_batch_idx();
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // The intermediate is max-sized and uniform; the pair guards need this image's true output
    // width so the second output of the last pair never indexes the next image's bounds.
    int out_height = dst.at_rows(batch_idx), out_width = out_cols_batch[batch_idx];
    int in_height = src.at_rows(batch_idx);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_hp2[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_hp2), x_offset, h_ksize,
                                           out_width, blockDim.x * 2, use_share_mem);

    if (pair0 < out_width && dst_y < out_height)
    {
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

        using vin_t   = cuda::MakeType<T1, NC>;
        using vout_t  = cuda::MakeType<T2, NC>;
        using accum_t = cuda::MakeType<work_type, NC>;

        const std::uintptr_t ibase = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0));
        const std::uintptr_t istep = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0)) - ibase;
        if (((ibase | istep) & (alignof(vin_t) - 1)) == 0)
        {
            accum_t s0 = cuda::SetAll<accum_t>(work_type{0});
            accum_t s1 = cuda::SetAll<accum_t>(work_type{0});
            if (dst_y < in_height)
            {
                const std::byte *base   = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, 0));
                const int        preEnd = min(xmin1, end0);
                for (int x = xmin0; x < preEnd; ++x)
                {
                    s0 = s0
                       + k0[x - xmin0]
                             * cuda::StaticCast<work_type>(
                                 *reinterpret_cast<const vin_t *>(base + static_cast<size_t>(x) * sizeof(vin_t)));
                }
                for (int x = xmin1; x < end0; ++x)
                {
                    const accum_t px = cuda::StaticCast<work_type>(
                        *reinterpret_cast<const vin_t *>(base + static_cast<size_t>(x) * sizeof(vin_t)));
                    s0 = s0 + k0[x - xmin0] * px;
                    s1 = s1 + k1[x - xmin1] * px;
                }
                const int sufBegin = max(xmin1, end0);
                for (int x = sufBegin; x < end1; ++x)
                {
                    s1 = s1
                       + k1[x - xmin1]
                             * cuda::StaticCast<work_type>(
                                 *reinterpret_cast<const vin_t *>(base + static_cast<size_t>(x) * sizeof(vin_t)));
                }
            }
            vout_t *out0 = reinterpret_cast<vout_t *>(dst.ptr(batch_idx, dst_y, pair0, 0));
            if (round_up)
            {
                work_type *acc0 = reinterpret_cast<work_type *>(&s0);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    reinterpret_cast<T2 *>(out0)[c] = cuda::SaturateCast<T2>(std::round(acc0[c]));
            }
            else
            {
                *out0 = cuda::SaturateCast<vout_t>(s0);
            }
            if (has1)
            {
                vout_t *out1 = reinterpret_cast<vout_t *>(dst.ptr(batch_idx, dst_y, pair0 + 1, 0));
                if (round_up)
                {
                    work_type *acc1 = reinterpret_cast<work_type *>(&s1);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        reinterpret_cast<T2 *>(out1)[c] = cuda::SaturateCast<T2>(std::round(acc1[c]));
                }
                else
                {
                    *out1 = cuda::SaturateCast<vout_t>(s1);
                }
            }
        }
        else
        {
            for (int o = 0; o < (has1 ? 2 : 1); ++o)
            {
                const int  dst_x = pair0 + o;
                const int  xmin  = o ? xmin1 : xmin0;
                const int  xmax  = o ? xmax1 : xmax0;
                work_type *hk    = o ? k1 : k0;
                for (int c = 0; c < src.nch; ++c)
                {
                    work_type h_ss = 0.0;
                    if (dst_y < in_height)
                    {
                        const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, c));
                        for (int x = 0; x < xmax; ++x)
                        {
                            h_ss = h_ss + *reinterpret_cast<const T1 *>(p) * hk[x];
                            p += src.nch * sizeof(T1);
                        }
                    }
                    if (round_up)
                        *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(std::round(h_ss));
                    else
                        *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(h_ss);
                }
            }
        }
    }
}

template<int NC, class T1, class T2, class Filter>
__global__ void vertical_pass_var_shape(const Ptr2dNHWC<T1> src, Ptr2dVarShapeNHWC<T2> dst, Filter &filterp,
                                        int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch,
                                        int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                        int *v_bounds_batch, int *v_bounds_offset, work_type *v_kk_batch,
                                        int *v_kk_offset, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y  = threadIdx.y;
    const int y_offset = blockIdx.y * blockDim.y;

    const int  batch_idx = get_batch_idx();
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_v[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_v), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        // The intermediate has output width and precompute clamps ymin plus the tap count to its
        // height, so neither coordinate can wrap. Keep the hot loop to direct 2-D addressing.

        // Interleaved channel-vectorization: load the NC-channel float intermediate pixel as one vector
        // (intermediate is 16-byte aligned by the host, always vector-safe) and write the NC-channel output
        // pixel as one vector. The per-image OUTPUT alignment is checked once (uniform across this block's
        // image); the scalar path runs when it is not vector-aligned.
        bool did_vec = false;
        if constexpr (NC > 0)
        {
            using vin_t                = cuda::MakeType<T1, NC>;
            using vout_t               = cuda::MakeType<T2, NC>;
            using accum_t              = cuda::MakeType<work_type, NC>;
            const std::uintptr_t obase = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0));
            const std::uintptr_t ostep
                = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0)) - obase; // row stride in bytes
            if (((obase | ostep) & (alignof(vout_t) - 1)) == 0)
            {
                const std::byte *p  = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x, 0));
                accum_t          ss = cuda::SetAll<accum_t>(init_buffer);
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + v_k[y] * cuda::StaticCast<work_type>(*reinterpret_cast<const vin_t *>(p));
                    p += src.rowStride;
                }
                vout_t *out = reinterpret_cast<vout_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
                if (round_up)
                {
                    work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        reinterpret_cast<T2 *>(out)[c] = cuda::SaturateCast<T2>(std::round(acc[c]));
                }
                else
                {
                    *out = cuda::SaturateCast<vout_t>(ss);
                }
                did_vec = true;
            }
        }
        if (!did_vec)
        {
            for (int c = 0; c < src.ch; ++c)
            {
                const std::byte *p  = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x, c));
                work_type        ss = init_buffer;
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + *reinterpret_cast<const T1 *>(p) * v_k[y];
                    p += src.rowStride;
                }

                if (round_up)
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(std::round(ss));
                else
                    *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T2>(ss);
            }
        }
    }
}

// Interleaved var-shape vertical pass vectorized along the flattened row (see vertical_pass_vec in
// pillow_resize.cu): all cols*nch scalars of an output row share v_k, so each thread accumulates VEC
// consecutive scalars with one wide load per tap. The float intermediate rows are 16-byte aligned by
// the host; the per-image OUTPUT alignment is checked once per thread (uniform across this block's
// image) and the row tail or an unaligned image falls back to a per-element loop with identical
// per-element arithmetic. Weights are read straight from L1: threads of a warp share dst_y, so the
// per-tap loads are broadcasts and shared-memory staging would only add a block barrier.
template<int VEC, class T1, class T2, class Filter>
__global__ void vertical_pass_var_shape_vec(const Ptr2dNHWC<T1> src, Ptr2dVarShapeNHWC<T2> dst, Filter &filterp,
                                            int *v_ksize_batch, int *v_bounds_batch, int *v_bounds_offset,
                                            work_type *v_kk_batch, int *v_kk_offset, work_type init_buffer,
                                            bool round_up)
{
    const int elem0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int  batch_idx = get_batch_idx();
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx);
    int row_elems  = dst.at_cols(batch_idx) * dst.nch;

    if (elem0 < row_elems && dst_y < out_height)
    {
        const int ymin = v_bounds[dst_y * 2];
        const int ymax = v_bounds[dst_y * 2 + 1];

        const work_type *v_k = &v_kk[dst_y * v_ksize];

        const std::byte *src_row
            = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, 0, 0)) + elem0 * sizeof(T1);
        T2 *dst_row = dst.ptr(batch_idx, dst_y, 0, 0) + elem0;

        using vin_t                 = cuda::MakeType<T1, VEC>;
        using vout_t                = cuda::MakeType<T2, VEC>;
        const std::uintptr_t obase  = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0));
        const std::uintptr_t ostep  = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0)) - obase;
        const bool           vec_ok = elem0 + VEC <= row_elems
                         && (((obase | ostep) & (alignof(vout_t) - 1))
                             | (static_cast<std::uintptr_t>(src.rowStride) & (alignof(vin_t) - 1)))
                                == 0;

        if (vec_ok)
        {
            using accum_t = cuda::MakeType<work_type, VEC>;

            accum_t ss = cuda::SetAll<accum_t>(init_buffer);
            for (int y = 0; y < ymax; ++y)
            {
                ss = ss + v_k[y] * cuda::StaticCast<work_type>(*reinterpret_cast<const vin_t *>(src_row));
                src_row += src.rowStride;
            }
            if (round_up)
            {
                work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
                for (int i = 0; i < VEC; ++i) dst_row[i] = cuda::SaturateCast<T2>(std::round(acc[i]));
            }
            else
            {
                *reinterpret_cast<vout_t *>(dst_row) = cuda::SaturateCast<vout_t>(ss);
            }
        }
        else
        {
            for (int i = 0; i < VEC && elem0 + i < row_elems; ++i)
            {
                const std::byte *p  = src_row + i * sizeof(T1);
                work_type        ss = init_buffer;
                for (int y = 0; y < ymax; ++y)
                {
                    ss = ss + *reinterpret_cast<const T1 *>(p) * v_k[y];
                    p += src.rowStride;
                }
                dst_row[i] = round_up ? cuda::SaturateCast<T2>(std::round(ss)) : cuda::SaturateCast<T2>(ss);
            }
        }
    }
}

// Fused single-pass var-shape resize for the interleaved FLOAT path. Mathematically identical,
// op-for-op, to horizontal_pass_var_shape + vertical_pass_var_shape: the var-shape intermediate is
// already work_type (float), so for each contributing source row the inner horizontal sum reproduces
// exactly what the float intermediate would hold, and accumulating v_k[y] * inner yields the same
// float result -- but the intermediate stays in registers, dropping its full-frame DRAM round-trip.
// Used only for small filter kernels (host-side max-k gate): the inner sum is recomputed v_ksize times,
// so large supports would go compute-bound and regress, and keep the separable passes. The per-image
// vector path is gated on input AND output alignment (vec and scalar give identical float results).
template<int NC, class T, class Filter>
__global__ void fused_pass_var_shape(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, Filter &filterp,
                                     int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch, int *h_bounds_offset,
                                     work_type *h_kk_batch, int *h_kk_offset, int *v_bounds_batch, int *v_bounds_offset,
                                     work_type *v_kk_batch, int *v_kk_offset, work_type init_buffer, bool round_up,
                                     bool use_share_mem)
{
    static_assert(NC > 0, "fused_pass_var_shape is the interleaved channel-vectorized path");
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    const int  h_ksize  = h_ksize_batch[batch_idx];
    const int  v_ksize  = v_ksize_batch[batch_idx];
    int       *h_bounds = h_bounds_batch + h_bounds_offset[batch_idx];
    int       *v_bounds = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *h_kk     = h_kk_batch + h_kk_offset[batch_idx];
    work_type *v_kk     = v_kk_batch + v_kk_offset[batch_idx];

    const int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);

    if (dst_x >= out_width || dst_y >= out_height)
        return;

    const int  xmin = h_bounds[dst_x * 2];
    const int  xmax = h_bounds[dst_x * 2 + 1];
    const int  ymin = v_bounds[dst_y * 2];
    const int  ymax = v_bounds[dst_y * 2 + 1];
    work_type *h_k  = &h_kk[dst_x * h_ksize];
    work_type *v_k  = &v_kk[dst_y * v_ksize];

    // Precomputed bounds clamp both support windows to the source image, so direct coordinates are
    // equivalent to the old quotient/remainder reconstruction and avoid divisions in the nested taps.

    using vin_t   = cuda::MakeType<T, NC>;
    using accum_t = cuda::MakeType<work_type, NC>;

    // Per-image alignment for the vectorized load (input) and store (output); uniform across the image.
    const std::uintptr_t ibase   = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0));
    const std::uintptr_t istep   = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0)) - ibase;
    const std::uintptr_t obase   = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0));
    const std::uintptr_t ostep   = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0)) - obase;
    const bool           aligned = (((ibase | istep | obase | ostep) & (alignof(vin_t) - 1)) == 0);

    if (aligned)
    {
        accum_t ss = cuda::SetAll<accum_t>(init_buffer);
        for (int y = 0; y < ymax; ++y)
        {
            // inner == the float intermediate at (ymin + y, dst_x): identical taps, order, and bounds.
            const std::byte *p     = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin + y, xmin, 0));
            accum_t          inner = cuda::SetAll<accum_t>(work_type{0});
            for (int x = 0; x < xmax; ++x)
            {
                inner = inner + h_k[x] * cuda::StaticCast<work_type>(*reinterpret_cast<const vin_t *>(p));
                p += sizeof(vin_t);
            }
            ss = ss + v_k[y] * inner;
        }
        vin_t *out = reinterpret_cast<vin_t *>(dst.ptr(batch_idx, dst_y, dst_x, 0));
        if (round_up)
        {
            work_type *acc = reinterpret_cast<work_type *>(&ss);
#pragma unroll
            for (int c = 0; c < NC; ++c) reinterpret_cast<T *>(out)[c] = cuda::SaturateCast<T>(std::round(acc[c]));
        }
        else
        {
            *out = cuda::SaturateCast<vin_t>(ss);
        }
    }
    else
    {
        for (int c = 0; c < src.nch; ++c)
        {
            work_type ss = init_buffer;
            for (int y = 0; y < ymax; ++y)
            {
                const std::byte *p     = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin + y, xmin, c));
                work_type        inner = 0;
                for (int x = 0; x < xmax; ++x)
                {
                    inner = inner + *reinterpret_cast<const T *>(p) * h_k[x];
                    p += src.nch * sizeof(T);
                }
                ss = ss + v_k[y] * inner;
            }
            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(std::round(ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, c) = cuda::SaturateCast<T>(ss);
        }
    }
}

// Planar (NCHW / CHW) var-shape passes. PillowResize resizes each channel independently. The external
// src/dst are per-plane wraps indexed (image, plane, y, x); the internal scratch buffer stays
// interleaved (indexed by plane in its channel slot). One grid-z slice is launched per image
// (blockIdx.z = image); the C channel planes are looped inside the thread so the per-output-pixel
// bounds / weight setup -- shared by all planes of an image -- is computed once and reused.
template<class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape_planar(const cuda::ImageBatchVarShapeWrap<const T1> src, Ptr2dNHWC<T2> dst,
                                                 Filter &filterp, int *h_ksize_batch, int channels, int *h_bounds_batch,
                                                 int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                                 work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = get_batch_idx();
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // All channel planes of an image share its size, so the per-image setup (bounds, coeffs, in-size)
    // is computed once and reused across the channels looped below.
    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);
    int in_height = src.height(batch_idx, 0);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_h), x_offset, h_ksize, out_width,
                                           blockDim.x, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int xmin = h_bounds[dst_x * 2];
        int xmax = h_bounds[dst_x * 2 + 1];

        work_type *h_k = &h_k_tmp[local_x * h_ksize];
        // The precompute step clamps xmin and the tap count to the source width, so the support stays
        // inside row dst_y of every plane: walk a hoisted row pointer instead of re-deriving the
        // coordinate (and its quotient/remainder) for every tap. The max-sized launch still needs the
        // per-image height guard: shorter images store zero rows, as the interleaved kernel does.
        const bool validRow = dst_y < in_height;
        for (int plane = 0; plane < channels; ++plane)
        {
            work_type h_ss = 0.0;
            if (validRow)
            {
                const T1 *row = src.ptr(batch_idx, plane, dst_y, xmin);
                for (int x = 0; x < xmax; ++x)
                {
                    h_ss = h_ss + row[x] * h_k[x];
                }
            }
            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, plane) = cuda::SaturateCast<T2>(std::round(h_ss));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, plane) = cuda::SaturateCast<T2>(h_ss);
        }
    }
}

template<int PC, class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape_planar_channels(const cuda::ImageBatchVarShapeWrap<const T1> src,
                                                          Ptr2dNHWC<T2> dst, Filter &filterp, int *h_ksize_batch,
                                                          int *h_bounds_batch, int *h_bounds_offset,
                                                          work_type *h_kk_batch, int *h_kk_offset,
                                                          work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = get_batch_idx();
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    int out_height = dst.at_rows(batch_idx), out_width = dst.at_cols(batch_idx);
    int in_height = src.height(batch_idx, 0);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_h), x_offset, h_ksize, out_width,
                                           blockDim.x, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int xmin = h_bounds[dst_x * 2];
        int xmax = h_bounds[dst_x * 2 + 1];

        work_type *h_k = &h_k_tmp[local_x * h_ksize];
        work_type  h_ss[PC];
        // The precompute step clamps xmin and the tap count to the source width, so the support stays
        // inside row dst_y of every plane: walk hoisted per-plane row pointers instead of re-deriving
        // the coordinate (and its quotient/remainder) for every tap.
        const T1  *row[PC];

#pragma unroll
        for (int plane = 0; plane < PC; ++plane)
        {
            h_ss[plane] = 0.0;
        }

        // The max-sized launch still needs the per-image height guard: shorter images store zero
        // rows, as the interleaved kernel does.
        if (dst_y < in_height)
        {
#pragma unroll
            for (int plane = 0; plane < PC; ++plane)
            {
                row[plane] = src.ptr(batch_idx, plane, dst_y, xmin);
            }

            for (int x = 0; x < xmax; ++x)
            {
                const work_type k = h_k[x];
#pragma unroll
                for (int plane = 0; plane < PC; ++plane)
                {
                    h_ss[plane] = h_ss[plane] + row[plane][x] * k;
                }
            }
        }

#pragma unroll
        for (int plane = 0; plane < PC; ++plane)
        {
            if (round_up)
                *dst.ptr(batch_idx, dst_y, dst_x, plane) = cuda::SaturateCast<T2>(std::round(h_ss[plane]));
            else
                *dst.ptr(batch_idx, dst_y, dst_x, plane) = cuda::SaturateCast<T2>(h_ss[plane]);
        }
    }
}

template<class T1, class T2, class Filter>
__global__ void vertical_pass_var_shape_planar(const Ptr2dNHWC<T1> src, cuda::ImageBatchVarShapeWrap<T2> dst,
                                               Filter &filterp, int *v_ksize_batch, int channels, int *v_bounds_batch,
                                               int *v_bounds_offset, work_type *v_kk_batch, int *v_kk_offset,
                                               work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y  = threadIdx.y;
    const int y_offset = blockIdx.y * blockDim.y;

    const int  batch_idx = get_batch_idx();
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    // All channel planes of an image share its size; setup is computed once and reused across planes.
    int out_height = dst.height(batch_idx, 0), out_width = dst.width(batch_idx, 0);

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_v[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_v), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (dst_x < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];
        // The precompute step clamps ymin and the tap count to the intermediate height and dst_x is
        // bounded by its width, so the support stays inside column dst_x: walk a hoisted column
        // pointer by row stride instead of re-deriving the coordinate for every tap.
        for (int plane = 0; plane < channels; ++plane)
        {
            const std::byte *p  = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x, plane));
            work_type        ss = init_buffer;
            for (int y = 0; y < ymax; ++y)
            {
                ss = ss + *reinterpret_cast<const T1 *>(p) * v_k[y];
                p += src.rowStride;
            }

            if (round_up)
                *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::SaturateCast<T2>(std::round(ss));
            else
                *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::SaturateCast<T2>(ss);
        }
    }
}

template<typename Filter, typename elem_type, bool kPlanar>
void pillow_resize_var_shape(const ImageBatchVarShape &inDataBase, const ImageBatchVarShape &outDataBase,
                             const Workspace &ws, bool normalize_coeff, work_type init_buffer, bool round_up,
                             cudaStream_t stream)
{
    if (ws.hostMem.ready != nullptr)
        checkCudaErrors(cudaEventSynchronize(ws.hostMem.ready));

    if (ws.cudaMem.ready != nullptr)
        checkCudaErrors(cudaStreamWaitEvent(stream, ws.cudaMem.ready));

    void *cpu_workspace = ws.hostMem.data;
    void *gpu_workspace = ws.cudaMem.data;

    auto inDataPtr = inDataBase.exportData<ImageBatchVarShapeDataStridedCuda>(stream);
    if (!inDataPtr)
    {
        throw LegacyImageBatchExportError("Something wrong happened during conversion of type...!!!");
    }

    auto outDataPtr = outDataBase.exportData<ImageBatchVarShapeDataStridedCuda>(stream);
    if (!outDataPtr)
    {
        throw LegacyImageBatchExportError("Something wrong happened during conversion of type...!!!");
    }

    const ImageBatchVarShapeDataStridedCuda &inData  = *inDataPtr;
    const ImageBatchVarShapeDataStridedCuda &outData = *outDataPtr;

    int channels = inData.uniqueFormat().numChannels();
    int batch    = inData.numImages();

    Filter filterp;

    Size2D outMaxSize = outData.maxSize();
    Size2D inMaxSize  = inData.maxSize();

    int max_height = outMaxSize.h, max_width = outMaxSize.w;
    int max_input_height = inMaxSize.h;

    const void **inputs              = (const void **)cpu_workspace;
    void       **outputs             = (void **)((char *)inputs + sizeof(void *) * batch);
    void       **hori                = (void **)((char *)outputs + sizeof(void *) * batch);
    int         *rows                = (int *)((char *)hori + sizeof(void *) * batch);
    int         *cols                = (int *)((char *)rows + sizeof(int) * batch);
    int         *out_rows            = (int *)((char *)cols + sizeof(int) * batch);
    int         *out_cols            = (int *)((char *)out_rows + sizeof(int) * batch);
    int         *roi_x               = (int *)((char *)out_cols + sizeof(int) * batch);
    int         *roi_y               = (int *)((char *)roi_x + sizeof(int) * batch);
    work_type   *h_scale_batch       = (work_type *)((char *)roi_y + sizeof(int) * batch);
    work_type   *v_scale_batch       = (work_type *)((char *)h_scale_batch + sizeof(work_type) * batch);
    work_type   *h_filterscale_batch = (work_type *)((char *)v_scale_batch + sizeof(work_type) * batch);
    work_type   *v_filterscale_batch = (work_type *)((char *)h_filterscale_batch + sizeof(work_type) * batch);
    work_type   *h_support_batch     = (work_type *)((char *)v_filterscale_batch + sizeof(work_type) * batch);
    work_type   *v_support_batch     = (work_type *)((char *)h_support_batch + sizeof(work_type) * batch);
    int         *h_k_size_batch      = (int *)((char *)v_support_batch + sizeof(work_type) * batch);
    int         *v_k_size_batch      = (int *)((char *)h_k_size_batch + sizeof(int) * batch);
    int         *h_bounds_offset     = (int *)((char *)v_k_size_batch + sizeof(int) * batch);
    int         *v_bounds_offset     = (int *)((char *)h_bounds_offset + sizeof(int) * batch);
    int         *h_kk_offset         = (int *)((char *)v_bounds_offset + sizeof(int) * batch);
    int         *v_kk_offset         = (int *)((char *)h_kk_offset + sizeof(int) * batch);

    int  h_kk_total = 0, v_kk_total = 0;
    int  max_h_k_size = 0, max_v_k_size = 0;
    int  h_bounds_total = 0, v_bounds_total = 0;
    bool all_upscale   = true;
    bool all_hcontract = true;

    for (int i = 0; i < batch; i++)
    {
        rows[i]     = inDataBase[i].size().h;
        cols[i]     = inDataBase[i].size().w;
        out_rows[i] = outDataBase[i].size().h;
        out_cols[i] = outDataBase[i].size().w;

        roi_x[i] = 0;
        roi_y[i] = 0;

        work_type h_scale = 0, v_scale = 0;
        work_type h_filterscale = 0, v_filterscale = 0;
        h_filterscale = h_scale = static_cast<work_type>(inDataBase[i].size().w) / out_cols[i];
        v_filterscale = v_scale = static_cast<work_type>(inDataBase[i].size().h) / out_rows[i];
        all_upscale             = all_upscale && h_scale < work_type(1) && v_scale < work_type(1);
        all_hcontract           = all_hcontract && h_scale > work_type(1);
        if (h_filterscale < 1.0)
        {
            h_filterscale = 1.0;
        }
        if (v_filterscale < 1.0)
        {
            v_filterscale = 1.0;
        }
        h_scale_batch[i]       = h_scale;
        v_scale_batch[i]       = v_scale;
        h_filterscale_batch[i] = h_filterscale;
        v_filterscale_batch[i] = v_filterscale;

        // Determine support size (length of resampling filter).
        work_type h_support = filterp.support() * h_filterscale;
        work_type v_support = filterp.support() * v_filterscale;
        // Maximum number of coeffs.
        int       h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
        int       v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;
        h_support_batch[i] = h_support;
        v_support_batch[i] = v_support;
        h_k_size_batch[i]  = h_k_size;
        v_k_size_batch[i]  = v_k_size;
        h_kk_offset[i]     = h_kk_total;
        v_kk_offset[i]     = v_kk_total;
        h_kk_total += out_cols[i] * h_k_size;
        v_kk_total += out_rows[i] * v_k_size;
        h_bounds_offset[i] = h_bounds_total;
        v_bounds_offset[i] = v_bounds_total;
        h_bounds_total += out_cols[i] * 2;
        v_bounds_total += out_rows[i] * 2;

        if (h_k_size > max_h_k_size)
            max_h_k_size = h_k_size;
        if (v_k_size > max_v_k_size)
            max_v_k_size = v_k_size;
    }

    const void **inputs_gpu              = (const void **)gpu_workspace;
    void       **outputs_gpu             = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    void       **hori_gpu                = (void **)((char *)outputs_gpu + sizeof(void *) * batch);
    int         *rows_gpu                = (int *)((char *)hori_gpu + sizeof(void *) * batch);
    int         *cols_gpu                = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int         *out_rows_gpu            = (int *)((char *)cols_gpu + sizeof(int) * batch);
    int         *out_cols_gpu            = (int *)((char *)out_rows_gpu + sizeof(int) * batch);
    int         *roi_x_gpu               = (int *)((char *)out_cols_gpu + sizeof(int) * batch);
    int         *roi_y_gpu               = (int *)((char *)roi_x_gpu + sizeof(int) * batch);
    work_type   *h_scale_batch_gpu       = (work_type *)((char *)roi_y_gpu + sizeof(int) * batch);
    work_type   *v_scale_batch_gpu       = (work_type *)((char *)h_scale_batch_gpu + sizeof(work_type) * batch);
    work_type   *h_filterscale_batch_gpu = (work_type *)((char *)v_scale_batch_gpu + sizeof(work_type) * batch);
    work_type   *v_filterscale_batch_gpu = (work_type *)((char *)h_filterscale_batch_gpu + sizeof(work_type) * batch);
    work_type   *h_support_batch_gpu     = (work_type *)((char *)v_filterscale_batch_gpu + sizeof(work_type) * batch);
    work_type   *v_support_batch_gpu     = (work_type *)((char *)h_support_batch_gpu + sizeof(work_type) * batch);
    int         *h_k_size_batch_gpu      = (int *)((char *)v_support_batch_gpu + sizeof(work_type) * batch);
    int         *v_k_size_batch_gpu      = (int *)((char *)h_k_size_batch_gpu + sizeof(int) * batch);
    int         *h_bounds_offset_gpu     = (int *)((char *)v_k_size_batch_gpu + sizeof(int) * batch);
    int         *v_bounds_offset_gpu     = (int *)((char *)h_bounds_offset_gpu + sizeof(int) * batch);
    int         *h_kk_offset_gpu         = (int *)((char *)v_bounds_offset_gpu + sizeof(int) * batch);
    int         *v_kk_offset_gpu         = (int *)((char *)h_kk_offset_gpu + sizeof(int) * batch);

    work_type *h_kk_batch_gpu     = (work_type *)((char *)v_kk_offset_gpu + sizeof(int) * batch);
    work_type *v_kk_batch_gpu     = (work_type *)((char *)h_kk_batch_gpu + sizeof(work_type) * h_kk_total);
    int       *h_bounds_batch_gpu = (int *)((char *)v_kk_batch_gpu + sizeof(work_type) * v_kk_total);
    int       *v_bounds_batch_gpu = (int *)((char *)h_bounds_batch_gpu + sizeof(int) * h_bounds_total);

    const size_t metadata_buffer_size = (sizeof(void *) * 3 + sizeof(int) * 12 + sizeof(work_type) * 6) * batch;
    const size_t current_buffer_size  = metadata_buffer_size + sizeof(work_type) * (h_kk_total + v_kk_total)
                                     + sizeof(int) * (h_bounds_total + v_bounds_total);

    // Buffer for storing results from the horizontal pass. The intermediate is elem_type: integer
    // dtypes quantize between passes exactly like the tensor path and Pillow's 8bpc pipeline (and move
    // a quarter of the float bytes); float keeps the full-precision work_type intermediate. Round up to
    // 16 bytes so the intermediate can be read/written with up-to-16-byte vector accesses
    // (getWorkspaceRequirements sizes the region for the float worst case and reserves +16).
    std::uintptr_t hori_raw      = reinterpret_cast<std::uintptr_t>((char *)gpu_workspace + current_buffer_size);
    void          *hori_gpu_data = reinterpret_cast<void *>((hori_raw + 15) & ~std::uintptr_t(15));

    // The large coefficient/bounds regions are produced by _precomputeCoeffsVarShape below; only the
    // small per-sample metadata prefix is initialized on the host and needs to be uploaded.
    checkCudaErrors(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, metadata_buffer_size,
                                    cudaMemcpyHostToDevice, stream));

    if (ws.hostMem.ready != nullptr)
        checkCudaErrors(cudaEventRecord(ws.hostMem.ready, stream));

    Ptr2dNHWC<elem_type> ptr_h_out(batch, max_input_height, max_width, channels, (elem_type *)hori_gpu_data);

    dim3      blockSize(BLOCK, BLOCK / 4, 1);
    // Both planar and interleaved passes launch one grid-z slice per image and loop the channel planes
    // internally, so the per-image / per-output-pixel filter setup is shared across channels.
    const int pass_z = batch;
    dim3      gridSizeH(divUp(max_width, blockSize.x), divUp(max_input_height, blockSize.y), pass_z);
    dim3      gridSizeV(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), pass_z);

    dim3 coef_block(BLOCK * 2, 1, 1);
    dim3 h_coef_grid(divUp(max_width, coef_block.x), 1, batch);
    dim3 v_coef_grid(divUp(max_height, coef_block.x), 1, batch);

    size_t h_sm_size       = coef_block.x * (max_h_k_size * sizeof(work_type));
    size_t v_sm_size       = coef_block.x * (max_v_k_size * sizeof(work_type));
    size_t hv_sm_size1     = max_h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2     = max_v_k_size * sizeof(work_type) * blockSize.y;
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

    // compute horizontal coef
    _precomputeCoeffsVarShape<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        cols_gpu, roi_x_gpu, h_scale_batch_gpu, h_filterscale_batch_gpu, h_support_batch_gpu, out_cols_gpu,
        h_k_size_batch_gpu, filterp, h_bounds_batch_gpu, h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu,
        normalize_coeff, h_use_share_mem);

    checkKernelErrors();
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // compute vertical coef
    _precomputeCoeffsVarShape<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        rows_gpu, roi_y_gpu, v_scale_batch_gpu, v_filterscale_batch_gpu, v_support_batch_gpu, out_rows_gpu,
        v_k_size_batch_gpu, filterp, v_bounds_batch_gpu, v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu,
        normalize_coeff, v_use_share_mem);
    checkKernelErrors();
    // checkCudaErrors(cudaStreamSynchronize(stream));
    if constexpr (kPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const elem_type> src_ptr(inData);
        cuda::ImageBatchVarShapeWrap<elem_type>       dst_ptr(outData);
        switch (channels)
        {
        case 3:
            horizontal_pass_var_shape_planar_channels<3, elem_type, elem_type, Filter>
                <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, h_bounds_batch_gpu, h_bounds_offset_gpu,
                    h_kk_batch_gpu, h_kk_offset_gpu, init_buffer, round_up, hv_use_share_mem);
            break;
        case 4:
            if constexpr (!std::is_same_v<elem_type, float>)
            {
                horizontal_pass_var_shape_planar_channels<4, elem_type, elem_type, Filter>
                    <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                        src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, h_bounds_batch_gpu, h_bounds_offset_gpu,
                        h_kk_batch_gpu, h_kk_offset_gpu, init_buffer, round_up, hv_use_share_mem);
            }
            else
            {
                horizontal_pass_var_shape_planar<elem_type, elem_type, Filter>
                    <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                        src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, channels, h_bounds_batch_gpu,
                        h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu, init_buffer, round_up, hv_use_share_mem);
            }
            break;
        default:
            horizontal_pass_var_shape_planar<elem_type, elem_type, Filter>
                <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, channels, h_bounds_batch_gpu, h_bounds_offset_gpu,
                    h_kk_batch_gpu, h_kk_offset_gpu, init_buffer, round_up, hv_use_share_mem);
            break;
        }
        checkKernelErrors();
        // Same L1-broadcast property as the interleaved vertical: skip coefficient staging.
        vertical_pass_var_shape_planar<elem_type, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
            ptr_h_out, dst_ptr, filterp, v_k_size_batch_gpu, channels, v_bounds_batch_gpu, v_bounds_offset_gpu,
            v_kk_batch_gpu, v_kk_offset_gpu, init_buffer, round_up, false);
    }
    else
    {
        Ptr2dVarShapeNHWC<elem_type> src_ptr(inData);
        Ptr2dVarShapeNHWC<elem_type> dst_ptr(outData);
        // NC = compile-time channel count enables the interleaved channel-vectorized fast path (the
        // kernels gate it per-image on alignment and fall back to the scalar loop otherwise). NC=0 keeps
        // the scalar path for unexpected channel counts.
        auto                         launch_v = [&](auto nc_const)
        {
            constexpr int  NC = decltype(nc_const)::value;
            // A single channel does too little work per staged coefficient to amortize the shared-memory
            // copy and synchronization. Multi-channel paths retain staging because all channels reuse it.
            const bool     pass_use_share_mem = NC > 1 && hv_use_share_mem;
            const size_t   pass_h_sm_size     = pass_use_share_mem ? hv_sm_size1 : 0;
            const size_t   pass_v_sm_size     = pass_use_share_mem ? hv_sm_size2 : 0;
            // The fused kernel removes the intermediate write/read, but recomputes each horizontal sum
            // for every vertical tap. Upscales win across the measured GPUs; small-kernel downscales win
            // only on the architecture families selected by PillowResizeSupportsFusedDownscale.
            // Bit-exact: for float the T-typed intermediate is work_type, so the in-register inner sum
            // reproduces it op-for-op.
            constexpr bool can_fuse = (NC > 0 && std::is_same_v<elem_type, float>);
            const bool     small_kernel
                = max_h_k_size <= kPillowResizeFuseMaxKSize && max_v_k_size <= kPillowResizeFuseMaxKSize;
            const bool fuse_pass = can_fuse && filterp.support() <= work_type(1)
                                && (all_upscale || (small_kernel && PillowResizeSupportsFusedDownscale()));
            if (fuse_pass)
            {
                if constexpr (can_fuse) // guard instantiation: fused_pass_var_shape static_asserts NC > 0
                {
                    fused_pass_var_shape<NC, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
                        src_ptr, dst_ptr, filterp, h_k_size_batch_gpu, v_k_size_batch_gpu, h_bounds_batch_gpu,
                        h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu, v_bounds_batch_gpu, v_bounds_offset_gpu,
                        v_kk_batch_gpu, v_kk_offset_gpu, init_buffer, round_up, pass_use_share_mem);
                    checkKernelErrors();
                }
            }
            else
            {
                // See horizontal_pass_paired: byte-dtype horizontal downscales load the shared
                // window segment once for two adjacent outputs.
                // Three-channel byte pixels only: single-channel var-shape rows regressed 22-40%
                // on the A100/H100 regen burnins when paired.
                bool paired = false;
                if constexpr (NC == 3 && sizeof(elem_type) == 1)
                {
                    if (all_hcontract)
                    {
                        const size_t paired_sm  = 2 * pass_h_sm_size;
                        const bool   paired_use = pass_use_share_mem && paired_sm <= SHARE_MEM_LIMIT;
                        dim3         gridSizeHP(divUp(max_width, 2 * static_cast<int>(blockSize.x)),
                                                divUp(max_input_height, blockSize.y), pass_z);
                        horizontal_pass_var_shape_paired<NC, elem_type, elem_type, Filter>
                            <<<gridSizeHP, blockSize, paired_use ? paired_sm : 0, stream>>>(
                                src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, out_cols_gpu, h_bounds_batch_gpu,
                                h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu, init_buffer, round_up,
                                paired_use);
                        paired = true;
                    }
                }
                if (!paired)
                {
                    horizontal_pass_var_shape<NC, elem_type, elem_type, Filter>
                        <<<gridSizeH, blockSize, pass_h_sm_size, stream>>>(
                            src_ptr, ptr_h_out, filterp, h_k_size_batch_gpu, v_k_size_batch_gpu, h_bounds_batch_gpu,
                            h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu, v_bounds_batch_gpu,
                            v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu, init_buffer, round_up,
                            pass_use_share_mem);
                }
                checkKernelErrors();
                // Upscale batches are bound by the vertical pass's per-element load/store issue
                // rate, where the row vectorization wins; downscale verticals sit at the DRAM
                // ridge and keep the per-pixel kernel.
                if (all_upscale)
                {
                    constexpr int VEC = 4;
                    dim3          gridSizeVV(divUp(divUp(max_width * channels, VEC), static_cast<int>(blockSize.x)),
                                             divUp(max_height, blockSize.y), pass_z);
                    vertical_pass_var_shape_vec<VEC, elem_type, elem_type, Filter>
                        <<<gridSizeVV, blockSize, 0, stream>>>(ptr_h_out, dst_ptr, filterp, v_k_size_batch_gpu,
                                                               v_bounds_batch_gpu, v_bounds_offset_gpu, v_kk_batch_gpu,
                                                               v_kk_offset_gpu, init_buffer, round_up);
                }
                else
                {
                    vertical_pass_var_shape<NC, elem_type, elem_type, Filter>
                        <<<gridSizeV, blockSize, pass_v_sm_size, stream>>>(
                            ptr_h_out, dst_ptr, filterp, h_k_size_batch_gpu, v_k_size_batch_gpu, h_bounds_batch_gpu,
                            h_bounds_offset_gpu, h_kk_batch_gpu, h_kk_offset_gpu, v_bounds_batch_gpu,
                            v_bounds_offset_gpu, v_kk_batch_gpu, v_kk_offset_gpu, init_buffer, round_up,
                            pass_use_share_mem);
                }
            }
        };
        switch (channels)
        {
        case 1:
            launch_v(std::integral_constant<int, 1>{});
            break;
        case 2:
            launch_v(std::integral_constant<int, 2>{});
            break;
        case 3:
            launch_v(std::integral_constant<int, 3>{});
            break;
        case 4:
            launch_v(std::integral_constant<int, 4>{});
            break;
        default:
            launch_v(std::integral_constant<int, 0>{});
            break;
        }
    }

    checkKernelErrors();

    if (ws.cudaMem.ready != nullptr)
        checkCudaErrors(cudaEventRecord(ws.cudaMem.ready, stream));
}

} // namespace

// Single dtype switch shared by the interleaved and planar paths via the kPlanar template parameter,
// so the launch table is written once.
template<typename Filter, bool kPlanar>
void pillow_resize_dispatch_dtype(const ImageBatchVarShape &inData, const ImageBatchVarShape &outData,
                                  const Workspace &ws, cudaStream_t stream)
{
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());
    switch (data_type)
    {
    case kCV_8U:
        pillow_resize_var_shape<Filter, unsigned char, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    case kCV_8S:
        pillow_resize_var_shape<Filter, signed char, kPlanar>(inData, outData, ws, false, 0., true, stream);
        break;
    case kCV_16U:
        pillow_resize_var_shape<Filter, std::uint16_t, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    case kCV_16S:
        pillow_resize_var_shape<Filter, std::int16_t, kPlanar>(inData, outData, ws, false, 0., true, stream);
        break;
    case kCV_32S:
        pillow_resize_var_shape<Filter, int, kPlanar>(inData, outData, ws, false, 0., true, stream);
        break;
    case kCV_32F:
        pillow_resize_var_shape<Filter, float, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    case kCV_64F:
    default:
        break;
    }
}

template<typename Filter>
void pillow_resize_filter_var_shape(const ImageBatchVarShape &inData, const ImageBatchVarShape &outData,
                                    const Workspace &ws, NVCVInterpolationType interpolation, bool isPlanar,
                                    cudaStream_t stream)
{
    if (isPlanar)
        pillow_resize_dispatch_dtype<Filter, true>(inData, outData, ws, stream);
    else
        pillow_resize_dispatch_dtype<Filter, false>(inData, outData, ws, stream);
}

WorkspaceRequirements PillowResizeVarShape::getWorkspaceRequirements(DataShape max_input_shape,
                                                                     DataShape max_output_shape, DataType max_data_type)
{
    constexpr size_t kDefaultDeviceAlignment = 256;

    WorkspaceRequirements req{};

    int    max_support = 3; // Needed for various filtes Cubic needs 2 and Lanczos needs 3. Just use worst case.
    size_t size        = std::ceil(
               max_output_shape.H
                   * (((1.0 * max_input_shape.H / max_output_shape.H + 1) * max_support * 2 + 1) * sizeof(work_type)
               + 2 * sizeof(int))
               + max_output_shape.W
                     * (((1.0 * max_input_shape.W / max_output_shape.W + 1) * max_support * 2 + 1) * sizeof(work_type)
                 + 2 * sizeof(int)));

    const size_t metadata_buffer_size
        = (sizeof(void *) * 3 + sizeof(int) * 12 + sizeof(work_type) * 6) * max_input_shape.N;
    size_t buffer_size = metadata_buffer_size + size * max_input_shape.N;

    req.hostMem.size      = metadata_buffer_size;
    req.hostMem.alignment = alignof(std::max_align_t);

    buffer_size += static_cast<size_t>(max_input_shape.N) * max_input_shape.C * max_input_shape.H * max_output_shape.W
                 * sizeof(float);
    buffer_size += 16; // padding to 16-byte-align the float intermediate buffer (vector loads/stores)

    req.cudaMem.size      = buffer_size;
    req.cudaMem.alignment = kDefaultDeviceAlignment;

    return req;
}

ErrorCode PillowResizeVarShape::infer(const nvcv::ImageBatchVarShape &inDataBase,
                                      const nvcv::ImageBatchVarShape &outDataBase,
                                      const NVCVInterpolationType interpolation, cudaStream_t stream,
                                      const NVCVWorkspace &ws)
{
    if (!inDataBase.uniqueFormat() || !outDataBase.uniqueFormat())
    {
        LOG_ERROR("Images in input and outut batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inDataBase.uniqueFormat() != outDataBase.uniqueFormat())
    {
        LOG_ERROR("Invalid DataFormat between input and output");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = GetLegacyDataFormat(inDataBase);

    if (!(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW))
    {
        LOG_ERROR("Invalid input DataFormat " << format
                                              << ", the valid DataFormats are: \"NHWC\", \"HWC\", \"NCHW\", \"CHW\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    const bool isPlanar = (format == kNCHW || format == kCHW);

    int channels = inDataBase.uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format (matches the
    // Resize and Normalize operators).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // The planar passes launch one grid-z slice per image (channels are looped inside the kernel, see
    // pass_z = batch below), so numImages must fit CUDA's 65535 grid-z limit.
    if (isPlanar && inDataBase.numImages() > 65535)
    {
        LOG_ERROR("Planar PillowResize requires numImages <= 65535 (CUDA grid-z limit)");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    DataType data_type = helpers::GetLegacyDataType(inDataBase.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    // The horizontally-resized intermediate is one dense float image slot per batch entry
    // (numImages x maxInH x maxOutW x C), and the kernels compute its per-image offsets as 32-bit
    // products (Ptr2dNHWC). Reject batches whose intermediate exceeds INT32_MAX bytes instead of
    // overflowing the addressing and corrupting memory.
    {
        constexpr int64_t kMaxByteExtent = std::numeric_limits<int32_t>::max();
        const int64_t     interExtent    = static_cast<int64_t>(inDataBase.numImages()) * inDataBase.maxSize().h
                                  * outDataBase.maxSize().w * channels * DataSize(data_type);
        if (interExtent > kMaxByteExtent)
        {
            LOG_ERROR("Intermediate byte extent " << interExtent << " exceeds the 32-bit addressing limit "
                                                  << kMaxByteExtent << "; split the batch into smaller submissions");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
    }

    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
        pillow_resize_filter_var_shape<BilinearFilter>(inDataBase, outDataBase, ws, interpolation, isPlanar, stream);
        break;
    case NVCV_INTERP_BOX:
        pillow_resize_filter_var_shape<BoxFilter>(inDataBase, outDataBase, ws, interpolation, isPlanar, stream);
        break;
    case NVCV_INTERP_HAMMING:
        pillow_resize_filter_var_shape<HammingFilter>(inDataBase, outDataBase, ws, interpolation, isPlanar, stream);
        break;
    case NVCV_INTERP_CUBIC:
        pillow_resize_filter_var_shape<BicubicFilter>(inDataBase, outDataBase, ws, interpolation, isPlanar, stream);
        break;
    case NVCV_INTERP_LANCZOS:
        pillow_resize_filter_var_shape<LanczosFilter>(inDataBase, outDataBase, ws, interpolation, isPlanar, stream);
        break;
    default:
        LOG_ERROR("Unsupported interpolation method " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
