/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Rect.h>

using namespace nvcv;
using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;

using work_type = float;

#define M_PI 3.14159265358979323846 /* pi */

namespace nvcv::legacy::cuda_op {

static constexpr int kPillowResizeFuseMaxKSize = 5;

inline bool PillowResizeSupportsFusedDownscale()
{
    // The fused kernel's compute-versus-memory tradeoff is not monotonic across GPU generations.
    // Keep downscale fusion on the validated architectures and use the separable path elsewhere.
    thread_local int cached_device = -1;
    thread_local int cached_major  = -1;

    int device;
    checkCudaErrors(cudaGetDevice(&device));
    if (device != cached_device)
    {
        checkCudaErrors(cudaDeviceGetAttribute(&cached_major, cudaDevAttrComputeCapabilityMajor, device));
        cached_device = device;
    }

    return cached_major == 8 || cached_major == 9;
}

static constexpr float        bilinear_filter_support = 1.f;
static constexpr float        box_filter_support      = 0.5f;
static constexpr float        hamming_filter_support  = 1.f;
static constexpr float        bicubic_filter_support  = 2.f;
static constexpr float        lanczos_filter_support  = 3.f;
static constexpr unsigned int precision_bits          = 32 - 8 - 2;

class BilinearFilter
{
public:
    __host__ __device__ BilinearFilter()
        : _support(bilinear_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        if (x < 0.0)
        {
            x = -x;
        }
        if (x < 1.0)
        {
            return 1.0 - x;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class BoxFilter
{
public:
    __host__ __device__ BoxFilter()
        : _support(box_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        const float half_pixel = 0.5;
        if (x > -half_pixel && x <= half_pixel)
        {
            return 1.0;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class HammingFilter
{
public:
    __host__ __device__ HammingFilter()
        : _support(hamming_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        if (x < 0.0)
        {
            x = -x;
        }
        if (x == 0.0)
        {
            return 1.0;
        }
        if (x >= 1.0)
        {
            return 0.0;
        }
        x = x * M_PI;
        return sin(x) / x * (0.54f + 0.46f * cos(x));
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class BicubicFilter
{
public:
    __host__ __device__ BicubicFilter()
        : _support(bicubic_filter_support){};

    __host__ __device__ work_type filter(work_type x)
    {
        const float a = -0.5f;
        if (x < 0.0)
        {
            x = -x;
        }
        if (x < 1.0)
        {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        }
        if (x < 2.0)
        {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

class LanczosFilter
{
public:
    __host__ __device__ LanczosFilter()
        : _support(lanczos_filter_support){};

    __host__ __device__ work_type _sincFilter(work_type x)
    {
        if (x == 0.0)
        {
            return 1.0;
        }
        x = x * M_PI;
        return sin(x) / x;
    }

    __host__ __device__ work_type filter(work_type x)
    {
        const float lanczos_a_param = 3.0;
        if (-lanczos_a_param <= x && x < lanczos_a_param)
        {
            return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
        }
        return 0.0;
    }

    __host__ __device__ work_type support() const
    {
        return _support;
    };

private:
    work_type _support;
};

// Per-thread resampling-coefficient computation for one output position, shared by the tensor
// (`_precomputeCoeffs`) and var-shape (`_precomputeCoeffsVarShape`) precompute kernels. The two only
// differ in how they fetch their scalar parameters (direct kernel args vs. per-image arrays) and in
// the fixed-point `precision` shift, so the math itself lives here once. Writes the resampling weights
// into `kk_out` (cooperatively staged through shared memory when `use_share_mem`) and the [xmin, xmax)
// support bounds for output index `xx` into `bounds_out`.
template<class Filter>
__device__ inline void PillowPrecomputeCoeffs(int xx, int local_id, int x_offset, int in_size, int in0, work_type scale,
                                              work_type filterscale, work_type support, int out_size, int k_size,
                                              Filter &filterp, int *bounds_out, work_type *kk_out, bool normalize_coeff,
                                              bool use_share_mem, unsigned int precision)
{
    work_type *kk = kk_out + x_offset * k_size;
    if (use_share_mem)
    {
        extern __shared__ __align__(sizeof(work_type)) unsigned char smem_raw[];
        kk = reinterpret_cast<work_type *>(smem_raw);
    }

    if (xx < out_size)
    {
        int             x          = 0;
        int             xmin       = 0;
        int             xmax       = 0;
        work_type       center     = 0;
        work_type       ww         = 0;
        work_type       ss         = 0;
        const work_type half_pixel = 0.5;

        center = in0 + (xx + half_pixel) * scale;
        ww     = 0.0;
        ss     = 1.0 / filterscale;
        // Round the value.
        xmin = static_cast<int>(center - support + half_pixel);
        if (xmin < 0)
        {
            xmin = 0;
        }
        // Round the value.
        xmax = static_cast<int>(center + support + half_pixel);
        if (xmax > in_size)
        {
            xmax = in_size;
        }
        xmax -= xmin;
        work_type *k = &kk[local_id * k_size];
        for (x = 0; x < xmax; ++x)
        {
            work_type w = filterp.filter((x + xmin - center + half_pixel) * ss);
            k[x]        = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
        {
            if (std::fabs(ww) > 1e-5)
            {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < k_size; ++x)
        {
            k[x] = .0f;
        }
        if (normalize_coeff)
        {
            for (int i = 0; i < k_size; i++)
            {
                work_type val = k[i];
                if (val < 0)
                {
                    k[i] = static_cast<int>(-half_pixel + val * (1U << precision));
                }
                else
                {
                    k[i] = static_cast<int>(half_pixel + val * (1U << precision));
                }
            }
        }

        bounds_out[xx * 2]     = xmin;
        bounds_out[xx * 2 + 1] = xmax;
    }
    if (use_share_mem)
    {
        __syncthreads();
        for (int i = local_id; i < (out_size - x_offset) * k_size && i < blockDim.x * k_size; i += blockDim.x)
        {
            kk_out[x_offset * k_size + i] = kk[i];
        }
    }
}

// Cooperatively stage this thread block's slice of the coefficient table into shared memory and return
// the pointer the thread should read its per-position weights from. When shared memory is disabled,
// returns the thread block's slice of the global table directly. `block_extent` is the block dimension
// spanning the staged axis (blockDim.x for the horizontal pass, blockDim.y for the vertical pass).
// Shared by the horizontal/vertical resize passes across the tensor, var-shape, and planar paths.
inline __device__ work_type *PillowStageCoeffs(work_type *kk_global, work_type *smem, int offset, int ksize,
                                               int out_extent, int block_extent, bool use_share_mem)
{
    if (!use_share_mem)
    {
        return kk_global + offset * ksize;
    }
    const int local_tid = threadIdx.x + blockDim.x * threadIdx.y;
    for (int i = local_tid; i < block_extent * ksize && i < (out_extent - offset) * ksize; i += blockDim.x * blockDim.y)
    {
        smem[i] = kk_global[offset * ksize + i];
    }
    __syncthreads();
    return smem;
}

} // namespace nvcv::legacy::cuda_op
