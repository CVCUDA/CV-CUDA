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
#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpPillowResize.hpp"
#include "PillowResizePolicy.hpp"

#include <cuda_runtime.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/ImageFormat.h>
#include <nvcv/Rect.h>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/detail/Align.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

using work_type = float;

// Launch geometry: 32x8 threads for the resize passes and 64x1 for the coefficient precompute.
constexpr int    kBlock         = 32;
constexpr size_t kShareMemLimit = 4096;

// Fixed-point scale for the normalized-coefficient path: a 32-bit accumulator less 8 bits of
// sample and 2 bits of headroom (Pillow's PRECISION_BITS).
constexpr unsigned int kPrecisionBits = 32 - 8 - 2;

// The intermediate buffer is read/written with up-to-16-byte vector loads/stores
// (MakeType<T, 4>), which require natural alignment; the coefficient/bounds regions preceding it
// are only work_type/int aligned. getWorkspaceRequirements reserves an extra kInterAlign bytes so
// the buffer can always be rounded up to this boundary.
constexpr std::uintptr_t kInterAlign = 16;

// Fusing the two passes recomputes each horizontal sum once per vertical tap, so it only pays
// while the tap windows stay small.
constexpr int kPillowResizeFuseMaxKSize = 5;

// The filters below promote to double through this constant exactly as the legacy `M_PI` macro
// did; keeping the double promotion is what makes sin/cos bit-identical to the legacy kernels.
constexpr double kPi = 3.14159265358979323846;

constexpr float bilinear_filter_support = 1.f;
constexpr float box_filter_support      = 0.5f;
constexpr float hamming_filter_support  = 1.f;
constexpr float bicubic_filter_support  = 2.f;
constexpr float lanczos_filter_support  = 3.f;

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
        x = x * kPi;
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
        x = x * kPi;
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

template<class T>
struct TypeTag
{
    using type = T;
};

// The tensor and var-shape entry points select their filter from the same five interpolation types
// and reject the same complement, so the mapping and its rejection message live here once.
template<class Cb>
void DispatchFilter(NVCVInterpolationType interpolation, const Cb &cb)
{
    switch (interpolation)
    {
    case NVCV_INTERP_LINEAR:
        cb(TypeTag<BilinearFilter>{});
        break;
    case NVCV_INTERP_CUBIC:
        cb(TypeTag<BicubicFilter>{});
        break;
    case NVCV_INTERP_LANCZOS:
        cb(TypeTag<LanczosFilter>{});
        break;
    case NVCV_INTERP_BOX:
        cb(TypeTag<BoxFilter>{});
        break;
    case NVCV_INTERP_HAMMING:
        cb(TypeTag<HammingFilter>{});
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Unsupported interpolation method %d",
                              static_cast<int>(interpolation));
    }
}

inline bool PillowResizeSupportsFusedDownscale()
{
    int sm = 0;
    NVCV_CHECK_THROW(cvcuda::priv::GetCurrentDeviceSM(sm));
    return cvcuda::priv::PillowResizeFusesDownscaleForSM(sm);
}

// A pass stages its coefficients through shared memory only when the block's whole slice fits in
// kShareMemLimit; otherwise it reads them from global memory and is launched with no dynamic shared
// memory at all. Zeroes the size(s) in that case and reports which way the pass went. The two-size
// overload is for the horizontal/vertical pair, which share one decision.
inline bool FitSharedMem(size_t &size)
{
    if (size <= kShareMemLimit)
        return true;
    size = 0;
    return false;
}

inline bool FitSharedMem(size_t &size1, size_t &size2)
{
    if (size1 <= kShareMemLimit && size2 <= kShareMemLimit)
        return true;
    size1 = 0;
    size2 = 0;
    return false;
}

// The channel-vectorized kernels take the channel count as a template parameter, so both the tensor
// and the var-shape launchers need the same runtime-to-compile-time step. A count outside 1..4 maps
// to `integral_constant<int, 0>`, which selects the scalar channel-loop instantiation.
template<class Cb>
void DispatchChannelCount(int channels, const Cb &cb)
{
    switch (channels)
    {
    case 1:
        cb(std::integral_constant<int, 1>{});
        break;
    case 2:
        cb(std::integral_constant<int, 2>{});
        break;
    case 3:
        cb(std::integral_constant<int, 3>{});
        break;
    case 4:
        cb(std::integral_constant<int, 4>{});
        break;
    default:
        cb(std::integral_constant<int, 0>{});
        break;
    }
}

// ---------------------------------------------------------------------------------------------
// Buffer views
//
// Both resize passes index (sample, row, column, channel) and read the extents inside their tap
// loops, so the views carry the extents next to the strides. Strides are int32: the operator
// rejects any tensor -- input, output, or intermediate -- whose byte extent exceeds INT32_MAX, so
// every address stays in 32-bit registers.
// ---------------------------------------------------------------------------------------------

// Interleaved (NHWC/HWC) view: the channels of one pixel are contiguous.
template<typename T>
struct InterleavedPtr
{
    __host__ __device__ InterleavedPtr(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(rows_ * cols_ * ch_ * static_cast<int>(sizeof(T)))
        , rowStride(cols_ * ch_ * static_cast<int>(sizeof(T)))
        , data(data_)
    {
    }

    explicit __host__ InterleavedPtr(const nvcv::TensorDataAccessStridedImagePlanar &tensor)
        : batches(tensor.numSamples())
        , rows(tensor.numRows())
        , cols(tensor.numCols())
        , ch(tensor.numChannels())
        , imgStride(static_cast<int>(tensor.sampleStride()))
        , rowStride(static_cast<int>(tensor.rowStride()))
        , data(reinterpret_cast<T *>(tensor.sampleData(0)))
    {
    }

    inline __host__ __device__ T *ptr(int b, int y, int x, int c) const
    {
        return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(data) + b * imgStride + y * rowStride
                                     + (x * ch + c) * static_cast<int>(sizeof(T)));
    }

    // Every address this view produces is `data` plus a multiple of the strides, so the whole buffer
    // is safe for `align`-wide vector accesses iff the base and both strides are aligned.
    __host__ bool isAlignedTo(std::uintptr_t align) const
    {
        return nvcv::detail::IsAligned(reinterpret_cast<std::uintptr_t>(data) | static_cast<std::uintptr_t>(imgStride)
                                           | static_cast<std::uintptr_t>(rowStride),
                                       align);
    }

    int batches;
    int rows;
    int cols;
    int ch;
    int imgStride;
    int rowStride;
    T  *data;
};

// Planar (NCHW/CHW) view: each channel is a separate plane of the sample.
template<typename T>
struct PlanarPtr
{
    __host__ __device__ PlanarPtr(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(rows_ * cols_ * ch_ * static_cast<int>(sizeof(T)))
        , rowStride(cols_ * static_cast<int>(sizeof(T)))
        , chStride(cols_ * static_cast<int>(sizeof(T)) * rows_)
        , data(data_)
    {
    }

    explicit __host__ PlanarPtr(const nvcv::TensorDataAccessStridedImagePlanar &tensor)
        : batches(tensor.numSamples())
        , rows(tensor.numRows())
        , cols(tensor.numCols())
        , ch(tensor.numChannels())
        , imgStride(static_cast<int>(tensor.sampleStride()))
        , rowStride(static_cast<int>(tensor.rowStride()))
        , chStride(static_cast<int>(tensor.planeStride()))
        , data(reinterpret_cast<T *>(tensor.sampleData(0)))
    {
    }

    inline __host__ __device__ T *ptr(int b, int y, int x, int c) const
    {
        return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(data) + b * imgStride + c * chStride + y * rowStride
                                     + x * static_cast<int>(sizeof(T)));
    }

    // See InterleavedPtr::isAlignedTo; the plane stride is the extra term here.
    __host__ bool isAlignedTo(std::uintptr_t align) const
    {
        return nvcv::detail::IsAligned(reinterpret_cast<std::uintptr_t>(data) | static_cast<std::uintptr_t>(imgStride)
                                           | static_cast<std::uintptr_t>(chStride)
                                           | static_cast<std::uintptr_t>(rowStride),
                                       align);
    }

    int batches;
    int rows;
    int cols;
    int ch;
    int imgStride;
    int rowStride;
    int chStride;
    T  *data;
};

// ---------------------------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------------------------

// Per-thread resampling-coefficient computation for one output position, shared by the tensor
// (`PrecomputeCoeffs`) and var-shape (`PrecomputeCoeffsVarShape`) precompute kernels. The two only
// differ in how they fetch their scalar parameters (direct kernel args vs. per-image arrays), so
// the math itself lives here once. Writes the resampling weights into `kk_out` (cooperatively
// staged through shared memory when `use_share_mem`) and the [xmin, xmax) support bounds for
// output index `xx` into `bounds_out`.
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

// ---------------------------------------------------------------------------------------------
// Tensor kernels
// ---------------------------------------------------------------------------------------------

template<class Filter>
__global__ void PrecomputeCoeffs(int in_size, int in0, work_type scale, work_type filterscale, work_type support,
                                 int out_size, int k_size, Filter filterp, int *bounds_out, work_type *kk_out,
                                 bool normalize_coeff, bool use_share_mem)
{
    const int xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    PillowPrecomputeCoeffs<Filter>(xx, local_id, x_offset, in_size, in0, scale, filterscale, support, out_size, k_size,
                                   filterp, bounds_out, kk_out, normalize_coeff, use_share_mem, kPrecisionBits);
}

// SrcPtr2d / DstPtr2d are InterleavedPtr (NHWC) or PlanarPtr (NCHW). Both expose the same
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
// OPT amortizes coefficient/bounds setup across adjacent outputs without changing each tap sum.
template<int NC, class T, class Filter, class SrcPtr2d, class DstPtr2d, int OPT = 1>
__global__ void horizontal_pass(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int first_x    = OPT == 1 ? blockIdx.x * blockDim.x + threadIdx.x : 0;
    const int lane_base  = static_cast<int>(threadIdx.x) * OPT;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_offset   = blockIdx.x * blockDim.x * OPT;
    const int batch_idx  = blockIdx.z;
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_h), x_offset, h_ksize, out_width,
                                           blockDim.x * OPT, use_share_mem);

    if ((OPT == 1 ? first_x : x_offset + lane_base) < out_width && dst_y < out_height)
    {
#pragma unroll
        for (int o = 0; o < OPT; ++o)
        {
            const int local = lane_base + o;
            const int dst_x = OPT == 1 ? first_x : x_offset + local;
            if (OPT > 1 && dst_x >= out_width)
            {
                break;
            }

            const int  xmin = h_bounds[dst_x * 2];
            const int  xmax = h_bounds[dst_x * 2 + 1];
            work_type *h_k  = &h_k_tmp[local * h_ksize];

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
                    for (int c = 0; c < NC; ++c)
                        reinterpret_cast<T *>(out)[c] = cuda::SaturateCast<T>(std::round(acc[c]));
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
    const int batch_idx  = blockIdx.z;
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

template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d, int OPT = 1, bool PACKED_TAPS = false>
__global__ void horizontal_pass_planar_channels(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp,
                                                int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds,
                                                work_type *v_kk, work_type init_buffer, bool round_up,
                                                bool use_share_mem)
{
    static_assert(
        !PACKED_TAPS
        || (PC == 3 && std::is_same_v<T, unsigned char> && std::is_same_v<Filter, BicubicFilter> && OPT == 1));
    const int first_x    = OPT == 1 ? blockIdx.x * blockDim.x + threadIdx.x : 0;
    const int lane_base  = static_cast<int>(threadIdx.x);
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_offset   = blockIdx.x * blockDim.x * OPT;
    const int batch_idx  = blockIdx.z;
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_hp[];
    work_type *h_k_tmp = PillowStageCoeffs(h_kk, reinterpret_cast<work_type *>(kk_smem_hp), x_offset, h_ksize,
                                           out_width, blockDim.x * OPT, use_share_mem);

    if ((OPT == 1 ? first_x : x_offset + lane_base) < out_width && dst_y < out_height)
    {
#pragma unroll
        for (int o = 0; o < OPT; ++o)
        {
            const int local = lane_base + o * static_cast<int>(blockDim.x);
            const int dst_x = OPT == 1 ? first_x : x_offset + local;
            if (OPT > 1 && dst_x >= out_width)
            {
                break;
            }

            const int  xmin = h_bounds[dst_x * 2];
            const int  xmax = h_bounds[dst_x * 2 + 1];
            work_type *h_k  = &h_k_tmp[local * h_ksize];

            work_type h_ss[PC];
#pragma unroll
            for (int c = 0; c < PC; ++c) h_ss[c] = 0.0;

            // Hoist the per-plane row pointers: the tap loop then advances plain pointers instead of
            // re-deriving the full (b, y, x, c) offset for every tap of every plane. Three byte planes
            // only: at four planes the extra pointer registers measured 3.5-6.6% slower on multi-tap
            // rows, and float planes regressed contract cubic 6-16%.
            if constexpr (PC == 3 && sizeof(T) == 1)
            {
                const T *row[PC];
#pragma unroll
                for (int c = 0; c < PC; ++c) row[c] = src.ptr(batch_idx, dst_y, xmin, c);

                if constexpr (PACKED_TAPS)
                {
                    int tap = 0;
                    if ((xmin & 1) != 0 && xmax > 0)
                    {
                        const work_type weight = h_k[0];
#pragma unroll
                        for (int channel = 0; channel < PC; ++channel)
                            h_ss[channel] = h_ss[channel] + row[channel][0] * weight;
                        tap = 1;
                    }
#pragma unroll 2
                    for (; tap <= xmax - 2; tap += 2)
                    {
                        uchar2 values[PC];
#pragma unroll
                        for (int channel = 0; channel < PC; ++channel)
                            values[channel] = *reinterpret_cast<const uchar2 *>(row[channel] + tap);
#pragma unroll
                        for (int offset = 0; offset < 2; ++offset)
                        {
                            const work_type weight = h_k[tap + offset];
#pragma unroll
                            for (int channel = 0; channel < PC; ++channel)
                                h_ss[channel]
                                    = h_ss[channel]
                                    + __uint2float_rn(reinterpret_cast<const unsigned char *>(&values[channel])[offset])
                                          * weight;
                        }
                    }
                    if (tap < xmax)
                    {
                        const work_type weight = h_k[tap];
#pragma unroll
                        for (int channel = 0; channel < PC; ++channel)
                            h_ss[channel] = h_ss[channel] + row[channel][tap] * weight;
                    }
                }
                else
                {
                    for (int tap = 0; tap < xmax; ++tap)
                    {
                        const work_type weight = h_k[tap];
#pragma unroll
                        for (int channel = 0; channel < PC; ++channel)
                            h_ss[channel] = h_ss[channel] + row[channel][tap] * weight;
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
    const int batch_idx  = blockIdx.z;
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

// Keep the per-element tap order while amortizing setup across an output span.
template<class T, int VEC>
struct alignas(sizeof(T) * VEC) PillowSpan
{
    static_assert(VEC % 4 == 0, "PillowSpan carries whole four-component groups");

    static constexpr int kGroups = VEC / 4;

    using group_type = nvcv::cuda::MakeType<T, 4>;

    group_type v[kGroups];
};

// Accumulates a flat span of VEC consecutive scalars that share one weight per tap. Used by the
// vertical passes, where all cols*ch scalars of an output row share the same v_k.
template<class T, int VEC>
struct PillowSpanAccum
{
    using pack_type  = PillowSpan<T, VEC>;
    using group_type = typename pack_type::group_type;
    using accum_type = nvcv::cuda::MakeType<work_type, 4>;

    static constexpr int kGroups = pack_type::kGroups;

    accum_type s[kGroups];

    inline __device__ void init(work_type seed)
    {
#pragma unroll
        for (int g = 0; g < kGroups; ++g) s[g] = nvcv::cuda::SetAll<accum_type>(seed);
    }

    inline __device__ void tap(work_type k, const pack_type &px)
    {
#pragma unroll
        for (int g = 0; g < kGroups; ++g) s[g] = s[g] + k * nvcv::cuda::StaticCast<work_type>(px.v[g]);
    }

    inline __device__ void store(T *dst, bool round_up) const
    {
        if (round_up)
        {
            // Signed-integer path keeps std::round (half away from zero) per element for bit
            // exactness with the scalar path; cuda::round has no half-away mode.
            const work_type *acc = reinterpret_cast<const work_type *>(s);
#pragma unroll
            for (int i = 0; i < VEC; ++i) dst[i] = nvcv::cuda::SaturateCast<T>(std::round(acc[i]));
        }
        else
        {
            pack_type out;
#pragma unroll
            for (int g = 0; g < kGroups; ++g) out.v[g] = nvcv::cuda::SaturateCast<group_type>(s[g]);
            *reinterpret_cast<pack_type *>(dst) = out;
        }
    }
};

template<class T, int VEC>
__device__ PillowSpanAccum<T, VEC> PillowVerticalSpan(const std::byte *p, int taps, int rowStride, const work_type *k,
                                                      work_type seed)
{
    PillowSpanAccum<T, VEC> a;
    a.init(seed);
    for (int y = 0; y < taps; ++y)
    {
        a.tap(k[y], *reinterpret_cast<const PillowSpan<T, VEC> *>(p));
        p += rowStride;
    }
    return a;
}

// Vertical weights are shared across an output row. Each thread accumulates several adjacent
// elements to amortize setup; vector-aligned rows use the span path and tails remain scalar.
template<int VEC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass_vec(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                  int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                  work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int elem0      = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y;
    const int batch_idx  = blockIdx.z;
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
            if constexpr (VEC == 4)
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
                PillowVerticalSpan<T, VEC>(src_row, ymax, src.rowStride, v_k, init_buffer).store(dst_row, round_up);
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

// Exact-quarter weights and byte inputs make every sum exactly representable in float.
// Quantize each horizontal result before the vertical blend, as the intermediate buffer does.
inline __device__ uchar3 PillowLinear2xBlend(uchar3 a, uchar3 b, float ka)
{
    return cuda::SaturateCast<uchar3>(ka * cuda::StaticCast<float>(a) + (1.0f - ka) * cuda::StaticCast<float>(b));
}

inline __device__ uchar3 PillowLinear2xHorizontal(const uchar3 *row, int x, int width, bool odd)
{
    return odd ? PillowLinear2xBlend(row[x], row[min(x + 1, width - 1)], 0.75f)
               : PillowLinear2xBlend(row[max(x - 1, 0)], row[x], 0.25f);
}

template<int COUNT>
__device__ void PillowStoreRGB8(std::byte *dst, const uchar3 (&pixels)[COUNT])
{
    unsigned char bytes[3 * COUNT];
#pragma unroll
    for (int i = 0; i < COUNT; ++i)
    {
        bytes[3 * i]     = pixels[i].x;
        bytes[3 * i + 1] = pixels[i].y;
        bytes[3 * i + 2] = pixels[i].z;
    }
#pragma unroll
    for (int i = 0; i + 4 <= 3 * COUNT; i += 4)
        *reinterpret_cast<uchar4 *>(dst + i) = make_uchar4(bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
    if constexpr (COUNT == 2)
        *reinterpret_cast<uchar2 *>(dst + 4) = make_uchar2(bytes[4], bytes[5]);
}

template<int COUNT>
__device__ void PillowLinear2xRows(const InterleavedPtr<unsigned char> &src, const InterleavedPtr<unsigned char> &dst,
                                   int batch, int x, int y)
{
    const auto *top = reinterpret_cast<const uchar3 *>(src.ptr(batch, max(y - 1, 0), 0, 0));
    const auto *mid = reinterpret_cast<const uchar3 *>(src.ptr(batch, y, 0, 0));
    const auto *bot = reinterpret_cast<const uchar3 *>(src.ptr(batch, min(y + 1, src.rows - 1), 0, 0));
    uchar3      even[COUNT], odd[COUNT];
#pragma unroll
    for (int i = 0; i < COUNT; ++i)
    {
        const uchar3 hmid = PillowLinear2xHorizontal(mid, x + i / 2, src.cols, i % 2);
        even[i]           = PillowLinear2xBlend(PillowLinear2xHorizontal(top, x + i / 2, src.cols, i % 2), hmid, 0.25f);
        odd[i]            = PillowLinear2xBlend(hmid, PillowLinear2xHorizontal(bot, x + i / 2, src.cols, i % 2), 0.75f);
    }
    PillowStoreRGB8(reinterpret_cast<std::byte *>(dst.ptr(batch, 2 * y, 2 * x, 0)), even);
    PillowStoreRGB8(reinterpret_cast<std::byte *>(dst.ptr(batch, 2 * y + 1, 2 * x, 0)), odd);
}

__global__ void fused_pass_linear_2x(InterleavedPtr<unsigned char> src, InterleavedPtr<unsigned char> dst)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.cols || y >= src.rows)
        return;
    if (x + 1 < src.cols)
        PillowLinear2xRows<4>(src, dst, blockIdx.z, x, y);
    else
        PillowLinear2xRows<2>(src, dst, blockIdx.z, x, y);
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
    const int batch_idx  = blockIdx.z;
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
// Occupancy target for the fused planar pass. Its PC == 4 instantiation settles at 56 registers on
// sm_80 (GA100 / A100) where the pre-refactor build got 48 -- four resident blocks per SM instead
// of five, at 256 threads -- and that alone cost the float4 planar upscale rows 7.7-9.3% on A100.
// Name the old occupancy there.
//
// Only there. The other Ampere variants (sm_86/87/88/89) already settle at 40 registers without
// any bound -- six resident blocks on their 1536-threads/SM SMs -- and a bound set for five blocks
// would increase their register count to 48 (five blocks, a regression). Hopper and Blackwell also
// settle at 40 registers naturally (six blocks). Turing cannot express a five-block bound at all:
// five blocks is 1280 threads against its 1024 ceiling, and ptxas rejects the bound outright. So
// the attribute is emitted for sm_80 (GA100) and nowhere else, leaving the others' device code
// byte-identical to the undecorated source.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#    define CVCUDA_FUSED_PLANAR_BOUNDS __launch_bounds__(kBlock *(kBlock / 4), 5)
#else
#    define CVCUDA_FUSED_PLANAR_BOUNDS
#endif

template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__device__ void fused_planar_body(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                  int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                  work_type init_buffer, bool round_up, bool use_share_mem)
{
    static_assert(PC > 0, "fused planar pass needs a compile-time plane count");
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = blockIdx.z;
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

// Two entry points into that body, because the fuse predicate admits two workloads with opposite
// demands on the register file and one attribute cannot serve both.
//
// Upscale is latency-bound: few taps per output pixel, scattered source and coefficient reads, so it
// is paid in warps in flight and wants the fifth resident block the bound buys. Downscale is the
// opposite by construction -- the fusion exists to trade the intermediate's DRAM round-trip for
// recomputing each horizontal sum per vertical tap, which only pays while those accumulators stay
// live in registers. Capping it at 48 costs more than the extra block returns.
//
// Measured on A100 against the pre-migration baseline, both CUDA majors and both language drivers:
// the bound is worth -7.0 to -8.2% on the expand-linear float4 rows and +21.4 to +22.7% on the
// contract-linear ones. Splitting keeps the first and gives back the second.
template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
CVCUDA_FUSED_PLANAR_BOUNDS __global__ void fused_pass_planar_upscale(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi,
                                                                     Filter &filterp, int h_ksize, int v_ksize,
                                                                     int *h_bounds, work_type *h_kk, int *v_bounds,
                                                                     work_type *v_kk, work_type init_buffer,
                                                                     bool round_up, bool use_share_mem)
{
    fused_planar_body<PC, T>(src, dst, roi, filterp, h_ksize, v_ksize, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                             round_up, use_share_mem);
}

template<int PC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void fused_pass_planar_downscale(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp,
                                            int h_ksize, int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds,
                                            work_type *v_kk, work_type init_buffer, bool round_up, bool use_share_mem)
{
    fused_planar_body<PC, T>(src, dst, roi, filterp, h_ksize, v_ksize, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                             round_up, use_share_mem);
}

// Within each plane, adjacent columns share vertical weights and can reuse the span accumulator.
// The launch verifies base, row, sample and plane alignment; partial spans remain scalar.
template<int VEC, class T, class Filter, class SrcPtr2d, class DstPtr2d>
__global__ void vertical_pass_planar_vec(const SrcPtr2d src, DstPtr2d dst, NVCVRectI roi, Filter &filterp, int h_ksize,
                                         int v_ksize, int *h_bounds, work_type *h_kk, int *v_bounds, work_type *v_kk,
                                         work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x0     = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y    = threadIdx.y;
    const int y_offset   = blockIdx.y * blockDim.y;
    const int batch_idx  = blockIdx.z;
    int       out_height = dst.rows, out_width = dst.cols;

    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_vp[];
    work_type *v_k_tmp = PillowStageCoeffs(v_kk, reinterpret_cast<work_type *>(kk_smem_vp), y_offset, v_ksize,
                                           out_height, blockDim.y, use_share_mem);

    if (dst_x0 < out_width && dst_y < out_height)
    {
        int ymin = v_bounds[dst_y * 2];
        int ymax = v_bounds[dst_y * 2 + 1];

        work_type *v_k = &v_k_tmp[local_y * v_ksize];

        const bool full = (dst_x0 + VEC <= out_width);
        for (int c = 0; c < src.ch; ++c)
        {
            if (full)
            {
                if constexpr (VEC == 4)
                {
                    using vec_t         = cuda::MakeType<T, VEC>;
                    using accum_t       = cuda::MakeType<work_type, VEC>;
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
                    const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin, dst_x0, c));
                    PillowVerticalSpan<T, VEC>(p, ymax, src.rowStride, v_k, init_buffer)
                        .store(dst.ptr(batch_idx, dst_y, dst_x0, c), round_up);
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
    const int batch_idx  = blockIdx.z;
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

        using pack_t = typename PillowSpanAccum<T, VEC>::pack_type;

        const bool full = (dst_x0 + VEC <= out_width);
        for (int c = 0; c < src.ch; ++c)
        {
            if (full)
            {
                PillowSpanAccum<T, VEC> s0, s1;
                s0.init(init_buffer);
                s1.init(init_buffer);

                const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin0, dst_x0, c));
                for (int y = ymin0; y < preEnd; ++y)
                {
                    s0.tap(k0[y - ymin0], *reinterpret_cast<const pack_t *>(p));
                    p += src.rowStride;
                }
                for (int y = preEnd; y < end0; ++y)
                {
                    const pack_t px = *reinterpret_cast<const pack_t *>(p);
                    s0.tap(k0[y - ymin0], px);
                    s1.tap(k1[y - ymin1], px);
                    p += src.rowStride;
                }
                const std::byte *ps = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, sufBegin, dst_x0, c));
                for (int y = sufBegin; y < end1; ++y)
                {
                    s1.tap(k1[y - ymin1], *reinterpret_cast<const pack_t *>(ps));
                    ps += src.rowStride;
                }

                s0.store(dst.ptr(batch_idx, row0, dst_x0, c), round_up);
                if (has1)
                {
                    s1.store(dst.ptr(batch_idx, row0 + 1, dst_x0, c), round_up);
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
void pillow_resize_v2(const nvcv::TensorDataAccessStridedImagePlanar &inData,
                      const nvcv::TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                      bool normalize_coeff, work_type init_buffer, bool round_up, bool planar, cudaStream_t stream)
{
    const int                 in_N = inData.numSamples(), in_C = inData.numChannels(), in_H = inData.numRows();
    InterleavedPtr<elem_type> src_ptr(inData);
    InterleavedPtr<elem_type> dst_ptr(outData);
    NVCVRectI                 roi = {0, 0, src_ptr.cols, src_ptr.rows};
    Filter                    filterp;
    work_type                 h_scale = 0, v_scale = 0;
    work_type                 h_filterscale = 0, v_filterscale = 0;
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
    std::uintptr_t d_h_raw  = reinterpret_cast<std::uintptr_t>((char *)v_bounds + dst_ptr.rows * 2 * sizeof(int));
    elem_type     *d_h_data = reinterpret_cast<elem_type *>(nvcv::detail::AlignUp(d_h_raw, kInterAlign));

    dim3 blockSize(kBlock, kBlock / 4, 1);
    dim3 gridSizeH(util::DivUp(out_width, blockSize.x), util::DivUp(in_H, blockSize.y), in_N);
    dim3 gridSizeV(util::DivUp(out_width, blockSize.x), util::DivUp(out_height, blockSize.y), in_N);

    dim3 coef_block(kBlock * 2, 1, 1);
    dim3 h_coef_grid(util::DivUp(dst_ptr.cols, coef_block.x), 1, 1);
    dim3 v_coef_grid(util::DivUp(dst_ptr.rows, coef_block.x), 1, 1);

    size_t h_sm_size = coef_block.x * (h_k_size * sizeof(work_type));
    size_t v_sm_size = coef_block.x * (v_k_size * sizeof(work_type));

    size_t hv_sm_size1 = h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2 = v_k_size * sizeof(work_type) * blockSize.y;

    bool h_use_share_mem  = FitSharedMem(h_sm_size);
    bool v_use_share_mem  = FitSharedMem(v_sm_size);
    bool hv_use_share_mem = FitSharedMem(hv_sm_size1, hv_sm_size2);

    // compute horizental coef
    PrecomputeCoeffs<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        src_ptr.cols, roi.x, h_scale, h_filterscale, h_support, dst_ptr.cols, h_k_size, filterp, h_bounds, h_kk,
        normalize_coeff, h_use_share_mem);
    NVCV_CHECK_THROW(cudaGetLastError());

    // compute vertical coef
    PrecomputeCoeffs<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        src_ptr.rows, roi.y, v_scale, v_filterscale, v_support, dst_ptr.rows, v_k_size, filterp, v_bounds, v_kk,
        normalize_coeff, v_use_share_mem);
    NVCV_CHECK_THROW(cudaGetLastError());

    // Horizontal then vertical separable pass. The intermediate buffer holds the horizontally-resized
    // image (out_width columns, source rows) in the SAME layout as the input so the channel loop and
    // grid (z = N) match. Interleaved uses InterleavedPtr; planar uses PlanarPtr, which keeps one
    // z-slice per sample and amortizes the per-pixel bounds/weight setup across the C planes.
    auto launch = [&](auto nc_const, auto src_p, auto inter_p, auto dst_p)
    {
        constexpr int NC = decltype(nc_const)::value;

        // The fused byte path preserves intermediate quantization. Packed RGB stores require
        // four-byte-aligned destination rows; source uchar3 accesses need no extra alignment.
        if constexpr (NC == 3 && std::is_same_v<elem_type, unsigned char> && std::is_same_v<Filter, BilinearFilter>)
        {
            // Above this extent, float coefficient centers cannot represent the quarter-pixel positions.
            if (src_ptr.cols <= (1 << 22) && src_ptr.rows <= (1 << 22) && out_width == 2 * src_ptr.cols
                && out_height == 2 * src_ptr.rows && dst_p.isAlignedTo(alignof(uchar4)))
            {
                (void)inter_p; // the intermediate buffer is unused on the fused path
                dim3 gridSize2x(util::DivUp(util::DivUp(src_ptr.cols, 2), blockSize.x),
                                util::DivUp(src_ptr.rows, blockSize.y), in_N);
                fused_pass_linear_2x<<<gridSize2x, blockSize, 0, stream>>>(src_p, dst_p);
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

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
                NVCV_CHECK_THROW(cudaGetLastError());
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
                    const bool   paired_use = hv_use_share_mem && paired_sm <= kShareMemLimit;
                    const dim3   pairedBlock
                        = std::is_same_v<elem_type,
                                         unsigned char> && std::is_same_v<Filter, BicubicFilter> && paired_use
                               && util::DivUp<int64_t>(out_width, kBlock) * util::DivUp<int64_t>(in_H, kBlock / 2)
                                      < util::DivUp<int64_t>(out_width, 2 * kBlock)
                                            * util::DivUp<int64_t>(in_H, kBlock / 4)
                            ? dim3(kBlock / 2, kBlock / 2, 1)
                            : blockSize;
                    dim3 gridSizeHP(util::DivUp(out_width, 2 * pairedBlock.x), util::DivUp(in_H, pairedBlock.y), in_N);
                    horizontal_pass_paired<NC, elem_type, Filter>
                        <<<gridSizeHP, pairedBlock, paired_use ? paired_sm : 0, stream>>>(
                            src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
                            init_buffer, round_up, paired_use);
                    paired = true;
                }
            }
            if (!paired)
            {
                // Byte pixels move so few bytes per tap that the per-thread prologue dominates;
                // give each thread a run of adjacent output columns so it is paid once for the run.
                // Three-byte pixels are excluded: uchar3 has no single-instruction load, so each tap
                // is already three byte loads and the tap loop -- not the prologue -- is the cost.
                // Running four outputs there measured 0.84x (expand cubic 4k) and 0.94x (expand
                // linear 4k) against the one-output launch. The staged coefficient slice grows with
                // the run, so fall back when it no longer fits the shared-memory budget.
                constexpr bool wide_pixel = NC > 0 && NC != 3;
                constexpr int  OPT        = cvcuda::priv::kPillowHorizontalOutputsPerThread;
                const size_t   wide_sm    = OPT * hv_sm_size1;
                const bool     wide_fit   = !hv_use_share_mem || wide_sm <= kShareMemLimit;
                // Upscale only: a thread's run of outputs then spans barely more source than one
                // output does, so a warp still reads a dense span. On downscale the runs are
                // disjoint and OPT multiplies the warp's source stride by four; that measured 0.91x
                // (contract linear 1080p uchar4) and no gain anywhere else.
                if (sizeof(elem_type) == 1 && wide_pixel && wide_fit && h_scale < work_type(1))
                {
                    dim3 gridSizeHW(util::DivUp(out_width, OPT * static_cast<int>(blockSize.x)),
                                    util::DivUp(in_H, blockSize.y), in_N);
                    horizontal_pass<NC, elem_type, Filter, decltype(src_p), decltype(inter_p), OPT>
                        <<<gridSizeHW, blockSize, hv_use_share_mem ? wide_sm : 0, stream>>>(
                            src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
                            init_buffer, round_up, hv_use_share_mem);
                }
                else
                {
                    horizontal_pass<NC, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                        src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                        round_up, hv_use_share_mem);
                }
            }
            NVCV_CHECK_THROW(cudaGetLastError());
            // Byte/short spans cover more outputs per thread. F32 already spans four elements.
            constexpr int WIDE      = cvcuda::priv::kPillowVerticalVecBytes / static_cast<int>(sizeof(elem_type));
            constexpr int NARROW    = cvcuda::priv::kPillowNarrowVecElems;
            const int     row_elems = out_width * in_C;
            auto          launchVerticalVec = [&](auto vec_const)
            {
                constexpr int V = decltype(vec_const)::value;
                dim3          gridSizeVV(util::DivUp(util::DivUp(row_elems, V), blockSize.x),
                                         util::DivUp(out_height, blockSize.y), in_N);
                // Threads of a warp share dst_y here, so the per-tap weight loads are L1
                // broadcasts; staging them through shared memory only adds the block barrier.
                vertical_pass_vec<V, elem_type, Filter>
                    <<<gridSizeVV, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            };
            // Four-channel pixels already move one whole vector per tap in vertical_pass, so the flat
            // mapping used to be excluded for them: at four elements per thread it was the same four
            // bytes reshuffled across threads and measured slower on A100 (expand uchar4). WIDE >
            // NARROW holds exactly for the sub-four-byte element types, which is where the flat span
            // is genuinely wider than the pixel vector, so four-channel byte and short pixels now
            // take it and four-byte ones still do not.
            constexpr std::uintptr_t kWideAlign   = alignof(PillowSpan<elem_type, WIDE>);
            constexpr std::uintptr_t kNarrowAlign = alignof(PillowSpan<elem_type, NARROW>);
            if (WIDE > NARROW
                && cvcuda::priv::PillowWideSpanKeepsBlockBusy(row_elems, WIDE, static_cast<int>(blockSize.x))
                && inter_p.isAlignedTo(kWideAlign) && dst_p.isAlignedTo(kWideAlign))
            {
                launchVerticalVec(std::integral_constant<int, WIDE>{});
            }
            else if (NC != 4 && inter_p.isAlignedTo(kNarrowAlign) && dst_p.isAlignedTo(kNarrowAlign))
            {
                launchVerticalVec(std::integral_constant<int, NARROW>{});
            }
            else
            {
                vertical_pass<NC, elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
                    inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
            NVCV_CHECK_THROW(cudaGetLastError());
        }
    };

    if (planar)
    {
        // Planar (NCHW): channels are in separate planes. The horizontal pass uses the scalar channel
        // loop (NC == 0); the vertical pass is x-vectorized (consecutive columns share v_k) when the
        // intermediate and output buffers are vector-aligned, else scalar.
        PlanarPtr<elem_type> src_p(inData);
        PlanarPtr<elem_type> inter_p(in_N, in_H, out_width, in_C, (elem_type *)d_h_data);
        PlanarPtr<elem_type> dst_p(outData);

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
                                  && ((is_upscale && moderate_up && in_C == 4)
                                      || (!is_upscale && small_kernel && PillowResizeSupportsFusedDownscale()
                                          && (in_C == 3 || in_C == 4)));
            if (fuse_planar)
            {
                // Same body either way; the entry point differs only in its launch bounds.
                if (is_upscale)
                {
                    if (in_C == 3)
                    {
                        fused_pass_planar_upscale<3, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
                            src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                            round_up, false);
                    }
                    else
                    {
                        fused_pass_planar_upscale<4, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
                            src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                            round_up, false);
                    }
                }
                else
                {
                    if (in_C == 3)
                    {
                        fused_pass_planar_downscale<3, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
                            src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                            round_up, false);
                    }
                    else
                    {
                        fused_pass_planar_downscale<4, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
                            src_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                            round_up, false);
                    }
                }
                NVCV_CHECK_THROW(cudaGetLastError());
                return;
            }
        }

        constexpr int OPT       = cvcuda::priv::kPillowHorizontalOutputsPerThread;
        const size_t  h_wide_sm = OPT * hv_sm_size1;
        const bool    h_fit     = !hv_use_share_mem || h_wide_sm <= kShareMemLimit;
        const bool    h_wide_run
            = sizeof(elem_type) == 1 && h_scale < work_type(1) && h_fit
           && cvcuda::priv::PillowWideSpanKeepsBlockBusy(out_width, OPT, static_cast<int>(blockSize.x));
        dim3 gridSizeHW(util::DivUp(out_width, OPT * static_cast<int>(blockSize.x)), util::DivUp(in_H, blockSize.y),
                        in_N);
        auto launchPlanarHorizontal = [&](auto pc_const)
        {
            constexpr int PC = decltype(pc_const)::value;
            if constexpr (PC == 3 && std::is_same_v<elem_type, unsigned char> && std::is_same_v<Filter, BicubicFilter>)
            {
                if (h_scale > work_type(1) && v_scale > work_type(1) && src_p.isAlignedTo(alignof(uchar2)))
                {
                    horizontal_pass_planar_channels<PC, elem_type, Filter, std::decay_t<decltype(src_p)>,
                                                    std::decay_t<decltype(inter_p)>, 1, true>
                        <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(src_p, inter_p, roi, filterp, h_k_size,
                                                                        v_k_size, h_bounds, h_kk, v_bounds, v_kk,
                                                                        init_buffer, round_up, hv_use_share_mem);
                    return;
                }
            }
            if (h_wide_run)
            {
                horizontal_pass_planar_channels<PC, elem_type, Filter, decltype(src_p), decltype(inter_p), OPT>
                    <<<gridSizeHW, blockSize, hv_use_share_mem ? h_wide_sm : 0, stream>>>(
                        src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                        round_up, hv_use_share_mem);
            }
            else
            {
                horizontal_pass_planar_channels<PC, elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_p, inter_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer,
                    round_up, hv_use_share_mem);
            }
        };
        switch (in_C)
        {
        case 1:
            launchPlanarHorizontal(std::integral_constant<int, 1>{});
            break;
        case 3:
            launchPlanarHorizontal(std::integral_constant<int, 3>{});
            break;
        case 4:
            if constexpr (!std::is_same_v<elem_type, float>)
            {
                launchPlanarHorizontal(std::integral_constant<int, 4>{});
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
        NVCV_CHECK_THROW(cudaGetLastError());

        // Use the same output-span policy for planar rows.
        constexpr int            WIDE   = cvcuda::priv::kPillowVerticalVecBytes / static_cast<int>(sizeof(elem_type));
        constexpr int            NARROW = cvcuda::priv::kPillowNarrowVecElems;
        constexpr std::uintptr_t kWideAlign   = alignof(PillowSpan<elem_type, WIDE>);
        constexpr std::uintptr_t kNarrowAlign = alignof(PillowSpan<elem_type, NARROW>);
        // Threads of a warp share the output row (or row pair), so per-tap weight loads are L1
        // broadcasts; skip the shared-memory staging and its block barrier. Row pairing only
        // pays where adjacent output rows actually share source taps AND the shared load also
        // shares a byte-to-float convert: three-plane byte linear upscale. Downscale windows are
        // disjoint, float rows have no convert to share, and four byte planes regressed 6% on
        // A100 (the extra plane iteration doubles the paired accumulator pressure).
        const bool               pairRows
            = sizeof(elem_type) == 1 && in_C == 3 && v_scale < work_type(1) && filterp.support() <= work_type(1);
        auto launchPlanarVertical = [&](auto vec_const)
        {
            constexpr int V = decltype(vec_const)::value;
            if (pairRows)
            {
                dim3 gridSizeVP(util::DivUp(util::DivUp(out_width, V), blockSize.x),
                                util::DivUp(out_height, 2 * blockSize.y), in_N);
                vertical_pass_planar_vec_paired<V, elem_type, Filter>
                    <<<gridSizeVP, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            }
            else
            {
                dim3 gridSizeVP(util::DivUp(util::DivUp(out_width, V), blockSize.x),
                                util::DivUp(out_height, blockSize.y), in_N);
                vertical_pass_planar_vec<V, elem_type, Filter>
                    <<<gridSizeVP, blockSize, 0, stream>>>(inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds,
                                                           h_kk, v_bounds, v_kk, init_buffer, round_up, false);
            }
        };
        if (WIDE > NARROW
            && !(sizeof(elem_type) == 1 && in_C == 4 && v_scale > work_type(1) && filterp.support() > work_type(1))
            && cvcuda::priv::PillowWideSpanKeepsBlockBusy(out_width, WIDE, static_cast<int>(blockSize.x))
            && inter_p.isAlignedTo(kWideAlign) && dst_p.isAlignedTo(kWideAlign))
        {
            launchPlanarVertical(std::integral_constant<int, WIDE>{});
        }
        else if (inter_p.isAlignedTo(kNarrowAlign) && dst_p.isAlignedTo(kNarrowAlign))
        {
            launchPlanarVertical(std::integral_constant<int, NARROW>{});
        }
        else
        {
            vertical_pass<0, elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
                inter_p, dst_p, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk, init_buffer, round_up,
                hv_use_share_mem);
        }
        NVCV_CHECK_THROW(cudaGetLastError());
    }
    else
    {
        // Interleaved (NHWC): vectorize over the contiguous channels at the known channel count.
        InterleavedPtr<elem_type> src_p(inData);
        InterleavedPtr<elem_type> inter_p(in_N, in_H, out_width, in_C, (elem_type *)d_h_data);
        InterleavedPtr<elem_type> dst_p(outData);

        // The vectorized pixel load/store requires the buffers be aligned to alignof(MakeType<T,NC>).
        // The intermediate is 16-byte aligned above, but the user input/output tensors may have
        // arbitrary row strides -- gate on all three and fall back to the scalar loop (NC=0) otherwise.
        DispatchChannelCount(
            in_C,
            [&](auto nc_const)
            {
                constexpr int NC = decltype(nc_const)::value;
                if constexpr (NC > 0)
                {
                    constexpr std::uintptr_t kAlign = alignof(cuda::MakeType<elem_type, NC>);
                    if (src_p.isAlignedTo(kAlign) && inter_p.isAlignedTo(kAlign) && dst_p.isAlignedTo(kAlign))
                    {
                        launch(nc_const, src_p, inter_p, dst_p);
                        return;
                    }
                }
                launch(std::integral_constant<int, 0>{}, src_p, inter_p, dst_p);
            });
    }
}

// ---------------------------------------------------------------------------------------------
// Var-shape kernels
// ---------------------------------------------------------------------------------------------

template<class Filter>
__global__ void PrecomputeCoeffsVarShape(int *in_size_batch, int *in0_batch, work_type *scale_batch,
                                         work_type *filterscale_batch, work_type *support_batch, int *out_size_batch,
                                         int *k_size_batch, Filter filterp, int *bounds_out_batch,
                                         int *bound_out_offset, work_type *kk_out_batch, int *kk_out_offset,
                                         bool normalize_coeff, bool use_share_mem)
{
    const int xx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx   = blockIdx.z;
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
                                   filterp, bounds_out, kk_out, normalize_coeff, use_share_mem, kPrecisionBits);
}

template<int NC, class T1, class T2, class Filter>
__global__ void horizontal_pass_var_shape(const cuda::ImageBatchVarShapeWrapNHWC<T1> src, InterleavedPtr<T2> dst,
                                          Filter &filterp, int *h_ksize_batch, int *out_cols_batch, int *v_ksize_batch,
                                          int *h_bounds_batch, int *h_bounds_offset, work_type *h_kk_batch,
                                          int *h_kk_offset, int *v_bounds_batch, int *v_bounds_offset,
                                          work_type *v_kk_batch, int *v_kk_offset, work_type init_buffer, bool round_up,
                                          bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = blockIdx.z;
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // Bounds-within-slice fix (GPU OOB read; no pixel regression test constructible — sanitizer only).
    int out_height = dst.rows, out_width = out_cols_batch[batch_idx];
    int in_height = src.height(batch_idx);

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
            const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0, 0));
            const std::uintptr_t step
                = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0, 0)) - base; // row stride in bytes
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
            for (int c = 0; c < src.numChannels(); ++c)
            {
                work_type h_ss = 0.0;
                if (dst_y < in_height)
                {
                    const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, c));
                    for (int x = 0; x < xmax; ++x)
                    {
                        h_ss = h_ss + *reinterpret_cast<const T1 *>(p) * h_k[x];
                        p += src.numChannels() * sizeof(T1);
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
__global__ void horizontal_pass_var_shape_paired(const cuda::ImageBatchVarShapeWrapNHWC<T1> src, InterleavedPtr<T2> dst,
                                                 Filter &filterp, int *h_ksize_batch, int *out_cols_batch,
                                                 int *h_bounds_batch, int *h_bounds_offset, work_type *h_kk_batch,
                                                 int *h_kk_offset, work_type init_buffer, bool round_up,
                                                 bool use_share_mem)
{
    static_assert(NC > 0, "paired var-shape horizontal pass is the interleaved channel-vectorized path");
    const int pair0    = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x * 2;

    const int  batch_idx = blockIdx.z;
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // The intermediate is max-sized and uniform; the pair guards need this image's true output
    // width so the second output of the last pair never indexes the next image's bounds.
    int out_height = dst.rows, out_width = out_cols_batch[batch_idx];
    int in_height = src.height(batch_idx);

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

        const std::uintptr_t ibase = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0, 0));
        const std::uintptr_t istep = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0, 0)) - ibase;
        if (((ibase | istep) & (alignof(vin_t) - 1)) == 0)
        {
            accum_t s0 = cuda::SetAll<accum_t>(work_type{0});
            accum_t s1 = cuda::SetAll<accum_t>(work_type{0});
            if (dst_y < in_height)
            {
                const std::byte *base   = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, 0, 0));
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
                for (int c = 0; c < src.numChannels(); ++c)
                {
                    work_type h_ss = 0.0;
                    if (dst_y < in_height)
                    {
                        const std::byte *p = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, dst_y, xmin, c));
                        for (int x = 0; x < xmax; ++x)
                        {
                            h_ss = h_ss + *reinterpret_cast<const T1 *>(p) * hk[x];
                            p += src.numChannels() * sizeof(T1);
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
__global__ void vertical_pass_var_shape(const InterleavedPtr<T1> src, cuda::ImageBatchVarShapeWrapNHWC<T2> dst,
                                        Filter &filterp, int *h_ksize_batch, int *v_ksize_batch, int *h_bounds_batch,
                                        int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                        int *v_bounds_batch, int *v_bounds_offset, work_type *v_kk_batch,
                                        int *v_kk_offset, work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y  = threadIdx.y;
    const int y_offset = blockIdx.y * blockDim.y;

    const int  batch_idx = blockIdx.z;
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);

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
            const std::uintptr_t obase = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0, 0));
            const std::uintptr_t ostep
                = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0, 0)) - obase; // row stride in bytes
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
// the tensor path): all cols*nch scalars of an output row share v_k, so each thread accumulates VEC
// consecutive scalars with one wide load per tap. The float intermediate rows are 16-byte aligned by
// the host; the per-image OUTPUT alignment is checked once per thread (uniform across this block's
// image) and the row tail or an unaligned image falls back to a per-element loop with identical
// per-element arithmetic. Weights are read straight from L1: threads of a warp share dst_y, so the
// per-tap loads are broadcasts and shared-memory staging would only add a block barrier.
template<int VEC, class T1, class T2, class Filter>
__global__ void vertical_pass_var_shape_vec(const InterleavedPtr<T1> src, cuda::ImageBatchVarShapeWrapNHWC<T2> dst,
                                            Filter &filterp, int *v_ksize_batch, int *v_bounds_batch,
                                            int *v_bounds_offset, work_type *v_kk_batch, int *v_kk_offset,
                                            work_type init_buffer, bool round_up)
{
    const int elem0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int  batch_idx = blockIdx.z;
    int        v_ksize   = v_ksize_batch[batch_idx];
    int       *v_bounds  = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *v_kk      = v_kk_batch + v_kk_offset[batch_idx];

    int out_height = dst.height(batch_idx);
    int row_elems  = dst.width(batch_idx) * dst.numChannels();

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
        const std::uintptr_t obase  = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0, 0));
        const std::uintptr_t ostep  = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0, 0)) - obase;
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
__global__ void fused_pass_var_shape(const cuda::ImageBatchVarShapeWrapNHWC<T> src,
                                     cuda::ImageBatchVarShapeWrapNHWC<T> dst, Filter &filterp, int *h_ksize_batch,
                                     int *v_ksize_batch, int *h_bounds_batch, int *h_bounds_offset,
                                     work_type *h_kk_batch, int *h_kk_offset, int *v_bounds_batch, int *v_bounds_offset,
                                     work_type *v_kk_batch, int *v_kk_offset, work_type init_buffer, bool round_up,
                                     bool use_share_mem)
{
    static_assert(NC > 0, "fused_pass_var_shape is the interleaved channel-vectorized path");
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int  h_ksize  = h_ksize_batch[batch_idx];
    const int  v_ksize  = v_ksize_batch[batch_idx];
    int       *h_bounds = h_bounds_batch + h_bounds_offset[batch_idx];
    int       *v_bounds = v_bounds_batch + v_bounds_offset[batch_idx];
    work_type *h_kk     = h_kk_batch + h_kk_offset[batch_idx];
    work_type *v_kk     = v_kk_batch + v_kk_offset[batch_idx];

    const int out_height = dst.height(batch_idx), out_width = dst.width(batch_idx);

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
    const std::uintptr_t ibase   = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 0, 0, 0));
    const std::uintptr_t istep   = reinterpret_cast<std::uintptr_t>(src.ptr(batch_idx, 1, 0, 0)) - ibase;
    const std::uintptr_t obase   = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 0, 0, 0));
    const std::uintptr_t ostep   = reinterpret_cast<std::uintptr_t>(dst.ptr(batch_idx, 1, 0, 0)) - obase;
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
        for (int c = 0; c < src.numChannels(); ++c)
        {
            work_type ss = init_buffer;
            for (int y = 0; y < ymax; ++y)
            {
                const std::byte *p     = reinterpret_cast<const std::byte *>(src.ptr(batch_idx, ymin + y, xmin, c));
                work_type        inner = 0;
                for (int x = 0; x < xmax; ++x)
                {
                    inner = inner + *reinterpret_cast<const T *>(p) * h_k[x];
                    p += src.numChannels() * sizeof(T);
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
__global__ void horizontal_pass_var_shape_planar(const cuda::ImageBatchVarShapeWrap<const T1> src,
                                                 InterleavedPtr<T2> dst, Filter &filterp, int *h_ksize_batch,
                                                 int *out_cols_batch, int channels, int *h_bounds_batch,
                                                 int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                                 work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = blockIdx.z;
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // All channel planes of an image share its size, so the per-image setup (bounds, coeffs, in-size)
    // is computed once and reused across the channels looped below.
    // Bounds-within-slice fix (GPU OOB read; no pixel regression test constructible — sanitizer only).
    int out_height = dst.rows, out_width = out_cols_batch[batch_idx];
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
                                                          InterleavedPtr<T2> dst, Filter &filterp, int *h_ksize_batch,
                                                          int *out_cols_batch, int *h_bounds_batch,
                                                          int *h_bounds_offset, work_type *h_kk_batch, int *h_kk_offset,
                                                          work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_x  = threadIdx.x;
    const int x_offset = blockIdx.x * blockDim.x;

    const int  batch_idx = blockIdx.z;
    int        h_ksize   = h_ksize_batch[batch_idx];
    int       *h_bounds  = h_bounds_batch + h_bounds_offset[batch_idx];
    work_type *h_kk      = h_kk_batch + h_kk_offset[batch_idx];

    // Bounds-within-slice fix (GPU OOB read; no pixel regression test constructible — sanitizer only).
    int out_height = dst.rows, out_width = out_cols_batch[batch_idx];
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

            constexpr bool is_rgb8 = PC == 3 && std::is_same_v<T1, uint8_t> && std::is_same_v<T2, uint8_t>;
            if constexpr (is_rgb8 && std::is_same_v<Filter, BicubicFilter>)
            {
#pragma unroll 1
                for (int tap = 0; tap < xmax; ++tap)
                {
                    const work_type coefficient = h_k[tap];
#pragma unroll
                    for (int plane = 0; plane < PC; ++plane)
                    {
                        h_ss[plane] = h_ss[plane] + row[plane][tap] * coefficient;
                    }
                }
            }
            else
            {
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
__global__ void vertical_pass_var_shape_planar(const InterleavedPtr<T1> src, cuda::ImageBatchVarShapeWrap<T2> dst,
                                               Filter &filterp, int *v_ksize_batch, int channels, int *v_bounds_batch,
                                               int *v_bounds_offset, work_type *v_kk_batch, int *v_kk_offset,
                                               work_type init_buffer, bool round_up, bool use_share_mem)
{
    const int dst_x    = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y    = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_y  = threadIdx.y;
    const int y_offset = blockIdx.y * blockDim.y;

    const int  batch_idx = blockIdx.z;
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

template<bool Planar>
struct PillowVarShapeU8TileAdapter
{
    cuda::ImageBatchVarShapeWrap<const unsigned char> source;
    cuda::ImageBatchVarShapeWrap<unsigned char>       destination;

    __device__ int width(int sample) const
    {
        return destination.width(sample);
    }

    __device__ int height(int sample) const
    {
        return destination.height(sample);
    }

    __device__ unsigned char load(int sample, int row, int column, int channel) const
    {
        return *source.ptr(sample, Planar ? channel : 0, row, Planar ? column : column * 3 + channel);
    }

    __device__ void store(int sample, int row, int column, int channel, unsigned char value) const
    {
        *destination.ptr(sample, Planar ? channel : 0, row, Planar ? column : column * 3 + channel) = value;
    }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#    define CVCUDA_PILLOW_U8_TILE_BOUNDS \
        __launch_bounds__(cvcuda::priv::kPillowU8TileThreads, MinBlocks < 4 ? MinBlocks : 4)
#else
#    define CVCUDA_PILLOW_U8_TILE_BOUNDS __launch_bounds__(cvcuda::priv::kPillowU8TileThreads, MinBlocks)
#endif

template<class ImageAdapter, int MinBlocks>
__global__ CVCUDA_PILLOW_U8_TILE_BOUNDS void pillow_fused_u8_tile(
    ImageAdapter images, const int *horizontalSize, const int *verticalSize, const int *horizontalBounds,
    const int *horizontalBoundsOffset, const work_type *horizontalWeights, const int *horizontalWeightsOffset,
    const int *verticalBounds, const int *verticalBoundsOffset, const work_type *verticalWeights,
    const int *verticalWeightsOffset)
{
    constexpr int            tileWidth  = cvcuda::priv::kPillowU8TileWidth;
    constexpr int            tileHeight = cvcuda::priv::kPillowU8TileHeight;
    __shared__ unsigned char intermediate[cvcuda::priv::kPillowU8TileMaxRows * tileWidth * 3];
    __shared__ int           firstRow;
    __shared__ int           rowCount;
    const int                sample = blockIdx.z;
    const int                width = images.width(sample), height = images.height(sample);
    const int                firstColumn = blockIdx.x * tileWidth, firstOutputRow = blockIdx.y * tileHeight;
    if (firstColumn >= width || firstOutputRow >= height)
        return;
    const int       *horizontalWindows = horizontalBounds + horizontalBoundsOffset[sample];
    const int       *verticalWindows   = verticalBounds + verticalBoundsOffset[sample];
    const work_type *horizontal        = horizontalWeights + horizontalWeightsOffset[sample];
    const work_type *vertical          = verticalWeights + verticalWeightsOffset[sample];
    if (threadIdx.x == 0)
    {
        const int lastOutputRow = min(firstOutputRow + tileHeight, height) - 1;
        firstRow                = verticalWindows[firstOutputRow * 2];
        rowCount = verticalWindows[lastOutputRow * 2] + verticalWindows[lastOutputRow * 2 + 1] - firstRow;
    }
    __syncthreads();
    for (int index = threadIdx.x; index < rowCount * tileWidth; index += blockDim.x)
    {
        const int sourceRow = firstRow + index / tileWidth;
        const int column    = firstColumn + index % tileWidth;
        if (column >= width)
            continue;
        const int        firstTap = horizontalWindows[column * 2], taps = horizontalWindows[column * 2 + 1];
        const work_type *weights = horizontal + column * horizontalSize[sample];
        work_type        sums[3] = {};
        for (int tap = 0; tap < taps; ++tap)
        {
#pragma unroll
            for (int channel = 0; channel < 3; ++channel)
                sums[channel] = sums[channel] + images.load(sample, sourceRow, firstTap + tap, channel) * weights[tap];
        }
#pragma unroll
        for (int channel = 0; channel < 3; ++channel)
            intermediate[index * 3 + channel] = cuda::SaturateCast<unsigned char>(sums[channel]);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < tileWidth * tileHeight; index += blockDim.x)
    {
        const int column = firstColumn + index % tileWidth;
        const int row    = firstOutputRow + index / tileWidth;
        if (column >= width || row >= height)
            continue;
        const int        firstTap = verticalWindows[row * 2], taps = verticalWindows[row * 2 + 1];
        const work_type *weights = vertical + row * verticalSize[sample];
        work_type        sums[3] = {};
        for (int tap = 0; tap < taps; ++tap)
        {
            const int middle = ((firstTap + tap - firstRow) * tileWidth + index % tileWidth) * 3;
#pragma unroll
            for (int channel = 0; channel < 3; ++channel)
                sums[channel] = sums[channel] + intermediate[middle + channel] * weights[tap];
        }
#pragma unroll
        for (int channel = 0; channel < 3; ++channel)
            images.store(sample, row, column, channel, cuda::SaturateCast<unsigned char>(sums[channel]));
    }
}

// The per-image metadata prefix of the var-shape workspace. The host fills it, the device reads the
// verbatim copy, and getWorkspaceRequirements sizes it -- so all three must agree on the layout.
// Carving it from one definition keeps that agreement structural rather than a three-way eyeball
// match between two 18-slot pointer ladders and a byte formula.
#undef CVCUDA_PILLOW_U8_TILE_BOUNDS

struct VarShapeMeta
{
    int       *rows;
    int       *cols;
    int       *out_rows;
    int       *out_cols;
    int       *roi_x;
    int       *roi_y;
    work_type *h_scale;
    work_type *v_scale;
    work_type *h_filterscale;
    work_type *v_filterscale;
    work_type *h_support;
    work_type *v_support;
    int       *h_k_size;
    int       *v_k_size;
    int       *h_bounds_offset;
    int       *v_bounds_offset;
    int       *h_kk_offset;
    int       *v_kk_offset;

    // Three pointer-sized slots precede the arrays: the legacy layout reserved them for per-image
    // input/output/intermediate pointers. Nothing reads or writes them any more, but they stay part
    // of the sized workspace, which is public contract.
    static constexpr size_t kReservedPtrSlots = 3;
    static constexpr size_t kBytesPerImage
        = sizeof(void *) * kReservedPtrSlots + sizeof(int) * 12 + sizeof(work_type) * 6;

    static VarShapeMeta Carve(void *base, int batch)
    {
        char *cursor = static_cast<char *>(base) + sizeof(void *) * kReservedPtrSlots * batch;

        auto take = [&](size_t elemSize)
        {
            char *begin = cursor;
            cursor += elemSize * batch;
            return begin;
        };
        auto takeInt = [&]
        {
            return reinterpret_cast<int *>(take(sizeof(int)));
        };
        auto takeWork = [&]
        {
            return reinterpret_cast<work_type *>(take(sizeof(work_type)));
        };

        VarShapeMeta m;
        m.rows            = takeInt();
        m.cols            = takeInt();
        m.out_rows        = takeInt();
        m.out_cols        = takeInt();
        m.roi_x           = takeInt();
        m.roi_y           = takeInt();
        m.h_scale         = takeWork();
        m.v_scale         = takeWork();
        m.h_filterscale   = takeWork();
        m.v_filterscale   = takeWork();
        m.h_support       = takeWork();
        m.v_support       = takeWork();
        m.h_k_size        = takeInt();
        m.v_k_size        = takeInt();
        m.h_bounds_offset = takeInt();
        m.v_bounds_offset = takeInt();
        m.h_kk_offset     = takeInt();
        m.v_kk_offset     = takeInt();
        return m;
    }
};

template<bool Planar>
bool PillowVarShapeStorageDisjoint(const nvcv::ImageBatchVarShape &input, const nvcv::ImageBatchVarShape &output,
                                   void *scratch, size_t scratchBytes)
{
    using Range              = cvcuda::priv::PillowStorageRange;
    constexpr int planes     = Planar ? 3 : 1;
    constexpr int pixelBytes = Planar ? 1 : 3;
    static_assert(3 * sizeof(Range) <= VarShapeMeta::kBytesPerImage);
    if (scratch == nullptr || input.numImages() <= 0 || input.numImages() != output.numImages()
        || static_cast<size_t>(input.numImages()) > scratchBytes / sizeof(Range) / planes)
        return false;
    size_t count = 0;
    Range  envelope{std::numeric_limits<uintptr_t>::max(), 0};
    for (int sample = 0; sample < input.numImages(); ++sample)
    {
        auto data = input[sample].exportData<nvcv::ImageDataStridedCuda>();
        if (!data || data->numPlanes() != planes)
            return false;
        for (int plane = 0; plane < planes; ++plane)
        {
            const auto &descriptor = data->plane(plane);
            Range       range;
            if (!cvcuda::priv::PillowPlaneStorageRange(reinterpret_cast<uintptr_t>(descriptor.basePtr),
                                                       descriptor.height, descriptor.rowStride,
                                                       static_cast<int64_t>(descriptor.width) * pixelBytes, range))
                return false;
            std::memcpy(static_cast<unsigned char *>(scratch) + count * sizeof(Range), &range, sizeof(range));
            ++count;
            envelope.begin = std::min(envelope.begin, range.begin);
            envelope.end   = std::max(envelope.end, range.end);
        }
    }
    for (int sample = 0; sample < output.numImages(); ++sample)
    {
        auto data = output[sample].exportData<nvcv::ImageDataStridedCuda>();
        if (!data || data->numPlanes() != planes)
            return false;
        for (int plane = 0; plane < planes; ++plane)
        {
            const auto &descriptor = data->plane(plane);
            Range       range;
            if (!cvcuda::priv::PillowPlaneStorageRange(reinterpret_cast<uintptr_t>(descriptor.basePtr),
                                                       descriptor.height, descriptor.rowStride,
                                                       static_cast<int64_t>(descriptor.width) * pixelBytes, range)
                || (!cvcuda::priv::PillowStorageRangesDisjoint(envelope, range)
                    && !cvcuda::priv::PillowStorageRangeDisjointFromAll(range, static_cast<const Range *>(scratch),
                                                                        count)))
                return false;
        }
    }
    return true;
}

template<typename Filter, typename elem_type, bool kPlanar>
void pillow_resize_var_shape(const nvcv::ImageBatchVarShape &inDataBase, const nvcv::ImageBatchVarShape &outDataBase,
                             const cvcuda::Workspace &ws, bool normalize_coeff, work_type init_buffer, bool round_up,
                             cudaStream_t stream)
{
    if (ws.hostMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaEventSynchronize(ws.hostMem.ready));

    if (ws.cudaMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ws.cudaMem.ready));

    void *cpu_workspace = ws.hostMem.data;
    void *gpu_workspace = ws.cudaMem.data;

    auto inDataPtr = inDataBase.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!inDataPtr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be a device-accessible, varshape pitch-linear image batch");
    }

    auto outDataPtr = outDataBase.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (!outDataPtr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be a device-accessible, varshape pitch-linear image batch");
    }

    const nvcv::ImageBatchVarShapeDataStridedCuda &inData  = *inDataPtr;
    const nvcv::ImageBatchVarShapeDataStridedCuda &outData = *outDataPtr;

    int channels = inData.uniqueFormat().numChannels();
    int batch    = inData.numImages();

    constexpr bool can_tile
        = kPlanar
       && std::is_same_v<
              elem_type,
              unsigned char> && (std::is_same_v<Filter, BilinearFilter> || std::is_same_v<Filter, BicubicFilter>);
    bool storage_disjoint = false;
    if constexpr (can_tile)
    {
        if (channels == 3)
            storage_disjoint
                = PillowVarShapeStorageDisjoint<kPlanar>(inDataBase, outDataBase, cpu_workspace, ws.hostMem.req.size);
    }

    Filter filterp;

    nvcv::Size2D outMaxSize = outData.maxSize();
    nvcv::Size2D inMaxSize  = inData.maxSize();

    int max_height = outMaxSize.h, max_width = outMaxSize.w;
    int max_input_height = inMaxSize.h;

    const VarShapeMeta host = VarShapeMeta::Carve(cpu_workspace, batch);

    int  h_kk_total = 0, v_kk_total = 0;
    int  max_h_k_size = 0, max_v_k_size = 0;
    int  h_bounds_total = 0, v_bounds_total = 0;
    bool all_upscale      = true;
    bool all_hcontract    = true;
    bool all_u8_tiles_fit = true;

    for (int i = 0; i < batch; i++)
    {
        // One `operator[]` per batch entry: it crosses the C ABI and takes a reference on the image,
        // so re-reading it for every field would pay that cost six times per image.
        const nvcv::Size2D inSize  = inDataBase[i].size();
        const nvcv::Size2D outSize = outDataBase[i].size();

        host.rows[i]     = inSize.h;
        host.cols[i]     = inSize.w;
        host.out_rows[i] = outSize.h;
        host.out_cols[i] = outSize.w;

        host.roi_x[i] = 0;
        host.roi_y[i] = 0;

        work_type h_scale = 0, v_scale = 0;
        work_type h_filterscale = 0, v_filterscale = 0;
        h_filterscale = h_scale = static_cast<work_type>(inSize.w) / host.out_cols[i];
        v_filterscale = v_scale = static_cast<work_type>(inSize.h) / host.out_rows[i];
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
        host.h_scale[i]       = h_scale;
        host.v_scale[i]       = v_scale;
        host.h_filterscale[i] = h_filterscale;
        host.v_filterscale[i] = v_filterscale;

        // Determine support size (length of resampling filter).
        work_type h_support = filterp.support() * h_filterscale;
        work_type v_support = filterp.support() * v_filterscale;
        // Maximum number of coeffs.
        int       h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
        int       v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;
        host.h_support[i]  = h_support;
        host.v_support[i]  = v_support;
        host.h_k_size[i]   = h_k_size;
        host.v_k_size[i]   = v_k_size;
        if constexpr (can_tile)
        {
            all_u8_tiles_fit = all_u8_tiles_fit && channels == 3
                            && cvcuda::priv::PillowU8FusedTileFits(inSize.h, outSize.h, v_k_size);
        }
        host.h_kk_offset[i] = h_kk_total;
        host.v_kk_offset[i] = v_kk_total;
        h_kk_total += host.out_cols[i] * h_k_size;
        v_kk_total += host.out_rows[i] * v_k_size;
        host.h_bounds_offset[i] = h_bounds_total;
        host.v_bounds_offset[i] = v_bounds_total;
        h_bounds_total += host.out_cols[i] * 2;
        v_bounds_total += host.out_rows[i] * 2;

        if (h_k_size > max_h_k_size)
            max_h_k_size = h_k_size;
        if (v_k_size > max_v_k_size)
            max_v_k_size = v_k_size;
    }

    const VarShapeMeta dev = VarShapeMeta::Carve(gpu_workspace, batch);

    const size_t metadata_buffer_size = VarShapeMeta::kBytesPerImage * batch;

    work_type *h_kk_batch_gpu     = (work_type *)((char *)gpu_workspace + metadata_buffer_size);
    work_type *v_kk_batch_gpu     = (work_type *)((char *)h_kk_batch_gpu + sizeof(work_type) * h_kk_total);
    int       *h_bounds_batch_gpu = (int *)((char *)v_kk_batch_gpu + sizeof(work_type) * v_kk_total);
    int       *v_bounds_batch_gpu = (int *)((char *)h_bounds_batch_gpu + sizeof(int) * h_bounds_total);

    const size_t current_buffer_size = metadata_buffer_size + sizeof(work_type) * (h_kk_total + v_kk_total)
                                     + sizeof(int) * (h_bounds_total + v_bounds_total);

    // Buffer for storing results from the horizontal pass. The intermediate is elem_type: integer
    // dtypes quantize between passes exactly like the tensor path and Pillow's 8bpc pipeline (and move
    // a quarter of the float bytes); float keeps the full-precision work_type intermediate.
    // getWorkspaceRequirements sizes this region for the float worst case.
    std::uintptr_t hori_raw      = reinterpret_cast<std::uintptr_t>((char *)gpu_workspace + current_buffer_size);
    void          *hori_gpu_data = reinterpret_cast<void *>(nvcv::detail::AlignUp(hori_raw, kInterAlign));

    // The large coefficient/bounds regions are produced by PrecomputeCoeffsVarShape below; only the
    // small per-sample metadata prefix is initialized on the host and needs to be uploaded.
    NVCV_CHECK_THROW(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, metadata_buffer_size,
                                     cudaMemcpyHostToDevice, stream));

    if (ws.hostMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaEventRecord(ws.hostMem.ready, stream));

    InterleavedPtr<elem_type> ptr_h_out(batch, max_input_height, max_width, channels, (elem_type *)hori_gpu_data);

    dim3      blockSize(kBlock, kBlock / 4, 1);
    // Both planar and interleaved passes launch one grid-z slice per image and loop the channel planes
    // internally, so the per-image / per-output-pixel filter setup is shared across channels.
    const int pass_z = batch;
    dim3      gridSizeH(util::DivUp(max_width, blockSize.x), util::DivUp(max_input_height, blockSize.y), pass_z);
    dim3      gridSizeV(util::DivUp(max_width, blockSize.x), util::DivUp(max_height, blockSize.y), pass_z);

    dim3 coef_block(kBlock * 2, 1, 1);
    dim3 h_coef_grid(util::DivUp(max_width, coef_block.x), 1, batch);
    dim3 v_coef_grid(util::DivUp(max_height, coef_block.x), 1, batch);

    size_t h_sm_size   = coef_block.x * (max_h_k_size * sizeof(work_type));
    size_t v_sm_size   = coef_block.x * (max_v_k_size * sizeof(work_type));
    size_t hv_sm_size1 = max_h_k_size * sizeof(work_type) * blockSize.x;
    size_t hv_sm_size2 = max_v_k_size * sizeof(work_type) * blockSize.y;

    bool h_use_share_mem  = FitSharedMem(h_sm_size);
    bool v_use_share_mem  = FitSharedMem(v_sm_size);
    bool hv_use_share_mem = FitSharedMem(hv_sm_size1, hv_sm_size2);

    // compute horizontal coef
    PrecomputeCoeffsVarShape<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        dev.cols, dev.roi_x, dev.h_scale, dev.h_filterscale, dev.h_support, dev.out_cols, dev.h_k_size, filterp,
        h_bounds_batch_gpu, dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, normalize_coeff, h_use_share_mem);
    NVCV_CHECK_THROW(cudaGetLastError());

    // compute vertical coef
    PrecomputeCoeffsVarShape<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        dev.rows, dev.roi_y, dev.v_scale, dev.v_filterscale, dev.v_support, dev.out_rows, dev.v_k_size, filterp,
        v_bounds_batch_gpu, dev.v_bounds_offset, v_kk_batch_gpu, dev.v_kk_offset, normalize_coeff, v_use_share_mem);
    NVCV_CHECK_THROW(cudaGetLastError());

    if constexpr (can_tile)
    {
        if (channels == 3 && all_u8_tiles_fit && storage_disjoint)
        {
            const PillowVarShapeU8TileAdapter<kPlanar> images{cuda::ImageBatchVarShapeWrap<const unsigned char>(inData),
                                                              cuda::ImageBatchVarShapeWrap<unsigned char>(outData)};
            const dim3 tileGrid(util::DivUp(max_width, cvcuda::priv::kPillowU8TileWidth),
                                util::DivUp(max_height, cvcuda::priv::kPillowU8TileHeight), batch);
            pillow_fused_u8_tile<decltype(images), kPlanar ? 4 : 6>
                <<<tileGrid, cvcuda::priv::kPillowU8TileThreads, 0, stream>>>(
                    images, dev.h_k_size, dev.v_k_size, h_bounds_batch_gpu, dev.h_bounds_offset, h_kk_batch_gpu,
                    dev.h_kk_offset, v_bounds_batch_gpu, dev.v_bounds_offset, v_kk_batch_gpu, dev.v_kk_offset);
            NVCV_CHECK_THROW(cudaGetLastError());
            if (ws.cudaMem.ready != nullptr)
                NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
            return;
        }
    }

    if constexpr (kPlanar)
    {
        cuda::ImageBatchVarShapeWrap<const elem_type> src_ptr(inData);
        cuda::ImageBatchVarShapeWrap<elem_type>       dst_ptr(outData);
        switch (channels)
        {
        case 3:
            horizontal_pass_var_shape_planar_channels<3, elem_type, elem_type, Filter>
                <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, h_bounds_batch_gpu, dev.h_bounds_offset,
                    h_kk_batch_gpu, dev.h_kk_offset, init_buffer, round_up, hv_use_share_mem);
            break;
        case 4:
            if constexpr (!std::is_same_v<elem_type, float>)
            {
                horizontal_pass_var_shape_planar_channels<4, elem_type, elem_type, Filter>
                    <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                        src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, h_bounds_batch_gpu,
                        dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, init_buffer, round_up, hv_use_share_mem);
            }
            else
            {
                horizontal_pass_var_shape_planar<elem_type, elem_type, Filter>
                    <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                        src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, channels, h_bounds_batch_gpu,
                        dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, init_buffer, round_up, hv_use_share_mem);
            }
            break;
        default:
            horizontal_pass_var_shape_planar<elem_type, elem_type, Filter>
                <<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
                    src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, channels, h_bounds_batch_gpu,
                    dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, init_buffer, round_up, hv_use_share_mem);
            break;
        }
        NVCV_CHECK_THROW(cudaGetLastError());
        // Same L1-broadcast property as the interleaved vertical: skip coefficient staging.
        vertical_pass_var_shape_planar<elem_type, elem_type, Filter><<<gridSizeV, blockSize, 0, stream>>>(
            ptr_h_out, dst_ptr, filterp, dev.v_k_size, channels, v_bounds_batch_gpu, dev.v_bounds_offset,
            v_kk_batch_gpu, dev.v_kk_offset, init_buffer, round_up, false);
    }
    else
    {
        cuda::ImageBatchVarShapeWrapNHWC<elem_type> src_ptr(inData, channels);
        cuda::ImageBatchVarShapeWrapNHWC<elem_type> dst_ptr(outData, channels);
        // NC = compile-time channel count enables the interleaved channel-vectorized fast path (the
        // kernels gate it per-image on alignment and fall back to the scalar loop otherwise). NC=0 keeps
        // the scalar path for unexpected channel counts.
        auto                                        launch_v = [&](auto nc_const)
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
                        src_ptr, dst_ptr, filterp, dev.h_k_size, dev.v_k_size, h_bounds_batch_gpu, dev.h_bounds_offset,
                        h_kk_batch_gpu, dev.h_kk_offset, v_bounds_batch_gpu, dev.v_bounds_offset, v_kk_batch_gpu,
                        dev.v_kk_offset, init_buffer, round_up, pass_use_share_mem);
                    NVCV_CHECK_THROW(cudaGetLastError());
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
                        const bool   paired_use = pass_use_share_mem && paired_sm <= kShareMemLimit;
                        const dim3   pairedBlock
                            = std::is_same_v<elem_type,
                                             unsigned char> && std::is_same_v<Filter, BicubicFilter> && paired_use
                                   && util::DivUp<int64_t>(max_width, kBlock)
                                              * util::DivUp<int64_t>(max_input_height, kBlock / 2)
                                          < util::DivUp<int64_t>(max_width, 2 * kBlock)
                                                * util::DivUp<int64_t>(max_input_height, kBlock / 4)
                                ? dim3(kBlock / 2, kBlock / 2, 1)
                                : blockSize;
                        dim3 gridSizeHP(util::DivUp(max_width, 2 * pairedBlock.x),
                                        util::DivUp(max_input_height, pairedBlock.y), pass_z);
                        horizontal_pass_var_shape_paired<NC, elem_type, elem_type, Filter>
                            <<<gridSizeHP, pairedBlock, paired_use ? paired_sm : 0, stream>>>(
                                src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, h_bounds_batch_gpu,
                                dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, init_buffer, round_up,
                                paired_use);
                        paired = true;
                    }
                }
                if (!paired)
                {
                    horizontal_pass_var_shape<NC, elem_type, elem_type, Filter>
                        <<<gridSizeH, blockSize, pass_h_sm_size, stream>>>(
                            src_ptr, ptr_h_out, filterp, dev.h_k_size, dev.out_cols, dev.v_k_size, h_bounds_batch_gpu,
                            dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, v_bounds_batch_gpu,
                            dev.v_bounds_offset, v_kk_batch_gpu, dev.v_kk_offset, init_buffer, round_up,
                            pass_use_share_mem);
                }
                NVCV_CHECK_THROW(cudaGetLastError());
                // Upscale batches are bound by the vertical pass's per-element load/store issue
                // rate, where the row vectorization wins; downscale verticals sit at the DRAM
                // ridge and keep the per-pixel kernel.
                if (all_upscale)
                {
                    constexpr int VEC = 4;
                    dim3          gridSizeVV(util::DivUp(util::DivUp(max_width * channels, VEC), blockSize.x),
                                             util::DivUp(max_height, blockSize.y), pass_z);
                    vertical_pass_var_shape_vec<VEC, elem_type, elem_type, Filter>
                        <<<gridSizeVV, blockSize, 0, stream>>>(ptr_h_out, dst_ptr, filterp, dev.v_k_size,
                                                               v_bounds_batch_gpu, dev.v_bounds_offset, v_kk_batch_gpu,
                                                               dev.v_kk_offset, init_buffer, round_up);
                }
                else
                {
                    vertical_pass_var_shape<NC, elem_type, elem_type, Filter>
                        <<<gridSizeV, blockSize, pass_v_sm_size, stream>>>(
                            ptr_h_out, dst_ptr, filterp, dev.h_k_size, dev.v_k_size, h_bounds_batch_gpu,
                            dev.h_bounds_offset, h_kk_batch_gpu, dev.h_kk_offset, v_bounds_batch_gpu,
                            dev.v_bounds_offset, v_kk_batch_gpu, dev.v_kk_offset, init_buffer, round_up,
                            pass_use_share_mem);
                }
            }
        };
        DispatchChannelCount(channels, launch_v);
    }

    NVCV_CHECK_THROW(cudaGetLastError());

    if (ws.cudaMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
}

// ---------------------------------------------------------------------------------------------
// Validation: layout and data-type classification
//
// The legacy path reached its support matrix through GetLegacyDataFormat / GetLegacyDataType.
// Those helpers are reproduced here rather than replaced by direct dtype equality tests, because
// they classify on *bits per channel plus data kind* -- so the 8-bit unsigned case also covers the
// packed TYPE_2U8/3U8/4U8 formats -- and because they *reject* every (kind, width) pair the legacy
// enum had no name for. A gate phrased as "not unsigned-32 and a whole number of bytes" would be
// wider than legacy and would silently admit TYPE_U64 / TYPE_S64.
// ---------------------------------------------------------------------------------------------

enum class Layout
{
    kNHWC,
    kHWC,
    kNCHW,
    kCHW,
};

// The element types the legacy DataType enum could name. PillowResize admits a subset of them
// (see PillowResizeTensor / PillowResizeVarShape below); the rest exist so that the two-stage
// legacy rejection order -- classify first, then apply the operator's own list -- is preserved.
enum class ElemType
{
    kU8,
    kS8,
    kU16,
    kS16,
    kS32,
    kF16,
    kF32,
    kF64,
};

inline size_t ElemSize(ElemType t)
{
    switch (t)
    {
    case ElemType::kU8:
    case ElemType::kS8:
        return 1;
    case ElemType::kU16:
    case ElemType::kS16:
    case ElemType::kF16:
        return 2;
    case ElemType::kS32:
    case ElemType::kF32:
        return 4;
    case ElemType::kF64:
        return 8;
    }
    return 0;
}

inline ElemType ClassifyElemType(int32_t bpc, nvcv::DataKind kind)
{
    switch (kind)
    {
    case nvcv::DataKind::FLOAT:
        if (bpc == 64)
            return ElemType::kF64;
        if (bpc == 32)
            return ElemType::kF32;
        if (bpc == 16)
            return ElemType::kF16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);

    case nvcv::DataKind::SIGNED:
        if (bpc == 8)
            return ElemType::kS8;
        if (bpc == 16)
            return ElemType::kS16;
        if (bpc == 32)
            return ElemType::kS32;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);

    case nvcv::DataKind::UNSIGNED:
        if (bpc == 8)
            return ElemType::kU8;
        if (bpc == 16)
            return ElemType::kU16;
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);

    case nvcv::DataKind::COMPLEX:
    case nvcv::DataKind::UNSPECIFIED:
        break;
    }
    throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                          "Only floating-point, signed integer and unsigned integer data kinds are supported ");
}

inline ElemType ClassifyElemType(const nvcv::DataType &dtype)
{
    auto bpc = dtype.bitsPerChannel();

    // NVCV_PACKING_X32_Y24b8 is a real packing, so the non-uniform width rejection is reachable.
    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return ClassifyElemType(bpc[0], dtype.dataKind());
}

inline ElemType ClassifyElemType(const nvcv::ImageFormat &fmt)
{
    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planeDataType(i) != fmt.planeDataType(0))
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "All planes must have the same data type");
        }
    }

    return ClassifyElemType(fmt.planeDataType(0));
}

inline Layout ClassifyLayout(const nvcv::TensorLayout &layout)
{
    if (layout == nvcv::TENSOR_NCHW)
    {
        return Layout::kNCHW;
    }
    else if (layout == nvcv::TENSOR_CHW)
    {
        return Layout::kCHW;
    }
    else if (layout == nvcv::TENSOR_NHWC)
    {
        return Layout::kNHWC;
    }
    else if (layout == nvcv::TENSOR_HWC)
    {
        return Layout::kHWC;
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }
}

inline Layout ClassifyLayout(const nvcv::ImageBatchVarShape &imgBatch)
{
    nvcv::ImageFormat fmt = cvcuda::priv::legacy_format::RequireUniformVarShapeFormat(imgBatch);

    if (fmt.numPlanes() >= 2)
    {
        return imgBatch.numImages() >= 2 ? Layout::kNCHW : Layout::kCHW;
    }
    else
    {
        return imgBatch.numImages() >= 2 ? Layout::kNHWC : Layout::kHWC;
    }
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

template<typename Filter>
void pillow_resize_filter(const nvcv::TensorDataAccessStridedImagePlanar &inData,
                          const nvcv::TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                          ElemType data_type, bool planar, cudaStream_t stream)
{
    switch (data_type)
    {
    case ElemType::kU8:
        pillow_resize_v2<Filter, unsigned char>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    case ElemType::kS8:
        pillow_resize_v2<Filter, signed char>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case ElemType::kU16:
        pillow_resize_v2<Filter, std::uint16_t>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    case ElemType::kS16:
        pillow_resize_v2<Filter, std::int16_t>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case ElemType::kS32:
        pillow_resize_v2<Filter, int>(inData, outData, gpu_workspace, false, 0., true, planar, stream);
        break;
    case ElemType::kF32:
        pillow_resize_v2<Filter, float>(inData, outData, gpu_workspace, false, 0., false, planar, stream);
        break;
    default:
        break;
    }
}

// Single dtype switch shared by the interleaved and planar paths via the kPlanar template parameter,
// so the launch table is written once.
template<typename Filter, bool kPlanar>
// Only the four types RunPillowResizeVarShape admits appear here. kS8/kS32 reach the tensor
// dispatch but are rejected before this one, so arms for them were 20 unreachable instantiations.
void pillow_resize_dispatch_dtype(const nvcv::ImageBatchVarShape &inData, const nvcv::ImageBatchVarShape &outData,
                                  const cvcuda::Workspace &ws, ElemType data_type, cudaStream_t stream)
{
    switch (data_type)
    {
    case ElemType::kU8:
        pillow_resize_var_shape<Filter, unsigned char, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    case ElemType::kU16:
        pillow_resize_var_shape<Filter, std::uint16_t, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    case ElemType::kS16:
        pillow_resize_var_shape<Filter, std::int16_t, kPlanar>(inData, outData, ws, false, 0., true, stream);
        break;
    case ElemType::kF32:
        pillow_resize_var_shape<Filter, float, kPlanar>(inData, outData, ws, false, 0., false, stream);
        break;
    default:
        break;
    }
}

template<typename Filter>
void pillow_resize_filter_var_shape(const nvcv::ImageBatchVarShape &inData, const nvcv::ImageBatchVarShape &outData,
                                    const cvcuda::Workspace &ws, ElemType data_type, bool isPlanar, cudaStream_t stream)
{
    if (isPlanar)
        pillow_resize_dispatch_dtype<Filter, true>(inData, outData, ws, data_type, stream);
    else
        pillow_resize_dispatch_dtype<Filter, false>(inData, outData, ws, data_type, stream);
}

inline void RunPillowResizeTensor(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                  const NVCVInterpolationType interpolation, cudaStream_t stream,
                                  const cvcuda::Workspace &ws)
{
    Layout format        = ClassifyLayout(inData.layout());
    Layout output_format = ClassifyLayout(outData.layout());

    if (ws.cudaMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaStreamWaitEvent(stream, ws.cudaMem.ready));

    void *gpu_workspace = ws.cudaMem.data;

    if (format != output_format)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input and output; they must match");
    }
    const bool planar = (format == Layout::kNCHW || format == Layout::kCHW);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    // Resize changes rows and columns only: the kernels take elem_type and the channel count from
    // the input and then wrap the output with them, so batch, channels and dtype must agree.
    if (inData.dtype() != outData.dtype() || inAccess->numSamples() != outAccess->numSamples()
        || inAccess->numChannels() != outAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same batch size, channel count and DataType");
    }

    ElemType  data_type = ClassifyElemType(inData.dtype());
    const int in_N = inAccess->numSamples(), in_C = inAccess->numChannels(), in_H = inAccess->numRows();

    if (in_C > 4)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", in_C);
    }

    // Covered by OpPillowResize_Negative.tensor_batch_over_65535_rejected.
    if (in_N > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "PillowResize requires numSamples <= 65535 (CUDA grid-z limit)");
    }

    if (!(data_type == ElemType::kU8 || data_type == ElemType::kS8 || data_type == ElemType::kU16
          || data_type == ElemType::kS16 || data_type == ElemType::kS32 || data_type == ElemType::kF32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    // The kernels compute addresses as 32-bit products of sample/row strides (InterleavedPtr/PlanarPtr),
    // so any tensor -- input, output, or the horizontally-resized intermediate -- whose byte extent
    // exceeds INT32_MAX overflows the addressing. Reject those instead of corrupting memory.
    {
        constexpr int64_t kMaxByteExtent = std::numeric_limits<int32_t>::max();
        const int64_t     interExtent
            = static_cast<int64_t>(in_N) * in_C * in_H * outAccess->numCols() * ElemSize(data_type);
        const int64_t maxExtent = std::max(
            {inAccess->numSamples() * inAccess->sampleStride(), outAccess->numSamples() * outAccess->sampleStride(),
             inAccess->numRows() * inAccess->rowStride(), outAccess->numRows() * outAccess->rowStride(), interExtent});
        if (maxExtent > kMaxByteExtent)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Tensor byte extent %ld exceeds the 32-bit addressing limit %ld; split the batch "
                                  "into smaller submissions",
                                  static_cast<long>(maxExtent), static_cast<long>(kMaxByteExtent));
        }
    }

    DispatchFilter(interpolation,
                   [&](auto filterTag)
                   {
                       pillow_resize_filter<typename decltype(filterTag)::type>(*inAccess, *outAccess, gpu_workspace,
                                                                                data_type, planar, stream);
                   });

    if (ws.cudaMem.ready != nullptr)
        NVCV_CHECK_THROW(cudaEventRecord(ws.cudaMem.ready, stream));
}

inline void RunPillowResizeVarShape(const nvcv::ImageBatchVarShape &inDataBase,
                                    const nvcv::ImageBatchVarShape &outDataBase,
                                    const NVCVInterpolationType interpolation, cudaStream_t stream,
                                    const cvcuda::Workspace &ws)
{
    if (!inDataBase.uniqueFormat() || !outDataBase.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in input and output batch must all have the same format");
    }

    if (inDataBase.uniqueFormat() != outDataBase.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataFormat between input and output");
    }

    // batch is taken from the input and used as the launch's z extent against the output batch.
    if (inDataBase.numImages() != outDataBase.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input and output must have the same number of images, got %d and %d",
                              inDataBase.numImages(), outDataBase.numImages());
    }

    Layout format = ClassifyLayout(inDataBase);

    const bool isPlanar = (format == Layout::kNCHW || format == Layout::kCHW);

    int channels = inDataBase.uniqueFormat().numChannels();

    // Planar 2-channel layout is rejected: there is no defined 2-plane planar format (matches the
    // Resize and Normalize operators).
    if (channels > 4 || (isPlanar && channels == 2))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }

    // Covered by OpPillowResize_Negative.varshape_batch_over_65535_rejected.
    if (inDataBase.numImages() > 65535)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "PillowResize requires numImages <= 65535 (CUDA grid-z limit)");
    }

    ElemType data_type = ClassifyElemType(inDataBase.uniqueFormat());

    if (!(data_type == ElemType::kU8 || data_type == ElemType::kU16 || data_type == ElemType::kS16
          || data_type == ElemType::kF32))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType");
    }

    // The horizontally-resized intermediate is one dense image slot per batch entry
    // (numImages x maxInH x maxOutW x C), and the kernels compute its per-image offsets as 32-bit
    // products (InterleavedPtr). Reject batches whose intermediate exceeds INT32_MAX bytes instead of
    // overflowing the addressing and corrupting memory.
    {
        constexpr int64_t kMaxByteExtent = std::numeric_limits<int32_t>::max();
        const int64_t     interExtent    = static_cast<int64_t>(inDataBase.numImages()) * inDataBase.maxSize().h
                                  * outDataBase.maxSize().w * channels * ElemSize(data_type);
        if (interExtent > kMaxByteExtent)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Intermediate byte extent %ld exceeds the 32-bit addressing limit %ld; split the "
                                  "batch into smaller submissions",
                                  static_cast<long>(interExtent), static_cast<long>(kMaxByteExtent));
        }
    }

    DispatchFilter(interpolation,
                   [&](auto filterTag)
                   {
                       pillow_resize_filter_var_shape<typename decltype(filterTag)::type>(inDataBase, outDataBase, ws,
                                                                                          data_type, isPlanar, stream);
                   });
}

// ---------------------------------------------------------------------------------------------
// Workspace sizing
//
// Kept exactly as the two legacy operators sized it -- the tensor path with max_support = 1 and the
// var-shape path with max_support = 3 (cubic needs 2 and Lanczos 3; worst case) -- and combined with
// MaxWorkspaceReq, because the public getWorkspaceRequirements entry points are shared by both.
// ---------------------------------------------------------------------------------------------

inline double CoeffRegionBytes(int max_support, int inH, int outH, int inW, int outW)
{
    return outH * (((1.0 * inH / outH + 1) * max_support * 2 + 1) * sizeof(work_type) + 2 * sizeof(int))
         + outW * (((1.0 * inW / outW + 1) * max_support * 2 + 1) * sizeof(work_type) + 2 * sizeof(int));
}

inline cvcuda::WorkspaceRequirements TensorWorkspaceRequirements(int N, int C, int inH, int inW, int outH, int outW,
                                                                 ElemType data_type)
{
    int    max_support = 1; //3
    size_t size        = std::ceil(CoeffRegionBytes(max_support, inH, outH, inW, outW))
                + static_cast<size_t>(N) * C * inH * outW * ElemSize(data_type)
                + kInterAlign; // padding to align the intermediate buffer (vector loads/stores)
    cvcuda::WorkspaceRequirements req{};
    req.cudaMem = {size, 256};
    return req;
}

inline cvcuda::WorkspaceRequirements VarShapeWorkspaceRequirements(int N, int C, int inH, int inW, int outH, int outW)
{
    constexpr size_t kDefaultDeviceAlignment = 256;

    cvcuda::WorkspaceRequirements req{};

    int    max_support = 3; // Needed for various filters: cubic needs 2 and Lanczos needs 3. Just use worst case.
    size_t size        = std::ceil(CoeffRegionBytes(max_support, inH, outH, inW, outW));

    const size_t metadata_buffer_size = VarShapeMeta::kBytesPerImage * N;
    size_t       buffer_size          = metadata_buffer_size + size * N;

    req.hostMem.size      = metadata_buffer_size;
    req.hostMem.alignment = alignof(std::max_align_t);

    buffer_size += static_cast<size_t>(N) * C * inH * outW * sizeof(float);
    buffer_size += kInterAlign; // padding to align the float intermediate buffer (vector loads/stores)

    req.cudaMem.size      = buffer_size;
    req.cudaMem.alignment = kDefaultDeviceAlignment;

    return req;
}

} // namespace

namespace cvcuda::priv {

PillowResize::PillowResize() = default;

WorkspaceRequirements PillowResize::getWorkspaceRequirements(int batchSize, const nvcv::Size2D *in_sizes,
                                                             const nvcv::Size2D *out_sizes, NVCVImageFormat fmt)
{
    nvcv::Size2D maxInSize{0, 0};
    nvcv::Size2D maxOutSize{0, 0};
    for (int i = 0; i < batchSize; i++)
    {
        maxInSize  = nvcv::MaxSize(in_sizes[i], maxInSize);
        maxOutSize = nvcv::MaxSize(out_sizes[i], maxOutSize);
    }
    return getWorkspaceRequirements(batchSize, maxInSize, maxOutSize, fmt);
}

WorkspaceRequirements PillowResize::getWorkspaceRequirements(int maxBatchSize, nvcv::Size2D maxInSize,
                                                             nvcv::Size2D maxOutSize, NVCVImageFormat fmt)
{
    std::array<int32_t, 4> bpc;
    nvcvImageFormatGetBitsPerChannel(fmt, bpc.data());
    int32_t maxChannel = 0;
    nvcvImageFormatGetNumChannels(fmt, &maxChannel);
    NVCVDataKind dataKind;
    nvcvImageFormatGetDataKind(fmt, &dataKind);
    auto     dkind    = static_cast<nvcv::DataKind>(dataKind);
    ElemType dataType = ClassifyElemType(bpc[0], dkind);

    auto req = TensorWorkspaceRequirements(maxBatchSize, maxChannel, maxInSize.h, maxInSize.w, maxOutSize.h,
                                           maxOutSize.w, dataType);
    auto reqVarShape
        = VarShapeWorkspaceRequirements(maxBatchSize, maxChannel, maxInSize.h, maxInSize.w, maxOutSize.h, maxOutSize.w);

    return MaxWorkspaceReq(req, reqVarShape);
}

void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::Tensor &in, const nvcv::Tensor &out,
                              const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::PillowResize::operator()[Tensor]");
    auto inData = in.exportData<nvcv::TensorDataStridedCuda>();
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-acessible, pitch-linear tensor");
    }

    auto outData = out.exportData<nvcv::TensorDataStridedCuda>();
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-acessible, pitch-linear tensor");
    }

    if (const nvcv::TensorLayout layout = inData->layout(); layout == nvcv::TENSOR_NCHW || layout == nvcv::TENSOR_CHW)
    {
        if (outData->layout() != layout)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output tensors must have the same layout");
        }

        auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        if (!inAccess || !outAccess)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible tensor layout");
        }

        if (inAccess->numChannels() != outAccess->numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                  "Input and output tensors must have the same number of channels");
        }
        // 2-channel planar is unsupported (matches Normalize and the var-shape planar path); 1, 3 and
        // 4 channel planes are valid.
        if (inAccess->numChannels() < 1 || inAccess->numChannels() > 4 || inAccess->numChannels() == 2)
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid number of channels");
        }
    }

    // Planar (NCHW/CHW) tensors resize natively through the same runner as the interleaved layouts:
    // one grid z-slice per sample, with the C channel planes looped inside the kernel so the
    // per-output-pixel filter setup is shared across channels. Plane strides are honored, so there is
    // no tight-packing requirement. The planar preconditions above stay in this function rather than
    // moving into the runner: the runner's own checks fire in a fixed order (and after it waits on
    // the workspace event), so relocating them would change which message a bad input produces.
    RunPillowResizeTensor(*inData, *outData, interpolation, stream, ws);
}

void PillowResize::operator()(cudaStream_t stream, const Workspace &ws, const nvcv::ImageBatchVarShape &in,
                              const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::PillowResize::operator()[ImageBatchVarShape]");
    RunPillowResizeVarShape(in, out, interpolation, stream, ws);
}

} // namespace cvcuda::priv
