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

#include "LegacyFormatClassify.hpp"
#include "Nvtx.hpp"
#include "OpRandomResizedCrop.hpp"
#include "TensorAccessValidation.hpp"

#include "PixelLayoutCommon.cuh"

#include <cuda_fp16.h>
#include <cvcuda/cuda_tools/ImageBatchVarShapeWrap.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/TensorWrap.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/CheckError.hpp>
#include <nvcv/util/Math.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <type_traits>

namespace {

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

using uchar  = unsigned char;
using ushort = unsigned short;

using cvcuda::priv::legacy_format::ClassifyDataFormat;
using cvcuda::priv::legacy_format::ClassifyDataType;
using cvcuda::priv::legacy_format::DataFormatCode;
using cvcuda::priv::legacy_format::DataTypeCode;
using cvcuda::priv::legacy_format::IsPlanar;

// The five dtypes the operator admits.  Every one of them interpolates in FP32 and rounds once on
// store, so F16 gets real half kernels rather than aliasing the 16-bit integer ones.
inline bool IsSupportedDataType(DataTypeCode code)
{
    return code == DataTypeCode::kU8 || code == DataTypeCode::kU16 || code == DataTypeCode::kS16
        || code == DataTypeCode::kF32 || code == DataTypeCode::kF16;
}

// Vectorized destination stores ------------------------------------------------------------------
//
// Same shape as the equivalents in OpResize.cu: 8-bit pixels are narrow enough that NIX of them fit
// in one uint3/uint4 store, so a thread can commit a whole run at once. Wider types stay scalar.

// Three-channel pixels need uint3-sized stores; other packed 8-bit pixels use uint4 to keep the
// vectorized write width naturally aligned.
template<typename T>
using DPT = std::conditional_t<cuda::NumElements<T> == 3, uint3, uint4>;

// Number of logical pixels carried by one vectorized destination store.
template<typename T>
constexpr int NIX = sizeof(DPT<T>) / sizeof(T);

// uint3 has 4-byte alignment despite its 12-byte size; uint4 uses its full width.
template<typename T>
constexpr unsigned int MSK = (sizeof(DPT<T>) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(DPT<T>)) - 1;

// True for 8-bit base types.  Gates two separate things: the packed linear store width, and the
// row-pointer hoisting in the cubic kernel (which packs nothing -- OpResize.cu calls that predicate
// USE_ROW_PTR).  The name is inherited from the legacy RRC_USE_NIX.
template<typename T>
constexpr bool USE_NIX = sizeof(cuda::BaseType<T>) == 1;

template<typename T>
__device__ __forceinline__ void WritePack(T &u, const T (&v)[NIX<T>])
{
    reinterpret_cast<DPT<T> &>(u) = reinterpret_cast<const DPT<T> &>(v);
}

template<typename T>
__device__ __forceinline__ bool CheckRowAlign(T *row)
{
    return (static_cast<unsigned int>(reinterpret_cast<size_t>(row)) & MSK<T>) == 0;
}

// Launch geometry ---------------------------------------------------------------------------------

// Interleaved kernels give one thread one destination pixel (or one NIX-wide run of them).
constexpr int kBlockDimX = 32;
constexpr int kBlockDimY = kBlockDimX / 4;

// Planar kernels are sized for a 32-byte contiguous x-run per warp row instead, so the block shape
// depends on the element width: 32x8 for 8-bit, 16x16 for 16-bit, 8x32 for 32-bit.
constexpr int kPlanarThreadsPerBlock = 256;

// Both paths carry one sample per grid-z entry, so both are capped by the CUDA grid-z limit; only
// the planar one rejects an over-large batch up front, as the legacy operator did.  On the
// interleaved path an over-large batch surfaces as a launch failure through cudaGetLastError.
constexpr int64_t kMaxGridZ = 65535;

template<typename T>
inline dim3 PlanarBlockSize()
{
    int width = 32 / static_cast<int>(sizeof(T));
    if (width < 1)
    {
        width = 1;
    }
    return dim3(width, kPlanarThreadsPerBlock / width, 1);
}

// Per-sample crop parameters ----------------------------------------------------------------------

struct alignas(16) SampleCrop
{
    float scaleX;
    float scaleY;
    int   top;
    int   left;
};

static_assert(sizeof(SampleCrop) == 2 * sizeof(float) + 2 * sizeof(int));

struct CropParams
{
    SampleCrop *samples;

    __device__ __forceinline__ SampleCrop operator[](int sample) const
    {
        return samples[sample];
    }
};

inline size_t CropParamBufferSize(int batch)
{
    return sizeof(SampleCrop) * static_cast<size_t>(batch);
}

inline CropParams CarveCropParams(std::byte *storage)
{
    return {reinterpret_cast<SampleCrop *>(storage)};
}

// Draws one crop window, following torchvision's RandomResizedCrop.get_params: up to ten attempts at
// a (scale, log-ratio) pair that fits inside the image, then a deterministic central-crop fallback.
//
// This is a verbatim port of the legacy operator's draw sequence and must stay one.  For a fixed
// seed the output pixels are a function of which distributions are constructed, where they are
// constructed, and the order of the draws -- two per attempt, plus two more on the attempt that
// succeeds.  Any restructuring that changes that sequence changes the operator's output.
inline void GetCropParams(std::mt19937 &generator, double minScale, double maxScale, double minRatio, double maxRatio,
                          int input_rows, int input_cols, int *top_indices, int *left_indices, int *crop_rows,
                          int *crop_cols)
{
    int                                    rows          = input_rows;
    int                                    cols          = input_cols;
    double                                 area          = rows * cols;
    const double                           log_min_ratio = std::log(minRatio);
    const double                           log_max_ratio = std::log(maxRatio);
    std::uniform_real_distribution<double> scale_dist(minScale, maxScale);
    std::uniform_real_distribution<double> ratio_dist(log_min_ratio, log_max_ratio);
    bool                                   got_params = false;
    for (int i = 0; i < 10; ++i)
    {
        if (got_params)
            return;
        int    target_area  = area * scale_dist(generator);
        double aspect_ratio = std::exp(ratio_dist(generator));

        *crop_cols = int(std::round(std::sqrt(target_area * aspect_ratio)));
        *crop_rows = int(std::round(std::sqrt(target_area / aspect_ratio)));

        if (*crop_cols > 0 && *crop_cols <= cols && *crop_rows > 0 && *crop_rows <= rows)
        {
            std::uniform_int_distribution<int> row_uni(0, rows - *crop_rows);
            std::uniform_int_distribution<int> col_uni(0, cols - *crop_cols);
            *top_indices  = row_uni(generator);
            *left_indices = col_uni(generator);
            got_params    = true;
        }
    }
    // Fallback to central crop
    if (!got_params)
    {
        double in_ratio = double(cols) / double(rows);
        if (in_ratio < minRatio)
        {
            *crop_cols = cols;
            *crop_rows = int(std::round(*crop_cols / minRatio));
        }
        else if (in_ratio > maxRatio)
        {
            *crop_rows = rows;
            *crop_cols = int(std::round(*crop_rows * maxRatio));
        }
        else // whole image
        {
            *crop_cols = cols;
            *crop_rows = rows;
        }
        *top_indices  = (rows - *crop_rows) / 2;
        *left_indices = (cols - *crop_cols) / 2;
    }
}

// Sample geometry ---------------------------------------------------------------------------------
//
// The tensor path has one size for the whole batch and the var-shape path one per image.  Selecting
// between them with a policy type keeps the choice at compile time, so each kernel still emits
// exactly the two specializations the two legacy translation units emitted between them.

struct UniformGeometry
{
    int2 src;
    int2 dst;

    template<class W>
    __device__ __forceinline__ int2 srcSize(const W &, int) const
    {
        return src;
    }

    template<class W>
    __device__ __forceinline__ int2 dstSize(const W &, int) const
    {
        return dst;
    }
};

struct VarShapeGeometry
{
    template<class W>
    __device__ __forceinline__ int2 srcSize(const W &w, int b) const
    {
        return int2{w.width(b), w.height(b)};
    }

    template<class W>
    __device__ __forceinline__ int2 dstSize(const W &w, int b) const
    {
        return int2{w.width(b), w.height(b)};
    }
};

// Source sampling ---------------------------------------------------------------------------------

// One axis of a bilinear tap: the clamped index of the near sample and the weight of the far one.
struct LinearTap
{
    int   s;
    float f;
};

// Maps a destination coordinate back through the crop window. Both neighbours (s and s + 1) are
// always read, so s is clamped to extent - 2 and the weight saturates to 0 or 1 outside the image.
__device__ __forceinline__ LinearTap LinearSourceTap(int dstIdx, float scale, int offset, int extent)
{
    float f = (float)((dstIdx + 0.5f) * scale - 0.5f + offset);
    int   s = cuda::round<cuda::RoundMode::DOWN, int>(f);

    f = ((s < 0) ? 0 : ((s > extent - 2) ? 1 : f - s));
    s = cuda::max(0, cuda::min(s, extent - 2));

    return {s, f};
}

// Catmull-Rom (A = -0.75), matching OpenCV's interpolateCubic. Fills the four tap weights and
// returns the index of the second tap; callers clamp the four positions against the image.
__device__ __forceinline__ int CubicSourceTaps(int dstIdx, float scale, int offset, float (&c)[4])
{
    float f = (float)((dstIdx + 0.5f) * scale - 0.5f + offset);
    int   s = cuda::round<cuda::RoundMode::DOWN, int>(f);
    f -= s;

    const float A = -0.75f;

    c[0] = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A;
    c[1] = ((A + 2) * f - (A + 3)) * f * f + 1;
    c[2] = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1;
    c[3] = 1.f - c[0] - c[1] - c[2];

    return s;
}

// Interleaved (NHWC / HWC) kernels -----------------------------------------------------------------

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeLinear(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) && (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const LinearTap ty = LinearSourceTap(dst_y, crop.scaleY, crop.top, srcSize.y);
        const LinearTap tx = LinearSourceTap(dst_x, crop.scaleX, crop.left, srcSize.x);

        const T *aPtr = src.ptr(batch_idx, ty.s, 0);     // upper row
        const T *bPtr = src.ptr(batch_idx, ty.s + 1, 0); // lower row

        *dst.ptr(batch_idx, dst_y, dst_x)
            = cuda::SaturateCast<T>((1.0f - tx.f) * (aPtr[tx.s] * (1.0f - ty.f) + bPtr[tx.s] * ty.f)
                                    + tx.f * (aPtr[tx.s + 1] * (1.0f - ty.f) + bPtr[tx.s + 1] * ty.f));
    }
}

template<typename T>
__device__ __forceinline__ auto LinearPixelValue(T value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    if constexpr (std::is_same_v<T, uchar3>)
    {
        return make_float3(__fsub_rn(__int_as_float(0x4b000000u | value.x), 8388608.0f),
                           __fsub_rn(__int_as_float(0x4b000000u | value.y), 8388608.0f),
                           __fsub_rn(__int_as_float(0x4b000000u | value.z), 8388608.0f));
    }
    else
    {
        return value;
    }
#else
    return value;
#endif
}

// 8-bit linear: one thread owns a contiguous run of NIX destination pixels sharing a source row
// pair, staged in registers and committed with one vector store.
template<int NIX_, class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeLinearPacked(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params)
{
    const int dst_x0    = (blockIdx.x * blockDim.x + threadIdx.x) * NIX_;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if (dst_y < dstSize.y)
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const LinearTap ty = LinearSourceTap(dst_y, crop.scaleY, crop.top, srcSize.y);

        const T *aPtr = src.ptr(batch_idx, ty.s, 0);
        const T *bPtr = src.ptr(batch_idx, ty.s + 1, 0);

        T dstPack[NIX_];
#pragma unroll
        for (int i = 0; i < NIX_; ++i)
        {
            const int dst_x = dst_x0 + i;
            if (dst_x < dstSize.x)
            {
                const LinearTap tx = LinearSourceTap(dst_x, crop.scaleX, crop.left, srcSize.x);

                dstPack[i] = cuda::SaturateCast<T>(
                    (1.0f - tx.f) * (LinearPixelValue(aPtr[tx.s]) * (1.0f - ty.f) + LinearPixelValue(bPtr[tx.s]) * ty.f)
                    + tx.f
                          * (LinearPixelValue(aPtr[tx.s + 1]) * (1.0f - ty.f)
                             + LinearPixelValue(bPtr[tx.s + 1]) * ty.f));
            }
        }

        T *dstRow = dst.ptr(batch_idx, dst_y, 0);
        // dst_x0 advances by NIX_, so an aligned row base keeps the packed write aligned.  Var-shape
        // rows start at arbitrary offsets, so the alignment has to be checked per row.
        if (dst_x0 + NIX_ - 1 < dstSize.x && CheckRowAlign(dstRow))
        {
            WritePack(dstRow[dst_x0], dstPack);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < NIX_; ++i)
            {
                const int dst_x = dst_x0 + i;
                if (dst_x < dstSize.x)
                {
                    dstRow[dst_x] = dstPack[i];
                }
            }
        }
    }
}

template<class Geometry, class SrcWrapper, class DstWrapper>
__global__ void ResizeNearest(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) && (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * crop.scaleX) + crop.left, srcSize.x - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * crop.scaleY) + crop.top, srcSize.y - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
}

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeCubic(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) & (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float     cY[4];
        const int sy = CubicSourceTaps(dst_y, crop.scaleY, crop.top, cY);

        float     cX[4];
        const int sx = CubicSourceTaps(dst_x, crop.scaleX, crop.left, cX);

        work_type accum = cuda::SetAll<work_type>(0);

        if constexpr (USE_NIX<T>)
        {
            // 8-bit taps are cheap enough that the 16 full address computations dominate; hoisting
            // one row pointer per tap row trades them for 4.  Same weights, same accumulation order.
            int csy[4];
            int csx[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                csy[k] = cuda::clamp(sy + k - 1, 0, srcSize.y - 1);
                csx[k] = cuda::clamp(sx + k - 1, 0, srcSize.x - 1);
            }

#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
                const T *srcRow = src.ptr(batch_idx, csy[ky], 0);
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    accum += cY[ky] * cX[kx] * srcRow[csx[kx]];
                }
            }
        }
        else
        {
#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
                int csy = cuda::clamp(sy + ky - 1, 0, srcSize.y - 1);
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    int csx = cuda::clamp(sx + kx - 1, 0, srcSize.x - 1);
                    accum += cY[ky] * cX[kx] * *src.ptr(batch_idx, csy, csx);
                }
            }
        }

        *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
    }
}

// Planar (NCHW / CHW) kernels ----------------------------------------------------------------------
//
// Planar kernels are templated on the scalar element type and take the channel count at runtime, so
// one instantiation per dtype covers every channel count.

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeLinearPlanar(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params,
                                   int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) && (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const LinearTap ty = LinearSourceTap(dst_y, crop.scaleY, crop.top, srcSize.y);
        const LinearTap tx = LinearSourceTap(dst_x, crop.scaleX, crop.left, srcSize.x);

        for (int plane = 0; plane < channels; ++plane)
        {
            const T *aPtr = src.ptr(batch_idx, plane, ty.s, 0);
            const T *bPtr = src.ptr(batch_idx, plane, ty.s + 1, 0);

            *dst.ptr(batch_idx, plane, dst_y, dst_x)
                = cuda::SaturateCast<T>((1.0f - tx.f) * (aPtr[tx.s] * (1.0f - ty.f) + bPtr[tx.s] * ty.f)
                                        + tx.f * (aPtr[tx.s + 1] * (1.0f - ty.f) + bPtr[tx.s + 1] * ty.f));
        }
    }
}

template<class Geometry, typename T>
__device__ __forceinline__ auto LinearPlanarInputValue(T value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    if constexpr (std::is_same_v<Geometry, UniformGeometry> && std::is_same_v<T, uchar>)
    {
        return __fsub_rn(__int_as_float(0x4b000000u | value), 8388608.0f);
    }
    else
#endif
    {
        return value;
    }
}

// 8-bit planar linear.  Unlike the interleaved packed kernel this does not vectorize the store: the
// thread's PNIX outputs are blockDim.x apart, so the warp writes contiguously while each thread
// writes single bytes.  The win is that one thread owns PNIX columns, so the y tap and the per-plane
// row pointers are amortized over that many outputs.
template<int PNIX, class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeLinearPlanarStrided(const SrcWrapper src, DstWrapper dst, const Geometry geom,
                                          const CropParams params, int channels)
{
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if (dst_y < dstSize.y)
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const LinearTap ty = LinearSourceTap(dst_y, crop.scaleY, crop.top, srcSize.y);

        const int xBase = blockIdx.x * blockDim.x * PNIX + threadIdx.x;
        if constexpr (std::is_same_v<Geometry, VarShapeGeometry>)
        {
            if (xBase >= dstSize.x)
                return;
        }

        int       dx[PNIX];
        LinearTap tx[PNIX];
#pragma unroll
        for (int i = 0; i < PNIX; ++i)
        {
            dx[i] = xBase + i * static_cast<int>(blockDim.x);
            tx[i] = LinearSourceTap(dx[i], crop.scaleX, crop.left, srcSize.x);
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            const T *aPtr = src.ptr(batch_idx, plane, ty.s, 0);
            const T *bPtr = src.ptr(batch_idx, plane, ty.s + 1, 0);

            T *outputColumns = nullptr;
            if constexpr (std::is_same_v<Geometry, VarShapeGeometry>)
                outputColumns = dst.ptr(batch_idx, plane, dst_y, xBase);
#pragma unroll
            for (int i = 0; i < PNIX; ++i)
            {
                if (dx[i] >= dstSize.x)
                    continue;

                if constexpr (std::is_same_v<Geometry, VarShapeGeometry>)
                {
                    T *outputPixel = outputColumns + i * static_cast<int>(blockDim.x);
                    *outputPixel   = cuda::SaturateCast<T>(
                        (1.0f - tx[i].f) * (aPtr[tx[i].s] * (1.0f - ty.f) + bPtr[tx[i].s] * ty.f)
                        + tx[i].f * (aPtr[tx[i].s + 1] * (1.0f - ty.f) + bPtr[tx[i].s + 1] * ty.f));
                }
                else
                {
                    *dst.ptr(batch_idx, plane, dst_y, dx[i]) = cuda::SaturateCast<T>(
                        (1.0f - tx[i].f)
                            * (LinearPlanarInputValue<Geometry>(aPtr[tx[i].s]) * (1.0f - ty.f)
                               + LinearPlanarInputValue<Geometry>(bPtr[tx[i].s]) * ty.f)
                        + tx[i].f
                              * (LinearPlanarInputValue<Geometry>(aPtr[tx[i].s + 1]) * (1.0f - ty.f)
                                 + LinearPlanarInputValue<Geometry>(bPtr[tx[i].s + 1]) * ty.f));
                }
            }
        }
    }
}

template<class Geometry, class SrcWrapper, class DstWrapper>
__global__ void ResizeNearestPlanar(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params,
                                    int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) && (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        const int sx = cuda::min(__float2int_rd((dst_x + 0.5f) * crop.scaleX) + crop.left, srcSize.x - 1);
        const int sy = cuda::min(__float2int_rd((dst_y + 0.5f) * crop.scaleY) + crop.top, srcSize.y - 1);

        for (int plane = 0; plane < channels; ++plane)
        {
            *dst.ptr(batch_idx, plane, dst_y, dst_x) = *src.ptr(batch_idx, plane, sy, sx);
        }
    }
}

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void ResizeCubicPlanar(const SrcWrapper src, DstWrapper dst, const Geometry geom, const CropParams params,
                                  int channels)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    const int2 dstSize = geom.dstSize(dst, batch_idx);

    if ((dst_x < dstSize.x) & (dst_y < dstSize.y))
    {
        const int2 srcSize = geom.srcSize(src, batch_idx);

        const SampleCrop crop = params[batch_idx];

        using work_type = cuda::ConvertBaseTypeTo<float, T>;

        float     cY[4];
        const int sy = CubicSourceTaps(dst_y, crop.scaleY, crop.top, cY);

        float     cX[4];
        const int sx = CubicSourceTaps(dst_x, crop.scaleX, crop.left, cX);

        // The tap positions do not depend on the plane, so the clamps are solved once for all of
        // them.
        int csy[4];
        int csx[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            csy[k] = cuda::clamp(sy + k - 1, 0, srcSize.y - 1);
            csx[k] = cuda::clamp(sx + k - 1, 0, srcSize.x - 1);
        }

        for (int plane = 0; plane < channels; ++plane)
        {
            work_type accum = cuda::SetAll<work_type>(0);

#pragma unroll
            for (int ky = 0; ky < 4; ++ky)
            {
#pragma unroll
                for (int kx = 0; kx < 4; ++kx)
                {
                    accum += cY[ky] * cX[kx] * *src.ptr(batch_idx, plane, csy[ky], csx[kx]);
                }
            }

            *dst.ptr(batch_idx, plane, dst_y, dst_x) = cuda::SaturateCast<T>(accum);
        }
    }
}

// Launchers -----------------------------------------------------------------------------------------

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
void LaunchResize(cudaStream_t stream, const SrcWrapper &src, const DstWrapper &dst, const Geometry &geom,
                  int2 gridExtent, int batchSize, const CropParams &params, NVCVInterpolationType interpolation)
{
    dim3 block(kBlockDimX, kBlockDimY, 1);
    dim3 grid(util::DivUp(gridExtent.x, kBlockDimX), util::DivUp(gridExtent.y, kBlockDimY), batchSize);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        constexpr int kNix = USE_NIX<T> ? NIX<T> : 1;
        if constexpr (kNix > 1)
        {
            // Each thread covers kNix columns, so the x grid shrinks by the same factor.
            dim3 packedGrid(util::DivUp(gridExtent.x, kBlockDimX * kNix), util::DivUp(gridExtent.y, kBlockDimY),
                            batchSize);
            ResizeLinearPacked<kNix><<<packedGrid, block, 0, stream>>>(src, dst, geom, params);
        }
        else
        {
            ResizeLinear<<<grid, block, 0, stream>>>(src, dst, geom, params);
        }
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        ResizeNearest<<<grid, block, 0, stream>>>(src, dst, geom, params);
    }
    else
    {
        ResizeCubic<<<grid, block, 0, stream>>>(src, dst, geom, params);
    }
}

template<class Geometry, class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
void LaunchResizePlanar(cudaStream_t stream, const SrcWrapper &src, const DstWrapper &dst, const Geometry &geom,
                        int2 gridExtent, int batchSize, int channels, const CropParams &params,
                        NVCVInterpolationType interpolation)
{
    dim3 block = PlanarBlockSize<T>();
    dim3 grid(util::DivUp(gridExtent.x, static_cast<int>(block.x)),
              util::DivUp(gridExtent.y, static_cast<int>(block.y)), batchSize);

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        // The planar element type is the scalar one, so this is 4 only for 8-bit data.
        constexpr int kPnix = (sizeof(T) == 1) ? 4 : 1;
        if constexpr (kPnix > 1)
        {
            dim3 stridedGrid(util::DivUp(gridExtent.x, static_cast<int>(block.x) * kPnix),
                             util::DivUp(gridExtent.y, static_cast<int>(block.y)), batchSize);
            ResizeLinearPlanarStrided<kPnix><<<stridedGrid, block, 0, stream>>>(src, dst, geom, params, channels);
        }
        else
        {
            ResizeLinearPlanar<<<grid, block, 0, stream>>>(src, dst, geom, params, channels);
        }
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        ResizeNearestPlanar<<<grid, block, 0, stream>>>(src, dst, geom, params, channels);
    }
    else
    {
        ResizeCubicPlanar<<<grid, block, 0, stream>>>(src, dst, geom, params, channels);
    }
}

// Dispatch ------------------------------------------------------------------------------------------

// Calls cb with a value-initialized pixel type for the given channel count.  Channels 1, 3 and 4 are
// the only ones that reach here: the legacy [dtype][channels-1] tables held a null entry for two
// channels, and the caller rejects that count before dispatching.
template<typename BaseType, class Cb>
inline void DispatchByChannels(int channels, const Cb &cb)
{
    switch (channels)
    {
    case 1:
        return cb(BaseType{});
    case 3:
        return cb(cuda::MakeType<BaseType, 3>{});
    default: // channels == 4
        // Stands in for the legacy NVCV_ASSERT(func != 0) on a null dispatch-table entry.
        NVCV_ASSERT(channels == 4);
        return cb(cuda::MakeType<BaseType, 4>{});
    }
}

// Calls cb with a value-initialized scalar of the element type that code names.  Planar kernels use
// it directly (they template on the scalar and take the channel count at runtime); interleaved
// kernels chain it into DispatchByChannels, which folds the channel count into the pixel type.
//
// The kF32 arm is the default because IsSupportedDataType runs first and admits nothing else -- the
// legacy function-pointer tables this replaces reserved entries for kS8, kS32 and kF64 that the
// same gate made unreachable.
template<class Cb>
inline void DispatchByDataType(DataTypeCode code, const Cb &cb)
{
    switch (code)
    {
    case DataTypeCode::kU8:
        return cb(uchar{});
    case DataTypeCode::kU16:
        return cb(ushort{});
    case DataTypeCode::kS16:
        return cb(short{});
    case DataTypeCode::kF16:
        return cb(__half{});
    default: // DataTypeCode::kF32
        NVCV_ASSERT(code == DataTypeCode::kF32);
        return cb(float{});
    }
}

// Shared validation ---------------------------------------------------------------------------------

inline void ValidateChannelCount(int channels)
{
    if (channels > 4 || channels == 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid channel number %d", channels);
    }
}

inline void ValidateDataTypes(DataTypeCode inCode, DataTypeCode outCode)
{
    if (!IsSupportedDataType(inCode))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid DataType %d", static_cast<int>(inCode));
    }
    if (inCode != outCode)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "DataType of input and output must be equal, but got %d and %d", static_cast<int>(inCode),
                              static_cast<int>(outCode));
    }
}

// See OpRandomResizedCrop_Negative.linear_interp_requires_2x2_source (and its var-shape twin).
inline void ValidateSourceAtLeast2x2(int width, int height)
{
    if (width < 2 || height < 2)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Linear interpolation requires source dimensions of at least 2x2");
    }
}

inline void ValidateBatchSize(int batch, int32_t maxBatchSize)
{
    if (maxBatchSize <= 0 || batch > maxBatchSize)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid maximum batch size %d", maxBatchSize);
    }
}

// The tensor wrappers index samples with int32 offsets, so the whole tensor has to fit.
inline void ValidateTensorExtent(int64_t srcMaxStride, int64_t dstMaxStride)
{
    constexpr int32_t kIntMax = cuda::TypeTraits<int32_t>::max;
    if (std::max(srcMaxStride, dstMaxStride) > kIntMax)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input or output size exceeds %d. Tensor is too large.", kIntMax);
    }
}

inline void ValidatePlanarBatch(int64_t batch, const char *countName)
{
    if (batch > kMaxGridZ)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Planar random resized crop requires %s <= %ld (CUDA grid-z limit)", countName,
                              static_cast<long>(kMaxGridZ));
    }
}

} // namespace

namespace cvcuda::priv {

RandomResizedCrop::CropState::CropState(int32_t maxBatchSize, uint32_t seed)
{
    if (maxBatchSize > 0)
    {
        const size_t bufferSize = CropParamBufferSize(maxBatchSize);

        NVCV_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&deviceParams), bufferSize));

        hostParams = static_cast<std::byte *>(std::malloc(bufferSize));
        if (hostParams == nullptr)
        {
            cudaFree(deviceParams);
            deviceParams = nullptr;
            throw nvcv::Exception(nvcv::Status::ERROR_OUT_OF_MEMORY, "Memory allocation error of size: %zu",
                                  bufferSize);
        }
    }

    generator = (seed == 0) ? std::mt19937(std::random_device{}()) : std::mt19937(seed);
}

RandomResizedCrop::CropState::~CropState()
{
    NVCV_CHECK_LOG(cudaFree(deviceParams));
    std::free(hostParams);
}

RandomResizedCrop::RandomResizedCrop(double minScale, double maxScale, double minRatio, double maxRatio,
                                     int32_t maxBatchSize, uint32_t seed)
    : m_minScale(minScale)
    , m_maxScale(maxScale)
    , m_minRatio(minRatio)
    , m_maxRatio(maxRatio)
    , m_maxBatchSize(maxBatchSize)
    , m_state([maxBatchSize, seed](int) { return std::make_unique<CropState>(maxBatchSize, seed); })
    , m_stateVarShape([maxBatchSize, seed](int) { return std::make_unique<CropState>(maxBatchSize, seed); })
{
    if (!std::isfinite(minScale) || !std::isfinite(maxScale) || !std::isfinite(minRatio) || !std::isfinite(maxRatio)
        || minScale <= 0 || maxScale <= 0 || minRatio <= 0 || maxRatio <= 0 || minScale > maxScale
        || minRatio > maxRatio)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "scale and ratio must be finite, positive, and satisfy min <= max");
    }
}

void RandomResizedCrop::operator()(cudaStream_t stream, const nvcv::Tensor &in, const nvcv::Tensor &out,
                                   const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::RandomResizedCrop::operator()[Tensor]");
    auto inData  = RequireCudaTensor(in, "Input");
    auto outData = RequireCudaTensor(out, "Output");

    const DataFormatCode inFormat  = ClassifyDataFormat(inData->layout());
    const DataFormatCode outFormat = ClassifyDataFormat(outData->layout());

    if (inFormat != outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inFormat),
                              static_cast<int>(outFormat));
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    NVCV_ASSERT(inAccess);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    NVCV_ASSERT(outAccess);

    const DataTypeCode inCode  = ClassifyDataType(inData->dtype());
    const DataTypeCode outCode = ClassifyDataType(outData->dtype());

    const int channels = inAccess->numChannels();
    ValidateChannelCount(channels);

    // See OpRandomResizedCrop_Negative.tensorSampleCountMismatch and .tensorChannelCountMismatch.
    if (inAccess->numSamples() != outAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output samples are different");
    }
    if (channels != outAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output channels are different");
    }

    ValidateDataTypes(inCode, outCode);
    cvcuda::priv::ValidateInterpolation(interpolation);

    const int2 srcSize{static_cast<int>(inAccess->numCols()), static_cast<int>(inAccess->numRows())};
    const int2 dstSize{static_cast<int>(outAccess->numCols()), static_cast<int>(outAccess->numRows())};

    if (interpolation == NVCV_INTERP_LINEAR)
    {
        ValidateSourceAtLeast2x2(srcSize.x, srcSize.y);
    }

    const int batch = static_cast<int>(inAccess->numSamples());
    ValidateBatchSize(batch, m_maxBatchSize);

    CropState &state = m_state.get();

    CropParams hostParams = CarveCropParams(state.hostParams);

    for (int i = 0; i < batch; ++i)
    {
        int top, left, crop_rows, crop_cols;
        GetCropParams(state.generator, m_minScale, m_maxScale, m_minRatio, m_maxRatio, srcSize.y, srcSize.x, &top,
                      &left, &crop_rows, &crop_cols);
        hostParams.samples[i].scaleX = ((float)crop_cols) / dstSize.x;
        hostParams.samples[i].scaleY = ((float)crop_rows) / dstSize.y;
        hostParams.samples[i].top    = top;
        hostParams.samples[i].left   = left;
    }

    NVCV_CHECK_THROW(cudaMemcpyAsync(state.deviceParams, state.hostParams, CropParamBufferSize(batch),
                                     cudaMemcpyHostToDevice, stream));

    const CropParams params = CarveCropParams(state.deviceParams);

    // Checked here rather than before the draw loop so a rejected submit leaves the generator where
    // the legacy operator left it -- the check sat inside the launch helper, past the staging copy.
    ValidateTensorExtent(inAccess->sampleStride() * inAccess->numSamples(),
                         outAccess->sampleStride() * outAccess->numSamples());

    if (IsPlanar(inFormat))
    {
        ValidatePlanarBatch(batch, "numSamples");

        DispatchByDataType(inCode,
                           [&](auto dummyType)
                           {
                               using T = decltype(dummyType);
                               LaunchResizePlanar(stream, cuda::CreateTensorWrapNCHW<T, int32_t>(*inData),
                                                  cuda::CreateTensorWrapNCHW<T, int32_t>(*outData),
                                                  UniformGeometry{srcSize, dstSize}, dstSize, batch, channels, params,
                                                  interpolation);
                           });
    }
    else
    {
        DispatchByDataType(inCode,
                           [&](auto base)
                           {
                               DispatchByChannels<decltype(base)>(
                                   channels,
                                   [&](auto dummyType)
                                   {
                                       using T = decltype(dummyType);
                                       LaunchResize(stream, cuda::CreateTensorWrapNHW<T, int32_t>(*inData),
                                                    cuda::CreateTensorWrapNHW<T, int32_t>(*outData),
                                                    UniformGeometry{srcSize, dstSize}, dstSize, batch, params,
                                                    interpolation);
                                   });
                           });
    }

    // cudaGetLastError() is sticky until read: one check here covers the launch above.
    NVCV_CHECK_THROW(cudaGetLastError());
}

void RandomResizedCrop::operator()(cudaStream_t stream, const nvcv::ImageBatchVarShape &in,
                                   const nvcv::ImageBatchVarShape &out, const NVCVInterpolationType interpolation) const
{
    CVCUDA_NVTX_RANGE("cvcuda::RandomResizedCrop::operator()[ImageBatchVarShape]");
    auto inDataPtr = in.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (inDataPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto outDataPtr = out.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    if (outDataPtr == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    const nvcv::ImageBatchVarShapeDataStridedCuda &inData  = *inDataPtr;
    const nvcv::ImageBatchVarShapeDataStridedCuda &outData = *outDataPtr;

    // Checked before anything dereferences uniqueFormat(), on both batches.
    if (!inData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in input batch must all have the same format ");
    }
    if (!outData.uniqueFormat())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Images in output batch must all have the same format ");
    }

    // Earlier than on the tensor path, where the batch check follows the format and dtype checks.
    // Both orders are observable through which rejection a doubly-invalid submit reports, so each
    // keeps the order its legacy operator used.
    const int batch = inData.numImages();
    ValidateBatchSize(batch, m_maxBatchSize);

    const DataFormatCode inFormat  = ClassifyDataFormat(inData);
    const DataFormatCode outFormat = ClassifyDataFormat(outData);

    if (inFormat != outFormat)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Invalid DataFormat between input (%d) and output (%d)", static_cast<int>(inFormat),
                              static_cast<int>(outFormat));
    }

    const int channels = inData.uniqueFormat().numChannels();
    ValidateChannelCount(channels);

    // See OpRandomResizedCropVarshape_Negative.channelCountMismatch: the batch formats agree on
    // layout and element width without agreeing on channel count, and the kernels take the pixel
    // type from the input alone.
    if (channels != outData.uniqueFormat().numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output channels are different");
    }

    const DataTypeCode inCode  = ClassifyDataType(inData.uniqueFormat());
    const DataTypeCode outCode = ClassifyDataType(outData.uniqueFormat());

    ValidateDataTypes(inCode, outCode);
    cvcuda::priv::ValidateInterpolation(interpolation);

    // The image counts only had to agree once the kernels launched, where an assertion aborted the
    // process; a mismatched pair that got past the format check reached that assertion.
    if (batch != outData.numImages())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output images are different");
    }

    ImageHandleBatch inHandles(in.handle(), batch);

    // A pre-pass, not a check inside the fill loop below: rejecting mid-loop would leave the
    // generator advanced for the images already visited.
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        for (int i = 0; i < batch; ++i)
        {
            const nvcv::Size2D size = inHandles.size(i);
            ValidateSourceAtLeast2x2(size.w, size.h);
        }
    }

    ImageHandleBatch outHandles(out.handle(), batch);

    CropState &state = m_stateVarShape.get();

    CropParams hostParams = CarveCropParams(state.hostParams);

    for (int i = 0; i < batch; ++i)
    {
        // Unreachable: uniqueFormat() above already pins every image to `channels`.  Kept because it
        // is the legacy rejection, and it costs one host compare per image.
        if (channels != nvcv::ImageFormat(inData.hostFormatList()[i]).numChannels())
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid Input");
        }
        const nvcv::Size2D srcSize = inHandles.size(i);
        const nvcv::Size2D dstSize = outHandles.size(i);

        int top, left, crop_rows, crop_cols;
        GetCropParams(state.generator, m_minScale, m_maxScale, m_minRatio, m_maxRatio, srcSize.h, srcSize.w, &top,
                      &left, &crop_rows, &crop_cols);
        hostParams.samples[i].scaleX = ((float)crop_cols) / dstSize.w;
        hostParams.samples[i].scaleY = ((float)crop_rows) / dstSize.h;
        hostParams.samples[i].top    = top;
        hostParams.samples[i].left   = left;
    }

    NVCV_CHECK_THROW(cudaMemcpyAsync(state.deviceParams, state.hostParams, CropParamBufferSize(batch),
                                     cudaMemcpyHostToDevice, stream));

    const CropParams   params     = CarveCropParams(state.deviceParams);
    const nvcv::Size2D outMaxSize = outData.maxSize();
    const int2         gridExtent{outMaxSize.w, outMaxSize.h};

    if (IsPlanar(inFormat))
    {
        ValidatePlanarBatch(batch, "numImages");

        DispatchByDataType(inCode,
                           [&](auto dummyType)
                           {
                               using T = decltype(dummyType);
                               LaunchResizePlanar(stream, cuda::ImageBatchVarShapeWrap<T>(inData),
                                                  cuda::ImageBatchVarShapeWrap<T>(outData), VarShapeGeometry{},
                                                  gridExtent, batch, channels, params, interpolation);
                           });
    }
    else
    {
        DispatchByDataType(inCode,
                           [&](auto base)
                           {
                               DispatchByChannels<decltype(base)>(
                                   channels,
                                   [&](auto dummyType)
                                   {
                                       using T = decltype(dummyType);
                                       LaunchResize(stream, cuda::ImageBatchVarShapeWrap<T>(inData),
                                                    cuda::ImageBatchVarShapeWrap<T>(outData), VarShapeGeometry{},
                                                    gridExtent, batch, params, interpolation);
                                   });
                           });
    }

    // cudaGetLastError() is sticky until read: one check here covers the launch above.
    NVCV_CHECK_THROW(cudaGetLastError());
}

} // namespace cvcuda::priv
