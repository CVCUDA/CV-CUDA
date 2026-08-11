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

#ifndef CV_CUDA_PRIV_LEGACY_WARP_CUBIC_CUH
#define CV_CUDA_PRIV_LEGACY_WARP_CUBIC_CUH

#include "CvCudaUtils.cuh"

#include <cstdint>

namespace nvcv::legacy::cuda_op {

// Fast CUBIC sampling for the warp kernels. The InterpolationWrap CUBIC path issues one
// border-resolved scalar load per tap (48 loads/pixel for uchar3), which leaves the kernel
// LSU-throttled. Interior 4x4 windows need no border handling and their taps are 4 contiguous
// pixels per row, so each tap row is fetched with a few 8B loads and realigned in registers.
// Values, weights (GetCubicCoeffs), accumulation order, and SaturateCast are identical to
// InterpolationWrap<BW, NVCV_INTERP_CUBIC>::operator[], so results stay bit-exact; non-interior
// pixels defer to the wrap itself. The 8B loads assume the data base address is at least
// 8-byte aligned (NVCV device buffers are at least 256-byte aligned).

// Wide pixels stay on the InterpolationWrap path: uchar4's aligned per-tap loads are already
// issue-efficient, and float3/float4-class pixels regressed on the reference SKUs (their taps
// are wide loads already, so the restructured gather only adds overhead where big L2s absorb
// the footprint). The fast sampler pays off for byte/short and scalar pixels.
template<typename T>
constexpr bool kCubicFastSampler = sizeof(T) < 12 && !(sizeof(T) == 4 && cuda::NumElements<T> == 4);

// Number of 8-byte loads covering one row of 4 T pixels at arbitrary 8B phase.
template<typename T>
constexpr int kCubicRowLoads = (7 + 4 * static_cast<int>(sizeof(T)) + 7) / 8;

// Right-edge margin (in pixels) covering the widened-load overread past the last tap.
template<typename T>
constexpr int kCubicRowMarginPx
    = (8 * kCubicRowLoads<T> - 4 * static_cast<int>(sizeof(T)) + static_cast<int>(sizeof(T)) - 1)
    / static_cast<int>(sizeof(T));

template<typename T>
inline __device__ void LoadTapRow(const T *rowp, T (&out)[4])
{
    using BT = cuda::BaseType<T>;

    constexpr int kSpan  = 4 * static_cast<int>(sizeof(T));
    constexpr int kLoads = kCubicRowLoads<T>;
    constexpr int kWords = kSpan / 4;

    const uintptr_t addr = reinterpret_cast<uintptr_t>(rowp);
    const uint2    *p8   = reinterpret_cast<const uint2 *>(addr & ~uintptr_t(7));
    const int       off  = static_cast<int>(addr & 7);

    uint32_t raw[2 * kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
    {
        const uint2 v  = __ldg(p8 + i);
        raw[2 * i]     = v.x;
        raw[2 * i + 1] = v.y;
    }

    // Realign in two steps with compile-time register indices: drop one 4B word when the 8B
    // phase is >= 4, then funnel-shift out the sub-word phase.
    const bool dropWord = (off & 4) != 0;
    uint32_t   w[kWords + 1];
#pragma unroll
    for (int i = 0; i < kWords + 1; ++i)
    {
        w[i] = dropWord ? raw[i + 1] : raw[i];
    }

    const uint32_t shift = static_cast<uint32_t>((off & 3) * 8);
    uint32_t       aligned[kWords];
#pragma unroll
    for (int i = 0; i < kWords; ++i)
    {
        aligned[i] = __funnelshift_r(w[i], w[i + 1], shift);
    }

#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
#pragma unroll
        for (int c = 0; c < cuda::NumElements<T>; ++c)
        {
            const int      bo   = k * static_cast<int>(sizeof(T)) + c * static_cast<int>(sizeof(BT));
            const uint32_t bits = aligned[bo >> 2] >> ((bo & 3) * 8);

            if constexpr (sizeof(BT) == 4)
            {
                cuda::GetElement(out[k], c) = cuda::BaseType<T>(__uint_as_float(aligned[bo >> 2]));
            }
            else if constexpr (sizeof(BT) == 2)
            {
                cuda::GetElement(out[k], c) = static_cast<BT>(bits & 0xFFFFu);
            }
            else
            {
                cuda::GetElement(out[k], c) = static_cast<BT>(bits & 0xFFu);
            }
        }
    }
}

// Interior taps default to individual loads from their row pointer: for coherent (gentle) maps
// the per-tap loads coalesce across the warp, and skipping border resolution is the entire win.
// WideInterior opts into the widened row loads for access patterns that scatter enough for the
// load-count reduction to dominate (measured per operator; rotate's diagonal access qualifies,
// perspective's gentle maps do not).
template<typename T, bool WideInterior, typename StrideType, class RowPtrOp>
inline __device__ T CubicAccumulate(RowPtrOp rowPtr, float x, float y, StrideType ix, StrideType iy)
{
    float wx[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
    float wy[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

    using FT = cuda::ConvertBaseTypeTo<float, T>;
    auto sum = cuda::SetAll<FT>(0);

#pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        if constexpr (WideInterior)
        {
            T row[4];
            LoadTapRow(rowPtr(static_cast<int>(iy) + r - 1, static_cast<int>(ix) - 1), row);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                sum += row[k] * (wx[k] * wy[r]);
            }
        }
        else
        {
            const T *row = rowPtr(static_cast<int>(iy) + r - 1, static_cast<int>(ix) - 1);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                sum += row[k] * (wx[k] * wy[r]);
            }
        }
    }

    return cuda::SaturateCast<T>(sum);
}

// Border-resolved CUBIC gather. Border remapping is separable per axis, so the 4 column and 4
// row indices are resolved once (8 GetIndexWithBorder calls — REFLECT costs an integer modulo
// each) instead of per tap (32), and each tap row is addressed from its row base pointer.
// Index math, tap values, weights, accumulation order, and SaturateCast match the
// InterpolationWrap CUBIC path exactly.
template<typename T, NVCVBorderType B, typename StrideType, class RowPtrOp>
inline __device__ T CubicAccumulateBorder(RowPtrOp rowPtr, T borderValue, int2 size, float x, float y, StrideType ix,
                                          StrideType iy)
{
    float wx[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
    float wy[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

    StrideType xr[4], yr[4];
    bool       xin[4], yin[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        if constexpr (B == NVCV_BORDER_CONSTANT)
        {
            xin[k] = !cuda::IsOutside(ix + k - 1, static_cast<StrideType>(size.x));
            yin[k] = !cuda::IsOutside(iy + k - 1, static_cast<StrideType>(size.y));
            xr[k]  = xin[k] ? ix + k - 1 : 0;
            yr[k]  = yin[k] ? iy + k - 1 : 0;
        }
        else
        {
            xin[k] = yin[k] = true;
            xr[k]           = cuda::GetIndexWithBorder<B>(ix + k - 1, static_cast<StrideType>(size.x));
            yr[k]           = cuda::GetIndexWithBorder<B>(iy + k - 1, static_cast<StrideType>(size.y));
        }
    }

    using FT = cuda::ConvertBaseTypeTo<float, T>;
    auto sum = cuda::SetAll<FT>(0);

    // Border remapping preserves adjacency away from fold points, so most resolved windows are
    // still 4 contiguous columns (ascending, or descending in reflected segments) and take the
    // widened row loads; windows straddling a fold or a constant-border edge stay scalar.
    const bool       xAsc  = xr[1] == xr[0] + 1 && xr[2] == xr[0] + 2 && xr[3] == xr[0] + 3;
    const bool       xDesc = xr[1] == xr[0] - 1 && xr[2] == xr[0] - 2 && xr[3] == xr[0] - 3;
    const bool       xIn   = xin[0] && xin[1] && xin[2] && xin[3] && yin[0] && yin[1] && yin[2] && yin[3];
    const StrideType x0    = xAsc ? xr[0] : xr[3];

    if ((xAsc || xDesc) && xIn && x0 + 3 + kCubicRowMarginPx<T> < size.x)
    {
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            T row[4];
            LoadTapRow(rowPtr(static_cast<int>(yr[r])) + x0, row);
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const T v = xAsc ? row[k] : row[3 - k];
                sum += v * (wx[k] * wy[r]);
            }
        }
    }
    else
    {
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            const T *row = rowPtr(static_cast<int>(yr[r]));
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const T v = (yin[r] && xin[k]) ? row[xr[k]] : borderValue;
                sum += v * (wx[k] * wy[r]);
            }
        }
    }

    return cuda::SaturateCast<T>(sum);
}

// Tensor (NHW wrap) CUBIC sample; srcSize is (numCols, numRows).
template<class SrcWrapper, bool WideInterior = false>
inline __device__ std::remove_cv_t<typename SrcWrapper::ValueType> CubicSampleTensor(const SrcWrapper &src, int z,
                                                                                     float2 coord, int2 srcSize)
{
    using T          = std::remove_cv_t<typename SrcWrapper::ValueType>;
    using StrideType = typename SrcWrapper::StrideType;

    const StrideType ix = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.x);
    const StrideType iy = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.y);

    const auto &tw = src.borderWrap().tensorWrap();

    if (ix >= 1 && iy >= 1 && iy + 2 < srcSize.y && ix + 2 + kCubicRowMarginPx<T> < srcSize.x)
    {
        return CubicAccumulate<T, WideInterior>([&](int yy, int xx) { return tw.ptr(z, yy, xx); }, coord.x, coord.y, ix,
                                                iy);
    }

    constexpr NVCVBorderType kB = SrcWrapper::BorderWrapper::kBorderType;
    return CubicAccumulateBorder<T, kB>([&](int yy) { return tw.ptr(z, yy); }, src.borderWrap().borderValue(), srcSize,
                                        coord.x, coord.y, ix, iy);
}

// Var-shape (image batch wrap) CUBIC sample; sizes come from the wrapped batch per sample.
template<class SrcWrapper, bool WideInterior = false>
inline __device__ std::remove_cv_t<typename SrcWrapper::ValueType> CubicSampleVarShape(const SrcWrapper &src, int z,
                                                                                       float2 coord)
{
    using T          = std::remove_cv_t<typename SrcWrapper::ValueType>;
    using StrideType = int;

    const StrideType ix = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.x);
    const StrideType iy = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.y);

    const auto &ibw     = src.borderWrap().imageBatchWrap();
    const int2  srcSize = {ibw.width(z), ibw.height(z)};

    if (ix >= 1 && iy >= 1 && iy + 2 < srcSize.y && ix + 2 + kCubicRowMarginPx<T> < srcSize.x)
    {
        return CubicAccumulate<T, WideInterior>([&](int yy, int xx) { return ibw.ptr(z, yy, xx); }, coord.x, coord.y,
                                                ix, iy);
    }

    constexpr NVCVBorderType kB = SrcWrapper::BorderWrapper::kBorderType;
    return CubicAccumulateBorder<T, kB>([&](int yy) { return ibw.ptr(z, yy); }, src.borderWrap().borderValue(), srcSize,
                                        coord.x, coord.y, ix, iy);
}

// Var-shape planar (plane-indexed) CUBIC sample for the per-plane kernel.
template<class SrcWrapper, bool WideInterior = false>
inline __device__ std::remove_cv_t<typename SrcWrapper::ValueType> CubicSampleVarShapePlane(const SrcWrapper &src,
                                                                                            int z, int plane,
                                                                                            float2 coord)
{
    using T          = std::remove_cv_t<typename SrcWrapper::ValueType>;
    using StrideType = int;

    const StrideType ix = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.x);
    const StrideType iy = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.y);

    const auto &ibw     = src.borderWrap().imageBatchWrap();
    const int2  srcSize = {ibw.width(z, plane), ibw.height(z, plane)};

    if (ix >= 1 && iy >= 1 && iy + 2 < srcSize.y && ix + 2 + kCubicRowMarginPx<T> < srcSize.x)
    {
        return CubicAccumulate<T, WideInterior>([&](int yy, int xx) { return ibw.ptr(z, plane, yy, xx); }, coord.x,
                                                coord.y, ix, iy);
    }

    constexpr NVCVBorderType kB = SrcWrapper::BorderWrapper::kBorderType;
    return CubicAccumulateBorder<T, kB>([&](int yy) { return ibw.ptr(z, plane, yy, 0); },
                                        src.borderWrap().borderValue(), srcSize, coord.x, coord.y, ix, iy);
}

// Fused planar CUBIC warp for one output pixel across all channel planes. Every plane samples
// the same transformed coordinate, so the interpolation indices, weights, and border resolution
// are computed once and only the gather + accumulate runs per plane. Per-plane values match the
// single-channel per-plane kernel launches bit-exactly (same index math, weights, accumulation
// order, and SaturateCast; scalar float math is identical to the single-channel vector math).
// rowPtr(p, yy) returns plane p's row base; store(p, v) writes the plane's output pixel.
template<typename BT, NVCVBorderType B, int NP, typename StrideType, class RowPtrOp, class StoreOp>
inline __device__ void CubicWarpPlanes(RowPtrOp rowPtr, StoreOp store, const float4 &borderValue4, int2 size,
                                       float2 coord)
{
    const StrideType ix = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.x);
    const StrideType iy = cuda::GetIndexForInterpolation<NVCV_INTERP_CUBIC, 1, StrideType>(coord.y);

    float wx[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(coord.x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
    float wy[4]; // NOSONAR: CUDA cubic coefficients are indexed in the unrolled loop.
    cuda::GetCubicCoeffs(coord.y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

    if (ix >= 1 && iy >= 1 && iy + 2 < size.y && ix + 2 + kCubicRowMarginPx<BT> < size.x)
    {
#pragma unroll
        for (int p = 0; p < NP; ++p)
        {
            float sum = 0;
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                BT row[4];
                LoadTapRow(rowPtr(p, static_cast<int>(iy) + r - 1) + (static_cast<int>(ix) - 1), row);
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    sum += row[k] * (wx[k] * wy[r]);
                }
            }
            store(p, cuda::SaturateCast<BT>(sum));
        }
        return;
    }

    StrideType xr[4], yr[4];
    bool       xin[4], yin[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        if constexpr (B == NVCV_BORDER_CONSTANT)
        {
            xin[k] = !cuda::IsOutside(ix + k - 1, static_cast<StrideType>(size.x));
            yin[k] = !cuda::IsOutside(iy + k - 1, static_cast<StrideType>(size.y));
            xr[k]  = xin[k] ? ix + k - 1 : 0;
            yr[k]  = yin[k] ? iy + k - 1 : 0;
        }
        else
        {
            xin[k] = yin[k] = true;
            xr[k]           = cuda::GetIndexWithBorder<B>(ix + k - 1, static_cast<StrideType>(size.x));
            yr[k]           = cuda::GetIndexWithBorder<B>(iy + k - 1, static_cast<StrideType>(size.y));
        }
    }

    const bool       xAsc   = xr[1] == xr[0] + 1 && xr[2] == xr[0] + 2 && xr[3] == xr[0] + 3;
    const bool       xDesc  = xr[1] == xr[0] - 1 && xr[2] == xr[0] - 2 && xr[3] == xr[0] - 3;
    const bool       allIn  = xin[0] && xin[1] && xin[2] && xin[3] && yin[0] && yin[1] && yin[2] && yin[3];
    const StrideType x0     = xAsc ? xr[0] : xr[3];
    const bool       vecRow = (xAsc || xDesc) && allIn && x0 + 3 + kCubicRowMarginPx<BT> < size.x;

#pragma unroll
    for (int p = 0; p < NP; ++p)
    {
        const BT bval = cuda::StaticCast<BT>(cuda::GetElement(borderValue4, p));

        float sum = 0;
        if (vecRow)
        {
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                BT row[4];
                LoadTapRow(rowPtr(p, static_cast<int>(yr[r])) + x0, row);
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    sum += (xAsc ? row[k] : row[3 - k]) * (wx[k] * wy[r]);
                }
            }
        }
        else
        {
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                const BT *row = rowPtr(p, static_cast<int>(yr[r]));
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const BT v = (yin[r] && xin[k]) ? row[xr[k]] : bval;
                    sum += v * (wx[k] * wy[r]);
                }
            }
        }
        store(p, cuda::SaturateCast<BT>(sum));
    }
}

} // namespace nvcv::legacy::cuda_op

#endif // CV_CUDA_PRIV_LEGACY_WARP_CUBIC_CUH
