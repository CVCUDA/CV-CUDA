/* Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#include <cfloat>

// NOTE: Below are the "standard" (NTSC and ITU Rec.601) RGB to luma conversion
// coefficients. More accurate coefficents, given as comments on the right, are
// found at http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html and
// https://www.imagemagick.org/include/api/pixel.php.
static constexpr float R2YF = 0.299f; // 0.298839
static constexpr float G2YF = 0.587f; // 0.586811
static constexpr float B2YF = 0.114f; // 0.114350

static constexpr int gray_shift = 15;
static constexpr int yuv_shift  = 14;
static constexpr int RY15       = 9798;  // == R2YF*32768 + 0.5
static constexpr int GY15       = 19235; // == G2YF*32768 + 0.5
static constexpr int BY15       = 3735;  // == B2YF*32768 + 0.5

static constexpr int R2Y  = 4899;  // == R2YF*16384
static constexpr int G2Y  = 9617;  // == G2YF*16384
static constexpr int B2Y  = 1868;  // == B2YF*16384
static constexpr int R2VI = 14369; // == R2VF*16384
static constexpr int B2UI = 8061;  // == B2UF*16384

static constexpr float B2UF = 0.492f; // 0.492111: U = (B - Y) * B2UF + 0.5
static constexpr float R2VF = 0.877f; // 0.877283: V = (R - Y) * R2VF + 0.5

static constexpr int U2BI = 33292;
static constexpr int U2GI = -6472;
static constexpr int V2GI = -9519;
static constexpr int V2RI = 18678;

static constexpr float U2BF = 2.032f;
static constexpr float U2GF = -0.395f;
static constexpr float V2GF = -0.581f;
static constexpr float V2RF = 1.140f;

// Coefficients for YUV420sp to RGB conversion
static constexpr int ITUR_BT_601_CY    = 1220542;
static constexpr int ITUR_BT_601_CUB   = 2116026;
static constexpr int ITUR_BT_601_CUG   = -409993;
static constexpr int ITUR_BT_601_CVG   = -852492;
static constexpr int ITUR_BT_601_CVR   = 1673527;
static constexpr int ITUR_BT_601_SHIFT = 20;
// Coefficients for RGB to YUV420p conversion
static constexpr int ITUR_BT_601_CRY = 269484;
static constexpr int ITUR_BT_601_CGY = 528482;
static constexpr int ITUR_BT_601_CBY = 102760;
static constexpr int ITUR_BT_601_CRU = -155188;
static constexpr int ITUR_BT_601_CGU = -305135;
static constexpr int ITUR_BT_601_CBU = 460324;
static constexpr int ITUR_BT_601_CGV = -385875;
static constexpr int ITUR_BT_601_CBV = -74448;

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

#define BLOCK 32

#define DEVICE_INLINE __device__ __forceinline__
#define GLOBAL_BOUNDS __global__ __launch_bounds__(Policy::BlockSize)

template<typename T, typename BT = nvcv::cuda::BaseType<T>>
constexpr BT Alpha = std::is_floating_point_v<BT> ? 1 : nvcv::cuda::TypeTraits<BT>::max;

namespace nvcv::legacy::cuda_op {

template<typename T, typename StrideT>
using TensorWrap3D = nvcv::cuda::Tensor3DWrap<T, StrideT>;

template<typename T, typename StrideT>
using TensorWrap4D = nvcv::cuda::Tensor4DWrap<T, StrideT>;

template<int BlockWidth_, int BlockHeight_, int RowsPerThread_>
struct CvtKernelPolicy
{
    static_assert(BlockWidth_ % 32 == 0);
    static constexpr int BlockWidth      = BlockWidth_;
    static constexpr int BlockHeight     = BlockHeight_;
    static constexpr int BlockSize       = BlockWidth * BlockHeight;
    static constexpr int RowsPerThread   = RowsPerThread_;
    static constexpr int TileWidth       = BlockWidth;
    static constexpr int TileHeight      = BlockHeight * RowsPerThread;
    static constexpr int ThreadRowStride = BlockHeight;
};

template<typename Policy, int N_IN, int N_OUT, typename EltT, typename LoadOpT, typename ConvOpT, typename StoreOpT>
DEVICE_INLINE void color_conversion_common(LoadOpT load_op, ConvOpT conv_op, StoreOpT store_op, int2 size)
{
    const int x         = blockIdx.x * Policy::TileWidth + threadIdx.x;
    const int y0        = blockIdx.y * Policy::TileHeight + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= size.x)
    {
        return;
    }

    // Branchless efficient path for inner blocks.
    if (y0 + Policy::TileHeight <= size.y)
    {
        EltT r_in[Policy::RowsPerThread][N_IN];
        EltT r_out[Policy::RowsPerThread][N_OUT];

#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++)
        {
            const int y = y0 + Policy::ThreadRowStride * i;
            load_op(r_in[i], batch_idx, x, y);
        }
#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++) conv_op(r_in[i], r_out[i]);
#pragma unroll
        for (int i = 0; i < Policy::RowsPerThread; i++)
        {
            const int y = y0 + Policy::ThreadRowStride * i;
            store_op(r_out[i], batch_idx, x, y);
        }
    }
    else
    {
        int y = y0;
        for (int i = 0; i < Policy::RowsPerThread && y < size.y; i++)
        {
            EltT r_in[N_IN];
            EltT r_out[N_OUT];

            load_op(r_in, batch_idx, x, y);
            conv_op(r_in, r_out);
            store_op(r_out, batch_idx, x, y);

            y += Policy::ThreadRowStride;
        }
    }
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load3_nhwc(const TensorWrap3D<const SrcT, StrideT> &src, EltT &C0, EltT &C1, EltT &C2, int batch_idx,
                              int x, int y)
{
    SrcT vec = *src.ptr(batch_idx, y, x);
    C0       = vec.x;
    C1       = vec.y;
    C2       = vec.z;
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store3_nhwc(const TensorWrap3D<DstT, StrideT> &dst, EltT C0, EltT C1, EltT C2, int batch_idx, int x,
                               int y)
{
    DstT vec;
    vec.x                     = C0;
    vec.y                     = C1;
    vec.z                     = C2;
    *dst.ptr(batch_idx, y, x) = vec;
}

template<typename SrcT, typename EltT, typename StrideT>
DEVICE_INLINE void load_bgra_nhwc(const TensorWrap3D<const SrcT, StrideT> &src, EltT &B, EltT &G, EltT &R, EltT &A,
                                  int batch_idx, int x, int y, int bidx)
{
    SrcT vec = *src.ptr(batch_idx, y, x);
    B        = bidx == 0 ? vec.x : vec.z;
    G        = vec.y;
    R        = bidx == 0 ? vec.z : vec.x;
    if constexpr (nvcv::cuda::NumComponents<SrcT> == 4)
    {
        A = vec.w;
    }
    else
    {
        A = Alpha<EltT>;
    }
}

template<typename DstT, typename EltT, typename StrideT>
DEVICE_INLINE void store_bgra_nhwc(const TensorWrap3D<DstT, StrideT> &dst, EltT B, EltT G, EltT R, EltT A,
                                   int batch_idx, int x, int y, int bidx)
{
    DstT vec;
    vec.x = bidx == 0 ? B : R;
    vec.y = G;
    vec.z = bidx == 0 ? R : B;
    if constexpr (nvcv::cuda::NumComponents<DstT> == 4)
    {
        vec.w = A;
    }
    *dst.ptr(batch_idx, y, x) = vec;
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void rgb_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 4, 4, EltT>(
        [&src, bidx] __device__(EltT(&r_in)[4], int batch_idx, int x, int y)
        { load_bgra_nhwc(src, r_in[0], r_in[1], r_in[2], r_in[3], batch_idx, x, y, bidx); },
        [] __device__(const EltT(&r_in)[4], EltT(&r_out)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_out[i] = r_in[i];
        },
        [&dst] __device__(const EltT(&r_out)[4], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_out[0], r_out[1], r_out[2], r_out[3], batch_idx, x, y, 0); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void gray_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                    int2 dstSize)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 1, 4, EltT>(
        [&src] __device__(EltT(&r_gray)[1], int batch_idx, int x, int y) { r_gray[0] = *src.ptr(batch_idx, y, x); },
        [] __device__(const EltT(&r_gray)[1], EltT(&r_BGRA)[4])
        {
#pragma unroll
            for (int i = 0; i < 4; i++) r_BGRA[i] = r_gray[0];
        },
        [&dst] __device__(const EltT(&r_BGRA)[4], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGRA[0], r_BGRA[1], r_BGRA[2], r_BGRA[3], batch_idx, x, y, 0); },
        dstSize);
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_gray_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                    int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 1, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_gray)[1])
        {
            if constexpr (std::is_integral_v<EltT>)
                r_gray[0]
                    = (EltT)CV_DESCALE((int)r_BGR[0] * BY15 + (int)r_BGR[1] * GY15 + (int)r_BGR[2] * RY15, gray_shift);
            else
                r_gray[0] = (EltT)(r_BGR[0] * B2YF + r_BGR[1] * G2YF + r_BGR[2] * R2YF);
        },
        [&dst] __device__(const EltT(&r_gray)[1], int batch_idx, int x, int y)
        { *dst.ptr(batch_idx, y, x) = r_gray[0]; },
        dstSize);
}

template<typename T>
DEVICE_INLINE void bgr_to_yuv_int(T B_, T G_, T R_, T &Y_, T &Cb_, T &Cr_)
{
    constexpr int C0 = R2Y, C1 = G2Y, C2 = B2Y, C3 = R2VI, C4 = B2UI;
    constexpr int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) << yuv_shift;

    const int B = B_, G = G_, R = R_;

    const int Y  = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    const int Cr = CV_DESCALE((R - Y) * C3 + delta, yuv_shift);
    const int Cb = CV_DESCALE((B - Y) * C4 + delta, yuv_shift);

    Y_  = cuda::SaturateCast<T>(Y);
    Cb_ = cuda::SaturateCast<T>(Cb);
    Cr_ = cuda::SaturateCast<T>(Cr);
}

DEVICE_INLINE void bgr_to_yuv_float(float B, float G, float R, float &Y, float &Cb, float &Cr)
{
    constexpr float C0 = R2YF, C1 = G2YF, C2 = B2YF, C3 = R2VF, C4 = B2UF;
    constexpr float delta = 0.5f;

    Y  = R * C0 + G * C1 + B * C2;
    Cr = (R - Y) * C3 + delta;
    Cb = (B - Y) * C4 + delta;
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_yuv_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_YCbCr)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_yuv_int(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
            else
                bgr_to_yuv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_YCbCr[0], r_YCbCr[1], r_YCbCr[2]);
        },
        [&dst] __device__(const EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { store3_nhwc(dst, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        dstSize);
}

template<typename T>
DEVICE_INLINE void yuv_to_bgr_int(T Y_, T Cb_, T Cr_, T &B_, T &G_, T &R_)
{
    constexpr int C0 = V2RI, C1 = V2GI, C2 = U2GI, C3 = U2BI;
    constexpr int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));

    const int Y = Y_, Cb = Cb_, Cr = Cr_;
    const int B = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    const int G = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    const int R = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    B_ = cuda::SaturateCast<T>(B);
    G_ = cuda::SaturateCast<T>(G);
    R_ = cuda::SaturateCast<T>(R);
}

DEVICE_INLINE void yuv_to_bgr_flt(float Y, float Cb, float Cr, float &B, float &G, float &R)
{
    constexpr float C0 = V2RF, C1 = V2GF, C2 = U2GF, C3 = U2BF;
    constexpr float delta = 0.5f;

    B = Y + (Cb - delta) * C3;
    G = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
    R = Y + (Cr - delta) * C0;
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void yuv_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_YCbCr)[3], int batch_idx, int x, int y)
        { load3_nhwc(src, r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], batch_idx, x, y); },
        [] __device__(const EltT(&r_YCbCr)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                yuv_to_bgr_int(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            else
                yuv_to_bgr_flt(r_YCbCr[0], r_YCbCr[1], r_YCbCr[2], r_BGR[0], r_BGR[1], r_BGR[2]);
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>, batch_idx, x, y, bidx); },
        dstSize);
}

DEVICE_INLINE void bgr_to_hsv_uchar(uchar b8, uchar g8, uchar r8, uchar &h8, uchar &s8, uchar &v8, bool isFullRange)
{
    const int hrange    = isFullRange ? 256 : 180;
    const int hsv_shift = 12;

    const int b = (int)b8;
    const int g = (int)g8;
    const int r = (int)r8;

    const int vmin = cuda::min(b, cuda::min(g, r));
    const int v    = cuda::max(b, cuda::max(g, r));

    const int diff = v - vmin;
    const int vr   = v == r ? -1 : 0;
    const int vg   = v == g ? -1 : 0;

    const int hdiv_table = diff == 0 ? 0 : cuda::SaturateCast<int>((hrange << hsv_shift) / (6.f * diff));
    const int sdiv_table = v == 0 ? 0 : cuda::SaturateCast<int>((255 << hsv_shift) / (float)v);

    const int s = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    int       h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));

    h = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += h < 0 ? hrange : 0;

    h8 = cuda::SaturateCast<uint8_t>(h);
    s8 = (uint8_t)s;
    v8 = (uint8_t)v;
}

DEVICE_INLINE void bgr_to_hsv_float(float b, float g, float r, float &h, float &s, float &v)
{
    float vmin = cuda::min(r, cuda::min(g, b));
    v          = cuda::max(r, cuda::max(g, b));
    float diff = v - vmin;
    s          = diff / (fabs(v) + FLT_EPSILON);
    diff       = 60.f / (diff + FLT_EPSILON);

    // clang-format off
    if      (v == r) h = (g - b) * diff;
    else if (v == g) h = (b - r) * diff + 120.f;
    else             h = (r - g) * diff + 240.f;

    if (h < 0.f) h += 360.f;
    // clang-format on
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_hsv_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [isFullRange] __device__(const EltT(&r_BGR)[3], EltT(&r_HSV)[3])
        {
            if constexpr (std::is_integral_v<EltT>)
                bgr_to_hsv_uchar(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2], isFullRange);
            else
                bgr_to_hsv_float(r_BGR[0], r_BGR[1], r_BGR[2], r_HSV[0], r_HSV[1], r_HSV[2]);
        },
        [&dst] __device__(const EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { store3_nhwc(dst, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        dstSize);
}

template<typename T>
DEVICE_INLINE T select4_reg(const T (&tab)[4], int idx)
{
    // Random access in a register array of size 4, with 6 instructions.
    // The compiler was generating 10 instructions for tab[idx].
    T out;
    out = idx == 1 ? tab[1] : tab[0];
    out = idx == 2 ? tab[2] : out;
    out = idx == 3 ? tab[3] : out;
    return out;
}

DEVICE_INLINE void hsv_to_bgr_float(float h, float s, float v, float &b, float &g, float &r)
{
    if (s == 0)
        b = g = r = v;
    else
    {
        h += 6 * (h < 0);
        int idx = static_cast<int>(h); // Sector index.
        h -= idx;                      // Fractional part of h.
        idx = (idx % 6) << 2;          // Shift index for sector LUT.

        // clang-format off
        const float tab[4] {v,
                            v * (1 - s),
                            v * (1 - s * h),
                            v * (1 - s * (1 - h))};
        // clang-format on

        constexpr int32_t idx_lutb = 0x00200311;
        constexpr int32_t idx_lutg = 0x00112003;
        constexpr int32_t idx_lutr = 0x00031120;

        b = select4_reg(tab, (idx_lutb >> idx) & 0xf);
        g = select4_reg(tab, (idx_lutg >> idx) & 0xf);
        r = select4_reg(tab, (idx_lutr >> idx) & 0xf);
    }
}

template<typename Policy, typename SrcT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void hsv_to_bgr_nhwc(const TensorWrap3D<const SrcT, StrideT> src, const TensorWrap3D<DstT, StrideT> dst,
                                   int2 dstSize, int bidx, bool isFullRange)
{
    using EltT = nvcv::cuda::BaseType<SrcT>;
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src] __device__(EltT(&r_HSV)[3], int batch_idx, int x, int y)
        { load3_nhwc(src, r_HSV[0], r_HSV[1], r_HSV[2], batch_idx, x, y); },
        [isFullRange] __device__(const EltT(&r_HSV)[3], EltT(&r_BGR)[3])
        {
            if constexpr (std::is_same_v<EltT, uchar>)
            {
                const float     scaleH  = isFullRange ? (6.0f / 256.0f) : (6.0f / 180.0f);
                constexpr float scaleSV = 1.0f / 255.0f;

                float Bf, Gf, Rf;

                hsv_to_bgr_float((float)r_HSV[0] * scaleH, r_HSV[1] * scaleSV, r_HSV[2] * scaleSV, Bf, Gf, Rf);

                r_BGR[0] = cuda::SaturateCast<uchar>(Bf * 255.0f);
                r_BGR[1] = cuda::SaturateCast<uchar>(Gf * 255.0f);
                r_BGR[2] = cuda::SaturateCast<uchar>(Rf * 255.0f);
            }
            else
            {
                constexpr float scaleH = 6.0f / 360.0f;

                hsv_to_bgr_float(r_HSV[0] * scaleH, r_HSV[1], r_HSV[2], r_BGR[0], r_BGR[1], r_BGR[2]);
            }
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>, batch_idx, x, y, bidx); },
        dstSize);
}

template<bool IsSemiPlanar, typename EltT, typename StrideT>
DEVICE_INLINE void load_yuv420(const nvcv::cuda::Tensor4DWrap<const EltT, StrideT> &src, EltT &Y, EltT &U, EltT &V,
                               int2 size, int batch_idx, int x, int y, int uidx)
{
    if constexpr (IsSemiPlanar)
    {
        // U and V are subsampled at half the full resolution (in both x and y), combined (i.e., interleaved), and
        // arranged as full rows after the full resolution Y data. Example memory layout for 4 x 4 image (NV12):
        //   Y_00 Y_01 Y_02 Y_03
        //   Y_10 Y_11 Y_12 Y_13
        //   Y_20 Y_21 Y_22 Y_23
        //   Y_30 Y_31 Y_32 Y_33
        //   U_00 V_00 U_02 V_02
        //   U_20 V_20 U_22 V_22
        // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
        // and Y_11. Each full U-V row represents 2 rows of Y values. Some layouts (e.g., NV21) swap the location
        // of the U and V values in each U-V pair (indicated by the uidx parameter).

        const int uv_y = size.y + y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
        const int uv_x = (x & ~1);       // Convert x to even # (set lowest bit to 0).

        Y = *src.ptr(batch_idx, y, x);                    // Y (luma) is at full resolution.
        U = *src.ptr(batch_idx, uv_y, uv_x + uidx);       // Some formats swap the U and V elements (as indicated
        V = *src.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)); //   by the uidx parameter).
    }
    else
    {
        // U and V are subsampled at half the full resolution (in both x and y) and arranged as non-interleaved planes
        // (i.e., planar format). Each subsampled U and V "plane" is arranged as full rows after the full resolution Y
        // data--so two consecutive subsampled U or V rows are combined into one row spanning the same width as the Y
        // plane. Example memory layout for 4 x 4 image (e.g. I420):
        //   Y_00 Y_01 Y_02 Y_03
        //   Y_10 Y_11 Y_12 Y_13
        //   Y_20 Y_21 Y_22 Y_23
        //   Y_30 Y_31 Y_32 Y_33
        //   U_00 U_02 U_20 U_22
        //   V_00 V_02 V_20 V_22
        // Each U and V value corresponds to a 2x2 block of Y values--e.g. U_00 and V_00 correspond to Y_00, Y_01, Y_10,
        // and Y_11. Each full U and V row represents 4 rows of Y values. Some layouts (e.g., YV12) swap the location
        // of the U and V planes (indicated by the uidx parameter).

        const int by = size.y + y / 4; // Base row coordinate for U and V: subsampled plane is 1/4 the height.
        const int h4 = size.y / 4;     // Height (# of rows) of each subsampled U and V plane.

        // Compute x position that combines two subsampled rows into one.
        const int uv_x = (x / 2) + ((size.x / 2) & -((y / 2) & 1)); // Second half of row for odd y coordinates.

        Y = *src.ptr(batch_idx, y, x);                       // Y (luma) is at full resolution.
        U = *src.ptr(batch_idx, by + h4 * uidx, uv_x);       // Some formats swap the U and V "planes" (as indicated
        V = *src.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x); //   by the uidx parameter).
    }
}

template<bool IsSemiPlanar, typename EltT, typename StrideT>
DEVICE_INLINE void store_yuv420(const TensorWrap4D<EltT, StrideT> &dst, EltT Y, EltT U, EltT V, int2 size,
                                int batch_idx, int x, int y, int uidx)
{
    if constexpr (IsSemiPlanar)
    {
        // See YUV420 semi-planar layout commments in load_yuv420 above.
        *dst.ptr(batch_idx, y, x) = Y; // Y (luma) is at full resolution.
        if (y % 2 == 0 && x % 2 == 0)
        {
            const int uv_y = size.y + y / 2; // The interleaved U-V semi-plane is 1/2 the height of the Y data.
            const int uv_x = (x & ~1);       // Convert x to even # (set lowest bit to 0).

            *dst.ptr(batch_idx, uv_y, uv_x + uidx)       = U; // Some formats swap the U and V elements (as indicated
            *dst.ptr(batch_idx, uv_y, uv_x + (uidx ^ 1)) = V; //   by the uidx parameter).
        }
    }
    else
    {
        // See YUV420 planar layout commments in load_yuv420 above.
        *dst.ptr(batch_idx, y, x, 0) = Y; // Y (luma) is at full resolution.
        if (y % 2 == 0 && x % 2 == 0)
        {
            const int by = size.y + y / 4; // Base row coordinate for U and V: subsampled plane is 1/4 the height.
            const int h4 = size.y / 4;     // Height (# of rows) of each subsampled U and V plane.

            // Compute x position that combines two subsampled rows into one.
            const int uv_x = (x / 2) + ((size.x / 2) & -((y / 2) & 1)); // Second half of row for odd y coordinates.

            *dst.ptr(batch_idx, by + h4 * uidx, uv_x)       = U; // Some formats swap the U and V "planes" (as indicated
            *dst.ptr(batch_idx, by + h4 * (uidx ^ 1), uv_x) = V; //   by the uidx parameter).
        }
    }
}

DEVICE_INLINE void bgr_to_yuv42xxp(const uchar &b, const uchar &g, const uchar &r, uchar &Y, uchar &U, uchar &V)
{
    const int shifted16 = (16 << ITUR_BT_601_SHIFT);
    const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
    int       yy        = ITUR_BT_601_CRY * r + ITUR_BT_601_CGY * g + ITUR_BT_601_CBY * b + halfShift + shifted16;

    Y = cuda::SaturateCast<uchar>(yy >> ITUR_BT_601_SHIFT);

    const int shifted128 = (128 << ITUR_BT_601_SHIFT);
    int       uu         = ITUR_BT_601_CRU * r + ITUR_BT_601_CGU * g + ITUR_BT_601_CBU * b + halfShift + shifted128;
    int       vv         = ITUR_BT_601_CBU * r + ITUR_BT_601_CGV * g + ITUR_BT_601_CBV * b + halfShift + shifted128;

    U = cuda::SaturateCast<uchar>(uu >> ITUR_BT_601_SHIFT);
    V = cuda::SaturateCast<uchar>(vv >> ITUR_BT_601_SHIFT);
}

DEVICE_INLINE void yuv42xxp_to_bgr(const int &Y, const int &U, const int &V, uchar &b, uchar &g, uchar &r)
{
    //R = 1.164(Y - 16) + 1.596(V - 128)
    //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    //B = 1.164(Y - 16)                  + 2.018(U - 128)

    //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
    //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
    //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
    const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG,
              C4           = ITUR_BT_601_CUB;
    const int yuv4xx_shift = ITUR_BT_601_SHIFT;

    int yy = cuda::max(0, Y - 16) * C0;
    int uu = U - 128;
    int vv = V - 128;

    r = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C1 * vv), yuv4xx_shift));
    g = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C2 * vv + C3 * uu), yuv4xx_shift));
    b = cuda::SaturateCast<uchar>(CV_DESCALE((yy + C4 * uu), yuv4xx_shift));
}

template<bool IsSemiPlanar, typename Policy, typename SrcT, typename EltT, typename StrideT>
GLOBAL_BOUNDS void bgr_to_yuv420_char_nhwc(const TensorWrap3D<const SrcT, StrideT> src,
                                           const TensorWrap4D<EltT, StrideT> dst, int2 size, int bidx, int uidx)
{
    static_assert(std::is_same_v<nvcv::cuda::BaseType<SrcT>, EltT>);
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, bidx] __device__(EltT(&r_BGR)[3], int batch_idx, int x, int y)
        {
            EltT A;
            load_bgra_nhwc(src, r_BGR[0], r_BGR[1], r_BGR[2], A, batch_idx, x, y, bidx);
        },
        [] __device__(const EltT(&r_BGR)[3], EltT(&r_YUV)[3])
        { bgr_to_yuv42xxp(r_BGR[0], r_BGR[1], r_BGR[2], r_YUV[0], r_YUV[1], r_YUV[2]); },
        [&dst, uidx, size] __device__(const EltT(&r_YUV)[3], int batch_idx, int x, int y)
        { store_yuv420<IsSemiPlanar>(dst, r_YUV[0], r_YUV[1], r_YUV[2], size, batch_idx, x, y, uidx); },
        size);
}

template<bool IsSemiPlanar, typename Policy, typename EltT, typename DstT, typename StrideT>
GLOBAL_BOUNDS void yuv420_to_bgr_char_nhwc(const TensorWrap4D<const EltT, StrideT> src,
                                           const TensorWrap3D<DstT, StrideT> dst, int2 size, int bidx, int uidx)
{
    static_assert(std::is_same_v<nvcv::cuda::BaseType<DstT>, EltT>);
    color_conversion_common<Policy, 3, 3, EltT>(
        [&src, uidx, size] __device__(EltT(&r_YUV)[3], int batch_idx, int x, int y)
        { load_yuv420<IsSemiPlanar>(src, r_YUV[0], r_YUV[1], r_YUV[2], size, batch_idx, x, y, uidx); },
        [] __device__(const EltT(&r_YUV)[3], EltT(&r_BGR)[3])
        {
            yuv42xxp_to_bgr(static_cast<int>(r_YUV[0]), static_cast<int>(r_YUV[1]), static_cast<int>(r_YUV[2]),
                            r_BGR[0], r_BGR[1], r_BGR[2]);
        },
        [&dst, bidx] __device__(const EltT(&r_BGR)[3], int batch_idx, int x, int y)
        { store_bgra_nhwc(dst, r_BGR[0], r_BGR[1], r_BGR[2], Alpha<EltT>, batch_idx, x, y, bidx); },
        size);
}

// YUV 422 interleaved formats (e.g., YUYV, YVYU, and UYVY) group 2 pixels into groups of 4 elements. Each group of two
// pixels has two distinct luma (Y) values, one for each pixel. The chromaticity values (U and V) are subsampled by a
// factor of two so that there is only one U and one V value for each group of 2 pixels. Example memory layout for
// 4 x 4 image (UYVY format):
//   U_00 Y_00 V_00 Y_01 U_02 Y_02 V_02 Y_03
//   U_10 Y_10 V_10 Y_11 U_12 Y_12 V_12 Y_13
//   U_20 Y_20 V_20 Y_21 U_22 Y_22 V_22 Y_23
//   U_30 Y_30 V_30 Y_31 U_32 Y_32 V_32 Y_33
// Each U and V value corresponds to two Y values--e.g. U_00 and V_00 correspond to Y_00 and Y_10 while U_12 and V_12
// correspond to Y_12 and Y_13. Thus, a given Y value, Y_rc = Y(r,c) (where r is the row, or y coordinate, and c is the
// column, or x coordinate), corresponds to U(r,c') and V(r,c') where c' is the even column coordinate <= c -- that is,
// c' = 2 * floor(c/2) = (c & ~1). Some layouts swap the positions of the chromaticity and luma values (e.g., YUYV)
// (indicated by the yidx parameter) and / or swap the the positions of the U and V chromaticity valus (e.g., YVYU)
// (indicated by the uidx parameter).
// The data layout is treated as a single channel tensor, so each group of 4 values corresponds to two pixels. As such,
// the tensor width is twice the actual pixel width. Thus, it's easiest to process 4 consecutive values (2 pixels) per
// thread.
template<class SrcWrapper, class DstWrapper>
__global__ void yuv422_to_bgr_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int dcn, int bidx, int yidx,
                                        int uidx)
{
    using T = typename DstWrapper::ValueType;

    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dstSize.y)
        return;

    int dst_x = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // Process 2 destination pixels/thread.
    if (dst_x >= dstSize.x)
        return;

    const int batch_idx = get_batch_idx();

    const int src_x = 2 * dst_x;    // Process 4 source elements/thread (i.e., 2 destination pixels).
    const int uv_x  = (src_x & ~3); // Compute "even" x coordinate for U and V (set lowest two bits to 0).

    const T Y0 = *src.ptr(batch_idx, dst_y, src_x + yidx);
    const T Y1 = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
    const T U  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + uidx);
    const T V  = *src.ptr(batch_idx, dst_y, uv_x + (yidx ^ 1) + (uidx ^ 2));

    T r{0}, g{0}, b{0};

    yuv42xxp_to_bgr(int(Y0), int(U), int(V), b, g, r);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = Alpha<T>;

    dst_x++; // Move to next output pixel.
    yuv42xxp_to_bgr(int(Y1), int(U), int(V), b, g, r);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = Alpha<T>;
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
__global__ void yuv422_to_gray_char_nhwc(SrcWrapper src, DstWrapper dst, int2 dstSize, int yidx)
{
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_y >= dstSize.y)
        return;

    int dst_x = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // Process 2 destination pixels/thread.
    if (dst_x >= dstSize.x)
        return;

    const int batch_idx = get_batch_idx();

    const int src_x = 2 * dst_x; // Process 4 source elements/thread.

    *dst.ptr(batch_idx, dst_y, dst_x++) = *src.ptr(batch_idx, dst_y, src_x + yidx);
    *dst.ptr(batch_idx, dst_y, dst_x)   = *src.ptr(batch_idx, dst_y, src_x + yidx + 2);
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_BGR_to_RGB(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   NVCVColorConversionCode code, cuda_op::DataShape shape, int bidx,
                                   cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);
    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    rgb_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_RGB(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    int sch  = (code == NVCV_COLOR_BGRA2BGR || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int dch  = (code == NVCV_COLOR_BGR2BGRA || code == NVCV_COLOR_BGR2RGBA || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int bidx = (code != NVCV_COLOR_BGRA2BGR && code != NVCV_COLOR_BGR2BGRA) ? 2 : 0;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != sch)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C << " -- expecting " << sch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outputShape.C != dch)
    {
        LOG_ERROR("Invalid output channel number " << outputShape.C << " -- expecting " << dch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Mismatched input / output DataTypes " << inDataType << " / " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Shape mismatch -- output tensor shape " << outputShape << " doesn't match input tensor shape "
                                                           << inputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType == kCV_16F && sch < 4 && dch == 4)
    {
        LOG_ERROR("Adding alpha to the output is not supported for " << outDataType);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

#define CVCUDA_BGR2RGB_IF(SCH, DCH, SRC_T, DST_T) \
    if (sch == SCH && dch == DCH)                 \
    return Launch_BGR_to_RGB<SRC_T, DST_T>(inData, outData, code, inputShape, bidx, stream)

#define CVCUDA_BGR2RGB_CASE(T3, T4)       \
    CVCUDA_BGR2RGB_IF(3, 3, T3, T3);      \
    else CVCUDA_BGR2RGB_IF(3, 4, T3, T4); \
    else CVCUDA_BGR2RGB_IF(4, 3, T4, T3); \
    else CVCUDA_BGR2RGB_IF(4, 4, T4, T4); \
    else return ErrorCode::INVALID_DATA_SHAPE

    switch (inDataType)
    {
    case kCV_8U:
    case kCV_8S:
        CVCUDA_BGR2RGB_CASE(uchar3, uchar4);
    case kCV_16F: // Not properly handled when adding alpha to the destination.
    case kCV_16U:
    case kCV_16S:
        CVCUDA_BGR2RGB_CASE(ushort3, ushort4);
    case kCV_32S:
        CVCUDA_BGR2RGB_CASE(int3, int4);
    case kCV_32F:
        CVCUDA_BGR2RGB_CASE(float3, float4);
    case kCV_64F:
        CVCUDA_BGR2RGB_CASE(double3, double4);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_BGR2RGB_CASE
#undef CVCUDA_BGR2RGB_IF
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_GRAY_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    cuda_op::DataShape shape, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 8>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    gray_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode GRAY_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             NVCVColorConversionCode code, cudaStream_t stream)
{
    int dch = (code == NVCV_COLOR_GRAY2BGRA) ? 4 : 3;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 1)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C << " -- expecting 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outputShape.C != dch)
    {
        LOG_ERROR("Invalid output channel number " << outputShape.C << " -- expecting " << dch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Mismatched input / output DataTypes " << inDataType << " / " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Shape mismatch -- output tensor shape " << outputShape << " doesn't match input tensor shape "
                                                           << inputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType == kCV_16F && dch == 4)
    {
        LOG_ERROR("Adding alpha to the output is not supported for " << outDataType);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

#define CVCUDA_GRAY2BGR_IF(DCH, SRC_T, DST_T) \
    if (dch == DCH)                           \
    return Launch_GRAY_to_BGR<SRC_T, DST_T>(inData, outData, inputShape, stream)

#define CVCUDA_GRAY2BGR_CASE(T, T3, T4) \
    CVCUDA_GRAY2BGR_IF(3, T, T3);       \
    else CVCUDA_GRAY2BGR_IF(4, T, T4);  \
    else return ErrorCode::INVALID_DATA_SHAPE

    switch (inDataType)
    {
    case kCV_8U:
    case kCV_8S:
        CVCUDA_GRAY2BGR_CASE(uchar, uchar3, uchar4);
    case kCV_16F: // Not properly handled when adding alpha to the destination.
    case kCV_16U:
    case kCV_16S:
        CVCUDA_GRAY2BGR_CASE(ushort, ushort3, ushort4);
    case kCV_32S:
        CVCUDA_GRAY2BGR_CASE(int, int3, int4);
    case kCV_32F:
        CVCUDA_GRAY2BGR_CASE(float, float3, float4);
    case kCV_64F:
        CVCUDA_GRAY2BGR_CASE(double, double3, double4);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_GRAY2BGR_CASE
#undef CVCUDA_GRAY2BGR_IF
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_BGR_to_GRAY(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                    cuda_op::DataShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    bgr_to_gray_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_GRAY(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                             NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_RGB2GRAY) ? 2 : 0;
    int sch  = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_BGRA2GRAY) ? 4 : 3;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != sch)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C << " -- expecting " << sch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outputShape.C != 1)
    {
        LOG_ERROR("Invalid output channel number " << outputShape.C << " -- expecting 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Mismatched input / output DataTypes " << inDataType << " / " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Shape mismatch -- output tensor shape " << outputShape << " doesn't match input tensor shape "
                                                           << inputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

#define CVCUDA_BGR2GRAY_IF(SCH, SRC_T, DST_T) \
    if (sch == SCH)                           \
    return Launch_BGR_to_GRAY<SRC_T, DST_T>(inData, outData, inputShape, bidx, stream)

#define CVCUDA_BGR2GRAY_CASE(T, T3, T4) \
    CVCUDA_BGR2GRAY_IF(3, T3, T);       \
    else CVCUDA_BGR2GRAY_IF(4, T4, T);  \
    else return ErrorCode::INVALID_DATA_SHAPE

    switch (inDataType)
    {
    case kCV_8U:
        CVCUDA_BGR2GRAY_CASE(uchar, uchar3, uchar4);
    case kCV_16U:
        CVCUDA_BGR2GRAY_CASE(ushort, ushort3, ushort4);
    case kCV_32F:
        CVCUDA_BGR2GRAY_CASE(float, float3, float4);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_BGR2GRAY_CASE
#undef CVCUDA_BGR2GRAY_IF
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_BGR_to_YUV(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   cuda_op::DataShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    bgr_to_yuv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_YUV(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_BGR2YUV ? 0 : 2;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 3)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape != inputShape)
    {
        LOG_ERROR("Invalid input shape " << inputShape << " different than output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

#define CVCUDA_BGR2YUV_CASE(T3) return Launch_BGR_to_YUV<T3, T3>(inData, outData, inputShape, bidx, stream)
    switch (inDataType)
    {
    case kCV_8U:
        CVCUDA_BGR2YUV_CASE(uchar3);
    case kCV_16U:
        CVCUDA_BGR2YUV_CASE(ushort3);
    case kCV_32F:
        CVCUDA_BGR2YUV_CASE(float3);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_BGR2YUV_CASE
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_YUV_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   cuda_op::DataShape shape, int bidx, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHW<DstT>(outData);
    yuv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_YUV2BGR ? 0 : 2;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 3)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape != inputShape)
    {
        LOG_ERROR("Invalid input shape " << inputShape << " different than output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

#define CVCUDA_YUV2BGR_CASE(T3) return Launch_YUV_to_BGR<T3, T3>(inData, outData, inputShape, bidx, stream)
    switch (inDataType)
    {
    case kCV_8U:
        CVCUDA_YUV2BGR_CASE(uchar3);
    case kCV_16U:
        CVCUDA_YUV2BGR_CASE(ushort3);
    case kCV_32F:
        CVCUDA_YUV2BGR_CASE(float3);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_YUV2BGR_CASE
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_BGR_to_HSV(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   cuda_op::DataShape shape, int bidx, bool isFullRange, bool strides_64b,
                                   cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        bgr_to_hsv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        bgr_to_hsv_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_HSV(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
    int  bidx        = (code == NVCV_COLOR_BGR2HSV || code == NVCV_COLOR_BGR2HSV_FULL) ? 0 : 2;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 3)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape != inputShape)
    {
        LOG_ERROR("Invalid input shape " << inputShape << " different than output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const bool strides_64b = std::max(inAccess->sampleStride() * inAccess->numSamples(),
                                      outAccess->sampleStride() * outAccess->numSamples())
                           > nvcv::cuda::TypeTraits<int32_t>::max;

#define CVCUDA_BGR2HSV_CASE(T3) \
    return Launch_BGR_to_HSV<T3, T3>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream)

    switch (inDataType)
    {
    case kCV_8U:
        CVCUDA_BGR2HSV_CASE(uchar3);
    case kCV_32F:
        CVCUDA_BGR2HSV_CASE(float3);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_BGR2HSV_CASE
    return ErrorCode::SUCCESS;
}

template<typename SrcT, typename DstT>
inline ErrorCode Launch_HSV_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                   cuda_op::DataShape shape, int bidx, bool isFullRange, bool strides_64b,
                                   cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        hsv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData);
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        hsv_to_bgr_nhwc<Policy><<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, isFullRange);
    }
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode HSV_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                            NVCVColorConversionCode code, cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL);
    int  bidx        = (code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL) ? 0 : 2;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outputShape.C != 3 && outputShape.C != 4)
    {
        LOG_ERROR("Invalid output channel number " << outputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inputShape.C != 3)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int  dcn         = outputShape.C;
    const bool strides_64b = std::max(inAccess->sampleStride() * inAccess->numSamples(),
                                      outAccess->sampleStride() * outAccess->numSamples())
                           > nvcv::cuda::TypeTraits<int32_t>::max;

#define CVCUDA_HSV2BGR_CASE(T3, T4)                                                                            \
    if (dcn == 3)                                                                                              \
        return Launch_HSV_to_BGR<T3, T3>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream); \
    else                                                                                                       \
        return Launch_HSV_to_BGR<T3, T4>(inData, outData, inputShape, bidx, isFullRange, strides_64b, stream)

    switch (inDataType)
    {
    case kCV_8U:
        CVCUDA_HSV2BGR_CASE(uchar3, uchar4);
    case kCV_32F:
        CVCUDA_HSV2BGR_CASE(float3, float4);
    default:
        LOG_ERROR("Unsupported DataType " << inDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
#undef CVCUDA_HSV2BGR_CASE
    return ErrorCode::SUCCESS;
}

template<bool IsSemiPlanar, typename SrcT, typename DstT>
inline ErrorCode Launch_YUV420xp_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                        cuda_op::DataShape shape, int bidx, int uidx, bool strides_64b,
                                        cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(shape.W, Policy::TileWidth), divUp(shape.H, Policy::TileHeight), shape.N);

    int2 dstSize{shape.W, shape.H};

    if (strides_64b)
    {
        // YUV420 input: 4D tensor with scalar type.
        auto srcWrap = cuda::CreateTensorWrapNHWC<const SrcT, int64_t>(inData);
        // BGR output: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int64_t>(outData);
        yuv420_to_bgr_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, uidx);
    }
    else
    {
        // YUV420 input: 4D tensor with scalar type.
        auto srcWrap = cuda::CreateTensorWrapNHWC<const SrcT, int32_t>(inData);
        // BGR output: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHW<DstT, int32_t>(outData);
        yuv420_to_bgr_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, bidx, uidx);
    }
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV420xp_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                 NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2BGR_NV21
           || code == NVCV_COLOR_YUV2BGRA_NV21 || code == NVCV_COLOR_YUV2BGR_YV12 || code == NVCV_COLOR_YUV2BGRA_YV12
           || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_NV12 || code == NVCV_COLOR_YUV2BGRA_NV12 || code == NVCV_COLOR_YUV2RGB_NV12
           || code == NVCV_COLOR_YUV2RGBA_NV12 || code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV
           || code == NVCV_COLOR_YUV2RGB_IYUV || code == NVCV_COLOR_YUV2RGBA_IYUV)
            ? 0
            : 1;

    // clang-format off
    bool p420 = (code == NVCV_COLOR_YUV2BGR_YV12 || code == NVCV_COLOR_YUV2BGRA_YV12 ||
                 code == NVCV_COLOR_YUV2RGB_YV12 || code == NVCV_COLOR_YUV2RGBA_YV12 ||
                 code == NVCV_COLOR_YUV2BGR_IYUV || code == NVCV_COLOR_YUV2BGRA_IYUV ||
                 code == NVCV_COLOR_YUV2RGB_IYUV || code == NVCV_COLOR_YUV2RGBA_IYUV);
    // clang-format on

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if ((code != NVCV_COLOR_YUV2GRAY_420 || outputShape.C != 1) && outputShape.C != 3 && outputShape.C != 4)
    {
        LOG_ERROR("Invalid output channel number " << outputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inputShape.C != 1)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inputShape.H % 3 != 0 || inputShape.W % 2 != 0)
    {
        LOG_ERROR("Invalid input shape " << inputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inDataType != kCV_8U || outDataType != kCV_8U)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    int rgb_width  = inputShape.W;
    int rgb_height = inputShape.H * 2 / 3;

    if (outputShape.H != rgb_height || outputShape.W != rgb_width || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (p420 && rgb_height % 4 != 0) // YUV 420 planar formats need 4 rows of Y for every full row of U or V.
    {
        LOG_ERROR(
            "Invalid input shape: to convert from YUV 420 planar formats, the output "
            "tensor height must be a multiple of 4; height = "
            << rgb_height);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int  dcn         = outputShape.C;
    const bool strides_64b = std::max(inAccess->sampleStride() * inAccess->numSamples(),
                                      outAccess->sampleStride() * outAccess->numSamples())
                           > nvcv::cuda::TypeTraits<int32_t>::max;

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_420:
    {
        int dpitch = static_cast<int>(outAccess->rowStride());
        int spitch = static_cast<int>(inAccess->rowStride());

        for (int i = 0; i < inputShape.N; i++)
        {
            const void *srcPtr = inData.basePtr() + (size_t)i * inAccess->sampleStride();

            void *dstPtr = outData.basePtr() + (size_t)i * outAccess->sampleStride();

            checkCudaErrors(cudaMemcpy2DAsync(dstPtr, dpitch, srcPtr, spitch, rgb_width, rgb_height,
                                              cudaMemcpyDeviceToDevice, stream));
        }
    }
    break;
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2BGRA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV21:
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2RGBA_NV21:
        if (dcn == 3)
            return Launch_YUV420xp_to_BGR<true, uchar, uchar3>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                               stream);
        else
            return Launch_YUV420xp_to_BGR<true, uchar, uchar4>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                               stream);
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
        if (dcn == 3)
            return Launch_YUV420xp_to_BGR<false, uchar, uchar3>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                                stream);
        else
            return Launch_YUV420xp_to_BGR<false, uchar, uchar4>(inData, outData, outputShape, bidx, uidx, strides_64b,
                                                                stream);
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

template<bool IsSemiPlanar, typename SrcT, typename DstT>
inline ErrorCode Launch_BGR_to_YUV420xp(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                        DataShape inputShape, int bidx, int uidx, bool strides_64b, cudaStream_t stream)
{
    using Policy = CvtKernelPolicy<32, 4, 4>;

    int2 srcSize{inputShape.W, inputShape.H};

    dim3 blockSize(Policy::BlockWidth, Policy::BlockHeight);
    dim3 gridSize(divUp(inputShape.W, Policy::TileWidth), divUp(inputShape.H, Policy::TileHeight), inputShape.N);

    if (strides_64b)
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int64_t>(inData); // RGB input: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHWC<DstT, int64_t>(outData);     // YUV420 output: 4D scalar tensor.

        bgr_to_yuv420_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, srcSize, bidx, uidx);
    }
    else
    {
        auto srcWrap = cuda::CreateTensorWrapNHW<const SrcT, int32_t>(inData); // RGB input: 3D tensor with vector type.
        auto dstWrap = cuda::CreateTensorWrapNHWC<DstT, int32_t>(outData);     // YUV420 output: 4D scalar tensor.

        bgr_to_yuv420_char_nhwc<IsSemiPlanar, Policy>
            <<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, srcSize, bidx, uidx);
    }
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_YUV420xp(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                 NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_NV21
           || code == NVCV_COLOR_BGRA2YUV_NV21 || code == NVCV_COLOR_BGR2YUV_YV12 || code == NVCV_COLOR_BGRA2YUV_YV12
           || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV)
            ? 0
            : 2;

    int uidx
        = (code == NVCV_COLOR_BGR2YUV_NV12 || code == NVCV_COLOR_BGRA2YUV_NV12 || code == NVCV_COLOR_RGB2YUV_NV12
           || code == NVCV_COLOR_RGBA2YUV_NV12 || code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV
           || code == NVCV_COLOR_RGB2YUV_IYUV || code == NVCV_COLOR_RGBA2YUV_IYUV)
            ? 0
            : 1;

    // clang-format off
    bool p420 = (code == NVCV_COLOR_BGR2YUV_YV12 || code == NVCV_COLOR_BGRA2YUV_YV12 ||
                 code == NVCV_COLOR_RGB2YUV_YV12 || code == NVCV_COLOR_RGBA2YUV_YV12 ||
                 code == NVCV_COLOR_BGR2YUV_IYUV || code == NVCV_COLOR_BGRA2YUV_IYUV ||
                 code == NVCV_COLOR_RGB2YUV_IYUV || code == NVCV_COLOR_RGBA2YUV_IYUV);
    // clang-format on

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.C != 3 && inputShape.C != 4)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inputShape.H % 2 != 0 || inputShape.W % 2 != 0)
    {
        LOG_ERROR("Invalid input shape " << inputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (p420 && inputShape.H % 4 != 0) // YUV 420 planar formats need 4 rows of Y for every full row of U or V.
    {
        LOG_ERROR(
            "Invalid input shape: to convert to YUV 420 planar formats, the input "
            "tensor height must be a multiple of 4; height = "
            << inputShape.H);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inDataType != kCV_8U || outDataType != kCV_8U)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    int yuv420_width  = inputShape.W;
    int yuv420_height = inputShape.H / 2 * 3;

    if (outputShape.H != yuv420_height || outputShape.W != yuv420_width || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const bool strides_64b = std::max(inAccess->sampleStride() * inAccess->numSamples(),
                                      outAccess->sampleStride() * outAccess->numSamples())
                           > nvcv::cuda::TypeTraits<int32_t>::max;

    switch (code)
    {
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV21:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV21:
        if (inputShape.C == 3)
            return Launch_BGR_to_YUV420xp<true, uchar3, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                               stream);
        else
            return Launch_BGR_to_YUV420xp<true, uchar4, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                               stream);
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
        if (inputShape.C == 3)
            return Launch_BGR_to_YUV420xp<false, uchar3, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                                stream);
        else
            return Launch_BGR_to_YUV420xp<false, uchar4, uchar>(inData, outData, inputShape, bidx, uidx, strides_64b,
                                                                stream);
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV422_to_BGR(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                               NVCVColorConversionCode code, cudaStream_t stream)
{
    int bidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU
           || code == NVCV_COLOR_YUV2BGRA_YVYU || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY)
            ? 0
            : 2;

    int yidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_YVYU || code == NVCV_COLOR_YUV2BGRA_YVYU
           || code == NVCV_COLOR_YUV2RGB_YVYU || code == NVCV_COLOR_YUV2RGBA_YVYU || code == NVCV_COLOR_YUV2GRAY_YUY2)
            ? 0
            : 1;

    int uidx
        = (code == NVCV_COLOR_YUV2BGR_YUY2 || code == NVCV_COLOR_YUV2BGRA_YUY2 || code == NVCV_COLOR_YUV2RGB_YUY2
           || code == NVCV_COLOR_YUV2RGBA_YUY2 || code == NVCV_COLOR_YUV2BGR_UYVY || code == NVCV_COLOR_YUV2BGRA_UYVY
           || code == NVCV_COLOR_YUV2RGB_UYVY || code == NVCV_COLOR_YUV2RGBA_UYVY)
            ? 0
            : 2;

    auto inAccess = TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (inputShape.W % 4 != 0)
    {
        LOG_ERROR("Invalid input shape " << inputShape << " -- width must be a multiple of 4");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if ((code != NVCV_COLOR_YUV2GRAY_UYVY && code != NVCV_COLOR_YUV2GRAY_YUY2 || outputShape.C != 1)
        && outputShape.C != 3 && outputShape.C != 4)
    {
        LOG_ERROR("Invalid output channel number "
                  << outputShape.C
                  << " -- RGB output must have 3 or 4 channels and grayscale output must have 1 channel");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inputShape.C != 1)
    {
        LOG_ERROR("Invalid input channel number " << inputShape.C << " -- input must have 1 channel");
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (inDataType != kCV_8U || outDataType != kCV_8U)
    {
        LOG_ERROR("Unsupported input / output DataType " << inDataType << " / " << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || 2 * outputShape.W != inputShape.W || outputShape.N != inputShape.N)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(inputShape.W / 4, blockSize.x), divUp(inputShape.H, blockSize.y), inputShape.N);

    int2 dstSize{outputShape.W, outputShape.H};
    int  dcn = outputShape.C;

    auto srcWrap = cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dstWrap = cuda::CreateTensorWrapNHWC<uint8_t>(outData);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_YUY2:
    case NVCV_COLOR_YUV2GRAY_UYVY:
    {
        yuv422_to_gray_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, yidx);
        checkKernelErrors();
    }
    break;
    case NVCV_COLOR_YUV2BGR_YUY2:
    case NVCV_COLOR_YUV2BGR_YVYU:
    case NVCV_COLOR_YUV2BGRA_YUY2:
    case NVCV_COLOR_YUV2BGRA_YVYU:
    case NVCV_COLOR_YUV2RGB_YUY2:
    case NVCV_COLOR_YUV2RGB_YVYU:
    case NVCV_COLOR_YUV2RGBA_YUY2:
    case NVCV_COLOR_YUV2RGBA_YVYU:
    case NVCV_COLOR_YUV2RGB_UYVY:
    case NVCV_COLOR_YUV2BGR_UYVY:
    case NVCV_COLOR_YUV2RGBA_UYVY:
    case NVCV_COLOR_YUV2BGRA_UYVY:
    {
        yuv422_to_bgr_char_nhwc<<<gridSize, blockSize, 0, stream>>>(srcWrap, dstWrap, dstSize, dcn, bidx, yidx, uidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_PARAMETER;
    }
    return ErrorCode::SUCCESS;
}

ErrorCode CvtColor::infer(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                          NVCVColorConversionCode code, cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid input DataFormat " << format << ", the valid DataFormats are: \"NHWC\", \"HWC\"");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    typedef ErrorCode (*func_t)(const TensorDataStridedCuda &inData, const TensorDataStridedCuda &outData,
                                NVCVColorConversionCode code, cudaStream_t stream);

    static const func_t funcs[] = {
        BGR_to_RGB, // CV_BGR2BGRA    =0
        BGR_to_RGB, // CV_BGRA2BGR    =1
        BGR_to_RGB, // CV_BGR2RGBA    =2
        BGR_to_RGB, // CV_RGBA2BGR    =3
        BGR_to_RGB, // CV_BGR2RGB     =4
        BGR_to_RGB, // CV_BGRA2RGBA   =5

        BGR_to_GRAY, // CV_BGR2GRAY    =6
        BGR_to_GRAY, // CV_RGB2GRAY    =7
        GRAY_to_BGR, // CV_GRAY2BGR    =8
        0,           //GRAY_to_BGRA,           // CV_GRAY2BGRA   =9
        0,           //BGRA_to_GRAY,           // CV_BGRA2GRAY   =10
        0,           //RGBA_to_GRAY,           // CV_RGBA2GRAY   =11

        0, //BGR_to_BGR565,          // CV_BGR2BGR565  =12
        0, //RGB_to_BGR565,          // CV_RGB2BGR565  =13
        0, //BGR565_to_BGR,          // CV_BGR5652BGR  =14
        0, //BGR565_to_RGB,          // CV_BGR5652RGB  =15
        0, //BGRA_to_BGR565,         // CV_BGRA2BGR565 =16
        0, //RGBA_to_BGR565,         // CV_RGBA2BGR565 =17
        0, //BGR565_to_BGRA,         // CV_BGR5652BGRA =18
        0, //BGR565_to_RGBA,         // CV_BGR5652RGBA =19

        0, //GRAY_to_BGR565,         // CV_GRAY2BGR565 =20
        0, //BGR565_to_GRAY,         // CV_BGR5652GRAY =21

        0, //BGR_to_BGR555,          // CV_BGR2BGR555  =22
        0, //RGB_to_BGR555,          // CV_RGB2BGR555  =23
        0, //BGR555_to_BGR,          // CV_BGR5552BGR  =24
        0, //BGR555_to_RGB,          // CV_BGR5552RGB  =25
        0, //BGRA_to_BGR555,         // CV_BGRA2BGR555 =26
        0, //RGBA_to_BGR555,         // CV_RGBA2BGR555 =27
        0, //BGR555_to_BGRA,         // CV_BGR5552BGRA =28
        0, //BGR555_to_RGBA,         // CV_BGR5552RGBA =29

        0, //GRAY_to_BGR555,         // CV_GRAY2BGR555 =30
        0, //BGR555_to_GRAY,         // CV_BGR5552GRAY =31

        0, //BGR_to_XYZ,             // CV_BGR2XYZ     =32
        0, //RGB_to_XYZ,             // CV_RGB2XYZ     =33
        0, //XYZ_to_BGR,             // CV_XYZ2BGR     =34
        0, //XYZ_to_RGB,             // CV_XYZ2RGB     =35

        0, //BGR_to_YCrCb,           // CV_BGR2YCrCb   =36
        0, //RGB_to_YCrCb,           // CV_RGB2YCrCb   =37
        0, //YCrCb_to_BGR,           // CV_YCrCb2BGR   =38
        0, //YCrCb_to_RGB,           // CV_YCrCb2RGB   =39

        BGR_to_HSV, //BGR_to_HSV,             // CV_BGR2HSV     =40
        BGR_to_HSV, //RGB_to_HSV,             // CV_RGB2HSV     =41

        0, //                =42
        0, //                =43

        0, //BGR_to_Lab,             // CV_BGR2Lab     =44
        0, //RGB_to_Lab,             // CV_RGB2Lab     =45

        0, //bayerBG_to_BGR,         // CV_BayerBG2BGR =46
        0, //bayeRGB_to_BGR,         // CV_BayeRGB2BGR =47
        0, //bayerRG_to_BGR,         // CV_BayerRG2BGR =48
        0, //bayerGR_to_BGR,         // CV_BayerGR2BGR =49

        0, //BGR_to_Luv,             // CV_BGR2Luv     =50
        0, //RGB_to_Luv,             // CV_RGB2Luv     =51

        0, //BGR_to_HLS,             // CV_BGR2HLS     =52
        0, //RGB_to_HLS,             // CV_RGB2HLS     =53

        HSV_to_BGR, // CV_HSV2BGR     =54
        HSV_to_BGR, // CV_HSV2RGB     =55

        0, //Lab_to_BGR,             // CV_Lab2BGR     =56
        0, //Lab_to_RGB,             // CV_Lab2RGB     =57
        0, //Luv_to_BGR,             // CV_Luv2BGR     =58
        0, //Luv_to_RGB,             // CV_Luv2RGB     =59

        0, //HLS_to_BGR,             // CV_HLS2BGR     =60
        0, //HLS_to_RGB,             // CV_HLS2RGB     =61

        0, // CV_BayerBG2BGR_VNG =62
        0, // CV_BayeRGB2BGR_VNG =63
        0, // CV_BayerRG2BGR_VNG =64
        0, // CV_BayerGR2BGR_VNG =65

        BGR_to_HSV, //BGR_to_HSV_FULL,        // CV_BGR2HSV_FULL = 66
        BGR_to_HSV, //RGB_to_HSV_FULL,        // CV_RGB2HSV_FULL = 67
        0,          //BGR_to_HLS_FULL,        // CV_BGR2HLS_FULL = 68
        0,          //RGB_to_HLS_FULL,        // CV_RGB2HLS_FULL = 69

        HSV_to_BGR, // CV_HSV2BGR_FULL = 70
        HSV_to_BGR, // CV_HSV2RGB_FULL = 71
        0,          //HLS_to_BGR_FULL,        // CV_HLS2BGR_FULL = 72
        0,          //HLS_to_RGB_FULL,        // CV_HLS2RGB_FULL = 73

        0, //LBGR_to_Lab,            // CV_LBGR2Lab     = 74
        0, //LRGB_to_Lab,            // CV_LRGB2Lab     = 75
        0, //LBGR_to_Luv,            // CV_LBGR2Luv     = 76
        0, //LRGB_to_Luv,            // CV_LRGB2Luv     = 77

        0, //Lab_to_LBGR,            // CV_Lab2LBGR     = 78
        0, //Lab_to_LRGB,            // CV_Lab2LRGB     = 79
        0, //Luv_to_LBGR,            // CV_Luv2LBGR     = 80
        0, //Luv_to_LRGB,            // CV_Luv2LRGB     = 81

        BGR_to_YUV, // CV_BGR2YUV      = 82
        BGR_to_YUV, // CV_RGB2YUV      = 83
        YUV_to_BGR, // CV_YUV2BGR      = 84
        YUV_to_BGR, // CV_YUV2RGB      = 85

        0, //bayerBG_to_gray,        // CV_BayerBG2GRAY = 86
        0, //bayeRGB_to_GRAY,        // CV_BayeRGB2GRAY = 87
        0, //bayerRG_to_gray,        // CV_BayerRG2GRAY = 88
        0, //bayerGR_to_gray,        // CV_BayerGR2GRAY = 89

        //! YUV 4:2:0 family to RGB
        YUV420xp_to_BGR, // CV_YUV2RGB_NV12 = 90,
        YUV420xp_to_BGR, // CV_YUV2BGR_NV12 = 91,
        YUV420xp_to_BGR, // CV_YUV2RGB_NV21 = 92, CV_YUV420sp2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_NV21 = 93, CV_YUV420sp2BGR

        YUV420xp_to_BGR, // CV_YUV2RGBA_NV12 = 94,
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV12 = 95,
        YUV420xp_to_BGR, // CV_YUV2RGBA_NV21 = 96, CV_YUV420sp2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_NV21 = 97, CV_YUV420sp2BGRA

        YUV420xp_to_BGR, // CV_YUV2RGB_YV12 = 98, CV_YUV420p2RGB
        YUV420xp_to_BGR, // CV_YUV2BGR_YV12 = 99, CV_YUV420p2BGR
        YUV420xp_to_BGR, // CV_YUV2RGB_IYUV = 100, CV_YUV2RGB_I420
        YUV420xp_to_BGR, // CV_YUV2BGR_IYUV = 101, CV_YUV2BGR_I420

        YUV420xp_to_BGR, // CV_YUV2RGBA_YV12 = 102, CV_YUV420p2RGBA
        YUV420xp_to_BGR, // CV_YUV2BGRA_YV12 = 103, CV_YUV420p2BGRA
        YUV420xp_to_BGR, // CV_YUV2RGBA_IYUV = 104, CV_YUV2RGBA_I420
        YUV420xp_to_BGR, // CV_YUV2BGRA_IYUV = 105, CV_YUV2BGRA_I420

        YUV420xp_to_BGR, // CV_YUV2GRAY_420 = 106,
        // CV_YUV2GRAY_NV21,
        // CV_YUV2GRAY_NV12,
        // CV_YUV2GRAY_YV12,
        // CV_YUV2GRAY_IYUV,
        // CV_YUV2GRAY_I420,
        // CV_YUV420sp2GRAY,
        // CV_YUV420p2GRAY ,

        //! YUV 4:2:2 family to RGB
        YUV422_to_BGR, // CV_YUV2RGB_UYVY = 107, CV_YUV2RGB_Y422, CV_YUV2RGB_UYNV
        YUV422_to_BGR, // CV_YUV2BGR_UYVY = 108, CV_YUV2BGR_Y422, CV_YUV2BGR_UYNV
        0,             // CV_YUV2RGB_VYUY = 109,
        0,             // CV_YUV2BGR_VYUY = 110,

        YUV422_to_BGR, // CV_YUV2RGBA_UYVY = 111, CV_YUV2RGBA_Y422, CV_YUV2RGBA_UYNV
        YUV422_to_BGR, // CV_YUV2BGRA_UYVY = 112, CV_YUV2BGRA_Y422, CV_YUV2BGRA_UYNV
        0,             // CV_YUV2RGBA_VYUY = 113,
        0,             // CV_YUV2BGRA_VYUY = 114,

        YUV422_to_BGR, // CV_YUV2RGB_YUY2 = 115, CV_YUV2RGB_YUYV, CV_YUV2RGB_YUNV
        YUV422_to_BGR, // CV_YUV2BGR_YUY2 = 116, CV_YUV2BGR_YUYV, CV_YUV2BGR_YUNV
        YUV422_to_BGR, // CV_YUV2RGB_YVYU = 117,
        YUV422_to_BGR, // CV_YUV2BGR_YVYU = 118,

        YUV422_to_BGR, // CV_YUV2RGBA_YUY2 = 119, CV_YUV2RGBA_YUYV, CV_YUV2RGBA_YUNV
        YUV422_to_BGR, // CV_YUV2BGRA_YUY2 = 120, CV_YUV2BGRA_YUYV, CV_YUV2BGRA_YUNV
        YUV422_to_BGR, // CV_YUV2RGBA_YVYU = 121,
        YUV422_to_BGR, // CV_YUV2BGRA_YVYU = 122,

        YUV422_to_BGR, // CV_YUV2GRAY_UYVY = 123, CV_YUV2GRAY_Y422, CV_YUV2GRAY_UYNV
        YUV422_to_BGR, // CV_YUV2GRAY_YUY2 = 124, CV_YUV2GRAY_YVYU, CV_YUV2GRAY_YUYV, CV_YUV2GRAY_YUNV

        //! alpha premultiplication
        0, //RGBA_to_mBGRA,         // CV_RGBA2mRGBA = 125,
        0, // CV_mRGBA2RGBA = 126,

        //! RGB to YUV 4:2:0 family (three plane YUV)
        BGR_to_YUV420xp, // CV_RGB2YUV_I420  = 127, CV_RGB2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGR2YUV_I420  = 128, CV_BGR2YUV_IYUV

        BGR_to_YUV420xp, // CV_RGBA2YUV_I420 = 129, CV_RGBA2YUV_IYUV
        BGR_to_YUV420xp, // CV_BGRA2YUV_I420 = 130, CV_BGRA2YUV_IYUV
        BGR_to_YUV420xp, // CV_RGB2YUV_YV12  = 131,
        BGR_to_YUV420xp, // CV_BGR2YUV_YV12  = 132,
        BGR_to_YUV420xp, // CV_RGBA2YUV_YV12 = 133,
        BGR_to_YUV420xp, // CV_BGRA2YUV_YV12 = 134,

        //! Edge-Aware Demosaicing
        0, // CV_BayerBG2BGR_EA  = 135,
        0, // CV_BayerGB2BGR_EA  = 136,
        0, // CV_BayerRG2BGR_EA  = 137,
        0, // CV_BayerGR2BGR_EA  = 138,

        0, // OpenCV COLORCVT_MAX = 139

        //! RGB to YUV 4:2:0 family (two plane YUV, not in OpenCV)
        BGR_to_YUV420xp, // CV_RGB2YUV_NV12 = 140,
        BGR_to_YUV420xp, // CV_BGR2YUV_NV12 = 141,
        BGR_to_YUV420xp, // CV_RGB2YUV_NV21 = 142, CV_RGB2YUV420sp
        BGR_to_YUV420xp, // CV_BGR2YUV_NV21 = 143, CV_BGR2YUV420sp

        BGR_to_YUV420xp, // CV_RGBA2YUV_NV12 = 144,
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV12 = 145,
        BGR_to_YUV420xp, // CV_RGBA2YUV_NV21 = 146, CV_RGBA2YUV420sp
        BGR_to_YUV420xp, // CV_BGRA2YUV_NV21 = 147, CV_BGRA2YUV420sp

        0, // CV_COLORCVT_MAX  = 148
    };

    func_t func = funcs[code];

    if (func == 0)
    {
        LOG_ERROR("Invalid convert color code: " << code);
        return ErrorCode::INVALID_PARAMETER;
    }

    return func(inData, outData, code, stream);
}

} // namespace nvcv::legacy::cuda_op
