/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static constexpr float B2YF = 0.114f;
static constexpr float G2YF = 0.587f;
static constexpr float R2YF = 0.299f;

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

static constexpr float B2UF = 0.492f;
static constexpr float R2VF = 0.877f;

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

namespace nvcv::legacy::cuda_op {

__device__ inline bool checkShapeFromYUV420(int rows, int cols, NVCVColorConversionCode code)
{
    int valid_row = 1, valid_col = 1;
    switch (code)
    {
    case NVCV_COLOR_YUV2BGR_NV12:
    case NVCV_COLOR_YUV2BGR_NV21:
    case NVCV_COLOR_YUV2BGRA_NV12:
    case NVCV_COLOR_YUV2BGRA_NV21:
    case NVCV_COLOR_YUV2RGB_NV12:
    case NVCV_COLOR_YUV2RGB_NV21:
    case NVCV_COLOR_YUV2RGBA_NV12:
    case NVCV_COLOR_YUV2RGBA_NV21:
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
    case NVCV_COLOR_YUV2GRAY_420:
        valid_row = 3;
        valid_col = 2;
        break;
    case NVCV_COLOR_BGR2YUV_NV12:
    case NVCV_COLOR_BGR2YUV_NV21:
    case NVCV_COLOR_BGRA2YUV_NV12:
    case NVCV_COLOR_BGRA2YUV_NV21:
    case NVCV_COLOR_RGB2YUV_NV12:
    case NVCV_COLOR_RGB2YUV_NV21:
    case NVCV_COLOR_RGBA2YUV_NV12:
    case NVCV_COLOR_RGBA2YUV_NV21:
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
        valid_row = 2;
        valid_col = 2;
        break;
    default:
        return false;
    }
    if (rows % valid_row != 0 || cols % valid_col != 0)
    {
        return false;
    }
    return true;
}

template<class T>
__global__ void rgb_to_bgr_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int sch, int dch, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    T g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = r;

    if (dch == 4)
    {
        T al = sch == 4 ? *src.ptr(batch_idx, dst_y, dst_x, 3) : cuda::TypeTraits<T>::max;
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = al;
    }
}

template<class T>
__global__ void gray_to_bgr_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dch)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T g = *src.ptr(batch_idx, dst_y, dst_x, 0);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = g;
    if (dch == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = g;
    }
}

template<class T>
__global__ void bgr_to_gray_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    int b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    T gray                               = (T)CV_DESCALE(b * BY15 + g * GY15 + r * RY15, gray_shift);
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = gray;
}

template<class T>
__global__ void bgr_to_gray_float_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    T g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    T gray = (T)(b * B2YF + g * G2YF + r * R2YF) * dst.ptr(batch_idx, dst_y, dst_x, 0) = gray;
}

template<class T>
__global__ void bgr_to_yuv_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    int B = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int G = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int R = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    int C0 = R2Y, C1 = G2Y, C2 = B2Y, C3 = R2VI, C4 = B2UI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1)) * (1 << yuv_shift);
    int Y     = CV_DESCALE(R * C0 + G * C1 + B * C2, yuv_shift);
    int Cr    = CV_DESCALE((R - Y) * C3 + delta, yuv_shift);
    int Cb    = CV_DESCALE((B - Y) * C4 + delta, yuv_shift);

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<T>(Y);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = cuda::SaturateCast<T>(Cb);
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = cuda::SaturateCast<T>(Cr);
}

template<class T>
__global__ void bgr_to_yuv_float_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T B = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    T G = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T R = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);

    T C0 = R2YF, C1 = G2YF, C2 = B2YF, C3 = R2VF, C4 = B2UF;
    T delta                              = 0.5f;
    T Y                                  = R * C0 + G * C1 + B * C2;
    T Cr                                 = (R - Y) * C3 + delta;
    T Cb                                 = (B - Y) * C4 + delta;
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = Y;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = Cb;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = Cr;
}

template<class T>
__global__ void yuv_to_bgr_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T Y  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    T Cb = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T Cr = *src.ptr(batch_idx, dst_y, dst_x, 2);

    int C0 = V2RI, C1 = V2GI, C2 = U2GI, C3 = U2BI;
    int delta = ((T)(cuda::TypeTraits<T>::max / 2 + 1));
    int b     = Y + CV_DESCALE((Cb - delta) * C3, yuv_shift);
    int g     = Y + CV_DESCALE((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
    int r     = Y + CV_DESCALE((Cr - delta) * C0, yuv_shift);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<T>(b);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<T>(g);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<T>(r);
}

template<class T>
__global__ void yuv_to_bgr_float_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    T Y  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    T Cb = *src.ptr(batch_idx, dst_y, dst_x, 1);
    T Cr = *src.ptr(batch_idx, dst_y, dst_x, 2);

    T C0 = V2RF, C1 = V2GF, C2 = U2GF, C3 = U2BF;
    T delta = 0.5f;
    T b     = Y + (Cb - delta) * C3;
    T g     = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
    T r     = Y + (Cr - delta) * C0;

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
}

template<class T>
__global__ void bgr_to_hsv_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx, bool isFullRange)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    int       b         = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    int       g         = *src.ptr(batch_idx, dst_y, dst_x, 1);
    int       r         = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);
    int       hrange    = isFullRange ? 256 : 180;
    int       hr        = hrange;
    const int hsv_shift = 12;
    int       h, s, v = b;
    int       vmin = b;
    int       vr, vg;

    v    = cuda::max(v, g);
    v    = cuda::max(v, r);
    vmin = min(vmin, g);
    vmin = min(vmin, r);

    unsigned char diff = cuda::SaturateCast<unsigned char>(v - vmin);
    vr                 = v == r ? -1 : 0;
    vg                 = v == g ? -1 : 0;

    int hdiv_table = diff == 0 ? 0 : cuda::SaturateCast<int>((hrange << hsv_shift) / (6. * diff));
    int sdiv_table = v == 0 ? 0 : cuda::SaturateCast<int>((255 << hsv_shift) / (1. * v));
    s              = (diff * sdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h              = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h              = (h * hdiv_table + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += h < 0 ? hr : 0;

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = cuda::SaturateCast<unsigned char>(h);
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = (unsigned char)s;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = (unsigned char)v;
}

template<class T>
__global__ void bgr_to_hsv_float_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    float b = *src.ptr(batch_idx, dst_y, dst_x, bidx);
    float g = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float r = *src.ptr(batch_idx, dst_y, dst_x, bidx ^ 2);
    float h, s, v;
    float hrange = 360.0;
    float hscale = hrange * (1.f / 360.f);

    float vmin, diff;

    v = vmin = r;
    if (v < g)
        v = g;
    if (v < b)
        v = b;
    if (vmin > g)
        vmin = g;
    if (vmin > b)
        vmin = b;

    diff = v - vmin;
    s    = diff / (float)(fabs(v) + FLT_EPSILON);
    diff = (float)(60. / (diff + FLT_EPSILON));
    if (v == r)
        h = (g - b) * diff;
    else if (v == g)
        h = (b - r) * diff + 120.f;
    else
        h = (r - g) * diff + 240.f;

    if (h < 0)
        h += 360.f;

    *dst.ptr(batch_idx, dst_y, dst_x, 0) = h * hscale;
    *dst.ptr(batch_idx, dst_y, dst_x, 1) = s;
    *dst.ptr(batch_idx, dst_y, dst_x, 2) = v;
}

__device__ inline void HSV2RGB_native_var_shape(float h, float s, float v, float &b, float &g, float &r,
                                                const float hscale)
{
    if (s == 0)
        b = g = r = v;
    else
    {
        static const int sector_data[][3] = {
            {1, 3, 0},
            {1, 0, 2},
            {3, 0, 1},
            {0, 2, 1},
            {0, 1, 3},
            {2, 1, 0}
        };
        float tab[4];
        int   sector;
        h *= hscale;
        h      = fmod(h, 6.f);
        sector = (int)floor(h);
        h -= sector;
        if ((unsigned)sector >= 6u)
        {
            sector = 0;
            h      = 0.f;
        }

        tab[0] = v;
        tab[1] = v * (1.f - s);
        tab[2] = v * (1.f - s * h);
        tab[3] = v * (1.f - s * (1.f - h));

        b = tab[sector_data[sector][0]];
        g = tab[sector_data[sector][1]];
        r = tab[sector_data[sector][2]];
    }
}

template<class T>
__global__ void hsv_to_bgr_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx, int dcn,
                                     bool isFullRange)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    float h = *src.ptr(batch_idx, dst_y, dst_x, 0);
    float s = *src.ptr(batch_idx, dst_y, dst_x, 1) * (1.0f / 255.0f);
    float v = *src.ptr(batch_idx, dst_y, dst_x, 2) * (1.0f / 255.0f);

    float         hrange = isFullRange ? 255 : 180;
    unsigned char alpha  = cuda::TypeTraits<T>::max;
    float         hs     = 6.f / hrange;

    float b, g, r;
    HSV2RGB_native_var_shape(h, s, v, b, g, r, hs);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = cuda::SaturateCast<uchar>(b * 255.0f);
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = cuda::SaturateCast<uchar>(g * 255.0f);
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = cuda::SaturateCast<uchar>(r * 255.0f);
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = alpha;
}

template<class T>
__global__ void hsv_to_bgr_float_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int bidx, int dcn)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    float h = *src.ptr(batch_idx, dst_y, dst_x, 0);
    float s = *src.ptr(batch_idx, dst_y, dst_x, 1);
    float v = *src.ptr(batch_idx, dst_y, dst_x, 2);

    float hrange = 360.0;
    float alpha  = 1.f;
    float hs     = 6.f / hrange;

    float b, g, r;
    HSV2RGB_native_var_shape(h, s, v, b, g, r, hs);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = alpha;
}

__device__ __forceinline__ void yuv42xxp_to_bgr_kernel(const int &Y, const int &U, const int &V, uchar &r, uchar &g,
                                                       uchar &b)
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

__device__ __forceinline__ void bgr_to_yuv42xxp_kernel(const uchar &r, const uchar &g, const uchar &b, uchar &Y,
                                                       uchar &U, uchar &V)
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

template<class T>
__global__ void bgr_to_yuv420sp_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dcn, int bidx,
                                          int uidx, NVCVColorConversionCode code)
{
    int       src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       src_cols  = src.at_cols(batch_idx);
    int       src_rows  = src.at_rows(batch_idx);
    if (src_x >= src_cols || src_y >= src_rows)
        return;

    assert(checkShapeFromYUV420(dst.at_rows(batch_idx), dst.at_cols(batch_idx), code));

    int uv_x = (src_x % 2 == 0) ? src_x : (src_x - 1);

    uchar b = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx));
    uchar g = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, 1));
    uchar r = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx ^ 2));
    // Ignore gray channel if input is RGBA

    uchar Y{0}, U{0}, V{0};
    bgr_to_yuv42xxp_kernel(r, g, b, Y, U, V);

    *dst.ptr(batch_idx, src_y, src_x, 0) = Y;
    if (src_y % 2 == 0 && src_x % 2 == 0)
    {
        *dst.ptr(batch_idx, src_rows + src_y / 2, uv_x + uidx)       = U;
        *dst.ptr(batch_idx, src_rows + src_y / 2, uv_x + (1 - uidx)) = V;
    }
}

template<class T>
__global__ void bgr_to_yuv420p_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dcn, int bidx,
                                         int uidx, NVCVColorConversionCode code)
{
    int       src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       src_cols  = src.at_cols(batch_idx);
    int       src_rows  = src.at_rows(batch_idx);
    if (src_x >= src_cols || src_y >= src_rows)
        return;

    assert(checkShapeFromYUV420(dst.at_rows(batch_idx), dst.at_cols(batch_idx), code));

    int plane_y_step  = src_rows * src_cols;
    int plane_uv_step = plane_y_step / 4;
    int uv_x          = (src_y % 4 < 2) ? src_x / 2 : (src_x / 2 + src_cols / 2);

    uchar b = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx));
    uchar g = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, 1));
    uchar r = static_cast<uchar>(*src.ptr(batch_idx, src_y, src_x, bidx ^ 2));
    // Ignore gray channel if input is RGBA

    uchar Y{0}, U{0}, V{0};
    bgr_to_yuv42xxp_kernel(r, g, b, Y, U, V);

    *dst.ptr(batch_idx, src_y, src_x, 0) = Y;
    if (src_y % 2 == 0 && src_x % 2 == 0)
    {
        *dst.ptr(batch_idx, src_rows + src_y / 4, uv_x + plane_uv_step * uidx)       = U;
        *dst.ptr(batch_idx, src_rows + src_y / 4, uv_x + plane_uv_step * (1 - uidx)) = V;
    }
}

template<class T>
__global__ void yuv420sp_to_bgr_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dcn, int bidx,
                                          int uidx, NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       dst_cols  = dst.at_cols(batch_idx);
    int       dst_rows  = dst.at_rows(batch_idx);
    if (dst_x >= dst_cols || dst_y >= dst_rows)
        return;

    assert(checkShapeFromYUV420(src.at_rows(batch_idx), src.at_cols(batch_idx), code));

    int uv_x = (dst_x % 2 == 0) ? dst_x : (dst_x - 1);

    T Y = *src.ptr(batch_idx, dst_y, dst_x);
    T U = *src.ptr(batch_idx, dst_rows + dst_y / 2, uv_x + uidx);
    T V = *src.ptr(batch_idx, dst_rows + dst_y / 2, uv_x + 1 - uidx);

    uchar r{0}, g{0}, b{0}, a{0xff};
    yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

template<class T>
__global__ void yuv420p_to_bgr_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dcn, int bidx,
                                         int uidx, NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       dst_cols  = dst.at_cols(batch_idx);
    int       dst_rows  = dst.at_rows(batch_idx);
    if (dst_x >= dst_cols || dst_y >= dst_rows)
        return;

    assert(checkShapeFromYUV420(src.at_rows(batch_idx), src.at_cols(batch_idx), code));

    int plane_y_step  = dst_rows * dst_cols;
    int plane_uv_step = plane_y_step / 4;
    int uv_x          = (dst_y % 4 < 2) ? dst_x / 2 : (dst_x / 2 + dst_cols / 2);

    T Y = *src.ptr(batch_idx, dst_y, dst_x);
    T U = *src.ptr(batch_idx, dst_rows + dst_y / 4, uv_x + plane_uv_step * uidx);
    T V = *src.ptr(batch_idx, dst_rows + dst_y / 4, uv_x + plane_uv_step * (1 - uidx));

    uchar r{0}, g{0}, b{0}, a{0xff};
    yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

template<class T>
__global__ void yuv422_to_bgr_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int dcn, int bidx, int yidx,
                                        int uidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       dst_cols  = dst.at_cols(batch_idx);
    int       dst_rows  = dst.at_rows(batch_idx);
    if (dst_x >= dst_cols || dst_y >= dst_rows)
        return;
    int uv_x = (dst_x % 2 == 0) ? dst_x : dst_x - 1;

    T Y = *src.ptr(batch_idx, dst_y, dst_x, yidx);
    T U = *src.ptr(batch_idx, dst_y, uv_x, (1 - yidx) + uidx);
    T V = *src.ptr(batch_idx, dst_y, uv_x, (1 - yidx) + uidx ^ 2);

    uchar r{0}, g{0}, b{0}, a{0xff};
    yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

    *dst.ptr(batch_idx, dst_y, dst_x, bidx)     = b;
    *dst.ptr(batch_idx, dst_y, dst_x, 1)        = g;
    *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
    if (dcn == 4)
    {
        *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
    }
}

template<class T>
__global__ void yuv420_to_gray_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst,
                                         NVCVColorConversionCode code)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;

    assert(checkShapeFromYUV420(src.at_rows(batch_idx), src.at_cols(batch_idx), code));

    T Y                                  = *src.ptr(batch_idx, dst_y, dst_x, 0);
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = Y;
}

template<class T>
__global__ void yuv422_to_gray_char_nhwc(Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, int yidx)
{
    int       dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int       dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (dst_x >= dst.at_cols(batch_idx) || dst_y >= dst.at_rows(batch_idx))
        return;
    T Y                                  = *src.ptr(batch_idx, dst_y, dst_x, yidx);
    *dst.ptr(batch_idx, dst_y, dst_x, 0) = Y;
}

inline ErrorCode BGR_to_RGB(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    int sch  = (code == NVCV_COLOR_BGRA2BGR || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int dch  = (code == NVCV_COLOR_BGR2BGRA || code == NVCV_COLOR_BGR2RGBA || code == NVCV_COLOR_BGRA2RGBA) ? 4 : 3;
    int bidx = (code == NVCV_COLOR_BGR2RGB || code == NVCV_COLOR_RGBA2BGR || code == NVCV_COLOR_BGRA2RGBA
                || code == NVCV_COLOR_BGR2RGBA)
                 ? 2
                 : 0;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != sch)
    {
        LOG_ERROR("Invalid input channel number " << channels << " expecting: " << sch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != dch)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " expecting: " << dch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    case kCV_8S:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        rgb_to_bgr_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, sch, dch, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_16U:
    case kCV_16F:
    case kCV_16S:
    {
        cuda_op::Ptr2dVarShapeNHWC<uint16_t> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<uint16_t> dst_ptr(outData);
        rgb_to_bgr_nhwc<uint16_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, sch, dch, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_32S:
    {
        cuda_op::Ptr2dVarShapeNHWC<int32_t> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<int32_t> dst_ptr(outData);
        rgb_to_bgr_nhwc<int32_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, sch, dch, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        rgb_to_bgr_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, sch, dch, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_64F:
    {
        cuda_op::Ptr2dVarShapeNHWC<double> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<double> dst_ptr(outData);
        rgb_to_bgr_nhwc<double><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, sch, dch, bidx);
        checkKernelErrors();
    }
    break;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode GRAY_to_BGR(const IImageBatchVarShapeDataStridedCuda &inData,
                             const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                             cudaStream_t stream)
{
    int dch = (code == NVCV_COLOR_GRAY2BGRA) ? 4 : 3;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 1)
    {
        LOG_ERROR("Invalid input channel number " << channels << " expecting: 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != dch)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " expecting: " << dch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    case kCV_8S:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        gray_to_bgr_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch);
        checkKernelErrors();
    }
    break;
    case kCV_16U:
    case kCV_16F:
    case kCV_16S:
    {
        cuda_op::Ptr2dVarShapeNHWC<uint16_t> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<uint16_t> dst_ptr(outData);
        gray_to_bgr_nhwc<uint16_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch);
        checkKernelErrors();
    }
    break;
    case kCV_32S:
    {
        cuda_op::Ptr2dVarShapeNHWC<int32_t> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<int32_t> dst_ptr(outData);
        gray_to_bgr_nhwc<int32_t><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        gray_to_bgr_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch);
        checkKernelErrors();
    }
    break;
    case kCV_64F:
    {
        cuda_op::Ptr2dVarShapeNHWC<double> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<double> dst_ptr(outData);
        gray_to_bgr_nhwc<double><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dch);
        checkKernelErrors();
    }
    break;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_GRAY(const IImageBatchVarShapeDataStridedCuda &inData,
                             const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                             cudaStream_t stream)
{
    int bidx = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_RGB2GRAY) ? 2 : 0;
    int sch  = (code == NVCV_COLOR_RGBA2GRAY || code == NVCV_COLOR_BGRA2GRAY) ? 4 : 3;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != sch)
    {
        LOG_ERROR("Invalid input channel number " << channels << " expecting: " << sch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 1)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " expecting: 1");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        bgr_to_gray_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_16U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> dst_ptr(outData);
        bgr_to_gray_char_nhwc<unsigned short><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        bgr_to_gray_char_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_YUV(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_BGR2YUV ? 0 : 2;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 3)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " expecting: 3");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        bgr_to_yuv_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_16U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> dst_ptr(outData);
        bgr_to_yuv_char_nhwc<unsigned short><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        bgr_to_yuv_float_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV_to_BGR(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    int bidx = code == NVCV_COLOR_YUV2BGR ? 0 : 2;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 3)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != channels)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " different than input channel " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        yuv_to_bgr_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_16U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned short> dst_ptr(outData);
        yuv_to_bgr_char_nhwc<unsigned short><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        yuv_to_bgr_float_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode BGR_to_HSV(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_BGR2HSV_FULL || code == NVCV_COLOR_RGB2HSV_FULL);
    int  bidx        = (code == NVCV_COLOR_BGR2HSV || code == NVCV_COLOR_BGR2HSV_FULL) ? 0 : 2;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 3)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != channels)
    {
        LOG_ERROR("Invalid output channel number " << dcn << " different than input channel " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        bgr_to_hsv_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, isFullRange);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        bgr_to_hsv_float_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode HSV_to_BGR(const IImageBatchVarShapeDataStridedCuda &inData,
                            const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                            cudaStream_t stream)
{
    bool isFullRange = (code == NVCV_COLOR_HSV2BGR_FULL || code == NVCV_COLOR_HSV2RGB_FULL);
    int  bidx        = (code == NVCV_COLOR_HSV2BGR || code == NVCV_COLOR_HSV2BGR_FULL) ? 0 : 2;

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 3)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3 && dcn != 4)
    {
        LOG_ERROR("Invalid output channel number " << dcn);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    switch (data_type)
    {
    case kCV_8U:
    {
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);
        hsv_to_bgr_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, dcn, isFullRange);
        checkKernelErrors();
    }
    break;
    case kCV_32F:
    {
        cuda_op::Ptr2dVarShapeNHWC<float> src_ptr(inData);
        cuda_op::Ptr2dVarShapeNHWC<float> dst_ptr(outData);
        hsv_to_bgr_float_nhwc<float><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, bidx, dcn);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV420xp_to_BGR(const IImageBatchVarShapeDataStridedCuda &inData,
                                 const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                                 cudaStream_t stream)
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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 1)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (data_type != kCV_8U)
    {
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3 && dcn != 4)
    {
        LOG_ERROR("Invalid output channel number " << dcn);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height * 2 / 3, blockSize.y), batch_size);

    cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
    cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_420:
    {
        yuv420_to_gray_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, code);
        checkKernelErrors();
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
    {
        yuv420sp_to_bgr_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dcn, bidx, uidx, code);
        checkKernelErrors();
    }
    break;
    case NVCV_COLOR_YUV2BGR_YV12:
    case NVCV_COLOR_YUV2BGR_IYUV:
    case NVCV_COLOR_YUV2BGRA_YV12:
    case NVCV_COLOR_YUV2BGRA_IYUV:
    case NVCV_COLOR_YUV2RGB_YV12:
    case NVCV_COLOR_YUV2RGB_IYUV:
    case NVCV_COLOR_YUV2RGBA_YV12:
    case NVCV_COLOR_YUV2RGBA_IYUV:
    {
        yuv420p_to_bgr_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dcn, bidx, uidx, code);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

inline ErrorCode YUV422_to_BGR(const IImageBatchVarShapeDataStridedCuda &inData,
                               const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                               cudaStream_t stream)
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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 2)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (data_type != kCV_8U)
    {
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!outData.uniqueFormat())
    {
        LOG_ERROR("Images in the output batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int dcn = outData.uniqueFormat().numChannels();

    if (dcn != 3 && dcn != 4)
    {
        LOG_ERROR("Invalid output channel number " << dcn);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int max_width  = inData.maxSize().w;
    int max_height = inData.maxSize().h;
    int batch_size = inData.numImages();

    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
    cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);

    switch (code)
    {
    case NVCV_COLOR_YUV2GRAY_YUY2:
    case NVCV_COLOR_YUV2GRAY_UYVY:
    {
        yuv422_to_gray_char_nhwc<unsigned char><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, yidx);
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
        yuv422_to_bgr_char_nhwc<unsigned char>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dcn, bidx, yidx, uidx);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

template<typename T>
inline static void bgr_to_yuv420p_launcher(Ptr2dVarShapeNHWC<T> src_ptr, Ptr2dVarShapeNHWC<T> dst_ptr,
                                           DataShape inputShape, int bidx, int uidx, NVCVColorConversionCode code,
                                           cudaStream_t stream)
{
    // method 1
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(inputShape.W, blockSize.x), divUp(inputShape.H, blockSize.y), inputShape.N);
    bgr_to_yuv420p_char_nhwc<unsigned char>
        <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, inputShape.C, bidx, uidx, code);
    checkKernelErrors();

    // method 2 (TODO)
    // NPP
}

template<typename T>
inline static void bgr_to_yuv420sp_launcher(Ptr2dVarShapeNHWC<T> src_ptr, Ptr2dVarShapeNHWC<T> dst_ptr,
                                            DataShape inputShape, int bidx, int uidx, NVCVColorConversionCode code,
                                            cudaStream_t stream)
{
    // method 1
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(inputShape.W, blockSize.x), divUp(inputShape.H, blockSize.y), inputShape.N);
    bgr_to_yuv420sp_char_nhwc<unsigned char>
        <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, inputShape.C, bidx, uidx, code);
    checkKernelErrors();

    // method 2 (TODO)
    // NPP
}

inline ErrorCode BGR_to_YUV420xp(const IImageBatchVarShapeDataStridedCuda &inData,
                                 const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                                 cudaStream_t stream)
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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int      channels  = inData.uniqueFormat().numChannels();
    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (channels != 3 && channels != 4)
    {
        LOG_ERROR("Invalid input channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }
    if (data_type != kCV_8U)
    {
        LOG_ERROR("Unsupported DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    // BGR input
    cuda_op::Ptr2dVarShapeNHWC<unsigned char> src_ptr(inData);
    // YUV420xp output
    cuda_op::Ptr2dVarShapeNHWC<unsigned char> dst_ptr(outData);

    cuda_op::DataShape maxInputShape(inData.numImages(), channels, inData.maxSize().h, inData.maxSize().w);

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
    {
        bgr_to_yuv420sp_launcher(src_ptr, dst_ptr, maxInputShape, bidx, uidx, code, stream);
        checkKernelErrors();
    }
    break;
    case NVCV_COLOR_BGR2YUV_YV12:
    case NVCV_COLOR_BGR2YUV_IYUV:
    case NVCV_COLOR_BGRA2YUV_YV12:
    case NVCV_COLOR_BGRA2YUV_IYUV:
    case NVCV_COLOR_RGB2YUV_YV12:
    case NVCV_COLOR_RGB2YUV_IYUV:
    case NVCV_COLOR_RGBA2YUV_YV12:
    case NVCV_COLOR_RGBA2YUV_IYUV:
    {
        bgr_to_yuv420p_launcher(src_ptr, dst_ptr, maxInputShape, bidx, uidx, code, stream);
        checkKernelErrors();
    }
    break;
    default:
        LOG_ERROR("Unsupported conversion code " << code);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    return ErrorCode::SUCCESS;
}

size_t CvtColorVarShape::calBufferSize(int batch_size)
{
    return 0;
}

ErrorCode CvtColorVarShape::infer(const IImageBatchVarShapeDataStridedCuda &inData,
                                  const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                                  cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    typedef ErrorCode (*func_t)(const IImageBatchVarShapeDataStridedCuda &inData,
                                const IImageBatchVarShapeDataStridedCuda &outData, NVCVColorConversionCode code,
                                cudaStream_t stream);

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

    NVCV_ASSERT(func != 0);

    return func(inData, outData, code, stream);
}

} // namespace nvcv::legacy::cuda_op
