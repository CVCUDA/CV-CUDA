/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvtColorUtils.hpp"

#include "TestUtils.hpp"

#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>

#include <array>
#include <cmath>   // For std::floor
#include <cstring> // For std::memcpy
#include <utility>

namespace cuda = nvcv::cuda;

template<typename T>
using Vector3 = cuda::math::Vector<T, 3>;

template<typename T>
using Matrix3x3 = cuda::math::Matrix<T, 3, 3>;

using Vec3f = Vector3<float>;
using Vec3d = Vector3<double>;

using Mat3f = Matrix3x3<float>;
using Mat3d = Matrix3x3<double>;

using std::vector;

// Accurate coefficients for converting RGB to ITU Rec.601 luma.
// Found at
//   http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html
// and
//   https://www.imagemagick.org/include/api/pixel.php.
// NOTE: These coefficients are more accurate than the standard [0.299, 0.587, 0.144] values used elsewhere and may
// results in slightly different floating point or large integer (e.g., uint16 or uint32) pixel values.
// static constexpr double Red2Y = 0.298839;  // Y = Red2Y * R
// static constexpr double Grn2Y = 0.586811;  //   + Grn2Y * G
// static constexpr double Blu2Y = 0.114350;  //   + Blu2Y * B
static constexpr double Red2Y = 0.299; // Y = Red2Y * R
static constexpr double Grn2Y = 0.587; //   + Grn2Y * G
static constexpr double Blu2Y = 0.114; //   + Blu2Y * B

// Coefficients to convert non-linear RGB to PAL (analog color TV standard) chromaticity (U and V) components.
// NOTE: Both PAL and NTSC use the ITU Rec.601 RGB coefficients to compute Y.
// static constexpr double Blu2U_PAL = 0.492111; // U = Blu2U_PAL * (B - Y) + 0.5
// static constexpr double Red2V_PAL = 0.877283; // V = Red2V_PAL * (R - Y) + 0.5
static constexpr double Blu2U_PAL = 0.492; // U = Blu2U_PAL * (B - Y) + 0.5
static constexpr double Red2V_PAL = 0.877; // V = Red2V_PAL * (R - Y) + 0.5

// Coefficients to convert non-linear RGB to ITU Rec.601 chromaticity (Cb and Cr) components.
static constexpr double Blu2Cb_601 = 0.56455710; // 1.0 / 1.7713   Cb/U
static constexpr double Red2Cr_601 = 0.71310298; // 1.0 / 1.402322 Cr/V

// clang-format off

// Coefficients to convert chromaticity (U and V) components to RGB.
static constexpr double U2Blu =  2.032;
static constexpr double U2Grn = -0.395;
static constexpr double V2Grn = -0.581;
static constexpr double V2Red =  1.140;

// Coefficients to convert RGB to ITU Rec.601 YCbCr.
static constexpr double R2Y_NV12 =  0.255785;
static constexpr double G2Y_NV12 =  0.502160;
static constexpr double B2Y_NV12 =  0.097523;

static constexpr double R2U_NV12 = -0.147644;
static constexpr double G2U_NV12 = -0.289856;
static constexpr double B2U_NV12 =  0.4375;

static constexpr double R2V_NV12 =  0.4375;
static constexpr double G2V_NV12 = -0.366352;
static constexpr double B2V_NV12 = -0.071148;

// Coefficients to convert RGB to ITU Rec.601 YCbCr.
static constexpr double Y2R_NV12 =  1.16895;
static constexpr double U2R_NV12 =  0.0;
static constexpr double V2R_NV12 =  1.60229;

static constexpr double Y2G_NV12 =  1.16895;
static constexpr double U2G_NV12 = -0.3933;
static constexpr double V2G_NV12 = -0.81616;

static constexpr double Y2B_NV12 =  1.16895;
static constexpr double U2B_NV12 =  2.02514;
static constexpr double V2B_NV12 =  0.0;

// Coefficients to add or subtract from YCbCr (abbreviated YUV)components to convert between RGB and ITU Rec.601 YCbCr.
static constexpr double Add2Y_NV12 =  16.0;
static constexpr double Add2U_NV12 = 128.0;
static constexpr double Add2V_NV12 = 128.0;

// clang-format on

template<typename T, typename BT = cuda::BaseType<T>>
constexpr BT Alpha = std::is_floating_point_v<BT> ? 1 : cuda::TypeTraits<BT>::max;

template<typename T>
struct RgbPixel
{
    T r;
    T g;
    T b;
};

struct YuvPair
{
    double y0;
    double y1;
    double u;
    double v;
};

template<typename T>
RgbPixel<T> ReadRgbPixel(const T *rgb, bool bgr)
{
    RgbPixel<T> pixel{rgb[0], rgb[1], rgb[2]};
    if (bgr)
    {
        std::swap(pixel.r, pixel.b);
    }
    return pixel;
}

template<typename T>
void StoreRgbPixel(T *&rgb, T r, T g, T b, bool rgba, bool bgr)
{
    if (bgr)
    {
        std::swap(r, b);
    }

    *rgb++ = r;
    *rgb++ = g;
    *rgb++ = b;
    if (rgba)
    {
        *rgb++ = Alpha<T>;
    }
}

inline double ClampedLuma(double y)
{
    y -= Add2Y_NV12;
    return y < 0.0 ? 0.0 : y;
}

template<typename T>
void StoreNv12RgbPixel(T *&rgb, double y, double u, double v, bool rgba, bool bgr)
{
    T r = cuda::SaturateCast<T>(Y2R_NV12 * y + U2R_NV12 * u + V2R_NV12 * v);
    T g = cuda::SaturateCast<T>(Y2G_NV12 * y + U2G_NV12 * u + V2G_NV12 * v);
    T b = cuda::SaturateCast<T>(Y2B_NV12 * y + U2B_NV12 * u + V2B_NV12 * v);

    StoreRgbPixel(rgb, r, g, b, rgba, bgr);
}

template<typename T>
void StoreYuv420ChromaIfNeeded(T *&u, T *&v, const RgbPixel<T> &rgb, unsigned int w, unsigned int h)
{
    if ((w & 1) != 0 || (h & 1) != 0)
    {
        return;
    }

    double chromaU = R2U_NV12 * rgb.r + G2U_NV12 * rgb.g + B2U_NV12 * rgb.b + Add2U_NV12;
    double chromaV = R2V_NV12 * rgb.r + G2V_NV12 * rgb.g + B2V_NV12 * rgb.b + Add2V_NV12;

    *u++ = cuda::SaturateCast<T>(chromaU);
    *v++ = cuda::SaturateCast<T>(chromaV);
}

template<typename T>
void StoreNv12ChromaIfNeeded(T *&uv, const RgbPixel<T> &rgb, unsigned int w, unsigned int h, bool yvu)
{
    if ((w & 1) != 0 || (h & 1) != 0)
    {
        return;
    }

    double u = R2U_NV12 * rgb.r + G2U_NV12 * rgb.g + B2U_NV12 * rgb.b + Add2U_NV12;
    double v = R2V_NV12 * rgb.r + G2V_NV12 * rgb.g + B2V_NV12 * rgb.b + Add2V_NV12;
    if (yvu)
    {
        std::swap(u, v);
    }

    *uv++ = cuda::SaturateCast<T>(u);
    *uv++ = cuda::SaturateCast<T>(v);
}

template<typename T>
std::pair<double, double> ReadNv12Chroma(const T *uv, bool yvu)
{
    double u = uv[0];
    double v = uv[1];
    if (yvu)
    {
        std::swap(u, v);
    }
    return {u - Add2U_NV12, v - Add2V_NV12};
}

template<typename T>
std::pair<const T *, const T *> Yuv420ChromaRows(const T *src, size_t imgPixels, unsigned int wdth, unsigned int h,
                                                 bool yvu)
{
    // NOTE: when computing subsampled row index, h needs to be integer divided by 4 before multiplying by width.
    const T *u = src + imgPixels + (h / 4) * wdth + ((h / 2) & 1) * (wdth / 2);
    const T *v = u + imgPixels / 4;
    if (yvu)
    {
        std::swap(u, v);
    }
    return {u, v};
}

template<typename T, bool LumaFirst>
YuvPair ReadYuv422Pair(const T *img, bool yvu)
{
    constexpr unsigned int idx0 = (LumaFirst ? 0 : 1); // First  luma value index.
    constexpr unsigned int idx1 = idx0 + 2;            // Second luma value index.
    constexpr unsigned int idxU = (LumaFirst ? 1 : 0); // U chroma value index.
    constexpr unsigned int idxV = idxU + 2;            // V chroma value index.

    YuvPair pair{ClampedLuma(img[idx0]), ClampedLuma(img[idx1]), img[idxU] - Add2U_NV12, img[idxV] - Add2V_NV12};
    if (yvu)
    {
        std::swap(pair.u, pair.v);
    }
    return pair;
}

template<typename T>
void StoreYuv422RgbPair(T *&rgb, const YuvPair &pair, bool rgba, bool bgr)
{
    StoreNv12RgbPixel(rgb, pair.y0, pair.u, pair.v, rgba, bgr);
    StoreNv12RgbPixel(rgb, pair.y1, pair.u, pair.v, rgba, bgr);
}

//-==================================================================================================================-//
// Set AlphaOnly to true to add/remove alpha channel to RGB/BGR image (without switching between RGB and BGR).
template<typename T, bool AlphaOnly>
static void convertRGBtoBGR(T *dst, const T *src, size_t numPixels, bool srcRGBA, bool dstRGBA)
{
    const unsigned int incr = srcRGBA ? 4 : 3;

    for (size_t i = 0; i < numPixels; i++, src += incr)
    {
        // clang-format off
        if constexpr (AlphaOnly) { *dst++ = src[0];  *dst++ = src[1];  *dst++ = src[2]; }
        else                     { *dst++ = src[2];  *dst++ = src[1];  *dst++ = src[0]; }
        if (dstRGBA) *dst++ = srcRGBA ? src[3] : Alpha<T>;
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoBGR(vector<T> &dst, const vector<T> &src, size_t numPixels, bool srcRGBA, bool dstRGBA)
{
    convertRGBtoBGR<T, false>(dst.data(), src.data(), numPixels, srcRGBA, dstRGBA);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoBGR(T) template void convertRGBtoBGR<T>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_RGBtoBGR(uint8_t);
MAKE_RGBtoBGR(uint16_t);
MAKE_RGBtoBGR(int32_t);
MAKE_RGBtoBGR(float);
MAKE_RGBtoBGR(double);

#undef MAKE_RGBtoBGR

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void changeAlpha(vector<T> &dst, const vector<T> &src, size_t numPixels, bool srcRGBA, bool dstRGBA)
{
    convertRGBtoBGR<T, true>(dst.data(), src.data(), numPixels, srcRGBA, dstRGBA);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_CHANGE_ALPHA(T) template void changeAlpha<T>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_CHANGE_ALPHA(uint8_t);
MAKE_CHANGE_ALPHA(uint16_t);
MAKE_CHANGE_ALPHA(int32_t);
MAKE_CHANGE_ALPHA(float);
MAKE_CHANGE_ALPHA(double);

#undef MAKE_CHANGE_ALPHA

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertRGBtoGray(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    const int incr = rgba ? 4 : 3;

    for (size_t i = 0; i < numPixels; i++, dst++, src += incr)
    {
        // clang-format off
        if (bgr) *dst = static_cast<T>(Blu2Y * src[0] + Grn2Y * src[1] + Red2Y * src[2]);
        else     *dst = static_cast<T>(Red2Y * src[0] + Grn2Y * src[1] + Blu2Y * src[2]);
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoGray(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba, bool bgr)
{
    convertRGBtoGray<T>(dst.data(), src.data(), numPixels, rgba, bgr);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoGray(T) template void convertRGBtoGray<T>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_RGBtoGray(uint8_t);
MAKE_RGBtoGray(uint16_t);
MAKE_RGBtoGray(int32_t);
MAKE_RGBtoGray(float);
MAKE_RGBtoGray(double);

#undef MAKE_RGBtoGray

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertGrayToRGB(T *dst, const T *src, size_t numPixels, bool rgba)
{
    for (size_t i = 0; i < numPixels; i++)
    {
        T val = *src++;

        // clang-format off
        *dst++ = val;  *dst++ = val;  *dst++ = val;
        if (rgba) *dst++ = val; // align with gpu code gray_to_bgr_nhwc
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertGrayToRGB(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba)
{
    convertGrayToRGB<T>(dst.data(), src.data(), numPixels, rgba);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_GrayToRGB(T) template void convertGrayToRGB<T>(vector<T> &, const vector<T> &, size_t, bool)

MAKE_GrayToRGB(uint8_t);
MAKE_GrayToRGB(uint16_t);
MAKE_GrayToRGB(int32_t);
MAKE_GrayToRGB(float);
MAKE_GrayToRGB(double);

#undef MAKE_GrayToRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T, bool FullRange>
void convertRGBtoHSV(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    // Set the hue range (e.g., 0-360 for float types) and scale factor (to convert the final value to output hue value).
    constexpr double range = HsvHueRange<T, FullRange>();
    constexpr double scale = range / 360.0;
    constexpr double norm  = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr double round = std::is_floating_point_v<T> ? 0 : 0.5;

    for (size_t i = 0; i < numPixels; i++)
    {
        double R = static_cast<double>(*src++) / norm;
        double G = static_cast<double>(*src++) / norm;
        double B = static_cast<double>(*src++) / norm;

        // clang-format off
        if (bgr) std::swap(R, B);
        if (rgba) src++;
        // clang-format on

        double Vmin = std::min(R, std::min(G, B));
        double V    = std::max(R, std::max(G, B));

        auto diff = V - Vmin;

        double S = V > DBL_EPSILON ? diff / V : 0.0;
        double H = 0.0;

        if (diff > DBL_EPSILON)
        {
            // clang-format off
            diff = 60.0 / diff;
            if      (V == R) H = (G - B) * diff;
            else if (V == G) H = (B - R) * diff + 120.0;
            else             H = (R - G) * diff + 240.0;
            // clang-format on
        }
        H *= scale;
        S *= norm;
        V *= norm;

        // Make sure hue falls within the proper range: the value 'range' (e.g., 360) should not appear since it's equivalent to 0.
        H += round;
        // clang-format off
        if      (H >= range) H -= range;  // For the case when T is uint8_t and FullRange is false, H can be > 180.
        else if (H <  0.0)   H += range;
        // clang-format on
        H -= round;

        *dst++ = static_cast<T>(H + round);
        *dst++ = static_cast<T>(S + round);
        *dst++ = static_cast<T>(V + round);
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool FullRange>
void convertRGBtoHSV(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba, bool bgr)
{
    convertRGBtoHSV<T, FullRange>(dst.data(), src.data(), numPixels, rgba, bgr);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Restricted range hue (FullRange = false): values between [0-180). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_RGBtoHSV(T) template void convertRGBtoHSV<T, false>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_RGBtoHSV(uint8_t);
MAKE_RGBtoHSV(uint16_t);
MAKE_RGBtoHSV(int32_t);
MAKE_RGBtoHSV(float);
MAKE_RGBtoHSV(double);

#undef MAKE_RGBtoHSV

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Full range hue (FullRange = false): values between [0-256). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_RGBtoHSV(T) template void convertRGBtoHSV<T, true>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_RGBtoHSV(uint8_t);
MAKE_RGBtoHSV(uint16_t);
MAKE_RGBtoHSV(int32_t);
MAKE_RGBtoHSV(float);
MAKE_RGBtoHSV(double);

#undef MAKE_RGBtoHSV

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
/* To convert HSV to RGB:
    1) Ensure that (H,S,V) range is (360.0, 1.0, 1.0)
    2) H' = H / 60
    3) C  = V * S
    4) I  = (int)H
    5) h  = H' - I  (fractional part of H')
    6) X  = C * (1 - fabs(fmod(H', 2.0) - 1.0))
          = C * (1 - fabs(H' - (I & ~1) - 1.0))
          = C * ((I & 1) ? 1 - h : h)
    5) m  = V - C
          = V - V * S
          = V * (1 - S)
    7) p  = X + m      (when I is even: (I & 1) == 0 (I = 0, 2, or 4))
          = C * h + V - C
          = V * S * h + V * (1 - S)
          = V * (S * h + 1 - S)
          = V * (1 - S + S * h)
          = V * (1 - S * (1 - h))
    8) q  = X + m      (when I is odd: (I & 1) == 1 (I = 1, 3, or 5))
          = C * (1 - h) + V - C
          = V * S * (1 - h) + V * (1 - S)
          = V * (S - S * h + 1 - S)
          = V * (1 - S * h)
    9) Cases: C + m = C + V - C = V
           I == 0: R = C + m = V
                   G = X + m = p  (even case)
                   B =     m

           I == 1: R = X + m = q  (odd case)
                   G = C + m = V
                   B =     m

           I == 2: R =     m
                   G = C + m = V
                   B = X + m = p  (even case)

           I == 3: R =     m
                   G = X + m = q  (odd case)
                   B = C + m = V

           I == 4: R = X + m = p  (even case)
                   G =     m
                   B = C + m = V

           I == 5: R = C + m = V
                   G =     m
                   B = X + m = q  (odd case)
*/
template<typename T, bool FullRange>
void convertHSVtoRGB(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    constexpr double range = HsvHueRange<T, FullRange>();
    constexpr double scale = 6.0 / range;
    constexpr double norm  = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr double round = std::is_floating_point_v<T> ? 0 : 0.5;

    constexpr std::array<unsigned int, 6> mapR = {0, 2, 1, 1, 3, 0};
    constexpr std::array<unsigned int, 6> mapG = {3, 0, 0, 2, 1, 1};
    constexpr std::array<unsigned int, 6> mapB = {1, 1, 3, 0, 0, 2};

    for (size_t i = 0; i < numPixels; i++)
    {
        double H = *src++ * scale; // 0 <= H <  6
        double S = *src++ / norm;  // 0 <= S <= 1
        double V = *src++ / norm;  // 0 <= V <= 1

        auto idx = static_cast<int>(std::floor(H));

        H -= idx;

        // clang-format off
        idx %= 6;
        if (idx < 0) idx += 6;

        std::array<double, 4> val = {V, V * (1 - S), V * (1 - S * H), V * (1 - S * (1 - H))};

        unsigned int r = mapR[idx];
        unsigned int g = mapG[idx];
        unsigned int b = mapB[idx];

        if (bgr) std::swap(r, b);
        *dst++ = static_cast<T>(val[r] * norm + round);
        *dst++ = static_cast<T>(val[g] * norm + round);
        *dst++ = static_cast<T>(val[b] * norm + round);
        if (rgba) *dst++ = Alpha<T>;
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool FullRange>
void convertHSVtoRGB(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba, bool bgr)
{
    convertHSVtoRGB<T, FullRange>(dst.data(), src.data(), numPixels, rgba, bgr);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Restricted range hue (FullRange = false): values between [0-180). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_HSVtoRGB(T) template void convertHSVtoRGB<T, false>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_HSVtoRGB(uint8_t);
MAKE_HSVtoRGB(uint16_t);
MAKE_HSVtoRGB(int32_t);
MAKE_HSVtoRGB(float);
MAKE_HSVtoRGB(double);

#undef MAKE_HSVtoRGB

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Full range hue (FullRange = false): values between [0-256). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_HSVtoRGB(T) template void convertHSVtoRGB<T, true>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_HSVtoRGB(uint8_t);
MAKE_HSVtoRGB(uint16_t);
MAKE_HSVtoRGB(int32_t);
MAKE_HSVtoRGB(float);
MAKE_HSVtoRGB(double);

#undef MAKE_HSVtoRGB

//-==================================================================================================================-//
template<typename T>
void convertRGBtoYUV_PAL(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    constexpr T max   = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr T delta = max / 2 + (std::is_floating_point_v<T> ? 0 : 1);

    for (size_t i = 0; i < numPixels; i++)
    {
        T red = *src++;
        T grn = *src++;
        T blu = *src++;

        // clang-format off
        if (bgr) std::swap(red, blu);
        if (rgba) src++;
        // clang-format on

        double Y = Red2Y * red + Grn2Y * grn + Blu2Y * blu;

        *dst++ = cuda::SaturateCast<T>(Y);
        *dst++ = cuda::SaturateCast<T>(Blu2U_PAL * (blu - Y) + delta);
        *dst++ = cuda::SaturateCast<T>(Red2V_PAL * (red - Y) + delta);
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoYUV_PAL(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba, bool bgr)
{
    convertRGBtoYUV_PAL<T>(dst.data(), src.data(), numPixels, rgba, bgr);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoYUV(T) template void convertRGBtoYUV_PAL<T>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_RGBtoYUV(uint8_t);
MAKE_RGBtoYUV(uint16_t);
MAKE_RGBtoYUV(int32_t);
MAKE_RGBtoYUV(float);
MAKE_RGBtoYUV(double);

#undef MAKE_RGBtoYUV

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertYUVtoRGB_PAL(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    constexpr T max   = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr T delta = max / 2 + (std::is_floating_point_v<T> ? 0 : 1);

    for (size_t i = 0; i < numPixels; i++)
    {
        double Y = *src++;
        double U = *src++;
        double V = *src++;

        U -= delta;
        V -= delta;

        double red = Y + V * V2Red;
        double grn = Y + U * U2Grn + V * V2Grn;
        double blu = Y + U * U2Blu;

        // clang-format off
        if (bgr) std::swap(red, blu);
        *dst++ = cuda::SaturateCast<T>(red);
        *dst++ = cuda::SaturateCast<T>(grn);
        *dst++ = cuda::SaturateCast<T>(blu);
        if (rgba) *dst++ = Alpha<T>;
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertYUVtoRGB_PAL(vector<T> &dst, const vector<T> &src, size_t numPixels, bool rgba, bool bgr)
{
    convertYUVtoRGB_PAL<T>(dst.data(), src.data(), numPixels, rgba, bgr);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_YUVtoRGB(T) template void convertYUVtoRGB_PAL<T>(vector<T> &, const vector<T> &, size_t, bool, bool)

MAKE_YUVtoRGB(uint8_t);
MAKE_YUVtoRGB(uint16_t);
MAKE_YUVtoRGB(int32_t);
MAKE_YUVtoRGB(float);
MAKE_YUVtoRGB(double);

#undef MAKE_YUVtoRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertRGBtoYUV_420(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs, bool rgba,
                         bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrPix   = rgba ? 4 : 3;
    const size_t incrSrc   = imgPixels * incrPix;
    const size_t incrDst   = imgPixels * 3 / 2;

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *y = dst;
        T *u = y + imgPixels;
        T *v = u + imgPixels / 4;

        const T *rgb = src;

        // clang-format off
        if (yvu) std::swap(u, v);
        // clang-format on

        for (unsigned int h = 0; h < hght; h++)
        {
            for (unsigned int w = 0; w < wdth; w++, rgb += incrPix)
            {
                RgbPixel<T> pixel = ReadRgbPixel(rgb, bgr);
                *y++ = cuda::SaturateCast<T>(R2Y_NV12 * pixel.r + G2Y_NV12 * pixel.g + B2Y_NV12 * pixel.b + Add2Y_NV12);
                StoreYuv420ChromaIfNeeded(u, v, pixel, w, h);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoYUV_420(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght,
                         unsigned int numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure input data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(rgba ? 4 : 3));

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertRGBtoYUV_420<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoYUV(T)                                                                                           \
    template void convertRGBtoYUV_420<T>(vector<T> &, const vector<T> &, unsigned int, unsigned int, unsigned int, \
                                         bool, bool, bool)

MAKE_RGBtoYUV(uint8_t);
MAKE_RGBtoYUV(uint16_t);
MAKE_RGBtoYUV(int32_t);
MAKE_RGBtoYUV(float);
MAKE_RGBtoYUV(double);

#undef MAKE_RGBtoYUV

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertYUVtoRGB_420(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs, bool rgba,
                         bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;
    const size_t incrDst   = imgPixels * (rgba ? 4 : 3);

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *y = src;

        for (unsigned int h = 0; h < hght; h++)
        {
            auto [u, v] = Yuv420ChromaRows(src, imgPixels, wdth, h, yvu);

            for (unsigned int w = 0; w < wdth; w++)
            {
                double Y = ClampedLuma(*y++);
                double U = *u - Add2U_NV12;
                double V = *v - Add2V_NV12;

                StoreNv12RgbPixel(rgb, Y, U, V, rgba, bgr);

                u += (w & 1);
                v += (w & 1);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertYUVtoRGB_420(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght,
                         unsigned int numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(rgba ? 4 : 3));

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertYUVtoRGB_420<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_NV12toRGB(T)                                                                                          \
    template void convertYUVtoRGB_420<T>(vector<T> &, const vector<T> &, unsigned int, unsigned int, unsigned int, \
                                         bool, bool, bool)

MAKE_NV12toRGB(uint8_t);
MAKE_NV12toRGB(uint16_t);
MAKE_NV12toRGB(int32_t);
MAKE_NV12toRGB(float);
MAKE_NV12toRGB(double);

#undef MAKE_NV12toRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertYUVtoGray_420(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs)
{
    // Ensure both width and height are multiples of 2.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += imgPixels)
    {
        std::memcpy(dst, src, imgPixels * sizeof(T)); // Copy Y plane of each image to destination tensor.
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertYUVtoGray_420(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght,
                          unsigned int numImgs)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth);

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertYUVtoGray_420<T>(dst.data(), src.data(), wdth, hght, numImgs);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_YUVtoGray(T) \
    template void convertYUVtoGray_420<T>(vector<T> &, const vector<T> &, unsigned int, unsigned int, unsigned int)

MAKE_YUVtoGray(uint8_t);
MAKE_YUVtoGray(uint16_t);
MAKE_YUVtoGray(int32_t);
MAKE_YUVtoGray(float);
MAKE_YUVtoGray(double);

#undef MAKE_YUVtoGray

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertRGBtoNV12(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs, bool rgba,
                      bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrPix   = rgba ? 4 : 3;
    const size_t incrSrc   = imgPixels * incrPix;
    const size_t incrDst   = imgPixels * 3 / 2;

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *y  = dst;
        T *uv = dst + imgPixels;

        const T *rgb = src;

        for (unsigned int h = 0; h < hght; h++)
        {
            for (unsigned int w = 0; w < wdth; w++, rgb += incrPix)
            {
                RgbPixel<T> pixel = ReadRgbPixel(rgb, bgr);
                *y++ = cuda::SaturateCast<T>(R2Y_NV12 * pixel.r + G2Y_NV12 * pixel.g + B2Y_NV12 * pixel.b + Add2Y_NV12);
                StoreNv12ChromaIfNeeded(uv, pixel, w, h, yvu);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoNV12(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght, unsigned int numImgs,
                      bool rgba, bool bgr, bool yvu)
{
    // Ensure input data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(rgba ? 4 : 3));

    // YUV NV12 needs 3 elements for each two RGB pixels.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertRGBtoNV12<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoNV12(T)                                                                                             \
    template void convertRGBtoNV12<T>(vector<T> &, const vector<T> &, unsigned int, unsigned int, unsigned int, bool, \
                                      bool, bool)

MAKE_RGBtoNV12(uint8_t);
MAKE_RGBtoNV12(uint16_t);
MAKE_RGBtoNV12(int32_t);
MAKE_RGBtoNV12(float);
MAKE_RGBtoNV12(double);

#undef MAKE_RGBtoNV12

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertNV12toRGB(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs, bool rgba,
                      bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;
    const size_t incrDst   = imgPixels * (rgba ? 4 : 3);

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *y = src;

        for (unsigned int h = 0; h < hght; h++)
        {
            // NOTE: when computing uv row index, h needs to be integer divided by 2 before multiplying by width.
            const T *uv = src + imgPixels + (h >> 1) * wdth;

            for (unsigned int w = 0; w < wdth; w++)
            {
                double Y    = ClampedLuma(*y++);
                auto [U, V] = ReadNv12Chroma(uv, yvu);

                StoreNv12RgbPixel(rgb, Y, U, V, rgba, bgr);

                uv += (w & 1) * 2;
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertNV12toRGB(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght, unsigned int numImgs,
                      bool rgba, bool bgr, bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(rgba ? 4 : 3));

    // YUV NV12 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertNV12toRGB<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_NV12toRGB(T)                                                                                             \
    template void convertNV12toRGB<T>(vector<T> &, const vector<T> &, unsigned int, unsigned int, unsigned int, bool, \
                                      bool, bool)

MAKE_NV12toRGB(uint8_t);
MAKE_NV12toRGB(uint16_t);
MAKE_NV12toRGB(int32_t);
MAKE_NV12toRGB(float);
MAKE_NV12toRGB(double);

#undef MAKE_NV12toRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T, bool LumaFirst>
void convertYUVtoRGB_422(T *dst, const T *src, unsigned int wdth, unsigned int hght, unsigned int numImgs, bool rgba,
                         bool bgr, bool yvu)
{
    // Ensure width is a multiple of 2.
    assert(wdth % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 2;
    const size_t incrDst   = imgPixels * (rgba ? 4 : 3);

    for (unsigned int n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *img = src;

        for (unsigned int h = 0; h < hght; h++)
        {
            for (unsigned int w = 0; w < wdth; w += 2, img += 4)
            {
                StoreYuv422RgbPair(rgb, ReadYuv422Pair<T, LumaFirst>(img, yvu), rgba, bgr);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool LumaFirst>
void convertYUVtoRGB_422(vector<T> &dst, const vector<T> &src, unsigned int wdth, unsigned int hght,
                         unsigned int numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA w/ or w/o alpha) values for the given width, height, & batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(rgba ? 4 : 3));
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 2); // 4 values for each two RGB pixels.

    convertYUVtoRGB_422<T, LumaFirst>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toRGB(T)                                                                                    \
    template void convertYUVtoRGB_422<T, false>(vector<T> &, const vector<T> &, unsigned int, unsigned int, \
                                                unsigned int, bool, bool, bool)

MAKE_422toRGB(uint8_t);
MAKE_422toRGB(uint16_t);
MAKE_422toRGB(int32_t);
MAKE_422toRGB(float);
MAKE_422toRGB(double);

#undef MAKE_422toRGB
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toRGB(T)                                                                                   \
    template void convertYUVtoRGB_422<T, true>(vector<T> &, const vector<T> &, unsigned int, unsigned int, \
                                               unsigned int, bool, bool, bool)

MAKE_422toRGB(uint8_t);
MAKE_422toRGB(uint16_t);
MAKE_422toRGB(int32_t);
MAKE_422toRGB(float);
MAKE_422toRGB(double);

#undef MAKE_422toRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T, bool LumaFirst>
void convertYUVtoGray_422(T *dst, const T *src, size_t numPixels)
{
    src += (LumaFirst ? 0 : 1); // Increment to first Y value if luma not first.

    for (size_t i = 0; i < numPixels; i++, src += 2) *dst++ = *src;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool LumaFirst>
void convertYUVtoGray_422(vector<T> &dst, const vector<T> &src, size_t numPixels)
{
    assert(dst.size() == numPixels);
    assert(src.size() == numPixels * 2); // YUV 422 needs 4 values for each two RGB pixels.

    convertYUVtoGray_422<T, LumaFirst>(dst.data(), src.data(), numPixels);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toGray(T) template void convertYUVtoGray_422<T, false>(vector<T> &, const vector<T> &, size_t)

MAKE_422toGray(uint8_t);
MAKE_422toGray(uint16_t);
MAKE_422toGray(int32_t);
MAKE_422toGray(float);
MAKE_422toGray(double);

#undef MAKE_422toGray
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toGray(T) template void convertYUVtoGray_422<T, true>(vector<T> &, const vector<T> &, size_t)

MAKE_422toGray(uint8_t);
MAKE_422toGray(uint16_t);
MAKE_422toGray(int32_t);
MAKE_422toGray(float);
MAKE_422toGray(double);

#undef MAKE_422toGray
//--------------------------------------------------------------------------------------------------------------------//
