/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>

#include <cmath>   // For std::floor
#include <cstring> // For std::memcpy

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

// Coefficients to convert chromaticity (U and V) components to RGB .
// static constexpr double U2Blu =  2.03211;
// static constexpr double U2Grn = -0.39465;
// static constexpr double V2Grn = -0.58060;
// static constexpr double V2Red =  1.13983;
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

//-==================================================================================================================-//
// Set AlphaOnly to true to add/remove alpha channel to RGB/BGR image (without switching between RGB and BGR).
template<typename T, bool AlphaOnly>
static void convertRGBtoBGR(T *dst, const T *src, size_t numPixels, bool srcRGBA, bool dstRGBA)
{
    const uint incr = 3 + srcRGBA;

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
    const int incr = 3 + rgba;

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
        if (rgba) *dst++ = Alpha<T>;
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
    constexpr double range = (sizeof(T) > 1) ? 360.0 : (FullRange ? 256.0 : 180.0);
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

        double diff = static_cast<double>(V - Vmin);

        double S = static_cast<double>(V) > DBL_EPSILON ? diff / V : 0.0;
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
    5) h  = H' - I  // Fractional part of H'
    6) X  = C * (1 - fabs(fmod(H', 2.0) - 1.0))
          = C * (1 - fabs(H' - (I & ~1) - 1.0))
          = C * ((I & 1) ? 1 - h : h)
    5) m  = V - C
          = V - V * S
          = V * (1 - S)
    7) p  = X + m      // When I is even: (I & 1) == 0 (I = 0, 2, or 4)
          = C * h + V - C
          = V * S * h + V * (1 - S)
          = V * (S * h + 1 - S)
          = V * (1 - S + S * h)
          = V * (1 - S * (1 - h))
    8) q  = X + m      // When I is odd: (I & 1) == 1 (I = 1, 3, or 5)
          = C * (1 - h) + V - C
          = V * S * (1 - h) + V * (1 - S)
          = V * (S - S * h + 1 - S)
          = V * (1 - S * h)
    9) Cases: // Note: C + m = C + V - C = V
           I == 0: R = C + m = V
                   G = X + m = p  // Even case
                   B =     m

           I == 1: R = X + m = q  // Odd case
                   G = C + m = V
                   B =     m

           I == 2: R =     m
                   G = C + m = V
                   B = X + m = p  // Even case

           I == 3: R =     m
                   G = X + m = q  // Odd case
                   B = C + m = V

           I == 4: R = X + m = p  // Even case
                   G =     m
                   B = C + m = V

           I == 5: R = C + m = V
                   G =     m
                   B = X + m = q  // Odd case
*/
template<typename T, bool FullRange>
void convertHSVtoRGB(T *dst, const T *src, size_t numPixels, bool rgba, bool bgr)
{
    constexpr double range = (sizeof(T) > 1) ? 360.0 : (FullRange ? 256.0 : 180.0);
    constexpr double scale = 6.0 / range;
    constexpr double norm  = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr double round = std::is_floating_point_v<T> ? 0 : 0.5;

    constexpr uint mapR[6] = {0, 2, 1, 1, 3, 0};
    constexpr uint mapG[6] = {3, 0, 0, 2, 1, 1};
    constexpr uint mapB[6] = {1, 1, 3, 0, 0, 2};

    for (size_t i = 0; i < numPixels; i++)
    {
        double H = *src++ * scale; // 0 <= H <  6
        double S = *src++ / norm;  // 0 <= S <= 1
        double V = *src++ / norm;  // 0 <= V <= 1

        int idx = static_cast<int>(std::floor(H));

        H -= idx;

        // clang-format off
        idx %= 6;
        if (idx < 0) idx += 6;

        double val[] = {V,
                        V * (1 - S),
                        V * (1 - S * H),
                        V * (1 - S * (1 - H))};

        uint r = mapR[idx];
        uint g = mapG[idx];
        uint b = mapB[idx];

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
void convertRGBtoYUV_420(T *dst, const T *src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrPix   = 3 + rgba;
    const size_t incrSrc   = imgPixels * incrPix;
    const size_t incrDst   = imgPixels * 3 / 2;

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *y = dst;
        T *u = y + imgPixels;
        T *v = u + imgPixels / 4;

        const T *rgb = src;

        // clang-format off
        if (yvu) std::swap(u, v);
        // clang-format on

        for (uint h = 0; h < hght; h++)
        {
            for (uint w = 0; w < wdth; w++, rgb += incrPix)
            {
                T R = rgb[0];
                T G = rgb[1];
                T B = rgb[2];

                // Convert all RGB values to Y values and store them.
                // clang-format off
                if (bgr) std::swap(R, B);
                *y++ = cuda::SaturateCast<T>(R2Y_NV12 * R + G2Y_NV12 * G + B2Y_NV12 * B + Add2Y_NV12);
                // clang-format on

                // Convert only even pixels (in width and height) to U and V values and store them.
                if ((w & 1) == 0 && (h & 1) == 0)
                {
                    double U = R2U_NV12 * R + G2U_NV12 * G + B2U_NV12 * B + Add2U_NV12;
                    double V = R2V_NV12 * R + G2V_NV12 * G + B2V_NV12 * B + Add2V_NV12;

                    *u++ = cuda::SaturateCast<T>(U);
                    *v++ = cuda::SaturateCast<T>(V);
                }
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoYUV_420(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr,
                         bool yvu)
{
    // Ensure input data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertRGBtoYUV_420<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoYUV(T) \
    template void convertRGBtoYUV_420<T>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

MAKE_RGBtoYUV(uint8_t);
MAKE_RGBtoYUV(uint16_t);
MAKE_RGBtoYUV(int32_t);
MAKE_RGBtoYUV(float);
MAKE_RGBtoYUV(double);

#undef MAKE_RGBtoYUV

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertYUVtoRGB_420(T *dst, const T *src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;
    const size_t incrDst   = imgPixels * (3 + rgba);

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *y = src;

        for (uint h = 0; h < hght; h++)
        {
            // clang-format off
            // NOTE: when computing subsampled row index, h needs to be integer divided by 4 before multiplying by width.
            const T *u = src + imgPixels + (h / 4) * wdth + ((h / 2) & 1) * (wdth / 2);
            const T *v = u   + imgPixels / 4;

            if (yvu) std::swap(u, v);
            // clang-format on

            for (uint w = 0; w < wdth; w++)
            {
                double Y = *y++;
                double U = *u;
                double V = *v;

                // Convert all YUV (ITU Rec.601) values to RGB values and store them.
                Y -= Add2Y_NV12;
                U -= Add2U_NV12;
                V -= Add2V_NV12;

                // clang-format off
                if (Y < 0.0) Y = 0.0;
                T R = cuda::SaturateCast<T>(Y2R_NV12 * Y + U2R_NV12 * U + V2R_NV12 * V);
                T G = cuda::SaturateCast<T>(Y2G_NV12 * Y + U2G_NV12 * U + V2G_NV12 * V);
                T B = cuda::SaturateCast<T>(Y2B_NV12 * Y + U2B_NV12 * U + V2B_NV12 * V);
                if (bgr) std::swap(R, B);
                *rgb++ = R;
                *rgb++ = G;
                *rgb++ = B;
                if (rgba) *rgb++ = Alpha<T>;
                // clang-format on

                u += (w & 1);
                v += (w & 1);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertYUVtoRGB_420(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr,
                         bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertYUVtoRGB_420<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_NV12toRGB(T) \
    template void convertYUVtoRGB_420<T>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

MAKE_NV12toRGB(uint8_t);
MAKE_NV12toRGB(uint16_t);
MAKE_NV12toRGB(int32_t);
MAKE_NV12toRGB(float);
MAKE_NV12toRGB(double);

#undef MAKE_NV12toRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertYUVtoGray_420(T *dst, const T *src, uint wdth, uint hght, uint numImgs)
{
    // Ensure both width and height are multiples of 2.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += imgPixels)
    {
        std::memcpy(dst, src, imgPixels * sizeof(T)); // Copy Y plane of each image to destination tensor.
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertYUVtoGray_420(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth);

    // YUV 420 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertYUVtoGray_420<T>(dst.data(), src.data(), wdth, hght, numImgs);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_YUVtoGray(T) template void convertYUVtoGray_420<T>(vector<T> &, const vector<T> &, uint, uint, uint)

MAKE_YUVtoGray(uint8_t);
MAKE_YUVtoGray(uint16_t);
MAKE_YUVtoGray(int32_t);
MAKE_YUVtoGray(float);
MAKE_YUVtoGray(double);

#undef MAKE_YUVtoGray

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertRGBtoNV12(T *dst, const T *src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrPix   = 3 + rgba;
    const size_t incrSrc   = imgPixels * incrPix;
    const size_t incrDst   = imgPixels * 3 / 2;

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *y  = dst;
        T *uv = dst + imgPixels;

        const T *rgb = src;

        for (uint h = 0; h < hght; h++)
        {
            for (uint w = 0; w < wdth; w++, rgb += incrPix)
            {
                T R = rgb[0];
                T G = rgb[1];
                T B = rgb[2];

                // Convert all RGB values to Y values and store them.
                // clang-format off
                if (bgr) std::swap(R, B);
                *y++ = cuda::SaturateCast<T>(R2Y_NV12 * R + G2Y_NV12 * G + B2Y_NV12 * B + Add2Y_NV12);
                // clang-format on

                // Convert only even pixels (in width and height) to U and V values and store them.
                if ((w & 1) == 0 && (h & 1) == 0)
                {
                    double U = R2U_NV12 * R + G2U_NV12 * G + B2U_NV12 * B + Add2U_NV12;
                    double V = R2V_NV12 * R + G2V_NV12 * G + B2V_NV12 * B + Add2V_NV12;

                    // clang-format off
                    if (yvu) std::swap(U, V);
                    // clang-format on
                    *uv++ = cuda::SaturateCast<T>(U);
                    *uv++ = cuda::SaturateCast<T>(V);
                }
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertRGBtoNV12(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr,
                      bool yvu)
{
    // Ensure input data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));

    // YUV NV12 needs 3 elements for each two RGB pixels.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertRGBtoNV12<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RGBtoNV12(T) \
    template void convertRGBtoNV12<T>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

MAKE_RGBtoNV12(uint8_t);
MAKE_RGBtoNV12(uint16_t);
MAKE_RGBtoNV12(int32_t);
MAKE_RGBtoNV12(float);
MAKE_RGBtoNV12(double);

#undef MAKE_RGBtoNV12

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void convertNV12toRGB(T *dst, const T *src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure both width and height are multiples of 2 since we're processing 2x2 blocks.
    assert(wdth % 2 == 0 && hght % 2 == 0);

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 3 / 2;
    const size_t incrDst   = imgPixels * (3 + rgba);

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *y = src;

        for (uint h = 0; h < hght; h++)
        {
            // NOTE: when computing uv row index, h needs to be integer divided by 2 before multiplying by width.
            const T *uv = src + imgPixels + (h >> 1) * wdth;

            for (uint w = 0; w < wdth; w++)
            {
                double Y = *y++;
                double U = uv[0];
                double V = uv[1];

                // clang-format off
                if (yvu) std::swap(U, V);

                // Convert all YUV (ITU Rec.601) values to RGB values and store them.
                Y -= Add2Y_NV12;
                U -= Add2U_NV12;
                V -= Add2V_NV12;
                if (Y < 0.0) Y = 0.0;

                T R = cuda::SaturateCast<T>(Y2R_NV12 * Y + U2R_NV12 * U + V2R_NV12 * V);
                T G = cuda::SaturateCast<T>(Y2G_NV12 * Y + U2G_NV12 * U + V2G_NV12 * V);
                T B = cuda::SaturateCast<T>(Y2B_NV12 * Y + U2B_NV12 * U + V2B_NV12 * V);

                if (bgr) std::swap(R, B);
                *rgb++ = R;
                *rgb++ = G;
                *rgb++ = B;
                if (rgba) *rgb++ = Alpha<T>;

                if (w & 1) uv += 2;
                // clang-format on
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T>
void convertNV12toRGB(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr,
                      bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA with or w/o alpha) values for the given width and height and batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));

    // YUV NV12 needs 3 elements for each two RGB pixels.
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 3 / 2);

    convertNV12toRGB<T>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_NV12toRGB(T) \
    template void convertNV12toRGB<T>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

MAKE_NV12toRGB(uint8_t);
MAKE_NV12toRGB(uint16_t);
MAKE_NV12toRGB(int32_t);
MAKE_NV12toRGB(float);
MAKE_NV12toRGB(double);

#undef MAKE_NV12toRGB

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T, bool LumaFirst>
void convertYUVtoRGB_422(T *dst, const T *src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr, bool yvu)
{
    // Ensure width is a multiple of 2.
    assert(wdth % 2 == 0);

    constexpr uint idx0 = (LumaFirst ? 0 : 1); // First  luma value index.
    constexpr uint idx1 = idx0 + 2;            // Second luma value index.
    constexpr uint idxU = (LumaFirst ? 1 : 0); // U chroma value index.
    constexpr uint idxV = idxU + 2;            // V chroma value index.

    const size_t imgPixels = (size_t)hght * (size_t)wdth;
    const size_t incrSrc   = imgPixels * 2;
    const size_t incrDst   = imgPixels * (3 + rgba);

    for (uint n = 0; n < numImgs; n++, src += incrSrc, dst += incrDst)
    {
        T *rgb = dst;

        const T *img = src;

        for (uint h = 0; h < hght; h++)
        {
            for (uint w = 0; w < wdth; w += 2, img += 4)
            {
                T R, G, B;

                // clang-format off
                double U  = img[idxU],
                       V  = img[idxV],
                       Y0 = img[idx0],
                       Y1 = img[idx1];

                if (yvu) std::swap(U, V);

                // Convert all YUV (ITU Rec.601) values to RGB values and store them.
                Y0 -= Add2Y_NV12;
                Y1 -= Add2Y_NV12;
                U  -= Add2U_NV12;
                V  -= Add2V_NV12;

                if (Y0 < 0.0) Y0 = 0.0;
                if (Y1 < 0.0) Y1 = 0.0;
                // clang-format on

                double Y_0  = Y2R_NV12 * Y0; // NOTE: Y2R_NV12 == Y2G_NV12 == Y2B_NV12.
                double Y_1  = Y2R_NV12 * Y1;
                double UV_r = U2R_NV12 * U + V2R_NV12 * V;
                double UV_g = U2G_NV12 * U + V2G_NV12 * V;
                double UV_b = U2B_NV12 * U + V2B_NV12 * V;

                R = cuda::SaturateCast<T>(Y_0 + UV_r);
                G = cuda::SaturateCast<T>(Y_0 + UV_g);
                B = cuda::SaturateCast<T>(Y_0 + UV_b);

                // clang-format off
                if (bgr) std::swap(R, B);
                *rgb++ = R;  *rgb++ = G;  *rgb++ = B;
                if (rgba) *rgb++ = Alpha<T>;
                // clang-format on

                R = cuda::SaturateCast<T>(Y_1 + UV_r);
                G = cuda::SaturateCast<T>(Y_1 + UV_g);
                B = cuda::SaturateCast<T>(Y_1 + UV_b);

                // clang-format off
                if (bgr) std::swap(R, B);
                *rgb++ = R;  *rgb++ = G;  *rgb++ = B;
                if (rgba) *rgb++ = Alpha<T>;
                // clang-format on
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
template<typename T, bool LumaFirst>
void convertYUVtoRGB_422(vector<T> &dst, const vector<T> &src, uint wdth, uint hght, uint numImgs, bool rgba, bool bgr,
                         bool yvu)
{
    // Ensure output data has sets of 3 or 4 (RGB/BGA w/ or w/o alpha) values for the given width, height, & batch size.
    assert(dst.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * (size_t)(3 + rgba));
    assert(src.size() == (size_t)numImgs * (size_t)hght * (size_t)wdth * 2); // 4 values for each two RGB pixels.

    convertYUVtoRGB_422<T, LumaFirst>(dst.data(), src.data(), wdth, hght, numImgs, rgba, bgr, yvu);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toRGB(T) \
    template void convertYUVtoRGB_422<T, false>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

MAKE_422toRGB(uint8_t);
MAKE_422toRGB(uint16_t);
MAKE_422toRGB(int32_t);
MAKE_422toRGB(float);
MAKE_422toRGB(double);

#undef MAKE_422toRGB
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_422toRGB(T) \
    template void convertYUVtoRGB_422<T, true>(vector<T> &, const vector<T> &, uint, uint, uint, bool, bool, bool)

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
    src += (1 - LumaFirst); // Increment to first Y value if luma not first.

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
