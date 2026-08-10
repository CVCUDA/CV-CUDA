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

#include "TestUtils.hpp"

#include "Definitions.hpp"

#include <cvcuda/cuda_tools/TypeTraits.hpp>

#include <array>

namespace cuda = nvcv::cuda;

using std::vector;

template<typename T>
struct RgbStoreParams
{
    T    alpha;
    bool rgba;
    bool bga;
};

template<typename T>
void StoreRgb(T *&dst, T red, T grn, T blu, const RgbStoreParams<T> &params)
{
    if (params.bga)
    {
        std::swap(red, blu);
    }

    *dst++ = red;
    *dst++ = grn;
    *dst++ = blu;
    if (params.rgba)
    {
        *dst++ = params.alpha;
    }
}

//-==================================================================================================================-//
// Generate an random image image vector.
template<typename T>
void generateRandVec( // NOSONAR: std::span is C++20.
    T *dst, size_t size, RandEng &eng)
{
    RandInt<T> rand(0, cuda::TypeTraits<T>::max);

    // clang-format off
    for (size_t i = 0; i < size; i++) dst[i] = rand(eng);
    // clang-format on
}

template<>
void generateRandVec( // NOSONAR: std::span is C++20.
    float *dst, size_t size, RandEng &eng)
{
    RandFlt<float> rand(0.0f, 1.0f);

    // clang-format off
    for (size_t i = 0; i < size; i++) dst[i] = rand(eng);
    // clang-format on
}

template<>
void generateRandVec( // NOSONAR: std::span is C++20.
    double *dst, size_t size, RandEng &eng)
{
    std::uniform_real_distribution rand(0.0, 1.0);

    // clang-format off
    for (size_t i = 0; i < size; i++) dst[i] = rand(eng);
    // clang-format on
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RAND_VEC(T) template void generateRandVec<T>(T *, size_t, RandEng &)

MAKE_RAND_VEC(uint8_t);
MAKE_RAND_VEC(uint16_t);
MAKE_RAND_VEC(int32_t);
MAKE_RAND_VEC(float);
MAKE_RAND_VEC(double);

#undef MAKE_RAND_VEC

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
template<typename T>
void generateRandTestRGB(T *dst, size_t size, RandEng &eng, bool rgba, bool bga)
{
    constexpr T                max = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr std::array<T, 3> val = {0, max / 2, max};

    const size_t minSize = 3 * 3 * 3 * (rgba ? 4 : 3);

    generateRandVec(dst, size, eng);

    if (size <= minSize)
    {
        return;
    }

    RgbStoreParams<T> params{max, rgba, bga};
    T                *out = dst;

    for (unsigned int r = 0; r < 3; r++)
    {
        const T red = val[r];

        for (unsigned int g = 0; g < 3; g++)
        {
            const T grn = val[g];

            for (unsigned int b = 0; b < 3; b++)
            {
                StoreRgb(out, red, grn, val[b], params);
            }
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_RAND_RGB_TEST(T) template void generateRandTestRGB<T>(T *, size_t, RandEng &, bool, bool)

MAKE_RAND_RGB_TEST(uint8_t);
MAKE_RAND_RGB_TEST(uint16_t);
MAKE_RAND_RGB_TEST(int32_t);
MAKE_RAND_RGB_TEST(float);
MAKE_RAND_RGB_TEST(double);

#undef MAKE_RAND_RGB_TEST

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
// Generate an image potentially containing all 16,777,216 RGB8 colors, assuming the image / tensor dimensions are
// sufficiently large (http://www.brucelindbloom.com/downloads/RGB16Million.png); otherwise the image is cropped to the
// provided sizes. Generates consecutive 256 x 256 image blocks where, within each block, red varies from 0 to 255
// horizontally and green varies from 0 to 255 vertically. Blue increments from 0 to 255 in consecutive blocks; partial
// blocks (i.e., those that may be cropped to specified dimensions) still increment the blue value. All values are then
// rescaled to fit the data type--e.g., floating point types are rescaled to be between 0 and 1.
// To get all 16,777,216 8-bit RGB colors, generate a single image (i.e., tensor batch = 1) of size 1 x 4096 x 4096,
// or generate a tensor of 4 x 2048 x 2048, 16 x 1024 x 1024, 64 x 512 x 512, or 256 x 256 x 256.
// Note: generates interleaved (non-planar) data.
template<typename T>
void generateAllRGB(T *dst, unsigned int wdth, unsigned int hght, unsigned int num, bool rgba, bool bga)
{
    constexpr T      max   = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr double round = std::is_floating_point_v<T> ? 0 : 0.5;
    constexpr double scale = (double)max / 255.0;

    const size_t incrH = wdth * (rgba ? 4 : 3);
    const size_t incrN = hght * incrH;

    unsigned int addB = 0;

    RgbStoreParams<T> params{max, rgba, bga};

    for (unsigned int i = 0; i < num; i++)
    {
        T *img = dst + i * incrN;

        for (unsigned int y = 0; y < hght; y++)
        {
            T *row = img + y * incrH;

            auto grn = static_cast<uint8_t>(y & 255);

            for (unsigned int x = 0; x < wdth; x++)
            {
                auto red = static_cast<uint8_t>(x & 255);
                auto blu = static_cast<uint8_t>(((x >> 8) + addB) & 255);

                StoreRgb(row, static_cast<T>(red * scale + round), static_cast<T>(grn * scale + round),
                         static_cast<T>(blu * scale + round), params);
            }
            // clang-format off
            if (grn == 255) addB += ((wdth + 255) >> 8);
            // clang-format on
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#define MAKE_ALL_RGB_TEST(T) \
    template void generateAllRGB<T>(T *, unsigned int, unsigned int, unsigned int, bool rgba, bool bga)

MAKE_ALL_RGB_TEST(uint8_t);
MAKE_ALL_RGB_TEST(uint16_t);
MAKE_ALL_RGB_TEST(int32_t);
MAKE_ALL_RGB_TEST(float);
MAKE_ALL_RGB_TEST(double);

#undef MAKE_ALL_RGB_TEST

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
// Generate a random HSV (Hue-Saturation-Value) image where the Hue range can be specified and the Saturation and Value
// ranges are scaled according to the data type. Since Hue is circular, it can be useful to generate Hue values outside
// the standard range (e.g., min to test if a function that processes HSV images properly accounts for wrap-around Hue values.
// Note: generates interleaved (non-planar) data.
template<typename T, bool FullRange>
void generateRandHSV(T *dst, size_t size, RandEng &eng, double minHueMult, double maxHueMult)
{
    ASSERT_EQ(size % 3, 0);

    constexpr T            max   = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr unsigned int range = HsvHueRange<T, FullRange>();
    constexpr double       scale = (double)range / 360.0;
    constexpr double       round = std::is_floating_point_v<T> ? 0 : 0.5;

    // clang-format off
    if (minHueMult > 1.0) minHueMult = 0.0;
    if (maxHueMult < 0.0) maxHueMult = 1.0;
    // clang-format on

    double minHue = minHueMult * range;
    double maxHue = maxHueMult * range;

    std::uniform_real_distribution randHue(minHue, maxHue);
    std::uniform_real_distribution randSV(0.0, 1.0);

    for (size_t i = 0; i < size; i += 3)
    {
        // clang-format off
        *dst++ = static_cast<T>(randHue(eng) * scale + round);
        *dst++ = static_cast<T>(randSV (eng) * max   + round);
        *dst++ = static_cast<T>(randSV (eng) * max   + round);
        // clang-format on
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Restricted range hue (FullRange = false): values between [0-180). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_RAND_HSV_TEST(T) template void generateRandHSV<T, false>(T *, size_t, RandEng &, double, double)

MAKE_RAND_HSV_TEST(uint8_t);
MAKE_RAND_HSV_TEST(uint16_t);
MAKE_RAND_HSV_TEST(int32_t);
MAKE_RAND_HSV_TEST(float);
MAKE_RAND_HSV_TEST(double);

#undef MAKE_RAND_HSV_TEST

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Full range hue (FullRange = false): values between [0-256). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_RAND_HSV_TEST(T) template void generateRandHSV<T, true>(T *, size_t, RandEng &, double, double)

MAKE_RAND_HSV_TEST(uint8_t);
MAKE_RAND_HSV_TEST(uint16_t);
MAKE_RAND_HSV_TEST(int32_t);
MAKE_RAND_HSV_TEST(float);
MAKE_RAND_HSV_TEST(double);

#undef MAKE_RAND_HSV_TEST

//--------------------------------------------------------------------------------------------------------------------//

//-==================================================================================================================-//
// Generate an HSV (Hue-Saturation-Value) image containing blocks of size H_range x 256 where H_range is either:
//   * 360 (for size(T) > 1),
//   * 255 (for size(T) == 1 and FullRange == true), or
//   * 180 (for size(T) == 1 and FullRange == false).
// Within each block, H (Hue) varies from 0 to H_range-1 horizontally and S (Saturation) varies from 0 to 255 vertically.
// V (Value) increments from 0 to 255 in consecutive blocks. The values for S and V are normalized (i.e., rescaled)
// according to the data type.
// To get all available HSV values, generate a single image (i.e., tensor batch = 1) of size 1 x (16*H_range) x 4096,
// or a tensor of 4 x (8*H_range) x 2048, 16 x (4*H_range) x 1024, 64 x (2*H_range) x 512, or 256 x H_range x 256.
// Note: generates interleaved (non-planar) data.
template<typename T, bool FullRange>
void generateAllHSV(T *dst, unsigned int wdth, unsigned int hght, unsigned int num)
{
    constexpr T            max   = std::is_floating_point_v<T> ? 1 : cuda::TypeTraits<T>::max;
    constexpr unsigned int range = HsvHueRange<T, FullRange>();
    constexpr double       scale = (double)range / 360.0;
    constexpr double       norm  = (double)max / 255.0;
    constexpr double       round = std::is_floating_point_v<T> ? 0 : 0.5;

    constexpr unsigned int stepV
        = 1; // Step size for V (value) from one block to the next. 17 is prime, so 256 % (17 * m) will
             // always be unique for 0 <= m < 256.
    const size_t incrH = wdth * 3;
    const size_t incrN = hght * incrH;

    unsigned int addV = 0;

    for (unsigned int i = 0; i < num; i++)
    {
        T *img = dst + i * incrN;

        for (unsigned int y = 0; y < hght; y++)
        {
            T *row = img + y * incrH;

            auto S = static_cast<uint8_t>(y & 255);

            // clang-format off
            for (unsigned int x = 0; x < wdth; x++)
            {
                auto H = static_cast<uint8_t>(x % range);
                auto V = static_cast<uint8_t>((((unsigned int)(x / range) + addV) * stepV) & 255);

                *row++ = static_cast<T>(H * scale + round);
                *row++ = static_cast<T>(S * norm  + round);
                *row++ = static_cast<T>(V * norm  + round);
            }
            if (S == 255) addV += ((wdth + range - 1) / range);
            // clang-format on
        }
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Restricted range hue (FullRange = false): values between [0-180). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_ALL_HSV_TEST(T) template void generateAllHSV<T, false>(T *, unsigned int, unsigned int, unsigned int)

MAKE_ALL_HSV_TEST(uint8_t);
MAKE_ALL_HSV_TEST(uint16_t);
MAKE_ALL_HSV_TEST(int32_t);
MAKE_ALL_HSV_TEST(float);
MAKE_ALL_HSV_TEST(double);

#undef MAKE_ALL_HSV_TEST

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Full range hue (FullRange = false): values between [0-256). Applies only to uint8_t, but still need to
// instantiate all the types.
#define MAKE_ALL_HSV_TEST(T) template void generateAllHSV<T, true>(T *, unsigned int, unsigned int, unsigned int)

MAKE_ALL_HSV_TEST(uint8_t);
MAKE_ALL_HSV_TEST(uint16_t);
MAKE_ALL_HSV_TEST(int32_t);
MAKE_ALL_HSV_TEST(float);
MAKE_ALL_HSV_TEST(double);

#undef MAKE_ALL_HSV_TEST

//--------------------------------------------------------------------------------------------------------------------//
