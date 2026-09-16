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

// Formula, constants, and channel-range provenance:
//   OpenCV, modules/imgproc/src/color_lab.cpp and imgproc color-conversion documentation (Apache-2.0).
//   https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_lab.cpp
// OpenCV further attributes its RGB-to-Lab conversion to RGB2Lab.m by Mark Ruzon, translated from C code by
// Yossi Rubner (23 September 1997). This file is a CUDA-oriented reimplementation of the documented equations.

#ifndef CVCUDA_PRIV_LAB_COLOR_CONVERSION_CUH
#define CVCUDA_PRIV_LAB_COLOR_CONVERSION_CUH

#include <cuda_fp16.h>
#include <cvcuda/cuda_tools/SaturateCast.hpp>

#include <cmath>
#include <type_traits>

namespace cvcuda::priv::lab {

constexpr float kEpsilon = 216.f / 24389.f;
constexpr float kKappa   = 24389.f / 27.f;
constexpr float kDelta   = 6.f / 29.f;

__device__ __forceinline__ float Clamp01(float value)
{
    return fminf(fmaxf(value, 0.f), 1.f);
}

template<bool InputIsU8, bool SRGB>
__device__ __forceinline__ float InputToLinear(float value)
{
    if constexpr (!InputIsU8)
    {
        value = Clamp01(value);
    }
    if constexpr (SRGB)
    {
        return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
    }
    else
    {
        return value;
    }
}

template<bool FastU8, bool SRGB>
__device__ __forceinline__ float LinearToOutput(float value)
{
    value = Clamp01(value);
    if constexpr (!SRGB)
    {
        return value;
    }
    if (value <= 0.0031308f)
    {
        return 12.92f * value;
    }
    if constexpr (FastU8)
    {
        return 1.055f * __powf(value, 1.f / 2.4f) - 0.055f;
    }
    else
    {
        return 1.055f * powf(value, 1.f / 2.4f) - 0.055f;
    }
}

__device__ __forceinline__ float XYZToLab(float value)
{
    return value > kEpsilon ? cbrtf(value) : (kKappa * value + 16.f) / 116.f;
}

__device__ __forceinline__ float XYZToLabFast(float value)
{
    return value > kEpsilon ? __powf(value, 1.f / 3.f) : (kKappa * value + 16.f) / 116.f;
}

__device__ __forceinline__ float LabToXYZ(float value)
{
    return value > kDelta ? value * value * value : 3.f * kDelta * kDelta * (value - 4.f / 29.f);
}

template<bool InputIsU8, bool FastCbrt, bool PreNormalizedMatrix, bool SRGB>
__device__ __forceinline__ void RGBToLabFloat(float red, float green, float blue, float &lightness, float &a, float &b)
{
    red   = InputToLinear<InputIsU8, SRGB>(red);
    green = InputToLinear<InputIsU8, SRGB>(green);
    blue  = InputToLinear<InputIsU8, SRGB>(blue);

    float x;
    if constexpr (PreNormalizedMatrix)
    {
        x = (0.412453f / 0.950456f) * red + (0.357580f / 0.950456f) * green + (0.180423f / 0.950456f) * blue;
    }
    else
    {
        x = (0.412453f * red + 0.357580f * green + 0.180423f * blue) / 0.950456f;
    }

    const float y = 0.212671f * red + 0.715160f * green + 0.072169f * blue;

    float z;
    if constexpr (PreNormalizedMatrix)
    {
        z = (0.019334f / 1.088754f) * red + (0.119193f / 1.088754f) * green + (0.950227f / 1.088754f) * blue;
    }
    else
    {
        z = (0.019334f * red + 0.119193f * green + 0.950227f * blue) / 1.088754f;
    }

    const float fx = FastCbrt ? XYZToLabFast(x) : XYZToLab(x);
    const float fy = FastCbrt ? XYZToLabFast(y) : XYZToLab(y);
    const float fz = FastCbrt ? XYZToLabFast(z) : XYZToLab(z);

    lightness = 116.f * fy - 16.f;
    a         = 500.f * (fx - fy);
    b         = 200.f * (fy - fz);
}

template<bool FastTransfer, bool PreNormalizedMatrix, bool SRGB>
__device__ __forceinline__ void LabToRGBFloat(float lightness, float a, float b, float &red, float &green, float &blue)
{
    const float fy = (lightness + 16.f) / 116.f;
    const float fx = fy + a / 500.f;
    const float fz = fy - b / 200.f;

    if constexpr (PreNormalizedMatrix)
    {
        const float x = LabToXYZ(fx);
        const float y = LabToXYZ(fy);
        const float z = LabToXYZ(fz);

        red   = LinearToOutput<FastTransfer, SRGB>((3.240479f * 0.950456f) * x - 1.537150f * y
                                                 - (0.498535f * 1.088754f) * z);
        green = LinearToOutput<FastTransfer, SRGB>((-0.969256f * 0.950456f) * x + 1.875991f * y
                                                   + (0.041556f * 1.088754f) * z);
        blue  = LinearToOutput<FastTransfer, SRGB>((0.055648f * 0.950456f) * x - 0.204043f * y
                                                  + (1.057311f * 1.088754f) * z);
    }
    else
    {
        const float x = 0.950456f * LabToXYZ(fx);
        const float y = LabToXYZ(fy);
        const float z = 1.088754f * LabToXYZ(fz);

        red   = LinearToOutput<FastTransfer, SRGB>(3.240479f * x - 1.537150f * y - 0.498535f * z);
        green = LinearToOutput<FastTransfer, SRGB>(-0.969256f * x + 1.875991f * y + 0.041556f * z);
        blue  = LinearToOutput<FastTransfer, SRGB>(0.055648f * x - 0.204043f * y + 1.057311f * z);
    }
}

template<typename T>
__device__ __forceinline__ float ToFloat(T value)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return __half2float(value);
    }
    else
    {
        return static_cast<float>(value);
    }
}

template<bool SRGB, bool FastF32, typename T>
__device__ __forceinline__ void RGBToLab(T red, T green, T blue, T &lightness, T &a, T &b)
{
    constexpr bool  kIsU8                = std::is_same_v<T, unsigned char>;
    constexpr bool  kIsF16               = std::is_same_v<T, __half>;
    constexpr bool  kPreNormalizedMatrix = kIsU8 || kIsF16 || SRGB || FastF32;
    constexpr float scale                = kIsU8 ? 1.f / 255.f : 1.f;

    float outL, outA, outB;
    RGBToLabFloat<kIsU8, kIsU8 || kIsF16, kPreNormalizedMatrix, SRGB>(ToFloat(red) * scale, ToFloat(green) * scale,
                                                                      ToFloat(blue) * scale, outL, outA, outB);

    if constexpr (kIsU8)
    {
        lightness = nvcv::cuda::SaturateCast<T>(outL * 255.f / 100.f);
        a         = nvcv::cuda::SaturateCast<T>(outA + 128.f);
        b         = nvcv::cuda::SaturateCast<T>(outB + 128.f);
    }
    else
    {
        lightness = nvcv::cuda::SaturateCast<T>(outL);
        a         = nvcv::cuda::SaturateCast<T>(outA);
        b         = nvcv::cuda::SaturateCast<T>(outB);
    }
}

template<bool SRGB, bool FastF16, typename T>
__device__ __forceinline__ void LabToRGB(T lightness, T a, T b, T &red, T &green, T &blue)
{
    constexpr bool kIsU8  = std::is_same_v<T, unsigned char>;
    constexpr bool kIsF16 = std::is_same_v<T, __half>;
    constexpr bool kFast  = SRGB && (kIsU8 || (kIsF16 && FastF16));

    float inL = ToFloat(lightness);
    float inA = ToFloat(a);
    float inB = ToFloat(b);
    if constexpr (kIsU8)
    {
        inL = inL * 100.f / 255.f;
        inA -= 128.f;
        inB -= 128.f;
    }

    float outR, outG, outB;
    LabToRGBFloat<kFast, kFast, SRGB>(inL, inA, inB, outR, outG, outB);

    constexpr float scale = kIsU8 ? 255.f : 1.f;
    red                   = nvcv::cuda::SaturateCast<T>(outR * scale);
    green                 = nvcv::cuda::SaturateCast<T>(outG * scale);
    blue                  = nvcv::cuda::SaturateCast<T>(outB * scale);
}

} // namespace cvcuda::priv::lab

#endif // CVCUDA_PRIV_LAB_COLOR_CONVERSION_CUH
