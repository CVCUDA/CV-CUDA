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

#ifndef NVCV_TEST_COMMON_HALF_TEST_UTILS_HPP
#define NVCV_TEST_COMMON_HALF_TEST_UTILS_HPP

// F16 tolerance policy — the single source of truth for CV-CUDA operator tests.
//
// F16 operators use native half arithmetic and are validated against an FP32-computed CPU
// reference. Bit-exact EXPECT_EQ is not applicable to that comparison because the reference is
// intentionally computed at higher precision than the kernel; the bound is kUlps half-ULPs at
// the reference's magnitude, where kUlps reflects the kernel's native-half rounding-step count.
// Each test states its own kUlps with a one-line justification:
//   - 0    -> use EXPECT_EQ on the 16-bit patterns instead (pure data movement, no arithmetic)
//   - 1    -> a single mul-add chain (e.g. Normalize, ConvertTo, BrightnessContrast)
//   - 2-4  -> n-tap interpolation or small filters (e.g. Resize, Warp, Gaussian)
//   - <=8  -> transcendental chains (e.g. GammaContrast, hue/saturation rotations)
// Tests referencing this policy satisfy the TST-8 / COV-BITEXACT adjacent-rationale rule.

#include <cuda_fp16.h>
#include <cuda_runtime.h> // for float4, etc.
#include <cvcuda/Types.h> // for NVCVInterpolationType, etc.
#include <gtest/gtest.h>
#include <nvcv/ImageFormat.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace nvcv::test {

// Round-trip a float through __half so the operator under test and the FP32 gold consume
// identical half-representable inputs; otherwise the comparison measures input quantization
// noise instead of kernel error.
inline float QuantizeToHalf(float v)
{
    return __half2float(__float2half(v));
}

inline void QuantizeToHalf(std::vector<float> &v)
{
    for (float &x : v)
    {
        x = QuantizeToHalf(x);
    }
}

// Component-wise overload for float4 border values: kernels convert the float border value to
// half (StaticCast<BaseType<T>>) before use, so an FP32 gold must consume the same
// half-quantized border.
inline float4 QuantizeToHalf(const float4 &v)
{
    return float4{QuantizeToHalf(v.x), QuantizeToHalf(v.y), QuantizeToHalf(v.z), QuantizeToHalf(v.w)};
}

// One ULP of FP16 at the magnitude of g (10 mantissa bits), flushing to the fixed subnormal
// spacing 2^-24 below the smallest positive normal half 2^-14.
inline float HalfUlp(float g)
{
    const float a = std::fabs(g);
    if (a < 0x1p-14f)
    {
        return 0x1p-24f;
    }
    return std::exp2f(std::floor(std::log2f(a)) - 10.f);
}

// Expect every element of test to be within kUlps half-ULPs of the FP32 reference gold.
inline void ExpectNearHalfUlps(const std::vector<float> &gold, const std::vector<float> &test, float kUlps,
                               float absToleranceFloor = 0.f)
{
    ASSERT_EQ(gold.size(), test.size());
    for (size_t i = 0; i < gold.size(); ++i)
    {
        EXPECT_NEAR(test[i], gold[i], std::max(kUlps * HalfUlp(gold[i]), absToleranceFloor))
            << "F16 output differs at index " << i;
    }
}

// Overload taking the raw F16 output as produced by the operator.
inline void ExpectNearHalfUlps(const std::vector<float> &gold, const std::vector<__half> &test, float kUlps,
                               float absToleranceFloor = 0.f)
{
    ASSERT_EQ(gold.size(), test.size());
    for (size_t i = 0; i < gold.size(); ++i)
    {
        EXPECT_NEAR(__half2float(test[i]), gold[i], std::max(kUlps * HalfUlp(gold[i]), absToleranceFloor))
            << "F16 output differs at index " << i;
    }
}

// Widen an F16 buffer (as raw bytes) to floats, e.g. to feed an FP32 CPU reference with the
// operator's half-quantized input.
inline std::vector<float> HalfBytesToFloat(const std::vector<uint8_t> &buf)
{
    if (buf.size() % sizeof(__half) != 0)
    {
        throw std::invalid_argument("F16 buffer size must be divisible by sizeof(__half)");
    }

    std::vector<float> out(buf.size() / sizeof(__half));
    for (size_t i = 0; i < out.size(); ++i)
    {
        __half value;
        std::memcpy(&value, buf.data() + i * sizeof(value), sizeof(value));
        out[i] = __half2float(value);
    }
    return out;
}

// Narrow a float buffer to F16 raw bytes, e.g. to build the F16 tensor input from float test data.
inline std::vector<uint8_t> FloatToHalfBytes(const std::vector<float> &v)
{
    std::vector<uint8_t> out(v.size() * sizeof(__half));
    for (size_t i = 0; i < v.size(); ++i)
    {
        const __half value = __float2half(v[i]);
        std::memcpy(out.data() + i * sizeof(value), &value, sizeof(value));
    }
    return out;
}

// True if the image format carries half-based pixels.
inline bool IsF16Format(nvcv::ImageFormat fmt)
{
    return fmt.planeDataType(0).channelType(0) == nvcv::TYPE_F16;
}

// F16 inputs cannot be raw random bytes: arbitrary bit patterns include NaN/Inf, which an
// FP32-gold comparison cannot bound. Generate finite values and quantize them to half so the
// operator and the gold consume identical half-representable inputs.
inline void FillRandomHalfBytes(
    std::vector<uint8_t>       &buf,
    std::default_random_engine &randEng, // NOSONAR: deterministic test data, not security-sensitive.
    float lo = 0.f, float hi = 255.f)
{
    if (buf.size() % sizeof(__half) != 0)
    {
        throw std::invalid_argument("F16 buffer size must be divisible by sizeof(__half)");
    }

    std::uniform_real_distribution<float> rand(lo, hi);
    std::vector<float>                    vals(buf.size() / sizeof(__half));
    std::ranges::generate(vals, [&rand, &randEng]() { return rand(randEng); });
    buf = FloatToHalfBytes(vals);
}

// Deterministic finite HWC test pattern quantized to half (random bytes would contain NaN/Inf,
// which an FP32-gold comparison cannot bound), so the FP32 gold consumes exactly the values
// the F16 kernel reads.
inline std::vector<float> MakeFiniteHalfHwc(int width, int height, int channels, int seed)
{
    std::vector<float> values(static_cast<size_t>(width) * height * channels);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = QuantizeToHalf(static_cast<float>((static_cast<int>(i) * 17 + seed) % 113) / 11.f);
    }
    return values;
}

// Deterministic half-representable test pattern lo + step * ((i * 7 + sample * 13) % mod),
// different per sample so batched samples do not alias; quantized to half so kernel and
// reference consume identical inputs. Callers pick lo/step/mod so the values suit their
// tolerance analysis (e.g. positive and bounded away from zero for relative ULP bounds).
inline std::vector<float> MakeHalfQuantizedPattern(int numElements, int sample, float lo, float step, int mod)
{
    std::vector<float> v(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        v[i] = lo + step * static_cast<float>((i * 7 + sample * 13) % mod);
    }
    QuantizeToHalf(v);
    return v;
}

// Interpolating-op F16 comparison policy: NEAREST is a pure gather of half-representable values
// (no arithmetic), so the FP32 gold narrows back to half exactly and the 16-bit patterns must
// match (kUlps = 0 policy). LINEAR/CUBIC accumulate the taps in FP32 (InterpolationWrap) and
// round to half once on store; 4 half-ULPs bounds that rounding
// plus device/host FP32 ordering differences in the coordinate and tap math.
inline void ExpectF16InterpOutput(const std::vector<float> &goldFloat, const std::vector<uint8_t> &testHalf,
                                  NVCVInterpolationType interpolation)
{
    if (interpolation == NVCV_INTERP_NEAREST)
    {
        EXPECT_EQ(FloatToHalfBytes(goldFloat), testHalf);
    }
    else
    {
        ExpectNearHalfUlps(goldFloat, HalfBytesToFloat(testHalf), 4.f);
    }
}

// Overload comparing two raw F16 buffers, for golds already produced in half precision.
inline void ExpectF16InterpOutput(const std::vector<uint8_t> &goldHalf, const std::vector<uint8_t> &testHalf,
                                  NVCVInterpolationType interpolation)
{
    if (interpolation == NVCV_INTERP_NEAREST)
    {
        EXPECT_EQ(goldHalf, testHalf);
    }
    else
    {
        ExpectNearHalfUlps(HalfBytesToFloat(goldHalf), HalfBytesToFloat(testHalf), 4.f);
    }
}

} // namespace nvcv::test

#endif // NVCV_TEST_COMMON_HALF_TEST_UTILS_HPP
