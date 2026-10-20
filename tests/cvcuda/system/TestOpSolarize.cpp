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

#include "ElementwiseOpHarness.hpp"
#include "PlanarParityUtils.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/OpSolarize.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold: out = (in >= threshold) ? (bound - in) : in, per element. Mirrors the
// documented oracle (torchvision solarize) and is computed independently of the kernel so the
// bit-exact EXPECT_EQ is a real regression check.
template<typename T>
T SolarizeScalarGold(T v, double threshold)
{
    const T bound = ew::Bound<T>();
    return (static_cast<double>(v) >= threshold) ? static_cast<T>(bound - v) : v;
}

// Mid-range threshold per element type, so the random data exercises both the invert branch and
// the pass-through branch (u8 -> 127.5, u16 -> 32767.5, f32 -> 0.5).
template<typename DT>
double MidThreshold()
{
    return std::is_floating_point_v<DT> ? 0.5 : static_cast<double>(ew::Bound<DT>()) / 2.0;
}

// Per-dtype gold + invoke factories: both pin the same mid-range threshold for a given element type
// so the kernel and the reference stay in lockstep across u8 / u16 / f32.
const auto goldFor = []<typename DT>(DT)
{
    const double thr = MidThreshold<DT>();
    return [thr](DT v)
    {
        return SolarizeScalarGold<DT>(v, thr);
    };
};

const auto invokeFor = []<typename DT>(DT)
{
    const double thr = MidThreshold<DT>();
    return [thr](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::Solarize op;
        op(s, in, out, thr);
    };
};

// A fixed-threshold invoker for the var-shape / parity / negative cases (which use one dtype or
// don't reach the kernel).
auto invokeThr(double thr)
{
    return [thr](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::Solarize op;
        op(s, in, out, thr);
    };
}

} // namespace

// Tensor correctness over the declared dtype × channel matrix -----------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpSolarize, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8     }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8   }, // u8  / 3ch
    {       42,     53,     4,   nvcv::FMT_RGBA8  }, // u8  / 4ch
    {       80,     40,     2,   nvcv::FMT_U16    }, // u16 / 1ch
    {       17,     19,     1,   nvcv::FMT_F32    }, // f32 / 1ch
    {      101,     33,     2,   nvcv::FMT_RGBf32 }, // f32 / 3ch
    {       64,     48,     3,   nvcv::FMT_RGBAf32}, // f32 / 4ch
});

// clang-format on
TEST_P(OpSolarize, tensor_correct_output)
{
    ew::RunTensorCorrectDispatch(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                 nvcv::ImageFormat{GetParamValue<3>()}, goldFor, invokeFor);
}

// VarShape correctness: bit-exact vs the CPU gold (fixed uint8 threshold) ------------------------
TEST(OpSolarize, varshape_correct_output)
{
    ew::RunVarShapeCorrect([](uint8_t v) { return SolarizeScalarGold<uint8_t>(v, 100.0); }, invokeThr(100.0));
}

TEST(OpSolarize, threshold_normalization_covers_round_up_and_nan)
{
    const double justAboveHalf = std::nextafter(0.5, 1.0);
    ew::RunTensorCorrect<float>(
        17, 13, 1, nvcv::FMT_F32, [justAboveHalf](float v) { return SolarizeScalarGold<float>(v, justAboveHalf); },
        invokeThr(justAboveHalf));

    const double nanThreshold = std::numeric_limits<double>::quiet_NaN();
    ew::RunTensorCorrect<uint8_t>(
        17, 13, 1, nvcv::FMT_U8, [nanThreshold](uint8_t v) { return SolarizeScalarGold<uint8_t>(v, nanThreshold); },
        invokeThr(nanThreshold));
}

TEST(OpSolarize, threshold_normalization_covers_finite_boundary_paths)
{
    const double justBelowHalf = std::nextafter(0.5, 0.0);
    ew::RunTensorCorrect<float>(
        17, 13, 1, nvcv::FMT_F32, [justBelowHalf](float v) { return SolarizeScalarGold<float>(v, justBelowHalf); },
        invokeThr(justBelowHalf));

    constexpr double finiteIntegerThreshold = 128.0;
    ew::RunTensorCorrect<uint8_t>(
        17, 13, 1, nvcv::FMT_U8, [](uint8_t v) { return SolarizeScalarGold<uint8_t>(v, finiteIntegerThreshold); },
        invokeThr(finiteIntegerThreshold));
}

TEST(OpSolarize, threshold_normalization_uses_finite_carrier_boundaries)
{
    constexpr double exactFloatThreshold = 0.5;
    ew::RunTensorCorrect<float>(
        15, 7, 2, nvcv::FMT_RGBf32, [](float v) { return SolarizeScalarGold<float>(v, exactFloatThreshold); },
        invokeThr(exactFloatThreshold));

    constexpr double negativeIntegerThreshold = -0.25;
    ew::RunTensorCorrect<uint8_t>(
        31, 7, 2, nvcv::FMT_RGB8, [](uint8_t v) { return SolarizeScalarGold<uint8_t>(v, negativeIntegerThreshold); },
        invokeThr(negativeIntegerThreshold));
}

// F16 correctness: the reference is FP32 on the half-quantized input; kUlps = 1 because the
// pass-through branch is exact and the inverted branch is a single correctly-rounded half subtract
// (one rounding step; see HalfTestUtils.hpp). The threshold is 0.5 — exactly representable in both
// half and float — so pixels sitting exactly on the boundary compare deterministically (a threshold
// between adjacent half values could otherwise flip branch between the half kernel and FP32 gold).
constexpr float  kSolarizeF16Ulps      = 1.f;
constexpr double kSolarizeF16Threshold = 0.5;

const auto goldSolarizeF32 = [](float v)
{
    return v >= static_cast<float>(kSolarizeF16Threshold) ? 1.0f - v : v;
};

// The single-channel FMT_F16 rows matter: 1-channel F16 must take the scalar kernel that the
// wide-load (ushort4) exclusion in OpSolarize.cu falls back to, in both Tensor and VarShape paths.
// clang-format off
NVCV_TEST_SUITE_P(OpSolarizeF16, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       61,     23,     2,   nvcv::FMT_F16    }, // f16 / 1ch
    {       55,     33,     2,   nvcv::FMT_RGBf16 }, // f16 / 3ch
    {       23,     54,     1,   nvcv::FMT_RGBAf16}, // f16 / 4ch
});

// clang-format on
TEST_P(OpSolarizeF16, tensor_correct_output)
{
    ew::RunTensorCorrectF16(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                            nvcv::ImageFormat{GetParamValue<3>()}, goldSolarizeF32, invokeThr(kSolarizeF16Threshold),
                            kSolarizeF16Ulps);
}

NVCV_TEST_SUITE_P(OpSolarizeF16VarShape, test::ValueList<nvcv::ImageFormat>{nvcv::FMT_F16, nvcv::FMT_RGBf16});

TEST_P(OpSolarizeF16VarShape, correct_output)
{
    ew::RunVarShapeCorrectF16(GetParam(), goldSolarizeF32, invokeThr(kSolarizeF16Threshold), kSolarizeF16Ulps);
}

// Planar ≡ interleaved parity (fake-planar) -----------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpSolarizePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64,  48, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {100,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 66,  49, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

NVCV_TEST_SUITE_P(OpSolarizePlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 66,  49, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

// F16 parity data lies in (0, 1], so the historical 100.0 threshold would leave the invert branch
// unexercised; the F16 rows use the mid-range 0.5 (both layouts see the same threshold, so parity
// is checked on both branches).
static double PlanarParityThreshold(nvcv::ImageFormat interleavedFmt)
{
    return interleavedFmt == nvcv::FMT_RGBAf16 ? kSolarizeF16Threshold : 100.0;
}

TEST_P(OpSolarizePlanar, tensor_matches_interleaved)
{
    const double thr = PlanarParityThreshold(GetParamValue<4>());
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [thr](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                        nvcv::ImageFormat) { EXPECT_NO_THROW(invokeThr(thr)(s, src, dst)); });
}

TEST_P(OpSolarizePlanarVarShape, varshape_matches_interleaved)
{
    const double thr = PlanarParityThreshold(GetParamValue<4>());
    test::planar::RunVarShapeParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                    GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                    [thr](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                          const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    { EXPECT_NO_THROW(invokeThr(thr)(s, src, dst)); });
}

// F64 dtype: call RunTensorCorrect<double> directly since RunTensorCorrectDispatch only dispatches
// over u8/u16/f32.  MidThreshold<double>() = 0.5, so both branches are exercised.
TEST(OpSolarize, tensor_correct_output_f64)
{
    ew::RunTensorCorrect<double>(41, 19, 1, nvcv::FMT_F64, goldFor(double{}), invokeFor(double{}));
}

// Regression for the interleaved single-channel F64 fast path: it packs four doubles into a
// double4 kernel whose threshold compare ran through SolarizeThreshold's float32 boundary.
// That boundary rounds up to the smallest float >= threshold (16777218.0f here) while a double
// at the threshold rounds down to the float below it (16777216.0f), so v == threshold escapes
// inversion; the scalar kernel's plain double compare inverts it. W = 8 keeps the row 32-byte
// aligned so pre-fix it is the vector kernel, not the scalar one, that runs.
TEST(OpSolarize, tensor_f64_vector_path_compares_in_double)
{
    constexpr double kThreshold = 16777217.0; // 2^24 + 1: exact in double, rounded in float

    // Gold is the scalar-kernel semantics: v >= threshold inverts to bound - v with bound = 1.
    // Element 0 is the discriminator: the double compare and the float-narrowed compare disagree.
    // The rest pin the exact double arithmetic and pass-through behavior around the boundary.
    std::vector<double> input{
        16777217.0,         // == threshold: must invert; the narrowed compare lets it through
        16777216.0,         // just below: pass through (both domains agree)
        16777218.0,         // just above: invert (both domains agree)
        -16777217.0,        // negative: pass through
        0.25,               // small: pass through
        9007199254740992.0, // 2^53: inverts to exactly 1 - 2^53 (exact only in double arithmetic)
        4503599627370495.5, // fractional near 2^52: inversion must keep the .5
        1.0,                // top of the unit range: pass through
    };
    std::vector<double> expected(input.size());
    for (size_t i = 0; i < input.size(); ++i) expected[i] = input[i] >= kThreshold ? 1.0 - input[i] : input[i];

    nvcv::Tensor src     = nvcv::util::CreateTensor(1, 8, 1, nvcv::FMT_F64);
    nvcv::Tensor dst     = nvcv::util::CreateTensor(1, 8, 1, nvcv::FMT_F64);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    const size_t bytes = input.size() * sizeof(double);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));

    cvcuda::Solarize op;
    ASSERT_NO_THROW(op(nullptr, src, dst, kThreshold));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));

    std::vector<double> actual(input.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
}

TEST(OpSolarize, f64_vector_body_and_varshape)
{
    const nvcv::ImageFormat rgb{NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZ1, ASSOCIATED, X64_Y64_Z64)};
    const nvcv::ImageFormat rgba{
        NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, UNDEFINED, PL, FLOAT, XYZW, ASSOCIATED, X64_Y64_Z64_W64)};
    for (auto fmt : {nvcv::FMT_F64, rgb, rgba})
    {
        ew::RunTensorCorrect<double>(256, 17, 2, fmt, goldFor(double{}), invokeThr(0.5));
        ew::RunTensorCorrect<double>(131, 17, 2, fmt, goldFor(double{}), invokeThr(0.5));
        ew::RunVarShapeCorrectTyped<double>(fmt, goldFor(double{}), invokeThr(0.5));
    }
}

static void RunSolarizeF64TensorCase(const std::vector<double> &values, int channels, bool planar)
{
    const double inf   = std::numeric_limits<double>::infinity();
    const double nan   = std::numeric_limits<double>::quiet_NaN();
    const int    width = static_cast<int>(values.size()) | 1;

    const nvcv::TensorShape shape = planar ? nvcv::TensorShape{{1, channels, 1, width}, "NCHW"}
                                        : nvcv::TensorShape{{1, 1, width, channels}, "NHWC"};
    nvcv::Tensor src(shape, nvcv::TYPE_F64);
    nvcv::Tensor dst(shape, nvcv::TYPE_F64);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    auto         access  = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(access);
    const size_t         bytes = planar ? access->planeStride() * channels : access->rowStride();
    std::vector<uint8_t> input(bytes, 0);
    std::vector<uint8_t> output(bytes);
    const auto           offset = [&](int x, int c)
    {
        return planar ? c * access->planeStride() + x * sizeof(double) : (x * channels + c) * sizeof(double);
    };
    for (int c = 0; c < channels; ++c)
        for (int x = 0; x < width; ++x)
            std::memcpy(input.data() + offset(x, c), &values[(x + c) % values.size()], sizeof(double));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    for (double threshold : {-inf, -1., -0., 0., std::nextafter(0.5, 0.), 0.5, std::nextafter(0.5, 1.), 0.75, 1.,
                             std::nextafter(1., inf), 16777217., inf, nan})
    {
        SCOPED_TRACE(testing::Message() << channels << ":" << planar << ":" << threshold);
        cvcuda::Solarize op;
        op(nullptr, src, dst, threshold);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < channels * width; ++i)
        {
            const int    c        = i / width;
            const int    x        = i % width;
            const double v        = values[(x + c) % values.size()];
            const double expected = v >= threshold ? 1. - v : v;
            double       actual;
            std::memcpy(&actual, output.data() + offset(x, c), sizeof(actual));
            EXPECT_EQ(0, std::memcmp(&expected, &actual, sizeof(actual)));
        }
    }
}

TEST(OpSolarize, f64_threshold_edges_and_tails)
{
    const double        inf = std::numeric_limits<double>::infinity();
    const double        nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> values{-inf,
                               -16777217.,
                               -1.,
                               -0.,
                               0.,
                               std::numeric_limits<double>::denorm_min(),
                               0.25,
                               std::nextafter(0.5, 0.),
                               0.5,
                               std::nextafter(0.5, 1.),
                               1.,
                               16777216.,
                               16777217.,
                               16777218.,
                               std::numeric_limits<double>::max(),
                               inf,
                               nan};
    for (int exponent = 1; exponent <= 53; ++exponent)
    {
        const double value = 1. - std::ldexp(1., -exponent);
        values.push_back(std::nextafter(value, 0.));
        values.push_back(value);
        values.push_back(std::nextafter(value, inf));
    }
    values.push_back(std::nextafter(1., inf));
    const auto cases = test::Combine(test::ValueList<int>{1, 3, 4}, test::ValueList<bool>{false, true});
    for (const auto &[channels, planar] : cases)
        ASSERT_NO_FATAL_FAILURE(RunSolarizeF64TensorCase(values, channels, planar));
}

static void RunSolarizeHalfThreshold(const nvcv::Tensor &src, nvcv::Tensor &dst,
                                     const nvcv::TensorDataStridedCuda &dstData, const std::vector<__half> &input,
                                     double threshold)
{
    SCOPED_TRACE(threshold);
    cvcuda::Solarize op;
    ASSERT_NO_THROW(op(nullptr, src, dst, threshold));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    std::vector<__half> got;
    nvcv::util::GetImageVectorFromTensor<__half>(dstData, 0, got);
    ASSERT_EQ(input.size(), got.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        const float  value = __half2float(input[i]);
        const __half gold  = static_cast<double>(value) >= threshold ? __float2half(1.f - value) : input[i];
        ASSERT_EQ(static_cast<__half_raw>(gold).x, static_cast<__half_raw>(got[i]).x) << i;
    }
}

TEST(OpSolarize, exhaustive_half_thresholds)
{
    constexpr int count = 65536;
    const double  inf   = std::numeric_limits<double>::infinity();
    const double  nan   = std::numeric_limits<double>::quiet_NaN();
    for (auto format : {nvcv::FMT_F16, nvcv::FMT_RGBf16, nvcv::FMT_RGBAf16, nvcv::FMT_RGBf16p, nvcv::FMT_RGBAf16p})
    {
        SCOPED_TRACE(format);
        std::vector<__half> input(count * format.numChannels());
        for (size_t i = 0; i < input.size(); ++i) input[i] = __half{__half_raw{static_cast<uint16_t>(i)}};
        auto src     = nvcv::util::CreateTensor(1, 256, 256, format);
        auto dst     = nvcv::util::CreateTensor(1, 256, 256, format);
        auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        nvcv::util::SetImageTensorFromVector<__half>(*srcData, input, 0);
        for (double threshold :
             {-inf, std::nextafter(-65504., -inf), -1., -0., 0., std::ldexp(1., -25), std::ldexp(1., -24),
              std::nextafter(0.5, 0.), 0.5, std::nextafter(0.5, 1.), std::nextafter(1., 0.), 1., std::nextafter(1., 2.),
              65504., std::nextafter(65504., inf), inf, nan})
        {
            ASSERT_NO_FATAL_FAILURE(RunSolarizeHalfThreshold(src, dst, *dstData, input, threshold));
        }
    }
}

static void RunSolarizeWideAlignmentCase(int srcOffset, int dstOffset)
{
    constexpr int     width    = 33;
    constexpr int     height   = 3;
    constexpr int     samples  = 2;
    constexpr int     channels = 4;
    constexpr int64_t row      = width * channels * sizeof(double) + 8;

    SCOPED_TRACE(testing::Message() << srcOffset << ":" << dstOffset);
    const size_t srcBytes = srcOffset + samples * height * row;
    const size_t dstBytes = dstOffset + samples * height * row;
    NVCVByte    *srcPtr{};
    NVCVByte    *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    auto wrap = [=](NVCVByte *ptr, int offset)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + offset;
        buffer.strides[0] = height * row;
        buffer.strides[1] = row;
        buffer.strides[2] = channels * sizeof(double);
        buffer.strides[3] = sizeof(double);
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, height, width, channels}, "NHWC"},
            nvcv::TYPE_F64, buffer
        });
    };
    auto                 src = wrap(srcPtr, srcOffset);
    auto                 dst = wrap(dstPtr, dstOffset);
    std::vector<uint8_t> input(srcBytes, 0xA5);
    std::vector<uint8_t> expected(dstBytes, 0xA5);
    std::vector<uint8_t> actual(dstBytes, 0xA5);
    for (int n = 0; n < samples; ++n)
        for (int y = 0; y < height; ++y)
            for (int i = 0; i < width * channels; ++i)
            {
                const double value = ((i + y + n) % 17) / 16.;
                const double gold  = value >= 0.5 ? 1. - value : value;
                const size_t off   = (n * height + y) * row + i * sizeof(double);
                std::memcpy(input.data() + srcOffset + off, &value, sizeof(value));
                std::memcpy(expected.data() + dstOffset + off, &gold, sizeof(gold));
            }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, dstBytes));
    cvcuda::Solarize op;
    ASSERT_NO_THROW(op(nullptr, src, dst, 0.5));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    EXPECT_EQ(expected, actual);
}

TEST(OpSolarize, f64_wide_scalar_alignment)
{
    const auto cases = test::Combine(test::ValueList<int>{0, 8}, test::ValueList<int>{0, 8});
    for (const auto &[srcOffset, dstOffset] : cases)
        ASSERT_NO_FATAL_FAILURE(RunSolarizeWideAlignmentCase(srcOffset, dstOffset));
}

namespace {

uint8_t SolarizePaddedValue(int64_t ordinal)
{
    if (ordinal == 0)
    {
        return 0;
    }
    if (ordinal == 1)
    {
        return 255;
    }
    return static_cast<uint8_t>((ordinal * 37 + 11) & 0xFF);
}

void CheckSolarizePaddedPlane(const uint8_t *srcBytes, const uint8_t *dstBytes, int height, int64_t rowBytes,
                              int64_t rowStride, int64_t planeStride, double threshold)
{
    for (int y = 0; y < height; ++y)
    {
        const auto rowOffset = static_cast<size_t>(y * rowStride);
        for (int64_t e = 0; e < rowBytes; ++e)
        {
            const size_t  offset   = rowOffset + static_cast<size_t>(e);
            const uint8_t expected = SolarizeScalarGold<uint8_t>(srcBytes[offset], threshold);
            ASSERT_EQ(expected, dstBytes[offset]) << "payload mismatch at row " << y << " byte " << e;
        }
        for (int64_t e = rowBytes; e < rowStride; ++e)
        {
            ASSERT_EQ(0xA5, dstBytes[rowOffset + static_cast<size_t>(e)]) << "row padding overwritten at row " << y;
        }
    }
    for (int64_t e = height * rowStride; e < planeStride; ++e)
    {
        ASSERT_EQ(0xA5, dstBytes[static_cast<size_t>(e)]) << "plane gap overwritten at byte " << e;
    }
}

void RunSolarizePaddedU8Tensor(nvcv::ImageFormat fmt, int width, int height, int batch, int rowPadding, int planeGap,
                               double threshold)
{
    const bool    planar       = fmt.numPlanes() > 1;
    const int     channels     = fmt.numChannels();
    const int64_t rowBytes     = static_cast<int64_t>(width) * (planar ? 1 : channels);
    const int64_t rowStride    = rowBytes + rowPadding;
    const int64_t planeStride  = rowStride * height + (planar ? planeGap : 0);
    const int64_t sampleStride = planeStride * (planar ? channels : 1);
    const int     numPlanes    = planar ? channels : 1;
    const size_t  totalBytes   = static_cast<size_t>(sampleStride) * batch;

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, totalBytes));

    auto wrapTensor = [&](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr = ptr;
        if (planar)
        {
            buffer.strides[0] = sampleStride;
            buffer.strides[1] = planeStride;
            buffer.strides[2] = rowStride;
            buffer.strides[3] = 1;
            return nvcv::TensorWrapData(
                nvcv::TensorDataStridedCuda{
                    nvcv::TensorShape{{batch, channels, height, width}, "NCHW"},
                    nvcv::DataType{            fmt.planeDataType(0)       },
                    buffer
            },
                nvcv::TensorDataCleanupCallback{[ptr](const nvcv::TensorData &)
                                                {
                                                    EXPECT_EQ(cudaSuccess, cudaFree(ptr));
                                                }});
        }

        buffer.strides[0] = sampleStride;
        buffer.strides[1] = rowStride;
        buffer.strides[2] = channels;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{   {batch, height, width, channels}, "NHWC"},
                nvcv::DataType{fmt.planeDataType(0).channelType(0)       },
                buffer
        },
            nvcv::TensorDataCleanupCallback{[ptr](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(ptr));
                                            }});
    };

    std::vector<uint8_t> srcBytes(totalBytes, 0x3D);
    for (int plane = 0; plane < batch * numPlanes; ++plane)
    {
        const int n = plane / numPlanes;
        const int p = plane % numPlanes;
        for (int y = 0; y < height; ++y)
        {
            const auto rowOffset = static_cast<size_t>(n * sampleStride + p * planeStride + y * rowStride);
            for (int64_t e = 0; e < rowBytes; ++e)
            {
                const int64_t ordinal = (static_cast<int64_t>(n * numPlanes + p) * height + y) * rowBytes + e;
                const auto    value   = SolarizePaddedValue(ordinal);
                srcBytes[rowOffset + static_cast<size_t>(e)] = value;
            }
        }
    }

    const std::vector<uint8_t> dstSentinel(totalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), totalBytes, cudaMemcpyHostToDevice));

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invokeThr(threshold)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));

    for (int plane = 0; plane < batch * numPlanes; ++plane)
    {
        const int n = plane / numPlanes;
        const int p = plane % numPlanes;
        SCOPED_TRACE(::testing::Message() << "sample " << n << " plane " << p);
        const auto offset = static_cast<size_t>(n * sampleStride + p * planeStride);
        ASSERT_NO_FATAL_FAILURE(CheckSolarizePaddedPlane(srcBytes.data() + offset, dstBytes.data() + offset, height,
                                                         rowBytes, rowStride, planeStride, threshold));
    }
}

} // namespace

TEST(OpSolarize, dispatch_gates_cover_padded_tensor_fallbacks)
{
    // A 36-byte U8 row is 4-byte aligned but not 16-byte aligned, so the carrier declines and the
    // single-channel Vec4 tensor path handles the payload and its scalar tail.
    RunSolarizePaddedU8Tensor(nvcv::FMT_U8, 30, 5, 1, 6, 0, 128.0);

    // The padded RGB8 row is neither carrier- nor Vec4-aligned, and its multi-channel instantiation
    // therefore reaches the scalar interleaved fallback.
    RunSolarizePaddedU8Tensor(nvcv::FMT_RGB8, 41, 13, 2, 2, 0, 128.0);

    // The one-byte gap makes the planar carrier plan inexpressible; the odd row stride also declines
    // the planar Vec4 path, leaving the scalar planar wrapper to process each plane.
    RunSolarizePaddedU8Tensor(nvcv::FMT_RGB8p, 41, 13, 2, 2, 1, 128.0);

    // F16 is excluded from both Vec4 tensor predicates; parity verifies its scalar planar fallback.
    test::planar::RunTensorParity(nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16, 17, 11, 17, 11, 1,
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeThr(0.5)(s, src, dst)); });
}

TEST(OpSolarize, coverage_planar_fallback_varshape_vector_finite_threshold)
{
    constexpr double finiteIntegerThreshold = 128.0;

    // The one-byte plane gap makes the carrier plan invalid, and the non-aligned plane stride
    // declines the planar Vec4 tensor launch, so the scalar planar fallback is selected.
    RunSolarizePaddedU8Tensor(nvcv::FMT_RGB8p, 37, 9, 2, 3, 1, finiteIntegerThreshold);

    // A regular planar batch satisfies the vector launch limits and returns through the planar
    // VarShape Vec4 path. The same finite threshold exercises both non-saturating integer bounds.
    test::planar::RunVarShapeParity(
        nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 67, 19, 67, 19, 2,
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeThr(finiteIntegerThreshold)(s, src, dst)); });
}

TEST(OpSolarize, threshold_normalization_covers_vec4_false_arms)
{
    // The padded U8 row declines the 16-byte carrier but remains eligible for the 4-byte Vec4
    // tensor path, so the finite integer threshold takes the false arms of all integer guards in
    // the device-side SolarizeThreshold constructor.
    RunSolarizePaddedU8Tensor(nvcv::FMT_U8, 30, 5, 1, 6, 0, 128.0);

    // One row beyond the carrier's 65535-region cap declines that path while four-row Vec4 blocks
    // still fit grid.y. The exact F32 threshold is representable, so float normalization does not
    // round it up before the vector kernel constructs its threshold on device.
    constexpr int                  width      = 4;
    constexpr int                  height     = 65536;
    constexpr size_t               rowBytes   = static_cast<size_t>(width) * sizeof(float);
    constexpr size_t               rowStride  = rowBytes + 16;
    constexpr size_t               totalBytes = rowStride * static_cast<size_t>(height);
    constexpr double               threshold  = 0.5;
    const std::array<float, width> values     = {0.25f, 0.5f, 0.75f, 1.0f};

    NVCVByte *srcPtr = nullptr;
    NVCVByte *dstPtr = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, totalBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, totalBytes));

    std::vector<uint8_t> srcBytes(totalBytes, 0x3D);
    for (int y = 0; y < height; ++y)
    {
        const size_t rowOffset = static_cast<size_t>(y) * rowStride;
        for (int x = 0; x < width; ++x)
        {
            std::memcpy(srcBytes.data() + rowOffset + static_cast<size_t>(x) * sizeof(float), &values[x],
                        sizeof(float));
        }
    }
    const std::vector<uint8_t> dstSentinel(totalBytes, 0xA5);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, srcBytes.data(), totalBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, dstSentinel.data(), totalBytes, cudaMemcpyHostToDevice));

    auto wrapTensor = [](NVCVByte *ptr)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr;
        buffer.strides[0] = rowStride * static_cast<int64_t>(height);
        buffer.strides[1] = rowStride;
        buffer.strides[2] = sizeof(float);
        buffer.strides[3] = sizeof(float);
        return nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{1, height, width, 1}, "NHWC"},
                nvcv::TYPE_F32, buffer
        },
            nvcv::TensorDataCleanupCallback{[ptr](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(ptr));
                                            }});
    };

    nvcv::Tensor src = wrapTensor(srcPtr);
    nvcv::Tensor dst = wrapTensor(dstPtr);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_NO_THROW(invokeThr(threshold)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(totalBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstPtr, totalBytes, cudaMemcpyDeviceToHost));
    for (int y = 0; y < height; ++y)
    {
        const size_t rowOffset = static_cast<size_t>(y) * rowStride;
        for (int x = 0; x < width; ++x)
        {
            float actual = 0.0f;
            std::memcpy(&actual, dstBytes.data() + rowOffset + static_cast<size_t>(x) * sizeof(float), sizeof(actual));
            ASSERT_EQ(SolarizeScalarGold<float>(values[x], threshold), actual)
                << "payload mismatch at row " << y << " column " << x;
        }
        for (size_t offset = rowBytes; offset < rowStride; ++offset)
        {
            ASSERT_EQ(0xA5, dstBytes[rowOffset + offset]) << "row padding overwritten at row " << y;
        }
    }
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpSolarize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_S16,   nvcv::FMT_S16  }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGB8p}, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpSolarize_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeThr(128.0));
}

TEST(OpSolarize_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invokeThr(0.5), 16, 16);
}

// The two rejections same_shape::ValidateSrcDstTensors owns. They are asserted per entry point
// rather than only where the fix was written, because that one validator now decides the tensor
// contract of four operators. Bodies and rationale: ElementwiseOpHarness.hpp.
TEST(OpSolarize_Negative, tensor_padded_column_stride)
{
    ew::ExpectPaddedColumnStrideTensorRejected(invokeThr(128.0));
}

TEST(OpSolarize_Negative, tensor_multi_channel_dtype)
{
    ew::ExpectMultiChannelDtypeTensorRejected(invokeThr(128.0));
}

// Regression: an aligned 1-channel F64 tensor once took the vectorized path, where
// SolarizeThreshold<double> narrows the threshold to float32 and misclassifies double values that
// collapse onto an adjacent float. v0 = 1 + 2^-24 is the midpoint of floats 1.0 and 1 + 2^-23:
// the float32 boundary rounds up to 1 + 2^-23 while v0 rounds down to 1.0f, so the narrowed compare
// leaves v0 un-inverted; the double gold inverts it. The F64 vec4 gate exclusion keeps the exact
// double compare (OpSolarize.cu). Width 4 keeps the row stride a multiple of sizeof(double4).
TEST(OpSolarize, tensor_f64_threshold_compare_stays_double)
{
    const double        v0        = 1.0 + 0x1p-24;
    const double        threshold = v0;
    std::vector<double> input{v0, 2.0, 0.25, 1.0 + 0x1p-23};
    std::vector<double> expected{1.0 - v0, -1.0, 0.25, 1.0 - (1.0 + 0x1p-23)}; // gold: v>=t ? 1-v : v
    nvcv::Tensor        src     = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F64);
    nvcv::Tensor        dst     = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_F64);
    auto                srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto                dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    const size_t bytes = input.size() * sizeof(double);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bytes, cudaMemcpyHostToDevice));
    cvcuda::Solarize op;
    op(nullptr, src, dst, threshold);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    std::vector<double> actual(input.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstData->basePtr(), bytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
}

TEST(OpSolarize_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaSolarizeCreate(nullptr));
}
