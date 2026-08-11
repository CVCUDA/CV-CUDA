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

#include <type_traits>

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

// Planar ≡ interleaved parity (fake-planar) -----------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpSolarizePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64,  48, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    {100,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

NVCV_TEST_SUITE_P(OpSolarizePlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on
TEST_P(OpSolarizePlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeThr(100.0)(s, src, dst)); });
}

TEST_P(OpSolarizePlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeThr(100.0)(s, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpSolarize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F16,   nvcv::FMT_F16  }, // unsupported dtype (16-bit float)
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
