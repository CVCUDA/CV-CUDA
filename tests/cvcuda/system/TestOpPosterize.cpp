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
#include <cvcuda/OpPosterize.hpp>

#include <cstdint>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// Independent CPU gold: keep the top `bits` bits of each channel value, zero the rest:
//   out = in & ~((1 << (W - bits)) - 1), where W is the type bit width. Mirrors the documented
// oracle (torchvision/PIL posterize, generalized to 16-bit) and is computed independently of the
// kernel so the bit-exact EXPECT_EQ is a real regression check.
template<typename T>
T PosterizeScalarGold(T v, int bits)
{
    constexpr auto W = static_cast<int>(sizeof(T) * 8);
    uint32_t       mask;
    if (bits <= 0)
    {
        mask = 0u;
    }
    else if (bits >= W)
    {
        mask = ~0u;
    }
    else
    {
        mask = ~((1u << (W - bits)) - 1u);
    }
    return static_cast<T>(v & static_cast<T>(mask));
}

// Generic invoker pinning a given bit count (Posterize is integer-only, so the same bits value is
// used for u8 and u16).
auto invokeBits(int bits)
{
    return [bits](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::Posterize op;
        op(s, in, out, bits);
    };
}

} // namespace

// Tensor correctness over the declared dtype × channel matrix (u8 / u16 only) -------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterize, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,          format        (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8    }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8  }, // u8  / 3ch
    {       42,     53,     4,   nvcv::FMT_RGBA8 }, // u8  / 4ch
    {       80,     40,     2,   nvcv::FMT_U16   }, // u16 / 1ch
});

// clang-format on
TEST_P(OpPosterize, tensor_correct_output)
{
    const int               width  = GetParamValue<0>();
    const int               height = GetParamValue<1>();
    const int               batch  = GetParamValue<2>();
    const nvcv::ImageFormat fmt{GetParamValue<3>()};
    const int               bits = 4;

    // Posterize is integer-only, so dispatch directly over u8 / u16 (no f32 branch).
    if (ew::BaseKind(fmt) == 1)
    {
        ew::RunTensorCorrect<uint16_t>(
            width, height, batch, fmt, [](uint16_t v) { return PosterizeScalarGold<uint16_t>(v, 4); },
            invokeBits(bits));
    }
    else
    {
        ew::RunTensorCorrect<uint8_t>(
            width, height, batch, fmt, [](uint8_t v) { return PosterizeScalarGold<uint8_t>(v, 4); }, invokeBits(bits));
    }
}

// VarShape correctness: bit-exact vs the CPU gold ------------------------------------------------
TEST(OpPosterize, varshape_correct_output)
{
    ew::RunVarShapeCorrect([](uint8_t v) { return PosterizeScalarGold<uint8_t>(v, 3); }, invokeBits(3));
}

// Planar ≡ interleaved parity (fake-planar) -----------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterizePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    { 64,  48, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
});

NVCV_TEST_SUITE_P(OpPosterizePlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on
TEST_P(OpPosterizePlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeBits(3)(s, src, dst)); });
}

TEST_P(OpPosterizePlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeBits(3)(s, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpPosterize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F32,   nvcv::FMT_F32  }, // unsupported dtype (32-bit float)
    {nvcv::FMT_F16,   nvcv::FMT_F16  }, // unsupported dtype (16-bit float)
    {nvcv::FMT_S16,   nvcv::FMT_S16  }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_RGB8,  nvcv::FMT_RGB8p}, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,  nvcv::FMT_U8   }, // input/output channel mismatch
});

// clang-format on
TEST_P(OpPosterize_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeBits(4));
}

TEST(OpPosterize_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2S16, nvcv::FMT_2S16, invokeBits(4), 16, 16);
}

// bits outside [0, W] must be rejected (u8 has W=8, so 9 and -1 are out of range).
TEST(OpPosterize_Negative, rejects_bits_out_of_range)
{
    ew::ExpectRejected(nvcv::FMT_U8, nvcv::FMT_U8, invokeBits(9), 16, 16);
    ew::ExpectRejected(nvcv::FMT_U8, nvcv::FMT_U8, invokeBits(-1), 16, 16);
}
