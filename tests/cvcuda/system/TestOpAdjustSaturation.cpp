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

#include "ElementwiseOpHarness.hpp" // BaseKind, ExpectRejected
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdjustSaturation.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

namespace {

// torchvision luminance coefficients, identical to the kernel (OpAdjustSaturation.cu).
constexpr float kR2Y = 0.2989f;
constexpr float kG2Y = 0.587f;
constexpr float kB2Y = 0.114f;

// Default device contraction can differ from the independent CPU evaluation by one output level.
constexpr double kU8MaxDiff  = 1;
constexpr double kF32MaxDiff = 1e-6;

// Clamp + store mirroring the kernel's StoreSaturation: integers truncate toward zero into the dtype
// range; floats clamp to [0, 1].
template<typename DT>
DT StoreGold(float v)
{
    if constexpr (std::is_floating_point_v<DT>)
    {
        return static_cast<DT>(fminf(fmaxf(v, 0.0f), 1.0f));
    }
    else
    {
        float t = truncf(v);
        t       = fminf(fmaxf(t, 0.0f), static_cast<float>(std::numeric_limits<DT>::max()));
        return static_cast<DT>(t);
    }
}

// Independent CPU gold on an interleaved HWC buffer: blend each 3-channel pixel toward its luminance;
// 1-channel is identity. Mirrors torchvision.transforms.v2.functional.adjust_saturation.
template<typename DT>
std::vector<DT> AdjustSaturationGold(const std::vector<DT> &in, int channels, double saturation)
{
    if (channels == 1)
    {
        return in; // identity
    }

    const auto      ratio    = static_cast<float>(saturation);
    const auto      oneMinus = static_cast<float>(1.0 - saturation);
    std::vector<DT> out(in.size());
    for (size_t p = 0; p + 3 <= in.size(); p += 3)
    {
        const auto r = static_cast<float>(in[p]);
        const auto g = static_cast<float>(in[p + 1]);
        const auto b = static_cast<float>(in[p + 2]);

        float gray = (kR2Y * r + kG2Y * g) + kB2Y * b;
        if constexpr (!std::is_floating_point_v<DT>)
        {
            gray = floorf(gray);
        }

        out[p]     = StoreGold<DT>(ratio * r + oneMinus * gray);
        out[p + 1] = StoreGold<DT>(ratio * g + oneMinus * gray);
        out[p + 2] = StoreGold<DT>(ratio * b + oneMinus * gray);
    }
    return out;
}

// Bind the operator with a fixed saturation for the parity / negative helpers.
auto invokeSat(double saturation)
{
    return [saturation](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::AdjustSaturation op;
        op(s, in, out, saturation);
    };
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix + edge factors --------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturation, test::ValueList<int, int, int, nvcv::ImageFormat, double>
{
    //   width, height, batch,            format            saturation
    {       66,     55,     1,   nvcv::FMT_U8,      0.5}, // u8  / 1ch (identity)
    {      123,     67,     3,   nvcv::FMT_RGB8,    0.5}, // u8  / 3ch blend
    {       50,     40,     2,   nvcv::FMT_RGB8,    0.0}, // u8  / 3ch -> grayscale
    {       50,     40,     2,   nvcv::FMT_RGB8,    1.7}, // u8  / 3ch over-saturate (clamp to 255)
    {       17,     19,     1,   nvcv::FMT_F32,     0.5}, // f32 / 1ch (identity)
    {      101,     33,     2,   nvcv::FMT_RGBf32,  0.5}, // f32 / 3ch blend
    {       60,     48,     1,   nvcv::FMT_RGBf32,  1.5}, // f32 / 3ch over-saturate (clamp to 1)
    {       60,     48,     1,   nvcv::FMT_RGBf32,  0.0}, // f32 / 3ch -> grayscale
});

// clang-format on
TEST_P(OpAdjustSaturation, correct_output)
{
    const int               w   = GetParamValue<0>();
    const int               h   = GetParamValue<1>();
    const int               b   = GetParamValue<2>();
    const nvcv::ImageFormat fmt = GetParamValue<3>();
    const double            sat = GetParamValue<4>();

    if (ew::BaseKind(fmt) == 2)
    {
        ew::RunTensorCorrectBuffer<float>(
            w, h, b, fmt,
            [sat](const std::vector<float> &in, int channels) { return AdjustSaturationGold(in, channels, sat); },
            invokeSat(sat), fmt.numChannels() == 3 ? kF32MaxDiff : 0);
    }
    else
    {
        ew::RunTensorCorrectBuffer<uint8_t>(
            w, h, b, fmt,
            [sat](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, sat); },
            invokeSat(sat), fmt.numChannels() == 3 ? kU8MaxDiff : 0);
    }
}

TEST(OpAdjustSaturation, varshape_correct_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8,
        [](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, 0.5); },
        invokeSat(0.5), kU8MaxDiff);
}

TEST(OpAdjustSaturation, varshape_grayscale_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8,
        [](const std::vector<uint8_t> &in, int channels) { return AdjustSaturationGold(in, channels, 0.0); },
        invokeSat(0.0), kU8MaxDiff);
}

TEST(OpAdjustSaturation, zero_extent_tensors_are_noops)
{
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {0, 3, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 3, 0, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeSat(0.5));
}

TEST(OpAdjustSaturation, empty_matching_varshape_is_noop)
{
    ew::ExpectEmptyVarShapeNoop(invokeSat(0.5));
}

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturationPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on
TEST_P(OpAdjustSaturationPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeSat(0.5)(s, src, dst)); });
}

TEST_P(OpAdjustSaturationPlanar, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeSat(0.5)(s, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustSaturation_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_U16,      nvcv::FMT_U16   }, // unsupported dtype (16-bit unsigned)
    {nvcv::FMT_F16,      nvcv::FMT_F16   }, // unsupported dtype (16-bit float)
    {nvcv::FMT_RGBA8,    nvcv::FMT_RGBA8 }, // unsupported channel count (4)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpAdjustSaturation_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeSat(0.5));
}

TEST(OpAdjustSaturation_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invokeSat(0.5), 16, 16);
}

TEST(OpAdjustSaturation_Negative, rejects_non_rgb_varshape_formats)
{
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8p, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_BGR8, invokeSat(0.5));
    ew::ExpectVarShapeRejected(nvcv::FMT_NV12, invokeSat(0.5));
}

TEST(OpAdjustSaturation_Negative, rejects_vector_tensor_dtype)
{
    nvcv::Tensor src(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    nvcv::Tensor dst(
        nvcv::TensorShape{
            {1, 2, 4, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { cvcuda::AdjustSaturation{}(stream, src, dst, 0.5); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustSaturation_Negative, rejects_negative_saturation)
{
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeSat(-1.0));
}

TEST(OpAdjustSaturation_Negative, rejects_mismatched_varshape_image_sizes)
{
    std::vector<nvcv::Image> srcImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{31, 37}, nvcv::FMT_RGB8}
    };
    std::vector<nvcv::Image> dstImages{
        nvcv::Image{{17, 19}, nvcv::FMT_RGB8},
        nvcv::Image{{23, 29}, nvcv::FMT_RGB8}
    };

    nvcv::ImageBatchVarShape src(2);
    nvcv::ImageBatchVarShape dst(2);
    src.pushBack(srcImages.begin(), srcImages.end());
    dst.pushBack(dstImages.begin(), dstImages.end());

    cvcuda::AdjustSaturation op;
    try
    {
        op(nullptr, src, dst, 0.5);
        FAIL() << "Expected mismatched image sizes to be rejected";
    }
    catch (const nvcv::Exception &e)
    {
        EXPECT_EQ(nvcv::Status::ERROR_INVALID_ARGUMENT, e.code());
        EXPECT_STREQ("Input and output image 1 sizes must match: input is 31x37, output is 23x29", e.msg());
    }
}

TEST(OpAdjustSaturation_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaAdjustSaturationCreate(nullptr));
}
