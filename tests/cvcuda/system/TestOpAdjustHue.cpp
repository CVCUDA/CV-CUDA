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
#include <cvcuda/OpAdjustHue.hpp>
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

// torchvision's float->uint8 scale factor, identical to the kernel (OpAdjustHue.cu).
constexpr float kU8Scale = 255.0f + 1.0f - 1e-3f;

// Independent device and host evaluation may differ slightly while following the same operation order.
constexpr double kU8MaxDiff  = 1;
constexpr double kF32MaxDiff = 2e-6;

// HSV hue rotation on a normalized RGB pixel — identical arithmetic to the kernel's HsvHueRotate.
void HsvHueRotateGold(float r, float g, float b, float hueShift, float &outR, float &outG, float &outB)
{
    const float maxc = fmaxf(fmaxf(r, g), b);
    const float minc = fminf(fminf(r, g), b);
    const float v    = maxc;
    const float cr   = maxc - minc;
    const bool  eqc  = (maxc == minc);

    const float s   = cr / (eqc ? 1.0f : maxc);
    const float crd = eqc ? 1.0f : cr;
    const float rc  = (maxc - r) / crd;
    const float gc  = (maxc - g) / crd;
    const float bc  = (maxc - b) / crd;

    float hh;
    if (maxc == r)
    {
        hh = bc - gc;
    }
    else if (maxc == g)
    {
        hh = 2.0f + rc - bc;
    }
    else
    {
        hh = 4.0f + gc - rc;
    }

    float h = hh / 6.0f + 1.0f;
    h       = h - floorf(h);

    h = h + hueShift;
    h = h - floorf(h);

    const float h6 = h * 6.0f;
    const float ii = floorf(h6);
    const float f  = h6 - ii;
    const int   i  = static_cast<int>(ii) % 6;

    const float sxf       = s * f;
    const float oneMinusS = 1.0f - s;
    const float p         = fminf(fmaxf(oneMinusS * v, 0.0f), 1.0f);
    const float q         = fminf(fmaxf((1.0f - sxf) * v, 0.0f), 1.0f);
    const float t         = fminf(fmaxf((sxf + oneMinusS) * v, 0.0f), 1.0f);

    switch (i)
    {
    case 0:
        outR = v;
        outG = t;
        outB = p;
        break;
    case 1:
        outR = q;
        outG = v;
        outB = p;
        break;
    case 2:
        outR = p;
        outG = v;
        outB = t;
        break;
    case 3:
        outR = p;
        outG = q;
        outB = v;
        break;
    case 4:
        outR = t;
        outG = p;
        outB = v;
        break;
    default:
        outR = v;
        outG = p;
        outB = q;
        break;
    }
}

template<typename DT>
float NormalizeGold(DT v)
{
    if constexpr (std::is_floating_point_v<DT>)
    {
        return static_cast<float>(v);
    }
    else
    {
        return static_cast<float>(v) * (1.0f / 255.0f);
    }
}

template<typename DT>
DT StoreGold(float x)
{
    if constexpr (std::is_floating_point_v<DT>)
    {
        return static_cast<DT>(x);
    }
    else
    {
        float t = truncf(x * kU8Scale);
        t       = fminf(fmaxf(t, 0.0f), static_cast<float>(std::numeric_limits<DT>::max()));
        return static_cast<DT>(t);
    }
}

// Independent CPU gold on an interleaved HWC buffer: rotate each 3-channel pixel's hue; 1-channel is
// identity. Mirrors torchvision.transforms.v2.functional.adjust_hue.
template<typename DT>
std::vector<DT> AdjustHueGold(const std::vector<DT> &in, int channels, double hue)
{
    if (channels == 1)
    {
        return in; // identity
    }

    const auto      hueShift = static_cast<float>(hue);
    std::vector<DT> out(in.size());
    for (size_t p = 0; p + 3 <= in.size(); p += 3)
    {
        float oR;
        float oG;
        float oB;
        HsvHueRotateGold(NormalizeGold<DT>(in[p]), NormalizeGold<DT>(in[p + 1]), NormalizeGold<DT>(in[p + 2]), hueShift,
                         oR, oG, oB);
        out[p]     = StoreGold<DT>(oR);
        out[p + 1] = StoreGold<DT>(oG);
        out[p + 2] = StoreGold<DT>(oB);
    }
    return out;
}

// Bind the operator with a fixed hue for the parity / negative helpers.
auto invokeHue(double hue)
{
    return [hue](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::AdjustHue op;
        op(s, in, out, hue);
    };
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix + several hue factors ---------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustHue, test::ValueList<int, int, int, nvcv::ImageFormat, double>
{
    //   width, height, batch,            format              hue
    {       66,     55,     1,   nvcv::FMT_U8,       0.25}, // u8  / 1ch (identity)
    {      123,     67,     3,   nvcv::FMT_RGB8,     0.25}, // u8  / 3ch
    {       50,     40,     2,   nvcv::FMT_RGB8,    -0.30}, // u8  / 3ch (negative shift wraps)
    {       50,     40,     2,   nvcv::FMT_RGB8,     0.50}, // u8  / 3ch (edge +0.5)
    {       17,     19,     1,   nvcv::FMT_F32,      0.25}, // f32 / 1ch (identity)
    {      101,     33,     2,   nvcv::FMT_RGBf32,   0.10}, // f32 / 3ch
    {       60,     48,     1,   nvcv::FMT_RGBf32,  -0.50}, // f32 / 3ch (edge -0.5)
    {       60,     48,     1,   nvcv::FMT_RGBf32,   0.00}, // f32 / 3ch (identity)
});

// clang-format on
TEST_P(OpAdjustHue, correct_output)
{
    const int               w   = GetParamValue<0>();
    const int               h   = GetParamValue<1>();
    const int               b   = GetParamValue<2>();
    const nvcv::ImageFormat fmt = GetParamValue<3>();
    const double            hue = GetParamValue<4>();

    if (ew::BaseKind(fmt) == 2)
    {
        ew::RunTensorCorrectBuffer<float>(
            w, h, b, fmt,
            [hue](const std::vector<float> &in, int channels) { return AdjustHueGold(in, channels, hue); },
            invokeHue(hue), fmt.numChannels() == 3 ? kF32MaxDiff : 0);
    }
    else
    {
        ew::RunTensorCorrectBuffer<uint8_t>(
            w, h, b, fmt,
            [hue](const std::vector<uint8_t> &in, int channels) { return AdjustHueGold(in, channels, hue); },
            invokeHue(hue), fmt.numChannels() == 3 ? kU8MaxDiff : 0);
    }
}

TEST(OpAdjustHue, varshape_correct_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8, [](const std::vector<uint8_t> &in, int channels) { return AdjustHueGold(in, channels, 0.25); },
        invokeHue(0.25), kU8MaxDiff);
}

TEST(OpAdjustHue, varshape_negative_shift_output)
{
    ew::RunVarShapeCorrectBufferTyped<uint8_t>(
        nvcv::FMT_RGB8, [](const std::vector<uint8_t> &in, int channels) { return AdjustHueGold(in, channels, -0.4); },
        invokeHue(-0.4), kU8MaxDiff);
}

TEST(OpAdjustHue, zero_extent_tensors_are_noops)
{
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {0, 3, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeHue(0.25));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 0, 5, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeHue(0.25));
    ew::ExpectZeroExtentTensorNoop(
        nvcv::TensorShape{
            {1, 3, 0, 1},
            "NHWC"
    },
        nvcv::TYPE_U8, invokeHue(0.25));
}

TEST(OpAdjustHue, empty_matching_varshape_is_noop)
{
    ew::ExpectEmptyVarShapeNoop(invokeHue(0.25));
}

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustHuePlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on
TEST_P(OpAdjustHuePlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeHue(0.25)(s, src, dst)); });
}

TEST_P(OpAdjustHuePlanar, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeHue(0.25)(s, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustHue_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_U16,      nvcv::FMT_U16   }, // unsupported dtype (16-bit unsigned)
    {nvcv::FMT_F16,      nvcv::FMT_F16   }, // unsupported dtype (16-bit float)
    {nvcv::FMT_RGBA8,    nvcv::FMT_RGBA8 }, // unsupported channel count (4)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,     nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpAdjustHue_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeHue(0.25));
}

TEST(OpAdjustHue_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invokeHue(0.25), 16, 16);
}

TEST(OpAdjustHue_Negative, rejects_non_rgb_varshape_formats)
{
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8, invokeHue(0.25));
    ew::ExpectVarShapeRejected(nvcv::FMT_YUV8p, invokeHue(0.25));
    ew::ExpectVarShapeRejected(nvcv::FMT_BGR8, invokeHue(0.25));
    ew::ExpectVarShapeRejected(nvcv::FMT_NV12, invokeHue(0.25));
}

TEST(OpAdjustHue_Negative, rejects_vector_tensor_dtype)
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
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { cvcuda::AdjustHue{}(stream, src, dst, 0.25); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustHue_Negative, rejects_out_of_range_hue)
{
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeHue(0.75));
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeHue(-0.75));
}

TEST(OpAdjustHue_Negative, rejects_mismatched_varshape_image_sizes)
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

    cvcuda::AdjustHue op;
    try
    {
        op(nullptr, src, dst, 0.25);
        FAIL() << "Expected mismatched image sizes to be rejected";
    }
    catch (const nvcv::Exception &e)
    {
        EXPECT_EQ(nvcv::Status::ERROR_INVALID_ARGUMENT, e.code());
        EXPECT_STREQ("Input and output image 1 sizes must match: input is 31x37, output is 23x29", e.msg());
    }
}

TEST(OpAdjustHue_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaAdjustHueCreate(nullptr));
}
