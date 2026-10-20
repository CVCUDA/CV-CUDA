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
#include <cstdint>
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
    {      128,      1,     2,   nvcv::FMT_RGBf32,   0.25},
    {      129,      1,     2,   nvcv::FMT_RGBf32,  -0.25},
    {      132,      3,     2,   nvcv::FMT_RGBf32,   0.10},
    {      256,      1,     2,   nvcv::FMT_RGBf32,   0.00},
    {      257,      3,     2,   nvcv::FMT_RGBf32,  -0.13},
    {     1024,    256,     1,   nvcv::FMT_RGBf32,   0.23},
    {     1024,    257,     1,   nvcv::FMT_RGBf32,  -0.13},
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

TEST(OpAdjustHue, hsv_sectors_cover_pixel_paths)
{
    auto runTensor = []<typename Input>(nvcv::ImageFormat fmt, Input input, double maxDiff)
    {
        using DT = typename decltype(input)::value_type;

        const int channels = fmt.numChannels();
        ASSERT_EQ(input.size() % static_cast<size_t>(channels), 0u);

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        nvcv::Tensor src     = nvcv::util::CreateTensor(1, static_cast<int>(input.size() / channels), 1, fmt);
        nvcv::Tensor dst     = nvcv::util::CreateTensor(1, static_cast<int>(input.size() / channels), 1, fmt);
        auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(srcData, nullptr);
        ASSERT_NE(dstData, nullptr);

        nvcv::util::SetImageTensorFromVector<DT>(*srcData, input, 0);
        ASSERT_NO_THROW(cvcuda::AdjustHue{}(stream, src, dst, 0.0));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<DT> got;
        nvcv::util::GetImageVectorFromTensor<DT>(*dstData, 0, got);
        ew::ExpectBuffer(AdjustHueGold(input, channels, 0.0), got, maxDiff);

        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    };

    runTensor(nvcv::FMT_RGB8,
              std::vector<uint8_t>{
                  255, 0,   0,   // output sector 0, max red
                  255, 0,   128, // output sector 5, max red
                  128, 255, 0,   // output sector 1, max green
                  0,   255, 128, // output sector 2, max green
                  0,   128, 255, // output sector 3, max blue
                  128, 0,   255, // output sector 4, max blue
                  64,  64,  64   // equal channels
              },
              kU8MaxDiff);
    runTensor(nvcv::FMT_RGBf32,
              std::vector<float>{
                  1.0f,  0.0f,  0.0f, // output sector 0, max red
                  1.0f,  0.0f,  0.5f, // output sector 5, max red
                  0.5f,  1.0f,  0.0f, // output sector 1, max green
                  0.0f,  1.0f,  0.5f, // output sector 2, max green
                  0.0f,  0.5f,  1.0f, // output sector 3, max blue
                  0.5f,  0.0f,  1.0f, // output sector 4, max blue
                  0.25f, 0.25f, 0.25f // equal channels
              },
              kF32MaxDiff);
    runTensor(nvcv::FMT_U8, std::vector<uint8_t>{0, 255, 127}, 0);
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

TEST(OpAdjustHue, varshape_float32_matches_gold)
{
    ew::RunVarShapeCorrectBufferTyped<float>(
        nvcv::FMT_RGBf32, [](const std::vector<float> &in, int channels) { return AdjustHueGold(in, channels, 0.25); },
        invokeHue(0.25), kF32MaxDiff);
}

TEST(OpAdjustHue, varshape_float32_group_boundaries_match_gold)
{
    ew::RunVarShapeCorrectBufferTyped<float>(
        nvcv::FMT_RGBf32, [](const std::vector<float> &in, int channels) { return AdjustHueGold(in, channels, -0.13); },
        invokeHue(-0.13), kF32MaxDiff, {128, 129, 257}, {3, 5, 7});
}

// F16 correctness: the kernel widens every half to float, runs the whole RGB->HSV->rotate->RGB
// chain in float (identical to the F32 path), and rounds once on the half store. kUlps = 8 per
// the HalfTestUtils.hpp policy: a transcendental-class HSV chain of divides, multiplies, and
// sector blends whose host/device float contraction may differ across several steps.
constexpr float kHueF16Ulps = 8.f;

static auto hueGoldF32(double hue)
{
    return [hue](const std::vector<float> &in, int channels)
    {
        return AdjustHueGold<float>(in, channels, hue);
    };
}

// clang-format off
NVCV_TEST_SUITE_P(OpAdjustHueF16, test::ValueList<int, int, int, nvcv::ImageFormat, double>
{
    //   width, height, batch,           format               hue
    {       33,     23,     2,   nvcv::FMT_F16,       0.25}, // f16 / 1ch (identity)
    {       57,     41,     2,   nvcv::FMT_RGBf16,    0.25}, // f16 / 3ch
    {       50,     40,     1,   nvcv::FMT_RGBf16,   -0.30}, // f16 / 3ch (negative shift wraps)
});

// clang-format on
TEST_P(OpAdjustHueF16, tensor_matches_fp32_gold)
{
    const double hue = GetParamValue<4>();
    ew::RunTensorCorrectBufferF16(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                  hueGoldF32(hue), invokeHue(hue), kHueF16Ulps);
}

TEST(OpAdjustHueF16, varshape_matches_fp32_gold)
{
    ew::RunVarShapeCorrectBufferF16(nvcv::FMT_RGBf16, hueGoldF32(0.25), invokeHue(0.25), kHueF16Ulps);
}

TEST(OpAdjustHueF16, hsv_sectors_cover_half_paths)
{
    constexpr int            width = 7;
    const std::vector<float> input{
        1.0f,  0.0f,  0.0f, // max red, output sector 0
        1.0f,  0.0f,  0.5f, // max red, output sector 5
        0.5f,  1.0f,  0.0f, // max green, output sector 1
        0.0f,  1.0f,  0.5f, // max green, output sector 2
        0.0f,  0.5f,  1.0f, // max blue, output sector 3
        0.5f,  0.0f,  1.0f, // max blue, output sector 4
        0.25f, 0.25f, 0.25f // equal channels
    };

    std::vector<__half> inputHalf;
    inputHalf.reserve(input.size());
    for (float value : input)
    {
        inputHalf.push_back(__float2half(value));
    }

    std::vector<float> quantized(inputHalf.size());
    for (size_t i = 0; i < inputHalf.size(); ++i)
    {
        quantized[i] = __half2float(inputHalf[i]);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src     = nvcv::util::CreateTensor(1, width, 1, nvcv::FMT_RGBf16);
    nvcv::Tensor dst     = nvcv::util::CreateTensor(1, width, 1, nvcv::FMT_RGBf16);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    nvcv::util::SetImageTensorFromVector<__half>(*srcData, inputHalf, 0);
    ASSERT_NO_THROW(cvcuda::AdjustHue{}(stream, src, dst, 0.0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<__half> got;
    nvcv::util::GetImageVectorFromTensor<__half>(*dstData, 0, got);
    ew::ExpectBuffer(AdjustHueGold(quantized, 3, 0.0), got, kHueF16Ulps);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
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
    {100,  80, 2, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
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

TEST(OpAdjustHuePlanar, single_channel_tensor_matches_interleaved)
{
    test::planar::RunTensorSingleChannelLayoutParity(
        37, 29, 37, 29, 2, nvcv::TYPE_U8,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        { EXPECT_NO_THROW(invokeHue(0.25)(stream, src, dst)); });
}

TEST(OpAdjustHuePlanar, single_channel_f32_tensor_copy_path)
{
    test::planar::RunTensorSingleChannelLayoutParity(
        41, 23, 41, 23, 2, nvcv::TYPE_F32,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        { EXPECT_NO_THROW(invokeHue(-0.25)(stream, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected -------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustHue_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_U16,      nvcv::FMT_U16   }, // unsupported dtype (16-bit unsigned)
    {nvcv::FMT_F64,      nvcv::FMT_F64   }, // unsupported dtype (64-bit float; F16 is now valid)
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

// Both tensor launch guards run before a kernel launch, so these cases use only token allocations.
TEST(OpAdjustHue_Negative, tensor_launch_guards_reject_oversized_batch_and_span)
{
    cvcuda::AdjustHue op;

    {
        constexpr int64_t kSamples = 65536;
        constexpr auto    kBytes   = static_cast<size_t>(kSamples);

        NVCVByte *srcPtr = nullptr;
        NVCVByte *dstPtr = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, kBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, kBytes));

        auto wrap = [](NVCVByte *ptr)
        {
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = ptr;
            buffer.strides[0] = 1;
            buffer.strides[1] = 1;
            buffer.strides[2] = 1;
            buffer.strides[3] = 1;
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{kSamples, 1, 1, 1}, "NHWC"},
                nvcv::TYPE_U8, buffer
            });
        };

        nvcv::Tensor src = wrap(srcPtr);
        nvcv::Tensor dst = wrap(dstPtr);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(nullptr, src, dst, 0.25); }));

        ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
        ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    }

    {
        constexpr int64_t kSamples      = 2;
        constexpr int64_t kRows         = 50000;
        constexpr int64_t kRowStride    = kRows;
        constexpr int64_t kSampleStride = kRows * kRowStride;

        NVCVByte *srcPtr = nullptr;
        NVCVByte *dstPtr = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, 1024));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, 1024));

        auto wrap = [](NVCVByte *ptr)
        {
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = ptr;
            buffer.strides[0] = kSampleStride;
            buffer.strides[1] = kRowStride;
            buffer.strides[2] = 1;
            buffer.strides[3] = 1;
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{kSamples, kRows, kRows, 1}, "NHWC"},
                nvcv::TYPE_U8, buffer
            });
        };

        nvcv::Tensor src = wrap(srcPtr);
        nvcv::Tensor dst = wrap(dstPtr);
        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { op(nullptr, src, dst, 0.25); }));

        ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
        ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    }
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
