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

#include "ElementwiseOpHarness.hpp" // for elementwise::Bound / ExpectRejected
#include "PlanarParityUtils.hpp"
#include "TestCAPIEnum.h"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdjustContrast.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

using AdjustContrastTensorCall
    = void (cvcuda::AdjustContrast::*)(cudaStream_t, const nvcv::Tensor &, const nvcv::Tensor &, double) const;
using AdjustContrastVarShapeCall
    = void (cvcuda::AdjustContrast::*)(cudaStream_t, const nvcv::ImageBatch &, const nvcv::ImageBatch &, double) const;
static_assert(std::is_same_v<decltype(static_cast<AdjustContrastTensorCall>(&cvcuda::AdjustContrast::operator())),
                             AdjustContrastTensorCall>);
static_assert(std::is_same_v<decltype(static_cast<AdjustContrastVarShapeCall>(&cvcuda::AdjustContrast::operator())),
                             AdjustContrastVarShapeCall>);

namespace {

// A fixed contrast factor exercised across the matrix. 1.5 increases contrast, so blending away from
// the grayscale mean pushes the darkest/brightest pixels below 0 / above bound -- exercising both
// clamp directions plus the round-to-nearest SaturateCast path in one factor.
constexpr double kFactor = 1.5;

// torchvision BT.601 luma weights (match AdjustColorCommon.cuh); the CPU gold is an independent
// reimplementation of the documented formula, not a call into the operator.
constexpr float kLumaR = 0.2989f;
constexpr float kLumaG = 0.587f;
constexpr float kLumaB = 0.114f;

constexpr int kMeanBlock = 256;

uint32_t FourCC(const char (&code)[5])
{
    uint32_t value{};
    std::memcpy(&value, code, sizeof(value));
    return value;
}

template<typename T>
T RoundSaturateGold(float value)
{
    if constexpr (std::is_floating_point_v<T>)
    {
        return value;
    }
    else
    {
        const float base     = std::floor(value);
        const float fraction = value - base;
        auto        rounded  = static_cast<unsigned int>(base);
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1U) != 0))
        {
            ++rounded;
        }
        return static_cast<T>(rounded);
    }
}

// Independent CPU gold for one interleaved HWC image. The host reduction mirrors the documented
// fixed 256-lane tree so both integer and deliberately exact float cases can be compared bit-for-bit.
template<typename T>
std::vector<T> AdjustContrastGoldImage(const std::vector<T> &in, int width, int height, int channels, double factor,
                                       bool truncate = false)
{
    const int  numPixels = width * height;
    const auto bound     = static_cast<float>(ew::Bound<T>());
    const auto ratio     = static_cast<float>(factor);

    std::array<float, kMeanBlock> partial{};
    for (int p = 0; p < numPixels; ++p)
    {
        float gray;
        if (channels == 1)
        {
            gray = static_cast<float>(in[p]);
        }
        else
        {
            const auto r = static_cast<float>(in[static_cast<size_t>(p) * channels + 0]);
            const auto g = static_cast<float>(in[static_cast<size_t>(p) * channels + 1]);
            const auto b = static_cast<float>(in[static_cast<size_t>(p) * channels + 2]);
            gray         = kLumaR * r + kLumaG * g + kLumaB * b;
            if constexpr (std::is_integral_v<T>)
            {
                gray = std::floor(gray);
            }
        }
        partial[p % kMeanBlock] += gray;
    }
    for (int stride = kMeanBlock / 2; stride > 0; stride >>= 1)
    {
        for (int i = 0; i < stride; ++i)
        {
            partial[i] += partial[i + stride];
        }
    }
    const float mean = partial[0] / static_cast<float>(numPixels);

    std::vector<T> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        float pixel = ratio * static_cast<float>(in[i]) + (1.0f - ratio) * mean;
        pixel       = std::clamp(pixel, 0.0f, bound);
        if (truncate && std::is_integral_v<T>)
        {
            // torchvision's ``.to(dtype)`` drops the fractional part instead of rounding.
            pixel = std::trunc(pixel);
        }
        out[i] = RoundSaturateGold<T>(pixel);
    }
    return out;
}

// Fixed-factor invoker for var-shape / parity / negative cases.
auto invokeFactor(double factor)
{
    return [factor](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::AdjustContrast op;
        op(s, in, out, factor);
    };
}

auto invokeFactorRound(double factor, NVCVRoundMode roundMode)
{
    return [factor, roundMode](cudaStream_t s, const auto &in, auto &out)
    {
        cvcuda::AdjustContrast op;
        op(s, in, out, factor, roundMode);
    };
}

template<typename DT>
void FillContrastFloatPattern(std::vector<DT> &input, int numPixels, int channels)
{
    static_assert(std::is_floating_point_v<DT>);
    for (int p = 0; p < numPixels; ++p)
    {
        input[static_cast<size_t>(p) * channels] = (p & 1) == 0 ? 0.0f : 1.0f;
        for (int c = 1; c < channels; ++c)
        {
            input[static_cast<size_t>(p) * channels + c] = 0.0f;
        }
    }
}

// Tensor correctness is bit-exact. Float inputs use a deterministic alternating pattern whose luma
// and fixed-tree mean have an exact host/device representation.
template<typename DT>
void RunContrastTensor(int width, int height, int batch, nvcv::ImageFormat fmt, double factor)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    channels = fmt.numChannels();
    const size_t count    = static_cast<size_t>(width) * height * channels;
    const DT     bound    = ew::Bound<DT>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batch, width, height, fmt);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         outData   = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    std::default_random_engine   rng(0);
    std::vector<std::vector<DT>> inputs(batch);
    for (int s = 0; s < batch; ++s)
    {
        auto &in = inputs[s];
        in.resize(count);
        if constexpr (std::is_floating_point_v<DT>)
        {
            FillContrastFloatPattern(in, width * height, channels);
        }
        else
        {
            std::uniform_int_distribution dist(0, static_cast<int>(bound));
            for (auto &v : in) v = static_cast<DT>(dist(rng));
        }
        nvcv::util::SetImageTensorFromVector<DT>(*inData, in, s);
    }

    ASSERT_NO_THROW(invokeFactor(factor)(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int s = 0; s < batch; ++s)
    {
        SCOPED_TRACE(s);
        std::vector<DT> got;
        nvcv::util::GetImageVectorFromTensor<DT>(*outData, s, got);
        std::vector<DT> gold = AdjustContrastGoldImage<DT>(inputs[s], width, height, channels, factor);
        ASSERT_EQ(gold.size(), got.size());
        EXPECT_EQ(gold, got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// VarShape correctness over one interleaved dtype/channel variant. Float inputs use the same
// deterministic pure-red/black pattern as the Tensor test; factor zero makes the fixed-tree mean
// itself the exact expected output. The largest image selects parallel reduction for the whole
// mixed-size batch; Tensor coverage independently exercises the fallback and nonzero blending.
template<typename DT>
void RunContrastVarShape(nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int n      = 3;
    const int     ch     = fmt.numChannels();
    const double  factor = std::is_floating_point_v<DT> ? 0.0 : kFactor;

    std::default_random_engine            rng(0);
    std::uniform_int_distribution         sizeDist(20, 90);
    std::uniform_int_distribution         byteDist(0, 255);
    constexpr std::array<nvcv::Size2D, n> floatSizes{
        {{1, 1}, {256, 32}, {257, 33}}
    };

    std::vector<nvcv::Image>     srcImgs;
    std::vector<nvcv::Image>     dstImgs;
    std::vector<int>             ws(n);
    std::vector<int>             hs(n);
    std::vector<std::vector<DT>> hostIn(n);
    for (int i = 0; i < n; ++i)
    {
        if constexpr (std::is_floating_point_v<DT>)
        {
            ws[i] = floatSizes[i].w;
            hs[i] = floatSizes[i].h;
        }
        else
        {
            ws[i] = sizeDist(rng);
            hs[i] = sizeDist(rng);
        }
        srcImgs.emplace_back(nvcv::Size2D{ws[i], hs[i]}, fmt);
        dstImgs.emplace_back(nvcv::Size2D{ws[i], hs[i]}, fmt);

        const size_t rowElements = static_cast<size_t>(ws[i]) * ch;
        const size_t rowBytes    = rowElements * sizeof(DT);
        hostIn[i].resize(rowElements * hs[i]);
        if constexpr (std::is_floating_point_v<DT>)
        {
            FillContrastFloatPattern(hostIn[i], ws[i] * hs[i], ch);
        }
        else
        {
            for (auto &value : hostIn[i]) value = static_cast<DT>(byteDist(rng));
        }

        auto idata = srcImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(idata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hostIn[i].data(),
                                            rowBytes, rowBytes, hs[i], cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape src(n);
    nvcv::ImageBatchVarShape dst(n);
    src.pushBack(srcImgs.begin(), srcImgs.end());
    dst.pushBack(dstImgs.begin(), dstImgs.end());

    ASSERT_NO_THROW(invokeFactor(factor)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < n; ++i)
    {
        SCOPED_TRACE(i);
        const size_t    rowElements = static_cast<size_t>(ws[i]) * ch;
        const size_t    rowBytes    = rowElements * sizeof(DT);
        std::vector<DT> got(rowElements * hs[i]);
        auto            odata = dstImgs[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(odata, nullptr);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(got.data(), rowBytes, odata->plane(0).basePtr, odata->plane(0).rowStride,
                                            rowBytes, hs[i], cudaMemcpyDeviceToHost));
        const std::vector<DT> gold = AdjustContrastGoldImage<DT>(hostIn[i], ws[i], hs[i], ch, factor);
        EXPECT_EQ(gold, got);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

// Tensor correctness over the declared dtype x channel matrix (u8 / f32; 1 / 3 channels) ----------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrast, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    //   width, height, batch,            format          (dtype / channels)
    {       66,     55,     1,   nvcv::FMT_U8     }, // u8  / 1ch
    {      123,     67,     3,   nvcv::FMT_RGB8   }, // u8  / 3ch
    {      257,     33,     1,   nvcv::FMT_F32    }, // f32 / 1ch; parallel reduction + tail
    {      257,     33,     2,   nvcv::FMT_RGBf32 }, // f32 / 3ch; parallel reduction + tail
    // Width 64 avoids allocator row padding, exercising whole-sample rather than per-row addressing.
    {       64,     17,     2,   nvcv::FMT_U8     },
    {       64,     17,     2,   nvcv::FMT_RGB8   },
    {       64,     17,     2,   nvcv::FMT_F32    },
    {       64,     17,     2,   nvcv::FMT_RGBf32 },
});

// clang-format on
TEST_P(OpAdjustContrast, tensor_correct_output)
{
    const auto fmt = nvcv::ImageFormat{GetParamValue<3>()};
    if (ew::BaseKind(fmt) == 2)
    {
        RunContrastTensor<float>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), fmt, kFactor);
        // The 0/1 pattern clamps back to itself at kFactor, so factor zero separately exposes the
        // reduction result and keeps the parallel lane-sharding path bit-exact against the CPU gold.
        RunContrastTensor<float>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), fmt, 0.0);
        // Retain exact-gold coverage of the small-image fallback on both float formats.
        RunContrastTensor<float>(16, 16, GetParamValue<2>(), fmt, 0.0);
    }
    else
    {
        RunContrastTensor<uint8_t>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), fmt, kFactor);
    }
}

// VarShape correctness over the declared dtype x channel matrix: bit-exact vs the CPU gold -------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrastVarShape, test::ValueList<nvcv::ImageFormat>
{
    nvcv::FMT_U8,
    nvcv::FMT_RGB8,
    nvcv::FMT_F32,
    nvcv::FMT_RGBf32,
});

// clang-format on
TEST_P(OpAdjustContrastVarShape, correct_output)
{
    const nvcv::ImageFormat fmt = GetParam();
    if (ew::BaseKind(fmt) == 2)
    {
        RunContrastVarShape<float>(fmt);
    }
    else
    {
        RunContrastVarShape<uint8_t>(fmt);
    }
}

// F16 correctness: the kernel widens every half to float, reduces the mean in the same fixed
// 256-lane tree as the FP32 gold, and rounds once on the blended store. kUlps = 1 per the
// HalfTestUtils.hpp policy: a single clamped mul-add blend; the float-domain mean difference
// (host/device FMA contraction) is orders of magnitude below one half-ULP.
constexpr float kContrastF16Ulps = 1.f;

// Buffer-level FP32 gold for the F16 runners: the float instantiation of the bit-exact gold,
// run on the widened half-quantized input (numPixels recovered from the flat buffer).
static auto contrastGoldF32(double factor)
{
    return [factor](const std::vector<float> &in, int channels)
    {
        return AdjustContrastGoldImage<float>(in, static_cast<int>(in.size()) / channels, 1, channels, factor);
    };
}

// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrastF16, test::ValueList<int, int, int, nvcv::ImageFormat>
{
    // Parameters: width, height, batch, format (channels).
    {       33,     23,     2,   nvcv::FMT_F16    }, // f16 / 1ch
    {       57,     41,     2,   nvcv::FMT_RGBf16 }, // f16 / 3ch
});

// clang-format on
TEST_P(OpAdjustContrastF16, tensor_matches_fp32_gold)
{
    ew::RunTensorCorrectBufferF16(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                                  contrastGoldF32(kFactor), invokeFactor(kFactor), kContrastF16Ulps);
}

TEST(OpAdjustContrastF16, varshape_matches_fp32_gold)
{
    ew::RunVarShapeCorrectBufferF16(nvcv::FMT_RGBf16, contrastGoldF32(kFactor), invokeFactor(kFactor),
                                    kContrastF16Ulps);
}

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrastPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,   nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
    {100,  80, 2, nvcv::FMT_RGBf16p, nvcv::FMT_RGBf16},
    {257,  33, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
    // These widths avoid planar row padding to exercise whole-sample addressing.
    { 96,  17, 2,   nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
    { 64,  17, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
});

// clang-format on
TEST_P(OpAdjustContrastPlanar, tensor_matches_interleaved)
{
    test::planar::RunTensorParity(GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(),
                                  GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                  [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeFactor(kFactor)(s, src, dst)); });
}

TEST_P(OpAdjustContrastPlanar, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        { EXPECT_NO_THROW(invokeFactor(kFactor)(s, src, dst)); });
}

TEST(OpAdjustContrastPlanar, single_sample_rgb8_chw_matches_interleaved)
{
    test::planar::RunTensorParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 257, 33, 257, 33, 1,
                                  [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeFactor(kFactor)(stream, src, dst)); });
}

TEST(OpAdjustContrastPlanar, small_rgb8_chw_uses_three_plane_mean)
{
    test::planar::RunTensorParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 37, 29, 37, 29, 1,
                                  [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst,
                                     nvcv::ImageFormat) { EXPECT_NO_THROW(invokeFactor(kFactor)(stream, src, dst)); });
}

TEST(OpAdjustContrastPlanar, single_channel_tensor_matches_interleaved)
{
    test::planar::RunTensorSingleChannelLayoutParity(
        37, 29, 37, 29, 2, nvcv::TYPE_U8,
        [](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        { EXPECT_NO_THROW(invokeFactor(kFactor)(stream, src, dst)); });
}

// Negative tests: the complement of the support matrix must be rejected with
// NVCV_ERROR_INVALID_ARGUMENT (ew::ExpectRejected asserts this via nvcv::ProtectCall).
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrast_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F64,    nvcv::FMT_F64   }, // unsupported dtype (64-bit float; F16 is now valid)
    {nvcv::FMT_S16,    nvcv::FMT_S16   }, // unsupported dtype (signed 16-bit)
    {nvcv::FMT_U16,    nvcv::FMT_U16   }, // unsupported dtype (unsigned 16-bit)
    {nvcv::FMT_RGBA8,  nvcv::FMT_RGBA8 }, // unsupported channel count (4, interleaved)
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8p}, // unsupported plane count (4-plane planar; not 1 or 3)
    {nvcv::FMT_RGB8,   nvcv::FMT_RGB8p }, // layout mismatch (interleaved in, planar out)
    {nvcv::FMT_RGB8,   nvcv::FMT_RGBf32}, // input/output data type mismatch
});

// clang-format on
TEST_P(OpAdjustContrast_Negative, rejects_unsupported)
{
    ew::ExpectRejected(GetParamValue<0>(), GetParamValue<1>(), invokeFactor(kFactor));
}

TEST(OpAdjustContrast_Negative, create_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaAdjustContrastCreate(nullptr));
}

TEST(OpAdjustContrast_Negative, rejects_two_channel)
{
    ew::ExpectRejected(nvcv::FMT_2F32, nvcv::FMT_2F32, invokeFactor(kFactor), 16, 16);
}

// 2-plane planar input (NCHW C=2) has no predefined ImageFormat, so build the tensors directly.
// Before the plane-count guard, this dispatched as planar RGB and read a nonexistent plane 2 in the
// grayscale-mean kernel (out-of-bounds); it must be rejected up front.
TEST(OpAdjustContrast_Negative, rejects_two_plane_planar)
{
    test::planar::ExpectPlanarTensorRejected({1, 2, 16, 16}, {1, 2, 16, 16}, invokeFactor(kFactor));
}

TEST(OpAdjustContrast_Negative, rejects_negative_factor)
{
    ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeFactor(-0.5));
    test::planar::ExpectVarShapeRejected({nvcv::FMT_RGB8}, {nvcv::FMT_RGB8}, invokeFactor(-0.5));
}

TEST(OpAdjustContrast_Negative, rejects_packed_tensor_dtype)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src(
        {
            {1, 8, 8, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);
    nvcv::Tensor dst(
        {
            {1, 8, 8, 1},
            "NHWC"
    },
        nvcv::TYPE_3U8);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invokeFactor(kFactor)(stream, src, dst); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast_Negative, rejects_tensor_shape_and_pixel_stride_mismatches)
{
    cvcuda::AdjustContrast op;
    cudaStream_t           stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    {
        nvcv::Tensor src(
            {
                {2, 16, 16, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
        nvcv::Tensor dst(
            {
                {1, 16, 16, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, kFactor); }));
    }

    {
        nvcv::Tensor src(
            {
                {1, 16, 16, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
        nvcv::Tensor dst(
            {
                {1, 16, 17, 1},
                "NHWC"
        },
            nvcv::TYPE_U8);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, kFactor); }));
    }

    {
        constexpr int64_t kWidth        = 16;
        constexpr int64_t kHeight       = 16;
        constexpr int64_t kRowStride    = 32;
        constexpr int64_t kSampleStride = kRowStride * kHeight;

        NVCVByte *srcPtr = nullptr;
        NVCVByte *dstPtr = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcPtr), kSampleStride));
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstPtr), kSampleStride));

        auto wrap = [=](NVCVByte *base, int64_t colStride)
        {
            nvcv::TensorDataStridedCuda::Buffer buffer{};
            buffer.basePtr    = base;
            buffer.strides[0] = kSampleStride;
            buffer.strides[1] = kRowStride;
            buffer.strides[2] = colStride;
            buffer.strides[3] = colStride;
            return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{1, kHeight, kWidth, 1}, "NHWC"},
                nvcv::TYPE_U8, buffer
            });
        };

        {
            nvcv::Tensor src = wrap(srcPtr, 2);
            nvcv::Tensor dst = wrap(dstPtr, 1);
            EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, src, dst, kFactor); }));
        }

        ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
        ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Both tensor launch guards run before a kernel launch, so these cases use only token allocations.
TEST(OpAdjustContrast_Negative, tensor_launch_guards_reject_oversized_batch_and_span)
{
    cvcuda::AdjustContrast op;

    {
        constexpr int64_t kSamples = 65536;
        constexpr auto    kBytes   = static_cast<size_t>(kSamples);

        NVCVByte *srcPtr = nullptr;
        NVCVByte *dstPtr = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcPtr), kBytes));
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstPtr), kBytes));

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
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(nullptr, src, dst, kFactor); }));

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
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcPtr), 1024));
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstPtr), 1024));

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
        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { op(nullptr, src, dst, kFactor); }));

        ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
        ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    }
}

TEST(OpAdjustContrast, empty_inputs_are_noops)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor srcTensor(
        {
            {1, 0, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor dstTensor(
        {
            {1, 0, 8, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    EXPECT_NO_THROW(invokeFactor(kFactor)(stream, srcTensor, dstTensor));

    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    EXPECT_NO_THROW(invokeFactor(kFactor)(stream, srcBatch, dstBatch));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast, same_handle_cross_stream_submissions)
{
    cudaStream_t stream1{};
    cudaStream_t stream2{};
    cudaStream_t gateStream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&gateStream, cudaStreamNonBlocking));

    cudaEvent_t gate{};
    cudaEvent_t start2{};
    cudaEvent_t done2{};
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&start2));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&done2));

    constexpr int width  = 257;
    constexpr int height = 129;
    constexpr int batch  = 3;
    constexpr int count  = width * height;

    nvcv::Tensor src1     = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor dst1     = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor src2     = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    nvcv::Tensor dst2     = nvcv::util::CreateTensor(batch, width, height, nvcv::FMT_U8);
    auto         srcData1 = src1.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData1 = dst1.exportData<nvcv::TensorDataStridedCuda>();
    auto         srcData2 = src2.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData2 = dst2.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData1 && dstData1 && srcData2 && dstData2);

    std::vector<uint8_t> input1(count, 17);
    std::vector<uint8_t> input2(count, 231);
    for (int sample = 0; sample < batch; ++sample)
    {
        nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData1, input1, sample);
        nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData2, input2, sample);
    }

    cvcuda::AdjustContrast op;
    ASSERT_NO_THROW(op(stream1, src1, dst1, 0.0));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream1));

    ASSERT_EQ(cudaSuccess,
              cudaLaunchHostFunc(
                  gateStream, [](void *) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }, nullptr));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(gate, gateStream));
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream1, gate));
    ASSERT_NO_THROW(op(stream1, src1, dst1, 0.0));

    ASSERT_EQ(cudaSuccess, cudaEventRecord(start2, stream2));
    ASSERT_NO_THROW(op(stream2, src2, dst2, 0.0));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(done2, stream2));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(done2));

    float elapsedMs{};
    ASSERT_EQ(cudaSuccess, cudaEventElapsedTime(&elapsedMs, start2, done2));
    EXPECT_GE(elapsedMs, 100.f);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream1));

    for (int sample = 0; sample < batch; ++sample)
    {
        std::vector<uint8_t> got1;
        std::vector<uint8_t> got2;
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*dstData1, sample, got1);
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*dstData2, sample, got2);
        EXPECT_EQ(input1, got1);
        EXPECT_EQ(input2, got2);
    }

    ASSERT_EQ(cudaSuccess, cudaEventDestroy(gate));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(start2));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(done2));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream1));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream2));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(gateStream));
}

TEST(OpAdjustContrast, cuda_graph_capture_replays)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr int        width  = 31;
    constexpr int        height = 17;
    std::vector<uint8_t> input(width * height, 73);

    nvcv::Tensor src     = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_U8);
    nvcv::Tensor dst     = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_U8);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, input, 0);

    cvcuda::AdjustContrast op;
    ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ASSERT_NO_THROW(op(stream, src, dst, 0.0));

    cudaGraph_t graph{};
    ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphExec{};
    ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Each replay must see the input the host staged for it. `stream` is non-blocking, so it is not
    // ordered against the default stream that SetImageTensorFromVector copies on; a replay that
    // outruns that copy reduces the previous replay's fill value and, at contrast factor zero,
    // writes it straight to the output. Several replays keep that window from closing by luck.
    std::vector<uint8_t> got;
    for (int replay = 0; replay < 16; ++replay)
    {
        std::ranges::fill(input, static_cast<uint8_t>(73 + replay));
        nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, input, 0);
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(graphExec, stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*dstData, 0, got);
        EXPECT_EQ(input, got) << "replay " << replay;
    }

    ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(graphExec));
    ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// The apply-grid guard runs after workspace acquisition. This forces cleanup through releaseNoThrow
// on both ordinary submissions and CUDA graph capture without requiring a large allocation.
TEST(OpAdjustContrast_Negative, late_validation_failure_releases_workspace)
{
    constexpr int64_t kHeight = 4 * 65535 + 1;

    NVCVByte *srcAllocation{};
    NVCVByte *dstAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), kHeight));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), kHeight));

    auto wrap = [](NVCVByte *base)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = base;
        buffer.strides[0] = kHeight;
        buffer.strides[1] = 1;
        buffer.strides[2] = 1;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{1, kHeight, 1, 1}, "NHWC"},
            nvcv::TYPE_U8, buffer
        });
    };

    {
        nvcv::Tensor src = wrap(srcAllocation);
        nvcv::Tensor dst = wrap(dstAllocation);

        cudaStream_t stream{};
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
        cvcuda::AdjustContrast op;

        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { op(stream, src, dst, kFactor); }));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    {
        nvcv::Tensor src = wrap(srcAllocation);
        nvcv::Tensor dst = wrap(dstAllocation);

        cudaStream_t stream{};
        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cvcuda::AdjustContrast op;

        ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        EXPECT_EQ(NVCV_ERROR_OVERFLOW, nvcv::ProtectCall([&] { op(stream, src, dst, kFactor); }));

        cudaGraph_t graph{};
        ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
        ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaFree(srcAllocation));
    ASSERT_EQ(cudaSuccess, cudaFree(dstAllocation));
}

TEST(OpAdjustContrast_Negative, rejects_zero_extent_varshape_image)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Image              srcImage({0, 1}, nvcv::FMT_RGB8);
    nvcv::Image              dstImage({0, 1}, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { invokeFactor(kFactor)(stream, src, dst); }));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast_Negative, rejects_subsampled_varshape)
{
    const nvcv::ImageFormat fmt = nvcv::ImageFormat::FromFourCC(FourCC("I420"), nvcv::CSPEC_BT601, nvcv::MemLayout::PL);
    ASSERT_EQ(3, fmt.numPlanes());
    ASSERT_EQ(nvcv::ChromaSubsampling::CSS_420, fmt.chromaSubsampling());
    test::planar::ExpectVarShapeRejected({fmt}, {fmt}, invokeFactor(kFactor));
}

TEST(OpAdjustContrast_Negative, rejects_non_rgb_channel_permutations)
{
    const std::array<nvcv::ImageFormat, 2> invalidFormats{
        nvcv::ImageFormat{nvcv::ColorModel::RGB, nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                          nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_ZYW1, nvcv::Packing::X8_Y8_Z8},
        nvcv::ImageFormat{nvcv::ColorModel::RGB, nvcv::CSPEC_UNDEFINED, nvcv::MemLayout::PITCH_LINEAR,
                          nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYXZ, nvcv::Packing::X8_Y8_Z8}
    };

    for (size_t i = 0; i < invalidFormats.size(); ++i)
    {
        SCOPED_TRACE(i);
        test::planar::ExpectVarShapeRejected({invalidFormats[i]}, {invalidFormats[i]}, invokeFactor(kFactor));
    }
}

TEST(OpAdjustContrast, varshape_bgr_uses_logical_rgb_order)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Image srcImage({1, 1}, nvcv::FMT_BGR8);
    nvcv::Image dstImage({1, 1}, nvcv::FMT_BGR8);
    auto        srcData = srcImage.exportData<nvcv::ImageDataStridedCuda>();
    auto        dstData = dstImage.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(srcData);
    ASSERT_TRUE(dstData);

    const std::array<uint8_t, 3> blue{255, 0, 0};
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->plane(0).basePtr, blue.data(), blue.size(), cudaMemcpyHostToDevice));

    nvcv::ImageBatchVarShape src(1);
    nvcv::ImageBatchVarShape dst(1);
    src.pushBack(srcImage);
    dst.pushBack(dstImage);
    ASSERT_NO_THROW(invokeFactor(0.0)(stream, src, dst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::array<uint8_t, 3> got{};
    ASSERT_EQ(cudaSuccess, cudaMemcpy(got.data(), dstData->plane(0).basePtr, got.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ((std::array<uint8_t, 3>{29, 29, 29}), got);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast, legacy_c_submits_use_nearest_even_at_exact_ties)
{
    const std::array<uint8_t, 2> input{1, 3};
    const std::array<uint8_t, 2> nearestExpected{2, 2};
    const std::array<uint8_t, 2> truncatedExpected{1, 2};
    cvcuda::AdjustContrast       op;

    nvcv::Tensor tensorIn      = nvcv::util::CreateTensor(1, 2, 1, nvcv::FMT_U8);
    nvcv::Tensor tensorOut     = nvcv::util::CreateTensor(1, 2, 1, nvcv::FMT_U8);
    auto         tensorInData  = tensorIn.exportData<nvcv::TensorDataStridedCuda>();
    auto         tensorOutData = tensorOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(tensorInData && tensorOutData);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(tensorInData->basePtr(), input.data(), input.size(), cudaMemcpyHostToDevice));

    ASSERT_EQ(NVCV_SUCCESS,
              cvcudaAdjustContrastSubmit(op.handle(), nullptr, tensorIn.handle(), tensorOut.handle(), 0.5));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    std::array<uint8_t, 2> actual{};
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tensorOutData->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestExpected, actual);

    ASSERT_EQ(NVCV_SUCCESS, cvcudaAdjustContrastRoundSubmit(op.handle(), nullptr, tensorIn.handle(), tensorOut.handle(),
                                                            0.5, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tensorOutData->basePtr(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(truncatedExpected, actual);

    nvcv::Image imageIn({2, 1}, nvcv::FMT_U8);
    nvcv::Image imageOut({2, 1}, nvcv::FMT_U8);
    auto        imageInData  = imageIn.exportData<nvcv::ImageDataStridedCuda>();
    auto        imageOutData = imageOut.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_TRUE(imageInData && imageOutData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(imageInData->plane(0).basePtr, input.data(), input.size(), cudaMemcpyHostToDevice));
    nvcv::ImageBatchVarShape batchIn(1);
    nvcv::ImageBatchVarShape batchOut(1);
    batchIn.pushBack(imageIn);
    batchOut.pushBack(imageOut);

    ASSERT_EQ(NVCV_SUCCESS,
              cvcudaAdjustContrastVarShapeSubmit(op.handle(), nullptr, batchIn.handle(), batchOut.handle(), 0.5));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), imageOutData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(nearestExpected, actual);

    ASSERT_EQ(NVCV_SUCCESS, cvcudaAdjustContrastVarShapeRoundSubmit(op.handle(), nullptr, batchIn.handle(),
                                                                    batchOut.handle(), 0.5, NVCV_ROUND_TRUNCATE));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(actual.data(), imageOutData->plane(0).basePtr, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(truncatedExpected, actual);
}

TEST(OpAdjustContrast, round_mode_selects_integer_conversion)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width    = 17;
    constexpr int height   = 11;
    constexpr int channels = 3;
    const size_t  count    = static_cast<size_t>(width) * height * channels;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGB8);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         outData   = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    std::vector<uint8_t> input(count);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<uint8_t>((i * 73 + 19) & 0xff);
    }
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<uint8_t>(*inData, input, 0));

    const auto nearestGold  = AdjustContrastGoldImage<uint8_t>(input, width, height, channels, kFactor, false);
    const auto truncateGold = AdjustContrastGoldImage<uint8_t>(input, width, height, channels, kFactor, true);
    // The factor has to actually separate the two conversions, or the test proves nothing.
    ASSERT_NE(nearestGold, truncateGold);

    const std::array cases{
        std::pair{ NVCV_ROUND_NEAREST,  std::cref(nearestGold)},
        std::pair{NVCV_ROUND_TRUNCATE, std::cref(truncateGold)}
    };
    for (const auto &[mode, gold] : cases)
    {
        SCOPED_TRACE(static_cast<int>(mode));
        ASSERT_NO_THROW(invokeFactorRound(kFactor, mode)(stream, inTensor, outTensor));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<uint8_t> got;
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*outData, 0, got);
        EXPECT_EQ(gold.get(), got);
    }

    ASSERT_NO_THROW(invokeFactor(kFactor)(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    std::vector<uint8_t> gotDefault;
    nvcv::util::GetImageVectorFromTensor<uint8_t>(*outData, 0, gotDefault);
    EXPECT_EQ(nearestGold, gotDefault);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast, round_mode_does_not_change_float_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width  = 2;
    constexpr int height = 2;

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor nearestT  = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    nvcv::Tensor truncateT = nvcv::util::CreateTensor(1, width, height, nvcv::FMT_RGBf32);
    auto         inData    = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         nearData  = nearestT.exportData<nvcv::TensorDataStridedCuda>();
    auto         truncData = truncateT.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(nearData, nullptr);
    ASSERT_NE(truncData, nullptr);

    std::vector<float> input{0.f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.f, 0.3f, 0.6f, 0.2f, 0.4f, 0.8f};
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<float>(*inData, input, 0));

    ASSERT_NO_THROW(invokeFactorRound(kFactor, NVCV_ROUND_NEAREST)(stream, inTensor, nearestT));
    ASSERT_NO_THROW(invokeFactorRound(kFactor, NVCV_ROUND_TRUNCATE)(stream, inTensor, truncateT));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<float> nearestGot;
    std::vector<float> truncateGot;
    nvcv::util::GetImageVectorFromTensor<float>(*nearData, 0, nearestGot);
    nvcv::util::GetImageVectorFromTensor<float>(*truncData, 0, truncateGot);
    EXPECT_EQ(nearestGot, truncateGot);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpAdjustContrast, explicit_round_modes_planar_and_varshape)
{
    for (auto mode : {NVCV_ROUND_NEAREST, NVCV_ROUND_TRUNCATE})
    {
        const auto invoke = [mode](cudaStream_t stream, const auto &src, const auto &dst, nvcv::ImageFormat)
        {
            cvcuda::AdjustContrast op;
            op(stream, src, dst, 0.5, mode);
        };
        // Equal RGB channels 1 and 4 floor to BT.601 grayscale values 0 and 3.
        const auto input = [](int, int width, int height, int channels)
        {
            std::vector<uint8_t> values(width * height * channels);
            for (size_t i = 0; i < values.size(); ++i) values[i] = (i / channels) % 2 ? 4 : 1;
            return values;
        };
        // Gray values are 0 and 3, mean 1.5; blended outputs are 1.25 and 2.75.
        const auto compare = [mode](const auto &interleaved, const auto &planar)
        {
            constexpr int channels = 3;
            EXPECT_EQ(interleaved, planar);
            for (size_t i = 0; i < interleaved.size(); ++i)
            {
                EXPECT_EQ(interleaved[i], (i / channels) % 2 ? (mode == NVCV_ROUND_NEAREST ? 3 : 2) : 1);
            }
        };
        test::planar::RunTensorParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 4, 2, 4, 2, 2, invoke, input, compare);
        test::planar::RunVarShapeParity(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 4, 2, 4, 2, 2, invoke, input, compare);
    }
}

TEST(OpAdjustContrast_Negative, rejects_non_finite_or_unrepresentable_factor)
{
    for (double factor : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::max()})
    {
        SCOPED_TRACE(factor);
        ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeFactor(factor));
        test::planar::ExpectVarShapeRejected({nvcv::FMT_RGB8}, {nvcv::FMT_RGB8}, invokeFactor(factor));
    }
}

TEST(OpAdjustContrast_Negative, rejects_invalid_round_mode)
{
    cvcuda::AdjustContrast   op;
    nvcv::Tensor             src = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_RGB8);
    nvcv::Tensor             dst = nvcv::util::CreateTensor(1, 4, 1, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape srcBatch(1);
    nvcv::ImageBatchVarShape dstBatch(1);
    srcBatch.pushBack(nvcv::Image({4, 1}, nvcv::FMT_RGB8));
    dstBatch.pushBack(nvcv::Image({4, 1}, nvcv::FMT_RGB8));

#ifndef ENABLE_SANITIZER
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              TestCAPIEnumAdjustContrastRoundSubmit(op.handle(), nullptr, src.handle(), dst.handle(), kFactor));
#endif

    for (auto mode : {NVCV_ROUND_NEAREST_WRAP, NVCV_ROUND_TRUNCATE_WRAP})
    {
        ew::ExpectRejected(nvcv::FMT_RGB8, nvcv::FMT_RGB8, invokeFactorRound(kFactor, mode));
        test::planar::ExpectVarShapeRejected({nvcv::FMT_RGB8}, {nvcv::FMT_RGB8}, invokeFactorRound(kFactor, mode));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaAdjustContrastRoundSubmit(op.handle(), nullptr, src.handle(), dst.handle(), kFactor, mode));
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  cvcudaAdjustContrastVarShapeRoundSubmit(op.handle(), nullptr, srcBatch.handle(), dstBatch.handle(),
                                                          kFactor, mode));
    }
}
