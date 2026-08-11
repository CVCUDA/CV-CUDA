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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdjustContrast.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

namespace test = nvcv::test;
namespace ew   = nvcv::test::elementwise;

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
std::vector<T> AdjustContrastGoldImage(const std::vector<T> &in, int width, int height, int channels, double factor)
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
        out[i]      = RoundSaturateGold<T>(pixel);
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

// Planar == interleaved parity ------------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrastPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 2,   nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
    {257,  33, 2, nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32},
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

// Negative tests: the complement of the support matrix must be rejected with
// NVCV_ERROR_INVALID_ARGUMENT (ew::ExpectRejected asserts this via nvcv::ProtectCall).
// clang-format off
NVCV_TEST_SUITE_P(OpAdjustContrast_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_F16,    nvcv::FMT_F16   }, // unsupported dtype (16-bit float)
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
