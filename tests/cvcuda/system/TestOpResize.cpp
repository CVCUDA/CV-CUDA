/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"
#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"
#include "ResizeUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpResize.hpp>
#include <cvcuda/OpRotate.hpp> // for the NEAREST_FLOOR rejection check
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>
#include <utility>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace t    = ::testing;

TEST(HalfTestUtils, fill_random_half_bytes_validates_buffer_size)
{
    std::default_random_engine randEng; // NOSONAR: deterministic test data, not security-sensitive.
    std::vector<uint8_t>       even(4);
    std::vector<uint8_t>       odd(3);

    EXPECT_NO_THROW(test::FillRandomHalfBytes(even, randEng));
    EXPECT_EQ(even.size(), 4);
    EXPECT_THROW(test::FillRandomHalfBytes(odd, randEng), std::invalid_argument);
    EXPECT_EQ(odd.size(), 3);
}

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(static_cast<double>(size) * scale);
}

TEST(ResizeUtilsF16, preserves_negative_interpolated_output)
{
    const std::vector<uint8_t> src = test::FloatToHalfBytes({-4.f, -3.f, -2.f, -1.f});
    std::vector<uint8_t>       resized(3 * 3 * sizeof(__half));
    std::vector<uint8_t>       cropped(resized.size());
    std::vector<uint8_t>       area(sizeof(__half));

    test::Resize(resized, 3 * sizeof(__half), {3, 3}, src, 2 * sizeof(__half), {2, 2}, nvcv::FMT_F16,
                 NVCV_INTERP_LINEAR, false);
    test::Resize(area, sizeof(__half), {1, 1}, src, 2 * sizeof(__half), {2, 2}, nvcv::FMT_F16, NVCV_INTERP_AREA, false);
    test::ResizedCrop(cropped, 3 * sizeof(__half), {3, 3}, src, 2 * sizeof(__half), {2, 2}, 0, 0, 2, 2, nvcv::FMT_F16,
                      NVCV_INTERP_LINEAR);

    EXPECT_THAT(test::HalfBytesToFloat(resized), t::Each(t::Lt(0.f)));
    EXPECT_THAT(test::HalfBytesToFloat(area), t::Each(t::Lt(0.f)));
    EXPECT_THAT(test::HalfBytesToFloat(cropped), t::Each(t::Lt(0.f)));
}

// clang-format off

#define NVCV_IMAGE_FORMAT_4U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8)
#define NVCV_IMAGE_FORMAT_3U16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_4U16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_3S16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, XYZ1, ASSOCIATED, X16_Y16_Z16)
#define NVCV_IMAGE_FORMAT_4S16 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, XYZW, ASSOCIATED, X16_Y16_Z16_W16)
#define NVCV_IMAGE_FORMAT_3F32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, XYZ1, ASSOCIATED, X32_Y32_Z32)
#define NVCV_IMAGE_FORMAT_4F32 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, FLOAT, XYZW, ASSOCIATED, X32_Y32_Z32_W32)

NVCV_TEST_SUITE_P(OpResize, test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,       interpolation, numberImages,    imageFormat
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGBA8},
    {        113,       12,       12,        36, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGBA8},
    // NEAREST_FLOOR drops the half-pixel offset; downscale, upscale, and mixed axes against the
    // host gold reference, which applies the same rule.
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST_FLOOR,     1,     nvcv::FMT_RGBA8},
    {        113,       12,       12,        36, NVCV_INTERP_NEAREST_FLOOR,     1,     nvcv::FMT_RGBA8},
    {        320,      320,      299,       299, NVCV_INTERP_NEAREST_FLOOR,     2,     nvcv::FMT_RGBA8},
    {         20,       20,       23,        23, NVCV_INTERP_NEAREST_FLOOR,     1,     nvcv::FMT_RGBA8},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST_FLOOR,     1,     nvcv::FMT_U8},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST_FLOOR,     1,     nvcv::FMT_RGB8},
    {        421,      148,      223,       124, NVCV_INTERP_NEAREST,           2,     nvcv::FMT_RGBA8},
    {        313,      212,      412,       336, NVCV_INTERP_NEAREST,           3,     nvcv::FMT_RGBA8},
    {         20,       20,       23,        23, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGBA8},
    {         42,       40,       21,        20,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {         21,       21,       42,        42,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {        420,      420,      210,       210,  NVCV_INTERP_LINEAR,           4,     nvcv::FMT_RGBA8},
    {        210,      210,      420,       420,  NVCV_INTERP_LINEAR,           5,     nvcv::FMT_RGBA8},
    {         37,       40,       19,        20,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {         35,       35,       22,        22,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {         42,       40,       21,        20,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGBA8},
    {         21,       21,       42,        42,   NVCV_INTERP_CUBIC,           6,     nvcv::FMT_RGBA8},
    {        420,      420,      420,       420,   NVCV_INTERP_CUBIC,           2,     nvcv::FMT_RGBA8},
    {        420,      420,      420,       420,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGBA8},
    {        420,      420,       40,        42,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGBA8},
    {       1920,     1080,      640,       320,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGBA8},
    {       1920,     1080,      640,       320,   NVCV_INTERP_CUBIC,           2,     nvcv::FMT_RGBA8},
    {         44,       40,       22,        20,    NVCV_INTERP_AREA,           2,     nvcv::FMT_RGBA8},
    {         30,       30,       20,        20,    NVCV_INTERP_AREA,           2,     nvcv::FMT_RGBA8},
    {         30,       30,       60,        60,    NVCV_INTERP_AREA,           4,     nvcv::FMT_RGBA8},
    {         60,       60,       20,        20,    NVCV_INTERP_AREA,           4,     nvcv::FMT_RGBA8},
    {       1080,     1920,      720,      1280,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {        720,     1280,      480,       854,  NVCV_INTERP_CUBIC,            1,     nvcv::FMT_RGBA8},
    {       1440,     2560,     1080,      1920,  NVCV_INTERP_AREA,             1,     nvcv::FMT_RGBA8},
    {       2160,     3840,     1080,      1920,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {       1080,     1920,      540,       960,  NVCV_INTERP_CUBIC,            1,     nvcv::FMT_RGBA8},
    {        720,     1280,      360,       640,  NVCV_INTERP_AREA,             1,     nvcv::FMT_RGBA8},
    {       2160,     3840,     1440,      2560,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBA8},
    {       1080,     1920,      360,       640,  NVCV_INTERP_CUBIC,            1,     nvcv::FMT_RGBA8},
    {       1440,     2560,      720,      1280,  NVCV_INTERP_AREA,             1,     nvcv::FMT_RGBA8},
    {         96,       72,       48,        36,  NVCV_INTERP_LINEAR,           2,     nvcv::FMT_U8},
    {         48,       36,       96,        72,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGB8},
    {         48,       36,       96,        72,  NVCV_INTERP_LINEAR,           2,     nvcv::FMT_U8},
    {         45,       30,       90,        60,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_U8},
    {         45,       30,       90,        60,  NVCV_INTERP_LINEAR,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         96,       72,       48,        36,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_U8},
    {         48,       36,       96,        72,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGB8},
    {         48,       36,       96,        72,   NVCV_INTERP_CUBIC,           2,     nvcv::FMT_U8},
    {         45,       30,       90,        60,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_U8},
    {         45,       30,       90,        60,   NVCV_INTERP_CUBIC,           2,     nvcv::FMT_RGB8},
    {         45,       30,       90,        60,   NVCV_INTERP_CUBIC,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {        128,       96,       32,        24,    NVCV_INTERP_AREA,           1,     nvcv::FMT_U8},
    {        128,       96,       32,        24,    NVCV_INTERP_AREA,           2,     nvcv::FMT_RGB8},
    {         90,       60,       45,        30,    NVCV_INTERP_AREA,           1,     nvcv::FMT_U8},
    // 8/7 rounds the final fractional endpoint one ULP past the source extent.
    {          8,        8,        7,         7,    NVCV_INTERP_AREA,           1,     nvcv::FMT_U8},
    {         90,       60,       45,        30,    NVCV_INTERP_AREA,           1,     nvcv::FMT_RGB8},
    {         90,       60,       45,        30,    NVCV_INTERP_AREA,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    // Byte non-integer UPSCALE. Every byte LINEAR/CUBIC upscale above is exactly 2x and therefore
    // diverts to the phase-fused LinearResizeExpand2x/CubicResizeExpand2x kernels; these rows are
    // the only host-gold coverage of the general byte upscale paths (LinearResize<true> and
    // CubicResizeUpscaleRows). The 54x30->240x135 scale mirrors the benchmarked
    // 864x480 -> 3840x2160 ratios (0.225 x, 0.2222 y). Destination widths 123 and 61 are not a
    // multiple of NIX (16 for U8, 4 for RGB8/4U8), so they also exercise the scalar tail branch.
    {         54,       30,      240,       135,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_U8},
    {         54,       30,      240,       135,  NVCV_INTERP_LINEAR,           2,     nvcv::FMT_RGB8},
    {         54,       30,      240,       135,  NVCV_INTERP_LINEAR,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         37,       25,      123,        71,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_U8},
    {         37,       25,      123,        71,  NVCV_INTERP_LINEAR,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         40,       30,       61,        47,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGB8},
    {         54,       30,      240,       135,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_U8},
    {         54,       30,      240,       135,   NVCV_INTERP_CUBIC,           2,     nvcv::FMT_RGB8},
    {         54,       30,      240,       135,   NVCV_INTERP_CUBIC,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         37,       25,      123,        71,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_U8},
    {         37,       25,      123,        71,   NVCV_INTERP_CUBIC,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         40,       30,       61,        47,   NVCV_INTERP_CUBIC,           1,     nvcv::FMT_RGB8},
    // Byte fractional DOWNSCALE with scale < 2, which selects LinearResize<true> (INTERSECT). The
    // existing rows that reach it are all 4-channel; these add the uchar1 and uchar3 element types.
    {         90,       60,       60,        40,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_U8},
    {         90,       60,       60,        40,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGB8},
    // (float LINEAR/CUBIC/AREA are NOT host-gold tested here: this test compares output bytes with a
    // +/-1 tolerance, which is only meaningful for integer outputs; interpolated float is covered
    // bit-exactly by the OpResizePlanar parity test instead.)
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_U8},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGB8},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_U16},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_3U16}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U16}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_S16},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_3S16}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4S16}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_F32},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_3F32}},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4F32}},
    // F16 NEAREST is pure data movement (compared bit-exactly); F16 LINEAR accumulates in FP32
    // and rounds to half once on store, so it CAN be host-gold tested against the FP32 reference
    // within half-ULP bounds (see the F16 comparison branch below). Dims reuse existing integer
    // rows, so tap selection is already validated by the byte suites. CUBIC/AREA F16 are covered
    // bit-exactly by the OpResizePlanar parity rows, mirroring the float treatment above.
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_F16},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGBf16},
    {         42,       48,       23,        24, NVCV_INTERP_NEAREST,           1,     nvcv::FMT_RGBAf16},
    {         42,       40,       21,        20,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_F16},
    {         45,       30,       90,        60,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_F16},
    {         48,       36,       96,        72,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBf16},
    {         37,       40,       19,        20,  NVCV_INTERP_LINEAR,           1,     nvcv::FMT_RGBAf16},
});

#undef NVCV_IMAGE_FORMAT_4U8
#undef NVCV_IMAGE_FORMAT_3U16
#undef NVCV_IMAGE_FORMAT_4U16
#undef NVCV_IMAGE_FORMAT_3S16
#undef NVCV_IMAGE_FORMAT_4S16
#undef NVCV_IMAGE_FORMAT_3F32
#undef NVCV_IMAGE_FORMAT_4F32

// clang-format oon

TEST_P(OpResize, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    const nvcv::ImageFormat fmt = GetParamValue<6>();

    // Generate input
    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numberOfImages, srcWidth, srcHeight, fmt);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecRowStride);
        if (test::IsF16Format(fmt))
        {
            test::FillRandomHalfBytes(srcVec[i], randEng);
        }
        else
        {
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numberOfImages, dstWidth, dstHeight, fmt);

    cvcuda::Resize resizeOp;
    EXPECT_NO_THROW(resizeOp(stream, imgSrc, imgDst, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);

        // Generate gold result
        test::Resize(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride, {srcWidth, srcHeight},
                     fmt, interpolation, false);

        if (test::IsF16Format(fmt))
        {
            test::ExpectF16InterpOutput(goldVec, testVec, interpolation);
            continue;
        }

        std::vector<int> absDiff(testVec.size());
        for (size_t idx = 0; idx < absDiff.size(); ++idx)
        {
            absDiff[idx] = abs(static_cast<int>(goldVec[idx]) - static_cast<int>(testVec[idx]));
        }

        int maxAbsDiff = 1;

        EXPECT_THAT(absDiff, t::Each(t::Le(maxAbsDiff)));
    }
}

TEST_P(OpResize, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();
    int dstWidthBase  = GetParamValue<2>();
    int dstHeightBase = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    const nvcv::ImageFormat fmt = GetParamValue<6>();

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

    std::uniform_int_distribution rndDstWidth(ScaledSize(dstWidthBase, 0.8), ScaledSize(dstWidthBase, 1.1));
    std::uniform_int_distribution rndDstHeight(ScaledSize(dstHeightBase, 0.8), ScaledSize(dstHeightBase, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    // The size of the first image is fixed: to cover area fast code path
    imgSrc.emplace_back(nvcv::Size2D{srcWidthBase, srcHeightBase}, fmt);
    imgDst.emplace_back(nvcv::Size2D{dstWidthBase, dstHeightBase}, fmt);
    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmt);
        imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowStride = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowStride);
        if (test::IsF16Format(fmt))
        {
            test::FillRandomHalfBytes(srcVec[i], randEng);
        }
        else
        {
            std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::Resize resizeOp;
    EXPECT_NO_THROW(resizeOp(stream, batchSrc, batchDst, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowStride = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);

        // Generate gold result
        test::Resize(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride[i], {srcWidth, srcHeight},
                     fmt, interpolation, true);

        if (test::IsF16Format(fmt))
        {
            test::ExpectF16InterpOutput(goldVec, testVec, interpolation);
            continue;
        }

        // maximum absolute error
        std::vector<int> absDiff(testVec.size());
        for (size_t idx = 0; idx < absDiff.size(); ++idx)
        {
            absDiff[idx] = abs(static_cast<int>(goldVec[idx]) - static_cast<int>(testVec[idx]));
        }

        int maxAbsDiff = 1;

        EXPECT_THAT(absDiff, t::Each(t::Le(maxAbsDiff)));
    }
}

// =============================================================================
// Cubic boundary interpolation tests
//
// These tests verify that bicubic resize correctly interpolates pixels near
// image boundaries using per-tap replicate-border clamping, instead of zeroing
// the fractional weight (which collapsed to nearest-neighbor at edges).
// =============================================================================

namespace {

double CubicWeight(double A, double t)
{
    if (double at = std::abs(t); at <= 1.0)
        return (A + 2.0) * at * at * at - (A + 3.0) * at * at + 1.0;
    else if (at < 2.0)
        return A * at * at * at - 5.0 * A * at * at + 8.0 * A * at - 4.0 * A;
    return 0.0;
}

template<typename Scalar>
double ReferenceCubicPixelReplicateBorder(const std::vector<Scalar> &src, int srcW, int srcH, int channels, int c,
                                          int ix, int iy, const std::array<double, 4> &wx,
                                          const std::array<double, 4> &wy)
{
    double val = 0.0;
    for (int ky = 0; ky < 4; ky++)
    {
        int sy = std::clamp(iy + ky - 1, 0, srcH - 1);
        for (int kx = 0; kx < 4; kx++)
        {
            int sx = std::clamp(ix + kx - 1, 0, srcW - 1);
            val += src[sy * srcW * channels + sx * channels + c] * wx[kx] * wy[ky];
        }
    }
    return val;
}

template<typename Scalar>
Scalar ConvertCubicReferenceValue(double value)
{
    if constexpr (std::is_same_v<Scalar, uint8_t>)
    {
        value = std::rint(std::max(0.0, std::min(255.0, value)));
    }
    return static_cast<Scalar>(value);
}

// Self-contained CPU reference for bicubic resize with replicate-border clamping.
// Does NOT depend on ResizeUtils (which had the same bug), so the test is
// independently correct even if both GPU kernel and ResizeUtils are broken.
template<typename Scalar>
void ReferenceCubicResizeReplicateBorder(std::vector<Scalar> &dst, int dstW, int dstH,
                                         const std::vector<Scalar> &src, int srcW, int srcH, int channels)
{
    double scaleX = static_cast<double>(srcW) / dstW;
    double scaleY = static_cast<double>(srcH) / dstH;
    const double A = -0.75;

    dst.resize(dstH * dstW * channels);

    for (int dy = 0; dy < dstH; dy++)
    {
        for (int dx = 0; dx < dstW; dx++)
        {
            double srcY = (dy + 0.5) * scaleY - 0.5;
            double srcX = (dx + 0.5) * scaleX - 0.5;
            auto   iy   = static_cast<int>(std::floor(srcY));
            auto   ix   = static_cast<int>(std::floor(srcX));
            double fy   = srcY - iy;
            double fx   = srcX - ix;

            std::array<double, 4> wy = {CubicWeight(A, fy + 1), CubicWeight(A, fy), CubicWeight(A, 1 - fy),
                                        CubicWeight(A, 2 - fy)};
            std::array<double, 4> wx = {CubicWeight(A, fx + 1), CubicWeight(A, fx), CubicWeight(A, 1 - fx),
                                        CubicWeight(A, 2 - fx)};

            for (int c = 0; c < channels; c++)
            {
                double val = ReferenceCubicPixelReplicateBorder(src, srcW, srcH, channels, c, ix, iy, wx, wy);
                dst[dy * dstW * channels + dx * channels + c] = ConvertCubicReferenceValue<Scalar>(val);
            }
        }
    }
}

// Create a gradient image where boundary pixels have distinct, non-uniform values.
// This makes boundary interpolation errors clearly visible: the old kernel would
// zero the fractional weight at edges, producing nearest-neighbor copies instead
// of proper cubic blends, yielding errors of 30-90+ on uint8 gradient images.
void FillGradientImage(std::vector<uint8_t> &img, int w, int h, int channels)
{
    img.resize(h * w * channels);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                int val = static_cast<int>(255.0 * x / std::max(w - 1, 1)) + c * 30 + y * 10 / std::max(h - 1, 1);
                img[y * w * channels + x * channels + c] = static_cast<uint8_t>(std::clamp(val, 0, 255));
            }
        }
    }
}

} // anonymous namespace

// clang-format off
NVCV_TEST_SUITE_P(OpResize_CubicBoundary, test::ValueList<int, int, int, int, int>
{
    //  srcW, srcH, dstW, dstH, channels
    {     8,    8,    5,    5,    3},  // small: every output pixel near boundary
    {     6,    6,   13,   13,    3},  // upscale: boundary taps dominate corners
    {    10,    4,    7,   11,    3},  // asymmetric: left/right edges stressed
    {     4,   10,   11,    7,    3},  // asymmetric: top/bottom edges stressed
    {     8,    8,    5,    5,    1},  // single channel
    {     8,    8,    5,    5,    4},  // RGBA
    {     6,    6,   13,   13,    1},  // upscale, single channel: the general cubic upscale on uchar1
    {     6,    6,   13,   13,    4},  // upscale, RGBA: the general cubic upscale on uchar4
    {     7,    5,   23,   17,    3},  // anisotropic upscale: window slides across owned columns
});

// clang-format on

TEST_P(OpResize_CubicBoundary, tensor_cubic_boundary)
{
    int srcW     = GetParamValue<0>();
    int srcH     = GetParamValue<1>();
    int dstW     = GetParamValue<2>();
    int dstH     = GetParamValue<3>();
    int channels = GetParamValue<4>();

    nvcv::ImageFormat fmt;
    switch (channels)
    {
    case 1:
        fmt = nvcv::FMT_U8;
        break;
    case 3:
        fmt = nvcv::FMT_RGB8;
        break;
    case 4:
        fmt = nvcv::FMT_RGBA8;
        break;
    default:
        FAIL() << "Unsupported channel count";
    }

    // Create gradient source
    std::vector<uint8_t> srcVec;
    FillGradientImage(srcVec, srcW, srcH, channels);

    // Compute expected with self-contained reference
    std::vector<uint8_t> goldVec;
    ReferenceCubicResizeReplicateBorder(goldVec, dstW, dstH, srcVec, srcW, srcH, channels);

    // Upload source to GPU tensor
    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(1, srcW, srcH, fmt);
    {
        auto srcData   = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        ASSERT_TRUE(srcAccess);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(0), srcAccess->rowStride(), srcVec.data(),
                                            srcW * channels, srcW * channels, srcH, cudaMemcpyHostToDevice));
    }

    // Run GPU resize
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(1, dstW, dstH, fmt);
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Resize resizeOp;
    EXPECT_NO_THROW(resizeOp(stream, imgSrc, imgDst, NVCV_INTERP_CUBIC));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Read back result
    std::vector<uint8_t> testVec(dstH * dstW * channels);
    {
        auto dstData   = imgDst.exportData<nvcv::TensorDataStridedCuda>();
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(dstAccess);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstW * channels, dstAccess->sampleData(0),
                                            dstAccess->rowStride(), dstW * channels, dstH, cudaMemcpyDeviceToHost));
    }

    // Compare: with the fix, max abs error should be <=1 (float rounding).
    // Without the fix, boundary pixels would differ by 30-90+.
    int maxDiff = 0;
    for (size_t i = 0; i < testVec.size(); i++)
    {
        int diff = abs(static_cast<int>(testVec[i]) - static_cast<int>(goldVec[i]));
        maxDiff  = std::max(maxDiff, diff);
    }
    EXPECT_LE(maxDiff, 1) << "Tensor cubic resize boundary error: max pixel diff = " << maxDiff << " (src=" << srcW
                          << "x" << srcH << " -> dst=" << dstW << "x" << dstH << ")";
}

TEST_P(OpResize_CubicBoundary, varshape_cubic_boundary)
{
    int srcW     = GetParamValue<0>();
    int srcH     = GetParamValue<1>();
    int dstW     = GetParamValue<2>();
    int dstH     = GetParamValue<3>();
    int channels = GetParamValue<4>();

    nvcv::ImageFormat fmt;
    switch (channels)
    {
    case 1:
        fmt = nvcv::FMT_U8;
        break;
    case 3:
        fmt = nvcv::FMT_RGB8;
        break;
    case 4:
        fmt = nvcv::FMT_RGBA8;
        break;
    default:
        FAIL() << "Unsupported channel count";
    }

    // Create gradient source
    std::vector<uint8_t> srcVec;
    FillGradientImage(srcVec, srcW, srcH, channels);

    // Compute expected with self-contained reference
    std::vector<uint8_t> goldVec;
    ReferenceCubicResizeReplicateBorder(goldVec, dstW, dstH, srcVec, srcW, srcH, channels);

    // Upload to ImageBatchVarShape (batch of 1)
    nvcv::Image imgSrc(nvcv::Size2D{srcW, srcH}, fmt);
    {
        auto srcData = imgSrc.exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec.data(),
                                            srcW * channels, srcW * channels, srcH, cudaMemcpyHostToDevice));
    }

    nvcv::Image imgDst(nvcv::Size2D{dstW, dstH}, fmt);

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc);
    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst);

    // Run GPU var-shape resize
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Resize resizeOp;
    EXPECT_NO_THROW(resizeOp(stream, batchSrc, batchDst, NVCV_INTERP_CUBIC));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Read back result
    std::vector<uint8_t> testVec(dstH * dstW * channels);
    {
        auto dstData = imgDst.exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(dstData);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstW * channels, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstW * channels, dstH, cudaMemcpyDeviceToHost));
    }

    // Compare: with the fix, max abs error should be <=1 (float rounding).
    // Without the fix, boundary pixels would differ by 30-90+.
    int maxDiff = 0;
    for (size_t i = 0; i < testVec.size(); i++)
    {
        int diff = abs(static_cast<int>(testVec[i]) - static_cast<int>(goldVec[i]));
        maxDiff  = std::max(maxDiff, diff);
    }
    EXPECT_LE(maxDiff, 1) << "VarShape cubic resize boundary error: max pixel diff = " << maxDiff << " (src=" << srcW
                          << "x" << srcH << " -> dst=" << dstW << "x" << dstH << ")";
}

TEST(OpResize_Negative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaResizeCreate(nullptr));
}

// clang-format off
NVCV_TEST_SUITE_P(OpResize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, NVCVInterpolationType>{
    {nvcv::FMT_U8, nvcv::FMT_U16, 1, 1, NVCV_INTERP_NEAREST}, // in/out image data type not same
    {nvcv::FMT_U8, nvcv::FMT_RGB8p, 1, 1, NVCV_INTERP_NEAREST}, // in/out image layout not same
    {nvcv::FMT_RGB8p, nvcv::FMT_U8, 1, 1, NVCV_INTERP_NEAREST}, // in/out image layout not same (planar in, interleaved out)
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 1, 2, NVCV_INTERP_NEAREST}, // in/out image num are different
    {nvcv::FMT_U8, nvcv::FMT_RGB8, 1, 1, NVCV_INTERP_NEAREST}, // in/out image channels are different
    {nvcv::FMT_F64, nvcv::FMT_F64, 1, 1, NVCV_INTERP_NEAREST}, // unsupported data type (64-bit float)
    {nvcv::FMT_F16, nvcv::FMT_F16, 1, 1, NVCV_INTERP_HAMMING}, // invalid interpolation (F16 dtype itself is valid)
    {nvcv::FMT_U8, nvcv::FMT_U8, 1, 1, NVCV_INTERP_HAMMING}, // invalid interpolation with supported dtype
});

NVCV_TEST_SUITE_P(OpResizeVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType>
{
    // invalid data format
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, NVCV_INTERP_NEAREST},
    // unsupported data type (64-bit float; F16 is now valid)
    {nvcv::FMT_F64, nvcv::FMT_F64, nvcv::FMT_F64, nvcv::FMT_F64, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_INTERP_HAMMING},
});

// clang-format on

TEST_P(OpResize_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = 42;
    int srcHeight = 48;
    int dstWidth  = 23;
    int dstHeight = 24;

    // The interpolation column must reach the operator, or interpolation negatives silently
    // degrade into (passing) NEAREST submissions.
    NVCVInterpolationType interpolation = GetParamValue<4>();

    const nvcv::ImageFormat inputFmt        = GetParamValue<0>();
    const nvcv::ImageFormat outputFmt       = GetParamValue<1>();
    int                     numInputImages  = GetParamValue<2>();
    int                     numOutputImages = GetParamValue<3>();

    // Generate input
    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numInputImages, srcWidth, srcHeight, inputFmt);

    // Generate test result
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numOutputImages, dstWidth, dstHeight, outputFmt);

    cvcuda::Resize resizeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&resizeOp, &stream, &imgSrc, &imgDst, &interpolation]
                                                             { resizeOp(stream, imgSrc, imgDst, interpolation); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResize_Negative, invalid_layout_and_planar_grid_z_are_rejected)
{
    cvcuda::Resize op;

    nvcv::Tensor invalidLayout(
        {
            {1, 4, 4, 3},
            "ABCD"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor invalidLayoutOut(invalidLayout.shape(), invalidLayout.dtype());
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, invalidLayout, invalidLayoutOut, NVCV_INTERP_NEAREST); }));

    constexpr int64_t samples = 21846; // samples * 3 channels = 65538, above CUDA grid.z.
    nvcv::TensorShape planarShape{
        {samples, 3, 1, 1},
        "NCHW"
    };
    nvcv::Tensor planarSrc(planarShape, nvcv::TYPE_U8);
    nvcv::Tensor planarDst(planarShape, nvcv::TYPE_U8);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, planarSrc, planarDst, NVCV_INTERP_NEAREST); }));
}

TEST_P(OpResizeVarshape_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase   = 42;
    int srcHeightBase  = 48;
    int dstWidthBase   = 23;
    int dstHeightBase  = 24;
    int numberOfImages = 5;

    nvcv::ImageFormat     inFmt         = GetParamValue<0>();
    nvcv::ImageFormat     outFmt        = GetParamValue<1>();
    nvcv::ImageFormat     inFmtExtra    = GetParamValue<2>();
    nvcv::ImageFormat     outFmtExtra   = GetParamValue<3>();
    NVCVInterpolationType interpolation = GetParamValue<4>();

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

    std::uniform_int_distribution rndDstWidth(ScaledSize(dstWidthBase, 0.8), ScaledSize(dstWidthBase, 1.1));
    std::uniform_int_distribution rndDstHeight(ScaledSize(dstHeightBase, 0.8), ScaledSize(dstHeightBase, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numberOfImages - 1; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, inFmt);
        imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, outFmt);
    }
    imgSrc.emplace_back(nvcv::Size2D{srcWidthBase, srcHeightBase}, inFmtExtra);
    imgDst.emplace_back(nvcv::Size2D{dstHeightBase, dstHeightBase}, outFmtExtra);

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // run operator
    cvcuda::Resize resizeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&resizeOp, &stream, &batchSrc, &batchDst, &interpolation]
                                                             { resizeOp(stream, batchSrc, batchDst, interpolation); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResize_Negative, linear_interp_requires_2x2_source)
{
    // Regression: LinearResize reads iSrcCoord+1 unconditionally; a 1x1 source causes GPU OOB.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src1x1 = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    nvcv::Tensor dst    = nvcv::util::CreateTensor(1, 16, 16, nvcv::FMT_U8);

    cvcuda::Resize resizeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&resizeOp, &stream, &src1x1, &dst]
                                                             { resizeOp(stream, src1x1, dst, NVCV_INTERP_LINEAR); }));

    // Width==1 with height>1 is also rejected.
    nvcv::Tensor src1xN = nvcv::util::CreateTensor(1, 1, 8, nvcv::FMT_U8);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&resizeOp, &stream, &src1xN, &dst]
                                                             { resizeOp(stream, src1xN, dst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResizeVarshape_Negative, linear_interp_requires_2x2_source)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batchSrc(1);

    nvcv::ImageBatchVarShape batchDst(1);
    batchSrc.pushBack(nvcv::Image({1, 1}, nvcv::FMT_U8));
    batchDst.pushBack(nvcv::Image({16, 16}, nvcv::FMT_U8));

    cvcuda::Resize resizeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&resizeOp, &stream, &batchSrc, &batchDst]
                                { resizeOp(stream, batchSrc, batchDst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Resize processes channels independently, so a planar input is resized plane-by-plane and must
// produce exactly the same pixels as the interleaved path. These tests feed identical data in both
// layouts through cvcuda::Resize and require the (re-interleaved) planar output to match the
// interleaved output bit-for-bit, which holds for every dtype and interpolation mode.
// =============================================================================

namespace {

// Resize identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
// The shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only
// bind the Resize call.
void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [interp](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::Resize op;
            EXPECT_NO_THROW(op(s, src, dst, interp));
        });
}

// Var-shape counterpart of RunPlanarParityTensorCase.
void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
                                    [interp](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                             const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::Resize op;
                                        EXPECT_NO_THROW(op(s, src, dst, interp));
                                    });
}

} // namespace

TEST(OpResizeRegression, wide_float_fractional_area_matches_planar_reference)
{
    constexpr int srcWidth  = 97;
    constexpr int srcHeight = 73;
    constexpr int dstWidth  = 37;
    constexpr int dstHeight = 29;

    for (const auto &[planarFormat, interleavedFormat] : std::array{
             std::pair{ nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
             std::pair{nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}
    })
    {
        RunPlanarParityTensorCase(planarFormat, interleavedFormat, srcWidth, srcHeight, dstWidth, dstHeight,
                                  NVCV_INTERP_AREA, 1);
        RunPlanarParityVarShapeCase(planarFormat, interleavedFormat, srcWidth, srcHeight, dstWidth, dstHeight,
                                    NVCV_INTERP_AREA, 1);
    }
}

TEST(OpResizeRegression, rgb8_planar_interleaved_boundary_kernel_paths)
{
    // Keep one parity test while selecting shapes for the generic LINEAR tail, exact-2x LINEAR
    // expansion, generic CUBIC tail, and aligned exact-2x AREA vector-load paths.
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 21, 21, 42, 42, NVCV_INTERP_LINEAR, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 42, 24, 21, 12, NVCV_INTERP_LINEAR, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 32, 24, 21, 15, NVCV_INTERP_CUBIC, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 32, 24, 16, 12, NVCV_INTERP_AREA, 1);
}

TEST(OpResizeRegression, cubic_expand2x_and_fractional_area_edge_branches)
{
    // Non-multiple launch dimensions exercise both the valid cubic expand body and its
    // over-provisioned-thread guard for the planar and interleaved RGB8 paths.
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 17, 15, 34, 30, NVCV_INTERP_CUBIC, 1);

    // A non-integer downscale with a final source-aligned row exercises both fractional AREA
    // endpoint cases while retaining exact planar/interleaved parity.
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 97, 73, 37, 29, NVCV_INTERP_AREA, 1);
}

TEST(OpResizeRegression, cubic_expand2x_scalar_planes_interior_border_and_tail)
{
    for (const auto &[planarFormat, interleavedFormat] : std::array{
             std::pair{   nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
             std::pair{  nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
             std::pair{ nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
             std::pair{nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}
    })
    {
        RunPlanarParityTensorCase(planarFormat, interleavedFormat, 521, 9, 1042, 18, NVCV_INTERP_CUBIC, 2);
    }
}

namespace {

template<typename Scalar>
void RunMisalignedPlanarCubicResizeCase(cudaStream_t stream, int numImages)
{
    constexpr int     srcWidth       = 521;
    constexpr int     srcHeight      = 9;
    constexpr int     dstWidth       = 2 * srcWidth;
    constexpr int     dstHeight      = 2 * srcHeight;
    constexpr int     channels       = 3;
    constexpr size_t  guardBytes     = 64;
    constexpr size_t  baseOffset     = guardBytes + sizeof(Scalar);
    constexpr int64_t srcRowStride   = (srcWidth + 5) * sizeof(Scalar);
    constexpr int64_t dstRowStride   = (dstWidth + 3) * sizeof(Scalar);
    constexpr int64_t srcPlaneStride = srcRowStride * srcHeight + 32;
    constexpr int64_t dstPlaneStride = dstRowStride * dstHeight + 32;
    const auto        dtype          = std::is_same_v<Scalar, float> ? nvcv::TYPE_F32 : nvcv::TYPE_U8;

    static_assert(srcRowStride % 16 != 0 && dstRowStride % 16 != 0);

    SCOPED_TRACE(t::Message() << "elementBytes=" << sizeof(Scalar) << ", numImages=" << numImages);
    const int            planes = numImages * channels;
    std::vector<uint8_t> srcHost(baseOffset + planes * srcPlaneStride + guardBytes, 0xA5);
    std::vector<uint8_t> expected(baseOffset + planes * dstPlaneStride + guardBytes, 0x5A);
    std::vector<uint8_t> actual(expected.size(), 0x5A);

    for (int plane = 0; plane < planes; ++plane)
    {
        std::vector<Scalar> input(srcWidth * srcHeight);
        for (int row = 0; row < srcHeight; ++row)
            for (int column = 0; column < srcWidth; ++column)
            {
                const int value                = (column * 13 + row * 7 + plane * 11) % 32;
                input[row * srcWidth + column] = static_cast<Scalar>(value - (std::is_same_v<Scalar, float> ? 16 : 0));
            }

        std::vector<Scalar> gold;
        ReferenceCubicResizeReplicateBorder(gold, dstWidth, dstHeight, input, srcWidth, srcHeight, 1);
        for (int row = 0; row < srcHeight; ++row)
            std::memcpy(srcHost.data() + baseOffset + plane * srcPlaneStride + row * srcRowStride,
                        input.data() + row * srcWidth, srcWidth * sizeof(Scalar));
        for (int row = 0; row < dstHeight; ++row)
            std::memcpy(expected.data() + baseOffset + plane * dstPlaneStride + row * dstRowStride,
                        gold.data() + row * dstWidth, dstWidth * sizeof(Scalar));
    }

    auto wrap = [&](NVCVByte *allocation, int width, int height, int64_t rowStride, int64_t planeStride)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr       = allocation + baseOffset;
        const int channelDim = numImages == 1 ? 0 : 1;
        if (numImages > 1)
            buffer.strides[0] = channels * planeStride;
        buffer.strides[channelDim]     = planeStride;
        buffer.strides[channelDim + 1] = rowStride;
        buffer.strides[channelDim + 2] = sizeof(Scalar);
        const auto shape = numImages == 1 ? nvcv::TensorShape{{channels, height, width}, "CHW"}
                                         : nvcv::TensorShape{{numImages, channels, height, width}, "NCHW"};
        auto cleanup = [allocation](const nvcv::TensorData &)
        {
            EXPECT_EQ(cudaSuccess, cudaFree(allocation));
        };
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, dtype, buffer},
                                    nvcv::TensorDataCleanupCallback{cleanup});
    };

    NVCVByte *srcAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), srcHost.size()));
    nvcv::Tensor src           = wrap(srcAllocation, srcWidth, srcHeight, srcRowStride, srcPlaneStride);
    NVCVByte    *dstAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), actual.size()));
    nvcv::Tensor dst = wrap(dstAllocation, dstWidth, dstHeight, dstRowStride, dstPlaneStride);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(srcAllocation + baseOffset) % alignof(Scalar), 0);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(dstAllocation + baseOffset) % alignof(Scalar), 0);
    ASSERT_NE(reinterpret_cast<uintptr_t>(srcAllocation + baseOffset) % 16, 0);
    ASSERT_NE(reinterpret_cast<uintptr_t>(dstAllocation + baseOffset) % 16, 0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcHost.data(), srcHost.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstAllocation, actual.data(), actual.size(), cudaMemcpyHostToDevice));

    cvcuda::Resize resizeOp;
    ASSERT_NO_THROW(resizeOp(stream, src, dst, NVCV_INTERP_CUBIC));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstAllocation, actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);

    std::vector<uint8_t> sourceAfter(srcHost.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(sourceAfter.data(), srcAllocation, sourceAfter.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(srcHost, sourceAfter);
}

} // namespace

TEST(OpResizeRegression, cubic_expand2x_misaligned_planar_matches_cpu_and_preserves_guards)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    RunMisalignedPlanarCubicResizeCase<uint8_t>(stream, 1);
    RunMisalignedPlanarCubicResizeCase<uint8_t>(stream, 2);
    RunMisalignedPlanarCubicResizeCase<float>(stream, 1);
    RunMisalignedPlanarCubicResizeCase<float>(stream, 2);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResizeRegression, u8_tensor_covers_linear_cubic_and_integer_area_kernel_edges)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Resize op;
    auto runCase = [&](int srcWidth, int srcHeight, int dstWidth, int dstHeight, NVCVInterpolationType interpolation)
    {
        nvcv::Tensor src = nvcv::util::CreateTensor(1, srcWidth, srcHeight, nvcv::FMT_U8);
        nvcv::Tensor dst = nvcv::util::CreateTensor(1, dstWidth, dstHeight, nvcv::FMT_U8);

        auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, srcData);

        std::vector<uint8_t> input(static_cast<size_t>(srcWidth) * srcHeight);
        for (int y = 0; y < srcHeight; ++y)
            for (int x = 0; x < srcWidth; ++x)
                input[static_cast<size_t>(y) * srcWidth + x] = static_cast<uint8_t>((x * 17 + y * 31 + 11) & 0xff);
        nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, input, 0);

        EXPECT_NO_THROW(op(stream, src, dst, interpolation));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        auto                 dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        std::vector<uint8_t> actual;
        ASSERT_TRUE(dstData);
        nvcv::util::GetImageVectorFromTensor<uint8_t>(*dstData, 0, actual);

        std::vector<uint8_t> expected(actual.size());
        test::Resize(expected, dstWidth, {dstWidth, dstHeight}, input, srcWidth, {srcWidth, srcHeight}, nvcv::FMT_U8,
                     interpolation, false);

        int maxAbsDiff = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            maxAbsDiff = std::max(maxAbsDiff, abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])));
        EXPECT_LE(maxAbsDiff, 1) << "interpolation=" << static_cast<int>(interpolation) << ", src=" << srcWidth << "x"
                                 << srcHeight << ", dst=" << dstWidth << "x" << dstHeight;
    };

    // The fractional downscale reaches the linear tail's non-negative source-coordinate arm.
    runCase(5, 5, 3, 3, NVCV_INTERP_LINEAR);
    // Exact 2x expansion reaches the interior vertical interpolation arm.
    runCase(5, 5, 10, 10, NVCV_INTERP_LINEAR);
    // A non-expanding cubic launch reaches the valid CubicResize body.
    runCase(5, 5, 3, 3, NVCV_INTERP_CUBIC);
    // A 3x integer AREA reduction fills one U8 vector and reaches AreaResizeFastDirect.
    runCase(48, 48, 16, 16, NVCV_INTERP_AREA);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResizeRegression, padded_u8_tensor_reaches_vector_and_tail_paths)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto runPaddedCase = [&](NVCVInterpolationType interpolation, int srcWidth, int srcHeight, int dstWidth,
                             int dstHeight, int srcPadding, int dstPadding)
    {
        const int64_t srcRowStride      = srcWidth + srcPadding;
        const int64_t dstRowStride      = dstWidth + dstPadding;
        const size_t  srcAllocationSize = static_cast<size_t>(srcRowStride) * srcHeight;
        const size_t  dstAllocationSize = static_cast<size_t>(dstRowStride) * dstHeight;

        NVCVByte *srcAllocation = nullptr;
        NVCVByte *dstAllocation = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), srcAllocationSize));
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), dstAllocationSize));

        nvcv::TensorDataStridedCuda::Buffer srcBuffer{};
        srcBuffer.basePtr    = srcAllocation;
        srcBuffer.strides[0] = srcRowStride * srcHeight;
        srcBuffer.strides[1] = srcRowStride;
        srcBuffer.strides[2] = 1;
        srcBuffer.strides[3] = 1;

        nvcv::TensorDataStridedCuda::Buffer dstBuffer{};
        dstBuffer.basePtr    = dstAllocation;
        dstBuffer.strides[0] = dstRowStride * dstHeight;
        dstBuffer.strides[1] = dstRowStride;
        dstBuffer.strides[2] = 1;
        dstBuffer.strides[3] = 1;

        nvcv::Tensor src = nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{1, srcHeight, srcWidth, 1}, "NHWC"},
                nvcv::TYPE_U8, srcBuffer
        },
            nvcv::TensorDataCleanupCallback{[srcAllocation](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(srcAllocation));
                                            }});
        nvcv::Tensor dst = nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{1, dstHeight, dstWidth, 1}, "NHWC"},
                nvcv::TYPE_U8, dstBuffer
        },
            nvcv::TensorDataCleanupCallback{[dstAllocation](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(dstAllocation));
                                            }});

        std::vector<uint8_t> srcHost(static_cast<size_t>(srcWidth) * srcHeight);
        for (int y = 0; y < srcHeight; ++y)
            for (int x = 0; x < srcWidth; ++x)
                srcHost[static_cast<size_t>(y) * srcWidth + x] = static_cast<uint8_t>((x * 17 + y * 31 + 11) & 0xff);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAllocation, srcRowStride, srcHost.data(), srcWidth, srcWidth, srcHeight,
                                            cudaMemcpyHostToDevice));

        cvcuda::Resize resizeOp;
        EXPECT_NO_THROW(resizeOp(stream, src, dst, interpolation));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        std::vector<uint8_t> actual(static_cast<size_t>(dstWidth) * dstHeight);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), dstWidth, dstAllocation, dstRowStride, dstWidth, dstHeight,
                                            cudaMemcpyDeviceToHost));

        std::vector<uint8_t> expected(actual.size());
        test::Resize(expected, dstWidth, {dstWidth, dstHeight}, srcHost, srcWidth, {srcWidth, srcHeight}, nvcv::FMT_U8,
                     interpolation, false);

        int maxAbsDiff = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            maxAbsDiff = std::max(maxAbsDiff, abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])));
        EXPECT_LE(maxAbsDiff, 1) << "interpolation=" << static_cast<int>(interpolation) << ", src=" << srcWidth << "x"
                                 << srcHeight << ", dst=" << dstWidth << "x" << dstHeight;
    };

    // Odd row strides alternate aligned and unaligned rows while width 17 forces a 16-byte pack plus a tail.
    runPaddedCase(NVCV_INTERP_NEAREST, 34, 8, 17, 4, 1, 1);
    runPaddedCase(NVCV_INTERP_LINEAR, 34, 8, 17, 4, 1, 1);
    runPaddedCase(NVCV_INTERP_LINEAR, 9, 5, 18, 10, 1, 1);
    runPaddedCase(NVCV_INTERP_CUBIC, 34, 8, 17, 4, 1, 1);
    runPaddedCase(NVCV_INTERP_AREA, 34, 8, 17, 4, 14, 15);
    runPaddedCase(NVCV_INTERP_AREA, 34, 8, 17, 4, 1, 1);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Parameters: srcW, srcH, dstW, dstH, interpolation, numImages, planarFmt, interleavedFmt
NVCV_TEST_SUITE_P(
    OpResizePlanar,
    test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // RGB8 (3 channel uint8), every interpolation, expand and contract.
        { 64,  48, 128,  96,       NVCV_INTERP_NEAREST, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        {128,  96,  64,  48,        NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 64,  48, 128,  96,        NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 64,  48, 128,  96,         NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        {128,  96,  64,  48,         NVCV_INTERP_CUBIC, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 90,  60,  45,  30,         NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        {128,  96,  32,  24,          NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 90,  60,  45,  30,          NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // 8/7 reaches the same endpoint-rounding edge in the separate varshape-planar kernel.
        {  8,   8,   7,   7,          NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 97,  73,  37,  29,          NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 31,  23,  79,  61,          NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // Byte non-integer upscale and sub-2x downscale on the flattened single-channel planar view --
  // the planar rows above are all exactly 2x or 1/2x, so these are the only parity coverage of
  // LinearResize<true> and the general cubic upscale on a uint8 plane.
        { 54,  30, 240, 135,        NVCV_INTERP_LINEAR, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 54,  30, 240, 135,         NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 90,  60,  60,  40,        NVCV_INTERP_LINEAR, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 37,  25, 123,  71,         NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // RGBA8 (4 channel uint8).
        { 50,  40, 100,  80,       NVCV_INTERP_NEAREST, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
        {100,  80,  50,  40,        NVCV_INTERP_LINEAR, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
        {120,  90,  30,  22,          NVCV_INTERP_AREA, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
        { 54,  30, 240, 135,        NVCV_INTERP_LINEAR, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
        { 54,  30, 240, 135,         NVCV_INTERP_CUBIC, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
 // Float planar, every interpolation, expand and contract -- exercises the float kernel paths
  // (vectorized LinearResize, gated CubicResize/AreaResizeVec) bit-exactly against interleaved.
        { 64,  48,  96,  72,       NVCV_INTERP_NEAREST, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 96,  72,  48,  36,        NVCV_INTERP_LINEAR, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 48,  36,  96,  72,        NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 64,  48,  96,  72,         NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 64,  48, 128,  96,         NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 96,  72,  32,  24,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 32,  24,  96,  72,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 64,  48, 128,  96,        NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
 // 320/299 is non-dyadic and wide enough that one ULP of the source coordinate (2^-15 at x ~ 268)
  // exceeds the interpolation weight's significance. The planar float path packs four output columns
  // per thread while interleaved float3/float4 pack one, so the two layouts only agree here if both
  // derive every column's coordinate the same way. NEAREST additionally moves a whole pixel.
        {320, 320, 299, 299,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        {320, 320, 299, 299,       NVCV_INTERP_NEAREST, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
 // NEAREST_FLOOR shares the pack structure, so it needs the same planar/interleaved parity cover.
        {320, 320, 299, 299, NVCV_INTERP_NEAREST_FLOOR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 64,  48, 128,  96, NVCV_INTERP_NEAREST_FLOOR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
        { 50,  40, 100,  80, NVCV_INTERP_NEAREST_FLOOR, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
        {128,  96,  64,  48,         NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 64,  48, 128,  96,         NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 45,  30,  90,  60,         NVCV_INTERP_CUBIC, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 31,  23,  73,  55,         NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        {128,  96,  32,  24,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 90,  60,  45,  30,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 97,  73,  29,  19,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
 // F16 planar, every interpolation, expand and contract -- the planar path flattens planes into a
  // single-channel view, so parity with interleaved stays bit-exact; this is the mandated F16
  // coverage for the interpolations without host-gold rows (CUBIC/AREA), mirroring float above.
        { 64,  48,  96,  72,       NVCV_INTERP_NEAREST, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
        { 96,  72,  48,  36,        NVCV_INTERP_LINEAR, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
        { 64,  48, 128,  96,        NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
        {128,  96,  64,  48,         NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
        { 64,  48, 128,  96,         NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
        {128,  96,  32,  24,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
        { 32,  24,  96,  72,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
 // Fractional (non-integer, sub-3x) AREA downscale. Every other wide-vector AREA row is an exact
  // integer ratio, a zoom-in, or above the 3x bound, so these are the only coverage of
  // AreaResizeFractional on an interleaved element whose NIX is below the group width GN (float3 and
  // float4 both have NIX=1) and of all three compile-time interior-column bounds: x scale 1.64
  // selects kMaxC=1, 2.62 selects kMaxC=2 and 2.90 selects kMaxC=3.
        { 97,  73,  37,  29,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 90,  60,  55,  37,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 90,  60,  31,  21,          NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        { 97,  73,  37,  29,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 90,  60,  55,  37,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 90,  60,  31,  21,          NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
 // Wide exact-2x float LINEAR upscale. The float exact-2x rows above are 48x36 -> 96x72 and
  // 64x48 -> 128x96, whose whole destination fits one 32-thread block column, so a
  // warp-span-uniform interior predicate is false everywhere and the destination width is always a
  // multiple of the flattened float plane's 4-element write pack. These rows are wide enough for
  // interior blocks to exist (a warp spans 64 source columns, so block b is interior only when
  // srcW >= 64*b + 65) and have odd source widths, so the 2-element scalar tail runs alongside the
  // vector body.
        {201,  67, 402, 134,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        {135,  51, 270, 102,        NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
 // Non-2x float LINEAR, both scale arms. The float LINEAR rows above are all exactly 2x or 1/2x,
  // so they divert to the exact-2x kernels or sit exactly on the scale >= 2 boundary; these cover
  // the general sub-2x (INTERSECT) arm, a 4x upscale, and a strictly-greater-than-2 downscale.
  // 42 and 46 are 2 mod 4, so the flattened float plane's scalar tail runs alongside its 4-element
  // vector body.
        { 63,  30,  42,  20,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 32,  24, 128,  96,        NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
        {115,  50,  46,  20,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 90,  60,  60,  40,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
 // Exact-2x float LINEAR downscale whose destination width is 2 mod 4. The existing exact-2x float
  // downscale rows (96x72 -> 48x36) have a destination width that is a multiple of the flattened
  // float plane's 4-element pack, so only its aligned vector body runs; these also run the guarded
  // per-element tail.
        { 92,  40,  46,  20,        NVCV_INTERP_LINEAR, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        {124,  40,  62,  20,        NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
 // Non-2x float CUBIC upscale with an odd destination height and a destination width that is not a
  // multiple of the flattened float plane's write pack. The only float CUBIC upscale rows are
  // 64x48 -> 96x72 and the exact-2x ones, all of which have an even height and a width that is a
  // multiple of 4, so nothing covers the row-blocked kernel's odd-last-row branch or its scalar
  // tail. The 54x30 -> 240x135 ratio mirrors the benchmarked 864x480 -> 3840x2160 upscale. Cubic
  // kernels derive every output coordinate directly from dstX.
        { 54,  30, 240, 135,         NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
        { 37,  25, 123,  71,         NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

TEST_P(OpResizePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

// Pin the reference U8 rounding where regrouping the cubic tap products changes the result.
TEST(OpResize, cubic_preserves_reference_rounding)
{
    constexpr int srcW = 37;
    constexpr int srcH = 25;
    constexpr int dstW = 123;
    constexpr int dstH = 71;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor src     = nvcv::util::CreateTensor(1, srcW, srcH, nvcv::FMT_U8);
    nvcv::Tensor dst     = nvcv::util::CreateTensor(1, dstW, dstH, nvcv::FMT_U8);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    std::vector<uint8_t> input(srcW * srcH);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>((i * 73 + i / 17) % 256);
    nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, input, 0);

    cvcuda::Resize op;
    ASSERT_NO_THROW(op(stream, src, dst, NVCV_INTERP_CUBIC));
    uint8_t value = 0;
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(&value, dstAccess->sampleData(0) + 19 * dstAccess->rowStride() + 75,
                                           sizeof(value), cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(value, 148);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Non-dyadic scales must use the same rounded coordinates in both tensor layouts.
TEST(OpResize, float_linear_non_dyadic_planar_parity)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 54, 30, 240, 135, NVCV_INTERP_LINEAR, 1);
    RunPlanarParityTensorCase(nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32, 37, 25, 123, 71, NVCV_INTERP_LINEAR, 1);
}

// Large coordinates can round a fractional AREA interval up to an extra whole column.
TEST(OpResize, wide_fractional_area_preserves_white)
{
    struct Case
    {
        int srcW;
        int dstW;
        int probeX;
    };

    for (const auto &c : {
             Case{4250000, 2250000, 1110257},
             Case{8500000, 3000000, 1480343}
    })
    {
        SCOPED_TRACE(c.srcW);
        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        nvcv::Tensor src     = nvcv::util::CreateTensor(1, c.srcW, 2, nvcv::FMT_U8);
        nvcv::Tensor dst     = nvcv::util::CreateTensor(1, c.dstW, 1, nvcv::FMT_U8);
        auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData && dstData);
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(srcAccess && dstAccess);
        ASSERT_EQ(cudaSuccess,
                  cudaMemset2DAsync(srcAccess->sampleData(0), srcAccess->rowStride(), 255, c.srcW, 2, stream));

        cvcuda::Resize op;
        ASSERT_NO_THROW(op(stream, src, dst, NVCV_INTERP_AREA));
        uint8_t value = 0;
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(&value, dstAccess->sampleData(0) + c.probeX, sizeof(value),
                                               cudaMemcpyDeviceToHost, stream));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(value, 255) << "Constant white input must remain white at x=" << c.probeX;
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}

TEST_P(OpResizePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

// The two nearest conventions must stay observably distinct: NEAREST samples the pixel covering
// the destination center, NEAREST_FLOOR drops the offset. They coincide whenever the scale makes
// the half-pixel shift land inside the same source pixel, so this uses a non-integer downscale
// where the two rules provably select different columns.
TEST(OpResize, nearest_floor_differs_from_nearest)
{
    constexpr int srcW = 320;
    constexpr int srcH = 320;
    constexpr int dstW = 299;
    constexpr int dstH = 299;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src        = nvcv::util::CreateTensor(1, srcW, srcH, nvcv::FMT_U8);
    nvcv::Tensor dstNearest = nvcv::util::CreateTensor(1, dstW, dstH, nvcv::FMT_U8);
    nvcv::Tensor dstFloor   = nvcv::util::CreateTensor(1, dstW, dstH, nvcv::FMT_U8);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    // A horizontal ramp makes any column-selection difference visible in the output bytes.
    std::vector<uint8_t> in(srcH * srcW);
    for (int y = 0; y < srcH; ++y)
        for (int x = 0; x < srcW; ++x) in[y * srcW + x] = static_cast<uint8_t>(x);
    nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, in, 0);

    cvcuda::Resize op;
    EXPECT_NO_THROW(op(stream, src, dstNearest, NVCV_INTERP_NEAREST));
    EXPECT_NO_THROW(op(stream, src, dstFloor, NVCV_INTERP_NEAREST_FLOOR));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> gotNearest;
    std::vector<uint8_t> gotFloor;
    auto                 nearestData = dstNearest.exportData<nvcv::TensorDataStridedCuda>();
    auto                 floorData   = dstFloor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(nearestData && floorData);
    nvcv::util::GetImageVectorFromTensor<uint8_t>(*nearestData, 0, gotNearest);
    nvcv::util::GetImageVectorFromTensor<uint8_t>(*floorData, 0, gotFloor);

    EXPECT_NE(gotNearest, gotFloor) << "NEAREST_FLOOR must not alias NEAREST";

    // Pin the rules themselves rather than merely asserting inequality. The ramp stores the
    // source column index truncated to 8 bits, so the expectation truncates the same way.
    const float scale = static_cast<float>(srcW) / dstW;
    for (int x = 0; x < dstW; ++x)
    {
        const int expNearest = std::min(static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * scale)), srcW - 1);
        const int expFloor   = std::min(static_cast<int>(std::floor(static_cast<float>(x) * scale)), srcW - 1);
        EXPECT_EQ(static_cast<uint8_t>(expNearest), gotNearest[x]) << "NEAREST column " << x;
        EXPECT_EQ(static_cast<uint8_t>(expFloor), gotFloor[x]) << "NEAREST_FLOOR column " << x;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// NEAREST_FLOOR is implemented by Resize only. Every other operator validates interpolation
// against an allowlist, so the new value must be rejected rather than silently treated as
// NEAREST. Rotate stands in for that group.
TEST(OpResize_Negative, nearest_floor_rejected_by_non_resize_operators)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor src = nvcv::util::CreateTensor(1, 32, 24, nvcv::FMT_RGBA8);
    nvcv::Tensor dst = nvcv::util::CreateTensor(1, 32, 24, nvcv::FMT_RGBA8);

    cvcuda::Rotate rotateOp(0);
    double2        shift = {0.0, 0.0};
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { rotateOp(stream, src, dst, 90.0, shift, NVCV_INTERP_NEAREST_FLOOR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpResize_Negative, planar_input_interleaved_output_layout_mismatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    // Planar input, interleaved output: layouts differ and must be rejected.
    nvcv::Tensor src = nvcv::util::CreateTensor(1, 32, 24, nvcv::FMT_RGB8p);
    nvcv::Tensor dst = nvcv::util::CreateTensor(1, 64, 48, nvcv::FMT_RGB8);

    cvcuda::Resize resizeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&resizeOp, &stream, &src, &dst] { resizeOp(stream, src, dst, NVCV_INTERP_NEAREST); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Pins the var-shape planar grid-z bound: the launched grid.z is numImages, not numImages*channels.
// The planar kernels loop channel planes per thread, so the true CUDA grid-z constraint is only on
// numImages. This test uses FMT_RGBA8p (4 channels) at N=16384, where the old numImages*channels
// guard would have thrown (16384*4 = 65536 > 65535) even though grid.z = 16384 is well within
// CUDA's limit. The call must succeed.
TEST(OpResize, varshape_planar_grid_z_is_images_not_planes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::Resize resizeOp;
    constexpr int  N = 16384; // 16384 * 4 channels = 65536 > 65535 (old guard would reject)

    nvcv::ImageBatchVarShape src(N);
    nvcv::ImageBatchVarShape dst(N);
    for (int i = 0; i < N; ++i)
    {
        src.pushBack(nvcv::Image(nvcv::Size2D{1, 1}, nvcv::FMT_RGBA8p));
        dst.pushBack(nvcv::Image(nvcv::Size2D{1, 1}, nvcv::FMT_RGBA8p));
    }
    EXPECT_NO_THROW(resizeOp(stream, src, dst, NVCV_INTERP_NEAREST));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
