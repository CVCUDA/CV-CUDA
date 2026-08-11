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
#include "PlanarParityUtils.hpp"
#include "ResizeUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpResize.hpp>
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

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace t    = ::testing;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(static_cast<double>(size) * scale);
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
    {         90,       60,       45,        30,    NVCV_INTERP_AREA,           1,     nvcv::FMT_RGB8},
    {         90,       60,       45,        30,    NVCV_INTERP_AREA,           1,     nvcv::ImageFormat{NVCV_IMAGE_FORMAT_4U8}},
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
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

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
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

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

double ReferenceCubicPixelReplicateBorder(const std::vector<uint8_t> &src, int srcW, int srcH, int channels, int c,
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

// Self-contained CPU reference for bicubic resize with replicate-border clamping.
// Does NOT depend on ResizeUtils (which had the same bug), so the test is
// independently correct even if both GPU kernel and ResizeUtils are broken.
void ReferenceCubicResizeReplicateBorder(std::vector<uint8_t> &dst, int dstW, int dstH,
                                         const std::vector<uint8_t> &src, int srcW, int srcH, int channels)
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
                // Clamp to [0, 255], no abs().
                val = std::rint(std::max(0.0, std::min(255.0, val)));
                dst[dy * dstW * channels + dx * channels + c] = static_cast<uint8_t>(val);
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
    {nvcv::FMT_F16, nvcv::FMT_F16, 1, 1, NVCV_INTERP_NEAREST}, // invalid datatype
    {nvcv::FMT_F16, nvcv::FMT_F16, 1, 1, NVCV_INTERP_HAMMING}, // invalid interpolation
});

NVCV_TEST_SUITE_P(OpResizeVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType>
{
    // invalid data format
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, NVCV_INTERP_NEAREST},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGB8, nvcv::FMT_RGBf16, nvcv::FMT_RGB8, NVCV_INTERP_NEAREST},
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

    NVCVInterpolationType interpolation = NVCV_INTERP_NEAREST;

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

// Parameters: srcW, srcH, dstW, dstH, interpolation, numImages, planarFmt, interleavedFmt
NVCV_TEST_SUITE_P(OpResizePlanar,
                  test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
  // RGB8 (3 channel uint8), every interpolation, expand and contract.
                      { 64, 48, 128, 96, NVCV_INTERP_NEAREST, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {128, 96,  64, 48,  NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 64, 48, 128, 96,  NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 64, 48, 128, 96,   NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {128, 96,  64, 48,   NVCV_INTERP_CUBIC, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 90, 60,  45, 30,   NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      {128, 96,  32, 24,    NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 90, 60,  45, 30,    NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 97, 73,  37, 29,    NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
                      { 31, 23,  79, 61,    NVCV_INTERP_AREA, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
 // RGBA8 (4 channel uint8).
                      { 50, 40, 100, 80, NVCV_INTERP_NEAREST, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
                      {100, 80,  50, 40,  NVCV_INTERP_LINEAR, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
                      {120, 90,  30, 22,    NVCV_INTERP_AREA, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
 // Float planar, every interpolation, expand and contract -- exercises the float kernel paths
  // (vectorized LinearResize, gated CubicResize/AreaResizeVec) bit-exactly against interleaved.
                      { 64, 48,  96, 72, NVCV_INTERP_NEAREST, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 96, 72,  48, 36,  NVCV_INTERP_LINEAR, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 48, 36,  96, 72,  NVCV_INTERP_LINEAR, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 64, 48,  96, 72,   NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 64, 48, 128, 96,   NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 96, 72,  32, 24,    NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 32, 24,  96, 72,    NVCV_INTERP_AREA, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
                      { 64, 48, 128, 96,  NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      {128, 96,  64, 48,   NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      { 64, 48, 128, 96,   NVCV_INTERP_CUBIC, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      { 45, 30,  90, 60,   NVCV_INTERP_CUBIC, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      {128, 96,  32, 24,    NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
                      { 90, 60,  45, 30,    NVCV_INTERP_AREA, 1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
});

TEST_P(OpResizePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

TEST_P(OpResizePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
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
