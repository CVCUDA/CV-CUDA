/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpRandomResizedCrop.hpp>
#include <cvcuda/cuda_tools/MathWrappers.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <random>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace t    = ::testing;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(static_cast<double>(size) * scale);
}

static void GetCropParams(std::mt19937 &generator, double minScale, double maxScale, double minRatio, double maxRatio,
                          int input_rows, int input_cols, int *top_indices, int *left_indices, int *crop_rows,
                          int *crop_cols)
{
    int          rows          = input_rows;
    int          cols          = input_cols;
    double       area          = rows * cols;
    const double log_min_ratio = std::log(minRatio);
    const double log_max_ratio = std::log(maxRatio);

    std::uniform_real_distribution scale_dist(minScale, maxScale);
    std::uniform_real_distribution ratio_dist(log_min_ratio, log_max_ratio);
    bool                           got_params = false;
    for (int i = 0; i < 10; ++i)
    {
        if (got_params)
            return;
        auto   target_area  = static_cast<int>(area * scale_dist(generator));
        double aspect_ratio = std::exp(ratio_dist(generator));

        *crop_cols = static_cast<int>(std::round(std::sqrt(static_cast<double>(target_area) * aspect_ratio)));
        *crop_rows = static_cast<int>(std::round(std::sqrt(static_cast<double>(target_area) / aspect_ratio)));

        if (*crop_cols > 0 && *crop_cols <= cols && *crop_rows > 0 && *crop_rows <= rows)
        {
            std::uniform_int_distribution row_uni(0, rows - *crop_rows);
            std::uniform_int_distribution col_uni(0, cols - *crop_cols);
            *top_indices  = row_uni(generator);
            *left_indices = col_uni(generator);
            got_params    = true;
        }
    }
    // Fallback to central crop
    if (!got_params)
    {
        if (double in_ratio = double(cols) / double(rows); in_ratio < minRatio)
        {
            *crop_cols = cols;
            *crop_rows = static_cast<int>(std::round(static_cast<double>(*crop_cols) / minRatio));
        }
        else if (in_ratio > maxRatio)
        {
            *crop_rows = rows;
            *crop_cols = static_cast<int>(std::round(static_cast<double>(*crop_rows) * maxRatio));
        }
        else // whole image
        {
            *crop_cols = cols;
            *crop_rows = rows;
        }
        *top_indices  = (rows - *crop_rows) / 2;
        *left_indices = (cols - *crop_cols) / 2;
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpRandomResizedCrop, test::ValueList<int, int, int, int, NVCVInterpolationType, int>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,       interpolation, numberImages
    {        42,        48,       23,        24, NVCV_INTERP_NEAREST,           1},
    {       113,        12,       12,        36, NVCV_INTERP_NEAREST,           1},
    {       421,       148,      223,       124, NVCV_INTERP_NEAREST,           2},
    {       313,       212,      412,       336, NVCV_INTERP_NEAREST,           3},
    {        42,        40,       21,        20,  NVCV_INTERP_LINEAR,           1},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           1},
    {       420,       420,      210,       210,  NVCV_INTERP_LINEAR,           4},
    {       210,       210,      420,       420,  NVCV_INTERP_LINEAR,           5},
    {        42,        40,       21,        20,   NVCV_INTERP_CUBIC,           1},
    {        21,        21,       42,        42,   NVCV_INTERP_CUBIC,           6},
    {        420,      420,      420,       420,   NVCV_INTERP_CUBIC,           2},
    {        420,      420,      420,       420,   NVCV_INTERP_CUBIC,           1},
    {        420,      420,       40,        42,   NVCV_INTERP_CUBIC,           1},
    {       1920,     1080,      640,       320,   NVCV_INTERP_CUBIC,           1},
    {       1920,     1080,      640,       320,   NVCV_INTERP_CUBIC,           2},
});

// clang-format on

static void RunTensorCorrectOutput(int srcWidth, int srcHeight, int dstWidth, int dstHeight,
                                   NVCVInterpolationType interpolation, int numberOfImages, nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4;
    double maxRatio = 4.0 / 3;

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

    // use fixed seed
    uint32_t                  seed = 1;
    cvcuda::RandomResizedCrop randomResizedCropOp(minScale, maxScale, minRatio, maxRatio, numberOfImages, seed);
    EXPECT_NO_THROW(randomResizedCropOp(stream, imgSrc, imgDst, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);

    std::mt19937 generator(seed);

    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        int top;
        int left;
        int crop_rows;
        int crop_cols;
        GetCropParams(generator, minScale, maxScale, minRatio, maxRatio, srcHeight, srcWidth, &top, &left, &crop_rows,
                      &crop_cols);

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);

        // Generate gold result
        test::ResizedCrop(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride,
                          {srcWidth, srcHeight}, top, left, crop_rows, crop_cols, fmt, interpolation);

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

TEST_P(OpRandomResizedCrop, tensor_correct_output)
{
    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();
    int                   numberImages  = GetParamValue<5>();

    RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_RGBA8);
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_RGB8);
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_U8);
        // F16 LINEAR runs the real half kernels (FP32 accumulation, one rounding to half) and is
        // checked against the FP32 gold within half-ULP bounds; see ExpectF16InterpOutput.
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_RGBf16);
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_F16);
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_U8);
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        // F16 NEAREST is pure data movement and compares bit-exactly.
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages,
                               nvcv::FMT_RGBAf16);
    }
}

static void RunVarShapeCorrectOutput(int srcWidthBase, int srcHeightBase, int dstWidthBase, int dstHeightBase,
                                     NVCVInterpolationType interpolation, int numberOfImages, nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4;
    double maxRatio = 4.0 / 3;

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndSrcWidth(ScaledSize(srcWidthBase, 0.8), ScaledSize(srcWidthBase, 1.1));
    std::uniform_int_distribution rndSrcHeight(ScaledSize(srcHeightBase, 0.8), ScaledSize(srcHeightBase, 1.1));

    std::uniform_int_distribution rndDstWidth(ScaledSize(dstWidthBase, 0.8), ScaledSize(dstWidthBase, 1.1));
    std::uniform_int_distribution rndDstHeight(ScaledSize(dstHeightBase, 0.8), ScaledSize(dstHeightBase, 1.1));

    std::vector<nvcv::Image> imgSrc;

    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numberOfImages; ++i)
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

    // Generate test result, using fixed seed
    uint32_t                  seed = 1;
    cvcuda::RandomResizedCrop randomResizedCropOp(minScale, maxScale, minRatio, maxRatio, numberOfImages, seed);
    EXPECT_NO_THROW(randomResizedCropOp(stream, batchSrc, batchDst, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::mt19937 generator(seed);

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

        int top;
        int left;
        int crop_rows;
        int crop_cols;
        GetCropParams(generator, minScale, maxScale, minRatio, maxRatio, srcHeight, srcWidth, &top, &left, &crop_rows,
                      &crop_cols);

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);

        // Generate gold result
        test::ResizedCrop(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride[i],
                          {srcWidth, srcHeight}, top, left, crop_rows, crop_cols, fmt, interpolation);

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

TEST_P(OpRandomResizedCrop, varshape_correct_output)
{
    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();
    int dstWidthBase  = GetParamValue<2>();
    int dstHeightBase = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();
    int                   numberImages  = GetParamValue<5>();

    RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                             nvcv::FMT_RGBA8);
    if (interpolation == NVCV_INTERP_LINEAR)
    {
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_RGB8);
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_U8);
        // F16 LINEAR runs the real half kernels (FP32 accumulation, one rounding to half) and is
        // checked against the FP32 gold within half-ULP bounds; see ExpectF16InterpOutput.
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_RGBf16);
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_F16);
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_U8);
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
        // F16 NEAREST is pure data movement and compares bit-exactly.
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_RGBAf16);
    }
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// RandomResizedCrop samples every channel from the same crop window, so a planar input is cropped
// and resized plane-by-plane and must produce the same pixels as the interleaved path. Each parity
// invocation constructs a fresh operator with the same seed so the interleaved and planar runs use
// identical crop parameters.
// =============================================================================

namespace {

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    constexpr double   minScale = 0.08;
    constexpr double   maxScale = 1.0;
    constexpr double   minRatio = 3.0 / 4.0;
    constexpr double   maxRatio = 4.0 / 3.0;
    constexpr uint32_t seed     = 11;

    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [interp, numImages](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::RandomResizedCrop op(minScale, maxScale, minRatio, maxRatio, numImages, seed);
            EXPECT_NO_THROW(op(s, src, dst, interp));
        });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int dstW, int dstH, NVCVInterpolationType interp, int numImages)
{
    constexpr double   minScale = 0.08;
    constexpr double   maxScale = 1.0;
    constexpr double   minRatio = 3.0 / 4.0;
    constexpr double   maxRatio = 4.0 / 3.0;
    constexpr uint32_t seed     = 11;

    test::planar::RunVarShapeParity(planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
                                    [interp, numImages](cudaStream_t s, const nvcv::ImageBatchVarShape &src,
                                                        const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
                                    {
                                        cvcuda::RandomResizedCrop op(minScale, maxScale, minRatio, maxRatio, numImages,
                                                                     seed);
                                        EXPECT_NO_THROW(op(s, src, dst, interp));
                                    });
}

} // namespace

// Parameters: srcW, srcH, dstW, dstH, interpolation, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpRandomResizedCropPlanar,
    test::ValueList<int, int, int, int, NVCVInterpolationType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 128, 96, NVCV_INTERP_NEAREST, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {128, 96,  64, 48,  NVCV_INTERP_LINEAR, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 80, 60,  40, 30,   NVCV_INTERP_CUBIC, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 50, 40, 100, 80, NVCV_INTERP_NEAREST, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 96, 64,  48, 32,  NVCV_INTERP_LINEAR, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 64, 48,  96, 72, NVCV_INTERP_NEAREST, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    { 72, 54,  45, 35,  NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 68, 52,  39, 31,   NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    // F16 planes take the same per-plane kernels as interleaved channels, so parity stays
    // bit-exact; CUBIC F16 (no host-gold row) gets its coverage here, mirroring float above.
    { 64, 48,  96, 72, NVCV_INTERP_NEAREST, 2, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
    { 72, 54,  45, 35,  NVCV_INTERP_LINEAR, 2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 68, 52,  39, 31,   NVCV_INTERP_CUBIC, 1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
});

// clang-format on

TEST_P(OpRandomResizedCropPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

TEST_P(OpRandomResizedCropPlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<6>(), GetParamValue<7>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>());
}

namespace {

struct StrictHostCrop
{
    int top;
    int left;
    int rows;
    int cols;
};

void SetStrictHostSentinels(const std::vector<nvcv::Image> &images, cudaStream_t stream)
{
    for (const auto &img : images)
    {
        const auto data       = img.exportData<nvcv::ImageDataStridedCuda>();
        const int  pixelBytes = img.format().planePixelStrideBytes(0);
        ASSERT_TRUE(data);
        for (int p = 0; p < img.format().numPlanes(); ++p)
        {
            ASSERT_EQ(cudaSuccess, cudaMemset2DAsync(data->plane(p).basePtr, data->plane(p).rowStride, 0xA5,
                                                     data->plane(p).width * pixelBytes, data->plane(p).height, stream));
        }
    }
}

void CheckStrictHostOutput(const nvcv::Image &src, const nvcv::Image &dst, const std::vector<uint8_t> &pixels,
                           const StrictHostCrop &crop)
{
    const auto srcSize    = src.size();
    const auto dstSize    = dst.size();
    const auto data       = dst.exportData<nvcv::ImageDataStridedCuda>();
    const int  pixelBytes = dst.format().planePixelStrideBytes(0);
    ASSERT_TRUE(data);
    const float scaleX   = static_cast<float>(crop.cols) / static_cast<float>(dstSize.w);
    const float scaleY   = static_cast<float>(crop.rows) / static_cast<float>(dstSize.h);
    const int   rowBytes = dstSize.w * pixelBytes;
    for (int p = 0; p < dst.format().numPlanes(); ++p)
    {
        SCOPED_TRACE(p);
        std::vector<uint8_t> actual(dstSize.h * rowBytes);
        std::vector<uint8_t> expected(actual.size());
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, data->plane(p).basePtr, data->plane(p).rowStride,
                                            rowBytes, dstSize.h, cudaMemcpyDeviceToHost));
        for (int y = 0; y < dstSize.h; ++y)
        {
            const int sy = std::min(static_cast<int>(std::floor((static_cast<float>(y) + 0.5f) * scaleY)) + crop.top,
                                    srcSize.h - 1);
            for (int x = 0; x < dstSize.w; ++x)
            {
                const int sx = std::min(
                    static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * scaleX)) + crop.left, srcSize.w - 1);
                std::memcpy(expected.data() + (y * dstSize.w + x) * pixelBytes,
                            pixels.data() + ((p * srcSize.h + sy) * srcSize.w + sx) * pixelBytes, pixelBytes);
            }
        }
        EXPECT_EQ(expected, actual);
    }
}

// The host submission selects each image's geometry, crop and plane. Nearest sampling has an
// exact CPU reference; a whole-image, unit-scale fallback additionally makes every
// interpolation mode an exact identity on nonuniform pixels, without changing resize tolerances.
void RunVarShapeStrictHostReference(nvcv::ImageFormat fmt, int count, NVCVInterpolationType interp, bool fallback,
                                    bool identity)
{
    const double       minScale = fallback ? 2.0 : 0.08;
    const double       maxScale = fallback ? 2.0 : 1.0;
    const double       minRatio = identity ? 0.01 : 0.75;
    const double       maxRatio = identity ? 100.0 : 4.0 / 3.0;
    constexpr uint32_t seed     = 1234;
    cudaStream_t       stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image>          src;
    std::vector<nvcv::Image>          dst;
    std::vector<std::vector<uint8_t>> pixels(count);
    const int                         planes     = fmt.numPlanes();
    const int                         pixelBytes = fmt.planePixelStrideBytes(0);
    for (int i = 0; i < count; ++i)
    {
        const nvcv::Size2D srcSize{17 + 3 * (i % 5), 13 + 2 * (i % 7)};
        const nvcv::Size2D dstSize = identity ? srcSize : nvcv::Size2D{9 + (7 * i) % 23, 7 + (5 * i) % 19};
        src.emplace_back(srcSize, fmt);
        dst.emplace_back(dstSize, fmt);
        const auto data = src.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(data);
        const int rowBytes   = srcSize.w * pixelBytes;
        const int planeBytes = srcSize.h * rowBytes;
        pixels[i].resize(planes * planeBytes);
        test::planar::FillDeterministicValues(pixels[i], 101 * i + 1, fmt.planeDataType(0).channelType(0));
        for (int p = 0; p < planes; ++p)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(data->plane(p).basePtr, data->plane(p).rowStride, pixels[i].data() + p * planeBytes,
                                   rowBytes, rowBytes, srcSize.h, cudaMemcpyHostToDevice));
        }
    }
    nvcv::ImageBatchVarShape batchSrc(count);
    nvcv::ImageBatchVarShape batchDst(count);
    batchSrc.pushBack(src.begin(), src.end());
    batchDst.pushBack(dst.begin(), dst.end());
    cvcuda::RandomResizedCrop op(minScale, maxScale, minRatio, maxRatio, count, seed);
    // Reproduce the operator's seeded crop sequence, not security-sensitive randomness.
    std::mt19937              generator(seed);

    // A late invalid image must be rejected before any preceding image consumes RNG state.
    nvcv::Image              invalidImage({1, 7}, fmt);
    nvcv::ImageBatchVarShape invalidSrc(count);
    invalidSrc.pushBack(src.begin(), src.end() - 1);
    invalidSrc.pushBack(invalidImage);
    std::vector<int> srcRefs;
    std::vector<int> dstRefs;
    for (int i = 0; i < count; ++i)
    {
        srcRefs.push_back(src[i].refCount());
        dstRefs.push_back(dst[i].refCount());
    }
    const int invalidRefs = invalidImage.refCount();
    // Both successful submission and the validation throw must release every bulk handle.
    auto      checkRefCounts = [&]
    {
        for (int i = 0; i < count; ++i)
        {
            EXPECT_EQ(srcRefs[i], src[i].refCount()) << i;
            EXPECT_EQ(dstRefs[i], dst[i].refCount()) << i;
        }
        EXPECT_EQ(invalidRefs, invalidImage.refCount());
    };
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, invalidSrc, batchDst, NVCV_INTERP_LINEAR); }));
    checkRefCounts();

    auto submitOnce = [&]
    {
        ASSERT_NO_FATAL_FAILURE(SetStrictHostSentinels(dst, stream));
        ASSERT_NO_THROW(op(stream, batchSrc, batchDst, interp));
        checkRefCounts();
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int i = 0; i < count; ++i)
        {
            SCOPED_TRACE(i);
            const auto     srcSize = src[i].size();
            StrictHostCrop crop;
            GetCropParams(generator, minScale, maxScale, minRatio, maxRatio, srcSize.h, srcSize.w, &crop.top,
                          &crop.left, &crop.rows, &crop.cols);
            ASSERT_NO_FATAL_FAILURE(CheckStrictHostOutput(src[i], dst[i], pixels[i], crop));
        }
    };
    for (int submit = 0; submit < 2; ++submit)
    {
        SCOPED_TRACE(submit);
        ASSERT_NO_FATAL_FAILURE(submitOnce());
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

// Covers all supported scalar types, channel counts (1, 3, 4), and native layout paths.
// clang-format off
NVCV_TEST_SUITE_P(OpRandomResizedCropStrictHost,
    test::ValueList<nvcv::ImageFormat, int>{
        {nvcv::FMT_U8, 1}, {nvcv::FMT_U16, 3}, {nvcv::FMT_S16, 3}, {nvcv::FMT_F16, 3}, {nvcv::FMT_F32, 1},
        {nvcv::FMT_RGB8, 257}, {nvcv::FMT_RGBA8, 3}, {nvcv::FMT_RGB8p, 3}, {nvcv::FMT_RGBA8p, 1},
        {nvcv::FMT_RGBf16, 3}, {nvcv::FMT_RGBAf16, 3}, {nvcv::FMT_RGBf16p, 1}, {nvcv::FMT_RGBAf16p, 3},
        {nvcv::FMT_RGBf32, 257}, {nvcv::FMT_RGBAf32, 3}, {nvcv::FMT_RGBf32p, 3}, {nvcv::FMT_RGBAf32p, 1},
    });

// clang-format on

TEST_P(OpRandomResizedCropStrictHost, seeded_nearest_reference)
{
    RunVarShapeStrictHostReference(GetParamValue<0>(), GetParamValue<1>(), NVCV_INTERP_NEAREST, false, false);
}

TEST_P(OpRandomResizedCropStrictHost, fallback_nearest_reference)
{
    RunVarShapeStrictHostReference(GetParamValue<0>(), GetParamValue<1>(), NVCV_INTERP_NEAREST, true, false);
}

TEST_P(OpRandomResizedCropStrictHost, whole_image_identity_reference)
{
    for (auto interp : {NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR, NVCV_INTERP_CUBIC})
    {
        SCOPED_TRACE(interp);
        RunVarShapeStrictHostReference(GetParamValue<0>(), GetParamValue<1>(), interp, true, true);
    }
}

TEST(OpRandomResizedCrop_Negative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(nullptr, 0.2, 1.0, 0.8, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop, createWithZeroSeed)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_SUCCESS, cvcudaRandomResizedCropCreate(&opHandle, 0.2, 1.0, 0.8, 1.3, 2, 0));
    EXPECT_NO_THROW(nvcvOperatorDestroy(opHandle));
}

TEST(OpRandomResizedCrop_Negative, createWithInvalidScale)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, 1.0, 0.2, 0.8, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop_Negative, createWithInvalidRatio)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, 0.2, 1.0, 1.3, 0.8, 2, 0));
}

// std::log(minRatio/maxRatio) and area sampling are undefined for non-positive bounds.
TEST(OpRandomResizedCrop_Negative, createWithZeroScale)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, 0.0, 1.0, 0.8, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop_Negative, createWithNegativeScale)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, -0.1, 1.0, 0.8, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop_Negative, createWithZeroRatio)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, 0.2, 1.0, 0.0, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop_Negative, createWithNegativeRatio)
{
    NVCVOperatorHandle opHandle;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaRandomResizedCropCreate(&opHandle, 0.2, 1.0, -0.5, 1.3, 2, 0));
}

TEST(OpRandomResizedCrop_Negative, tensorBatchLargerThanMaxBatchSizeRejected)
{
    constexpr int maxBatchSize   = 1;
    constexpr int numberOfImages = 2;

    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numberOfImages, 24, 24, nvcv::FMT_RGBA8);
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numberOfImages, 12, 12, nvcv::FMT_RGBA8);

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, maxBatchSize, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&randomResizedCropOp, &imgSrc, &imgDst]
                                { randomResizedCropOp(nullptr, imgSrc, imgDst, NVCV_INTERP_NEAREST); }));
}

// clang-format off
NVCV_TEST_SUITE_P(OpRandomResizedCrop_Negative, nvcv::test::ValueList<std::string, nvcv::DataType, std::string, nvcv::DataType, NVCVInterpolationType, int>
{
    //   in_layout,        in_data_type,   out_layout,     out_data_type,         interpolation, channels
    {        "CHW",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 5},
    {        "HWC",       nvcv::TYPE_U8,        "CHW",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 2},
    {        "CHW",       nvcv::TYPE_U8,        "CHW",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",      nvcv::TYPE_F64,        "HWC",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",    nvcv::TYPE_F64,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",      nvcv::TYPE_U32,        "HWC",     nvcv::TYPE_U8,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",    nvcv::TYPE_U32,   NVCV_INTERP_NEAREST, 2},
    {        "HWC",       nvcv::TYPE_U8,        "HWC",     nvcv::TYPE_U8,      NVCV_INTERP_AREA, 2},
});

NVCV_TEST_SUITE_P(OpRandomResizedCropVarshape_Negative, nvcv::test::ValueList<int, nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // exceed max batch size
    {10, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, NVCV_INTERP_NEAREST, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8},
    // invalid format
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, NVCV_INTERP_NEAREST, nvcv::FMT_RGB8, nvcv::FMT_RGBA8},
    {10, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, NVCV_INTERP_NEAREST, nvcv::FMT_RGBA8, nvcv::FMT_RGB8},
    {2, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8p, NVCV_INTERP_NEAREST, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, NVCV_INTERP_NEAREST, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
    // in/out data type mismatch (F16 is a supported dtype, but it must match on both sides)
    {2, nvcv::FMT_RGBAf16, nvcv::FMT_RGBA8, NVCV_INTERP_NEAREST, nvcv::FMT_RGBAf16, nvcv::FMT_RGBA8},
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBAf16, NVCV_INTERP_NEAREST, nvcv::FMT_RGBA8, nvcv::FMT_RGBAf16},
    // invalid interpolation
    {2, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, NVCV_INTERP_HAMMING, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8},

});

// clang-format on

TEST_P(OpRandomResizedCrop_Negative, infer_negative_parameter)
{
    std::string           in_layout     = GetParamValue<0>();
    nvcv::DataType        in_data_type  = GetParamValue<1>();
    std::string           out_layout    = GetParamValue<2>();
    nvcv::DataType        out_data_type = GetParamValue<3>();
    NVCVInterpolationType interpolation = GetParamValue<4>();
    int                   channels      = GetParamValue<5>();

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4;
    double maxRatio = 4.0 / 3;

    nvcv::Tensor imgSrc(
        {
            {24, 24, channels},
            in_layout.c_str()
    },
        in_data_type);
    nvcv::Tensor imgDst(
        {
            {24, 24, channels},
            out_layout.c_str()
    },
        out_data_type);

    // Create and Call operator
    int      numberOfImages = 4;
    uint32_t seed           = 1;

    cvcuda::RandomResizedCrop randomResizedCropOp(minScale, maxScale, minRatio, maxRatio, numberOfImages, seed);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&randomResizedCropOp, &imgSrc, &imgDst, &interpolation]
                                { randomResizedCropOp(nullptr, imgSrc, imgDst, interpolation); }));
}

TEST_P(OpRandomResizedCropVarshape_Negative, infer_negative_parameter)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                   numberOfImages = GetParamValue<0>();
    nvcv::ImageFormat     fmtIn          = GetParamValue<1>();
    nvcv::ImageFormat     fmtOut         = GetParamValue<2>();
    NVCVInterpolationType interpolation  = GetParamValue<3>();
    nvcv::ImageFormat     fmtInExtra     = GetParamValue<4>();
    nvcv::ImageFormat     fmtOutExtra    = GetParamValue<5>();

    const int srcWidthBase      = 24;
    const int srcHeightBase     = 24;
    const int dstWidthBase      = 24;
    const int dstHeightBase     = 24;
    const int maxNumberOfImages = 6;

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4.0;
    double maxRatio = 4.0 / 3.0;

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
        imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmtIn);
        imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmtOut);
    }

    imgSrc.emplace_back(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmtInExtra);
    imgDst.emplace_back(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmtOutExtra);

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    uint32_t                  seed = 1;
    cvcuda::RandomResizedCrop randomResizedCropOp(minScale, maxScale, minRatio, maxRatio, maxNumberOfImages, seed);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&randomResizedCropOp, &stream, &batchSrc, &batchDst, &interpolation]
                                { randomResizedCropOp(stream, batchSrc, batchDst, interpolation); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCrop, linearPackedAndPlanarStridedCoverage)
{
    // The non-multiple output width supplies complete packed groups and a scalar tail, while the
    // planar side reaches the 8-bit strided linear kernel with both in-range and out-of-range y
    // threads. The shared parity helper also exercises an interior bilinear source tap.
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 64, 48, 67, 51, NVCV_INTERP_LINEAR, 1);
}

static std::vector<uint8_t> ExactU8LinearReference(const std::vector<uint8_t> &input, nvcv::Size2D srcSize,
                                                   nvcv::Size2D dstSize, int channels)
{
    std::vector<uint8_t> expected(static_cast<size_t>(dstSize.w) * dstSize.h * channels);
    for (int row = 0; row < dstSize.h; ++row)
    {
        const double sourceRow
            = std::clamp((row + 0.5) * srcSize.h / dstSize.h - 0.5, 0.0, static_cast<double>(srcSize.h - 1));
        const auto   upper   = static_cast<int>(std::floor(sourceRow));
        const int    lower   = std::min(upper + 1, srcSize.h - 1);
        const double weightY = sourceRow - upper;
        for (int column = 0; column < dstSize.w; ++column)
        {
            const double sourceColumn
                = std::clamp((column + 0.5) * srcSize.w / dstSize.w - 0.5, 0.0, static_cast<double>(srcSize.w - 1));
            const auto   left    = static_cast<int>(std::floor(sourceColumn));
            const int    right   = std::min(left + 1, srcSize.w - 1);
            const double weightX = sourceColumn - left;
            for (int channel = 0; channel < channels; ++channel)
            {
                const auto sample = [&](int sourceY, int sourceX)
                {
                    return input[(static_cast<size_t>(sourceY) * srcSize.w + sourceX) * channels + channel];
                };
                const double value = sample(upper, left) * (1.0 - weightX) * (1.0 - weightY)
                                   + sample(upper, right) * weightX * (1.0 - weightY)
                                   + sample(lower, left) * (1.0 - weightX) * weightY
                                   + sample(lower, right) * weightX * weightY;
                const auto   integral = static_cast<int>(std::floor(value));
                const double fraction = value - integral;
                const bool   roundUp  = fraction > 0.5 || (fraction == 0.5 && integral % 2 != 0);
                const int    rounded  = integral + (roundUp ? 1 : 0);
                expected[(static_cast<size_t>(row) * dstSize.w + column) * channels + channel]
                    = static_cast<uint8_t>(rounded);
            }
        }
    }
    return expected;
}

static void RunExactRgb8LinearReference(const std::vector<nvcv::Size2D> &srcSizes,
                                        const std::vector<nvcv::Size2D> &dstSizes, bool varShape,
                                        nvcv::ImageFormat planarFormat      = nvcv::FMT_RGB8p,
                                        nvcv::ImageFormat interleavedFormat = nvcv::FMT_RGB8)
{
    std::vector<std::vector<uint8_t>> expected;
    auto makeInput = [&expected, &srcSizes, &dstSizes](int sample, int width, int height, int channels)
    {
        std::vector<uint8_t> input(static_cast<size_t>(width) * height * channels);
        for (size_t offset = 0; offset < input.size(); ++offset)
        {
            input[offset] = static_cast<uint8_t>(offset * 73 + sample * 19);
        }
        expected.push_back(ExactU8LinearReference(input, srcSizes[sample], dstSizes[sample], channels));
        return input;
    };
    size_t sample  = 0;
    auto   compare = [&expected, &sample](const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
    {
        ASSERT_LT(sample, expected.size());
        EXPECT_EQ(expected[sample], interleaved);
        EXPECT_EQ(expected[sample], planar);
        ++sample;
    };
    auto invoke = [count = static_cast<int>(srcSizes.size())](cudaStream_t stream, const auto &src, const auto &dst,
                                                              nvcv::ImageFormat)
    {
        cvcuda::RandomResizedCrop op(2.0, 2.0, 0.01, 100.0, count, 1234);
        ASSERT_NO_THROW(op(stream, src, dst, NVCV_INTERP_LINEAR));
    };
    if (varShape)
    {
        test::planar::RunVarShapeParity(planarFormat, interleavedFormat, srcSizes, dstSizes, invoke, makeInput,
                                        compare);
    }
    else
    {
        test::planar::RunTensorParity(planarFormat, interleavedFormat, srcSizes[0].w, srcSizes[0].h, dstSizes[0].w,
                                      dstSizes[0].h, static_cast<int>(srcSizes.size()), invoke, makeInput, compare);
    }
    EXPECT_EQ(sample, srcSizes.size());
}

TEST(OpRandomResizedCrop, rgb8_linear_dyadic_exact_reference_tensor)
{
    const std::vector<nvcv::Size2D> sourceSizes{
        { 2,  2},
        {64, 16},
        {65, 17},
        {66, 18},
        {67, 19},
        {10, 14}
    };
    const std::vector<nvcv::Size2D> outputSizes{
        {  1,  1},
        {128, 32},
        {130, 34},
        { 33,  9},
        { 67, 19},
        {  5,  7}
    };
    for (size_t index = 0; index < sourceSizes.size(); ++index)
    {
        SCOPED_TRACE(index);
        for (int count : {1, 3})
        {
            SCOPED_TRACE(count);
            RunExactRgb8LinearReference(std::vector<nvcv::Size2D>(count, sourceSizes[index]),
                                        std::vector<nvcv::Size2D>(count, outputSizes[index]), false);
        }
    }
}

TEST(OpRandomResizedCrop, rgb8_linear_dyadic_exact_reference_varshape)
{
    RunExactRgb8LinearReference(
        {
            { 2,  2},
            {64, 16},
            {65, 17},
            {66, 18},
            {67, 19},
            {10, 14}
    },
        {{1, 1}, {128, 32}, {130, 34}, {33, 9}, {67, 19}, {5, 7}}, true);
}

TEST(OpRandomResizedCrop, rgb8_linear_full_byte_range_multiblock_exact_reference)
{
    const std::vector<nvcv::Size2D> sourceSizes{
        {256, 17},
        {257, 19},
        {259, 21}
    };
    const std::vector<nvcv::Size2D> outputSizes{
        {512, 34},
        {514, 38},
        {518, 42}
    };
    for (size_t index = 0; index < sourceSizes.size(); ++index)
    {
        SCOPED_TRACE(index);
        RunExactRgb8LinearReference(std::vector<nvcv::Size2D>(3, sourceSizes[index]),
                                    std::vector<nvcv::Size2D>(3, outputSizes[index]), false);
    }
    RunExactRgb8LinearReference(sourceSizes, outputSizes, true);
}

TEST(OpRandomResizedCrop, rgba8_linear_dyadic_exact_reference)
{
    const std::vector<nvcv::Size2D> sourceSizes{
        { 2,  2},
        {64, 16},
        {65, 17},
        {66, 18},
        {67, 19},
        {10, 14}
    };
    const std::vector<nvcv::Size2D> outputSizes{
        {  1,  1},
        {128, 32},
        {130, 34},
        { 33,  9},
        { 67, 19},
        {  5,  7}
    };
    for (size_t index = 0; index < sourceSizes.size(); ++index)
    {
        SCOPED_TRACE(index);
        for (int count : {1, 3})
        {
            SCOPED_TRACE(count);
            RunExactRgb8LinearReference(std::vector<nvcv::Size2D>(count, sourceSizes[index]),
                                        std::vector<nvcv::Size2D>(count, outputSizes[index]), false, nvcv::FMT_RGBA8p,
                                        nvcv::FMT_RGBA8);
        }
    }
    RunExactRgb8LinearReference(sourceSizes, outputSizes, true, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8);
}

static void RunExactSingleChannelLinearReference(nvcv::Size2D sourceSize, nvcv::Size2D outputSize, int count)
{
    nvcv::Tensor source     = count == 1 ? nvcv::Tensor(
                                  {
                                      {1, sourceSize.h, sourceSize.w},
                                      "CHW"
    },
                                  nvcv::TYPE_U8)
                                         : nvcv::Tensor({{count, 1, sourceSize.h, sourceSize.w}, "NCHW"}, nvcv::TYPE_U8);
    nvcv::Tensor output     = count == 1 ? nvcv::Tensor(
                                  {
                                      {1, outputSize.h, outputSize.w},
                                      "CHW"
    },
                                  nvcv::TYPE_U8)
                                         : nvcv::Tensor({{count, 1, outputSize.h, outputSize.w}, "NCHW"}, nvcv::TYPE_U8);
    auto         sourceData = source.exportData<nvcv::TensorDataStridedCuda>();
    auto         outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(sourceData && outputData);
    auto sourceAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*sourceData);
    auto outputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
    ASSERT_TRUE(sourceAccess && outputAccess);
    std::vector<std::vector<uint8_t>> expected;
    for (int sample = 0; sample < count; ++sample)
    {
        std::vector<uint8_t> input(static_cast<size_t>(sourceSize.w) * sourceSize.h);
        for (size_t offset = 0; offset < input.size(); ++offset)
            input[offset] = static_cast<uint8_t>(offset * 73 + sample * 19);
        expected.push_back(ExactU8LinearReference(input, sourceSize, outputSize, 1));
        test::planar::UploadPlanarSample(*sourceAccess, sample, input, sourceSize.w, sourceSize.h, 1, 1);
        test::planar::UploadPlanarSample(*outputAccess, sample, std::vector<uint8_t>(outputSize.w * outputSize.h, 0xA5),
                                         outputSize.w, outputSize.h, 1, 1);
    }
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::RandomResizedCrop op(2.0, 2.0, 0.01, 100.0, count, 1234);
    ASSERT_NO_THROW(op(stream, source, output, NVCV_INTERP_LINEAR));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    for (int sample = 0; sample < count; ++sample)
        EXPECT_EQ(expected[sample],
                  test::planar::DownloadPlanarSample(*outputAccess, sample, outputSize.w, outputSize.h, 1, 1));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCrop, u8_single_channel_planar_linear_dyadic_exact_reference)
{
    const std::vector<nvcv::Size2D> sourceSizes{
        { 2,  2},
        {64, 16},
        {65, 17},
        {66, 18},
        {67, 19},
        {10, 14}
    };
    const std::vector<nvcv::Size2D> outputSizes{
        {  1,  1},
        {128, 32},
        {130, 34},
        { 33,  9},
        { 67, 19},
        {  5,  7}
    };
    for (size_t index = 0; index < sourceSizes.size(); ++index)
    {
        SCOPED_TRACE(index);
        const auto sourceSize = sourceSizes[index];
        const auto outputSize = outputSizes[index];
        for (int count : {1, 3})
        {
            SCOPED_TRACE(count);
            RunExactSingleChannelLinearReference(sourceSize, outputSize, count);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
        }
    }
}

TEST(OpRandomResizedCrop, u8_planar_linear_cropped_boundary_tail_parity)
{
    const std::vector<nvcv::Size2D> sourceSizes{
        {13,  9},
        {65, 31},
        {67, 45},
        { 2,  2},
        {32, 17},
        {91, 29}
    };
    const std::vector<nvcv::Size2D> outputSizes{
        { 41, 23},
        {129, 61},
        { 19, 17},
        {  7,  5},
        {130, 39},
        {127, 11}
    };
    for (size_t index = 0; index < sourceSizes.size(); ++index)
    {
        SCOPED_TRACE(index);
        const auto sourceSize = sourceSizes[index];
        const auto outputSize = outputSizes[index];
        for (int count : {1, 3})
        {
            SCOPED_TRACE(count);
            test::planar::RunTensorSingleChannelLayoutParity(
                sourceSize.w, sourceSize.h, outputSize.w, outputSize.h, count, nvcv::TYPE_U8,
                [count](cudaStream_t stream, const nvcv::Tensor &source, const nvcv::Tensor &output)
                {
                    cvcuda::RandomResizedCrop op(0.08, 1.0, 0.5, 2.0, count, 1234);
                    EXPECT_NO_THROW(op(stream, source, output, NVCV_INTERP_LINEAR));
                });
            for (auto planarFormat : {nvcv::FMT_RGB8p, nvcv::FMT_RGBA8p})
            {
                const auto interleavedFormat = planarFormat == nvcv::FMT_RGB8p ? nvcv::FMT_RGB8 : nvcv::FMT_RGBA8;
                RunPlanarParityTensorCase(planarFormat, interleavedFormat, sourceSize.w, sourceSize.h, outputSize.w,
                                          outputSize.h, NVCV_INTERP_LINEAR, count);
                RunPlanarParityVarShapeCase(planarFormat, interleavedFormat, sourceSize.w, sourceSize.h, outputSize.w,
                                            outputSize.h, NVCV_INTERP_LINEAR, count);
            }
        }
    }
}

TEST(OpRandomResizedCrop, linearPackedRowAlignmentAndInteriorTap)
{
    // The full-image crop keeps interior source taps in range; the one-byte row pad alternates
    // row-base alignment so complete packed groups take both alignment arms.
    constexpr int    kWidth           = 8;
    constexpr int    kHeight          = 8;
    constexpr int    kChannels        = 3;
    constexpr size_t kLogicalRowBytes = static_cast<size_t>(kWidth) * kChannels;
    constexpr size_t kRowStride       = kLogicalRowBytes + 1;
    constexpr size_t kSampleStride    = kRowStride * kHeight;

    NVCVByte *srcAllocation = nullptr;
    NVCVByte *dstAllocation = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&srcAllocation), kSampleStride));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&dstAllocation), kSampleStride));

    auto wrapTensor = [](NVCVByte *allocation)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = allocation;
        buffer.strides[0] = kSampleStride;
        buffer.strides[1] = kRowStride;
        buffer.strides[2] = kChannels;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{1, kHeight, kWidth, kChannels}, "NHWC"},
                nvcv::TYPE_U8, buffer
        },
            nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                            }});
    };

    nvcv::Tensor src = wrapTensor(srcAllocation);
    nvcv::Tensor dst = wrapTensor(dstAllocation);

    std::vector<uint8_t> srcBytes(kSampleStride, 0xC7);
    for (int y = 0; y < kHeight; ++y)
    {
        for (int x = 0; x < kWidth; ++x)
        {
            for (int c = 0; c < kChannels; ++c)
            {
                srcBytes[static_cast<size_t>(y) * kRowStride + static_cast<size_t>(x) * kChannels + c]
                    = static_cast<uint8_t>(17 + y * 13 + x * 3 + c);
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcAllocation, srcBytes.data(), srcBytes.size(), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstAllocation, 0xA5, kSampleStride));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::RandomResizedCrop op(1.0, 1.0, 1.0, 1.0, 1, 1);
    EXPECT_NO_THROW(op(stream, src, dst, NVCV_INTERP_LINEAR));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> dstBytes(kSampleStride);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstBytes.data(), dstAllocation, dstBytes.size(), cudaMemcpyDeviceToHost));

    for (int y = 0; y < kHeight; ++y)
    {
        const size_t rowStart = static_cast<size_t>(y) * kRowStride;
        for (size_t byte = 0; byte < kLogicalRowBytes; ++byte)
        {
            EXPECT_EQ(srcBytes[rowStart + byte], dstBytes[rowStart + byte]) << "row " << y << ", byte " << byte;
        }
        EXPECT_EQ(static_cast<uint8_t>(0xA5), dstBytes[rowStart + kLogicalRowBytes]) << "row " << y;
    }
}

// unique format test

// =============================================================================
// Regressions
//
// Each of these asserts a rejection rather than comparing pixels. The defects below are all
// out-of-bounds accesses that the submit reported as NVCV_SUCCESS; the reads and writes land in
// mapped memory just past a row or a sample and do not reliably fault, so only compute-sanitizer
// would see the corruption. The missing precondition, by contrast, is observable from the public
// API, which is what these pin.
// =============================================================================

TEST(OpRandomResizedCrop_Negative, linear_interp_requires_2x2_source)
{
    // The bug: the linear kernels derive a source row from the crop window, clamp it with
    // sy = max(0, min(sy, height - 2)), and then unconditionally read both row sy and row sy + 1
    // (and columns sx / sx + 1).  On a source with fewer than two rows or columns, height - 2 is
    // negative, the clamp collapses to 0, and the kernel reads one row (or column) past the image --
    // past the whole allocation when the tensor holds a single sample.  The submit reported
    // NVCV_SUCCESS, so nothing downstream could notice.  Resize carries the same kernel shape and
    // was fixed for it; RandomResizedCrop was not.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 1, 1);

    nvcv::Tensor dst = nvcv::util::CreateTensor(1, 16, 16, nvcv::FMT_U8);

    nvcv::Tensor src1x1 = nvcv::util::CreateTensor(1, 1, 1, nvcv::FMT_U8);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, src1x1, dst, NVCV_INTERP_LINEAR); }));

    // Width == 1 with height > 1 is also rejected.
    nvcv::Tensor src1xN = nvcv::util::CreateTensor(1, 1, 8, nvcv::FMT_U8);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, src1xN, dst, NVCV_INTERP_LINEAR); }));

    // Height == 1 with width > 1 is also rejected.
    nvcv::Tensor srcNx1 = nvcv::util::CreateTensor(1, 8, 1, nvcv::FMT_U8);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, srcNx1, dst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCropVarshape_Negative, linear_interp_requires_2x2_source)
{
    // Var-shape twin of the tensor case above; here the offending image can be any member of the
    // batch, so the precondition is checked per image.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batchSrc(2);
    nvcv::ImageBatchVarShape batchDst(2);
    batchSrc.pushBack(nvcv::Image({8, 8}, nvcv::FMT_U8));
    batchSrc.pushBack(nvcv::Image({1, 1}, nvcv::FMT_U8));
    batchDst.pushBack(nvcv::Image({16, 16}, nvcv::FMT_U8));
    batchDst.pushBack(nvcv::Image({16, 16}, nvcv::FMT_U8));

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 2, 1);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, batchSrc, batchDst, NVCV_INTERP_LINEAR); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCrop_Negative, tensorSampleCountMismatch)
{
    // The bug: OpRandomResizedCrop.h documents "Input/Output dependency ... Number | Yes", but the
    // tensor path only ever compared layouts, which carry no sample count -- an NHWC pair with
    // different N passed validation.  Crop parameters were then drawn and staged for the *input's*
    // sample count while the kernel grid took its z extent from the *output*, so every sample past
    // the input's last one read crop parameters from uninitialized staging memory and addressed
    // source rows outside the input tensor.
    nvcv::Tensor imgSrc(
        {
            {2, 24, 24, 4},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgDst(
        {
            {3, 24, 24, 4},
            "NHWC"
    },
        nvcv::TYPE_U8);

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 4, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(nullptr, imgSrc, imgDst, NVCV_INTERP_NEAREST); }));
}

TEST(OpRandomResizedCrop_Negative, tensorChannelCountMismatch)
{
    // The bug: OpRandomResizedCrop.h documents "Input/Output dependency ... Channels | Yes", but the
    // tensor path took the channel count from the input alone and used it to pick the kernel's pixel
    // type for *both* tensors.  A 4-channel input against a 3-channel output selected the uchar4
    // kernel, which then wrote 4 bytes per pixel into rows sized for 3 -- past the end of the last
    // row of the last sample.  The layout comparison did not catch it because "HWC" is the same
    // layout at either channel count.
    nvcv::Tensor imgSrc(
        {
            {24, 24, 4},
            "HWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgDst(
        {
            {24, 24, 3},
            "HWC"
    },
        nvcv::TYPE_U8);

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 4, 1);

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(nullptr, imgSrc, imgDst, NVCV_INTERP_NEAREST); }));
}

TEST(OpRandomResizedCropVarshape_Negative, channelCountMismatch)
{
    // The var-shape sibling of tensorChannelCountMismatch, and it survived the same way. The two
    // batches are compared through ClassifyDataFormat, which only reports interleaved-vs-planar and
    // 1-vs-many images, and through the element type, which for RGBA8 and RGB8 is "8-bit unsigned"
    // on both sides. Neither carries the channel count, so an RGBA8 input against an RGB8 output
    // passed validation, selected the uchar4 kernel from the *input* format, and wrote 4 bytes per
    // pixel into rows sized for 3.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batchSrc(2);
    nvcv::ImageBatchVarShape batchDst(2);
    for (int i = 0; i < 2; ++i)
    {
        batchSrc.pushBack(nvcv::Image({24, 24}, nvcv::FMT_RGBA8));
        batchDst.pushBack(nvcv::Image({12, 12}, nvcv::FMT_RGB8));
    }

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 4, 1);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, batchSrc, batchDst, NVCV_INTERP_NEAREST); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCropVarshape_Negative, imageCountMismatch)
{
    // Documents a third rejection this operator did not have. Unlike the two cases above this one
    // is not a silent NVCV_SUCCESS: a batch pair where one side has 1 image and the other has 2+
    // was already caught (the format code encodes "N"-prefixed vs not), and any other mismatch ran
    // into NVCV_ASSERT(in.numImages() == out.numImages()) inside the launch helper, which abort()s
    // -- NVCV_DEBUG is on in release builds. So the pre-fix behavior was a process abort or an
    // incidental out-of-range index error, not a wrong answer, and this test pins the contract
    // rather than reproducing a silent corruption.
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::RandomResizedCrop randomResizedCropOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 8, 1);

    nvcv::ImageBatchVarShape batchSrc(2);
    nvcv::ImageBatchVarShape batchDst(3);
    for (int i = 0; i < 2; ++i)
    {
        batchSrc.pushBack(nvcv::Image({24, 24}, nvcv::FMT_RGBA8));
    }
    for (int i = 0; i < 3; ++i)
    {
        batchDst.pushBack(nvcv::Image({12, 12}, nvcv::FMT_RGBA8));
    }
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { randomResizedCropOp(stream, batchSrc, batchDst, NVCV_INTERP_NEAREST); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpRandomResizedCrop, hostClassificationEdgeCasesAndCropFallback)
{
    cvcuda::RandomResizedCrop validationOp(0.08, 1.0, 3.0 / 4.0, 4.0 / 3.0, 1, 7);

    auto expectInvalidTensorType = [&validationOp](nvcv::DataType dtype)
    {
        nvcv::Tensor imgSrc(
            {
                {8, 8, dtype.numChannels()},
                "HWC"
        },
            dtype);
        nvcv::Tensor imgDst(
            {
                {8, 8, dtype.numChannels()},
                "HWC"
        },
            dtype);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { validationOp(nullptr, imgSrc, imgDst, NVCV_INTERP_NEAREST); }));
    };

    expectInvalidTensorType(nvcv::TYPE_S8);
    expectInvalidTensorType(nvcv::TYPE_S32);
    expectInvalidTensorType(nvcv::TYPE_S64);
    expectInvalidTensorType(nvcv::TYPE_C64);
    expectInvalidTensorType(nvcv::DataType(nvcv::DataKind::FLOAT, nvcv::Packing::X8));
    expectInvalidTensorType(nvcv::DataType(nvcv::DataKind::UNSPECIFIED, nvcv::Packing::X8));
    expectInvalidTensorType(nvcv::DataType(nvcv::DataKind::UNSIGNED, nvcv::Packing::X5Y5Z6));

    nvcv::Tensor invalidLayoutSrc(
        {
            {8, 8},
            "HW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor invalidLayoutDst(
        {
            {8, 8},
            "HW"
    },
        nvcv::TYPE_U8);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&] { validationOp(nullptr, invalidLayoutSrc, invalidLayoutDst, NVCV_INTERP_NEAREST); }));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto runSupportedTensorType = [&validationOp, stream](nvcv::DataType dtype)
    {
        nvcv::Tensor imgSrc(
            {
                {8, 8, dtype.numChannels()},
                "HWC"
        },
            dtype);
        nvcv::Tensor imgDst(
            {
                {8, 8, dtype.numChannels()},
                "HWC"
        },
            dtype);

        EXPECT_NO_THROW(validationOp(stream, imgSrc, imgDst, NVCV_INTERP_NEAREST));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    };

    runSupportedTensorType(nvcv::TYPE_S16);
    runSupportedTensorType(nvcv::TYPE_U16);

    auto expectInvalidVarShapeFormat = [&validationOp, stream](nvcv::ImageFormat format)
    {
        nvcv::ImageBatchVarShape batchSrc(1);
        nvcv::ImageBatchVarShape batchDst(1);
        batchSrc.pushBack(nvcv::Image({8, 8}, format));
        batchDst.pushBack(nvcv::Image({8, 8}, format));

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { validationOp(stream, batchSrc, batchDst, NVCV_INTERP_NEAREST); }));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    };

    expectInvalidVarShapeFormat(nvcv::FMT_NV12);
    expectInvalidVarShapeFormat(nvcv::FMT_NV24);

    const std::vector<nvcv::Size2D> srcSizes{
        { 8, 16},
        {16,  8},
        { 8,  8}
    };
    const std::vector<nvcv::Size2D> dstSizes{
        { 8, 11},
        {11,  8},
        { 8,  8}
    };
    std::vector<nvcv::Image>          srcImages;
    std::vector<nvcv::Image>          dstImages;
    std::vector<std::vector<uint8_t>> expected;

    for (int i = 0; i < static_cast<int>(srcSizes.size()); ++i)
    {
        srcImages.emplace_back(srcSizes[i], nvcv::FMT_U8);
        dstImages.emplace_back(dstSizes[i], nvcv::FMT_U8);

        const int            srcRowStride = srcSizes[i].w;
        std::vector<uint8_t> srcValues(srcSizes[i].w * srcSizes[i].h);
        for (int y = 0; y < srcSizes[i].h; ++y)
        {
            for (int x = 0; x < srcSizes[i].w; ++x)
            {
                srcValues[y * srcRowStride + x] = static_cast<uint8_t>(17 + i * 31 + y * 5 + x);
            }
        }

        const auto srcData = srcImages.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(srcData);
        ASSERT_EQ(1, srcData->numPlanes());
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcValues.data(),
                                            srcRowStride, srcRowStride, srcSizes[i].h, cudaMemcpyHostToDevice));

        const int top        = i == 0 ? 2 : 0;
        const int left       = i == 1 ? 2 : 0;
        const int cropHeight = dstSizes[i].h;
        const int cropWidth  = dstSizes[i].w;
        expected.emplace_back(cropWidth * cropHeight);
        for (int y = 0; y < cropHeight; ++y)
        {
            for (int x = 0; x < cropWidth; ++x)
            {
                expected.back()[y * cropWidth + x] = srcValues[(top + y) * srcRowStride + left + x];
            }
        }
    }

    nvcv::ImageBatchVarShape batchSrc(3);
    nvcv::ImageBatchVarShape batchDst(3);
    batchSrc.pushBack(srcImages.begin(), srcImages.end());
    batchDst.pushBack(dstImages.begin(), dstImages.end());

    cvcuda::RandomResizedCrop fallbackOp(2.0, 2.0, 3.0 / 4.0, 4.0 / 3.0, 3, 19);
    EXPECT_NO_THROW(fallbackOp(stream, batchSrc, batchDst, NVCV_INTERP_NEAREST));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < static_cast<int>(dstImages.size()); ++i)
    {
        const auto dstData = dstImages[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_TRUE(dstData);
        ASSERT_EQ(1, dstData->numPlanes());

        std::vector<uint8_t> actual(dstSizes[i].w * dstSizes[i].h);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(actual.data(), dstSizes[i].w, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstSizes[i].w, dstSizes[i].h, cudaMemcpyDeviceToHost));
        EXPECT_EQ(expected[i], actual);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
