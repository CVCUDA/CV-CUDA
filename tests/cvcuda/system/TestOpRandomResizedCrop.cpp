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
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

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
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
        RunTensorCorrectOutput(srcWidth, srcHeight, dstWidth, dstHeight, interpolation, numberImages, nvcv::FMT_U8);
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
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

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
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
        RunVarShapeCorrectOutput(srcWidthBase, srcHeightBase, dstWidthBase, dstHeightBase, interpolation, numberImages,
                                 nvcv::FMT_U8);
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
    // invalid data type
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

// unique format test
