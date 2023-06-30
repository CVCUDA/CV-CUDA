/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "ResizeUtils.hpp"

#include <common/InterpUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpRandomResizedCrop.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/MathWrappers.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <util/TensorDataUtils.hpp>

#include <cmath>
#include <random>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;
namespace t    = ::testing;

static void GetCropParams(std::mt19937 &generator, double minScale, double maxScale, double minRatio, double maxRatio,
                          int input_rows, int input_cols, int *top_indices, int *left_indices, int *crop_rows,
                          int *crop_cols)
{
    int          rows          = input_rows;
    int          cols          = input_cols;
    double       area          = rows * cols;
    const double log_min_ratio = std::log(minRatio);
    const double log_max_ratio = std::log(maxRatio);

    std::uniform_real_distribution<double> scale_dist(minScale, maxScale);
    std::uniform_real_distribution<double> ratio_dist(log_min_ratio, log_max_ratio);
    bool                                   got_params = false;
    for (int i = 0; i < 10; ++i)
    {
        if (got_params)
            return;
        int    target_area  = area * scale_dist(generator);
        double aspect_ratio = std::exp(ratio_dist(generator));

        *crop_cols = int(std::round(std::sqrt(target_area * aspect_ratio)));
        *crop_rows = int(std::round(std::sqrt(target_area / aspect_ratio)));

        if (*crop_cols > 0 && *crop_cols <= cols && *crop_rows > 0 && *crop_rows <= rows)
        {
            std::uniform_int_distribution<int> row_uni(0, rows - *crop_rows);
            std::uniform_int_distribution<int> col_uni(0, cols - *crop_cols);
            *top_indices  = row_uni(generator);
            *left_indices = col_uni(generator);
            got_params    = true;
        }
    }
    // Fallback to central crop
    if (!got_params)
    {
        double in_ratio = double(cols) / double(rows);
        if (in_ratio < minRatio)
        {
            *crop_cols = cols;
            *crop_rows = int(std::round(*crop_cols / minRatio));
        }
        else if (in_ratio > maxRatio)
        {
            *crop_rows = rows;
            *crop_cols = int(std::round(*crop_rows * maxRatio));
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

TEST_P(OpRandomResizedCrop, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4;
    double maxRatio = 4.0 / 3;

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

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
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

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

        int top, left, crop_rows, crop_cols;
        GetCropParams(generator, minScale, maxScale, minRatio, maxRatio, srcHeight, srcWidth, &top, &left, &crop_rows,
                      &crop_cols);

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);

        // Generate gold result
        test::ResizedCrop(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride,
                          {srcWidth, srcHeight}, top, left, crop_rows, crop_cols, fmt, interpolation);

        // maximum absolute error
        std::vector<int> mae(testVec.size());
        for (size_t i = 0; i < mae.size(); ++i)
        {
            mae[i] = abs(static_cast<int>(goldVec[i]) - static_cast<int>(testVec[i]));
        }

        int maeThreshold = 1;

        EXPECT_THAT(mae, t::Each(t::Le(maeThreshold)));
    }
}

TEST_P(OpRandomResizedCrop, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();
    int dstWidthBase  = GetParamValue<2>();
    int dstHeightBase = GetParamValue<3>();

    double minScale = 0.08;
    double maxScale = 1.0;
    double minRatio = 3.0 / 4;
    double maxRatio = 4.0 / 3;

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);

    std::uniform_int_distribution<int> rndDstWidth(dstWidthBase * 0.8, dstWidthBase * 1.1);
    std::uniform_int_distribution<int> rndDstHeight(dstHeightBase * 0.8, dstHeightBase * 1.1);

    std::vector<nvcv::Image> imgSrc, imgDst;
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
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

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

        int top, left, crop_rows, crop_cols;
        GetCropParams(generator, minScale, maxScale, minRatio, maxRatio, srcHeight, srcWidth, &top, &left, &crop_rows,
                      &crop_cols);

        std::vector<uint8_t> goldVec(dstHeight * dstRowStride);

        // Generate gold result
        test::ResizedCrop(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride[i],
                          {srcWidth, srcHeight}, top, left, crop_rows, crop_cols, fmt, interpolation);

        // maximum absolute error
        std::vector<int> mae(testVec.size());
        for (size_t i = 0; i < mae.size(); ++i)
        {
            mae[i] = abs(static_cast<int>(goldVec[i]) - static_cast<int>(testVec[i]));
        }

        int maeThreshold = 1;

        EXPECT_THAT(mae, t::Each(t::Le(maeThreshold)));
    }
}
