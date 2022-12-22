/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/ValueTests.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <cmath>
#include <random>

namespace test = nvcv::test;
namespace t    = ::testing;

static void Resize(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
                   int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, NVCVInterpolationType interpolation)
{
    double iScale = static_cast<double>(srcSize.h) / dstSize.h;
    double jScale = static_cast<double>(srcSize.w) / dstSize.w;

    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t       *dstPtr = hDst.data();
    const uint8_t *srcPtr = hSrc.data();

    for (int di = 0; di < dstSize.h; di++)
    {
        for (int dj = 0; dj < dstSize.w; dj++)
        {
            if (interpolation == NVCV_INTERP_NEAREST)
            {
                double fi = iScale * di;
                double fj = jScale * dj;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                si = std::min(si, srcSize.h - 1);
                sj = std::min(sj, srcSize.w - 1);

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k]
                        = srcPtr[si * srcRowStride + sj * elementsPerPixel + k];
                }
            }
            else if (interpolation == NVCV_INTERP_LINEAR)
            {
                double fi = iScale * (di + 0.5) - 0.5;
                double fj = jScale * (dj + 0.5) - 0.5;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                fi -= si;
                fj -= sj;

                fj = (sj < 0 || sj >= srcSize.w - 1) ? 0 : fj;

                si = std::max(0, std::min(si, srcSize.h - 2));
                sj = std::max(0, std::min(sj, srcSize.w - 2));

                double iWeights[2] = {1 - fi, fi};
                double jWeights[2] = {1 - fj, fj};

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    double res = std::rint(
                        srcPtr[(si + 0) * srcRowStride + (sj + 0) * elementsPerPixel + k] * iWeights[0] * jWeights[0]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * iWeights[1] * jWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 1) * elementsPerPixel + k] * iWeights[0] * jWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * iWeights[1]
                              * jWeights[1]);

                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
            else if (interpolation == NVCV_INTERP_CUBIC)
            {
                double fi = iScale * (di + 0.5) - 0.5;
                double fj = jScale * (dj + 0.5) - 0.5;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                fi -= si;
                fj -= sj;

                fj = (sj < 1 || sj >= srcSize.w - 3) ? 0 : fj;

                si = std::max(1, std::min(si, srcSize.h - 3));
                sj = std::max(1, std::min(sj, srcSize.w - 3));

                const double A = -0.75;
                double       iWeights[4];
                iWeights[0] = ((A * (fi + 1) - 5 * A) * (fi + 1) + 8 * A) * (fi + 1) - 4 * A;
                iWeights[1] = ((A + 2) * fi - (A + 3)) * fi * fi + 1;
                iWeights[2] = ((A + 2) * (1 - fi) - (A + 3)) * (1 - fi) * (1 - fi) + 1;
                iWeights[3] = 1 - iWeights[0] - iWeights[1] - iWeights[2];

                double jWeights[4];
                jWeights[0] = ((A * (fj + 1) - 5 * A) * (fj + 1) + 8 * A) * (fj + 1) - 4 * A;
                jWeights[1] = ((A + 2) * fj - (A + 3)) * fj * fj + 1;
                jWeights[2] = ((A + 2) * (1 - fj) - (A + 3)) * (1 - fj) * (1 - fj) + 1;
                jWeights[3] = 1 - jWeights[0] - jWeights[1] - jWeights[2];

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    double res = std::abs(std::rint(
                        srcPtr[(si - 1) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                              * iWeights[3]));

                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpResize, test::ValueList<int, int, int, int, NVCVInterpolationType, int>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,       interpolation, numberImages
    {        42,        48,       23,        24, NVCV_INTERP_NEAREST,           1},
    {       113,        12,      212,        36, NVCV_INTERP_NEAREST,           1},
    {       421,       148,      223,       124, NVCV_INTERP_NEAREST,           2},
    {       313,       212,      412,       336, NVCV_INTERP_NEAREST,           3},
    {        42,        40,       21,        20,  NVCV_INTERP_LINEAR,           1},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           1},
    {       420,       420,      210,       210,  NVCV_INTERP_LINEAR,           4},
    {       210,       210,      420,       420,  NVCV_INTERP_LINEAR,           5},
    {        42,        40,       21,        20,   NVCV_INTERP_CUBIC,           1},
    {        21,        21,       42,        42,   NVCV_INTERP_CUBIC,           6},
});

// clang-format on

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

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgSrc.exportData());

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
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    cvcuda::Resize resizeOp;
    EXPECT_NO_THROW(resizeOp(stream, imgSrc, imgDst, interpolation));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgDst.exportData());
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
        Resize(goldVec, dstVecRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride, {srcWidth, srcHeight}, fmt,
               interpolation);

        EXPECT_EQ(goldVec, testVec);
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

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);

    std::uniform_int_distribution<int> rndDstWidth(dstWidthBase * 0.8, dstWidthBase * 1.1);
    std::uniform_int_distribution<int> rndDstHeight(dstHeightBase * 0.8, dstHeightBase * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc, imgDst;
    for (int i = 0; i < numberOfImages; ++i)
    {
        imgSrc.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{rndSrcWidth(randEng), rndSrcHeight(randEng)}, fmt));
        imgDst.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{rndDstWidth(randEng), rndDstHeight(randEng)}, fmt));
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
        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
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

        const auto *srcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDst[i]->exportData());
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
        Resize(goldVec, dstRowStride, {dstWidth, dstHeight}, srcVec[i], srcVecRowStride[i], {srcWidth, srcHeight}, fmt,
               interpolation);

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
