/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpBilateralFilter.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>
#include <vector>

#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)

namespace gt   = ::testing;
namespace test = nvcv::test;

static uint32_t saturate_cast(float n)
{
    return static_cast<uint32_t>(std::min(255.0f, std::round(n)));
}

static bool CompareImages(uint8_t *pTest, uint8_t *pGold, size_t columns, size_t rows, size_t rowStride,
                          size_t channels, float delta)
{
    for (size_t j = 0; j < rows; j++)
    {
        for (size_t k = 0; k < columns; k++)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                size_t offset = j * rowStride + k * channels + c;
                float  diff   = std::abs(static_cast<float>(pTest[offset]) - static_cast<float>(pGold[offset]));
                if (diff > delta)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool CompareTensors(std::vector<uint8_t> &vTest, std::vector<uint8_t> &vGold, size_t columns, size_t rows,
                           size_t batch, size_t rowStride, size_t channels, size_t sampleStride, float delta)
{
    for (size_t i = 0; i < batch; i++)
    {
        uint8_t *pTest = vTest.data() + i * sampleStride;
        uint8_t *pGold = vGold.data() + i * sampleStride;
        if (!CompareImages(pTest, pGold, columns, rows, rowStride, channels, delta))
            return false;
    }
    return true;
}

static bool CompareVarShapes(std::vector<std::vector<uint8_t>> &vTest, std::vector<std::vector<uint8_t>> &vGold,
                             std::vector<int> &vColumns, std::vector<int> &vRows, std::vector<int> &vRowStride,
                             std::vector<int> &vChannels, float delta)
{
    for (size_t i = 0; i < vTest.size(); i++)
    {
        if (!CompareImages(vTest[i].data(), vGold[i].data(), vColumns[i], vRows[i], vRowStride[i], vChannels[i], delta))
            return false;
    }
    return true;
}

static void CPUBilateralFilter(uint8_t *pIn, uint8_t *pOut, int columns, int rows, int rowStride, int channels,
                               int radius, float colorCoefficient, float spaceCoefficient)
{
    float radiusSquared = radius * radius;
    for (int j = 0; j < rows; j++)
    {
        for (int k = 0; k < columns; k++)
        {
            std::vector<float> numerators(channels, 0.0f);
            float              denominator = 0.0f;
            std::vector<float> centers{static_cast<float>(pIn[j * rowStride + k * channels]),
                                       channels > 1 ? static_cast<float>(pIn[j * rowStride + k * channels + 1]) : 0,
                                       channels > 2 ? static_cast<float>(pIn[j * rowStride + k * channels + 2]) : 0,
                                       channels > 3 ? static_cast<float>(pIn[j * rowStride + k * channels + 3]) : 0};

            for (int y = j - radius; y <= j + radius; y++)
            {
                for (int x = k - radius; x <= k + radius; x++)
                {
                    float distanceSquared = (k - x) * (k - x) + (j - y) * (j - y);
                    if (distanceSquared <= radiusSquared)
                    {
                        std::vector<float> pixels;
                        for (auto c = 0; c < channels; ++c)
                        {
                            float pixel = ((x >= 0) && (x < columns) && (y >= 0) && (y < rows))
                                            ? static_cast<float>(pIn[y * rowStride + x * channels + c])
                                            : 0.0f;
                            pixels.emplace_back(pixel);
                        }
                        float e_space = distanceSquared * spaceCoefficient;
                        float e_color = 0.0f;

                        for (auto c = 0; c < channels; ++c)
                        {
                            e_color += std::abs(pixels[c] - centers[c]);
                        }
                        e_color = e_color * e_color * colorCoefficient;

                        float weight = std::exp(e_space + e_color);
                        denominator += weight;
                        for (auto c = 0; c < channels; ++c)
                        {
                            numerators[c] += weight * pixels[c];
                        }
                    }
                }
            }

            for (auto c = 0; c < channels; ++c)
            {
                pOut[j * rowStride + k * channels + c] = saturate_cast(numerators[c] / denominator);
            }
        }
    }
}

static void CPUBilateralFilterTensor(std::vector<uint8_t> &vIn, std::vector<uint8_t> &vOut, int columns, int rows,
                                     int batch, int rowStride, int channels, int sampleStride, int diameter,
                                     float sigmaColor, float sigmaSpace)
{
    int   radius           = diameter / 2;
    float spaceCoefficient = -1 / (2 * sigmaSpace * sigmaSpace);
    float colorCoefficient = -1 / (2 * sigmaColor * sigmaColor);
    for (int i = 0; i < batch; i++)
    {
        uint8_t *pIn  = vIn.data() + i * sampleStride;
        uint8_t *pOut = vOut.data() + i * sampleStride;
        CPUBilateralFilter(pIn, pOut, columns, rows, rowStride, channels, radius, colorCoefficient, spaceCoefficient);
    }
}

static void CPUBilateralFilterVarShape(std::vector<std::vector<uint8_t>> &vIn, std::vector<std::vector<uint8_t>> &vOut,
                                       std::vector<int> &vColumns, std::vector<int> &vRows,
                                       std::vector<int> &vRowStride, std::vector<int> &vChannels,
                                       std::vector<int> &vDiameter, std::vector<float> &vSigmaColor,
                                       std::vector<float> &vSigmaSpace)
{
    for (size_t i = 0; i < vIn.size(); i++)
    {
        int   radius           = vDiameter[i] / 2;
        float spaceCoefficient = -1 / (2 * vSigmaSpace[i] * vSigmaSpace[i]);
        float colorCoefficient = -1 / (2 * vSigmaColor[i] * vSigmaColor[i]);
        CPUBilateralFilter(vIn[i].data(), vOut[i].data(), vColumns[i], vRows[i], vRowStride[i], vChannels[i], radius,
                           colorCoefficient, spaceCoefficient);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpBilateralFilter, test::ValueList<int, int, int, float, float, int>
{
    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          1},
    {    48,     32, 4, 5,          3,          1},
    {    64,     32, 4, 5,          3,          1},
    {    32,    128, 4, 5,          3,          1},

    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          5},
    {    12,    32,  4, 5,          3,          5},
    {    64,    32,  4, 5,          3,          5},
    {    32,    128, 4, 5,          3,          5},

    //width, height, d, SigmaCol or, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          9},
    {    48,     32, 4, 5,          3,          9},
    {    64,     32, 4, 5,          3,          9},
    {    32,    128, 4, 5,          3,          9},
});

// clang-format on
TEST_P(OpBilateralFilter, BilateralFilter_packed)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int                            width          = GetParamValue<0>();
    int                            height         = GetParamValue<1>();
    int                            d              = GetParamValue<2>();
    float                          sigmaColor     = GetParamValue<3>();
    float                          sigmaSpace     = GetParamValue<4>();
    int                            numberOfImages = GetParamValue<5>();
    std::vector<nvcv::ImageFormat> fmts{nvcv::FMT_U8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_2U8}, nvcv::FMT_RGB8,
                                        nvcv::FMT_RGBA8};
    for (nvcv::ImageFormat fmt : fmts)
    {
        nvcv::Tensor imgOut   = nvcv::util::CreateTensor(numberOfImages, width, height, fmt);
        nvcv::Tensor imgIn    = nvcv::util::CreateTensor(numberOfImages, width, height, fmt);
        const int    channels = fmt.numChannels();

        auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();

        ASSERT_NE(nullptr, inData);
        ASSERT_NE(nullptr, outData);

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        ASSERT_TRUE(inAccess);

        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        ASSERT_TRUE(outAccess);

        int inSampleStride  = inAccess->numRows() * inAccess->rowStride();
        int outSampleStride = outAccess->numRows() * outAccess->rowStride();

        int inBufSize  = inSampleStride * inAccess->numSamples();
        int outBufSize = outSampleStride * outAccess->numSamples();

        std::vector<uint8_t> vIn(inBufSize);
        std::vector<uint8_t> vOut(outBufSize);

        std::vector<uint8_t> inGold(inBufSize, 0);
        std::vector<uint8_t> outGold(outBufSize, 0);
        for (int i = 0; i < inBufSize; i++) inGold[i] = i % 113; // Use prime number to prevent weird tiling patterns

        EXPECT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inGold.data(), inBufSize, cudaMemcpyHostToDevice));
        CPUBilateralFilterTensor(inGold, outGold, inAccess->numCols(), inAccess->numRows(), inAccess->numSamples(),
                                 inAccess->rowStride(), channels, inSampleStride, d, sigmaColor, sigmaSpace);

        // run operator
        cvcuda::BilateralFilter bilateralFilterOp;

        EXPECT_NO_THROW(bilateralFilterOp(stream, imgIn, imgOut, d, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT));

        // check cdata
        std::vector<uint8_t> outTest(outBufSize);

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaMemcpy(outTest.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));
        ASSERT_TRUE(CompareTensors(outTest, outGold, inAccess->numCols(), inAccess->numRows(), inAccess->numSamples(),
                                   inAccess->rowStride(), channels, inSampleStride, 0.9f));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpBilateralFilter, BilateralFilter_VarShape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int                            width          = GetParamValue<0>();
    int                            height         = GetParamValue<1>();
    int                            diameter       = GetParamValue<2>();
    float                          sigmaColor     = GetParamValue<3>();
    float                          sigmaSpace     = GetParamValue<4>();
    int                            numberOfImages = GetParamValue<5>();
    std::vector<nvcv::ImageFormat> fmts{nvcv::FMT_U8, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_2U8}, nvcv::FMT_RGB8,
                                        nvcv::FMT_RGBA8};
    for (nvcv::ImageFormat fmt : fmts)
    {
        // Create input varshape
        std::default_random_engine         rng;
        std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
        std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

        std::vector<nvcv::Image> imgSrc;

        std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
        std::vector<int>                  srcVecRowStride(numberOfImages);
        std::vector<int>                  srcVecRows(numberOfImages);
        std::vector<int>                  srcVecColumns(numberOfImages);
        std::vector<int>                  channelsVec(numberOfImages);
        std::vector<std::vector<uint8_t>> goldVec(numberOfImages);
        std::vector<std::vector<uint8_t>> dstVec(numberOfImages);
        for (int i = 0; i < numberOfImages; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
            int srcRowStride   = imgSrc[i].size().w * fmt.planePixelStrideBytes(0);
            srcVecRowStride[i] = srcRowStride;
            srcVecRows[i]      = imgSrc[i].size().h;
            srcVecColumns[i]   = imgSrc[i].size().w;
            channelsVec[i]     = fmt.numChannels();
            std::uniform_int_distribution<uint8_t> udist(0, 255);

            srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
            goldVec[i].resize(imgSrc[i].size().h * srcRowStride);
            dstVec[i].resize(imgSrc[i].size().h * srcRowStride);
            std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });
            std::generate(goldVec[i].begin(), goldVec[i].end(), [&]() { return 0; });
            std::generate(dstVec[i].begin(), dstVec[i].end(), [&]() { return 0; });
            auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(imgData, nvcv::NullOpt);

            // Copy input data to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride,
                                                     srcVec[i].data(), srcRowStride, srcRowStride, imgSrc[i].size().h,
                                                     cudaMemcpyHostToDevice, stream));
        }

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        // Create output varshape
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < numberOfImages; ++i)
        {
            imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
        }
        nvcv::ImageBatchVarShape batchDst(numberOfImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create diameter tensor
        std::vector<int> vDiameter(numberOfImages, diameter);
        nvcv::Tensor     diameterTensor({{numberOfImages}, "N"}, nvcv::TYPE_S32);
        {
            auto dev = diameterTensor.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_NE(dev, nullptr);

            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vDiameter.data(), vDiameter.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, stream));
        }

        // Create sigmaColor tensor
        std::vector<float> vSigmaColor(numberOfImages, sigmaColor);
        nvcv::Tensor       sigmaColorTensor({{numberOfImages}, "N"}, nvcv::TYPE_F32);
        {
            auto dev = sigmaColorTensor.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_NE(dev, nullptr);

            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vSigmaColor.data(),
                                                   vSigmaColor.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        // Create sigmaSpace tensor
        std::vector<float> vSigmaSpace(numberOfImages, sigmaSpace);
        nvcv::Tensor       sigmaSpaceTensor({{numberOfImages}, "N"}, nvcv::TYPE_F32);
        {
            auto dev = sigmaSpaceTensor.exportData<nvcv::TensorDataStridedCuda>();
            ASSERT_NE(dev, nullptr);

            ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vSigmaSpace.data(),
                                                   vSigmaSpace.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        // Create gold data
        CPUBilateralFilterVarShape(srcVec, goldVec, srcVecColumns, srcVecRows, srcVecRowStride, channelsVec, vDiameter,
                                   vSigmaColor, vSigmaSpace);

        // Run operator
        cvcuda::BilateralFilter bilateralFilterOp;
        EXPECT_NO_THROW(bilateralFilterOp(stream, batchSrc, batchDst, diameterTensor, sigmaColorTensor,
                                          sigmaSpaceTensor, NVCV_BORDER_CONSTANT));

        // Retrieve data from GPU
        for (int i = 0; i < numberOfImages; i++)
        {
            auto imgData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(imgData, nvcv::NullOpt);

            // Copy input data to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(dstVec[i].data(), srcVecRowStride[i], imgData->plane(0).basePtr,
                                                     imgData->plane(0).rowStride, srcVecRowStride[i],
                                                     imgDst[i].size().h, cudaMemcpyDeviceToHost, stream));
        }
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Compare data
        ASSERT_TRUE(CompareVarShapes(dstVec, goldVec, srcVecColumns, srcVecRows, srcVecRowStride, channelsVec, 0.9f));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

#undef NVCV_IMAGE_FORMAT_2U8
