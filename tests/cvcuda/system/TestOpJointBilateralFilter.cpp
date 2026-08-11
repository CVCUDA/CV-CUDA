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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpJointBilateralFilter.hpp>
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

using TensorDim = nvcv::TensorShape::DimType;

static uint8_t saturate_cast(float n)
{
    return static_cast<uint8_t>(std::min(255.0f, std::round(n)));
}

static bool CompareImages(const uint8_t *pTest, const uint8_t *pGold, size_t columns, size_t rows, size_t rowStride,
                          size_t channels, float delta)
{
    const size_t activeRowBytes = columns * channels;
    for (size_t j = 0; j < rows; j++)
    {
        const size_t rowOffset = j * rowStride;
        for (size_t k = 0; k < activeRowBytes; k++)
        {
            const size_t offset = rowOffset + k;
            float        diff   = std::abs(static_cast<float>(pTest[offset]) - static_cast<float>(pGold[offset]));
            if (diff > delta)
            {
                const size_t column  = k / channels;
                const size_t channel = k % channels;
                std::cout << " o = " << offset << " j = " << j << " k = " << column << " c = " << channel
                          << " rowS = " << rowStride << std::endl;
                std::cout << " test = " << static_cast<float>(pTest[offset])
                          << " gold = " << static_cast<float>(pGold[offset]) << std::endl;

                return false;
            }
        }
    }
    return true;
}

static bool CompareTensors(const std::vector<uint8_t> &vTest, const std::vector<uint8_t> &vGold, size_t columns,
                           size_t rows, size_t batch, size_t rowStride, size_t channels, size_t sampleStride,
                           float delta)
{
    for (size_t i = 0; i < batch; i++)
    {
        const uint8_t *pTest = vTest.data() + i * sampleStride;
        const uint8_t *pGold = vGold.data() + i * sampleStride;
        if (!CompareImages(pTest, pGold, columns, rows, rowStride, channels, delta))
            return false;
    }
    return true;
}

static bool CompareVarShapes(const std::vector<std::vector<uint8_t>> &vTest,
                             const std::vector<std::vector<uint8_t>> &vGold, const std::vector<int> &vColumns,
                             const std::vector<int> &vRows, const std::vector<int> &vRowStride,
                             const std::vector<int> &vChannels, float delta)
{
    for (size_t i = 0; i < vTest.size(); i++)
    {
        if (!CompareImages(vTest[i].data(), vGold[i].data(), vColumns[i], vRows[i], vRowStride[i], vChannels[i], delta))
        {
            return false;
        }
    }
    return true;
}

static float ReadPixel(const uint8_t *pIn, TensorDim x, TensorDim y, int c, TensorDim columns, TensorDim rows,
                       int rowStride, int channels)
{
    return ((x >= 0) && (x < columns) && (y >= 0) && (y < rows))
             ? static_cast<float>(pIn[y * rowStride + x * channels + c])
             : 0.0f;
}

static std::vector<float> ReadChannels(const uint8_t *pIn, TensorDim x, TensorDim y, TensorDim columns, TensorDim rows,
                                       int rowStride, int channels)
{
    std::vector<float> values(channels);

    for (int c = 0; c < channels; ++c)
    {
        values[c] = ReadPixel(pIn, x, y, c, columns, rows, rowStride, channels);
    }

    return values;
}

static void AccumulateJointBilateralSample(std::vector<float> &numerators, float &denominator,
                                           const std::vector<float> &centerColors, const uint8_t *pIn,
                                           const uint8_t *pInColor, TensorDim x, TensorDim y, TensorDim columns,
                                           TensorDim rows, int rowStride, int channels, float distanceSquared,
                                           float colorCoefficient, float spaceCoefficient)
{
    std::vector<float> pixels      = ReadChannels(pIn, x, y, columns, rows, rowStride, channels);
    std::vector<float> pixelColors = ReadChannels(pInColor, x, y, columns, rows, rowStride, channels);
    float              eColor      = 0.0f;

    for (int c = 0; c < channels; ++c)
    {
        eColor += std::abs(pixelColors[c] - centerColors[c]);
    }

    float weight = std::exp(distanceSquared * spaceCoefficient + eColor * eColor * colorCoefficient);
    denominator += weight;

    for (int c = 0; c < channels; ++c)
    {
        numerators[c] += weight * pixels[c];
    }
}

static void AccumulateJointBilateralWindow(std::vector<float> &numerators, float &denominator,
                                           const std::vector<float> &centerColors, const uint8_t *pIn,
                                           const uint8_t *pInColor, TensorDim column, TensorDim row, TensorDim columns,
                                           TensorDim rows, int rowStride, int channels, int radius, float radiusSquared,
                                           float colorCoefficient, float spaceCoefficient)
{
    for (TensorDim y = row - radius; y <= row + radius; y++)
    {
        for (TensorDim x = column - radius; x <= column + radius; x++)
        {
            auto distanceSquared = static_cast<float>((column - x) * (column - x) + (row - y) * (row - y));

            if (distanceSquared > radiusSquared)
            {
                continue;
            }

            AccumulateJointBilateralSample(numerators, denominator, centerColors, pIn, pInColor, x, y, columns, rows,
                                           rowStride, channels, distanceSquared, colorCoefficient, spaceCoefficient);
        }
    }
}

static void CPUJointBilateralFilter(const uint8_t *pIn, const uint8_t *pInColor, uint8_t *pOut, TensorDim columns,
                                    TensorDim rows, int rowStride, int channels, int radius, float colorCoefficient,
                                    float spaceCoefficient)
{
    auto radiusSquared = static_cast<float>(radius * radius);
    for (TensorDim j = 0; j < rows; j++)
    {
        for (TensorDim k = 0; k < columns; k++)
        {
            std::vector<float> numerators(channels, 0.0f);
            float              denominator  = 0;
            std::vector<float> centerColors = ReadChannels(pInColor, k, j, columns, rows, rowStride, channels);

            AccumulateJointBilateralWindow(numerators, denominator, centerColors, pIn, pInColor, k, j, columns, rows,
                                           rowStride, channels, radius, radiusSquared, colorCoefficient,
                                           spaceCoefficient);

            denominator = (denominator != 0) ? denominator : 1.0f;
            for (auto c = 0; c < channels; ++c)
            {
                pOut[j * rowStride + k * channels + c] = saturate_cast(numerators[c] / denominator);
            }
        }
    }
}

static void CPUJointBilateralFilterTensor(std::vector<uint8_t> &vIn, std::vector<uint8_t> &vInColor,
                                          std::vector<uint8_t> &vOut, TensorDim columns, TensorDim rows,
                                          TensorDim batch, int rowStride, int channels, int sampleStride, int diameter,
                                          float sigmaColor, float sigmaSpace)
{
    if (sigmaColor <= 0)
    {
        sigmaColor = 1;
    }
    if (sigmaSpace <= 0)
    {
        sigmaSpace = 1;
    }

    int radius;
    if (diameter <= 0)
    {
        radius = static_cast<int>(std::roundf(sigmaSpace * 1.5f));
    }
    else
    {
        radius = diameter / 2;
    }
    if (radius < 1)
    {
        radius = 1;
    }

    float spaceCoefficient = -1.f / (2.f * sigmaSpace * sigmaSpace);
    float colorCoefficient = -1.f / (2.f * sigmaColor * sigmaColor);
    for (TensorDim i = 0; i < batch; i++)
    {
        const uint8_t *pIn      = vIn.data() + i * sampleStride;
        const uint8_t *pInColor = vInColor.data() + i * sampleStride;
        uint8_t       *pOut     = vOut.data() + i * sampleStride;
        CPUJointBilateralFilter(pIn, pInColor, pOut, columns, rows, rowStride, channels, radius, colorCoefficient,
                                spaceCoefficient);
    }
}

static void CPUJointBilateralFilterVarShape(std::vector<std::vector<uint8_t>> &vIn,
                                            std::vector<std::vector<uint8_t>> &vInColor,
                                            std::vector<std::vector<uint8_t>> &vOut, std::vector<int> &vColumns,
                                            std::vector<int> &vRows, std::vector<int> &vRowStride,
                                            std::vector<int> &vChannels, std::vector<int> &vDiameter,
                                            std::vector<float> &vSigmaColor, std::vector<float> &vSigmaSpace)
{
    for (size_t i = 0; i < vIn.size(); i++)
    {
        float sigmaColor = vSigmaColor[i];
        float sigmaSpace = vSigmaSpace[i];
        int   diameter   = vDiameter[i];

        if (sigmaColor <= 0)
        {
            sigmaColor = 1;
        }
        if (sigmaSpace <= 0)
        {
            sigmaSpace = 1;
        }

        int radius;
        if (diameter <= 0)
        {
            radius = static_cast<int>(std::roundf(sigmaSpace * 1.5f));
        }
        else
        {
            radius = diameter / 2;
        }
        if (radius < 1)
        {
            radius = 1;
        }

        float spaceCoefficient = -1.f / (2.f * sigmaSpace * sigmaSpace);
        float colorCoefficient = -1.f / (2.f * sigmaColor * sigmaColor);
        CPUJointBilateralFilter(vIn[i].data(), vInColor[i].data(), vOut[i].data(), vColumns[i], vRows[i], vRowStride[i],
                                vChannels[i], radius, colorCoefficient, spaceCoefficient);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpJointBilateralFilter, test::ValueList<int, int, int, float, float, int>
{
    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          1},
    {    48,     32, 4, 5,          3,          1},
    {    64,     32, 4, 5,          3,          1},
    {    32,    128, 4, 5,          3,          1},
    {    32,    128, 4, 0,          3,          1},
    {    32,    128, 4, 5,          0,          1},
    {    32,    128, 0, 5,          3,          1},
    {    32,    128, 1, 5,          3,          1},

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

TEST_P(OpJointBilateralFilter, JointBilateralFilter_packed)
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

    for (nvcv::ImageFormat fmt : fmts) // NOSONAR
    {
        nvcv::Tensor imgOut     = nvcv::util::CreateTensor(numberOfImages, width, height, fmt);
        nvcv::Tensor imgIn      = nvcv::util::CreateTensor(numberOfImages, width, height, fmt);
        nvcv::Tensor imgInColor = nvcv::util::CreateTensor(numberOfImages, width, height, fmt);
        const int    channels   = fmt.numChannels();

        auto inData      = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto inColorData = imgInColor.exportData<nvcv::TensorDataStridedCuda>();
        auto outData     = imgOut.exportData<nvcv::TensorDataStridedCuda>();

        ASSERT_NE(nullptr, inData);
        ASSERT_NE(nullptr, inColorData);
        ASSERT_NE(nullptr, outData);

        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
        ASSERT_TRUE(inAccess);

        auto inColorAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inColorData);
        ASSERT_TRUE(inColorAccess);

        auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
        ASSERT_TRUE(outAccess);

        auto inSampleStride      = static_cast<int>(inAccess->numRows() * inAccess->rowStride());
        auto inColorSampleStride = static_cast<int>(inColorAccess->numRows() * inColorAccess->rowStride());
        auto outSampleStride     = static_cast<int>(outAccess->numRows() * outAccess->rowStride());

        int inBufSize      = inSampleStride * static_cast<int>(inAccess->numSamples());
        int inColorBufSize = inColorSampleStride * static_cast<int>(inColorAccess->numSamples());
        int outBufSize     = outSampleStride * static_cast<int>(outAccess->numSamples());

        std::vector<uint8_t> vIn(inBufSize);
        std::vector<uint8_t> vInColor(inColorBufSize);
        std::vector<uint8_t> vOut(outBufSize);

        std::vector<uint8_t> inGold(inBufSize, 0);
        std::vector<uint8_t> inColorGold(inColorBufSize, 0);
        std::vector<uint8_t> outGold(outBufSize, 0);
        for (int i = 0; i < inBufSize; i++) inGold[i] = i % 113; // Use prime number to prevent weird tiling patterns
        for (int i = 0; i < inColorBufSize; i++)
            inColorGold[i] = i % 109; // Use prime number to prevent weird tiling patterns
        EXPECT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inGold.data(), inBufSize, cudaMemcpyHostToDevice));
        EXPECT_EQ(cudaSuccess,
                  cudaMemcpy(inColorData->basePtr(), inColorGold.data(), inColorBufSize, cudaMemcpyHostToDevice));
        const int rowStride{static_cast<int>(inAccess->rowStride())};
        CPUJointBilateralFilterTensor(inGold, inColorGold, outGold, inAccess->numCols(), inAccess->numRows(),
                                      inAccess->numSamples(), rowStride, channels, inSampleStride, d, sigmaColor,
                                      sigmaSpace);

        // run operator
        cvcuda::JointBilateralFilter jointBilateralFilterOp;

        EXPECT_NO_THROW(
            jointBilateralFilterOp(stream, imgIn, imgInColor, imgOut, d, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT));

        // check cdata
        std::vector<uint8_t> outTest(outBufSize);

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaMemcpy(outTest.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));
        ASSERT_TRUE(CompareTensors(
            outTest, outGold, static_cast<size_t>(inAccess->numCols()), static_cast<size_t>(inAccess->numRows()),
            static_cast<size_t>(inAccess->numSamples()), static_cast<size_t>(inAccess->rowStride()),
            static_cast<size_t>(channels), static_cast<size_t>(inSampleStride), 0.9f));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpJointBilateralFilter, varshape_correct_output)
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

    for (nvcv::ImageFormat fmt : fmts) // NOSONAR
    {
        // Create input varshape
        std::default_random_engine    rng;
        std::uniform_int_distribution udistWidth(static_cast<int>(static_cast<double>(width) * 0.8),
                                                 static_cast<int>(static_cast<double>(width) * 1.1));
        std::uniform_int_distribution udistHeight(static_cast<int>(static_cast<double>(height) * 0.8),
                                                  static_cast<int>(static_cast<double>(height) * 1.1));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgSrcColor;

        std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
        std::vector<std::vector<uint8_t>> srcColorVec(numberOfImages);
        std::vector<int>                  srcVecRowStride(numberOfImages);
        std::vector<int>                  srcVecRows(numberOfImages);
        std::vector<int>                  srcVecColumns(numberOfImages);
        std::vector<int>                  channelsVec(numberOfImages);
        std::vector<std::vector<uint8_t>> goldVec(numberOfImages);
        std::vector<std::vector<uint8_t>> dstVec(numberOfImages);
        for (int i = 0; i < numberOfImages; ++i)
        {
            int          w = udistWidth(rng);
            int          h = udistHeight(rng);
            nvcv::Size2D sz(w, h);
            imgSrc.emplace_back(sz, fmt);
            imgSrcColor.emplace_back(sz, fmt);
            int srcRowStride   = imgSrc[i].size().w * fmt.planePixelStrideBytes(0);
            srcVecRowStride[i] = srcRowStride;
            srcVecRows[i]      = imgSrc[i].size().h;
            srcVecColumns[i]   = imgSrc[i].size().w;
            channelsVec[i]     = fmt.numChannels();
            std::uniform_int_distribution<uint8_t> udist(0, 255);

            srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
            srcColorVec[i].resize(imgSrcColor[i].size().h * srcRowStride);
            goldVec[i].resize(imgSrc[i].size().h * srcRowStride);
            dstVec[i].resize(imgSrc[i].size().h * srcRowStride);
            std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });
            std::ranges::generate(srcColorVec[i], [&udist, &rng]() { return udist(rng); });
            std::ranges::generate(goldVec[i], []() { return 0; });
            std::ranges::generate(dstVec[i], []() { return 0; });
            auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(imgData, nvcv::NullOpt);
            auto imgColorData = imgSrcColor[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(imgColorData, nvcv::NullOpt);

            // Copy input data to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride,
                                                     srcVec[i].data(), srcRowStride, srcRowStride, imgSrc[i].size().h,
                                                     cudaMemcpyHostToDevice, stream));

            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(imgColorData->plane(0).basePtr, imgColorData->plane(0).rowStride,
                                                     srcColorVec[i].data(), srcRowStride, srcRowStride,
                                                     imgSrcColor[i].size().h, cudaMemcpyHostToDevice, stream));
        }

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        nvcv::ImageBatchVarShape batchSrcColor(numberOfImages);
        batchSrcColor.pushBack(imgSrcColor.begin(), imgSrcColor.end());

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
        CPUJointBilateralFilterVarShape(srcVec, srcColorVec, goldVec, srcVecColumns, srcVecRows, srcVecRowStride,
                                        channelsVec, vDiameter, vSigmaColor, vSigmaSpace);

        // Run operator
        cvcuda::JointBilateralFilter jointBilateralFilterOp;
        EXPECT_NO_THROW(jointBilateralFilterOp(stream, batchSrc, batchSrcColor, batchDst, diameterTensor,
                                               sigmaColorTensor, sigmaSpaceTensor, NVCV_BORDER_CONSTANT));

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
        ASSERT_TRUE(CompareVarShapes(dstVec, goldVec, srcVecColumns, srcVecRows, srcVecRowStride, channelsVec, 1.0f));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunJointBilateralTensorPlanarParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                                NVCVBorderType border)
{
    cvcuda::JointBilateralFilter op;
    nvcv::test::planar::RunTensorParity(
        planarFmt, interleavedFmt, 33, 25, 33, 25, 2,
        [&op, border](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, src, dst, 5, 15.f, 3.f, border); });
}

static void RunJointBilateralVarShapePlanarParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                                  NVCVBorderType border)
{
    cvcuda::JointBilateralFilter op;
    nvcv::test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, 31, 23, 31, 23, 2,
        [&op, border](cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                      nvcv::ImageFormat)
        {
            const int numImages = src.numImages();

            auto diameter   = nvcv::test::planar::MakePerImageTensor(numImages, nvcv::TYPE_S32, 5);
            auto sigmaColor = nvcv::test::planar::MakePerImageTensor(numImages, nvcv::TYPE_F32, 15.f);
            auto sigmaSpace = nvcv::test::planar::MakePerImageTensor(numImages, nvcv::TYPE_F32, 3.f);

            op(stream, src, src, dst, diameter, sigmaColor, sigmaSpace, border);
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        });
}

TEST(OpJointBilateralFilterPlanar, tensor_rgb8_matches_interleaved)
{
    RunJointBilateralTensorPlanarParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_REFLECT);
}

TEST(OpJointBilateralFilterPlanar, tensor_rgba8_matches_interleaved)
{
    RunJointBilateralTensorPlanarParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, NVCV_BORDER_CONSTANT);
}

TEST(OpJointBilateralFilterPlanar, varshape_rgb8_matches_interleaved)
{
    RunJointBilateralVarShapePlanarParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_REFLECT);
}

TEST(OpJointBilateralFilterPlanar, varshape_rgba8_matches_interleaved)
{
    RunJointBilateralVarShapePlanarParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, NVCV_BORDER_CONSTANT);
}

static auto OpJointBilateralFilterVarshapeNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType,
                          nvcv::DataType, nvcv::DataType, nvcv::DataType, int, int>
        params{
            {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,    nvcv::FMT_U8,  nvcv::FMT_U16, NVCV_BORDER_CONSTANT,
             nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // in/out image format not same
            {NVCV_ERROR_INVALID_ARGUMENT,   nvcv::FMT_U16,    nvcv::FMT_U8,  nvcv::FMT_U16, NVCV_BORDER_CONSTANT,
             nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // inColor/out image format not same
            {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT,
             nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // in/out data format not same
            {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT,
             nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // inColor/out data format not same
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8,
                        static_cast<NVCVBorderType>(255), nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, nvcv::FMT_F16, NVCV_BORDER_CONSTANT,
                        nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT,
                        nvcv::TYPE_F32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT,
                        nvcv::TYPE_S32, nvcv::TYPE_S32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT,
                        nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_S32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT,
                        nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 6, 5);
    return params;
}

NVCV_TEST_SUITE_P(OpJointBilateralFilterVarshape_Negative, OpJointBilateralFilterVarshapeNegativeParams());

static auto OpJointBilateralFilterNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType> params{
        {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,    nvcv::FMT_U8,   nvcv::FMT_U16,
         NVCV_BORDER_CONSTANT}, // in/out image format not same
        {NVCV_ERROR_INVALID_ARGUMENT,   nvcv::FMT_U16,    nvcv::FMT_U8,   nvcv::FMT_U16,
         NVCV_BORDER_CONSTANT}, // inColor/out image format not same
        {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,  nvcv::FMT_RGB8,
         NVCV_BORDER_CONSTANT}, // in/out data format not same
        {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,
         NVCV_BORDER_CONSTANT}, // inColor/out data format not same
        {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p,
         NVCV_BORDER_CONSTANT}, // inColor not kHWC/kNHWC
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_U8,
                        static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, nvcv::FMT_F16, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpJointBilateralFilter_Negative, OpJointBilateralFilterNegativeParams());

#undef NVCV_IMAGE_FORMAT_2U8

TEST(OpJointBilateralFilter_Negative, tensor_planar_2channel_rejected)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(
        {
            {5, 2, 24, 24},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgIn(
        {
            {5, 2, 24, 24},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgInColor(
        {
            {5, 2, 24, 24},
            "NCHW"
    },
        nvcv::TYPE_U8);

    int   diameter   = 4;
    float sigmaColor = 5;
    float sigmaSpace = 3;

    cvcuda::JointBilateralFilter jointBilateralFilterOp;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&jointBilateralFilterOp, &stream, &imgIn, &imgInColor, &imgOut, &diameter, &sigmaColor, &sigmaSpace]
                  {
                      jointBilateralFilterOp(stream, imgIn, imgInColor, imgOut, diameter, sigmaColor, sigmaSpace,
                                             NVCV_BORDER_CONSTANT);
                  }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpJointBilateralFilter_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat inputColorFmt      = GetParamValue<2>();
    nvcv::ImageFormat outputFmt          = GetParamValue<3>();
    NVCVBorderType    borderType         = GetParamValue<4>();

    int   width          = 24;
    int   height         = 24;
    int   diameter       = 4;
    float sigmaColor     = 5;
    float sigmaSpace     = 3;
    int   numberOfImages = 5;

    nvcv::Tensor imgOut     = nvcv::util::CreateTensor(numberOfImages, width, height, outputFmt);
    nvcv::Tensor imgIn      = nvcv::util::CreateTensor(numberOfImages, width, height, inputFmt);
    nvcv::Tensor imgInColor = nvcv::util::CreateTensor(numberOfImages, width, height, inputColorFmt);

    // run operator
    cvcuda::JointBilateralFilter jointBilateralFilterOp;

    EXPECT_EQ(expectedReturnCode, nvcv::ProtectCall(
                                      [&jointBilateralFilterOp, &stream, &imgIn, &imgInColor, &imgOut, &diameter,
                                       &sigmaColor, &sigmaSpace, &borderType] {
                                          jointBilateralFilterOp(stream, imgIn, imgInColor, imgOut, diameter,
                                                                 sigmaColor, sigmaSpace, borderType);
                                      }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpJointBilateralFilterVarshape_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode   = GetParamValue<0>();
    nvcv::ImageFormat inputFmt             = GetParamValue<1>();
    nvcv::ImageFormat inputColorFmt        = GetParamValue<2>();
    nvcv::ImageFormat outputFmt            = GetParamValue<3>();
    NVCVBorderType    borderType           = GetParamValue<4>();
    nvcv::DataType    diameterDataType     = GetParamValue<5>();
    nvcv::DataType    sigmaColorDataType   = GetParamValue<6>();
    nvcv::DataType    sigmaSpaceDataType   = GetParamValue<7>();
    int               numberOfInputImages  = GetParamValue<8>();
    int               numberOfOutputImages = GetParamValue<9>();

    int   width      = 24;
    int   height     = 24;
    int   diameter   = 4;
    float sigmaColor = 5;
    float sigmaSpace = 3;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(width * 8 / 10, width * 11 / 10);
    std::uniform_int_distribution udistHeight(height * 8 / 10, height * 11 / 10);

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgSrcColor;

    for (int i = 0; i < numberOfInputImages; ++i)
    {
        int          w = udistWidth(rng);
        int          h = udistHeight(rng);
        nvcv::Size2D sz(w, h);
        imgSrc.emplace_back(sz, inputFmt);
        imgSrcColor.emplace_back(sz, inputColorFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfInputImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchSrcColor(numberOfInputImages);
    batchSrcColor.pushBack(imgSrcColor.begin(), imgSrcColor.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numberOfOutputImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outputFmt);
    }
    nvcv::ImageBatchVarShape batchDst(numberOfOutputImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create diameter tensor
    std::vector<int> vDiameter(numberOfInputImages, diameter);
    nvcv::Tensor     diameterTensor({{numberOfInputImages}, "N"}, diameterDataType);
    {
        auto dev = diameterTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vDiameter.data(), vDiameter.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create sigmaColor tensor
    std::vector<float> vSigmaColor(numberOfInputImages, sigmaColor);
    nvcv::Tensor       sigmaColorTensor({{numberOfInputImages}, "N"}, sigmaColorDataType);
    {
        auto dev = sigmaColorTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vSigmaColor.data(), vSigmaColor.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create sigmaSpace tensor
    std::vector<float> vSigmaSpace(numberOfInputImages, sigmaSpace);
    nvcv::Tensor       sigmaSpaceTensor({{numberOfInputImages}, "N"}, sigmaSpaceDataType);
    {
        auto dev = sigmaSpaceTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vSigmaSpace.data(), vSigmaSpace.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::JointBilateralFilter jointBilateralFilterOp;
    EXPECT_EQ(expectedReturnCode, nvcv::ProtectCall(
                                      [&jointBilateralFilterOp, &stream, &batchSrc, &batchSrcColor, &batchDst,
                                       &diameterTensor, &sigmaColorTensor, &sigmaSpaceTensor, &borderType]
                                      {
                                          jointBilateralFilterOp(stream, batchSrc, batchSrcColor, batchDst,
                                                                 diameterTensor, sigmaColorTensor, sigmaSpaceTensor,
                                                                 borderType);
                                      }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpJointBilateralFilter_Negative, varshape_hasDifferentFormat)
{
    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt,          fmt},
        {         fmt, nvcv::FMT_U8,          fmt},
        {         fmt,          fmt, nvcv::FMT_U8}
    };

    for (const auto &[inputFmtExtra, inputColorFmtExtra, outputFmtExtra] : testSet)
    {
        cudaStream_t stream;
        EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        int   width          = 24;
        int   height         = 24;
        int   diameter       = 4;
        float sigmaColor     = 5;
        float sigmaSpace     = 3;
        int   numberOfImages = 5;

        // Create input varshape
        std::default_random_engine    rng;
        std::uniform_int_distribution udistWidth(static_cast<int>(static_cast<double>(width) * 0.8),
                                                 static_cast<int>(static_cast<double>(width) * 1.1));
        std::uniform_int_distribution udistHeight(static_cast<int>(static_cast<double>(height) * 0.8),
                                                  static_cast<int>(static_cast<double>(height) * 1.1));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgSrcColor;

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            int          w = udistWidth(rng);
            int          h = udistHeight(rng);
            nvcv::Size2D sz(w, h);
            imgSrc.emplace_back(sz, fmt);
            imgSrcColor.emplace_back(sz, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgSrcColor.emplace_back(imgSrc[0].size(), inputColorFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        nvcv::ImageBatchVarShape batchSrcColor(numberOfImages);
        batchSrcColor.pushBack(imgSrcColor.begin(), imgSrcColor.end());

        // Create output varshape
        std::vector<nvcv::Image> imgDst;
        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            imgDst.emplace_back(imgSrc[i].size(), fmt);
        }
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

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

        // Run operator
        cvcuda::JointBilateralFilter jointBilateralFilterOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&jointBilateralFilterOp, &stream, &batchSrc, &batchSrcColor,
                                                    &batchDst, &diameterTensor, &sigmaColorTensor, &sigmaSpaceTensor]
                                                   {
                                                       jointBilateralFilterOp(stream, batchSrc, batchSrcColor, batchDst,
                                                                              diameterTensor, sigmaColorTensor,
                                                                              sigmaSpaceTensor, NVCV_BORDER_CONSTANT);
                                                   }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}

TEST(OpJointBilateralFilter_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaJointBilateralFilterCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
