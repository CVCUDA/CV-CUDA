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
#include <cvcuda/OpBilateralFilter.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <cstring>
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
            return false;
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

static void AccumulateBilateralSample(std::vector<float> &numerators, float &denominator,
                                      const std::vector<float> &centers, const uint8_t *pIn, TensorDim x, TensorDim y,
                                      TensorDim columns, TensorDim rows, int rowStride, int channels,
                                      float distanceSquared, float colorCoefficient, float spaceCoefficient)
{
    std::vector<float> pixels = ReadChannels(pIn, x, y, columns, rows, rowStride, channels);
    float              eColor = 0.0f;

    for (int c = 0; c < channels; ++c)
    {
        eColor += std::abs(pixels[c] - centers[c]);
    }

    float weight = std::exp(distanceSquared * spaceCoefficient + eColor * eColor * colorCoefficient);
    denominator += weight;

    for (int c = 0; c < channels; ++c)
    {
        numerators[c] += weight * pixels[c];
    }
}

static void AccumulateBilateralWindow(std::vector<float> &numerators, float &denominator,
                                      const std::vector<float> &centers, const uint8_t *pIn, TensorDim column,
                                      TensorDim row, TensorDim columns, TensorDim rows, int rowStride, int channels,
                                      int radius, float radiusSquared, float colorCoefficient, float spaceCoefficient)
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

            AccumulateBilateralSample(numerators, denominator, centers, pIn, x, y, columns, rows, rowStride, channels,
                                      distanceSquared, colorCoefficient, spaceCoefficient);
        }
    }
}

static void CPUBilateralFilter(const uint8_t *pIn, uint8_t *pOut, TensorDim columns, TensorDim rows, int rowStride,
                               int channels, int radius, float colorCoefficient, float spaceCoefficient)
{
    auto radiusSquared = static_cast<float>(radius * radius);
    for (TensorDim j = 0; j < rows; j++)
    {
        for (TensorDim k = 0; k < columns; k++)
        {
            std::vector<float> numerators(channels, 0.0f);
            float              denominator = 0.0f;
            std::vector<float> centers     = ReadChannels(pIn, k, j, columns, rows, rowStride, channels);

            AccumulateBilateralWindow(numerators, denominator, centers, pIn, k, j, columns, rows, rowStride, channels,
                                      radius, radiusSquared, colorCoefficient, spaceCoefficient);

            for (auto c = 0; c < channels; ++c)
            {
                pOut[j * rowStride + k * channels + c] = saturate_cast(numerators[c] / denominator);
            }
        }
    }
}

static void CPUBilateralFilterTensor(std::vector<uint8_t> &vIn, std::vector<uint8_t> &vOut, TensorDim columns,
                                     TensorDim rows, TensorDim batch, int rowStride, int channels, int sampleStride,
                                     int diameter, float sigmaColor, float sigmaSpace)
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
        const uint8_t *pIn  = vIn.data() + i * sampleStride;
        uint8_t       *pOut = vOut.data() + i * sampleStride;
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
    for (nvcv::ImageFormat fmt : fmts) // NOSONAR
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

        auto inSampleStride  = static_cast<int>(inAccess->numRows() * inAccess->rowStride());
        auto outSampleStride = static_cast<int>(outAccess->numRows() * outAccess->rowStride());

        int inBufSize  = inSampleStride * static_cast<int>(inAccess->numSamples());
        int outBufSize = outSampleStride * static_cast<int>(outAccess->numSamples());

        std::vector<uint8_t> vIn(inBufSize);
        std::vector<uint8_t> vOut(outBufSize);

        std::vector<uint8_t> inGold(inBufSize, 0);
        std::vector<uint8_t> outGold(outBufSize, 0);
        for (int i = 0; i < inBufSize; i++) inGold[i] = i % 113; // Use prime number to prevent weird tiling patterns

        EXPECT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inGold.data(), inBufSize, cudaMemcpyHostToDevice));
        const int rowStride{static_cast<int>(inAccess->rowStride())};
        CPUBilateralFilterTensor(inGold, outGold, inAccess->numCols(), inAccess->numRows(), inAccess->numSamples(),
                                 rowStride, channels, inSampleStride, d, sigmaColor, sigmaSpace);

        // run operator
        cvcuda::BilateralFilter bilateralFilterOp;

        EXPECT_NO_THROW(bilateralFilterOp(stream, imgIn, imgOut, d, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT));

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

TEST_P(OpBilateralFilter, varshape_correct_output)
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
            std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });
            std::ranges::generate(goldVec[i], []() { return 0; });
            std::ranges::generate(dstVec[i], []() { return 0; });
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

static void FillBilateralParamTensor(cudaStream_t stream, nvcv::Tensor &tensor, const std::vector<int> &values)
{
    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), values.data(), values.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
}

static void FillBilateralParamTensor(cudaStream_t stream, nvcv::Tensor &tensor, const std::vector<float> &values)
{
    auto dev = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(dev, nullptr);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), values.data(), values.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
}

static void RunBilateralTensorPlanarParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                           NVCVBorderType border)
{
    cvcuda::BilateralFilter op;
    nvcv::test::planar::RunTensorParity(
        planarFmt, interleavedFmt, 33, 25, 33, 25, 2,
        [&op, border](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, dst, 5, 15.f, 3.f, border); });
}

static void RunBilateralVarShapePlanarParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                             NVCVBorderType border)
{
    cvcuda::BilateralFilter op;
    nvcv::test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, 31, 23, 31, 23, 2,
        [&op, border](cudaStream_t stream, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst,
                      nvcv::ImageFormat)
        {
            const int numImages = src.numImages();

            nvcv::Tensor diameter({{numImages}, "N"}, nvcv::TYPE_S32);
            nvcv::Tensor sigmaColor({{numImages}, "N"}, nvcv::TYPE_F32);
            nvcv::Tensor sigmaSpace({{numImages}, "N"}, nvcv::TYPE_F32);

            FillBilateralParamTensor(stream, diameter, std::vector<int>(numImages, 5));
            FillBilateralParamTensor(stream, sigmaColor, std::vector<float>(numImages, 15.f));
            FillBilateralParamTensor(stream, sigmaSpace, std::vector<float>(numImages, 3.f));

            op(stream, src, dst, diameter, sigmaColor, sigmaSpace, border);
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        });
}

static std::vector<uint8_t> MakeFiniteFloatHwc(int width, int height, int channels, int seed)
{
    std::vector<float> values(width * height * channels);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<float>((static_cast<int>(i) * 17 + seed) % 113) / 11.f;
    }

    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

static float ReadFloatHwc(const std::vector<float> &src, int width, int height, int channels, int x, int y, int c)
{
    return ((x >= 0) && (x < width) && (y >= 0) && (y < height)) ? src[(y * width + x) * channels + c] : 0.f;
}

static std::vector<float> ReadFloatHwcPixel(const std::vector<float> &src, int width, int height, int channels, int x,
                                            int y)
{
    std::vector<float> pixel(channels);
    for (int c = 0; c < channels; ++c)
    {
        pixel[c] = ReadFloatHwc(src, width, height, channels, x, y, c);
    }
    return pixel;
}

static void AccumulateBilateralFloatHwcSample(std::vector<float> &numerator, float &denominator,
                                              const std::vector<float> &src, const std::vector<float> &center,
                                              int width, int height, int channels, int x, int y, int sampleX,
                                              int sampleY, float radiusSquared, float spaceCoefficient,
                                              float colorCoefficient)
{
    const auto dx              = x - sampleX;
    const auto dy              = y - sampleY;
    const auto distanceSquared = static_cast<float>(dx * dx + dy * dy);
    if (distanceSquared > radiusSquared)
    {
        return;
    }

    float oneNorm = 0.f;
    for (int c = 0; c < channels; ++c)
    {
        oneNorm += std::abs(ReadFloatHwc(src, width, height, channels, sampleX, sampleY, c) - center[c]);
    }

    const float weight = std::exp(distanceSquared * spaceCoefficient + oneNorm * oneNorm * colorCoefficient);
    denominator += weight;
    for (int c = 0; c < channels; ++c)
    {
        numerator[c] += weight * ReadFloatHwc(src, width, height, channels, sampleX, sampleY, c);
    }
}

static void WriteBilateralFloatHwcPixel(std::vector<float> &dst, const std::vector<float> &src, int width, int height,
                                        int channels, int x, int y, int radius, float radiusSquared,
                                        float spaceCoefficient, float colorCoefficient)
{
    const auto center      = ReadFloatHwcPixel(src, width, height, channels, x, y);
    auto       numerator   = std::vector<float>(channels, 0.f);
    float      denominator = 0.f;

    for (int sampleY = y - radius; sampleY <= y + radius; ++sampleY)
    {
        for (int sampleX = x - radius; sampleX <= x + radius; ++sampleX)
        {
            AccumulateBilateralFloatHwcSample(numerator, denominator, src, center, width, height, channels, x, y,
                                              sampleX, sampleY, radiusSquared, spaceCoefficient, colorCoefficient);
        }
    }

    for (int c = 0; c < channels; ++c)
    {
        dst[(y * width + x) * channels + c] = numerator[c] / denominator;
    }
}

static std::vector<uint8_t> CPUBilateralFilterFloatHwc(const std::vector<uint8_t> &input, int width, int height,
                                                       int channels, int diameter, float sigmaColor, float sigmaSpace)
{
    std::vector<float> src(width * height * channels);
    std::memcpy(src.data(), input.data(), input.size());

    if (sigmaColor <= 0)
    {
        sigmaColor = 1;
    }
    if (sigmaSpace <= 0)
    {
        sigmaSpace = 1;
    }

    int radius = diameter <= 0 ? static_cast<int>(std::roundf(sigmaSpace * 1.5f)) : diameter / 2;
    if (radius < 1)
    {
        radius = 1;
    }

    const auto  radiusSquared    = static_cast<float>(radius * radius);
    const float spaceCoefficient = -1.f / (2.f * sigmaSpace * sigmaSpace);
    const float colorCoefficient = -1.f / (2.f * sigmaColor * sigmaColor);

    std::vector<float> dst(src.size());
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            WriteBilateralFloatHwcPixel(dst, src, width, height, channels, x, y, radius, radiusSquared,
                                        spaceCoefficient, colorCoefficient);
        }
    }

    std::vector<uint8_t> output(dst.size() * sizeof(float));
    std::memcpy(output.data(), dst.data(), output.size());
    return output;
}

static void ExpectFloatHwcNear(const std::vector<uint8_t> &test, const std::vector<uint8_t> &gold,
                               float tolerance = 1e-5f)
{
    ASSERT_EQ(test.size(), gold.size());
    ASSERT_EQ(test.size() % sizeof(float), 0);
    const auto numValues = test.size() / sizeof(float);

    for (size_t i = 0; i < numValues; ++i)
    {
        float testValue;
        float goldValue;
        std::memcpy(&testValue, test.data() + i * sizeof(float), sizeof(float));
        std::memcpy(&goldValue, gold.data() + i * sizeof(float), sizeof(float));

        ASSERT_TRUE(std::isfinite(testValue));
        ASSERT_TRUE(std::isfinite(goldValue));
        EXPECT_NEAR(testValue, goldValue, tolerance) << "at float index " << i;
    }
}

static void RunBilateralPackedTensorFloatReference(nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width     = 33;
    constexpr int height    = 25;
    constexpr int numImages = 2;

    const int channels    = fmt.numChannels();
    const int pixelStride = fmt.planePixelStrideBytes(0);
    const int rowStride   = width * pixelStride;
    ASSERT_EQ(pixelStride, channels * static_cast<int>(sizeof(float)));

    nvcv::Tensor src = nvcv::util::CreateTensor(numImages, width, height, fmt);
    nvcv::Tensor dst = nvcv::util::CreateTensor(numImages, width, height, fmt);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData && dstData);

    auto srcAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(srcAcc && dstAcc);

    std::vector<std::vector<uint8_t>> gold(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        auto hwc = MakeFiniteFloatHwc(width, height, channels, i * 101 + 31);
        gold[i]  = CPUBilateralFilterFloatHwc(hwc, width, height, channels, 5, 15.f, 3.f);
        nvcv::test::planar::UploadInterleavedSample(*srcAcc, i, hwc, width, height, rowStride);
    }

    cvcuda::BilateralFilter op;
    op(stream, src, dst, 5, 15.f, 3.f, NVCV_BORDER_CONSTANT);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto gpu = nvcv::test::planar::DownloadInterleavedSample(*dstAcc, i, width, height, rowStride);
        ExpectFloatHwcNear(gpu, gold[i], 1e-4f);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunBilateralPackedVarShapeFloatReference(nvcv::ImageFormat fmt)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const std::vector<nvcv::Size2D> sizes{
        {31, 23},
        {35, 25}
    };
    const auto numImages   = static_cast<int>(sizes.size());
    const int  channels    = fmt.numChannels();
    const int  pixelStride = fmt.planePixelStrideBytes(0);
    ASSERT_EQ(pixelStride, channels * static_cast<int>(sizeof(float)));

    std::vector<nvcv::Image>          src;
    std::vector<nvcv::Image>          dst;
    std::vector<std::vector<uint8_t>> gold(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        src.emplace_back(sizes[i], fmt);
        dst.emplace_back(sizes[i], fmt);
    }

    for (int i = 0; i < numImages; ++i)
    {
        const int width     = sizes[i].w;
        const int height    = sizes[i].h;
        const int rowStride = width * pixelStride;

        auto hwc = MakeFiniteFloatHwc(width, height, channels, i * 101 + 37);
        gold[i]  = CPUBilateralFilterFloatHwc(hwc, width, height, channels, 5, 15.f, 3.f);

        auto data = src[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(0).basePtr, data->plane(0).rowStride, hwc.data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(src.begin(), src.end());
    batchDst.pushBack(dst.begin(), dst.end());

    nvcv::Tensor diameter({{numImages}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor sigmaColor({{numImages}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaSpace({{numImages}, "N"}, nvcv::TYPE_F32);
    FillBilateralParamTensor(stream, diameter, std::vector<int>(numImages, 5));
    FillBilateralParamTensor(stream, sigmaColor, std::vector<float>(numImages, 15.f));
    FillBilateralParamTensor(stream, sigmaSpace, std::vector<float>(numImages, 3.f));

    cvcuda::BilateralFilter op;
    op(stream, batchSrc, batchDst, diameter, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        const int width     = sizes[i].w;
        const int height    = sizes[i].h;
        const int rowStride = width * pixelStride;

        std::vector<uint8_t> gpu(height * rowStride);
        auto                 data = dst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(gpu.data(), rowStride, data->plane(0).basePtr, data->plane(0).rowStride,
                                            rowStride, height, cudaMemcpyDeviceToHost));
        ExpectFloatHwcNear(gpu, gold[i], 1e-4f);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunBilateralTensorPlanarFloatParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                                NVCVBorderType border)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width     = 33;
    constexpr int height    = 25;
    constexpr int numImages = 2;

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int rowStride = width * channels * elemSize;
    ASSERT_EQ(elemSize, static_cast<int>(sizeof(float)));

    nvcv::Tensor srcI = nvcv::util::CreateTensor(numImages, width, height, interleavedFmt);
    nvcv::Tensor dstI = nvcv::util::CreateTensor(numImages, width, height, interleavedFmt);
    nvcv::Tensor srcP = nvcv::util::CreateTensor(numImages, width, height, planarFmt);
    nvcv::Tensor dstP = nvcv::util::CreateTensor(numImages, width, height, planarFmt);

    auto srcIData = srcI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPData = srcP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcIData && dstIData && srcPData && dstPData);

    auto srcIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcIData);
    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto srcPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(srcIAcc && dstIAcc && srcPAcc && dstPAcc);

    for (int i = 0; i < numImages; ++i)
    {
        auto hwc = MakeFiniteFloatHwc(width, height, channels, i * 101 + 13);
        nvcv::test::planar::UploadInterleavedSample(*srcIAcc, i, hwc, width, height, rowStride);
        nvcv::test::planar::UploadPlanarSample(
            *srcPAcc, i, nvcv::test::planar::DeinterleaveToPlanes(hwc, width, height, channels, elemSize), width,
            height, channels, elemSize);
    }

    cvcuda::BilateralFilter op;
    op(stream, srcI, dstI, 5, 15.f, 3.f, border);
    op(stream, srcP, dstP, 5, 15.f, 3.f, border);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = nvcv::test::planar::DownloadInterleavedSample(*dstIAcc, i, width, height, rowStride);
        auto planesOut   = nvcv::test::planar::DownloadPlanarSample(*dstPAcc, i, width, height, channels, elemSize);
        auto planarInter = nvcv::test::planar::InterleaveFromPlanes(planesOut, width, height, channels, elemSize);
        ExpectFloatHwcNear(planarInter, gpuInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void RunBilateralVarShapePlanarFloatParity(nvcv::ImageFormat interleavedFmt, nvcv::ImageFormat planarFmt,
                                                  NVCVBorderType border)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width     = 31;
    constexpr int height    = 23;
    constexpr int numImages = 2;

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int rowStride = width * channels * elemSize;
    ASSERT_EQ(elemSize, static_cast<int>(sizeof(float)));

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> dstI;
    std::vector<nvcv::Image> srcP;
    std::vector<nvcv::Image> dstP;
    for (int i = 0; i < numImages; ++i)
    {
        srcI.emplace_back(nvcv::Size2D{width, height}, interleavedFmt);
        dstI.emplace_back(nvcv::Size2D{width, height}, interleavedFmt);
        srcP.emplace_back(nvcv::Size2D{width, height}, planarFmt);
        dstP.emplace_back(nvcv::Size2D{width, height}, planarFmt);
    }

    nvcv::ImageBatchVarShape batchSrcI(numImages);
    nvcv::ImageBatchVarShape batchDstI(numImages);
    nvcv::ImageBatchVarShape batchSrcP(numImages);
    nvcv::ImageBatchVarShape batchDstP(numImages);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchDstI.pushBack(dstI.begin(), dstI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());
    batchDstP.pushBack(dstP.begin(), dstP.end());

    for (int i = 0; i < numImages; ++i)
    {
        auto hwc = MakeFiniteFloatHwc(width, height, channels, i * 101 + 17);

        auto idata = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(idata, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));

        auto      planes     = nvcv::test::planar::DeinterleaveToPlanes(hwc, width, height, channels, elemSize);
        auto      pdata      = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        const int planeBytes = width * height * elemSize;
        ASSERT_NE(pdata, nvcv::NullOpt);
        ASSERT_EQ(pdata->numPlanes(), channels);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(pdata->plane(c).basePtr, pdata->plane(c).rowStride, planes.data() + c * planeBytes,
                                   width * elemSize, width * elemSize, height, cudaMemcpyHostToDevice));
        }
    }

    nvcv::Tensor diameter({{numImages}, "N"}, nvcv::TYPE_S32);
    nvcv::Tensor sigmaColor({{numImages}, "N"}, nvcv::TYPE_F32);
    nvcv::Tensor sigmaSpace({{numImages}, "N"}, nvcv::TYPE_F32);
    FillBilateralParamTensor(stream, diameter, std::vector<int>(numImages, 5));
    FillBilateralParamTensor(stream, sigmaColor, std::vector<float>(numImages, 15.f));
    FillBilateralParamTensor(stream, sigmaSpace, std::vector<float>(numImages, 3.f));

    cvcuda::BilateralFilter op;
    op(stream, batchSrcI, batchDstI, diameter, sigmaColor, sigmaSpace, border);
    op(stream, batchSrcP, batchDstP, diameter, sigmaColor, sigmaSpace, border);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const int planeBytes = width * height * elemSize;
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        std::vector<uint8_t> gpuInter(height * rowStride);
        auto                 idata = dstI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(idata, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(gpuInter.data(), rowStride, idata->plane(0).basePtr,
                                            idata->plane(0).rowStride, rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> planesOut(width * height * channels * elemSize);
        auto                 pdata = dstP[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(pdata, nvcv::NullOpt);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(planesOut.data() + c * planeBytes, width * elemSize, pdata->plane(c).basePtr,
                                   pdata->plane(c).rowStride, width * elemSize, height, cudaMemcpyDeviceToHost));
        }
        auto planarInter = nvcv::test::planar::InterleaveFromPlanes(planesOut, width, height, channels, elemSize);
        ExpectFloatHwcNear(planarInter, gpuInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBilateralFilterPackedFloat, tensor_rgbf32_matches_cpu_reference)
{
    RunBilateralPackedTensorFloatReference(nvcv::FMT_RGBf32);
}

TEST(OpBilateralFilterPackedFloat, tensor_rgbaf32_matches_cpu_reference)
{
    RunBilateralPackedTensorFloatReference(nvcv::FMT_RGBAf32);
}

TEST(OpBilateralFilterPackedFloat, varshape_rgbf32_matches_cpu_reference)
{
    RunBilateralPackedVarShapeFloatReference(nvcv::FMT_RGBf32);
}

TEST(OpBilateralFilterPackedFloat, varshape_rgbaf32_matches_cpu_reference)
{
    RunBilateralPackedVarShapeFloatReference(nvcv::FMT_RGBAf32);
}

TEST(OpBilateralFilterPlanar, tensor_rgb8_matches_interleaved)
{
    RunBilateralTensorPlanarParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_REFLECT);
}

TEST(OpBilateralFilterPlanar, tensor_rgba8_matches_interleaved)
{
    RunBilateralTensorPlanarParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, NVCV_BORDER_CONSTANT);
}

TEST(OpBilateralFilterPlanar, tensor_rgbf32_matches_interleaved)
{
    RunBilateralTensorPlanarFloatParity(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p, NVCV_BORDER_REFLECT);
}

TEST(OpBilateralFilterPlanar, tensor_rgbaf32_matches_interleaved)
{
    RunBilateralTensorPlanarFloatParity(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32p, NVCV_BORDER_CONSTANT);
}

TEST(OpBilateralFilterPlanar, varshape_rgb8_matches_interleaved)
{
    RunBilateralVarShapePlanarParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, NVCV_BORDER_REFLECT);
}

TEST(OpBilateralFilterPlanar, varshape_rgba8_matches_interleaved)
{
    RunBilateralVarShapePlanarParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, NVCV_BORDER_CONSTANT);
}

TEST(OpBilateralFilterPlanar, varshape_rgbf32_matches_interleaved)
{
    RunBilateralVarShapePlanarFloatParity(nvcv::FMT_RGBf32, nvcv::FMT_RGBf32p, NVCV_BORDER_REFLECT);
}

TEST(OpBilateralFilterPlanar, varshape_rgbaf32_matches_interleaved)
{
    RunBilateralVarShapePlanarFloatParity(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32p, NVCV_BORDER_CONSTANT);
}

#undef NVCV_IMAGE_FORMAT_2U8

static auto OpBilateralFilterVarshapeNegativeParams()
{
    test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType, nvcv::DataType, nvcv::DataType,
                    nvcv::DataType, int, int>
        params{
            {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,  nvcv::FMT_U16, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
             nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // in/out image format not same
            {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
             nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5}, // in/out data format not same
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, static_cast<NVCVBorderType>(255),
                        nvcv::TYPE_S32, nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
                        nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT, nvcv::TYPE_F32,
                        nvcv::TYPE_F32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
                        nvcv::TYPE_S32, nvcv::TYPE_F32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
                        nvcv::TYPE_F32, nvcv::TYPE_S32, 5, 5);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, NVCV_BORDER_CONSTANT, nvcv::TYPE_S32,
                        nvcv::TYPE_F32, nvcv::TYPE_F32, 6, 5);
    return params;
}

NVCV_TEST_SUITE_P(OpBilateralFilterVarshape_Negative, OpBilateralFilterVarshapeNegativeParams());

static auto OpBilateralFilterNegativeParams()
{
    nvcv::test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType> params{
        {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,  nvcv::FMT_U16,
         NVCV_BORDER_CONSTANT}, // in/out image datatype not same
        {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8,
         NVCV_BORDER_CONSTANT}, // in/out data format not same
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpBilateralFilter_Negative, OpBilateralFilterNegativeParams());

TEST_P(OpBilateralFilter_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    NVCVBorderType    borderType         = GetParamValue<3>();

    int   width          = 24;
    int   height         = 24;
    int   diameter       = 4;
    float sigmaColor     = 5;
    float sigmaSpace     = 3;
    int   numberOfImages = 5;

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, width, height, outputFmt);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(numberOfImages, width, height, inputFmt);

    // run operator
    cvcuda::BilateralFilter bilateralFilterOp;

    EXPECT_EQ(expectedReturnCode,
              nvcv::ProtectCall(
                  [&bilateralFilterOp, &stream, &imgIn, &imgOut, &diameter, &sigmaColor, &sigmaSpace, &borderType]
                  { bilateralFilterOp(stream, imgIn, imgOut, diameter, sigmaColor, sigmaSpace, borderType); }));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBilateralFilter_Negative, rejects_mismatched_output_shape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int   width          = 24;
    int   height         = 24;
    int   diameter       = 4;
    float sigmaColor     = 5;
    float sigmaSpace     = 3;
    int   numberOfImages = 5;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(numberOfImages, width, height, nvcv::FMT_U8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, width + 1, height, nvcv::FMT_U8);

    cvcuda::BilateralFilter bilateralFilterOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&bilateralFilterOp, &stream, &imgIn, &imgOut, &diameter, &sigmaColor, &sigmaSpace] {
                      bilateralFilterOp(stream, imgIn, imgOut, diameter, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT);
                  }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpBilateralFilterVarshape_Negative, varshape)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode   = GetParamValue<0>();
    nvcv::ImageFormat inputFmt             = GetParamValue<1>();
    nvcv::ImageFormat outputFmt            = GetParamValue<2>();
    NVCVBorderType    borderType           = GetParamValue<3>();
    nvcv::DataType    diameterDataType     = GetParamValue<4>();
    nvcv::DataType    sigmaColorDataType   = GetParamValue<5>();
    nvcv::DataType    sigmaSpaceDataType   = GetParamValue<6>();
    int               numberOfInputImages  = GetParamValue<7>();
    int               numberOfOutputImages = GetParamValue<8>();

    int   width      = 24;
    int   height     = 24;
    int   diameter   = 4;
    float sigmaColor = 5;
    float sigmaSpace = 3;

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(static_cast<int>(static_cast<double>(width) * 0.8),
                                             static_cast<int>(static_cast<double>(width) * 1.1));
    std::uniform_int_distribution udistHeight(static_cast<int>(static_cast<double>(height) * 0.8),
                                              static_cast<int>(static_cast<double>(height) * 1.1));

    std::vector<nvcv::Image> imgSrc;

    for (int i = 0; i < numberOfInputImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfInputImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

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
    cvcuda::BilateralFilter bilateralFilterOp;
    EXPECT_EQ(expectedReturnCode, nvcv::ProtectCall(
                                      [&bilateralFilterOp, &stream, &batchSrc, &batchDst, &diameterTensor,
                                       &sigmaColorTensor, &sigmaSpaceTensor, &borderType] {
                                          bilateralFilterOp(stream, batchSrc, batchDst, diameterTensor,
                                                            sigmaColorTensor, sigmaSpaceTensor, borderType);
                                      }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBilateralFilterVarshape_Negative, varshape_hasDifferentFormat)
{
    nvcv::ImageFormat                                             fmt = nvcv::FMT_RGB8;
    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt},
        {         fmt, nvcv::FMT_U8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
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

        for (int i = 0; i < numberOfImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        nvcv::ImageBatchVarShape batchSrc(numberOfImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

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
        cvcuda::BilateralFilter bilateralFilterOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&bilateralFilterOp, &stream, &batchSrc, &batchDst, &diameterTensor,
                                                    &sigmaColorTensor, &sigmaSpaceTensor]
                                                   {
                                                       bilateralFilterOp(stream, batchSrc, batchDst, diameterTensor,
                                                                         sigmaColorTensor, sigmaSpaceTensor,
                                                                         NVCV_BORDER_CONSTANT);
                                                   }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}

TEST(OpBilateralFilter_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaBilateralFilterCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
