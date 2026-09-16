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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpBndBox.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
using namespace cvcuda::priv;

static std::mt19937 &Rng()
{
    static std::mt19937 rng;
    return rng;
}

static int randl(int l, int h)
{
    return std::uniform_int_distribution<int>(l, h)(Rng());
}

static void runOp(cudaStream_t &stream, const cvcuda::BndBox &op, int inN, int inW, int inH, int num, int sed,
                  const nvcv::ImageFormat &format)
{
    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;

    Rng().seed(sed);
    for (int n = 0; n < inN; n++)
    {
        std::vector<NVCVBndBoxI> curVec;
        for (int i = 0; i < num; i++)
        {
            NVCVBndBoxI bndBox;
            bndBox.box.x       = randl(0, inW - 1);
            bndBox.box.y       = randl(0, inH - 1);
            bndBox.box.width   = randl(1, inW);
            bndBox.box.height  = randl(1, inH);
            bndBox.thickness   = randl(-1, 30);
            bndBox.fillColor   = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
            bndBox.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)};
            curVec.push_back(bndBox);
        }
        bndBoxVec.push_back(curVec);
    }

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, inW, inH, format);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    const size_t inSampleStride = static_cast<size_t>(inAccess->numRows()) * static_cast<size_t>(inAccess->rowStride());
    const size_t outSampleStride
        = static_cast<size_t>(outAccess->numRows()) * static_cast<size_t>(outAccess->rowStride());

    const size_t inBufSize  = inSampleStride * static_cast<size_t>(inAccess->numSamples());
    const size_t outBufSize = outSampleStride * static_cast<size_t>(outAccess->numSamples());

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()));

    std::vector<uint8_t> outHost(outBufSize);
    std::vector<uint8_t> inHost(inBufSize, 0xFF);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outHost.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_NE(inHost, outHost) << "Output should differ from input after drawing bounding boxes";
}

static size_t hostPixelOffset(const nvcv::TensorDataAccessStridedImagePlanar &access, int64_t sampleStride, int sample,
                              int y, int x, int channel)
{
    return static_cast<size_t>(sample * sampleStride + channel * access.chStride() + y * access.rowStride()
                               + x * access.colStride());
}

static void fillHostPixel(const nvcv::TensorDataAccessStridedImagePlanar &inputAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t inputSampleStride,
                          int64_t outputSampleStride, int sample, int y, int x, std::vector<uint8_t> &inputHost,
                          std::vector<uint8_t> &expectedHost)
{
    for (int channel = 0; channel < inputAccess.numChannels(); ++channel)
    {
        const auto inputOffset     = hostPixelOffset(inputAccess, inputSampleStride, sample, y, x, channel);
        const auto outputOffset    = hostPixelOffset(outputAccess, outputSampleStride, sample, y, x, channel);
        inputHost[inputOffset]     = static_cast<uint8_t>((sample * 53 + y * 19 + x * 7 + channel * 29) & 0xFF);
        expectedHost[outputOffset] = inputHost[inputOffset];
    }
}

static void fillHostPattern(const nvcv::TensorDataAccessStridedImagePlanar &inputAccess,
                            const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t inputSampleStride,
                            int64_t outputSampleStride, int numSamples, int height, int width,
                            std::vector<uint8_t> &inputHost, std::vector<uint8_t> &expectedHost)
{
    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                fillHostPixel(inputAccess, outputAccess, inputSampleStride, outputSampleStride, sample, y, x, inputHost,
                              expectedHost);
            }
        }
    }
}

static void fillHostAlpha(const nvcv::TensorDataAccessStridedImagePlanar &inputAccess,
                          const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t inputSampleStride,
                          int64_t outputSampleStride, int numSamples, int height, int width, uint8_t alpha,
                          std::vector<uint8_t> &inputHost, std::vector<uint8_t> &expectedHost)
{
    const int sampleSize = height * width;
    for (int pixel = 0; pixel < numSamples * sampleSize; ++pixel)
    {
        const int sample = pixel / sampleSize;
        const int y      = pixel % sampleSize / width;
        const int x      = pixel % width;

        inputHost[hostPixelOffset(inputAccess, inputSampleStride, sample, y, x, 3)]      = alpha;
        expectedHost[hostPixelOffset(outputAccess, outputSampleStride, sample, y, x, 3)] = alpha;
    }
}

static bool shouldDrawHostPixel(const NVCVBndBoxI &box, bool filled, int x, int y)
{
    const int left   = box.box.x;
    const int top    = box.box.y;
    const int right  = left + box.box.width - 1;
    const int bottom = top + box.box.height - 1;
    return filled ? (x > left && x < right && y > top && y < bottom)
                  : (x < left + 2 || x > right - 2 || y < top + 2 || y > bottom - 2);
}

static void drawHostBox(const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t outputSampleStride,
                        int sample, const NVCVBndBoxI &box, bool filled, std::vector<uint8_t> &expectedHost)
{
    const int     left    = box.box.x;
    const int     top     = box.box.y;
    const int     right   = left + box.box.width - 1;
    const int     bottom  = top + box.box.height - 1;
    const uint8_t color[] = {box.borderColor.r, box.borderColor.g, box.borderColor.b, box.borderColor.a};

    for (int y = top; y <= bottom; ++y)
    {
        for (int x = left; x <= right; ++x)
        {
            if (const bool draw = shouldDrawHostPixel(box, filled, x, y); !draw)
            {
                continue;
            }
            for (int channel = 0; channel < outputAccess.numChannels(); ++channel)
            {
                const auto offset    = hostPixelOffset(outputAccess, outputSampleStride, sample, y, x, channel);
                expectedHost[offset] = color[channel];
            }
        }
    }
}

static void drawHostBoxes(const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t outputSampleStride,
                          const std::vector<std::vector<NVCVBndBoxI>> &boxes, bool filled,
                          std::vector<uint8_t> &expectedHost)
{
    for (int sample = 0; sample < static_cast<int>(boxes.size()); ++sample)
    {
        for (const auto &box : boxes[sample])
        {
            drawHostBox(outputAccess, outputSampleStride, sample, box, filled, expectedHost);
        }
    }
}

static void expectHostPixel(const nvcv::TensorDataAccessStridedImagePlanar &inputAccess,
                            const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t inputSampleStride,
                            int64_t outputSampleStride, int sample, int y, int x, const std::vector<uint8_t> &inputHost,
                            const std::vector<uint8_t> &inputAfterHost, const std::vector<uint8_t> &expectedHost,
                            const std::vector<uint8_t> &outputHost)
{
    for (int channel = 0; channel < outputAccess.numChannels(); ++channel)
    {
        const auto outputOffset = hostPixelOffset(outputAccess, outputSampleStride, sample, y, x, channel);
        EXPECT_EQ(outputHost[outputOffset], expectedHost[outputOffset])
            << "sample=" << sample << ", row=" << y << ", col=" << x << ", channel=" << channel;

        const auto inputOffset = hostPixelOffset(inputAccess, inputSampleStride, sample, y, x, channel);
        EXPECT_EQ(inputAfterHost[inputOffset], inputHost[inputOffset])
            << "input modified at sample=" << sample << ", row=" << y << ", col=" << x << ", channel=" << channel;
    }
}

static void expectHostGold(const nvcv::TensorDataAccessStridedImagePlanar &inputAccess,
                           const nvcv::TensorDataAccessStridedImagePlanar &outputAccess, int64_t inputSampleStride,
                           int64_t outputSampleStride, int numSamples, int height, int width,
                           const std::vector<uint8_t> &inputHost, const std::vector<uint8_t> &inputAfterHost,
                           const std::vector<uint8_t> &expectedHost, const std::vector<uint8_t> &outputHost)
{
    for (int sample = 0; sample < numSamples; ++sample)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                expectHostPixel(inputAccess, outputAccess, inputSampleStride, outputSampleStride, sample, y, x,
                                inputHost, inputAfterHost, expectedHost, outputHost);
            }
        }
    }
}

static void compareOutOfPlaceToHostGold(cudaStream_t stream, const nvcv::ImageFormat &format, bool filled,
                                        int width = 18, int height = 14, int inputAlpha = -1, bool expectDense = false)
{
    constexpr int numSamples = 4;

    nvcv::Tensor input  = nvcv::util::CreateTensor(numSamples, width, height, format);
    nvcv::Tensor output = nvcv::util::CreateTensor(numSamples, width, height, format);

    auto inputData  = input.exportData<nvcv::TensorDataStridedCuda>();
    auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inputData, nullptr);
    ASSERT_NE(outputData, nullptr);

    auto inputAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    auto outputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
    ASSERT_TRUE(inputAccess);
    ASSERT_TRUE(outputAccess);

    ASSERT_EQ(inputAccess->numSamples(), numSamples);
    ASSERT_EQ(inputAccess->numRows(), height);
    ASSERT_EQ(inputAccess->numCols(), width);
    ASSERT_EQ(inputAccess->numChannels(), outputAccess->numChannels());
    ASSERT_TRUE(inputAccess->numChannels() == 3 || inputAccess->numChannels() == 4);

    const auto rowBytes = static_cast<int64_t>(width * inputAccess->numChannels());
    EXPECT_EQ(inputAccess->rowStride() == rowBytes && outputAccess->rowStride() == rowBytes
                  && inputAccess->sampleStride() == inputAccess->rowStride() * height
                  && outputAccess->sampleStride() == outputAccess->rowStride() * height,
              expectDense);

    const int64_t inputSampleStride  = inputAccess->sampleStride() == 0
                                         ? inputAccess->rowStride() * inputAccess->numRows()
                                         : inputAccess->sampleStride();
    const int64_t outputSampleStride = outputAccess->sampleStride() == 0
                                         ? outputAccess->rowStride() * outputAccess->numRows()
                                         : outputAccess->sampleStride();
    const auto    inputSize          = static_cast<size_t>(inputSampleStride * numSamples);
    const auto    outputSize         = static_cast<size_t>(outputSampleStride * numSamples);

    std::vector<uint8_t> inputHost(inputSize, 0xA5);
    std::vector<uint8_t> expectedHost(outputSize, 0x5A);
    fillHostPattern(*inputAccess, *outputAccess, inputSampleStride, outputSampleStride, numSamples, height, width,
                    inputHost, expectedHost);
    if (inputAlpha >= 0)
    {
        ASSERT_EQ(inputAccess->numChannels(), 4);
        fillHostAlpha(*inputAccess, *outputAccess, inputSampleStride, outputSampleStride, numSamples, height, width,
                      static_cast<uint8_t>(inputAlpha), inputHost, expectedHost);
    }

    const int                             filledAligned = filled ? 1 : 2;
    std::vector<std::vector<NVCVBndBoxI>> boxes(numSamples);
    boxes[1].push_back({
        {filledAligned, filledAligned,   8,   8},
        filled ? -1 : 2,
        {           31,            97, 211, 255},
        {            0,             0,   0,   0}
    });
    boxes[2].push_back({
        {filledAligned, filledAligned,   6,   6},
        filled ? -1 : 2,
        {           32,            98, 210, 255},
        {            0,             0,   0,   0}
    });
    boxes[2].push_back({
        {filled ? 11 : 10, filled ? 3 : 4,   6,   8},
        filled ? -1 : 2,
        {              67,            149, 193, 255},
        {               0,              0,   0,   0}
    });
    drawHostBoxes(*outputAccess, outputSampleStride, boxes, filled, expectedHost);

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(boxes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputData->basePtr(), inputHost.data(), inputSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(outputData->basePtr(), 0, outputSize));

    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, input, output, (NVCVBndBoxesI)bndBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> outputHost(outputSize);
    std::vector<uint8_t> inputAfterHost(inputSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(outputHost.data(), outputData->basePtr(), outputSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputAfterHost.data(), inputData->basePtr(), inputSize, cudaMemcpyDeviceToHost));

    expectHostGold(*inputAccess, *outputAccess, inputSampleStride, outputSampleStride, numSamples, height, width,
                   inputHost, inputAfterHost, expectedHost, outputHost);
}

static void compareOddExtentToEvenReference(cudaStream_t stream, const nvcv::ImageFormat &format, bool filled)
{
    constexpr int oddWidth        = 17;
    constexpr int oddHeight       = 15;
    constexpr int referenceWidth  = 18;
    constexpr int referenceHeight = 16;

    nvcv::Tensor oddInput        = nvcv::util::CreateTensor(1, oddWidth, oddHeight, format);
    nvcv::Tensor oddOutput       = nvcv::util::CreateTensor(1, oddWidth, oddHeight, format);
    nvcv::Tensor referenceInput  = nvcv::util::CreateTensor(1, referenceWidth, referenceHeight, format);
    nvcv::Tensor referenceOutput = nvcv::util::CreateTensor(1, referenceWidth, referenceHeight, format);

    auto oddInputData        = oddInput.exportData<nvcv::TensorDataStridedCuda>();
    auto oddOutputData       = oddOutput.exportData<nvcv::TensorDataStridedCuda>();
    auto referenceInputData  = referenceInput.exportData<nvcv::TensorDataStridedCuda>();
    auto referenceOutputData = referenceOutput.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(oddInputData && oddOutputData && referenceInputData && referenceOutputData);

    auto oddInputAccess        = nvcv::TensorDataAccessStridedImagePlanar::Create(*oddInputData);
    auto oddOutputAccess       = nvcv::TensorDataAccessStridedImagePlanar::Create(*oddOutputData);
    auto referenceInputAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*referenceInputData);
    auto referenceOutputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*referenceOutputData);
    ASSERT_TRUE(oddInputAccess && oddOutputAccess && referenceInputAccess && referenceOutputAccess);

    auto effectiveSampleStride = [](const nvcv::TensorDataAccessStridedImagePlanar &access)
    {
        return access.sampleStride() == 0 ? access.rowStride() * access.numRows() : access.sampleStride();
    };

    const int64_t oddInputStride        = effectiveSampleStride(*oddInputAccess);
    const int64_t oddOutputStride       = effectiveSampleStride(*oddOutputAccess);
    const int64_t referenceInputStride  = effectiveSampleStride(*referenceInputAccess);
    const int64_t referenceOutputStride = effectiveSampleStride(*referenceOutputAccess);

    auto initializeInput = [](const nvcv::TensorDataAccessStridedImagePlanar &access, int64_t sampleStride, int width,
                              int height, nvcv::TensorDataStridedCuda &data)
    {
        std::vector<uint8_t> input(static_cast<size_t>(sampleStride), 0xA5);
        std::vector<uint8_t> expected(static_cast<size_t>(sampleStride), 0);
        fillHostPattern(access, access, sampleStride, sampleStride, 1, height, width, input, expected);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(data.basePtr(), input.data(), input.size(), cudaMemcpyHostToDevice));
    };

    initializeInput(*oddInputAccess, oddInputStride, oddWidth, oddHeight, *oddInputData);
    initializeInput(*referenceInputAccess, referenceInputStride, referenceWidth, referenceHeight, *referenceInputData);
    ASSERT_EQ(cudaSuccess, cudaMemset(oddOutputData->basePtr(), 0, oddOutputStride));
    ASSERT_EQ(cudaSuccess, cudaMemset(referenceOutputData->basePtr(), 0, referenceOutputStride));

    std::vector<std::vector<NVCVBndBoxI>> boxes(1);
    boxes[0].push_back({
        {oddWidth - 5, oddHeight - 5,   5,   5},
        filled ? -1 : 2,
        {          83,           173, 227, 255},
        {           0,             0,   0,   0}
    });
    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(boxes);

    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, oddInput, oddOutput, (NVCVBndBoxesI)bndBoxes.get()));
    EXPECT_NO_THROW(op(stream, referenceInput, referenceOutput, (NVCVBndBoxesI)bndBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> oddOutputHost(static_cast<size_t>(oddOutputStride));
    std::vector<uint8_t> referenceOutputHost(static_cast<size_t>(referenceOutputStride));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(oddOutputHost.data(), oddOutputData->basePtr(), oddOutputHost.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(referenceOutputHost.data(), referenceOutputData->basePtr(),
                                      referenceOutputHost.size(), cudaMemcpyDeviceToHost));

    for (int y = 0; y < oddHeight; ++y)
    {
        for (int x = 0; x < oddWidth; ++x)
        {
            for (int channel = 0; channel < oddOutputAccess->numChannels(); ++channel)
            {
                const auto oddOffset = hostPixelOffset(*oddOutputAccess, oddOutputStride, 0, y, x, channel);
                const auto referenceOffset
                    = hostPixelOffset(*referenceOutputAccess, referenceOutputStride, 0, y, x, channel);
                EXPECT_EQ(oddOutputHost[oddOffset], referenceOutputHost[referenceOffset])
                    << "row=" << y << ", col=" << x << ", channel=" << channel;
            }
        }
    }
}

static void compareInPlaceToHostGold(cudaStream_t stream)
{
    constexpr int numSamples = 2;
    constexpr int width      = 18;
    constexpr int height     = 14;

    nvcv::Tensor tensor = nvcv::util::CreateTensor(numSamples, width, height, nvcv::FMT_RGB8);
    auto         data   = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(data, nullptr);

    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    ASSERT_TRUE(access);

    const int64_t        sampleStride = access->sampleStride();
    const auto           size         = static_cast<size_t>(sampleStride * numSamples);
    std::vector<uint8_t> inputHost(size, 0xA5);
    std::vector<uint8_t> expectedHost(size, 0xA5);
    fillHostPattern(*access, *access, sampleStride, sampleStride, numSamples, height, width, inputHost, expectedHost);

    std::vector<std::vector<NVCVBndBoxI>> boxes(numSamples);
    boxes[0].push_back({
        { 2,  2,   8,   8},
        2,
        {31, 97, 211, 255},
        { 0,  0,   0,   0}
    });
    drawHostBoxes(*access, sampleStride, boxes, false, expectedHost);

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(boxes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(data->basePtr(), inputHost.data(), size, cudaMemcpyHostToDevice));

    cvcuda::BndBox op;
    EXPECT_NO_THROW(op(stream, tensor, tensor, (NVCVBndBoxesI)bndBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> actualHost(size);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actualHost.data(), data->basePtr(), size, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expectedHost, actualHost);
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox, test::ValueList<int, int, int, int, int, nvcv::ImageFormat>
{
    //  inN,    inW,    inH,    num,    seed,   format
    {   1,      224,    224,    100,    3,      nvcv::FMT_RGBA8 },
    {   8,      224,    224,    100,    7,      nvcv::FMT_RGBA8 },
    {   16,     224,    224,    100,    11,     nvcv::FMT_RGBA8 },
    {   1,      224,    224,    100,    3,      nvcv::FMT_RGB8  },
    {   8,      224,    224,    100,    7,      nvcv::FMT_RGB8  },
    {   16,     224,    224,    100,    11,     nvcv::FMT_RGB8  },
    {   1,      1280,   720,    100,    23,     nvcv::FMT_RGBA8 },
    {   1,      1920,   1080,   200,    37,     nvcv::FMT_RGBA8 },
    {   1,      3840,   2160,   200,    59,     nvcv::FMT_RGBA8 },
    {   1,      1280,   720,    100,    23,     nvcv::FMT_RGB8  },
    {   1,      1920,   1080,   200,    37,     nvcv::FMT_RGB8  },
    {   1,      3840,   2160,   200,    59,     nvcv::FMT_RGB8  },
});

// clang-format on
TEST_P(OpBndBox, BndBox_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = GetParamValue<0>();
    int               inW    = GetParamValue<1>();
    int               inH    = GetParamValue<2>();
    int               num    = GetParamValue<3>();
    int               sed    = GetParamValue<4>();
    nvcv::ImageFormat format = GetParamValue<5>();
    cvcuda::BndBox    op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBoxPlanar, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int>
{
    // planar format, interleaved format, batch
    {nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,  2},
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, 1},
});

// clang-format on

TEST_P(OpBndBoxPlanar, tensor_output_matches_interleaved)
{
    constexpr int width  = 64;
    constexpr int height = 48;
    int           batch  = GetParamValue<2>();

    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;
    for (int n = 0; n < batch; ++n)
    {
        std::vector<NVCVBndBoxI> curVec;
        NVCVBndBoxI              bndBox;
        bndBox.box         = {width / 4, height / 4, width / 2, height / 2};
        bndBox.thickness   = 3;
        bndBox.borderColor = {255, 255, 0, 255};
        bndBox.fillColor   = {0, 128, 255, 0};
        curVec.push_back(bndBox);
        bndBoxVec.push_back(curVec);
    }
    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    cvcuda::BndBox op;
    nvcv::test::planar::RunTensorParity(
        GetParamValue<0>(), GetParamValue<1>(), width, height, width, height, batch,
        [&op, &bndBoxes](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, dst, (NVCVBndBoxesI)bndBoxes.get()); });
}

// clang-format on
TEST(OpBndBox, BndBox_memory)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = 1;
    int               inW    = 224;
    int               inH    = 224;
    int               num    = 100;
    int               sed    = 3;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;
    cvcuda::BndBox    op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    //check if data is cleared
    sed++;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, planar_workspace_reallocation_preserves_draw)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::BndBox op;
    runOp(stream, op, 2, 18, 14, 4, 11, nvcv::FMT_RGB8p);
    runOp(stream, op, 2, 22, 16, 4, 17, nvcv::FMT_RGB8p);

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_out_of_place_matches_host_gold_rgb8)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8, true);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8, false);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8, false, 256, 14, -1, true);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_out_of_place_matches_host_gold_rgba8)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, true);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, false);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, false, 256, 14, -1, true);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_out_of_place_odd_extents_match_host_gold)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    for (bool filled : {false, true})
    {
        compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8, filled, 17, 14);
        compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8, filled, 18, 15);
        compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, filled, 17, 15);
        compareOddExtentToEvenReference(stream, nvcv::FMT_RGB8, filled);
        compareOddExtentToEvenReference(stream, nvcv::FMT_RGBA8, filled);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_transparent_pixels_preserved)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, false, 17, 15, 0);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8, false, 17, 15, 1);

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_out_of_place_matches_host_gold_rgb8p)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8p, true);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGB8p, false);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_out_of_place_matches_host_gold_rgba8p)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8p, true);
    compareOutOfPlaceToHostGold(stream, nvcv::FMT_RGBA8p, false);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBndBox, BndBox_in_place_matches_host_gold_rgb8)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    compareInPlaceToHostGold(stream);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, int>
    {
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 10, 8, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 10, 10, 8},
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, 10, 8, 10},
    });

// clang-format on

TEST(OpBndBox_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaBndBoxCreate(nullptr));
}

TEST(OpBndBox_Negative, rejects_non_image_tensor)
{
    nvcv::Tensor inTensor{
        {{16}, "N"},
        nvcv::TYPE_U8
    };
    nvcv::Tensor outTensor{
        {{16}, "N"},
        nvcv::TYPE_U8
    };
    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(std::vector<std::vector<NVCVBndBoxI>>(1));

    cvcuda::BndBox op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, inTensor, outTensor, (NVCVBndBoxesI)bndBoxes.get()); }));
}

TEST(OpBndBox_Negative, rejects_single_channel_nhwc_tensor)
{
    const nvcv::TensorShape shape{
        {1, 12, 16, 1},
        "NHWC"
    };
    nvcv::Tensor src(shape, nvcv::TYPE_U8);
    nvcv::Tensor dst(shape, nvcv::TYPE_U8);

    std::vector<std::vector<NVCVBndBoxI>> boxes(1);
    boxes[0].push_back({
        { 2,  2,   8,   6},
        2,
        {31, 97, 211, 255},
        { 0,  0,   0,   0}
    });
    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(boxes);

    cvcuda::BndBox op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, src, dst, (NVCVBndBoxesI)bndBoxes.get()); }));
}

TEST_P(OpBndBox_Negative, invalid_parameters)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFormat  = GetParamValue<0>();
    nvcv::ImageFormat outFormat = GetParamValue<1>();
    int               inN       = GetParamValue<2>();
    int               outN      = GetParamValue<3>();
    int               bboxesN   = GetParamValue<4>();

    int inW = 224;
    int inH = 224;
    int num = 50;

    std::vector<std::vector<NVCVBndBoxI>> bndBoxVec;

    Rng().seed(0);
    for (int n = 0; n < bboxesN; n++)
    {
        std::vector<NVCVBndBoxI> curVec;
        for (int i = 0; i < num; i++)
        {
            NVCVBndBoxI bndBox;
            bndBox.box.x       = randl(0, inW - 1);
            bndBox.box.y       = randl(0, inH - 1);
            bndBox.box.width   = randl(1, inW);
            bndBox.box.height  = randl(1, inH);
            bndBox.thickness   = randl(-1, 30);
            bndBox.fillColor   = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
            bndBox.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)};
            curVec.push_back(bndBox);
        }
        bndBoxVec.push_back(curVec);
    }

    auto bndBoxes = std::make_shared<NVCVBndBoxesImpl>(bndBoxVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, inFormat);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(outN, inW, inH, outFormat);

    cvcuda::BndBox op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &imgIn, &imgOut, &bndBoxes]
                                { op(stream, imgIn, imgOut, (NVCVBndBoxesI)bndBoxes.get()); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
