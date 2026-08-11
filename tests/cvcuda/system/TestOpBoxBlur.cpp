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
#include <cvcuda/OpBoxBlur.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

namespace gt   = ::testing;
namespace test = nvcv::test;
using namespace cvcuda::priv;

constexpr int kBoxBlurCropSize      = 32;
constexpr int kBoxBlurColorChannels = 3;

static std::shared_ptr<NVCVBlurBoxesImpl> MakeBlurBoxes(int numImages, const std::vector<NVCVBlurBoxI> &boxes)
{
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec(numImages, boxes);
    return std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);
}

template<typename T>
static std::vector<T> DownloadTensorPixelsAs(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    static_assert(sizeof(T) == 1);

    const int64_t samples  = access.numSamples();
    const int64_t channels = access.numChannels();
    const int64_t rows     = access.numRows();
    const int64_t cols     = access.numCols();
    const bool    planar   = access.chStride() != 1;

    std::vector<T> pixels;
    pixels.resize(samples * rows * cols * channels);

    auto      *dst      = pixels.data();
    const auto rowBytes = static_cast<size_t>(cols * (planar ? 1 : channels));
    const auto dstPitch = rowBytes;
    const auto height   = static_cast<size_t>(rows);

    for (int64_t n = 0; n < samples; ++n)
    {
        if (planar)
        {
            for (int64_t c = 0; c < channels; ++c)
            {
                const auto *src = access.sampleData(static_cast<int>(n)) + c * access.chStride();
                EXPECT_EQ(cudaSuccess, cudaMemcpy2D(dst, dstPitch, src, static_cast<size_t>(access.rowStride()),
                                                    rowBytes, height, cudaMemcpyDeviceToHost));
                dst += rows * cols;
            }
        }
        else
        {
            EXPECT_EQ(cudaSuccess,
                      cudaMemcpy2D(dst, dstPitch, access.sampleData(static_cast<int>(n)),
                                   static_cast<size_t>(access.rowStride()), rowBytes, height, cudaMemcpyDeviceToHost));
            dst += rows * cols * channels;
        }
    }

    return pixels;
}

static std::vector<uint8_t> DownloadTensorPixels(const nvcv::TensorDataAccessStridedImagePlanar &access)
{
    return DownloadTensorPixelsAs<uint8_t>(access);
}

static size_t TensorPixelOffset(int64_t sample, int64_t channel, int64_t row, int64_t col, int64_t channels,
                                int64_t rows, int64_t cols, bool planar)
{
    if (planar)
    {
        return ((static_cast<size_t>(sample) * static_cast<size_t>(channels) + static_cast<size_t>(channel))
                    * static_cast<size_t>(rows)
                + static_cast<size_t>(row))
                 * static_cast<size_t>(cols)
             + static_cast<size_t>(col);
    }
    return ((static_cast<size_t>(sample) * static_cast<size_t>(rows) + static_cast<size_t>(row))
                * static_cast<size_t>(cols)
            + static_cast<size_t>(col))
             * static_cast<size_t>(channels)
         + static_cast<size_t>(channel);
}

struct TensorPixelCoordinates
{
    int sample;
    int channel;
    int row;
    int col;
};

static TensorPixelCoordinates GetTensorPixelCoordinates(size_t index, int channels, int rows, int cols)
{
    const auto col = static_cast<int>(index % static_cast<size_t>(cols));
    index /= static_cast<size_t>(cols);
    const auto row = static_cast<int>(index % static_cast<size_t>(rows));
    index /= static_cast<size_t>(rows);
    const auto channel = static_cast<int>(index % static_cast<size_t>(channels));
    const auto sample  = static_cast<int>(index / static_cast<size_t>(channels));
    return {sample, channel, row, col};
}

static std::vector<uint8_t> MakeBoxBlurInput(int samples, int channels, int rows, int cols, bool planar)
{
    std::vector<uint8_t> pixels(static_cast<size_t>(samples) * channels * rows * cols);
    for (size_t index = 0; index < pixels.size(); ++index)
    {
        const auto [sample, channel, row, col] = GetTensorPixelCoordinates(index, channels, rows, cols);
        pixels[TensorPixelOffset(sample, channel, row, col, channels, rows, cols, planar)]
            = static_cast<uint8_t>((sample * 53 + channel * 71 + row * 29 + col * 17 + 13) & 0xFF);
    }
    return pixels;
}

template<typename T>
static void UploadTensorPixels(const nvcv::TensorDataAccessStridedImagePlanar &access, const std::vector<T> &pixels)
{
    static_assert(sizeof(T) == 1);

    const int64_t samples  = access.numSamples();
    const int64_t channels = access.numChannels();
    const int64_t rows     = access.numRows();
    const int64_t cols     = access.numCols();
    const bool    planar   = access.chStride() != 1;

    ASSERT_EQ(pixels.size(), static_cast<size_t>(samples) * channels * rows * cols);
    for (int64_t sample = 0; sample < samples; ++sample)
    {
        if (planar)
        {
            for (int64_t channel = 0; channel < channels; ++channel)
            {
                const auto offset = TensorPixelOffset(sample, channel, 0, 0, channels, rows, cols, planar);
                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy2D(access.sampleData(static_cast<int>(sample)) + channel * access.chStride(),
                                       static_cast<size_t>(access.rowStride()), pixels.data() + offset,
                                       static_cast<size_t>(cols), static_cast<size_t>(cols), static_cast<size_t>(rows),
                                       cudaMemcpyHostToDevice));
            }
        }
        else
        {
            const auto offset   = TensorPixelOffset(sample, 0, 0, 0, channels, rows, cols, planar);
            const auto rowBytes = static_cast<size_t>(cols * channels);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(access.sampleData(static_cast<int>(sample)),
                                                static_cast<size_t>(access.rowStride()), pixels.data() + offset,
                                                rowBytes, rowBytes, static_cast<size_t>(rows), cudaMemcpyHostToDevice));
        }
    }
}

template<typename T>
using BoxBlurCropBuffer = std::array<T, kBoxBlurCropSize * kBoxBlurCropSize * kBoxBlurColorChannels>;

static size_t BoxBlurCropOffset(int channel, int row, int col)
{
    return (static_cast<size_t>(channel) * kBoxBlurCropSize + row) * kBoxBlurCropSize + col;
}

static int BoxToCropCoordinate(int relativeCoordinate, int boxExtent)
{
    return static_cast<int>(static_cast<float>(relativeCoordinate) / static_cast<float>(boxExtent)
                            * static_cast<float>(kBoxBlurCropSize));
}

static int CropToImageCoordinate(int cropCoordinate, int boxExtent, int origin, int imageExtent)
{
    const float coordinate
        = static_cast<float>(cropCoordinate) / static_cast<float>(kBoxBlurCropSize) * static_cast<float>(boxExtent)
        + 0.5F + static_cast<float>(origin);
    return std::clamp(static_cast<int>(coordinate), 0, imageExtent - 1);
}

template<typename T>
static void CopyBoxToCrop(const std::vector<T> &input, BoxBlurCropBuffer<T> &crop, int sample, int channels, int rows,
                          int cols, bool planar, int left, int top, int boxWidth, int boxHeight)
{
    for (int channel = 0; channel < std::min(channels, kBoxBlurColorChannels); ++channel)
    {
        for (int cropRow = 0; cropRow < kBoxBlurCropSize; ++cropRow)
        {
            for (int cropCol = 0; cropCol < kBoxBlurCropSize; ++cropCol)
            {
                const int srcCol = CropToImageCoordinate(cropCol, boxWidth, left, cols);
                const int srcRow = CropToImageCoordinate(cropRow, boxHeight, top, rows);
                crop[BoxBlurCropOffset(channel, cropRow, cropCol)]
                    = input[TensorPixelOffset(sample, channel, srcRow, srcCol, channels, rows, cols, planar)];
            }
        }
    }
}

template<typename T>
static T BlurCropPixel(const BoxBlurCropBuffer<T> &crop, int channel, int cropRow, int cropCol, int radius)
{
    using SumType = std::conditional_t<std::is_signed_v<T>, int32_t, uint32_t>;
    SumType sum   = 0;
    int     count = 0;
    for (int filterRow = -radius; filterRow <= radius; ++filterRow)
    {
        for (int filterCol = -radius; filterCol <= radius; ++filterCol)
        {
            const int sampleRow = cropRow + filterRow;
            const int sampleCol = cropCol + filterCol;
            if (sampleRow < 0 || sampleRow >= kBoxBlurCropSize || sampleCol < 0 || sampleCol >= kBoxBlurCropSize)
            {
                continue;
            }
            sum += crop[BoxBlurCropOffset(channel, sampleRow, sampleCol)];
            ++count;
        }
    }
    if (count == 0)
    {
        return T{};
    }
    return static_cast<T>(sum / count);
}

template<typename T>
static void BlurCrop(const BoxBlurCropBuffer<T> &crop, BoxBlurCropBuffer<T> &blurred, int channels, int radius)
{
    for (int channel = 0; channel < std::min(channels, kBoxBlurColorChannels); ++channel)
    {
        for (int cropRow = 0; cropRow < kBoxBlurCropSize; ++cropRow)
        {
            for (int cropCol = 0; cropCol < kBoxBlurCropSize; ++cropCol)
            {
                blurred[BoxBlurCropOffset(channel, cropRow, cropCol)]
                    = BlurCropPixel(crop, channel, cropRow, cropCol, radius);
            }
        }
    }
}

template<typename T>
static void CopyCropToBox(const BoxBlurCropBuffer<T> &blurred, std::vector<T> &output, int sample, int channels,
                          int rows, int cols, bool planar, int left, int top, int boxWidth, int boxHeight)
{
    const int gapWidth     = (boxWidth + kBoxBlurCropSize - 1) / kBoxBlurCropSize;
    const int gapHeight    = (boxHeight + kBoxBlurCropSize - 1) / kBoxBlurCropSize;
    const int coveredWidth = kBoxBlurCropSize * gapWidth;

    for (int relativeRow = 0; relativeRow < kBoxBlurCropSize * gapHeight; ++relativeRow)
    {
        const int dstRow = relativeRow + top;
        if (dstRow < 0 || dstRow >= rows)
        {
            continue;
        }
        const int srcRow = BoxToCropCoordinate(relativeRow, boxHeight);
        if (srcRow >= kBoxBlurCropSize)
        {
            continue;
        }

        for (int relativeCol = 0; relativeCol < coveredWidth; ++relativeCol)
        {
            const int dstCol = relativeCol + left;
            if (dstCol < 0 || dstCol >= cols)
            {
                continue;
            }
            const int srcCol = BoxToCropCoordinate(relativeCol, boxWidth);
            if (srcCol >= kBoxBlurCropSize)
            {
                continue;
            }

            for (int channel = 0; channel < std::min(channels, kBoxBlurColorChannels); ++channel)
            {
                output[TensorPixelOffset(sample, channel, dstRow, dstCol, channels, rows, cols, planar)]
                    = blurred[BoxBlurCropOffset(channel, srcRow, srcCol)];
            }
            if (channels == 4)
            {
                output[TensorPixelOffset(sample, 3, dstRow, dstCol, channels, rows, cols, planar)]
                    = std::numeric_limits<T>::max();
            }
        }
    }
}

template<typename T>
static std::vector<T> BoxBlurReference(const std::vector<T> &input, int samples, int channels, int rows, int cols,
                                       bool planar, const std::vector<NVCVBlurBoxI> &boxes)
{
    static_assert(sizeof(T) == 1 && std::is_integral_v<T>);

    BoxBlurCropBuffer<T> crop{};
    BoxBlurCropBuffer<T> blurred{};
    std::vector<T>       output = input;

    for (int sample = 0; sample < samples; ++sample)
    {
        for (const NVCVBlurBoxI &box : boxes)
        {
            const int left   = std::clamp(box.box.x, 0, cols - 1);
            const int top    = std::clamp(box.box.y, 0, rows - 1);
            const int right  = std::clamp(left + box.box.width - 1, 0, cols - 1);
            const int bottom = std::clamp(top + box.box.height - 1, 0, rows - 1);
            if (left == right || top == bottom || box.box.width < 3 || box.box.height < 3 || box.kernelSize < 1)
            {
                continue;
            }

            const int boxWidth  = right - left;
            const int boxHeight = bottom - top;
            CopyBoxToCrop(input, crop, sample, channels, rows, cols, planar, left, top, boxWidth, boxHeight);
            BlurCrop(crop, blurred, channels, box.kernelSize / 2);
            CopyCropToBox(blurred, output, sample, channels, rows, cols, planar, left, top, boxWidth, boxHeight);
        }
    }
    return output;
}

static std::vector<int8_t> MakeSignedBoxBlurInput(int samples, int channels, int rows, int cols, bool planar)
{
    std::vector<int8_t> pixels(static_cast<size_t>(samples) * channels * rows * cols);
    for (size_t index = 0; index < pixels.size(); ++index)
    {
        const auto [sample, channel, row, col] = GetTensorPixelCoordinates(index, channels, rows, cols);
        const int value                        = (sample * 43 + channel * 61 + row * 23 + col * 17) % 191 - 95;
        pixels[TensorPixelOffset(sample, channel, row, col, channels, rows, cols, planar)] = static_cast<int8_t>(value);
    }
    return pixels;
}

static void runOp(cudaStream_t &stream, const cvcuda::BoxBlur &op, int inN, int inW, int inH, int cols, int rows,
                  int wBox, int hBox, int ks, const nvcv::ImageFormat &format)
{
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;

    for (int n = 0; n < inN; n++)
    {
        std::vector<NVCVBlurBoxI> curVec;
        for (int i = 0; i < cols; i++)
        {
            int x = (inW / cols) * i + wBox / 2;
            for (int j = 0; j < rows; j++)
            {
                NVCVBlurBoxI blurBox;
                blurBox.box.x      = x;
                blurBox.box.y      = (inH / rows) * j + hBox / 2;
                blurBox.box.width  = wBox;
                blurBox.box.height = hBox;
                blurBox.kernelSize = ks;
                curVec.push_back(blurBox);
            }
        }
        blurBoxVec.push_back(curVec);
    }

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

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

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    auto inBufSize  = static_cast<int>(inSampleStride * inAccess->numSamples());
    auto outBufSize = static_cast<int>(outSampleStride * outAccess->numSamples());

    std::vector<uint8_t>          inVec(inBufSize);
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);
    std::ranges::generate(inVec, [&rand, &randEng]() { return rand(randEng); });

    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output->basePtr(), inVec.data(), outBufSize, cudaMemcpyHostToDevice));

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()));

    std::vector<uint8_t> outHost(outBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outHost.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_NE(inVec, outHost) << "Output should differ from input after applying box blur";
}

TEST(OpBoxBlur, BoxBlur_memory)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               inN    = 1;
    int               inW    = 224;
    int               inH    = 224;
    int               cols   = 5;
    int               rows   = 5;
    int               wBox   = 16;
    int               hBox   = 16;
    int               ks     = 7;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;
    cvcuda::BoxBlur   op;
    runOp(stream, op, inN, inW, inH, cols, rows, wBox, hBox, ks, format);
    hBox += 3;
    runOp(stream, op, inN, inW, inH, cols, rows, wBox, hBox, ks, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlurInPlaceParity, test::ValueList<nvcv::ImageFormat>
{
    nvcv::FMT_RGB8,
    nvcv::FMT_RGBA8,
    nvcv::FMT_RGB8p,
    nvcv::FMT_RGBA8p,
});

// clang-format on

TEST_P(OpBoxBlurInPlaceParity, out_of_place_matches_in_place)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int     numImages = 2;
    constexpr int     width     = 128;
    constexpr int     height    = 96;
    nvcv::ImageFormat format    = GetParam();

    nvcv::Tensor src        = nvcv::util::CreateTensor(numImages, width, height, format);
    nvcv::Tensor outOfPlace = nvcv::util::CreateTensor(numImages, width, height, format);
    nvcv::Tensor inPlace    = nvcv::util::CreateTensor(numImages, width, height, format);

    auto srcData        = src.exportData<nvcv::TensorDataStridedCuda>();
    auto outOfPlaceData = outOfPlace.exportData<nvcv::TensorDataStridedCuda>();
    auto inPlaceData    = inPlace.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(outOfPlaceData, nullptr);
    ASSERT_NE(inPlaceData, nullptr);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    const int64_t bufferSize = srcAccess->sampleStride() * srcAccess->numSamples();
    ASSERT_GT(bufferSize, 0);

    std::vector<uint8_t> input(bufferSize);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcData->basePtr(), input.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inPlaceData->basePtr(), input.data(), bufferSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(outOfPlaceData->basePtr(), 0xCD, bufferSize));

    std::vector<NVCVBlurBoxI> boxes{
        {  {8, 7, 32, 31}, 5},
        {{72, 41, 29, 23}, 7},
    };
    auto blurBoxes = MakeBlurBoxes(numImages, boxes);

    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, src, outOfPlace, (NVCVBlurBoxesI)blurBoxes.get()));
    EXPECT_NO_THROW(op(stream, inPlace, inPlace, (NVCVBlurBoxesI)blurBoxes.get()));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto outOfPlaceAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outOfPlaceData);
    auto inPlaceAccess    = nvcv::TensorDataAccessStridedImagePlanar::Create(*inPlaceData);
    ASSERT_TRUE(outOfPlaceAccess);
    ASSERT_TRUE(inPlaceAccess);

    EXPECT_EQ(DownloadTensorPixels(*inPlaceAccess), DownloadTensorPixels(*outOfPlaceAccess));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlurCpuGold, test::ValueList<nvcv::ImageFormat>
{
    nvcv::FMT_RGB8,
    nvcv::FMT_RGBA8,
    nvcv::FMT_RGB8p,
    nvcv::FMT_RGBA8p,
});

// clang-format on

TEST_P(OpBoxBlurCpuGold, tensor_correct_output_matches_independent_cpu_reference)
{
    constexpr int samples = 2;
    constexpr int width   = 128;
    constexpr int height  = 96;

    const nvcv::ImageFormat format = GetParam();
    nvcv::Tensor            input  = nvcv::util::CreateTensor(samples, width, height, format);
    nvcv::Tensor            output = nvcv::util::CreateTensor(samples, width, height, format);

    auto inputData  = input.exportData<nvcv::TensorDataStridedCuda>();
    auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData);
    ASSERT_TRUE(outputData);

    auto inputAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    auto outputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
    ASSERT_TRUE(inputAccess);
    ASSERT_TRUE(outputAccess);

    const int  channels  = inputAccess->numChannels();
    const bool planar    = inputAccess->chStride() != 1;
    const auto inputHost = MakeBoxBlurInput(samples, channels, height, width, planar);
    UploadTensorPixels(*inputAccess, inputHost);
    ASSERT_EQ(cudaSuccess, cudaMemset(outputData->basePtr(), 0xCD,
                                      static_cast<size_t>(outputAccess->sampleStride() * outputAccess->numSamples())));

    const std::vector<NVCVBlurBoxI> boxes{
        {  {-4, -3, 7, 6}, 3},
        {  {8, 7, 32, 31}, 5},
        {{72, 41, 29, 23}, 7},
    };
    const auto reference = BoxBlurReference(inputHost, samples, channels, height, width, planar, boxes);
    auto       blurBoxes = MakeBlurBoxes(samples, boxes);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, input, output, (NVCVBlurBoxesI)blurBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(reference, DownloadTensorPixels(*outputAccess));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlurSignedCpuGold, test::ValueList<nvcv::TensorLayout, int>
{
    {nvcv::TENSOR_NHWC, 3},
    {nvcv::TENSOR_NHWC, 4},
    {nvcv::TENSOR_NCHW, 3},
    {nvcv::TENSOR_NCHW, 4},
});

// clang-format on

TEST_P(OpBoxBlurSignedCpuGold, negative_s8_values_match_independent_cpu_reference)
{
    constexpr int samples = 1;
    constexpr int width   = 64;
    constexpr int height  = 48;

    const nvcv::TensorLayout layout   = GetParamValue<0>();
    const int                channels = GetParamValue<1>();
    const bool               planar   = layout == nvcv::TENSOR_NCHW;
    const nvcv::TensorShape  shape
        = planar ? nvcv::TensorShape{{samples, channels, height, width}, layout}
                 : nvcv::TensorShape{{samples, height, width, channels}, layout};

    nvcv::Tensor input(shape, nvcv::TYPE_S8);
    nvcv::Tensor output(shape, nvcv::TYPE_S8);

    auto inputData  = input.exportData<nvcv::TensorDataStridedCuda>();
    auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(inputData);
    ASSERT_TRUE(outputData);

    auto inputAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inputData);
    auto outputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
    ASSERT_TRUE(inputAccess);
    ASSERT_TRUE(outputAccess);

    const auto inputHost = MakeSignedBoxBlurInput(samples, channels, height, width, planar);
    ASSERT_TRUE(std::ranges::any_of(inputHost, [](int8_t value) { return value < 0; }));
    UploadTensorPixels(*inputAccess, inputHost);
    ASSERT_EQ(cudaSuccess, cudaMemset(outputData->basePtr(), 0xCD,
                                      static_cast<size_t>(outputAccess->sampleStride() * outputAccess->numSamples())));

    const std::vector<NVCVBlurBoxI> boxes{
        {{7, 5, 41, 31}, 5}
    };
    const auto reference = BoxBlurReference(inputHost, samples, channels, height, width, planar, boxes);
    auto       blurBoxes = MakeBlurBoxes(samples, boxes);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, input, output, (NVCVBlurBoxesI)blurBoxes.get()));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(reference, DownloadTensorPixelsAs<int8_t>(*outputAccess));
    EXPECT_EQ(inputHost, DownloadTensorPixelsAs<int8_t>(*inputAccess));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlurPlanar,
                  test::ValueList<int, int, int, int, int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>
{
    // numImages, width, height, box x/y/w/h, kernel, planar format, interleaved format
    {2, 37, 41, 3, 5, 24, 25, 5,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8 },
    {2, 37, 41, 3, 5, 24, 25, 5,  nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {1, 64, 48, 7, 9, 31, 29, 17, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8 },
    {1, 64, 48, 7, 9, 31, 29, 17, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
});

// clang-format on

TEST_P(OpBoxBlurPlanar, tensor_matches_interleaved)
{
    int               numImages      = GetParamValue<0>();
    int               width          = GetParamValue<1>();
    int               height         = GetParamValue<2>();
    int               boxX           = GetParamValue<3>();
    int               boxY           = GetParamValue<4>();
    int               boxW           = GetParamValue<5>();
    int               boxH           = GetParamValue<6>();
    int               kernelSize     = GetParamValue<7>();
    nvcv::ImageFormat planarFmt      = GetParamValue<8>();
    nvcv::ImageFormat interleavedFmt = GetParamValue<9>();

    NVCVBlurBoxI blurBox{
        {boxX, boxY, boxW, boxH},
        kernelSize
    };

    auto            blurBoxes = MakeBlurBoxes(numImages, {blurBox});
    cvcuda::BoxBlur op;

    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, width, height, width, height, numImages,
        [&op, &blurBoxes](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, dst, (NVCVBlurBoxesI)blurBoxes.get()); });
}

// clang-format off
NVCV_TEST_SUITE_P(OpBoxBlur_Negative, test::ValueList<int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>
{
    {2, 2, 224, 224, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
    {2, 2, 224, 224, nvcv::FMT_RGB8, nvcv::FMT_RGB8p},
    {2, 2, 224, 224, nvcv::FMT_RGB8, nvcv::FMT_RGBf32},
    {2, 2, 224, 224, nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32},
    {2, 3, 224, 224, nvcv::FMT_RGB8, nvcv::FMT_RGB8},
    {2, 2, 224, 230, nvcv::FMT_RGB8, nvcv::FMT_RGB8},
    {2, 2, 224, 230, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8},
});

// clang-format on

TEST(OpBoxBlur_Negative, createWillNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaBoxBlurCreate(nullptr));
}

TEST_P(OpBoxBlur_Negative, invalid_parameters)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               inN       = GetParamValue<0>();
    int               bboxesN   = GetParamValue<1>();
    int               inW       = GetParamValue<2>();
    int               outW      = GetParamValue<3>();
    nvcv::ImageFormat inFormat  = GetParamValue<4>();
    nvcv::ImageFormat outFormat = GetParamValue<5>();

    int inH  = 224;
    int cols = 5;
    int rows = 5;
    int wBox = 16;
    int hBox = 16;
    int ks   = 7;

    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;

    for (int n = 0; n < bboxesN; n++)
    {
        std::vector<NVCVBlurBoxI> curVec;
        for (int i = 0; i < cols; i++)
        {
            int x = (inW / cols) * i + wBox / 2;
            for (int j = 0; j < rows; j++)
            {
                NVCVBlurBoxI blurBox;
                blurBox.box.x      = x;
                blurBox.box.y      = (inH / rows) * j + hBox / 2;
                blurBox.box.width  = wBox;
                blurBox.box.height = hBox;
                blurBox.kernelSize = ks;
                curVec.push_back(blurBox);
            }
        }
        blurBoxVec.push_back(curVec);
    }

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, inFormat);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, outW, inH, outFormat);

    // run operator
    cvcuda::BoxBlur op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &imgIn, &imgOut, &blurBoxes]
                                { op(stream, imgIn, imgOut, (NVCVBlurBoxesI)blurBoxes.get()); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpBoxBlur, test_nothing_to_apply)
{
    int               inN    = 1;
    int               inW    = 100;
    int               inH    = 100;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    // Create a blur box that won't be applied by osd (width < 3)
    std::vector<std::vector<NVCVBlurBoxI>> blurBoxVec;
    std::vector<NVCVBlurBoxI>              curVec;

    // left == right
    {
        NVCVBlurBoxI blurBox;
        blurBox.box.x      = 10;
        blurBox.box.y      = 10;
        blurBox.box.width  = 1; // width = 1, right = left + 1 - 1 = left
        blurBox.box.height = 10;
        blurBox.kernelSize = 7;
        curVec.push_back(blurBox);
    }

    // top == bottom
    {
        NVCVBlurBoxI blurBox;
        blurBox.box.x      = 30;
        blurBox.box.y      = 30;
        blurBox.box.width  = 10;
        blurBox.box.height = 1; // height = 1, bottom = top + 1 - 1 = top
        blurBox.kernelSize = 7;
        curVec.push_back(blurBox);
    }

    // width < 3
    {
        NVCVBlurBoxI blurBox;
        blurBox.box.x      = 10;
        blurBox.box.y      = 10;
        blurBox.box.width  = 2;
        blurBox.box.height = 10;
        blurBox.kernelSize = 7;
        curVec.push_back(blurBox);
    }

    blurBoxVec.push_back(curVec);

    auto blurBoxes = std::make_shared<NVCVBlurBoxesImpl>(blurBoxVec);

    nvcv::Tensor img = nvcv::util::CreateTensor(inN, inW, inH, format);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto input = img.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    long inSampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, inSampleStride));

    cvcuda::BoxBlur op;
    EXPECT_NO_THROW(op(stream, img, img, (NVCVBlurBoxesI)blurBoxes.get()));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
