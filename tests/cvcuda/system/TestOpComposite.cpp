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

#include <common/ValueTests.hpp>
#include <cvcuda/OpComposite.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

//#define DBG_COMPOSITE 1

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

template<typename T>
static void print_img(const std::vector<T> &vec, int rowStride, int height, const std::string &message)
{
    std::cout << std::endl;
    std::cout << message << std::endl;
    for (int k = 0; k < height; k++)
    {
        for (int j = 0; j < rowStride; j++)
        {
            std::cout << static_cast<int>(vec[k * rowStride + j]) << ",";
        }
        std::cout << std::endl;
    }
}

static void setGoldBuffer(std::vector<uint8_t> &gold, std::vector<uint8_t> &fg, std::vector<uint8_t> bg,
                          std::vector<uint8_t> fgMask, int width, int height, int inVecRowStride,
                          int fgMaskVecRowStride, int outVecRowStride, int inChannels, int outChannels)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int            fg_offset     = r * inVecRowStride + c * inChannels;
            int            fgMask_offset = r * fgMaskVecRowStride + c;
            int            dst_offset    = r * outVecRowStride + c * outChannels;
            uint8_t       *ptrGold       = gold.data() + dst_offset;
            const uint8_t *ptrFg         = fg.data() + fg_offset;
            const uint8_t *ptrBg         = bg.data() + fg_offset;
            const uint8_t *ptrMat        = fgMask.data() + fgMask_offset;
            uint8_t        a             = *ptrMat;
            for (int k = 0; k < inChannels; k++)
            {
                auto c0    = static_cast<float>(ptrBg[k]);
                auto c1    = static_cast<float>(ptrFg[k]);
                auto alpha = static_cast<float>(a) / 255.f;
                ptrGold[k] = static_cast<uint8_t>(std::lerp(c0, c1, alpha) + 0.5f);
            }
            if (inChannels == 3 && outChannels == 4)
            {
                ptrGold[3] = 255;
            }
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpComposite, test::ValueList<int, int, int, int, int>
{
    //inWidth, inHeight, in_channels, out_channels, numberImages
    {       8,        6,          3,             3,          1},
    {       8,        6,          3,             3,          4},
    {       8,        6,          3,             4,          1},
    {       8,        6,          3,             4,          4},

    //inWidth, inHeight, in_channels, out_channels, numberImages
    {      16,       16,           3,            3,          1},
    {      16,       16,           3,            3,          4},
    {      16,       16,           3,            4,          1},
    {      16,       16,           3,            4,          4},
});

// clang-format on
TEST_P(OpComposite, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int inWidth        = GetParamValue<0>();
    int inHeight       = GetParamValue<1>();
    int inChannels     = GetParamValue<2>();
    int outChannels    = GetParamValue<3>();
    int numberOfImages = GetParamValue<4>();

    int outWidth  = inWidth;
    int outHeight = inHeight;

    nvcv::ImageFormat inFormat;
    nvcv::ImageFormat outFormat;

    if (inChannels == 3)
        inFormat = nvcv::FMT_RGB8;
    if (inChannels == 4)
        inFormat = nvcv::FMT_RGBA8;
    if (outChannels == 3)
        outFormat = nvcv::FMT_RGB8;
    if (outChannels == 4)
        outFormat = nvcv::FMT_RGBA8;

    assert(inChannels <= outChannels);

    nvcv::Tensor foregroundImg(numberOfImages, {inWidth, inHeight}, inFormat);
    nvcv::Tensor backgroundImg(numberOfImages, {inWidth, inHeight}, inFormat);
    nvcv::Tensor fgMaskImg(numberOfImages, {inWidth, inHeight}, nvcv::FMT_U8);
    nvcv::Tensor outImg(numberOfImages, {outWidth, outHeight}, outFormat);

    auto foregroundData = foregroundImg.exportData<nvcv::TensorDataStridedCuda>();
    auto backgroundData = backgroundImg.exportData<nvcv::TensorDataStridedCuda>();
    auto fgMaskData     = fgMaskImg.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, foregroundData);
    ASSERT_NE(nullptr, backgroundData);
    ASSERT_NE(nullptr, fgMaskData);

    auto foregroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*foregroundData);
    ASSERT_TRUE(foregroundAccess);

    auto backgroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*backgroundData);
    ASSERT_TRUE(foregroundAccess);

    auto fgMaskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*fgMaskData);
    ASSERT_TRUE(fgMaskAccess);

    auto foregroundBufSize = static_cast<size_t>(foregroundAccess->sampleStride() * foregroundAccess->numSamples());
    auto fgMaskBufSize     = static_cast<size_t>(fgMaskAccess->sampleStride() * fgMaskAccess->numSamples());

    EXPECT_EQ(cudaSuccess, cudaMemset(foregroundData->basePtr(), 0x00, foregroundBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(backgroundData->basePtr(), 0x00, foregroundBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(fgMaskData->basePtr(), 0x00, fgMaskBufSize));

    std::vector<std::vector<uint8_t>> foregroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> backgroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> fgMaskVec(numberOfImages);

    std::default_random_engine rng;

    int inVecRowStride     = inWidth * inFormat.planePixelStrideBytes(0);
    int fgMaskVecRowStride = inWidth * nvcv::FMT_U8.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; i++)
    {
        foregroundVec[i].resize(inHeight * inVecRowStride);
        backgroundVec[i].resize(inHeight * inVecRowStride);
        fgMaskVec[i].resize(inHeight * fgMaskVecRowStride);

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        std::ranges::generate(foregroundVec[i], [&udist, &rng]() { return udist(rng); });
        std::ranges::generate(backgroundVec[i], [&udist, &rng]() { return udist(rng); });
        std::ranges::generate(fgMaskVec[i], [&udist, &rng]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(foregroundAccess->sampleData(i), foregroundAccess->rowStride(), foregroundVec[i].data(),
                               inVecRowStride, inVecRowStride, inHeight, cudaMemcpyHostToDevice));
        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(backgroundAccess->sampleData(i), backgroundAccess->rowStride(), backgroundVec[i].data(),
                               inVecRowStride, inVecRowStride, inHeight, cudaMemcpyHostToDevice));
        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(fgMaskAccess->sampleData(i), fgMaskAccess->rowStride(), fgMaskVec[i].data(),
                                            fgMaskVecRowStride, fgMaskVecRowStride, inHeight, cudaMemcpyHostToDevice));
    }

    // run operator
    cvcuda::Composite compositeOp;

    EXPECT_NO_THROW(compositeOp(stream, foregroundImg, backgroundImg, fgMaskImg, outImg));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // check cdata
    auto outData = outImg.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int outVecRowStride = outWidth * outFormat.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(inHeight * outVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), outVecRowStride, outAccess->sampleData(i), outAccess->rowStride(),
                               outVecRowStride, outHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(outHeight * outVecRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // generate gold result
        setGoldBuffer(goldVec, foregroundVec[i], backgroundVec[i], fgMaskVec[i], inWidth, inHeight, inVecRowStride,
                      fgMaskVecRowStride, outVecRowStride, inChannels, outChannels);

#ifdef DBG_COMPOSITE
        print_img<uint8_t>(foregroundVec[i], inVecRowStride, inHeight, "Foreground");
        print_img<uint8_t>(backgroundVec[i], inVecRowStride, inHeight, "Background");
        print_img<uint8_t>(fgMaskVec[i], fgMaskVecRowStride, inHeight, "Foreground Mask");
        print_img<uint8_t>(goldVec, outVecRowStride, outHeight, "Golden output");
        print_img<uint8_t>(testVec, outVecRowStride, outHeight, "Test output");
#endif
        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpComposite, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int inWidth        = GetParamValue<0>();
    int inHeight       = GetParamValue<1>();
    int inChannels     = GetParamValue<2>();
    int outChannels    = GetParamValue<3>();
    int numberOfImages = GetParamValue<4>();

    nvcv::ImageFormat inFormat;
    nvcv::ImageFormat outFormat;
    nvcv::ImageFormat maskFormat = nvcv::FMT_U8;

    if (inChannels == 3)
        inFormat = nvcv::FMT_RGB8;
    if (inChannels == 4)
        inFormat = nvcv::FMT_RGBA8;
    if (outChannels == 3)
        outFormat = nvcv::FMT_RGB8;
    if (outChannels == 4)
        outFormat = nvcv::FMT_RGBA8;

    assert(inChannels <= outChannels);

    std::default_random_engine rng;

    // Create input varshape

    std::uniform_int_distribution udistWidth(ScaledSize(inWidth, 0.8), ScaledSize(inWidth, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(inHeight, 0.8), ScaledSize(inHeight, 1.1));

    std::vector<nvcv::Image> imgForeground;
    std::vector<nvcv::Image> imgBackground;
    std::vector<nvcv::Image> imgFgMask;
    std::vector<nvcv::Image> imgOut;

    std::vector<std::vector<uint8_t>> foregroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> backgroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> fgMaskVec(numberOfImages);

    std::vector<nvcv::Size2D> sizeVec(numberOfImages);

    std::vector<int> inVecRowStride(numberOfImages);
    std::vector<int> fgMaskVecRowStride(numberOfImages);

    for (int i = 0; i < numberOfImages; i++)
    {
        nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};
        sizeVec[i] = size;

        imgForeground.emplace_back(size, inFormat);
        imgBackground.emplace_back(size, inFormat);
        imgFgMask.emplace_back(size, maskFormat);
        imgOut.emplace_back(size, outFormat);

        int foregroundRowStride = size.w * inFormat.numChannels();
        int fgMaskRowStride     = size.w * maskFormat.numChannels();

        inVecRowStride[i]     = foregroundRowStride;
        fgMaskVecRowStride[i] = fgMaskRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        // resize the vectors to proper size
        foregroundVec[i].resize(size.h * foregroundRowStride);
        backgroundVec[i].resize(size.h * foregroundRowStride);
        fgMaskVec[i].resize(size.h * fgMaskRowStride);

        // populate the vector entries
        std::ranges::generate(foregroundVec[i], [&udist, &rng]() { return udist(rng); });
        std::ranges::generate(backgroundVec[i], [&udist, &rng]() { return udist(rng); });
        std::ranges::generate(fgMaskVec[i], [&udist, &rng]() { return udist(rng); });

        auto imgDataForeground = imgForeground[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(imgDataForeground);

        auto imgDataBackground = imgBackground[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(imgDataBackground);

        auto imgDataFgMask = imgFgMask[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(imgDataFgMask);

        // Copy foreground image data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(imgDataForeground->plane(0).basePtr, imgDataForeground->plane(0).rowStride,
                                            foregroundVec[i].data(), foregroundRowStride, foregroundRowStride, size.h,
                                            cudaMemcpyHostToDevice));

        // Copy background image data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(imgDataBackground->plane(0).basePtr, imgDataBackground->plane(0).rowStride,
                                            backgroundVec[i].data(), foregroundRowStride, foregroundRowStride, size.h,
                                            cudaMemcpyHostToDevice));

        // Copy foreground mask image data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(imgDataFgMask->plane(0).basePtr, imgDataFgMask->plane(0).rowStride, fgMaskVec[i].data(),
                               fgMaskRowStride, fgMaskRowStride, size.h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape batchForeground(numberOfImages);
    nvcv::ImageBatchVarShape batchBackground(numberOfImages);
    nvcv::ImageBatchVarShape batchFgMask(numberOfImages);
    nvcv::ImageBatchVarShape batchOutput(numberOfImages);

    batchForeground.pushBack(imgForeground.begin(), imgForeground.end());
    batchBackground.pushBack(imgBackground.begin(), imgBackground.end());
    batchFgMask.pushBack(imgFgMask.begin(), imgFgMask.end());
    batchOutput.pushBack(imgOut.begin(), imgOut.end());

    // Generate test result
    cvcuda::Composite compositeOp;
    EXPECT_NO_THROW(compositeOp(stream, batchForeground, batchBackground, batchFgMask, batchOutput));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        int width  = sizeVec[i].w;
        int height = sizeVec[i].h;

        const auto outData = imgOut[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(outData->numPlanes() == 1);

        int outRowStride        = width * outFormat.numChannels();
        int foregroundRowStride = width * inFormat.numChannels();
        int fgMaskRowStride     = width * maskFormat.numChannels();

        std::vector<uint8_t> testVec(height * outRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), outRowStride, outData->plane(0).basePtr, outData->plane(0).rowStride,
                               outRowStride, // vec has no padding
                               height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * outRowStride);

        // generate gold result
        setGoldBuffer(goldVec, foregroundVec[i], backgroundVec[i], fgMaskVec[i], width, height, foregroundRowStride,
                      fgMaskRowStride, outRowStride, inChannels, outChannels);

        EXPECT_EQ(goldVec, testVec);
    }
}

namespace {

template<typename T>
std::vector<T> InterleavedToPlanar(const std::vector<T> &hwc, int width, int height, int channels)
{
    std::vector<T> chw(hwc.size());
    const int      pixels = width * height;
    for (int p = 0; p < pixels; ++p)
    {
        for (int c = 0; c < channels; ++c)
        {
            chw[c * pixels + p] = hwc[p * channels + c];
        }
    }
    return chw;
}

template<typename T>
void UploadTensorInterleaved(const nvcv::TensorDataStridedCuda &tensorData, int sample, const std::vector<T> &hwc,
                             int width, int height, int channels)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);
    ASSERT_TRUE(access);

    const size_t rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(access->sampleData(sample), access->rowStride(), hwc.data(), rowBytes, rowBytes,
                                        height, cudaMemcpyHostToDevice));
}

template<typename T>
void UploadTensorPlanar(const nvcv::TensorDataStridedCuda &tensorData, int sample, const std::vector<T> &chw, int width,
                        int height, int channels)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);
    ASSERT_TRUE(access);

    const size_t rowBytes   = static_cast<size_t>(width) * sizeof(T);
    auto        *sampleData = access->sampleData(sample);
    for (int c = 0; c < channels; ++c)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(access->chData(c, sampleData), access->rowStride(),
                                            chw.data() + static_cast<size_t>(c) * width * height, rowBytes, rowBytes,
                                            height, cudaMemcpyHostToDevice));
    }
}

template<typename T = uint8_t>
std::vector<T> DownloadTensorInterleaved(const nvcv::TensorDataStridedCuda &tensorData, int sample, int width,
                                         int height, int channels)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);
    if (!access)
    {
        ADD_FAILURE() << "Expected image-like tensor access";
        return {};
    }

    const size_t   rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    std::vector<T> hwc(static_cast<size_t>(height) * width * channels);
    EXPECT_EQ(cudaSuccess, cudaMemcpy2D(hwc.data(), rowBytes, access->sampleData(sample), access->rowStride(), rowBytes,
                                        height, cudaMemcpyDeviceToHost));
    return hwc;
}

template<typename T = uint8_t>
std::vector<T> DownloadTensorPlanar(const nvcv::TensorDataStridedCuda &tensorData, int sample, int width, int height,
                                    int channels)
{
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(tensorData);
    if (!access)
    {
        ADD_FAILURE() << "Expected image-like tensor access";
        return {};
    }

    const size_t   rowBytes = static_cast<size_t>(width) * sizeof(T);
    std::vector<T> chw(static_cast<size_t>(height) * width * channels);
    auto          *sampleData = access->sampleData(sample);
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_EQ(cudaSuccess, cudaMemcpy2D(chw.data() + static_cast<size_t>(c) * width * height, rowBytes,
                                            access->chData(c, sampleData), access->rowStride(), rowBytes, height,
                                            cudaMemcpyDeviceToHost));
    }
    return chw;
}

template<typename T>
void UploadImageInterleaved(const nvcv::Image &image, const std::vector<T> &hwc, int width, int height, int channels)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(data, nvcv::NullOpt);

    const size_t rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(0).basePtr, data->plane(0).rowStride, hwc.data(), rowBytes,
                                        rowBytes, height, cudaMemcpyHostToDevice));
}

template<typename T>
void UploadImagePlanar(const nvcv::Image &image, const std::vector<T> &chw, int width, int height, int channels)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(data, nvcv::NullOpt);
    ASSERT_EQ(data->numPlanes(), channels);

    const size_t rowBytes = static_cast<size_t>(width) * sizeof(T);
    for (int c = 0; c < channels; ++c)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(c).basePtr, data->plane(c).rowStride,
                                            chw.data() + static_cast<size_t>(c) * width * height, rowBytes, rowBytes,
                                            height, cudaMemcpyHostToDevice));
    }
}

template<typename T = uint8_t>
std::vector<T> DownloadImageInterleaved(const nvcv::Image &image, int width, int height, int channels)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    if (data == nvcv::NullOpt)
    {
        ADD_FAILURE() << "Expected CUDA-accessible image data";
        return {};
    }

    const size_t   rowBytes = static_cast<size_t>(width) * channels * sizeof(T);
    std::vector<T> hwc(static_cast<size_t>(height) * width * channels);
    EXPECT_EQ(cudaSuccess, cudaMemcpy2D(hwc.data(), rowBytes, data->plane(0).basePtr, data->plane(0).rowStride,
                                        rowBytes, height, cudaMemcpyDeviceToHost));
    return hwc;
}

template<typename T = uint8_t>
std::vector<T> DownloadImagePlanar(const nvcv::Image &image, int width, int height, int channels)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    if (data == nvcv::NullOpt)
    {
        ADD_FAILURE() << "Expected CUDA-accessible image data";
        return {};
    }
    if (data->numPlanes() != channels)
    {
        ADD_FAILURE() << "Expected " << channels << " planes, got " << data->numPlanes();
        return {};
    }

    const size_t   rowBytes = static_cast<size_t>(width) * sizeof(T);
    std::vector<T> chw(static_cast<size_t>(height) * width * channels);
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_EQ(cudaSuccess,
                  cudaMemcpy2D(chw.data() + static_cast<size_t>(c) * width * height, rowBytes, data->plane(c).basePtr,
                               data->plane(c).rowStride, rowBytes, height, cudaMemcpyDeviceToHost));
    }
    return chw;
}

std::vector<uint8_t> MakePatternBytes(size_t count, uint32_t seed)
{
    std::vector<uint8_t> values(count);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<uint8_t>((i * 37 + seed * 17) & 0xFF);
    }
    return values;
}

void RunCompositePlanarTensorParity(nvcv::ImageFormat interleavedOutFormat, nvcv::ImageFormat planarOutFormat,
                                    int width, int height, int numImages, int outChannels)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor fgI(numImages, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor bgI(numImages, {width, height}, nvcv::FMT_RGB8);
    nvcv::Tensor maskI(numImages, {width, height}, nvcv::FMT_U8);
    nvcv::Tensor dstI(numImages, {width, height}, interleavedOutFormat);

    nvcv::Tensor fgP(numImages, {width, height}, nvcv::FMT_RGB8p);
    nvcv::Tensor bgP(numImages, {width, height}, nvcv::FMT_RGB8p);
    nvcv::Tensor maskP(numImages, {width, height}, nvcv::FMT_U8);
    nvcv::Tensor dstP(numImages, {width, height}, planarOutFormat);

    auto fgIData   = fgI.exportData<nvcv::TensorDataStridedCuda>();
    auto bgIData   = bgI.exportData<nvcv::TensorDataStridedCuda>();
    auto maskIData = maskI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstIData  = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto fgPData   = fgP.exportData<nvcv::TensorDataStridedCuda>();
    auto bgPData   = bgP.exportData<nvcv::TensorDataStridedCuda>();
    auto maskPData = maskP.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData  = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(fgIData && bgIData && maskIData && dstIData && fgPData && bgPData && maskPData && dstPData);

    for (int sample = 0; sample < numImages; ++sample)
    {
        const auto           sampleSeed = static_cast<uint32_t>(sample + 1);
        std::vector<uint8_t> fg         = MakePatternBytes(static_cast<size_t>(width) * height * 3, sampleSeed);
        std::vector<uint8_t> bg         = MakePatternBytes(static_cast<size_t>(width) * height * 3, sampleSeed + 11);
        std::vector<uint8_t> mask       = MakePatternBytes(static_cast<size_t>(width) * height, sampleSeed + 23);

        UploadTensorInterleaved(*fgIData, sample, fg, width, height, 3);
        UploadTensorInterleaved(*bgIData, sample, bg, width, height, 3);
        UploadTensorInterleaved(*maskIData, sample, mask, width, height, 1);
        UploadTensorPlanar(*fgPData, sample, InterleavedToPlanar(fg, width, height, 3), width, height, 3);
        UploadTensorPlanar(*bgPData, sample, InterleavedToPlanar(bg, width, height, 3), width, height, 3);
        UploadTensorInterleaved(*maskPData, sample, mask, width, height, 1);
    }

    cvcuda::Composite compositeOp;
    EXPECT_NO_THROW(compositeOp(stream, fgI, bgI, maskI, dstI));
    EXPECT_NO_THROW(compositeOp(stream, fgP, bgP, maskP, dstP));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < numImages; ++sample)
    {
        SCOPED_TRACE(sample);
        const auto           sampleSeed = static_cast<uint32_t>(sample + 1);
        std::vector<uint8_t> fg         = MakePatternBytes(static_cast<size_t>(width) * height * 3, sampleSeed);
        std::vector<uint8_t> bg         = MakePatternBytes(static_cast<size_t>(width) * height * 3, sampleSeed + 11);
        std::vector<uint8_t> mask       = MakePatternBytes(static_cast<size_t>(width) * height, sampleSeed + 23);
        std::vector<uint8_t> gold(static_cast<size_t>(width) * height * outChannels);
        setGoldBuffer(gold, fg, bg, mask, width, height, width * 3, width, width * outChannels, 3, outChannels);

        std::vector<uint8_t> interleaved
            = DownloadTensorInterleaved<uint8_t>(*dstIData, sample, width, height, outChannels);
        std::vector<uint8_t> planar = DownloadTensorPlanar<uint8_t>(*dstPData, sample, width, height, outChannels);

        EXPECT_EQ(gold, interleaved);
        EXPECT_EQ(InterleavedToPlanar(gold, width, height, outChannels), planar);
        EXPECT_EQ(InterleavedToPlanar(interleaved, width, height, outChannels), planar);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

void RunCompositePlanarVarShapeParity(nvcv::ImageFormat interleavedOutFormat, nvcv::ImageFormat planarOutFormat,
                                      int width, int height, int numImages, int outChannels)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> fgIImgs;
    std::vector<nvcv::Image> bgIImgs;
    std::vector<nvcv::Image> maskIImgs;
    std::vector<nvcv::Image> dstIImgs;
    std::vector<nvcv::Image> fgPImgs;
    std::vector<nvcv::Image> bgPImgs;
    std::vector<nvcv::Image> maskPImgs;
    std::vector<nvcv::Image> dstPImgs;

    for (int sample = 0; sample < numImages; ++sample)
    {
        const nvcv::Size2D size{width + sample, height + sample};

        fgIImgs.emplace_back(size, nvcv::FMT_RGB8);
        bgIImgs.emplace_back(size, nvcv::FMT_RGB8);
        maskIImgs.emplace_back(size, nvcv::FMT_U8);
        dstIImgs.emplace_back(size, interleavedOutFormat);
        fgPImgs.emplace_back(size, nvcv::FMT_RGB8p);
        bgPImgs.emplace_back(size, nvcv::FMT_RGB8p);
        maskPImgs.emplace_back(size, nvcv::FMT_U8);
        dstPImgs.emplace_back(size, planarOutFormat);

        const auto           sampleSeed = static_cast<uint32_t>(sample + 1);
        std::vector<uint8_t> fg         = MakePatternBytes(static_cast<size_t>(size.w) * size.h * 3, sampleSeed);
        std::vector<uint8_t> bg         = MakePatternBytes(static_cast<size_t>(size.w) * size.h * 3, sampleSeed + 11);
        std::vector<uint8_t> mask       = MakePatternBytes(static_cast<size_t>(size.w) * size.h, sampleSeed + 23);

        UploadImageInterleaved(fgIImgs[sample], fg, size.w, size.h, 3);
        UploadImageInterleaved(bgIImgs[sample], bg, size.w, size.h, 3);
        UploadImageInterleaved(maskIImgs[sample], mask, size.w, size.h, 1);
        UploadImagePlanar(fgPImgs[sample], InterleavedToPlanar(fg, size.w, size.h, 3), size.w, size.h, 3);
        UploadImagePlanar(bgPImgs[sample], InterleavedToPlanar(bg, size.w, size.h, 3), size.w, size.h, 3);
        UploadImageInterleaved(maskPImgs[sample], mask, size.w, size.h, 1);
    }

    auto makeBatch = [numImages](std::vector<nvcv::Image> &images)
    {
        nvcv::ImageBatchVarShape batch(numImages);
        batch.pushBack(images.begin(), images.end());
        return batch;
    };

    nvcv::ImageBatchVarShape fgI   = makeBatch(fgIImgs);
    nvcv::ImageBatchVarShape bgI   = makeBatch(bgIImgs);
    nvcv::ImageBatchVarShape maskI = makeBatch(maskIImgs);
    nvcv::ImageBatchVarShape dstI  = makeBatch(dstIImgs);
    nvcv::ImageBatchVarShape fgP   = makeBatch(fgPImgs);
    nvcv::ImageBatchVarShape bgP   = makeBatch(bgPImgs);
    nvcv::ImageBatchVarShape maskP = makeBatch(maskPImgs);
    nvcv::ImageBatchVarShape dstP  = makeBatch(dstPImgs);

    cvcuda::Composite compositeOp;
    EXPECT_NO_THROW(compositeOp(stream, fgI, bgI, maskI, dstI));
    EXPECT_NO_THROW(compositeOp(stream, fgP, bgP, maskP, dstP));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < numImages; ++sample)
    {
        SCOPED_TRACE(sample);
        const nvcv::Size2D   size       = fgIImgs[sample].size();
        const auto           sampleSeed = static_cast<uint32_t>(sample + 1);
        std::vector<uint8_t> fg         = MakePatternBytes(static_cast<size_t>(size.w) * size.h * 3, sampleSeed);
        std::vector<uint8_t> bg         = MakePatternBytes(static_cast<size_t>(size.w) * size.h * 3, sampleSeed + 11);
        std::vector<uint8_t> mask       = MakePatternBytes(static_cast<size_t>(size.w) * size.h, sampleSeed + 23);
        std::vector<uint8_t> gold(static_cast<size_t>(size.w) * size.h * outChannels);
        setGoldBuffer(gold, fg, bg, mask, size.w, size.h, size.w * 3, size.w, size.w * outChannels, 3, outChannels);

        std::vector<uint8_t> interleaved
            = DownloadImageInterleaved<uint8_t>(dstIImgs[sample], size.w, size.h, outChannels);
        std::vector<uint8_t> planar = DownloadImagePlanar<uint8_t>(dstPImgs[sample], size.w, size.h, outChannels);

        EXPECT_EQ(gold, interleaved);
        EXPECT_EQ(InterleavedToPlanar(gold, size.w, size.h, outChannels), planar);
        EXPECT_EQ(InterleavedToPlanar(interleaved, size.w, size.h, outChannels), planar);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

TEST(OpCompositePlanar, tensor_rgb_output_matches_interleaved)
{
    RunCompositePlanarTensorParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 31, 19, 2, 3);
}

TEST(OpCompositePlanar, tensor_rgba_output_matches_interleaved)
{
    RunCompositePlanarTensorParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, 29, 17, 2, 4);
}

TEST(OpCompositePlanar, varshape_rgb_output_matches_interleaved)
{
    RunCompositePlanarVarShapeParity(nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 31, 19, 2, 3);
}

TEST(OpCompositePlanar, varshape_rgba_output_matches_interleaved)
{
    RunCompositePlanarVarShapeParity(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, 29, 17, 2, 4);
}

// clang-format off
NVCV_TEST_SUITE_P(OpComposite_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, bool>
    {
        // image format
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, nvcv::FMT_U8, nvcv::FMT_RGB8, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8p, false},
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8, false},
        // data type
        {nvcv::FMT_RGBf16, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGBf16, nvcv::FMT_U8, nvcv::FMT_RGB8, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_F16, nvcv::FMT_RGB8, false},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGBf16, false},
        // different format
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_U8, nvcv::FMT_RGB8, true},
    });

// clang-format on

TEST(OpComposite_Negative, createWithNullHandle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaCompositeCreate(nullptr));
}

struct CompositeMismatchTestCase
{
    int          foregroundImages;
    int          backgroundImages;
    int          maskImages;
    int          outputImages;
    nvcv::Size2D foregroundSize;
    nvcv::Size2D backgroundSize;
    nvcv::Size2D maskSize;
    nvcv::Size2D outputSize;
    const char  *name;
};

static CompositeMismatchTestCase MakeCompositeMismatchTestCase(const char *name)
{
    const nvcv::Size2D size{24, 24};
    return {2, 2, 2, 2, size, size, size, size, name};
}

static std::array<CompositeMismatchTestCase, 10> MakeCompositeMismatchTestCases(bool useSmallerOutputBatch)
{
    const std::array<const char *, 10> names{
        {
         "output_batch", "output_width",
         "output_height", "foreground_batch",
         "foreground_width", "foreground_height",
         "background_batch", "fgMask_batch",
         "background_size", "fgMask_size",
         }
    };

    std::array<CompositeMismatchTestCase, 10> testCases{};
    std::ranges::transform(names, testCases.begin(), MakeCompositeMismatchTestCase);

    if (useSmallerOutputBatch)
    {
        testCases[0].foregroundImages = 3;
        testCases[0].backgroundImages = 3;
        testCases[0].maskImages       = 3;
    }
    else
    {
        testCases[0].outputImages = 3;
    }

    testCases[1].outputSize = nvcv::Size2D{16, 24};
    testCases[2].outputSize = nvcv::Size2D{24, 16};
    testCases[3].foregroundImages++;
    testCases[4].foregroundSize = nvcv::Size2D{16, 24};
    testCases[5].foregroundSize = nvcv::Size2D{24, 16};
    testCases[6].backgroundImages++;
    testCases[7].maskImages++;
    testCases[8].backgroundSize = nvcv::Size2D{16, 24};
    testCases[9].maskSize       = nvcv::Size2D{24, 16};

    return testCases;
}

TEST(OpComposite_Negative, tensor_rejects_mismatched_shape)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const std::array<CompositeMismatchTestCase, 10> testCases = MakeCompositeMismatchTestCases(false);

    cvcuda::Composite compositeOp;

    for (const CompositeMismatchTestCase &testCase : testCases)
    {
        SCOPED_TRACE(testCase.name);

        nvcv::Tensor foregroundImg(testCase.foregroundImages, testCase.foregroundSize, nvcv::FMT_RGB8);
        nvcv::Tensor backgroundImg(testCase.backgroundImages, testCase.backgroundSize, nvcv::FMT_RGB8);
        nvcv::Tensor fgMaskImg(testCase.maskImages, testCase.maskSize, nvcv::FMT_U8);
        nvcv::Tensor outImg(testCase.outputImages, testCase.outputSize, nvcv::FMT_RGB8);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { compositeOp(stream, foregroundImg, backgroundImg, fgMaskImg, outImg); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static std::vector<nvcv::Image> CreateCompositeImages(int numImages, nvcv::Size2D size, nvcv::ImageFormat format)
{
    std::vector<nvcv::Image> images;
    images.reserve(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        images.emplace_back(size, format);
    }

    return images;
}

TEST(OpComposite_Negative, varshape_rejects_mismatched_shape)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const std::array<CompositeMismatchTestCase, 10> testCases = MakeCompositeMismatchTestCases(true);

    cvcuda::Composite compositeOp;

    for (const CompositeMismatchTestCase &testCase : testCases)
    {
        SCOPED_TRACE(testCase.name);

        std::vector<nvcv::Image> imgForeground
            = CreateCompositeImages(testCase.foregroundImages, testCase.foregroundSize, nvcv::FMT_RGB8);
        std::vector<nvcv::Image> imgBackground
            = CreateCompositeImages(testCase.backgroundImages, testCase.backgroundSize, nvcv::FMT_RGB8);
        std::vector<nvcv::Image> imgFgMask
            = CreateCompositeImages(testCase.maskImages, testCase.maskSize, nvcv::FMT_U8);
        std::vector<nvcv::Image> imgOutput
            = CreateCompositeImages(testCase.outputImages, testCase.outputSize, nvcv::FMT_RGB8);

        int batchCapacity = std::max(
            {testCase.foregroundImages, testCase.backgroundImages, testCase.maskImages, testCase.outputImages});

        nvcv::ImageBatchVarShape batchForeground(batchCapacity);
        nvcv::ImageBatchVarShape batchBackground(batchCapacity);
        nvcv::ImageBatchVarShape batchFgMask(batchCapacity);
        nvcv::ImageBatchVarShape batchOutput(batchCapacity);

        batchForeground.pushBack(imgForeground.begin(), imgForeground.end());
        batchBackground.pushBack(imgBackground.begin(), imgBackground.end());
        batchFgMask.pushBack(imgFgMask.begin(), imgFgMask.end());
        batchOutput.pushBack(imgOutput.begin(), imgOutput.end());

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&] { compositeOp(stream, batchForeground, batchBackground, batchFgMask, batchOutput); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpComposite_Negative, invalid_parameters)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int inWidth        = 24;
    int inHeight       = 24;
    int numberOfImages = 3;
    int outWidth       = inWidth;
    int outHeight      = inHeight;

    nvcv::ImageFormat foregroundFormat = GetParamValue<0>();
    nvcv::ImageFormat backgroundFormat = GetParamValue<1>();
    nvcv::ImageFormat fgMaskFormat     = GetParamValue<2>();
    nvcv::ImageFormat outFormat        = GetParamValue<3>();
    if (bool isDiffFormatTest = GetParamValue<4>(); isDiffFormatTest)
    {
        GTEST_SKIP() << "Skipping diff format test for image input";
    }

    nvcv::Tensor foregroundImg(numberOfImages, {inWidth, inHeight}, foregroundFormat);
    nvcv::Tensor backgroundImg(numberOfImages, {inWidth, inHeight}, backgroundFormat);
    nvcv::Tensor fgMaskImg(numberOfImages, {inWidth, inHeight}, fgMaskFormat);
    nvcv::Tensor outImg(numberOfImages, {outWidth, outHeight}, outFormat);

    // run operator
    cvcuda::Composite compositeOp;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&compositeOp, &stream, &foregroundImg, &backgroundImg, &fgMaskImg, &outImg]
                                { compositeOp(stream, foregroundImg, backgroundImg, fgMaskImg, outImg); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpComposite_Negative, varshape_invalid_parameters)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int inWidth        = 24;
    int inHeight       = 24;
    int numberOfImages = 3;

    nvcv::ImageFormat foregroundFormat = GetParamValue<0>();
    nvcv::ImageFormat backgroundFormat = GetParamValue<1>();
    nvcv::ImageFormat fgMaskFormat     = GetParamValue<2>();
    nvcv::ImageFormat outFormat        = GetParamValue<3>();
    bool              isDiffFormatTest = GetParamValue<4>();

    std::default_random_engine rng;

    // Create input varshape

    std::uniform_int_distribution udistWidth(ScaledSize(inWidth, 0.8), ScaledSize(inWidth, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(inHeight, 0.8), ScaledSize(inHeight, 1.1));

    std::vector<nvcv::Image> imgForeground;
    std::vector<nvcv::Image> imgBackground;
    std::vector<nvcv::Image> imgFgMask;
    std::vector<nvcv::Image> imgOut;

    for (int i = 0; i < numberOfImages - 1; i++)
    {
        nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};

        imgForeground.emplace_back(size, foregroundFormat);
        imgBackground.emplace_back(size, backgroundFormat);
        imgFgMask.emplace_back(size, fgMaskFormat);
        imgOut.emplace_back(size, outFormat);
    }

    if (isDiffFormatTest)
    {
        nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};
        imgForeground.emplace_back(size, nvcv::FMT_U8);
        imgBackground.emplace_back(size, nvcv::FMT_U8);
        imgFgMask.emplace_back(size, nvcv::FMT_RGB8);
        imgOut.emplace_back(size, nvcv::FMT_U8);
    }
    else
    {
        nvcv::Size2D size{udistWidth(rng), udistHeight(rng)};
        imgForeground.emplace_back(size, foregroundFormat);
        imgBackground.emplace_back(size, backgroundFormat);
        imgFgMask.emplace_back(size, fgMaskFormat);
        imgOut.emplace_back(size, outFormat);
    }

    nvcv::ImageBatchVarShape batchForeground(numberOfImages);
    nvcv::ImageBatchVarShape batchBackground(numberOfImages);
    nvcv::ImageBatchVarShape batchFgMask(numberOfImages);
    nvcv::ImageBatchVarShape batchOutput(numberOfImages);

    batchForeground.pushBack(imgForeground.begin(), imgForeground.end());
    batchBackground.pushBack(imgBackground.begin(), imgBackground.end());
    batchFgMask.pushBack(imgFgMask.begin(), imgFgMask.end());
    batchOutput.pushBack(imgOut.begin(), imgOut.end());

    // Run operator
    cvcuda::Composite compositeOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&compositeOp, &stream, &batchForeground, &batchBackground, &batchFgMask, &batchOutput]
                                { compositeOp(stream, batchForeground, batchBackground, batchFgMask, batchOutput); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
