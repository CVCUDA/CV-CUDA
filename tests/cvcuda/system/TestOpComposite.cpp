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
#include <cvcuda/OpComposite.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

//#define DBG_COMPOSITE 1

template<typename T>
static void print_img(std::vector<T> vec, int rowStride, int height, std::string message)
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
            int      fg_offset     = r * inVecRowStride + c * inChannels;
            int      fgMask_offset = r * fgMaskVecRowStride + c;
            int      dst_offset    = r * outVecRowStride + c * outChannels;
            uint8_t *ptrGold       = gold.data() + dst_offset;
            uint8_t *ptrFg         = fg.data() + fg_offset;
            uint8_t *ptrBg         = bg.data() + fg_offset;
            uint8_t *ptrMat        = fgMask.data() + fgMask_offset;
            uint8_t  a             = *ptrMat;
            for (int k = 0; k < inChannels; k++)
            {
                int c0     = ptrBg[k];
                int c1     = ptrFg[k];
                ptrGold[k] = (uint8_t)(((int)c1 - (int)c0) * (int)a * (1.0f / 255.0f) + c0 + 0.5f);
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

    nvcv::ImageFormat inFormat, outFormat;

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

    const auto *foregroundData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(foregroundImg.exportData());
    const auto *backgroundData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(backgroundImg.exportData());
    const auto *fgMaskData     = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(fgMaskImg.exportData());

    ASSERT_NE(nullptr, foregroundData);
    ASSERT_NE(nullptr, backgroundData);
    ASSERT_NE(nullptr, fgMaskData);

    auto foregroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*foregroundData);
    ASSERT_TRUE(foregroundAccess);

    auto backgroundAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*backgroundData);
    ASSERT_TRUE(foregroundAccess);

    auto fgMaskAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*fgMaskData);
    ASSERT_TRUE(fgMaskAccess);

    int foregroundBufSize = foregroundAccess->sampleStride()
                          * foregroundAccess->numSamples(); //img pitch bytes can be more than the image 64, 128, etc
    int fgMaskBufSize = fgMaskAccess->sampleStride() * fgMaskAccess->numSamples();

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

        std::generate(foregroundVec[i].begin(), foregroundVec[i].end(), [&]() { return udist(rng); });
        std::generate(backgroundVec[i].begin(), backgroundVec[i].end(), [&]() { return udist(rng); });
        std::generate(fgMaskVec[i].begin(), fgMaskVec[i].end(), [&]() { return udist(rng); });

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
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(outImg.exportData());
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
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

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

    nvcv::ImageFormat inFormat, outFormat;
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

    std::uniform_int_distribution<int> udistWidth(inWidth * 0.8, inWidth * 1.1);
    std::uniform_int_distribution<int> udistHeight(inHeight * 0.8, inHeight * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgForeground;
    std::vector<std::unique_ptr<nvcv::Image>> imgBackground;
    std::vector<std::unique_ptr<nvcv::Image>> imgFgMask;
    std::vector<std::unique_ptr<nvcv::Image>> imgOut;

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

        imgForeground.emplace_back(std::make_unique<nvcv::Image>(size, inFormat));
        imgBackground.emplace_back(std::make_unique<nvcv::Image>(size, inFormat));
        imgFgMask.emplace_back(std::make_unique<nvcv::Image>(size, maskFormat));
        imgOut.emplace_back(std::make_unique<nvcv::Image>(size, outFormat));

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
        generate(foregroundVec[i].begin(), foregroundVec[i].end(), [&]() { return udist(rng); });
        generate(backgroundVec[i].begin(), backgroundVec[i].end(), [&]() { return udist(rng); });
        generate(fgMaskVec[i].begin(), fgMaskVec[i].end(), [&]() { return udist(rng); });

        auto *imgDataForeground = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgForeground[i]->exportData());
        assert(imgDataForeground != nullptr);

        auto *imgDataBackground = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgBackground[i]->exportData());
        assert(imgDataBackground != nullptr);

        auto *imgDataFgMask = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgFgMask[i]->exportData());
        assert(imgDataFgMask != nullptr);

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

        const auto *outData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgOut[i]->exportData());
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
