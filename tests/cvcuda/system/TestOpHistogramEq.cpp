/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpHistogramEq.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

static void histogramEqualization(const std::vector<uint8_t> &inputImage, int width, int height, int numChannels,
                                  std::vector<uint8_t> &outputImage)
{
    // Initialize the output image
    outputImage = inputImage;

    for (int channel = 0; channel < numChannels; ++channel)
    {
        // Create histogram for the current channel
        std::array<int, 256> histogram = {};

        for (size_t i = channel; i < inputImage.size(); i += numChannels)
        {
            ++histogram[inputImage[i]];
        }

        // Compute the cumulative histogram
        std::array<int, 256> cumulativeHistogram = histogram;
        for (int i = 1; i < 256; ++i)
        {
            cumulativeHistogram[i] += cumulativeHistogram[i - 1];
        }
        auto smallest
            = *std::find_if(cumulativeHistogram.begin(), cumulativeHistogram.end(), [](int i) { return i != 0; });

        // Normalize the cumulative histogram to get the mapping values
        std::array<uint8_t, 256> equalizeMap = {};
        for (int i = 0; i < 256; ++i)
        {
            int   tmpT     = (cumulativeHistogram[i] - smallest);
            int   tmpB     = ((width * height) - smallest);
            float ratio    = (float)(tmpT * 255) / (float)tmpB;
            equalizeMap[i] = nvcv::cuda::SaturateCast<int>(ratio);
        }

        // Map the input image pixel values into the equalized values
        for (size_t i = channel; i < inputImage.size(); i += numChannels)
        {
            outputImage[i] = equalizeMap[inputImage[i]];
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogramEq, test::ValueList<int, int, NVCVImageFormat, int>
{
    //inWidth, inHeight,                  format,   numberInBatch
    {       20,     428,  NVCV_IMAGE_FORMAT_BGR8,              1},
    {       101,     99,  NVCV_IMAGE_FORMAT_RGB8,              3},
    {       256,    256,  NVCV_IMAGE_FORMAT_RGB8,              1},
    {        12,    512,  NVCV_IMAGE_FORMAT_RGB8,              5},

    {       5,       22,  NVCV_IMAGE_FORMAT_BGRA8,             1},
    {     120,      103,  NVCV_IMAGE_FORMAT_BGRA8,             2},
    {      56,        2,  NVCV_IMAGE_FORMAT_BGRA8,             3},
    {      12,       52,  NVCV_IMAGE_FORMAT_BGRA8,             1},

    {       2,        4,     NVCV_IMAGE_FORMAT_U8,             1},
    {     100,      101,     NVCV_IMAGE_FORMAT_U8,             4},
    {      56,       35,     NVCV_IMAGE_FORMAT_U8,             2},
    {      12,       51,     NVCV_IMAGE_FORMAT_U8,             3},
});

// clang-format on
TEST_P(OpHistogramEq, HistogramEq_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    nvcv::ImageFormat format{GetParamValue<2>()};
    int               batches = GetParamValue<3>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    std::vector<uint8_t>          goldImage;
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);
    int                           colorChannels  = inTensor.shape()[inTensor.shape().rank() - 1];
    size_t                        imageSizeBytes = width * height * colorChannels * sizeof(uint8_t);

    for (int i = 0; i < batches; ++i)
    {
        // generate random input image
        std::vector<uint8_t> imageVec(imageSizeBytes);
        std::generate(imageVec.begin(), imageVec.end(), [&]() { return rand(randEng); });
        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
    }

    // run operator
    cvcuda::HistogramEq op(batches);
    EXPECT_NO_THROW(op(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < batches; ++i)
    {
        // get output image
        std::vector<uint8_t> outImage;
        std::vector<uint8_t> input;
        std::vector<uint8_t> goldImage;

        EXPECT_NO_THROW(util::GetImageVectorFromTensor(inTensor.exportData(), i, input));
        histogramEqualization(input, width, height, colorChannels, goldImage);
        EXPECT_NO_THROW(util::GetImageVectorFromTensor(outTensor.exportData(), i, outImage));
        ASSERT_EQ(outImage, goldImage);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogreamEqVarShape, test::ValueList<int, int, NVCVImageFormat, int>
{
    //inWidth, inHeight,                  format,   numberInBatch
    {       4,        4,  NVCV_IMAGE_FORMAT_RGB8,              4},
    {     101,       99,  NVCV_IMAGE_FORMAT_RGB8,              3},
    {     256,      256,  NVCV_IMAGE_FORMAT_RGB8,              1},
    {     12,       512,  NVCV_IMAGE_FORMAT_RGB8,              5},

    {      32,       21,  NVCV_IMAGE_FORMAT_BGRA8,             1},
    {     120,      103,  NVCV_IMAGE_FORMAT_BGRA8,             2},
    {      56,        2,  NVCV_IMAGE_FORMAT_BGRA8,             3},
    {      12,       52,  NVCV_IMAGE_FORMAT_BGRA8,             1},

    {       2,        4,     NVCV_IMAGE_FORMAT_U8,             1},
    {     100,      101,     NVCV_IMAGE_FORMAT_U8,             4},
    {      56,       35,     NVCV_IMAGE_FORMAT_U8,             2},
    {      12,      51 ,     NVCV_IMAGE_FORMAT_U8,             3},

});

// clang-format on

TEST_P(OpHistogreamEqVarShape, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int               width  = GetParamValue<0>();
    const int               height = GetParamValue<1>();
    const nvcv::ImageFormat format{GetParamValue<2>()};
    const int               batches = GetParamValue<3>();

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowStride(batches);

    //setup the input images
    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, format);

        int srcRowStride   = imgSrc[i].size().w * format.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i].size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    // Push the images on the varShape Batch.
    nvcv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Run operator set the max batches
    cvcuda::HistogramEq op(batches);
    EXPECT_NO_THROW(op(stream, batchSrc, batchDst));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int32_t numCh = srcData->format().numChannels();
        ASSERT_LT(numCh, 5);
        ASSERT_GT(numCh, 0);
        int3 shape{srcData->plane(0).width, srcData->plane(0).height, numCh};

        // Create test vector just want the image data.
        std::vector<uint8_t> opOutVec(shape.y * shape.x * numCh);
        std::vector<uint8_t> goldVec(shape.y * shape.x * numCh);
        std::vector<uint8_t> inputVec(shape.y * shape.x * numCh);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(opOutVec.data(), shape.x * numCh, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               shape.x * numCh, shape.y, cudaMemcpyDeviceToHost));

        // Copy input data to Host to generate goldVector
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(inputVec.data(), shape.x * numCh, srcData->plane(0).basePtr, srcData->plane(0).rowStride,
                               shape.x * numCh, shape.y, cudaMemcpyDeviceToHost));
        histogramEqualization(inputVec, shape.x, shape.y, numCh, goldVec);
        EXPECT_EQ(opOutVec, goldVec);
    }
}
