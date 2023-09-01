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
#include <cvcuda/OpHistogram.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

static void computeHistogram(std::vector<uint8_t> imageVec, std::vector<uint32_t> &goldHistogram)
{
    // Assuming grayscale image, the histogram will be of size 256
    std::vector<uint32_t> histogram(256, 0);

    // Compute the histogram
    for (auto pixel : imageVec)
    {
        histogram[pixel]++;
    }

    // Append the computed histogram to the goldHistogram
    goldHistogram.insert(goldHistogram.end(), histogram.begin(), histogram.end());
};

static void computeHistogramWithMask(std::vector<uint8_t> imageVec, std::vector<uint8_t> maskVec,
                                     std::vector<uint32_t> &goldHistogram)
{
    // Assuming grayscale image, the histogram will be of size 256
    std::vector<uint32_t> histogram(256, 0);

    // Compute the histogram
    for (size_t i = 0; i < imageVec.size(); ++i)
    {
        if (maskVec[i])
            histogram[imageVec[i]]++;
    }

    // Append the computed histogram to the goldHistogram
    goldHistogram.insert(goldHistogram.end(), histogram.begin(), histogram.end());
};

// clang-format off
NVCV_TEST_SUITE_P(OpHistogram, test::ValueList<int, int, NVCVImageFormat, int>
{
    //inWidth, inHeight,                format,   numberInBatch
    {       2,        2,  NVCV_IMAGE_FORMAT_U8,              1},
    {      10,       10,  NVCV_IMAGE_FORMAT_U8,              1},
    {      11,       13,  NVCV_IMAGE_FORMAT_U8,              2},
    {     320,      240,  NVCV_IMAGE_FORMAT_U8,              3},
    {     640,      480,  NVCV_IMAGE_FORMAT_U8,              2},
    {     800,      600,  NVCV_IMAGE_FORMAT_U8,              1},
    {     1024,     768,  NVCV_IMAGE_FORMAT_U8,              1},
    {     1280,     720,  NVCV_IMAGE_FORMAT_U8,              1},
    {     1920,    1080,  NVCV_IMAGE_FORMAT_U8,              1},
    {     2048,    1536,  NVCV_IMAGE_FORMAT_U8,              1},
    {     2592,    1944,  NVCV_IMAGE_FORMAT_U8,              1},
    {     3840,    2160,  NVCV_IMAGE_FORMAT_U8,              1},
    {     4096,    3072,  NVCV_IMAGE_FORMAT_U8,              1},
});

// clang-format on
TEST_P(OpHistogram, Histogram)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    nvcv::ImageFormat format{GetParamValue<2>()};
    int               batches = GetParamValue<3>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, batches, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_S32));

    std::vector<uint32_t>         goldHistogram;
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    for (int i = 0; i < batches; ++i)
    {
        // generate random input image
        std::vector<uint8_t> imageVec(width * height);
        std::generate(imageVec.begin(), imageVec.end(), [&]() { return rand(randEng); });

        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
        // Compute histogram and add to vector
        computeHistogram(imageVec, goldHistogram);
    }

    // run operator
    cvcuda::Histogram op;
    EXPECT_NO_THROW(op(stream, inTensor, nvcv::NullOpt, histogram));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint32_t> opHistogram;
    // get 0th sample since histogram is just a 2d tensor
    EXPECT_NO_THROW(util::GetImageVectorFromTensor(histogram.exportData(), 0, opHistogram));

    // Compare the computed histogram with the output histogram
    ASSERT_EQ(opHistogram, goldHistogram);
}

TEST_P(OpHistogram, Histogram_mask)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    nvcv::ImageFormat format{GetParamValue<2>()};
    int               batches = GetParamValue<3>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor inMask    = nvcv::util::CreateTensor(batches, width, height, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_U8));
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, batches, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_S32));

    std::vector<uint32_t>         goldHistogram;
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);
    std::uniform_int_distribution randMask(0u, 1u); // any value other than 0 is considered as 1 but want some 0s too

    for (int i = 0; i < batches; ++i)
    {
        // generate random input image
        std::vector<uint8_t> imageVec(width * height);
        std::generate(imageVec.begin(), imageVec.end(), [&]() { return rand(randEng); });
        //generate random mask
        std::vector<uint8_t> maskVec(width * height);
        std::generate(maskVec.begin(), maskVec.end(), [&]() { return randMask(randEng); });

        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
        // copy mask input to tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inMask.exportData(), maskVec, i));

        // Compute histogram and add to vector
        computeHistogramWithMask(imageVec, maskVec, goldHistogram);
    }

    // run operator
    cvcuda::Histogram op;
    EXPECT_NO_THROW(op(stream, inTensor, inMask, histogram));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint32_t> opHistogram;
    // get 0th sample since histogram is just a 2d tensor
    EXPECT_NO_THROW(util::GetImageVectorFromTensor(histogram.exportData(), 0, opHistogram));

    // Compare the computed histogram with the output histogram
    ASSERT_EQ(opHistogram, goldHistogram);
}
