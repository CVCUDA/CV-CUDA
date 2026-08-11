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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpStack.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

// clang-format off
NVCV_TEST_SUITE_P(OpStack, test::ValueList<int, int, nvcv::ImageFormat, int, int>
{
    //inWidth, inHeight,                format,    numberOfTensors, maxNumberInBatch
    {     320,      240,         nvcv::FMT_U8,                   5,                2},
    {      40,       81,         nvcv::FMT_RGB8,                 1,                3},
    {     800,      600,         nvcv::FMT_BGR8,                 1,                4},
    {    1024,      768,         nvcv::FMT_RGBA8,                2,                1},
    {      12,      720,         nvcv::FMT_BGRA8,                3,                5},
    {     160,      121,         nvcv::FMT_BGR8p,                2,                2},
    {     920,       80,         nvcv::FMT_RGB8p,                1,                3},
    {      41,      536,         nvcv::FMT_RGBA8p,               1,                4},
    {     592,      944,         nvcv::FMT_BGRA8p,               2,                5},
    {       1,        2,         nvcv::FMT_U32,                  1,                1},
    {      48,       36,         nvcv::FMT_RGBf32,               1,                2},
    {     192,     1944,         nvcv::FMT_BGRf32,               1,                3},
    {    1920,     1080,         nvcv::FMT_RGBAf32,              4,                4},
    {    2048,     1536,         nvcv::FMT_BGRAf32,              1,                5},
    {    1024,      768,         nvcv::FMT_RGBA8p,               3,                1},
    {    1280,      720,         nvcv::FMT_RGBf32p,              1,                2},
    {     192,       80,         nvcv::FMT_BGRf32p,              1,                3},
    {    2048,      536,         nvcv::FMT_RGBAf32p,             1,                4},
    {     259,      194,         nvcv::FMT_BGRAf32p,             1,                5},
    {    1921,     1080,         nvcv::FMT_F64,                  1,                1},
    {    1920,     1080,         nvcv::FMT_F16,                  2,                2},
    {      48,       36,         nvcv::FMT_BGRAf32,              1,                3},
});

// clang-format on
TEST_P(OpStack, test_NCHW_tensors)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width            = GetParamValue<0>();
    int               height           = GetParamValue<1>();
    nvcv::ImageFormat format           = GetParamValue<2>();
    int               numberOfTensors  = GetParamValue<3>();
    int               maxNumberInBatch = GetParamValue<4>();

    int numChannels          = format.numChannels();
    int bytesPerPixel        = 0;
    int totalNumberOfTensors = 0;

    for (int32_t i = 0; i < numChannels; i++)
    {
        bytesPerPixel += format.bitsPerChannel()[i] / 8;
    }

    // generate the output tensor to contain all of the input tensors

    auto                                 reqs = nvcv::TensorBatch::CalcRequirements(numberOfTensors);
    nvcv::TensorBatch                    inTensorBatch(reqs);
    std::vector<std::vector<nvcv::Byte>> inputVecs;

    // generate random input images
    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);
    std::uniform_int_distribution distribution(1, maxNumberInBatch);
    int                           numberInBatch = distribution(randEng);

    for (int i = 0; i < numberOfTensors; ++i)
    {
        nvcv::Tensor inTensor(numberInBatch, {width, height}, format);
        totalNumberOfTensors += numberInBatch; // include individual tensors and tensors in N > 1 tensor(s)

        for (int j = 0; j < numberInBatch; j++) // NOSONAR
        {
            // generate random input image in bytes
            std::vector<nvcv::Byte> imageVec((width * height) * bytesPerPixel);
            std::ranges::generate(imageVec, [&rand, &randEng]() { return (nvcv::Byte)rand(randEng); });
            // copy random input to device tensor
            EXPECT_NO_THROW(util::SetImageTensorFromByteVector(inTensor.exportData(), imageVec, j));
            // add tensor to batch and input vector
            inputVecs.push_back(imageVec);
        }
        inTensorBatch.pushBack(inTensor);
    }

    nvcv::Tensor  outTensor(totalNumberOfTensors, {width, height}, format);
    // run operator
    cvcuda::Stack op;
    EXPECT_NO_THROW(op(stream, inTensorBatch, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // go through each sample of the output tensor and compare vals.
    for (int i = 0; i < totalNumberOfTensors; ++i) // NOSONAR
    {
        // generate random input image in bytes
        std::vector<nvcv::Byte> outSample;
        EXPECT_NO_THROW(util::GetImageByteVectorFromTensor(outTensor.exportData(), i, outSample));
        // Compare the computed histogram with the output histogram
        ASSERT_EQ(inputVecs[i], outSample);
    }
}

TEST_P(OpStack, test_CHW_tensors)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width           = GetParamValue<0>();
    int               height          = GetParamValue<1>();
    nvcv::ImageFormat format          = GetParamValue<2>();
    int               numberOfTensors = GetParamValue<3>();

    int numChannels   = format.numChannels();
    int bytesPerPixel = 0;

    for (int32_t i = 0; i < numChannels; i++)
    {
        bytesPerPixel += format.bitsPerChannel()[i] / 8;
    }

    // generate the output tensor to contain all of the input tensors

    auto              reqs = nvcv::TensorBatch::CalcRequirements(numberOfTensors);
    nvcv::TensorBatch inTensorBatch(reqs);

    // generate random input images
    std::default_random_engine           randEng(0);
    std::uniform_int_distribution        rand(0u, 255u);
    std::vector<std::vector<nvcv::Byte>> inputVecs;

    for (int i = 0; i < numberOfTensors; ++i) // NOSONAR
    {
        nvcv::Tensor inTensor = nvcv::util::CreateTensor(1, width, height, format); //this will create a CHW/HWC tensor
        // generate random input image in bytes
        std::vector<nvcv::Byte> imageVec((width * height) * bytesPerPixel);
        std::ranges::generate(imageVec, [&rand, &randEng]() { return (nvcv::Byte)rand(randEng); });
        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromByteVector(inTensor.exportData(), imageVec));
        // add tensor to batch and input vector
        inputVecs.push_back(imageVec);
        inTensorBatch.pushBack(inTensor);
    }

    nvcv::Tensor  outTensor(numberOfTensors, {width, height}, format);
    // run operator
    cvcuda::Stack op;
    EXPECT_NO_THROW(op(stream, inTensorBatch, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // go through each sample of the output tensor and compare vals.
    for (int i = 0; i < numberOfTensors; ++i) // NOSONAR
    {
        // generate random input image in bytes
        std::vector<nvcv::Byte> outSample;
        EXPECT_NO_THROW(util::GetImageByteVectorFromTensor(outTensor.exportData(), i, outSample));
        // Compare the computed histogram with the output histogram
        ASSERT_EQ(inputVecs[i], outSample);
    }
}

TEST(OpStack_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaStackCreate(nullptr));
}

TEST(OpStack_Negative, invalid_parameters)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width           = 24;
    int               height          = 24;
    nvcv::ImageFormat format          = nvcv::FMT_U8;
    int               numberOfTensors = 10;

    int numChannels   = format.numChannels();
    int bytesPerPixel = 0;

    for (int32_t i = 0; i < numChannels; i++)
    {
        bytesPerPixel += format.bitsPerChannel()[i] / 8;
    }

    // generate the output tensor to contain all of the input tensors
    auto              reqs = nvcv::TensorBatch::CalcRequirements(numberOfTensors);
    nvcv::TensorBatch inTensorBatch(reqs);

    for (int i = 0; i < numberOfTensors; ++i)
    {
        nvcv::Tensor inTensor = nvcv::util::CreateTensor(1, width, height, format);
        inTensorBatch.pushBack(inTensor);
    }

    cvcuda::Stack op;

    nvcv::Tensor invalidRankOutTensor(
        {
            {height, width},
            "HW"
    },
        nvcv::TYPE_U8);

    nvcv::Tensor smallOutTensor(numberOfTensors - 1, {width, height}, format);
    nvcv::Tensor misMatchOutTensor(numberOfTensors, {width - 1, height}, format);
    // run operator

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &inTensorBatch, &invalidRankOutTensor]
                                                             { op(stream, inTensorBatch, invalidRankOutTensor); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &inTensorBatch, &smallOutTensor]
                                                             { op(stream, inTensorBatch, smallOutTensor); }));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &inTensorBatch, &misMatchOutTensor]
                                                             { op(stream, inTensorBatch, misMatchOutTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpStack, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width          = GetParamValue<0>();
    int               height         = GetParamValue<1>();
    nvcv::ImageFormat format         = GetParamValue<2>();
    int               numberOfImages = GetParamValue<3>();

    int numChannels   = format.numChannels();
    int bytesPerPixel = 0;

    for (int32_t i = 0; i < numChannels; i++)
    {
        bytesPerPixel += format.bitsPerChannel()[i] / 8;
    }

    // Create input varshape image batch
    std::vector<nvcv::Image> imgSrc;
    for (int i = 0; i < numberOfImages; ++i) // NOSONAR
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, format);
    }

    nvcv::ImageBatchVarShape inBatch(numberOfImages);
    inBatch.pushBack(imgSrc.begin(), imgSrc.end());

    // generate random input images
    std::default_random_engine           randEng(0);
    std::uniform_int_distribution        rand(0u, 255u);
    std::vector<std::vector<nvcv::Byte>> inputVecs(numberOfImages);

    for (int i = 0; i < numberOfImages; ++i) // NOSONAR
    {
        // generate random input image in bytes
        inputVecs[i].resize((width * height) * bytesPerPixel);
        std::ranges::generate(inputVecs[i], [&rand, &randEng]() { return (nvcv::Byte)rand(randEng); });

        // copy random input to device image
        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nullptr);

        const NVCVImageBufferStrided &buf = imgData->cdata().buffer.strided;
        for (int p = 0; p < buf.numPlanes; ++p)
        {
            const NVCVImagePlaneStrided &plane = buf.planes[p];
            int offset   = (format.numPlanes() > 1) ? (p * width * height * (bytesPerPixel / numChannels)) : 0;
            int rowBytes = plane.width * (bytesPerPixel / buf.numPlanes);
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(plane.basePtr, plane.rowStride, inputVecs[i].data() + offset, rowBytes,
                                                rowBytes, plane.height, cudaMemcpyHostToDevice));
        }
    }

    nvcv::Tensor outTensor(numberOfImages, {width, height}, format);

    // run operator
    cvcuda::Stack op;
    EXPECT_NO_THROW(op(stream, inBatch, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // go through each sample of the output tensor and compare vals.
    for (int i = 0; i < numberOfImages; ++i) // NOSONAR
    {
        std::vector<nvcv::Byte> outSample;
        EXPECT_NO_THROW(util::GetImageByteVectorFromTensor(outTensor.exportData(), i, outSample));
        ASSERT_EQ(inputVecs[i], outSample);
    }
}

TEST(OpStack_Negative, varshape_invalid_parameters)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width          = 24;
    int               height         = 24;
    nvcv::ImageFormat format         = nvcv::FMT_U8;
    int               numberOfImages = 10;

    // Create input varshape image batch
    std::vector<nvcv::Image> imgSrc;
    for (int i = 0; i < numberOfImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{width, height}, format);
    }

    nvcv::ImageBatchVarShape inBatch(numberOfImages);
    inBatch.pushBack(imgSrc.begin(), imgSrc.end());

    cvcuda::Stack op;

    // Invalid rank output tensor
    nvcv::Tensor invalidRankOutTensor(
        {
            {height, width},
            "HW"
    },
        nvcv::TYPE_U8);

    // Output tensor too small
    nvcv::Tensor smallOutTensor(numberOfImages - 1, {width, height}, format);

    // Dimension mismatch
    nvcv::Tensor misMatchOutTensor(numberOfImages, {width - 1, height}, format);

    // Test invalid cases
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &inBatch, &invalidRankOutTensor]
                                                             { op(stream, inBatch, invalidRankOutTensor); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &inBatch, &smallOutTensor] { op(stream, inBatch, smallOutTensor); }));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&op, &stream, &inBatch, &misMatchOutTensor]
                                                             { op(stream, inBatch, misMatchOutTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpStack_Negative, varshape_plane_mismatch_does_not_access_missing_planes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int kWidth  = 24;
    constexpr int kHeight = 24;

    nvcv::Image              image(nvcv::Size2D{kWidth, kHeight}, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape inBatch(1);
    inBatch.pushBack(image);
    nvcv::Tensor outTensor(
        {
            {1, 3, kHeight, kWidth},
            "NCHW"
    },
        nvcv::TYPE_U8);

    cvcuda::Stack op;
    EXPECT_NO_THROW(op(stream, inBatch, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
