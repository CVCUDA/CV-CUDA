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
#include <cvcuda/OpHistogramEq.hpp>
#include <cvcuda/cuda_tools/SaturateCast.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

static int ScaledSize(int size, double scale)
{
    return static_cast<int>(size * scale);
}

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
        auto smallestIt = std::ranges::find_if(cumulativeHistogram, [](int i) { return i != 0; });
        auto smallest   = *smallestIt;

        // Normalize the cumulative histogram to get the mapping values
        std::array<uint8_t, 256> equalizeMap = {};
        if ((width * height - smallest) == 0)
        {
            for (int i = 0; i < 256; ++i)
            {
                equalizeMap[i] = static_cast<uint8_t>(smallestIt - cumulativeHistogram.begin());
            }
        }
        else
        {
            for (int i = 0; i < 256; ++i)
            {
                int   tmpT     = (cumulativeHistogram[i] - smallest);
                int   tmpB     = ((width * height) - smallest);
                float ratio    = (float)(tmpT * 255) / (float)tmpB;
                equalizeMap[i] = nvcv::cuda::SaturateCast<uint8_t>(ratio);
            }
        }

        // Map the input image pixel values into the equalized values
        for (size_t i = channel; i < inputImage.size(); i += numChannels)
        {
            outputImage[i] = equalizeMap[inputImage[i]];
        }
    }
}

#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)

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

    {       6,        4,     NVCV_IMAGE_FORMAT_2U8,            1},
    {     121,      124,     NVCV_IMAGE_FORMAT_2U8,            2},
    {      43,       38,     NVCV_IMAGE_FORMAT_2U8,            1},
    {      36,       12,     NVCV_IMAGE_FORMAT_2U8,            3},
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
    for (bool AllPixelsHaveSameValue : std::vector<bool>{false, true}) // NOSONAR
    {
        nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
        nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

        std::default_random_engine             randEng(0);
        std::uniform_int_distribution<uint8_t> rand(0, 255);
        auto   colorChannels  = static_cast<int>(inTensor.shape()[inTensor.shape().rank() - 1]);
        size_t imageSizeBytes = width * height * colorChannels * sizeof(uint8_t);

        for (int i = 0; i < batches; ++i) // NOSONAR
        {
            // generate random input image
            std::vector<uint8_t> imageVec(imageSizeBytes);
            // all pixels have same value
            if (AllPixelsHaveSameValue)
            {
                std::ranges::generate(imageVec, [&rand, &randEng]() { return rand(randEng); });
            }
            else
            {
                std::ranges::generate(imageVec, []() { return 5; });
            }
            // copy random input to device tensor
            EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
        }

        // run operator
        cvcuda::HistogramEq op(batches);
        EXPECT_NO_THROW(op(stream, inTensor, outTensor));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        for (int batchIndex = 0; batchIndex < batches; ++batchIndex) // NOSONAR
        {
            // get output image
            std::vector<uint8_t> outImage;
            std::vector<uint8_t> input;
            std::vector<uint8_t> goldImage;

            EXPECT_NO_THROW(util::GetImageVectorFromTensor(inTensor.exportData(), batchIndex, input));
            histogramEqualization(input, width, height, colorChannels, goldImage);
            EXPECT_NO_THROW(util::GetImageVectorFromTensor(outTensor.exportData(), batchIndex, outImage));
            ASSERT_EQ(outImage, goldImage);
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogramEq, tensor_c1_nchw_widened_body_and_tail)
{
    constexpr int width   = 16 * 8 + 3;
    constexpr int height  = 17;
    constexpr int batches = 2;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor inTensor(
        {
            {batches, 1, height, width},
            "NCHW"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor outTensor(
        {
            {batches, 1, height, width},
            "NCHW"
    },
        nvcv::TYPE_U8);

    auto inData  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = outTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess);
    ASSERT_TRUE(outAccess);

    std::vector<std::vector<uint8_t>> inputs(batches, std::vector<uint8_t>(width * height));
    for (int sample = 0; sample < batches; ++sample)
    {
        test::planar::FillDeterministicValues(inputs[sample], sample * 101 + 13, nvcv::TYPE_U8);
        test::planar::UploadPlanarSample(*inAccess, sample, inputs[sample], width, height, 1, 1);
    }

    cvcuda::HistogramEq op(batches);
    EXPECT_NO_THROW(op(stream, inTensor, outTensor));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < batches; ++sample)
    {
        SCOPED_TRACE(sample);
        std::vector<uint8_t> gold;
        histogramEqualization(inputs[sample], width, height, 1, gold);
        auto output = test::planar::DownloadPlanarSample(*outAccess, sample, width, height, 1, 1);
        EXPECT_EQ(output, gold);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogramEqVarShape, test::ValueList<int, int, NVCVImageFormat, int>
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

    {       6,        4,     NVCV_IMAGE_FORMAT_2U8,            1},
    {     121,      124,     NVCV_IMAGE_FORMAT_2U8,            2},
    {      43,       38,     NVCV_IMAGE_FORMAT_2U8,            1},
    {      36,       12,     NVCV_IMAGE_FORMAT_2U8,            3},
});

// clang-format on

TEST_P(OpHistogramEqVarShape, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int               width  = GetParamValue<0>();
    const int               height = GetParamValue<1>();
    const nvcv::ImageFormat format{GetParamValue<2>()};
    const int               batches = GetParamValue<3>();
    for (bool AllPixelsHaveSameValue : std::vector<bool>{false, true}) // NOSONAR
    {
        // Create input varshape
        std::default_random_engine    rng;
        std::uniform_int_distribution udistWidth(ScaledSize(width, 0.8), ScaledSize(width, 1.1));
        std::uniform_int_distribution udistHeight(ScaledSize(height, 0.8), ScaledSize(height, 1.1));

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
            // all pixels have same value
            if (AllPixelsHaveSameValue)
            {
                std::ranges::generate(srcVec[i], []() { return 5; });
            }
            else
            {
                std::ranges::generate(srcVec[i], [&udist, &rng]() { return udist(rng); });
            }

            auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
            ASSERT_NE(imgData, nvcv::NullOpt);

            // Copy input data to the GPU
            ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride,
                                                     srcVec[i].data(), srcRowStride, srcRowStride, imgSrc[i].size().h,
                                                     cudaMemcpyHostToDevice, stream));
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
                      cudaMemcpy2D(opOutVec.data(), shape.x * numCh, dstData->plane(0).basePtr,
                                   dstData->plane(0).rowStride, shape.x * numCh, shape.y, cudaMemcpyDeviceToHost));

            // Copy input data to Host to generate goldVector
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(inputVec.data(), shape.x * numCh, srcData->plane(0).basePtr,
                                   srcData->plane(0).rowStride, shape.x * numCh, shape.y, cudaMemcpyDeviceToHost));
            histogramEqualization(inputVec, shape.x, shape.y, numCh, goldVec);
            EXPECT_EQ(opOutVec, goldVec);
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Planar == interleaved parity ---------------------------------------------------------------
// clang-format off
NVCV_TEST_SUITE_P(OpHistogramEqPlanar,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {37, 29, 2,  nvcv::FMT_RGB8p,  nvcv::FMT_RGB8},
    {32, 25, 1, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
});

NVCV_TEST_SUITE_P(OpHistogramEqPlanarVarShape,
                  test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {37, 29, 2, nvcv::FMT_RGB8p, nvcv::FMT_RGB8},
});

// clang-format on

TEST_P(OpHistogramEqPlanar, tensor_matches_interleaved)
{
    const int numImages = GetParamValue<2>();
    test::planar::RunTensorParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), numImages,
        [numImages](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::HistogramEq op(numImages);
            EXPECT_NO_THROW(op(s, src, dst));
        });
}

TEST_P(OpHistogramEqPlanarVarShape, varshape_matches_interleaved)
{
    test::planar::RunVarShapeParity(
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<0>(),
        GetParamValue<1>(), GetParamValue<2>(),
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::HistogramEq op(src.numImages());
            EXPECT_NO_THROW(op(s, src, dst));
        });
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogramEq_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf16},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p},
});

NVCV_TEST_SUITE_P(OpHistogramEqVarshape_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int>{
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 2, 2},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 2, 2},
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf16, 2, 2},
    {nvcv::FMT_RGBf16, nvcv::FMT_RGBf16, 2, 2},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 4, 3},
});

// clang-format on

TEST_P(OpHistogramEq_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFmt  = GetParamValue<0>();
    nvcv::ImageFormat outFmt = GetParamValue<1>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(2, 24, 24, inFmt);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(2, 24, 24, outFmt);

    // run operator
    cvcuda::HistogramEq op(2);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &inTensor, &outTensor] { op(stream, inTensor, outTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogramEq_Negative, rejects_two_channel_planar_tensor)
{
    test::planar::ExpectPlanarTensorRejected({2, 2, 24, 24}, {2, 2, 24, 24},
                                             [](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst)
                                             {
                                                 cvcuda::HistogramEq op(2);
                                                 op(s, src, dst);
                                             });
}

TEST(OpHistogramEq_Negative, rejects_two_channel_planar_varshape)
{
    const nvcv::ImageFormat twoChannelPlanar{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X8)};

    test::planar::ExpectVarShapeRejected(
        {twoChannelPlanar, twoChannelPlanar}, {twoChannelPlanar, twoChannelPlanar},
        [](cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst)
        {
            cvcuda::HistogramEq op(2);
            op(s, src, dst);
        });
}

#undef NVCV_IMAGE_FORMAT_2U8

TEST_P(OpHistogramEqVarshape_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFmt        = GetParamValue<0>();
    nvcv::ImageFormat outFmt       = GetParamValue<1>();
    int               inputImages  = GetParamValue<2>();
    int               outputImages = GetParamValue<3>();

    // Create input varshape
    std::default_random_engine    rng;
    std::uniform_int_distribution udistWidth(ScaledSize(24, 0.8), ScaledSize(24, 1.1));
    std::uniform_int_distribution udistHeight(ScaledSize(24, 0.8), ScaledSize(24, 1.1));

    std::vector<nvcv::Image> imgSrc;
    //setup the input images
    for (int i = 0; i < inputImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(inputImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < outputImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), outFmt);
    }
    // Push the images on the varShape Batch.
    nvcv::ImageBatchVarShape batchDst(outputImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Run operator set the max batches
    cvcuda::HistogramEq op(std::max(inputImages, outputImages));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst] { op(stream, batchSrc, batchDst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogramEq_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaHistogramEqCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpHistogramEq_Negative, create_invalid_maxBatch)
{
    NVCVOperatorHandle handle;
    EXPECT_EQ(cvcudaHistogramEqCreate(&handle, 0), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpHistogramEq_Negative, tensor_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(2, 4, 4, nvcv::FMT_U8);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(2, 4, 4, nvcv::FMT_U8);

    cvcuda::HistogramEq op(1);
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &inTensor, &outTensor] { op(stream, inTensor, outTensor); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogramEqVarshape_Negative, varshape_batch_exceeds_maxBatch)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < 2; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrc(2);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    nvcv::ImageBatchVarShape batchDst(2);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::HistogramEq op(1);
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst] { op(stream, batchSrc, batchDst); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogramEqVarshape_Negative, varshape_hasDifferentFormat)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    std::vector<std::tuple<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {nvcv::FMT_U8,          fmt},
        {         fmt, nvcv::FMT_U8}
    };

    for (const auto &[inputFmtExtra, outputFmtExtra] : testSet)
    {
        int numImages = 3;

        // Create input varshape
        std::default_random_engine    rng;
        std::uniform_int_distribution udistWidth(ScaledSize(24, 0.8), ScaledSize(24, 1.1));
        std::uniform_int_distribution udistHeight(ScaledSize(24, 0.8), ScaledSize(24, 1.1));

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;
        //setup the input images
        for (int i = 0; i < numImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);
            imgDst.emplace_back(imgSrc[i].size(), fmt);
        }
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Run operator set the max batches
        cvcuda::HistogramEq op(numImages);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst] { op(stream, batchSrc, batchDst); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
