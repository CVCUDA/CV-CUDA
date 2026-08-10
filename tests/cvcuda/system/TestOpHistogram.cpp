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
#include <cvcuda/OpHistogram.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <exception>
#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
namespace util = nvcv::util;

static void computeHistogram(const std::vector<uint8_t> &imageVec, std::vector<uint32_t> &goldHistogram)
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

    for (int i = 0; i < batches; ++i) // NOSONAR
    {
        // generate random input image
        std::vector<uint8_t> imageVec(width * height);
        std::ranges::generate(imageVec, [&rand, &randEng]() { return rand(randEng); });

        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
        // Compute histogram and add to vector
        computeHistogram(imageVec, goldHistogram);
    }

    // run operator
    cvcuda::Histogram op;
    EXPECT_NO_THROW(op(stream, inTensor, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, histogram));
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

    for (int i = 0; i < batches; ++i) // NOSONAR
    {
        // generate random input image
        std::vector<uint8_t> imageVec(width * height);
        std::ranges::generate(imageVec, [&rand, &randEng]() { return rand(randEng); });
        //generate random mask
        std::vector<uint8_t> maskVec(width * height);
        std::ranges::generate(maskVec, [&randMask, &randEng]() { return randMask(randEng); });

        // copy random input to device tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, i));
        // copy mask input to tensor
        EXPECT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inMask.exportData(), maskVec, i));

        // Compute histogram and add to vector
        computeHistogramWithMask(imageVec, maskVec, goldHistogram);
    }

    // run operator
    cvcuda::Histogram op;
    EXPECT_NO_THROW(op(stream, inTensor, nvcv::OptionalTensorConstRef{inMask}, histogram));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint32_t> opHistogram;
    // get 0th sample since histogram is just a 2d tensor
    EXPECT_NO_THROW(util::GetImageVectorFromTensor(histogram.exportData(), 0, opHistogram));

    // Compare the computed histogram with the output histogram
    ASSERT_EQ(opHistogram, goldHistogram);
}

static nvcv::Tensor CreateSingleChannelTensor(int batches, int width, int height, bool planar, bool batched)
{
    if (planar && batched)
    {
        nvcv::TensorShape shape{
            {batches, 1, height, width},
            "NCHW"
        };
        return nvcv::Tensor(shape, nvcv::TYPE_U8);
    }
    if (planar)
    {
        nvcv::TensorShape shape{
            {1, height, width},
            "CHW"
        };
        return nvcv::Tensor(shape, nvcv::TYPE_U8);
    }
    if (batched)
    {
        return nvcv::util::CreateTensor(batches, width, height, nvcv::FMT_U8);
    }

    nvcv::TensorShape shape{
        {height, width, 1},
        "HWC"
    };
    return nvcv::Tensor(shape, nvcv::TYPE_U8);
}

template<typename T>
static gt::AssertionResult SetImageTensor(nvcv::Tensor &tensor, std::vector<T> &values, int sample)
{
    try
    {
        util::SetImageTensorFromVector<T>(tensor.exportData(), values, sample);
    }
    catch (const util::TensorDataUtilsError &e)
    {
        return gt::AssertionFailure() << e.what();
    }
    return gt::AssertionSuccess();
}

static std::vector<uint8_t> MakeHistogramParityInput(int width, int height, int sample)
{
    // Use standard LCG constants and the top state byte for reproducible, well-spread intensities.
    constexpr uint32_t kLcgMultiplier = 1664525u;
    constexpr uint32_t kLcgIncrement  = 1013904223u;

    std::vector<uint8_t> imageVec(width * height);
    for (size_t i = 0; i < imageVec.size(); ++i)
    {
        const uint32_t state = kLcgMultiplier * static_cast<uint32_t>(i + sample * imageVec.size()) + kLcgIncrement;
        imageVec[i]          = static_cast<uint8_t>(state >> 24);
    }
    return imageVec;
}

static std::vector<uint8_t> MakeHistogramParityMask(int width, int height, int sample)
{
    std::vector<uint8_t> maskVec(width * height);
    for (size_t i = 0; i < maskVec.size(); ++i)
    {
        maskVec[i] = ((i + static_cast<size_t>(sample)) % 3) == 0 ? 0 : 1;
    }
    return maskVec;
}

static gt::AssertionResult FillHistogramParityInputs(nvcv::Tensor &interleaved, nvcv::Tensor &planar, int width,
                                                     int height, int batches)
{
    for (int i = 0; i < batches; ++i)
    {
        std::vector<uint8_t> imageVec = MakeHistogramParityInput(width, height, i);

        if (auto result = SetImageTensor(interleaved, imageVec, i); !result)
        {
            return result << " while filling interleaved sample " << i;
        }
        if (auto result = SetImageTensor(planar, imageVec, i); !result)
        {
            return result << " while filling planar sample " << i;
        }
    }
    return gt::AssertionSuccess();
}

static gt::AssertionResult FillHistogramParityMasks(nvcv::Tensor &interleaved, nvcv::Tensor &planar, int width,
                                                    int height, int batches)
{
    for (int i = 0; i < batches; ++i)
    {
        std::vector<uint8_t> maskVec = MakeHistogramParityMask(width, height, i);

        if (auto result = SetImageTensor(interleaved, maskVec, i); !result)
        {
            return result << " while filling interleaved mask sample " << i;
        }
        if (auto result = SetImageTensor(planar, maskVec, i); !result)
        {
            return result << " while filling planar mask sample " << i;
        }
    }
    return gt::AssertionSuccess();
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogramPlanar, test::ValueList<int, int, int, bool, bool>
{
    // width, height, batches, useMask, batched
    {    17,     13,       3,   false,    true},
    {    17,     13,       3,    true,    true},
    {    19,     11,       1,   false,   false},
    {    19,     11,       1,    true,   false},
});

// clang-format on

TEST_P(OpHistogramPlanar, output_matches_interleaved)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int  width   = GetParamValue<0>();
    int  height  = GetParamValue<1>();
    int  batches = GetParamValue<2>();
    bool useMask = GetParamValue<3>();
    bool batched = GetParamValue<4>();

    nvcv::Tensor srcI  = CreateSingleChannelTensor(batches, width, height, false, batched);
    nvcv::Tensor srcP  = CreateSingleChannelTensor(batches, width, height, true, batched);
    nvcv::Tensor histI = nvcv::util::CreateTensor(1, 256, batches, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_S32));
    nvcv::Tensor histP = nvcv::util::CreateTensor(1, 256, batches, nvcv::ImageFormat(NVCV_IMAGE_FORMAT_S32));

    ASSERT_TRUE(FillHistogramParityInputs(srcI, srcP, width, height, batches));

    cvcuda::Histogram op;
    if (useMask)
    {
        nvcv::Tensor maskI = CreateSingleChannelTensor(batches, width, height, false, batched);
        nvcv::Tensor maskP = CreateSingleChannelTensor(batches, width, height, true, batched);
        ASSERT_TRUE(FillHistogramParityMasks(maskI, maskP, width, height, batches));

        EXPECT_NO_THROW(op(stream, srcI, nvcv::OptionalTensorConstRef{maskI}, histI));
        EXPECT_NO_THROW(op(stream, srcP, nvcv::OptionalTensorConstRef{maskP}, histP));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    else
    {
        EXPECT_NO_THROW(op(stream, srcI, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, histI));
        EXPECT_NO_THROW(op(stream, srcP, nvcv::OptionalTensorConstRef{nvcv::NullOpt}, histP));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint32_t> histIVec;
    std::vector<uint32_t> histPVec;
    EXPECT_NO_THROW(util::GetImageVectorFromTensor(histI.exportData(), 0, histIVec));
    EXPECT_NO_THROW(util::GetImageVectorFromTensor(histP.exportData(), 0, histPVec));
    ASSERT_EQ(histIVec, histPVec);
}

static void ExpectHistogramInvalidArgument(const nvcv::Tensor &histogram, nvcv::OptionalTensorConstRef mask)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor         inTensor = nvcv::util::CreateTensor(1, 2, 2, nvcv::FMT_U8);
    std::vector<uint8_t> imageVec(4, 0);
    ASSERT_NO_THROW(util::SetImageTensorFromVector<uint8_t>(inTensor.exportData(), imageVec, 0));

    cvcuda::Histogram op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { op(stream, inTensor, mask, histogram); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpHistogram_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, int, int>{
    // inFmt, histFmt, batches, histHeight
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 3, 3},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 3, 3},
    {nvcv::FMT_F16, nvcv::FMT_F16, nvcv::FMT_S32, 3, 3},
    {nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_S32, 3, 5},
    {nvcv::FMT_U8, nvcv::FMT_U8, nvcv::FMT_S32, 5, 3},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_S32, 3, 3},
    {nvcv::FMT_U8, nvcv::FMT_RGB8, nvcv::FMT_S32, 3, 3},
});

// clang-format on

TEST_P(OpHistogram_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat maskFmt    = GetParamValue<1>();
    nvcv::ImageFormat histFmt    = GetParamValue<2>();
    int               batches    = GetParamValue<3>();
    int               histHeight = GetParamValue<4>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, 16, 16, inputFmt);
    nvcv::Tensor inMask    = nvcv::util::CreateTensor(batches, 16, 16, maskFmt);
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, histHeight, histFmt);

    // run operator
    cvcuda::Histogram op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &inTensor, &inMask, &histogram]
                                { op(stream, inTensor, nvcv::OptionalTensorConstRef{inMask}, histogram); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogram_Negative, rejects_invalid_histogram_dtype)
{
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, 1, nvcv::FMT_U8);
    ExpectHistogramInvalidArgument(histogram, nvcv::OptionalTensorConstRef{nvcv::NullOpt});
}

TEST(OpHistogram_Negative, rejects_histogram_width_less_than_256)
{
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 255, 1, nvcv::FMT_S32);
    ExpectHistogramInvalidArgument(histogram, nvcv::OptionalTensorConstRef{nvcv::NullOpt});
}

TEST(OpHistogram_Negative, rejects_histogram_with_multiple_channels)
{
    nvcv::Tensor histogram(
        {
            {1, 256, 2},
            "HWC"
    },
        nvcv::TYPE_S32);
    ExpectHistogramInvalidArgument(histogram, nvcv::OptionalTensorConstRef{nvcv::NullOpt});
}

TEST(OpHistogram_Negative, rejects_invalid_mask_dtype)
{
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, 1, nvcv::FMT_S32);
    nvcv::Tensor mask      = nvcv::util::CreateTensor(1, 2, 2, nvcv::FMT_S32);

    std::vector<int32_t> maskVec(4, 1);
    ASSERT_NO_THROW(util::SetImageTensorFromVector<int32_t>(mask.exportData(), maskVec, 0));

    ExpectHistogramInvalidArgument(histogram, nvcv::OptionalTensorConstRef{mask});
}

TEST(OpHistogram_Negative, rejects_mask_layout_different_from_input)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor inTensor{
        {{1, 2, 2, 1}, "NHWC"},
        nvcv::TYPE_U8
    };
    nvcv::Tensor mask{
        {{2, 2, 1}, "HWC"},
        nvcv::TYPE_U8
    };
    nvcv::Tensor histogram = nvcv::util::CreateTensor(1, 256, 1, nvcv::FMT_S32);

    cvcuda::Histogram op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(stream, inTensor, nvcv::OptionalTensorConstRef{mask}, histogram); }));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpHistogram_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaHistogramCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
