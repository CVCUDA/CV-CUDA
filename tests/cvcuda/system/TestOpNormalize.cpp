/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpNormalize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cmath>
#include <random>

namespace test = nvcv::test;
namespace t    = ::testing;

static void Normalize(std::vector<uint8_t> &hDst, int dstRowStride, const std::vector<uint8_t> &hSrc, int srcRowStride,
                      nvcv::Size2D size, nvcv::ImageFormat fmt, const std::vector<float> &hBase, int baseRowStride,
                      nvcv::Size2D baseSize, nvcv::ImageFormat baseFormat, const std::vector<float> &hScale,
                      int scaleRowStride, nvcv::Size2D scaleSize, nvcv::ImageFormat scaleFormat,
                      const float globalScale, const float globalShift, const float epsilon, const uint32_t flags)
{
    using FT = float;

    for (int i = 0; i < size.h; i++)
    {
        const int bi = baseSize.h == 1 ? 0 : i;
        const int si = scaleSize.h == 1 ? 0 : i;

        for (int j = 0; j < size.w; j++)
        {
            const int bj = baseSize.w == 1 ? 0 : j;
            const int sj = scaleSize.w == 1 ? 0 : j;

            for (int k = 0; k < fmt.numChannels(); k++)
            {
                const int bk = (baseFormat.numChannels() == 1 ? 0 : k);
                const int sk = (scaleFormat.numChannels() == 1 ? 0 : k);

                FT mul;

                if (flags & CVCUDA_NORMALIZE_SCALE_IS_STDDEV)
                {
                    FT s = hScale.at(si * scaleRowStride + sj * scaleFormat.numChannels() + sk);
                    FT x = s * s + epsilon;
                    mul  = FT{1} / std::sqrt(x);
                }
                else
                {
                    mul = hScale.at(si * scaleRowStride + sj * scaleFormat.numChannels() + sk);
                }

                FT res = std::rint((hSrc.at(i * srcRowStride + j * fmt.numChannels() + k)
                                    - hBase.at(bi * baseRowStride + bj * baseFormat.numChannels() + bk))
                                       * mul * globalScale
                                   + globalShift);

                hDst.at(i * dstRowStride + j * fmt.numChannels() + k) = res < 0 ? 0 : (res > 255 ? 255 : res);
            }
        }
    }
}

static uint32_t normalScale   = 0;
static uint32_t scaleIsStdDev = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;

// clang-format off

NVCV_TEST_SUITE_P(OpNormalize, test::ValueList<int, int, int, bool, bool, uint32_t, float, float, float>
{
    // width, height, numImages, scalarBase, scalarScale,         flags, globalScale, globalShift, epsilon,
    {     32,     33,         1,       true,        true,   normalScale,         0.f,         0.f,     0.f, },
    {     66,     55,         1,       true,        true,   normalScale,         1.f,         0.f,     0.f, },
    {    122,    212,         2,       true,        true,   normalScale,      1.234f,      43.21f,     0.f, },
    {    211,    102,         3,      false,       false,   normalScale,        1.1f,        0.1f,     0.f, },
    {     21,     12,         5,       true,        true, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     22,     12,         5,      false,        true, scaleIsStdDev,        1.3f,        0.3f,     0.f, },
    {     55,     23,         5,       true,       false, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     72,     88,         5,      false,       false, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     63,     32,         7,      false,        true,   normalScale,        1.3f,        0.3f,     0.f, },
    {     22,     13,         9,       true,       false,   normalScale,        1.4f,        0.4f,     0.f, },
    {     55,     33,         2,       true,       false, scaleIsStdDev,        2.1f,        1.1f,   1.23f, },
    {    444,    222,         4,       true,       false, scaleIsStdDev,        2.2f,        2.2f,   12.3f, }
});

// clang-format on

TEST_P(OpNormalize, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      width       = GetParamValue<0>();
    int      height      = GetParamValue<1>();
    int      numImages   = GetParamValue<2>();
    bool     scalarBase  = GetParamValue<3>();
    bool     scalarScale = GetParamValue<4>();
    uint32_t flags       = GetParamValue<5>();
    float    globalScale = GetParamValue<6>();
    float    globalShift = GetParamValue<7>();
    float    epsilon     = GetParamValue<8>();

    int baseWidth      = (scalarBase ? 1 : width);
    int scaleWidth     = (scalarScale ? 1 : width);
    int baseHeight     = (scalarBase ? 1 : height);
    int scaleHeight    = (scalarScale ? 1 : height);
    int baseNumImages  = (scalarBase ? 1 : numImages);
    int scaleNumImages = (scalarScale ? 1 : numImages);

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);

    nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    std::default_random_engine rng;

    // Create input tensor
    nvcv::Tensor imgSrc  = nvcv::util::CreateTensor(numImages, width, height, fmt);
    auto         srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    int                               srcVecRowStride = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(height * srcVecRowStride);
        generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               height, cudaMemcpyHostToDevice));
    }

    // Create base tensor
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    auto         baseData = imgBase.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, baseData);
    auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
    ASSERT_TRUE(baseAccess);

    std::vector<std::vector<float>> baseVec(baseNumImages);
    int                             baseVecRowStride = baseWidth * baseFormat.numChannels();
    for (int i = 0; i < baseNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 255.f);

        baseVec[i].resize(baseHeight * baseVecRowStride);
        generate(baseVec[i].begin(), baseVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(i), baseAccess->rowStride(), baseVec[i].data(),
                                            baseVecRowStride * sizeof(float),
                                            baseVecRowStride * sizeof(float), // vec has no padding
                                            baseHeight, cudaMemcpyHostToDevice));
    }

    // Create scale tensor
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);
    auto         scaleData = imgScale.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, scaleData);
    auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
    ASSERT_TRUE(scaleAccess);

    std::vector<std::vector<float>> scaleVec(scaleNumImages);
    assert(scaleFormat.numPlanes() == 1);
    int scaleVecRowStride = scaleWidth * scaleFormat.numChannels();
    for (int i = 0; i < scaleNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 1.f);

        scaleVec[i].resize(scaleHeight * scaleVecRowStride);
        generate(scaleVec[i].begin(), scaleVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(i), scaleAccess->rowStride(), scaleVec[i].data(),
                                            scaleVecRowStride * sizeof(float),
                                            scaleVecRowStride * sizeof(float), // vec has no padding
                                            scaleHeight, cudaMemcpyHostToDevice));
    }

    // Create dest tensor
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, fmt);

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * dstVecRowStride);

        int bi = baseNumImages == 1 ? 0 : i;
        int si = scaleNumImages == 1 ? 0 : i;

        // Generate gold result
        Normalize(goldVec, dstVecRowStride, srcVec[i], srcVecRowStride, {width, height}, fmt, baseVec[bi],
                  baseVecRowStride, {baseWidth, baseHeight}, baseFormat, scaleVec[si], scaleVecRowStride,
                  {scaleWidth, scaleHeight}, scaleFormat, globalScale, globalShift, epsilon, flags);

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpNormalize, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      width       = GetParamValue<0>();
    int      height      = GetParamValue<1>();
    int      numImages   = GetParamValue<2>();
    bool     scalarBase  = GetParamValue<3>();
    bool     scalarScale = GetParamValue<4>();
    uint32_t flags       = GetParamValue<5>();
    float    globalScale = GetParamValue<6>();
    float    globalShift = GetParamValue<7>();
    float    epsilon     = GetParamValue<8>();

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);

    nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    std::default_random_engine rng;

    // Create input varshape

    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<nvcv::Image> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowStride(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt);

        int srcRowStride   = imgSrc[i].size().w * fmt.numChannels();
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i].size().h * srcRowStride);
        generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto imgData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               imgSrc[i].size().h, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create base tensor
    nvcv::Tensor imgBase(
        {
            {1, 1, 1, baseFormat.numChannels()},
            nvcv::TENSOR_NHWC
    },
        baseFormat.planeDataType(0));
    std::vector<float> baseVec(baseFormat.numChannels());
    {
        auto baseData = imgBase.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, baseData);
        auto baseAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*baseData);
        ASSERT_TRUE(baseAccess);

        std::uniform_real_distribution<float> udist(0, 255.f);
        generate(baseVec.begin(), baseVec.end(), [&]() { return udist(rng); });

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(0), baseAccess->rowStride(), baseVec.data(),
                                            baseVec.size() * sizeof(float),
                                            baseVec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice));
    }

    // Create scale tensor
    nvcv::Tensor imgScale(
        {
            {1, 1, 1, scaleFormat.numChannels()},
            nvcv::TENSOR_NHWC
    },
        scaleFormat.planeDataType(0));
    std::vector<float> scaleVec(scaleFormat.numChannels());
    {
        auto scaleData = imgScale.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, scaleData);
        auto scaleAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*scaleData);
        ASSERT_TRUE(scaleAccess);

        std::uniform_real_distribution<float> udist(0, 1.f);
        generate(scaleVec.begin(), scaleVec.end(), [&]() { return udist(rng); });

        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(0), scaleAccess->rowStride(), scaleVec.data(),
                                            scaleVec.size() * sizeof(float),
                                            scaleVec.size() * sizeof(float), // vec has no padding
                                            1, cudaMemcpyHostToDevice));
    }

    // Create output varshape
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(imgSrc[i].size(), imgSrc[i].format());
    }
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_NO_THROW(
        normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int width  = srcData->plane(0).width;
        int height = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int dstRowStride = srcVecRowStride[i];

        std::vector<uint8_t> testVec(height * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * dstRowStride);

        // Generate gold result
        Normalize(goldVec, dstRowStride, srcVec[i], srcVecRowStride[i], {width, height}, fmt, baseVec, 0, {1, 1},
                  baseFormat, scaleVec, 0, {1, 1}, scaleFormat, globalScale, globalShift, epsilon, flags);

        EXPECT_THAT(testVec, t::ElementsAreArray(goldVec));
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpNormalize_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, bool>{
    // inFmt, outFmt, isVarShapeDifferentFormatTest
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, false},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, false},
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, false},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, true},
});

// clang-format on

TEST_P(OpNormalize_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFmt                         = GetParamValue<0>();
    nvcv::ImageFormat outFmt                        = GetParamValue<1>();
    bool              isVarShapeDifferentFormatTest = GetParamValue<2>();

    if (isVarShapeDifferentFormatTest)
    {
        GTEST_SKIP() << "Skip varshape different format test for tensor test";
    }

    int width     = 24;
    int height    = 24;
    int numImages = 2;

    uint32_t flags       = normalScale;
    float    globalScale = 0.f;
    float    globalShift = 0.f;
    float    epsilon     = 0.f;

    int baseWidth      = 1;
    int scaleWidth     = 1;
    int baseHeight     = 1;
    int scaleHeight    = 1;
    int baseNumImages  = 1;
    int scaleNumImages = 1;

    nvcv::ImageFormat baseFormat  = nvcv::FMT_F32;
    nvcv::ImageFormat scaleFormat = nvcv::FMT_F32;

    // Create input and output tensors
    nvcv::Tensor imgSrc = nvcv::util::CreateTensor(numImages, width, height, inFmt);
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);
    nvcv::Tensor imgDst = nvcv::util::CreateTensor(numImages, width, height, outFmt);

    // Generate test result
    cvcuda::Normalize normalizeOp;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&] { normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags); }));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpNormalize_Negative, varshape_op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inFmt                         = GetParamValue<0>();
    nvcv::ImageFormat outFmt                        = GetParamValue<1>();
    bool              isVarShapeDifferentFormatTest = GetParamValue<2>();

    std::vector<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>> testSet{
        {isVarShapeDifferentFormatTest ? nvcv::FMT_U16 : inFmt,                                                 outFmt},
        {                                                inFmt, isVarShapeDifferentFormatTest ? nvcv::FMT_U16 : outFmt}
    };

    // check
    if (isVarShapeDifferentFormatTest)
    {
        ASSERT_NE(testSet[0].first, inFmt);
        ASSERT_EQ(testSet[0].second, outFmt);
        ASSERT_EQ(testSet[1].first, inFmt);
        ASSERT_NE(testSet[1].second, outFmt);
    }
    else
    {
        testSet.pop_back();
        ASSERT_EQ(testSet[0].first, inFmt);
        ASSERT_EQ(testSet[0].second, outFmt);
    }

    for (auto testCase : testSet)
    {
        nvcv::ImageFormat inputFmtExtra  = std::get<0>(testCase);
        nvcv::ImageFormat outputFmtExtra = std::get<1>(testCase);

        int width     = 24;
        int height    = 24;
        int numImages = 5;

        uint32_t flags       = normalScale;
        float    globalScale = 0.f;
        float    globalShift = 0.f;
        float    epsilon     = 0.f;

        nvcv::ImageFormat baseFormat  = nvcv::FMT_F32;
        nvcv::ImageFormat scaleFormat = nvcv::FMT_F32;

        std::default_random_engine rng;

        // Create input and output varshape

        std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
        std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < numImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, inFmt);
            imgDst.emplace_back(imgSrc[i].size(), outFmt);
        }
        imgSrc.emplace_back(imgSrc[0].size(), inputFmtExtra);
        imgDst.emplace_back(imgSrc.back().size(), outputFmtExtra);

        nvcv::ImageBatchVarShape batchSrc(numImages);
        nvcv::ImageBatchVarShape batchDst(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        // Create base tensor
        nvcv::Tensor imgBase(
            {
                {1, 1, 1, baseFormat.numChannels()},
                nvcv::TENSOR_NHWC
        },
            baseFormat.planeDataType(0));

        // Create scale tensor
        nvcv::Tensor imgScale(
            {
                {1, 1, 1, scaleFormat.numChannels()},
                nvcv::TENSOR_NHWC
        },
            scaleFormat.planeDataType(0));

        // Generate test result
        cvcuda::Normalize normalizeOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&] {
                                                       normalizeOp(stream, batchSrc, imgBase, imgScale, batchDst,
                                                                   globalScale, globalShift, epsilon, flags);
                                                   }));

        // Get test data back
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpNormalize_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaNormalizeCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
