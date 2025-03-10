/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpPadAndStack.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>

namespace test = nvcv::test;

static void PadAndStack(std::vector<uint8_t> &hDst, const std::vector<std::vector<uint8_t>> &hBatchSrc,
                        const nvcv::TensorDataAccessStridedImagePlanar &dDstData, const int srcWidth,
                        const int srcHeight, const int srcRowStride, const int srcPixPitch,
                        const std::vector<int> &topVec, const std::vector<int> &leftVec,
                        const NVCVBorderType borderType, const float borderValue)
{
    int dstPixPitch  = dDstData.numChannels();
    int dstRowStride = dDstData.rowStride() / sizeof(uint8_t);
    int dstImgPitch  = dDstData.sampleStride() / sizeof(uint8_t);

    int2 coord, size{srcWidth, srcHeight};

    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coord.y = di - topVec[db];

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coord.x = dj - leftVec[db];

                for (int dk = 0; dk < dDstData.numChannels(); dk++)
                {
                    uint8_t out = 0;

                    if (coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y)
                    {
                        out = hBatchSrc[db][coord.y * srcRowStride + coord.x * srcPixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<uint8_t>(borderValue);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                            {
                                test::ReplicateBorderIndex(coord, size);
                            }
                            else if (borderType == NVCV_BORDER_WRAP)
                            {
                                test::WrapBorderIndex(coord, size);
                            }
                            else if (borderType == NVCV_BORDER_REFLECT)
                            {
                                test::ReflectBorderIndex(coord, size);
                            }
                            else if (borderType == NVCV_BORDER_REFLECT101)
                            {
                                test::Reflect101BorderIndex(coord, size);
                            }

                            out = hBatchSrc[db][coord.y * srcRowStride + coord.x * srcPixPitch + dk];
                        }
                    }

                    hDst[db * dstImgPitch + di * dstRowStride + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStack, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         NVCVBorderType, borderValue
    {       212,       113,          1,      111,       132,      0,       0,   NVCV_BORDER_CONSTANT,         0.f},
    {        12,        13,          2,      211,       232,      0,       3,   NVCV_BORDER_CONSTANT,        12.f},
    {       212,       113,          3,       11,       432,      5,       0,   NVCV_BORDER_CONSTANT,        13.f},
    {       212,       613,          4,      311,       532,      7,       7,   NVCV_BORDER_CONSTANT,       134.f},

    {       234,       131,          2,      131,       130,     33,      22,  NVCV_BORDER_REPLICATE,         0.f},
    {       234,       131,          2,      123,       132,     41,      42,    NVCV_BORDER_REFLECT,         0.f},
    {       234,       131,          2,      134,       131,     53,      62,       NVCV_BORDER_WRAP,         0.f},
    {       243,       123,          2,      132,       123,     77,      98, NVCV_BORDER_REFLECT101,         0.f},

});

// clang-format on

TEST_P(OpPadAndStack, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int dstWidth   = GetParamValue<3>();
    int dstHeight  = GetParamValue<4>();
    int topPad     = GetParamValue<5>();
    int leftPad    = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();

    float borderValue = GetParamValue<8>();

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    auto inTopData  = inTop.exportData<nvcv::TensorDataStridedCuda>();
    auto inLeftData = inLeft.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    auto inTopAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inTopData);
    ASSERT_TRUE(inTopAccess);

    auto inLeftAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inLeftData);
    ASSERT_TRUE(inLeftAccess);

    int inTopBufSize  = (inTopAccess->sampleStride() / sizeof(int)) * inTopAccess->numSamples();
    int inLeftBufSize = (inLeftAccess->sampleStride() / sizeof(int)) * inLeftAccess->numSamples();

    ASSERT_EQ(inTopBufSize, inLeftBufSize);

    std::vector<int> topVec(inTopBufSize);
    std::vector<int> leftVec(inLeftBufSize);

    for (int b = 0; b < numBatches; ++b)
    {
        topVec[b]  = topPad;
        leftVec[b] = leftPad;
    }

    // Copy vectors with top and left padding to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<nvcv::Image> srcImgVec;

    std::vector<std::vector<uint8_t>> batchSrcVec;

    std::default_random_engine randEng{0};

    int srcStride = 0, srcRowStride = 0, srcPixPitch = 0;

    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGBA8);

        auto imgSrcData = srcImgVec.back().exportData<nvcv::ImageDataStridedCuda>();

        srcStride      = imgSrcData->plane(0).rowStride;
        srcRowStride   = srcStride / sizeof(uint8_t);
        srcPixPitch    = 4;
        int srcBufSize = srcRowStride * imgSrcData->plane(0).height;

        std::vector<uint8_t> srcVec(srcBufSize);

        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(),
                                               srcVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

        batchSrcVec.push_back(srcVec);
    }

    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);

    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);

    int dstBufSize = (dstAccess->sampleStride() / sizeof(uint8_t)) * dstAccess->numSamples();

    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(uint8_t), stream));

    std::vector<uint8_t> testVec(dstBufSize);
    std::vector<uint8_t> goldVec(dstBufSize);

    // Generate gold result
    PadAndStack(goldVec, batchSrcVec, *dstAccess, srcWidth, srcHeight, srcRowStride, srcPixPitch, topVec, leftVec,
                borderType, borderValue);

    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_NO_THROW(padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(goldVec, testVec);
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStack_Negative, test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType>
{
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // input is not nhwc/hwc
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, nvcv::FMT_S32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // output is not nhwc/hwc
#ifndef ENABLE_SANITIZER
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_S32, static_cast<NVCVBorderType>(255)}, // invalid borderType
#endif
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, nvcv::FMT_S32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // invalid datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_F16, nvcv::FMT_S32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // invalid datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_U8, nvcv::FMT_S32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // invalid datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_F32, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // invalid top datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_F32, NVCV_BORDER_CONSTANT}, // invalid left datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, nvcv::FMT_S32, NVCV_BORDER_CONSTANT}, // invalid top datatype
    {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_RGBA8p, NVCV_BORDER_CONSTANT}, // invalid left datatype
});

// clang-format on

TEST_P(OpPadAndStack_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    nvcv::ImageFormat topFmt             = GetParamValue<3>();
    nvcv::ImageFormat leftFmt            = GetParamValue<4>();
    NVCVBorderType    borderType         = GetParamValue<5>();

    int   srcWidth    = 12;
    int   srcHeight   = 13;
    int   numBatches  = 3;
    int   dstWidth    = 111;
    int   dstHeight   = 131;
    float borderValue = 0.f;

    nvcv::Tensor inTop(1, {numBatches, 1}, topFmt);
    nvcv::Tensor inLeft(1, {numBatches, 1}, leftFmt);

    std::vector<nvcv::Image> srcImgVec;
    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, inputFmt);
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor        imgDst(numBatches, {dstWidth, dstHeight}, outputFmt);
    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_EQ(
        expectedReturnCode,
        nvcv::ProtectCall([&] { padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue); }));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPadAndStack_Negative, input_format_not_same)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int   srcWidth    = 12;
    int   srcHeight   = 13;
    int   numBatches  = 3;
    int   dstWidth    = 111;
    int   dstHeight   = 131;
    float borderValue = 0.f;

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    std::vector<nvcv::Image> srcImgVec;
    for (int b = 0; b < numBatches - 1; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGBA8);
    }
    srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor        imgDst(numBatches, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);
    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&] { padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, NVCV_BORDER_CONSTANT, borderValue); }));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
