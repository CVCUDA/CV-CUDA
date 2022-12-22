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

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <nvcv/DataLayout.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/IImageData.hpp>
#include <nvcv/ITensorData.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <random>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

namespace test = nvcv::test;

template<typename T>
static void CopyMakeBorder(std::vector<T> &hDst, const std::vector<T> &hSrc,
                           const nvcv::TensorDataAccessStridedImagePlanar &dDstData, const int srcWidth,
                           const int srcHeight, const int srcRowStride, const int srcPixPitch, const int srcImgPitch,
                           const int top, const int left, const NVCVBorderType borderType, const float4 borderValue)
{
    int dstPixPitch  = dDstData.numChannels();
    int dstRowStride = dDstData.rowStride() / sizeof(T);
    int dstImgPitch  = dDstData.sampleStride() / sizeof(T);

    int2 coords, size{srcWidth, srcHeight};
    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coords.y = di - top;

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coords.x = dj - left;

                for (int dk = 0; dk < dDstData.numChannels(); dk++)
                {
                    T out = 0;

                    if (coords.x >= 0 && coords.x < srcWidth && coords.y >= 0 && coords.y < srcHeight)
                    {
                        out = hSrc[db * srcImgPitch + coords.y * srcRowStride + coords.x * srcPixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<T>(reinterpret_cast<const float *>(&borderValue)[dk]);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                                test::ReplicateBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_WRAP)
                                test::WrapBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT)
                                test::ReflectBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT101)
                                test::Reflect101BorderIndex(coords, size);

                            out = hSrc[db * srcImgPitch + coords.y * srcRowStride + coords.x * srcPixPitch + dk];
                        }
                    }

                    hDst[db * dstImgPitch + di * dstRowStride + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

template<typename T>
static void CopyMakeBorder(std::vector<std::vector<T>> &hBatchDst, const std::vector<std::vector<T>> &hBatchSrc,
                           const std::vector<std::unique_ptr<nvcv::Image>> &dBatchDstData,
                           const std::vector<std::unique_ptr<nvcv::Image>> &dBatchSrcData, const std::vector<int> &top,
                           const std::vector<int> &left, const NVCVBorderType borderType, const float4 borderValue)
{
    int2 coords;
    for (size_t db = 0; db < dBatchDstData.size(); db++)
    {
        auto &hDst         = hBatchDst[db];
        auto &dDst         = dBatchDstData[db];
        auto *imgDstData   = dynamic_cast<const nvcv::IImageDataStridedCuda *>(dDst->exportData());
        int   dstRowStride = imgDstData->plane(0).rowStride / sizeof(T);
        int   dstPixPitch  = dDst->format().numChannels();

        auto &hSrc       = hBatchSrc[db];
        auto &dSrc       = dBatchSrcData[db];
        auto *imgSrcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(dSrc->exportData());
        int   rowStride  = imgSrcData->plane(0).rowStride / sizeof(T);
        int   pixPitch   = imgSrcData->format().numChannels();

        auto imgSize = dBatchSrcData[db]->size();
        int2 size{imgSize.w, imgSize.h};
        for (int di = 0; di < imgDstData->plane(0).height; di++) //for rows
        {
            coords.y = di - top[db];

            for (int dj = 0; dj < imgDstData->plane(0).width; dj++) // for columns
            {
                coords.x = dj - left[db];

                for (int dk = 0; dk < dstPixPitch; dk++)
                {
                    T out = 0;

                    if (coords.x >= 0 && coords.x < size.x && coords.y >= 0 && coords.y < size.y)
                    {
                        out = hSrc[coords.y * rowStride + coords.x * pixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<T>(reinterpret_cast<const float *>(&borderValue)[dk]);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                                test::ReplicateBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_WRAP)
                                test::WrapBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT)
                                test::ReflectBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT101)
                                test::Reflect101BorderIndex(coords, size);

                            out = hSrc[coords.y * rowStride + coords.x * pixPitch + dk];
                        }
                    }

                    hDst[di * dstRowStride + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

template<typename T>
static void CopyMakeBorder(std::vector<T> &hDst, const std::vector<std::vector<T>> &hBatchSrc,
                           const nvcv::TensorDataAccessStridedImagePlanar  &dDstData,
                           const std::vector<std::unique_ptr<nvcv::Image>> &dBatchSrcData, const std::vector<int> &top,
                           const std::vector<int> &left, const NVCVBorderType borderType, const float4 borderValue)
{
    int dstPixPitch  = dDstData.numChannels();
    int dstRowStride = dDstData.rowStride() / sizeof(T);
    int dstImgPitch  = dDstData.sampleStride() / sizeof(T);

    int2 coords;
    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        auto &hSrc       = hBatchSrc[db];
        auto *imgSrcData = dynamic_cast<const nvcv::IImageDataStridedCuda *>(dBatchSrcData[db]->exportData());
        int   rowStride  = imgSrcData->plane(0).rowStride / sizeof(T);
        int   pixPitch   = imgSrcData->format().numChannels();

        auto imgSize = dBatchSrcData[db]->size();
        int2 size{imgSize.w, imgSize.h};
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coords.y = di - top[db];

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coords.x = dj - left[db];

                for (int dk = 0; dk < dDstData.numChannels(); dk++)
                {
                    T out = 0;

                    if (coords.x >= 0 && coords.x < size.x && coords.y >= 0 && coords.y < size.y)
                    {
                        out = hSrc[coords.y * rowStride + coords.x * pixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<T>(reinterpret_cast<const float *>(&borderValue)[dk]);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                                test::ReplicateBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_WRAP)
                                test::WrapBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT)
                                test::ReflectBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT101)
                                test::Reflect101BorderIndex(coords, size);

                            out = hSrc[coords.y * rowStride + coords.x * pixPitch + dk];
                        }
                    }

                    hDst[db * dstImgPitch + di * dstRowStride + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpCopyMakeBorder, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, float, float, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad,         NVCVBorderType,    bValue1, bValue2, bValue3, bValue4, ImageFormat
    {       212,       113,          1,        0,         0,      0,       0,   NVCV_BORDER_CONSTANT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        12,        13,          2,       12,        16,      0,       3,   NVCV_BORDER_CONSTANT,       12.f,   100.f,   245.f,     0.f, nvcv::FMT_RGB8},
    {       212,       113,          3,        0,       113,      5,       0,   NVCV_BORDER_CONSTANT,       13.f,     5.f,     4.f,     0.f, nvcv::FMT_RGB8},
    {       212,       613,          4,       19,        20,      7,       7,   NVCV_BORDER_CONSTANT,      255.f,   255.f,   255.f,     0.f, nvcv::FMT_RGB8},

    {       234,       131,          2,       44,        55,     33,      22,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {       234,       131,          2,       33,        20,     41,      42,    NVCV_BORDER_REFLECT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBA8},
    {       234,       131,          2,      100,        85,     53,      62,       NVCV_BORDER_WRAP,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {       243,       123,          2,       56,       123,     77,      98, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf32},

});

// clang-format on

template<typename T>
void StartTest(int srcWidth, int srcHeight, int numBatches, int topPad, int bottomPad, int leftPad, int rightPad,
               NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int dstWidth  = srcWidth + leftPad + rightPad;
    int dstHeight = srcHeight + topPad + bottomPad;

    std::vector<T> srcVec;

    nvcv::Tensor imgSrc(numBatches, {srcWidth, srcHeight}, format);
    const auto  *srcData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgSrc.exportData());
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcData);
    int srcBufSize = (srcAccess->sampleStride() / sizeof(T)) * srcAccess->numSamples();
    srcVec.resize(srcBufSize);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    if (std::is_same<T, float>::value)
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng) / 255.0f; });
    else
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    // Copy each input image with random data to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec.data(), srcBufSize * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);
    const auto  *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);
    int dstBufSize = (dstAccess->sampleStride() / sizeof(T)) * dstAccess->numSamples();
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(T), stream));

    std::vector<T> testVec(dstBufSize);
    std::vector<T> goldVec(dstBufSize);

    int srcPixPitch  = srcAccess->numChannels();
    int srcRowStride = srcAccess->rowStride() / sizeof(T);
    int srcImgPitch  = srcAccess->sampleStride() / sizeof(T);

    // Generate gold result
    CopyMakeBorder(goldVec, srcVec, *dstAccess, srcWidth, srcHeight, srcRowStride, srcPixPitch, srcImgPitch, topPad,
                   leftPad, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgSrc, imgDst, topPad, leftPad, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize * sizeof(T), cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec);
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}

TEST_P(OpCopyMakeBorder, tensor_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    if (nvcv::FMT_RGB8 == format || nvcv::FMT_RGBA8 == format)
        StartTest<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (nvcv::FMT_RGBf32 == format || nvcv::FMT_RGBAf32 == format)
        StartTest<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType, borderValue,
                         format);
}

template<typename T>
void StartTestVarShape(int srcWidthBase, int srcHeightBase, int numBatches, int topPad, int bottomPad, int leftPad,
                       int rightPad, NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::default_random_engine         randEng{0};
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.2);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.2);

    std::uniform_int_distribution<int> rndTop(topPad * 0.8, topPad * 1.2);
    std::uniform_int_distribution<int> rndBottom(bottomPad * 0.8, bottomPad * 1.2);

    std::uniform_int_distribution<int> rndLeft(leftPad * 0.8, leftPad * 1.2);
    std::uniform_int_distribution<int> rndRight(rightPad * 0.8, rightPad * 1.2);

    //Prepare input and output buffer
    std::vector<std::unique_ptr<nvcv::Image>> imgSrcVec, imgDstVec;
    std::vector<std::vector<T>>               hImgSrcVec, hImgDstVec, batchGoldVec;
    std::vector<int>                          topVec(numBatches);
    std::vector<int>                          leftVec(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        int srcWidth  = rndSrcWidth(randEng);
        int srcHeight = rndSrcHeight(randEng);
        int top       = rndTop(randEng);
        int left      = rndLeft(randEng);
        int bottom    = rndBottom(randEng);
        int right     = rndRight(randEng);

        int dstWidth  = srcWidth + left + right;
        int dstHeight = srcHeight + top + bottom;
        topVec[i]     = top;
        leftVec[i]    = left;
        //prepare input buffers
        imgSrcVec.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{srcWidth, srcHeight}, format));

        auto *imgSrcData   = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrcVec.back()->exportData());
        int   srcStride    = imgSrcData->plane(0).rowStride;
        int   srcRowStride = srcStride / sizeof(T);
        int   srcBufSize   = srcRowStride * imgSrcData->plane(0).height;

        std::vector<T>                         srcVec(srcBufSize);
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        if (std::is_same<T, float>::value)
            std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng) / 255.0f; });
        else
            std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(), srcBufSize * sizeof(T),
                                               cudaMemcpyHostToDevice, stream));
        hImgSrcVec.push_back(std::move(srcVec));

        //prepare output Buffers
        imgDstVec.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{dstWidth, dstHeight}, format));
        auto *imgDstData   = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgDstVec.back()->exportData());
        int   dstStride    = imgDstData->plane(0).rowStride;
        int   dstRowStride = dstStride / sizeof(T);
        int   dstBufSize   = dstRowStride * imgDstData->plane(0).height;

        std::vector<T> dstVec(dstBufSize);
        std::vector<T> goldVec(dstBufSize);
        hImgDstVec.push_back(std::move(dstVec));
        batchGoldVec.push_back(std::move(goldVec));
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(imgSrcVec.begin(), imgSrcVec.end());
    nvcv::ImageBatchVarShape imgBatchDst(numBatches);
    imgBatchDst.pushBack(imgDstVec.begin(), imgDstVec.end());

    // Prepare top and left inputs
    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    const auto *inTopData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTop.exportData());
    const auto *inLeftData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inLeft.exportData());

    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    auto inTopAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inTopData);
    ASSERT_TRUE(inTopAccess);

    auto inLeftAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inLeftData);
    ASSERT_TRUE(inLeftAccess);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    CopyMakeBorder(batchGoldVec, hImgSrcVec, imgDstVec, imgSrcVec, topVec, leftVec, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgBatchSrc, imgBatchDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    int idx = 0;
    for (auto &img : imgBatchDst)
    {
        auto &testVec   = hImgDstVec[idx];
        auto &goldVec   = batchGoldVec[idx];
        auto  imgAccess = dynamic_cast<const nvcv::IImageDataStridedCuda *>(img.exportData());

        ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), imgAccess->plane(0).basePtr, testVec.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
        for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
        test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
        test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
        if (goldVec != testVec)
        {
            test::DebugPrintDiff(testVec, goldVec);
        }
#endif

        EXPECT_EQ(goldVec, testVec);
        idx++;
    }
}

TEST_P(OpCopyMakeBorder, varshape_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    if (nvcv::FMT_RGB8 == format || nvcv::FMT_RGBA8 == format)
        StartTestVarShape<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                   borderValue, format);
    else if (nvcv::FMT_RGBf32 == format || nvcv::FMT_RGBAf32 == format)
        StartTestVarShape<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                 borderValue, format);
}

template<typename T>
void StartTestStack(int srcWidthBase, int srcHeightBase, int numBatches, int topPad, int bottomPad, int leftPad,
                    int rightPad, NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    //make sure the random pad settings did not exceed the limit.
    int dstWidth  = (srcWidthBase + leftPad + rightPad) * 1.2;
    int dstHeight = (srcHeightBase + topPad + bottomPad) * 1.2;

    std::default_random_engine         randEng{0};
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.2);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.2);

    std::uniform_int_distribution<int> rndTop(topPad * 0.8, topPad * 1.2);
    std::uniform_int_distribution<int> rndLeft(leftPad * 0.8, leftPad * 1.2);

    //Prepare input and output buffer
    std::vector<std::unique_ptr<nvcv::Image>> imgSrcVec, imgDstVec;
    std::vector<std::vector<T>>               hImgSrcVec;
    std::vector<int>                          topVec(numBatches);
    std::vector<int>                          leftVec(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        int srcWidth  = rndSrcWidth(randEng);
        int srcHeight = rndSrcHeight(randEng);
        int top       = rndTop(randEng);
        int left      = rndLeft(randEng);

        topVec[i]  = top;
        leftVec[i] = left;
        //prepare input buffers
        imgSrcVec.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{srcWidth, srcHeight}, format));

        auto *imgSrcData   = dynamic_cast<const nvcv::IImageDataStridedCuda *>(imgSrcVec.back()->exportData());
        int   srcStride    = imgSrcData->plane(0).rowStride;
        int   srcRowStride = srcStride / sizeof(T);
        int   srcBufSize   = srcRowStride * imgSrcData->plane(0).height;

        std::vector<T>                         srcVec(srcBufSize);
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        if (std::is_same<T, float>::value)
            std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng) / 255.0f; });
        else
            std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(), srcBufSize * sizeof(T),
                                               cudaMemcpyHostToDevice, stream));
        hImgSrcVec.push_back(std::move(srcVec));
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(imgSrcVec.begin(), imgSrcVec.end());

    //prepare output buffer
    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);
    const auto  *dstData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);
    int dstBufSize = (dstAccess->sampleStride() / sizeof(T)) * dstAccess->numSamples();
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(T), stream));

    std::vector<T> testVec(dstBufSize);
    std::vector<T> goldVec(dstBufSize);

    // Prepare top and left inputs
    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    const auto *inTopData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTop.exportData());
    const auto *inLeftData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inLeft.exportData());

    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    auto inTopAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inTopData);
    ASSERT_TRUE(inTopAccess);

    auto inLeftAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inLeftData);
    ASSERT_TRUE(inLeftAccess);

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    CopyMakeBorder(goldVec, hImgSrcVec, *dstAccess, imgSrcVec, topVec, leftVec, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize * sizeof(T), cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec);
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}

TEST_P(OpCopyMakeBorder, stack_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    if (nvcv::FMT_RGB8 == format || nvcv::FMT_RGBA8 == format)
        StartTestStack<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                borderValue, format);
    else if (nvcv::FMT_RGBf32 == format || nvcv::FMT_RGBAf32 == format)
        StartTestStack<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                              borderValue, format);
}
