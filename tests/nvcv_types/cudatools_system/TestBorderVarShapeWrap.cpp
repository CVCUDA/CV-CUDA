/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DeviceBorderVarShapeWrap.hpp" // to test in the device

#include <common/BorderUtils.hpp>               // for test::IsInside, etc.
#include <common/Printers.hpp>                  // for stream operator, etc.
#include <common/TypedTests.hpp>                // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Image.hpp>                       // for Image, etc.
#include <nvcv/ImageBatch.hpp>                  // for ImageBatchVarShape, etc.
#include <nvcv/cuda/BorderVarShapeWrap.hpp>     // the object of this test
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp> // for ImageBatchVarShapeWrap, etc.
#include <nvcv/cuda/MathOps.hpp>                // for operator == to allow EXPECT_EQ

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

// ----------------------- Testing BorderVarShapeWrap --------------------------

// Shortcuts to easy write each test case

constexpr auto S16     = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16   = NVCV_IMAGE_FORMAT_2S16;
constexpr auto RGB8    = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8   = NVCV_IMAGE_FORMAT_RGBA8;
constexpr auto RGBf32  = NVCV_IMAGE_FORMAT_RGBf32;
constexpr auto RGBAf32 = NVCV_IMAGE_FORMAT_RGBAf32;

#define NVCV_TEST_ROW(WIDTH, HEIGHT, VARSIZE, SAMPLES, BORDERSIZE, FORMAT, VALUETYPE, BORDERTYPE)         \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<VARSIZE>, ttype::Value<SAMPLES>, \
                 ttype::Value<BORDERSIZE>, ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>>

NVCV_TYPED_TEST_SUITE(BorderVarShapeWrapTest,
                      ttype::Types<NVCV_TEST_ROW(22, 33, 0, 1, 0, RGBA8, uchar4, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(33, 22, 2, 3, 3, _2S16, short2, NVCV_BORDER_CONSTANT),
                                   NVCV_TEST_ROW(11, 44, 3, 5, 55, RGBAf32, float4, NVCV_BORDER_REPLICATE),
                                   NVCV_TEST_ROW(122, 66, 7, 9, 7, RGBf32, float3, NVCV_BORDER_WRAP),
                                   NVCV_TEST_ROW(66, 163, 12, 6, 9, RGB8, uchar3, NVCV_BORDER_REFLECT),
                                   NVCV_TEST_ROW(199, 99, 23, 4, 19, S16, short1, NVCV_BORDER_REFLECT101)>);

#undef NVCV_TEST_ROW

TYPED_TEST(BorderVarShapeWrapTest, correct_fill)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width      = ttype::GetValue<TypeParam, 0>;
    int height     = ttype::GetValue<TypeParam, 1>;
    int varSize    = ttype::GetValue<TypeParam, 2>;
    int samples    = ttype::GetValue<TypeParam, 3>;
    int borderSize = ttype::GetValue<TypeParam, 4>;

    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 5>};

    using ValueType            = ttype::GetType<TypeParam, 6>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 7>;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    int2 bSize{borderSize, borderSize};

    nvcv::ImageBatchVarShape srcImageBatch(samples);
    nvcv::ImageBatchVarShape dstImageBatch(samples);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<int>     randSize{-varSize, varSize};
    std::uniform_int_distribution<uint8_t> randValues{0, 255};

    std::vector<nvcv::Image>          srcImageList;
    std::vector<std::vector<uint8_t>> srcVec(samples);

    for (int i = 0; i < srcImageBatch.capacity(); ++i)
    {
        srcImageList.emplace_back(nvcv::Size2D{width + randSize(randEng), height + randSize(randEng)}, format);

        auto srcData = srcImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcData, nvcv::NullOpt);

        int srcRowStride = srcData->plane(0).rowStride;
        int srcHeight    = srcImageList[i].size().h;

        srcVec[i].resize(srcHeight * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return randValues(randEng); });

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(srcData->plane(0).basePtr, srcRowStride, srcVec[i].data(), srcRowStride,
                                    srcRowStride, srcHeight, cudaMemcpyHostToDevice, stream));
    }

    srcImageBatch.pushBack(srcImageList.begin(), srcImageList.end());

    std::vector<nvcv::Image> dstImageList;

    for (int i = 0; i < samples; ++i)
    {
        nvcv::Size2D size = srcImageList[i].size();

        dstImageList.emplace_back(nvcv::Size2D{size.w + borderSize * 2, size.h + borderSize * 2},
                                  srcImageList[i].format());
    }

    dstImageBatch.pushBack(dstImageList.begin(), dstImageList.end());

    auto srcImageBatchData = srcImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(srcImageBatchData, nullptr);

    auto dstImageBatchData = dstImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(dstImageBatchData, nullptr);

    int3 dstMaxSize{dstImageBatchData->maxSize().w, dstImageBatchData->maxSize().h, dstImageBatchData->numImages()};

    cuda::BorderVarShapeWrap<const ValueType, kBorderType> srcWrap(*srcImageBatchData, borderValue);
    cuda::ImageBatchVarShapeWrap<ValueType>                dstWrap(*dstImageBatchData);

    DeviceRunFillBorderVarShape(dstWrap, srcWrap, dstMaxSize, bSize, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = srcImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = dstImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        nvcv::Size2D srcSize = srcImageList[i].size();
        nvcv::Size2D dstSize = dstImageList[i].size();

        int srcRowStride = srcData->plane(0).rowStride;
        int dstRowStride = dstData->plane(0).rowStride;

        std::vector<uint8_t> testVec(dstSize.h * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstRowStride,
                                            dstRowStride, dstSize.h, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstSize.h * dstRowStride);

        // Run gold fill border
        int2 srcCoord;

        for (int y = 0; y < dstSize.h; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.w; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {srcSize.w, srcSize.h}, kBorderType);

                *reinterpret_cast<ValueType *>(&goldVec[y * dstRowStride + x * sizeof(ValueType)])
                    = isInside ? *reinterpret_cast<ValueType *>(
                          &srcVec[i][srcCoord.y * srcRowStride + srcCoord.x * sizeof(ValueType)])
                               : borderValue;
            }
        }

        EXPECT_EQ(testVec, goldVec);
    }
}

// ----------------------- Testing BorderVarShapeWrapNHWC --------------------------

#define NVCV_TEST_ROW(WIDTH, HEIGHT, VARSIZE, SAMPLES, BORDERSIZE, FORMAT, VALUETYPE, BORDERTYPE, NUMCHANNELS) \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<VARSIZE>, ttype::Value<SAMPLES>,      \
                 ttype::Value<BORDERSIZE>, ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>,          \
                 ttype::Value<NUMCHANNELS>>

NVCV_TYPED_TEST_SUITE(BorderVarShapeWrapNHWCTest,
                      ttype::Types<NVCV_TEST_ROW(22, 33, 0, 1, 0, RGBA8, uchar1, NVCV_BORDER_CONSTANT, 3),
                                   NVCV_TEST_ROW(33, 22, 2, 3, 3, _2S16, short1, NVCV_BORDER_CONSTANT, 2),
                                   NVCV_TEST_ROW(11, 44, 3, 5, 55, RGBAf32, float1, NVCV_BORDER_REPLICATE, 4),
                                   NVCV_TEST_ROW(122, 66, 7, 9, 7, RGBf32, float1, NVCV_BORDER_WRAP, 3),
                                   NVCV_TEST_ROW(66, 163, 12, 6, 9, RGB8, uchar1, NVCV_BORDER_REFLECT, 3),
                                   NVCV_TEST_ROW(199, 99, 23, 4, 19, S16, short1, NVCV_BORDER_REFLECT101, 1)>);

#undef NVCV_TEST_ROW

TYPED_TEST(BorderVarShapeWrapNHWCTest, correct_fill)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width      = ttype::GetValue<TypeParam, 0>;
    int               height     = ttype::GetValue<TypeParam, 1>;
    int               varSize    = ttype::GetValue<TypeParam, 2>;
    int               samples    = ttype::GetValue<TypeParam, 3>;
    int               borderSize = ttype::GetValue<TypeParam, 4>;
    nvcv::ImageFormat format{ttype::GetValue<TypeParam, 5>};
    using ValueType            = ttype::GetType<TypeParam, 6>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 7>;
    int            numChannels = ttype::GetValue<TypeParam, 8>;

    ValueType borderValue = cuda::SetAll<ValueType>(123);

    int2 bSize{borderSize, borderSize};

    nvcv::ImageBatchVarShape srcImageBatch(samples);
    nvcv::ImageBatchVarShape dstImageBatch(samples);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<int>     randSize{-varSize, varSize};
    std::uniform_int_distribution<uint8_t> randValues{0, 255};

    std::vector<nvcv::Image>          srcImageList;
    std::vector<std::vector<uint8_t>> srcVec(samples);

    for (int i = 0; i < srcImageBatch.capacity(); ++i)
    {
        srcImageList.emplace_back(nvcv::Size2D{width + randSize(randEng), height + randSize(randEng)}, format);

        auto srcData = srcImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcData, nvcv::NullOpt);

        int srcRowStride = srcData->plane(0).rowStride;
        int srcHeight    = srcImageList[i].size().h;

        srcVec[i].resize(srcHeight * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return randValues(randEng); });

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(srcData->plane(0).basePtr, srcRowStride, srcVec[i].data(), srcRowStride,
                                    srcRowStride, srcHeight, cudaMemcpyHostToDevice, stream));
    }

    srcImageBatch.pushBack(srcImageList.begin(), srcImageList.end());

    std::vector<nvcv::Image> dstImageList;

    for (int i = 0; i < samples; ++i)
    {
        nvcv::Size2D size = srcImageList[i].size();

        dstImageList.emplace_back(nvcv::Size2D{size.w + borderSize * 2, size.h + borderSize * 2},
                                  srcImageList[i].format());
    }

    dstImageBatch.pushBack(dstImageList.begin(), dstImageList.end());

    auto srcImageBatchData = srcImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(srcImageBatchData, nullptr);

    auto dstImageBatchData = dstImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(dstImageBatchData, nullptr);

    int3 dstMaxSize{dstImageBatchData->maxSize().w, dstImageBatchData->maxSize().h, dstImageBatchData->numImages()};

    cuda::BorderVarShapeWrapNHWC<const ValueType, kBorderType> srcWrap(*srcImageBatchData, numChannels, borderValue);
    cuda::ImageBatchVarShapeWrapNHWC<ValueType>                dstWrap(*dstImageBatchData, numChannels);

    DeviceRunFillBorderVarShapeNHWC(dstWrap, srcWrap, dstMaxSize, bSize, numChannels, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < samples; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = srcImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = dstImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        nvcv::Size2D srcSize = srcImageList[i].size();
        nvcv::Size2D dstSize = dstImageList[i].size();

        int srcRowStride = srcData->plane(0).rowStride;
        int dstRowStride = dstData->plane(0).rowStride;

        std::vector<uint8_t> testVec(dstSize.h * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstRowStride,
                                            dstRowStride, dstSize.h, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstSize.h * dstRowStride);

        // Run gold fill border
        int2 srcCoord;

        for (int y = 0; y < dstSize.h; ++y)
        {
            srcCoord.y = y - borderSize;

            for (int x = 0; x < dstSize.w; ++x)
            {
                srcCoord.x = x - borderSize;

                bool isInside = test::IsInside(srcCoord, {srcSize.w, srcSize.h}, kBorderType);
                for (int c = 0; c < numChannels; ++c)
                {
                    *reinterpret_cast<ValueType *>(
                        &goldVec[y * dstRowStride + x * sizeof(ValueType) * numChannels + c * sizeof(ValueType)])
                        = isInside ? *reinterpret_cast<ValueType *>(
                              &srcVec[i][srcCoord.y * srcRowStride + srcCoord.x * sizeof(ValueType) * numChannels
                                         + c * sizeof(ValueType)])
                                   : borderValue;
                }
            }
        }

        EXPECT_EQ(testVec, goldVec);
    }
}
