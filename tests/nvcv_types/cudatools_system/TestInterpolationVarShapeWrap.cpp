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

#include "DeviceInterpolationVarShapeWrap.hpp" // to test in the device

#include <common/InterpUtils.hpp>                  // for test::GoldInterp, etc.
#include <common/Printers.hpp>                     // for stream operator, etc.
#include <common/TypedTests.hpp>                   // for NVCV_TYPED_TEST_SUITE, etc.
#include <nvcv/Image.hpp>                          // for Image, etc.
#include <nvcv/ImageBatch.hpp>                     // for ImageBatchVarShape, etc.
#include <nvcv/cuda/BorderVarShapeWrap.hpp>        // for ImageBatchVarShapeWrap, etc.
#include <nvcv/cuda/ImageBatchVarShapeWrap.hpp>    // for ImageBatchVarShapeWrap, etc.
#include <nvcv/cuda/InterpolationVarShapeWrap.hpp> // the object of this test
#include <nvcv/cuda/MathOps.hpp>                   // for operator == to allow EXPECT_EQ

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

namespace cuda  = nvcv::cuda;
namespace test  = nvcv::test;
namespace ttype = nvcv::test::type;

// -------------------- Testing InterpolationVarShapeWrap ----------------------

// Shortcuts to easy write each test case

constexpr auto Y8    = NVCV_IMAGE_FORMAT_Y8;
constexpr auto U8    = NVCV_IMAGE_FORMAT_U8;
constexpr auto S16   = NVCV_IMAGE_FORMAT_S16;
constexpr auto _2S16 = NVCV_IMAGE_FORMAT_2S16;
constexpr auto RGB8  = NVCV_IMAGE_FORMAT_RGB8;
constexpr auto RGBA8 = NVCV_IMAGE_FORMAT_RGBA8;

#define NVCV_TEST_ROW(WIDTH, HEIGHT, VARSIZE, BATCHES, SHIFTX, SHIFTY, SCALEX, SCALEY, FORMAT, VALUETYPE, BORDERTYPE, \
                      INTERPTYPE)                                                                                     \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<VARSIZE>, ttype::Value<BATCHES>,             \
                 ttype::Value<SHIFTX>, ttype::Value<SHIFTY>, ttype::Value<SCALEX>, ttype::Value<SCALEY>,              \
                 ttype::Value<FORMAT>, VALUETYPE, ttype::Value<BORDERTYPE>, ttype::Value<INTERPTYPE>>

NVCV_TYPED_TEST_SUITE(
    InterpolationVarShapeWrapTest,
    ttype::Types<
        NVCV_TEST_ROW(13, 18, 0, 1, 0.f, 0.f, 0.f, 0.f, Y8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST),

        NVCV_TEST_ROW(23, 14, 2, 2, 1.f, 1.f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(22, 13, 1, 2, 1.f, 1.f, 0.f, 0.f, S16, short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(31, 23, 4, 2, 3.f, 3.f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_REFLECT, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(23, 32, 2, 2, 1.f, 2.f, 0.f, 0.f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_NEAREST),
        NVCV_TEST_ROW(18, 17, 3, 3, 2.f, 1.f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_NEAREST),

        NVCV_TEST_ROW(13, 17, 3, 2, 13.5f, 17.5f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(13, 42, 4, 2, 3.f, 4.f, 0.f, 0.f, S16, short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(13, 12, 5, 3, 4.25f, 4.25f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_REFLECT, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(16, 25, 4, 2, 4.5f, 3.5f, 0.f, 0.f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_LINEAR),
        NVCV_TEST_ROW(11, 12, 3, 2, 11.f, 12.f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_LINEAR),

        NVCV_TEST_ROW(14, 18, 3, 3, 6.f, 7.f, 0.f, 0.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(12, 42, 4, 2, 12.f, 9.f, 0.f, 0.f, S16, short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(14, 15, 5, 3, 4.75f, 4.75f, 0.f, 0.f, _2S16, short2, NVCV_BORDER_REFLECT, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(17, 26, 4, 2, 2.5f, 3.5f, 0.f, 0.f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_CUBIC),
        NVCV_TEST_ROW(18, 19, 3, 2, 1.25f, 2.f, 0.f, 0.f, RGBA8, uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_CUBIC),

        NVCV_TEST_ROW(27, 13, 2, 2, 1.5f, 1.5f, 1.f, 1.f, U8, uchar1, NVCV_BORDER_CONSTANT, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(28, 10, 2, 3, 1.25f, 2.25f, 1.f, 2.f, S16, short1, NVCV_BORDER_REPLICATE, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(29, 33, 3, 2, 3.75f, 1.5f, 2.f, 1.f, _2S16, short2, NVCV_BORDER_REFLECT, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(18, 17, 2, 3, 4.f, 6.f, 2.5f, 1.5f, RGB8, uchar3, NVCV_BORDER_WRAP, NVCV_INTERP_AREA),
        NVCV_TEST_ROW(31, 23, 2, 2, 7.f, 9.f, 1.25f, 2.75f, RGBA8, uchar4, NVCV_BORDER_REFLECT101, NVCV_INTERP_AREA)>);

#undef NVCV_TEST_ROW

TYPED_TEST(InterpolationVarShapeWrapTest, correct_shift)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int   width   = ttype::GetValue<TypeParam, 0>;
    const int   height  = ttype::GetValue<TypeParam, 1>;
    const int   varSize = ttype::GetValue<TypeParam, 2>;
    const int   batches = ttype::GetValue<TypeParam, 3>;
    const float shiftX  = ttype::GetValue<TypeParam, 4>;
    const float shiftY  = ttype::GetValue<TypeParam, 5>;
    const float scaleX  = ttype::GetValue<TypeParam, 6>;
    const float scaleY  = ttype::GetValue<TypeParam, 7>;

    const nvcv::ImageFormat format{ttype::GetValue<TypeParam, 8>};

    using ValueType            = ttype::GetType<TypeParam, 9>;
    constexpr auto kBorderType = ttype::GetValue<TypeParam, 10>;
    constexpr auto kInterpType = ttype::GetValue<TypeParam, 11>;

    const ValueType borderValue = cuda::SetAll<ValueType>(123);

    const float2 shift{shiftX, shiftY};
    const float2 scale{scaleX, scaleY};

    nvcv::ImageBatchVarShape srcImageBatch(batches);
    nvcv::ImageBatchVarShape dstImageBatch(batches);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<int>     randSize{-varSize, varSize};
    std::uniform_int_distribution<uint8_t> randValues{0, 255};

    std::vector<nvcv::Image>          srcImageList;
    std::vector<std::vector<uint8_t>> srcVec(batches);

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

    for (int i = 0; i < dstImageBatch.capacity(); ++i)
    {
        dstImageList.emplace_back(srcImageList[i].size(), srcImageList[i].format());

        auto dstData = dstImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(dstData, nvcv::NullOpt);

        int dstRowStride = dstData->plane(0).rowStride;
        int dstHeight    = dstImageList[i].size().h;

        std::vector<uint8_t> dstVec(dstHeight * dstRowStride, 0);

        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(dstData->plane(0).basePtr, dstRowStride, dstVec.data(), dstRowStride,
                                                 dstRowStride, dstHeight, cudaMemcpyHostToDevice, stream));
    }

    dstImageBatch.pushBack(dstImageList.begin(), dstImageList.end());

    auto srcImageBatchData = srcImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(srcImageBatchData, nullptr);

    auto dstImageBatchData = dstImageBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);
    ASSERT_NE(dstImageBatchData, nullptr);

    cuda::InterpolationVarShapeWrap<const ValueType, kBorderType, kInterpType> srcWrap(*srcImageBatchData, borderValue,
                                                                                       scaleX, scaleY);
    cuda::ImageBatchVarShapeWrap<ValueType>                                    dstWrap(*dstImageBatchData);

    int3 dstMaxSize{dstImageBatchData->maxSize().w, dstImageBatchData->maxSize().h, dstImageBatchData->numImages()};

    DeviceRunInterpVarShapeShift(dstWrap, srcWrap, dstMaxSize, shift, stream);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = srcImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = dstImageList[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int2 srcSize = int2{srcImageList[i].size().w, srcImageList[i].size().h};
        int2 dstSize = int2{dstImageList[i].size().w, dstImageList[i].size().h};
        ASSERT_EQ(srcSize, dstSize);

        int srcRowStride = srcData->plane(0).rowStride;
        int dstRowStride = dstData->plane(0).rowStride;

        long2 srcStrides = long2{srcRowStride, sizeof(ValueType)};
        long2 dstStrides = long2{dstRowStride, sizeof(ValueType)};

        std::vector<uint8_t> testVec(dstSize.y * dstRowStride, 0);
        std::vector<uint8_t> goldVec(dstSize.y * dstRowStride, 0);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstRowStride,
                                            dstRowStride, dstSize.y, cudaMemcpyDeviceToHost));

        // Run gold interpolation shift
        float2 srcCoord = {};
        int2   dstCoord = {};

        for (dstCoord.y = 0; dstCoord.y < dstSize.y; ++dstCoord.y)
        {
            srcCoord.y = dstCoord.y + shiftY;

            for (dstCoord.x = 0; dstCoord.x < dstSize.x; ++dstCoord.x)
            {
                srcCoord.x = dstCoord.x + shiftX;

                test::ValueAt<ValueType>(goldVec, dstStrides, dstCoord) = test::GoldInterp<kInterpType, kBorderType>(
                    srcVec[i], srcStrides, srcSize, borderValue, scale, srcCoord);
            }
        }

        VEC_EXPECT_NEAR(testVec, goldVec, 1);
    }
}
