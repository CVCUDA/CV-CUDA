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

#include "Definitions.hpp"

#include <common/TypedTests.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpFindContours.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

namespace gt    = ::testing;
namespace test  = nvcv::test;
namespace ttest = test::type;

using CPUImage = std::vector<uint8_t>;

// clang-format off

using Types = ttest::Concat<
    // ttest::Combine<ttest::Zip<ttest::Values<32, 64>,
    //                           ttest::Values<32, 64>>,
    //                ttest::Values<1, 2, 4, 8, 16>>,
    // ttest::Combine<ttest::Zip<ttest::Values<128, 256>,
    //                           ttest::Values<128, 256>>,
    //                ttest::Values<1, 2, 4>>,
    // ttest::Combine<ttest::Zip<ttest::Values<512>,
    //                           ttest::Values<512>>,
    //                ttest::Values<1, 2>>,
    // ttest::Combine<ttest::Zip<ttest::Values<1024>,
    //                           ttest::Values<1024>>,
    //                ttest::Values<1>>
    ttest::Combine<ttest::Zip<ttest::Values<32, 64, 128, 256, 512, 1024>,
                              ttest::Values<32, 64, 128, 256, 512, 1024>>,
                   ttest::Values<1, 2, 4, 8, 16, 32, 64, 128>>,
    ttest::Combine<ttest::Zip<ttest::Values<1920, 3840>,
                              ttest::Values<1080, 2160>>,
                   ttest::Values<1, 2, 4, 8>>,
    ttest::Combine<ttest::Zip<ttest::Values<7680>,
                              ttest::Values<4320>>,
                   ttest::Values<1, 2>>
>;
NVCV_TYPED_TEST_SUITE(OpFindContours, Types);

void generateRectangle(CPUImage &image, nvcv::Size2D boundary, nvcv::Size2D anchor = {0, 0}, nvcv::Size2D size = {5, 5},
                       double angle = 0.0, bool fill = true, uint8_t setValue = 1);

void generateRectangle(CPUImage &image, nvcv::Size2D boundary, nvcv::Size2D anchor, nvcv::Size2D size, double angle,
                       bool fill, uint8_t setValue)
{
    auto rad      = angle * (M_PI / 180.0);
    auto cosAngle = std::cos(rad);
    auto sinAngle = std::sin(rad);

    auto transformed = anchor;
    for (auto y = 0; y < size.h; ++y)
    {
        for (auto x = 0; x < size.w; ++x)
        {
            transformed.w = anchor.w + (x * cosAngle - y * sinAngle);
            transformed.h = anchor.h + (x * sinAngle + y * cosAngle);

            if (fill || y == 0 || y == size.h - 1 || x == 0 || x == size.w - 1)
            {
                if (transformed.w >= 0 && transformed.w < boundary.w && transformed.h >= 0
                    && transformed.h < boundary.h)
                {
                    image[transformed.h * boundary.w + transformed.w] = setValue;
                }
            }
        }
    }
}

TYPED_TEST(OpFindContours, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width          = ttest::GetValue<TypeParam, 0>;
    int height         = ttest::GetValue<TypeParam, 1>;
    int numberOfImages = ttest::GetValue<TypeParam, 2>;

    nvcv::Tensor imgIn         = nvcv::util::CreateTensor(numberOfImages, width, height, nvcv::FMT_U8);
    auto         dtype         = nvcv::TYPE_S32;
    auto         tshape_points = nvcv::TensorShape{
        {numberOfImages, 1024, 2},
        nvcv::TENSOR_NCW
    };
    auto tshape_counts = nvcv::TensorShape{
        {numberOfImages, 4},
        nvcv::TENSOR_NW
    };
    nvcv::Tensor points{tshape_points, dtype};
    nvcv::Tensor counts{tshape_counts, dtype};

    auto inData = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    ASSERT_EQ(numberOfImages, inAccess->numSamples());

    auto imgPtr = make_cudaPitchedPtr(reinterpret_cast<void *>(inAccess->sampleData(0)), inAccess->rowStride(), width, height);
    auto extent = make_cudaExtent(sizeof(uint8_t) * width, height, numberOfImages);
    ASSERT_EQ(cudaSuccess, cudaMemset3DAsync(imgPtr, 0, extent));

    //Generate input
    CPUImage srcVec(height * width, 0);

    // Creating a 16-pixel contour (simple)
    // Head Node at (5, 5)
    generateRectangle(srcVec, {width, height}, {5, 5});

    // Creating a 26-pixel contour (complex)
    // Head Node at (17, 17)
    generateRectangle(srcVec, {width, height}, {17, 17});
    generateRectangle(srcVec, {width, height}, {20, 20});

    // Creating a 12-pixel contour (simple rotated)
    // Head Node at (12, 12)
    generateRectangle(srcVec, {width, height}, {12, 12}, {5, 5}, 45.0);

    for (auto i = 0; i < numberOfImages; ++i)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec.data(), width, width,
                                            height, cudaMemcpyHostToDevice));
    }

    // Creating contour validator
    std::unordered_set<int> expectedSizes{{0, 16, 26, 12}};

    // run operator
    cvcuda::FindContours findContoursOp(nvcv::Size2D{width, height}, numberOfImages);
    EXPECT_NO_THROW(findContoursOp(stream, imgIn, points, counts));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto             outData = counts.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);
    auto outAccess = nvcv::TensorDataAccessStrided::Create(*outData);
    ASSERT_TRUE(outAccess);

    std::vector<int> hcounts(4, 0);
    for (auto i = 0; i < numberOfImages; ++i)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpy(hcounts.data(), outAccess->sampleData(i),
                                          hcounts.size() * sizeof(int), cudaMemcpyDeviceToHost));

        std::unordered_set<int> resultSizes{hcounts.begin(), hcounts.end()};
        EXPECT_EQ(resultSizes, expectedSizes);
    }
}
