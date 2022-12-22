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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpChannelReorder.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/cuda/MathOps.hpp>

namespace test = nvcv::test;

TEST(TestOpChannelReorder, smoke_test_works)
{
    // Let's set up input and output images
    nvcv::Image inImages[2] = {
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8},
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8}
    };

    nvcv::Image outImages[2] = {
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_BGRA8},
        nvcv::Image{nvcv::Size2D{4, 2}, nvcv::FMT_RGBA8}
    };

    nvcv::ImageBatchVarShape in(2), out(2);

    // Create the input and output varshapes
    in.pushBack(inImages[0]);
    in.pushBack(inImages[1]);

    out.pushBack(outImages[0]);
    out.pushBack(outImages[1]);

    // Populate input images
    std::vector<uchar4> inImageValues0;
    auto               &inImageData0 = dynamic_cast<const nvcv::IImageDataStrided &>(*inImages[0].exportData());
    inImageValues0.resize(inImageData0.plane(0).rowStride / sizeof(uchar4) * inImageData0.size().h);
    inImageValues0[0] = {1, 2, 3, 7};
    inImageValues0[1] = {7, 3, 2, 9};
    test::SetTensorFromVector<uchar4>(nvcv::TensorWrapImage(inImages[0]).exportData(), inImageValues0, -1);

    std::vector<uchar4> inImageValues1;
    auto               &inImageData1 = dynamic_cast<const nvcv::IImageDataStrided &>(*inImages[1].exportData());
    inImageValues1.resize(inImageData1.plane(0).rowStride / sizeof(uchar4) * inImageData1.size().h);
    inImageValues1[0] = {3, 2, 1, 4};
    inImageValues1[1] = {1, 3, 10, 28};
    test::SetTensorFromVector<uchar4>(nvcv::TensorWrapImage(inImages[1]).exportData(), inImageValues1, -1);

    // Populate the order tensor
    // clang-format off
    nvcv::Tensor inOrders(
        {
            {2, 4},
            "NC"
        },
        nvcv::TYPE_S32);
    // clang-format on

    auto             &inOrderData = dynamic_cast<const nvcv::ITensorDataStrided &>(*inOrders.exportData());
    std::vector<int4> inOrderValues(inOrderData.stride(0) / sizeof(int4));

    // N==0
    inOrderValues[0] = {2, -1, 1, 3};
    test::SetTensorFromVector<int4>(inOrders.exportData(), inOrderValues, 0);

    // N=1
    inOrderValues[0] = {3, 2, 1, -1};
    test::SetTensorFromVector<int4>(inOrders.exportData(), inOrderValues, 1);

    // Execute operation
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::ChannelReorder chReorder;

    chReorder(stream, in, out, inOrders);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Fetch results
    std::vector<uchar4> outImageValues;

    // image0 x order[0]
    test::GetVectorFromTensor<uchar4>(nvcv::TensorWrapImage(outImages[0]).exportData(), 0, outImageValues);
    EXPECT_EQ(make_uchar4(3, 0, 2, 7), outImageValues[0]);
    EXPECT_EQ(make_uchar4(2, 0, 3, 9), outImageValues[1]);

    // image1 x order[1]
    outImageValues.clear();
    test::GetVectorFromTensor<uchar4>(nvcv::TensorWrapImage(outImages[1]).exportData(), 0, outImageValues);
    EXPECT_EQ(make_uchar4(4, 1, 2, 0), outImageValues[0]);
    EXPECT_EQ(make_uchar4(28, 10, 3, 0), outImageValues[1]);
}
