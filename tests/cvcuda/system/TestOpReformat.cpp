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

#include <common/TypedTests.hpp>
#include <cvcuda/OpReformat.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <iostream>
#include <random>

namespace test  = nvcv::test;
namespace cuda  = nvcv::cuda;
namespace ttype = nvcv::test::type;

using uchar = unsigned char;

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long4 pitches, int b, int y, int x, int c, nvcv::TensorLayout layout)
{
    if (layout == nvcv::TensorLayout::NHWC || layout == nvcv::TensorLayout::HWC)
    {
        return *reinterpret_cast<T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z + c * pitches.w]);
    }
    else if (layout == nvcv::TensorLayout::NCHW || layout == nvcv::TensorLayout::CHW)
    {
        return *reinterpret_cast<T *>(&vec[b * pitches.x + c * pitches.y + y * pitches.z + x * pitches.w]);
    }
    return *reinterpret_cast<T *>(&vec[0]);
}

template<typename T>
inline void Reformat(std::vector<uint8_t> &hDst, long4 dstStrides, nvcv::TensorLayout dstLayout,
                     std::vector<uint8_t> &hSrc, long4 srcStrides, nvcv::TensorLayout srcLayout, int numBatches,
                     int numRows, int numCols, int numChannels)
{
    for (int b = 0; b < numBatches; ++b)
    {
        for (int y = 0; y < numRows; ++y)
        {
            for (int x = 0; x < numCols; ++x)
            {
                for (int c = 0; c < numChannels; ++c)
                {
                    ValueAt<T>(hDst, dstStrides, b, y, x, c, dstLayout)
                        = ValueAt<T>(hSrc, srcStrides, b, y, x, c, srcLayout);
                }
            }
        }
    }
}

#define NVCV_TEST_ROW(WIDTH, HEIGHT, BATCHES, INFORMAT, OUTFORMAT, VALUETYPE)                              \
    ttype::Types<ttype::Value<WIDTH>, ttype::Value<HEIGHT>, ttype::Value<BATCHES>, ttype::Value<INFORMAT>, \
                 ttype::Value<OUTFORMAT>, VALUETYPE>

NVCV_TYPED_TEST_SUITE(
    OpReformat, ttype::Types<NVCV_TEST_ROW(176, 113, 1, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8, uchar),
                             NVCV_TEST_ROW(23, 43, 23, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_RGB8p, uchar),
                             NVCV_TEST_ROW(7, 4, 7, NVCV_IMAGE_FORMAT_RGBf32, NVCV_IMAGE_FORMAT_RGBf32, float),
                             NVCV_TEST_ROW(56, 49, 2, NVCV_IMAGE_FORMAT_RGBA8p, NVCV_IMAGE_FORMAT_RGBA8, uchar),
                             NVCV_TEST_ROW(56, 49, 3, NVCV_IMAGE_FORMAT_RGB8, NVCV_IMAGE_FORMAT_RGB8p, uchar),
                             NVCV_TEST_ROW(31, 30, 3, NVCV_IMAGE_FORMAT_RGBAf32, NVCV_IMAGE_FORMAT_RGBAf32p, float),
                             NVCV_TEST_ROW(30, 31, 3, NVCV_IMAGE_FORMAT_RGBf32p, NVCV_IMAGE_FORMAT_RGBf32, float)>);

#undef NVCV_TEST_ROW

TYPED_TEST(OpReformat, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = ttype::GetValue<TypeParam, 0>;
    int height  = ttype::GetValue<TypeParam, 1>;
    int batches = ttype::GetValue<TypeParam, 2>;

    nvcv::ImageFormat inFormat{ttype::GetValue<TypeParam, 3>};
    nvcv::ImageFormat outFormat{ttype::GetValue<TypeParam, 4>};

    using ValueType = ttype::GetType<TypeParam, 5>;

    nvcv::Tensor inTensor(batches, {width, height}, inFormat);
    nvcv::Tensor outTensor(batches, {width, height}, outFormat);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(inTensor.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(outTensor.exportData());

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    long4 inStrides{inData->stride(0), inData->stride(1), inData->stride(2), inData->stride(3)};
    long4 outStrides{outData->stride(0), outData->stride(1), outData->stride(2), outData->stride(3)};

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    int numBatches  = inAccess->numSamples();
    int numRows     = inAccess->numRows();
    int numCols     = inAccess->numCols();
    int numChannels = inAccess->numChannels();

    long inBufSize  = inStrides.x * inData->shape(0);
    long outBufSize = outStrides.x * outData->shape(0);

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::Reformat reformatOp;

    EXPECT_NO_THROW(reformatOp(stream, inTensor, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    Reformat<ValueType>(goldVec, outStrides, outData->layout(), inVec, inStrides, inData->layout(), numBatches, numRows,
                        numCols, numChannels);

    EXPECT_EQ(testVec, goldVec);
}
