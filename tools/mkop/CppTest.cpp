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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <cvcuda/Op__OPNAME__.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

// clang-format off
NVCV_TEST_SUITE_P(Op__OPNAME__, test::ValueList<int, int, NVCVImageFormat, int>
{
    //inWidth, inHeight,                format,   numberInBatch
    {       2,        2,  NVCV_IMAGE_FORMAT_U8,              1},
});

// clang-format on
TEST_P(Op__OPNAME__, __OPNAME___sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int               width  = GetParamValue<0>();
    int               height = GetParamValue<1>();
    nvcv::ImageFormat format{GetParamValue<2>()};
    int               batches = GetParamValue<3>();

    nvcv::Tensor inTensor  = nvcv::util::CreateTensor(batches, width, height, format);
    nvcv::Tensor outTensor = nvcv::util::CreateTensor(batches, width, height, format);

    auto input  = inTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto output = outTensor.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    int inBufSize  = inSampleStride * inAccess->numSamples();
    int outBufSize = outSampleStride * outAccess->numSamples();

    std::vector<uint8_t> inVec(inBufSize);

    std::vector<uint8_t> goldVec(outBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    // TODO populate gold vector with expected results
    std::generate(goldVec.begin(), goldVec.end(), [&]() { return rand(randEng); });
    // populate input with random
    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    cvcuda::__OPNAME__ op;
    EXPECT_NO_THROW(op(stream, inTensor, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy output back to host
    std::vector<uint8_t> testVec(outBufSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    // TODO make test pass
    ASSERT_EQ(goldVec, testVec);
}
