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

#include <common/ValueTests.hpp>
#include <cvcuda/OpConvertTo.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/SaturateCast.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

template<typename DT_DEST>
static void setGoldBuffer(std::vector<DT_DEST> &vect, DT_DEST val, int width, int height, int rowStride, int imgStride,
                          int numImages)
{
    for (int img = 0; img < numImages; img++)
    {
        int imgStart = imgStride * img;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                vect[imgStart + x] = val;
            }
            imgStart += rowStride;
        }
    }
}

template<typename DT_SOURCE, typename DT_DEST>
const void testConvertTo(nvcv::ImageFormat fmtIn, nvcv::ImageFormat fmtOut, int batch, int width, int height,
                         double alpha, double beta, DT_SOURCE setVal, DT_DEST expVal)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(batch, {width, height}, fmtOut);
    nvcv::Tensor imgIn(batch, {width, height}, fmtIn);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgOut.exportData());

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int inBufSizeElements  = (inAccess->sampleStride() / sizeof(DT_SOURCE)) * inAccess->numSamples();
    int outBufSizeElements = (outAccess->sampleStride() / sizeof(DT_DEST)) * outAccess->numSamples();
    int inBufSizeBytes     = inAccess->sampleStride() * inAccess->numSamples();
    int outBufSizeBytes    = outAccess->sampleStride() * outAccess->numSamples();

    std::vector<DT_SOURCE> srcVec(inBufSizeElements, setVal);
    std::vector<DT_DEST>   goldVec(outBufSizeElements);
    std::vector<DT_DEST>   testVec(outBufSizeElements);

    setGoldBuffer<DT_DEST>(goldVec, expVal, width * outAccess->numChannels(), height,
                           (outAccess->rowStride() / sizeof(DT_DEST)), (outAccess->sampleStride() / sizeof(DT_DEST)),
                           batch);

    // Copy input data to the GPU
    EXPECT_EQ(cudaSuccess,
              cudaMemcpyAsync(inData->basePtr(), srcVec.data(), inBufSizeBytes, cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0x0, outBufSizeBytes, stream));

    // run operator
    cvcuda::ConvertTo convertToOp;

    EXPECT_NO_THROW(convertToOp(stream, imgIn, imgOut, alpha, beta));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->basePtr(), outBufSizeBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    //dbgImage(goldVec, inData->rowStride());
    //dbgImage(testVec, outData->rowStride());
    EXPECT_EQ(goldVec, testVec);
}

// clang-format off
NVCV_TEST_SUITE_P(OpConvertTo, test::ValueList<int, int, double, double, int>
{
         //   width,     height,       alpha,          beta,   batch
         {       5,        5,          1.0,           1.0,       1 },
         {       5,        5,          2.1,           2.0,       5 },
         {       1,        1,          2.1,           -1.0,      1 }
});

// clang-format on

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBA8)
{
    using toType   = uint8_t;
    using fromType = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBAf32)
{
    using toType   = float;
    using fromType = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBA8)
{
    using toType   = uint8_t;
    using fromType = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBAf32t)
{
    using toType   = float;
    using fromType = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val,
                                    valExp);
}
