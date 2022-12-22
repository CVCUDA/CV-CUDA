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
#include <cvcuda/OpCenterCrop.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>

#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

//#define DBG_CROP_RECT

#ifdef DBG_CROP_RECT
static void dbgImage(std::vector<uint8_t> &in, int rowStride)
{
    std::cout << "\n IMG -- " << rowStride << " in " << in.size() << "\n";
    for (size_t i = 0; i < in.size(); i++)
    {
        if (i % rowStride == 0)
            std::cout << "\n";
        printf("%02x,", in[i]);
    }
}
#endif

// Width is in bytes or pixels..
static void WriteData(const nvcv::TensorDataAccessStridedImagePlanar &data, uint8_t val, int crop_rows,
                      int crop_columns)
{
    EXPECT_EQ(NVCV_TENSOR_NHWC, data.layout());
    EXPECT_LE(crop_columns, data.numCols());
    EXPECT_LE(crop_rows, data.numRows());

    int      bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int      bytesPerPixel = data.numChannels() * bytesPerChan;
    uint8_t *impPtrTop     = (uint8_t *)data.sampleData(0);
    uint8_t *impPtr        = nullptr;
    int      numImages     = data.numSamples();
    int      rowStride     = data.rowStride();

    EXPECT_NE(nullptr, impPtrTop);
    int top_indices  = (data.numRows() - crop_rows) / 2;
    int left_indices = (data.numCols() - crop_columns) / 2;

    for (int img = 0; img < numImages; img++)
    {
        impPtr = impPtrTop + (data.sampleStride() * img) + (left_indices * bytesPerPixel) + (top_indices * rowStride);
        EXPECT_EQ(cudaSuccess, cudaMemset2D((void *)impPtr, rowStride, val, crop_columns * bytesPerPixel, crop_rows));
    }
}

static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
                          int crop_rows, int crop_columns, uint8_t val)
{
    int bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int bytesPerPixel = data.numChannels() * bytesPerChan;
    //int top_indices = (data.numRows() - crop_rows) / 2;
    //int left_indices = (data.numCols() - crop_columns) / 2;

    uint8_t *ptrTop = vect.data();
    for (int img = 0; img < data.numSamples(); img++)
    {
        uint8_t *ptr = ptrTop + data.sampleStride() * img;
        for (int i = 0; i < crop_rows; i++)
        {
            memset(ptr, val, crop_columns * bytesPerPixel);
            ptr += data.rowStride();
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpCenterCrop, test::ValueList<int, int, int, int, int>
{
    //width, height, crop_columns, crop_rows, numberImages
    {     3,      3,            3,         3,            1},
    {     3,      3,            3,         1,            1},
    {     3,      3,            1,         3,            1},
    {     3,      3,            1,         1,            1},

    //width, height, crop_columns, crop_rows, numberImages
    {     5,      5,            5,         5,            1},
    {     5,      5,            5,         3,            1},
    {     5,      5,            3,         5,            1},
    {     5,      5,            3,         3,            1},

    //width, height, crop_columns, crop_rows, numberImages
    {     5,      7,            5,         7,            1},
    {     5,      7,            5,         3,            1},
    {     5,      7,            3,         7,            1},
    {     5,      7,            3,         3,            1},

    //width, height, crop_columns, crop_rows, numberImages
    {     5,      5,            5,         5,            5},
    {     5,      5,            5,         3,            5},
    {     5,      5,            3,         5,            5},
    {     5,      5,            3,         3,            5},

    //width, height, crop_columns, crop_rows, numberImages
    {     5,      5,            5,         5,            2},
    {     5,      5,            5,         3,            2},
    {     5,      5,            3,         5,            2},
    {     5,      5,            3,         3,            2},

    //width, height, crop_columns, crop_rows, numberImages
    {     5,      7,            5,         7,            2},
    {     5,      7,            5,         3,            2},
    {     5,      7,            3,         7,            2},
    {     5,      7,            3,         3,            2},

});

// clang-format on
TEST_P(OpCenterCrop, CenterCrop_packed)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int     inWidth        = GetParamValue<0>();
    int     inHeight       = GetParamValue<1>();
    int     crop_columns   = GetParamValue<2>();
    int     crop_rows      = GetParamValue<3>();
    int     numberOfImages = GetParamValue<4>();
    uint8_t cropVal        = 0x56;

    nvcv::Tensor imgOut(numberOfImages, {inWidth, inHeight}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgIn(numberOfImages, {inWidth, inHeight}, nvcv::FMT_RGBA8);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(imgOut.exportData());

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int inBufSize
        = inAccess->sampleStride() * inAccess->numSamples(); //img pitch bytes can be more than the image 64, 128, etc
    int outBufSize = outAccess->sampleStride() * outAccess->numSamples();

    EXPECT_EQ(cudaSuccess, cudaMemset(inData->basePtr(), 0x00, inAccess->sampleStride() * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0x00, outAccess->sampleStride() * outAccess->numSamples()));
    WriteData(*inAccess, cropVal, crop_rows, crop_columns); // write data to be cropped

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, *outAccess, crop_rows, crop_columns, cropVal);

    // run operator
    cvcuda::CenterCrop cropOp;

    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, {crop_columns, crop_rows}));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), inData->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

#ifdef DBG_CROP_RECT
    dbgImage(testIn, inData->rowStride());
    dbgImage(test, outData->rowStride());
    dbgImage(gold, outData->rowStride());
#endif
    EXPECT_EQ(gold, test);
}
