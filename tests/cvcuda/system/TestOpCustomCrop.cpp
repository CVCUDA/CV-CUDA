/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "PlanarParityUtils.hpp"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpCustomCrop.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

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
static void WriteData(const nvcv::TensorDataAccessStridedImagePlanar &data, uint8_t val, NVCVRectI region)
{
    EXPECT_TRUE(data.layout() == NVCV_TENSOR_NHWC || data.layout() == NVCV_TENSOR_HWC);
    EXPECT_LE(region.x + region.width, data.numCols());
    EXPECT_LE(region.y + region.height, data.numRows());

    int        bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int        bytesPerPixel = data.numChannels() * bytesPerChan;
    auto      *impPtrTop     = reinterpret_cast<std::byte *>(data.sampleData(0));
    std::byte *impPtr        = nullptr;
    auto       numImages     = static_cast<int>(data.numSamples());
    auto       rowStride     = static_cast<int>(data.rowStride());

    EXPECT_NE(nullptr, impPtrTop);
    for (int img = 0; img < numImages; img++)
    {
        impPtr = impPtrTop + (data.sampleStride() * img) + (region.x * bytesPerPixel) + (rowStride * region.y);
        EXPECT_EQ(cudaSuccess,
                  cudaMemset2D((void *)impPtr, rowStride, val, region.width * bytesPerPixel, region.height));
    }
}

static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::TensorDataAccessStridedImagePlanar &data,
                          NVCVRectI region, uint8_t val)
{
    int bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int bytesPerPixel = data.numChannels() * bytesPerChan;

    uint8_t *ptrTop    = vect.data();
    auto     numImages = static_cast<int>(data.numSamples());
    for (int img = 0; img < numImages; img++)
    {
        uint8_t *ptr = ptrTop + data.sampleStride() * img;
        for (int i = 0; i < region.height; i++)
        {
            memset(ptr, val, region.width * bytesPerPixel);
            ptr += data.rowStride();
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpCustomCrop, test::ValueList<int, int, int, int, int, int, int, int, int>
{
    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       2,        2,        2,        2,          1,          1,     0,     0,            1},
    {       2,        2,        2,        2,          1,          1,     0,     1,            1},
    {       2,        2,        2,        2,          1,          1,     1,     0,            1},
    {       2,        2,        2,        2,          1,          1,     1,     1,            1},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        2,        2,          2,          2,     0,     0,            1},
    {       5,        5,        2,        2,          2,          2,     0,     1,            1},
    {       5,        5,        2,        2,          2,          2,     1,     0,            1},
    {       5,        5,        2,        2,          2,          2,     1,     1,            1},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        2,        2,          2,          2,     0,     0,            5},
    {       5,        5,        2,        2,          2,          2,     0,     3,            5},
    {       5,        5,        2,        2,          2,          2,     3,     0,            5},
    {       5,        5,        2,        2,          2,          2,     3,     3,            5},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        5,        5,          1,          2,     0,     0,            2},
    {       5,        5,        5,        5,          1,          2,     0,     3,            2},
    {       5,        5,        5,        5,          1,          2,     4,     0,            2},
    {       5,        5,        5,        5,          1,          2,     4,     3,            2},

});

// clang-format on

TEST_P(OpCustomCrop, CustomCrop_packed)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int     inWidth        = GetParamValue<0>();
    int     inHeight       = GetParamValue<1>();
    int     outWidth       = GetParamValue<2>();
    int     outHeight      = GetParamValue<3>();
    int     cropWidth      = GetParamValue<4>();
    int     cropHeight     = GetParamValue<5>();
    int     cropX          = GetParamValue<6>();
    int     cropY          = GetParamValue<7>();
    int     numberOfImages = GetParamValue<8>();
    uint8_t cropVal        = 0x56;

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, outWidth, outHeight, nvcv::FMT_RGBA8);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(numberOfImages, inWidth, inHeight, nvcv::FMT_RGBA8);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    auto inSampleStride  = static_cast<size_t>(inAccess->numRows() * inAccess->rowStride());
    auto outSampleStride = static_cast<size_t>(outAccess->numRows() * outAccess->rowStride());
    auto numSamples      = static_cast<size_t>(inAccess->numSamples());

    size_t inBufSize  = inSampleStride * numSamples;
    size_t outBufSize = outSampleStride * numSamples;

    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    EXPECT_EQ(cudaSuccess, cudaMemset(inData->basePtr(), 0x00, inBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0x00, outBufSize));
    WriteData(*inAccess, cropVal, crpRect); // write data to be cropped

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, *outAccess, crpRect, cropVal);

    // run operator
    cvcuda::CustomCrop cropOp;

    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, crpRect));

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

// Parameters: dtype, channels, input width, input height, crop width, crop height, crop x, crop y, batch
// clang-format off
NVCV_TEST_SUITE_P(OpCustomCropGold,
                  test::ValueList<nvcv::DataType, int, int, int, int, int, int, int, int>{
    // Exercise every byte-width/channel dispatch cell. Odd widths and unaligned origins protect
    // vectorized implementations' body and scalar-tail paths.
    { nvcv::TYPE_U8, 1, 19,  9,  1,  1,  0, 0, 1},
    { nvcv::TYPE_S8, 2, 23, 11,  7,  3,  1, 2, 2},
    { nvcv::TYPE_U8, 3, 41, 13, 17,  5,  3, 1, 3},
    { nvcv::TYPE_S8, 4, 59, 15, 33,  7,  5, 4, 2},

    {nvcv::TYPE_U16, 1, 31, 10,  5,  4,  2, 3, 2},
    {nvcv::TYPE_S16, 2, 38, 12, 16,  6,  1, 1, 1},
    {nvcv::TYPE_F16, 3, 47, 14, 31,  5,  7, 6, 2},
    {nvcv::TYPE_U16, 4, 69, 16, 35,  8,  9, 3, 3},

    {nvcv::TYPE_U32, 1,  5,  5,  2,  2,  0, 0, 1},
    {nvcv::TYPE_S32, 1, 29,  9,  3,  3,  4, 2, 2},
    {nvcv::TYPE_F32, 2, 43, 11, 15,  5,  2, 3, 1},
    {nvcv::TYPE_S32, 3, 61, 13, 32,  7, 11, 2, 3},
    {nvcv::TYPE_F32, 4, 71, 17, 37,  9, 13, 5, 2},

    {nvcv::TYPE_F64, 1, 27,  8,  2,  2,  1, 1, 1},
    {nvcv::TYPE_F64, 2, 45, 12, 13,  4,  5, 4, 2},
    {nvcv::TYPE_F64, 3, 63, 14, 34,  6,  7, 5, 3},
    {nvcv::TYPE_F64, 4, 79, 18, 39, 10, 17, 3, 2},
});

// clang-format on

TEST_P(OpCustomCropGold, tensor_correct_output)
{
    const nvcv::DataType dtype          = GetParamValue<0>();
    const int            channels       = GetParamValue<1>();
    const int            inWidth        = GetParamValue<2>();
    const int            inHeight       = GetParamValue<3>();
    const int            cropWidth      = GetParamValue<4>();
    const int            cropHeight     = GetParamValue<5>();
    const int            cropX          = GetParamValue<6>();
    const int            cropY          = GetParamValue<7>();
    const int            numberOfImages = GetParamValue<8>();
    const NVCVRectI      cropRect{cropX, cropY, cropWidth, cropHeight};

    const nvcv::TensorShape inShape{
        {numberOfImages, inHeight, inWidth, channels},
        "NHWC"
    };
    const nvcv::TensorShape outShape{
        {numberOfImages, cropHeight, cropWidth, channels},
        "NHWC"
    };
    nvcv::Tensor imgIn(inShape, dtype);
    nvcv::Tensor imgOut(outShape, dtype);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess);
    ASSERT_TRUE(outAccess);

    const auto   bytesPerChannel = static_cast<size_t>(dtype.bitsPerChannel()[0] / 8);
    const size_t bytesPerPixel   = static_cast<size_t>(channels) * bytesPerChannel;
    const auto   inRowStride     = static_cast<size_t>(inAccess->rowStride());
    const auto   outRowStride    = static_cast<size_t>(outAccess->rowStride());
    const auto   inSampleStride  = static_cast<size_t>(inAccess->sampleStride());
    const auto   outSampleStride = static_cast<size_t>(outAccess->sampleStride());
    const size_t inBufSize       = inSampleStride * (numberOfImages - 1) + inRowStride * inHeight;
    const size_t outBufSize      = outSampleStride * (numberOfImages - 1) + outRowStride * cropHeight;

    std::vector<uint8_t> src(inBufSize, 0xC3);
    for (int n = 0; n < numberOfImages; ++n)
    {
        for (int y = 0; y < inHeight; ++y)
        {
            for (size_t byte = 0; byte < static_cast<size_t>(inWidth) * bytesPerPixel; ++byte)
            {
                const size_t x           = byte / bytesPerPixel;
                const size_t channelByte = byte % bytesPerPixel;
                const size_t c           = channelByte / bytesPerChannel;
                const size_t b           = channelByte % bytesPerChannel;
                const size_t offset
                    = static_cast<size_t>(n) * inSampleStride + static_cast<size_t>(y) * inRowStride + byte;
                src[offset] = static_cast<uint8_t>((97 * n + 31 * y + 17 * x + 7 * c + 3 * b + 11) % 251);
            }
        }
    }

    // Independent host oracle: copy the addressed byte rectangle row by row. This deliberately does
    // not reuse the device kernel's wrappers, launch geometry, or dispatch table.
    std::vector<uint8_t> gold(outBufSize, 0xA5);
    for (int n = 0; n < numberOfImages; ++n)
    {
        for (int y = 0; y < cropHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(n) * inSampleStride
                                   + static_cast<size_t>(y + cropY) * inRowStride
                                   + static_cast<size_t>(cropX) * bytesPerPixel;
            const size_t dstOffset = static_cast<size_t>(n) * outSampleStride + static_cast<size_t>(y) * outRowStride;
            std::memcpy(gold.data() + dstOffset, src.data() + srcOffset,
                        static_cast<size_t>(cropWidth) * bytesPerPixel);
        }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), src.data(), inBufSize, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0xA5, outBufSize, stream));

    cvcuda::CustomCrop cropOp;
    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, cropRect));

    std::vector<uint8_t> test(outBufSize);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(test.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(gold, test);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Byte-copy coverage: source phases 0..15, body/tail lengths, dtype widths and ROI boundaries.
// Tensor allocation may pad rows: these cases do not assert rotating destination alignment or
// exact allocation ends. Caller-owned tightly bounded views are checked separately.

// Parameters: dtype, channels, inWidth, inHeight, cropWidth, cropHeight, cropX, cropY, batch
// clang-format off
NVCV_TEST_SUITE_P(OpCustomCropAlign,
                  test::ValueList<nvcv::DataType, int, int, int, int, int, int, int, int>{
    // Every source byte phase 0..15 for a 1-byte pixel, on a crop wide enough to hold several
    // whole 16-byte units plus a remainder.
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  0, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  1, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  2, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  3, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  4, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  5, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  6, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  7, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  8, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4,  9, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 10, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 11, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 12, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 13, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 14, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 5, 61, 4, 15, 0, 2},

    // Destination row lengths on both sides of the 16-byte unit boundary.
    { nvcv::TYPE_U8, 1, 96, 5, 15, 3,  7, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5, 16, 3,  7, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5, 17, 3,  7, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5, 31, 3,  5, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5, 32, 3,  5, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5, 33, 3,  5, 1, 1},
    { nvcv::TYPE_U8, 1, 96, 5,  1, 3,  9, 1, 1},

    // Multi-byte pixels verify byte-width dispatch and packed-channel copies.
    { nvcv::TYPE_U8, 3, 71, 8, 37, 5,  3, 2, 2},
    { nvcv::TYPE_U8, 3, 71, 8, 37, 5,  6, 2, 2},
    { nvcv::TYPE_U8, 3, 71, 8, 32, 5, 11, 2, 2},
    { nvcv::TYPE_U8, 4, 67, 6, 33, 5,  7, 1, 2},
    { nvcv::TYPE_U8, 4, 67, 6, 32, 5,  9, 1, 2},
    {nvcv::TYPE_U16, 3, 59, 6, 29, 5,  5, 1, 2},
    {nvcv::TYPE_F16, 1, 83, 6, 41, 5,  7, 1, 2},
    {nvcv::TYPE_F32, 3, 47, 6, 23, 5,  3, 1, 2},
    {nvcv::TYPE_F32, 4, 43, 6, 21, 5,  5, 1, 2},
    {nvcv::TYPE_F64, 3, 37, 6, 17, 5,  3, 1, 2},

    // Crops ending on the final logical source pixel; allocator row padding may follow it.
    { nvcv::TYPE_U8, 1, 96, 4, 81, 4, 15, 0, 2},
    { nvcv::TYPE_U8, 1, 96, 4, 89, 4,  7, 0, 2},
    { nvcv::TYPE_U8, 3, 71, 5, 68, 5,  3, 0, 2},
    {nvcv::TYPE_F32, 1, 51, 4, 44, 4,  7, 0, 2},

    // Full-width crops cover both dense and allocator-padded rows.
    { nvcv::TYPE_U8, 1, 96, 7, 96, 7,  0, 0, 3},
    { nvcv::TYPE_U8, 1, 97, 7, 97, 7,  0, 0, 3},
    { nvcv::TYPE_U8, 3, 71, 7, 71, 7,  0, 0, 3},
    {nvcv::TYPE_F32, 3, 47, 7, 47, 7,  0, 0, 3},
    { nvcv::TYPE_U8, 1, 96, 7, 96, 4,  0, 3, 3},
});

// clang-format on

static void RunCropByteCase(nvcv::DataType dtype, int channels, int inWidth, int inHeight, int cropWidth,
                            int cropHeight, int cropX, int cropY, int numberOfImages, bool planar = false)
{
    const NVCVRectI cropRect{cropX, cropY, cropWidth, cropHeight};

    const auto shape = [=](int width, int height)
    {
        return planar ? nvcv::TensorShape{{numberOfImages, channels, height, width}, "NCHW"}
                      : nvcv::TensorShape{{numberOfImages, height, width, channels}, "NHWC"};
    };
    nvcv::Tensor imgIn(shape(inWidth, inHeight), dtype);
    nvcv::Tensor imgOut(shape(cropWidth, cropHeight), dtype);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess);
    ASSERT_TRUE(outAccess);

    const size_t bytesPerPixel   = static_cast<size_t>(planar ? 1 : channels) * (dtype.bitsPerChannel()[0] / 8);
    const auto   inRowStride     = static_cast<size_t>(inAccess->rowStride());
    const auto   outRowStride    = static_cast<size_t>(outAccess->rowStride());
    const auto   inSampleStride  = static_cast<size_t>(inAccess->sampleStride());
    const auto   outSampleStride = static_cast<size_t>(outAccess->sampleStride());
    const int    planes          = planar ? channels : 1;
    const auto   inPlaneStride   = static_cast<size_t>(inAccess->planeStride());
    const auto   outPlaneStride  = static_cast<size_t>(outAccess->planeStride());
    const size_t inBufSize
        = inSampleStride * (numberOfImages - 1) + (planes - 1) * inPlaneStride + inRowStride * inHeight;
    const size_t outBufSize
        = outSampleStride * (numberOfImages - 1) + (planes - 1) * outPlaneStride + outRowStride * cropHeight;

    std::vector<uint8_t> src(inBufSize);
    for (size_t i = 0; i < inBufSize; ++i)
    {
        src[i] = static_cast<uint8_t>((i * 131u + 17u) % 251u);
    }

    // Independent host oracle: memcpy the addressed byte rectangle row by row. It shares no
    // addressing, launch geometry, or dispatch with the device implementation.
    std::vector<uint8_t> gold(outBufSize, 0x5C);
    for (int n = 0; n < numberOfImages; ++n)
    {
        for (int plane = 0; plane < planes; ++plane)
            for (int y = 0; y < cropHeight; ++y)
            {
                const size_t srcOffset = static_cast<size_t>(n) * inSampleStride + plane * inPlaneStride
                                       + static_cast<size_t>(y + cropY) * inRowStride
                                       + static_cast<size_t>(cropX) * bytesPerPixel;
                const size_t dstOffset = static_cast<size_t>(n) * outSampleStride + plane * outPlaneStride
                                       + static_cast<size_t>(y) * outRowStride;
                std::memcpy(gold.data() + dstOffset, src.data() + srcOffset,
                            static_cast<size_t>(cropWidth) * bytesPerPixel);
            }
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), src.data(), inBufSize, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0x5C, outBufSize, stream));

    cvcuda::CustomCrop cropOp;
    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, cropRect));

    std::vector<uint8_t> test(outBufSize);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(test.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(gold, test);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// A crop that starts on the very first byte of the source and ends on its very last byte, on a
// tensor whose allocation the implementation may not read outside of. Kept separate from the value
// list so it also runs under compute-sanitizer as a single named case.
TEST_P(OpCustomCropAlign, tensor_correct_output)
{
    RunCropByteCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(),
                    GetParamValue<5>(), GetParamValue<6>(), GetParamValue<7>(), GetParamValue<8>());
}

TEST(OpCustomCropAlign, planar_u16_matches_byte_gold)
{
    RunCropByteCase(nvcv::TYPE_U16, 3, 59, 6, 29, 5, 5, 1, 2, true);
}

static void RunWidePixelCropCase(nvcv::DataType dtype, int width, int x, int padding, int srcOffset, int dstOffset)
{
    constexpr int srcW       = 128;
    constexpr int srcH       = 5;
    constexpr int dstH       = 3;
    constexpr int samples    = 2;
    constexpr int pixelBytes = 24;

    SCOPED_TRACE(testing::Message() << dtype << ":" << width << ":" << x << ":" << padding << ":" << srcOffset << ":"
                                    << dstOffset);
    const int64_t srcRow   = srcW * pixelBytes + padding;
    const int64_t dstRow   = ((width * pixelBytes + 15) / 16) * 16 + padding;
    const size_t  srcBytes = srcOffset + samples * srcH * srcRow;
    const size_t  dstBytes = dstOffset + samples * dstH * dstRow;
    NVCVByte     *srcPtr{};
    NVCVByte     *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    const auto wrap = [dtype](NVCVByte *ptr, int offset, int w, int h, int64_t row)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + offset;
        buffer.strides[0] = h * row;
        buffer.strides[1] = row;
        buffer.strides[2] = pixelBytes;
        buffer.strides[3] = 8;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{samples, h, w, 3}, "NHWC"},
            dtype, buffer
        });
    };
    auto                 src = wrap(srcPtr, srcOffset, srcW, srcH, srcRow);
    auto                 dst = wrap(dstPtr, dstOffset, width, dstH, dstRow);
    std::vector<uint8_t> input(srcBytes);
    std::vector<uint8_t> gold(dstBytes, 0x5C);
    std::vector<uint8_t> actual(dstBytes);
    for (size_t i = 0; i < srcBytes; ++i) input[i] = static_cast<uint8_t>((i * 131u + 17u) % 251u);
    // Copy only the CPU-addressed rectangle; compare the offsets and row padding too.
    for (int n = 0; n < samples; ++n)
        for (int y = 0; y < dstH; ++y)
            std::memcpy(gold.data() + dstOffset + (n * dstH + y) * dstRow,
                        input.data() + srcOffset + (n * srcH + y + 1) * srcRow + x * pixelBytes, width * pixelBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0x5C, dstBytes));
    cvcuda::CustomCrop op;
    ASSERT_NO_THROW(op(nullptr, src, dst, NVCVRectI{x, 1, width, dstH}));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(gold, actual);
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

TEST(OpCustomCropAlign, wide_pixel_phases_and_tails)
{
    const auto cases
        = test::Combine(test::ValueList<nvcv::DataType>{nvcv::TYPE_U64, nvcv::TYPE_S64, nvcv::TYPE_F64},
                        test::ValueList<int>{63, 64, 65}, test::ValueList<int>{0, 1}, test::ValueList<int>{0, 8},
                        test::ValueList<int>{0, 8}, test::ValueList<int>{0, 8});
    for (const auto &[dtype, width, x, padding, srcOffset, dstOffset] : cases)
        ASSERT_NO_FATAL_FAILURE(RunWidePixelCropCase(dtype, width, x, padding, srcOffset, dstOffset));
}

TEST(OpCustomCropAlign, tensor_spans_whole_source)
{
    const int       kWidth  = 33;
    const int       kHeight = 5;
    const NVCVRectI cropRect{0, 0, kWidth, kHeight};

    nvcv::Tensor imgIn(
        nvcv::TensorShape{
            {1, kHeight, kWidth, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgOut(
        nvcv::TensorShape{
            {1, kHeight, kWidth, 3},
            "NHWC"
    },
        nvcv::TYPE_U8);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess  = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(inAccess);
    ASSERT_TRUE(outAccess);
    const auto   inRowStride  = static_cast<size_t>(inAccess->rowStride());
    const auto   outRowStride = static_cast<size_t>(outAccess->rowStride());
    const size_t inBufSize    = inRowStride * kHeight;
    const size_t outBufSize   = outRowStride * kHeight;
    const size_t rowBytes     = static_cast<size_t>(kWidth) * 3;

    std::vector<uint8_t> src(inBufSize);
    for (size_t i = 0; i < inBufSize; ++i) src[i] = static_cast<uint8_t>(i * 7u + 3u);

    std::vector<uint8_t> gold(outBufSize, 0x5C);
    for (int y = 0; y < kHeight; ++y)
        std::memcpy(gold.data() + y * outRowStride, src.data() + y * inRowStride, rowBytes);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inData->basePtr(), src.data(), inBufSize, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outData->basePtr(), 0x5C, outBufSize, stream));

    cvcuda::CustomCrop cropOp;
    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, cropRect));

    std::vector<uint8_t> test(outBufSize);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(test.data(), outData->basePtr(), outBufSize, cudaMemcpyDeviceToHost, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(gold, test);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCustomCropCoverage, non_dense_nhwc_single_channel_matches_reference)
{
    constexpr int    kSamples            = 2;
    constexpr int    kInputWidth         = 9;
    constexpr int    kInputHeight        = 5;
    constexpr int    kCropWidth          = 5;
    constexpr int    kCropHeight         = 3;
    constexpr int    kCropX              = 2;
    constexpr int    kCropY              = 1;
    constexpr size_t kInputRowStride     = kInputWidth + 3;
    constexpr size_t kOutputRowStride    = kCropWidth + 2;
    constexpr size_t kInputSampleStride  = kInputRowStride * kInputHeight + 5;
    constexpr size_t kOutputSampleStride = kOutputRowStride * kCropHeight + 7;
    constexpr size_t kInputBytes         = kInputSampleStride * kSamples;
    constexpr size_t kOutputBytes        = kOutputSampleStride * kSamples;

    static_assert(kInputSampleStride != kInputRowStride * kInputHeight);
    static_assert(kOutputSampleStride != kOutputRowStride * kCropHeight);

    NVCVByte *inputAllocation{};
    NVCVByte *outputAllocation{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&inputAllocation), kInputBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void **>(&outputAllocation), kOutputBytes));

    auto wrapTensor = [](NVCVByte *allocation, int height, int width, size_t rowStride, size_t sampleStride)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = allocation;
        buffer.strides[0] = sampleStride;
        buffer.strides[1] = rowStride;
        buffer.strides[2] = 1;
        buffer.strides[3] = 1;
        return nvcv::TensorWrapData(
            nvcv::TensorDataStridedCuda{
                nvcv::TensorShape{{kSamples, height, width, 1}, "NHWC"},
                nvcv::TYPE_U8, buffer
        },
            nvcv::TensorDataCleanupCallback{[allocation](const nvcv::TensorData &)
                                            {
                                                EXPECT_EQ(cudaSuccess, cudaFree(allocation));
                                            }});
    };

    nvcv::Tensor input  = wrapTensor(inputAllocation, kInputHeight, kInputWidth, kInputRowStride, kInputSampleStride);
    nvcv::Tensor output = wrapTensor(outputAllocation, kCropHeight, kCropWidth, kOutputRowStride, kOutputSampleStride);

    std::vector<uint8_t> inputHost(kInputBytes, 0xC3);
    for (int sample = 0; sample < kSamples; ++sample)
    {
        for (int y = 0; y < kInputHeight; ++y)
        {
            for (int x = 0; x < kInputWidth; ++x)
            {
                inputHost[static_cast<size_t>(sample) * kInputSampleStride + static_cast<size_t>(y) * kInputRowStride
                          + static_cast<size_t>(x)]
                    = static_cast<uint8_t>((37 * sample + 11 * y + 3 * x + 7) % 251);
            }
        }
    }

    std::vector<uint8_t> expected(kOutputBytes, 0xA5);
    for (int sample = 0; sample < kSamples; ++sample)
    {
        for (int y = 0; y < kCropHeight; ++y)
        {
            for (int x = 0; x < kCropWidth; ++x)
            {
                const size_t srcOffset = static_cast<size_t>(sample) * kInputSampleStride
                                       + static_cast<size_t>(y + kCropY) * kInputRowStride + kCropX + x;
                const size_t dstOffset
                    = static_cast<size_t>(sample) * kOutputSampleStride + static_cast<size_t>(y) * kOutputRowStride + x;
                expected[dstOffset] = inputHost[srcOffset];
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputAllocation, inputHost.data(), kInputBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(outputAllocation, 0xA5, kOutputBytes));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::CustomCrop cropOp;
    const NVCVRectI    cropRect{kCropX, kCropY, kCropWidth, kCropHeight};
    ASSERT_NO_THROW(cropOp(stream, input, output, cropRect));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> actual(kOutputBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outputAllocation, kOutputBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// CustomCrop copies a sub-region of each channel plane independently, so cropping a planar image is
// just copying N*C single-channel planes. These tests feed identical data through cvcuda::CustomCrop
// in interleaved and planar layout and require the (re-interleaved) planar output to match the
// interleaved output bit-for-bit, for every dtype and crop geometry. CustomCrop is tensor-only, so
// there is no var-shape parity case.
// =============================================================================

// Parameters: inWidth, inHeight, cropWidth, cropHeight, cropX, cropY, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpCustomCropPlanar,
                  test::ValueList<int, int, int, int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 113, 100, 64, 20, 10, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, offset crop, batch
    {123,  66,  50, 40,  0,  0, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, top-left crop
    { 64,  48,  32, 24, 16, 12, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8 (uchar4 planar tensor)
    { 50,  40,  25, 20,  5,  5, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, interior crop
    { 64,  48,  40, 30,  8,  6, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // float planar
});

// clang-format on

TEST_P(OpCustomCropPlanar, tensor_matches_interleaved)
{
    // Crop identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
    // The shared upload/run/download/compare scaffolding lives in PlanarParityUtils.hpp; here we only
    // bind the CustomCrop call and its ROI.
    const NVCVRectI crpRect = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<2>(), GetParamValue<3>()};
    test::planar::RunTensorParity(
        GetParamValue<7>(), GetParamValue<8>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
        GetParamValue<3>(), GetParamValue<6>(),
        [crpRect](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::CustomCrop op;
            EXPECT_NO_THROW(op(s, src, dst, crpRect));
        });
}

TEST_P(OpCustomCropPlanar, rejects_grid_z_overflow)
{
    const NVCVRectI    crpRect = {0, 0, 1, 1};
    cvcuda::CustomCrop cropOp;
    test::planar::ExpectPlanarTensorRejected(
        {21846, 3, 1, 1}, {21846, 3, 1, 1},
        [&cropOp, &crpRect](cudaStream_t s, const nvcv::Tensor &in, const nvcv::Tensor &out)
        { cropOp(s, in, out, crpRect); });
}

TEST_P(OpCustomCropPlanar, rejects_non_tightly_packed_batched_tensors)
{
    auto makePaddedNCHW = [](int numSamples, int numChannels, int height, int width)
    {
        nvcv::TensorDataStridedCuda::Buffer buf{};
        const int64_t                       channelStride = static_cast<int64_t>(height) * width;
        buf.basePtr                                       = reinterpret_cast<NVCVByte *>(0xDEADBEEFULL);
        buf.strides[0]                                    = numChannels * channelStride + 1;
        buf.strides[1]                                    = channelStride;
        buf.strides[2]                                    = width;
        buf.strides[3]                                    = 1;
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
            nvcv::TensorShape{{numSamples, numChannels, height, width}, "NCHW"},
            nvcv::TYPE_U8, buf
        });
    };

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn   = makePaddedNCHW(2, 3, 5, 5);
    nvcv::Tensor imgOut  = makePaddedNCHW(2, 3, 2, 2);
    NVCVRectI    crpRect = {0, 0, 2, 2};

    cvcuda::CustomCrop cropOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&cropOp, &stream, &imgIn, &imgOut, &crpRect]
                                                             { cropOp(stream, imgIn, imgOut, crpRect); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpCustomCrop_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, int, int, int, int>{
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 5, 5, 2, 2, 0, 0},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 5, 5, 2, 2, 0, 0},
    {nvcv::FMT_RGBf32, nvcv::FMT_RGB8, 5, 5, 2, 2, 0, 0},
    {nvcv::FMT_C64, nvcv::FMT_C64, 5, 5, 2, 2, 0, 0}, // unsupported complex data type
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 7, 2, 0, 0}, // invalid width
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 7, 0, 0}, // invalid height
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 2, -1, 0}, // invalid x
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 2, 0, -1}, // invalid y
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 2, 6, 0}, // invalid x
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 2, 0, 6}, // invalid y
    {nvcv::FMT_U8,   nvcv::FMT_U8,   5, 5, 0, 2, 0, 0}, // zero width, dense-copy path
    {nvcv::FMT_U8,   nvcv::FMT_U8,   5, 5, -1, 2, 0, 0}, // negative width, dense-copy path
    {nvcv::FMT_U8,   nvcv::FMT_U8,   5, 5, 2, 0, 0, 0}, // zero height, dense-copy path
    {nvcv::FMT_U8,   nvcv::FMT_U8,   5, 5, 2, -1, 0, 0}, // negative height, dense-copy path
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 0, 2, 0, 0}, // zero width, kernel path
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, -1, 2, 0, 0}, // negative width, kernel path
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, 0, 0, 0}, // zero height, kernel path
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 5, 2, -1, 0, 0}, // negative height, kernel path
});

// clang-format on

TEST_P(OpCustomCrop_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat outputFmt  = GetParamValue<1>();
    const int         inWidth    = GetParamValue<2>();
    const int         inHeight   = GetParamValue<3>();
    const int         cropWidth  = GetParamValue<4>();
    const int         cropHeight = GetParamValue<5>();
    const int         cropX      = GetParamValue<6>();
    const int         cropY      = GetParamValue<7>();

    const int outWidth       = inWidth;
    const int outHeight      = inHeight;
    int       numberOfImages = 5;

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, outWidth, outHeight, inputFmt);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(numberOfImages, inWidth, inHeight, outputFmt);

    NVCVRectI          crpRect = {cropX, cropY, cropWidth, cropHeight};
    // run operator
    cvcuda::CustomCrop cropOp;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&cropOp, &stream, &imgIn, &imgOut, &crpRect]
                                                             { cropOp(stream, imgIn, imgOut, crpRect); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCustomCrop_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaCustomCropCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpCustomCrop_Negative, rejects_mixed_width_tensor_dtype)
{
    const nvcv::DataType mixedWidth{nvcv::DataKind::UNSIGNED, static_cast<nvcv::Packing>(NVCV_PACKING_X32_Y24b8)};
    nvcv::Tensor         imgIn(
                {
                    {1, 5, 5, 2},
                    "NHWC"
    },
                mixedWidth);
    nvcv::Tensor imgOut(
        {
            {1, 2, 2, 2},
            "NHWC"
    },
        mixedWidth);
    const NVCVRectI cropRect{0, 0, 2, 2};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::CustomCrop cropOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&] { cropOp(stream, imgIn, imgOut, cropRect); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Parameters: input samples, input channels, output samples, output channels
// clang-format off
NVCV_TEST_SUITE_P(OpCustomCropInterleavedShapeNegative, test::ValueList<int, int, int, int>{
    {1, 3, 1, 4},
    {1, 3, 2, 3},
});

// clang-format on

TEST_P(OpCustomCropInterleavedShapeNegative, rejects_mismatched_sample_or_channel_count)
{
    const int       inSamples   = GetParamValue<0>();
    const int       inChannels  = GetParamValue<1>();
    const int       outSamples  = GetParamValue<2>();
    const int       outChannels = GetParamValue<3>();
    const NVCVRectI cropRect{0, 0, 2, 2};

    nvcv::Tensor imgIn(
        nvcv::TensorShape{
            {inSamples, 5, 5, inChannels},
            "NHWC"
    },
        nvcv::TYPE_U8);
    nvcv::Tensor imgOut(
        nvcv::TensorShape{
            {outSamples, 2, 2, outChannels},
            "NHWC"
    },
        nvcv::TYPE_U8);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::CustomCrop cropOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall([&cropOp, stream, &imgIn, &imgOut, &cropRect]
                                                             { cropOp(stream, imgIn, imgOut, cropRect); }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCustomCrop_Negative, planar_rejects_mismatched_sample_channel_shape)
{
    // Flattening planar tensors as N*C single-channel planes would make these both look like six
    // samples. CustomCrop must preserve the original sample/channel contract before flattening.
    const NVCVRectI    crpRect = {0, 0, 2, 2};
    cvcuda::CustomCrop cropOp;
    test::planar::ExpectPlanarTensorRejected(
        {2, 3, 5, 5}, {3, 2, 2, 2},
        [&cropOp, &crpRect](cudaStream_t s, const nvcv::Tensor &in, const nvcv::Tensor &out)
        { cropOp(s, in, out, crpRect); });
}
