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
#include <cuda_runtime_api.h>
#include <cvcuda/OpCenterCrop.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

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

// Width is specified in pixels.
static void WriteData(const nvcv::TensorDataAccessStridedImagePlanar &data, uint8_t val, int crop_rows,
                      int crop_columns)
{
    EXPECT_TRUE(data.layout() == NVCV_TENSOR_NHWC || data.layout() == NVCV_TENSOR_HWC);
    EXPECT_LE(crop_columns, data.numCols());
    EXPECT_LE(crop_rows, data.numRows());

    int        bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int        bytesPerPixel = data.numChannels() * bytesPerChan;
    auto      *impPtrTop     = reinterpret_cast<std::byte *>(data.sampleData(0));
    std::byte *impPtr        = nullptr;
    auto       numImages     = static_cast<int>(data.numSamples());
    auto       rowStride     = static_cast<int>(data.rowStride());

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
    int      bytesPerChan  = data.dtype().bitsPerChannel()[0] / 8;
    int      bytesPerPixel = data.numChannels() * bytesPerChan;
    uint8_t *ptrTop        = vect.data();
    auto     numImages     = static_cast<int>(data.numSamples());
    for (int img = 0; img < numImages; img++)
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

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, inWidth, inHeight, nvcv::FMT_RGBA8);
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

    EXPECT_EQ(cudaSuccess, cudaMemset(inData->basePtr(), 0x00, inBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(outData->basePtr(), 0x00, outBufSize));
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

namespace {

size_t ImageByteOffset(int x, int y, int channel, int width, int height, int channels, int bytesPerChannel, bool planar)
{
    const size_t element = planar ? (channel * height + y) * width + x : (y * width + x) * channels + channel;
    return element * bytesPerChannel;
}

std::vector<nvcv::Byte> MakeExactOutputInput(int width, int height, int channels, int bytesPerChannel, int seed)
{
    std::vector<nvcv::Byte> input(static_cast<size_t>(width) * height * channels * bytesPerChannel);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<nvcv::Byte>((i * 37 + seed * 53 + 11) & 0xff);
    }
    return input;
}

std::vector<nvcv::Byte> CenterCropReference(const std::vector<nvcv::Byte> &input, int inWidth, int inHeight,
                                            int cropWidth, int cropHeight, int channels, int bytesPerChannel,
                                            bool planar)
{
    std::vector<nvcv::Byte> output(static_cast<size_t>(cropWidth) * cropHeight * channels * bytesPerChannel);
    const int               left = (inWidth - cropWidth) / 2;
    const int               top  = (inHeight - cropHeight) / 2;

    for (int channel = 0; channel < channels; ++channel)
    {
        for (int y = 0; y < cropHeight; ++y)
        {
            for (int x = 0; x < cropWidth; ++x)
            {
                const size_t src
                    = ImageByteOffset(x + left, y + top, channel, inWidth, inHeight, channels, bytesPerChannel, planar);
                const size_t dst
                    = ImageByteOffset(x, y, channel, cropWidth, cropHeight, channels, bytesPerChannel, planar);
                std::copy_n(input.begin() + src, bytesPerChannel, output.begin() + dst);
            }
        }
    }
    return output;
}

void RunExactOutputCase(nvcv::ImageFormat fmt, int inWidth, int inHeight, int crop_columns, int crop_rows,
                        int numberOfImages)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(numberOfImages, crop_columns, crop_rows, fmt);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(numberOfImages, inWidth, inHeight, fmt);

    auto inData  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    const int  channels        = inAccess->numChannels();
    const int  bytesPerChannel = inAccess->dtype().bitsPerChannel()[0] / 8;
    const bool planar          = inAccess->layout() == NVCV_TENSOR_NCHW || inAccess->layout() == NVCV_TENSOR_CHW;

    std::vector<std::vector<nvcv::Byte>> gold(numberOfImages);
    for (int image = 0; image < numberOfImages; ++image)
    {
        auto input = MakeExactOutputInput(inWidth, inHeight, channels, bytesPerChannel, image);
        gold[image]
            = CenterCropReference(input, inWidth, inHeight, crop_columns, crop_rows, channels, bytesPerChannel, planar);
        nvcv::util::SetImageTensorFromByteVector(*inData, input, image);
    }

    cvcuda::CenterCrop cropOp;
    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, {crop_columns, crop_rows}));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int image = 0; image < numberOfImages; ++image)
    {
        std::vector<nvcv::Byte> test;
        nvcv::util::GetImageByteVectorFromTensor(*outData, image, test);
        EXPECT_EQ(gold[image], test);
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

} // namespace

#define NVCV_IMAGE_FORMAT_2U8 NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)

// clang-format off
NVCV_TEST_SUITE_P(OpCenterCropExactOutput,
                  test::ValueList<nvcv::ImageFormat, int, int, int, int, int>{
    {                                     nvcv::FMT_U8, 18, 14,  9, 7, 3}, // NHWC, byte copy, C1, odd remainder
    {                                     nvcv::FMT_S8, 19, 15, 11, 9, 2}, // NHWC, signed byte copy
    {nvcv::ImageFormat{NVCV_IMAGE_FORMAT_2U8}, 21, 17, 13, 9, 2}, // NHWC, byte C2 fallback
    {                                   nvcv::FMT_RGB8, 23, 19, 11, 9, 2}, // NHWC, byte C3 fallback
    {                                  nvcv::FMT_RGBA8, 25, 21, 13, 9, 2}, // NHWC, byte C4 fallback
    {                                nvcv::FMT_RGBAf32, 13, 11,  7, 5, 2}, // NHWC, wider-type fallback
    {                                  nvcv::FMT_RGB8, 23, 19, 11, 9, 1}, // HWC rank-3 fallback
    {                                 nvcv::FMT_RGB8p, 23, 19, 11, 9, 2}, // NCHW, byte copy, C3
    {                                nvcv::FMT_RGBA8p, 25, 21, 13, 9, 2}, // NCHW, byte copy, C4
    {                                 nvcv::FMT_RGB8p, 23, 19, 11, 9, 1}, // CHW flattened byte copy
    {                              nvcv::FMT_RGBAf32p, 13, 11,  7, 5, 2}, // NCHW, wider-type fallback
});

// clang-format on

#undef NVCV_IMAGE_FORMAT_2U8

TEST_P(OpCenterCropExactOutput, matches_host_reference)
{
    RunExactOutputCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(), GetParamValue<3>(),
                       GetParamValue<4>(), GetParamValue<5>());
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// CenterCrop copies a centered sub-region of each channel plane independently, so cropping a planar
// image is just copying N*C single-channel planes (the centering offset is the same on every plane).
// These tests feed identical data through cvcuda::CenterCrop in interleaved and planar layout and
// require the (re-interleaved) planar output to match the interleaved output bit-for-bit, for every
// dtype and crop geometry. CenterCrop is tensor-only, so there is no var-shape parity case.
// =============================================================================

namespace {

// Center-crop identical data in interleaved and planar tensor layout; outputs must match bit-for-bit.
// The shared scaffolding (upload/run/download/compare) lives in PlanarParityUtils.hpp; here we only
// bind the CenterCrop call and its crop size.
void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int inW, int inH,
                               int cropW, int cropH, int numImages)
{
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, inW, inH, cropW, cropH, numImages,
        [cropW, cropH](cudaStream_t s, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::CenterCrop op;
            EXPECT_NO_THROW(op(s, src, dst, {cropW, cropH}));
        });
}

} // namespace

// Parameters: inWidth, inHeight, cropWidth, cropHeight, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpCenterCropPlanar,
                  test::ValueList<int, int, int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    {176, 112, 100, 64, 2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, even centering, batch
    {123,  67,  50, 40, 1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8}, // RGB8, odd dims (centering truncation)
    { 64,  48,  32, 24, 2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8 (uchar4 planar tensor)
    { 50,  40,  26, 20, 1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8}, // RGBA8, interior crop
    { 64,  48,  40, 30, 2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32}, // float planar
});

// clang-format on

TEST_P(OpCenterCropPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<5>(), GetParamValue<6>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpCenterCrop_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, int ,int>{
    // inFmt, outFmt, width, height, cropCols, cropRows
    {nvcv::FMT_RGB8, nvcv::FMT_RGBf16, 5, 7, 5, 7},
    {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 5, 7, 5, 7},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 5, 7, 5, 7},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 7, 6, 7},
    {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 5, 7, 5, 8},
});

// clang-format on

TEST_P(OpCenterCrop_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::ImageFormat inFmt        = GetParamValue<0>();
    nvcv::ImageFormat outFmt       = GetParamValue<1>();
    int               inWidth      = GetParamValue<2>();
    int               inHeight     = GetParamValue<3>();
    int               crop_columns = GetParamValue<4>();
    int               crop_rows    = GetParamValue<5>();

    nvcv::Tensor imgOut = nvcv::util::CreateTensor(2, inWidth, inHeight, inFmt);
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(2, inWidth, inHeight, outFmt);

    // run operator
    cvcuda::CenterCrop cropOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&cropOp, &stream, &imgIn, &imgOut, &crop_columns, &crop_rows] {
                                                   cropOp(stream, imgIn, imgOut, {crop_columns, crop_rows});
                                               }));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCenterCrop_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaCenterCropCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpCenterCrop_Negative, planar_rejects_two_channels)
{
    cvcuda::CenterCrop cropOp;
    test::planar::ExpectPlanarTensorRejected({2, 2, 5, 5}, {2, 2, 2, 2},
                                             [&cropOp](cudaStream_t s, const nvcv::Tensor &in, const nvcv::Tensor &out)
                                             {
                                                 cropOp(s, in, out, {2, 2});
                                             });
}

TEST(OpCenterCrop_Negative, planar_rejects_mismatched_sample_channel_shape)
{
    // Flattening planar tensors as N*C single-channel planes would make these both look like six
    // samples. CenterCrop must preserve the original sample/channel contract before flattening.
    cvcuda::CenterCrop cropOp;
    test::planar::ExpectPlanarTensorRejected({2, 3, 5, 5}, {3, 2, 2, 2},
                                             [&cropOp](cudaStream_t s, const nvcv::Tensor &in, const nvcv::Tensor &out)
                                             {
                                                 cropOp(s, in, out, {2, 2});
                                             });
}
