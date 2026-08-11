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

#include <common/BorderUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpPadAndStack.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <random>
#include <type_traits>

namespace test = nvcv::test;
namespace util = nvcv::util;

struct PadAndStackRefData
{
    std::vector<uint8_t>                    &hDst;
    const std::vector<std::vector<uint8_t>> &hBatchSrc;
    int                                      dstImgPitch;
    int                                      dstRowStride;
    int                                      dstPixPitch;
    int                                      srcRowStride;
    int                                      srcPixPitch;
    int2                                     size;
    const std::vector<int>                  &topVec;
    const std::vector<int>                  &leftVec;
    NVCVBorderType                           borderType;
    float                                    borderValue;
};

static void ApplyBorderIndex(int2 &coord, int2 size, NVCVBorderType borderType)
{
    if (borderType == NVCV_BORDER_REPLICATE)
    {
        test::ReplicateBorderIndex(coord, size);
    }
    else if (borderType == NVCV_BORDER_WRAP)
    {
        test::WrapBorderIndex(coord, size);
    }
    else if (borderType == NVCV_BORDER_REFLECT)
    {
        test::ReflectBorderIndex(coord, size);
    }
    else if (borderType == NVCV_BORDER_REFLECT101)
    {
        test::Reflect101BorderIndex(coord, size);
    }
}

static uint8_t PadAndStackValue(const PadAndStackRefData &ref, int db, int2 coord, int channel)
{
    const auto &hSrc = ref.hBatchSrc[db];

    if (coord.x >= 0 && coord.x < ref.size.x && coord.y >= 0 && coord.y < ref.size.y)
    {
        return hSrc[coord.y * ref.srcRowStride + coord.x * ref.srcPixPitch + channel];
    }

    if (ref.borderType == NVCV_BORDER_CONSTANT)
    {
        return static_cast<uint8_t>(ref.borderValue);
    }

    ApplyBorderIndex(coord, ref.size, ref.borderType);
    return hSrc[coord.y * ref.srcRowStride + coord.x * ref.srcPixPitch + channel];
}

static void WritePadAndStackPixel(const PadAndStackRefData &ref, int db, int di, int dj)
{
    int2 coord{dj - ref.leftVec[db], di - ref.topVec[db]};

    for (int dk = 0; dk < ref.dstPixPitch; dk++)
    {
        ref.hDst[db * ref.dstImgPitch + di * ref.dstRowStride + dj * ref.dstPixPitch + dk]
            = PadAndStackValue(ref, db, coord, dk);
    }
}

static void PadAndStack(std::vector<uint8_t> &hDst, const std::vector<std::vector<uint8_t>> &hBatchSrc,
                        const nvcv::TensorDataAccessStridedImagePlanar &dDstData, const int srcWidth,
                        const int srcHeight, const int srcRowStride, const int srcPixPitch,
                        const std::vector<int> &topVec, const std::vector<int> &leftVec,
                        const NVCVBorderType borderType, const float borderValue)
{
    auto dstPixPitch  = dDstData.numChannels();
    auto dstRowStride = static_cast<int>(dDstData.rowStride() / sizeof(uint8_t));
    auto dstImgPitch  = static_cast<int>(dDstData.sampleStride() / sizeof(uint8_t));

    PadAndStackRefData ref{
        hDst,   hBatchSrc, dstImgPitch, dstRowStride, dstPixPitch, srcRowStride, srcPixPitch, {srcWidth, srcHeight},
        topVec, leftVec,   borderType,  borderValue
    };

    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                WritePadAndStackPixel(ref, db, di, dj);
            }
        }
    }
}

template<typename T>
static std::vector<T> InterleavedToPlanar(const std::vector<T> &hwc, int width, int height, int channels)
{
    std::vector<T> chw(hwc.size());
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                chw[(static_cast<size_t>(c) * height + y) * width + x]
                    = hwc[(static_cast<size_t>(y) * width + x) * channels + c];
            }
        }
    }
    return chw;
}

template<typename T>
static std::vector<T> MakePlanarParityInput(size_t count, int sample)
{
    std::vector<T> values(count);

    size_t i = 0;
    for (T &value : values)
    {
        value = static_cast<T>((i * 7 + static_cast<size_t>(sample) * 31 + 13) % 251);
        ++i;
    }
    return values;
}

template<typename T>
static void UploadImage(const nvcv::Image &image, const std::vector<T> &values, int width, int height, int channels,
                        cudaStream_t stream)
{
    auto data = image.exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(data, nvcv::NullOpt);
    ASSERT_TRUE(data->numPlanes() == 1 || data->numPlanes() == channels);

    for (int plane = 0; plane < data->numPlanes(); ++plane)
    {
        const int    planeChannels = data->numPlanes() == 1 ? channels : 1;
        const size_t rowBytes      = static_cast<size_t>(width) * planeChannels * sizeof(T);
        const auto  *src = values.data() + (data->numPlanes() == 1 ? 0 : static_cast<size_t>(plane) * width * height);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(data->plane(plane).basePtr, data->plane(plane).rowStride, src,
                                                 rowBytes, rowBytes, height, cudaMemcpyHostToDevice, stream));
    }
}

template<typename T>
static void RunPadAndStackPlanarParity(int srcWidth, int srcHeight, int numBatches, int dstWidth, int dstHeight,
                                       int topPad, int leftPad, NVCVBorderType borderType, float borderValue,
                                       nvcv::ImageFormat interleavedFormat, nvcv::ImageFormat planarFormat)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels = interleavedFormat.numChannels();

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);
    auto         inTopData  = inTop.exportData<nvcv::TensorDataStridedCuda>();
    auto         inLeftData = inLeft.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    std::vector<int> topVec(numBatches, topPad);
    std::vector<int> leftVec(numBatches, leftPad);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<nvcv::Image> srcIImgs;
    std::vector<nvcv::Image> srcPImgs;
    for (int b = 0; b < numBatches; ++b)
    {
        auto hwc = MakePlanarParityInput<T>(static_cast<size_t>(srcWidth) * srcHeight * channels, b);
        srcIImgs.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, interleavedFormat);
        srcPImgs.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, planarFormat);
        UploadImage(srcIImgs.back(), hwc, srcWidth, srcHeight, channels, stream);
        UploadImage(srcPImgs.back(), InterleavedToPlanar(hwc, srcWidth, srcHeight, channels), srcWidth, srcHeight,
                    channels, stream);
    }

    nvcv::ImageBatchVarShape srcI(numBatches);
    nvcv::ImageBatchVarShape srcP(numBatches);
    srcI.pushBack(srcIImgs.begin(), srcIImgs.end());
    srcP.pushBack(srcPImgs.begin(), srcPImgs.end());

    nvcv::Tensor dstI(numBatches, {dstWidth, dstHeight}, interleavedFormat);
    nvcv::Tensor dstP(numBatches, {dstWidth, dstHeight}, planarFormat);
    auto         dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto         dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstIData);
    ASSERT_NE(nullptr, dstPData);

    cvcuda::PadAndStack op;
    EXPECT_NO_THROW(op(stream, srcI, dstI, inTop, inLeft, borderType, borderValue));
    EXPECT_NO_THROW(op(stream, srcP, dstP, inTop, inLeft, borderType, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int sample = 0; sample < numBatches; ++sample)
    {
        SCOPED_TRACE(sample);
        std::vector<T> interleavedOut;
        std::vector<T> planarOut;
        util::GetImageVectorFromTensor<T>(*dstIData, sample, interleavedOut);
        util::GetImageVectorFromTensor<T>(*dstPData, sample, planarOut);
        EXPECT_EQ(InterleavedToPlanar(interleavedOut, dstWidth, dstHeight, channels), planarOut);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStack, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         NVCVBorderType, borderValue
    {       212,       113,          1,      111,       132,      0,       0,   NVCV_BORDER_CONSTANT,         0.f},
    {        12,        13,          2,      211,       232,      0,       3,   NVCV_BORDER_CONSTANT,        12.f},
    {       212,       113,          3,       11,       432,      5,       0,   NVCV_BORDER_CONSTANT,        13.f},
    {       212,       613,          4,      311,       532,      7,       7,   NVCV_BORDER_CONSTANT,       134.f},

    {       234,       131,          2,      131,       130,     33,      22,  NVCV_BORDER_REPLICATE,         0.f},
    {       234,       131,          2,      123,       132,     41,      42,    NVCV_BORDER_REFLECT,         0.f},
    {       234,       131,          2,      134,       131,     53,      62,       NVCV_BORDER_WRAP,         0.f},
    {       243,       123,          2,      132,       123,     77,      98, NVCV_BORDER_REFLECT101,         0.f},

});

// clang-format on

TEST_P(OpPadAndStack, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int dstWidth   = GetParamValue<3>();
    int dstHeight  = GetParamValue<4>();
    int topPad     = GetParamValue<5>();
    int leftPad    = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();

    float borderValue = GetParamValue<8>();

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    auto inTopData  = inTop.exportData<nvcv::TensorDataStridedCuda>();
    auto inLeftData = inLeft.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    auto inTopAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inTopData);
    ASSERT_TRUE(inTopAccess);

    auto inLeftAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inLeftData);
    ASSERT_TRUE(inLeftAccess);

    auto inTopBufSize  = static_cast<int>((inTopAccess->sampleStride() / sizeof(int)) * inTopAccess->numSamples());
    auto inLeftBufSize = static_cast<int>((inLeftAccess->sampleStride() / sizeof(int)) * inLeftAccess->numSamples());

    ASSERT_EQ(inTopBufSize, inLeftBufSize);

    std::vector<int> topVec(inTopBufSize);
    std::vector<int> leftVec(inLeftBufSize);

    for (int b = 0; b < numBatches; ++b)
    {
        topVec[b]  = topPad;
        leftVec[b] = leftPad;
    }

    // Copy vectors with top and left padding to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<nvcv::Image> srcImgVec;

    std::vector<std::vector<uint8_t>> batchSrcVec;

    std::default_random_engine randEng{0};

    int srcStride    = 0;
    int srcRowStride = 0;
    int srcPixPitch  = 0;

    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGBA8);

        auto imgSrcData = srcImgVec.back().exportData<nvcv::ImageDataStridedCuda>();

        srcStride      = imgSrcData->plane(0).rowStride;
        srcRowStride   = srcStride / sizeof(uint8_t);
        srcPixPitch    = 4;
        int srcBufSize = srcRowStride * imgSrcData->plane(0).height;

        std::vector<uint8_t> srcVec(srcBufSize);

        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        std::ranges::generate(srcVec, [&srcRand, &randEng]() { return srcRand(randEng); });

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(),
                                               srcVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

        batchSrcVec.push_back(srcVec);
    }

    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);

    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);

    auto dstBufSize = static_cast<int>((dstAccess->sampleStride() / sizeof(uint8_t)) * dstAccess->numSamples());

    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(uint8_t), stream));

    std::vector<uint8_t> testVec(dstBufSize);
    std::vector<uint8_t> goldVec(dstBufSize);

    // Generate gold result
    PadAndStack(goldVec, batchSrcVec, *dstAccess, srcWidth, srcHeight, srcRowStride, srcPixPitch, topVec, leftVec,
                borderType, borderValue);

    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_NO_THROW(padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(goldVec, testVec);
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStackPlanar,
                  test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, nvcv::ImageFormat,
                                  nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         borderType, borderValue,   interleavedFormat,      planarFormat
    {        17,        13,          2,       23,        21,      2,       3,   NVCV_BORDER_CONSTANT,       11.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
    {        19,        11,          3,       27,        18,      1,       0, NVCV_BORDER_REFLECT101,        0.f,       nvcv::FMT_RGBA8,   nvcv::FMT_RGBA8p},
    {        15,        14,          2,       20,        19,      4,       2,  NVCV_BORDER_REPLICATE,        0.f,     nvcv::FMT_RGBf32,  nvcv::FMT_RGBf32p},
});

// clang-format on

TEST_P(OpPadAndStackPlanar, varshape_matches_interleaved)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int dstWidth   = GetParamValue<3>();
    int dstHeight  = GetParamValue<4>();
    int topPad     = GetParamValue<5>();
    int leftPad    = GetParamValue<6>();

    NVCVBorderType    borderType        = GetParamValue<7>();
    float             borderValue       = GetParamValue<8>();
    nvcv::ImageFormat interleavedFormat = GetParamValue<9>();
    nvcv::ImageFormat planarFormat      = GetParamValue<10>();

    if (interleavedFormat == nvcv::FMT_RGBf32)
    {
        RunPadAndStackPlanarParity<float>(srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,
                                          borderType, borderValue, interleavedFormat, planarFormat);
    }
    else
    {
        RunPadAndStackPlanarParity<uint8_t>(srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,
                                            borderType, borderValue, interleavedFormat, planarFormat);
    }
}

static auto OpPadAndStackNegativeParams()
{
    test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat,
                    NVCVBorderType>
        params{};
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_S32,
                        static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_F16, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_F16, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_U8, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_F32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_F32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_RGBA8p,
                        NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpPadAndStack_Negative, OpPadAndStackNegativeParams());

TEST_P(OpPadAndStack_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    nvcv::ImageFormat topFmt             = GetParamValue<3>();
    nvcv::ImageFormat leftFmt            = GetParamValue<4>();
    NVCVBorderType    borderType         = GetParamValue<5>();

    int   srcWidth    = 12;
    int   srcHeight   = 13;
    int   numBatches  = 3;
    int   dstWidth    = 111;
    int   dstHeight   = 131;
    float borderValue = 0.f;

    nvcv::Tensor inTop(1, {numBatches, 1}, topFmt);
    nvcv::Tensor inLeft(1, {numBatches, 1}, leftFmt);

    std::vector<nvcv::Image> srcImgVec;
    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, inputFmt);
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor        imgDst(numBatches, {dstWidth, dstHeight}, outputFmt);
    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_EQ(
        expectedReturnCode,
        nvcv::ProtectCall([&padAndStackOp, &stream, &imgBatchSrc, &imgDst, &inTop, &inLeft, &borderType, &borderValue]
                          { padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue); }));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPadAndStack_Negative, input_format_not_same)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int   srcWidth    = 12;
    int   srcHeight   = 13;
    int   numBatches  = 3;
    int   dstWidth    = 111;
    int   dstHeight   = 131;
    float borderValue = 0.f;

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    std::vector<nvcv::Image> srcImgVec;
    for (int b = 0; b < numBatches - 1; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGBA8);
    }
    srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGB8);
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor        imgDst(numBatches, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);
    // Generate test result
    cvcuda::PadAndStack padAndStackOp;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&padAndStackOp, &stream, &imgBatchSrc, &imgDst, &inTop, &inLeft, &borderValue]
                  { padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, NVCV_BORDER_CONSTANT, borderValue); }));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpPadAndStack_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPadAndStackCreate(nullptr));
}
