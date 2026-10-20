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
#include "HalfTestUtils.hpp"

#include <common/BorderUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpPadAndStack.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cstring>
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
static std::vector<T> MakePadAndStackPlanarGold(const std::vector<T> &input, int2 srcSize, int2 dstSize, int channels,
                                                int2 padding, NVCVBorderType borderType, T padValue)
{
    std::vector<T> expected(static_cast<size_t>(dstSize.x) * dstSize.y * channels);
    for (int y = 0; y < dstSize.y; ++y)
        for (int x = 0; x < dstSize.x; ++x)
        {
            int2       coord{x - padding.x, y - padding.y};
            const bool outside = coord.x < 0 || coord.x >= srcSize.x || coord.y < 0 || coord.y >= srcSize.y;
            if (outside && borderType != NVCV_BORDER_CONSTANT)
                ApplyBorderIndex(coord, srcSize, borderType);
            for (int c = 0; c < channels; ++c)
                expected[(static_cast<size_t>(y) * dstSize.x + x) * channels + c]
                    = outside && borderType == NVCV_BORDER_CONSTANT
                        ? padValue
                        : input[(static_cast<size_t>(coord.y) * srcSize.x + coord.x) * channels + c];
        }
    return expected;
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
    for (int b = 0; b < numBatches; ++b)
    {
        topVec[b] -= b % 2;
        leftVec[b] -= b % 3;
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<nvcv::Image> srcIImgs;
    std::vector<nvcv::Image> srcPImgs;
    for (int b = 0; b < numBatches; ++b)
    {
        const int width  = srcWidth - b % 3;
        const int height = srcHeight - b % 2;
        auto      hwc    = MakePlanarParityInput<T>(static_cast<size_t>(width) * height * channels, b);
        srcIImgs.emplace_back(nvcv::Size2D{width, height}, interleavedFormat);
        srcPImgs.emplace_back(nvcv::Size2D{width, height}, planarFormat);
        UploadImage(srcIImgs.back(), hwc, width, height, channels, stream);
        UploadImage(srcPImgs.back(), InterleavedToPlanar(hwc, width, height, channels), width, height, channels,
                    stream);
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
        const int width    = srcWidth - sample % 3;
        const int height   = srcHeight - sample % 2;
        auto      input    = MakePlanarParityInput<T>(static_cast<size_t>(width) * height * channels, sample);
        auto      padValue = static_cast<T>(borderValue);
        if constexpr (std::is_same_v<T, uint16_t>)
        {
            if (interleavedFormat.planeDataType(0).dataKind() == nvcv::DataKind::FLOAT)
            {
                const __half halfValue = __float2half(borderValue);
                std::memcpy(&padValue, &halfValue, sizeof(padValue));
            }
        }
        auto           expected = MakePadAndStackPlanarGold(input, {width, height}, {dstWidth, dstHeight}, channels,
                                                            {leftVec[sample], topVec[sample]}, borderType, padValue);
        std::vector<T> interleavedOut;
        std::vector<T> planarOut;
        util::GetImageVectorFromTensor<T>(*dstIData, sample, interleavedOut);
        util::GetImageVectorFromTensor<T>(*dstPData, sample, planarOut);
        ASSERT_EQ(expected.size(), interleavedOut.size());
        EXPECT_EQ(0, std::memcmp(expected.data(), interleavedOut.data(), expected.size() * sizeof(T)));
        expected = InterleavedToPlanar(expected, dstWidth, dstHeight, channels);
        ASSERT_EQ(expected.size(), planarOut.size());
        EXPECT_EQ(0, std::memcmp(expected.data(), planarOut.data(), expected.size() * sizeof(T)));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStack, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         NVCVBorderType, borderValue,          format
    {         1,         1,          1,        1,    131071,      0,       0,   NVCV_BORDER_CONSTANT,       137.f,    nvcv::FMT_U8},
    {       212,       113,          1,      111,       132,      0,       0,   NVCV_BORDER_CONSTANT,         0.f, nvcv::FMT_RGBA8},
    {        12,        13,          2,      211,       232,      0,       3,   NVCV_BORDER_CONSTANT,        12.f, nvcv::FMT_RGBA8},
    {       212,       113,          3,       11,       432,      5,       0,   NVCV_BORDER_CONSTANT,        13.f, nvcv::FMT_RGBA8},
    {       212,       613,          4,      311,       532,      7,       7,   NVCV_BORDER_CONSTANT,       134.f, nvcv::FMT_RGBA8},
    {        64,        32,          2,       96,        64,     16,      16,   NVCV_BORDER_CONSTANT,         0.f,  nvcv::FMT_RGB8},
    {        17,        13,          3,       23,        21,     -2,       3,   NVCV_BORDER_CONSTANT,       137.f,  nvcv::FMT_RGB8},
    {        17,        13,          3,       23,        21,     -2,       3,   NVCV_BORDER_CONSTANT,      137.5f,
        nvcv::ImageFormat{nvcv::MemLayout::PL, nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XY00, nvcv::Packing::X8_Y8}},

    {       234,       131,          2,      131,       130,     33,      22,  NVCV_BORDER_REPLICATE,         0.f, nvcv::FMT_RGBA8},
    {       234,       131,          2,      123,       132,     41,      42,    NVCV_BORDER_REFLECT,         0.f, nvcv::FMT_RGBA8},
    {       234,       131,          2,      134,       131,     53,      62,       NVCV_BORDER_WRAP,         0.f, nvcv::FMT_RGBA8},
    {       243,       123,          2,      132,       123,     77,      98, NVCV_BORDER_REFLECT101,         0.f, nvcv::FMT_RGBA8},

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

    float             borderValue = GetParamValue<8>();
    nvcv::ImageFormat format      = GetParamValue<9>();

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
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, format);

        auto imgSrcData = srcImgVec.back().exportData<nvcv::ImageDataStridedCuda>();

        srcStride      = imgSrcData->plane(0).rowStride;
        srcRowStride   = srcStride / sizeof(uint8_t);
        srcPixPitch    = format.numChannels();
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

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);

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

NVCV_TEST_SUITE_P(OpPadAndStackScalarStrides, test::ValueList<int, int, int, int>{
                                                  { 1, 0, 0, 0},
                                                  { 3, 1, 1, 0},
                                                  { 4, 0, 0, 0},
                                                  { 5, 1, 0, 1},
                                                  {17, 0, 1, 2},
                                                  {17, 4, 0, 6},
});

TEST_P(OpPadAndStackScalarStrides, constant_padding_preserves_storage_guards)
{
    const int                      width     = GetParamValue<0>();
    const int                      srcOffset = GetParamValue<1>();
    const int                      dstOffset = GetParamValue<2>();
    const int                      padding   = GetParamValue<3>();
    constexpr int                  batches   = 3;
    constexpr int                  srcHeight = 3;
    constexpr int                  dstHeight = 7;
    const int                      dstWidth  = width + 9;
    const int                      srcRow    = width + padding;
    const int                      dstRow    = dstWidth + padding;
    const int                      srcBytes  = srcOffset + srcHeight * srcRow + 4;
    const int                      dstBytes  = dstOffset + batches * dstHeight * dstRow + 4;
    const std::array<int, batches> top{1, -1, 5};
    const std::array<int, batches> left{4, -2, 3};

    nvcv::Tensor inTop(1, {batches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {batches, 1}, nvcv::FMT_S32);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inTop.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), top.data(),
                                      top.size() * sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inLeft.exportData<nvcv::TensorDataStridedCuda>()->basePtr(), left.data(),
                                      left.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<nvcv::Tensor>         storage;
    std::vector<std::vector<uint8_t>> input(batches, std::vector<uint8_t>(srcBytes, 0xA5));
    nvcv::ImageBatchVarShape          images(batches);
    for (int b = 0; b < batches; ++b)
    {
        for (int y = 0; y < srcHeight; ++y)
            for (int x = 0; x < width; ++x)
                input[b][srcOffset + y * srcRow + x] = static_cast<uint8_t>(b * 53 + y * 17 + x);
        storage.emplace_back(1, nvcv::Size2D{srcBytes, 1}, nvcv::FMT_U8);
        auto data = storage.back().exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(data);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(data->basePtr(), input[b].data(), srcBytes, cudaMemcpyHostToDevice));
        nvcv::ImageDataStridedCuda::Buffer buffer{};
        buffer.numPlanes           = 1;
        buffer.planes[0].width     = width;
        buffer.planes[0].height    = srcHeight;
        buffer.planes[0].rowStride = srcRow;
        buffer.planes[0].basePtr   = reinterpret_cast<NVCVByte *>(data->basePtr()) + srcOffset;
        images.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{nvcv::FMT_U8, buffer}));
    }

    nvcv::Tensor outputStorage(1, {dstBytes, 1}, nvcv::FMT_U8);
    auto         outputData = outputStorage.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(outputData);
    ASSERT_EQ(cudaSuccess, cudaMemset(outputData->basePtr(), 0xA5, dstBytes));
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr             = reinterpret_cast<NVCVByte *>(outputData->basePtr()) + dstOffset;
    buffer.strides[0]          = dstHeight * dstRow;
    buffer.strides[1]          = dstRow;
    buffer.strides[2]          = 1;
    buffer.strides[3]          = 1;
    auto                output = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{batches, dstHeight, dstWidth, 1}, "NHWC"},
        nvcv::TYPE_U8, buffer
    });
    cvcuda::PadAndStack op;
    ASSERT_NO_THROW(op(nullptr, images, output, inTop, inLeft, NVCV_BORDER_CONSTANT, 137.5f));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));

    std::vector<uint8_t> expected(dstBytes, 0xA5);
    std::vector<uint8_t> actual(dstBytes);
    for (int b = 0; b < batches; ++b)
        for (int y = 0; y < dstHeight; ++y)
            for (int x = 0; x < dstWidth; ++x)
            {
                const int sx = x - left[b];
                const int sy = y - top[b];
                expected[dstOffset + (b * dstHeight + y) * dstRow + x]
                    = sx >= 0 && sx < width && sy >= 0 && sy < srcHeight ? input[b][srcOffset + sy * srcRow + sx] : 137;
            }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), outputData->basePtr(), dstBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
}

TEST(OpPadAndStack, two_channel_u8_interleaved_correct_output)
{
    constexpr int srcWidth   = 3;
    constexpr int srcHeight  = 2;
    constexpr int numBatches = 2;
    constexpr int dstWidth   = 5;
    constexpr int dstHeight  = 4;

    const nvcv::ImageFormat twoChannelInterleaved{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XY00, ASSOCIATED, X8_Y8)};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor top(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor left(1, {numBatches, 1}, nvcv::FMT_S32);
    auto         topData  = top.exportData<nvcv::TensorDataStridedCuda>();
    auto         leftData = left.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, topData);
    ASSERT_NE(nullptr, leftData);

    const std::vector<int> topVec{1, 0};
    const std::vector<int> leftVec{1, 2};
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(topData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(leftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<nvcv::Image>          srcImages;
    std::vector<std::vector<uint8_t>> batchSrc;
    const int                         channels     = twoChannelInterleaved.numChannels();
    const int                         srcPixPitch  = channels;
    int                               srcRowStride = 0;

    auto fillSample = [&](std::vector<uint8_t> &values, int rowStride, int b)
    {
        for (int y = 0; y < srcHeight; ++y)
        {
            for (int x = 0; x < srcWidth; ++x)
            {
                for (int c = 0; c < channels; ++c)
                {
                    values[y * rowStride + x * channels + c] = static_cast<uint8_t>(17 + b * 80 + y * 23 + x * 7 + c);
                }
            }
        }
    };

    for (int b = 0; b < numBatches; ++b)
    {
        srcImages.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, twoChannelInterleaved);
        auto srcData = srcImages.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcData, nvcv::NullOpt);

        const int rowStride = srcData->plane(0).rowStride;
        if (b == 0)
        {
            srcRowStride = rowStride;
        }
        else
        {
            ASSERT_EQ(srcRowStride, rowStride);
        }

        std::vector<uint8_t> values(static_cast<size_t>(rowStride) * srcHeight, 0);
        fillSample(values, rowStride, b);

        const size_t rowBytes = static_cast<size_t>(srcWidth) * channels * sizeof(uint8_t);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(srcData->plane(0).basePtr, srcData->plane(0).rowStride, values.data(),
                                                 rowStride, rowBytes, srcHeight, cudaMemcpyHostToDevice, stream));
        batchSrc.push_back(std::move(values));
    }

    nvcv::ImageBatchVarShape input(numBatches);
    input.pushBack(srcImages.begin(), srcImages.end());

    nvcv::Tensor output(numBatches, {dstWidth, dstHeight}, twoChannelInterleaved);
    auto         outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outputData);
    auto outputAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outputData);
    ASSERT_TRUE(outputAccess);

    const auto outputBufSize
        = static_cast<int>((outputAccess->sampleStride() / sizeof(uint8_t)) * outputAccess->numSamples());
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(outputData->basePtr(), 0, outputBufSize, stream));

    std::vector<uint8_t> testValues(outputBufSize);
    std::vector<uint8_t> goldValues(outputBufSize);
    PadAndStack(goldValues, batchSrc, *outputAccess, srcWidth, srcHeight, srcRowStride, srcPixPitch, topVec, leftVec,
                NVCV_BORDER_CONSTANT, 99.f);

    cvcuda::PadAndStack op;
    EXPECT_NO_THROW(op(stream, input, output, top, left, NVCV_BORDER_CONSTANT, 99.f));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testValues.data(), outputData->basePtr(), outputBufSize, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    EXPECT_EQ(goldValues, testValues);
}

TEST(OpPadAndStack, integer_planar_interleaved_parity)
{
    const nvcv::ImageFormat rgbU16{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, UNSIGNED, XYZ0, ASSOCIATED, X16_Y16_Z16)};
    const nvcv::ImageFormat rgbU16Planar{NVCV_DETAIL_MAKE_NONCOLOR_FMT3(PL, UNSIGNED, XYZ0, ASSOCIATED, X16, X16, X16)};
    const nvcv::ImageFormat rgbS16{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, XYZ0, ASSOCIATED, X16_Y16_Z16)};
    const nvcv::ImageFormat rgbS16Planar{NVCV_DETAIL_MAKE_NONCOLOR_FMT3(PL, SIGNED, XYZ0, ASSOCIATED, X16, X16, X16)};
    const nvcv::ImageFormat rgbS32{NVCV_DETAIL_MAKE_NONCOLOR_FMT1(PL, SIGNED, XYZ0, ASSOCIATED, X32_Y32_Z32)};
    const nvcv::ImageFormat rgbS32Planar{NVCV_DETAIL_MAKE_NONCOLOR_FMT3(PL, SIGNED, XYZ0, ASSOCIATED, X32, X32, X32)};

    RunPadAndStackPlanarParity<uint16_t>(7, 5, 2, 11, 9, 2, 3, NVCV_BORDER_CONSTANT, 17.f, rgbU16, rgbU16Planar);
    RunPadAndStackPlanarParity<int16_t>(7, 5, 2, 11, 9, 2, 3, NVCV_BORDER_CONSTANT, 17.f, rgbS16, rgbS16Planar);
    RunPadAndStackPlanarParity<int32_t>(7, 5, 2, 11, 9, 2, 3, NVCV_BORDER_CONSTANT, 17.f, rgbS32, rgbS32Planar);
    RunPadAndStackPlanarParity<uint16_t>(529, 7, 3, 593, 15, 4, 32, NVCV_BORDER_REFLECT101, 0.f, rgbU16, rgbU16Planar);
    RunPadAndStackPlanarParity<int16_t>(529, 7, 3, 593, 15, 4, 32, NVCV_BORDER_REFLECT101, 0.f, rgbS16, rgbS16Planar);
    RunPadAndStackPlanarParity<uint16_t>(529, 7, 3, 517, 15, 4, -4, NVCV_BORDER_REFLECT101, 0.f, rgbU16, rgbU16Planar);
    RunPadAndStackPlanarParity<int16_t>(529, 7, 3, 517, 15, 4, -4, NVCV_BORDER_REFLECT101, 0.f, rgbS16, rgbS16Planar);
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStackPlanar,
                  test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, nvcv::ImageFormat,
                                  nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         borderType, borderValue,   interleavedFormat,      planarFormat
    {         1,         1,          1,        1,     65536,      0,       0,  NVCV_BORDER_REPLICATE,        0.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
    {       529,         7,          3,      517,        15,      4,      -4, NVCV_BORDER_REFLECT101,        0.f,     nvcv::FMT_RGBf16,  nvcv::FMT_RGBf16p},
    {        17,        13,          2,       23,        21,      2,       3,   NVCV_BORDER_CONSTANT,       11.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
    {        19,        11,          3,       27,        18,      1,       0, NVCV_BORDER_REFLECT101,        0.f,       nvcv::FMT_RGBA8,   nvcv::FMT_RGBA8p},
    {        15,        14,          2,       20,        19,      4,       2,  NVCV_BORDER_REPLICATE,        0.f,     nvcv::FMT_RGBf32,  nvcv::FMT_RGBf32p},
    {        17,        13,          2,       23,        21,      2,       3,   NVCV_BORDER_CONSTANT,       11.f,      nvcv::FMT_RGBf16,  nvcv::FMT_RGBf16p},
    {        15,        14,          2,       20,        19,      4,       2,  NVCV_BORDER_REPLICATE,        0.f,     nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16p},
    {       533,        14,          3,      597,        19,      4,      32,  NVCV_BORDER_REPLICATE,        0.f,     nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16p},
    {      1031,         9,          3,     1065,        17,      4,      16, NVCV_BORDER_REFLECT101,        0.f,       nvcv::FMT_RGBA8,   nvcv::FMT_RGBA8p},
    {       269,        11,          3,      301,        19,      4,      16,       NVCV_BORDER_WRAP,        0.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
    {        19,        11,          3,       13,         7,     -1,      -2,    NVCV_BORDER_REFLECT,        0.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
    {      1031,        11,          3,      533,         7,     -1,     -16,    NVCV_BORDER_REFLECT,        0.f,        nvcv::FMT_RGB8,    nvcv::FMT_RGB8p},
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
    else if (interleavedFormat == nvcv::FMT_RGBf16 || interleavedFormat == nvcv::FMT_RGBAf16)
    {
        // PadAndStack does no arithmetic, so F16 parity runs on opaque 16-bit patterns; both
        // layouts round the CONSTANT borderValue float -> half the same way.
        RunPadAndStackPlanarParity<uint16_t>(srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,
                                             borderType, borderValue, interleavedFormat, planarFormat);
    }
    else
    {
        RunPadAndStackPlanarParity<uint8_t>(srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,
                                            borderType, borderValue, interleavedFormat, planarFormat);
    }
}

// PadAndStack is pure data movement, so F16 outputs must match a 16-bit-pattern gold bit-exactly
// (EXPECT_EQ; kUlps = 0 per the HalfTestUtils.hpp policy). The CONSTANT borderValue is the only
// conversion in the operator: it rounds the float value to half once (SetAll<T>(borderValue) /
// static_cast<T>(borderValue)), mirrored here by __float2half.
static uint16_t PadAndStackValueF16(const std::vector<uint16_t> &hSrc, int2 coord, int2 size, int srcRowElems,
                                    int channels, int channel, NVCVBorderType borderType, float borderValue)
{
    if (!(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y))
    {
        if (borderType == NVCV_BORDER_CONSTANT)
        {
            const __half borderHalf = __float2half(borderValue);
            return reinterpret_cast<const uint16_t &>(borderHalf);
        }
        ApplyBorderIndex(coord, size, borderType);
    }
    return hSrc[static_cast<size_t>(coord.y) * srcRowElems + static_cast<size_t>(coord.x) * channels + channel];
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStackF16,
                  test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         NVCVBorderType, borderValue,            format
    {       212,       113,          1,      111,       132,      0,       0,   NVCV_BORDER_CONSTANT,         0.f,     nvcv::FMT_F16},
    {       533,         7,          3,      597,        15,      4,      32,   NVCV_BORDER_CONSTANT,        0.1f,     nvcv::FMT_F16},
    {        12,        13,          2,      211,       232,      0,       3,   NVCV_BORDER_CONSTANT,        0.1f,  nvcv::FMT_RGBf16}, // 0.1 is not half-representable
    {       212,       613,          4,      311,       532,      7,       7,   NVCV_BORDER_CONSTANT,       134.f, nvcv::FMT_RGBAf16},
    {        17,        13,          3,       23,        21,     -2,       3,   NVCV_BORDER_CONSTANT,        0.1f, nvcv::FMT_RGBAf16},
    {       234,       131,          2,      131,       130,     33,      22,  NVCV_BORDER_REPLICATE,         0.f,  nvcv::FMT_RGBf16},
    {       234,       131,          2,      134,       131,     53,      62,       NVCV_BORDER_WRAP,         0.f,     nvcv::FMT_F16},
    {       243,       123,          2,      132,       123,     77,      98, NVCV_BORDER_REFLECT101,         0.f, nvcv::FMT_RGBAf16},
});

// clang-format on

TEST_P(OpPadAndStackF16, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int               srcWidth    = GetParamValue<0>();
    const int               srcHeight   = GetParamValue<1>();
    const int               numBatches  = GetParamValue<2>();
    const int               dstWidth    = GetParamValue<3>();
    const int               dstHeight   = GetParamValue<4>();
    const int               topPad      = GetParamValue<5>();
    const int               leftPad     = GetParamValue<6>();
    const NVCVBorderType    borderType  = GetParamValue<7>();
    const float             borderValue = GetParamValue<8>();
    const nvcv::ImageFormat fmt         = GetParamValue<9>();

    const int channels = fmt.numChannels();

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    auto inTopData  = inTop.exportData<nvcv::TensorDataStridedCuda>();
    auto inLeftData = inLeft.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    std::vector<int> topVec(numBatches, topPad);
    std::vector<int> leftVec(numBatches, leftPad);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::default_random_engine            randEng{0}; // NOSONAR: deterministic test data, not security-sensitive.
    std::uniform_real_distribution<float> srcRand(0.f, 1.f);

    const int srcRowElems = srcWidth * channels;

    std::vector<nvcv::Image>           srcImgVec;
    std::vector<std::vector<uint16_t>> batchSrcBits;

    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, fmt);

        std::vector<float> srcFloat(static_cast<size_t>(srcRowElems) * srcHeight);
        std::ranges::generate(srcFloat, [&srcRand, &randEng]() { return srcRand(randEng); });
        const std::vector<uint8_t> srcHalf = test::FloatToHalfBytes(srcFloat);

        auto imgSrcData = srcImgVec.back().exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgSrcData, nvcv::NullOpt);
        const size_t rowBytes = static_cast<size_t>(srcRowElems) * sizeof(__half);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgSrcData->plane(0).basePtr, imgSrcData->plane(0).rowStride, srcHalf.data(),
                                    rowBytes, rowBytes, srcHeight, cudaMemcpyHostToDevice, stream));

        std::vector<uint16_t> bits(srcHalf.size() / sizeof(uint16_t));
        std::memcpy(bits.data(), srcHalf.data(), srcHalf.size());
        batchSrcBits.push_back(std::move(bits));
    }

    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, fmt);

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    cvcuda::PadAndStack padAndStackOp;
    EXPECT_NO_THROW(padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const int dstRowElems = dstWidth * channels;
    for (int b = 0; b < numBatches; ++b)
    {
        SCOPED_TRACE(b);

        std::vector<uint16_t> testBits(static_cast<size_t>(dstRowElems) * dstHeight);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testBits.data(), dstRowElems * sizeof(uint16_t), dstAccess->sampleData(b),
                                            dstAccess->rowStride(), dstRowElems * sizeof(uint16_t), dstHeight,
                                            cudaMemcpyDeviceToHost));

        std::vector<uint16_t> goldBits(testBits.size());
        const int             numValues = dstWidth * dstHeight * channels;
        for (int dstIndex = 0; dstIndex < numValues; ++dstIndex)
        {
            const int c = dstIndex % channels;
            const int x = (dstIndex / channels) % dstWidth;
            const int y = dstIndex / channels / dstWidth;
            goldBits[dstIndex]
                = PadAndStackValueF16(batchSrcBits[b], int2{x - leftPad, y - topPad}, int2{srcWidth, srcHeight},
                                      srcRowElems, channels, c, borderType, borderValue);
        }

        EXPECT_EQ(goldBits, testBits);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<typename T>
static void RunPadAndStackConstantGold(nvcv::DataType dtype, float borderValue)
{
    const int               channels            = dtype.numChannels();
    const auto              multiChannelSwizzle = channels == 3 ? nvcv::Swizzle::S_XYZ0 : nvcv::Swizzle::S_XYZW;
    const nvcv::ImageFormat format{nvcv::MemLayout::PL, dtype.dataKind(),
                                   channels == 1 ? nvcv::Swizzle::S_X000 : multiChannelSwizzle, dtype.packing()};
    constexpr int           batches   = 4;
    constexpr int           dstWidth  = 23;
    constexpr int           dstHeight = 21;
    const std::vector<int>  top{-2, 3, 1, 0};
    const std::vector<int>  left{3, -2, 2, 0};
    cudaStream_t            stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor inTop(1, {batches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {batches, 1}, nvcv::FMT_S32);
    auto         topData  = inTop.exportData<nvcv::TensorDataStridedCuda>();
    auto         leftData = inLeft.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(topData);
    ASSERT_TRUE(leftData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(topData->basePtr(), top.data(), batches * sizeof(int), cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(leftData->basePtr(), left.data(), batches * sizeof(int), cudaMemcpyHostToDevice, stream));
    std::vector<std::vector<T>> source(batches);
    nvcv::ImageBatchVarShape    input(batches);
    for (int b = 0; b < batches; ++b)
    {
        const int width     = 17 + b;
        const int height    = 13 + 2 * b;
        const int numValues = width * height * channels;
        source[b].resize(numValues);
        for (int i = 0; i < numValues; ++i)
        {
            const int value = i % 97 - (std::is_signed_v<T> ? 48 : 0);
            if constexpr (std::is_integral_v<T>)
                source[b][i] = static_cast<T>(value * (sizeof(T) == 2 ? 257 : 65537));
            else
                source[b][i] = static_cast<T>(static_cast<float>(value) * 0.25f);
        }
        nvcv::Image image({width, height}, format);
        UploadImage(image, source[b], width, height, channels, stream);
        input.pushBack(image);
    }
    nvcv::Tensor        output(batches, {dstWidth, dstHeight}, format);
    cvcuda::PadAndStack op;
    ASSERT_NO_THROW(op(stream, input, output, inTop, inLeft, NVCV_BORDER_CONSTANT, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    auto outputData = output.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(outputData);
    for (int b = 0; b < batches; ++b)
    {
        SCOPED_TRACE(b);
        const int      width  = 17 + b;
        const int      height = 13 + 2 * b;
        std::vector<T> actual;
        util::GetImageVectorFromTensor<T>(*outputData, b, actual);
        std::vector<T> expected(dstWidth * dstHeight * channels, static_cast<T>(borderValue));
        for (int i = 0; i < dstWidth * dstHeight * channels; ++i)
        {
            const int sy = i / (dstWidth * channels) - top[b];
            const int sx = i / channels % dstWidth - left[b];
            const int c  = i % channels;
            if (sx >= 0 && sx < width && sy >= 0 && sy < height)
                expected[i] = source[b][(sy * width + sx) * channels + c];
        }
        ASSERT_EQ(expected.size(), actual.size());
        EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(T)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

NVCV_TEST_SUITE_P(OpPadAndStackConstantTyped, test::ValueList<nvcv::DataType, float>{
                                                  {  nvcv::TYPE_U8,  137.5f},
                                                  { nvcv::TYPE_U16,  137.5f},
                                                  { nvcv::TYPE_S16, -137.5f},
                                                  {nvcv::TYPE_3U16,  137.5f},
                                                  {nvcv::TYPE_4U16,  137.5f},
                                                  {nvcv::TYPE_3S16, -137.5f},
                                                  {nvcv::TYPE_4S16, -137.5f},
                                                  {nvcv::TYPE_3S32, -137.5f},
                                                  {nvcv::TYPE_4S32, -137.5f},
                                                  {nvcv::TYPE_3F32,   -0.1f},
                                                  {nvcv::TYPE_4F32,   -0.1f},
});

TEST_P(OpPadAndStackConstantTyped, correct_output)
{
    const auto  dtype       = GetParamValue<0>();
    const float borderValue = GetParamValue<1>();
    if (dtype.channelType(0) == nvcv::TYPE_U8)
        RunPadAndStackConstantGold<uint8_t>(dtype, borderValue);
    else if (dtype.channelType(0) == nvcv::TYPE_U16)
        RunPadAndStackConstantGold<uint16_t>(dtype, borderValue);
    else if (dtype.channelType(0) == nvcv::TYPE_S16)
        RunPadAndStackConstantGold<int16_t>(dtype, borderValue);
    else if (dtype.channelType(0) == nvcv::TYPE_S32)
        RunPadAndStackConstantGold<int32_t>(dtype, borderValue);
    else if (dtype.channelType(0) == nvcv::TYPE_F32)
        RunPadAndStackConstantGold<float>(dtype, borderValue);
    else
        FAIL() << "Unhandled data type " << dtype;
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
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT_COVERAGE);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8p, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT_COVERAGE);
    // F16 is now supported; these rows exercise the input/output dtype-mismatch check.
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_F16, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_F16, nvcv::FMT_U8, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p, nvcv::FMT_RGB8, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U16, nvcv::FMT_S32, nvcv::FMT_S32,
                        NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_2S16, nvcv::FMT_2S16, nvcv::FMT_S32, nvcv::FMT_S32,
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

TEST(OpPadAndStack_Negative, rejects_two_channel_planar)
{
    constexpr int           numBatches = 2;
    const nvcv::ImageFormat twoChannelPlanar{NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X8)};

    nvcv::Tensor top(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor left(1, {numBatches, 1}, nvcv::FMT_S32);

    nvcv::ImageBatchVarShape input(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        input.pushBack(nvcv::Image(nvcv::Size2D{12, 13}, twoChannelPlanar));
    }
    nvcv::Tensor output(numBatches, {16, 17}, twoChannelPlanar);

    cvcuda::PadAndStack op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, input, output, top, left, NVCV_BORDER_CONSTANT, 0.f); }));
}

TEST(OpPadAndStack_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaPadAndStackCreate(nullptr));
}

TEST(OpPadAndStack_Negative, rejects_unhandled_scalar_data_kinds)
{
    cvcuda::PadAndStack op;

    for (const auto format : {nvcv::FMT_C64, nvcv::FMT_U32, nvcv::FMT_S8, nvcv::FMT_F64})
    {
        nvcv::ImageBatchVarShape input(1);
        input.pushBack(nvcv::Image(nvcv::Size2D{4, 4}, format));

        nvcv::Tensor output(1, {8, 8}, format);
        nvcv::Tensor top(1, {1, 1}, nvcv::FMT_S32);
        nvcv::Tensor left(1, {1, 1}, nvcv::FMT_S32);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(nullptr, input, output, top, left, NVCV_BORDER_CONSTANT, 0.f); }));
    }
}

// Hardening: grid.z is the output sample count and every thread reads top/left at its own sample,
// so padding tensors shorter than the batch used to be read past their end.
TEST(OpPadAndStack_Negative, rejects_short_padding_tensors)
{
    constexpr int numBatches = 3;

    nvcv::ImageBatchVarShape input(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        input.pushBack(nvcv::Image(nvcv::Size2D{12, 13}, nvcv::FMT_RGBA8));
    }
    nvcv::Tensor output(numBatches, {16, 17}, nvcv::FMT_RGBA8);

    // One entry short of the batch.
    nvcv::Tensor shortTop(1, {numBatches - 1, 1}, nvcv::FMT_S32);
    nvcv::Tensor fullLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    cvcuda::PadAndStack op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, input, output, shortTop, fullLeft, NVCV_BORDER_CONSTANT, 0.f); }));
}

TEST(OpPadAndStack_Negative, rejects_output_sample_count_mismatch)
{
    constexpr int inputSamples  = 2;
    constexpr int outputSamples = 3;

    nvcv::ImageBatchVarShape input(inputSamples);
    for (int i = 0; i < inputSamples; ++i)
    {
        input.pushBack(nvcv::Image(nvcv::Size2D{12, 13}, nvcv::FMT_RGBA8));
    }

    nvcv::Tensor output(outputSamples, {16, 17}, nvcv::FMT_RGBA8);
    nvcv::Tensor top(1, {outputSamples, 1}, nvcv::FMT_S32);
    nvcv::Tensor left(1, {outputSamples, 1}, nvcv::FMT_S32);

    cvcuda::PadAndStack op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, input, output, top, left, NVCV_BORDER_CONSTANT, 0.f); }));
}

TEST(OpPadAndStack_Negative, rejects_invalid_multi_plane_formats)
{
    constexpr int batch = 1;

    const nvcv::ImageFormat mixedPlaneDataTypes{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X16)};
    const nvcv::ImageFormat multiChannelPlanes{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8, X8_Y8)};

    nvcv::Tensor        top(1, {batch, 1}, nvcv::FMT_S32);
    nvcv::Tensor        left(1, {batch, 1}, nvcv::FMT_S32);
    cvcuda::PadAndStack op;

    const auto expectInvalidFormat = [&](nvcv::ImageFormat inputFormat, nvcv::ImageFormat outputFormat)
    {
        nvcv::ImageBatchVarShape input(batch);
        input.pushBack(nvcv::Image(nvcv::Size2D{8, 8}, inputFormat));
        nvcv::Tensor output(batch, {10, 10}, outputFormat);

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(nullptr, input, output, top, left, NVCV_BORDER_CONSTANT, 0.f); }));
    };

    expectInvalidFormat(mixedPlaneDataTypes, nvcv::FMT_U8);
    expectInvalidFormat(multiChannelPlanes, nvcv::FMT_RGBA8);
}
