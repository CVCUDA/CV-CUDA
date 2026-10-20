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

#include <common/BorderUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cuda_fp16.h>
#include <cvcuda/OpCopyMakeBorder.hpp>
#include <nvcv/DataLayout.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <cstring>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

namespace test = nvcv::test;

inline bool IsInside(int2 coords, int2 size)
{
    return coords.x >= 0 && coords.x < size.x && coords.y >= 0 && coords.y < size.y;
}

inline void ApplyBorderIndex(int2 &coords, int2 size, NVCVBorderType borderType)
{
    if (borderType == NVCV_BORDER_REPLICATE)
        test::ReplicateBorderIndex(coords, size);
    else if (borderType == NVCV_BORDER_WRAP)
        test::WrapBorderIndex(coords, size);
    else if (borderType == NVCV_BORDER_REFLECT)
        test::ReflectBorderIndex(coords, size);
    else if (borderType == NVCV_BORDER_REFLECT101)
        test::Reflect101BorderIndex(coords, size);
}

inline int ScaleDimension(int value, double scale)
{
    return static_cast<int>(static_cast<double>(value) * scale);
}

inline float BorderComponent(const float4 &borderValue, int channel)
{
    switch (channel)
    {
    case 0:
        return borderValue.x;
    case 1:
        return borderValue.y;
    case 2:
        return borderValue.z;
    case 3:
        return borderValue.w;
    default:
        return 0.0f;
    }
}

template<typename T, typename Size>
static int ElementCountFromBytes(Size numBytes)
{
    return static_cast<int>(numBytes / sizeof(T));
}

template<typename T>
static void FillRandom(std::vector<T> &values, std::uniform_int_distribution<uint8_t> &srcRand,
                       std::default_random_engine &randEng)
{
    for (T &value : values)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            value = static_cast<T>(srcRand(randEng)) / 255.0f;
        }
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>)
        {
            value = static_cast<T>(std::uniform_int_distribution<int>{std::numeric_limits<T>::lowest(),
                                                                      std::numeric_limits<T>::max()}(randEng));
        }
        else
        {
            value = static_cast<T>(srcRand(randEng));
        }
    }
}

template<typename T, class GetSrcOffset>
static T CopyMakeBorderValue(const std::vector<T> &hSrc, int2 coords, int2 size, int channel, NVCVBorderType borderType,
                             const float4 &borderValue, GetSrcOffset getSrcOffset)
{
    if (IsInside(coords, size))
    {
        return hSrc[getSrcOffset(coords, channel)];
    }

    if (borderType == NVCV_BORDER_CONSTANT)
    {
        return static_cast<T>(BorderComponent(borderValue, channel));
    }

    ApplyBorderIndex(coords, size, borderType);
    return hSrc[getSrcOffset(coords, channel)];
}

template<typename T, class GetSrcOffset>
static void CopyMakeBorderPixel(std::vector<T> &hDst, int dstOffset, const std::vector<T> &hSrc, int2 coords, int2 size,
                                int channels, NVCVBorderType borderType, const float4 &borderValue,
                                GetSrcOffset getSrcOffset)
{
    for (int dk = 0; dk < channels; dk++)
    {
        hDst[dstOffset + dk] = CopyMakeBorderValue(hSrc, coords, size, dk, borderType, borderValue, getSrcOffset);
    }
}

template<typename T>
static void CopyMakeBorder(std::vector<T> &hDst, const std::vector<T> &hSrc,
                           const nvcv::TensorDataAccessStridedImagePlanar &dDstData, const int srcWidth,
                           const int srcHeight, const int srcRowStride, const int srcPixPitch, const int srcImgPitch,
                           const int top, const int left, const NVCVBorderType borderType, const float4 borderValue)
{
    int dstPixPitch  = dDstData.numChannels();
    int dstRowStride = ElementCountFromBytes<T>(dDstData.rowStride());
    int dstImgPitch  = ElementCountFromBytes<T>(dDstData.sampleStride());

    int2 coords{};
    int2 size{srcWidth, srcHeight};
    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coords.y       = di - top;
            auto srcOffset = [db, srcImgPitch, srcRowStride, srcPixPitch](int2 srcCoords, int channel)
            {
                return db * srcImgPitch + srcCoords.y * srcRowStride + srcCoords.x * srcPixPitch + channel;
            };

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coords.x = dj - left;

                int dstOffset = db * dstImgPitch + di * dstRowStride + dj * dstPixPitch;
                CopyMakeBorderPixel(hDst, dstOffset, hSrc, coords, size, dstPixPitch, borderType, borderValue,
                                    srcOffset);
            }
        }
    }
}

template<typename T>
static void CopyMakeBorder(std::vector<std::vector<T>> &hBatchDst, const std::vector<std::vector<T>> &hBatchSrc,
                           const std::vector<nvcv::Image> &dBatchDstData, const std::vector<nvcv::Image> &dBatchSrcData,
                           const std::vector<int> &top, const std::vector<int> &left, const NVCVBorderType borderType,
                           const float4 borderValue)
{
    int2 coords;
    for (size_t db = 0; db < dBatchDstData.size(); db++)
    {
        auto &hDst         = hBatchDst[db];
        auto &dDst         = dBatchDstData[db];
        auto  imgDstData   = dDst.exportData<nvcv::ImageDataStridedCuda>();
        int   dstRowStride = ElementCountFromBytes<T>(imgDstData->plane(0).rowStride);
        int   dstPixPitch  = dDst.format().numChannels();

        auto &hSrc       = hBatchSrc[db];
        auto &dSrc       = dBatchSrcData[db];
        auto  imgSrcData = dSrc.exportData<nvcv::ImageDataStridedCuda>();
        int   rowStride  = ElementCountFromBytes<T>(imgSrcData->plane(0).rowStride);
        int   pixPitch   = imgSrcData->format().numChannels();

        auto imgSize = dBatchSrcData[db].size();
        int2 size{imgSize.w, imgSize.h};
        auto srcOffset = [rowStride, pixPitch](int2 srcCoords, int channel)
        {
            return srcCoords.y * rowStride + srcCoords.x * pixPitch + channel;
        };

        for (int di = 0; di < imgDstData->plane(0).height; di++) //for rows
        {
            coords.y = di - top[db];

            for (int dj = 0; dj < imgDstData->plane(0).width; dj++) // for columns
            {
                coords.x = dj - left[db];

                int dstOffset = di * dstRowStride + dj * dstPixPitch;
                CopyMakeBorderPixel(hDst, dstOffset, hSrc, coords, size, dstPixPitch, borderType, borderValue,
                                    srcOffset);
            }
        }
    }
}

template<typename T>
static void CopyMakeBorder(std::vector<T> &hDst, const std::vector<std::vector<T>> &hBatchSrc,
                           const nvcv::TensorDataAccessStridedImagePlanar &dDstData,
                           const std::vector<nvcv::Image> &dBatchSrcData, const std::vector<int> &top,
                           const std::vector<int> &left, const NVCVBorderType borderType, const float4 borderValue)
{
    int dstPixPitch  = dDstData.numChannels();
    int dstRowStride = ElementCountFromBytes<T>(dDstData.rowStride());
    int dstImgPitch  = ElementCountFromBytes<T>(dDstData.sampleStride());

    int2 coords;
    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        auto &hSrc       = hBatchSrc[db];
        auto  imgSrcData = dBatchSrcData[db].exportData<nvcv::ImageDataStridedCuda>();
        int   rowStride  = ElementCountFromBytes<T>(imgSrcData->plane(0).rowStride);
        int   pixPitch   = imgSrcData->format().numChannels();

        auto imgSize = dBatchSrcData[db].size();
        int2 size{imgSize.w, imgSize.h};
        auto srcOffset = [rowStride, pixPitch](int2 srcCoords, int channel)
        {
            return srcCoords.y * rowStride + srcCoords.x * pixPitch + channel;
        };

        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coords.y = di - top[db];

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coords.x = dj - left[db];

                int dstOffset = db * dstImgPitch + di * dstRowStride + dj * dstPixPitch;
                CopyMakeBorderPixel(hDst, dstOffset, hSrc, coords, size, dstPixPitch, borderType, borderValue,
                                    srcOffset);
            }
        }
    }
}

const nvcv::ImageFormat kU8C2{nvcv::MemLayout::PL, nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XY00,
                              nvcv::TYPE_2U8.packing()};
const nvcv::ImageFormat kU16C3{nvcv::MemLayout::PL, nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZ0,
                               nvcv::TYPE_3U16.packing()};
const nvcv::ImageFormat kU16C4{nvcv::MemLayout::PL, nvcv::DataKind::UNSIGNED, nvcv::Swizzle::S_XYZW,
                               nvcv::TYPE_4U16.packing()};
const nvcv::ImageFormat kS16C3{nvcv::MemLayout::PL, nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZ0,
                               nvcv::TYPE_3S16.packing()};
const nvcv::ImageFormat kS16C4{nvcv::MemLayout::PL, nvcv::DataKind::SIGNED, nvcv::Swizzle::S_XYZW,
                               nvcv::TYPE_4S16.packing()};

// clang-format off

NVCV_TEST_SUITE_P(OpCopyMakeBorder, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, float, float, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad,         NVCVBorderType,    bValue1, bValue2, bValue3, bValue4, ImageFormat
    {       212,       113,          1,        0,         0,      0,       0,   NVCV_BORDER_CONSTANT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        12,        13,          2,       12,        16,      0,       3,   NVCV_BORDER_CONSTANT,       12.f,   100.f,   245.f,     0.f, nvcv::FMT_RGB8},
    {       212,       113,          3,        0,       113,      5,       0,   NVCV_BORDER_CONSTANT,       13.f,     5.f,     4.f,     0.f, nvcv::FMT_RGB8},
    {       212,       613,          4,       19,        20,      7,       7,   NVCV_BORDER_CONSTANT,      255.f,   255.f,   255.f,     0.f, nvcv::FMT_RGB8},

    {       234,       131,          2,       44,        55,     33,      22,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {       234,       131,          2,       33,        20,     41,      42,    NVCV_BORDER_REFLECT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBA8},
    {       234,       131,          2,      100,        85,     53,      62,       NVCV_BORDER_WRAP,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {       243,       123,          2,       56,       123,     77,      98, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf32},

    {        31,        17,          3,        2,         3,      5,       4,   NVCV_BORDER_CONSTANT,       37.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        35,        19,          3,        3,         2,      4,       6,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        37,        23,          3,        4,         5,      3,       2, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        41,        27,          2,        5,         3,      4,       6, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        29,        21,          3,        5,         4,      2,       3, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_F32},
    {        31,        17,          3,        2,         3,      5,       4,   NVCV_BORDER_CONSTANT,      -37.f,     0.f,     0.f,     0.f,  nvcv::FMT_S8},
    {        33,        19,          2,        4,         3,      2,       5,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f,  nvcv::FMT_S8},
    {        35,        23,          3,        3,         4,      6,       2,   NVCV_BORDER_CONSTANT,   100000.f,     0.f,     0.f,     0.f, nvcv::FMT_S32},
    {        37,        29,          2,        5,         2,      3,       4, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_S32},

    // F16 is a copy plus a single float->half constant-border conversion; host and device both
    // round to nearest even, so results are deterministic and compare bit-exactly via EXPECT_EQ.
    // The fractional border components exercise that rounding parity.
    {        55,        33,          2,        4,         5,      3,       2,   NVCV_BORDER_CONSTANT,      12.5f,  100.7f,   245.f,     0.f, nvcv::FMT_RGBf16},
    {        64,        21,          3,        3,         2,      4,       6,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_F16},
    {        23,        54,          2,        6,         2,      4,       3, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf16},

    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,       13.f,    57.f,   211.f,    83.f, nvcv::FMT_RGBA8},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,       13.f,    57.f,   211.f,    83.f, nvcv::FMT_RGBA8},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,      12.5f,  100.7f,   245.f,     0.f, nvcv::FMT_RGBf16},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,      12.5f,  100.7f,   245.f,   3.25f, nvcv::FMT_RGBAf16},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,      12.5f,  100.7f,   245.f,   3.25f, nvcv::FMT_RGBAf16},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,     13.25f,   -3.5f, 2048.75f,     0.f, nvcv::FMT_RGBf32},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,     13.25f,   -3.5f, 2048.75f,     0.f, nvcv::FMT_RGBf32},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,     13.25f,   -3.5f, 2048.75f,  255.5f, nvcv::FMT_RGBAf32},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,     13.25f,   -3.5f, 2048.75f,  255.5f, nvcv::FMT_RGBAf32},

    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,       13.f,    57.f,     0.f,     0.f, kU8C2},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,       13.f,    57.f,     0.f,     0.f, kU8C2},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,       13.f, 50003.f, 65530.f,     0.f, kU16C3},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,       13.f, 50003.f, 65530.f,     0.f, kU16C3},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,       13.f, 50003.f, 65530.f, 40007.f, kU16C4},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,       13.f, 50003.f, 65530.f, 40007.f, kU16C4},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,   -30001.f,  1234.f, 32003.f,     0.f, kS16C3},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,   -30001.f,  1234.f, 32003.f,     0.f, kS16C3},
    {        64,        24,          2,        4,         4,      4,       4,   NVCV_BORDER_CONSTANT,   -30001.f,  1234.f, 32003.f,   -17.f, kS16C4},
    {        31,        17,          3,        2,         5,      3,       3,   NVCV_BORDER_CONSTANT,   -30001.f,  1234.f, 32003.f,   -17.f, kS16C4},

});

// clang-format on

template<typename T>
void StartTest(int srcWidth, int srcHeight, int numBatches, int topPad, int bottomPad, int leftPad, int rightPad,
               NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int dstWidth  = srcWidth + leftPad + rightPad;
    int dstHeight = srcHeight + topPad + bottomPad;

    std::vector<T> srcVec;

    nvcv::Tensor imgSrc(numBatches, {srcWidth, srcHeight}, format);
    auto         srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcData);
    int srcBufSize = ElementCountFromBytes<T>(srcAccess->sampleStride()) * static_cast<int>(srcAccess->numSamples());
    srcVec.resize(srcBufSize);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    FillRandom(srcVec, srcRand, randEng);

    // Copy each input image with random data to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->basePtr(), srcVec.data(), srcBufSize * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);
    auto         dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);
    int dstBufSize = ElementCountFromBytes<T>(dstAccess->sampleStride()) * static_cast<int>(dstAccess->numSamples());
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(T), stream));

    std::vector<T> testVec(dstBufSize);
    std::vector<T> goldVec(dstBufSize);

    int srcPixPitch  = srcAccess->numChannels();
    int srcRowStride = ElementCountFromBytes<T>(srcAccess->rowStride());
    int srcImgPitch  = ElementCountFromBytes<T>(srcAccess->sampleStride());

    // Generate gold result
    CopyMakeBorder(goldVec, srcVec, *dstAccess, srcWidth, srcHeight, srcRowStride, srcPixPitch, srcImgPitch, topPad,
                   leftPad, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgSrc, imgDst, topPad, leftPad, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize * sizeof(T), cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec);
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}

TEST_P(OpCopyMakeBorder, tensor_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    const auto dtype = format.planeDataType(0).channelType(0);
    if (dtype == nvcv::TYPE_U8)
        StartTest<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (dtype == nvcv::TYPE_F32)
        StartTest<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType, borderValue,
                         format);
    else if (dtype == nvcv::TYPE_F16)
        StartTest<__half>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                          borderValue, format);
    else if (dtype == nvcv::TYPE_U16)
        StartTest<uint16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                            borderValue, format);
    else if (dtype == nvcv::TYPE_S8)
        StartTest<int8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                          borderValue, format);
    else if (dtype == nvcv::TYPE_S32)
        StartTest<int32_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (dtype == nvcv::TYPE_S16)
        StartTest<int16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else
        FAIL() << "Unsupported test format: " << format;
}

// clang-format off

// Exact pad sizes distinguish affine carrier strips from alignment and multi-fold fallbacks.
// The independent StartTest oracle compares the entire destination, including row padding.
NVCV_TEST_SUITE_P(OpCopyMakeBorderEdge, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, float, float, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad,         NVCVBorderType,    bValue1, bValue2, bValue3, bValue4, ImageFormat

    // Aligned right origins and both wide and shifted-middle strips, with body and tail bytes.
    {        48,         5,          2,        3,         4,      16,       17, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,         5,          2,        3,         4,      11,        7, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        48,         5,          2,        3,         4,      16,       17, NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,         5,          2,        3,         4,      11,        7, NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        48,         5,          2,        3,         4,      16,       17, NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,         5,          2,        3,         4,      11,        7, NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        48,         5,          2,        3,         4,      16,       17, NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,         5,          2,        3,         4,      11,        7, NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    // Unaligned right origin declines the permutation carrier; CONSTANT remains on its own path.
    {        37,         5,          2,        3,         4,      10,        7, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,         5,          2,        3,         4,      15,        7, NVCV_BORDER_CONSTANT,      7.f,    91.f,   201.f,     0.f, nvcv::FMT_RGB8},

    // Remaining pixel sizes at assorted phases: 1, 2, 4, 6, 8, 12 and 16 bytes.
    {        33,        17,          2,        3,         5,       1,        6, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        33,        17,          2,        3,         5,       7,        6,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        33,        17,          2,        3,         5,      15,        6,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        33,        17,          2,        3,         5,      13,        6,  NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        33,        17,          2,        3,         5,      11,        6,   NVCV_BORDER_CONSTANT,      42.f,    0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        29,        13,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U16},
    {        29,        13,          2,        4,         3,       5,        5,  NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U16},
    {        29,        13,          2,        4,         3,       3,        5,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_S16},
    {        29,        13,          2,        4,         3,       7,        5,   NVCV_BORDER_CONSTANT,      33.f,    0.f,     0.f,     0.f, nvcv::FMT_S16},
    {        27,        15,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_F16},
    {        27,        15,          2,        4,         3,       3,        5,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_F16},
    {        23,        19,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBA8},
    {        23,        19,          2,        4,         3,       3,        5,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBA8},
    {        23,        19,          2,        4,         3,       2,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_F32},
    {        23,        19,          2,        4,         3,       5,        5,  NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_F32},
    {        21,        11,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf16},
    {        21,        11,          2,        4,         3,       2,        5,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf16},
    {        21,        11,          2,        4,         3,       3,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf16},
    {        19,         9,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {        19,         9,          2,        4,         3,       2,        5,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {        19,         9,          2,        4,         3,       3,        5,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {        19,         9,          2,        4,         3,       1,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf32},

    // Zero-width borders, one side at a time, then whole axes, then a pure copy.
    {        37,        11,          2,        0,         5,       3,        7, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,        11,          2,        4,         0,       3,        7,  NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,        11,          2,        4,         5,       0,        7,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        37,        11,          2,        4,         5,       3,        0,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        33,        17,          2,        4,         5,       0,        0, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        33,        17,          2,        0,         0,       3,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {        19,         9,          2,        0,         0,       0,        0, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {        37,        11,          2,        0,         0,       0,        0,   NVCV_BORDER_CONSTANT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},

    // Border exactly at, and past, the source extent: a single reflect/wrap fold stops being enough.
    {        16,        12,          2,       11,        12,      15,       16, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        16,        12,          2,       12,        13,      16,       17, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        16,        12,          2,       12,        13,      16,       17,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        16,        12,          2,       12,        13,      16,       17,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {         5,         4,          2,        9,        11,      13,        7, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {         6,         5,          2,       12,         8,       9,       14,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {         7,         3,          2,       10,         5,      17,        6,       NVCV_BORDER_WRAP,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {         4,         4,          2,       13,         2,       9,       11,  NVCV_BORDER_REPLICATE,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf32},

    // Degenerate 1-pixel extents (REFLECT101 has a dedicated size==1 branch).
    {         1,         1,          2,        3,         2,       4,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_U8},
    {         1,         9,          2,        3,         2,       4,        5, NVCV_BORDER_REFLECT101,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {         9,         1,          2,        3,         2,       4,        5,    NVCV_BORDER_REFLECT,     0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {         1,         1,          2,        3,         2,       4,        5,   NVCV_BORDER_CONSTANT,     9.f,    19.f,    29.f,     0.f, nvcv::FMT_RGB8},


});

// clang-format on

TEST_P(OpCopyMakeBorderEdge, tensor_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    if (nvcv::FMT_U8 == format || nvcv::FMT_RGB8 == format || nvcv::FMT_RGBA8 == format)
        StartTest<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (nvcv::FMT_U16 == format)
        StartTest<uint16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                            borderValue, format);
    else if (nvcv::FMT_S16 == format)
        StartTest<int16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (nvcv::FMT_F32 == format || nvcv::FMT_RGBf32 == format || nvcv::FMT_RGBAf32 == format)
        StartTest<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType, borderValue,
                         format);
    else if (nvcv::FMT_F16 == format || nvcv::FMT_RGBf16 == format || nvcv::FMT_RGBAf16 == format)
        StartTest<__half>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                          borderValue, format);
    else
        FAIL() << "unhandled image format in OpCopyMakeBorderEdge";
}

template<typename T>
void StartTestVarShape(int srcWidthBase, int srcHeightBase, int numBatches, int topPad, int bottomPad, int leftPad,
                       int rightPad, NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::default_random_engine    randEng{0};
    std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.2));
    std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8), ScaleDimension(srcHeightBase, 1.2));

    std::uniform_int_distribution rndTop(ScaleDimension(topPad, 0.8), ScaleDimension(topPad, 1.2));
    std::uniform_int_distribution rndBottom(ScaleDimension(bottomPad, 0.8), ScaleDimension(bottomPad, 1.2));

    std::uniform_int_distribution rndLeft(ScaleDimension(leftPad, 0.8), ScaleDimension(leftPad, 1.2));
    std::uniform_int_distribution rndRight(ScaleDimension(rightPad, 0.8), ScaleDimension(rightPad, 1.2));

    //Prepare input and output buffer
    std::vector<nvcv::Image>    imgSrcVec;
    std::vector<nvcv::Image>    imgDstVec;
    std::vector<std::vector<T>> hImgSrcVec;
    std::vector<std::vector<T>> hImgDstVec;
    std::vector<std::vector<T>> batchGoldVec;
    std::vector<int>            topVec(numBatches);
    std::vector<int>            leftVec(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        int srcWidth  = rndSrcWidth(randEng);
        int srcHeight = rndSrcHeight(randEng);
        int top       = rndTop(randEng);
        int left      = rndLeft(randEng);
        int bottom    = rndBottom(randEng);
        int right     = rndRight(randEng);

        int dstWidth  = srcWidth + left + right;
        int dstHeight = srcHeight + top + bottom;
        topVec[i]     = top;
        leftVec[i]    = left;
        //prepare input buffers
        imgSrcVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, format);

        auto imgSrcData   = imgSrcVec.back().exportData<nvcv::ImageDataStridedCuda>();
        int  srcStride    = imgSrcData->plane(0).rowStride;
        int  srcRowStride = ElementCountFromBytes<T>(srcStride);
        int  srcBufSize   = srcRowStride * imgSrcData->plane(0).height;

        std::vector<T>                         srcVec(srcBufSize);
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        FillRandom(srcVec, srcRand, randEng);

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(), srcBufSize * sizeof(T),
                                               cudaMemcpyHostToDevice, stream));
        hImgSrcVec.push_back(std::move(srcVec));

        //prepare output Buffers
        imgDstVec.emplace_back(nvcv::Size2D{dstWidth, dstHeight}, format);
        auto imgDstData   = imgDstVec.back().exportData<nvcv::ImageDataStridedCuda>();
        int  dstStride    = imgDstData->plane(0).rowStride;
        int  dstRowStride = ElementCountFromBytes<T>(dstStride);
        int  dstBufSize   = dstRowStride * imgDstData->plane(0).height;

        std::vector<T> dstVec(dstBufSize);
        std::vector<T> goldVec(dstBufSize);
        hImgDstVec.push_back(std::move(dstVec));
        batchGoldVec.push_back(std::move(goldVec));
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(imgSrcVec.begin(), imgSrcVec.end());
    nvcv::ImageBatchVarShape imgBatchDst(numBatches);
    imgBatchDst.pushBack(imgDstVec.begin(), imgDstVec.end());

    // Prepare top and left inputs
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

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    CopyMakeBorder(batchGoldVec, hImgSrcVec, imgDstVec, imgSrcVec, topVec, leftVec, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgBatchSrc, imgBatchDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    int idx = 0;
    for (auto &img : imgBatchDst)
    {
        auto &testVec   = hImgDstVec[idx];
        auto &goldVec   = batchGoldVec[idx];
        auto  imgAccess = img.exportData<nvcv::ImageDataStridedCuda>();

        ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), imgAccess->plane(0).basePtr, testVec.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
        for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
        test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
        test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
        if (goldVec != testVec)
        {
            test::DebugPrintDiff(testVec, goldVec);
        }
#endif

        const int dstRowStride = ElementCountFromBytes<T>(imgAccess->plane(0).rowStride);
        const int validRowSize = ElementCountFromBytes<T>(img.size().w * format.planePixelStrideBytes(0));
        for (int row = 0; row < img.size().h; ++row)
        {
            SCOPED_TRACE(row);
            EXPECT_TRUE(std::equal(goldVec.begin() + row * dstRowStride,
                                   goldVec.begin() + row * dstRowStride + validRowSize,
                                   testVec.begin() + row * dstRowStride));
        }
        idx++;
    }
}

TEST_P(OpCopyMakeBorder, varshape_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    const auto dtype = format.planeDataType(0).channelType(0);
    if (dtype == nvcv::TYPE_U8)
        StartTestVarShape<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                   borderValue, format);
    else if (dtype == nvcv::TYPE_F32)
        StartTestVarShape<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                 borderValue, format);
    else if (dtype == nvcv::TYPE_F16)
        StartTestVarShape<__half>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                  borderValue, format);
    else if (dtype == nvcv::TYPE_U16)
        StartTestVarShape<uint16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                    borderValue, format);
    else if (dtype == nvcv::TYPE_S8)
        StartTestVarShape<int8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                  borderValue, format);
    else if (dtype == nvcv::TYPE_S32)
        StartTestVarShape<int32_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                   borderValue, format);
    else if (dtype == nvcv::TYPE_S16)
        StartTestVarShape<int16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                   borderValue, format);
    else
        FAIL() << "Unsupported test format: " << format;
}

TEST(OpCopyMakeBorder, varshape_16bit_integer_correct_output)
{
    const float4 u16BorderValue{4095.f, 1024.f, 257.f, 0.f};
    const float4 s16BorderValue{-1234.f, 234.f, -56.f, 0.f};

    StartTestVarShape<uint16_t>(37, 23, 3, 4, 5, 3, 2, NVCV_BORDER_CONSTANT, u16BorderValue, nvcv::FMT_U16);
    StartTestVarShape<int16_t>(31, 19, 2, 3, 2, 4, 3, NVCV_BORDER_CONSTANT, s16BorderValue, nvcv::FMT_S16);
}

template<typename T>
void StartTestStack(int srcWidthBase, int srcHeightBase, int numBatches, int topPad, int bottomPad, int leftPad,
                    int rightPad, NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    //make sure the random pad settings did not exceed the limit.
    int dstWidth  = ScaleDimension(srcWidthBase + leftPad + rightPad, 1.2);
    int dstHeight = ScaleDimension(srcHeightBase + topPad + bottomPad, 1.2);

    std::default_random_engine    randEng{0};
    std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.2));
    std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8), ScaleDimension(srcHeightBase, 1.2));

    std::uniform_int_distribution rndTop(ScaleDimension(topPad, 0.8), ScaleDimension(topPad, 1.2));
    std::uniform_int_distribution rndLeft(ScaleDimension(leftPad, 0.8), ScaleDimension(leftPad, 1.2));

    //Prepare input and output buffer
    std::vector<nvcv::Image>    imgSrcVec;
    std::vector<nvcv::Image>    imgDstVec;
    std::vector<std::vector<T>> hImgSrcVec;
    std::vector<int>            topVec(numBatches);
    std::vector<int>            leftVec(numBatches);
    for (int i = 0; i < numBatches; ++i)
    {
        int srcWidth  = rndSrcWidth(randEng);
        int srcHeight = rndSrcHeight(randEng);
        int top       = rndTop(randEng);
        int left      = rndLeft(randEng);

        topVec[i]  = top;
        leftVec[i] = left;
        //prepare input buffers
        imgSrcVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, format);

        auto imgSrcData   = imgSrcVec.back().exportData<nvcv::ImageDataStridedCuda>();
        int  srcStride    = imgSrcData->plane(0).rowStride;
        int  srcRowStride = ElementCountFromBytes<T>(srcStride);
        int  srcBufSize   = srcRowStride * imgSrcData->plane(0).height;

        std::vector<T>                         srcVec(srcBufSize);
        std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
        FillRandom(srcVec, srcRand, randEng);

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).basePtr, srcVec.data(), srcBufSize * sizeof(T),
                                               cudaMemcpyHostToDevice, stream));
        hImgSrcVec.push_back(std::move(srcVec));
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(imgSrcVec.begin(), imgSrcVec.end());

    //prepare output buffer
    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);
    auto         dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);
    int dstBufSize = ElementCountFromBytes<T>(dstAccess->sampleStride()) * static_cast<int>(dstAccess->numSamples());
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->basePtr(), 0, dstBufSize * sizeof(T), stream));

    std::vector<T> testVec(dstBufSize);
    std::vector<T> goldVec(dstBufSize);

    // Prepare top and left inputs
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

    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->basePtr(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->basePtr(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    CopyMakeBorder(goldVec, hImgSrcVec, *dstAccess, imgSrcVec, topVec, leftVec, borderType, borderValue);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(testVec.data(), dstData->basePtr(), dstBufSize * sizeof(T), cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcStride / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowStride() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowStride() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec);
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}

TEST_P(OpCopyMakeBorder, stack_correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int bottomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    const auto dtype = format.planeDataType(0).channelType(0);
    if (dtype == nvcv::TYPE_U8)
        StartTestStack<uint8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                borderValue, format);
    else if (dtype == nvcv::TYPE_F32)
        StartTestStack<float>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                              borderValue, format);
    else if (dtype == nvcv::TYPE_F16)
        StartTestStack<__half>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                               borderValue, format);
    else if (dtype == nvcv::TYPE_U16)
        StartTestStack<uint16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                 borderValue, format);
    else if (dtype == nvcv::TYPE_S8)
        StartTestStack<int8_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                               borderValue, format);
    else if (dtype == nvcv::TYPE_S32)
        StartTestStack<int32_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                borderValue, format);
    else if (dtype == nvcv::TYPE_S16)
        StartTestStack<int16_t>(srcWidth, srcHeight, numBatches, topPad, bottomPad, leftPad, rightPad, borderType,
                                borderValue, format);
    else
        FAIL() << "Unsupported test format: " << format;
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// CopyMakeBorder remaps each output pixel independently of its channel, including per-channel
// constant border values. A planar input must therefore match the equivalent interleaved result
// exactly once the planar output is re-interleaved.
// =============================================================================

namespace {

void FillPadTensor(nvcv::Tensor &tensor, int value, int count)
{
    auto data = tensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(data, nullptr);

    std::vector<int> values(static_cast<size_t>(count), value);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(data->basePtr(), values.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice));
}

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int top, int bottom, int left, int right, NVCVBorderType borderType,
                               const float4 &borderValue, int numImages)
{
    test::planar::RunTensorParity(planarFmt, interleavedFmt, srcW, srcH, srcW + left + right, srcH + top + bottom,
                                  numImages,
                                  [top, left, borderType, borderValue](cudaStream_t stream, const nvcv::Tensor &src,
                                                                       const nvcv::Tensor &dst, nvcv::ImageFormat)
                                  {
                                      cvcuda::CopyMakeBorder op;
                                      EXPECT_NO_THROW(op(stream, src, dst, top, left, borderType, borderValue));
                                  });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int top, int bottom, int left, int right, NVCVBorderType borderType,
                                 const float4 &borderValue, int numImages)
{
    nvcv::Tensor topTensor(1, {numImages, 1}, nvcv::FMT_S32);
    nvcv::Tensor leftTensor(1, {numImages, 1}, nvcv::FMT_S32);
    FillPadTensor(topTensor, top, numImages);
    FillPadTensor(leftTensor, left, numImages);

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, srcW, srcH, srcW + left + right, srcH + top + bottom, numImages,
        [&topTensor, &leftTensor, borderType, borderValue](cudaStream_t stream, const nvcv::ImageBatchVarShape &src,
                                                           const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::CopyMakeBorder op;
            EXPECT_NO_THROW(op(stream, src, dst, topTensor, leftTensor, borderType, borderValue));
        });
}

void RunPlanarParityStackCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                              int top, int bottom, int left, int right, NVCVBorderType borderType,
                              const float4 &borderValue, int numImages)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int channels  = planarFmt.numChannels();
    const int elemSize  = planarFmt.planePixelStrideBytes(0);
    const int srcRowStr = srcW * channels * elemSize;
    const int dstW      = srcW + left + right;
    const int dstH      = srcH + top + bottom;
    const int dstRowStr = dstW * channels * elemSize;

    std::vector<nvcv::Image> srcI;
    std::vector<nvcv::Image> srcP;
    for (int i = 0; i < numImages; ++i)
    {
        srcI.emplace_back(nvcv::Size2D{srcW, srcH}, interleavedFmt);
        srcP.emplace_back(nvcv::Size2D{srcW, srcH}, planarFmt);
    }

    nvcv::ImageBatchVarShape batchSrcI(numImages);
    nvcv::ImageBatchVarShape batchSrcP(numImages);
    batchSrcI.pushBack(srcI.begin(), srcI.end());
    batchSrcP.pushBack(srcP.begin(), srcP.end());

    for (int i = 0; i < numImages; ++i)
    {
        std::vector<uint8_t> hwc(srcH * srcRowStr);
        test::planar::FillDeterministicValues(hwc, static_cast<size_t>(i) * 101 + 17, planarFmt.planeDataType(0));

        auto idata = srcI[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(idata->plane(0).basePtr, idata->plane(0).rowStride, hwc.data(), srcRowStr,
                                            srcRowStr, srcH, cudaMemcpyHostToDevice));

        auto      planes     = test::planar::DeinterleaveToPlanes(hwc, srcW, srcH, channels, elemSize);
        auto      pdata      = srcP[i].exportData<nvcv::ImageDataStridedCuda>();
        const int planeBytes = srcW * srcH * elemSize;
        ASSERT_EQ(pdata->numPlanes(), channels);
        for (int c = 0; c < channels; ++c)
        {
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(pdata->plane(c).basePtr, pdata->plane(c).rowStride, planes.data() + c * planeBytes,
                                   srcW * elemSize, srcW * elemSize, srcH, cudaMemcpyHostToDevice));
        }
    }

    nvcv::Tensor dstI(numImages, {dstW, dstH}, interleavedFmt);
    nvcv::Tensor dstP(numImages, {dstW, dstH}, planarFmt);

    nvcv::Tensor topTensor(1, {numImages, 1}, nvcv::FMT_S32);
    nvcv::Tensor leftTensor(1, {numImages, 1}, nvcv::FMT_S32);
    FillPadTensor(topTensor, top, numImages);
    FillPadTensor(leftTensor, left, numImages);

    cvcuda::CopyMakeBorder op;
    EXPECT_NO_THROW(op(stream, batchSrcI, dstI, topTensor, leftTensor, borderType, borderValue));
    EXPECT_NO_THROW(op(stream, batchSrcP, dstP, topTensor, leftTensor, borderType, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto dstIData = dstI.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPData = dstP.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(dstIData && dstPData);

    auto dstIAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstIData);
    auto dstPAcc = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPData);
    ASSERT_TRUE(dstIAcc && dstPAcc);

    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);
        auto gpuInter    = test::planar::DownloadInterleavedSample(*dstIAcc, i, dstW, dstH, dstRowStr);
        auto planesOut   = test::planar::DownloadPlanarSample(*dstPAcc, i, dstW, dstH, channels, elemSize);
        auto planarInter = test::planar::InterleaveFromPlanes(planesOut, dstW, dstH, channels, elemSize);

        EXPECT_EQ(gpuInter, planarInter);
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

float4 MakeBorderValue(float x, float y, float z, float w)
{
    return float4{x, y, z, w};
}

} // namespace

// clang-format off
NVCV_TEST_SUITE_P(OpCopyMakeBorderPlanar,
                  test::ValueList<int, int, int, int, int, int, NVCVBorderType, float, float, float, float, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    // srcW, srcH, top, bottom, left, right,             border,      b0,    b1,    b2,    b3, images,          planarFmt,      interleavedFmt
    {   24,   17,   3,      2,    4,     1,   NVCV_BORDER_CONSTANT,  12.f, 100.f, 245.f,   0.f,      2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   31,   19,   0,      5,    2,     3,  NVCV_BORDER_REPLICATE,   0.f,   0.f,   0.f,   0.f,      2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   28,   23,   4,      1,    0,     5,    NVCV_BORDER_REFLECT,   0.f,   0.f,   0.f,   0.f,      1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   26,   21,   5,      4,    3,     2,       NVCV_BORDER_WRAP,   0.f,   0.f,   0.f,   0.f,      2,   nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {   64,   32,  16,      0,   32,     0, NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,      2,   nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {   63,   31,  15,      1,   33,     1, NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,      1,   nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {   25,   18,   2,      3,    5,     4, NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,      2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {   18,   16,   1,      2,    2,     1,   NVCV_BORDER_CONSTANT,   3.f,   7.f,  11.f,  13.f,      1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {   26,   21,   5,      4,    3,     2,       NVCV_BORDER_WRAP,   0.f,   0.f,   0.f,   0.f,      2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    {   18,   16,   1,      2,    2,     1,   NVCV_BORDER_CONSTANT,   3.f,   7.f,  11.f,  13.f,      1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},

    // Planar plane sizes are the interleaved ones divided by the channel count, so a plane row is
    // one third / one quarter as wide.  These rows put the plane row length and the left border on
    // both sides of a 16-byte boundary, drop each border to zero in turn, and push the border past
    // the source extent, which is where a widened plane access breaks.
    {   64,   12,   3,      5,   16,     7,   NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,    2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   63,   12,   3,      5,   15,     9,   NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,    2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   65,   11,   0,      4,    1,     6,       NVCV_BORDER_REPLICATE, 0.f,   0.f,   0.f,   0.f,    2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   33,   13,   4,      0,    5,     0,          NVCV_BORDER_WRAP,   0.f,   0.f,   0.f,   0.f,    2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {   17,   11,   3,      2,    0,     0,      NVCV_BORDER_REFLECT,    0.f,   0.f,   0.f,   0.f,    1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {    6,    5,  12,      8,    9,    14,   NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,    2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    {    5,    4,   9,     11,   13,     7,          NVCV_BORDER_WRAP,   0.f,   0.f,   0.f,   0.f,    2,   nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    {   35,   17,   4,      3,    2,     5,       NVCV_BORDER_CONSTANT,  9.f,  19.f,  29.f,  39.f,    2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    {   35,   17,   4,      3,    2,     5,   NVCV_BORDER_REFLECT101,   0.f,   0.f,   0.f,   0.f,    2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
});

// clang-format on

TEST_P(OpCopyMakeBorderPlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(
        GetParamValue<12>(), GetParamValue<13>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
        MakeBorderValue(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>(), GetParamValue<10>()),
        GetParamValue<11>());
}

TEST(OpCopyMakeBorder, large_planar_f32_reflect101_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 1920, 1080, 540, 0, 960, 0, NVCV_BORDER_REFLECT101,
                              float4{}, 8);
}

TEST(OpCopyMakeBorder, large_rgb8_reflect101_matches_gold)
{
    StartTest<uint8_t>(1920, 1080, 8, 540, 0, 960, 0, NVCV_BORDER_REFLECT101, float4{}, nvcv::FMT_RGB8);
}

TEST(OpCopyMakeBorder, large_rgb8_reflect101_matches_planar)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 1920, 1080, 540, 0, 960, 0, NVCV_BORDER_REFLECT101,
                              float4{}, 8);
}

TEST(OpCopyMakeBorder, rgb8_reflect101_two_aligned_strips_matches_gold)
{
    StartTest<uint8_t>(1920, 512, 16, 256, 128, 32, 64, NVCV_BORDER_REFLECT101, float4{}, nvcv::FMT_RGB8);
}

TEST(OpCopyMakeBorder, rgb8_reflect101_two_aligned_strips_matches_planar)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 1920, 512, 256, 128, 32, 64, NVCV_BORDER_REFLECT101,
                              float4{}, 16);
}

TEST(OpCopyMakeBorder, planar_f32_reflect101_two_aligned_strips_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 1920, 512, 256, 128, 32, 64, NVCV_BORDER_REFLECT101,
                              float4{}, 16);
}

TEST(OpCopyMakeBorder, planar_f32_reflect101_right_aligned_strip_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 1920, 512, 256, 128, 0, 96, NVCV_BORDER_REFLECT101,
                              float4{}, 16);
}

TEST(OpCopyMakeBorder, planar_f32_reflect101_wide_carrier_cta_matches_interleaved)
{
    RunPlanarParityTensorCase(nvcv::FMT_RGBf32p, nvcv::FMT_RGBf32, 6144, 64, 3, 5, 2048, 0, NVCV_BORDER_REFLECT101,
                              float4{}, 32);
}

template<typename T>
static void RunBatchlessReflect101GoldAndParity(nvcv::DataType dtype, int srcWidth, int srcHeight, int top, int bottom,
                                                int left, int right)
{
    constexpr int channels    = 3;
    const int     dstWidth    = srcWidth + left + right;
    const int     dstHeight   = srcHeight + top + bottom;
    const int     srcRowBytes = srcWidth * channels * sizeof(T);
    const int     dstRowBytes = dstWidth * channels * sizeof(T);
    SCOPED_TRACE(testing::Message() << dtype << ":" << srcWidth << ":" << srcHeight << ":" << left << ":" << right);

    nvcv::Tensor srcInterleaved(
        {
            {srcHeight, srcWidth, channels},
            "HWC"
    },
        dtype);
    nvcv::Tensor dstInterleaved(
        {
            {dstHeight, dstWidth, channels},
            "HWC"
    },
        dtype);
    nvcv::Tensor srcPlanar(
        {
            {channels, srcHeight, srcWidth},
            "CHW"
    },
        dtype);
    nvcv::Tensor dstPlanar(
        {
            {channels, dstHeight, dstWidth},
            "CHW"
    },
        dtype);
    auto srcInterleavedData = srcInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto dstInterleavedData = dstInterleaved.exportData<nvcv::TensorDataStridedCuda>();
    auto srcPlanarData      = srcPlanar.exportData<nvcv::TensorDataStridedCuda>();
    auto dstPlanarData      = dstPlanar.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcInterleavedData && dstInterleavedData && srcPlanarData && dstPlanarData);
    ASSERT_EQ(srcInterleavedData->layout(), nvcv::TENSOR_HWC);
    ASSERT_EQ(srcPlanarData->layout(), nvcv::TENSOR_CHW);
    auto srcInterleavedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcInterleavedData);
    auto dstInterleavedAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstInterleavedData);
    auto srcPlanarAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcPlanarData);
    auto dstPlanarAccess      = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstPlanarData);
    ASSERT_TRUE(srcInterleavedAccess && dstInterleavedAccess && srcPlanarAccess && dstPlanarAccess);
    ASSERT_EQ(srcInterleavedAccess->numSamples(), 1);
    ASSERT_EQ(srcPlanarAccess->numSamples(), 1);

    std::vector<uint8_t> input(static_cast<size_t>(srcHeight) * srcRowBytes);
    test::planar::FillDeterministicValues(input, 0xC0FFEE, dtype);
    std::vector<uint8_t> expected(static_cast<size_t>(dstHeight) * dstRowBytes);
    for (int dstRow = 0; dstRow < dstHeight; ++dstRow)
    {
        for (int dstCol = 0; dstCol < dstWidth; ++dstCol)
        {
            int2 source{dstCol - left, dstRow - top};
            ApplyBorderIndex(source, int2{srcWidth, srcHeight}, NVCV_BORDER_REFLECT101);
            std::memcpy(expected.data() + static_cast<size_t>(dstRow) * dstRowBytes + dstCol * channels * sizeof(T),
                        input.data() + static_cast<size_t>(source.y) * srcRowBytes + source.x * channels * sizeof(T),
                        channels * sizeof(T));
        }
    }
    test::planar::UploadInterleavedSample(*srcInterleavedAccess, 0, input, srcWidth, srcHeight, srcRowBytes);
    test::planar::UploadPlanarSample(
        *srcPlanarAccess, 0, test::planar::DeinterleaveToPlanes(input, srcWidth, srcHeight, channels, sizeof(T)),
        srcWidth, srcHeight, channels, sizeof(T));
    test::planar::UploadInterleavedSample(*dstInterleavedAccess, 0, std::vector<uint8_t>(expected.size(), 0xA5),
                                          dstWidth, dstHeight, dstRowBytes);
    test::planar::UploadPlanarSample(*dstPlanarAccess, 0, std::vector<uint8_t>(expected.size(), 0x5A), dstWidth,
                                     dstHeight, channels, sizeof(T));
    cvcuda::CopyMakeBorder op;
    ASSERT_NO_THROW(op(nullptr, srcInterleaved, dstInterleaved, top, left, NVCV_BORDER_REFLECT101, float4{}));
    ASSERT_NO_THROW(op(nullptr, srcPlanar, dstPlanar, top, left, NVCV_BORDER_REFLECT101, float4{}));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    const auto interleaved
        = test::planar::DownloadInterleavedSample(*dstInterleavedAccess, 0, dstWidth, dstHeight, dstRowBytes);
    const auto planar = test::planar::InterleaveFromPlanes(
        test::planar::DownloadPlanarSample(*dstPlanarAccess, 0, dstWidth, dstHeight, channels, sizeof(T)), dstWidth,
        dstHeight, channels, sizeof(T));
    EXPECT_EQ(expected, interleaved);
    EXPECT_EQ(expected, planar);
    EXPECT_EQ(interleaved, planar);
}

template<typename T>
static void RunBatchlessReflect101Coverage(nvcv::DataType dtype)
{
    ASSERT_NO_FATAL_FAILURE(RunBatchlessReflect101GoldAndParity<T>(dtype, 1, 2, 5, 4, 3, 2));
    ASSERT_NO_FATAL_FAILURE(RunBatchlessReflect101GoldAndParity<T>(dtype, 33, 7, 15, 11, 17, 19));
    ASSERT_NO_FATAL_FAILURE(RunBatchlessReflect101GoldAndParity<T>(dtype, 64, 13, 3, 5, 16, 16));
    int device{};
    int cacheBytes{};
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&cacheBytes, cudaDevAttrL2CacheSize, device));
    ASSERT_GT(cacheBytes, 0);
    constexpr int srcWidth       = 6144;
    constexpr int budgetRowBytes = srcWidth * sizeof(T) * (std::is_same_v<T, float> ? 2 : 3);
    const int     srcHeight      = cacheBytes / budgetRowBytes + 1;
    ASSERT_GT(static_cast<int64_t>(srcHeight) * budgetRowBytes, cacheBytes);
    ASSERT_NO_FATAL_FAILURE(RunBatchlessReflect101GoldAndParity<T>(dtype, srcWidth, srcHeight, 3, 5, 2048, 0));
}

TEST(OpCopyMakeBorder, batchless_f32_reflect101_matches_gold_and_hwc)
{
    RunBatchlessReflect101Coverage<float>(nvcv::TYPE_F32);
}

TEST(OpCopyMakeBorder, batchless_rgb8_reflect101_matches_gold_and_chw)
{
    RunBatchlessReflect101Coverage<uint8_t>(nvcv::TYPE_U8);
}

template<typename T>
static void RunStreamingReflect101Edges(nvcv::DataType dtype)
{
    int device{};
    int cacheBytes{};
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&cacheBytes, cudaDevAttrL2CacheSize, device));
    ASSERT_GT(cacheBytes, 0);
    for (const auto &[width, left, right] : {std::make_tuple(6143, 2047, 19), std::make_tuple(639, 801, 23)})
    {
        const int budgetRowBytes = (2 * width + left + right) * sizeof(T) * (std::is_same_v<T, float> ? 1 : 3);
        const int height         = cacheBytes / budgetRowBytes + 1;
        ASSERT_GT(static_cast<int64_t>(height) * budgetRowBytes, cacheBytes);
        ASSERT_NO_FATAL_FAILURE(RunBatchlessReflect101GoldAndParity<T>(dtype, width, height, 3, 5, left, right));
    }
}

TEST(OpCopyMakeBorder, streaming_f32_reflect101_odd_and_repeated_folds_match_gold)
{
    RunStreamingReflect101Edges<float>(nvcv::TYPE_F32);
}

TEST(OpCopyMakeBorder, streaming_rgb8_reflect101_odd_and_repeated_folds_match_gold)
{
    RunStreamingReflect101Edges<uint8_t>(nvcv::TYPE_U8);
}

TEST_P(OpCopyMakeBorderPlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(
        GetParamValue<12>(), GetParamValue<13>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
        MakeBorderValue(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>(), GetParamValue<10>()),
        GetParamValue<11>());
}

TEST_P(OpCopyMakeBorderPlanar, stack_matches_interleaved)
{
    RunPlanarParityStackCase(
        GetParamValue<12>(), GetParamValue<13>(), GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
        GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
        MakeBorderValue(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>(), GetParamValue<10>()),
        GetParamValue<11>());
}

static auto OpCopyMakeBorderNegativeParams()
{
    test::ValueList<NVCVStatus, nvcv::ImageFormat, nvcv::ImageFormat, int, int, NVCVBorderType> params{
        {NVCV_ERROR_INVALID_ARGUMENT,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 0, 0,
         NVCV_BORDER_CONSTANT}, // data format is different
        {NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_RGB8p,  nvcv::FMT_RGB8, 0, 0,
         NVCV_BORDER_CONSTANT}, // data format is different
        {NVCV_ERROR_INVALID_ARGUMENT,    nvcv::FMT_U8,   nvcv::FMT_U16, 0, 0,
         NVCV_BORDER_CONSTANT}, // data type is different
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 0, 0,
                        static_cast<NVCVBorderType>(255));
#endif
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, -1, 0, NVCV_BORDER_CONSTANT);
    params.emplace_back(NVCV_ERROR_INVALID_ARGUMENT, nvcv::FMT_U8, nvcv::FMT_U8, 0, -1, NVCV_BORDER_CONSTANT);
    return params;
}

NVCV_TEST_SUITE_P(OpCopyMakeBorder_Negative, OpCopyMakeBorderNegativeParams());

TEST_P(OpCopyMakeBorder_Negative, op)
{
    NVCVStatus        expectedReturnCode = GetParamValue<0>();
    nvcv::ImageFormat inputFmt           = GetParamValue<1>();
    nvcv::ImageFormat outputFmt          = GetParamValue<2>();
    int               topPad             = GetParamValue<3>();
    int               leftPad            = GetParamValue<4>();
    NVCVBorderType    borderType         = GetParamValue<5>();

    int    srcWidth   = 24;
    int    srcHeight  = 24;
    int    dstWidth   = srcWidth + leftPad;
    int    dstHeight  = srcHeight + topPad;
    int    numBatches = 3;
    float4 borderValue;
    borderValue.x = 1.f;
    borderValue.y = 1.f;
    borderValue.z = 1.f;
    borderValue.w = 1.f;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgSrc(numBatches, {srcWidth, srcHeight}, inputFmt);
    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, outputFmt);

    // Generate test result
    cvcuda::CopyMakeBorder cpyMakeBorderOp;
    EXPECT_EQ(
        expectedReturnCode,
        nvcv::ProtectCall([&cpyMakeBorderOp, &stream, &imgSrc, &imgDst, &topPad, &leftPad, &borderType, &borderValue]
                          { cpyMakeBorderOp(stream, imgSrc, imgDst, topPad, leftPad, borderType, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCopyMakeBorder_Negative, single_image_layout_mismatch)
{
    const nvcv::TensorShape hwcShape{
        {4, 4, 3},
        nvcv::TENSOR_HWC
    };
    const nvcv::TensorShape chwShape{
        {3, 6, 6},
        nvcv::TENSOR_CHW
    };
    nvcv::Tensor input(hwcShape, nvcv::TYPE_U8);
    nvcv::Tensor output(chwShape, nvcv::TYPE_U8);
    float4       borderValue{1.f, 1.f, 1.f, 1.f};

    cvcuda::CopyMakeBorder op;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, input, output, 1, 1, NVCV_BORDER_CONSTANT, borderValue); }));
}

TEST(OpCopyMakeBorder_Negative, invalid_out_size)
{
    nvcv::ImageFormat fmt     = nvcv::FMT_U8;
    int               topPad  = 2;
    int               leftPad = 2;

    std::vector<std::pair<int, int>> testSet{
        {-1,  0}, // invalid dst width
        { 0, -1}  // invalid dst height
    };

    for (const auto &[leftPadExtra, topPadExtra] : testSet)
    {
        int    srcWidth   = 24;
        int    srcHeight  = 24;
        int    dstWidth   = srcWidth + leftPad + leftPadExtra;
        int    dstHeight  = srcHeight + topPad + topPadExtra;
        int    numBatches = 3;
        float4 borderValue;
        borderValue.x = 1.f;
        borderValue.y = 1.f;
        borderValue.z = 1.f;
        borderValue.w = 1.f;

        cudaStream_t stream;
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        nvcv::Tensor imgSrc(numBatches, {srcWidth, srcHeight}, fmt);
        nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, fmt);

        // Run operator
        cvcuda::CopyMakeBorder cpyMakeBorderOp;
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall(
                      [&cpyMakeBorderOp, &stream, &imgSrc, &imgDst, &topPad, &leftPad, &borderValue] {
                          cpyMakeBorderOp(stream, imgSrc, imgDst, topPad, leftPad, NVCV_BORDER_CONSTANT, borderValue);
                      }));

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}

// 64-bit elements are the widest the operator takes, and a 4-channel one is 32 bytes -- wider than
// the 16-byte vector pack the interior fast path uses. Its elements-per-thread count is then zero,
// which divided by zero when building the grid, so this pins every 64-bit channel count.
template<typename T>
static void RunCopyMakeBorder64Bit(nvcv::DataType dtype, int channels, int width = 9, int height = 5,
                                   NVCVBorderType borderType = NVCV_BORDER_CONSTANT, float4 borderValue = {0, 0, 0, 0})
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int kSrcW   = width;
    const int kSrcH   = height;
    const int kTop    = 2;
    const int kLeft   = 3;
    const int kBottom = 1;
    const int kRight  = 4;
    const int dstW    = kSrcW + kLeft + kRight;
    const int dstH    = kSrcH + kTop + kBottom;

    nvcv::Tensor src(
        {
            {1, kSrcH, kSrcW, channels},
            "NHWC"
    },
        dtype);
    nvcv::Tensor dst(
        {
            {1, dstH, dstW, channels},
            "NHWC"
    },
        dtype);

    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    ASSERT_TRUE(dstData);

    std::vector<T> input(static_cast<size_t>(kSrcW) * kSrcH * channels);
    size_t         inputValue = 1;
    for (T &value : input)
    {
        value = static_cast<T>(inputValue);
        ++inputValue;
    }
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<T>(*srcData, input, 0));

    cvcuda::CopyMakeBorder op;
    ASSERT_NO_THROW(op(stream, src, dst, kTop, kLeft, borderType, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<T> got;
    nvcv::util::GetImageVectorFromTensor<T>(*dstData, 0, got);
    ASSERT_EQ(static_cast<size_t>(dstW) * dstH * channels, got.size());

    for (size_t o = 0; o < got.size(); ++o)
    {
        const auto c = static_cast<int>(o % channels);
        const auto x = static_cast<int>((o / channels) % dstW);
        const auto y = static_cast<int>(o / (channels * dstW));
        const T    expected
            = CopyMakeBorderValue<T>(input, int2{x - kLeft, y - kTop}, int2{kSrcW, kSrcH}, c, borderType, borderValue,
                                     [kSrcW, channels](int2 coord, int channel)
                                     { return (static_cast<size_t>(coord.y) * kSrcW + coord.x) * channels + channel; });
        ASSERT_EQ(expected, got[o]) << "at " << x << "," << y << " ch " << c;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCopyMakeBorder, signed64_matches_the_reference)
{
    RunCopyMakeBorder64Bit<int64_t>(nvcv::TYPE_S64, 1);
    RunCopyMakeBorder64Bit<int64_t>(nvcv::TYPE_3S64, 3);
    RunCopyMakeBorder64Bit<int64_t>(nvcv::TYPE_4S64, 4);
}

TEST(OpCopyMakeBorder, float64_matches_the_reference)
{
    RunCopyMakeBorder64Bit<double>(nvcv::TYPE_F64, 1);
    RunCopyMakeBorder64Bit<double>(nvcv::TYPE_3F64, 3);
    RunCopyMakeBorder64Bit<double>(nvcv::TYPE_4F64, 4);
}

TEST(OpCopyMakeBorder, wide_pixels_all_borders_and_tails)
{
    for (auto border :
         {NVCV_BORDER_CONSTANT, NVCV_BORDER_REPLICATE, NVCV_BORDER_REFLECT, NVCV_BORDER_REFLECT101, NVCV_BORDER_WRAP})
        for (int width : {9, 127, 256})
        {
            SCOPED_TRACE(testing::Message() << border << ":" << width);
            const float4 value{7.f, -11.f, 13.f, -17.f};
            RunCopyMakeBorder64Bit<int64_t>(nvcv::TYPE_4S64, 4, width, 17, border, value);
            RunCopyMakeBorder64Bit<double>(nvcv::TYPE_4F64, 4, width, 17, border, value);
        }
}

template<typename T>
static void RunCopyMakeBorder64ReflectionCase(nvcv::DataType dtype, bool planar, int width, int padding, int srcOffset,
                                              int dstOffset, NVCVBorderType border, int height = 3, int samples = 2,
                                              int planeGap = 0, int sampleGap = 0)
{
    constexpr int channels = 3;
    constexpr int top      = 1;

    SCOPED_TRACE(testing::Message() << planar << ":" << width << ":" << padding << ":" << srcOffset << ":" << dstOffset
                                    << ":" << border);
    const int     left        = width / 2;
    const int     right       = width == 33 ? 3 : 0;
    const int     dstW        = width + left + right;
    const int     dstH        = height + top + 1;
    const int     planes      = planar ? channels : 1;
    const int     rowChannels = planar ? 1 : channels;
    const int64_t srcRow      = (width * rowChannels + padding) * sizeof(T);
    const int64_t dstRow      = (dstW * rowChannels + padding) * sizeof(T);
    const size_t  srcBytes    = srcOffset + samples * (planes * (height * srcRow + planeGap) + sampleGap);
    const size_t  dstBytes    = dstOffset + samples * (planes * (dstH * dstRow + planeGap) + sampleGap);
    NVCVByte     *srcPtr{};
    NVCVByte     *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    const auto wrap
        = [planes, planar, dtype, samples, planeGap, sampleGap](NVCVByte *ptr, int offset, int w, int h, int64_t row)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + offset;
        buffer.strides[0] = planes * (h * row + planeGap) + sampleGap;
        buffer.strides[1] = planar ? h * row + planeGap : row;
        buffer.strides[2] = planar ? row : channels * sizeof(T);
        buffer.strides[3] = sizeof(T);
        nvcv::TensorShape shape = planar
        ? nvcv::TensorShape{{samples, channels, h, w}, "NCHW"}
        : nvcv::TensorShape{{samples, h, w, channels}, "NHWC"};
        return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, dtype, buffer});
    };
    auto       src    = wrap(srcPtr, srcOffset, width, height, srcRow);
    auto       dst    = wrap(dstPtr, dstOffset, dstW, dstH, dstRow);
    const auto offset = [planar, planes, planeGap, sampleGap](int n, int c, int y, int x, int h, int64_t row)
    {
        return n * (planes * (h * row + planeGap) + sampleGap) + (planar ? c * (h * row + planeGap) : 0) + y * row
             + (planar ? x : x * channels + c) * sizeof(T);
    };
    std::vector<uint8_t> input(srcBytes, 0xA5);
    std::vector<uint8_t> expected(dstBytes, 0xA5);
    std::vector<uint8_t> actual(dstBytes);
    std::vector<T>       dense(width * height * channels);
    for (int n = 0; n < samples; ++n)
    {
        for (int i = 0; i < height * width * channels; ++i)
        {
            const int  c     = i % channels;
            const int  x     = (i / channels) % width;
            const int  y     = i / (channels * width);
            const auto value = static_cast<T>(n * 10000 + y * width * channels + x * channels + c - 100);
            dense[(y * width + x) * channels + c] = value;
            std::memcpy(input.data() + srcOffset + offset(n, c, y, x, height, srcRow), &value, sizeof(T));
        }
        for (int i = 0; i < dstH * dstW * channels; ++i)
        {
            const int c     = i % channels;
            const int x     = (i / channels) % dstW;
            const int y     = i / (channels * dstW);
            const T   value = CopyMakeBorderValue<T>(
                dense, int2{x - left, y - top}, int2{width, height}, c, border, float4{},
                [width](int2 xy, int channel) { return (xy.y * width + xy.x) * channels + channel; });
            std::memcpy(expected.data() + dstOffset + offset(n, c, y, x, dstH, dstRow), &value, sizeof(T));
        }
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0xA5, dstBytes));
    cvcuda::CopyMakeBorder op;
    ASSERT_NO_THROW(op(nullptr, src, dst, top, left, border, float4{}));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

template<typename T>
static void RunCopyMakeBorder64Reflection(nvcv::DataType dtype)
{
    const auto cases = test::Combine(test::ValueList<bool>{false, true}, test::ValueList<int>{32, 33, 508, 512, 516},
                                     test::ValueList<int>{0, 1}, test::ValueList<int>{0, 8}, test::ValueList<int>{0, 8},
                                     test::ValueList<NVCVBorderType>{NVCV_BORDER_REFLECT, NVCV_BORDER_REFLECT101});
    for (const auto &[planar, width, padding, srcOffset, dstOffset, border] : cases)
        ASSERT_NO_FATAL_FAILURE(
            RunCopyMakeBorder64ReflectionCase<T>(dtype, planar, width, padding, srcOffset, dstOffset, border));
}

TEST(OpCopyMakeBorder, rgb8_row_carrier_strides_and_fallbacks)
{
    RunCopyMakeBorder64ReflectionCase<uint8_t>(nvcv::TYPE_U8, false, 1920, 16, 1, 0, NVCV_BORDER_REFLECT101, 512, 16,
                                               16, 32);
    RunCopyMakeBorder64ReflectionCase<uint8_t>(nvcv::TYPE_U8, false, 1920, 5, 1, 1, NVCV_BORDER_REFLECT101, 512, 16, 3,
                                               7);
    for (int width : {1, 2, 31, 32, 33})
        RunCopyMakeBorder64ReflectionCase<uint8_t>(nvcv::TYPE_U8, false, width, 5, 1, 3, NVCV_BORDER_REFLECT101);
}

TEST(OpCopyMakeBorder, planar_f32_row_carrier_strides_and_fallbacks)
{
    RunCopyMakeBorder64ReflectionCase<float>(nvcv::TYPE_F32, true, 1920, 4, 4, 0, NVCV_BORDER_REFLECT101, 512, 16, 16,
                                             32);
    RunCopyMakeBorder64ReflectionCase<float>(nvcv::TYPE_F32, true, 1920, 5, 4, 4, NVCV_BORDER_REFLECT101, 512, 16, 12,
                                             28);
    for (int width : {1, 2, 31, 32, 33})
        RunCopyMakeBorder64ReflectionCase<float>(nvcv::TYPE_F32, true, width, 5, 4, 12, NVCV_BORDER_REFLECT101);
}

TEST(OpCopyMakeBorder, s64_reflection_carrier_boundaries)
{
    RunCopyMakeBorder64Reflection<int64_t>(nvcv::TYPE_S64);
}

TEST(OpCopyMakeBorder, f64_reflection_carrier_boundaries)
{
    RunCopyMakeBorder64Reflection<double>(nvcv::TYPE_F64);
}

template<typename T>
static void RunCopyMakeBorderScalarAlignedWideCase(nvcv::DataType dtype, int baseOffset, NVCVBorderType border)
{
    constexpr int srcW     = 8;
    constexpr int srcH     = 2;
    constexpr int dstW     = 12;
    constexpr int dstH     = 4;
    constexpr int samples  = 2;
    constexpr int channels = 4;
    constexpr int left     = 2;
    constexpr int top      = 1;
    const float4  borderValue{7.f, -11.f, 13.f, -17.f};

    SCOPED_TRACE(testing::Message() << baseOffset << ":" << border);
    const int64_t srcRow   = srcW * channels * sizeof(T) + 8;
    const int64_t dstRow   = dstW * channels * sizeof(T) + 8;
    const size_t  srcBytes = baseOffset + samples * srcH * srcRow;
    const size_t  dstBytes = baseOffset + samples * dstH * dstRow;
    NVCVByte     *srcPtr{};
    NVCVByte     *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    auto wrap = [baseOffset](NVCVByte *ptr, int height, int64_t row)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = ptr + baseOffset;
        buffer.strides[0] = height * row;
        buffer.strides[1] = row;
        buffer.strides[2] = channels * sizeof(T);
        buffer.strides[3] = sizeof(T);
        return buffer;
    };
    auto                 src = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{samples, srcH, srcW, channels}, "NHWC"},
        dtype, wrap(srcPtr, srcH, srcRow)
    });
    auto                 dst = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{
        nvcv::TensorShape{{samples, dstH, dstW, channels}, "NHWC"},
        dtype, wrap(dstPtr, dstH, dstRow)
    });
    std::vector<uint8_t> input(srcBytes, 0xA5);
    std::vector<uint8_t> expected(dstBytes, 0xA5);
    std::vector<uint8_t> actual(dstBytes, 0xA5);
    std::vector<T>       dense(srcW * srcH * channels);
    for (int n = 0; n < samples; ++n)
    {
        int inputIndex = 0;
        for (T &value : dense)
        {
            value = static_cast<T>(inputIndex - 100 + n * 1000);
            ++inputIndex;
        }
        for (int y = 0; y < srcH; ++y)
            std::memcpy(input.data() + baseOffset + (n * srcH + y) * srcRow, dense.data() + y * srcW * channels,
                        srcW * channels * sizeof(T));
        for (int i = 0; i < dstH * dstW * channels; ++i)
        {
            const int c     = i % channels;
            const int x     = (i / channels) % dstW;
            const int y     = i / (channels * dstW);
            const T   value = CopyMakeBorderValue<T>(
                dense, int2{x - left, y - top}, int2{srcW, srcH}, c, border, borderValue,
                [](int2 coord, int channel)
                { return (static_cast<size_t>(coord.y) * srcW + coord.x) * channels + channel; });
            std::memcpy(expected.data() + baseOffset + (n * dstH + y) * dstRow + (x * channels + c) * sizeof(T), &value,
                        sizeof(value));
        }
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstPtr, actual.data(), dstBytes, cudaMemcpyHostToDevice));
    cvcuda::CopyMakeBorder op;
    ASSERT_NO_THROW(op(nullptr, src, dst, top, left, border, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
    EXPECT_EQ(expected, actual);
}

template<typename T>
static void RunCopyMakeBorderScalarAlignedWide(nvcv::DataType dtype)
{
    const auto cases = test::Combine(test::ValueList<int>{0, 8},
                                     test::ValueList<NVCVBorderType>{NVCV_BORDER_CONSTANT, NVCV_BORDER_REPLICATE});
    for (const auto &[baseOffset, border] : cases)
        ASSERT_NO_FATAL_FAILURE(RunCopyMakeBorderScalarAlignedWideCase<T>(dtype, baseOffset, border));
}

TEST(OpCopyMakeBorder, s64_wide_scalar_alignment)
{
    RunCopyMakeBorderScalarAlignedWide<int64_t>(nvcv::TYPE_S64);
}

TEST(OpCopyMakeBorder, f64_wide_scalar_alignment)
{
    RunCopyMakeBorderScalarAlignedWide<double>(nvcv::TYPE_F64);
}

static void RunCopyMakeBorderConstantByteGold(bool fixedOutput, bool planar, int width, int height, int left, int right,
                                              int rowPadding, int srcOffset, int dstOffset, uint8_t borderByte)
{
    constexpr int samples   = 2;
    constexpr int top       = 2;
    const int     planes    = planar ? 3 : 1;
    const int     dstWidth  = width + left + right + 1;
    const int     dstHeight = height + top + 3;
    const int     srcRow    = ((width + 3) & ~3) + rowPadding;
    const int     dstRow    = ((dstWidth + 3) & ~3) + rowPadding;
    const int     srcPlane  = height * srcRow + 4;
    const int     dstPlane  = dstHeight * dstRow + 4;
    const int     srcSample = planes * srcPlane + 4;
    const int     dstSample = planes * dstPlane + 4;
    const size_t  srcBytes  = srcOffset + samples * srcSample;
    const size_t  dstBytes  = dstOffset + samples * dstSample;
    const auto    format    = planar ? nvcv::FMT_RGB8p : nvcv::FMT_U8;
    const float4  borderValue{static_cast<float>(borderByte), 91.f, 201.f, 255.f};

    SCOPED_TRACE(testing::Message() << fixedOutput << ":" << planar << ":" << width << ":" << height << ":" << left
                                    << ":" << right << ":" << rowPadding << ":" << srcOffset << ":" << dstOffset << ":"
                                    << static_cast<int>(borderByte));
    NVCVByte *srcPtr{};
    NVCVByte *dstPtr{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(&srcPtr, srcBytes));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstPtr, dstBytes));
    std::vector<uint8_t>     input(srcBytes, 0xA5);
    std::vector<uint8_t>     expected(dstBytes, 0x5A);
    std::vector<uint8_t>     actual(dstBytes);
    std::vector<int>         topValues(samples);
    std::vector<int>         leftValues(samples);
    nvcv::ImageBatchVarShape source(samples);
    nvcv::ImageBatchVarShape destination(samples);

    for (int sample = 0; sample < samples; ++sample)
    {
        const int srcWidth         = std::max(1, width - sample);
        const int srcHeight        = std::max(1, height - sample);
        const int outputSizeOffset = fixedOutput ? 0 : sample;
        const int outWidth         = dstWidth - outputSizeOffset;
        const int outHeight        = dstHeight - outputSizeOffset;
        topValues[sample]          = top + sample;
        leftValues[sample]         = left + sample;
        nvcv::ImageDataStridedCuda::Buffer srcBuffer{};
        nvcv::ImageDataStridedCuda::Buffer dstBuffer{};
        srcBuffer.numPlanes = planes;
        dstBuffer.numPlanes = planes;
        for (int plane = 0; plane < planes; ++plane)
        {
            const int srcBase                 = srcOffset + sample * srcSample + plane * srcPlane;
            const int dstBase                 = dstOffset + sample * dstSample + plane * dstPlane;
            srcBuffer.planes[plane].basePtr   = srcPtr + srcBase;
            srcBuffer.planes[plane].width     = srcWidth;
            srcBuffer.planes[plane].height    = srcHeight;
            srcBuffer.planes[plane].rowStride = srcRow;
            dstBuffer.planes[plane].basePtr   = dstPtr + dstBase;
            dstBuffer.planes[plane].width     = outWidth;
            dstBuffer.planes[plane].height    = outHeight;
            dstBuffer.planes[plane].rowStride = dstRow;
            for (int index = 0; index < srcHeight * srcWidth; ++index)
            {
                const int row    = index / srcWidth;
                const int column = index % srcWidth;
                input[srcBase + row * srcRow + column]
                    = static_cast<uint8_t>(sample * 37 + plane * 61 + row * 13 + column * 3);
            }
            for (int index = 0; index < outHeight * outWidth; ++index)
            {
                const int row                             = index / outWidth;
                const int column                          = index % outWidth;
                expected[dstBase + row * dstRow + column] = CopyMakeBorderValue<uint8_t>(
                    input, int2{column - leftValues[sample], row - topValues[sample]}, int2{srcWidth, srcHeight}, plane,
                    NVCV_BORDER_CONSTANT, borderValue,
                    [srcBase, srcRow](int2 coords, int) { return srcBase + coords.y * srcRow + coords.x; });
            }
        }
        source.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, srcBuffer}));
        if (!fixedOutput)
            destination.pushBack(nvcv::ImageWrapData(nvcv::ImageDataStridedCuda{format, dstBuffer}));
    }

    nvcv::Tensor topTensor(1, {samples, 1}, nvcv::FMT_S32);
    nvcv::Tensor leftTensor(1, {samples, 1}, nvcv::FMT_S32);
    auto         topData  = topTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto         leftData = leftTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(topData && leftData);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(topData->basePtr(), topValues.data(), samples * sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(leftData->basePtr(), leftValues.data(), samples * sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcPtr, input.data(), srcBytes, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(dstPtr, 0x5A, dstBytes));
    cvcuda::CopyMakeBorder op;
    if (fixedOutput)
    {
        nvcv::TensorDataStridedCuda::Buffer buffer{};
        buffer.basePtr    = dstPtr + dstOffset;
        buffer.strides[0] = dstSample;
        buffer.strides[1] = planar ? dstPlane : dstRow;
        buffer.strides[2] = planar ? dstRow : 1;
        buffer.strides[3] = 1;
        const nvcv::TensorShape shape = planar
            ? nvcv::TensorShape{{samples, planes, dstHeight, dstWidth}, "NCHW"}
            : nvcv::TensorShape{{samples, dstHeight, dstWidth, 1}, "NHWC"};
        auto output = nvcv::TensorWrapData(nvcv::TensorDataStridedCuda{shape, nvcv::TYPE_U8, buffer});
        ASSERT_NO_THROW(op(nullptr, source, output, topTensor, leftTensor, NVCV_BORDER_CONSTANT, borderValue));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    }
    else
    {
        ASSERT_NO_THROW(op(nullptr, source, destination, topTensor, leftTensor, NVCV_BORDER_CONSTANT, borderValue));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));
    }
    ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), dstPtr, dstBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected, actual);
    std::vector<uint8_t> inputAfter(srcBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inputAfter.data(), srcPtr, srcBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(input, inputAfter);
    ASSERT_EQ(cudaSuccess, cudaFree(srcPtr));
    ASSERT_EQ(cudaSuccess, cudaFree(dstPtr));
}

static void RunCopyMakeBorderConstantByteCoverage(bool fixedOutput)
{
    const test::ValueList<int, int, int, int, int, int, int, uint8_t> cases{
        {8, 3, 4,  3, 0, 0, 0,   7},
        {8, 3, 4,  3, 0, 0, 1,   7},
        {8, 3, 4,  3, 0, 0, 2,   7},
        {8, 3, 4,  3, 0, 0, 3,   7},
        {9, 4, 5,  4, 1, 1, 0, 128},
        {7, 2, 1,  2, 2, 3, 1, 255},
        {1, 1, 9, 10, 0, 0, 0, 255},
        {2, 2, 0,  0, 1, 1, 0,   0}
    };
    for (bool planar : {false, true})
    {
        for (const auto &[width, height, left, right, rowPadding, srcOffset, dstOffset, borderByte] : cases)
        {
            ASSERT_NO_FATAL_FAILURE(RunCopyMakeBorderConstantByteGold(fixedOutput, planar, width, height, left, right,
                                                                      rowPadding, srcOffset, dstOffset, borderByte));
        }
    }
}

TEST(OpCopyMakeBorder, varshape_constant_byte_packed_matches_gold)
{
    RunCopyMakeBorderConstantByteCoverage(false);
}

TEST(OpCopyMakeBorder, stack_constant_byte_packed_matches_gold)
{
    RunCopyMakeBorderConstantByteCoverage(true);
}

static auto OpCopyMakeBorderVarshapeNegativeParams()
{
    test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, nvcv::ImageFormat, NVCVBorderType> params{
        { nvcv::FMT_RGB8, nvcv::FMT_RGB8p,   nvcv::FMT_S32,   nvcv::FMT_S32, NVCV_BORDER_CONSTANT},
        {nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,   nvcv::FMT_S32,   nvcv::FMT_S32, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8,   nvcv::FMT_F32,   nvcv::FMT_S32, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8,   nvcv::FMT_S32,   nvcv::FMT_F32, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8, nvcv::FMT_RGB8p,   nvcv::FMT_S32, NVCV_BORDER_CONSTANT},
        { nvcv::FMT_RGB8,  nvcv::FMT_RGB8,   nvcv::FMT_S32, nvcv::FMT_RGB8p, NVCV_BORDER_CONSTANT},
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(nvcv::FMT_RGB8, nvcv::FMT_RGB8, nvcv::FMT_S32, nvcv::FMT_S32, static_cast<NVCVBorderType>(255));
#endif
    return params;
}

NVCV_TEST_SUITE_P(OpCopyMakeBorderVarshape_Negative, OpCopyMakeBorderVarshapeNegativeParams());

TEST_P(OpCopyMakeBorderVarshape_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = 24;
    int srcHeightBase = 24;
    int left          = 1;
    int top           = 1;
    int numBatches    = 3;

    float4 borderValue;
    borderValue.x = 1.f;
    borderValue.y = 1.f;
    borderValue.z = 1.f;
    borderValue.w = 1.f;

    nvcv::ImageFormat inputFmt   = GetParamValue<0>();
    nvcv::ImageFormat outputFmt  = GetParamValue<1>();
    nvcv::ImageFormat topPadFmt  = GetParamValue<2>();
    nvcv::ImageFormat leftPadDmt = GetParamValue<3>();
    NVCVBorderType    borderType = GetParamValue<4>();

    std::default_random_engine    randEng{0};
    std::uniform_int_distribution rndSrcWidth(ScaleDimension(srcWidthBase, 0.8), ScaleDimension(srcWidthBase, 1.2));
    std::uniform_int_distribution rndSrcHeight(ScaleDimension(srcHeightBase, 0.8), ScaleDimension(srcHeightBase, 1.2));

    //Prepare input and output buffer
    std::vector<nvcv::Image> imgSrcVec;
    std::vector<nvcv::Image> imgDstVec;
    for (int i = 0; i < numBatches; ++i)
    {
        int srcWidth  = rndSrcWidth(randEng);
        int srcHeight = rndSrcHeight(randEng);

        int dstWidth  = srcWidth + left;
        int dstHeight = srcHeight + top;
        //prepare input buffers
        imgSrcVec.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, inputFmt);
        //prepare output Buffers
        imgDstVec.emplace_back(nvcv::Size2D{dstWidth, dstHeight}, outputFmt);
    }
    nvcv::ImageBatchVarShape imgBatchSrc(numBatches);
    imgBatchSrc.pushBack(imgSrcVec.begin(), imgSrcVec.end());
    nvcv::ImageBatchVarShape imgBatchDst(numBatches);
    imgBatchDst.pushBack(imgDstVec.begin(), imgDstVec.end());

    // Prepare top and left inputs
    nvcv::Tensor inTop(1, {numBatches, 1}, topPadFmt);
    nvcv::Tensor inLeft(1, {numBatches, 1}, leftPadDmt);

    // Run operator
    cvcuda::CopyMakeBorder cpyMakeBorderOp;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&cpyMakeBorderOp, &stream, &imgBatchSrc, &imgBatchDst, &inTop, &inLeft, &borderType, &borderValue]
                  { cpyMakeBorderOp(stream, imgBatchSrc, imgBatchDst, inTop, inLeft, borderType, borderValue); }));
}

TEST(OpCopyMakeBorderVarshape_Negative, empty_input_rejected)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape imgBatchSrc(1);
    nvcv::ImageBatchVarShape imgBatchDst(1);
    imgBatchDst.pushBack(nvcv::Image({24, 24}, nvcv::FMT_U8));

    nvcv::Tensor inTop(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {1, 1}, nvcv::FMT_S32);
    float4       borderValue{1.f, 1.f, 1.f, 1.f};

    cvcuda::CopyMakeBorder cpyMakeBorderOp;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&cpyMakeBorderOp, &stream, &imgBatchSrc, &imgBatchDst, &inTop, &inLeft, &borderValue]
            { cpyMakeBorderOp(stream, imgBatchSrc, imgBatchDst, inTop, inLeft, NVCV_BORDER_CONSTANT, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCopyMakeBorderVarshape_Negative, rejects_invalid_multi_plane_formats)
{
    constexpr int batch = 1;

    const nvcv::ImageFormat mixedPlaneDataTypes{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, X8, X16)};
    const nvcv::ImageFormat multiChannelPlanes{
        NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8, X8_Y8)};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::CopyMakeBorder op;
    const auto             expectInvalidFormat = [&](nvcv::ImageFormat format)
    {
        nvcv::ImageBatchVarShape input(batch);
        nvcv::ImageBatchVarShape output(batch);
        input.pushBack(nvcv::Image(nvcv::Size2D{8, 8}, format));
        output.pushBack(nvcv::Image(nvcv::Size2D{10, 10}, format));

        nvcv::Tensor top(1, {batch, 1}, nvcv::FMT_S32);
        nvcv::Tensor left(1, {batch, 1}, nvcv::FMT_S32);
        const float4 borderValue{1.f, 1.f, 1.f, 1.f};

        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
                  nvcv::ProtectCall([&] { op(stream, input, output, top, left, NVCV_BORDER_CONSTANT, borderValue); }));
    };

    expectInvalidFormat(mixedPlaneDataTypes);
    expectInvalidFormat(multiChannelPlanes);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpCopyMakeBorder_Negative, rejects_complex_dtype)
{
    cvcuda::CopyMakeBorder op;
    nvcv::Tensor           input(1, {4, 4}, nvcv::FMT_C64);
    nvcv::Tensor           output(1, {6, 6}, nvcv::FMT_C64);
    const float4           borderValue{1.f, 1.f, 1.f, 1.f};

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { op(nullptr, input, output, 1, 1, NVCV_BORDER_CONSTANT, borderValue); }));
}

TEST(OpCopyMakeBorder_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaCopyMakeBorderCreate(nullptr), NVCV_ERROR_INVALID_ARGUMENT);
}
