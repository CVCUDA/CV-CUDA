/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpOSD.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
using namespace cvcuda::priv;

static std::mt19937 &Rng()
{
    static std::mt19937 rng;
    return rng;
}

static int randl(int l, int h)
{
    return std::uniform_int_distribution<int>(l, h)(Rng());
}

static std::time_t CurrentTime()
{
    return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

static void FillRandomSegment(std::vector<float> &hSeg)
{
    for (auto &v : hSeg)
    {
        v = static_cast<float>(randl(0, 100)) / 100.0f;
    }
}

#pragma GCC push_options
#pragma GCC optimize("O1")

#pragma GCC pop_options

static void runOp(cudaStream_t &stream, const cvcuda::OSD &op, int inN, int inW, int inH, int num, int sed,
                  const nvcv::ImageFormat &format)
{
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    Rng().seed(sed);
    for (int n = 0; n < inN; n++)
    {
        std::vector<std::shared_ptr<NVCVElement>> curVec;
        for (int i = 0; i < num; i++)
        {
            auto type = static_cast<NVCVOSDType>(randl(int(NVCV_OSD_NONE) + 1, int(NVCV_OSD_MAX) - 1));
            std::shared_ptr<NVCVElement> element;
            switch (type)
            {
            case NVCVOSDType::NVCV_OSD_RECT:
            {
                NVCVBndBoxI bndBox;
                bndBox.box.x       = randl(0, inW - 1);
                bndBox.box.y       = randl(0, inH - 1);
                bndBox.box.width   = randl(1, inW);
                bndBox.box.height  = randl(1, inH);
                bndBox.thickness   = randl(-1, 30);
                bndBox.fillColor   = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
                bndBox.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                      (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)}; // alpha: 1-255
                element            = std::make_shared<NVCVElement>(type, &bndBox);
                break;
            }
            case NVCVOSDType::NVCV_OSD_TEXT:
            {
                auto text = NVCVText("abcdefghijklmnopqrstuvwxyz", 5 * randl(1, 10), DEFAULT_OSD_FONT,
                                     NVCVPointI({randl(0, inW - 1), randl(0, inH - 1)}),
                                     NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                    (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                     NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                    (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}));
                element   = std::make_shared<NVCVElement>(type, &text);
                break;
            }
            case NVCVOSDType::NVCV_OSD_SEGMENT:
            {
                int32_t            segW = randl(1, 8);
                int32_t            segH = randl(1, 8);
                std::vector<float> hSeg(segW * segH);
                FillRandomSegment(hSeg);
                NVCVSegment seg(NVCVBoxI{randl(0, inW - 1), randl(0, inH - 1), randl(1, inW), randl(1, inH)},
                                randl(-1, 5), hSeg.data(), segW, segH, 0.5f,
                                NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                               (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)}),
                                NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                               (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}));
                element = std::make_shared<NVCVElement>(type, &seg);
                break;
            }
            case NVCVOSDType::NVCV_OSD_POINT:
            {
                NVCVPoint point;
                point.centerPos.x = randl(0, inW - 1);
                point.centerPos.y = randl(0, inH - 1);
                point.radius      = randl(1, 50);
                point.color = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                               (unsigned char)randl(0, 255)};
                element     = std::make_shared<NVCVElement>(type, &point);
                break;
            }
            case NVCVOSDType::NVCV_OSD_LINE:
            {
                NVCVLine line;
                line.pos0.x    = randl(0, inW - 1);
                line.pos0.y    = randl(0, inH - 1);
                line.pos1.x    = randl(0, inW - 1);
                line.pos1.y    = randl(0, inH - 1);
                line.thickness = randl(1, 5);
                line.color = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                              (unsigned char)randl(0, 255)};
                line.interpolation = true;
                element            = std::make_shared<NVCVElement>(type, &line);
                break;
            }
            case NVCVOSDType::NVCV_OSD_POLYLINE:
            {
                std::vector<int32_t> pts = {randl(0, inW - 1), randl(0, inH - 1), randl(0, inW - 1),
                                            randl(0, inH - 1), randl(0, inW - 1), randl(0, inH - 1)};
                NVCVPolyLine         polyLine(pts.data(), 3, randl(1, 5), (bool)randl(0, 1),
                                              NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                             (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                              NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                             (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                              true);
                element = std::make_shared<NVCVElement>(type, &polyLine);
                break;
            }
            case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
            {
                NVCVRotatedBox rb;
                rb.centerPos.x = randl(0, inW - 1);
                rb.centerPos.y = randl(0, inH - 1);
                rb.width       = randl(1, inW);
                rb.height      = randl(1, inH);
                rb.yaw         = 0.02f * static_cast<float>(randl(1, 314));
                rb.thickness   = randl(-1, 5);
                rb.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)}; // alpha: 1-255
                rb.bgColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                              (unsigned char)randl(0, 255)};
                rb.interpolation = (bool)randl(0, 1);
                element          = std::make_shared<NVCVElement>(type, &rb);
                break;
            }
            case NVCVOSDType::NVCV_OSD_CIRCLE:
            {
                NVCVCircle circle;
                circle.centerPos.x = randl(0, inW - 1);
                circle.centerPos.y = randl(0, inH - 1);
                circle.radius      = randl(1, 50);
                circle.thickness   = randl(1, 5);
                circle.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                      (unsigned char)randl(0, 255), (unsigned char)randl(1, 255)}; // alpha: 1-255
                circle.bgColor     = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
                element            = std::make_shared<NVCVElement>(type, &circle);
                break;
            }
            case NVCVOSDType::NVCV_OSD_ARROW:
            {
                NVCVArrow arrow;
                arrow.pos0.x    = randl(0, inW - 1);
                arrow.pos0.y    = randl(0, inH - 1);
                arrow.pos1.x    = randl(0, inW - 1);
                arrow.pos1.y    = randl(0, inH - 1);
                arrow.arrowSize = randl(1, 5);
                arrow.thickness = randl(1, 5);
                arrow.color = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                               (unsigned char)randl(0, 255)};
                arrow.interpolation = false;
                element             = std::make_shared<NVCVElement>(type, &arrow);
                break;
            }
            case NVCVOSDType::NVCV_OSD_CLOCK:
            {
                auto clock = NVCVClock{static_cast<NVCVClockFormat>(randl(1, 3)),
                                       CurrentTime(),
                                       5 * randl(1, 10),
                                       DEFAULT_OSD_FONT,
                                       NVCVPointI({randl(0, inW - 1), randl(0, inH - 1)}),
                                       NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                       NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)})};
                element    = std::make_shared<NVCVElement>(type, &clock);
                break;
            }
            default:
                break;
            }
            curVec.push_back(element);
        }
        elementVec.push_back(curVec);
    }

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, inW, inH, format);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    auto inBufSize  = static_cast<int>(inSampleStride * inAccess->numSamples());
    auto outBufSize = static_cast<int>(outSampleStride * outAccess->numSamples());

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));

    std::vector<uint8_t> outHost(outBufSize);
    std::vector<uint8_t> inHost(inBufSize, 0xFF);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outHost.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_NE(inHost, outHost) << "Output should differ from input after drawing OSD elements";
}

// clang-format off
NVCV_TEST_SUITE_P(OpOSD, test::ValueList<int, int, int, int, int, nvcv::ImageFormat>
{
    //  inN,    inW,    inH,    num,    seed,   format
    {   1,      224,    224,    100,    3,      nvcv::FMT_RGBA8 },
    {   2,      224,    224,    100,    3,      nvcv::FMT_RGBA8 },
    {   8,      224,    224,    100,    7,      nvcv::FMT_RGBA8 },
    {   16,     224,    224,    100,    11,     nvcv::FMT_RGBA8 },
    {   1,      224,    224,    100,    3,      nvcv::FMT_RGB8  },
    {   8,      224,    224,    100,    7,      nvcv::FMT_RGB8  },
    {   16,     224,    224,    100,    11,     nvcv::FMT_RGB8  },
    // disable 1280x720, 1920x1080, 3840x2160 tests due to flakiness (platform-specific numerical differences)
    //{   1,      1280,   720,    100,    23,     nvcv::FMT_RGBA8 },
    //{   1,      1920,   1080,   200,    37,     nvcv::FMT_RGBA8 },
    //{   1,      3840,   2160,   200,    59,     nvcv::FMT_RGBA8 },
    //{   1,      1280,   720,    100,    23,     nvcv::FMT_RGB8  },
    //{   1,      1920,   1080,   200,    37,     nvcv::FMT_RGB8  },
    //{   1,      3840,   2160,   200,    59,     nvcv::FMT_RGB8  },
});

// clang-format on

TEST_P(OpOSD, OSD_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = GetParamValue<0>();
    int               inW    = GetParamValue<1>();
    int               inH    = GetParamValue<2>();
    int               num    = GetParamValue<3>();
    int               sed    = GetParamValue<4>();
    nvcv::ImageFormat format = GetParamValue<5>();
    cvcuda::OSD       op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpOSDPlanar, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int>
{
    // planar format, interleaved format, batch
    {nvcv::FMT_RGB8p,  nvcv::FMT_RGB8,  2},
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8, 1},
});

// clang-format on

TEST_P(OpOSDPlanar, tensor_output_matches_interleaved)
{
    constexpr int width  = 64;
    constexpr int height = 48;
    int           batch  = GetParamValue<2>();

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    for (int n = 0; n < batch; ++n)
    {
        NVCVPoint point;
        point.centerPos.x = width / 2;
        point.centerPos.y = height / 2;
        point.radius      = 7;
        point.color       = {255, 128, 0, 255};

        std::vector<std::shared_ptr<NVCVElement>> curVec;
        curVec.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point));
        elementVec.push_back(curVec);
    }
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    nvcv::test::planar::RunTensorParity(
        GetParamValue<0>(), GetParamValue<1>(), width, height, width, height, batch,
        [&op, &ctx](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst, nvcv::ImageFormat)
        { op(stream, src, dst, (NVCVElements)ctx.get()); });
}

// clang-format on

TEST(OpOSD, OSD_memory)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = 1;
    int               inW    = 224;
    int               inH    = 224;
    int               num    = 100;
    int               sed    = 3;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;
    cvcuda::OSD       op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    //check if data is cleared
    sed++;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD, stb_backend)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = 1;
    int               inW    = 224;
    int               inH    = 224;
    int               sed    = 22;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;
    cvcuda::OSD       op;
    NVCVOSDType       type = NVCVOSDType::NVCV_OSD_TEXT;

    Rng().seed(sed);

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    std::vector<std::string> testStrings{
        // valid
        "Hello!", "\u00E9", "\u20AC", "\U0001F600",
        // invalid
        "\xC0\x80", "\xc2\x00", // second bytes
        "\xde\x82\xa0", "\xe2\x00\xa0", "\xe2\x82\x00", "\xe0\x9f\xa0",
        "\xed\xbf\xa0", // three bytes
        "\xf4\x90\x84\x9e", "\xf0\x9d\x04\x9e", "\xf0\x80\x84\x9e", "\xf4\xbf\x84\x9e", "\xf0\x9d\x84\x0e",
        "\xf5\xc0\x84\x9e", "\xf3\xc0\x84\x9e" // four bytes
    };

    std::vector<std::shared_ptr<NVCVElement>> textVec;
    for (const auto &testStr : testStrings)
    {
        std::shared_ptr<NVCVElement> element;
        auto                         text = NVCVText(testStr.c_str(), 5 * randl(1, 10), DEFAULT_OSD_FONT,
                                                     NVCVPointI({randl(0, inW - 1), randl(0, inH - 1)}),
                                                     NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                                    (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                                     NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                                    (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}));
        element                           = std::make_shared<NVCVElement>(type, &text);
        textVec.push_back(element);
    }

    elementVec.push_back(textVec);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, inW, inH, format);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(outAccess);

    long inSampleStride  = inAccess->numRows() * inAccess->rowStride();
    long outSampleStride = outAccess->numRows() * outAccess->rowStride();

    auto inBufSize  = static_cast<int>(inSampleStride * inAccess->numSamples());
    auto outBufSize = static_cast<int>(outSampleStride * outAccess->numSamples());

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));

    std::vector<uint8_t> outHost(outBufSize);
    std::vector<uint8_t> inHost(inBufSize, 0xFF);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outHost.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_NE(inHost, outHost) << "Output should differ from input after rendering text (STB backend)";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Negative, create_with_null_handle)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, cvcudaOSDCreate(nullptr));
}

TEST(OpOSD_Negative, rejects_hwc_to_nhwc_layout_mismatch)
{
    constexpr int width  = 32;
    constexpr int height = 32;

    NVCVPoint point;
    point.centerPos.x = width / 2;
    point.centerPos.y = height / 2;
    point.radius      = 3;
    point.color       = {255, 128, 0, 255};

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec{
        {std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point)}};
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto dtype = nvcv::FMT_RGB8.planeDataType(0).channelType(0);

    nvcv::Tensor imgIn(
        {
            {height, width, 3},
            "HWC"
    },
        dtype);
    nvcv::Tensor imgOut(
        {
            {1, height, width, 3},
            "NHWC"
    },
        dtype);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::OSD op;
    EXPECT_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()), nvcv::Exception);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

static void runOSDOperation(const nvcv::Tensor &imgIn, const nvcv::Tensor &imgOut,
                            std::shared_ptr<NVCVElementsImpl> ctx, bool isNegativeTest = false)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);

    long inSampleStride = inAccess->numRows() * inAccess->rowStride();

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, inSampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, inSampleStride));

    cvcuda::OSD op;
    if (isNegativeTest)
    {
        EXPECT_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()), nvcv::Exception);
    }
    else
    {
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD, test_polyLine_cornor_tests)
{
    int               inN    = 1;
    int               inW    = 200;
    int               inH    = 200;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    std::vector<std::shared_ptr<NVCVElement>>              curVec;
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    // is ray intersects segment
    {
        // triangle
        std::vector<int> points = {50, 120, 150, 100, 100, 50};
        NVCVPolyLine     polyLine(points.data(), 3, 1, true, {255, 0, 0, 255}, {0, 255, 0, 128}, true);

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POLYLINE, &polyLine);
        curVec.push_back(element);
    }

#ifndef ENABLE_SANITIZER
    // invalid odsType
    {
        auto text = NVCVText("Hello", 20, DEFAULT_OSD_FONT, NVCVPointI({10, 10}), NVCVColorRGBA({255, 0, 0, 255}),
                             NVCVColorRGBA({0, 0, 0, 0}));

        auto element = std::make_shared<NVCVElement>(static_cast<NVCVOSDType>(255), &text);
        curVec.push_back(element);
    }
#endif

    // invalid font size
    {
        auto text = NVCVText("Hello", 0, DEFAULT_OSD_FONT, NVCVPointI({10, 10}), NVCVColorRGBA({255, 0, 0, 255}),
                             NVCVColorRGBA({0, 0, 0, 0}));

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
        curVec.push_back(element);
    }

    // invalid font size for clock
    {
        auto clock = NVCVClock{
            static_cast<NVCVClockFormat>(randl(1, 3)),
            CurrentTime(),
            0,
            DEFAULT_OSD_FONT,
            NVCVPointI({randl(0, 10), randl(0, 10)}
            ),
            {    255,   0,       0, 255},
            {      0, 255,       0, 128}
        };
        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CLOCK, &clock);
        curVec.push_back(element);
    }

    elementVec.push_back(curVec);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, inW, inH, format);

    runOSDOperation(imgIn, imgOut, ctx);
}

TEST(OpOSD, test_inplace)
{
    int               inN    = 1;
    int               inW    = 100;
    int               inH    = 100;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    NVCVPoint point;
    point.centerPos.x = 10;
    point.centerPos.y = 10;
    point.radius      = 2;
    point.color       = {255, 0, 0, 255};

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              pointVec;
    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
    pointVec.push_back(element);
    elementVec.push_back(pointVec);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor img = nvcv::util::CreateTensor(inN, inW, inH, format);

    runOSDOperation(img, img, ctx);
}

TEST(OpOSD, test_nothing_to_draw)
{
    int               inN    = 1;
    int               inW    = 100;
    int               inH    = 100;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    // triangle
    std::vector<int> points = {50, 120, 150, 100, 100, 50};

    // numPoints < 2
    NVCVPolyLine polyLine(points.data(), 1, 1, true, {255, 0, 0, 255}, {0, 255, 0, 128}, true);

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              curVec;
    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POLYLINE, &polyLine);
    curVec.push_back(element);
    elementVec.push_back(curVec);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor img = nvcv::util::CreateTensor(inN, inW, inH, format);

    runOSDOperation(img, img, ctx);
}

// clang-format off
NVCV_TEST_SUITE_P(OpOSD_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, int, int, int>
    {
        {nvcv::FMT_RGB8p, nvcv::FMT_RGB8, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8p, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGBf32, 10, 10, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 10, 8, 10},
        {nvcv::FMT_RGB8, nvcv::FMT_RGB8, 10, 10, 8},
    });

// clang-format on

TEST_P(OpOSD_Negative, invalid_parameters)
{
    nvcv::ImageFormat inFormat  = GetParamValue<0>();
    nvcv::ImageFormat outFormat = GetParamValue<1>();
    int               inN       = GetParamValue<2>();
    int               outN      = GetParamValue<3>();
    int               elementsN = GetParamValue<4>();

    int inW = 224;
    int inH = 224;
    int num = 5;

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    for (int n = 0; n < elementsN; n++)
    {
        std::vector<std::shared_ptr<NVCVElement>> curVec;
        for (int i = 0; i < num; i++)
        {
            auto text = NVCVText("Hello", 2, DEFAULT_OSD_FONT, NVCVPointI({10, 10}), NVCVColorRGBA({255, 0, 0, 255}),
                                 NVCVColorRGBA({0, 0, 0, 0}));

            auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
            curVec.push_back(element);
        }
        elementVec.push_back(curVec);
    }

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, inFormat);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(outN, inW, inH, outFormat);

    runOSDOperation(imgIn, imgOut, ctx, true);
}

TEST(OpOSD_Negative, invalid_osd_type)
{
    int               inN    = 1;
    int               inW    = 100;
    int               inH    = 100;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;

    std::vector<std::shared_ptr<NVCVElement>>              curVec;
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    // invalid odsType
    {
        auto text = NVCVText("Hello", 20, DEFAULT_OSD_FONT, NVCVPointI({10, 10}), NVCVColorRGBA({255, 0, 0, 255}),
                             NVCVColorRGBA({0, 0, 0, 0}));

        auto element1 = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_NONE, &text);
        curVec.push_back(element1);
    }

    elementVec.push_back(curVec);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(inN, inW, inH, format);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(inN, inW, inH, format);

    runOSDOperation(imgIn, imgOut, ctx, true);
}

TEST(OpOSD_Smoke, operator_creation)
{
    // Verify operator can be created and destroyed
    EXPECT_NO_THROW(cvcuda::OSD op);
}

TEST(OpOSD_Smoke, rectangle_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 224, 224, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with gray background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 128, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with rectangle
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVBndBoxI bbox;
    bbox.box.x       = 50;
    bbox.box.y       = 50;
    bbox.box.width   = 100;
    bbox.box.height  = 80;
    bbox.thickness   = 2;
    bbox.fillColor   = {255, 0, 0, 128};
    bbox.borderColor = {0, 255, 0, 255};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
    elements.push_back(element);
    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Validation strategy: Verify OSD rectangle element is drawn
    // Check that output differs from zeroed buffer (rectangle was drawn)
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    // Check that some pixels were drawn (output is not all zeros)
    bool hasNonZero = false;
    for (uint8_t value : outData)
    {
        if (value != 0)
        {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero) << "Output should contain pixels after drawing OSD rectangle";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, text_element)
{
    // REVISIT: This test has known issues on Jetson platforms (aarch64)
    // Skip on aarch64 until the underlying text rendering issue is resolved
#if defined(__aarch64__)
    GTEST_SKIP() << "Skipped: OpOSD text_element test has known issues on Jetson/aarch64 platforms";
#endif

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGBA8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize RGBA input with dark gray background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long                 sampleStride = inAccess->numRows() * inAccess->rowStride();
    std::vector<uint8_t> inData(sampleStride);
    for (size_t i = 0; i < inData.size(); i += 4)
    {
        inData[i]     = 50;  // R
        inData[i + 1] = 50;  // G
        inData[i + 2] = 50;  // B
        inData[i + 3] = 255; // A
    }
    EXPECT_EQ(cudaSuccess, cudaMemcpy(input->basePtr(), inData.data(), sampleStride, cudaMemcpyHostToDevice));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with text
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    auto text = NVCVText("Test", 20, DEFAULT_OSD_FONT, NVCVPointI({10, 10}), NVCVColorRGBA({255, 255, 255, 255}),
                         NVCVColorRGBA({0, 0, 0, 128}));

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text);
    elements.push_back(element);
    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Verify text was rendered: white text pixels (255,255,255) must exist in output
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    bool hasWhiteText = false;
    for (size_t i = 0; i + 3 < outData.size(); i += 4)
    {
        if (outData[i] == 255 && outData[i + 1] == 255 && outData[i + 2] == 255)
        {
            hasWhiteText = true;
            break;
        }
    }
    EXPECT_TRUE(hasWhiteText) << "Output should contain white text pixels (255,255,255)";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, line_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 200, 200, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 200, 200, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 100, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with line
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVLine line;
    line.pos0.x        = 20;
    line.pos0.y        = 20;
    line.pos1.x        = 180;
    line.pos1.y        = 180;
    line.thickness     = 3;
    line.color         = {255, 0, 255, 255};
    line.interpolation = NVCV_INTERP_LINEAR;

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
    elements.push_back(element);
    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, point_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 150, 150, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 150, 150, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 80, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with point
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVPoint point;
    point.centerPos.x = 75;
    point.centerPos.y = 75;
    point.radius      = 5;
    point.color       = {255, 128, 0, 255};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point);
    elements.push_back(element);
    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, circle_element)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_RGBA8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 256, 256, nvcv::FMT_RGBA8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 60, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with circle
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    NVCVCircle circle;
    circle.centerPos.x = 128;
    circle.centerPos.y = 128;
    circle.radius      = 50;
    circle.thickness   = 3;
    circle.borderColor = {0, 255, 255, 255};
    circle.bgColor     = {255, 0, 255, 100};

    auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
    elements.push_back(element);
    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, multiple_elements)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 640, 480, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with black background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, sampleStride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

    // Create OSD context with multiple different element types
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;

    // Rectangle
    NVCVBndBoxI bbox;
    bbox.box.x       = 50;
    bbox.box.y       = 50;
    bbox.box.width   = 200;
    bbox.box.height  = 150;
    bbox.thickness   = 2;
    bbox.fillColor   = {255, 0, 0, 0};
    bbox.borderColor = {255, 0, 0, 255};
    auto element1    = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
    elements.push_back(element1);

    // Line
    NVCVLine line;
    line.pos0.x        = 300;
    line.pos0.y        = 100;
    line.pos1.x        = 500;
    line.pos1.y        = 300;
    line.thickness     = 2;
    line.color         = {0, 255, 0, 255};
    line.interpolation = NVCV_INTERP_LINEAR;
    auto element2      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line);
    elements.push_back(element2);

    // Circle
    NVCVCircle circle;
    circle.centerPos.x = 400;
    circle.centerPos.y = 250;
    circle.radius      = 40;
    circle.thickness   = 2;
    circle.borderColor = {0, 0, 255, 255};
    circle.bgColor     = {0, 0, 0, 0};
    auto element3      = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_CIRCLE, &circle);
    elements.push_back(element3);

    elementVec.push_back(elements);

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Basic validation: Check output contains colors from the different elements
    std::vector<uint8_t> outData(sampleStride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));

    bool hasRed   = false; // From rectangle border
    bool hasGreen = false; // From line
    bool hasBlue  = false; // From circle border

    for (size_t i = 0; i + 2 < outData.size(); i += 3)
    {
        uint8_t r = outData[i];
        uint8_t g = outData[i + 1];
        uint8_t b = outData[i + 2];

        if (r == 255 && g == 0 && b == 0)
            hasRed = true;
        if (r == 0 && g == 255 && b == 0)
            hasGreen = true;
        if (r == 0 && g == 0 && b == 255)
            hasBlue = true;

        if (hasRed && hasGreen && hasBlue)
            break;
    }

    EXPECT_TRUE(hasRed) << "Output should contain red pixels from rectangle";
    EXPECT_TRUE(hasGreen) << "Output should contain green pixels from line";
    EXPECT_TRUE(hasBlue) << "Output should contain blue pixels from circle";

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, memory_management)
{
    // Run operator multiple times to verify no memory leaks
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::OSD op;

    for (int iter = 0; iter < 5; iter++) // NOSONAR
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 320, 240, nvcv::FMT_RGB8);

        auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);

        // Initialize input with known background
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
        ASSERT_TRUE(inAccess);
        long sampleStride = inAccess->numRows() * inAccess->rowStride();
        EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 90, sampleStride));
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 30 + iter * 20;
        bbox.box.y       = 30 + iter * 15;
        bbox.box.width   = 100;
        bbox.box.height  = 80;
        bbox.thickness   = 2;
        bbox.fillColor   = {255, 0, 0, 128};
        bbox.borderColor = {0, 255, 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, edge_cases)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input with known background
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 70, sampleStride));

    cvcuda::OSD op;

    // Empty elements list - output should match input
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;
        // Empty elements
        elementVec.push_back(elements);
        auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Empty elements list: operator returns early, output remains uninitialized
        // For a smoke test, we just verify the operation doesn't crash
    }

    // Element at image boundary - should draw without crashing
    {
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 0;
        bbox.box.y       = 0;
        bbox.box.width   = 100;
        bbox.box.height  = 100;
        bbox.thickness   = 1;
        bbox.fillColor   = {0, 0, 0, 0};
        bbox.borderColor = {255, 255, 255, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);
        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        // Verify white border was drawn at boundary
        std::vector<uint8_t> outData(sampleStride);
        EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), sampleStride, cudaMemcpyDeviceToHost));
        bool hasWhite = false;
        for (size_t i = 0; i + 2 < outData.size(); i += 3)
        {
            if (outData[i] == 255 && outData[i + 1] == 255 && outData[i + 2] == 255)
            {
                hasWhite = true;
                break;
            }
        }
        EXPECT_TRUE(hasWhite) << "Output should contain white border pixels at image boundary";
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, batch_processing)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    batchSize = 3;
    nvcv::Tensor imgIn     = nvcv::util::CreateTensor(batchSize, 224, 224, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut    = nvcv::util::CreateTensor(batchSize, 224, 224, nvcv::FMT_RGB8);

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Initialize input tensors for batch
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    ASSERT_TRUE(inAccess);
    long sampleStride = inAccess->numRows() * inAccess->rowStride();
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 110, sampleStride * batchSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * batchSize));

    // Create OSD elements for each batch item
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    for (int b = 0; b < batchSize; b++)
    {
        std::vector<std::shared_ptr<NVCVElement>> elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = 40 + b * 20;
        bbox.box.y       = 40 + b * 20;
        bbox.box.width   = 120;
        bbox.box.height  = 100;
        bbox.thickness   = 2;
        bbox.fillColor   = {static_cast<unsigned char>(50 * b), static_cast<unsigned char>(100 + 50 * b), 200, 128};
        bbox.borderColor = {255, static_cast<unsigned char>(255 - 50 * b), 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);
    }

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpOSD_Smoke, various_image_sizes)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::OSD op;

    // Test different image sizes
    std::vector<std::pair<int, int>> sizes = {
        { 128,  128},
        { 256,  256},
        { 640,  480},
        {1920, 1080}
    };

    for (auto [w, h] : sizes) // NOSONAR
    {
        nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, w, h, nvcv::FMT_RGB8);
        nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, w, h, nvcv::FMT_RGB8);

        auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
        auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);

        // Initialize input for each size
        auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
        ASSERT_TRUE(inAccess);
        long sampleStride = inAccess->numRows() * inAccess->rowStride();
        EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 120, sampleStride));
        EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride));

        std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
        std::vector<std::shared_ptr<NVCVElement>>              elements;

        NVCVBndBoxI bbox;
        bbox.box.x       = w / 4;
        bbox.box.y       = h / 4;
        bbox.box.width   = w / 2;
        bbox.box.height  = h / 2;
        bbox.thickness   = 2;
        bbox.fillColor   = {100, 150, 200, 50};
        bbox.borderColor = {255, 255, 0, 255};

        auto element = std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox);
        elements.push_back(element);
        elementVec.push_back(elements);

        auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

        EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get())) << "Failed with size " << w << "x" << h;
        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// ============================================================
// Pixel-precise rendering tests
//
// These tests verify that rendered elements appear at exact pixel
// coordinates with exact colors, using the kernel's blending formula:
//
//   blend_alpha = ((bg_a * (255 - fg_a)) >> 8) + fg_a
//   out_c = ((in_c * bg_a * (255 - fg_a)) >> 8 + fg_c * fg_a) / blend_alpha
//
// For RGB8, bg_a is hardcoded to 255 in the kernel. When fg_a == 255,
// blend_alpha == 255 and out_c == fg_c exactly.
// ============================================================

// Helper: run a single-batch OSD op and return the output as a host buffer.
// imgIn is initialized to bgFill before calling the op.
static std::vector<uint8_t> runOSDAndGetOutput(const nvcv::Tensor &imgIn, const nvcv::Tensor &imgOut,
                                               std::shared_ptr<NVCVElementsImpl> ctx, uint8_t bgFill = 0)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*input);
    long stride   = inAccess->numRows() * inAccess->rowStride();

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), bgFill, stride));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), bgFill, stride));

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> outData(stride);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), stride, cudaMemcpyDeviceToHost));
    return outData;
}

// Returns the row stride of the first sample of a tensor.
static long getRowStride(const nvcv::Tensor &t)
{
    auto data   = t.exportData<nvcv::TensorDataStridedCuda>();
    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*data);
    return access->rowStride();
}

TEST(OpOSD_Pixel, batch_commands_are_pixel_exact_and_isolated)
{
    constexpr int width     = 96;
    constexpr int height    = 96;
    constexpr int batchSize = 5;

    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(batchSize, width, height, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(batchSize, width, height, nvcv::FMT_RGB8);

    constexpr std::array<std::array<int, 2>, batchSize> centers = {
        {{20, 20}, {34, 20}, {48, 48}, {76, 76}, {34, 76}}
    };
    constexpr std::array<std::array<unsigned char, 3>, batchSize> colors = {
        {{255, 0, 0}, {255, 255, 0}, {0, 255, 0}, {0, 0, 255}, {255, 0, 255}}
    };
    constexpr std::array<bool, batchSize> hasCommand = {true, false, true, true, false};

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    for (int batch = 0; batch < batchSize; ++batch)
    {
        if (!hasCommand[batch])
        {
            elementVec.emplace_back();
            continue;
        }

        NVCVPoint point;
        point.centerPos.x = centers[batch][0];
        point.centerPos.y = centers[batch][1];
        point.radius      = 6;
        point.color       = {colors[batch][0], colors[batch][1], colors[batch][2], 255};

        std::vector<std::shared_ptr<NVCVElement>> elements;
        elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point));

        if (batch == 0 || batch == 2)
        {
            // The first text is fully outside the image and must not advance the GPU text-line index.
            NVCVText text("A", 20, DEFAULT_OSD_FONT,
                          batch == 0 ? NVCVPointI({width - 1, height - 1}) : NVCVPointI({4, 70}),
                          NVCVColorRGBA({255, 255, 255, 255}), NVCVColorRGBA({0, 0, 0, 0}));
            elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_TEXT, &text));
        }

        elementVec.push_back(std::move(elements));
    }

    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto input  = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    auto output = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    auto access = nvcv::TensorDataAccessStridedImagePlanar::Create(*output);
    ASSERT_TRUE(access);

    long rowStride    = access->rowStride();
    long sampleStride = access->numRows() * rowStride;
    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0, sampleStride * batchSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0, sampleStride * batchSize));

    cvcuda::OSD op;
    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> outData(sampleStride * batchSize);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outData.data(), output->basePtr(), outData.size(), cudaMemcpyDeviceToHost));

    auto px = [&outData, rowStride, sampleStride](int batch, int x, int y, int channel)
    {
        return outData[batch * sampleStride + y * rowStride + x * 3 + channel];
    };

    for (int batch = 0; batch < batchSize; ++batch)
    {
        for (int commandBatch = 0; commandBatch < batchSize; ++commandBatch)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                unsigned char expected = hasCommand[batch] && batch == commandBatch ? colors[batch][channel] : 0;
                EXPECT_EQ(px(batch, centers[commandBatch][0], centers[commandBatch][1], channel), expected)
                    << "batch=" << batch << " commandBatch=" << commandBatch << " channel=" << channel;
            }
        }
    }
}

TEST(OpOSD_Pixel, point_exact_color)
{
    // Draw a red point (rendered as filled circle) at (50,50) with radius=10
    // on a black RGB8 background. Alpha=255 → no blending, center pixel = source color.
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    NVCVPoint point;
    point.centerPos.x = 50;
    point.centerPos.y = 50;
    point.radius      = 10;
    point.color       = {255, 0, 0, 255};

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_POINT, &point));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, 0);
    long rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Center of point — all MSAA sub-samples are well inside radius=10, so alpha is 255.
    EXPECT_EQ(px(50, 50, 0), 255) << "center: red channel";
    EXPECT_EQ(px(50, 50, 1), 0) << "center: green channel";
    EXPECT_EQ(px(50, 50, 2), 0) << "center: blue channel";

    // Exterior — untouched
    EXPECT_EQ(px(5, 5, 0), 0) << "exterior: red channel";
    EXPECT_EQ(px(5, 5, 1), 0) << "exterior: green channel";
    EXPECT_EQ(px(5, 5, 2), 0) << "exterior: blue channel";
}

TEST(OpOSD_Pixel, alpha_blend_known_value)
{
    // Draw a filled rect with color=(200,0,0,128) on a black RGB8 background.
    // RGB8 kernel hardcodes bg_alpha=255. With fg_alpha=128, in_c=0:
    //   blend_alpha = ((255 * 127) >> 8) + 128 = 126 + 128 = 254
    //   out_r = (200 * 128) / 254 = 100
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    NVCVBndBoxI bbox;
    bbox.box.x       = 20;
    bbox.box.y       = 20;
    bbox.box.width   = 60;
    bbox.box.height  = 60;
    bbox.thickness   = 2;
    bbox.fillColor   = {200, 0, 0, 128};
    bbox.borderColor = {0, 0, 0, 0}; // transparent border

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, 0);
    long rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Interior pixel — blended result from the formula above
    EXPECT_EQ(px(50, 50, 0), 100) << "interior: red channel (alpha-blended)";
    EXPECT_EQ(px(50, 50, 1), 0) << "interior: green channel";
    EXPECT_EQ(px(50, 50, 2), 0) << "interior: blue channel";

    // Exterior — unchanged
    EXPECT_EQ(px(5, 5, 0), 0) << "exterior unchanged";
}

TEST(OpOSD_Pixel, rect_border_position)
{
    // Draw a border-only rect at (20,20) size 60x60, thickness=1, border=(0,255,0,255).
    // Fill is transparent. Verify border pixels and interior/exterior are correct.
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    NVCVBndBoxI bbox;
    bbox.box.x       = 20;
    bbox.box.y       = 20;
    bbox.box.width   = 60;
    bbox.box.height  = 60;
    bbox.thickness   = 1;
    bbox.fillColor   = {0, 0, 0, 0}; // transparent fill
    bbox.borderColor = {0, 255, 0, 255};

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, 0);
    long rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Left border, midway — green
    EXPECT_EQ(px(20, 30, 0), 0) << "left border: red";
    EXPECT_EQ(px(20, 30, 1), 255) << "left border: green";
    EXPECT_EQ(px(20, 30, 2), 0) << "left border: blue";

    // Right border, midway — green (box spans x=[20,79])
    EXPECT_EQ(px(79, 30, 0), 0) << "right border: red";
    EXPECT_EQ(px(79, 30, 1), 255) << "right border: green";
    EXPECT_EQ(px(79, 30, 2), 0) << "right border: blue";

    // Interior — fill is transparent, should be black
    EXPECT_EQ(px(50, 50, 0), 0) << "interior: red";
    EXPECT_EQ(px(50, 50, 1), 0) << "interior: green";
    EXPECT_EQ(px(50, 50, 2), 0) << "interior: blue";

    // Exterior — untouched
    EXPECT_EQ(px(5, 5, 0), 0) << "exterior: red";
    EXPECT_EQ(px(5, 5, 1), 0) << "exterior: green";
    EXPECT_EQ(px(5, 5, 2), 0) << "exterior: blue";
}

TEST(OpOSD_Pixel, segment_full_mask_draws_color)
{
    // Segment with all mask values = 1.0f (> threshold 0.5f). Interior pixels should
    // be colored. The kernel uses strict > comparison, and bilinear interpolation on
    // the thresholded mask gives full alpha at the center of the box.
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    const int32_t      segW = 8;
    const int32_t      segH = 8;
    std::vector<float> mask(segW * segH, 1.0f);

    NVCVSegment seg(NVCVBoxI{20, 20, 40, 40}, 0, mask.data(), segW, segH, 0.5f,
                    NVCVColorRGBA({0, 0, 0, 0}),      // transparent border
                    NVCVColorRGBA({0, 0, 255, 255})); // blue fill

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_SEGMENT, &seg));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, 0);
    long rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Center of bounding box — mask is fully above threshold, blue should be drawn
    EXPECT_GT(px(30, 30, 2), 0) << "center of box: blue channel should be non-zero";

    // Exterior — untouched
    EXPECT_EQ(px(5, 5, 0), 0) << "exterior: red";
    EXPECT_EQ(px(5, 5, 1), 0) << "exterior: green";
    EXPECT_EQ(px(5, 5, 2), 0) << "exterior: blue";
}

TEST(OpOSD_Pixel, segment_empty_mask_unchanged)
{
    // Segment with all mask values = 0.0f (< threshold 0.5f). Kernel draws nothing —
    // output inside the bounding box must be identical to input.
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    const int32_t      segW = 8;
    const int32_t      segH = 8;
    std::vector<float> mask(segW * segH, 0.0f);

    NVCVSegment seg(NVCVBoxI{20, 20, 40, 40}, 0, mask.data(), segW, segH, 0.5f, NVCVColorRGBA({0, 0, 0, 0}),
                    NVCVColorRGBA({0, 0, 255, 255}));

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_SEGMENT, &seg));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    const uint8_t bgFill    = 50;
    auto          outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, bgFill);
    long          rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Center of bounding box — mask all below threshold, should be unchanged background
    EXPECT_EQ(px(30, 30, 0), bgFill) << "box center: red should equal background";
    EXPECT_EQ(px(30, 30, 1), bgFill) << "box center: green should equal background";
    EXPECT_EQ(px(30, 30, 2), bgFill) << "box center: blue should equal background";

    // Full buffer should equal the input (bgFill everywhere)
    EXPECT_TRUE(std::all_of(outData.begin(), outData.end(), [bgFill](uint8_t v) { return v == bgFill; }))
        << "entire output should equal background when mask is all-zero";
}

TEST(OpOSD_Pixel, horizontal_line_exact_pixels)
{
    // Draw a horizontal line from (10,50) to (90,50), interpolation=false, color=(0,0,255,255).
    // The kernel forces interpolation=false for horizontal lines (pos0.y == pos1.y),
    // producing hard-edge fill. With fg_alpha=255, pixels on the line are exactly (0,0,255).
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, 100, 100, nvcv::FMT_RGB8);

    NVCVLine line;
    line.pos0.x        = 10;
    line.pos0.y        = 50;
    line.pos1.x        = 90;
    line.pos1.y        = 50;
    line.thickness     = 1;
    line.color         = {0, 0, 255, 255};
    line.interpolation = false;

    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;
    std::vector<std::shared_ptr<NVCVElement>>              elements;
    elements.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_LINE, &line));
    elementVec.push_back(elements);
    auto ctx = std::make_shared<NVCVElementsImpl>(elementVec);

    auto outData   = runOSDAndGetOutput(imgIn, imgOut, ctx, 0);
    long rowStride = getRowStride(imgOut);

    auto px = [&outData, &rowStride](int x, int y, int c)
    {
        return outData[y * rowStride + x * 3 + c];
    };

    // Midpoint of line
    EXPECT_EQ(px(50, 50, 0), 0) << "line midpoint: red";
    EXPECT_EQ(px(50, 50, 1), 0) << "line midpoint: green";
    EXPECT_EQ(px(50, 50, 2), 255) << "line midpoint: blue";

    // 5 rows above — untouched
    EXPECT_EQ(px(50, 45, 0), 0) << "above line: red";
    EXPECT_EQ(px(50, 45, 1), 0) << "above line: green";
    EXPECT_EQ(px(50, 45, 2), 0) << "above line: blue";
}

TEST(OpOSD_Pixel, inplace_equals_outofplace)
{
    // Draw the same element out-of-place and in-place; results must be byte-identical.
    const int W = 100;
    const int H = 100;

    NVCVBndBoxI bbox;
    bbox.box.x       = 20;
    bbox.box.y       = 20;
    bbox.box.width   = 60;
    bbox.box.height  = 60;
    bbox.thickness   = 2;
    bbox.fillColor   = {180, 90, 30, 200};
    bbox.borderColor = {255, 0, 128, 255};

    auto makeCtx = [&bbox]()
    {
        std::vector<std::vector<std::shared_ptr<NVCVElement>>> ev;
        std::vector<std::shared_ptr<NVCVElement>>              elems;
        elems.push_back(std::make_shared<NVCVElement>(NVCVOSDType::NVCV_OSD_RECT, &bbox));
        ev.push_back(elems);
        return std::make_shared<NVCVElementsImpl>(ev);
    };

    // Out-of-place
    nvcv::Tensor imgIn  = nvcv::util::CreateTensor(1, W, H, nvcv::FMT_RGB8);
    nvcv::Tensor imgOut = nvcv::util::CreateTensor(1, W, H, nvcv::FMT_RGB8);
    auto         oop    = runOSDAndGetOutput(imgIn, imgOut, makeCtx(), 128);

    // In-place (same tensor for input and output)
    nvcv::Tensor img = nvcv::util::CreateTensor(1, W, H, nvcv::FMT_RGB8);
    auto         inp = runOSDAndGetOutput(img, img, makeCtx(), 128);

    EXPECT_EQ(oop, inp) << "in-place and out-of-place results must be byte-identical";
}
