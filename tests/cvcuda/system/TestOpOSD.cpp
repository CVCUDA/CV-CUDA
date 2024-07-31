/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "OsdUtils.cuh"

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpOSD.hpp>
#include <cvcuda/priv/Types.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
using namespace cvcuda::priv;

static int randl(int l, int h)
{
    int value = rand() % (h - l + 1);
    return l + value;
}

#pragma GCC push_options
#pragma GCC optimize("O1")

static void setGoldBuffer(std::vector<uint8_t> &vect, nvcv::ImageFormat format,
                          const nvcv::TensorDataAccessStridedImagePlanar &data, nvcv::Byte *inBuf,
                          std::shared_ptr<NVCVElementsImpl> ctx, cudaStream_t stream)
{
    auto context = cuosd_context_create();

    for (int n = 0; n < ctx->batch(); n++)
    {
        test::osd::Image *image = test::osd::create_image(
            data.numCols(), data.numRows(),
            format == nvcv::FMT_RGBA8 ? test::osd::ImageFormat::RGBA : test::osd::ImageFormat::RGB);
        int bufSize = data.numCols() * data.numRows() * data.numChannels();
        EXPECT_EQ(cudaSuccess, cudaMemcpy(image->data0, inBuf + n * bufSize, bufSize, cudaMemcpyDeviceToDevice));

        auto numElements = ctx->numElementsAt(n);

        for (int i = 0; i < numElements; i++)
        {
            auto element = ctx->elementAt(n, i);
            switch (element->type())
            {
            case NVCVOSDType::NVCV_OSD_RECT:
            {
                auto bbox = *((NVCVBndBoxI *)element->ptr());

                int left   = std::max(std::min(bbox.box.x, data.numCols() - 1), 0);
                int top    = std::max(std::min(bbox.box.y, data.numRows() - 1), 0);
                int right  = std::max(std::min(left + bbox.box.width - 1, data.numCols() - 1), 0);
                int bottom = std::max(std::min(top + bbox.box.height - 1, data.numRows() - 1), 0);

                if (left == right || top == bottom || bbox.box.width <= 0 || bbox.box.height <= 0)
                {
                    continue;
                }

                cuOSDColor borderColor
                    = {bbox.borderColor.r, bbox.borderColor.g, bbox.borderColor.b, bbox.borderColor.a};
                cuOSDColor fillColor = {bbox.fillColor.r, bbox.fillColor.g, bbox.fillColor.b, bbox.fillColor.a};
                cuosd_draw_rectangle(context, left, top, right, bottom, bbox.thickness, borderColor, fillColor);
                break;
            }
            case NVCVOSDType::NVCV_OSD_TEXT:
            {
                auto       text      = *((NVCVText *)element->ptr());
                cuOSDColor fontColor = *(cuOSDColor *)(&text.fontColor);
                cuOSDColor bgColor   = *(cuOSDColor *)(&text.bgColor);
                cuosd_draw_text(context, text.utf8Text, text.fontSize, text.fontName, text.tlPos.x, text.tlPos.y,
                                fontColor, bgColor);
                break;
            }
            case NVCVOSDType::NVCV_OSD_SEGMENT:
            {
                auto segment = (NVCVSegment *)element->ptr();

                int left   = segment->box.x;
                int top    = segment->box.y;
                int right  = left + segment->box.width - 1;
                int bottom = top + segment->box.height - 1;

                if (left == right || top == bottom || segment->box.width <= 0 || segment->box.height <= 0)
                {
                    continue;
                }
                cuOSDColor borderColor = *(cuOSDColor *)(&segment->borderColor);
                cuOSDColor segColor    = *(cuOSDColor *)(&segment->segColor);
                cuosd_draw_segmentmask(context, left, top, right, bottom, segment->thickness, segment->dSeg,
                                       segment->segWidth, segment->segHeight, segment->segThreshold, borderColor,
                                       segColor);
                break;
            }
            case NVCVOSDType::NVCV_OSD_POINT:
            {
                auto       point = *((NVCVPoint *)element->ptr());
                cuOSDColor color = *(cuOSDColor *)(&point.color);
                cuosd_draw_point(context, point.centerPos.x, point.centerPos.y, point.radius, color);
                break;
            }
            case NVCVOSDType::NVCV_OSD_LINE:
            {
                auto       line  = *((NVCVLine *)element->ptr());
                cuOSDColor color = *(cuOSDColor *)(&line.color);
                cuosd_draw_line(context, line.pos0.x, line.pos0.y, line.pos1.x, line.pos1.y, line.thickness, color,
                                line.interpolation);
                break;
            }
            case NVCVOSDType::NVCV_OSD_POLYLINE:
            {
                auto       pl          = (NVCVPolyLine *)element->ptr();
                cuOSDColor borderColor = *(cuOSDColor *)(&pl->borderColor);
                cuOSDColor fill_color  = *(cuOSDColor *)(&pl->fillColor);
                cuosd_draw_polyline(context, pl->hPoints, pl->dPoints, pl->numPoints, pl->thickness, pl->isClosed,
                                    borderColor, pl->interpolation, fill_color);
                break;
            }
            case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
            {
                auto       rb          = *((NVCVRotatedBox *)element->ptr());
                cuOSDColor borderColor = *(cuOSDColor *)(&rb.borderColor);
                cuOSDColor bgColor     = *(cuOSDColor *)(&rb.bgColor);
                cuosd_draw_rotationbox(context, rb.centerPos.x, rb.centerPos.y, rb.width, rb.height, rb.yaw,
                                       rb.thickness, borderColor, rb.interpolation, bgColor);
                break;
            }
            case NVCVOSDType::NVCV_OSD_CIRCLE:
            {
                auto       circle      = *((NVCVCircle *)element->ptr());
                cuOSDColor borderColor = *(cuOSDColor *)(&circle.borderColor);
                cuOSDColor bgColor     = *(cuOSDColor *)(&circle.bgColor);
                cuosd_draw_circle(context, circle.centerPos.x, circle.centerPos.y, circle.radius, circle.thickness,
                                  borderColor, bgColor);
                break;
            }
            case NVCVOSDType::NVCV_OSD_ARROW:
            {
                auto       arrow = *((NVCVArrow *)element->ptr());
                cuOSDColor color = *(cuOSDColor *)(&arrow.color);
                cuosd_draw_arrow(context, arrow.pos0.x, arrow.pos0.y, arrow.pos1.x, arrow.pos1.y, arrow.arrowSize,
                                 arrow.thickness, color, arrow.interpolation);
                break;
            }
            case NVCVOSDType::NVCV_OSD_CLOCK:
            {
                auto             clock       = *((NVCVClock *)element->ptr());
                cuOSDClockFormat clockFormat = (cuOSDClockFormat)(int)(clock.clockFormat);
                cuOSDColor       fontColor   = *(cuOSDColor *)(&clock.fontColor);
                cuOSDColor       bgColor     = *(cuOSDColor *)(&clock.bgColor);
                cuosd_draw_clock(context, clockFormat, clock.time, clock.fontSize, clock.font, clock.tlPos.x,
                                 clock.tlPos.y, fontColor, bgColor);
                break;
            }
            default:
                break;
            }
        }

        test::osd::cuosd_apply(context, image, stream);
        EXPECT_EQ(cudaSuccess, cudaMemcpy(vect.data() + n * bufSize, image->data0, bufSize, cudaMemcpyDeviceToHost));
        test::osd::free_image(image);
    }

    cudaStreamSynchronize(stream);
    cuosd_context_destroy(context);
}

#pragma GCC pop_options

// run operator
static void runOp(cudaStream_t &stream, cvcuda::OSD &op, int &inN, int &inW, int &inH, int &num, int &sed,
                  nvcv::ImageFormat &format)
{
    std::vector<std::vector<std::shared_ptr<NVCVElement>>> elementVec;

    test::osd::Segment  *test_segment  = test::osd::create_segment();
    test::osd::Polyline *test_polyline = test::osd::create_polyline();

    srand(sed);
    for (int n = 0; n < inN; n++)
    {
        std::vector<std::shared_ptr<NVCVElement>> curVec;
        for (int i = 0; i < num; i++)
        {
            NVCVOSDType                  type = (NVCVOSDType)randl(int(NVCV_OSD_NONE) + 1, int(NVCV_OSD_MAX) - 1);
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
                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
                element            = std::make_shared<NVCVElement>(type, &bndBox);
                break;
            }
            case NVCVOSDType::NVCV_OSD_TEXT:
            {
                NVCVText text = NVCVText("abcdefghijklmnopqrstuvwxyz", 5 * randl(1, 10), DEFAULT_OSD_FONT,
                                         NVCVPointI({randl(0, inW - 1), randl(0, inH - 1)}),
                                         NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                        (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                         NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                        (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}));
                element       = std::make_shared<NVCVElement>(type, &text);
                break;
            }
            case NVCVOSDType::NVCV_OSD_SEGMENT:
            {
                NVCVSegment segment = NVCVSegment(
                    NVCVBoxI({randl(0, inW - 1), randl(0, inH - 1), randl(1, inW), randl(1, inH)}), randl(-1, 5),
                    test_segment->data, test_segment->width, test_segment->height, 0.1 * randl(1, 5),
                    NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                   (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                    NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                   (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}));
                element = std::make_shared<NVCVElement>(type, &segment);
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
                NVCVPolyLine pl
                    = NVCVPolyLine(test_polyline->h_pts, test_polyline->n_pts, randl(1, 5), randl(0, 1),
                                   NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                   NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                   true);
                element = std::make_shared<NVCVElement>(type, &pl);
                break;
            }
            case NVCVOSDType::NVCV_OSD_ROTATED_RECT:
            {
                NVCVRotatedBox rb;
                rb.centerPos.x = randl(0, inW - 1);
                rb.centerPos.y = randl(0, inH - 1);
                rb.width       = randl(1, inW);
                rb.height      = randl(1, inH);
                rb.yaw         = 0.02 * randl(1, 314);
                rb.thickness   = randl(1, 5);
                rb.borderColor = {(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                  (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
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
                                      (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)};
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
                NVCVClock clock
                    = NVCVClock{(NVCVClockFormat)(randl(1, 3)),
                                time(0),
                                5 * randl(1, 10),
                                DEFAULT_OSD_FONT,
                                NVCVPointI({randl(0, inW - 1), randl(0, inH - 1)}),
                                NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                               (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)}),
                                NVCVColorRGBA({(unsigned char)randl(0, 255), (unsigned char)randl(0, 255),
                                               (unsigned char)randl(0, 255), (unsigned char)randl(0, 255)})};
                element = std::make_shared<NVCVElement>(type, &clock);
                break;
            }
            default:
                break;
            }
            curVec.push_back(element);
        }
        elementVec.push_back(curVec);
    }

    std::shared_ptr<NVCVElementsImpl> ctx = std::make_shared<NVCVElementsImpl>(elementVec);

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

    int inBufSize  = inSampleStride * inAccess->numSamples();
    int outBufSize = outSampleStride * outAccess->numSamples();

    EXPECT_EQ(cudaSuccess, cudaMemset(input->basePtr(), 0xFF, inSampleStride * inAccess->numSamples()));
    EXPECT_EQ(cudaSuccess, cudaMemset(output->basePtr(), 0xFF, outSampleStride * outAccess->numSamples()));

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, (NVCVElements)ctx.get()));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, format, *inAccess, input->basePtr(), ctx, stream);

    test::osd::free_segment(test_segment);
    test::osd::free_polyline(test_polyline);

    EXPECT_EQ(gold, test);
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
    {   1,      1280,   720,    100,    23,     nvcv::FMT_RGBA8 },
    {   1,      1920,   1080,   200,    37,     nvcv::FMT_RGBA8 },
    {   1,      3840,   2160,   200,    59,     nvcv::FMT_RGBA8 },
    {   1,      1280,   720,    100,    23,     nvcv::FMT_RGB8  },
    {   1,      1920,   1080,   200,    37,     nvcv::FMT_RGB8  },
    {   1,      3840,   2160,   200,    59,     nvcv::FMT_RGB8  },
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
