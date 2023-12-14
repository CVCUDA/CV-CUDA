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

#include "OsdUtils.cuh"

#include <common/ValueTests.hpp>
#include <cvcuda/OpBndBox.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;

static int randl(int l, int h)
{
    int value = rand() % (h - l + 1);
    return l + value;
}

static void setGoldBuffer(std::vector<uint8_t> &vect, nvcv::ImageFormat format,
                          const nvcv::TensorDataAccessStridedImagePlanar &data, nvcv::Byte *inBuf, NVCVBndBoxesI bboxes,
                          cudaStream_t stream)
{
    auto context = cuosd_context_create();

    for (int n = 0; n < bboxes.batch; n++)
    {
        test::osd::Image *image = test::osd::create_image(
            data.numCols(), data.numRows(),
            format == nvcv::FMT_RGBA8 ? test::osd::ImageFormat::RGBA : test::osd::ImageFormat::RGB);
        int bufSize = data.numCols() * data.numRows() * data.numChannels();
        EXPECT_EQ(cudaSuccess, cudaMemcpy(image->data0, inBuf + n * bufSize, bufSize, cudaMemcpyDeviceToDevice));

        auto numBoxes = bboxes.numBoxes[n];

        for (int i = 0; i < numBoxes; i++)
        {
            auto bbox = bboxes.boxes[i];

            int left   = std::max(std::min(bbox.box.x, data.numCols() - 1), 0);
            int top    = std::max(std::min(bbox.box.y, data.numRows() - 1), 0);
            int right  = std::max(std::min(left + bbox.box.width - 1, data.numCols() - 1), 0);
            int bottom = std::max(std::min(top + bbox.box.height - 1, data.numRows() - 1), 0);

            if (left == right || top == bottom || bbox.box.width <= 0 || bbox.box.height <= 0)
            {
                continue;
            }

            int thickness = bbox.thickness;

            cuOSDColor borderColor = {bbox.borderColor.r, bbox.borderColor.g, bbox.borderColor.b, bbox.borderColor.a};
            cuOSDColor fillColor   = {bbox.fillColor.r, bbox.fillColor.g, bbox.fillColor.b, bbox.fillColor.a};

            cuosd_draw_rectangle(context, left, top, right, bottom, thickness, borderColor, fillColor);
        }

        test::osd::cuosd_apply(context, image, stream);

        bboxes.boxes = (NVCVBndBoxI *)((unsigned char *)bboxes.boxes + numBoxes * sizeof(NVCVBndBoxI));
        EXPECT_EQ(cudaSuccess, cudaMemcpy(vect.data() + n * bufSize, image->data0, bufSize, cudaMemcpyDeviceToHost));

        test::osd::free_image(image);
    }
    cudaStreamSynchronize(stream);
    cuosd_context_destroy(context);
}

// run operator
static void runOp(cudaStream_t &stream, cvcuda::BndBox &op, int &inN, int &inW, int &inH, int &num, int &sed,
                  nvcv::ImageFormat &format)
{
    NVCVBndBoxesI            bndBoxes;
    std::vector<int>         numBoxVec;
    std::vector<NVCVBndBoxI> bndBoxVec;

    srand(sed);
    for (int n = 0; n < inN; n++)
    {
        numBoxVec.push_back(num);
        for (int i = 0; i < num; i++)
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
            bndBoxVec.push_back(bndBox);
        }
    }

    bndBoxes.batch    = inN;
    bndBoxes.numBoxes = numBoxVec.data();
    bndBoxes.boxes    = bndBoxVec.data();

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

    EXPECT_NO_THROW(op(stream, imgIn, imgOut, bndBoxes));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), input->basePtr(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), output->basePtr(), outBufSize, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, format, *inAccess, input->basePtr(), bndBoxes, stream);

    EXPECT_EQ(gold, test);
}

// clang-format off
NVCV_TEST_SUITE_P(OpBndBox, test::ValueList<int, int, int, int, int, nvcv::ImageFormat>
{
    //  inN,    inW,    inH,    num,    seed,   format
    {   1,      224,    224,    100,    3,      nvcv::FMT_RGBA8 },
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
TEST_P(OpBndBox, BndBox_sanity)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = GetParamValue<0>();
    int               inW    = GetParamValue<1>();
    int               inH    = GetParamValue<2>();
    int               num    = GetParamValue<3>();
    int               sed    = GetParamValue<4>();
    nvcv::ImageFormat format = GetParamValue<5>();
    cvcuda::BndBox    op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format on
TEST(OpBndBox, BndBox_memory)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int               inN    = 1;
    int               inW    = 224;
    int               inH    = 224;
    int               num    = 100;
    int               sed    = 3;
    nvcv::ImageFormat format = nvcv::FMT_RGBA8;
    cvcuda::BndBox    op;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    //check if data is cleared
    sed++;
    runOp(stream, op, inN, inW, inH, num, sed, format);
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
