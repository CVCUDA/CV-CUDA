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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpNonMaximumSuppression.hpp>
#include <cvcuda/cuda_tools/DropCast.hpp>
#include <cvcuda/cuda_tools/MathOps.hpp>
#include <cvcuda/cuda_tools/StaticCast.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <malloc.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <iostream>
#include <vector>

namespace test = nvcv::test;
namespace util = nvcv::util;

static std::default_random_engine g_rng(std::random_device{}());

template<typename T>
float GoldArea(const T &bbox)
{
    return bbox.z * bbox.w;
}

template<typename T>
float GoldIoU(const T &box1, const T &box2)
{
    int   xInterLeft   = std::max(box1.x, box2.x);
    int   yInterTop    = std::max(box1.y, box2.y);
    int   xInterRight  = std::min(box1.x + box1.z, box2.x + box2.z);
    int   yInterBottom = std::min(box1.y + box1.w, box2.y + box2.w);
    int   widthInter   = xInterRight - xInterLeft;
    int   heightInter  = yInterBottom - yInterTop;
    float interArea    = widthInter * heightInter;
    float iou          = 0.f;
    if (widthInter > 0.f && heightInter > 0.f)
    {
        float unionArea = GoldArea(box1) + GoldArea(box2) - interArea;
        if (unionArea > 0.f)
        {
            iou = interArea / unionArea;
        }
    }
    return iou;
}

inline void GoldNMS(const std::vector<uint8_t> &srcBBVec, std::vector<uint8_t> &dstMkVec,
                    const std::vector<uint8_t> &srcScVec, const long2 &srcBBStrides, const long2 &dstMkStrides,
                    const long2 &srcScStrides, const int2 &shape, float scoreThreshold, float iouThreshold)
{
    for (int x = 0; x < shape.x; ++x)
    {
        for (int y1 = 0; y1 < shape.y; ++y1)
        {
            const float &score1 = util::ValueAt<float>(srcScVec, srcScStrides, int2{x, y1});
            uint8_t     &dst    = util::ValueAt<uint8_t>(dstMkVec, dstMkStrides, int2{x, y1});

            if (score1 < scoreThreshold)
            {
                dst = 0;
                continue;
            }

            const short4 &src1    = util::ValueAt<short4>(srcBBVec, srcBBStrides, int2{x, y1});
            bool          discard = false;

            for (int y2 = 0; y2 < shape.y; ++y2)
            {
                if (y1 == y2)
                {
                    continue;
                }

                const short4 &src2 = util::ValueAt<short4>(srcBBVec, srcBBStrides, int2{x, y2});

                if (GoldIoU(src1, src2) > iouThreshold)
                {
                    const float &score2 = util::ValueAt<float>(srcScVec, srcScStrides, int2{x, y2});
                    if (score1 < score2 || (score1 == score2 && GoldArea(src1) < GoldArea(src2)))
                    {
                        discard = true;
                        break;
                    }
                }
            }

            dst = discard ? 0 : 1;
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpNonMaximumSuppression, test::ValueList<int, int, float, float>
{
    // numSamples, numBBoxes, scThresh, iouThresh,
    {           1,         5,     .50f,      .75f,},
    {           3,        23,     .25f,      .50f,},
    {          10,       123,     .35f,      .45f,},
    {          15,      1234,     .45f,      .65f,},
    {           2,      8765,     .95f,      .85f,},
    {        2000,         4,     .55f,      .25f,},
});

// clang-format on

TEST_P(OpNonMaximumSuppression, correct_output)
{
    int   numSamples = GetParamValue<0>();
    int   numBBoxes  = GetParamValue<1>();
    float scThresh   = GetParamValue<2>();
    float iouThresh  = GetParamValue<3>();

    int2 inShape{numSamples, numBBoxes};

    // clang-format off

    nvcv::Tensor srcBB({{numSamples, numBBoxes}, "NW"}, nvcv::TYPE_4S16);
    nvcv::Tensor dstMk({{numSamples, numBBoxes}, "NW"}, nvcv::TYPE_U8);
    nvcv::Tensor srcSc({{numSamples, numBBoxes}, "NW"}, nvcv::TYPE_F32);

    // clang-format on

    auto srcBBData = srcBB.exportData<nvcv::TensorDataStridedCuda>();
    auto srcScData = srcSc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcScData && srcScData);
    ASSERT_EQ(srcBBData->shape(0), srcScData->shape(0));
    ASSERT_EQ(srcBBData->shape(1), srcScData->shape(1));

    long2 srcBBStrides{srcBBData->stride(0), srcBBData->stride(1)};
    long2 srcScStrides{srcScData->stride(0), srcScData->stride(1)};

    int2 shape = nvcv::cuda::StaticCast<int>(long2{srcBBData->shape(0), srcScData->shape(1)});
    ASSERT_EQ(shape, inShape);

    std::uniform_int_distribution<int16_t> randPos(0, 128), randSize(50, 100), randScore(0, 1024);

    long srcBBBufSize{srcBBStrides.x * shape.x};
    long srcScBufSize{srcScStrides.x * shape.x};

    std::vector<uint8_t> srcBBVec(srcBBBufSize);
    std::vector<uint8_t> srcScVec(srcScBufSize);

    short4 bbox;
    int    halfBBoxes = static_cast<int>(std::ceil(shape.y / 2.f)); // repeat bboxes after pass half total

    for (int x = 0; x < shape.x; ++x)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            if (y < halfBBoxes)
            {
                bbox = short4{randPos(g_rng), randPos(g_rng), randSize(g_rng), randSize(g_rng)};
            }
            else
            {
                bbox = util::ValueAt<short4>(srcBBVec, srcBBStrides, int2{x, y - halfBBoxes});
            }

            util::ValueAt<short4>(srcBBVec, srcBBStrides, int2{x, y}) = bbox;
            util::ValueAt<float>(srcScVec, srcScStrides, int2{x, y})  = randScore(g_rng) / 1024.f;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcBBData->basePtr(), srcBBVec.data(), srcBBBufSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(srcScData->basePtr(), srcScVec.data(), srcScBufSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::NonMaximumSuppression nms;

    EXPECT_NO_THROW(nms(stream, srcBB, dstMk, srcSc, scThresh, iouThresh));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto dstMkData = dstMk.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(dstMkData);
    ASSERT_EQ(dstMkData->shape(0), srcBBData->shape(0));
    ASSERT_EQ(dstMkData->shape(1), srcBBData->shape(1));

    long2 dstMkStrides{dstMkData->stride(0), dstMkData->stride(1)};

    long dstMkBufSize{dstMkStrides.x * shape.x};

    std::vector<uint8_t> dstMkVecTest(dstMkBufSize);
    std::vector<uint8_t> dstMkVecGold(dstMkBufSize);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dstMkVecTest.data(), dstMkData->basePtr(), dstMkBufSize, cudaMemcpyDeviceToHost));

    GoldNMS(srcBBVec, dstMkVecGold, srcScVec, srcBBStrides, dstMkStrides, srcScStrides, shape, scThresh, iouThresh);

    EXPECT_EQ(dstMkVecTest, dstMkVecGold);
}

// clang-format off
NVCV_TEST_SUITE_P(OpNonMaximumSuppression_Negative, test::ValueList<std::string, nvcv::DataType, std::string, nvcv::DataType, std::string, nvcv::DataType, float, int, int, int, int, int, int, int>{
    {"NWC", nvcv::TYPE_S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 5, 5}, // in: rank3 + S16 + last shape is not 4
    {"NWC", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 4, 3, 3, 3, 5, 5, 5}, // in: rank3 + S16 + last shape is not 1
    {"NW", nvcv::TYPE_S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 5, 5}, // in: rank2 + S16
    {"NW", nvcv::TYPE_4U8, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 5, 5}, // in: not S16
    {"NW", nvcv::TYPE_4S16, "NWC", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 4, 3, 3, 3, 5, 5, 5}, // out: rank3 + last shape not 1
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_S16, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 5, 5}, // out: not U8
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NWC", nvcv::TYPE_F32, 0.1f, 4, 3, 3, 3, 5, 5, 5}, // scores : rank3 + last shape not 1
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F64, 0.1f, 1, 3, 3, 3, 5, 5, 5}, // scores: not F32
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.f, 1, 3, 3, 3, 5, 5, 5}, // invalid iou threshold
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 1.5f, 1, 3, 3, 3, 5, 5, 5}, // invalid iou threshold
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 2, 3, 5, 5, 5}, // input, output number of batches is not equal
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 2, 5, 5, 5}, // input, scores number of batches is not equal
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 4, 5}, // input, scores number of boxes is not equal
    {"NW", nvcv::TYPE_4S16, "NW", nvcv::TYPE_U8, "NW", nvcv::TYPE_F32, 0.1f, 1, 3, 3, 3, 5, 5, 4}, // input, scores number of boxes is not equal
});

// clang-format on

TEST_P(OpNonMaximumSuppression_Negative, op)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::string    srcBBLayout     = GetParamValue<0>();
    nvcv::DataType srcBBDatatype   = GetParamValue<1>();
    std::string    dstMkLayout     = GetParamValue<2>();
    nvcv::DataType dstMkDatatype   = GetParamValue<3>();
    std::string    srcScLayout     = GetParamValue<4>();
    nvcv::DataType srcScDatatype   = GetParamValue<5>();
    const float    iouThresh       = GetParamValue<6>();
    const int      lastShape       = GetParamValue<7>();
    const int      numSamplesSrcBB = GetParamValue<8>();
    const int      numSamplesDstMk = GetParamValue<9>();
    const int      numSamplesSrcSc = GetParamValue<10>();
    const int      numBBoxesSrcBB  = GetParamValue<11>();
    const int      numBBoxesDstMk  = GetParamValue<12>();
    const int      numBBoxesSrcSc  = GetParamValue<13>();

    float scThresh = 0.5f;

    nvcv::TensorShape srcBBShape = srcBBLayout.size() == 3 ? nvcv::TensorShape{{numSamplesSrcBB, numBBoxesSrcBB, lastShape}, srcBBLayout.c_str()} : nvcv::TensorShape{{numSamplesSrcBB, numBBoxesSrcBB}, srcBBLayout.c_str()};
    nvcv::TensorShape dstMkShape = dstMkLayout.size() == 3 ? nvcv::TensorShape{{numSamplesDstMk, numBBoxesDstMk, lastShape}, dstMkLayout.c_str()} : nvcv::TensorShape{{numSamplesDstMk, numBBoxesDstMk}, dstMkLayout.c_str()};
    nvcv::TensorShape srcScShape = srcScLayout.size() == 3 ? nvcv::TensorShape{{numSamplesSrcSc, numBBoxesSrcSc, lastShape}, srcScLayout.c_str()} : nvcv::TensorShape{{numSamplesSrcSc, numBBoxesSrcSc}, srcScLayout.c_str()};

    nvcv::Tensor srcBB(srcBBShape, srcBBDatatype);
    nvcv::Tensor dstMk(dstMkShape, dstMkDatatype);
    nvcv::Tensor srcSc(srcScShape, srcScDatatype);

    cvcuda::NonMaximumSuppression nms;

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&] { nms(stream, srcBB, dstMk, srcSc, scThresh, iouThresh); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
