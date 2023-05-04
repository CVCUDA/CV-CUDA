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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpNonMaximumSuppression.hpp>
#include <nvcv/Rect.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <iostream>
#include <vector>

namespace test = nvcv::test;

constexpr bool operator==(NVCVRectI box1, NVCVRectI box2);

constexpr bool operator!=(NVCVRectI box1, NVCVRectI box2);

std::ostream &operator<<(std::ostream &stream, const NVCVRectI &box);

float IntersectionOverUnion(const NVCVRectI &box1, const NVCVRectI &box2);

void CPUNonMaximumSuppression(const std::vector<NVCVRectI> &in, std::vector<NVCVRectI> &out,
                              const std::vector<float> &scores, float score_threshold, float iou_threshold);

constexpr bool operator==(NVCVRectI box1, NVCVRectI box2)
{
    return box1.x == box2.x && box1.y == box2.y && box1.width == box2.width && box1.height == box2.height;
}

constexpr bool operator!=(NVCVRectI box1, NVCVRectI box2)
{
    return !(box1 == box2);
}

std::ostream &operator<<(std::ostream &stream, const NVCVRectI &box)
{
    stream << "NVCVRectI(x=" << box.x << ", y=" << box.y << ", width=" << box.width << ", height=" << box.height << ")";
    return stream;
}

float IntersectionOverUnion(const NVCVRectI &box1, const NVCVRectI &box2)
{
    float box1Area = box1.width * box1.height;
    float box2Area = box2.width * box2.height;

    auto xInterLeft   = std::max(box1.x, box2.x);
    auto yInterTop    = std::max(box1.y, box2.y);
    auto xInterRight  = std::min(box1.x + box1.width, box2.x + box2.width);
    auto yInterBottom = std::min(box1.y + box1.height, box2.y + box2.height);

    auto widthInter  = xInterRight - xInterLeft;
    auto heightInter = yInterBottom - yInterTop;

    float interArea = widthInter * heightInter;

    return interArea / (box1Area + box2Area - interArea);
}

void CPUNonMaximumSuppression(const std::vector<NVCVRectI> &in, std::vector<NVCVRectI> &out,
                              const std::vector<float> &scores, float score_threshold, float iou_threshold)
{
    const NVCVRectI ZERO_BBOX = {0, 0, 0, 0};

    out = in;
    for (size_t i = 0; i < 5; ++i)
    {
        if (scores[i] < score_threshold)
        {
            out[i] = ZERO_BBOX;
        }
    }

    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            if (i != j && out[i] != ZERO_BBOX && in[j] != ZERO_BBOX && scores[i] < scores[j]
                && IntersectionOverUnion(out[i], in[j]) > iou_threshold)
            {
                out[i] = ZERO_BBOX;
            }
        }
    }
}

TEST(OpNonMaximumSuppression, correct_output)
{
    const NVCVRectI ZERO_BBOX = {0, 0, 0, 0};

    auto tshape = nvcv::TensorShape{
        {1, 5, 4},
        nvcv::TENSOR_NCW
    };
    auto         dtype = nvcv::TYPE_S32;
    nvcv::Tensor inBBoxes{tshape, dtype};
    nvcv::Tensor outBBoxes{tshape, dtype};
    nvcv::Tensor inScores{
        nvcv::TensorShape{{1, 5}, nvcv::TENSOR_NW},
        nvcv::TYPE_F32
    };

    auto                   inBBAccess    = nvcv::TensorDataAccessStrided::Create(inBBoxes.exportData());
    auto                   numBBElements = inBBAccess->sampleStride() / sizeof(NVCVRectI);
    std::vector<NVCVRectI> inBBoxValues(numBBElements, ZERO_BBOX);

    auto               inScoreAccess    = nvcv::TensorDataAccessStrided::Create(inScores.exportData());
    auto               numScoreElements = inScoreAccess->sampleStride() / sizeof(float);
    std::vector<float> inScoreValues(numScoreElements, 0.0f);

    inBBoxValues[0]  = {0, 0, 0, 0};
    inScoreValues[0] = .999f;
    inBBoxValues[1]  = {0, 0, 0, 0};
    inScoreValues[1] = 0.f;
    inBBoxValues[2]  = {0, 0, 0, 0};
    inScoreValues[2] = .6f;
    inBBoxValues[3]  = {0, 0, 0, 0};
    inScoreValues[3] = .5f;
    inBBoxValues[4]  = {0, 0, 0, 0};
    inScoreValues[4] = .8f;

    float scoreThresh = 0.5f;
    float iouThresh   = 0.75f;

    decltype(inBBoxValues)  verBBoxValues(numBBElements, ZERO_BBOX);
    decltype(inScoreValues) verScoreValues(numScoreElements, 0.0f);

    test::SetTensorFromVector<NVCVRectI>(inBBoxes.exportData(), inBBoxValues, -1);
    test::GetVectorFromTensor<NVCVRectI>(inBBoxes.exportData(), 0, verBBoxValues);
    ASSERT_EQ(inBBoxValues, verBBoxValues);

    test::SetTensorFromVector<float>(inScores.exportData(), inScoreValues, -1);
    test::GetVectorFromTensor<float>(inScores.exportData(), 0, verScoreValues);
    ASSERT_EQ(inScoreValues, verScoreValues);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::NonMaximumSuppression nms;

    nms(stream, inBBoxes, outBBoxes, inScores, scoreThresh, iouThresh);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    decltype(inBBoxValues) outBBoxValues(numBBElements, ZERO_BBOX);
    decltype(inBBoxValues) outBBoxValues2(numBBElements, ZERO_BBOX);

    test::GetVectorFromTensor<NVCVRectI>(outBBoxes.exportData(), 0, outBBoxValues);

    CPUNonMaximumSuppression(inBBoxValues, outBBoxValues2, inScoreValues, scoreThresh, iouThresh);

    ASSERT_EQ(outBBoxValues.size(), outBBoxValues2.size());

    for (size_t i = 0; i < outBBoxValues.size(); ++i)
    {
        EXPECT_EQ(outBBoxValues[i], outBBoxValues2[i]);
    }
}
