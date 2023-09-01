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

#include <common/ValueTests.hpp>
#include <cvcuda/OpMinAreaRect.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <util/TensorDataUtils.hpp>

#include <fstream>
#include <iostream>
#include <random>

namespace gt   = ::testing;
namespace test = nvcv::test;
void formatPoints(std::vector<std::pair<float, float>> points, std::vector<std::pair<float, float>> &format_points);
bool isNearOpenCvResults(std::vector<float> opencvRes, std::vector<float> cvcudaRes);

void formatPoints(std::vector<std::pair<float, float>> points, std::vector<std::pair<float, float>> &format_points)
{
    std::sort(points.begin(), points.end(),
              [](std::pair<float, float> &a, const std::pair<float, float> &b) { return a.first < b.first; });

    if (points[0].second <= points[1].second)
    {
        format_points[0] = points[0];
        format_points[3] = points[1];
    }
    else
    {
        format_points[0] = points[1];
        format_points[3] = points[0];
    }

    if (points[2].second <= points[3].second)
    {
        format_points[1] = points[2];
        format_points[2] = points[3];
    }
    else
    {
        format_points[1] = points[3];
        format_points[2] = points[2];
    }

    return;
}

bool isNearOpenCvResults(std::vector<float> opencvRes, std::vector<float> cvcudaRes)
{
    std::vector<std::pair<float, float>> goldVec{
        std::make_pair(opencvRes[0], opencvRes[1]), std::make_pair(opencvRes[2], opencvRes[3]),
        std::make_pair(opencvRes[4], opencvRes[5]), std::make_pair(opencvRes[6], opencvRes[7])};
    std::vector<std::pair<float, float>> predVec{
        std::pair<float, float>{cvcudaRes[0], cvcudaRes[1]},
         std::pair<float, float>{cvcudaRes[2], cvcudaRes[3]},
        std::pair<float, float>{cvcudaRes[4], cvcudaRes[5]},
         std::pair<float, float>{cvcudaRes[6], cvcudaRes[7]}
    };
    std::vector<std::pair<float, float>> goldVec_format(4, std::pair<float, float>{0, 0});
    std::vector<std::pair<float, float>> predVec_format(4, std::pair<float, float>{0, 0});
    formatPoints(goldVec, goldVec_format);
    formatPoints(predVec, predVec_format);

    for (size_t i = 0; i < predVec_format.size(); i++)
    {
        if (std::abs(goldVec_format[i].first - predVec_format[i].first) > 5.0
            || std::abs(goldVec_format[i].second - predVec_format[i].second) > 5.0)
        {
            return false;
        }
    }
    return true;
}

TEST(OpMinAreaRect, MinAreaRect_sanity)
{
    int batchsize = 3;

    std::vector<std::vector<short>> contourPointsData;

    contourPointsData.push_back(
        {845, 600, 845, 601, 847, 603, 859, 603, 860, 604, 865, 604, 866, 603, 867, 603, 868, 602, 868, 601, 867, 600});
    contourPointsData.push_back({965,  489, 964,  490, 963,  490, 962,  491, 962,  494, 963,  495,
                                 963,  499, 964,  500, 964,  501, 966,  503, 1011, 503, 1012, 504,
                                 1013, 503, 1027, 503, 1027, 502, 1028, 501, 1028, 490, 1027, 489});
    contourPointsData.push_back({1050, 198, 1049, 199, 1040, 199, 1040, 210, 1041, 211, 1040, 212, 1040, 214, 1045, 214,
                                 1046, 213, 1049, 213, 1050, 212, 1051, 212, 1052, 211, 1053, 211, 1054, 210, 1055, 210,
                                 1056, 209, 1058, 209, 1059, 208, 1059, 200, 1058, 200, 1057, 199, 1051, 199});
    //
    std::vector<std::vector<float>> openCV_minAreaRect_results;
    openCV_minAreaRect_results.push_back({868.0, 604.0, 845.0, 604.0, 845.0, 600.0, 868.0, 600.0});
    openCV_minAreaRect_results.push_back({962.0, 504.0, 962.0, 489.0, 1028.0, 489.0, 1028.0, 504.0});
    openCV_minAreaRect_results.push_back({1040.0, 214.0, 1040.0, 198.0, 1059.0, 198.0, 1059.0, 214.0});

    // point number in each contour
    nvcv::Tensor inPointNumInContour{
        nvcv::TensorShape{{1, batchsize}, nvcv::TENSOR_NW},
        nvcv::TYPE_S32
    };
    auto inPointNumInContourAccess    = nvcv::TensorDataAccessStrided::Create(inPointNumInContour.exportData());
    auto numPointNumInContourElements = inPointNumInContourAccess->sampleStride() / sizeof(int);
    std::vector<int> inPointNumInContourValues(numPointNumInContourElements, 0);

    for (int i = 0; i < batchsize; i++)
    {
        inPointNumInContourValues[i] = contourPointsData[i].size() / 2;
    }
    int maxPointsNumInCountour = *std::max_element(inPointNumInContourValues.begin(), inPointNumInContourValues.end());

    // inTensor
    auto tshapeIn = nvcv::TensorShape{
        {batchsize, maxPointsNumInCountour, 2},
        nvcv::TENSOR_NWC
    };
    nvcv::DataType dtypeIn = nvcv::TYPE_S16;
    nvcv::Tensor   inContours{tshapeIn, dtypeIn};
    auto           inContursAccess     = nvcv::TensorDataAccessStrided::Create(inContours.exportData());
    auto           numContoursElements = inContursAccess->sampleStride() / (2 * dtypeIn.strideBytes());

    // outTensor
    // 8 is the tl tr bl br cooridinates
    auto tshapeOut = nvcv::TensorShape{
        {batchsize, 8},
        nvcv::TENSOR_NW
    };
    auto                            dtypeOut = nvcv::TYPE_F32;
    nvcv::Tensor                    outMinAreaRect{tshapeOut, dtypeOut};
    std::vector<std::vector<float>> testVec(batchsize, std::vector<float>(8, 0));
    auto                            outAccess = nvcv::TensorDataAccessStrided::Create(outMinAreaRect.exportData());

    for (int i = 0; i < batchsize; i++)
    {
        contourPointsData[i].resize(numContoursElements * 2);
        nvcv::util::SetTensorFromVector<short>(inContours.exportData(), contourPointsData[i], i);
    }
    nvcv::util::SetTensorFromVector<int>(inPointNumInContour.exportData(), inPointNumInContourValues, -1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinAreaRect minAreaRectOp(batchsize);
    EXPECT_NO_THROW(minAreaRectOp(stream, inContours, outMinAreaRect, inPointNumInContour, batchsize));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy output back to host
    for (size_t i = 0; i < testVec.size(); i++)
    {
        nvcv::util::GetVectorFromTensor<float>(outMinAreaRect.exportData(), i, testVec[i]);
        ASSERT_PRED2(isNearOpenCvResults, openCV_minAreaRect_results[i], testVec[i]);
    }
}
