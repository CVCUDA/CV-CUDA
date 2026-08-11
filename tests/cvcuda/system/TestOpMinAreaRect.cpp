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

#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpMinAreaRect.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <random>

namespace test = nvcv::test;
namespace t    = ::testing;

void formatPoints(std::vector<std::pair<float, float>> points, std::vector<std::pair<float, float>> &format_points);
bool isNearOpenCvResults(std::vector<float> opencvRes, std::vector<float> cvcudaRes);

template<typename T>
void SetRectanglePoints(nvcv::Tensor &tensor, size_t contourElements, int numPoints)
{
    std::vector<T>                           contour(contourElements, 0);
    constexpr std::array<std::pair<T, T>, 4> corners{
        {{10, 20}, {110, 20}, {110, 70}, {10, 70}}
    };
    for (int i = 0; i < numPoints; ++i)
    {
        contour[2 * i]     = corners[i % corners.size()].first;
        contour[2 * i + 1] = corners[i % corners.size()].second;
    }
    nvcv::util::SetTensorFromVector<T>(tensor.exportData(), contour, 0);
}

void formatPoints(std::vector<std::pair<float, float>> points, std::vector<std::pair<float, float>> &format_points)
{
    std::ranges::sort(
        points, [](const std::pair<float, float> &a, const std::pair<float, float> &b) { return a.first < b.first; });

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
        inPointNumInContourValues[i] = static_cast<int>(contourPointsData[i].size() / 2);
    }
    int maxPointsNumInCountour = *std::ranges::max_element(inPointNumInContourValues);

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
        nvcv::util::GetVectorFromTensor<float>(outMinAreaRect.exportData(), static_cast<int>(i), testVec[i]);
        ASSERT_PRED2(isNearOpenCvResults, openCV_minAreaRect_results[i], testVec[i]);
    }
}

TEST(OpMinAreaRect, MinAreaRect_multiple_contours_odd_stride)
{
    int batchsize = 5;

    std::vector<std::vector<short>> contourPointsData;

    contourPointsData.push_back({0, 0, 200, 0, 200, 100, 0, 100});
    contourPointsData.push_back({100, 0, 0, -100, -100, 0, 0, 100});
    contourPointsData.push_back({0, 0, 150, 75, 145, 85, -5, 10});
    contourPointsData.push_back({0, 0, 100, 0, 50, 100});
    contourPointsData.push_back({10, 10, 90, 20, 80, 80, 20, 70, 15, 40});

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
        inPointNumInContourValues[i] = static_cast<int>(contourPointsData[i].size() / 2);
    }
    int maxPointsNumInCountour = *std::ranges::max_element(inPointNumInContourValues);

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
    auto tshapeOut = nvcv::TensorShape{
        {batchsize, 8},
        nvcv::TENSOR_NW
    };
    auto                            dtypeOut = nvcv::TYPE_F32;
    nvcv::Tensor                    outMinAreaRect{tshapeOut, dtypeOut};
    std::vector<std::vector<float>> testVec(batchsize, std::vector<float>(8, 0));

    for (int i = 0; i < batchsize; i++)
    {
        contourPointsData[i].resize(numContoursElements * 2, 0);
        nvcv::util::SetTensorFromVector<short>(inContours.exportData(), contourPointsData[i], i);
    }
    nvcv::util::SetTensorFromVector<int>(inPointNumInContour.exportData(), inPointNumInContourValues, -1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinAreaRect minAreaRectOp(batchsize);
    EXPECT_NO_THROW(minAreaRectOp(stream, inContours, outMinAreaRect, inPointNumInContour, batchsize));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

NVCV_TEST_SUITE_P(OpMinAreaRectCorrectness, test::ValueList<nvcv::DataType, int>{
                                                {nvcv::TYPE_S16, 1024},
                                                {nvcv::TYPE_U16,  128},
                                                {nvcv::TYPE_U16,  512},
                                                {nvcv::TYPE_S32,  128},
                                                {nvcv::TYPE_S32,  512},
});

TEST_P(OpMinAreaRectCorrectness, tensor_correct_output)
{
    nvcv::DataType dtype       = GetParamValue<0>();
    int            numOfPoints = GetParamValue<1>();

    nvcv::Tensor inPointNumInContour{
        nvcv::TensorShape{{1, 1}, nvcv::TENSOR_NW},
        nvcv::TYPE_S32
    };
    auto             pointCountAccess = nvcv::TensorDataAccessStrided::Create(inPointNumInContour.exportData());
    std::vector<int> pointCounts(pointCountAccess->sampleStride() / sizeof(int), 0);
    pointCounts[0] = numOfPoints;
    nvcv::util::SetTensorFromVector<int>(inPointNumInContour.exportData(), pointCounts, -1);

    nvcv::Tensor inContours{
        nvcv::TensorShape{{1, numOfPoints, 2}, nvcv::TENSOR_NWC},
        dtype
    };
    auto contourAccess   = nvcv::TensorDataAccessStrided::Create(inContours.exportData());
    auto contourElements = contourAccess->sampleStride() / dtype.strideBytes();

    if (dtype == nvcv::TYPE_S16)
    {
        SetRectanglePoints<int16_t>(inContours, contourElements, numOfPoints);
    }
    else if (dtype == nvcv::TYPE_U16)
    {
        SetRectanglePoints<uint16_t>(inContours, contourElements, numOfPoints);
    }
    else
    {
        ASSERT_EQ(dtype, nvcv::TYPE_S32);
        SetRectanglePoints<int32_t>(inContours, contourElements, numOfPoints);
    }

    nvcv::Tensor outMinAreaRect{
        nvcv::TensorShape{{1, 8}, nvcv::TENSOR_NW},
        nvcv::TYPE_F32
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::MinAreaRect minAreaRectOp(1);
    EXPECT_NO_THROW(minAreaRectOp(stream, inContours, outMinAreaRect, inPointNumInContour, 1));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<float> output(8);
    nvcv::util::GetVectorFromTensor<float>(outMinAreaRect.exportData(), 0, output);

    std::vector<std::pair<float, float>> expected{
        { 10, 20},
        {110, 20},
        {110, 70},
        { 10, 70}
    };
    std::vector<std::pair<float, float>> actual{
        {output[0], output[1]},
        {output[2], output[3]},
        {output[4], output[5]},
        {output[6], output[7]}
    };
    std::vector<std::pair<float, float>> expectedFormatted(4);
    std::vector<std::pair<float, float>> actualFormatted(4);
    formatPoints(expected, expectedFormatted);
    formatPoints(actual, actualFormatted);

    for (size_t i = 0; i < expectedFormatted.size(); ++i)
    {
        EXPECT_EQ(expectedFormatted[i].first, actualFormatted[i].first);
        EXPECT_EQ(expectedFormatted[i].second, actualFormatted[i].second);
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpMinAreaRect_Negative, test::ValueList<int, nvcv::TensorLayout, nvcv::TensorLayout, nvcv::TensorLayout, nvcv::DataType, nvcv::DataType, nvcv::DataType>
{
    // batchsize, inLayout, numPointsInContourLayout, outLayout, inDataType, numPointsInContourDataType, outDataType
    {        10, nvcv::TENSOR_NWC, nvcv::TENSOR_NW, nvcv::TENSOR_NW, nvcv::TYPE_S16, nvcv::TYPE_S32, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NCW, nvcv::TENSOR_NW, nvcv::TENSOR_NW, nvcv::TYPE_S16, nvcv::TYPE_S32, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NWC, nvcv::TENSOR_CW, nvcv::TENSOR_NW, nvcv::TYPE_S16, nvcv::TYPE_S32, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NWC, nvcv::TENSOR_NW, nvcv::TENSOR_CW, nvcv::TYPE_S16, nvcv::TYPE_S32, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NWC, nvcv::TENSOR_NW, nvcv::TENSOR_NW, nvcv::TYPE_F16, nvcv::TYPE_S32, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NWC, nvcv::TENSOR_NW, nvcv::TENSOR_NW, nvcv::TYPE_S16, nvcv::TYPE_F16, nvcv::TYPE_F32},
    {         2, nvcv::TENSOR_NWC, nvcv::TENSOR_NW, nvcv::TENSOR_NW, nvcv::TYPE_S16, nvcv::TYPE_S32, nvcv::TYPE_F16},
});

// clang-format on

TEST_P(OpMinAreaRect_Negative, tensor_correct_output)
{
    int                batchsize                  = GetParamValue<0>();
    nvcv::TensorLayout inLayout                   = GetParamValue<1>();
    nvcv::TensorLayout numPointsInContourLayout   = GetParamValue<2>();
    nvcv::TensorLayout outLayout                  = GetParamValue<3>();
    nvcv::DataType     inDataType                 = GetParamValue<4>();
    nvcv::DataType     numPointsInContourDataType = GetParamValue<5>();
    nvcv::DataType     outDataType                = GetParamValue<6>();

    const int maxPointsNumInCountour = 50;
    const int maxContourNum          = 5;

    nvcv::Tensor inPointNumInContour{
        nvcv::TensorShape{{1, batchsize}, numPointsInContourLayout},
        numPointsInContourDataType
    };

    // inTensor
    auto tshapeIn = nvcv::TensorShape{
        {batchsize, maxPointsNumInCountour, 2},
        inLayout
    };
    nvcv::Tensor inContours{tshapeIn, inDataType};

    // outTensor
    // 8 is the tl tr bl br cooridinates
    auto tshapeOut = nvcv::TensorShape{
        {batchsize, 8},
        outLayout
    };
    nvcv::Tensor outMinAreaRect{tshapeOut, outDataType};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinAreaRect minAreaRectOp(maxContourNum);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&minAreaRectOp, &stream, &inContours, &outMinAreaRect, &inPointNumInContour, &batchsize]
                          { minAreaRectOp(stream, inContours, outMinAreaRect, inPointNumInContour, batchsize); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMinAreaRect, invalid_create)
{
    EXPECT_EQ(cvcudaMinAreaRectCreate(nullptr, 1), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpMinAreaRect, numPointsInContour_exceeds_tensor_width)
{
    // Regression test for GPU heap overread: numPointsInContour > max_pts must not
    // cause out-of-bounds GPU reads in calculateRotateArea.
    int batchsize    = 1;
    int max_pts      = 4;
    int reported_pts = 100; // intentionally larger than max_pts

    auto tshapeIn = nvcv::TensorShape{
        {batchsize, max_pts, 2},
        nvcv::TENSOR_NWC
    };
    nvcv::DataType dtypeIn = nvcv::TYPE_S16;
    nvcv::Tensor   inContours{tshapeIn, dtypeIn};
    auto           inContoursAccess    = nvcv::TensorDataAccessStrided::Create(inContours.exportData());
    auto           numContoursElements = inContoursAccess->sampleStride() / (2 * dtypeIn.strideBytes());

    // Use a simple axis-aligned rectangle.
    std::vector<short> pts = {0, 0, 100, 0, 100, 50, 0, 50};
    pts.resize(numContoursElements * 2, 0);
    nvcv::util::SetTensorFromVector<short>(inContours.exportData(), pts, 0);

    nvcv::Tensor inPointNumInContour{
        nvcv::TensorShape{{1, batchsize}, nvcv::TENSOR_NW},
        nvcv::TYPE_S32
    };
    auto inPointNumInContourAccess    = nvcv::TensorDataAccessStrided::Create(inPointNumInContour.exportData());
    auto numPointNumInContourElements = inPointNumInContourAccess->sampleStride() / sizeof(int);
    std::vector<int> numPtsVec(numPointNumInContourElements, 0);
    numPtsVec[0] = reported_pts;
    nvcv::util::SetTensorFromVector<int>(inPointNumInContour.exportData(), numPtsVec, -1);

    auto tshapeOut = nvcv::TensorShape{
        {batchsize, 8},
        nvcv::TENSOR_NW
    };
    nvcv::Tensor outMinAreaRect{tshapeOut, nvcv::TYPE_F32};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cvcuda::MinAreaRect minAreaRectOp(batchsize);
    EXPECT_NO_THROW(minAreaRectOp(stream, inContours, outMinAreaRect, inPointNumInContour, batchsize));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
