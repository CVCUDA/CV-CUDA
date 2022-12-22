/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/TensorShapeInfo.hpp>

namespace test = nvcv::test;
namespace t    = ::testing;

// TensorShapeInfo::numSamples ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfo_NumBatch_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorShape{{4,2,13,4,5},"HNWCF"},1},
        {nvcv::TensorShape{{4,2,13,4,5},"PWxCN"},1},
        {nvcv::TensorShape{{4,2,13,4,5},"NWxCF"},4},
        {nvcv::TensorShape{{4,2,13},"HWC"},1},
        {nvcv::TensorShape{{},""},0},
      });

// clang-format on

TEST_P(TensorShapeInfo_NumBatch_ExecTests, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const int              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfo::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numSamples());
}

// TensorShapeInfoImage::numChannels ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImage_NumChannels_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorShape{{4,2,13,4,5},"HNWCF"},4},
        {nvcv::TensorShape{{4,2,13,4,5},"PWxNC"},5},
        {nvcv::TensorShape{{4,2,13,4,5},"NWxCF"},4},
        {nvcv::TensorShape{{4,2,13,4,5,23},"NCWxpF"},2},
        {nvcv::TensorShape{{4,2,13},"HWC"},13},
        {nvcv::TensorShape{{4,13},"HW"},1},
      });

// clang-format on

TEST_P(TensorShapeInfoImage_NumChannels_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const int              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numChannels());
}

// TensorShapeInfoImage::width ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImage_NumCols_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorShape{{4,2,13,4,5},"HNWCF"},13},
        {nvcv::TensorShape{{4,2,13,4,5},"PWxNC"},2},
        {nvcv::TensorShape{{4,2,13,4,5},"WNxCF"},4},
        {nvcv::TensorShape{{4,2,13},"HWC"},2},
        {nvcv::TensorShape{{4,13},"HW"},13},
      });

// clang-format on

TEST_P(TensorShapeInfoImage_NumCols_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const int              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numCols());
}

// TensorShapeInfoImage::height ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImage_NumRows_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorShape{{4,2,13,4,5},"HNWCF"},4},
        {nvcv::TensorShape{{4,2,13,4,5},"PWHNC"},13},
        {nvcv::TensorShape{{4,2,13,4,5},"WNxCH"},5},
        {nvcv::TensorShape{{3,13},"HW"},3},
        {nvcv::TensorShape{{3,13,2},"HWC"},3},
        {nvcv::TensorShape{{3,13,2},"WHC"},13},
        {nvcv::TensorShape{{3},"W"},1},
        {nvcv::TensorShape{{3,13,8,5},"sWm2"},1},
      });

// clang-format on

TEST_P(TensorShapeInfoImage_NumRows_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const int              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numRows());
}

// TensorShapeInfoImage::size2d ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImage_Size_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",nvcv::Size2D>>
      {
        {nvcv::TensorShape{{4,2,13,4,5},"HNWCF"},nvcv::Size2D{13,4}},
        {nvcv::TensorShape{{4,2,13,4,5},"PWHNC"},nvcv::Size2D{2,13}},
        {nvcv::TensorShape{{4,2,13,4,5},"WNxCH"},nvcv::Size2D{4,5}},
        {nvcv::TensorShape{{3,13},"HW"},nvcv::Size2D{13,3}},
        {nvcv::TensorShape{{3,7,2},"HWC"},nvcv::Size2D{7,3}},
        {nvcv::TensorShape{{3,13,2},"WHC"},nvcv::Size2D{3,13}},
        {nvcv::TensorShape{{3},"W"},nvcv::Size2D{3,1}}
      });

// clang-format on

TEST_P(TensorShapeInfoImage_Size_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const nvcv::Size2D     &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->size());
}

// TensorShapeInfoImagePlanar::IsCompatible ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImagePlanar_IsCompatible_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorShape{{5},"W"},true},
        {nvcv::TensorShape{{4,5},"HW"},true},
        {nvcv::TensorShape{{4,2,13},"CHW"},true},

        {nvcv::TensorShape{{4,5},"WC"},true},
        {nvcv::TensorShape{{4,5},"CW"},true},
        {nvcv::TensorShape{{4,2,13},"HWC"},true},

        {nvcv::TensorShape{{5,8},"NW"},true},
        {nvcv::TensorShape{{4,5,2},"NHW"},true},
        {nvcv::TensorShape{{4,2,13,4},"NCHW"},true},

        {nvcv::TensorShape{{4,5,8},"NWC"},true},
        {nvcv::TensorShape{{4,5,4},"NCW"},true},
        {nvcv::TensorShape{{4,2,13,4},"NHWC"},true},

        {nvcv::TensorShape{{5,4},"Wx"},false},
        {nvcv::TensorShape{{4,5},"WH"},false},
        {nvcv::TensorShape{{4,2,13},"CWH"},false},
        {nvcv::TensorShape{{4,2,13},"WCH"},false},
        {nvcv::TensorShape{{4,2,13},"WHC"},false},

        {nvcv::TensorShape{{4,5,2},"WxC"},false},
        {nvcv::TensorShape{{4,5,5},"CxW"},false},
        {nvcv::TensorShape{{4,2,13,2},"HWxC"},false},
        {nvcv::TensorShape{{4,2,13,4},"HxWC"},false},

        {nvcv::TensorShape{{5,8},"WN"},false},
        {nvcv::TensorShape{{4,5,2},"NWH"},false},
        {nvcv::TensorShape{{4,2,13,4},"NCWH"},false},

        {nvcv::TensorShape{{4,5,2},"WNC"},false},
        {nvcv::TensorShape{{4,5,6},"CNW"},false},
        {nvcv::TensorShape{{4,2,13,4},"HNWC"},false},
        {nvcv::TensorShape{{4,2,13,4},"HWNC"},false},
        {nvcv::TensorShape{{4,2,13,4},"HWCN"},false},
      });

// clang-format on

TEST_P(TensorShapeInfoImagePlanar_IsCompatible_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const bool             &gold = std::get<1>(GetParam());

    EXPECT_EQ(gold, nvcv::TensorShapeInfoImagePlanar::IsCompatible(input));
}

// TensorShapeInfoImagePlanar::numPlanes ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorShapeInfoImagePlanar_NumPlanes_ExecTest,
      test::ValueList<test::Param<"layout",nvcv::TensorShape>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorShape{{5},"W"},1},
        {nvcv::TensorShape{{4,5},"HW"},1},
        {nvcv::TensorShape{{4,2,13},"CHW"},4},

        {nvcv::TensorShape{{4,5},"WC"},1},
        {nvcv::TensorShape{{4,5},"CW"},4},
        {nvcv::TensorShape{{4,2,13},"HWC"},1},

        {nvcv::TensorShape{{5,8},"NW"},1},
        {nvcv::TensorShape{{4,5,2},"NHW"},1},
        {nvcv::TensorShape{{4,2,13,4},"NCHW"},2},

        {nvcv::TensorShape{{4,5,8},"NWC"},1},
        {nvcv::TensorShape{{4,5,4},"NCW"},5},
        {nvcv::TensorShape{{4,2,13,4},"NHWC"},1},
      });

// clang-format on

TEST_P(TensorShapeInfoImagePlanar_NumPlanes_ExecTest, works)
{
    const nvcv::TensorShape input{std::get<0>(GetParam())};
    const int              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorShapeInfoImagePlanar::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numPlanes());
}
