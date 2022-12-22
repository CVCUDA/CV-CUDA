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

#include <common/ValueTests.hpp>
#include <nvcv/TensorLayoutInfo.hpp>

namespace test = nvcv::test;
namespace t    = ::testing;

// TensorLayoutInfo::isBatch ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfo_IsBatch_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorLayout("Nabc"),true},
        {nvcv::TensorLayout("abc"),false},
        {nvcv::TensorLayout("aNbc"),false},
        {nvcv::TensorLayout("abcN"),false},
        {nvcv::TensorLayout(""),false},
      });

// clang-format on

TEST_P(TensorLayoutInfo_IsBatch_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const bool              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfo::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->isBatch());
}

// TensorLayoutInfo::idxSample ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfo_IdxSample_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("Nabc"),0},
        {nvcv::TensorLayout("abc"),-1},
        {nvcv::TensorLayout("aNbc"),-1},
        {nvcv::TensorLayout("abcN"),-1},
        {nvcv::TensorLayout("ABCDEFGHIJKLMOP"),-1},
        {nvcv::TensorLayout("QRSTUVWXYZ01234"),-1},
        {nvcv::TensorLayout("abcdefghijklmno"),-1},
        {nvcv::TensorLayout("pqrstuvwxyz?.!@"),-1},
        {nvcv::TensorLayout(""),-1},
      });

// clang-format on

TEST_P(TensorLayoutInfo_IdxSample_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfo::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->idxSample());
}

// TensorLayoutInfo::isImage ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfo_IsImage_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorLayout("aWbc"),true},
        {nvcv::TensorLayout("W"),true},
        {nvcv::TensorLayout("NW"),true},
        {nvcv::TensorLayout("N"),false},
        {nvcv::TensorLayout(""),false},
      });

// clang-format on

TEST_P(TensorLayoutInfo_IsImage_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const bool              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfo::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->isImage());
}

// TensorLayoutInfoImage::numSpatialDims ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_NumSpatialDims_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("aWbc"),1},
        {nvcv::TensorLayout("W"),1},
        {nvcv::TensorLayout("DW"),2},
        {nvcv::TensorLayout("DWH"),3},
        {nvcv::TensorLayout("DWxH"),3},
        {nvcv::TensorLayout("DNWx."),2},
        {nvcv::TensorLayout("ABCEFGWIJKLMNOP"),1},
        {nvcv::TensorLayout("QRSTUVXYZ01234W"),1},
        {nvcv::TensorLayout("Wabcdefghijklmn"),1},
        {nvcv::TensorLayout("opqrstuvwxyz?@W"),1},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_NumSpatialDims_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->numSpatialDims());
}

// TensorLayoutInfoImage::isRowMajor ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IsRowMajor_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorLayout("W"),true},
        {nvcv::TensorLayout("HW"),true},
        {nvcv::TensorLayout("WC"),true},
        {nvcv::TensorLayout("HWC"),true},
        {nvcv::TensorLayout(".W"),true},
        {nvcv::TensorLayout(".HW"),true},
        {nvcv::TensorLayout(".WC"),true},
        {nvcv::TensorLayout(".HWC"),true},
        {nvcv::TensorLayout("CW"),true},
        {nvcv::TensorLayout("H.WC"),true},
        {nvcv::TensorLayout("H.W"),true},

        {nvcv::TensorLayout("WH"),false},
        {nvcv::TensorLayout("WHC"),false},
        {nvcv::TensorLayout("WCH"),false},

        {nvcv::TensorLayout("W."),false},
        {nvcv::TensorLayout("WH"),false},
        {nvcv::TensorLayout("W.H"),false},
        {nvcv::TensorLayout("HW.C"),false},
        {nvcv::TensorLayout("WH.C"),false},
        {nvcv::TensorLayout("H.W.C"),false},
        {nvcv::TensorLayout("W.H.C"),false},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IsRowMajor_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const bool              &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->isRowMajor());
}

// TensorLayoutInfo::idxChannel ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IdxChannel_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("CaWc"),0},
        {nvcv::TensorLayout("WbC"),2},
        {nvcv::TensorLayout("WCbc"),1},
        {nvcv::TensorLayout("ABWDEFGHIJKLMOP"),-1},
        {nvcv::TensorLayout("QRSTUVWXYZ0123W"),-1},
        {nvcv::TensorLayout("abcdefghijklmnW"),-1},
        {nvcv::TensorLayout("opqrstuvwxyz?.W"),-1},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IdxChannel_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->idxChannel());
}

// TensorLayoutInfo::idxWidth ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IdxWidth_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("Wabc"),0},
        {nvcv::TensorLayout("abW"),2},
        {nvcv::TensorLayout("aWbc"),1},
        {nvcv::TensorLayout("ABCDEFGHIJKLMNW"),14},
        {nvcv::TensorLayout("OPQRSTUVXYZ012W"),14},
        {nvcv::TensorLayout("abcdefghijklmnW"),14},
        {nvcv::TensorLayout("opqrstuvwxyz?!W"),14},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IdxWidth_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->idxWidth());
}

// TensorLayoutInfo::idxHeight ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IdxHeight_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("HaWc"),0},
        {nvcv::TensorLayout("WbH"),2},
        {nvcv::TensorLayout("aHbW"),1},
        {nvcv::TensorLayout("ABCDEFGIJKLMNOW"),-1},
        {nvcv::TensorLayout("PQRSTUVWXYZ0123"),-1},
        {nvcv::TensorLayout("abcdefghijklmnW"),-1},
        {nvcv::TensorLayout("opqrstuvwxyz?.W"),-1},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IdxHeight_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->idxHeight());
}

// TensorLayoutInfo::idxDepth ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IdxDepth_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",int>>
      {
        {nvcv::TensorLayout("DabW"),0},
        {nvcv::TensorLayout("aWD"),2},
        {nvcv::TensorLayout("aDWc"),1},
        {nvcv::TensorLayout("ABCEFGHIJKLMNOW"),-1},
        {nvcv::TensorLayout("PQRSTUVWXYZ0123"),-1},
        {nvcv::TensorLayout("abcdefghijklmnW"),-1},
        {nvcv::TensorLayout("opqrstuvwxyz?.W"),-1},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IdxDepth_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->idxDepth());
}

// TensorLayoutInfo::isChannelFirst ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IsChannelFirst_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorLayout("CzWc"),true},
        {nvcv::TensorLayout("NCzxW"),true},
        {nvcv::TensorLayout("N.CzWc"),false},
        {nvcv::TensorLayout("W.C"),false},
        {nvcv::TensorLayout("WC"),false},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IsChannelFirst_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->isChannelFirst());
}

// TensorLayoutInfo::isChannelLast ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutInfoImage_IsChannelLast_ExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold",bool>>
      {
        {nvcv::TensorLayout("CzxW"),false},
        {nvcv::TensorLayout("NCzWc"),false},
        {nvcv::TensorLayout("N.CzxW"),false},
        {nvcv::TensorLayout("N.WxC"),true},
        {nvcv::TensorLayout("W.C"),true},
        {nvcv::TensorLayout("WC"),true},
        {nvcv::TensorLayout("NW.C"),true},
        {nvcv::TensorLayout("HW"),true},
      });

// clang-format on

TEST_P(TensorLayoutInfoImage_IsChannelLast_ExecTests, works)
{
    const nvcv::TensorLayout input{std::get<0>(GetParam())};
    const int               &gold = std::get<1>(GetParam());

    auto info = nvcv::TensorLayoutInfoImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->isChannelLast());
}
