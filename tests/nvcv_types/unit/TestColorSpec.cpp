/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcv/src/priv/ColorSpec.hpp"

#include <nvcv/ColorSpec.hpp>

TEST(ColorSpecTests, whitePoint)
{
    EXPECT_EQ(nvcv::priv::ColorSpec{NVCV_COLOR_SPEC_BT601}.whitePoint(), NVCV_WHITE_POINT_D65);
}

TEST(ColorSpecTests, GetName)
{
    EXPECT_STREQ("NVCV_COLOR_SPACE_BT601", nvcv::priv::GetName(NVCV_COLOR_SPACE_BT601));
    EXPECT_STREQ("NVCV_COLOR_SPACE_BT709", nvcv::priv::GetName(NVCV_COLOR_SPACE_BT709));
    EXPECT_STREQ("NVCV_COLOR_SPACE_BT2020", nvcv::priv::GetName(NVCV_COLOR_SPACE_BT2020));
    EXPECT_STREQ("NVCV_COLOR_SPACE_DCIP3", nvcv::priv::GetName(NVCV_COLOR_SPACE_DCIP3));
    EXPECT_STREQ("NVCVColorSpace(-1)", nvcv::priv::GetName(static_cast<NVCVColorSpace>(-1)));
}

TEST(ColorSpecTests, GetColorSpecName)
{
    EXPECT_STREQ("NVCV_COLOR_SPEC_BT601", nvcv::priv::GetName(NVCV_COLOR_SPEC_BT601));
    EXPECT_STREQ("NVCVColorSpec(SPACE_DCIP3,ENC_BT2020c,XFER_sYCC,RANGE_LIMITED,LOC_ODD,LOC_CENTER)",
                 nvcv::priv::GetName(NVCV_MAKE_COLOR_SPEC(NVCV_COLOR_SPACE_DCIP3, NVCV_YCbCr_ENC_BT2020c,
                                                          NVCV_COLOR_XFER_sYCC, NVCV_COLOR_RANGE_LIMITED,
                                                          NVCV_CHROMA_LOC_ODD, NVCV_CHROMA_LOC_CENTER)));
}

TEST(ColorSpecTests, operator_insertion_NVCVWhitePoint)
{
    auto testOperatorInsertion = [](const std::string &expectedStr, NVCVWhitePoint whitePoint)
    {
        std::ostringstream ss;
        ss << whitePoint;
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("NVCV_WHITE_POINT_D65", NVCV_WHITE_POINT_D65);
    testOperatorInsertion("NVCVWhitePoint(255)", NVCV_WHITE_POINT_FORCE8);
}

TEST(ColorSpecTests, operator_insertion_NVCVColorSpec)
{
    auto testOperatorInsertion = [](const std::string &expectedStr, NVCVColorSpec colorSpec)
    {
        std::ostringstream ss;
        ss << colorSpec;
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("NVCV_COLOR_SPEC_UNDEFINED", NVCV_COLOR_SPEC_UNDEFINED);
    testOperatorInsertion("NVCV_COLOR_SPEC_BT601", NVCV_COLOR_SPEC_BT601);
}

TEST(ColorSpecTests, StrNVCVColorSpec)
{
    EXPECT_EQ("NVCV_COLOR_SPEC_UNDEFINED", StrNVCVColorSpec(NVCV_COLOR_SPEC_UNDEFINED));
    EXPECT_EQ("NVCV_COLOR_SPEC_BT601", StrNVCVColorSpec(NVCV_COLOR_SPEC_BT601));
}

// Regression: operator<<(nvcv::ColorSpace) previously dispatched to
// nvcvColorSpecGetName with a cast through NVCVColorSpec (a wider packed
// type), producing garbage names for the four small ColorSpace values.
TEST(ColorSpecTests, operator_insertion_nvcv_ColorSpace)
{
    auto check = [](const std::string &expected, nvcv::ColorSpace cspace)
    {
        std::ostringstream ss;
        ss << cspace;
        EXPECT_EQ(expected, ss.str());
    };

    check("NVCV_COLOR_SPACE_BT601", nvcv::ColorSpace::BT601);
    check("NVCV_COLOR_SPACE_BT709", nvcv::ColorSpace::BT709);
    check("NVCV_COLOR_SPACE_BT2020", nvcv::ColorSpace::BT2020);
    check("NVCV_COLOR_SPACE_DCIP3", nvcv::ColorSpace::DCIP3);
}
