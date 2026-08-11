/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Version.h>

#include <string>

TEST(VersionTests, version_numeric)
{
    EXPECT_EQ(NVCV_VERSION, nvcvGetVersion());
}

TEST(VersionTests, get_version_components)
{
    EXPECT_EQ(NVCV_VERSION_MAJOR, nvcvGetVersion() / 1000000);
    EXPECT_EQ(NVCV_VERSION_MINOR, nvcvGetVersion() / 10000 % 100);
    EXPECT_EQ(NVCV_VERSION_PATCH, nvcvGetVersion() / 100 % 100);
    EXPECT_EQ(NVCV_VERSION_TWEAK, nvcvGetVersion() % 100);
}

// macro to stringify a macro-expanded expression
#define STR(a)        STR_HELPER(a)
#define STR_HELPER(a) #a

TEST(VersionTests, get_version_string)
{
    std::string ver = std::string(STR(NVCV_VERSION_MAJOR)) + "." + STR(NVCV_VERSION_MINOR) + "."
                    + STR(NVCV_VERSION_PATCH)
#if NVCV_VERSION_TWEAK
                    + "." + STR(NVCV_VERSION_TWEAK)
#endif
        ;
    if (!std::string(NVCV_VERSION_SUFFIX).empty())
        ver += std::string("-") + NVCV_VERSION_SUFFIX;

    EXPECT_STREQ(ver.c_str(), NVCV_VERSION_STRING);
}

TEST(VersionTests, make_version4_macro)
{
    EXPECT_EQ(1020304, NVCV_MAKE_VERSION4(1, 2, 3, 4));
}

TEST(VersionTests, make_version3_macro)
{
    EXPECT_EQ(1020300, NVCV_MAKE_VERSION3(1, 2, 3));
}

TEST(VersionTests, make_version2_macro)
{
    EXPECT_EQ(1020000, NVCV_MAKE_VERSION2(1, 2));
}

TEST(VersionTests, make_version1_macro)
{
    EXPECT_EQ(1000000, NVCV_MAKE_VERSION1(1));
}

TEST(VersionTests, make_version_macro)
{
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 3, 4), NVCV_MAKE_VERSION(1, 2, 3, 4));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 3, 0), NVCV_MAKE_VERSION(1, 2, 3));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 0, 0), NVCV_MAKE_VERSION(1, 2));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 0, 0, 0), NVCV_MAKE_VERSION(1));
}

TEST(VersionTests, api_version_macro)
{
    EXPECT_EQ(NVCV_MAKE_VERSION(NVCV_VERSION_MAJOR, NVCV_VERSION_MINOR), NVCV_VERSION_API);
}
