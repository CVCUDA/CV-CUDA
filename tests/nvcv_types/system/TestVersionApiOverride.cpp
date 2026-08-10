/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(VersionTests, api_version_macro_override)
{
    EXPECT_EQ(NVCV_MAKE_VERSION(0, 16), NVCV_VERSION_API);
}

TEST(VersionTests, api_version_at_least)
{
    EXPECT_TRUE(NVCV_VERSION_API_AT_LEAST(0, 16));
    EXPECT_TRUE(NVCV_VERSION_API_AT_LEAST(0, 15));

    EXPECT_FALSE(NVCV_VERSION_API_AT_LEAST(0, 17));
    EXPECT_FALSE(NVCV_VERSION_API_AT_LEAST(1, 0));
}

TEST(VersionTests, api_version_at_most)
{
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(0, 16));
    EXPECT_FALSE(NVCV_VERSION_API_AT_MOST(0, 15));
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(0, 17));
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(1, 0));
}

TEST(VersionTests, api_version_in_range)
{
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 16, 0, 16));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 15, 0, 16));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 15, 0, 17));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 15, 1, 0));

    EXPECT_FALSE(NVCV_VERSION_API_IN_RANGE(0, 14, 0, 15));
    EXPECT_FALSE(NVCV_VERSION_API_IN_RANGE(0, 17, 1, 0));
}

TEST(VersionTests, api_version_is)
{
    EXPECT_TRUE(NVCV_VERSION_API_IS(0, 16));
    EXPECT_FALSE(NVCV_VERSION_API_IS(0, 17));
    EXPECT_FALSE(NVCV_VERSION_API_IS(0, 15));
}
