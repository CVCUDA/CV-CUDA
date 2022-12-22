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
#include <util/Assert.h>
#include <util/String.hpp>

namespace util = nvcv::util;
namespace test = nvcv::test;
namespace t    = ::testing;

NVCV_TEST_SUITE_P(ReplaceAllInlineTests,
                  // input, bufSize, what, replace, gold
                  test::ValueList<const char *, int, const char *, const char *, const char *>{
                      {      "test", 16, "ROD",    "lima",         "test"}, // pattern not found

                      {   "RODtest", 16, "ROD",        "",         "test"}, // remove pattern once begin
                      {   "teRODst", 16, "ROD",        "",         "test"}, // remove pattern once middle
                      {   "testROD", 16, "ROD",        "",         "test"}, // remove pattern once end
                      {"tRODeRODst", 16, "ROD",        "",         "test"}, // remove pattern twice

                      {   "RODtest", 16, "ROD",      "AB",       "ABtest"}, // replace pattern with smaller once begin
                      {   "teRODst", 16, "ROD",      "AB",       "teABst"}, // replace pattern with smaller once middle
                      {   "testROD", 16, "ROD",      "AB",       "testAB"}, // replace pattern with smaller once end
                      {"tRODeRODst", 16, "ROD",      "AB",     "tABeABst"}, // replace pattern with smaller twice

                      {   "RODtest", 16, "ROD",    "ABCD",     "ABCDtest"}, // replace pattern with larger once begin
                      {   "teRODst", 16, "ROD",    "ABCD",     "teABCDst"}, // replace pattern with larger once middle
                      {   "testROD", 16, "ROD",    "ABCD",     "testABCD"}, // replace pattern with larger once end
                      {"tRODeRODst", 16, "ROD",    "ABCD", "tABCDeABCDst"}, // replace pattern with larger twice

                      {   "tRODest",  7, "ROD",    "ABCD",      "tABCDes"}, // buffer size too small for replacement

                      {   "tRODest", 32, "ROD", "RODOLFO",  "tRODOLFOest"}, // 'replacement' contains 'what'
});

TEST_P(ReplaceAllInlineTests, test)
{
    const char *input   = GetParamValue<0>();
    const int   bufSize = GetParamValue<1>();
    const char *what    = GetParamValue<2>();
    const char *replace = GetParamValue<3>();
    const char *gold    = GetParamValue<4>();

    char buffer[256];
    // +1 for sentinel
    NVCV_ASSERT(sizeof(buffer) + 1 >= strlen(input));
    NVCV_ASSERT(sizeof(buffer) + 1 >= strlen(gold));

    strncpy(buffer, input, sizeof(buffer));
    char *sentinel = buffer + std::max(strlen(input), strlen(gold)) + 1;
    *sentinel      = '\xFF';

    ASSERT_NO_THROW(util::ReplaceAllInline(buffer, bufSize, what, replace));

    EXPECT_STREQ(gold, buffer);
    EXPECT_EQ('\xFF', *sentinel) << "buffer overrun";
}

TEST(BufferOStreamTests, is_zero_terminated_on_dtor)
{
    char buf[] = "rod";

    {
        util::BufferOStream str(buf, sizeof(buf));
    }
    EXPECT_EQ('\0', buf[0]);
}

TEST(BufferOStreamTests, is_flushed_on_dtor)
{
    char buf[] = "rod";

    {
        util::BufferOStream str(buf, sizeof(buf));
        str << 'x';
    }
    EXPECT_STREQ("x", buf);
}

TEST(BufferOStreamTests, data_is_written)
{
    char buf[] = "rod";

    util::BufferOStream str(buf, sizeof(buf));
    str << "123" << '\0' << std::flush;
    EXPECT_STREQ("123", buf);
}

TEST(BufferOStreamTests, overflow)
{
    char buf[] = "rodlima";

    util::BufferOStream str(buf, sizeof(buf) - 1);
    str << "12345678\0" << std::flush;
    EXPECT_FALSE(str.good());
    EXPECT_TRUE(str.fail());
    EXPECT_STREQ("1234567", buf);
}
