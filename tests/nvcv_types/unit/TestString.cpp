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

#include <common/ValueTests.hpp>
#include <nvcv/util/Assert.h>
#include <nvcv/util/String.hpp>

#include <algorithm>
#include <array>
#include <string>

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

                      {   "RODtest", 16, "ROD",     "ROD",      "RODtest"}, // replace pattern with smaller once begin
                      {   "RODtest", 16, "ROD",      "AB",       "ABtest"}, // replace pattern with smaller once begin
                      {   "teRODst", 16, "ROD",      "AB",       "teABst"}, // replace pattern with smaller once middle
                      {   "testROD", 16, "ROD",      "AB",       "testAB"}, // replace pattern with smaller once end
                      {"tRODeRODst", 16, "ROD",      "AB",     "tABeABst"}, // replace pattern with smaller twice

                      {   "RODtest", 16, "ROD",    "ABCD",     "ABCDtest"}, // replace pattern with larger once begin
                      {   "teRODst", 16, "ROD",    "ABCD",     "teABCDst"}, // replace pattern with larger once middle
                      {   "testROD", 16, "ROD",    "ABCD",     "testABCD"}, // replace pattern with larger once end
                      {"tRODeRODst", 16, "ROD",    "ABCD", "tABCDeABCDst"}, // replace pattern with larger twice

                      {   "testROD",  8, "ROD",    "ABCD",      "testABC"}, // buffer size too small for replacement
                      {   "tRODest",  8, "ROD",    "ABCD",      "tABCDes"}, // buffer size too small for replacement
                      {   "RODtest",  8, "ROD",    "ABCD",      "ABCDtes"}, // buffer size too small for replacement
                      {   "RODtesT",  8,   "T",    "ABCD",      "RODtesA"}, // buffer size too small for replacement
                      {   "RODtesT",  0,   "T",    "ABCD",      "RODtesT"}, // buffer size too small for replacement
                      {   "RODTest",  8,   "T",    "ABCD",      "RODABCD"}, // buffer size too small for replacement
                      {   "ROTests",  8,   "T",    "ABCD",      "ROABCDe"}, // buffer size too small for replacement
                      {   "RODTest",  8,  "es",    "ABCD",      "RODTABC"}, // buffer size too small for replacement
                      {"RODtestROD", 11, "ROD",    "ABCD",   "ABCDtestRO"}, // buffer bounds
                      {"RODtestROV", 11,   "V",    "ABCD",   "RODtestROA"}, // buffer bounds
                      {   "tRODest", 32, "ROD", "RODOLFO",  "tRODOLFOest"}, // 'replacement' contains 'what'
});

TEST_P(ReplaceAllInlineTests, test)
{
    const char *input   = GetParamValue<0>();
    const int   bufSize = GetParamValue<1>();
    const char *what    = GetParamValue<2>();
    const char *replace = GetParamValue<3>();
    const char *gold    = GetParamValue<4>();

    const std::string inputText{input};
    const std::string goldText{gold};

    std::array<char, 256> buffer{};
    NVCV_ASSERT(buffer.size() > std::max(inputText.size(), goldText.size()) + 1);

    std::ranges::copy(inputText, buffer.begin());
    buffer[inputText.size()] = '\0';
    char *sentinel           = buffer.data() + std::max(inputText.size(), goldText.size()) + 1;
    *sentinel                = '\xFF';

    ASSERT_NO_THROW(util::ReplaceAllInline(buffer.data(), bufSize, what, replace));
    EXPECT_STREQ(gold, buffer.data());
    EXPECT_EQ('\xFF', *sentinel) << "buffer overrun";
}

TEST(BufferOStreamTests, is_zero_terminated_on_dtor)
{
    std::array<char, 4> buf = {"rod"};

    {
        util::BufferOStream str(buf.data(), buf.size());
    }
    EXPECT_EQ('\0', buf[0]);
}

TEST(BufferOStreamTests, is_flushed_on_dtor)
{
    std::array<char, 4> buf = {"rod"};

    {
        util::BufferOStream str(buf.data(), buf.size());
        str << 'x';
    }
    EXPECT_STREQ("x", buf.data());
}

TEST(BufferOStreamTests, data_is_written)
{
    std::array<char, 4> buf = {"rod"};

    util::BufferOStream str(buf.data(), buf.size());
    str << "123" << '\0' << std::flush;
    EXPECT_STREQ("123", buf.data());
}

TEST(BufferOStreamTests, overflow)
{
    std::array<char, 8> buf = {"rodlima"};

    util::BufferOStream str(buf.data(), buf.size() - 1);
    str << "12345678\0" << std::flush;
    EXPECT_FALSE(str.good());
    EXPECT_TRUE(str.fail());
    EXPECT_STREQ("1234567", buf.data());
}

TEST(ReplaceAllInlineTests, unterminated_buffer_is_terminated)
{
    std::array<char, 4> buf = {'a', 'b', 'c', 'd'};

    util::ReplaceAllInline(buf.data(), buf.size(), "missing", "replacement");

    EXPECT_EQ((std::array<char, 4>{'a', 'b', 'c', '\0'}), buf);
}

TEST(FixedBufferStreamBufTests, invalid_reset_and_seek)
{
    std::array<char, 4>        buf{};
    util::FixedBufferStreamBuf streamBuf(nullptr, 0);

    EXPECT_EQ(std::streampos{std::streamoff{-1}}, streamBuf.pubseekpos(0, std::ios_base::out));

    streamBuf.reset(buf.data(), buf.size());
    EXPECT_EQ(std::streampos{std::streamoff{-1}}, streamBuf.pubseekpos(0, std::ios_base::in));
    EXPECT_EQ(std::streampos{std::streamoff{-1}}, streamBuf.pubseekpos(-1, std::ios_base::out));
    EXPECT_EQ(std::streampos{std::streamoff{-1}}, streamBuf.pubseekpos(buf.size(), std::ios_base::out));
}

TEST(FixedBufferStreamBufTests, eof_overflow_is_not_an_error)
{
    class TestStreamBuf : public util::FixedBufferStreamBuf
    {
    public:
        using FixedBufferStreamBuf::FixedBufferStreamBuf;
        using FixedBufferStreamBuf::overflow;
    };

    std::array<char, 1> buf{};
    TestStreamBuf       streamBuf(buf.data(), buf.size());

    EXPECT_FALSE(TestStreamBuf::traits_type::eq_int_type(streamBuf.overflow(TestStreamBuf::traits_type::eof()),
                                                         TestStreamBuf::traits_type::eof()));
}
