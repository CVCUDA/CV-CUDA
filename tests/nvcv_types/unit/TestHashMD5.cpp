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

#include <util/HashMD5.hpp>

namespace util = nvcv::util;
namespace t    = ::testing;

namespace {

// clang-format off
const auto g_matchMd5Empty = t::ElementsAre
(
    0xD4, 0x1D, 0x8C, 0xD9,
    0x8F, 0x00, 0xB2, 0x04,
    0xE9, 0x80, 0x09, 0x98,
    0xeC, 0xF8, 0x42, 0x7E
);

const auto g_matchMd5NullOptional = t::ElementsAre
(
    0x9C, 0x3D, 0x0F, 0xC4,
    0x11, 0x08, 0x87, 0xEF,
    0xDC, 0xFE, 0x04, 0x73,
    0x3B, 0x41, 0x8B, 0xF2
);

const char *const g_str1     = "cv-cuda rules";
const auto g_matchMd5String1 = t::ElementsAre
(
    0x4B, 0x04, 0xB9, 0xEE,
    0xB1, 0x59, 0x35, 0x98,
    0xEB, 0xDD, 0x8B, 0x09,
    0x4D, 0xB3, 0x78, 0x41
);

const char *const g_str2     = "cv-cuda rules!";
const auto g_matchMd5String2 = t::ElementsAre
(
    0x55, 0xAB, 0x40, 0x9E,
    0x46, 0x03, 0xC6, 0x6B,
    0xD3, 0xDD, 0xE6, 0x69,
    0x43, 0x08, 0xFA, 0x20
);

auto g_matchFPzero = t::ElementsAre
(
    0x7D, 0xEA, 0x36, 0x2B,
    0x3F, 0xAC, 0x8E, 0x00,
    0x95, 0x6A, 0x49, 0x52,
    0xA3, 0xD4, 0xF4, 0x74
);
// clang-format on

} // namespace

TEST(HashMD5Tests, empty)
{
    util::HashMD5 hash;

    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5Empty);
}

TEST(HashMD5Tests, fp32)
{
    util::HashMD5 hash;

    Update(hash, 3.141f);
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0x37, 0x43, 0x6F, 0x40, 0x85, 0x70, 0xD3, 0xD1, 0xFC, 0xC7, 0x24,
                                                       0x0E, 0x85, 0xD3, 0x35, 0x25));
}

TEST(HashMD5Tests, fp32_different_mem_repr)
{
    util::HashMD5 hash;

    float posZero = +0.f;
    float negZero = -0.f;
    ASSERT_NE(0, memcmp(&posZero, &negZero, sizeof(posZero)))
        << "+0.f and -0.f must have different memory representations";

    Update(hash, posZero);
    EXPECT_THAT(hash.getHashAndReset(), g_matchFPzero);

    Update(hash, negZero);
    EXPECT_THAT(hash.getHashAndReset(), g_matchFPzero);
}

TEST(HashMD5Tests, fp64_different_mem_repr)
{
    util::HashMD5 hash;

    double posZero = +0.0;
    double negZero = -0.0;
    ASSERT_NE(0, memcmp(&posZero, &negZero, sizeof(posZero)))
        << "+0.0 and -0.0 must have different memory representations";

    Update(hash, posZero);
    EXPECT_THAT(hash.getHashAndReset(), g_matchFPzero);

    Update(hash, negZero);
    EXPECT_THAT(hash.getHashAndReset(), g_matchFPzero);
}

TEST(HashMD5Tests, fp64)
{
    util::HashMD5 hash;

    Update(hash, 3.141);
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0x18, 0x3B, 0x85, 0x61, 0xD3, 0xEB, 0x14, 0x40, 0x0C, 0xE7, 0x9E,
                                                       0xC9, 0x24, 0x67, 0x08, 0x8F));
}

TEST(HashMD5Tests, vector)
{
    util::HashMD5 hash;

    // clang-format off
    const std::vector<float> vec1 = {0.5, -0.2, 3.141};
    const auto matcher1 =
    t::ElementsAre(
        0xCC, 0xC2, 0xAB, 0x13,
        0x7F, 0x76, 0xA4, 0x86,
        0x13, 0x64, 0x7B, 0x34,
        0x63, 0xF8, 0xE1, 0x94
    );

    const std::vector<float> vec2 = {0.5, -0.2, 3.142};
    const auto matcher2 =
    t::ElementsAre(
        0xA8, 0xFA, 0x59, 0x16,
        0x33, 0x51, 0xF1, 0xCE,
        0xC2, 0x63, 0x65, 0x87,
        0x18, 0xEC, 0xF8, 0x50
    );
    // clang-format on

    Update(hash, vec1);
    EXPECT_THAT(hash.getHashAndReset(), matcher1);

    Update(hash, vec2);
    EXPECT_THAT(hash.getHashAndReset(), matcher2);
}

TEST(HashMD5Tests, reset_works)
{
    util::HashMD5 hash;

    Update(hash, g_str1);
    EXPECT_THAT(hash.getHashAndReset(), t::Not(g_matchMd5Empty));

    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5Empty);
}

TEST(HashMD5Tests, optional_empty)
{
    util::HashMD5 hash;

    std::optional<float> ofloat;
    Update(hash, ofloat);

    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5NullOptional);
}

TEST(HashMD5Tests, optional_empty_same_hash_no_matter_what_type)
{
    util::HashMD5 hash;

    struct Foo
    {
        int foo = 419;
    };

    std::optional<float> ofloat;
    Update(hash, ofloat);
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5NullOptional);

    std::optional<Foo> ofoo;
    Update(hash, ofoo);
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5NullOptional);
}

TEST(HashMD5Tests, char_ptr)
{
    util::HashMD5 hash;

    Update(hash, g_str1);
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5String1);
    Update(hash, g_str2);
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5String2);
}

TEST(HashMD5Tests, std_string)
{
    util::HashMD5 hash;

    Update(hash, std::string{g_str1});
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5String1);
}

TEST(HashMD5Tests, std_string_view_char_ptr)
{
    util::HashMD5 hash;

    Update(hash, std::string_view{g_str1});
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5String1);
}

TEST(HashMD5Tests, std_string_view_string)
{
    util::HashMD5 hash;

    Update(hash, std::string_view{std::string{g_str1}});
    EXPECT_THAT(hash.getHashAndReset(), g_matchMd5String1);
}

TEST(HashMD5Tests, null_char_ptr)
{
    util::HashMD5 hash;

    Update(hash, (const char *)nullptr);
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0xB0, 0x4D, 0x55, 0x50, 0x34, 0x11, 0x5F, 0x3D, 0xD7, 0xD8, 0xF3,
                                                       0xB7, 0x4E, 0xFF, 0xDE, 0x5A));
}

TEST(HashMD5Tests, multiple)
{
    util::HashMD5 hash;

    Update(hash, (const char *)nullptr, 5.0, std::string("rod"), -78, std::vector{4, -5, 3});
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0x82, 0xBC, 0xC1, 0x63, 0xF9, 0x39, 0x85, 0x96, 0x5B, 0x7F, 0x45,
                                                       0xBB, 0x0A, 0x2B, 0xFF, 0x71));

    Update(hash, (const char *)nullptr, 5.0, std::string("rod"), -78, std::vector{4, -4, 3});
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0xB9, 0x2F, 0x16, 0x3D, 0xBA, 0xA9, 0xDA, 0x60, 0x4B, 0x2B, 0x8F,
                                                       0xAB, 0x7E, 0xF0, 0xA4, 0x42));

    Update(hash, (const char *)nullptr, 5.0);
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0x44, 0x65, 0x1E, 0xE0, 0x08, 0x26, 0xE8, 0xCB, 0xB8, 0x8E, 0x63,
                                                       0xF8, 0x22, 0x79, 0xB6, 0x8A));

    Update(hash, (const char *)nullptr, 5.1);
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0x0A, 0x7C, 0x56, 0x14, 0x96, 0xEA, 0x08, 0x1C, 0x12, 0x1E, 0xC1,
                                                       0x1D, 0xF7, 0x33, 0x79, 0x80));
}

template<class T>
struct Foo
{
};

TEST(HashMD5Tests, typeids)
{
    util::HashMD5 hash;

    Update(hash, typeid(Foo<signed int>));
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0xF3, 0xCA, 0xA5, 0xA5, 0x81, 0x1E, 0x00, 0xD6, 0x05, 0x6E, 0x8F,
                                                       0x3D, 0x3D, 0xE3, 0x1A, 0x10));

    Update(hash, typeid(Foo<unsigned int>));
    EXPECT_THAT(hash.getHashAndReset(), t::ElementsAre(0xD1, 0x5B, 0xF4, 0x45, 0x25, 0x91, 0xD3, 0xCB, 0x05, 0x39, 0x64,
                                                       0x73, 0x0B, 0x62, 0x8C, 0x0A));
}

TEST(HashMD5Tests, unique_mem_repr)
{
    util::HashMD5 hash;

    struct Foo
    {
        char bar;
        int  foo;
    } __attribute__((packed));

    auto matcher_1_2 = t::ElementsAre(0x31, 0x7B, 0xC3, 0x40, 0xA8, 0x86, 0xB3, 0x65, 0x49, 0x4D, 0x77, 0x0E, 0x05,
                                      0xFE, 0x61, 0x19);
    for (int i = 0; i < 2; ++i)
    {
        Update(hash, Foo{1, 2});
        EXPECT_THAT(hash.getHashAndReset(), matcher_1_2);
    }

    auto matcher_1_3 = t::ElementsAre(0x4E, 0xFB, 0xCF, 0x54, 0x8D, 0x72, 0x8B, 0xDB, 0x7A, 0x32, 0xAE, 0x37, 0xBC,
                                      0x8D, 0x47, 0xDB);
    for (int i = 0; i < 2; ++i)
    {
        Update(hash, Foo{1, 3});
        EXPECT_THAT(hash.getHashAndReset(), matcher_1_3);
    }
}
