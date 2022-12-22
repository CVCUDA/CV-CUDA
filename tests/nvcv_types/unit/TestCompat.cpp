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
#if __has_include(<sys/random.h>)
#    include <sys/random.h>
#    define HAS_SYS_RANDOM_H 1
#endif
#include <util/Compat.h>

namespace test = nvcv::test;
namespace t    = ::testing;

#if HAS_SYS_RANDOM_H

static char g_buf[256] = {};

class CompatGetRandomParamTest
    : public t::TestWithParam<
          std::tuple<test::Param<"buffer", void *>, test::Param<"length", size_t>, test::Param<"flags", unsigned int>,
                     test::Param<"gold_retval", ssize_t>, test::Param<"gold_errno", int>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Negative, CompatGetRandomParamTest,
{
    //       buffer,            length,  flags, gold_retval, gold_errno
    {          NULL,                 1,      0,          -1,     EFAULT},
    { (void *)0x666,                 1,      0,          -1,     EFAULT},
    { (void *)0x666,                -1,      0,          -1,     EFAULT},
    {         g_buf,                 1,  0x666,          -1,     EINVAL},
});

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, CompatGetRandomParamTest,
{
    //       buffer,        length,  flags,   gold_retval,    gold_errno
    {(void *)&g_buf, sizeof(g_buf),      0, sizeof(g_buf), 0 /*ignored*/},
    {(void *)&g_buf,             1,      0,             1, 0 /*ignored*/},
    {(void *)&g_buf,             0,      0,             0, 0 /*ignored*/},
});

// clang-format on

TEST_P(CompatGetRandomParamTest, test)
{
    void        *buf         = std::get<0>(GetParam());
    size_t       length      = std::get<1>(GetParam());
    unsigned int flags       = std::get<2>(GetParam());
    int          gold_retval = std::get<3>(GetParam());
    int          gold_errno  = std::get<4>(GetParam());

    ssize_t ret = Compat_getrandom(buf, length, flags);
    EXPECT_EQ(gold_retval, ret);
    if (ret < 0)
    {
        EXPECT_EQ(gold_errno, errno);
    }
}

class CompatGetRandomExecTest : public t::TestWithParam<test::Param<"flags", unsigned int>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Flags, CompatGetRandomExecTest,
{
    //                    flags
    {                         0 },
    {             GRND_NONBLOCK },
    {               GRND_RANDOM },
    { GRND_RANDOM|GRND_NONBLOCK },
});

// clang-format on

TEST_P(CompatGetRandomExecTest, works)
{
    unsigned int flags = GetParam();

    // when using urandom, it's guaranteed that it'll return at least 256 bytes
    // so let's use 256.
    char    buf1[256 + 1] = {};
    ssize_t n             = Compat_getrandom(buf1, 256, flags);
    ASSERT_EQ(256, n);
    ASSERT_EQ(0, buf1[256]);

    char buf2[256 + 1] = {};
    n                  = Compat_getrandom(buf2, 256, flags);
    ASSERT_EQ(256, n);
    ASSERT_EQ(0, buf2[256]);
    ASSERT_THAT(buf1, t::Not(t::ElementsAreArray(buf2)));
}

class CompatGetEntropyParamTest
    : public t::TestWithParam<std::tuple<test::Param<"buffer", void *>, test::Param<"length", size_t>,
                                         test::Param<"gold_retval", int>, test::Param<"gold_errno", int>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Negative, CompatGetEntropyParamTest,
{
    //       buffer,            length,  gold_retval, gold_errno
    {          NULL,                 1,           -1,     EFAULT},
    { (void *)0x666,                 1,           -1,     EFAULT},
    {         g_buf,               257,           -1,     EIO},
});

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, CompatGetEntropyParamTest,
{
    //       buffer,        length,  gold_retval,    gold_errno
    {(void *)&g_buf, sizeof(g_buf),            0, 0 /*ignored*/},
    {(void *)&g_buf,             1,            0, 0 /*ignored*/},
    {(void *)&g_buf,             0,            0, 0 /*ignored*/},
});

// clang-format on

TEST_P(CompatGetEntropyParamTest, test)
{
    void  *buf         = std::get<0>(GetParam());
    size_t length      = std::get<1>(GetParam());
    int    gold_retval = std::get<2>(GetParam());
    int    gold_errno  = std::get<3>(GetParam());

    ssize_t ret = Compat_getentropy(buf, length);
    EXPECT_EQ(gold_retval, ret);
    if (ret < 0)
    {
        EXPECT_EQ(gold_errno, errno);
    }
}

TEST(CompatGetEntropyExecTest, works)
{
    char buf1[256] = {};
    ASSERT_EQ(0, Compat_getentropy(buf1, sizeof(buf1) - 1));
    ASSERT_EQ(0, buf1[sizeof(buf1) - 1]);

    char buf2[256] = {};
    ASSERT_EQ(0, Compat_getentropy(buf2, sizeof(buf2) - 1));
    ASSERT_EQ(0, buf1[sizeof(buf2) - 1]);

    ASSERT_THAT(buf1, t::Not(t::ElementsAreArray(buf2)));
}

#endif
