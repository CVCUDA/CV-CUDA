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
#include <nvcv/TensorLayout.h>
#include <nvcv/TensorLayout.hpp>

namespace test = nvcv::test;
namespace t    = ::testing;

// nvcvTensorLayoutMake / nvcvTensorLayoutMakeRange ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutMakeExecTests,
      test::ValueList<test::Param<"str",const char *>,
                      test::Param<"gold",NVCVTensorLayout>>
      {
        {"ABC", NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {"", NVCV_TENSOR_LAYOUT_MAKE("")},
        {"0123456789ABCDE", NVCV_TENSOR_LAYOUT_MAKE("0123456789ABCDE")}
      });

// clang-format on

TEST_P(TensorLayoutMakeExecTests, from_string)
{
    const char             *input = std::get<0>(GetParam());
    const NVCVTensorLayout &gold  = std::get<1>(GetParam());

    NVCVTensorLayout test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorLayoutMake(input, &test));
    EXPECT_EQ(gold, test);
}

TEST_P(TensorLayoutMakeExecTests, from_range)
{
    const char             *input = std::get<0>(GetParam());
    const NVCVTensorLayout &gold  = std::get<1>(GetParam());

    NVCVTensorLayout test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorLayoutMakeRange(input, input + strlen(input), &test));
    EXPECT_EQ(gold, test);
}

// nvcvTensorLayoutMakeFirst =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutMakeFirstExecTests,
      test::ValueList<test::Param<"input",NVCVTensorLayout>,
                      test::Param<"len",int>,
                      test::Param<"gold",NVCVTensorLayout>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 0, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 1, NVCV_TENSOR_LAYOUT_MAKE("A")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 2, NVCV_TENSOR_LAYOUT_MAKE("AB")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 3, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 4, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -1, NVCV_TENSOR_LAYOUT_MAKE("C")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -2, NVCV_TENSOR_LAYOUT_MAKE("BC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -3, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -4, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
      });

// clang-format on

TEST_P(TensorLayoutMakeFirstExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &n     = std::get<1>(GetParam());
    const NVCVTensorLayout &gold  = std::get<2>(GetParam());

    NVCVTensorLayout test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorLayoutMakeFirst(input, n, &test));
    EXPECT_EQ(gold, test);
}

// nvcvTensorLayoutMakeLast =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutMakeLastExecTests,
      test::ValueList<test::Param<"input",NVCVTensorLayout>,
                      test::Param<"len",int>,
                      test::Param<"gold", NVCVTensorLayout>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 0, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 1, NVCV_TENSOR_LAYOUT_MAKE("C")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 2, NVCV_TENSOR_LAYOUT_MAKE("BC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 3, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), 4, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -1, NVCV_TENSOR_LAYOUT_MAKE("A")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -2, NVCV_TENSOR_LAYOUT_MAKE("AB")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -3, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), -4, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
      });

// clang-format on

TEST_P(TensorLayoutMakeLastExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &n     = std::get<1>(GetParam());
    const NVCVTensorLayout &gold  = std::get<2>(GetParam());

    NVCVTensorLayout test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorLayoutMakeLast(input, n, &test));
    EXPECT_EQ(gold, test);
}

// nvcvTensorLayoutMakeSubRange =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutMakeSubRangeExecTests,
                              test::ValueList<test::Param<"input",NVCVTensorLayout>,
                                              test::Param<"beg",int>,
                                              test::Param<"end",int>,
                                              test::Param<"gold",NVCVTensorLayout>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 0, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 1, NVCV_TENSOR_LAYOUT_MAKE("A")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 3, NVCV_TENSOR_LAYOUT_MAKE("ABC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 4, NVCV_TENSOR_LAYOUT_MAKE("ABCD")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 5, NVCV_TENSOR_LAYOUT_MAKE("ABCD")},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -1, 0, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 1, 1, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 1, 2, NVCV_TENSOR_LAYOUT_MAKE("B")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 1, 3, NVCV_TENSOR_LAYOUT_MAKE("BC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 3, 4, NVCV_TENSOR_LAYOUT_MAKE("D")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 3, 5, NVCV_TENSOR_LAYOUT_MAKE("D")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 4, 5, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 5, 7, NVCV_TENSOR_LAYOUT_MAKE("")},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -2, -1, NVCV_TENSOR_LAYOUT_MAKE("C")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -3, -1, NVCV_TENSOR_LAYOUT_MAKE("BC")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -3, -3, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -1, -2, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -4, -3, NVCV_TENSOR_LAYOUT_MAKE("A")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -5, -3, NVCV_TENSOR_LAYOUT_MAKE("A")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -5, -5, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -6, -5, NVCV_TENSOR_LAYOUT_MAKE("")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -4, -1, NVCV_TENSOR_LAYOUT_MAKE("ABC")},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -4, 4, NVCV_TENSOR_LAYOUT_MAKE("ABCD")},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -3, 4, NVCV_TENSOR_LAYOUT_MAKE("BCD")},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 1, -1, NVCV_TENSOR_LAYOUT_MAKE("BC")}
      });

// clang-format on

TEST_P(TensorLayoutMakeSubRangeExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &beg   = std::get<1>(GetParam());
    const int              &end   = std::get<2>(GetParam());
    const NVCVTensorLayout &gold  = std::get<3>(GetParam());

    NVCVTensorLayout test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvTensorLayoutMakeSubRange(input, beg, end, &test));
    EXPECT_EQ(gold, test);
}

// nvcvTensorLayoutFindDimIndex =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutFindDimIndexExecTests,
      test::ValueList<test::Param<"input",NVCVTensorLayout>,
                      test::Param<"label",int>,
                      test::Param<"start",int>,
                      test::Param<"gold", int>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'C', 0, 2},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'C', 2, 2},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'C', 3, -1},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'z', 0, -1},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'D', 0, 3},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'A', 0, 0},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'D', -2, 3},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'D', -1, 3},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 'C', -1, -1},
      });

// clang-format on

TEST_P(TensorLayoutFindDimIndexExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &label = std::get<1>(GetParam());
    const int              &start = std::get<2>(GetParam());
    const int              &gold  = std::get<3>(GetParam());

    EXPECT_EQ(gold, nvcvTensorLayoutFindDimIndex(input, label, start));
}

// nvcvTensorLayoutGetLabel =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutGetLabelExecTests,
      test::ValueList<test::Param<"input",NVCVTensorLayout>,
                      test::Param<"idx",int>,
                      test::Param<"gold", char>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0, 'A'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 1, 'B'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 3, 'D'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 4, '\0'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 5, '\0'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -1, 'D'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -2, 'C'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -4, 'A'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -5, '\0'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), -6, '\0'},
      });

// clang-format on

TEST_P(TensorLayoutGetLabelExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &idx   = std::get<1>(GetParam());
    const char             &gold  = std::get<2>(GetParam());

    EXPECT_EQ(gold, nvcvTensorLayoutGetLabel(input, idx));
}

// nvcvTensorLayoutGetNumDim =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutGetNumDimExecTests,
      test::ValueList<test::Param<"input",NVCVTensorLayout>,
                      test::Param<"gold", int>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 4},
        {NVCV_TENSOR_LAYOUT_MAKE(""), 0},
        {NVCV_TENSOR_LAYOUT_MAKE("."), 1},
        {NVCV_TENSOR_LAYOUT_MAKE("\0"), 1},
        {NVCV_TENSOR_LAYOUT_MAKE("0123456789ABCDE"), 15},
      });

// clang-format on

TEST_P(TensorLayoutGetNumDimExecTests, works)
{
    const NVCVTensorLayout &input = std::get<0>(GetParam());
    const int              &gold  = std::get<1>(GetParam());

    EXPECT_EQ(gold, nvcvTensorLayoutGetNumDim(input));
}

// nvcvTensorLayoutCompare =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutCompareExecTests,
      test::ValueList<test::Param<"a",NVCVTensorLayout>,
                      test::Param<"b",NVCVTensorLayout>,
                      test::Param<"gold", int>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ABCD"), 0},
        {NVCV_TENSOR_LAYOUT_MAKE(""), NVCV_TENSOR_LAYOUT_MAKE(""), 0},
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("\0"), 'A'},
        {NVCV_TENSOR_LAYOUT_MAKE("\0"), NVCV_TENSOR_LAYOUT_MAKE("A"), -'A'},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), NVCV_TENSOR_LAYOUT_MAKE("ABD"), -1},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), NVCV_TENSOR_LAYOUT_MAKE("ABB"), 1},
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("ABC"), -2},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), NVCV_TENSOR_LAYOUT_MAKE("A"), 2},
        {NVCV_TENSOR_LAYOUT_MAKE(""), NVCV_TENSOR_LAYOUT_MAKE("ABC"), -3},
        {NVCV_TENSOR_LAYOUT_MAKE("ABC"), NVCV_TENSOR_LAYOUT_MAKE(""), 3}
      });

// clang-format on

TEST_P(TensorLayoutCompareExecTests, works)
{
    const NVCVTensorLayout &a    = std::get<0>(GetParam());
    const NVCVTensorLayout &b    = std::get<1>(GetParam());
    const int              &gold = std::get<2>(GetParam());

    EXPECT_EQ(gold, nvcvTensorLayoutCompare(a, b));
}

// nvcvTensorLayoutGetName =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutGetNameExecTests,
      test::ValueList<test::Param<"layout",NVCVTensorLayout>,
                      test::Param<"gold", const char *>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), "ABCD"},
        {NVCV_TENSOR_LAYOUT_MAKE(""), ""},
        {NVCV_TENSOR_LAYOUT_MAKE("0"), "0"},
        {NVCV_TENSOR_LAYOUT_MAKE("\0"), ""},
        {NVCV_TENSOR_LAYOUT_MAKE("0123456789ABCDE"), "0123456789ABCDE"},
        {NVCV_TENSOR_LAYOUT_MAKE(" AB"), " AB"},
        {NVCV_TENSOR_LAYOUT_MAKE("AB "), "AB "},
        {NVCV_TENSOR_LAYOUT_MAKE("A B"), "A B"},
        {NVCV_TENSOR_LAYOUT_MAKE("A B"), "A B"},
      });

// clang-format on

TEST_P(TensorLayoutGetNameExecTests, works)
{
    const NVCVTensorLayout &layout = std::get<0>(GetParam());
    const char             *gold   = std::get<1>(GetParam());

    EXPECT_STREQ(gold, nvcvTensorLayoutGetName(&layout));
}

// nvcvTensorLayoutStartsWith =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutStartsWithExecTests,
      test::ValueList<test::Param<"a",NVCVTensorLayout>,
                      test::Param<"b",NVCVTensorLayout>,
                      test::Param<"gold", bool>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("A"), true},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ABCD"), true},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ABC"), true},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("A"), true},
        {NVCV_TENSOR_LAYOUT_MAKE(""), NVCV_TENSOR_LAYOUT_MAKE(""), true},
        {NVCV_TENSOR_LAYOUT_MAKE("23l2"), NVCV_TENSOR_LAYOUT_MAKE(""), true},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ACD"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ABCDE"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("E"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("AE"), NVCV_TENSOR_LAYOUT_MAKE("E"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("AE"), NVCV_TENSOR_LAYOUT_MAKE("EA"), false},
      });

// clang-format on

TEST_P(TensorLayoutStartsWithExecTests, works)
{
    const NVCVTensorLayout &a    = std::get<0>(GetParam());
    const NVCVTensorLayout &b    = std::get<1>(GetParam());
    const bool             &gold = std::get<2>(GetParam());

    EXPECT_EQ(gold, (bool)nvcvTensorLayoutStartsWith(a, b));
}

// nvcvTensorLayoutEndsWith =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutEndsWithExecTests,
      test::ValueList<test::Param<"a",NVCVTensorLayout>,
                      test::Param<"b",NVCVTensorLayout>,
                      test::Param<"gold", bool>>
      {
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("A"), true},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ABCD"), true},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("BCD"), true},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("D"), true},
        {NVCV_TENSOR_LAYOUT_MAKE(""), NVCV_TENSOR_LAYOUT_MAKE(""), true},
        {NVCV_TENSOR_LAYOUT_MAKE("23l2"), NVCV_TENSOR_LAYOUT_MAKE(""), true},

        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ACD"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("ABCD"), NVCV_TENSOR_LAYOUT_MAKE("ZABCD"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("A"), NVCV_TENSOR_LAYOUT_MAKE("E"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("AE"), NVCV_TENSOR_LAYOUT_MAKE("A"), false},
        {NVCV_TENSOR_LAYOUT_MAKE("AE"), NVCV_TENSOR_LAYOUT_MAKE("EA"), false},
      });

// clang-format on

TEST_P(TensorLayoutEndsWithExecTests, works)
{
    const NVCVTensorLayout &a    = std::get<0>(GetParam());
    const NVCVTensorLayout &b    = std::get<1>(GetParam());
    const bool             &gold = std::get<2>(GetParam());

    EXPECT_EQ(gold, (bool)nvcvTensorLayoutEndsWith(a, b));
}

// TensorLayout ostream =======================================================

// clang-format off
NVCV_TEST_SUITE_P(TensorLayoutOStreamExecTests,
      test::ValueList<test::Param<"layout",nvcv::TensorLayout>,
                      test::Param<"gold", const char *>>
      {
        {nvcv::TensorLayout("ABCD"), "ABCD"},
        {nvcv::TensorLayout(""), ""},
        {nvcv::TensorLayout("0"), "0"},
        {nvcv::TensorLayout("\0"), ""},
        {nvcv::TensorLayout("0123456789ABCDE"), "0123456789ABCDE"},
        {nvcv::TensorLayout(" AB"), " AB"},
        {nvcv::TensorLayout("AB "), "AB "},
        {nvcv::TensorLayout("A B"), "A B"},
        {nvcv::TensorLayout("A B"), "A B"},
      });

// clang-format on

TEST_P(TensorLayoutOStreamExecTests, works)
{
    const nvcv::TensorLayout &layout = std::get<0>(GetParam());
    const char               *gold   = std::get<1>(GetParam());

    std::ostringstream ss;
    ss << layout;

    EXPECT_STREQ(gold, ss.str().c_str());
}
