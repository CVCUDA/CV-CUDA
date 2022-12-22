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

#include <common/ValueList.hpp>

#include <type_traits>

namespace t    = ::testing;
namespace test = nvcv::test;

TEST(JoinTupleTests, only_values)
{
    std::tuple<int, float, char> gold{1, 3.2f, 'a'};

    EXPECT_EQ(gold, test::detail::JoinTuple(1, 3.2f, 'a'));
}

TEST(JoinTupleTests, all_tuples)
{
    std::tuple<int, float, char, long, signed, double> gold{1, 3.2f, 'a', 43l, 1, 3.2};

    EXPECT_EQ(gold,
              test::detail::JoinTuple(std::make_tuple(1, 3.2f), std::make_tuple('a'), std::make_tuple(43l, 1, 3.2)));
}

TEST(JoinTupleTests, tuples_and_values_value_first)
{
    std::tuple<int, float, char, long, signed, double> gold{1, 3.2f, 'a', 43l, 1, 3.2};

    EXPECT_EQ(gold, test::detail::JoinTuple(1, std::make_tuple(3.2f, 'a'), std::make_tuple(43l, 1, 3.2)));
}

TEST(JoinTupleTests, tuples_and_values_tuple_first)
{
    std::tuple<int, float, char, long, signed, double> gold{1, 3.2f, 'a', 43l, 1, 3.2};

    EXPECT_EQ(gold, test::detail::JoinTuple(std::make_tuple(1, 3.2f), 'a', std::make_tuple(43l, 1, 3.2)));
}

TEST(ExtractTupleTests, extract_first_element_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<int>                                    gold{1};

    EXPECT_EQ(gold, test::detail::ExtractTuple<0>(input));
}

TEST(ExtractTupleTests, extract_middle_element_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<char>                                   gold{'a'};

    EXPECT_EQ(gold, test::detail::ExtractTuple<2>(input));
}

TEST(ExtractTupleTests, extract_last_element_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<double>                                 gold{3.2};

    EXPECT_EQ(gold, test::detail::ExtractTuple<5>(input));
}

TEST(ExtractTupleTests, extract_first_few_elements_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<int, float, char>                       gold{1, 3.2f, 'a'};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<0, 1, 2>(input)));
}

TEST(ExtractTupleTests, extract_middle_few_elements_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<char, long, signed>                     gold{'a', 43l, 1};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<2, 3, 4>(input)));
}

TEST(ExtractTupleTests, extract_last_few_elements_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<long, signed, double>                   gold{43l, 1, 3.2};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<3, 4, 5>(input)));
}

TEST(ExtractTupleTests, extract_all_elements_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};

    EXPECT_EQ(input, (test::detail::ExtractTuple<0, 1, 2, 3, 4, 5>(input)));
}

TEST(ExtractTupleTests, extract_element_of_tuple_with_one)
{
    std::tuple<int> input{1};

    EXPECT_EQ(input, test::detail::ExtractTuple<0>(input));
}

TEST(ExtractTupleTests, extract_no_elements_of_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<>                                       gold;

    EXPECT_EQ(gold, test::detail::ExtractTuple<>(input));
}

TEST(ExtractTupleTests, extract_no_elements_of_tuple_with_one)
{
    std::tuple<int> input{1};
    std::tuple<>    gold;

    EXPECT_EQ(gold, test::detail::ExtractTuple<>(input));
}

TEST(ExtractTupleTests, extract_same_element_multiple_times_from_tuple_with_one)
{
    std::tuple<int>           input{1};
    std::tuple<int, int, int> gold{1, 1, 1};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<0, 0, 0>(input)));
}

TEST(ExtractTupleTests, extract_same_element_multiple_times_from_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<char, char, char>                       gold{'a', 'a', 'a'};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<2, 2, 2>(input)));
}

TEST(ExtractTupleTests, extract_multiple_elements_with_repetition_from_tuple_with_many)
{
    std::tuple<int, float, char, long, signed, double> input{1, 3.2f, 'a', 43l, 1, 3.2};
    std::tuple<char, long, char>                       gold{'a', 43l, 'a'};

    EXPECT_EQ(gold, (test::detail::ExtractTuple<2, 3, 2>(input)));
}

TEST(ValueListTests, concat_lists)
{
    test::ValueList<int> a = {6, 9, 5};
    test::ValueList<int> b = {4, 2, 1};
    test::ValueList<int> c = {10, 8, 3};

    test::ValueList<int> gold = {6, 9, 5, 4, 2, 1, 10, 8, 3};

    EXPECT_EQ(gold, Concat(a, b, c));
}

TEST(ValueListTests, concat_one_list)
{
    test::ValueList<int> a = {6, 9, 5};

    EXPECT_EQ(a, Concat(a));
}

TEST(ValueListTests, concat_one_value)
{
    test::ValueList<int> gold = {4};

    EXPECT_EQ(gold, test::Concat(4));
}

TEST(ValueListTests, concat_multiple_value)
{
    test::ValueList<int> gold = {4, 5, 6, 7};

    EXPECT_EQ(gold, test::Concat(4, 5, 6, 7));
}

TEST(ValueListTests, concat_with_value_at_beginning)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int> gold = {4, 6, 3, 5};

    EXPECT_EQ(gold, Concat(4, a));
}

TEST(ValueListTests, concat_with_value_at_in_the_middle)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> c = {10, 8, 3};

    test::ValueList<int> gold = {6, 3, 5, 4, 10, 8, 3};

    EXPECT_EQ(gold, Concat(a, 4, c));
}

TEST(ValueListTests, concat_with_value_at_end)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> c = {10, 8, 3};

    test::ValueList<int> gold = {6, 3, 5, 10, 8, 3, 4};

    EXPECT_EQ(gold, Concat(a, c, 4));
}

TEST(ValueListTests, concat_with_multiple_value)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> c = {10, 8, 3};

    test::ValueList<int> gold = {20, 34, 6, 3, 5, 56, 10, 8, 3, 44};

    EXPECT_EQ(gold, Concat(20, 34, a, 56, c, 44));
}

TEST(ValueListTests, concat_keep_repeated_values)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int> gold = {6, 3, 5, 6, 3, 5, 6, 3, 5};

    EXPECT_EQ(gold, Concat(a, a, a));
}

TEST(ValueListTests, combine_one_list_is_identity)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(a, Combine(a));
}

TEST(ValueListTests, combine_same_list_twice)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int, int> gold = {std::make_tuple(6, 6), std::make_tuple(6, 3), std::make_tuple(6, 5),
                                      std::make_tuple(3, 6), std::make_tuple(3, 3), std::make_tuple(3, 5),
                                      std::make_tuple(5, 6), std::make_tuple(5, 3), std::make_tuple(5, 5)};

    EXPECT_EQ(gold, Combine(a, a));
}

TEST(ValueListTests, combine_with_value_begin)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int, int> gold = {std::make_tuple(6, 6), std::make_tuple(6, 3), std::make_tuple(6, 5)};

    EXPECT_EQ(gold, Combine(6, a));
}

TEST(ValueListTests, combine_with_value_at_end)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int, int> gold = {std::make_tuple(6, 6), std::make_tuple(3, 6), std::make_tuple(5, 6)};

    EXPECT_TRUE((std::is_same_v<test::ValueList<int, int>, decltype(Combine(a, 6))>));

    EXPECT_EQ(gold, Combine(a, 6));
}

TEST(ValueListTests, combine_complex_lists)
{
    test::ValueList<int, int> a = {std::make_tuple(6, 6), std::make_tuple(6, 3)};
    test::ValueList<int>      b = {4, 5};

    test::ValueList<int, int, int> gold
        = {std::make_tuple(6, 6, 4), std::make_tuple(6, 6, 5), std::make_tuple(6, 3, 4), std::make_tuple(6, 3, 5)};

    EXPECT_TRUE((std::is_same_v<test::ValueList<int, int, int>, decltype(Combine(a, b))>));
    EXPECT_EQ(gold, Combine(a, b));
}

TEST(ValueListTests, combine_associativity)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {1, 2};
    test::ValueList<int> c = {10, 8, 3};

    EXPECT_EQ(Combine(a, b, c), Combine(Combine(a, b), c));
    EXPECT_EQ(Combine(a, b, c), Combine(a, Combine(b, c)));
}

TEST(ValueListTests, difference_subtract_empty)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(a, Difference(a, test::ValueList<int>()));
}

TEST(ValueListTests, difference_from_empty_is_empty)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(test::ValueList<int>(), Difference(test::ValueList<int>(), a));
}

TEST(ValueListTests, difference_works1)
{
    test::ValueList<int> a = {7, 3, 5, 8, 6};
    test::ValueList<int> b = {6, 5};

    test::ValueList<int> gold = {7, 3, 8};

    EXPECT_EQ(gold, Difference(a, b));
}

TEST(ValueListTests, difference_result_empty)
{
    test::ValueList<int> a = {6, 5};
    test::ValueList<int> b = {6, 3, 5, 7};

    EXPECT_EQ(test::ValueList<int>(), Difference(a, b));
}

TEST(ValueListTests, nested_parameters)
{
    test::ValueList<int> a = {6, 2};
    test::ValueList<int> b = {1, 3};

    test::ValueList<int, int, int, int, int> c = Combine(b, Combine(a, a), Combine(b, b));

    test::ValueList<int, int> aa;
    for (int a1 : a)
    {
        for (int a2 : a)
        {
            aa.emplace_back(a1, a2);
        }
    }

    test::ValueList<int, int> bb;
    for (int b1 : b)
    {
        for (int b2 : b)
        {
            bb.emplace_back(b1, b2);
        }
    }

    test::ValueList<int, int, int, int, int> gold;

    for (int b1 : b)
    {
        for (std::tuple<int, int> a2 : aa)
        {
            for (std::tuple<int, int> b2 : bb)
            {
                gold.emplace_back(b1, std::get<0>(a2), std::get<1>(a2), std::get<0>(b2), std::get<1>(b2));
            }
        }
    }

    EXPECT_EQ(gold, c);
}

TEST(ValueListTests, select_if_one_match_simple)
{
    test::ValueList<int> a    = {6, 2};
    test::ValueList<int> gold = {6};

    EXPECT_EQ(gold, SelectIf(test::Match<0>(6), a));
}

TEST(ValueListTests, select_if_no_match_simple)
{
    test::ValueList<int> a = {6, 2};
    test::ValueList<int> gold;

    EXPECT_EQ(gold, SelectIf(test::Match<0>(3), a));
}

TEST(ValueListTests, select_if_all_match_simple)
{
    test::ValueList<int> a    = {6, 6};
    test::ValueList<int> gold = a;

    EXPECT_EQ(gold, SelectIf(test::Match<0>(6), a));
}

TEST(ValueListTests, select_if_one_match_list_simple)
{
    test::ValueList<int> a    = {6, 2};
    test::ValueList<int> gold = {6};

    EXPECT_EQ(gold, SelectIf(test::Match<0>(test::ValueList<int>{6, 3}), a));
}

TEST(ValueListTests, select_if_no_match_list_simple)
{
    test::ValueList<int> a = {6, 2};
    test::ValueList<int> gold;

    EXPECT_EQ(gold, SelectIf(test::Match<0>(test::ValueList<int>{5, 3}), a));
}

TEST(ValueListTests, select_if_all_match_list_simple)
{
    test::ValueList<int> a    = {6, 2};
    test::ValueList<int> gold = a;

    EXPECT_EQ(gold, SelectIf(test::Match<0>(test::ValueList<int>{2, 6}), a));
}

TEST(ValueListTests, select_if_one_match_multi)
{
    test::ValueList<int, float> a = {
        {5,   0.5},
        {3, 1.125}
    };
    test::ValueList<int, float> gold = {
        {5, 0.5}
    };

    EXPECT_EQ(gold, SelectIf(test::Match<1>(0.5), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0>(5), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(5, 0.5), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(0.5, 5), a));
}

TEST(ValueListTests, select_if_no_match_multi)
{
    test::ValueList<int, float> a = {
        {5,   0.5},
        {3, 1.125}
    };
    test::ValueList<int, float> gold;

    EXPECT_EQ(gold, SelectIf(test::Match<1>(0.2), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0>(1), a));

    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(1, 0.5), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(5, 0.51), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(4, 1.125), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(3, 1.126), a));

    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(0.5, 1), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(0.51, 5), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(1.125, 4), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(1.126, 3), a));
}

TEST(ValueListTests, select_if_match_some_multi)
{
    test::ValueList<int, float> a = {
        {5,   0.5},
        {3, 1.125},
        {4, 1.125},
        {3, 1.126}
    };
    test::ValueList<int, float> gold = {
        {3, 1.125},
        {4, 1.125}
    };

    test::ValueList<int, float> match01 = {
        {3, 1.125},
        {4, 1.125},
        {0,  -0.5}
    };
    test::ValueList<float, int> match10
        = test::Transform(match01, [](int i, float f) { return std::make_tuple(f, i); });

    EXPECT_EQ(gold, SelectIf(Match(match01), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(match01), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(match10), a));
}

TEST(ValueListTests, select_if_match_some_multi_repeated_args)
{
    test::ValueList<int>      a     = {5, 2, 3, 8, 7};
    test::ValueList<int>      match = {5, 8, 7};
    test::ValueList<int, int> gold  = {
         {5, 8},
         {5, 7},
         {8, 5},
         {8, 7},
         {7, 5},
         {7, 8}
    };

    EXPECT_EQ(gold, SelectIf(Match(match * match - test::IsSameArgs), a * a));
}

TEST(ValueListTests, select_if_match_some_value)
{
    test::ValueList<int, float> a = {
        {5,   0.5},
        {3, 1.125},
        {4, 1.125},
        {3, 1.126}
    };
    test::ValueList<int, float> gold = {
        {3, 1.125}
    };

    EXPECT_EQ(gold, SelectIf(test::Match(3, 1.125), a));
    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(3, 1.125), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(1.125, 3), a));
}

TEST(ValueListTests, select_if_all_match_multi_first_same_value)
{
    test::ValueList<int, float> a = {
        {5,   0.5},
        {5, 1.125}
    };
    test::ValueList<int, float> gold = a;

    EXPECT_EQ(gold, SelectIf(test::Match<0>(5), a));
}

TEST(ValueListTests, select_if_all_match_multi_first_different_values)
{
    test::ValueList<int, float> a = {
        {4,   0.5},
        {5, 1.125}
    };
    test::ValueList<int, float> gold = a;

    EXPECT_EQ(gold, SelectIf(test::Match<0, 1>(a), a));
    EXPECT_EQ(gold, SelectIf(test::Match<1, 0>(test::ValueList<float, int>{
                                 {  0.5, 4},
                                 {1.125, 5}
    }),
                             a));
}

TEST(ValueListTests, select_fn_one_arg)
{
    test::ValueList<int, float> a = {
        {4,   0.5},
        {7, 1.125},
        {2,  54.3},
        {5,   3.2}
    };
    test::ValueList<int, float> gold = {
        {7, 1.125},
        {5,   3.2}
    };

    EXPECT_EQ(gold, test::SelectIf<0>([](float v) { return v >= 5; }, a));
}

TEST(ValueListTests, select_fn_multiple_args)
{
    test::ValueList<int, float, char, unsigned> a = {
        {-4,   0.5, 'a',   5},
        { 7, 1.125, 'z',   9},
        { 2,  54.3, 'g', 634},
        { 5,   3.2, 'y',  32}
    };
    test::ValueList<int, float, char, unsigned> gold = {
        {7, 1.125, 'z',  9},
        {5,   3.2, 'y', 32}
    };

    EXPECT_EQ(gold, (test::SelectIf<1, 3>([](float a, unsigned b) { return a >= 1 && a <= 10 && b <= 100; }, a)));
}

TEST(ValueListTests, select_any_single_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {1, 4, 5};

    EXPECT_EQ(gold, SelectIfAny([](int v) { return v != 3; }, a));
}

TEST(ValueListTests, select_any_single_no_match)
{
    test::ValueList<int> a = {1, 4, 3, 5, 3};

    EXPECT_EQ(a, SelectIfAny([](int v) { return true; }, a));
}

TEST(ValueListTests, select_any_single_all_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {};

    EXPECT_EQ(gold, SelectIfAny([](int v) { return false; }, a));
}

TEST(ValueListTests, select_any_multi_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };

    test::ValueList<int, char> gold = {
        {3, 'a'},
        {3, 'c'}
    };

    EXPECT_EQ(gold, SelectIfAny([](int v) { return v == 3; }, a));
}

TEST(ValueListTests, select_any_multi_no_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };

    EXPECT_EQ(a, SelectIfAny([](int v) { return true; }, a));
}

TEST(ValueListTests, select_any_multi_all_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold = {};

    EXPECT_EQ(gold, SelectIfAny([](int v) { return false; }, a));
}

TEST(ValueListTests, select_all_single_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {1, 4, 5};

    EXPECT_EQ(gold, SelectIfAll([](int v) { return v != 3; }, a));
}

TEST(ValueListTests, select_all_single_no_match)
{
    test::ValueList<int> a = {1, 4, 3, 5, 3};
    EXPECT_EQ(a, SelectIfAll([](int v) { return true; }, a));
}

TEST(ValueListTests, select_all_single_all_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {};
    EXPECT_EQ(gold, SelectIfAll([](int v) { return false; }, a));
}

TEST(ValueListTests, select_all_multi_no_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    EXPECT_EQ(a, SelectIfAll([](int v) { return true; }, a));
}

TEST(ValueListTests, select_all_multi_all_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold;

    EXPECT_EQ(gold, SelectIfAll([](int v) { return false; }, a));
}

TEST(ValueListTests, select_all_multi)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold = {
        {3, 'c'}
    };

    EXPECT_EQ(gold, SelectIfAll([](int v) { return v == 3 || v == 'c'; }, a));
}

TEST(ValueListTests, remove_all_args_same_one_arg)
{
    test::ValueList<int> a = {6, 2};

    EXPECT_EQ(a, RemoveIf(test::IsSameArgs, a));
}

TEST(ValueListTests, remove_all_args_same_several_different_args)
{
    test::ValueList<int, int> a = {std::make_tuple(1, 4), std::make_tuple(4, 1), std::make_tuple(4, 7)};

    EXPECT_EQ(a, RemoveIf(test::IsSameArgs, a));
}

TEST(ValueListTests, remove_all_args_same_all_args_are_the_same)
{
    test::ValueList<int, int> a = {std::make_tuple(1, 1), std::make_tuple(2, 2), std::make_tuple(3, 3)};

    EXPECT_EQ((test::ValueList<int, int>{}), RemoveIf(test::IsSameArgs, a));
}

TEST(ValueListTests, remove_all_args_same_some_args_are_the_same)
{
    test::ValueList<int, int> a
        = {std::make_tuple(1, 1), std::make_tuple(3, 2), std::make_tuple(1, 2), std::make_tuple(3, 3)};

    test::ValueList<int, int> gold = {std::make_tuple(3, 2), std::make_tuple(1, 2)};

    EXPECT_EQ(gold, RemoveIf(test::IsSameArgs, a));
}

TEST(ValueListTests, remove_any_single_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {1, 4, 5};

    EXPECT_EQ(gold, RemoveIfAny([](int v) { return v == 3; }, a));
}

TEST(ValueListTests, remove_any_single_no_match)
{
    test::ValueList<int> a = {1, 4, 3, 5, 3};

    EXPECT_EQ(a, RemoveIfAny([](int v) { return false; }, a));
}

TEST(ValueListTests, remove_any_single_all_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {};

    EXPECT_EQ(gold, RemoveIfAny([](int v) { return true; }, a));
}

TEST(ValueListTests, remove_any_multi_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };

    test::ValueList<int, char> gold = {
        {3, 'a'}
    };

    EXPECT_EQ(gold, RemoveIfAny([](int v) { return v != 3 && v != 'a'; }, a));
}

TEST(ValueListTests, remove_any_multi_no_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };

    EXPECT_EQ(a, RemoveIfAny([](int v) { return false; }, a));
}

TEST(ValueListTests, remove_any_multi_all_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold = {};

    EXPECT_EQ(gold, RemoveIfAny([](int v) { return true; }, a));
}

TEST(ValueListTests, remove_all_single_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {1, 4, 5};

    EXPECT_EQ(gold, RemoveIfAll([](int v) { return v == 3; }, a));
}

TEST(ValueListTests, remove_all_single_no_match)
{
    test::ValueList<int> a = {1, 4, 3, 5, 3};
    EXPECT_EQ(a, RemoveIfAll([](int v) { return false; }, a));
}

TEST(ValueListTests, remove_all_single_all_match)
{
    test::ValueList<int> a    = {1, 4, 3, 5, 3};
    test::ValueList<int> gold = {};
    EXPECT_EQ(gold, RemoveIfAll([](int v) { return true; }, a));
}

TEST(ValueListTests, remove_all_multi_no_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    EXPECT_EQ(a, RemoveIfAll([](int v) { return false; }, a));
}

TEST(ValueListTests, remove_all_multi_all_match)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold;

    EXPECT_EQ(gold, RemoveIfAll([](int v) { return true; }, a));
}

TEST(ValueListTests, remove_all_multi)
{
    test::ValueList<int, char> a = {
        {1, 'c'},
        {3, 'a'},
        {5, 'b'},
        {3, 'c'}
    };
    test::ValueList<int, char> gold = {
        {3, 'a'},
        {3, 'c'}
    };

    EXPECT_EQ(gold, RemoveIfAll([](int v) { return v != 3 || v == 'c'; }, a));
}

TEST(ValueListTests, exists_positive)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 9, 2};

    EXPECT_TRUE(list.exists(4));
}

TEST(ValueListTests, exists_negative)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 9, 2};

    EXPECT_FALSE(list.exists(666));
}

TEST(ValueListTests, exists_empty_list_negative)
{
    test::ValueList<int> list = {};

    EXPECT_FALSE(list.exists(666));
}

TEST(ValueListTests, combine_with_only_values)
{
    test::ValueList<int, float, double, char> gold = {std::make_tuple(4, 2.2f, 10.5, 'c')};

    EXPECT_EQ(gold, test::Combine(4, 2.2f, 10.5, 'c'));
}

TEST(ValueListTests, erase_only_one)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 9, 2};

    test::ValueList<int> gold = {1, 5, 6, 3, 9, 2};

    EXPECT_TRUE(list.erase(4));
    EXPECT_EQ(gold, list);
}

TEST(ValueListTests, erase_several)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 4, 9, 2, 4};

    test::ValueList<int> gold = {1, 5, 6, 3, 9, 2};

    EXPECT_TRUE(list.erase(4));

    EXPECT_EQ(gold, list);
}

TEST(ValueListTests, erase_doesnt_exist)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 9, 2, 4};

    test::ValueList<int> gold = list;

    EXPECT_FALSE(list.erase(666));

    EXPECT_EQ(gold, list);
}

TEST(ValueListTests, transform_same_arg_count_1)
{
    test::ValueList<int> list = {1, 5, 4, 6, 3, 9, 2};

    auto xform = [](int i)
    {
        return i * 2;
    };

    test::ValueList<int> gold = {2, 10, 8, 12, 6, 18, 4};

    EXPECT_EQ(gold, Transform(list, xform));
}

TEST(ValueListTests, transform_same_arg_count_2)
{
    enum class A
    {
        a1,
        a2,
        a3,
    };

    test::ValueList<int, A> list = {std::make_tuple(1, A::a1), std::make_tuple(3, A::a1), std::make_tuple(5, A::a3)};

    auto xform = [](int i, A a)
    {
        return std::make_tuple(i * 3, a == A::a1 ? A::a2 : a);
    };

    test::ValueList<int, A> gold = {std::make_tuple(3, A::a2), std::make_tuple(9, A::a2), std::make_tuple(15, A::a3)};

    EXPECT_TRUE((std::is_same_v<test::ValueList<int, A>, decltype(Transform(list, xform))>));

    EXPECT_EQ(gold, Transform(list, xform));
}

TEST(ValueListTests, transform_increase_arg_count)
{
    test::ValueList<int> list = {1, 5, 4, 6};

    enum class Parity
    {
        even,
        odd
    };

    auto xform = [](int i)
    {
        if (i & 1)
        {
            return std::make_tuple(i * 2, Parity::odd);
        }
        else
        {
            return std::make_tuple(i * 2, Parity::even);
        }
    };

    test::ValueList<int, Parity> gold = {std::make_tuple(2, Parity::odd), std::make_tuple(10, Parity::odd),
                                         std::make_tuple(8, Parity::even), std::make_tuple(12, Parity::even)};

    EXPECT_EQ(gold, Transform(list, xform));
}

TEST(ValueListTests, transform_decrease_arg_count)
{
    enum class Parity
    {
        even,
        odd
    };

    test::ValueList<int, Parity> list = {std::make_tuple(1, Parity::odd), std::make_tuple(4, Parity::odd),
                                         std::make_tuple(2, Parity::even), std::make_tuple(5, Parity::even)};

    auto xform = [](int i, Parity p)
    {
        if (p == Parity::even)
        {
            return i + 1;
        }
        else
        {
            return i * 2;
        }
    };

    test::ValueList<int> gold = {2, 8, 3, 6};

    EXPECT_EQ(gold, Transform(list, xform));
}

TEST(ValueListTests, transform_multiple_params_per_input)
{
    test::ValueList<int> list = {1, 2, 3};

    auto xform = [](int i)
    {
        test::ValueList<int, int> out;
        out.emplace_back(-i, i);
        out.emplace_back(i, -i);
        return out;
    };

    test::ValueList<int, int> gold = {
        {-1,  1},
        { 1, -1},
        {-2,  2},
        { 2, -2},
        {-3,  3},
        { 3, -3}
    };

    EXPECT_EQ(gold, Transform(list, xform));
}

TEST(ValueListTests, intersection_with_empty_is_empty)
{
    test::ValueList<int> a = {1, 2, 3};
    test::ValueList<int> b = {};

    EXPECT_EQ(b, Intersection(a, b));
    EXPECT_EQ(b, Intersection(b, a));
}

TEST(ValueListTests, intersection_disjoint_sets)
{
    test::ValueList<int> a = {3, 2, 1};
    test::ValueList<int> b = {4, 6, 5};

    test::ValueList<int> gold = {};

    EXPECT_EQ(gold, Intersection(a, b));
    EXPECT_EQ(gold, Intersection(b, a));
}

TEST(ValueListTests, intersection_not_sorted_some_overlapping_elements)
{
    test::ValueList<int> a = {3, 2, 1};
    test::ValueList<int> b = {2, 4, 3};

    test::ValueList<int> gold = {3, 2};

    EXPECT_EQ(gold, Intersection(a, b));
}

TEST(ValueListTests, intersection_sorted_some_overlapping_elements)
{
    test::ValueList<int> a = {1, 2, 3};
    test::ValueList<int> b = {2, 3, 4};

    test::ValueList<int> gold = {2, 3};

    EXPECT_EQ(gold, Intersection(a, b));
    EXPECT_EQ(gold, Intersection(b, a));
}

TEST(ValueListTests, intersection_sorted_one_set_includes_the_other)
{
    test::ValueList<int> a = {1, 2, 3, 4};
    test::ValueList<int> b = {2, 3, 4};

    EXPECT_EQ(b, Intersection(a, b));
    EXPECT_EQ(b, Intersection(b, a));
}

TEST(ValueListTests, intersection_sorted_more_than_two_sets)
{
    test::ValueList<int> a = {1, 2, 3, 4};
    test::ValueList<int> b = {2, 3, 4, 10, 42};
    test::ValueList<int> c = {2, 3, 4, 5, 6};

    test::ValueList<int> gold = {2, 3, 4};

    EXPECT_EQ(gold, Intersection(a, b, c));
    EXPECT_EQ(gold, Intersection(a, c, b));
    EXPECT_EQ(gold, Intersection(b, a, c));
    EXPECT_EQ(gold, Intersection(b, c, a));
    EXPECT_EQ(gold, Intersection(c, a, b));
    EXPECT_EQ(gold, Intersection(c, b, a));
}

TEST(ValueListTests, zip_one_list_is_identity)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(a, Zip(a));
}

TEST(ValueListTests, zip_same_list_twice)
{
    test::ValueList<int> a = {6, 3, 5};

    test::ValueList<int, int> gold = {std::make_tuple(6, 6), std::make_tuple(3, 3), std::make_tuple(5, 5)};

    EXPECT_EQ(gold, Zip(a, a));
}

TEST(ValueListTests, zip_complex_lists)
{
    test::ValueList<int, int> a = {std::make_tuple(6, 6), std::make_tuple(6, 3)};
    test::ValueList<int>      b = {4, 5};

    test::ValueList<int, int, int> gold = {std::make_tuple(6, 6, 4), std::make_tuple(6, 3, 5)};

    EXPECT_TRUE((std::is_same_v<test::ValueList<int, int, int>, decltype(Zip(a, b))>));
    EXPECT_EQ(gold, Zip(a, b));
}

TEST(ValueListTests, zip_associativity)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {1, 2, 8};
    test::ValueList<int> c = {10, 8, 3};

    EXPECT_EQ(Zip(a, b, c), Zip(Zip(a, b), c));
    EXPECT_EQ(Zip(a, b, c), Zip(a, Zip(b, c)));
}

TEST(ValueListTests, zip_error_lists_size_dont_match)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {1, 2};

    EXPECT_THROW(Zip(a, b), std::logic_error);
}

TEST(ValueListTests, conversion_from_vector_one_parameter)
{
    std::vector<int> v = {2, 3, 1};

    test::ValueList<int> gold = {2, 3, 1};

    EXPECT_EQ(gold, test::ValueList(v));
}

TEST(ValueListTests, conversion_from_vector_multiple_parameters)
{
    enum class E
    {
        A,
        B,
        C
    };

    std::vector<std::tuple<int, E>> v = {std::make_tuple(2, E::A), std::make_tuple(3, E::C), std::make_tuple(1, E::B)};

    test::ValueList<int, E> gold = {std::make_tuple(2, E::A), std::make_tuple(3, E::C), std::make_tuple(1, E::B)};

    EXPECT_EQ(gold, test::ValueList(v));
}

TEST(ValueListTests, lists_are_not_sorted)
{
    std::vector<int>     gold = {3, 1, 2, 7, 1};
    test::ValueList<int> a    = gold;

    EXPECT_THAT(a, t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_simple)
{
    std::vector<int>     gold = {1, 2, 3, 7};
    test::ValueList<int> a    = {3, 1, 2, 7, 1};

    EXPECT_THAT(UniqueSort(a), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_complex)
{
    enum class E
    {
        A,
        B,
        C
    };

    test::ValueList<int, E> v = {std::make_tuple(2, E::A), std::make_tuple(3, E::C), std::make_tuple(1, E::B)};

    test::ValueList<int, E> gold = {std::make_tuple(1, E::B), std::make_tuple(2, E::A), std::make_tuple(3, E::C)};

    EXPECT_EQ(UniqueSort(v), gold);
}

TEST(ValueListTests, unique_sort_subset_simple)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 3.5},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {9, 'c', 1.7}
    };
    std::vector<std::tuple<int, char, float>> gold = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {7, 'c', 1.7}
    };

    EXPECT_THAT(test::UniqueSort<1>(a), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_subset_multiple)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 3.5},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {9, 'c', 1.7}
    };
    std::vector<std::tuple<int, char, float>> gold = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 3.5},
        {7, 'c', 1.7}
    };

    EXPECT_THAT((test::UniqueSort<1, 2>(a)), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_with_extractor)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 2.3},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {7, 'c', 1.7}
    };

    auto extractor = [](int i, char c, float f)
    {
        return std::make_tuple(i, f);
    };

    std::vector<std::tuple<int, char, float>> gold = {
        {1, 'b', 3.1},
        {2, 'b', 3.1},
        {3, 'a', 2.3},
        {7, 'c', 1.7}
    };

    EXPECT_THAT(test::UniqueSort(extractor, a), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_subset_simple_with_extractor_simple)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 2.3},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {7, 'c', 1.7}
    };

    auto extractor = [](char c)
    {
        return c;
    };

    std::vector<std::tuple<int, char, float>> gold = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {7, 'c', 1.7}
    };

    EXPECT_THAT(test::UniqueSort<1>(extractor, a), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_subset_complex_with_extractor_simple)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 2.3},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {7, 'c', 1.7}
    };

    auto extractor = [](char c, float f)
    {
        return c;
    };

    std::vector<std::tuple<int, char, float>> gold = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {7, 'c', 1.7}
    };

    EXPECT_THAT((test::UniqueSort<1, 2>(extractor, a)), t::ElementsAreArray(gold));
}

TEST(ValueListTests, unique_sort_subset_complex_with_extractor_complex)
{
    test::ValueList<int, char, float> a = {
        {3, 'a', 2.3},
        {1, 'b', 3.1},
        {3, 'b', 2.3},
        {7, 'c', 1.7},
        {2, 'b', 3.1},
        {7, 'c', 1.8}
    };

    auto extractor = [](char c, float f)
    {
        return std::make_tuple(f, c);
    };

    std::vector<std::tuple<int, char, float>> gold = {
        {7, 'c', 1.7},
        {7, 'c', 1.8},
        {3, 'a', 2.3},
        {3, 'b', 2.3},
        {1, 'b', 3.1}
    };

    EXPECT_THAT((test::UniqueSort<1, 2>(extractor, a)), t::ElementsAreArray(gold));
}

TEST(ValueListTests, symmetric_difference_list_list_notempty)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {5, 2, 8};

    test::ValueList<int> gold = {6, 3, 2, 8};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
}

TEST(ValueListTests, symmetric_difference_list_list_disjoint)
{
    test::ValueList<int> a = {6, 3, 2};
    test::ValueList<int> b = {5, 7, 8};

    test::ValueList<int> gold = {6, 3, 2, 5, 7, 8};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
}

TEST(ValueListTests, symmetric_difference_list_list_same_different_order)
{
    test::ValueList<int> a = {6, 3, 2};
    test::ValueList<int> b = {2, 6, 3};

    test::ValueList<int> gold = {};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
}

TEST(ValueListTests, symmetric_difference_list_list_same_same_order)
{
    test::ValueList<int> a = {6, 3, 2};

    test::ValueList<int> gold = {};

    EXPECT_EQ(gold, SymmetricDifference(a, a));
}

TEST(ValueListTests, symmetric_difference_list_list_empty_input)
{
    test::ValueList<int> a = {6, 3, 2};
    test::ValueList<int> b = {};

    test::ValueList<int> gold = {6, 3, 2};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
    EXPECT_EQ(gold, SymmetricDifference(b, a));
}

TEST(ValueListTests, symmetric_difference_list_val)
{
    test::ValueList<int> a = {6, 3, 5};
    int                  b = 3;

    test::ValueList<int> gold = {6, 5};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
}

TEST(ValueListTests, symmetric_difference_val_list)
{
    int                  a = 3;
    test::ValueList<int> b = {6, 3, 5};

    test::ValueList<int> gold = {6, 5};

    EXPECT_EQ(gold, SymmetricDifference(a, b));
}

TEST(ValueListTests, extract_single_from_non_empty_simple_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(a, test::Extract<0>(a));
}

TEST(ValueListTests, extract_single_from_non_empty_homogeneous_multi_list)
{
    test::ValueList<int, int> list = {std::make_tuple(4, 2), std::make_tuple(4, 4), std::make_tuple(2, 4)};
    test::ValueList<int>      a    = {4, 4, 2};
    test::ValueList<int>      b    = {2, 4, 4};

    EXPECT_EQ(a, test::Extract<0>(list));
    EXPECT_EQ(b, test::Extract<1>(list));
}

TEST(ValueListTests, extract_single_from_non_empty_heterogeneous_multi_list)
{
    test::ValueList<int, float> list = {std::make_tuple(4, 2.1), std::make_tuple(4, 4.2), std::make_tuple(2, 4.3)};
    test::ValueList<int>        a    = {4, 4, 2};
    test::ValueList<float>      b    = {2.1, 4.2, 4.3};

    EXPECT_EQ(a, test::Extract<0>(list));
    EXPECT_EQ(b, test::Extract<1>(list));
}

TEST(ValueListTests, extract_multi_from_non_empty_homogeneous_multi_list)
{
    test::ValueList<int, int, int> list
        = {std::make_tuple(4, 2, 5), std::make_tuple(4, 4, 1), std::make_tuple(2, 4, 8)};
    test::ValueList<int, int> a = {std::make_tuple(4, 2), std::make_tuple(4, 4), std::make_tuple(2, 4)};
    test::ValueList<int, int> b = {std::make_tuple(2, 5), std::make_tuple(4, 1), std::make_tuple(4, 8)};

    EXPECT_EQ(a, (test::Extract<0, 1>(list)));
    EXPECT_EQ(b, (test::Extract<1, 2>(list)));
}

TEST(ValueListTests, extract_multi_from_non_empty_heterogeneous_multi_list)
{
    test::ValueList<int, float, int> list
        = {std::make_tuple(4, 2.1, 5), std::make_tuple(4, 4.2, 1), std::make_tuple(2, 4.3, 8)};
    test::ValueList<int, float> a = {std::make_tuple(4, 2.1), std::make_tuple(4, 4.2), std::make_tuple(2, 4.3)};
    test::ValueList<float, int> b = {std::make_tuple(2.1, 5), std::make_tuple(4.2, 1), std::make_tuple(4.3, 8)};

    EXPECT_EQ(a, (test::Extract<0, 1>(list)));
    EXPECT_EQ(b, (test::Extract<1, 2>(list)));
}

TEST(ValueListTests, extract_from_empty_list)
{
    test::ValueList<int> b;

    EXPECT_EQ(0u, test::Extract<0>(b).size());
}

TEST(ValueListTests, extract_nothing_from_list)
{
    test::ValueList<int> a = {6, 3, 5};
    std::tuple<>         nothing;

    test::ValueList<> gold = {nothing, nothing, nothing};

    EXPECT_EQ(gold, test::Extract<>(a));
}

TEST(ValueListTests, extract_nothing_from_void_list)
{
    EXPECT_EQ(test::ValueList<>(), test::Extract<>(test::ValueList<>()));
}

TEST(ValueListTests, operator_sub_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {6};

    EXPECT_EQ(Difference(a, b), a - b);
}

TEST(ValueListTests, operator_sub_list_val)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Difference(a, 6), a - 6);
}

TEST(ValueListTests, operator_sub_val_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Difference(6, a), 6 - a);
}

TEST(ValueListTests, operator_sub_functor)
{
    test::ValueList<int, int> a = {std::make_tuple(4, 2), std::make_tuple(4, 4), std::make_tuple(2, 4)};

    EXPECT_EQ(RemoveIf(test::IsSameArgs, a), a - test::IsSameArgs);
}

TEST(ValueListTests, operator_or_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {7};

    EXPECT_EQ(Concat(a, b), a | b);
}

TEST(ValueListTests, operator_or_list_val)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Concat(a, 7), a | 7);
}

TEST(ValueListTests, operator_or_val_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Concat(7, a), 7 | a);
}

TEST(ValueListTests, operator_and_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {7};

    EXPECT_EQ(Intersection(a, b), a & b);
}

TEST(ValueListTests, operator_and_list_val)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Intersection(a, 6), a & 6);
}

TEST(ValueListTests, operator_and_val_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Intersection(6, a), 6 & a);
}

TEST(ValueListTests, operator_mul_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {7};

    EXPECT_EQ(Combine(a, b), a * b);
}

TEST(ValueListTests, operator_mul_list_val)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Combine(a, 7), a * 7);
}

TEST(ValueListTests, operator_mul_val_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(Combine(7, a), 7 * a);
}

TEST(ValueListTests, operator_modulo_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {7, 2, 8};

    EXPECT_EQ(Zip(a, b), a % b);
}

TEST(ValueListTests, operator_xor_list_list)
{
    test::ValueList<int> a = {6, 3, 5};
    test::ValueList<int> b = {5, 2, 8};

    EXPECT_EQ(SymmetricDifference(a, b), a ^ b);
}

TEST(ValueListTests, operator_xor_list_val)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(SymmetricDifference(a, 3), a ^ 3);
}

TEST(ValueListTests, operator_xor_val_list)
{
    test::ValueList<int> a = {6, 3, 5};

    EXPECT_EQ(SymmetricDifference(3, a), 3 ^ a);
}

TEST(ValueListTests, make_simple_empty)
{
    test::ValueList<int> a = {};

    EXPECT_EQ(a, test::Make<int>(a));
}

TEST(ValueListTests, make_simple_identity)
{
    test::ValueList<int> a = {1, 4, 2, 6, 8, 4};

    EXPECT_EQ(a, test::Make<int>(a));
}

TEST(ValueListTests, make_struct)
{
    struct Foo
    {
        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3},
        {4, -2.4},
        {5,    8}
    };

    test::ValueList<Foo> gold = {
        Foo{1,  4.3f},
        Foo{4, -2.4f},
        Foo{5,     8}
    };

    EXPECT_EQ(gold, test::Make<Foo>(a));
}

TEST(ValueListTests, make_struct_implicit_ctor)
{
    struct Foo
    {
        Foo(int a_, float b_)
            : a(a_)
            , b(b_)
        {
        }

        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3f},
        {4, -2.4f},
        {5,     8}
    };

    test::ValueList<Foo> gold = {
        Foo{1,  4.3},
        Foo{4, -2.4},
        Foo{5,    8}
    };

    EXPECT_EQ(gold, test::Make<Foo>(a));
}

TEST(ValueListTests, make_struct_explicit_ctor)
{
    struct Foo
    {
        explicit Foo(int a_, float b_)
            : a(a_)
            , b(b_)
        {
        }

        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3f},
        {4, -2.4f},
        {5,     8}
    };

    test::ValueList<Foo> gold = {
        Foo{1,  4.3f},
        Foo{4, -2.4f},
        Foo{5,     8}
    };

    EXPECT_EQ(gold, test::Make<Foo>(a));
}

TEST(ValueListTests, make_optional_struct_implicit_ctor)
{
    struct Foo
    {
        Foo(int a_, float b_)
            : a(a_)
            , b(b_)
        {
        }

        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3f},
        {4, -2.4f},
        {5,     8}
    };

    test::ValueList<std::optional<Foo>> gold = {
        Foo{1,  4.3f},
        Foo{4, -2.4f},
        Foo{5,     8}
    };

    EXPECT_EQ(gold, test::Make<std::optional<Foo>>(a));
}

TEST(ValueListTests, make_optional_struct_explicit_ctor)
{
    struct Foo
    {
        explicit Foo(int a_, float b_)
            : a(a_)
            , b(b_)
        {
        }

        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3f},
        {4, -2.4f},
        {5,     8}
    };

    test::ValueList<std::optional<Foo>> gold = {
        Foo{1,  4.3f},
        Foo{4, -2.4f},
        Foo{5,     8}
    };

    EXPECT_EQ(gold, test::Make<std::optional<Foo>>(a));
}

TEST(ValueListTests, make_optional_struct_no_ctor)
{
    struct Foo
    {
        int   a;
        float b;

        bool operator==(const Foo &f) const
        {
            return a == f.a && b == f.b;
        };
    };

    test::ValueList<int, float> a = {
        {1,  4.3f},
        {4, -2.4f},
        {5,     8}
    };

    test::ValueList<std::optional<Foo>> gold = {
        Foo{1,  4.3f},
        Foo{4, -2.4f},
        Foo{5,     8}
    };

    EXPECT_EQ(gold, test::Make<std::optional<Foo>>(a));
}

TEST(ValueListTests, not_works)
{
    EXPECT_TRUE(test::Not([](int a, int b) { return a == b; })(3, 5));
    EXPECT_FALSE(test::Not([](int a, int b) { return a == b; })(5, 5));
}

TEST(ValueListTests, and_works)
{
    EXPECT_TRUE(test::And([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(4, 3));
    EXPECT_FALSE(test::And([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(5, 3));
    EXPECT_FALSE(test::And([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(4, 6));
    EXPECT_FALSE(test::And([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(5, 6));
}

TEST(ValueListTests, or_works)
{
    EXPECT_TRUE(test::Or([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(4, 3));
    EXPECT_TRUE(test::Or([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(5, 3));
    EXPECT_TRUE(test::Or([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(4, 6));
    EXPECT_FALSE(test::Or([](int a, int b) { return a == 4; }, [](int a, int b) { return b == 3; })(5, 6));
}

TEST(ValueListTests, implicit_conversion_different_types_multiple)
{
    struct Foo
    {
        Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo, char> list{test::ValueList<int, char>{{5, 'c'}}};

    EXPECT_EQ(5, std::get<0>(*list.begin()).value);
    EXPECT_EQ('c', std::get<1>(*list.begin()));
}

TEST(ValueListTests, explicit_conversion_different_types_multiple)
{
    struct Foo
    {
        explicit Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo, char> list{test::ValueList<int, char>{{5, 'c'}}};

    EXPECT_EQ(5, std::get<0>(*list.begin()).value);
    EXPECT_EQ('c', std::get<1>(*list.begin()));
}

TEST(ValueListTests, implicit_conversion_different_types_single)
{
    struct Foo
    {
        Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo> list{test::ValueList<int>{{5}}};

    EXPECT_EQ(5, list.begin()->value);
}

TEST(ValueListTests, default_ctor_implicit_conversion_different_types_single)
{
    struct Foo
    {
        Foo()
            : value(123)
        {
        }

        Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo> list{test::ValueList<int>{{5}}};

    EXPECT_EQ(5, list.begin()->value);
}

TEST(ValueListTests, explicit_conversion_different_types_single)
{
    struct Foo
    {
        explicit Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo> list{test::ValueList<int>{{5}}};

    EXPECT_EQ(5, list.begin()->value);
}

TEST(ValueListTests, default_ctor_explicit_conversion_different_types_single)
{
    struct Foo
    {
        Foo()
            : value(123)
        {
        }

        explicit Foo(int value_)
            : value(value_)
        {
        }

        int value;
    };

    test::ValueList<Foo> list{test::ValueList<int>{{5}}};

    EXPECT_EQ(5, list.begin()->value);
}

TEST(ValueListTests, create_with_default_parameters)
{
    struct Foo
    {
        Foo()
            : value(123)
        {
        }

        int value;
    };

    test::ValueList<Foo> list{test::ValueDefault()};

    ASSERT_EQ(1, list.size());

    EXPECT_EQ(123, list.begin()->value);
}

TEST(ValueListTests, create_with_default_parameters_mixed)
{
    struct Foo
    {
        Foo()
            : value(123)
        {
        }

        int value;
    };

    test::ValueList<int, Foo, char> list{5 * test::ValueDefault() * 'r'};

    ASSERT_EQ(1, list.size());

    EXPECT_EQ(5, std::get<0>(*list.begin()));
    EXPECT_EQ(123, std::get<1>(*list.begin()).value);
    EXPECT_EQ('r', std::get<2>(*list.begin()));
}

TEST(ValueListTests, dup_zero)
{
    EXPECT_EQ((test::ValueList<>{std::tuple<>(), std::tuple<>(), std::tuple<>()}),
              Dup<0>(test::ValueList<int>{1, 2, 3}));
}

TEST(ValueListTests, dup_one)
{
    EXPECT_EQ((test::ValueList<int>{1, 2, 3}), Dup<1>(test::ValueList<int>{1, 2, 3}));
}

TEST(ValueListTests, dup_many)
{
    EXPECT_EQ((test::ValueList<int, int>{
                  {1, 1},
                  {2, 2},
                  {3, 3}
    }),
              Dup<2>(test::ValueList<int>{1, 2, 3}));
}
