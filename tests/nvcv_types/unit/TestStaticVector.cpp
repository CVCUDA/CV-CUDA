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

#include <util/StaticVector.hpp>

namespace util = nvcv::util;

TEST(StaticVector, default_constructed_is_empty)
{
    util::StaticVector<short, 5> v;
    EXPECT_TRUE(v.empty());
}

TEST(StaticVector, destructor_destroys_items)
{
    std::weak_ptr<short> w[2];

    {
        util::StaticVector<std::shared_ptr<short>, 2> v = {std::make_shared<short>(2), std::make_shared<short>(4)};
        w[0]                                            = v[0];
        w[1]                                            = v[1];
    }

    EXPECT_TRUE(w[0].expired());
    EXPECT_TRUE(w[1].expired());
}

TEST(StaticVector, capacity_is_correct)
{
    util::StaticVector<short, 6> v;
    EXPECT_EQ(6, v.capacity());
}

TEST(StaticVector, value_type_is_correct)
{
    EXPECT_TRUE((std::is_same_v<short, util::StaticVector<short, 3>::value_type>));
}

TEST(StaticVector, empty_has_size_zero)
{
    util::StaticVector<short, 5> v;
    EXPECT_EQ(0u, v.size());
}

TEST(StaticVector, ctor_with_size_has_correct_size)
{
    util::StaticVector<short, 4> v(3);

    EXPECT_EQ(3u, v.size());
}

TEST(StaticVector, ctor_with_size_is_default_initialized)
{
    util::StaticVector<std::shared_ptr<short>, 4> v(3);

    ASSERT_EQ(3u, v.size());
    EXPECT_FALSE(v[0]);
    EXPECT_FALSE(v[1]);
    EXPECT_FALSE(v[2]);
}

TEST(StaticVector, ctor_with_size_and_value)
{
    util::StaticVector<short, 3> v(3, 5);

    ASSERT_EQ(3u, v.size());
    EXPECT_EQ(5, v[0]);
    EXPECT_EQ(5, v[1]);
    EXPECT_EQ(5, v[2]);
}

TEST(StaticVector, resize_to_big_throws_bad_alloc)
{
    util::StaticVector<std::shared_ptr<short>, 2> v;

    ASSERT_THROW(v.resize(3), std::bad_alloc);
}

TEST(StaticVector, resize_to_smaller_destroy_objects)
{
    util::StaticVector<std::shared_ptr<short>, 2> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    std::weak_ptr<short> w = v[1];

    ASSERT_NO_THROW(v.resize(1));

    EXPECT_TRUE(w.expired());
}

TEST(StaticVector, resize_to_smaller_doesnt_touch_non_destroyed_objects)
{
    util::StaticVector<std::shared_ptr<short>, 2> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    std::weak_ptr<short> w = v[0];

    ASSERT_NO_THROW(v.resize(1));

    ASSERT_FALSE(w.expired());
    EXPECT_EQ(0, *v[0]);
}

TEST(StaticVector, resize_to_smaller_updates_size)
{
    util::StaticVector<std::shared_ptr<short>, 2> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    ASSERT_NO_THROW(v.resize(1));

    EXPECT_EQ(1u, v.size());
}

TEST(StaticVector, resize_to_bigger_default_construct_new_objects)
{
    util::StaticVector<std::shared_ptr<short>, 3> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    ASSERT_NO_THROW(v.resize(3));

    ASSERT_EQ(3, v.size());
    EXPECT_FALSE(v[2]);
}

TEST(StaticVector, resize_to_bigger_doesnt_touch_existing_objects)
{
    util::StaticVector<std::shared_ptr<short>, 3> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    std::weak_ptr<short> w[] = {v[0], v[1]};

    ASSERT_NO_THROW(v.resize(3));

    EXPECT_FALSE(w[0].expired());
    EXPECT_FALSE(w[1].expired());
    EXPECT_EQ(0, *v[0]);
    EXPECT_EQ(1, *v[1]);
}

TEST(StaticVector, resize_to_bigger_updates_size)
{
    util::StaticVector<std::shared_ptr<short>, 3> v = {std::make_shared<short>(0), std::make_shared<short>(1)};

    ASSERT_NO_THROW(v.resize(3));

    EXPECT_EQ(3u, v.size());
}

TEST(StaticVector, ctor_with_too_large_size_throws_bad_alloc)
{
    EXPECT_THROW(({ util::StaticVector<short, 4> v(5); }), std::bad_alloc);
}

TEST(StaticVector, populated_vector_isnt_empty)
{
    util::StaticVector<short, 5> v = {3};
    EXPECT_FALSE(v.empty());
}

TEST(StaticVector, ctor_with_empty_initializer_list)
{
    util::StaticVector<short, 5> v = {};

    EXPECT_TRUE(v.empty());
}

TEST(StaticVector, ctor_with_nonempty_initializer_list_has_correct_size)
{
    util::StaticVector<short, 5> v = {4, 1, 3};
    EXPECT_EQ(3u, v.size());
}

TEST(StaticVector, ctor_with_too_big_initializer_list_throws_exception)
{
    EXPECT_THROW(({ util::StaticVector<short, 5> v = {4, 1, 3, 3, 4, 2}; }), std::bad_alloc);
}

TEST(StaticVector, assign_with_empty_initializer_list)
{
    util::StaticVector<short, 5> v = {2, 3};

    v = {};

    EXPECT_TRUE(v.empty());
}

TEST(StaticVector, assign_with_nonempty_initializer_list_has_correct_size)
{
    util::StaticVector<short, 5> v = {4, 1, 3};

    v = {2, 3};
    EXPECT_EQ(2u, v.size());
}

TEST(StaticVector, assign_with_too_big_initializer_list_throws_exception)
{
    util::StaticVector<short, 5> v = {4, 4, 2};

    EXPECT_THROW((v = {1, 2, 3, 4, 5, 6}), std::bad_alloc);
}

TEST(StaticVector, push_back_increment_size)
{
    util::StaticVector<short, 5> v = {2, 3, 4};

    ASSERT_NO_THROW(v.push_back(10));
    EXPECT_EQ(4u, v.size());
}

TEST(StaticVector, pop_back_decrement_size)
{
    util::StaticVector<short, 5> v = {2, 3, 4};

    ASSERT_NO_THROW(v.pop_back());
    EXPECT_EQ(2u, v.size());
}

TEST(StaticVector, pop_back_destroys_removed_item)
{
    util::StaticVector<std::shared_ptr<short>, 2> v = {std::make_shared<short>(2), std::make_shared<short>(3)};

    std::weak_ptr<short> w[] = {v[0], v[1]};

    ASSERT_NO_THROW(v.pop_back());

    EXPECT_FALSE(w[0].expired());
    EXPECT_TRUE(w[1].expired());
}

TEST(StaticVector, push_back_when_its_full_throws_exception)
{
    util::StaticVector<short, 5> v = {2, 3, 4, 4, 2};

    ASSERT_THROW(v.push_back(1), std::bad_alloc);
}

TEST(StaticVector, const_array_access_on_nonempty_works)
{
    const util::StaticVector<short, 5> v = {2, 3, 4, 4, 2};
    EXPECT_EQ(3, v[1]);
}

TEST(StaticVector, array_access_on_nonempty_works)
{
    util::StaticVector<short, 5> v = {2, 3, 4, 4, 2};
    EXPECT_EQ(3, v[1]);
}

TEST(StaticVector, move_ctor_src_isnt_emptied)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};

    [[maybe_unused]] util::StaticVector<std::shared_ptr<short>, 2> dst(std::move(src));

    EXPECT_EQ(1u, src.size());
}

TEST(StaticVector, move_ctor_dst_has_same_size_as_src_before_move)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst(std::move(src));

    EXPECT_EQ(1u, dst.size());
}

TEST(StaticVector, move_ctor_contents_are_moved)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst(std::move(src));

    ASSERT_FALSE(dst.empty());
    ASSERT_NE(nullptr, dst[0]);
    EXPECT_EQ(2, *dst[0]);
}

TEST(StaticVector, move_assign_src_isnt_emptied)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;

    dst = std::move(src);

    EXPECT_EQ(1u, src.size());
}

TEST(StaticVector, move_assign_dst_has_same_size_as_src_before_move)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;

    dst = std::move(src);

    EXPECT_EQ(1u, dst.size());
}

TEST(StaticVector, move_assign_contents_are_moved)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;

    dst = std::move(src);

    ASSERT_FALSE(dst.empty());
    ASSERT_NE(nullptr, dst[0]);
    EXPECT_EQ(2, *dst[0]);
}

TEST(StaticVector, copy_ctor_not_available_when_elements_arent_copiable)
{
    EXPECT_FALSE((std::is_copy_constructible_v<util::StaticVector<std::unique_ptr<short>, 3>>));
}

TEST(StaticVector, copy_assign_not_available_when_elements_arent_copiable)
{
    EXPECT_FALSE((std::is_copy_assignable_v<util::StaticVector<std::unique_ptr<short>, 3>>));
}

TEST(StaticVector, copy_ctor_available_when_elements_are_copiable)
{
    EXPECT_TRUE((std::is_copy_constructible_v<util::StaticVector<std::shared_ptr<short>, 3>>));
}

TEST(StaticVector, copy_assign_available_when_elements_are_copiable)
{
    EXPECT_TRUE((std::is_copy_assignable_v<util::StaticVector<std::shared_ptr<short>, 3>>));
}

TEST(StaticVector, move_ctor_available_when_elements_arent_copiable)
{
    EXPECT_TRUE((std::is_move_constructible_v<util::StaticVector<std::unique_ptr<short>, 3>>));
}

TEST(StaticVector, move_assign_available_when_elements_arent_copiable)
{
    EXPECT_TRUE((std::is_move_assignable_v<util::StaticVector<std::unique_ptr<short>, 3>>));
}

TEST(StaticVector, move_ctor_available_when_elements_are_copiable)
{
    EXPECT_TRUE((std::is_move_constructible_v<util::StaticVector<std::shared_ptr<short>, 3>>));
}

TEST(StaticVector, move_assign_available_when_elements_are_copiable)
{
    EXPECT_TRUE((std::is_move_assignable_v<util::StaticVector<std::shared_ptr<short>, 3>>));
}

TEST(StaticVector, copy_ctor_src_size_is_unchanged)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst(src);

    EXPECT_EQ(1u, src.size());
}

TEST(StaticVector, copy_ctor_dst_and_src_have_same_size)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst(src);

    EXPECT_EQ(src.size(), dst.size());
}

TEST(StaticVector, copy_ctor_contents_are_copied)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst(src);

    ASSERT_FALSE(dst.empty());
    ASSERT_FALSE(src.empty());
    EXPECT_EQ(src[0], dst[0]);
}

TEST(StaticVector, copy_assign_src_size_is_unchanged)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;
    dst = src;

    EXPECT_EQ(1u, src.size());
}

TEST(StaticVector, copy_assign_dst_and_src_have_same_size)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;

    dst = src;

    EXPECT_EQ(src.size(), dst.size());
}

TEST(StaticVector, copy_assign_contents_are_copied)
{
    util::StaticVector<std::shared_ptr<short>, 2> src = {std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst;

    dst = src;

    ASSERT_FALSE(dst.empty());
    ASSERT_FALSE(src.empty());
    EXPECT_EQ(src[0], dst[0]);
}

TEST(StaticVector, begin_and_end_are_equal_on_empty_static_vector)
{
    util::StaticVector<short, 4> v;

    EXPECT_EQ(v.begin(), v.end());
}

TEST(StaticVector, begin_and_end_range_maps_contained_values)
{
    util::StaticVector<short, 4> v = {2, 5, 1, 3};

    ASSERT_EQ(v.size(), std::distance(v.begin(), v.end()));
    EXPECT_EQ(2, v.begin()[0]);
    EXPECT_EQ(5, v.begin()[1]);
    EXPECT_EQ(1, v.begin()[2]);
    EXPECT_EQ(3, v.begin()[3]);

    EXPECT_EQ(2, v.end()[-4]);
    EXPECT_EQ(5, v.end()[-3]);
    EXPECT_EQ(1, v.end()[-2]);
    EXPECT_EQ(3, v.end()[-1]);
}

TEST(StaticVector, const_begin_and_end_are_equal_on_empty_static_vector)
{
    const util::StaticVector<short, 4> v;

    EXPECT_EQ(v.begin(), v.end());
}

TEST(StaticVector, const_begin_and_end_range_maps_contained_values)
{
    const util::StaticVector<short, 4> v = {2, 5, 1, 3};

    ASSERT_EQ(v.size(), std::distance(v.begin(), v.end()));
    EXPECT_EQ(2, v.begin()[0]);
    EXPECT_EQ(5, v.begin()[1]);
    EXPECT_EQ(1, v.begin()[2]);
    EXPECT_EQ(3, v.begin()[3]);

    EXPECT_EQ(2, v.end()[-4]);
    EXPECT_EQ(5, v.end()[-3]);
    EXPECT_EQ(1, v.end()[-2]);
    EXPECT_EQ(3, v.end()[-1]);
}

TEST(StaticVector, cbegin_and_mutable_begin_are_the_same)
{
    util::StaticVector<short, 4> v = {3, 1, 2};

    EXPECT_EQ(v.cbegin(), v.begin());
}

TEST(StaticVector, cbegin_and_const_begin_are_the_same)
{
    const util::StaticVector<short, 4> v = {3, 1, 2};

    EXPECT_EQ(v.cbegin(), v.begin());
}

TEST(StaticVector, iterator_converts_to_const_iterator)
{
    using Vector = util::StaticVector<short, 4>;
    Vector v     = {3, 1, 2};

    EXPECT_TRUE((std::is_convertible_v<Vector::iterator, Vector::const_iterator>));
}

TEST(StaticVector, const_iterator_doesnt_convert_to_iterator)
{
    using Vector = util::StaticVector<short, 4>;
    Vector v     = {3, 1, 2};

    EXPECT_FALSE((std::is_convertible_v<Vector::const_iterator, Vector::iterator>));
}

TEST(StaticVector, front_is_the_first_element)
{
    const util::StaticVector<short, 4> v = {3, 1, 2};

    EXPECT_EQ(3, v.front());
}

TEST(StaticVector, back_is_the_last_element)
{
    const util::StaticVector<short, 4> v = {3, 1, 2};

    EXPECT_EQ(2, v.back());
}

TEST(StaticVector, clear_makes_vector_empty)
{
    util::StaticVector<short, 4> v = {3, 1, 2};

    ASSERT_NO_THROW(v.clear());
    EXPECT_TRUE(v.empty());
}

TEST(StaticVector, clear_properly_deletes_objects)
{
    util::StaticVector<std::shared_ptr<short>, 1> v = {std::make_shared<short>(3)};

    std::weak_ptr<short> w = v[0];

    ASSERT_NO_THROW(v.clear());

    EXPECT_TRUE(w.expired());
}

TEST(StaticVector, move_assignment_src_is_larger_than_src_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0), std::make_shared<short>(1)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc[2] = {src[0], src[1]};
    std::weak_ptr<short> wdst    = {dst[0]};

    dst = std::move(src);

    EXPECT_FALSE(wsrc[0].expired());
    EXPECT_FALSE(wsrc[1].expired());
    EXPECT_TRUE(wdst.expired());
}

TEST(StaticVector, move_assignment_src_is_smaller_than_src_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(1), std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc    = {src[0]};
    std::weak_ptr<short> wdst[2] = {dst[0], dst[1]};

    dst = std::move(src);

    EXPECT_FALSE(wsrc.expired());
    EXPECT_TRUE(wdst[0].expired());
    EXPECT_TRUE(wdst[1].expired());
}

TEST(StaticVector, move_assignment_src_has_same_element_count_as_dst_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0), std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(1), std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc[2] = {src[0], src[1]};
    std::weak_ptr<short> wdst[2] = {dst[0], dst[1]};

    dst = std::move(src);

    EXPECT_FALSE(wsrc[0].expired());
    EXPECT_FALSE(wsrc[1].expired());
    EXPECT_TRUE(wdst[0].expired());
    EXPECT_TRUE(wdst[1].expired());
}

TEST(StaticVector, copy_assignment_src_is_larger_than_src_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0), std::make_shared<short>(1)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc[2] = {src[0], src[1]};
    std::weak_ptr<short> wdst    = {dst[0]};

    dst = src;

    EXPECT_FALSE(wsrc[0].expired());
    EXPECT_FALSE(wsrc[1].expired());
    EXPECT_TRUE(wdst.expired());
}

TEST(StaticVector, copy_assignment_src_is_smaller_than_src_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(1), std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc    = {src[0]};
    std::weak_ptr<short> wdst[2] = {dst[0], dst[1]};

    dst = src;

    EXPECT_FALSE(wsrc.expired());
    EXPECT_TRUE(wdst[0].expired());
    EXPECT_TRUE(wdst[1].expired());
}

TEST(StaticVector, copy_assignment_src_has_same_element_count_as_src_items_lifetime_are_handled_correctly)
{
    util::StaticVector<std::shared_ptr<short>, 2> src{std::make_shared<short>(0), std::make_shared<short>(2)};
    util::StaticVector<std::shared_ptr<short>, 2> dst{std::make_shared<short>(1), std::make_shared<short>(3)};

    std::weak_ptr<short> wsrc[2] = {src[0], src[1]};
    std::weak_ptr<short> wdst[2] = {dst[0], dst[1]};

    dst = src;

    EXPECT_FALSE(wsrc[0].expired());
    EXPECT_FALSE(wsrc[1].expired());
    EXPECT_TRUE(wdst[0].expired());
    EXPECT_TRUE(wdst[1].expired());
}

TEST(StaticVector, swap_non_trivial_type_a_is_larger_than_b_dont_destroy_objects)
{
    util::StaticVector<std::shared_ptr<short>, 2> a{std::make_shared<short>(0), std::make_shared<short>(1)};
    util::StaticVector<std::shared_ptr<short>, 2> b{std::make_shared<short>(3)};

    std::weak_ptr<short> w[] = {a[0], a[1], b[0]};

    swap(a, b);

    EXPECT_FALSE(w[0].expired());
    EXPECT_FALSE(w[1].expired());
    EXPECT_FALSE(w[2].expired());
}

TEST(StaticVector, swap_non_trivial_type_a_is_smaller_than_b_dont_destroy_objects)
{
    util::StaticVector<std::shared_ptr<short>, 2> a{std::make_shared<short>(3)};
    util::StaticVector<std::shared_ptr<short>, 2> b{std::make_shared<short>(0), std::make_shared<short>(1)};

    std::weak_ptr<short> w[] = {a[0], b[0], b[1]};

    swap(a, b);

    EXPECT_FALSE(w[0].expired());
    EXPECT_FALSE(w[1].expired());
    EXPECT_FALSE(w[2].expired());
}

TEST(StaticVector, swap_non_trivial_type_a_has_same_element_count_as_b_dont_destroy_objects)
{
    util::StaticVector<std::shared_ptr<short>, 2> a{std::make_shared<short>(3), std::make_shared<short>(4)};
    util::StaticVector<std::shared_ptr<short>, 2> b{std::make_shared<short>(0), std::make_shared<short>(1)};

    std::weak_ptr<short> w[] = {a[0], a[1], b[0], b[1]};

    swap(a, b);

    EXPECT_FALSE(w[0].expired());
    EXPECT_FALSE(w[1].expired());
    EXPECT_FALSE(w[2].expired());
    EXPECT_FALSE(w[2].expired());
}

TEST(StaticVector, swap_trivial_type_a_is_larger_than_b_dont_destroy_objects)
{
    util::StaticVector<short, 2> a{0, 1};
    util::StaticVector<short, 2> b{2};

    swap(a, b);

    ASSERT_EQ(1u, a.size());
    ASSERT_EQ(2u, b.size());

    EXPECT_EQ(2, a[0]);
    EXPECT_EQ(0, b[0]);
    EXPECT_EQ(1, b[1]);
}

TEST(StaticVector, swap_trivial_type_a_is_smaller_than_b_dont_destroy_objects)
{
    util::StaticVector<short, 2> a{0};
    util::StaticVector<short, 2> b{1, 2};

    swap(a, b);

    ASSERT_EQ(2u, a.size());
    ASSERT_EQ(1u, b.size());

    EXPECT_EQ(1, a[0]);
    EXPECT_EQ(2, a[1]);
    EXPECT_EQ(0, b[0]);
}

TEST(StaticVector, swap_trivial_type_a_has_same_element_count_as_b_dont_destroy_objects)
{
    util::StaticVector<short, 2> a{0, 1};
    util::StaticVector<short, 2> b{2, 3};

    swap(a, b);

    ASSERT_EQ(2u, a.size());
    ASSERT_EQ(2u, b.size());

    EXPECT_EQ(2, a[0]);
    EXPECT_EQ(3, a[1]);
    EXPECT_EQ(0, b[0]);
    EXPECT_EQ(1, b[1]);
}

TEST(StaticVector, copy_ctor_trivial_type_does_copy)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst(src);

    ASSERT_EQ(2u, src.size());
    ASSERT_EQ(2u, dst.size());

    EXPECT_EQ(src[0], dst[0]);
    EXPECT_EQ(src[1], dst[1]);
}

TEST(StaticVector, copy_assign_trivial_type_does_copy)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst;

    dst = src;

    ASSERT_EQ(2u, src.size());
    ASSERT_EQ(2u, dst.size());

    EXPECT_EQ(src[0], dst[0]);
    EXPECT_EQ(src[1], dst[1]);
}

TEST(StaticVector, move_ctor_trivial_type_doesnt_reset_src)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst(std::move(src));

    EXPECT_EQ(2u, src.size());
    EXPECT_EQ(2u, dst.size());
}

TEST(StaticVector, move_assign_trivial_type_doesnt_reset_src)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst;

    dst = std::move(src);

    EXPECT_EQ(2u, src.size());
    EXPECT_EQ(2u, dst.size());
}

TEST(StaticVector, move_ctor_trivial_type_copy_items)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst(std::move(src));

    ASSERT_EQ(2u, dst.size());
    EXPECT_EQ(1, dst[0]);
    EXPECT_EQ(2, dst[1]);
}

TEST(StaticVector, move_assign_trivial_type_copy_items)
{
    util::StaticVector<short, 2> src{1, 2};
    util::StaticVector<short, 2> dst;

    dst = std::move(src);

    ASSERT_EQ(2u, dst.size());
    EXPECT_EQ(1, dst[0]);
    EXPECT_EQ(2, dst[1]);
}

TEST(StaticVector, cant_copy_ctor_from_different_type)
{
    EXPECT_FALSE((std::is_constructible_v<util::StaticVector<short, 4>, const util::StaticVector<char, 4> &>));
}

TEST(StaticVector, cant_move_ctor_from_different_type)
{
    EXPECT_FALSE((std::is_constructible_v<util::StaticVector<short, 4>, util::StaticVector<char, 4> &&>));
}

TEST(StaticVector, cant_copy_assign_from_different_type)
{
    EXPECT_FALSE((std::is_assignable_v<util::StaticVector<short, 4>, const util::StaticVector<char, 4> &>));
}

TEST(StaticVector, cant_move_assign_from_different_type)
{
    EXPECT_FALSE((std::is_assignable_v<util::StaticVector<short, 4>, util::StaticVector<char, 4> &&>));
}

TEST(StaticVector, cant_copy_ctor_from_different_capacity)
{
    EXPECT_FALSE((std::is_constructible_v<util::StaticVector<short, 4>, const util::StaticVector<short, 2> &>));
}

TEST(StaticVector, cant_move_ctor_from_different_capacity)
{
    EXPECT_FALSE((std::is_constructible_v<util::StaticVector<short, 4>, util::StaticVector<short, 2> &&>));
}

TEST(StaticVector, cant_copy_assign_from_different_capacity)
{
    EXPECT_FALSE((std::is_assignable_v<util::StaticVector<short, 4>, const util::StaticVector<short, 2> &>));
}

TEST(StaticVector, cant_move_assign_from_different_capacity)
{
    EXPECT_FALSE((std::is_assignable_v<util::StaticVector<short, 4>, util::StaticVector<short, 2> &&>));
}

TEST(StaticVector, erase_single_element_at_beginning)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.begin());

    EXPECT_EQ(2, vec.size());

    EXPECT_TRUE(items[0].expired());
    EXPECT_FALSE(items[1].expired());
    EXPECT_FALSE(items[2].expired());

    EXPECT_EQ(2, *vec[0]);
    EXPECT_EQ(3, *vec[1]);
}

TEST(StaticVector, erase_single_element_in_the_middle)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.begin() + 1);

    EXPECT_EQ(2, vec.size());

    EXPECT_FALSE(items[0].expired());
    EXPECT_TRUE(items[1].expired());
    EXPECT_FALSE(items[2].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(3, *vec[1]);
}

TEST(StaticVector, erase_single_element_at_the_end)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.begin() + 2);

    EXPECT_EQ(2, vec.size());

    EXPECT_FALSE(items[0].expired());
    EXPECT_FALSE(items[1].expired());
    EXPECT_TRUE(items[2].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(2, *vec[1]);
}

TEST(StaticVector, erase_single_element_past_end_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.end()), ".*");
}

TEST(StaticVector, erase_single_element_before_beginning_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.begin() - 1), ".*");
}

TEST(StaticVector, erase_range_begin_before_start_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.begin() - 1, vec.end()), ".*");
}

TEST(StaticVector, erase_range_begin_after_end_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.end() + 1, vec.end() + 1), ".*");
}

TEST(StaticVector, erase_empty_range_noop)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.begin(), vec.begin());

    EXPECT_FALSE(items[0].expired());
    EXPECT_FALSE(items[1].expired());
    EXPECT_FALSE(items[2].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(2, *vec[1]);
    EXPECT_EQ(3, *vec[2]);
}

TEST(StaticVector, erase_empty_range_at_end_noop)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.end(), vec.end());

    EXPECT_FALSE(items[0].expired());
    EXPECT_FALSE(items[1].expired());
    EXPECT_FALSE(items[2].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(2, *vec[1]);
    EXPECT_EQ(3, *vec[2]);
}

TEST(StaticVector, erase_empty_range_at_end_returns_end)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    EXPECT_EQ(vec.end(), vec.erase(vec.end(), vec.end()));
}

TEST(StaticVector, erase_empty_range_returns_begin_range)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    EXPECT_EQ(vec.begin(), vec.erase(vec.begin(), vec.begin()));
}

TEST(StaticVector, erase_range_end_before_begin_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.begin() + 2, vec.begin()), ".*");
}

TEST(StaticVector, erase_range_end_after_vector_end_segfaults)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    ASSERT_DEATH(vec.erase(vec.begin() + 2, vec.end() + 1), ".*");
}

TEST(StaticVector, erase_all_elements_empties_container)
{
    util::StaticVector<std::shared_ptr<int>, 3> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    vec.erase(vec.begin(), vec.end());

    EXPECT_EQ(0, vec.size());

    EXPECT_TRUE(items[0].expired());
    EXPECT_TRUE(items[1].expired());
    EXPECT_TRUE(items[2].expired());
}

TEST(StaticVector, erase_range_at_beginning_with_bigger_range_remaining)
{
    util::StaticVector<std::shared_ptr<int>, 5> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3), std::make_shared<int>(4),
                                                    std::make_shared<int>(5)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2], vec[3], vec[4]};

    vec.erase(vec.begin(), vec.begin() + 2);

    EXPECT_EQ(3, vec.size());

    EXPECT_TRUE(items[0].expired());
    EXPECT_TRUE(items[1].expired());
    EXPECT_FALSE(items[2].expired());
    EXPECT_FALSE(items[3].expired());
    EXPECT_FALSE(items[4].expired());

    EXPECT_EQ(3, *vec[0]);
    EXPECT_EQ(4, *vec[1]);
    EXPECT_EQ(5, *vec[2]);
}

TEST(StaticVector, erase_range_at_beginning_with_smaller_range_remaining)
{
    util::StaticVector<std::shared_ptr<int>, 5> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3), std::make_shared<int>(4),
                                                    std::make_shared<int>(5)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2], vec[3], vec[4]};

    vec.erase(vec.begin(), vec.begin() + 3);

    EXPECT_EQ(2, vec.size());

    EXPECT_TRUE(items[0].expired());
    EXPECT_TRUE(items[1].expired());
    EXPECT_TRUE(items[2].expired());
    EXPECT_FALSE(items[3].expired());
    EXPECT_FALSE(items[4].expired());

    EXPECT_EQ(4, *vec[0]);
    EXPECT_EQ(5, *vec[1]);
}

TEST(StaticVector, erase_range_at_end)
{
    util::StaticVector<std::shared_ptr<int>, 5> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3), std::make_shared<int>(4),
                                                    std::make_shared<int>(5)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2], vec[3], vec[4]};

    vec.erase(vec.end() - 2, vec.end());

    EXPECT_EQ(3, vec.size());

    EXPECT_FALSE(items[0].expired());
    EXPECT_FALSE(items[1].expired());
    EXPECT_FALSE(items[2].expired());
    EXPECT_TRUE(items[3].expired());
    EXPECT_TRUE(items[4].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(2, *vec[1]);
    EXPECT_EQ(3, *vec[2]);
}

TEST(StaticVector, erase_range_in_the_middle)
{
    util::StaticVector<std::shared_ptr<int>, 5> vec{std::make_shared<int>(1), std::make_shared<int>(2),
                                                    std::make_shared<int>(3), std::make_shared<int>(4),
                                                    std::make_shared<int>(5)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2], vec[3], vec[4]};

    vec.erase(vec.begin() + 1, vec.begin() + 3);

    EXPECT_EQ(3, vec.size());

    EXPECT_FALSE(items[0].expired());
    EXPECT_TRUE(items[1].expired());
    EXPECT_TRUE(items[2].expired());
    EXPECT_FALSE(items[3].expired());
    EXPECT_FALSE(items[4].expired());

    EXPECT_EQ(1, *vec[0]);
    EXPECT_EQ(4, *vec[1]);
    EXPECT_EQ(5, *vec[2]);
}

TEST(StaticVector, erase_in_middle_return_iterator_to_next_element)
{
    using Vector = util::StaticVector<std::shared_ptr<int>, 5>;

    Vector vec{std::make_shared<int>(1), std::make_shared<int>(2), std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    Vector::iterator it = vec.erase(vec.begin(), vec.begin() + 2);

    EXPECT_EQ(3, **it);
    EXPECT_EQ(vec.begin(), it);
}

TEST(StaticVector, erase_at_end_return_end_iterator)
{
    using Vector = util::StaticVector<std::shared_ptr<int>, 5>;

    Vector vec{std::make_shared<int>(1), std::make_shared<int>(2), std::make_shared<int>(3)};

    std::weak_ptr<int> items[] = {vec[0], vec[1], vec[2]};

    Vector::iterator it = vec.erase(vec.begin() + 2, vec.begin() + 3);

    EXPECT_EQ(vec.end(), it);
}

TEST(StaticVector, rbegin_and_rend_are_equal_on_empty_container)
{
    util::StaticVector<int, 5> v;
    EXPECT_EQ(v.rbegin(), v.rend());
}

TEST(StaticVector, rbegin_point_to_last_element)
{
    util::StaticVector<int, 3> v = {1, 2, 3};
    EXPECT_EQ(3, *v.rbegin());
}

TEST(StaticVector, rend_point_to_previous_to_first_element)
{
    util::StaticVector<int, 3> v = {1, 2, 3};
    EXPECT_EQ(1, *std::prev(v.rend()));
}

TEST(StaticVector, reverse_iteration_works)
{
    using Vector = util::StaticVector<int, 5>;
    Vector v     = {1, 2, 3, 4, 5};

    Vector::reverse_iterator it = v.rbegin();

    EXPECT_EQ(5, it[0]);
    EXPECT_EQ(4, it[1]);
    EXPECT_EQ(3, it[2]);
    EXPECT_EQ(2, it[3]);
    EXPECT_EQ(1, it[4]);
}

template<class FROM, class TO, class EN = void>
struct IsAssignableImpl : std::false_type
{
};

template<class FROM, class TO>
struct IsAssignableImpl<FROM, TO, decltype(std::declval<TO>() = std::declval<FROM>())> : std::true_type
{
};

template<class FROM, class TO>
constexpr bool IsAssignable = IsAssignableImpl<FROM, TO>::value;

TEST(StaticVector, reverse_iterator_implicitly_converts_to_const_reverse_iterator)
{
    using Vector = util::StaticVector<int, 5>;

    EXPECT_FALSE((IsAssignable<Vector::reverse_iterator, Vector::const_reverse_iterator>));
}

TEST(StaticVector, const_reverse_iterator_doesnt_convert_to_reverse_iterator)
{
    using Vector = util::StaticVector<int, 5>;

    // is_convertible isn't working for some reason...
    EXPECT_FALSE((IsAssignable<Vector::const_reverse_iterator, Vector::reverse_iterator>));
}

struct NonDefaultConstructible
{
    NonDefaultConstructible() = delete;

    NonDefaultConstructible(int d)
        : dummy(d)
    {
    }

    int dummy;
};

TEST(StaticVector, cant_increase_size_of_non_default_constructible_type_vector)
{
    util::StaticVector<NonDefaultConstructible, 5> v;
    ASSERT_THROW(v.resize(1), std::runtime_error);
    EXPECT_EQ(0u, v.size());
}

TEST(StaticVector, can_decrease_size_of_non_default_constructible_type_vector)
{
    util::StaticVector<NonDefaultConstructible, 5> v;
    ASSERT_NO_THROW(v.push_back(1));
    ASSERT_EQ(1u, v.size());

    EXPECT_NO_THROW(v.resize(0));
    EXPECT_EQ(0u, v.size());
}

TEST(StaticVector, can_add_non_default_constructible_type)
{
    util::StaticVector<NonDefaultConstructible, 5> v;
    ASSERT_NO_THROW(v.push_back(1));
    EXPECT_EQ(1u, v.size());
    EXPECT_EQ(1, v[0].dummy);
}

TEST(StaticVector, can_erase_non_default_constructible_type)
{
    util::StaticVector<NonDefaultConstructible, 5> v;
    ASSERT_NO_THROW(v.push_back(1));
    EXPECT_NO_THROW(v.erase(v.begin()));

    EXPECT_EQ(0u, v.size());
}

struct NoThrowableMoveCtor
{
    // use the opposite exception spec to check whether it influences the result or not
    NoThrowableMoveCtor(const NoThrowableMoveCtor &){};

    NoThrowableMoveCtor &operator=(const NoThrowableMoveCtor &)
    {
        return *this;
    }

    NoThrowableMoveCtor &operator=(NoThrowableMoveCtor &&)
    {
        return *this;
    }

    NoThrowableMoveCtor(NoThrowableMoveCtor &&) noexcept {};
};

TEST(StaticVector, has_nothrow_move_ctor_when_type_has_it)
{
    EXPECT_TRUE((std::is_nothrow_move_constructible_v<util::StaticVector<NoThrowableMoveCtor, 5>>));
}

struct NoThrowableMoveAssign
{
    // use the opposite exception spec to check whether it influences the result or not
    NoThrowableMoveAssign(const NoThrowableMoveAssign &){};

    NoThrowableMoveAssign &operator=(const NoThrowableMoveAssign &)
    {
        return *this;
    }

    NoThrowableMoveAssign(NoThrowableMoveAssign &&){};

    NoThrowableMoveAssign &operator=(NoThrowableMoveAssign &&) noexcept
    {
        return *this;
    }
};

TEST(StaticVector, has_nothrow_move_assign_when_type_has_it)
{
    EXPECT_TRUE((std::is_nothrow_move_assignable_v<util::StaticVector<NoThrowableMoveAssign, 5>>));
}

struct ThrowableMoveCtor
{
    // use the opposite exception spec to check whether it influences the result or not
    ThrowableMoveCtor(const ThrowableMoveCtor &) noexcept {};

    ThrowableMoveCtor operator=(const ThrowableMoveCtor &) noexcept
    {
        return *this;
    }

    ThrowableMoveCtor operator=(ThrowableMoveCtor &&) noexcept
    {
        return *this;
    }

    ThrowableMoveCtor(ThrowableMoveCtor &&){};
};

TEST(StaticVector, has_throwable_move_ctor_when_type_has_it)
{
    EXPECT_FALSE((std::is_nothrow_move_constructible_v<util::StaticVector<ThrowableMoveCtor, 5>>));
}

struct ThrowableMoveAssign
{
    // use the opposite exception spec to check whether it influences the result or not
    ThrowableMoveAssign(const ThrowableMoveAssign &) noexcept {};

    ThrowableMoveAssign operator=(const ThrowableMoveAssign &) noexcept
    {
        return *this;
    }

    ThrowableMoveAssign(ThrowableMoveAssign &&) noexcept {};

    ThrowableMoveAssign operator=(ThrowableMoveAssign &&)
    {
        return *this;
    }
};

TEST(StaticVector, has_throwable_move_assign_when_type_has_it)
{
    EXPECT_FALSE((std::is_nothrow_move_assignable_v<util::StaticVector<ThrowableMoveAssign, 5>>));
}

struct NoThrowableCopyCtor
{
    // use the opposite exception spec to check whether it influences the result or not
    NoThrowableCopyCtor(NoThrowableCopyCtor &&){};

    NoThrowableCopyCtor &operator=(const NoThrowableCopyCtor &)
    {
        return *this;
    }

    NoThrowableCopyCtor &operator=(NoThrowableCopyCtor &&)
    {
        return *this;
    }

    NoThrowableCopyCtor(const NoThrowableCopyCtor &) noexcept {};
};

TEST(StaticVector, has_nothrow_copy_ctor_when_type_has_it)
{
    EXPECT_TRUE((std::is_nothrow_copy_constructible_v<util::StaticVector<NoThrowableCopyCtor, 5>>));
}

struct NoThrowableCopyAssign
{
    // use the opposite exception spec to check whether it influences the result or not
    NoThrowableCopyAssign(const NoThrowableCopyAssign &){};
    NoThrowableCopyAssign(NoThrowableCopyAssign &&){};

    NoThrowableCopyAssign &operator=(NoThrowableCopyAssign &&)
    {
        return *this;
    }

    NoThrowableCopyAssign &operator=(const NoThrowableCopyAssign &) noexcept
    {
        return *this;
    }
};

TEST(StaticVector, has_nothrow_copy_assign_when_type_has_it)
{
    EXPECT_TRUE((std::is_nothrow_copy_assignable_v<util::StaticVector<NoThrowableCopyAssign, 5>>));
}

struct ThrowableCopyCtor
{
    // use the opposite exception spec to check whether it influences the result or not
    ThrowableCopyCtor(ThrowableCopyCtor &&) noexcept {};

    ThrowableCopyCtor operator=(const ThrowableCopyCtor &) noexcept
    {
        return *this;
    }

    ThrowableCopyCtor operator=(ThrowableCopyCtor &&) noexcept
    {
        return *this;
    }

    ThrowableCopyCtor(const ThrowableCopyCtor &){};
};

TEST(StaticVector, has_throwable_copy_ctor_when_type_has_it)
{
    EXPECT_FALSE((std::is_nothrow_copy_constructible_v<util::StaticVector<ThrowableCopyCtor, 5>>));
}

struct ThrowableCopyAssign
{
    // use the opposite exception spec to check whether it influences the result or not
    ThrowableCopyAssign(const ThrowableCopyAssign &) noexcept {};

    ThrowableCopyAssign operator=(ThrowableCopyAssign &) noexcept
    {
        return *this;
    }

    ThrowableCopyAssign(ThrowableCopyAssign &&) noexcept {};

    ThrowableCopyAssign operator=(const ThrowableCopyAssign &&)
    {
        return *this;
    }
};

TEST(StaticVector, has_throwable_copy_assign_when_type_has_it)
{
    EXPECT_FALSE((std::is_nothrow_copy_assignable_v<util::StaticVector<ThrowableCopyAssign, 5>>));
}

TEST(StaticVector, construct_from_empty_range)
{
    auto data = std::make_shared<short>(0);

    std::weak_ptr<short> wdata = data;

    util::StaticVector<std::shared_ptr<short>, 2> v(&data, &data);

    EXPECT_EQ(0u, v.size());
    EXPECT_FALSE(wdata.expired());
}

TEST(StaticVector, construct_from_range)
{
    util::StaticVector<std::shared_ptr<int>, 5> data{std::make_shared<int>(1), std::make_shared<int>(2)};

    std::weak_ptr<int> wdata[2] = {data[0], data[1]};

    util::StaticVector<std::shared_ptr<int>, 2> v(data.begin(), data.end());

    ASSERT_EQ(2u, v.size());
    EXPECT_EQ(data[0], v[0]);
    EXPECT_EQ(data[1], v[1]);

    EXPECT_FALSE(wdata[0].expired());
    EXPECT_FALSE(wdata[1].expired());
}

TEST(StaticVector, construct_from_range_bigger_than_capacity_fails)
{
    util::StaticVector<std::shared_ptr<int>, 5> data{std::make_shared<int>(1), std::make_shared<int>(2)};

    EXPECT_THROW((util::StaticVector<std::shared_ptr<int>, 1>(data.begin(), data.end())), std::bad_alloc);
}
