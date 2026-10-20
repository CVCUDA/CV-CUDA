/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cvcuda/priv/OsdContext.hpp>
#include <gtest/gtest.h>

#include <type_traits>

TEST(OsdMemory, does_not_duplicate_allocation_ownership)
{
    using Memory = cvcuda::priv::osd::Memory<cvcuda::priv::osd::TextLocation>;

    EXPECT_TRUE(std::is_default_constructible_v<Memory>);
    EXPECT_FALSE(std::is_copy_constructible_v<Memory>);
    EXPECT_FALSE(std::is_copy_assignable_v<Memory>);
    EXPECT_FALSE(std::is_move_constructible_v<Memory>);
    EXPECT_FALSE(std::is_move_assignable_v<Memory>);
}
