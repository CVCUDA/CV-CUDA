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

#include <nvcv_types/priv/LockFreeStack.hpp>

namespace priv = nvcv::priv;

struct Node
{
    Node(int v = 0)
        : value(v){};
    int value;

    Node *next = nullptr;
};

TEST(LockFreeStack, wip_push)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i].value = i;
        stack.push(n + i);
    }

    ASSERT_EQ(n + 2, stack.top());
    ASSERT_EQ(n + 1, n[2].next);
    ASSERT_EQ(n + 0, n[1].next);
    ASSERT_EQ(nullptr, n[0].next);

    ASSERT_EQ(n + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n + 0, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, wip_push_stack)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node n(0);
    stack.push(&n);

    Node nn[3];
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn, nn + 2);

    ASSERT_EQ(nn + 0, stack.top());
    ASSERT_EQ(nn + 1, nn[0].next);
    ASSERT_EQ(nn + 2, nn[1].next);
    ASSERT_EQ(&n, nn[2].next);
    ASSERT_EQ(nullptr, n.next);

    ASSERT_EQ(nn + 0, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(&n, stack.pop());
    ASSERT_TRUE(stack.empty());

    ASSERT_EQ(nullptr, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, wip_release)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node nn[3];
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn, nn + 2);

    Node *h = stack.release();

    EXPECT_EQ(nullptr, stack.top());

    EXPECT_EQ(nn + 0, h);
    EXPECT_EQ(nn + 1, nn[0].next);
    EXPECT_EQ(nn + 2, nn[1].next);
    EXPECT_EQ(nullptr, nn[2].next);
}
