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

#include <nvcv/src/priv/LockFreeStack.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

namespace priv = nvcv::priv;

struct Node
{
    explicit Node(int v = 0)
        : value(v){};
    int value;

    Node *next = nullptr;
};

TEST(LockFreeStack, smoke_push)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    std::array<Node, 3> n;
    for (int i = 0; i < 3; ++i)
    {
        n[i].value = i;
        stack.push(n.data() + i);
    }

    ASSERT_EQ(n.data() + 2, stack.top());
    ASSERT_EQ(n.data() + 1, n[2].next);
    ASSERT_EQ(n.data() + 0, n[1].next);
    ASSERT_EQ(nullptr, n[0].next);

    ASSERT_EQ(n.data() + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n.data() + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(n.data() + 0, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, smoke_push_stack)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    Node n(0);
    stack.push(&n);

    std::array<Node, 3> nn;
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn.data(), nn.data() + 2);

    ASSERT_EQ(nn.data() + 0, stack.top());
    ASSERT_EQ(nn.data() + 1, nn[0].next);
    ASSERT_EQ(nn.data() + 2, nn[1].next);
    ASSERT_EQ(&n, nn[2].next);
    ASSERT_EQ(nullptr, n.next);

    ASSERT_EQ(nn.data() + 0, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn.data() + 1, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(nn.data() + 2, stack.pop());
    ASSERT_FALSE(stack.empty());

    ASSERT_EQ(&n, stack.pop());
    ASSERT_TRUE(stack.empty());

    ASSERT_EQ(nullptr, stack.pop());
    ASSERT_TRUE(stack.empty());
}

TEST(LockFreeStack, smoke_release)
{
    priv::LockFreeStack<Node> stack;
    ASSERT_TRUE(stack.empty());

    std::array<Node, 3> nn;
    for (int i = 0; i < 3; ++i)
    {
        nn[i].value = i;
        nn[i].next  = i + 1 < 3 ? &nn[i + 1] : nullptr;
    }

    stack.pushStack(nn.data(), nn.data() + 2);

    Node *h = stack.release();

    EXPECT_EQ(nullptr, stack.top());

    EXPECT_EQ(nn.data() + 0, h);
    EXPECT_EQ(nn.data() + 1, nn[0].next);
    EXPECT_EQ(nn.data() + 2, nn[1].next);
    EXPECT_EQ(nullptr, nn[2].next);
}

TEST(LockFreeStack, publishes_nodes_between_threads)
{
    constexpr int kNumNodes = 1000;

    priv::LockFreeStack<Node>   stack;
    std::array<Node, kNumNodes> nodes;
    std::atomic<bool>           valid{true};

    std::jthread producer(
        [&nodes, &stack]
        {
            for (int i = 0; i < kNumNodes; ++i)
            {
                nodes[i].value = i + 1;
                stack.push(&nodes[i]);
            }
        });

    std::jthread consumer(
        [&stack, &valid]
        {
            std::array<bool, kNumNodes> seen{};
            const auto                  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            for (int count = 0; count < kNumNodes;)
            {
                const Node *node = stack.pop();
                if (node == nullptr)
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        FAIL() << "consumer received only " << count << " of " << seen.size()
                               << " published nodes before the deadline";
                        return;
                    }
                    std::this_thread::yield();
                    continue;
                }

                if (node->value < 1 || node->value > kNumNodes || seen[node->value - 1])
                {
                    valid.store(false, std::memory_order_relaxed); // NOSONAR: observed only after join.
                }
                else
                {
                    seen[node->value - 1] = true;
                }
                ++count;
            }
        });

    producer.join();
    consumer.join();

    EXPECT_TRUE(valid.load(std::memory_order_relaxed)); // NOSONAR: producer and consumer are joined.
    EXPECT_TRUE(stack.empty());
}

TEST(LockFreeStack, publishes_detached_stack_between_threads)
{
    constexpr int kNumNodes = 1000;

    priv::LockFreeStack<Node>   stack;
    std::array<Node, kNumNodes> nodes;
    std::atomic<bool>           valid{true};

    std::jthread producer(
        [&nodes, &stack]
        {
            for (int i = 0; i < kNumNodes; ++i)
            {
                nodes[i].value = i + 1;
                nodes[i].next  = i + 1 < kNumNodes ? &nodes[i + 1] : nullptr;
            }
            stack.pushStack(nodes.data(), &nodes.back());
        });

    std::jthread consumer(
        [&stack, &valid]
        {
            // S1121: keep the release out of the loop condition.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            Node      *node     = stack.release();
            while (node == nullptr)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    FAIL() << "consumer received no detached stack before the deadline";
                    return;
                }
                std::this_thread::yield();
                node = stack.release();
            }

            for (int expected = 1; expected <= kNumNodes; ++expected)
            {
                if (node == nullptr || node->value != expected)
                {
                    valid.store(false, std::memory_order_relaxed); // NOSONAR: observed only after join.
                    return;
                }
                node = node->next;
            }
            valid.store(node == nullptr, std::memory_order_relaxed); // NOSONAR: observed only after join.
        });

    producer.join();
    consumer.join();

    EXPECT_TRUE(valid.load(std::memory_order_relaxed)); // NOSONAR: producer and consumer are joined.
    EXPECT_TRUE(stack.empty());
}
