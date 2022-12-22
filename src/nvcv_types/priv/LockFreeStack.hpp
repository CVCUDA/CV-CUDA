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

#ifndef NVCV_PRIV_CORE_LOCK_FREE_STACK_HPP
#define NVCV_PRIV_CORE_LOCK_FREE_STACK_HPP

#include <atomic>
#include <cassert>
#include <concepts>
#include <stack>

namespace nvcv::priv {

template<class T>
concept ForwardListNode = requires(T n)
{
    // clang-format off
    { n.next } -> std::convertible_to<T *>;
    // clang-format on
};

template<ForwardListNode Node>
class LockFreeStack
{
public:
    Node *pop() noexcept
    {
        // Lock the stack's head so that we can pop current head and set the
        // new one to curhead->next atomically below.
        for (;;)
        {
            Node *head = doGetUnlocked(m_head.load(std::memory_order_relaxed));
            if (!head)
            {
                return nullptr;
            }

            if (m_head.compare_exchange_weak(head, doGetLocked(head), std::memory_order_relaxed,
                                             std::memory_order_release))
            {
                break;
            }
        }

        // Set the newHead to oldHead->next and return oldHead
        Node *oldHead, *newHead;
        do
        {
            oldHead = m_head.load(std::memory_order_relaxed);
            assert(doGetLocked(oldHead)); // must have been locked above

            newHead = doGetUnlocked(oldHead)->next;
        }
        while (!m_head.compare_exchange_weak(oldHead, newHead, std::memory_order_relaxed, std::memory_order_release));

        return doGetUnlocked(oldHead);
    }

    void push(Node *newNode) noexcept
    {
        Node *oldHead;
        do
        {
            oldHead       = doGetUnlocked(m_head.load(std::memory_order_relaxed));
            newNode->next = oldHead;
        }
        while (!m_head.compare_exchange_weak(oldHead, newNode, std::memory_order_relaxed, std::memory_order_release));
    }

    Node *release() noexcept
    {
        Node *h = m_head.load(std::memory_order_relaxed);
        while (!m_head.compare_exchange_weak(h, nullptr, std::memory_order_relaxed, std::memory_order_release))
        {
        }

        return h;
    }

    void pushStack(Node *newHead, Node *last) noexcept
    {
        Node *oldHead;
        do
        {
            oldHead    = doGetUnlocked(m_head.load(std::memory_order_relaxed));
            last->next = oldHead;
        }
        while (!m_head.compare_exchange_weak(oldHead, newHead, std::memory_order_relaxed, std::memory_order_release));
    }

    Node *top() const
    {
        return doGetUnlocked(m_head.load());
    }

    void clear()
    {
        m_head = nullptr;
    }

    bool empty() const
    {
        return m_head == nullptr;
    }

private:
    std::atomic<Node *> m_head = nullptr;

    template<typename T>
    static T *doGetUnlocked(T *maybe_locked)
    {
        static_assert(alignof(Node) >= 2);
        return reinterpret_cast<T *>(reinterpret_cast<intptr_t>(maybe_locked) & -2);
    }

    template<typename T>
    static T *doGetLocked(T *maybe_unlocked)
    {
        static_assert(alignof(Node) >= 2);
        return reinterpret_cast<T *>(reinterpret_cast<intptr_t>(maybe_unlocked) | 1);
    }
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_LOCK_FREE_STACK_HPP
