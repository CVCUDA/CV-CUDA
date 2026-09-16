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

#include "StreamStack.hpp"

#include "Stream.hpp"

namespace nvcvpy::priv {

void StreamStack::push(Stream &stream)
{
    m_stack.push(stream.sharedStream());
}

void StreamStack::pop()
{
    if (!m_stack.empty())
    {
        m_stack.pop();
    }
}

std::shared_ptr<Stream> StreamStack::top()
{
    while (!m_stack.empty())
    {
        if (std::shared_ptr<Stream> stream = m_stack.top().lock())
        {
            return stream;
        }
        m_stack.pop();
    }
    return nullptr;
}

StreamStack &StreamStack::Instance()
{
    thread_local StreamStack stack;
    return stack;
}

} // namespace nvcvpy::priv
