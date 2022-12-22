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

#include "StreamStack.hpp"

#include "Stream.hpp"

namespace nvcvpy::priv {

void StreamStack::push(Stream &stream)
{
    std::unique_lock lk(m_mtx);
    m_stack.push(stream.shared_from_this());
}

void StreamStack::pop()
{
    std::unique_lock lk(m_mtx);
    m_stack.pop();
}

std::shared_ptr<Stream> StreamStack::top()
{
    std::unique_lock lk(m_mtx);
    if (!m_stack.empty())
    {
        return m_stack.top().lock();
    }
    else
    {
        return nullptr;
    }
}

StreamStack &StreamStack::Instance()
{
    static StreamStack stack;
    return stack;
}

} // namespace nvcvpy::priv
