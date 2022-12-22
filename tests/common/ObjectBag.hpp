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

#ifndef NVVPI_TEST_UTIL_OBJECTBAG_HPP
#define NVVPI_TEST_UTIL_OBJECTBAG_HPP

#include <nvcv/alloc/Fwd.h>

#include <functional>
#include <stack>

namespace nvcv::test {

// Bag of NVCV objects, destroys them in its dtor in reverse
// order of insertion.
class ObjectBag final
{
public:
    ObjectBag()                  = default;
    ObjectBag(const ObjectBag &) = delete;

    ~ObjectBag();

    void insert(NVCVAllocatorHandle handle);

private:
    std::stack<std::function<void()>> m_objs;
};

} // namespace nvcv::test

#endif // NVVPI_TEST_UTIL_OBJECTBAG_HPP
