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

#include "ObjectBag.hpp"

#include <nvcv/alloc/Allocator.h>

namespace nvcv::test {

ObjectBag::~ObjectBag()
{
    // Destroy from back to front
    while (!m_objs.empty())
    {
        m_objs.top()(); // call object destructor
        m_objs.pop();
    }
}

void ObjectBag::insert(NVCVAllocatorHandle handle)
{
    m_objs.push([handle]() { nvcvAllocatorDestroy(handle); });
}

} // namespace nvcv::test
