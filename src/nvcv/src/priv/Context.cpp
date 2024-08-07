/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Context.hpp"

#include "HandleManagerImpl.hpp"

#include <nvcv/util/Assert.h>

namespace nvcv::priv {

IContext &GlobalContext()
{
    static Context g_ctx;
    return g_ctx;
}

Context::Context()
    : m_allocatorManager("Allocator")
    , m_imageManager("Image")
    , m_imageBatchManager("ImageBatch")
    , m_tensorManager("Tensor")
    , m_tensorBatchManager("TensorBatch")
    , m_arrayManager("Array")
    , m_managerList{m_allocatorManager, m_imageManager,       m_imageBatchManager,
                    m_tensorManager,    m_tensorBatchManager, m_arrayManager}
{
}

Context::~Context()
{
    // empty
}

IAllocator &Context::allocDefault()
{
    return m_allocDefault;
}

auto Context::managerList() const -> const Managers &
{
    return m_managerList;
}

template class HandleManager<IImage>;
template class HandleManager<IImageBatch>;
template class HandleManager<ITensor>;
template class HandleManager<IArray>;
template class HandleManager<IAllocator>;
template class HandleManager<ITensorBatch>;

} // namespace nvcv::priv
