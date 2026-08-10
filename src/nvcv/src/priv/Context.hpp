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

#ifndef NVCV_PRIV_CORE_CONTEXT_HPP
#define NVCV_PRIV_CORE_CONTEXT_HPP

#include "AllocatorManager.hpp"
#include "ArrayManager.hpp"
#include "DefaultAllocator.hpp"
#include "IContext.hpp"
#include "ImageBatchManager.hpp"
#include "ImageManager.hpp"
#include "TensorBatchManager.hpp"
#include "TensorManager.hpp"

namespace nvcv::priv {

class Context final : public IContext
{
public:
    Context();
    ~Context() override;

    const Managers &managerList() const override;
    IAllocator     &allocDefault() override;

private:
    // Order is important due to inter-dependencies
    DefaultAllocator   m_allocDefault;
    AllocatorManager   m_allocatorManager{"Allocator"};
    ImageManager       m_imageManager{"Image"};
    ImageBatchManager  m_imageBatchManager{"ImageBatch"};
    TensorManager      m_tensorManager{"Tensor"};
    TensorBatchManager m_tensorBatchManager{"TensorBatch"};
    ArrayManager       m_arrayManager{"Array"};

    Managers m_managerList{m_allocatorManager, m_imageManager,       m_imageBatchManager,
                           m_tensorManager,    m_tensorBatchManager, m_arrayManager};
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_CONTEXT_HPP
