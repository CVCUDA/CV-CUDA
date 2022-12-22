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

#ifndef NVCV_PRIV_CORE_CONTEXT_HPP
#define NVCV_PRIV_CORE_CONTEXT_HPP

#include "AllocatorManager.hpp"
#include "DefaultAllocator.hpp"
#include "IContext.hpp"
#include "ImageBatchManager.hpp"
#include "ImageManager.hpp"
#include "TensorManager.hpp"

namespace nvcv::priv {

class Context final : public IContext
{
public:
    Context();
    ~Context();

    const Managers &managerList() const override;
    IAllocator     &allocDefault() override;

private:
    // Order is important due to inter-dependencies
    DefaultAllocator  m_allocDefault;
    AllocatorManager  m_allocatorManager;
    ImageManager      m_imageManager;
    ImageBatchManager m_imageBatchManager;
    TensorManager     m_tensorManager;

    Managers m_managerList;
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_CONTEXT_HPP
