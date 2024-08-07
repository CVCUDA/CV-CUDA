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

#ifndef NVCV_PRIV_CORE_ICONTEXT_HPP
#define NVCV_PRIV_CORE_ICONTEXT_HPP

#include <nvcv/Fwd.h>

#include <tuple>

namespace nvcv::priv {

// Forward declaration
template<class HandleType>
class CoreObjManager;

using ImageManager       = CoreObjManager<NVCVImageHandle>;
using ImageBatchManager  = CoreObjManager<NVCVImageBatchHandle>;
using TensorManager      = CoreObjManager<NVCVTensorHandle>;
using TensorBatchManager = CoreObjManager<NVCVTensorBatchHandle>;
using ArrayManager       = CoreObjManager<NVCVArrayHandle>;
using AllocatorManager   = CoreObjManager<NVCVAllocatorHandle>;

class IAllocator;

class IContext
{
public:
    using Managers = std::tuple<AllocatorManager &, ImageManager &, ImageBatchManager &, TensorManager &,
                                TensorBatchManager &, ArrayManager &>;

    template<class HandleType>
    CoreObjManager<HandleType> &manager()
    {
        return std::get<CoreObjManager<HandleType> &>(managerList());
    }

    virtual const Managers &managerList() const = 0;
    virtual IAllocator     &allocDefault()      = 0;
};

// Defined in Context.cpp
IContext &GlobalContext();

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_ICONTEXT_HPP
