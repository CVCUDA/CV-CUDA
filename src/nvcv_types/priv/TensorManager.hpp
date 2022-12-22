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

#ifndef NVCV_PRIV_CORE_TENSORMANAGER_HPP
#define NVCV_PRIV_CORE_TENSORMANAGER_HPP

#include "IContext.hpp"
#include "Tensor.hpp"
#include "TensorWrapDataStrided.hpp"

namespace nvcv::priv {

using TensorManager = CoreObjManager<NVCVTensorHandle>;

using TensorStorage = CompatibleStorage<Tensor, TensorWrapDataStrided>;

template<>
class CoreObjManager<NVCVTensorHandle> : public HandleManager<ITensor, TensorStorage>
{
    using Base = HandleManager<ITensor, TensorStorage>;

public:
    using Base::Base;
};

} // namespace nvcv::priv

#endif // NVCV_PRIV_CORE_TENSORMANAGER_HPP
