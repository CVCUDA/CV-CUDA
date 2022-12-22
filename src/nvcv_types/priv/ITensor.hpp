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

#ifndef NVCV_CORE_PRIV_ITENSOR_HPP
#define NVCV_CORE_PRIV_ITENSOR_HPP

#include "ICoreObject.hpp"
#include "ImageFormat.hpp"

#include <nvcv/Tensor.h>

namespace nvcv::priv {

class IAllocator;

class ITensor : public ICoreObjectHandle<ITensor, NVCVTensorHandle>
{
public:
    virtual int32_t        rank() const  = 0;
    virtual const int64_t *shape() const = 0;

    virtual const NVCVTensorLayout &layout() const = 0;

    virtual DataType dtype() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(NVCVTensorData &data) const = 0;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_ITENSOR_HPP
