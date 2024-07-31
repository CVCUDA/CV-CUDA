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

#ifndef NVCV_CORE_PRIV_IARRAY_HPP
#define NVCV_CORE_PRIV_IARRAY_HPP

#include "ICoreObject.hpp"
#include "ImageFormat.hpp"
#include "SharedCoreObj.hpp"

#include <nvcv/Array.h>

namespace nvcv::priv {

class IAllocator;

class IArray : public ICoreObjectHandle<IArray, NVCVArrayHandle>
{
public:
    virtual int32_t rank() const     = 0;
    virtual int64_t capacity() const = 0;
    virtual int64_t length() const   = 0;

    virtual DataType dtype() const = 0;

    virtual SharedCoreObj<IAllocator> alloc() const = 0;

    virtual NVCVResourceType target() const = 0;

    virtual void exportData(NVCVArrayData &data) const = 0;

    virtual void resize(int64_t length) = 0;
};

template<>
class CoreObjManager<NVCVArrayHandle> : public HandleManager<IArray>
{
    using Base = HandleManager<IArray>;

public:
    using Base::Base;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_IARRAY_HPP
