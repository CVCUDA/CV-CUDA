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

#ifndef NVCV_ARRAY_HPP
#define NVCV_ARRAY_HPP

#include "Array.h"
#include "ArrayData.hpp"
#include "CoreResource.hpp"
#include "alloc/Allocator.hpp"
#include "detail/Callback.hpp"

namespace nvcv {

NVCV_IMPL_SHARED_HANDLE(Array);

// Array definition -------------------------------------
class Array : public CoreResource<NVCVArrayHandle, Array>
{
public:
    using HandleType   = NVCVArrayHandle;
    using Base         = CoreResource<NVCVArrayHandle, Array>;
    using Requirements = NVCVArrayRequirements;

    int      rank() const;
    DataType dtype() const;

    int64_t length() const;
    int64_t capacity() const;

    NVCVResourceType target() const;

    ArrayData exportData() const;

    void resize(int64_t length);

    template<typename DerivedArrayData>
    Optional<DerivedArrayData> exportData() const
    {
        return exportData().cast<DerivedArrayData>();
    }

    void  setUserPointer(void *ptr);
    void *userPointer() const;

    static Requirements CalcRequirements(int64_t capacity, DataType dtype, int32_t alignment = 0,
                                         NVCVResourceType target = NVCV_RESOURCE_MEM_CUDA);

    NVCV_IMPLEMENT_SHARED_RESOURCE(Array, Base);

    explicit Array(const Requirements &reqs, NVCVResourceType target = NVCV_RESOURCE_MEM_CUDA,
                   const Allocator &alloc = nullptr);
    explicit Array(int64_t capacity, DataType dtype, int32_t alignment = 0,
                   NVCVResourceType target = NVCV_RESOURCE_MEM_CUDA, const Allocator &alloc = nullptr);
};

// ArrayWrapData definition -------------------------------------
using ArrayDataCleanupFunc = void(const ArrayData &);

struct TranslateArrayDataCleanup
{
    template<typename CppCleanup>
    void operator()(CppCleanup &&c, const NVCVArrayData *data) const noexcept
    {
        c(ArrayData(*data));
    }
};

using ArrayDataCleanupCallback
    = CleanupCallback<ArrayDataCleanupFunc, detail::RemovePointer_t<NVCVArrayDataCleanupFunc>,
                      TranslateArrayDataCleanup>;

// ArrayWrapImage definition -------------------------------------

inline Array ArrayWrapData(const ArrayData &data, ArrayDataCleanupCallback &&cleanup = {});

using ArrayWrapHandle = NonOwningResource<Array>;

} // namespace nvcv

#include "detail/ArrayImpl.hpp"

#endif // NVCV_ARRAY_HPP
