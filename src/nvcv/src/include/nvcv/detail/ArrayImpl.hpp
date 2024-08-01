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

#ifndef NVCV_ARRAY_IMPL_HPP
#define NVCV_ARRAY_IMPL_HPP

#ifndef NVCV_ARRAY_HPP
#    error "You must not include this header directly"
#endif

namespace nvcv {

// Array implementation -------------------------------------

inline int64_t Array::length() const
{
    NVCVArrayHandle harray = this->handle();

    int64_t length = 0;
    detail::CheckThrow(nvcvArrayGetLength(harray, &length));

    return length;
}

inline void Array::resize(int64_t length)
{
    NVCVArrayHandle harray = this->handle();

    detail::CheckThrow(nvcvArrayResize(harray, length));
}

inline int64_t Array::capacity() const
{
    NVCVArrayHandle harray = this->handle();

    int64_t capacity = 0;
    detail::CheckThrow(nvcvArrayGetCapacity(harray, &capacity));

    return capacity;
}

inline int Array::rank() const
{
    return 1;
}

inline DataType Array::dtype() const
{
    NVCVDataType out;
    detail::CheckThrow(nvcvArrayGetDataType(this->handle(), &out));
    return DataType{out};
}

inline NVCVResourceType Array::target() const
{
    NVCVArrayHandle harray = this->handle();

    NVCVResourceType target = NVCV_RESOURCE_MEM_CUDA;
    detail::CheckThrow(nvcvArrayGetTarget(harray, &target));

    return target;
}

inline ArrayData Array::exportData() const
{
    NVCVArrayData data;
    detail::CheckThrow(nvcvArrayExportData(this->handle(), &data));

    return ArrayData(data);
}

inline void Array::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvArraySetUserPointer(this->handle(), ptr));
}

inline void *Array::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvArrayGetUserPointer(this->handle(), &ptr));
    return ptr;
}

inline auto Array::CalcRequirements(int64_t capacity, DataType dtype, int32_t alignment, NVCVResourceType target)
    -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvArrayCalcRequirementsWithTarget(capacity, dtype, alignment, target, &reqs));
    return reqs;
}

inline Array::Array(const Requirements &reqs, NVCVResourceType target, const Allocator &alloc)
{
    NVCVArrayHandle handle;
    detail::CheckThrow(nvcvArrayConstructWithTarget(&reqs, alloc.handle(), target, &handle));
    reset(std::move(handle));
}

inline Array::Array(int64_t capacity, DataType dtype, int32_t alignment, NVCVResourceType target,
                    const Allocator &alloc)
    : Array(CalcRequirements(capacity, dtype, alignment, target), target, alloc)
{
}

// Factory functions --------------------------------------------------

inline Array ArrayWrapData(const ArrayData &data, ArrayDataCleanupCallback &&cleanup)
{
    NVCVArrayHandle handle;
    detail::CheckThrow(
        nvcvArrayWrapDataConstruct(&data.cdata(), cleanup.targetFunc(), cleanup.targetHandle(), &handle));
    cleanup.release(); // already owned by the array
    return Array(std::move(handle));
}

} // namespace nvcv

#endif // NVCV_ARRAY_IMPL_HPP
