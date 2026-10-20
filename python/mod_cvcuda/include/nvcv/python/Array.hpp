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

#ifndef NVCV_PYTHON_ARRAY_HPP
#define NVCV_PYTHON_ARRAY_HPP

#include "CAPI.hpp"
#include "DataType.hpp"
#include "Resource.hpp"
#include "Shape.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <nvcv/Array.hpp>

#include <cassert>

namespace nvcvpy {

namespace py = nanobind;

class Array
    : public Resource
    , public nvcv::Array
{
public:
    static Array Create(int64_t length, nvcv::DataType dtype)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *oarray = capi().Array_Create(length, static_cast<NVCVDataType>(dtype)); // NOSONAR
        CheckCAPIError();
        NVCV_ASSERT(oarray == nullptr);
        py::object pyarray = py::reinterpret_steal<py::object>(oarray);
        return Array(pyarray);
    }

    static Array Create(const Shape &shape, nvcv::DataType dtype)
    {
        return Create(LengthIf1D(shape), dtype);
    }

private:
    friend struct py::detail::type_caster<Array>;

    Array() = default;

    explicit Array(py::object obj)
        : Resource(obj)
        , nvcv::Array(FromHandle(CheckCAPIError(capi().Array_GetHandle(this->ptr())), true))
    {
    }
};

} // namespace nvcvpy

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Array>
{
    NB_TYPE_CASTER(cvpy::Array, const_name("cvcuda.Array"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::ARRAY>(src.ptr()))
        {
            value = cvpy::Array(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::Array array, rv_policy, cleanup_list *) noexcept
    {
        return static_cast<object &>(array).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_ARRAY_HPP
