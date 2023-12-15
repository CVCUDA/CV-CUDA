/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Array.hpp>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cassert>

namespace nvcvpy {

namespace py = pybind11;

class Array
    : public Resource
    , public nvcv::Array
{
public:
    static Array Create(int64_t length, nvcv::DataType dtype)
    {
        PyObject *oarray = capi().Array_Create(length, dtype);

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
        , nvcv::Array(FromHandle(capi().Array_GetHandle(this->ptr()), true))
    {
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Array> : type_caster_base<cvpy::Array>
{
    PYBIND11_TYPE_CASTER(cvpy::Array, const_name("nvcv.Array"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::Array(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::Array array, return_value_policy /* policy */, handle /*parent */)
    {
        array.inc_ref(); // for some reason this is needed
        return array;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_ARRAY_HPP
