/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_TENSORBATCH_HPP
#define NVCV_PYTHON_TENSORBATCH_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <nvcv/TensorBatch.hpp>
#include <nvcv/python/Tensor.hpp>

#include <cassert>

namespace nvcvpy {

namespace py = pybind11;

class TensorBatch
    : public Resource
    , public nvcv::TensorBatch
{
public:
    static TensorBatch Create(int capacity)
    {
        PyObject *tensorBatch = capi().TensorBatch_Create(capacity);

        py::object pytensorBatch = py::reinterpret_steal<py::object>(tensorBatch);

        return TensorBatch(pytensorBatch);
    }

    void pushBack(Tensor tensor)
    {
        capi().TensorBatch_PushBack(this->ptr(), tensor.ptr());
    }

    void popBack(int cnt)
    {
        capi().TensorBatch_PopBack(this->ptr(), cnt);
    }

    void clear()
    {
        capi().TensorBatch_Clear(this->ptr());
    }

    using nvcv::TensorBatch::operator[];
    using nvcv::TensorBatch::begin;
    using nvcv::TensorBatch::end;

private:
    friend struct py::detail::type_caster<TensorBatch>;

    TensorBatch() = default;

    explicit TensorBatch(py::object obj)
        : Resource(obj)
        , nvcv::TensorBatch(FromHandle(capi().TensorBatch_GetHandle(this->ptr()), true))
    {
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::TensorBatch> : type_caster_base<cvpy::TensorBatch>
{
    PYBIND11_TYPE_CASTER(cvpy::TensorBatch, const_name("nvcv.TensorBatch"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::TensorBatch(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::TensorBatch tensor, return_value_policy /* policy */, handle /*parent */)
    {
        tensor.inc_ref(); // for some reason this is needed
        return tensor;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_TENSORBATCH_HPP
