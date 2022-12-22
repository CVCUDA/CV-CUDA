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

#ifndef NVCV_PYTHON_STREAM_HPP
#define NVCV_PYTHON_STREAM_HPP

#include "CAPI.hpp"

#include <cuda_runtime.h>

namespace nvcvpy {

namespace py = pybind11;

class Stream : public py::object
{
public:
    Stream() = default;

    static Stream Current()
    {
        return Stream(py::reinterpret_borrow<py::object>(capi().Stream_GetCurrent()));
    }

    cudaStream_t cudaHandle() const
    {
        return capi().Stream_GetCudaHandle(this->ptr());
    }

private:
    friend struct py::detail::type_caster<Stream>;

    py::object m_pyStream;

    explicit Stream(py::object obj)
        : py::object(obj)
    {
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Stream> : type_caster_base<cvpy::Stream>
{
    PYBIND11_TYPE_CASTER(cvpy::Stream, const_name("nvcv.cuda.Stream"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::Stream(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::Stream stream, return_value_policy /* policy */, handle /*parent */)
    {
        stream.inc_ref(); // for some reason this is needed
        return stream;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_STREAM_HPP
