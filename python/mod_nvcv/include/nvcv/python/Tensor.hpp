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

#ifndef NVCV_PYTHON_TENSOR_HPP
#define NVCV_PYTHON_TENSOR_HPP

#include "CAPI.hpp"
#include "DataType.hpp"
#include "Resource.hpp"

#include <nvcv/Tensor.hpp>
#include <nvcv/TensorLayout.hpp>
#include <pybind11/stl.h>

#include <cassert>
#include <vector>

namespace nvcvpy {

namespace py = pybind11;

using Shape = std::vector<int64_t>;

class Tensor
    : public Resource
    , public nvcv::ITensor
{
public:
    static Tensor Create(const nvcv::TensorShape &tshape, nvcv::DataType dtype)
    {
        PyObject *otensor = capi().Tensor_Create(tshape.size(), &tshape[0], static_cast<NVCVDataType>(dtype),
                                                 static_cast<NVCVTensorLayout>(tshape.layout()));

        py::object pytensor = py::reinterpret_steal<py::object>(otensor);

        return Tensor(pytensor);
    }

    static Tensor CreateForImageBatch(int numImages, nvcv::Size2D size, nvcv::ImageFormat fmt)
    {
        PyObject *otensor
            = capi().Tensor_CreateForImageBatch(numImages, size.w, size.h, static_cast<NVCVImageFormat>(fmt));

        py::object pytensor = py::reinterpret_steal<py::object>(otensor);

        return Tensor(pytensor);
    }

private:
    friend struct py::detail::type_caster<Tensor>;
    NVCVTensorHandle m_handle;

    Tensor() = default;

    explicit Tensor(py::object obj)
        : Resource(obj)
        , m_handle(capi().Tensor_GetHandle(this->ptr()))
    {
    }

    NVCVTensorHandle doGetHandle() const override
    {
        return m_handle;
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Tensor> : type_caster_base<cvpy::Tensor>
{
    PYBIND11_TYPE_CASTER(cvpy::Tensor, const_name("nvcv.Tensor"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::Tensor(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::Tensor tensor, return_value_policy /* policy */, handle /*parent */)
    {
        tensor.inc_ref(); // for some reason this is needed
        return tensor;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_TENSOR_HPP
