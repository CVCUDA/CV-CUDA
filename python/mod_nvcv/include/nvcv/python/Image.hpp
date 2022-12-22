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

#ifndef NVCV_PYTHON_IMAGE_HPP
#define NVCV_PYTHON_IMAGE_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>
#include <pybind11/stl.h>

#include <cassert>
#include <vector>

namespace nvcvpy {

namespace py = pybind11;

namespace priv {
class Image
{
};
} // namespace priv

class Image
    : public Resource
    , public nvcv::IImage
{
public:
    using PrivateImpl = priv::Image;

    static Image Create(nvcv::Size2D size, nvcv::ImageFormat fmt)
    {
        PyObject *oimg = capi().Image_Create(size.w, size.h, static_cast<NVCVImageFormat>(fmt));

        py::object pyimg = py::reinterpret_steal<py::object>(oimg);

        return Image(pyimg);
    }

private:
    friend struct py::detail::type_caster<Image>;
    NVCVImageHandle m_handle;

    Image() = default;

    explicit Image(py::object obj)
        : Resource(obj)
        , m_handle(capi().Image_GetHandle(this->ptr()))
    {
    }

    NVCVImageHandle doGetHandle() const override
    {
        return m_handle;
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Image> : type_caster_base<cvpy::Image>
{
    PYBIND11_TYPE_CASTER(cvpy::Image, const_name("nvcv.Image"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::Image(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::Image tensor, return_value_policy /* policy */, handle /*parent */)
    {
        tensor.inc_ref(); // for some reason this is needed
        return tensor;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_IMAGE_HPP
