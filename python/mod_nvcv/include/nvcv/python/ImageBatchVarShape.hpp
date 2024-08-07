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

#ifndef NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP
#define NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/python/Image.hpp>

#include <cassert>

namespace nvcvpy {

namespace py = pybind11;

class ImageBatchVarShape
    : public Resource
    , public nvcv::ImageBatchVarShape
{
public:
    static ImageBatchVarShape Create(int capacity)
    {
        PyObject *ovarshape = capi().ImageBatchVarShape_Create(capacity);
        CheckCAPIError();
        NVCV_ASSERT(ovarshape != nullptr);
        py::object pyvarshape = py::reinterpret_steal<py::object>(ovarshape);

        return ImageBatchVarShape(pyvarshape);
    }

    // For manipulating the image list we can't call directly the
    // nvcv::ImageBatchVarShape methods, it must go through the python
    // bindings because it ends up storing a reference to the added images, to
    // keep them alive. We can't do it here, things must be consistent.
    // PROBLEM: these functions should be virtual but they aren't.
    // We can't modify the image list through nvcv::ImageBatchVarShape
    // or else the this image list to keep their alive won't be updated.
    // We currently can't avoid this issue.
    void pushBack(Image img)
    {
        capi().ImageBatchVarShape_PushBack(this->ptr(), img.ptr());
        CheckCAPIError();
    }

    void popBack(int cnt)
    {
        capi().ImageBatchVarShape_PopBack(this->ptr(), cnt);
        CheckCAPIError();
    }

    void clear()
    {
        capi().ImageBatchVarShape_Clear(this->ptr());
        CheckCAPIError();
    }

    // By default we use the varshape interface.
    using nvcv::ImageBatchVarShape::operator[];
    using nvcv::ImageBatchVarShape::begin;
    using nvcv::ImageBatchVarShape::cbegin;
    using nvcv::ImageBatchVarShape::cend;
    using nvcv::ImageBatchVarShape::end;

private:
    friend struct py::detail::type_caster<ImageBatchVarShape>;

    ImageBatchVarShape() = default;

    explicit ImageBatchVarShape(py::object obj)
        : Resource(obj)
        , nvcv::ImageBatchVarShape(FromHandle(CheckCAPIError(capi().ImageBatchVarShape_GetHandle(this->ptr())), true))
    {
    }
};

} // namespace nvcvpy

namespace pybind11::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::ImageBatchVarShape> : type_caster_base<cvpy::ImageBatchVarShape>
{
    PYBIND11_TYPE_CASTER(cvpy::ImageBatchVarShape, const_name("nvcv.ImageBatchVarShape"));

    bool load(handle src, bool)
    {
        // Does it have the correct object type?
        PyTypeObject *srctype = Py_TYPE(src.ptr());
        if (strcmp(name.text, srctype->tp_name) == 0)
        {
            value = cvpy::ImageBatchVarShape(reinterpret_borrow<object>(src));
            return true;
        }
        else
        {
            return false;
        }
    }

    static handle cast(cvpy::ImageBatchVarShape tensor, return_value_policy /* policy */, handle /*parent */)
    {
        tensor.inc_ref(); // for some reason this is needed
        return tensor;
    }
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP
