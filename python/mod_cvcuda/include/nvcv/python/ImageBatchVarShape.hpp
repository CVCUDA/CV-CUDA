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

#ifndef NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP
#define NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/python/Image.hpp>

#include <cassert>

namespace nvcvpy {

namespace py = nanobind;

class ImageBatchVarShape
    : public Resource
    , public nvcv::ImageBatchVarShape
{
public:
    ImageBatchVarShape(const ImageBatchVarShape &)     = default;
    ImageBatchVarShape(ImageBatchVarShape &&) noexcept = default;

    ImageBatchVarShape &operator=(const ImageBatchVarShape &)     = default;
    ImageBatchVarShape &operator=(ImageBatchVarShape &&) noexcept = default;

    static ImageBatchVarShape Create(int capacity)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *ovarshape = capi().ImageBatchVarShape_Create(capacity); // NOSONAR
        CheckCAPIError();
        NVCV_ASSERT(ovarshape != nullptr);
        py::object pyvarshape = py::reinterpret_steal<py::object>(ovarshape);

        return ImageBatchVarShape(pyvarshape);
    }

    // Python image wrappers must go through CAPI so the backing Python objects
    // stay alive for as long as the batch references them.
    void pushBackImage(const Image &img) // NOSONAR: mutates state through the wrapped C handle.
    {
        capi().ImageBatchVarShape_PushBack(this->ptr(), img.ptr());
        CheckCAPIError();
    }

    void popBackImages(int cnt) // NOSONAR: mutates state through the wrapped C handle.
    {
        capi().ImageBatchVarShape_PopBack(this->ptr(), cnt);
        CheckCAPIError();
    }

    void clearImages() // NOSONAR: mutates state through the wrapped C handle.
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

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::ImageBatchVarShape>
{
    NB_TYPE_CASTER(cvpy::ImageBatchVarShape, const_name("cvcuda.ImageBatchVarShape"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::IMAGE_BATCH_VAR_SHAPE>(src.ptr()))
        {
            value = cvpy::ImageBatchVarShape(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::ImageBatchVarShape tensor, rv_policy, cleanup_list *) noexcept
    {
        return static_cast<object &>(tensor).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_IMAGEBATCHVARSHAPE_HPP
