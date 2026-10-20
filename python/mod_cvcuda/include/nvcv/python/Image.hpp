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

#ifndef NVCV_PYTHON_IMAGE_HPP
#define NVCV_PYTHON_IMAGE_HPP

#include "CAPI.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageFormat.hpp>

#include <cassert>
#include <vector>

namespace nvcvpy {

namespace py = nanobind;

namespace priv {
class Image
{
};
} // namespace priv

class Image
    : public Resource
    , public nvcv::Image
{
public:
    using PrivateImpl = priv::Image;

    static Image Create(nvcv::Size2D size, nvcv::ImageFormat fmt, int rowAlign = 0)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *oimg = capi().Image_Create(size.w, size.h, static_cast<NVCVImageFormat>(fmt), rowAlign); // NOSONAR
        CheckCAPIError();
        NVCV_ASSERT(oimg != nullptr);
        py::object pyimg = py::reinterpret_steal<py::object>(oimg);

        return Image(pyimg);
    }

private:
    friend struct py::detail::type_caster<Image>;

    Image() = default;

    explicit Image(py::object obj)
        : Resource(obj)
        , nvcv::Image(FromHandle(CheckCAPIError(capi().Image_GetHandle(this->ptr())), true))
    {
    }
};

} // namespace nvcvpy

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Image>
{
    NB_TYPE_CASTER(cvpy::Image, const_name("cvcuda.Image"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::IMAGE>(src.ptr()))
        {
            value = cvpy::Image(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::Image tensor, rv_policy, cleanup_list *) noexcept
    {
        return static_cast<object &>(tensor).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_IMAGE_HPP
