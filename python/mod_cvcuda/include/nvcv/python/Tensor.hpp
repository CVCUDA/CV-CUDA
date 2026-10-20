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

#ifndef NVCV_PYTHON_TENSOR_HPP
#define NVCV_PYTHON_TENSOR_HPP

#include "CAPI.hpp"
#include "DataType.hpp"
#include "Resource.hpp"
#include "Shape.hpp"

#include <common/Assert.hpp>
#include <common/Binding.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorLayout.hpp>

#include <cassert>
#include <optional>

namespace nvcvpy {

namespace py = nanobind;

class Tensor
    : public Resource
    , public nvcv::Tensor
{
public:
    Tensor(const Tensor &)     = default;
    Tensor(Tensor &&) noexcept = default;

    Tensor &operator=(const Tensor &)     = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    static Tensor Create(const nvcv::TensorShape &tshape, nvcv::DataType dtype, int rowalign = 0)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *otensor // NOSONAR
            = capi().Tensor_Create(tshape.size(), &tshape[0], static_cast<NVCVDataType>(dtype),
                                   static_cast<NVCVTensorLayout>(tshape.layout()), rowalign);
        CheckCAPIError();
        NVCV_ASSERT(otensor != nullptr);
        py::object pytensor = py::reinterpret_steal<py::object>(otensor);

        return Tensor(pytensor);
    }

    static Tensor Create(const Shape &shape, nvcv::DataType dtype, nvcv::TensorLayout layout = nvcv::TENSOR_NONE,
                         int rowalign = 0)
    {
        return Create(CreateNVCVTensorShape(shape, layout), dtype, rowalign);
    }

    static Tensor CreateForImageBatch(int numImages, nvcv::Size2D size, nvcv::ImageFormat fmt, int rowalign = 0)
    {
        // reinterpret_steal below takes a non-const PyObject *.
        PyObject *otensor // NOSONAR
            = capi().Tensor_CreateForImageBatch(numImages, size.w, size.h, static_cast<NVCVImageFormat>(fmt), rowalign);
        CheckCAPIError();
        NVCV_ASSERT(otensor != nullptr);
        py::object pytensor = py::reinterpret_steal<py::object>(otensor);

        return Tensor(pytensor);
    }

private:
    friend struct py::detail::type_caster<Tensor>;

    Tensor() = default;

    explicit Tensor(py::object obj)
        : Resource(obj)
        , nvcv::Tensor(FromHandle(CheckCAPIError(capi().Tensor_GetHandle(this->ptr())), true))
    {
    }
};

// Forward an optional Python Tensor wrapper to a C++ submit() taking
// const nvcv::Tensor&, substituting a caller-owned null-handle fallback when
// the optional is empty.  Centralizes the explicit cast that silences Sonar's
// object-slicing finding at the operator call sites.
inline const nvcv::Tensor &AsNvcvTensor(const std::optional<Tensor> &opt, const nvcv::Tensor &nullTensor)
{
    return opt ? static_cast<const nvcv::Tensor &>(*opt) : nullTensor;
}

} // namespace nvcvpy

namespace nanobind::detail {

namespace cvpy = nvcvpy;

template<>
struct type_caster<cvpy::Tensor>
{
    NB_TYPE_CASTER(cvpy::Tensor, const_name("cvcuda.Tensor"));

    bool from_python(handle src, uint32_t, cleanup_list *)
    {
        if (cvpy::IsFacadeInstance<cvpy::FacadeType::TENSOR>(src.ptr()))
        {
            value = cvpy::Tensor(reinterpret_borrow<object>(src));
            return true;
        }
        return false;
    }

    static handle from_cpp(cvpy::Tensor tensor, rv_policy, cleanup_list *) noexcept
    {
        return static_cast<object &>(tensor).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_TENSOR_HPP
