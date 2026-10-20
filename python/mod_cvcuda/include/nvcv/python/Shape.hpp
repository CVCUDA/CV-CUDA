/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_PYTHON_SHAPE_HPP
#define NVCV_PYTHON_SHAPE_HPP

#include <common/Binding.hpp>
#include <nvcv/TensorShape.hpp>

#include <stdexcept>
#include <type_traits>

namespace nvcvpy {

class Shape : public nanobind::tuple
{
    static PyObject *CreateTuple(size_t size)
    {
        PyObject *tuple = PyTuple_New(static_cast<Py_ssize_t>(size));
        if (tuple == nullptr)
        {
            nanobind::raise_python_error();
        }
        return tuple;
    }

public:
    using nanobind::tuple::tuple;
    using nanobind::tuple::operator[];

    explicit Shape(size_t size)
        : nanobind::tuple(nanobind::steal<nanobind::tuple>(CreateTuple(size)))
    {
    }

    class item_proxy
    {
    public:
        item_proxy(Shape &shape, size_t index)
            : m_shape(shape)
            , m_index(index)
        {
            if (index >= shape.size())
            {
                throw std::out_of_range("tuple index out of range");
            }
        }

        template<class Derived>
        item_proxy &operator=(const nanobind::detail::api<Derived> &value)
        {
            return set(value.derived());
        }

        template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
        item_proxy &operator=(T value)
        {
            nanobind::object converted = nanobind::cast(value);
            return set(converted);
        }

        operator nanobind::handle() const // NOSONAR: reading a subscript proxy relies on implicit conversion.
        {
            return nanobind::handle(PyTuple_GET_ITEM(m_shape.ptr(), m_index));
        }

    private:
        item_proxy &set(nanobind::handle value)
        {
            value.inc_ref();
            if (PyTuple_SetItem(m_shape.ptr(), m_index, value.ptr()) != 0)
            {
                nanobind::raise_python_error();
            }
            return *this;
        }

        Shape &m_shape;
        size_t m_index;
    };

    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    item_proxy operator[](T index)
    {
        if constexpr (std::is_signed_v<T>)
        {
            if (index < 0)
            {
                throw std::out_of_range("tuple index out of range");
            }
        }
        return item_proxy(*this, static_cast<size_t>(index));
    }
};

inline Shape CreateShape(const nvcv::TensorShape &tshape)
{
    const auto &shape = tshape.shape();

    Shape s(static_cast<size_t>(shape.rank()));
    for (int i = 0; i < shape.rank(); ++i)
    {
        PyTuple_SET_ITEM(s.ptr(), i, nanobind::cast(shape[i]).release().ptr());
    }
    return s;
}

inline Shape CreateShape(int64_t length)
{
    Shape s(1);
    PyTuple_SET_ITEM(s.ptr(), 0, nanobind::cast(length).release().ptr());
    return s;
}

inline nvcv::TensorShape CreateNVCVTensorShape(const Shape &shape, nvcv::TensorLayout layout = nvcv::TENSOR_NONE)
{
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (const nanobind::handle dim : shape)
    {
        dims.push_back(nanobind::cast<int64_t>(dim));
    }

    return nvcv::TensorShape(dims.data(), static_cast<int32_t>(dims.size()), layout);
}

inline int64_t LengthIf1D(const Shape &shape)
{
    int64_t largest = 1;
    for (const nanobind::handle dim : shape)
    {
        int64_t length = nanobind::cast<int64_t>(dim);
        if (length > 1)
        {
            if (largest > 1)
            {
                throw std::invalid_argument("Non-supported array shape");
            }
            largest = length;
        }
    }

    return largest;
}

} // namespace nvcvpy

#endif // NVCV_PYTHON_SHAPE_HPP
