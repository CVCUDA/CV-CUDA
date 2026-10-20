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

#ifndef NVCV_PYTHON_PRIV_SIZE_HPP
#define NVCV_PYTHON_PRIV_SIZE_HPP

#include <common/Binding.hpp>

#include <limits>

namespace nvcvpy::priv {

struct Size2D
{
    int width;
    int height;

    bool operator==(const Size2D &that) const
    {
        return width == that.width && height == that.height;
    }
};

inline size_t ComputeHash(const Size2D &size)
{
    return std::hash<int>{}(size.width) ^ (std::hash<int>{}(size.height) << 1);
}

} // namespace nvcvpy::priv

namespace nanobind::detail {

template<>
struct type_caster<nvcvpy::priv::Size2D>
{
    NB_TYPE_CASTER(nvcvpy::priv::Size2D, const_name("tuple[int, int]"));

    bool from_python(handle src, uint32_t /*flags*/, cleanup_list * /*cleanup*/) noexcept
    {
        object sequence = steal<object>(PySequence_Tuple(src.ptr()));
        if (!sequence.is_valid())
        {
            PyErr_Clear();
            return false;
        }
        if (PyTuple_GET_SIZE(sequence.ptr()) != 2)
        {
            return false;
        }

        PyObject *widthObject  = PyTuple_GET_ITEM(sequence.ptr(), 0);
        PyObject *heightObject = PyTuple_GET_ITEM(sequence.ptr(), 1);
        long      width        = PyLong_AsLong(widthObject);
        if (width == -1 && PyErr_Occurred())
        {
            PyErr_Clear();
            return false;
        }
        long height = PyLong_AsLong(heightObject);
        if (height == -1 && PyErr_Occurred())
        {
            PyErr_Clear();
            return false;
        }
        if (width < std::numeric_limits<int>::min() || width > std::numeric_limits<int>::max()
            || height < std::numeric_limits<int>::min() || height > std::numeric_limits<int>::max())
        {
            return false;
        }

        value = {static_cast<int>(width), static_cast<int>(height)};
        return true;
    }

    static handle from_cpp(const nvcvpy::priv::Size2D &size, rv_policy, cleanup_list *) noexcept
    {
        return make_tuple(size.width, size.height).release();
    }
};

} // namespace nanobind::detail

#endif // NVCV_PYTHON_PRIV_SIZE_HPP
