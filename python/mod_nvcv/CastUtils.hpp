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

#ifndef NVCV_PYTHON_PRIV_CAST_UTILS_HPP
#define NVCV_PYTHON_PRIV_CAST_UTILS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace nvcvpy::priv {
namespace py = pybind11;

// pybind11 2.10.3 can't convert an item from the input list into another type
// automatically. It won't be able to match the call to current method definition.
// We have to accept std::vector<py::object> and try to cast them manually.

template<typename T>
std::shared_ptr<T> cast_py_object_as(py::object &obj)
{
    py::detail::type_caster<T> caster;
    if (!caster.load(obj, true))
    {
        return {};
    }
    std::shared_ptr<T> buf = caster;
    return buf;
}

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CAST_UTILS_HPP
