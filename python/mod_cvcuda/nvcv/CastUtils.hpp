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

#ifndef NVCV_PYTHON_PRIV_CAST_UTILS_HPP
#define NVCV_PYTHON_PRIV_CAST_UTILS_HPP

#include <common/Binding.hpp>

#include <memory>

namespace nvcvpy::priv {
namespace py = nanobind;

// Convert either an already-wrapped object or an object implementing one of
// ExternalBuffer's supported interchange protocols.

template<typename T>
std::shared_ptr<T> cast_py_object_as(py::object &obj)
{
    if (py::isinstance<T>(obj))
    {
        return py::cast<std::shared_ptr<T>>(obj, false);
    }

    std::shared_ptr<T> out = std::make_shared<T>();
    return out->load(obj.ptr()) ? out : std::shared_ptr<T>{};
}

} // namespace nvcvpy::priv

#endif // NVCV_PYTHON_PRIV_CAST_UTILS_HPP
