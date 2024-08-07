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

#ifndef NVCV_PYTHON_CONTAINER_HPP
#define NVCV_PYTHON_CONTAINER_HPP

#include "CAPI.hpp"
#include "Cache.hpp"
#include "Resource.hpp"

#include <common/Assert.hpp>
#include <pybind11/pybind11.h>

namespace nvcvpy {

namespace py = pybind11;

class Container
    : public Resource
    , public ICacheItem
{
public:
    Container(py::object o)
        : Resource(o)
    {
    }

    explicit Container()
    {
        PyObject *raw_obj = capi().Container_Create(this);
        CheckCAPIError();
        NVCV_ASSERT(raw_obj != nullptr);
        py::object temp = py::reinterpret_steal<py::object>(raw_obj);
        new (static_cast<Resource *>(this)) Resource(temp);
    }
};

} // namespace nvcvpy

#endif // NVCV_PYTHON_CONTAINER_HPP
