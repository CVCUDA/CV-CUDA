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

#ifndef NVCV_PYTHON_PYUTIL_HPP
#define NVCV_PYTHON_PYUTIL_HPP

#include <nvcv/DataType.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <sstream>

namespace nvcvpy::util {
namespace py = pybind11;

// Adds a method to an existing class
template<class T, class Func, class... Extra>
void DefClassMethod(const char *name, Func &&f, const Extra &...extra)
{
    py::type class_ = py::type::of<T>();

    // got from pt::class_<...>::def
    py::cpp_function cf(py::method_adaptor<py::type>(std::forward<Func>(f)), py::name(name), py::is_method(class_),
                        py::sibling(py::getattr(class_, name, py::none())), extra...);
    py::detail::add_class_method(class_, name, std::move(cf));
}

// Adds a static method to an existing class
template<class T, class Func, class... Extra>
void DefClassStaticMethod(const char *name, Func &&f, const Extra &...extra)
{
    py::type class_ = py::type::of<T>();

    // got from pt::class_<...>::def
    py::cpp_function cf(std::forward<Func>(f), py::name(name), py::scope(class_),
                        py::sibling(py::getattr(class_, name, py::none())), extra...);
    class_.attr(cf.name()) = py::staticmethod(cf);
}

void RegisterCleanup(py::module &m, std::function<void()> fn);

std::string GetFullyQualifiedName(py::handle h);

// We need to support array_interface formats and some numpy typestr
// formats, in addition to formats described in PEP 3118.
// This function will handle them all. pybind11 as of v2.9.1 isn't capable
// of doing that.
//
// ref: https://peps.python.org/pep-3118/#additions-to-the-struct-string-syntax
// ref: https://numpy.org/doc/stable/reference/arrays.interface.html#object.__array_interface__
// ref: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
py::dtype ToDType(const std::string &str);
py::dtype ToDType(const py::buffer_info &info);

} // namespace nvcvpy::util

#endif // NVCV_PYTHON_PYUTIL_HPP
