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

#include "Rect.hpp"

#include <common/String.hpp>
#include <nvcv/Rect.h>

static std::ostream &operator<<(std::ostream &out, const NVCVRectI &rc)
{
    return out << "RectI(x=" << rc.x << ",y=" << rc.y << ",width=" << rc.width << ",height=" << rc.height << ')';
}

namespace nvcvpy::priv {

void ExportRect(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVRectI>(m, "RectI", "RectI")
        .def(py::init([]() { return NVCVRectI{}; }), "Default constructor")
        .def(py::init(
                 [](int x, int y, int w, int h)
                 {
                     NVCVRectI r;
                     r.x      = x;
                     r.y      = y;
                     r.width  = w;
                     r.height = h;
                     return r;
                 }),
             "x"_a, "y"_a, "width"_a, "height"_a, "Constructor with x, y, width, height parameters")
        .def_readwrite("x", &NVCVRectI::x, "X coordinate of the rectangle's top-left corner")
        .def_readwrite("y", &NVCVRectI::y, "Y coordinate of the rectangle's top-left corner")
        .def_readwrite("width", &NVCVRectI::width, "Width of the rectangle")
        .def_readwrite("height", &NVCVRectI::height, "Height of the rectangle")
        .def("__repr__", &util::ToString<NVCVRectI>, "Returns a string representation of the rectangle");
}

} // namespace nvcvpy::priv
