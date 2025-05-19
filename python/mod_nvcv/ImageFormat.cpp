/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ImageFormat.hpp"

#include <nvcv/ImageFormat.hpp>

#include <sstream>

// So that pybind can export nvcv::ImageFormat as python enum
namespace std {
template<>
struct underlying_type<nvcv::ImageFormat>
{
    using type = uint64_t;
};
} // namespace std

namespace nvcv {
size_t ComputeHash(const nvcv::ImageFormat &fmt)
{
    return std::hash<uint64_t>()(static_cast<uint64_t>(fmt));
}

} // namespace nvcv

namespace nvcvpy::priv {

static std::string ImageFormatToString(nvcv::ImageFormat fmt)
{
    const char *str = nvcvImageFormatGetName(fmt);

    std::string_view prefix = "NVCV_IMAGE_FORMAT_";

    std::ostringstream out;

    out << "nvcv.";

    if (prefix == str)
    {
        out << "Format." << str + prefix.length();
    }
    else
    {
        prefix = "ImageFormat";
        if (prefix == str)
        {
            out << "Format" << str + prefix.length();
        }
        else
        {
            out << "<Unknown image format: " << str << '>';
        }
    }

    return out.str();
}

void ExportImageFormat(py::module &m)
{
    py::enum_<nvcv::ImageFormat> fmt(m, "Format");

#define DEF(F)     fmt.value(#F, nvcv::FMT_##F);
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) fmt.value("_" #F, nvcv::FMT_##F);

#include "NVCVPythonImageFormatDefs.inc"

#undef DEF
#undef DEF_NUM

    fmt.export_values()
        .def_property_readonly("planes", &nvcv::ImageFormat::numPlanes,
                               "Read-only property that returns the number of planes in the image")
        .def_property_readonly("channels", &nvcv::ImageFormat::numChannels,
                               "Read-only property that returns the number of color channels in the image");

    // Need to do this way because pybind11 doesn't allow enums to have methods.
    fmt.attr("__repr__") = py::cpp_function(&ImageFormatToString, py::name("__repr__"), py::is_method(fmt),
                                            "String representation of the image format.");
}

} // namespace nvcvpy::priv
