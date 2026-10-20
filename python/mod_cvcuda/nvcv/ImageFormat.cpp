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

#include "ImageFormat.hpp"

#include <nvcv/ImageFormat.hpp>

#include <new>
#include <sstream>
#include <string_view>
#include <vector>

namespace nvcv {
size_t ComputeHash(const nvcv::ImageFormat &fmt)
{
    return std::hash<uint64_t>()(static_cast<uint64_t>(fmt));
}

} // namespace nvcv

namespace nvcvpy::priv {

using namespace py::literals;

struct RegisteredImageFormat
{
    NVCVImageFormat  value;
    std::string_view name;
};

static std::vector<RegisteredImageFormat> &RegisteredImageFormats()
{
    static std::vector<RegisteredImageFormat> formats;
    return formats;
}

static std::string ImageFormatToString(nvcv::ImageFormat fmt)
{
    const char *str = nvcvImageFormatGetName(static_cast<NVCVImageFormat>(fmt));

    auto starts_with = [](const char *s, std::string_view p)
    {
        return std::string_view{s}.compare(0, p.size(), p) == 0;
    };

    std::ostringstream out;
    out << "nvcv.";

    if (constexpr std::string_view nvcvPrefix = "NVCV_IMAGE_FORMAT_"; starts_with(str, nvcvPrefix))
    {
        out << "Format." << str + nvcvPrefix.length();
    }
    else
    {
        constexpr std::string_view cppPrefix = "ImageFormat";
        if (starts_with(str, cppPrefix))
        {
            out << "Format" << str + cppPrefix.length();
        }
        else
        {
            out << "<Unknown image format: " << str << '>';
        }
    }

    return out.str();
}

static std::string ImageFormatName(nvcv::ImageFormat fmt)
{
    auto value = static_cast<NVCVImageFormat>(fmt);
    for (const RegisteredImageFormat &entry : RegisteredImageFormats())
    {
        if (entry.value == value)
        {
            return std::string(entry.name);
        }
    }
    return "???";
}

static std::string ImageFormatStr(nvcv::ImageFormat fmt)
{
    return "Format." + ImageFormatName(fmt);
}

void ExportImageFormat(py::module_ &m)
{
    py::class_<nvcv::ImageFormat> fmt(m, "Format", py::is_weak_referenceable());
    py::dict                      members;

    RegisteredImageFormats().clear();

#define REGISTER_FORMAT(NAME, VALUE)                                                     \
    {                                                                                    \
        RegisteredImageFormats().push_back({static_cast<NVCVImageFormat>(VALUE), NAME}); \
        py::object value = py::cast(VALUE);                                              \
        fmt.attr(NAME)   = value;                                                        \
        m.attr(NAME)     = value;                                                        \
        members[NAME]    = value;                                                        \
    }
#define DEF(F)     REGISTER_FORMAT(#F, nvcv::FMT_##F)
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) REGISTER_FORMAT("_" #F, nvcv::FMT_##F)

#include "NVCVPythonImageFormatDefs.inc"

    // FMT_NONE is a sentinel ("no format"), not part of the generated defs.
    // Expose it so signature defaults like `format=FMT_NONE` render as the
    // valid Python expression `nvcv.Format.NONE` instead of an unrepresentable
    // `<Unknown image format>` string.
    DEF(NONE)

#undef DEF
#undef DEF_NUM
#undef REGISTER_FORMAT

    std::ostringstream doc;
    doc << "Members:";
    for (const RegisteredImageFormat &entry : RegisteredImageFormats())
    {
        doc << "\n\n  " << entry.name;
    }

    fmt.attr("__doc__")     = doc.str();
    fmt.attr("__members__") = members;
    fmt.def(py::new_([](uint64_t value) { return nvcv::ImageFormat{value}; }), "value"_a)
        .def_prop_ro("name", &ImageFormatName)
        .def_prop_ro("value", [](nvcv::ImageFormat value) { return static_cast<uint64_t>(value); })
        .def_prop_ro("planes", &nvcv::ImageFormat::numPlanes,
                     "Read-only property that returns the number of planes in the image")
        .def_prop_ro("channels", &nvcv::ImageFormat::numChannels,
                     "Read-only property that returns the number of color channels in the image")
        .def("__repr__", &ImageFormatToString, "String representation of the image format.")
        .def("__str__", &ImageFormatStr)
        .def("__int__", [](nvcv::ImageFormat value) { return static_cast<uint64_t>(value); })
        .def("__index__", [](nvcv::ImageFormat value) { return static_cast<uint64_t>(value); })
        .def("__getstate__", [](nvcv::ImageFormat value) { return static_cast<uint64_t>(value); })
        .def("__setstate__", [](nvcv::ImageFormat *self, uint64_t state) { new (self) nvcv::ImageFormat{state}; })
        .def("__hash__", [](nvcv::ImageFormat value) { return nvcv::ComputeHash(value); })
        .def(
            "__eq__", [](nvcv::ImageFormat a, nvcv::ImageFormat b) { return a == b; }, py::is_operator())
        .def(
            "__ne__", [](nvcv::ImageFormat a, nvcv::ImageFormat b) { return a != b; }, py::is_operator());
}

} // namespace nvcvpy::priv
