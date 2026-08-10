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

#include "DataType.hpp"

#include <common/Assert.hpp>
#include <common/String.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <complex>
#include <sstream>
#include <string_view>

// DataType is implicitly convertible from/to numpy types such as
// numpy.int8, numpy.complex64, numpy.dtype, etc.

namespace nvcv {
size_t ComputeHash(const nvcv::DataType &dtype)
{
    return std::hash<uint64_t>()(static_cast<uint64_t>(dtype));
}

} // namespace nvcv

namespace nvcvpy::priv {

namespace {

template<class T>
struct IsComplex : std::false_type
{
};

template<class T>
struct IsComplex<std::complex<T>> : std::true_type
{
};

// Marker type for numpy float16 (no native C++ 16-bit float equivalent)
struct Float16Tag
{
};

// Type traits to abstract differences between standard types and Float16Tag
template<class T>
struct DTypeTraits
{
    static bool matches(const py::dtype &dtbase)
    {
        return dtbase.equal(py::dtype::of<T>());
    }

    static py::dtype create()
    {
        return py::dtype::of<T>();
    }

    static constexpr int itemsize()
    {
        return sizeof(T);
    }

    static int bits_per_component(int itemsize_param)
    {
        return itemsize_param * 8;
    }

    static nvcv::DataKind infer_kind()
    {
        if (IsComplex<T>::value)
            return nvcv::DataKind::COMPLEX;
        else if (std::is_floating_point_v<T>)
            return nvcv::DataKind::FLOAT;
        else if (std::is_signed_v<T>)
            return nvcv::DataKind::SIGNED;
        else if (std::is_unsigned_v<T>)
            return nvcv::DataKind::UNSIGNED;
        else
        {
            NVCV_ASSERT(!"Invalid type");
        }
    }
};

// Specialization for Float16Tag
template<>
struct DTypeTraits<Float16Tag>
{
    static bool matches(const py::dtype &dtbase)
    {
        return dtbase.itemsize() == 2 && dtbase.kind() == 'f';
    }

    static py::dtype create()
    {
        return py::dtype("e");
    }

    static constexpr int itemsize()
    {
        return 2;
    }

    static int bits_per_component(int /*itemsize_param*/)
    {
        return 16;
    }

    static nvcv::DataKind infer_kind()
    {
        return nvcv::DataKind::FLOAT;
    }
};

template<class T>
bool FindDataType(const py::dtype &dt, nvcv::DataType *dtype)
{
    int       nchannels = 1;
    py::dtype dtbase    = dt;
    if (hasattr(dt, "subdtype"))
    {
        py::object obj = dt.attr("subdtype");
        if (!obj.equal(py::none()))
        {
            auto subdt = py::cast<py::tuple>(obj);
            if (subdt.size() != 2)
            {
                // Malformed? subdtype tuple must have 2 elements.
                return false;
            }
            dtbase = subdt[0];

            // only 1d shape for now
            auto shape = py::cast<py::tuple>(subdt[1]);
            if (shape.size() >= 2)
            {
                return false;
            }

            nchannels = shape.empty() ? 1 : py::cast<int>(shape[0]);
        }
    }

    auto itemsize = static_cast<int>(dtbase.itemsize());

    // Use traits to check if this type matches
    if (!DTypeTraits<T>::matches(dtbase))
    {
        return false;
    }

    // get the data kind from the traits
    auto                dataKind = DTypeTraits<T>::infer_kind();
    nvcv::PackingParams pp       = {};
    pp.byteOrder                 = nvcv::ByteOrder::MSB;

    switch (nchannels)
    {
    case 1:
        pp.swizzle = nvcv::Swizzle::S_X000;
        break;
    case 2:
        pp.swizzle = nvcv::Swizzle::S_XY00;
        break;
    case 3:
        pp.swizzle = nvcv::Swizzle::S_XYZ0;
        break;
    case 4:
        pp.swizzle = nvcv::Swizzle::S_XYZW;
        break;
    default:
        NVCV_ASSERT(!"Invalid number of channels");
    }

    // Use traits to get bits per component
    for (int i = 0; i < nchannels; ++i)
    {
        pp.bits[i] = DTypeTraits<T>::bits_per_component(itemsize);
    }

    nvcv::Packing packing = MakePacking(pp);

    // Finally, infer the data type
    NVCV_ASSERT(dtype != nullptr);
    *dtype = nvcv::DataType{dataKind, packing};
    return true;
}

// clang-format off
using SupportedBaseTypes = std::tuple<
      Float16Tag,  // Explicit marker for numpy float16
      std::complex<float>,
      std::complex<double>,
      float, double,
      uint8_t, int8_t,
      uint16_t, int16_t,
      uint32_t, int32_t,
      uint64_t, int64_t
>;

// clang-format on

template<class... TT>
std::optional<nvcv::DataType> SelectDataType(std::tuple<TT...>, const py::dtype &dt)
{
    nvcv::DataType dtype;

    if ((FindDataType<TT>(dt, &dtype) || ...))
    {
        return dtype;
    }
    else
    {
        return std::nullopt;
    }
}

template<class T>
bool FindDType(T *, const nvcv::DataType &dtype, py::dtype *dt)
{
    int            nchannels = dtype.numChannels();
    int            itemsize  = dtype.bitsPerPixel() / 8;
    nvcv::DataKind dataKind  = dtype.dataKind();

    if (DTypeTraits<T>::itemsize() != itemsize / nchannels)
    {
        return false;
    }

    if (DTypeTraits<T>::infer_kind() != dataKind)
    {
        return false;
    }

    NVCV_ASSERT(dt != nullptr);
    *dt = DTypeTraits<T>::create();

    if (nchannels > 1)
    {
        *dt = py::dtype(util::ConcatString(nchannels, dt->char_()));
    }
    return true;
}

template<class... TT>
py::dtype SelectDType(std::tuple<TT...>, const nvcv::DataType &dtype)
{
    py::dtype dt;

    (FindDType((TT *)nullptr, dtype, &dt) || ...);

    return dt;
}

} // namespace

//

std::optional<nvcv::DataType> ToNVCVDataType(const py::dtype &dt)
{
    return SelectDataType(SupportedBaseTypes(), dt);
}

py::dtype ToDType(nvcv::DataType dtype)
{
    return SelectDType(SupportedBaseTypes(), dtype);
}

static std::string DataTypeToString(nvcv::DataType type)
{
    const char *str = nvcvDataTypeGetName(type);

    std::string_view prefix = "NVCV_DATA_TYPE_";

    std::ostringstream out;

    out << "nvcv.";

    auto starts_with = [](const char *s, std::string_view p)
    {
        return std::string_view{s}.rfind(p, 0) == 0;
    };

    if (starts_with(str, prefix))
    {
        out << "Type." << str + prefix.length();
    }
    else
    {
        prefix = "DataType";
        if (starts_with(str, prefix))
        {
            out << "Type" << str + prefix.length();
        }
        else
        {
            out << "<Unknown type: " << str << '>';
        }
    }

    return out.str();
}

void ExportDataType(py::module &m)
{
    py::class_<nvcv::DataType> type(m, "Type");

#define DEF(F)     type.def_readonly_static(#F, &nvcv::TYPE_##F);
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) type.def_readonly_static("_" #F, &nvcv::TYPE_##F);

#include "NVCVPythonDataTypeDefs.inc"

#undef DEF
#undef DEF_NUM

    type.def_property_readonly("components", &nvcv::DataType::numChannels);
    type.def(py::init<nvcv::DataType>());
    type.def(py::init<>());

    type.def("__repr__", &DataTypeToString);
    type.def(
        "__eq__", [](const nvcv::DataType &a, const nvcv::DataType &b) { return a == b; }, py::is_operator());
    type.def(
        "__ne__", [](const nvcv::DataType &a, const nvcv::DataType &b) { return a != b; }, py::is_operator());
    type.def(
        "__lt__", [](const nvcv::DataType &a, const nvcv::DataType &b) { return a < b; }, py::is_operator());
    // Type values surface to Python as numpy.dtype (see the DataType type_caster
    // below), so a Type constructed via Type(...) must hash identically to the
    // equivalent numpy.dtype to stay consistent with __eq__ across both forms.
    // Fall back to the packed value for the no-dtype sentinel so hashing never
    // dereferences a null object.
    type.def(
        "__hash__",
        [](const nvcv::DataType &a)
        {
            if (py::dtype dt = ToDType(a))
            {
                return PyObject_Hash(dt.ptr());
            }
            return static_cast<Py_hash_t>(std::hash<uint64_t>{}(static_cast<uint64_t>(a)));
        },
        "Return a value-based hash matching the equivalent numpy.dtype, so Type can be used as a dict key.");

    py::implicitly_convertible<py::dtype, nvcv::DataType>();
}

} // namespace nvcvpy::priv

namespace pybind11::detail {

namespace priv = nvcvpy::priv;

bool type_caster<nvcv::DataType>::load(handle src, bool)
{
    const type_info *tinfo = get_type_info(typeid(nvcv::DataType));
    if (Py_TYPE(src.ptr()) == tinfo->type)
    {
        value_and_holder vh = reinterpret_cast<instance *>(src.ptr())->get_value_and_holder();
        value               = *vh.template holder<nvcv::DataType *>();
        return true;
    }
    else
    {
        PyObject *ptr = nullptr;
        if (detail::npy_api::get().PyArray_DescrConverter_(src.ptr(), &ptr) == 0 || !ptr)
        {
            PyErr_Clear();
            return false;
        }
        dtype dt = dtype::from_args(reinterpret_steal<object>(ptr));

        if (std::optional<nvcv::DataType> _dt = priv::ToNVCVDataType(dt))
        {
            value = *_dt;
            return true;
        }
        else
        {
            return false;
        }
    }
}

handle type_caster<nvcv::DataType>::cast(nvcv::DataType type, return_value_policy /* policy */, handle /*parent */)
{
    dtype dt = priv::ToDType(type);

    // without the increfs, we get 6 of these...
    // *** Reference count error detected: an attempt was made to deallocate the dtype 6 (I) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 3 (h) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 4 (H) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 1 (b) ***
    // *** Reference count error detected: an attempt was made to deallocate the dtype 2 (B) ***
    // and also a segfault when using the nvcv struct types in some tests.
    // It *really* looks like we have to incref here.

    if (dt)
    {
        Py_INCREF(dt.ptr());
    }

    return dt;
}
} // namespace pybind11::detail
