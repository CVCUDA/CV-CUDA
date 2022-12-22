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

#include "DataType.hpp"

#include <common/Assert.hpp>
#include <common/String.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <complex>
#include <sstream>

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

    int itemsize = dtbase.itemsize();

    if (dtbase.equal(py::dtype::of<T>()))
    {
        nvcv::DataKind dataKind;
        if (IsComplex<T>::value)
        {
            nchannels = 2;
            itemsize /= 2;
            dataKind = nvcv::DataKind::FLOAT;
        }
        else if (std::is_floating_point<T>::value)
        {
            dataKind = nvcv::DataKind::FLOAT;
        }
        else if (std::is_signed<T>::value)
        {
            dataKind = nvcv::DataKind::SIGNED;
        }
        else if (std::is_unsigned<T>::value)
        {
            dataKind = nvcv::DataKind::UNSIGNED;
        }
        else
        {
            NVCV_ASSERT(!"Invalid type");
        }

        // Infer the packing
        nvcv::PackingParams pp = {};

        pp.byteOrder = nvcv::ByteOrder::MSB;

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
        for (int i = 0; i < nchannels; ++i)
        {
            pp.bits[i] = static_cast<int>(itemsize * 8);
        }
        nvcv::Packing packing = MakePacking(pp);

        // Finally, infer the data type
        NVCV_ASSERT(dtype != nullptr);
        *dtype = nvcv::DataType{dataKind, packing};
        return true;
    }
    else
    {
        return false;
    }
}

// clang-format off
using SupportedBaseTypes = std::tuple<
      std::complex<float>, // must come before float
      std::complex<double>, // must come before double
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

std::optional<nvcv::DataType> ToDataType(const py::dtype &dt)
{
    return SelectDataType(SupportedBaseTypes(), dt);
}

template<class T>
bool FindDType(T *, const nvcv::DataType &dtype, py::dtype *dt)
{
    int nchannels = dtype.numChannels();
    int itemsize  = dtype.bitsPerPixel() / 8;

    if (sizeof(T) != itemsize / nchannels)
    {
        return false;
    }

    nvcv::DataKind dataKind = dtype.dataKind();

    if ((std::is_floating_point_v<T> && dataKind == nvcv::DataKind::FLOAT)
        || (std::is_integral_v<T> && std::is_signed_v<T> && dataKind == nvcv::DataKind::SIGNED)
        || (std::is_integral_v<T> && std::is_unsigned_v<T> && dataKind == nvcv::DataKind::UNSIGNED))
    {
        NVCV_ASSERT(dt != nullptr);

        *dt = py::dtype::of<T>();

        // data type has multiple components?
        if (nvcv::cuda::NumElements<T> != nchannels)
        {
            // Create a dtype with multiple components too, with shape argument
            *dt = py::dtype(util::FormatString("%d%c", nchannels, dt->char_()));
        }
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
bool FindDType(std::complex<T> *, const nvcv::DataType &dtype, py::dtype *dt)
{
    nvcv::DataKind dataKind  = dtype.dataKind();
    int            nchannels = dtype.numChannels();
    int            itemsize  = dtype.bitsPerPixel() / 8;

    if (dataKind == nvcv::DataKind::FLOAT && sizeof(std::complex<T>) == itemsize && nchannels == 2)
    {
        NVCV_ASSERT(dt != nullptr);
        *dt = py::dtype::of<std::complex<T>>();
        return true;
    }
    else
    {
        return false;
    }
}

template<class... TT>
py::dtype SelectDType(std::tuple<TT...>, const nvcv::DataType &dtype)
{
    py::dtype dt;

    (FindDType((TT *)nullptr, dtype, &dt) || ...);

    return dt;
}

py::dtype ToDType(nvcv::DataType dtype)
{
    return SelectDType(SupportedBaseTypes(), dtype);
}

} // namespace

static std::string DataTypeToString(nvcv::DataType type)
{
    const char *str = nvcvDataTypeGetName(type);

    const char *prefix = "NVCV_DATA_TYPE_";

    std::ostringstream out;

    out << "nvcv.";

    if (strncmp(str, prefix, strlen(prefix)) == 0)
    {
        out << "Type." << str + strlen(prefix);
    }
    else
    {
        prefix = "DataType";
        if (strncmp(str, prefix, strlen(prefix)) == 0)
        {
            out << "Type" << str + strlen(prefix);
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
    type.def(py::self == py::self);
    type.def(py::self != py::self);
    type.def(py::self < py::self);

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

        if (std::optional<nvcv::DataType> _dt = priv::ToDataType(dt))
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
