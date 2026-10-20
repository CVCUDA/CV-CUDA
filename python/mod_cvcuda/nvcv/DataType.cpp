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
#include <common/Binding.hpp>
#include <common/String.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>

#include <complex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

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
    static bool matches(const py::numpy_dtype &dtbase)
    {
        return dtbase.equal(py::numpy_dtype::of<T>());
    }

    static py::numpy_dtype create()
    {
        return py::numpy_dtype::of<T>();
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
    static bool matches(const py::numpy_dtype &dtbase)
    {
        return dtbase.equal(create());
    }

    static py::numpy_dtype create()
    {
        return py::numpy_dtype("e");
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
bool CreateDataType(int itemSize, int numChannels, nvcv::DataType *dtype)
{
    if (numChannels < 1 || numChannels > 4)
    {
        return false;
    }

    auto          dataKind = DTypeTraits<T>::infer_kind();
    nvcv::Packing packing  = MakeHomogeneousPacking(DTypeTraits<T>::bits_per_component(itemSize), numChannels);

    NVCV_ASSERT(dtype != nullptr);
    *dtype = nvcv::DataType{dataKind, packing};
    return true;
}

template<class T>
bool FindDataType(const py::numpy_dtype &dt, nvcv::DataType *dtype)
{
    int             numChannels = 1;
    py::numpy_dtype baseType    = dt;
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
            baseType = subdt[0];

            // only 1d shape for now
            auto shape = py::cast<py::tuple>(subdt[1]);
            if (shape.size() >= 2)
            {
                return false;
            }

            numChannels = shape.empty() ? 1 : py::cast<int>(shape[0]);
        }
    }

    if (!DTypeTraits<T>::matches(baseType))
    {
        return false;
    }

    return CreateDataType<T>(static_cast<int>(baseType.itemsize()), numChannels, dtype);
}

bool KindMatches(char kind, nvcv::DataKind dataKind)
{
    switch (dataKind)
    {
    case nvcv::DataKind::UNSIGNED:
        return kind == 'u';
    case nvcv::DataKind::SIGNED:
        return kind == 'i';
    case nvcv::DataKind::FLOAT:
        return kind == 'f';
    case nvcv::DataKind::COMPLEX:
        return kind == 'c';
    default:
        return false;
    }
}

template<class T>
bool FindDataType(char kind, int itemSize, nvcv::DataType *dtype)
{
    if (DTypeTraits<T>::itemsize() != itemSize || !KindMatches(kind, DTypeTraits<T>::infer_kind()))
    {
        return false;
    }

    return CreateDataType<T>(itemSize, 1, dtype);
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

// Raised when a built-in data type cannot be constructed; indicates corrupted
// tables rather than bad user input, hence a dedicated type.
class BuiltinDataTypeInitError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// NumPy interns its built-in scalar dtype objects. Keep one strong reference
// to each supported scalar so the common cvcuda.Type.* path is a pointer
// lookup instead of rebuilding and comparing several NumPy dtype objects on
// every argument conversion. The cache is deliberately immortal: dtype
// conversion remains reachable during Python module teardown.
class BuiltinDataTypeCache
{
public:
    template<class... TT>
    explicit BuiltinDataTypeCache(std::tuple<TT...>)
    {
        m_entries.reserve(sizeof...(TT));
        (add<TT>(), ...);
    }

    bool find(PyObject *object, nvcv::DataType *result) const noexcept
    {
        for (const Entry &entry : m_entries)
        {
            if (entry.object.ptr() == object)
            {
                *result = entry.value;
                return true;
            }
        }
        return false;
    }

private:
    struct Entry
    {
        py::numpy_dtype object;
        nvcv::DataType  value;
    };

    template<class T>
    void add()
    {
        nvcv::DataType value;
        if (!CreateDataType<T>(DTypeTraits<T>::itemsize(), 1, &value))
        {
            throw BuiltinDataTypeInitError("Failed to initialize a built-in data type");
        }
        m_entries.push_back({DTypeTraits<T>::create(), value});
    }

    std::vector<Entry> m_entries;
};

BuiltinDataTypeCache &BuiltinDataTypes()
{
    static auto *cache = new BuiltinDataTypeCache(SupportedBaseTypes{}); // NOSONAR: deliberately immortal
    return *cache;
}

template<class... TT>
std::optional<nvcv::DataType> SelectDataType(std::tuple<TT...>, const py::numpy_dtype &dt)
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

template<class... TT>
std::optional<nvcv::DataType> SelectDataType(std::tuple<TT...>, char kind, int itemSize)
{
    nvcv::DataType dtype;

    if ((FindDataType<TT>(kind, itemSize, &dtype) || ...))
    {
        return dtype;
    }
    else
    {
        return std::nullopt;
    }
}

template<class T>
bool FindDType(T *, const nvcv::DataType &dtype, py::numpy_dtype *dt)
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
        *dt = py::numpy_dtype(util::ConcatString(nchannels, dt->char_()));
    }
    return true;
}

// Direct equivalent of the DataType -> numpy dtype -> DataType round trip:
// same match predicate as FindDType, same reconstruction as FindDataType, but
// without building the intermediate Python object. Wrapping a buffer runs this
// on every call, and materializing a numpy dtype there dominated the cost.
template<class T>
bool FindNormalizedDataType(T *, const nvcv::DataType &dtype, nvcv::DataType *out)
{
    int nchannels = dtype.numChannels();
    int itemsize  = dtype.bitsPerPixel() / 8;

    if (nchannels < 1 || nchannels > 4)
    {
        return false;
    }

    int componentSize = itemsize / nchannels;

    if (DTypeTraits<T>::itemsize() != componentSize)
    {
        return false;
    }

    if (DTypeTraits<T>::infer_kind() != dtype.dataKind())
    {
        return false;
    }

    return CreateDataType<T>(componentSize, nchannels, out);
}

template<class... TT>
std::optional<nvcv::DataType> SelectNormalizedDataType(std::tuple<TT...>, const nvcv::DataType &dtype)
{
    nvcv::DataType out;

    if ((FindNormalizedDataType((TT *)nullptr, dtype, &out) || ...))
    {
        return out;
    }
    else
    {
        return std::nullopt;
    }
}

template<class... TT>
py::numpy_dtype SelectDType(std::tuple<TT...>, const nvcv::DataType &dtype)
{
    py::numpy_dtype dt;

    (FindDType((TT *)nullptr, dtype, &dt) || ...);

    return dt;
}

} // namespace

//

nvcv::Packing MakeHomogeneousPacking(int bits, int numChannels)
{
    NVCV_ASSERT(numChannels >= 1 && numChannels <= 4);
    // These are the canonical MSB packings selected by MakePacking with alignment 0.
    static constexpr nvcv::Packing common[][4] = {
        { nvcv::Packing::X8,   nvcv::Packing::X8_Y8,    nvcv::Packing::X8_Y8_Z8,     nvcv::Packing::X8_Y8_Z8_W8},
        {nvcv::Packing::X16, nvcv::Packing::X16_Y16, nvcv::Packing::X16_Y16_Z16, nvcv::Packing::X16_Y16_Z16_W16},
        {nvcv::Packing::X32, nvcv::Packing::X32_Y32, nvcv::Packing::X32_Y32_Z32, nvcv::Packing::X32_Y32_Z32_W32},
        {nvcv::Packing::X64, nvcv::Packing::X64_Y64, nvcv::Packing::X64_Y64_Z64, nvcv::Packing::X64_Y64_Z64_W64},
    };
    switch (bits)
    {
    case 8:
        return common[0][numChannels - 1];
    case 16:
        return common[1][numChannels - 1];
    case 32:
        return common[2][numChannels - 1];
    case 64:
        return common[3][numChannels - 1];
    default:
        break;
    }

    nvcv::PackingParams pp = {};
    pp.byteOrder           = nvcv::ByteOrder::MSB;
    static constexpr nvcv::Swizzle swizzles[]
        = {nvcv::Swizzle::S_X000, nvcv::Swizzle::S_XY00, nvcv::Swizzle::S_XYZ0, nvcv::Swizzle::S_XYZW};
    pp.swizzle = swizzles[numChannels - 1];
    std::fill_n(pp.bits.begin(), numChannels, bits);
    return nvcv::MakePacking(pp);
}

std::optional<nvcv::DataType> ToNVCVDataType(const py::numpy_dtype &dt)
{
    return SelectDataType(SupportedBaseTypes(), dt);
}

std::optional<nvcv::DataType> ToNVCVDataType(char kind, int itemSize)
{
    return SelectDataType(SupportedBaseTypes(), kind, itemSize);
}

std::optional<nvcv::DataType> NormalizeDataType(const nvcv::DataType &dtype)
{
    return SelectNormalizedDataType(SupportedBaseTypes(), dtype);
}

py::numpy_dtype ToDType(nvcv::DataType dtype)
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

void ExportDataType(py::module_ &m)
{
    // Build this while module initialization can report Python exceptions.
    // Later caster lookups are allocation-free and noexcept.
    (void)BuiltinDataTypes();

    py::class_<PyDataType> type(m, "Type", py::is_weak_referenceable());

#define DEF(F)     type.def_prop_ro_static(#F, [](py::handle) { return nvcv::TYPE_##F; });
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) type.def_prop_ro_static("_" #F, [](py::handle) { return nvcv::TYPE_##F; });

#include "NVCVPythonDataTypeDefs.inc"

#undef DEF
#undef DEF_NUM

    type.def_prop_ro("components", [](const PyDataType &self) { return self.value.numChannels(); });
    type.def(py::new_([]() { return PyDataType{}; }));
    type.def(py::new_([](nvcv::DataType value) { return PyDataType{value}; }));

    type.def("__repr__", [](const PyDataType &self) { return DataTypeToString(self.value); });
    type.def(
        "__eq__", [](const PyDataType &a, nvcv::DataType b) { return a.value == b; }, py::is_operator());
    type.def(
        "__ne__", [](const PyDataType &a, nvcv::DataType b) { return a.value != b; }, py::is_operator());
    type.def(
        "__lt__", [](const PyDataType &a, nvcv::DataType b) { return a.value < b; }, py::is_operator());
    // Type values surface to Python as numpy.dtype (see the DataType type_caster
    // below), so a Type constructed via Type(...) must hash identically to the
    // equivalent numpy.dtype to stay consistent with __eq__ across both forms.
    // Fall back to the packed value for the no-dtype sentinel so hashing never
    // dereferences a null object.
    type.def(
        "__hash__",
        [](const PyDataType &a)
        {
            if (py::numpy_dtype dt = ToDType(a.value))
            {
                return PyObject_Hash(dt.ptr());
            }
            return static_cast<Py_hash_t>(std::hash<uint64_t>{}(static_cast<uint64_t>(a.value)));
        },
        "Return a value-based hash matching the equivalent numpy.dtype, so Type can be used as a dict key.");
}

} // namespace nvcvpy::priv

namespace nanobind::detail {

namespace priv = nvcvpy::priv;

bool type_caster<nvcv::DataType>::from_python(handle src, uint32_t, cleanup_list *) noexcept
{
    if (nanobind::isinstance<priv::PyDataType>(src))
    {
        value = nanobind::inst_ptr<priv::PyDataType>(src)->value;
        return true;
    }

    if (priv::BuiltinDataTypes().find(src.ptr(), &value))
    {
        return true;
    }

    try
    {
        numpy_dtype dt = numpy_dtype::from_args(src);
        if (std::optional<nvcv::DataType> converted = priv::ToNVCVDataType(dt))
        {
            value = *converted;
            return true;
        }
    }
    catch (const python_error &)
    {
        PyErr_Clear();
    }
    return false;
}

handle type_caster<nvcv::DataType>::from_cpp(nvcv::DataType type, rv_policy, cleanup_list *) noexcept
{
    try
    {
        return priv::ToDType(type).release();
    }
    catch (...)
    {
        return {};
    }
}
} // namespace nanobind::detail
