/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CVCUDA_PYTHON_COMMON_BINDING_HPP
#define CVCUDA_PYTHON_COMMON_BINDING_HPP

#include <common/CompatInteger.hpp>
#include <cvcuda/Types.h>
#include <nanobind/eval.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nvcv/BorderType.h>
#include <nvcv/ColorSpec.h>
#include <nvcv/RoundMode.h>

#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nanobind {

using error_already_set   = python_error;
using return_value_policy = rv_policy;

namespace detail {

// Preserve nanobind's native maps for registered values, then use the
// pybind11-compatible wrapper value only when the native lookup misses.
template<class Enum>
struct enum_compat_type_caster
{
    NB_INLINE bool from_python(handle src, uint32_t flags, cleanup_list *cleanup) noexcept
    {
        if (int64_t result; NB_CALL(enum_from_python)(NB_CTX_C(cleanup), &typeid(Enum), src.ptr(), &result, flags))
        {
            value = static_cast<Enum>(result);
            return true;
        }

        if (handle enumType = nanobind::type<Enum>();
            enumType.is_valid() && PyObject_TypeCheck(src.ptr(), reinterpret_cast<PyTypeObject *>(enumType.ptr())))
        {
            PyObject *valueObject = PyObject_GetAttrString(src.ptr(), "_value_");
            if (valueObject == nullptr)
            {
                PyErr_Clear();
                return false;
            }

            using Underlying = std::underlying_type_t<Enum>;
            if constexpr (std::is_signed_v<Underlying>)
            {
                long long result = PyLong_AsLongLong(valueObject);
                Py_DECREF(valueObject);
                if (result == -1 && PyErr_Occurred())
                {
                    PyErr_Clear();
                    return false;
                }
                if (result < std::numeric_limits<Underlying>::lowest()
                    || result > std::numeric_limits<Underlying>::max())
                {
                    return false;
                }
                value = static_cast<Enum>(static_cast<Underlying>(result));
            }
            else
            {
                unsigned long long result = PyLong_AsUnsignedLongLong(valueObject);
                Py_DECREF(valueObject);
                if (result == static_cast<unsigned long long>(-1) && PyErr_Occurred())
                {
                    PyErr_Clear();
                    return false;
                }
                if (result > std::numeric_limits<Underlying>::max())
                {
                    return false;
                }
                value = static_cast<Enum>(static_cast<Underlying>(result));
            }
            return true;
        }
        return false;
    }

    NB_INLINE static handle from_cpp(Enum src, rv_policy, cleanup_list *cleanup) noexcept
    {
        PyObject *result // NOSONAR: returned as a nanobind handle, whose constructor requires a non-const PyObject*.
            = NB_CALL(enum_from_cpp)(NB_CTX_C(cleanup), &typeid(Enum), static_cast<int64_t>(src));
        if (result != nullptr)
        {
            return result;
        }
        if (!PyErr_ExceptionMatches(PyExc_ValueError))
        {
            return nullptr;
        }

        // Stock nanobind rejects unregistered enum values. pybind11 instead
        // returns a fresh wrapper, so use the compatibility constructor only
        // on this cold path while retaining nanobind's forward-map fast path.
        handle enumType = nanobind::type<Enum>();
        if (!enumType.is_valid())
        {
            return nullptr;
        }
        PyErr_Clear();

        PyObject *valueObject;
        using Underlying = std::underlying_type_t<Enum>;
        if constexpr (std::is_signed_v<Underlying>)
        {
            valueObject = PyLong_FromLongLong(static_cast<long long>(static_cast<Underlying>(src)));
        }
        else
        {
            valueObject = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(static_cast<Underlying>(src)));
        }
        if (valueObject == nullptr)
        {
            return nullptr;
        }

        PyObject *newFunction = PyObject_GetAttrString(enumType.ptr(), "__new__");
        if (newFunction == nullptr)
        {
            Py_DECREF(valueObject);
            return nullptr;
        }

        std::array<PyObject *, 2> arguments = {enumType.ptr(), valueObject};
        result = PyObject_Vectorcall(newFunction, arguments.data(), arguments.size(), nullptr);
        Py_DECREF(newFunction);
        Py_DECREF(valueObject);
        return result;
    }

    NB_TYPE_CASTER(Enum, const_name<Enum>())
};

#define CVCUDA_PYTHON_ENUM_COMPAT_CASTER(Enum)               \
    template<>                                               \
    struct type_caster<Enum> : enum_compat_type_caster<Enum> \
    {                                                        \
    }

// nanobind's generic enum caster cannot be partially specialized further, so
// opt the public CV-CUDA enum types into the compatibility caster explicitly.
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVAdaptiveThresholdType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVBorderType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVChannelManip);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVClockFormat);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVColorConversionCode);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVColorSpec);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVConnectivityType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVHistogramEqMethod);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVInterpolationType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVLabelType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVLabelMaskType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVPairwiseMatcherType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVMorphologyType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVNormType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVRemapMapValueType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVRoundMode);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVSIFTFlagType);
CVCUDA_PYTHON_ENUM_COMPAT_CASTER(NVCVThresholdType);

#undef CVCUDA_PYTHON_ENUM_COMPAT_CASTER

inline object enum_compat_install_helper(handle scope)
{
    constexpr const char *kInstallAttribute = "_cvcuda_enum_compat_install";

    if (hasattr(scope, kInstallAttribute))
    {
        return scope.attr(kInstallAttribute);
    }

    dict namespace_;
    namespace_["__name__"] = "nanobind";
    exec(R"(
import operator as _operator


class _CVCUDAEnumCompatBase:
    def __init__(self, value):
        pass

    @staticmethod
    def _coerce_value(cls, value):
        try:
            value = _operator.index(value)
        except TypeError:
            # pybind11's integral caster accepts the SupportsInt fallback but
            # deliberately rejects the built-in floating-point conversion.
            if isinstance(value, (float, str, bytes, bytearray)):
                raise
            convert = getattr(value, "__int__", None)
            if convert is None:
                raise
            value = convert()
            if not isinstance(value, int):
                raise TypeError("__int__ returned non-int")

        if not cls._cvcuda_enum_min <= value <= cls._cvcuda_enum_max:
            raise TypeError("enum value is outside the underlying integer range")
        return value

    def __new__(cls, value):
        value = _CVCUDAEnumCompatBase._coerce_value(cls, value)
        self = object.__new__(cls)
        self._value_ = value
        member = cls._value2member_map_.get(value)
        self._name_ = "???" if member is None else member._name_
        return self

    @property
    def name(self):
        member = type(self)._value2member_map_.get(self._value_)
        return "???" if member is None else member._name_

    @property
    def value(self):
        return self._value_

    def __hash__(self):
        return hash(self._value_)

    def __bool__(self):
        return True

    def __int__(self):
        return self._value_

    def __index__(self):
        return self._value_

    def __reduce_ex__(self, protocol):
        return type(self), (self._value_,)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            other = other._value_
        elif not type(self)._cvcuda_enum_integer_comparable:
            return NotImplemented
        return self._value_ == other

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __str__(self):
        return type(self).__name__ + "." + self.name

    def __repr__(self):
        return "<" + type(self).__name__ + "." + self.name + ": " + str(self._value_) + ">"


class _CVCUDAEnumCompatMeta(type):
    @property
    def __members__(cls):
        return dict(cls._member_map_)

    def __dir__(cls):
        names = super().__dir__()
        seen = set(names)
        for name in (*cls.__members__, "name", "value"):
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @property
    def __doc__(cls):
        base_doc = cls._cvcuda_enum_compat_doc
        doc = "" if base_doc is None else base_doc + "\n\n"
        doc += "Members:"
        for name, member in cls.__members__.items():
            doc += "\n\n  " + name
            if member.__doc__ is not None:
                doc += " : " + member.__doc__
        return doc


def _cvcuda_enum_compat_install(cls, minimum, maximum, integer_comparable, arithmetic):
    cls._cvcuda_enum_compat_doc = cls.__doc__
    cls._cvcuda_enum_min = minimum
    cls._cvcuda_enum_max = maximum
    cls._cvcuda_enum_integer_comparable = integer_comparable

    # Both are heap types with the same instance layout. Keeping the original
    # type object preserves nanobind's native caster registration and maps.
    cls.__bases__ = (_CVCUDAEnumCompatBase,)
    cls.__class__ = _CVCUDAEnumCompatMeta
    cls.__orig_bases__ = (object,)

    for name in (
        "__new__", "__init__", "name", "value", "__int__", "__index__",
        "__hash__", "__bool__", "__reduce_ex__", "__eq__", "__ne__",
        "__str__", "__repr__",
    ):
        if name in _CVCUDAEnumCompatBase.__dict__:
            setattr(cls, name, _CVCUDAEnumCompatBase.__dict__[name])

    if arithmetic:
        cls.__lt__ = lambda self, other: self._value_ < other
        cls.__gt__ = lambda self, other: self._value_ > other
        cls.__le__ = lambda self, other: self._value_ <= other
        cls.__ge__ = lambda self, other: self._value_ >= other
        cls.__and__ = lambda self, other: self._value_ & other
        cls.__rand__ = lambda self, other: other & self._value_
        cls.__or__ = lambda self, other: self._value_ | other
        cls.__ror__ = lambda self, other: other | self._value_
        cls.__xor__ = lambda self, other: self._value_ ^ other
        cls.__rxor__ = lambda self, other: other ^ self._value_
        cls.__invert__ = lambda self: ~self._value_
)",
         namespace_);

    object install                = namespace_["_cvcuda_enum_compat_install"];
    scope.attr(kInstallAttribute) = install;
    return install;
}

template<class Extra>
auto enum_compat_constructor_extra(const Extra &extra)
{
    if constexpr (std::is_same_v<std::decay_t<Extra>, is_arithmetic>)
    {
        return std::tuple<>();
    }
    else
    {
        return std::tie(extra);
    }
}

template<class... Extra>
inline constexpr bool enum_compat_arithmetic = (std::is_same_v<std::decay_t<Extra>, is_arithmetic> || ...);

template<class Enum, class... Extra>
inline constexpr bool enum_compat_integer_equality
    = std::is_convertible_v<Enum, std::underlying_type_t<Enum>> || enum_compat_arithmetic<Extra...>;

template<class Enum>
void enum_compat_install(handle scope, handle enumType, bool integerComparable, bool arithmetic)
{
    using Underlying = std::underlying_type_t<Enum>;
    enum_compat_install_helper(scope)(enumType, int_(std::numeric_limits<Underlying>::lowest()),
                                      int_(std::numeric_limits<Underlying>::max()), bool_(integerComparable),
                                      bool_(arithmetic));
}

} // namespace detail

template<class Enum>
class enum_compat_builder final : public enum_<Enum>
{
    using Base = enum_<Enum>;

public:
    template<class... Extra>
    enum_compat_builder(handle scope, const char *name, bool integerComparable, bool arithmetic, const Extra &...extra)
        : Base(scope, name, extra...)
        , m_scope(borrow<object>(scope))
    {
        // ptr() rather than *this: the helper only needs the Python object, and
        // passing the builder would slice it down to its handle base.
        detail::enum_compat_install<Enum>(scope, this->ptr(), integerComparable, arithmetic);
    }

    enum_compat_builder &value(const char *name, Enum value, const char *doc = nullptr)
    {
        Base::value(name, value, doc);
        return *this;
    }

    enum_compat_builder &str_value(const char *name, Enum value, const char *strValue, const char *doc = nullptr)
    {
        Base::str_value(name, value, strValue, doc);
        return *this;
    }

    enum_compat_builder &export_values()
    {
        dict members = borrow<dict>(this->attr("_member_map_"));
        for (auto [name, member] : members)
        {
            setattr(m_scope, name, member);
        }
        return *this;
    }

private:
    object m_scope;
};

// pybind11 enums behave like their underlying integer for conversion,
// comparison, hashing, and repr while remaining distinct Python classes.
// nanobind's default enum intentionally follows enum.Enum more closely, so
// restore the established CV-CUDA behavior at each enum declaration.
template<class Enum, class... Extra>
enum_compat_builder<Enum> enum_compat(handle scope, const char *name, const Extra &...extra)
{
    static_assert(std::is_base_of_v<detail::enum_compat_type_caster<Enum>, detail::type_caster<Enum>>,
                  "enum_compat types must opt into enum_compat_type_caster");

    auto           constructorExtra   = std::tuple_cat(detail::enum_compat_constructor_extra(extra)...);
    constexpr bool kArithmetic        = detail::enum_compat_arithmetic<Extra...>;
    constexpr bool kIntegerComparable = detail::enum_compat_integer_equality<Enum, Extra...>;

    return std::apply([scope, name](const auto &...keptExtra)
                      { return enum_compat_builder<Enum>(scope, name, kIntegerComparable, kArithmetic, keptExtra...); },
                      constructorExtra);
}

// Small compatibility wrappers for the buffer and NumPy dtype facilities used
// by CV-CUDA. nanobind deliberately keeps these out of its core object API.
struct buffer_info // NOSONAR: mirrors pybind11's public aggregate; only the Py_buffer lifetime owner stays private.
{
    struct view
    {
        Py_buffer value{};

        view() = default;

        // The destructor releases the Py_buffer, so copying one would release
        // it twice. Instances are only ever shared through m_view.
        view(const view &)            = delete;
        view &operator=(const view &) = delete;
        view(view &&)                 = delete;
        view &operator=(view &&)      = delete;

        ~view()
        {
            if (value.obj != nullptr)
            {
                PyBuffer_Release(&value);
            }
        }
    };

    buffer_info() = default;

    explicit buffer_info(handle h)
        : m_view(std::make_shared<view>())
    {
        if (PyObject_GetBuffer(h.ptr(), &m_view->value, PyBUF_STRIDES | PyBUF_FORMAT) != 0)
        {
            raise_python_error();
        }

        const Py_buffer &v = m_view->value;
        ptr                = v.buf;
        itemsize           = v.itemsize;
        size               = itemsize == 0 ? 0 : v.len / itemsize;
        ndim               = v.ndim;
        format             = v.format == nullptr ? "B" : v.format;
        readonly           = v.readonly != 0;
        shape.assign(v.shape, v.shape + v.ndim);

        if (v.strides != nullptr)
        {
            strides.assign(v.strides, v.strides + v.ndim);
        }
        else
        {
            strides.resize(v.ndim);
            ssize_t stride = itemsize;
            for (ssize_t i = v.ndim; i-- > 0;)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
    }

    // The untyped pointer mirrors Py_buffer::buf; the buffer protocol carries
    // its element type in `format`, not in the pointer.
    buffer_info(void *data, ssize_t elementSize, std::string elementFormat, ssize_t dimensions, // NOSONAR
                std::vector<ssize_t> dimensionsShape, std::vector<ssize_t> dimensionsStrides, bool isReadonly = false)
        : ptr(data)
        , itemsize(elementSize)
        , size(1)
        , ndim(dimensions)
        , format(std::move(elementFormat))
        , shape(std::move(dimensionsShape))
        , strides(std::move(dimensionsStrides))
        , readonly(isReadonly)
    {
        for (ssize_t extent : shape)
        {
            size *= extent;
        }
    }

    void                *ptr      = nullptr; // NOSONAR: mirrors Py_buffer::buf, see the constructor above.
    ssize_t              itemsize = 0;
    ssize_t              size     = 0;
    ssize_t              ndim     = 0;
    std::string          format;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    bool                 readonly = false;

private:
    std::shared_ptr<view> m_view;
};

class NB_EXPORT_SHARED buffer : public object
{
    NB_OBJECT(buffer, object, "collections.abc.Buffer", PyObject_CheckBuffer)

public:
    buffer_info request() const
    {
        return buffer_info(*this);
    }
};

inline bool numpy_dtype_check(PyObject *o) noexcept
{
    return o != nullptr;
}

class NB_EXPORT_SHARED numpy_dtype : public object // NOSONAR: forwarding ctors come from nanobind's macro.
{
    NB_OBJECT_DEFAULT(numpy_dtype, object, "numpy.dtype", numpy_dtype_check)

public:
    explicit numpy_dtype(handle spec)
    {
        object value = module_::import_("numpy").attr("dtype")(spec);
        m_ptr        = value.release().ptr();
    }

    explicit numpy_dtype(const char *spec)
        : numpy_dtype(str(spec))
    {
    }

    explicit numpy_dtype(const std::string &spec)
        : numpy_dtype(str(spec.c_str()))
    {
    }

    static numpy_dtype from_args(handle spec)
    {
        return numpy_dtype(spec);
    }

    template<class T>
    static numpy_dtype of()
    {
        if constexpr (std::is_same_v<T, bool>)
            return numpy_dtype("bool");
        else if constexpr (std::is_same_v<T, float>)
            return numpy_dtype("float32");
        else if constexpr (std::is_same_v<T, double>)
            return numpy_dtype("float64");
        else if constexpr (std::is_same_v<T, std::complex<float>>)
            return numpy_dtype("complex64");
        else if constexpr (std::is_same_v<T, std::complex<double>>)
            return numpy_dtype("complex128");
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
            return numpy_dtype((std::string("int") + std::to_string(sizeof(T) * 8)).c_str());
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
            return numpy_dtype((std::string("uint") + std::to_string(sizeof(T) * 8)).c_str());
        else
            static_assert(!sizeof(T), "Unsupported NumPy dtype");
    }

    ssize_t itemsize() const
    {
        return cast<ssize_t>(attr("itemsize"));
    }

    char kind() const
    {
        return cast<std::string>(attr("kind"))[0];
    }

    char char_() const
    {
        return cast<std::string>(attr("char"))[0];
    }
};

template<class T>
T reinterpret_borrow(handle h)
{
    return borrow<T>(h);
}

template<class T>
T reinterpret_steal(handle h)
{
    return steal<T>(h);
}

} // namespace nanobind

#endif // CVCUDA_PYTHON_COMMON_BINDING_HPP
