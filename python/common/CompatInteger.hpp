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

#ifndef CVCUDA_PYTHON_COMMON_COMPAT_INTEGER_HPP
#define CVCUDA_PYTHON_COMMON_COMPAT_INTEGER_HPP

#include <nanobind/nanobind.h>

#include <cstdint>
#include <type_traits>

namespace cvcudapy::compat {

inline bool IsPybindCompatibleIntegralInput(nanobind::handle src, uint32_t flags) noexcept
{
    if (!src)
    {
        return false;
    }
    if (PyLong_Check(src.ptr()))
    {
        return true;
    }
    if (PyFloat_Check(src.ptr()))
    {
        return false;
    }
    if (PyIndex_Check(src.ptr()))
    {
        return true;
    }
    return (flags & static_cast<uint32_t>(nanobind::detail::cast_flags::convert)) && PyNumber_Check(src.ptr());
}

} // namespace cvcudapy::compat

namespace nanobind::detail {

// nanobind deliberately accepts several string-like inputs through
// PyNumber_Long(), while pybind11 limits fallback conversion to objects that
// satisfy PyNumber_Check(). Keep nanobind's optimized integer loaders and add
// only that established input gate.
template<typename T>
struct PybindCompatibleIntegerCaster
{
    NB_INLINE bool from_python(handle src, uint32_t flags, cleanup_list *cleanup) noexcept
    {
        if (!::cvcudapy::compat::IsPybindCompatibleIntegralInput(src, flags))
        {
            return false;
        }

        // Preserve the established __int__ fallback; nanobind 3 only accepts __index__.
        object converted;
        if (!PyLong_Check(src.ptr()) && !PyIndex_Check(src.ptr()))
        {
            converted = steal<object>(PyNumber_Long(src.ptr()));
            if (!converted.is_valid())
            {
                PyErr_Clear();
                return false;
            }
            src = converted;
        }

        if constexpr (std::is_signed_v<T>)
        {
            if constexpr (sizeof(T) == 8)
                return NB_CALL(load_i64)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<int64_t *>(&value));
            else if constexpr (sizeof(T) == 4)
                return NB_CALL(load_i32)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<int32_t *>(&value));
            else if constexpr (sizeof(T) == 2)
                return NB_CALL(load_i16)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<int16_t *>(&value));
            else
                // 8-bit integer conversion, not byte-oriented memory access.
                return NB_CALL(load_i8)(NB_CTX_C(cleanup), src.ptr(), flags,
                                        reinterpret_cast<int8_t *>(&value)); // NOSONAR
        }
        else
        {
            if constexpr (sizeof(T) == 8)
                return NB_CALL(load_u64)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<uint64_t *>(&value));
            else if constexpr (sizeof(T) == 4)
                return NB_CALL(load_u32)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<uint32_t *>(&value));
            else if constexpr (sizeof(T) == 2)
                return NB_CALL(load_u16)(NB_CTX_C(cleanup), src.ptr(), flags, reinterpret_cast<uint16_t *>(&value));
            else
                // 8-bit integer conversion, not byte-oriented memory access.
                return NB_CALL(load_u8)(NB_CTX_C(cleanup), src.ptr(), flags,
                                        reinterpret_cast<uint8_t *>(&value)); // NOSONAR
        }
    }

    NB_INLINE static handle from_cpp(T src, rv_policy, cleanup_list *) noexcept
    {
        if constexpr (std::is_signed_v<T>)
        {
            if constexpr (sizeof(T) <= sizeof(long))
                return PyLong_FromLong(static_cast<long>(src));
            else
                return PyLong_FromLongLong(static_cast<long long>(src));
        }
        else
        {
            if constexpr (sizeof(T) <= sizeof(unsigned long))
                return PyLong_FromUnsignedLong(static_cast<unsigned long>(src));
            else
                return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(src));
        }
    }

    NB_TYPE_CASTER(T, io_name("typing.SupportsInt | typing.SupportsIndex", "int"))
};

#define CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(Type)                  \
    template<>                                                     \
    struct type_caster<Type> : PybindCompatibleIntegerCaster<Type> \
    {                                                              \
    }

CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(signed char);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(unsigned char);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(short);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(unsigned short);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(int);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(unsigned int);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(long);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(unsigned long);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(long long);
CVCUDA_DEFINE_COMPAT_INTEGER_CASTER(unsigned long long);

#undef CVCUDA_DEFINE_COMPAT_INTEGER_CASTER

} // namespace nanobind::detail

#endif // CVCUDA_PYTHON_COMMON_COMPAT_INTEGER_HPP
