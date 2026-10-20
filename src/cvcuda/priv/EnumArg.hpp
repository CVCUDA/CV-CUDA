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

#ifndef CVCUDA_PRIV_ENUMARG_HPP
#define CVCUDA_PRIV_ENUMARG_HPP

#include <nvcv/Exception.hpp>

#include <cstring>
#include <type_traits>

namespace cvcuda::priv {

/**
 * @brief Reject a C-API enum argument whose value lies outside its enum's range.
 *
 * A C caller can put any integer into an enum-typed parameter; the ABI only
 * moves bytes. In C++ an unscoped enum with no fixed underlying type has a
 * value range derived from its enumerators, not from int, so *reading* such a
 * parameter is undefined once the value falls outside that range -- which is
 * what -fsanitize=enum reports as "load of value N, which is not a valid value
 * for type E". The read happens in the callee, so declaring the parameter as
 * the enum does not prevent it.
 *
 * Inspect the object representation instead of loading the enum, and throw if
 * the value is out of range. Once this returns, the argument holds a value the
 * enum can represent, so every later read of it is well defined and needs no
 * further care -- callers keep using the parameter directly.
 *
 * @param arg      The enum-typed parameter, inspected without being loaded.
 * @param maxValue Largest value the enum may legitimately hold: the highest
 *                 enumerator for a sequential enum, or the union of every flag
 *                 for a bitmask.
 * @param what     Argument name used in the error message.
 */
template<class Enum>
void CheckEnumArg(const Enum &arg, long long maxValue, const char *what)
{
    static_assert(std::is_enum_v<Enum>, "CheckEnumArg is only meaningful for enum arguments");

    using Raw = std::underlying_type_t<Enum>;
    static_assert(sizeof(Raw) == sizeof(Enum), "enum must share its underlying type's object size");

    Raw raw;
    std::memcpy(&raw, &arg, sizeof(raw));

    const auto value = static_cast<long long>(raw);
    if (value < 0 || value > maxValue)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid %s: %lld", what, value);
    }
}

} // namespace cvcuda::priv

#endif // CVCUDA_PRIV_ENUMARG_HPP
