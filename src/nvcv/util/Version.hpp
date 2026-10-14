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

#ifndef NVCV_UTIL_VERSION_HPP
#define NVCV_UTIL_VERSION_HPP

#include <cstdint>
#include <iosfwd>
#include <stdexcept>

// WAR for some unwanted macros
#include <sys/types.h>
#ifdef major
#    undef major
#endif
#ifdef minor
#    undef minor
#endif

namespace nvcv::util {

class Version
{
public:
    constexpr explicit Version(int major, int minor, int patch, int tweak = 0)
        : m_code(major * 1000000 + minor * 10000 + patch * 100 + tweak)
    {
        // Major can be > 99, the rest are limited to 99
        if (major < 0 || minor < 0 || minor > 99 || patch < 0 || patch > 99 || tweak < 0 || tweak > 99)
        {
            throw std::invalid_argument("Invalid version code");
        }
    }

    constexpr explicit Version(uint32_t versionCode)
        : m_code(versionCode)
    {
    }

    constexpr int major() const
    {
        return m_code / 1000000;
    }

    constexpr int minor() const
    {
        return (m_code % 1000000) / 10000;
    }

    constexpr int patch() const
    {
        return (m_code % 10000) / 100;
    }

    constexpr int tweak() const
    {
        return m_code % 100;
    }

    constexpr uint32_t code() const
    {
        return m_code;
    }

    // C++20's spaceship operator would help here
    // constexpr bool operator<=>(const Version &that) const = default;

    constexpr bool operator<(const Version &that) const
    {
        return m_code < that.m_code;
    }

    constexpr bool operator<=(const Version &that) const
    {
        return m_code <= that.m_code;
    }

    constexpr bool operator>(const Version &that) const
    {
        return m_code > that.m_code;
    }

    constexpr bool operator>=(const Version &that) const
    {
        return m_code >= that.m_code;
    }

    constexpr bool operator==(const Version &that) const
    {
        return m_code == that.m_code;
    }

    constexpr bool operator!=(const Version &that) const
    {
        return m_code != that.m_code;
    }

    // needs to be public so that type can be passed as non-type template parameter
    int m_code;
};

std::ostream &operator<<(std::ostream &out, const Version &ver);

} // namespace nvcv::util

#endif // NVCV_UTIL_VERSION_HPP
