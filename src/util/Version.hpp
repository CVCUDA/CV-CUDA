/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVCV_UTIL_VERSION_HPP
#define NVCV_UTIL_VERSION_HPP

#include <compare>
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
    {
        // Major can be > 99, the rest are limited to 99
        if (major < 0 || minor < 0 || minor > 99 || patch < 0 || patch > 99 || tweak < 0 || tweak > 99)
        {
            throw std::invalid_argument("Invalid version code");
        }

        m_code = major * 1000000 + minor * 10000 + patch * 100 + tweak;
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

    constexpr auto operator<=>(const Version &that) const = default;

    // needs to be public so that type can be passed as non-type template parameter
    int m_code;
};

std::ostream &operator<<(std::ostream &out, const Version &ver);

} // namespace nvcv::util

#endif // NVCV_UTIL_VERSION_HPP
