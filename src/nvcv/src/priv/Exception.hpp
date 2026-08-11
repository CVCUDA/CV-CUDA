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

#ifndef NVCV_CORE_PRIV_EXCEPTION_HPP
#define NVCV_CORE_PRIV_EXCEPTION_HPP

#include "Status.hpp"

#include <nvcv/Status.h>
#include <nvcv/detail/Format.hpp>
#include <nvcv/util/String.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

namespace nvcv::priv {

class Exception : public std::exception
{
public:
    explicit Exception(NVCVStatus code, const char *fmt, va_list va);

    explicit Exception(NVCVStatus code, const char *msg);

    template<size_t N, class... Args>
    explicit Exception(NVCVStatus code, const char (&fmt)[N], Args &&...args)
        : m_code(code)
    {
        initStreamBuffer();
        formatMessage(fmt, std::forward<Args>(args)...);
    }

    explicit Exception(NVCVStatus code);

    Exception(const Exception &that) noexcept;
    Exception(Exception &&that) noexcept;
    Exception &operator=(const Exception &that) noexcept;
    Exception &operator=(Exception &&that) noexcept;

    ~Exception() noexcept override;

    NVCVStatus  code() const;
    const char *msg() const;

    const char *what() const noexcept override;

    template<class T>
    Exception &&operator<<(const T &v) &&
    {
        // REVISIT: must avoid allocating memory from heap, can't use ostringstream
        std::ostream ss(&m_strbuf);
        ss << v << std::flush;
        return std::move(*this);
    }

private:
    void copyFrom(const Exception &that) noexcept;
    void initStreamBuffer() noexcept;
    void resetStreamPosition() noexcept;

    template<size_t N, class... Args>
    void formatMessage(const char (&fmt)[N], Args &&...args)
    {
        detail::FormatTo(m_buffer.data(), m_buffer.size(), "%s: ", GetName(m_code));

        size_t len = std::char_traits<char>::length(m_buffer.data());
        detail::FormatTo(m_buffer.data() + len, m_buffer.size() - len, fmt, std::forward<Args>(args)...);

        resetStreamPosition();
    }

    NVCVStatus                                                m_code   = NVCV_ERROR_INTERNAL;
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2> m_buffer = {};

    util::FixedBufferStreamBuf m_strbuf;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_EXCEPTION_HPP
