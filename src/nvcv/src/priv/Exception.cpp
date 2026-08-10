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

#include "Exception.hpp"

#include "Status.hpp"

#include <nvcv/util/Assert.h>

#include <cstdarg>

namespace nvcv::priv {

Exception::Exception(NVCVStatus code)
    : Exception(code, "")
{
}

Exception::Exception(NVCVStatus code, const char *fmt, va_list va)
    : m_code(code)
{
    initStreamBuffer();

    detail::FormatTo(m_buffer.data(), m_buffer.size(), "%s: ", GetName(code));

    size_t len = std::char_traits<char>::length(m_buffer.data());
    detail::VFormatTo(m_buffer.data() + len, m_buffer.size() - len, fmt, va);

    // Next character written will be appended to m_buffer
    m_strbuf.seekpos(std::char_traits<char>::length(m_buffer.data()), std::ios_base::out);
}

Exception::Exception(NVCVStatus code, const char *msg)
    : m_code(code)
{
    initStreamBuffer();
    formatMessage("%s", msg != nullptr ? msg : "");
}

Exception::Exception(const Exception &that) noexcept
{
    initStreamBuffer();
    copyFrom(that);
}

Exception::Exception(Exception &&that) noexcept
    : Exception(static_cast<const Exception &>(that))
{
}

Exception &Exception::operator=(const Exception &that) noexcept
{
    copyFrom(that);
    return *this;
}

Exception &Exception::operator=(Exception &&that) noexcept
{
    copyFrom(that);
    return *this;
}

Exception::~Exception() noexcept
{
    m_buffer.back() = '\0';
}

void Exception::copyFrom(const Exception &that) noexcept
{
    m_code = that.m_code;
    std::memcpy(m_buffer.data(), that.m_buffer.data(), m_buffer.size());
    resetStreamPosition();
}

void Exception::initStreamBuffer() noexcept
{
    m_strbuf.reset(m_buffer.data(), static_cast<std::streamsize>(m_buffer.size()));
}

void Exception::resetStreamPosition() noexcept
{
    m_strbuf.seekpos(std::char_traits<char>::length(m_buffer.data()), std::ios_base::out);
}

NVCVStatus Exception::code() const
{
    return m_code;
}

const char *Exception::msg() const
{
    // Only return the message part
    const char *out = strchr(m_buffer.data(), ':');
    NVCV_ASSERT(out != nullptr);

    return out + 2; // skip ': '
}

const char *Exception::what() const noexcept
{
    return m_buffer.data();
}

} // namespace nvcv::priv
