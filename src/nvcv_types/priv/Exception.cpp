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

#include "Exception.hpp"

#include "Status.hpp"

#include <util/Assert.h>

#include <cstdarg>

namespace nvcv::priv {

Exception::Exception(NVCVStatus code)
    : Exception(code, "%s", "")
{
}

Exception::Exception(NVCVStatus code, const char *fmt, va_list va)
    : m_code(code)
    , m_strbuf{m_buffer, sizeof(m_buffer), m_buffer}
{
    snprintf(m_buffer, sizeof(m_buffer) - 1, "%s: ", GetName(code));

    size_t len = strlen(m_buffer);
    vsnprintf(m_buffer + len, sizeof(m_buffer) - len - 1, fmt, va);

    // Next character written will be appended to m_buffer
    m_strbuf.seekpos(strlen(m_buffer), std::ios_base::out);
}

Exception::Exception(NVCVStatus code, const char *fmt, ...)
    : m_code(code)
    , m_strbuf{m_buffer, sizeof(m_buffer), m_buffer}
{
    va_list va;
    va_start(va, fmt);

    snprintf(m_buffer, sizeof(m_buffer) - 1, "%s: ", GetName(code));

    size_t len = strlen(m_buffer);
    vsnprintf(m_buffer + len, sizeof(m_buffer) - len - 1, fmt, va);

    va_end(va);

    // Next character written will be appended to m_buffer
    m_strbuf.seekpos(strlen(m_buffer), std::ios_base::out);
}

NVCVStatus Exception::code() const
{
    return m_code;
}

const char *Exception::msg() const
{
    // Only return the message part
    const char *out = strchr(m_buffer, ':');
    NVCV_ASSERT(out != nullptr);

    return out += 2; // skip ': '
}

const char *Exception::what() const noexcept
{
    return m_buffer;
}

} // namespace nvcv::priv
