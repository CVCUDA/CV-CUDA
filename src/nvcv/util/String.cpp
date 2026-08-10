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

#include "String.hpp"

#include "Assert.h"

#include <algorithm>
#include <cstring>
#include <ios>

namespace nvcv::util {

void ReplaceAllInline(char *strBuffer, int bufferSize, std::string_view what, std::string_view replace) noexcept
{
    if (strBuffer == nullptr || what.empty() || bufferSize <= 0)
    {
        return;
    }

    auto *bufferEnd = strBuffer + bufferSize;
    auto *nulPos    = std::find(strBuffer, bufferEnd, '\0');
    if (nulPos == bufferEnd)
    {
        nulPos  = bufferEnd - 1;
        *nulPos = '\0';
    }

    char *searchStart = strBuffer;
    while (searchStart < nulPos)
    {
        auto *foundPos = std::search(searchStart, nulPos, what.begin(), what.end());
        if (foundPos == nulPos)
        {
            return;
        }

        const char *tailStart       = foundPos + what.size();
        auto        tailSize        = nulPos - tailStart;
        auto        replacementRoom = bufferEnd - foundPos - 1;
        size_t      replacementSize = std::min(replace.size(), static_cast<size_t>(replacementRoom));
        char       *tailWritePos    = foundPos + replacementSize;
        auto        tailRoom        = bufferEnd - tailWritePos - 1;
        auto        movedTailSize   = std::min(tailSize, tailRoom);

        std::memmove(tailWritePos, tailStart, movedTailSize);
        if (replacementSize > 0)
        {
            std::memcpy(foundPos, replace.data(), replacementSize);
        }
        nulPos      = tailWritePos + movedTailSize;
        *nulPos     = '\0';
        searchStart = tailWritePos;
    }
}

FixedBufferStreamBuf::FixedBufferStreamBuf(char *buffer, std::streamsize bufferSize)
{
    reset(buffer, bufferSize);
}

void FixedBufferStreamBuf::reset(char *buffer, std::streamsize bufferSize) noexcept
{
    m_buffer     = buffer;
    m_bufferSize = bufferSize;

    if (m_buffer != nullptr && m_bufferSize > 0)
    {
        setp(m_buffer, m_buffer + m_bufferSize);
        *m_buffer = '\0';
    }
}

std::streampos FixedBufferStreamBuf::seekpos(std::streampos pos, std::ios_base::openmode which) noexcept
{
    auto offset = static_cast<std::streamoff>(pos);
    if ((which & std::ios_base::out) == 0 || m_buffer == nullptr || offset < 0 || offset >= m_bufferSize)
    {
        return std::streampos{std::streamoff{-1}};
    }

    setp(m_buffer, m_buffer + m_bufferSize);
    pbump(static_cast<int>(offset));
    return std::streampos{offset};
}

FixedBufferStreamBuf::int_type FixedBufferStreamBuf::overflow(int_type ch) noexcept
{
    if (traits_type::eq_int_type(ch, traits_type::eof()))
    {
        return traits_type::not_eof(ch);
    }

    return traits_type::eof();
}

int FixedBufferStreamBuf::sync() noexcept
{
    if (m_buffer != nullptr && m_bufferSize > 0 && pptr() < epptr())
    {
        *pptr() = '\0';
    }
    return 0;
}

BufferOStream::BufferOStream(char *buffer, int len)
    : m_buf(buffer, len)
{
    this->init(&m_buf);
}

BufferOStream::~BufferOStream()
{
    // Make sure the buffer is 0-terminated and flushed
    *this << '\0' << std::flush;
}

} // namespace nvcv::util
