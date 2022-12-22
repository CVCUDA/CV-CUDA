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

#include "String.hpp"

#include "Assert.h"

#include <cstring>

namespace nvcv::util {

void ReplaceAllInline(char *strBuffer, int bufferSize, std::string_view what, std::string_view replace) noexcept
{
    NVCV_ASSERT(strBuffer != nullptr);
    NVCV_ASSERT(bufferSize >= 1);

    std::string_view str(strBuffer);

    std::string_view::size_type pos;
    while ((pos = str.find(what, 0)) != std::string_view::npos)
    {
        // First create some space to write 'replace'.

        // Number of bytes to move
        int count_orig = str.size() - pos - what.size();
        // Make sure we won't write past end of buffer
        int count = std::min<int>(pos + replace.size() + count_orig, bufferSize) - replace.size() - pos;
        NVCV_ASSERT(count >= 0);
        // Since buffers might overlap, let's use memmove
        memmove(strBuffer + pos + replace.size(), strBuffer + pos + what.size(), count);

        // Now copy the new string, replacing 'what'
        replace.copy(strBuffer + pos, replace.size());

        // Let's set strBuffer/set to where next search must start so that we don't search into the
        // replaced string.
        strBuffer += pos + replace.size();
        str = std::string_view(strBuffer, count);

        strBuffer[str.size()] = '\0'; // make sure strBuffer is zero-terminated
    }
}

BufferOStream::BufferOStream(char *buffer, int len)
    : m_buf(buffer, len, buffer)
{
    this->init(&m_buf);
}

BufferOStream::~BufferOStream()
{
    // Make sure the buffer is 0-terminated and flushed
    *this << '\0' << std::flush;
}

} // namespace nvcv::util
