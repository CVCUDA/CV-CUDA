/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void ReplaceAllInline(char *strBuffer, int bufferSize, const char *what, const char *replace) noexcept
{
    if (strBuffer == nullptr || what == nullptr || replace == nullptr || bufferSize <= 0)
    {
        return;
    }

    size_t whatSize    = std::strlen(what);
    size_t replaceSize = std::strlen(replace);
    size_t strSize     = std::strlen(strBuffer);

    char *searchStart    = strBuffer;
    char *writePos       = nullptr;
    char *endOfNewString = nullptr;
    char *endPos         = strBuffer + bufferSize - 1; //to make sure we do not overflow.

    while (searchStart < strBuffer + strSize)
    {
        char *foundPos = std::strstr(searchStart, what);
        if (foundPos == nullptr)
        {
            // No more occurrences of 'what' found
            return;
        }
        searchStart += replaceSize; // update for next token

        ptrdiff_t sizeOfRest = 0;
        // Move string after token only if there is data after the token.
        if (foundPos + (replaceSize - 1) < endPos)
        {
            char     *restOfString = (foundPos + whatSize); // string after the what token.
            ptrdiff_t moveAmount   = static_cast<std::size_t>(
                replaceSize);                     // how far from beginning of token to move the rest of the string.
            writePos   = foundPos + moveAmount;     // where to start writing the rest of the string.
            sizeOfRest = std::strlen(restOfString); //size of rest of string.

            // Move string after token
            // check for overflow we just want to write to buffer size
            if (writePos + sizeOfRest > endPos)
            {
                sizeOfRest = endPos - writePos;
            }
            NVCV_ASSERT(writePos <= endPos);
            NVCV_ASSERT(writePos + (sizeOfRest - 1) <= endPos);
            std::memmove(writePos, restOfString,
                         sizeOfRest); // move the remainder of the string to allow for replacement of what.
        }
        // Replace token
        // check for overflow
        if (foundPos + replaceSize > endPos)
        {
            replaceSize = endPos - foundPos;
        }
        NVCV_ASSERT(foundPos <= endPos);
        NVCV_ASSERT(foundPos + (replaceSize - 1) <= endPos);
        std::memmove(foundPos, replace, replaceSize); // replace the found token with the replacement string.
        endOfNewString  = std::max(foundPos + replaceSize,
                                   writePos + sizeOfRest); // update the end position to the new end of the string.
        *endOfNewString = '\0';                            // Null-terminate the output in case token is last.
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
