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

#ifndef NVCV_UTIL_STRING_HPP
#define NVCV_UTIL_STRING_HPP

#include <ostream>
#include <streambuf>
#include <string_view>

namespace nvcv::util {

void ReplaceAllInline(char *strBuffer, int bufferSize, std::string_view what, std::string_view replace) noexcept;

class FixedBufferStreamBuf : public std::streambuf
{
public:
    FixedBufferStreamBuf() = default;
    FixedBufferStreamBuf(char *buffer, std::streamsize bufferSize);

    FixedBufferStreamBuf(const FixedBufferStreamBuf &)            = delete;
    FixedBufferStreamBuf &operator=(const FixedBufferStreamBuf &) = delete;
    FixedBufferStreamBuf(FixedBufferStreamBuf &&)                 = delete;
    FixedBufferStreamBuf &operator=(FixedBufferStreamBuf &&)      = delete;

    void reset(char *buffer, std::streamsize bufferSize) noexcept;

    std::streampos seekpos(std::streampos pos, std::ios_base::openmode which = std::ios_base::out) noexcept override;

protected:
    int_type overflow(int_type ch) noexcept override;
    int      sync() noexcept override;

private:
    char           *m_buffer     = nullptr;
    std::streamsize m_bufferSize = 0;
};

class BufferOStream : public std::ostream
{
public:
    BufferOStream(char *buffer, int len);
    ~BufferOStream() override;

private:
    FixedBufferStreamBuf m_buf;
};

} // namespace nvcv::util

#endif // NVCV_UTIL_STRING_HPP
