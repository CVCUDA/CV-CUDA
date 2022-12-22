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

#ifndef NVCV_CORE_PRIV_EXCEPTION_HPP
#define NVCV_CORE_PRIV_EXCEPTION_HPP

#include <nvcv/Status.h>

#include <cstring>

#ifdef __GNUC__
#    undef __DEPRECATED
#endif
#include <strstream>

namespace nvcv::priv {

class Exception : public std::exception
{
public:
    explicit Exception(NVCVStatus code, const char *fmt, va_list va);

    explicit Exception(NVCVStatus code, const char *fmt, ...)
#if __GNUC__
        // first argument is actually 'this'
        __attribute__((format(printf, 3, 4)));
#else
        ;
#endif

    explicit Exception(NVCVStatus code);

    NVCVStatus  code() const;
    const char *msg() const;

    const char *what() const noexcept override;

    template<class T>
    Exception &&operator<<(const T &v) &&
    {
        // TODO: must avoid allocating memory from heap, can't use ostringstream
        std::ostream ss(&m_strbuf);
        ss << v << std::flush;
        return std::move(*this);
    }

private:
    NVCVStatus m_code;
    char       m_buffer[NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2];

    class StrBuffer : public std::strstreambuf
    {
    public:
        using std::strstreambuf::seekpos;
        using std::strstreambuf::strstreambuf;
    };

    StrBuffer m_strbuf;
};

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_EXCEPTION_HPP
