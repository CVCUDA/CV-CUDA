/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_DETAIL_FORMAT_HPP
#define NVCV_DETAIL_FORMAT_HPP

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <ostream>
#include <streambuf>
#include <string>
#include <type_traits>
#include <utility>

namespace nvcv { namespace detail {

class FixedBufferStreamBuf : public std::streambuf
{
public:
    FixedBufferStreamBuf(char *buffer, std::size_t bufferSize)
        : m_buffer(buffer)
        , m_bufferSize(bufferSize)
    {
        if (m_buffer != nullptr && m_bufferSize > 0)
        {
            setp(m_buffer, m_buffer + m_bufferSize - 1);
            *m_buffer = '\0';
        }
    }

protected:
    int sync() noexcept override
    {
        if (m_buffer != nullptr && m_bufferSize > 0)
        {
            if (pptr() < epptr())
            {
                *pptr() = '\0';
            }
            else
            {
                m_buffer[m_bufferSize - 1] = '\0';
            }
        }
        return 0;
    }

    int_type overflow(int_type ch) noexcept override
    {
        return traits_type::eq_int_type(ch, traits_type::eof()) ? traits_type::not_eof(ch) : traits_type::eof();
    }

private:
    char       *m_buffer     = nullptr;
    std::size_t m_bufferSize = 0;
};

class FixedBufferOStream
{
public:
    FixedBufferOStream(char *buffer, std::size_t bufferSize)
        : m_buf(buffer, bufferSize)
        , m_out(&m_buf)
    {
    }

    std::ostream &get()
    {
        return m_out;
    }

private:
    FixedBufferStreamBuf m_buf;
    std::ostream         m_out;
};

enum class PrintfLength
{
    kNone,
    kHH,
    kH,
    kL,
    kLL,
    kJ,
    kZ,
    kT,
    kLongDouble
};

struct PrintfSpec
{
    bool left      = false;
    bool showpos   = false;
    bool space     = false;
    bool alt       = false;
    bool zero      = false;
    int  width     = -1;
    int  precision = -1;

    PrintfLength length     = PrintfLength::kNone;
    char         conversion = '\0';
};

inline bool IsDigit(char ch)
{
    return ch >= '0' && ch <= '9';
}

inline int ParseInt(const char *&fmt)
{
    int value = 0;
    while (IsDigit(*fmt))
    {
        value = value * 10 + (*fmt - '0');
        ++fmt;
    }
    return value;
}

inline const char *ParsePrintfSpec(const char *fmt, PrintfSpec &spec)
{
    for (;;)
    {
        switch (*fmt)
        {
        case '-':
            spec.left = true;
            ++fmt;
            break;
        case '+':
            spec.showpos = true;
            ++fmt;
            break;
        case ' ':
            spec.space = true;
            ++fmt;
            break;
        case '#':
            spec.alt = true;
            ++fmt;
            break;
        case '0':
            spec.zero = true;
            ++fmt;
            break;
        default:
            goto parse_width;
        }
    }

parse_width:
    if (IsDigit(*fmt))
    {
        spec.width = ParseInt(fmt);
    }

    if (*fmt == '.')
    {
        ++fmt;
        spec.precision = IsDigit(*fmt) ? ParseInt(fmt) : 0;
    }

    switch (*fmt)
    {
    case 'h':
        ++fmt;
        if (*fmt == 'h')
        {
            ++fmt;
            spec.length = PrintfLength::kHH;
        }
        else
        {
            spec.length = PrintfLength::kH;
        }
        break;
    case 'l':
        ++fmt;
        if (*fmt == 'l')
        {
            ++fmt;
            spec.length = PrintfLength::kLL;
        }
        else
        {
            spec.length = PrintfLength::kL;
        }
        break;
    case 'j':
        ++fmt;
        spec.length = PrintfLength::kJ;
        break;
    case 'z':
        ++fmt;
        spec.length = PrintfLength::kZ;
        break;
    case 't':
        ++fmt;
        spec.length = PrintfLength::kT;
        break;
    case 'L':
        ++fmt;
        spec.length = PrintfLength::kLongDouble;
        break;
    default:
        break;
    }

    spec.conversion = *fmt;
    if (*fmt != '\0')
    {
        ++fmt;
    }
    return fmt;
}

inline void ApplyPrintfSpec(std::ostream &out, const PrintfSpec &spec)
{
    if (spec.width >= 0)
    {
        out.width(spec.width);
    }
    if (spec.precision >= 0)
    {
        out.precision(spec.precision);
    }
    if (spec.left)
    {
        out.setf(std::ios_base::left, std::ios_base::adjustfield);
    }
    if (spec.showpos)
    {
        out.setf(std::ios_base::showpos);
    }
    if (spec.alt)
    {
        out.setf(std::ios_base::showbase);
    }
    if (spec.zero && !spec.left)
    {
        out.fill('0');
    }
}

inline void AppendLiteral(std::ostream &out, const char *begin, const char *end)
{
    if (begin < end)
    {
        out.write(begin, end - begin);
    }
}

inline const char *AppendLiteralUntilPercent(std::ostream &out, const char *fmt, const char *&literalBegin)
{
    while (*fmt != '\0')
    {
        if (*fmt == '%')
        {
            AppendLiteral(out, literalBegin, fmt);
            return fmt + 1;
        }
        ++fmt;
    }

    AppendLiteral(out, literalBegin, fmt);
    return nullptr;
}

struct StringSlice
{
    const char *data;
    std::size_t size;
};

inline StringSlice ToStringSlice(const char *value)
{
    const char *out = value != nullptr ? value : "(null)";
    return StringSlice{out, std::char_traits<char>::length(out)};
}

template<std::size_t N>
StringSlice ToStringSlice(const char (&value)[N])
{
    return StringSlice{value, std::char_traits<char>::length(value)};
}

inline StringSlice ToStringSlice(const std::string &value)
{
    return StringSlice{value.data(), value.size()};
}

#if __cplusplus >= 201402L
template<class T>
using DecayT = std::decay_t<T>;

template<bool Cond, class T = void>
using EnableIfT = std::enable_if_t<Cond, T>;

template<bool Cond, class If, class Else>
using ConditionalT = std::conditional_t<Cond, If, Else>;

template<class T>
using UnderlyingTypeT = std::underlying_type_t<T>;

template<class T>
using MakeUnsignedT = std::make_unsigned_t<T>;
#else
template<class T>
using DecayT = typename std::decay<T>::type;

template<bool Cond, class T = void>
using EnableIfT = typename std::enable_if<Cond, T>::type;

template<bool Cond, class If, class Else>
using ConditionalT = typename std::conditional<Cond, If, Else>::type;

template<class T>
using UnderlyingTypeT = typename std::underlying_type<T>::type;

template<class T>
using MakeUnsignedT = typename std::make_unsigned<T>::type;
#endif

template<class T, class U>
constexpr bool IsSame()
{
#if __cplusplus >= 201703L
    return std::is_same_v<T, U>;
#else
    return std::is_same<T, U>::value;
#endif
}

template<class T>
constexpr bool IsEnum()
{
#if __cplusplus >= 201703L
    return std::is_enum_v<T>;
#else
    return std::is_enum<T>::value;
#endif
}

template<class T>
constexpr bool IsIntegral()
{
#if __cplusplus >= 201703L
    return std::is_integral_v<T>;
#else
    return std::is_integral<T>::value;
#endif
}

template<class T>
constexpr bool IsArithmetic()
{
#if __cplusplus >= 201703L
    return std::is_arithmetic_v<T>;
#else
    return std::is_arithmetic<T>::value;
#endif
}

template<class T>
struct IsStringLike
{
    using Decayed = DecayT<T>;
    static constexpr bool value
        = IsSame<Decayed, char *>() || IsSame<Decayed, const char *>() || IsSame<Decayed, std::string>();
};

inline void AppendSpaces(std::ostream &out, std::size_t count)
{
    while (count-- > 0)
    {
        out.put(' ');
    }
}

inline void AppendString(std::ostream &out, const PrintfSpec &spec, StringSlice value)
{
    if (spec.precision >= 0)
    {
        value.size = std::min<std::size_t>(value.size, static_cast<std::size_t>(spec.precision));
    }

    if (spec.width > static_cast<int>(value.size) && !spec.left)
    {
        AppendSpaces(out, static_cast<std::size_t>(spec.width) - value.size);
    }

    out.write(value.data, value.size);

    if (spec.width > static_cast<int>(value.size) && spec.left)
    {
        AppendSpaces(out, static_cast<std::size_t>(spec.width) - value.size);
    }
}

template<class T, bool = IsEnum<T>()>
struct PrintfIntegralBase
{
    using type = T;
};

template<class T>
struct PrintfIntegralBase<T, true>
{
    using type = UnderlyingTypeT<T>;
};

template<class T>
struct PrintfValueType
{
    using Decayed = DecayT<T>;
    using Base    = typename PrintfIntegralBase<Decayed>::type;
    using Signed  = ConditionalT<(sizeof(Base) < sizeof(int)), int, Base>;
};

template<class T, bool IsBool = IsSame<T, bool>()>
struct PrintfUnsignedBase
{
    using type = MakeUnsignedT<T>;
};

template<class T>
struct PrintfUnsignedBase<T, true>
{
    using type = unsigned int;
};

template<class T>
struct PrintfUnsignedValueType
{
    using Decayed      = DecayT<T>;
    using Base         = typename PrintfIntegralBase<Decayed>::type;
    using UnsignedBase = typename PrintfUnsignedBase<Base>::type;
    using Unsigned     = ConditionalT<(sizeof(UnsignedBase) < sizeof(unsigned int)), unsigned int, UnsignedBase>;
};

template<class T>
EnableIfT<IsIntegral<DecayT<T>>() || IsEnum<DecayT<T>>(), void> AppendSigned(std::ostream &out, const T &value)
{
    using Signed = typename PrintfValueType<T>::Signed;
    out << static_cast<Signed>(value);
}

template<class T>
EnableIfT<!IsIntegral<DecayT<T>>() && !IsEnum<DecayT<T>>(), void> AppendSigned(std::ostream &out, T &&value)
{
    out << std::forward<T>(value);
}

template<class T>
EnableIfT<IsIntegral<DecayT<T>>() || IsEnum<DecayT<T>>(), void> AppendUnsigned(std::ostream &out, const T &value)
{
    using Unsigned = typename PrintfUnsignedValueType<T>::Unsigned;
    out << static_cast<Unsigned>(value);
}

template<class T>
EnableIfT<!IsIntegral<DecayT<T>>() && !IsEnum<DecayT<T>>(), void> AppendUnsigned(std::ostream &out, T &&value)
{
    out << std::forward<T>(value);
}

template<class T>
EnableIfT<IsStringLike<T>::value, void> AppendStringValue(std::ostream &out, const PrintfSpec &spec, T &&value)
{
    AppendString(out, spec, ToStringSlice(std::forward<T>(value)));
}

template<class T>
EnableIfT<!IsStringLike<T>::value, void> AppendStringValue(std::ostream &out, const PrintfSpec &, T &&value)
{
    out << std::forward<T>(value);
}

template<class T>
EnableIfT<IsArithmetic<DecayT<T>>() || IsEnum<DecayT<T>>(), void> AppendChar(std::ostream &out, const T &value)
{
    out << static_cast<char>(value);
}

template<class T>
EnableIfT<!IsArithmetic<DecayT<T>>() && !IsEnum<DecayT<T>>(), void> AppendChar(std::ostream &out, T &&value)
{
    out << std::forward<T>(value);
}

template<class T>
void AppendPointer(std::ostream &out, T *value)
{
    out << static_cast<const void *>(value);
}

template<class T>
void AppendPointer(std::ostream &out, const T &value)
{
    out << value;
}

inline void AppendUnknownPrintfConversion(std::ostream &out, const PrintfSpec &spec)
{
    out.put('%');
    if (spec.conversion != '\0')
    {
        out.put(spec.conversion);
    }
}

template<class T>
void AppendFormattedValue(std::ostream &out, const PrintfSpec &spec, T &&value)
{
    const auto oldFlags     = out.flags();
    const auto oldPrecision = out.precision();
    const auto oldFill      = out.fill();

    ApplyPrintfSpec(out, spec);

    switch (spec.conversion)
    {
    case 'd':
    case 'i':
        AppendSigned(out, std::forward<T>(value));
        break;
    case 'u':
        AppendUnsigned(out, std::forward<T>(value));
        break;
    case 'o':
        out.setf(std::ios_base::oct, std::ios_base::basefield);
        AppendUnsigned(out, std::forward<T>(value));
        break;
    case 'x':
    case 'X':
        out.setf(std::ios_base::hex, std::ios_base::basefield);
        if (spec.conversion == 'X')
        {
            out.setf(std::ios_base::uppercase);
        }
        AppendUnsigned(out, std::forward<T>(value));
        break;
    case 'f':
    case 'F':
        out.setf(std::ios_base::fixed, std::ios_base::floatfield);
        if (spec.conversion == 'F')
        {
            out.setf(std::ios_base::uppercase);
        }
        out << value;
        break;
    case 'e':
    case 'E':
        out.setf(std::ios_base::scientific, std::ios_base::floatfield);
        if (spec.conversion == 'E')
        {
            out.setf(std::ios_base::uppercase);
        }
        out << value;
        break;
    case 'g':
    case 'G':
        if (spec.conversion == 'G')
        {
            out.setf(std::ios_base::uppercase);
        }
        out << value;
        break;
    case 's':
        AppendStringValue(out, spec, std::forward<T>(value));
        break;
    case 'c':
        AppendChar(out, std::forward<T>(value));
        break;
    case 'p':
        AppendPointer(out, std::forward<T>(value));
        break;
    default:
        AppendUnknownPrintfConversion(out, spec);
        break;
    }

    out.flags(oldFlags);
    out.precision(oldPrecision);
    out.fill(oldFill);
}

inline void FormatImpl(std::ostream &out, const char *fmt)
{
    if (fmt == nullptr)
    {
        return;
    }

    const char *literalBegin = fmt;
    while ((fmt = AppendLiteralUntilPercent(out, fmt, literalBegin)) != nullptr)
    {
        out.put('%');
        if (*fmt == '%')
        {
            ++fmt;
        }
        literalBegin = fmt;
    }
}

template<class Arg, class... Args>
void FormatImpl(std::ostream &out, const char *fmt, Arg &&arg, Args &&...args)
{
    if (fmt == nullptr)
    {
        return;
    }

    const char *literalBegin = fmt;
    while ((fmt = AppendLiteralUntilPercent(out, fmt, literalBegin)) != nullptr)
    {
        if (*fmt == '%')
        {
            ++fmt;
            out.put('%');
            literalBegin = fmt;
            continue;
        }

        PrintfSpec  spec;
        const char *next = ParsePrintfSpec(fmt, spec);
        AppendFormattedValue(out, spec, std::forward<Arg>(arg));
        FormatImpl(out, next, std::forward<Args>(args)...);
        return;
    }
}

template<class... Args>
void FormatTo(char *buffer, std::size_t bufferSize, const char *fmt, Args &&...args)
{
    if (buffer == nullptr || bufferSize == 0)
    {
        return;
    }

    FixedBufferOStream out(buffer, bufferSize);
    FormatImpl(out.get(), fmt, std::forward<Args>(args)...);
    out.get().flush();
    buffer[bufferSize - 1] = '\0';
}

inline void AppendVaArg(std::ostream &out, const PrintfSpec &spec, va_list &va)
{
    switch (spec.conversion)
    {
    case 'd':
    case 'i':
        switch (spec.length)
        {
        case PrintfLength::kLL:
            AppendFormattedValue(out, spec, va_arg(va, long long));
            break;
        case PrintfLength::kL:
            AppendFormattedValue(out, spec, va_arg(va, long));
            break;
        case PrintfLength::kJ:
            AppendFormattedValue(out, spec, va_arg(va, std::intmax_t));
            break;
        case PrintfLength::kZ:
            AppendFormattedValue(out, spec, va_arg(va, std::ptrdiff_t));
            break;
        case PrintfLength::kT:
            AppendFormattedValue(out, spec, va_arg(va, std::ptrdiff_t));
            break;
        default:
            AppendFormattedValue(out, spec, va_arg(va, int));
            break;
        }
        break;
    case 'u':
    case 'o':
    case 'x':
    case 'X':
        switch (spec.length)
        {
        case PrintfLength::kLL:
            AppendFormattedValue(out, spec, va_arg(va, unsigned long long));
            break;
        case PrintfLength::kL:
            AppendFormattedValue(out, spec, va_arg(va, unsigned long));
            break;
        case PrintfLength::kJ:
            AppendFormattedValue(out, spec, va_arg(va, std::uintmax_t));
            break;
        case PrintfLength::kZ:
            AppendFormattedValue(out, spec, va_arg(va, std::size_t));
            break;
        default:
            AppendFormattedValue(out, spec, va_arg(va, unsigned int));
            break;
        }
        break;
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
        if (spec.length == PrintfLength::kLongDouble)
        {
            AppendFormattedValue(out, spec, va_arg(va, long double));
        }
        else
        {
            AppendFormattedValue(out, spec, va_arg(va, double));
        }
        break;
    case 's':
        AppendFormattedValue(out, spec, va_arg(va, const char *));
        break;
    case 'c':
        AppendFormattedValue(out, spec, va_arg(va, int));
        break;
    case 'p':
        AppendFormattedValue(out, spec, va_arg(va, void *));
        break;
    default:
        AppendUnknownPrintfConversion(out, spec);
        break;
    }
}

inline void VFormatTo(char *buffer, std::size_t bufferSize, const char *fmt, va_list va)
{
    if (buffer == nullptr || bufferSize == 0 || fmt == nullptr)
    {
        return;
    }

    FixedBufferOStream out(buffer, bufferSize);
    va_list            vaCopy;
    va_copy(vaCopy, va);

    const char *literalBegin = fmt;
    while ((fmt = AppendLiteralUntilPercent(out.get(), fmt, literalBegin)) != nullptr)
    {
        if (*fmt == '%')
        {
            ++fmt;
            out.get().put('%');
            literalBegin = fmt;
            continue;
        }

        PrintfSpec spec;
        fmt = ParsePrintfSpec(fmt, spec);
        AppendVaArg(out.get(), spec, vaCopy);
        literalBegin = fmt;
    }

    va_end(vaCopy);
    out.get().flush();
    buffer[bufferSize - 1] = '\0';
}

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_FORMAT_HPP
