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

/**
 * @file Exception.hpp
 *
 * @brief Declaration of NVCV C++ exception classes.
 */

#ifndef NVCV_EXCEPTION_HPP
#define NVCV_EXCEPTION_HPP

#include <nvcv/Status.hpp>
#include <nvcv/detail/Format.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace nvcv {

namespace detail {
[[noreturn]] void ThrowException(NVCVStatus status);
}

/**
 * @defgroup NVCV_CPP_UTIL_EXCEPTION Exception
 * @{
*/
/**
 * @class Exception
 * @brief Custom exception class to represent errors specific to this application.
 *
 * This class extends the standard exception class and is designed to encapsulate error codes
 * and messages specific to this application's context.
 */
class Exception : public std::exception
{
public:
    /**
     * @brief Constructs an exception with a status code and a formatted message.
     *
     * @param code The error status code.
     * @param fmt The format string for the error message.
     * @param args The format arguments.
     */
    explicit Exception(Status code)
        : Exception(code, "%s", "")
    {
    }

    explicit Exception(Status code, const char *msg)
        : Exception(code, "%s", msg != nullptr ? msg : "")
    {
    }

    template<size_t N, class... Args>
    explicit Exception(Status code, const char (&fmt)[N], Args &&...args)
        : m_code(code)
    {
        doSetMessage(fmt, std::forward<Args>(args)...);
        nvcvSetThreadStatus(static_cast<NVCVStatus>(code), "%s", m_msg);
    }

    /**
     * @brief Retrieves the status code of the exception.
     *
     * @return The error status code.
     */
    Status code() const
    {
        return m_code;
    }

    /**
     * @brief Retrieves the message of the exception.
     *
     * @return The error message.
     */
    const char *msg() const
    {
        return m_msg;
    }

    /**
     * @brief Retrieves the exception message.
     *
     * This function overrides the standard exception's what() method.
     *
     * @return The error message.
     */
    const char *what() const noexcept override
    {
        return m_msgBuffer.data();
    }

private:
    Status      m_code;
    const char *m_msg;

    // 64: maximum size of string representation of a status enum
    // 2: ': '
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2> m_msgBuffer;

    friend void detail::ThrowException(NVCVStatus status);

    struct InternalCtorTag
    {
    };

    Exception(InternalCtorTag, Status code)
        : Exception(InternalCtorTag{}, code, "%s", "")
    {
    }

    Exception(InternalCtorTag, Status code, const char *msg)
        : Exception(InternalCtorTag{}, code, "%s", msg != nullptr ? msg : "")
    {
    }

    // Constructor that doesn't set the C thread status. Used when converting C statuses to C++.
    template<size_t N, class... Args>
    Exception(InternalCtorTag, Status code, const char (&fmt)[N], Args &&...args)
        : m_code(code)
    {
        doSetMessage(fmt, std::forward<Args>(args)...);
    }

    template<size_t N, class... Args>
    void doSetMessage(const char (&fmt)[N], Args &&...args)
    {
        auto buflen = static_cast<int>(m_msgBuffer.size());

        // no truncation?
        detail::FormatTo(m_msgBuffer.data(), m_msgBuffer.size(), "%s: ", GetName(m_code));

        int nwritten = static_cast<int>(std::char_traits<char>::length(m_msgBuffer.data()));
        if (nwritten < buflen - 1)
        {
            buflen -= nwritten;
            m_msg = m_msgBuffer.data() + nwritten;
            detail::FormatTo(m_msgBuffer.data() + nwritten, static_cast<std::size_t>(buflen), fmt,
                             std::forward<Args>(args)...);
        }

        m_msgBuffer.back() = '\0';
    }
};

/**
 * @brief Sets the thread's error status based on a captured exception.
 *
 * This function tries to rethrow the given exception and based on its type, it sets the appropriate
 * error status for the current thread using the `nvcvSetThreadStatus` function.
 *
 * @param e The captured exception to be rethrown and processed.
 */
inline void SetThreadError(std::exception_ptr e)
{
    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            nvcvSetThreadStatus(NVCV_SUCCESS, nullptr);
        }
    }
    catch (const Exception &e)
    {
        nvcvSetThreadStatus(static_cast<NVCVStatus>(e.code()), "%s", e.msg());
    }
    catch (const std::invalid_argument &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INVALID_ARGUMENT, "%s", e.what());
    }
    catch (const std::domain_error &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (const std::length_error &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (const std::out_of_range &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (const std::bad_alloc &)
    {
        nvcvSetThreadStatus(NVCV_ERROR_OUT_OF_MEMORY, "Not enough space for resource allocation");
    }
    catch (const std::range_error &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (const std::overflow_error &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (const std::underflow_error &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (...) // NOSONAR: API boundary converts any non-standard exception to NVCV status.
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "Unexpected error");
    }
}

/**
 * @brief Safely executes a function, capturing and setting any exceptions that arise.
 *
 * This function acts as a wrapper to safely execute a given function or lambda (`fn`).
 * If the function throws any exception, the exception is captured, and the error status is
 * set for the current thread using the `SetThreadError` function.
 *
 * @param fn The function or lambda to be executed.
 * @return NVCV_SUCCESS if `fn` executed without exceptions, otherwise the error code from the caught exception.
 *
 * @tparam F The type of the function or lambda.
 */
template<class F>
NVCVStatus ProtectCall(F &&fn)
{
    try
    {
        fn();
        return NVCV_SUCCESS;
    }
    catch (...) // NOSONAR: this API boundary translates any exception to an NVCVStatus.
    {
        SetThreadError(std::current_exception());
        return nvcvPeekAtLastError();
    }
}

/**@}*/

} // namespace nvcv

#endif // NVCV_EXCEPTION_HPP
