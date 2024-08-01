/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <cstring>

namespace nvcv {

namespace detail {
void ThrowException(NVCVStatus status);
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
     * @param ... The format arguments.
     */
    explicit Exception(Status code, const char *fmt = nullptr, ...)
#if __GNUC__
        __attribute__((format(printf, 3, 4)))
#endif
        : m_code(code)
    {
        va_list va;
        va_start(va, fmt);
        nvcvSetThreadStatusVarArgList(static_cast<NVCVStatus>(code), fmt, va);
        va_end(va);

        va_start(va, fmt);
        doSetMessage(fmt, va);
        va_end(va);
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
        return m_msgBuffer;
    }

private:
    Status      m_code;
    const char *m_msg;

    // 64: maximum size of string representation of a status enum
    // 2: ': '
    char m_msgBuffer[NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2];

    friend void detail::ThrowException(NVCVStatus status);

    struct InternalCtorTag
    {
    };

    // Constructor that doesn't set the C thread status.
    // Used when converting C statuses to C++.
    Exception(InternalCtorTag, Status code, const char *fmt = nullptr, ...)
#if __GNUC__
        __attribute__((format(printf, 4, 5)))
#endif
        : m_code(code)
    {
        va_list va;
        va_start(va, fmt);

        doSetMessage(fmt, va);

        va_end(va);
    }

    void doSetMessage(const char *fmt, va_list va)
    {
        int buflen   = sizeof(m_msgBuffer);
        int nwritten = snprintf(m_msgBuffer, buflen, "%s: ", GetName(m_code));

        // no truncation?
        if (nwritten < buflen)
        {
            buflen -= nwritten;
            m_msg = m_msgBuffer + nwritten;
            vsnprintf(m_msgBuffer + nwritten, buflen, fmt, va);
        }

        m_msgBuffer[sizeof(m_msgBuffer) - 1] = '\0';
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
    catch (const std::bad_alloc &)
    {
        nvcvSetThreadStatus(NVCV_ERROR_OUT_OF_MEMORY, "Not enough space for resource allocation");
    }
    catch (const std::exception &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (...)
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
    catch (...)
    {
        SetThreadError(std::current_exception());
        return nvcvPeekAtLastError();
    }
}

/**@}*/

} // namespace nvcv

#endif // NVCV_EXCEPTION_HPP
