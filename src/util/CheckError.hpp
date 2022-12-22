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

#ifndef NVCV_UTIL_CHECK_ERROR_HPP
#define NVCV_UTIL_CHECK_ERROR_HPP

#include "Assert.h"

#include <driver_types.h> // for cudaError

#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#if NVCV_EXPORTING
#    include <nvcv_types/priv/Exception.hpp>
#else
#    include <nvcv/Exception.hpp>
#endif

// Here we define the CHECK_ERROR macro that converts error values into exceptions
// or log messages. It can be extended to other errors. At minimum, you need to define:
// * inline bool CheckSucceeded(ErrorType err)
// * const char *ToString(NvError err, const char **perrdescr=nullptr);
//
// Optionally, you can define:
// * NVCVStatus TranslateError(NvError err);
//   by default it translates success to NVCV_SUCCESS, failure to NVCV_ERROR_INTERNAL
// * void PreprocessError(T err);
//   If you need to swallow up the error, CUDA needs that. Or any other kind of
//   processing when the error occurs

namespace nvcv::util {

namespace detail {
const char *GetCheckMessage(char *buf, int buflen);
char       *GetCheckMessage(char *buf, int buflen, const char *fmt, ...);
std::string FormatErrorMessage(const std::string_view &errname, const std::string_view &callstr,
                               const std::string_view &msg);
} // namespace detail

// CUDA -----------------------

inline bool CheckSucceeded(cudaError_t err)
{
    return err == cudaSuccess;
}

NVCVStatus  TranslateError(cudaError_t err);
const char *ToString(cudaError_t err, const char **perrdescr = nullptr);
void        PreprocessError(cudaError_t err);

// Default implementation --------------------

template<class T>
NVCVStatus TranslateError(T err)
{
    if (CheckSucceeded(err))
    {
        return NVCV_SUCCESS;
    }
    else
    {
        return NVCV_ERROR_INTERNAL;
    }
}

template<class T>
inline void PreprocessError(T err)
{
}

namespace detail {

template<class T>
void DoThrow(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
#if NVCV_EXPORTING
    using nvcv::priv::Exception;
    using StatusType = NVCVStatus;
#else
    using nvcv::Exception;
    using StatusType = nvcv::Status;
#endif

    // Can we expose source file data?
    if (file != nullptr)
    {
        throw Exception((StatusType)TranslateError(error), "%s:%d %s", file, line,
                        FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
    else
    {
        throw Exception((StatusType)TranslateError(error), "%s",
                        FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
}

template<class T>
void DoLog(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
    // TODO: replace with a real log facility

    // Can we expose source file data?
    if (file != nullptr)
    {
        std::cerr << file << ":" << line << ' ';
    }
    std::cerr << FormatErrorMessage(ToString(error), stmt, errmsg);
}

} // namespace detail

#define NVCV_CHECK_THROW(STMT, ...)                                                                                \
    [&]()                                                                                                          \
    {                                                                                                              \
        using ::nvcv::util::PreprocessError;                                                                       \
        using ::nvcv::util::CheckSucceeded;                                                                        \
        auto status = (STMT);                                                                                      \
        PreprocessError(status);                                                                                   \
        if (!CheckSucceeded(status))                                                                               \
        {                                                                                                          \
            char buf[NVCV_MAX_STATUS_MESSAGE_LENGTH];                                                              \
            ::nvcv::util::detail::DoThrow(status, NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO,                  \
                                          NVCV_OPTIONAL_STRINGIFY(STMT),                                           \
                                          ::nvcv::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
        }                                                                                                          \
    }()

#define NVCV_CHECK_LOG(STMT, ...)                                                                                \
    [&]()                                                                                                        \
    {                                                                                                            \
        using ::nvcv::util::PreprocessError;                                                                     \
        using ::nvcv::util::CheckSucceeded;                                                                      \
        auto status = (STMT);                                                                                    \
        PreprocessError(status);                                                                                 \
        if (!CheckSucceeded(status))                                                                             \
        {                                                                                                        \
            char buf[NVCV_MAX_STATUS_MESSAGE_LENGTH];                                                            \
            ::nvcv::util::detail::DoLog(status, NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO,                  \
                                        NVCV_OPTIONAL_STRINGIFY(STMT),                                           \
                                        ::nvcv::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
            return false;                                                                                        \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
            return true;                                                                                         \
        }                                                                                                        \
    }()

} // namespace nvcv::util

#endif // NVCV_UTIL_CHECK_ERROR_HPP
