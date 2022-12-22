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

#include "CheckError.hpp"

#include <cuda_runtime.h>

#include <cstdarg>
#include <regex>

namespace nvcv::util {

static std::string_view GetFunctionName(const std::string_view &stmt)
{
    static std::regex rgx("^([A-Za-z0-9_]+)\\(.*$");

    std::match_results<std::string_view::const_iterator> match;
    if (regex_match(stmt.begin(), stmt.end(), match, rgx))
    {
        return std::string_view(match[1].first, match[1].second);
    }
    else
    {
        return "";
    }
}

namespace detail {
const char *GetCheckMessage(char *buf, int bufsize)
{
    NVCV_ASSERT(buf != nullptr);
    (void)buf;
    (void)bufsize;

    return "";
}

char *GetCheckMessage(char *buf, int bufsize, const char *fmt, ...)
{
    NVCV_ASSERT(buf != nullptr);
    NVCV_ASSERT(fmt != nullptr);

    va_list va;
    va_start(va, fmt);

    vsnprintf(buf, bufsize - 1, fmt, va);

    va_end(va);

    return buf;
}

std::string FormatErrorMessage(const std::string_view &errname, const std::string_view &callstr,
                               const std::string_view &msg)
{
    std::string_view funcName = GetFunctionName(callstr);

    // TODO: avoid heap memory allocation here
    std::ostringstream ss;
    ss << '(';
    if (!funcName.empty())
    {
        ss << funcName << ':';
    }

    ss << errname << ')';
    if (!msg.empty())
    {
        ss << ' ' << msg;
    }

    return ss.str();
}

} // namespace detail

NVCVStatus TranslateError(cudaError_t err)
{
    switch (err)
    {
    case cudaErrorMemoryAllocation:
        return NVCV_ERROR_OUT_OF_MEMORY;

    case cudaErrorNotReady:
        return NVCV_ERROR_NOT_READY;

    case cudaErrorInvalidValue:
        return NVCV_ERROR_INVALID_ARGUMENT;

    default:
        return NVCV_ERROR_INTERNAL;
    }
}

void PreprocessError(cudaError_t err)
{
    // consume the error
    cudaGetLastError();
}

const char *ToString(cudaError_t err, const char **perrdescr)
{
    if (perrdescr != nullptr)
    {
        *perrdescr = cudaGetErrorString(err);
    }

    return cudaGetErrorName(err);
}

} // namespace nvcv::util
