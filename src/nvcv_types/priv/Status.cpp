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

#include "Status.hpp"

#include "Exception.hpp"
#include "TLS.hpp"

#include <nvcv/Exception.hpp>
#include <util/Assert.h>

#include <cstring>

namespace nvcv::priv {

void SetThreadError(std::exception_ptr e)
{
    CoreTLS &tls = GetCoreTLS();

    const int errorMessageLen = sizeof(tls.lastErrorMessage) - 1;

    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            tls.lastErrorStatus = NVCV_SUCCESS;
            strncpy(tls.lastErrorMessage, "success", errorMessageLen);
        }
    }
    catch (const ::nvcv::Exception &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        NVCV_ASSERT(!"Exception from public API cannot be originated from internal library implementation");
    }
    catch (const Exception &e)
    {
        tls.lastErrorStatus = e.code();
        strncpy(tls.lastErrorMessage, e.msg(), errorMessageLen);
    }
    catch (const std::invalid_argument &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INVALID_ARGUMENT;
        strncpy(tls.lastErrorMessage, e.what(), errorMessageLen);
    }
    catch (const std::bad_alloc &)
    {
        tls.lastErrorStatus = NVCV_ERROR_OUT_OF_MEMORY;
        strncpy(tls.lastErrorMessage, "Not enough space for resource allocation", errorMessageLen);
    }
    catch (const std::exception &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        strncpy(tls.lastErrorMessage, e.what(), errorMessageLen);
    }
    catch (...)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        strncpy(tls.lastErrorMessage, "Unexpected error", errorMessageLen);
    }

    tls.lastErrorMessage[errorMessageLen] = '\0'; // Make sure it's null-terminated
}

NVCVStatus GetLastThreadError() noexcept
{
    return GetLastThreadError(nullptr, 0);
}

NVCVStatus PeekAtLastThreadError() noexcept
{
    return PeekAtLastThreadError(nullptr, 0);
}

NVCVStatus GetLastThreadError(char *outMessage, int outMessageLen) noexcept
{
    NVCVStatus status = PeekAtLastThreadError(outMessage, outMessageLen);

    SetThreadError(std::exception_ptr{});

    return status;
}

NVCVStatus PeekAtLastThreadError(char *outMessage, int outMessageLen) noexcept
{
    CoreTLS &tls = GetCoreTLS();

    if (outMessage != nullptr && outMessageLen > 0)
    {
        strncpy(outMessage, tls.lastErrorMessage, outMessageLen);
        outMessage[outMessageLen - 1] = '\0'; // Make sure it's null-terminated
    }

    return tls.lastErrorStatus;
}

const char *GetName(NVCVStatus status) noexcept
{
#define CASE(ERR) \
    case ERR:     \
        return #ERR

    // written this way, without a default case,
    // the compiler can warn us if we forgot to add a new error here.
    switch (status)
    {
        CASE(NVCV_SUCCESS);
        CASE(NVCV_ERROR_NOT_IMPLEMENTED);
        CASE(NVCV_ERROR_INVALID_ARGUMENT);
        CASE(NVCV_ERROR_INVALID_IMAGE_FORMAT);
        CASE(NVCV_ERROR_INVALID_OPERATION);
        CASE(NVCV_ERROR_DEVICE);
        CASE(NVCV_ERROR_NOT_READY);
        CASE(NVCV_ERROR_OUT_OF_MEMORY);
        CASE(NVCV_ERROR_INTERNAL);
        CASE(NVCV_ERROR_NOT_COMPATIBLE);
        CASE(NVCV_ERROR_OVERFLOW);
        CASE(NVCV_ERROR_UNDERFLOW);
    }

    // Status not found?
    return "Unknown error";
#undef CASE
}

} // namespace nvcv::priv
