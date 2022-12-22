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

#include "priv/Status.hpp"

#include "priv/Exception.hpp"
#include "priv/SymbolVersioning.hpp"

#include <nvcv/Status.h>
#include <util/Assert.h>

namespace priv = nvcv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvGetLastError, ())
{
    return priv::GetLastThreadError(); // noexcept
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvGetLastErrorMessage, (char *msgBuffer, int32_t lenBuffer))
{
    return priv::GetLastThreadError(msgBuffer, lenBuffer);
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPeekAtLastError, ())
{
    return priv::PeekAtLastThreadError(); // noexcept
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPeekAtLastErrorMessage, (char *msgBuffer, int32_t lenBuffer))
{
    return priv::PeekAtLastThreadError(msgBuffer, lenBuffer); // noexcept
}

NVCV_DEFINE_API(0, 0, const char *, nvcvStatusGetName, (NVCVStatus err))
{
    return priv::GetName(err); // noexcept
}

NVCV_DEFINE_API(0, 2, void, nvcvSetThreadStatusVarArgList, (NVCVStatus status, const char *fmt, va_list va))
{
    NVCVStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });

    NVCV_ASSERT(ret == status);
}

NVCV_DEFINE_API(0, 2, void, nvcvSetThreadStatus, (NVCVStatus status, const char *fmt, ...))
{
    va_list va;
    va_start(va, fmt);

    NVCVStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });

    va_end(va);

    NVCV_ASSERT(ret == status);
}
