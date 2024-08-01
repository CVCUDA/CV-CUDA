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

#ifndef NVCV_DETAIL_CHECKERROR_HPP
#define NVCV_DETAIL_CHECKERROR_HPP

#include "../Exception.hpp"
#include "../Status.h"

#include <cassert>

namespace nvcv { namespace detail {

inline void ThrowException(NVCVStatus status)
{
    // Because of this stack allocation, compiler might
    // not inline this call. This it happens only in
    // error cases, it's ok.
    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];

    NVCVStatus tmp = nvcvGetLastErrorMessage(msg, sizeof(msg));
    (void)tmp;
    assert(tmp == status);

    throw Exception(Exception::InternalCtorTag{}, static_cast<Status>(status), "%s", msg);
}

inline void CheckThrow(NVCVStatus status)
{
    // This check gets inlined easier, and it's normal code path.
    if (status != NVCV_SUCCESS)
    {
        ThrowException(status);
    }
}

}} // namespace nvcv::detail

#endif // NVCV_DETAIL_CHECKERROR_HPP
