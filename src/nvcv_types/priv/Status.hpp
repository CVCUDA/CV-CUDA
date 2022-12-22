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

#ifndef NVCV_CORE_PRIV_STATUS_HPP
#define NVCV_CORE_PRIV_STATUS_HPP

#include <nvcv/Status.h>

#include <exception>
#include <iosfwd>

namespace nvcv::priv {

NVCVStatus GetLastThreadError(char *outMessage, int outMessageLen) noexcept;
NVCVStatus PeekAtLastThreadError(char *outMessage, int outMessageLen) noexcept;

NVCVStatus GetLastThreadError() noexcept;
NVCVStatus PeekAtLastThreadError() noexcept;

const char *GetName(NVCVStatus status) noexcept;

void SetThreadError(std::exception_ptr e);

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
        return PeekAtLastThreadError();
    }
}

} // namespace nvcv::priv

#endif // NVCV_CORE_PRIV_STATUS_HPP
