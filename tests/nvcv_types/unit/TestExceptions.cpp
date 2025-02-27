/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "Definitions.hpp"
#include "nvcv/src/priv/Exception.hpp"

#include <nvcv/Exception.hpp>

#if !NVCV_DEBUG
TEST(ExceptionTest, priv_protect_call_public_exception)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw nvcv::Exception(nvcv::Status::ERROR_DEVICE); });
    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);
}
#endif

TEST(ExceptionTest, exception_what)
{
    try
    {
        throw nvcv::priv::Exception(NVCV_ERROR_DEVICE);
    }
    catch (const nvcv::priv::Exception &e)
    {
        EXPECT_STREQ(e.what(), "NVCV_ERROR_DEVICE: ");
    }
}

TEST(ExceptionTest, priv_protect_call_invalid_argument)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw std::invalid_argument(""); });
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, status);
}

TEST(ExceptionTest, priv_protect_call_bad_alloc)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw std::bad_alloc(); });
    EXPECT_EQ(NVCV_ERROR_OUT_OF_MEMORY, status);
}

TEST(ExceptionTest, priv_protect_call_std_exception)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw std::exception(); });
    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);
}

TEST(ExceptionTest, priv_protect_call_unexpected)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw 5; });
    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);
}
