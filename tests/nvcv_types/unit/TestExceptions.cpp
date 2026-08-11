/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <exception>
#include <stdexcept>
#include <utility>

namespace {

class ProtectCallTestException : public std::exception
{
};

} // namespace

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

TEST(ExceptionTest, exception_format_numeric_modifiers)
{
    nvcv::priv::Exception e(NVCV_ERROR_DEVICE, "test error %ld %.2f %x %%", 123L, 4.5, 255U);

    EXPECT_STREQ("NVCV_ERROR_DEVICE: test error 123 4.50 ff %", e.what());
    EXPECT_STREQ("test error 123 4.50 ff %", e.msg());
}

TEST(ExceptionTest, exception_stream_append)
{
    auto e = nvcv::priv::Exception(NVCV_ERROR_DEVICE) << "streamed " << 7;

    EXPECT_STREQ("NVCV_ERROR_DEVICE: streamed 7", e.what());
    EXPECT_STREQ("streamed 7", e.msg());
}

TEST(ExceptionTest, exception_null_message)
{
    nvcv::priv::Exception e(NVCV_ERROR_DEVICE, static_cast<const char *>(nullptr));

    EXPECT_STREQ("NVCV_ERROR_DEVICE: ", e.what());
}

TEST(ExceptionTest, exception_copy_and_move_assignment)
{
    nvcv::priv::Exception source(NVCV_ERROR_DEVICE, "source");
    nvcv::priv::Exception copy(NVCV_ERROR_INTERNAL, "copy");
    nvcv::priv::Exception moved(NVCV_ERROR_INTERNAL, "moved");

    copy = source;
    EXPECT_EQ(NVCV_ERROR_DEVICE, copy.code());
    EXPECT_STREQ("source", copy.msg());

    moved = std::move(source);
    EXPECT_EQ(NVCV_ERROR_DEVICE, moved.code());
    EXPECT_STREQ("source", moved.msg());
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

TEST(ExceptionTest, priv_protect_call_standard_logic_and_runtime_errors)
{
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::domain_error("domain"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::length_error("length"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::out_of_range("out of range"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::range_error("range"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::overflow_error("overflow"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::priv::ProtectCall([] { throw std::underflow_error("underflow"); }));
}

TEST(ExceptionTest, priv_protect_call_std_exception)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw ProtectCallTestException(); });
    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);
}

TEST(ExceptionTest, priv_protect_call_unexpected)
{
    NVCVStatus status = nvcv::priv::ProtectCall([] { throw 5; });
    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);
}
