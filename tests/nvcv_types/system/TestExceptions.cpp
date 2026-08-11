/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Definitions.hpp"

#include <nvcv/Exception.hpp>

#include <array>

// REVISIT: once we have functions that generate errors, we should
// extend these tests to cover more scenarios

namespace t = ::testing;

TEST(ExceptionTest, exception_updates_internal_status)
{
    try
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "test error");
    }
    catch (const nvcv::Exception &e)
    {
        // The thread-local status update is the behavior under test.
        EXPECT_EQ(nvcv::Status::ERROR_DEVICE, e.code());
        EXPECT_STREQ("test error", e.msg());
    }

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test error", msg.data());
}

TEST(ExceptionTest, protect_call_nvcv_exception)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "test error"); });

    EXPECT_EQ(NVCV_ERROR_DEVICE, status);
}

TEST(ExceptionTest, protect_call_invalid_argument)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw std::invalid_argument("test invalid argument"); });

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, status);

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test invalid argument", msg.data());
}

TEST(ExceptionTest, protect_call_bad_alloc)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw std::bad_alloc(); });

    EXPECT_EQ(NVCV_ERROR_OUT_OF_MEMORY, status);

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_OUT_OF_MEMORY, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("Not enough space for resource allocation", msg.data());
}

TEST(ExceptionTest, protect_call_standard_logic_and_runtime_errors)
{
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::domain_error("domain"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::length_error("length"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::out_of_range("out of range"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::range_error("range"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::overflow_error("overflow"); }));
    EXPECT_EQ(NVCV_ERROR_INTERNAL, nvcv::ProtectCall([] { throw std::underflow_error("underflow"); }));
}

TEST(ExceptionTest, protect_call_unexpected)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw 5; });

    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("Unexpected error", msg.data());
}

TEST(ExceptionTest, exception_format_multiple_args)
{
    nvcv::Exception e(nvcv::Status::ERROR_DEVICE, "test error %d %s %c", 123, "rod", 'l');

    EXPECT_STREQ("NVCV_ERROR_DEVICE: test error 123 rod l", e.what());
    EXPECT_STREQ("test error 123 rod l", e.msg());

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test error 123 rod l", msg.data());
}

TEST(ExceptionTest, exception_format_numeric_modifiers)
{
    nvcv::Exception e(nvcv::Status::ERROR_DEVICE, "test error %ld %.2f %x %%", 123L, 4.5, 255U);

    EXPECT_STREQ("NVCV_ERROR_DEVICE: test error 123 4.50 ff %", e.what());
    EXPECT_STREQ("test error 123 4.50 ff %", e.msg());
}
