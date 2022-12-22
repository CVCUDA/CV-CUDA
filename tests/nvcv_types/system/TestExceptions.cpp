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

// TODO: once we have functions that generate errors, we should
// extend these tests to cover more scenarios

namespace t = ::testing;

TEST(ExceptionTest, exception_updates_internal_status)
{
    try
    {
        throw nvcv::Exception(nvcv::Status::ERROR_DEVICE, "test error");
    }
    catch (...)
    {
    }

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test error", msg);
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

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test invalid argument", msg);
}

TEST(ExceptionTest, protect_call_bad_alloc)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw std::bad_alloc(); });

    EXPECT_EQ(NVCV_ERROR_OUT_OF_MEMORY, status);

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_OUT_OF_MEMORY, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("Not enough space for resource allocation", msg);
}

TEST(ExceptionTest, protect_call_unexpected)
{
    NVCVStatus status = nvcv::ProtectCall([] { throw 5; });

    EXPECT_EQ(NVCV_ERROR_INTERNAL, status);

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("Unexpected error", msg);
}

TEST(ExceptionTest, exception_format_multiple_args)
{
    nvcv::Exception e(nvcv::Status::ERROR_DEVICE, "test error %d %s %c", 123, "rod", 'l');

    EXPECT_STREQ("NVCV_ERROR_DEVICE: test error 123 rod l", e.what());
    EXPECT_STREQ("test error 123 rod l", e.msg());

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test error 123 rod l", msg);
}
