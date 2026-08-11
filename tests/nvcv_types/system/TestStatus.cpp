/* Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Image.h>
#include <nvcv/Status.h>

#include <array>
#include <vector>

// REVISIT: once we have functions that generate errors, we should
// extend these tests to cover more scenarios

namespace t = ::testing;

class StatusNameTest : public t::TestWithParam<std::tuple<NVCVStatus, const char *>>
{
};

#define MAKE_STATUS_NAME(X) std::make_tuple(X, #X)

static auto StatusNameParams()
{
    std::vector<std::tuple<NVCVStatus, const char *>> params{
        MAKE_STATUS_NAME(NVCV_SUCCESS),
        MAKE_STATUS_NAME(NVCV_ERROR_NOT_IMPLEMENTED),
        MAKE_STATUS_NAME(NVCV_ERROR_INVALID_ARGUMENT),
        MAKE_STATUS_NAME(NVCV_ERROR_INVALID_IMAGE_FORMAT),
        MAKE_STATUS_NAME(NVCV_ERROR_INVALID_OPERATION),
        MAKE_STATUS_NAME(NVCV_ERROR_DEVICE),
        MAKE_STATUS_NAME(NVCV_ERROR_NOT_READY),
        MAKE_STATUS_NAME(NVCV_ERROR_OUT_OF_MEMORY),
        MAKE_STATUS_NAME(NVCV_ERROR_INTERNAL),
        MAKE_STATUS_NAME(NVCV_ERROR_NOT_COMPATIBLE),
        MAKE_STATUS_NAME(NVCV_ERROR_OVERFLOW),
        MAKE_STATUS_NAME(NVCV_ERROR_UNDERFLOW),
    };
#ifndef ENABLE_SANITIZER
    params.emplace_back(static_cast<NVCVStatus>(255), "Unknown error");
#endif
    return params;
}

INSTANTIATE_TEST_SUITE_P(AllStatuses, StatusNameTest, t::ValuesIn(StatusNameParams()));

#undef MAKE_STATUS_NAME

TEST_P(StatusNameTest, get_name)
{
    NVCVStatus  status = std::get<0>(GetParam());
    const char *gold   = std::get<1>(GetParam());

    EXPECT_STREQ(gold, nvcvStatusGetName(status));
}

TEST(StatusTest, main_thread_has_success_status_by_default)
{
    EXPECT_EQ(NVCV_SUCCESS, nvcvGetLastError());
    EXPECT_EQ(NVCV_SUCCESS, nvcvPeekAtLastError());
}

TEST(StatusTest, get_last_status_msg_success_has_correct_message)
{
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_SUCCESS, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("success", msg.data());
}

TEST(StatusTest, get_last_status_resets_error_state)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, "%s", "");
    EXPECT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastError());
    EXPECT_EQ(NVCV_SUCCESS, nvcvGetLastError());
}

TEST(StatusTest, peek_last_status_doesnt_reset_error_state)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, "%s", "");
    EXPECT_EQ(NVCV_ERROR_DEVICE, nvcvPeekAtLastError());
    EXPECT_EQ(NVCV_ERROR_DEVICE, nvcvPeekAtLastError());
}

TEST(StatusTest, get_last_status_msg_error_has_correct_message)
{
    nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "test message");

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_INTERNAL, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message", msg.data());

    ASSERT_EQ(NVCV_SUCCESS, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("success", msg.data());
}

TEST(StatusTest, peek_at_last_status_msg_success_has_correct_message)
{
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPeekAtLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("success", msg.data());
}

TEST(StatusTest, function_success_doesnot_reset_status)
{
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageCalcRequirements(640, 480, NVCV_IMAGE_FORMAT_U8, 0, 0, nullptr));

    NVCVImageRequirements reqs;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageCalcRequirements(640, 480, NVCV_IMAGE_FORMAT_U8, 0, 0, &reqs));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvGetLastError());
}

TEST(StatusTest, peek_at_last_status_msg_error_has_correct_message)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, "test message");

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvPeekAtLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message", msg.data());

    msg[0] = '\0';
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvPeekAtLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message", msg.data());
}

TEST(StatusTest, set_thread_status_var_arg)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, "test message %d %c %s", 123, 'r', "lima");

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message 123 r lima", msg.data());
}

TEST(StatusTest, set_thread_status_var_arg_numeric_modifiers)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, "test message %ld %.2f %x %%", 123L, 4.5, 255U);

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message 123 4.50 ff %", msg.data());
}

TEST(StatusTest, set_thread_status_var_arg_list)
{
    auto fn = [](const char *fmt, ...)
    {
        va_list va;
        va_start(va, fmt);

        nvcvSetThreadStatusVarArgList(NVCV_ERROR_DEVICE, fmt, va);
        va_end(va);
    };

    fn("test message %d %s %c", 321, "rod", 'l');

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("test message 321 rod l", msg.data());
}

TEST(StatusTest, set_thread_status_var_arg_list_1)
{
    va_list va;
    nvcvSetThreadStatusVarArgList(NVCV_ERROR_DEVICE, nullptr, va);
    EXPECT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastError());
}

TEST(StatusTest, set_thread_status_null_message)
{
    nvcvSetThreadStatus(NVCV_ERROR_DEVICE, nullptr);

    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> msg;
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg.data(), msg.size()));
    EXPECT_STREQ("", msg.data());
}
