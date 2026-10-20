/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "TestCAPIEnum.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace {

struct CAPIEnumCase
{
    TestCAPIEnumCase testCase;
    const char      *name;
    const char      *message;
};

constexpr std::array kCAPIEnumCases = {
#define TEST_CAPI_ENUM_VALUE(Id, Name, Message) CAPIEnumCase{Id, Name, Message},
    TEST_CAPI_ENUM_CASES(TEST_CAPI_ENUM_VALUE)
#undef TEST_CAPI_ENUM_VALUE
};

class CAPIEnumTest : public testing::TestWithParam<CAPIEnumCase>
{
};

TEST_P(CAPIEnumTest, rejects_invalid_object_representation_at_c_api_boundary)
{
    std::array<char, NVCV_MAX_STATUS_MESSAGE_LENGTH> message{};
    nvcvGetLastError();

    // Non-enum resources are intentionally unusable dummy values. The argument-specific diagnostic proves the raw
    // enum guard ran before any such resource could be used, without relying on sanitizer instrumentation.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, TestCAPIEnumCallWithInvalidValue(GetParam().testCase));
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvGetLastErrorMessage(message.data(), static_cast<int32_t>(message.size())));
    EXPECT_NE(std::string::npos, std::string{message.data()}.find(GetParam().message)) << message.data();
}

INSTANTIATE_TEST_SUITE_P(EnumArguments, CAPIEnumTest, testing::ValuesIn(kCAPIEnumCases),
                         [](const testing::TestParamInfo<CAPIEnumCase> &info) { return info.param.name; });

} // namespace
