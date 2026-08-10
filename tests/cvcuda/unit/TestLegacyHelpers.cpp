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

#include "Definitions.hpp"

#include <cvcuda/priv/legacy/CvCudaLegacyHelpers.hpp>
#include <nvcv/Exception.hpp>

namespace {

using nvcv::legacy::cuda_op::ErrorCode;
using nvcv::legacy::helpers::GetLegacyDataFormat;
using nvcv::legacy::helpers::GetLegacyDataType;

TEST(LegacyHelpersTest, RejectsUnsupportedFormatsAndTypes)
{
    EXPECT_THROW(GetLegacyDataFormat(3, 2, 1), nvcv::Exception);
    EXPECT_THROW(GetLegacyDataFormat(nvcv::TensorLayout("HW")), nvcv::Exception);

    EXPECT_THROW(GetLegacyDataType(24, nvcv::DataKind::FLOAT), nvcv::Exception);
    EXPECT_THROW(GetLegacyDataType(24, nvcv::DataKind::SIGNED), nvcv::Exception);
    EXPECT_THROW(GetLegacyDataType(8, nvcv::DataKind::COMPLEX), nvcv::Exception);
    EXPECT_THROW(GetLegacyDataType(8, nvcv::DataKind::UNSPECIFIED), nvcv::Exception);
}

TEST(LegacyHelpersTest, TranslatesSuccessAndUnknownErrors)
{
    EXPECT_EQ(nvcv::util::TranslateError(ErrorCode::SUCCESS), NVCV_SUCCESS);
    EXPECT_EQ(nvcv::util::TranslateError(static_cast<ErrorCode>(99)), NVCV_ERROR_INTERNAL);
}

TEST(LegacyHelpersTest, DescribesSuccessAndUnknownErrors)
{
    const char *description = nullptr;

    EXPECT_STREQ(nvcv::util::ToString(ErrorCode::SUCCESS, &description), "SUCCESS");
    EXPECT_STREQ(description, "Operation executed successfully");

    EXPECT_STREQ(nvcv::util::ToString(static_cast<ErrorCode>(99), &description), "UNKNOWN");
    EXPECT_STREQ(description, "Unknown error");
}

} // namespace
