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

#include "Definitions.hpp"

#include <common/ValueTests.hpp>
#include <operators/priv/legacy/CvCudaLegacyHelpers.hpp>

namespace gt      = ::testing;
namespace test    = nvcv::test;
namespace util    = nvcv::util;
namespace legOp   = nvcv::legacy::cuda_op;
namespace helpers = nvcv::legacy::helpers;

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyFormatHelpers, test::ValueList<legOp::DataFormat, int32_t, int32_t, int32_t>
{
    // ret, chan, plane, batch
     { legOp::DataFormat::kNCHW, 4, 4, 2},
     { legOp::DataFormat::kNHWC, 4, 1, 2},
     { legOp::DataFormat::kCHW,  4, 4, 1},
     { legOp::DataFormat::kHWC,  4, 1, 1},

     { legOp::DataFormat::kNCHW, 3, 3, 2},
     { legOp::DataFormat::kNHWC, 3, 1, 2},
     { legOp::DataFormat::kCHW,  3, 3, 1},
     { legOp::DataFormat::kHWC,  3, 1, 1},

     { legOp::DataFormat::kNCHW, 2, 2, 2},
     { legOp::DataFormat::kNHWC, 2, 1, 2},
     { legOp::DataFormat::kCHW,  2, 2, 1},
     { legOp::DataFormat::kHWC,  2, 1, 1},

     { legOp::DataFormat::kNHWC, 1, 1, 2},
     { legOp::DataFormat::kHWC,  1, 1, 1},
});

// clang-format on

TEST_P(CheckLegacyFormatHelpers, check_conversion_to_legacy_data_format)
{
    legOp::DataFormat gold      = GetParamValue<0>();
    int32_t           numCh     = GetParamValue<1>();
    int32_t           numPlanes = GetParamValue<2>();
    int32_t           numBatch  = GetParamValue<3>();
    EXPECT_EQ(gold, helpers::GetLegacyDataFormat(numCh, numPlanes, numBatch));
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyFormatHelpersInvalid, test::ValueList<legOp::DataFormat, int32_t, int32_t, int32_t>
{
    // semi planar not supported
     { legOp::DataFormat::kNCHW, 4, 3, 1},

});

// clang-format on
TEST_P(CheckLegacyFormatHelpersInvalid, check_conversion_to_legacy_data_format_invalid)
{
    int32_t numCh     = GetParamValue<1>();
    int32_t numPlanes = GetParamValue<2>();
    int32_t numBatch  = GetParamValue<3>();
    EXPECT_THROW(helpers::GetLegacyDataFormat(numCh, numPlanes, numBatch), nvcv::Exception);
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyHelpersDataType, test::ValueList<legOp::DataType, int32_t, nvcv::DataKind>
{
    // type, bpp, cv type
     { legOp::DataType::kCV_8U , 8, nvcv::DataKind::UNSIGNED},
     { legOp::DataType::kCV_8S , 8, nvcv::DataKind::SIGNED},
     { legOp::DataType::kCV_16U, 16, nvcv::DataKind::UNSIGNED},
     { legOp::DataType::kCV_16S, 16, nvcv::DataKind::SIGNED},
     { legOp::DataType::kCV_32S, 32, nvcv::DataKind::SIGNED},
     { legOp::DataType::kCV_32F, 32, nvcv::DataKind::FLOAT},
     { legOp::DataType::kCV_64F, 64, nvcv::DataKind::FLOAT},
     { legOp::DataType::kCV_16F, 16, nvcv::DataKind::FLOAT},
});

// clang-format on
TEST_P(CheckLegacyHelpersDataType, check_conversion_to_legacy_data_type)
{
    legOp::DataType expect = GetParamValue<0>();
    int32_t         bpp    = GetParamValue<1>();
    nvcv::DataKind  type   = GetParamValue<2>();

    EXPECT_EQ(expect, helpers::GetLegacyDataType(bpp, type));
}
