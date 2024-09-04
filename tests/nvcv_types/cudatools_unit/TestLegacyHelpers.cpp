/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/ValueTests.hpp>
#include <cvcuda/priv/legacy/CvCudaLegacyHelpers.hpp>

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
    nvcv::DataKind  kind   = GetParamValue<2>();

    EXPECT_EQ(expect, helpers::GetLegacyDataType(bpp, kind));
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyHelpersDataTypeInvalid, test::ValueList<int32_t, nvcv::DataKind>
{
    {8, nvcv::DataKind::FLOAT},
    {64, nvcv::DataKind::SIGNED},
    {64, nvcv::DataKind::UNSIGNED},
    {32, nvcv::DataKind::COMPLEX},
    {32, nvcv::DataKind::UNSPECIFIED}
});

// clang-format on

TEST_P(CheckLegacyHelpersDataTypeInvalid, check_conversion_to_legacy_data_type_invalid)
{
    int32_t        bpp  = GetParamValue<0>();
    nvcv::DataKind kind = GetParamValue<1>();
    EXPECT_THROW(helpers::GetLegacyDataType(bpp, kind), nvcv::Exception);
}

TEST(CheckLegacyHelpersDataFormat, check_image_batch_invalid_different_fmt)
{
    nvcv::ImageBatchVarShape imgBatch(2);
    imgBatch.pushBack(nvcv::Image{
        nvcv::Size2D{24, 24},
        nvcv::FMT_NV12
    });
    imgBatch.pushBack(nvcv::Image{
        nvcv::Size2D{24, 24},
        nvcv::FMT_U8
    });
    EXPECT_THROW(helpers::GetLegacyDataFormat(imgBatch), nvcv::Exception);
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyHelpersDataFormat, test::ValueList<legOp::DataFormat, int32_t, nvcv::ImageFormat>
{
    {legOp::DataFormat::kNCHW, 2, nvcv::FMT_RGB8p},
    {legOp::DataFormat::kCHW, 1, nvcv::FMT_RGB8p},
    {legOp::DataFormat::kNHWC, 2, nvcv::FMT_RGB8},
    {legOp::DataFormat::kHWC, 1, nvcv::FMT_RGB8},
});

// clang-format on

TEST_P(CheckLegacyHelpersDataFormat, check_image_batch_conversion)
{
    legOp::DataFormat expect    = GetParamValue<0>();
    int32_t           batchSize = GetParamValue<1>();
    nvcv::ImageFormat fmt       = GetParamValue<2>();

    nvcv::ImageBatchVarShape imgBatch(batchSize);
    for (auto i = 0; i < batchSize; ++i)
    {
        imgBatch.pushBack(nvcv::Image{
            nvcv::Size2D{24, 24},
            fmt
        });
    }

    EXPECT_EQ(helpers::GetLegacyDataFormat(imgBatch), expect);
}

TEST_P(CheckLegacyHelpersDataFormat, check_image_batch_conversion_exported)
{
    legOp::DataFormat expect    = GetParamValue<0>();
    int32_t           batchSize = GetParamValue<1>();
    nvcv::ImageFormat fmt       = GetParamValue<2>();

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape imgBatch(batchSize);
    for (auto i = 0; i < batchSize; ++i)
    {
        imgBatch.pushBack(nvcv::Image{
            nvcv::Size2D{24, 24},
            fmt
        });
    }

    auto exportedData = imgBatch.exportData<nvcv::ImageBatchVarShapeDataStridedCuda>(stream);

    EXPECT_EQ(helpers::GetLegacyDataFormat(exportedData.value()), expect);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyTranslateError, test::ValueList<NVCVStatus, legOp::ErrorCode>
{
    {NVCV_SUCCESS, legOp::ErrorCode::SUCCESS},
    {NVCV_ERROR_INVALID_ARGUMENT, legOp::ErrorCode::INVALID_PARAMETER},
    {NVCV_ERROR_INVALID_ARGUMENT, legOp::ErrorCode::INVALID_DATA_FORMAT},
    {NVCV_ERROR_INVALID_ARGUMENT, legOp::ErrorCode::INVALID_DATA_SHAPE},
    {NVCV_ERROR_INVALID_ARGUMENT, legOp::ErrorCode::INVALID_DATA_TYPE}
});

// clang-format on

TEST_P(CheckLegacyTranslateError, check_error_conversion)
{
    NVCVStatus       expect = GetParamValue<0>();
    legOp::ErrorCode err    = GetParamValue<1>();
    EXPECT_EQ(nvcv::util::TranslateError(err), expect);
}

// clang-format off
NVCV_TEST_SUITE_P(CheckLegacyToString, test::ValueList<legOp::ErrorCode, std::string, std::string>
{
    {legOp::ErrorCode::SUCCESS, "SUCCESS", "Operation executed successfully"},
    {legOp::ErrorCode::INVALID_PARAMETER, "INVALID_PARAMETER", "Some parameter is outside its acceptable range"},
    {legOp::ErrorCode::INVALID_DATA_FORMAT, "INVALID_DATA_FORMAT", "Data format is outside its acceptable range"},
    {legOp::ErrorCode::INVALID_DATA_SHAPE, "INVALID_DATA_SHAPE", "Tensor shape is outside its acceptable range"},
    {legOp::ErrorCode::INVALID_DATA_TYPE, "INVALID_DATA_TYPE", "Data type is outside its acceptable range"}
});

// clang-format on

TEST_P(CheckLegacyToString, check_error_to_string_conversion)
{
    legOp::ErrorCode err               = GetParamValue<0>();
    std::string      expectedErrorName = GetParamValue<1>();
    std::string      expectedDescr     = GetParamValue<2>();

    char        bufferDesc[256];
    const char *bufferDescPtr = bufferDesc;
    const char *buffer        = nvcv::util::ToString(err, &bufferDescPtr);

    EXPECT_STREQ(bufferDescPtr, expectedDescr.c_str());
    EXPECT_STREQ(buffer, expectedErrorName.c_str());
}
