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
#include <nvcv/DataType.h>
#include <nvcv/DataType.hpp>
#include <nvcv/ImageFormat.hpp>

namespace t    = ::testing;
namespace util = nvcv::util;
namespace test = nvcv::test;

namespace {

struct Params
{
    NVCVDataType dtype;
    NVCVPacking  packing;
    NVCVDataKind dataKind;
    int          channels;
    int          bpp;
};

std::ostream &operator<<(std::ostream &out, const Params &p)
{
    return out << "dtype=" << nvcvDataTypeGetName(p.dtype) << ", dataKind=" << p.dataKind << ", packing=" << p.packing;
}

} // namespace

class DataTypeTests : public t::TestWithParam<Params>
{
};

#define FMT_DATA_PARAMS(DataKind, Packing) NVCV_PACKING_##Packing, NVCV_DATA_KIND_##DataKind

#define MAKE_DATA_TYPE(DataKind, Packing) \
    NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_##DataKind, NVCV_PACKING_##Packing), FMT_DATA_PARAMS(DataKind, Packing)

INSTANTIATE_TEST_SUITE_P(Range, DataTypeTests, t::Values(Params{MAKE_DATA_TYPE(FLOAT, X32_Y32_Z32_W32), 4, 128}));

INSTANTIATE_TEST_SUITE_P(ExplicitTypes, DataTypeTests,
                         t::Values(Params{NVCV_DATA_TYPE_U8, FMT_DATA_PARAMS(UNSIGNED, X8), 1, 8},
                                   Params{NVCV_DATA_TYPE_S8, FMT_DATA_PARAMS(SIGNED, X8), 1, 8},
                                   Params{NVCV_DATA_TYPE_U16, FMT_DATA_PARAMS(UNSIGNED, X16), 1, 16},
                                   Params{NVCV_DATA_TYPE_S16, FMT_DATA_PARAMS(SIGNED, X16), 1, 16},
                                   Params{NVCV_DATA_TYPE_2U8, FMT_DATA_PARAMS(UNSIGNED, X8_Y8), 2, 16},
                                   Params{NVCV_DATA_TYPE_3U8, FMT_DATA_PARAMS(UNSIGNED, X8_Y8_Z8), 3, 24},
                                   Params{NVCV_DATA_TYPE_4U8, FMT_DATA_PARAMS(UNSIGNED, X8_Y8_Z8_W8), 4, 32},
                                   Params{NVCV_DATA_TYPE_F32, FMT_DATA_PARAMS(FLOAT, X32), 1, 32},
                                   Params{NVCV_DATA_TYPE_F64, FMT_DATA_PARAMS(FLOAT, X64), 1, 64},
                                   Params{MAKE_DATA_TYPE(UNSIGNED, X10Y11Z11), 3, 32},
                                   Params{MAKE_DATA_TYPE(UNSIGNED, X5Y6Z5), 3, 16},
                                   Params{MAKE_DATA_TYPE(UNSIGNED, X10b6_Y10b6), 2, 32},
                                   Params{NVCV_DATA_TYPE_2F32, FMT_DATA_PARAMS(FLOAT, X32_Y32), 2, 64}));

TEST_P(DataTypeTests, get_comp_packing_works)
{
    const Params &p = GetParam();

    NVCVPacking packing;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetPacking(p.dtype, &packing));
    EXPECT_EQ(p.packing, packing);
}

TEST_P(DataTypeTests, get_bpp_works)
{
    const Params &p = GetParam();

    int32_t bpp;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetBitsPerPixel(p.dtype, &bpp));
    EXPECT_EQ(p.bpp, bpp);
}

TEST_P(DataTypeTests, get_bpc_works)
{
    const Params &p = GetParam();

    int32_t bits[4];
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetBitsPerChannel(p.dtype, bits));

    NVCVPacking packing;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetPacking(p.dtype, &packing));

    int32_t goldbits[4];
    ASSERT_EQ(NVCV_SUCCESS, nvcvPackingGetBitsPerComponent(packing, goldbits));

    EXPECT_EQ(bits[0], goldbits[0]);
    EXPECT_EQ(bits[1], goldbits[1]);
    EXPECT_EQ(bits[2], goldbits[2]);
    EXPECT_EQ(bits[3], goldbits[3]);
}

TEST_P(DataTypeTests, get_datatype_works)
{
    const Params &p = GetParam();

    NVCVDataKind dataKind;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetDataKind(p.dtype, &dataKind));
    EXPECT_EQ(p.dataKind, dataKind);
}

TEST_P(DataTypeTests, get_channel_count)
{
    const Params &p = GetParam();

    int channels;

    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetNumChannels(p.dtype, &channels));
    EXPECT_EQ(p.channels, channels);
}

TEST_P(DataTypeTests, make_pixel_type)
{
    const Params &p = GetParam();

    NVCVDataType pix;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeDataType(&pix, p.dataKind, p.packing));
    EXPECT_EQ(p.dtype, pix);
}

class ImageDataTypeTests : public t::TestWithParam<std::tuple<nvcv::ImageFormat, nvcv::DataType>>
{
};

INSTANTIATE_TEST_SUITE_P(ExplicitTypes, ImageDataTypeTests,
                         t::Values(std::make_tuple(NVCV_IMAGE_FORMAT_U8, NVCV_DATA_TYPE_U8),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_S8, NVCV_DATA_TYPE_S8),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_U16, NVCV_DATA_TYPE_U16),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_S16, NVCV_DATA_TYPE_S16),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_F32, NVCV_DATA_TYPE_F32),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_2F32, NVCV_DATA_TYPE_2F32)));

TEST_P(ImageDataTypeTests, pixel_type_matches_corresponding_image_type)
{
    NVCVImageFormat imgFormat = std::get<0>(GetParam());
    NVCVDataType    dtype     = std::get<1>(GetParam());

    EXPECT_EQ((uint64_t)imgFormat, (uint64_t)dtype);
}

TEST(DataTypeTests, pixel_type_none_must_be_0)
{
    EXPECT_EQ(0, (int)NVCV_DATA_TYPE_NONE);
}

TEST(DataTypeTests, make_pixel_type_macro)
{
    NVCVDataType dtype;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeDataType(&dtype, NVCV_DATA_KIND_UNSIGNED, NVCV_PACKING_X8_Y8_Z8));

    EXPECT_EQ(dtype, NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_UNSIGNED, NVCV_PACKING_X8_Y8_Z8));
}

TEST(DataTypeTests, get_name_predefined)
{
    EXPECT_STREQ("NVCV_DATA_TYPE_U8", nvcvDataTypeGetName(NVCV_DATA_TYPE_U8));
}

TEST(DataTypeTests, get_name_not_predefined)
{
    NVCVDataType pix = NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_FLOAT, NVCV_PACKING_X8_Y8_Z8);

    EXPECT_STREQ("NVCVDataType(FLOAT,X8_Y8_Z8)", nvcvDataTypeGetName(pix));
}

class ChannelDataTypeTests : public t::TestWithParam<std::tuple<NVCVDataType, int, NVCVDataType, NVCVStatus>>
{
};

#define MAKE_DATA_TYPE_ABBREV(datatype, packing) NVCV_MAKE_DATA_TYPE(NVCV_DATA_KIND_##datatype, NVCV_PACKING_##packing)

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(
    Positive, ChannelDataTypeTests,
    test::ValueList<NVCVDataType, int, NVCVDataType>
    {
        {NVCV_DATA_TYPE_U8,                          0, NVCV_DATA_TYPE_U8},
        {MAKE_DATA_TYPE_ABBREV(FLOAT, X32),      0, MAKE_DATA_TYPE_ABBREV(FLOAT, X32)},
        {MAKE_DATA_TYPE_ABBREV(FLOAT, X8_Y8_Z8), 2, MAKE_DATA_TYPE_ABBREV(FLOAT, X8)},
        {NVCV_DATA_TYPE_2F32,                        0, NVCV_DATA_TYPE_F32},
        {NVCV_DATA_TYPE_2F32,                        1, NVCV_DATA_TYPE_F32},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X64_Y64_Z64_W64), 3, NVCV_DATA_TYPE_S64},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X4Y4),  0, MAKE_DATA_TYPE_ABBREV(UNSIGNED, X4)},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, b4X4Y4Z4), 2, MAKE_DATA_TYPE_ABBREV(UNSIGNED, X4)},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X3Y3Z2), 2, MAKE_DATA_TYPE_ABBREV(UNSIGNED, X2)},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X1),    0, MAKE_DATA_TYPE_ABBREV(UNSIGNED, X1)},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X2),    0, MAKE_DATA_TYPE_ABBREV(UNSIGNED, X2)},
        {NVCV_DATA_TYPE_3S16,                        2, NVCV_DATA_TYPE_S16},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X24),     0, MAKE_DATA_TYPE_ABBREV(SIGNED, X24)},
        {NVCV_DATA_TYPE_2F64,                        1, NVCV_DATA_TYPE_F64},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X96),     0, MAKE_DATA_TYPE_ABBREV(SIGNED, X96)},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X128),    0, MAKE_DATA_TYPE_ABBREV(SIGNED, X128)},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X192),    0, MAKE_DATA_TYPE_ABBREV(SIGNED, X192)},
        {MAKE_DATA_TYPE_ABBREV(SIGNED, X256),    0, MAKE_DATA_TYPE_ABBREV(SIGNED, X256)},
        {NVCV_DATA_TYPE_U8,                          1, NVCV_DATA_TYPE_NONE},
        {NVCV_DATA_TYPE_2F32,                        2, NVCV_DATA_TYPE_NONE},
        {MAKE_DATA_TYPE_ABBREV(UNSIGNED, b4X4Y4Z4), 3, NVCV_DATA_TYPE_NONE},
    }
        * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(
    Negative, ChannelDataTypeTests,
    test::ValueList<NVCVDataType, int>
    {
          {NVCV_DATA_TYPE_U8, -1},
          // WAR {MAKE_DATA_TYPE_ABBREV(UNSIGNED, b2X14), 0},
          {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X5Y5b1Z5), 0},
          {MAKE_DATA_TYPE_ABBREV(UNSIGNED, X3Y3Z2), 1}
    }
    * NVCV_DATA_TYPE_NONE * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(ChannelDataTypeTests, get_channel_type)
{
    NVCVDataType test       = std::get<0>(GetParam());
    int          channel    = std::get<1>(GetParam());
    NVCVDataType gold       = std::get<2>(GetParam());
    NVCVStatus   goldStatus = std::get<3>(GetParam());

    NVCVDataType pix;
    ASSERT_EQ(goldStatus, nvcvDataTypeGetChannelType(test, channel, &pix));
    if (goldStatus == NVCV_SUCCESS)
    {
        EXPECT_EQ(gold, pix);
    }
}

class DataTypeStrideTests
    : public t::TestWithParam<std::tuple<test::Param<"pix", NVCVDataType>, test::Param<"goldStride", int>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_,DataTypeStrideTests,
                              test::ValueList<NVCVDataType, int>
                              {
                                {NVCV_DATA_TYPE_U8, 1},
                                {NVCV_DATA_TYPE_U16, 2},
                                {NVCV_DATA_TYPE_3S8, 3},
                                {NVCV_DATA_TYPE_2U16, 4},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X256), 32},
                                {NVCV_DATA_TYPE_2F32, 8},
                                {MAKE_DATA_TYPE_ABBREV(FLOAT, X8_Y8_Z8), 3},
                              });

// clang-format on

TEST_P(DataTypeStrideTests, works)
{
    const NVCVDataType dtype      = std::get<0>(GetParam());
    const int          goldStride = std::get<1>(GetParam());

    int32_t testStride;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetStrideBytes(dtype, &testStride));
    EXPECT_EQ(goldStride, testStride);
}

// clang-format off
NVCV_TEST_SUITE_P(DataTypeAlignmentTests,
                              test::ValueList<test::Param<"pix", NVCVDataType>, test::Param<"goldAlign", int>>
                              {
                                {NVCV_DATA_TYPE_U8, 1},
                                {NVCV_DATA_TYPE_U16, 2},
                                {NVCV_DATA_TYPE_3S8, 1},
                                {NVCV_DATA_TYPE_2U16, 2},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X256), 32},
                                {NVCV_DATA_TYPE_2F32, 4},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X5Y1Z5W5), 2},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X3Y3Z2), 1},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X4b4), 1},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, b4X4), 1},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, b4X12), 2},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X10b6), 2},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X4Y4Z4W4), 2},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X8_Y8__X8_Z8), 1},
                                {MAKE_DATA_TYPE_ABBREV(SIGNED, X32_Y24b8), 4},
                              });

// clang-format on

TEST_P(DataTypeAlignmentTests, works)
{
    const NVCVDataType dtype     = std::get<0>(GetParam());
    const int          goldAlign = std::get<1>(GetParam());

    int32_t testAlign;
    ASSERT_EQ(NVCV_SUCCESS, nvcvDataTypeGetAlignment(dtype, &testAlign));
    EXPECT_EQ(goldAlign, testAlign);
}
