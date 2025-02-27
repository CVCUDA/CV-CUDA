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
#include "nvcv/src/priv/ColorFormat.hpp"
#include "nvcv/src/priv/ImageFormat.hpp"

TEST(ImageFormatTests, blockHeightLog2)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8_BL}.blockHeightLog2(), 1);
}

TEST(ImageFormatTests, colorRange)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}.colorRange(), NVCV_COLOR_RANGE_LIMITED);
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12_ER}.colorRange(), NVCV_COLOR_RANGE_FULL);

    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV21}.colorRange(), NVCV_COLOR_RANGE_LIMITED);
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV21_ER}.colorRange(), NVCV_COLOR_RANGE_FULL);
}

TEST(ImageFormatTests, newColorRange)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}.colorRange(NVCV_COLOR_RANGE_FULL),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12_ER});
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV21_ER}.colorRange(NVCV_COLOR_RANGE_LIMITED),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV21});

    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NONE}.colorRange(NVCV_COLOR_RANGE_FULL),
                 nvcv::priv::Exception);
}

TEST(ImageFormatTests, bitDepth)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.bitDepth(), 8);
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_2S16}.bitDepth(), 16);

    // The depth is different
#define NVCV_IMAGE_FORMAT_DIFF_DEPTH \
    NVCV_DETAIL_MAKE_COLOR_FMT3(RGB, UNDEFINED, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X32, X16)

    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_DIFF_DEPTH}.bitDepth(), 0);

#undef NVCV_IMAGE_FORMAT_DIFF_DEPTH
}

TEST(ImageFormatTests, newColorFormat)
{
    // Color test
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}.colorFormat(
                  nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12_ER}.colorFormat()),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12_ER});

    // Raw test
#define NVCV_IMAGE_FORMAT_TEST_RAW1 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_RGGB, PL, UNSIGNED, X000, ASSOCIATED, X8)
#define NVCV_IMAGE_FORMAT_TEST_RAW2 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_BGGR, PL, UNSIGNED, X000, ASSOCIATED, X8)

    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW1}.colorFormat(
                  nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW2}.colorFormat()),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW2});

#undef NVCV_IMAGE_FORMAT_TEST_RAW1
#undef NVCV_IMAGE_FORMAT_TEST_RAW2
}

TEST(ImageFormatTests, compareColorSpec)
{
    EXPECT_TRUE(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}.colorFormat()
                != nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.colorFormat());
}

TEST(ImageFormatTests, UpdateColorSpec)
{
#define NVCV_IMAGE_FORMAT_sRGB8 NVCV_DETAIL_MAKE_COLOR_FMT1(RGB, sRGB, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8)
#define NVCV_IMAGE_FORMAT_UYVY_UNDEFINED_SPEC \
    NVCV_DETAIL_MAKE_YCbCr_FMT1(UNDEFINED, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8)

    // Update None format
    EXPECT_THROW(nvcv::priv::UpdateColorSpec(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NONE},
                                             nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}.colorSpec()),
                 nvcv::priv::Exception);

    EXPECT_EQ(UpdateColorSpec(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}, NVCV_COLOR_SPEC_BT601),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8});

    EXPECT_EQ(UpdateColorSpec(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8}, NVCV_COLOR_SPEC_sRGB),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_sRGB8});

    EXPECT_EQ(UpdateColorSpec(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY_UNDEFINED_SPEC}, NVCV_COLOR_SPEC_BT601),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY});

#undef NVCV_IMAGE_FORMAT_sRGB8
#undef NVCV_IMAGE_FORMAT_UYVY_UNDEFINED_SPEC
}

TEST(ImageFormatTests, newDataKind)
{
    EXPECT_NE(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.dataKind(NVCV_DATA_KIND_UNSPECIFIED),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8});
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_S8}.dataKind(NVCV_DATA_KIND_UNSIGNED),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8});
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.dataKind(NVCV_DATA_KIND_SIGNED),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_S8});
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_C64}.dataKind(NVCV_DATA_KIND_FLOAT),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_F64});
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_F64}.dataKind(NVCV_DATA_KIND_COMPLEX),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_C64});
}

TEST(ImageFormatTests, StrNVCVImageFormat)
{
    EXPECT_EQ(StrNVCVImageFormat(NVCV_IMAGE_FORMAT_U8), "NVCV_IMAGE_FORMAT_U8");
    EXPECT_EQ(StrNVCVImageFormat(NVCV_IMAGE_FORMAT_BGRA8), "NVCV_IMAGE_FORMAT_BGRA8");
    EXPECT_EQ(StrNVCVImageFormat(NVCV_IMAGE_FORMAT_HSV8), "NVCV_IMAGE_FORMAT_HSV8");
    EXPECT_EQ(StrNVCVImageFormat(NVCV_IMAGE_FORMAT_UYVY), "NVCV_IMAGE_FORMAT_UYVY");
}

TEST(ImageFormatTests, newCSS)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.css(NVCV_CSS_NONE),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8});

    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.css(NVCV_CSS_420), nvcv::priv::Exception);
}

TEST(ImageFormatTests, newExtraChannelInfo_const)
{
    const NVCVExtraChannelInfo invalidDataKindInfo{1, 8, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_D};
    const NVCVExtraChannelInfo validInfo_2bits{1, 2, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};
    const NVCVExtraChannelInfo validInfo_32bits{1, 32, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};
    const NVCVExtraChannelInfo validInfo_64bits{1, 64, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};
    const NVCVExtraChannelInfo validInfo_128bits{1, 128, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};

    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.extraChannelInfo(&invalidDataKindInfo),
                 nvcv::priv::Exception);
    EXPECT_NO_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.extraChannelInfo(&validInfo_2bits));
    EXPECT_NO_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.extraChannelInfo(&validInfo_32bits));
    EXPECT_NO_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.extraChannelInfo(&validInfo_64bits));
    EXPECT_NO_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.extraChannelInfo(&validInfo_128bits));
}

TEST(ImageFormatTests, newRawPattern)
{
#define NVCV_IMAGE_FORMAT_TEST_RAW1 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_RGGB, PL, UNSIGNED, X000, ASSOCIATED, X8)
#define NVCV_IMAGE_FORMAT_TEST_RAW2 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_BGGR, PL, UNSIGNED, X000, ASSOCIATED, X8)

    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW1}.rawPattern(NVCV_RAW_BAYER_BGGR),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW2});

#undef NVCV_IMAGE_FORMAT_TEST_RAW1
#undef NVCV_IMAGE_FORMAT_TEST_RAW2

    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NONE}.rawPattern(NVCV_RAW_BAYER_RGGB),
                 nvcv::priv::Exception);
    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.rawPattern(NVCV_RAW_BAYER_RGGB), nvcv::priv::Exception);
}

TEST(ImageFormatTests, newSwizzleAndPacking)
{
    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_NONE}.swizzleAndPacking(
                     NVCV_SWIZZLE_X000, NVCV_PACKING_X32, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0),
                 nvcv::priv::Exception);

    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}.swizzleAndPacking(
                  NVCV_SWIZZLE_X000, NVCV_PACKING_X32, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0),
              nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U32});
}

TEST(ImageFormatTests, validateExtraChannelInfo)
{
    nvcv::priv::ColorFormat colorFmtRaw;
    colorFmtRaw.model = NVCV_COLOR_MODEL_RAW;
    colorFmtRaw.raw   = NVCV_RAW_BAYER_RGGB;

    // Invalid extra channel: numChannels > 7
    NVCVExtraChannelInfo invalidNumChannelsInfo{8, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};
    // Invalid extra channel: bitsPerPixel > 128
    NVCVExtraChannelInfo invalidBitsPerPixelInfo{2, 256, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};
    // Invalid extra channel: numChannels > 7
    NVCVExtraChannelInfo validExtraChannelInfo{2, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_D};

    EXPECT_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                         NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &invalidNumChannelsInfo),
                 nvcv::priv::Exception);

    EXPECT_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                         NVCV_SWIZZLE_0X00, NVCV_PACKING_0, NVCV_PACKING_X8, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &validExtraChannelInfo),
                 nvcv::priv::Exception);

    EXPECT_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                         NVCV_SWIZZLE_XY00, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &validExtraChannelInfo),
                 nvcv::priv::Exception);

    EXPECT_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                         NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &invalidBitsPerPixelInfo),
                 nvcv::priv::Exception);
}

TEST(ImageFormatTests, planeSwizzle)
{
    EXPECT_EQ(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGBAf16p}.planeSwizzle(3), NVCV_SWIZZLE_000X);
}

TEST(ImageFormatTests, bpc)
{
#define NVCV_IMAGE_FORMAT_INVALID_BPC \
    NVCV_DETAIL_MAKE_NONCOLOR_FMT4(PL, UNSIGNED, XYZW, ASSOCIATED, X16_Y16, X16_Y16, X16_Y16, X16_Y16)

    EXPECT_THROW(nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_INVALID_BPC}.bpc(), nvcv::priv::Exception);

#undef NVCV_IMAGE_FORMAT_INVALID_BPC
}

TEST(ImageFormatTests, FromPlanes)
{
#define NVCV_IMAGE_FORMAT_U8_INVALID NVCV_DETAIL_MAKE_NONCOLOR_FMT2(PL, UNSIGNED, XY00, ASSOCIATED, 0, X8)

    EXPECT_THROW(
        nvcv::priv::ImageFormat::FromPlanes(
            {nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8_INVALID}, nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8},
             nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}, nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_U8}}),
        nvcv::priv::Exception);

#undef NVCV_IMAGE_FORMAT_U8_INVALID

    EXPECT_THROW(
        nvcv::priv::ImageFormat::FromPlanes({nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8},
                                             nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8_UNASSOCIATED_ALPHA},
                                             nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8},
                                             nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8}}),
        nvcv::priv::Exception);

#define NVCV_IMAGE_FORMAT_TEST_RAW1 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_RGGB, PL, UNSIGNED, X000, ASSOCIATED, X8)
#define NVCV_IMAGE_FORMAT_TEST_RAW2 NVCV_DETAIL_MAKE_RAW_FMT1(BAYER_BGGR, PL, UNSIGNED, X000, ASSOCIATED, X8)

    EXPECT_THROW(nvcv::priv::ImageFormat::FromPlanes({nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW1},
                                                      nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW1},
                                                      nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW1},
                                                      nvcv::priv::ImageFormat{NVCV_IMAGE_FORMAT_TEST_RAW2}}),
                 nvcv::priv::Exception);

#undef NVCV_IMAGE_FORMAT_TEST_RAW1
#undef NVCV_IMAGE_FORMAT_TEST_RAW2
}

TEST(ImageFormatTests, constructor_0)
{
    nvcv::priv::ColorFormat colorFmtRaw;
    colorFmtRaw.model = NVCV_COLOR_MODEL_RAW;
    colorFmtRaw.raw   = NVCV_RAW_BAYER_RGGB;

    EXPECT_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                         NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
                 nvcv::priv::Exception);

    EXPECT_NO_THROW(nvcv::priv::ImageFormat(colorFmtRaw, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                            NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0,
                                            NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr));
}

TEST(ImageFormatTests, constructor_1)
{
    EXPECT_THROW(nvcv::priv::ImageFormat(NVCV_COLOR_MODEL_RAW, NVCV_COLOR_SPEC_BT601, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                         NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                         NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
                 nvcv::priv::Exception);

    EXPECT_THROW(
        nvcv::priv::ImageFormat(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_BT601, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
        nvcv::priv::Exception);

    EXPECT_THROW(
        nvcv::priv::ImageFormat(NVCV_COLOR_MODEL_UNDEFINED, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
        nvcv::priv::Exception);

    EXPECT_THROW(
        nvcv::priv::ImageFormat(NVCV_COLOR_MODEL_HSV, NVCV_COLOR_SPEC_BT601, NVCV_CSS_NONE, NVCV_MEM_LAYOUT_PL,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
        nvcv::priv::Exception);

    EXPECT_THROW(
        nvcv::priv::ImageFormat(NVCV_COLOR_MODEL_HSV, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, nullptr),
        nvcv::priv::Exception);
}

TEST(ImageFormatTests, operator_insertion)
{
    auto testOperatorInsertion = [](std::string expectedStr, NVCVImageFormat fmt) -> void
    {
        std::ostringstream ss;
        ss << nvcv::priv::ImageFormat{fmt};
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("NVCV_IMAGE_FORMAT_NONE", NVCV_IMAGE_FORMAT_NONE);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_U8", NVCV_IMAGE_FORMAT_U8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_U8_BL", NVCV_IMAGE_FORMAT_U8_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_S8", NVCV_IMAGE_FORMAT_S8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_U16", NVCV_IMAGE_FORMAT_U16);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_S16", NVCV_IMAGE_FORMAT_S16);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_S16_BL", NVCV_IMAGE_FORMAT_S16_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_U32", NVCV_IMAGE_FORMAT_U32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_S32", NVCV_IMAGE_FORMAT_S32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y8", NVCV_IMAGE_FORMAT_Y8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y8_BL", NVCV_IMAGE_FORMAT_Y8_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y8_ER", NVCV_IMAGE_FORMAT_Y8_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y8_ER_BL", NVCV_IMAGE_FORMAT_Y8_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y16", NVCV_IMAGE_FORMAT_Y16);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y16_BL", NVCV_IMAGE_FORMAT_Y16_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y16_ER", NVCV_IMAGE_FORMAT_Y16_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_Y16_ER_BL", NVCV_IMAGE_FORMAT_Y16_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV12", NVCV_IMAGE_FORMAT_NV12);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV12_BL", NVCV_IMAGE_FORMAT_NV12_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV12_ER", NVCV_IMAGE_FORMAT_NV12_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV12_ER_BL", NVCV_IMAGE_FORMAT_NV12_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV24", NVCV_IMAGE_FORMAT_NV24);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV24_BL", NVCV_IMAGE_FORMAT_NV24_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV24_ER", NVCV_IMAGE_FORMAT_NV24_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_NV24_ER_BL", NVCV_IMAGE_FORMAT_NV24_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGB8", NVCV_IMAGE_FORMAT_RGB8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGBA8", NVCV_IMAGE_FORMAT_RGBA8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_BGR8", NVCV_IMAGE_FORMAT_BGR8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_BGRA8", NVCV_IMAGE_FORMAT_BGRA8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_F32", NVCV_IMAGE_FORMAT_F32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_F64", NVCV_IMAGE_FORMAT_F64);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_2S16", NVCV_IMAGE_FORMAT_2S16);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_2S16_BL", NVCV_IMAGE_FORMAT_2S16_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_2F32", NVCV_IMAGE_FORMAT_2F32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_C64", NVCV_IMAGE_FORMAT_C64);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_2C64", NVCV_IMAGE_FORMAT_2C64);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_C128", NVCV_IMAGE_FORMAT_C128);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_2C128", NVCV_IMAGE_FORMAT_2C128);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_UYVY", NVCV_IMAGE_FORMAT_UYVY);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_UYVY_BL", NVCV_IMAGE_FORMAT_UYVY_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_UYVY_ER", NVCV_IMAGE_FORMAT_UYVY_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_UYVY_ER_BL", NVCV_IMAGE_FORMAT_UYVY_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_VYUY", NVCV_IMAGE_FORMAT_VYUY);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUYV_BL", NVCV_IMAGE_FORMAT_YUYV_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_VYUY_ER", NVCV_IMAGE_FORMAT_VYUY_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_VYUY_ER_BL", NVCV_IMAGE_FORMAT_VYUY_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUYV", NVCV_IMAGE_FORMAT_YUYV);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUYV_BL", NVCV_IMAGE_FORMAT_YUYV_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUYV_ER", NVCV_IMAGE_FORMAT_YUYV_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUYV_ER_BL", NVCV_IMAGE_FORMAT_YUYV_ER_BL);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUV8p", NVCV_IMAGE_FORMAT_YUV8p);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YUV8p_ER", NVCV_IMAGE_FORMAT_YUV8p_ER);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGB8_1U_U8", NVCV_IMAGE_FORMAT_RGB8_1U_U8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGB8_7U_U8", NVCV_IMAGE_FORMAT_RGB8_7U_U8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGBA8_3U_U16", NVCV_IMAGE_FORMAT_RGBA8_3U_U16);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGBA8_3POS3D_U32", NVCV_IMAGE_FORMAT_RGBA8_3POS3D_U32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGB8_3D_F32", NVCV_IMAGE_FORMAT_RGB8_3D_F32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_YCCK8", NVCV_IMAGE_FORMAT_YCCK8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_CMYK8", NVCV_IMAGE_FORMAT_CMYK8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_HSV8", NVCV_IMAGE_FORMAT_HSV8);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGBAf32", NVCV_IMAGE_FORMAT_RGBAf32);
    testOperatorInsertion("NVCV_IMAGE_FORMAT_RGBAf32p", NVCV_IMAGE_FORMAT_RGBAf32p);
}
