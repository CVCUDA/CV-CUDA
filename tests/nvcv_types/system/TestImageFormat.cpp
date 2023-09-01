/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvcv/ImageFormat.h>
#include <nvcv/ImageFormat.hpp>
#include <util/Assert.h>
#include <util/Compiler.hpp>
#include <util/Size.hpp>

#include <unordered_set>

namespace t    = ::testing;
namespace util = nvcv::util;
namespace test = nvcv::test;

namespace {

struct Params
{
    NVCVImageFormat       imgFormat;
    NVCVColorModel        colorModel;
    NVCVColorSpec         colorSpec;
    NVCVChromaSubsampling chromaSubSamp;
    NVCVMemLayout         memLayout;
    NVCVSwizzle           swizzle;
    NVCVAlphaType         alphaType;
    NVCVPacking           packing0, packing1, packing2, packing3;
    NVCVDataKind          dataKind;
    int                   samplesHoriz, samplesVert;
    NVCVChromaLocation    locHoriz, locVert;
    int                   bitsPerChannel[4] = {};
    int                   planeCount;

    struct
    {
        int          bpp       = 0;
        int          channels  = 0;
        NVCVDataType pixFormat = NVCV_DATA_TYPE_NONE;
        NVCVSwizzle  swizzle   = NVCV_SWIZZLE_0000;
    } planes[4];

    NVCVExtraChannelInfo exChannelInfo;
};

std::ostream &operator<<(std::ostream &out, const Params &p)
{
    return out << "imgFormat=" << nvcvImageFormatGetName(p.imgFormat) << ", colorModel=" << p.colorModel
               << ", colorSpec=" << p.colorSpec << ", chromaSubSamp=" << p.chromaSubSamp
               << ", memLayout=" << p.memLayout << ", swizzle=" << p.swizzle << ", packing0=" << p.packing0
               << ", packing1=" << p.packing1 << ", packing2=" << p.packing2 << ", packing3=" << p.packing3
               << ", dataKind=" << p.dataKind << ", exChannelInfo.numChannels= " << p.exChannelInfo.numChannels
               << ", exChannelInfo.bitsPerPixel= " << p.exChannelInfo.bitsPerPixel
               << ", exChannelInfo.datakind= " << p.exChannelInfo.datakind
               << ", exChannelInfo.channelType = " << p.exChannelInfo.channelType << ", alphaType= " << p.alphaType;
}

} // namespace

class ImageFormatTests : public t::TestWithParam<Params>
{
};

#define FMT_IMAGE_PARAMS(ColorModel, ColorStd, Subsampling, MemLayout, DataKind, Swizzle, AlphaType, Packing0,      \
                         Packing1, Packing2, Packing3)                                                              \
    NVCV_COLOR_MODEL_##ColorModel, NVCV_COLOR_SPEC_##ColorStd, NVCV_CSS_##Subsampling, NVCV_MEM_LAYOUT_##MemLayout, \
        NVCV_SWIZZLE_##Swizzle, NVCV_ALPHA_##AlphaType, NVCV_PACKING_##Packing0, NVCV_PACKING_##Packing1,           \
        NVCV_PACKING_##Packing2, NVCV_PACKING_##Packing3, NVCV_DATA_KIND_##DataKind

#define MAKE_IMAGE_FORMAT_ENUM(ColorModel, ColorStd, Subsampling, Layout, DataKind, Swizzle, AlphaType, Packing0,     \
                               Packing1, Packing2, Packing3)                                                          \
    NVCV_MAKE_IMAGE_FORMAT_ENUM(ColorModel, ColorStd, Subsampling, Layout, DataKind, Swizzle, AlphaType, 4, Packing0, \
                                Packing1, Packing2, Packing3),                                                        \
        FMT_IMAGE_PARAMS(ColorModel, ColorStd, Subsampling, Layout, DataKind, Swizzle, AlphaType, Packing0, Packing1, \
                         Packing2, Packing3)

INSTANTIATE_TEST_SUITE_P(
    ExplicitTypes, ImageFormatTests,
    t::Values(
        Params{
            NVCV_IMAGE_FORMAT_U8,
            FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, UNSIGNED, X000, ASSOCIATED, X8, 0, 0, 0),
            4,
            4,
            NVCV_CHROMA_LOC_EVEN,
            NVCV_CHROMA_LOC_EVEN,
            {                                           8, 0,                          0,                    0},
            1,
            {{8, 1, NVCV_DATA_TYPE_U8, NVCV_SWIZZLE_X000}  },
            {                                           0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U }
},
        Params{NVCV_IMAGE_FORMAT_S8,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, SIGNED, X000, ASSOCIATED, X8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 0, 0, 0},
               1,
               {{8, 1, NVCV_DATA_TYPE_S8, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_U16,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, UNSIGNED, X000, ASSOCIATED, X16, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {16, 0, 0, 0},
               1,
               {{16, 1, NVCV_DATA_TYPE_U16, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_S16,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, SIGNED, X000, ASSOCIATED, X16, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {16, 0, 0, 0},
               1,
               {{16, 1, NVCV_DATA_TYPE_S16, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_NV12_ER,
               FMT_IMAGE_PARAMS(YCbCr, BT601_ER, 420, PL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8, 0, 0),
               2,
               2,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               2,
               {{8, 1, NVCV_DATA_TYPE_U8, NVCV_SWIZZLE_X000}, {16, 2, NVCV_DATA_TYPE_2U8, NVCV_SWIZZLE_0XY0}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_YUYV_ER,
               FMT_IMAGE_PARAMS(YCbCr, BT601_ER, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8__X8_Z8, 0, 0, 0),
               2,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               1,
               {{16, 3, NVCV_DATA_TYPE_2U8, NVCV_SWIZZLE_XYZ1}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_UYVY_ER,
               FMT_IMAGE_PARAMS(YCbCr, BT601_ER, 422, PL, UNSIGNED, XYZ1, ASSOCIATED, Y8_X8__Z8_X8, 0, 0, 0),
               2,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               1,
               {{16, 3, NVCV_DATA_TYPE_2U8, NVCV_SWIZZLE_XYZ1}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_RGB8,
               FMT_IMAGE_PARAMS(RGB, UNDEFINED, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               1,
               {{24, 3, NVCV_DATA_TYPE_3U8, NVCV_SWIZZLE_XYZ1}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_RGBA8,
               FMT_IMAGE_PARAMS(RGB, UNDEFINED, NONE, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 8},
               1,
               {{32, 4, NVCV_DATA_TYPE_4U8, NVCV_SWIZZLE_XYZW}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_F16,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, FLOAT, X000, ASSOCIATED, X16, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {16, 0, 0, 0},
               1,
               {{16, 1, NVCV_DATA_TYPE_F16, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_F32,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, FLOAT, X000, ASSOCIATED, X32, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {32, 0, 0, 0},
               1,
               {{32, 1, NVCV_DATA_TYPE_F32, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_F64,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, FLOAT, X000, ASSOCIATED, X64, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {64, 0, 0, 0},
               1,
               {{64, 1, NVCV_DATA_TYPE_F64, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_2F32,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, FLOAT, XY00, ASSOCIATED, X32_Y32, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {32, 32, 0, 0},
               1,
               {{64, 2, NVCV_DATA_TYPE_2F32, NVCV_SWIZZLE_XY00}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_C64,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, COMPLEX, X000, ASSOCIATED, X64, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {64, 0, 0, 0},
               1,
               {{64, 1, NVCV_DATA_TYPE_C64, NVCV_SWIZZLE_X000}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_2C64,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, COMPLEX, XY00, ASSOCIATED, X64_Y64, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {64, 64, 0, 0},
               1,
               {{128, 2, NVCV_DATA_TYPE_2C64, NVCV_SWIZZLE_XY00}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_BGR8,
               FMT_IMAGE_PARAMS(RGB, UNDEFINED, NONE, PL, UNSIGNED, ZYX1, ASSOCIATED, X8_Y8_Z8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               1,
               {{24, 3, NVCV_DATA_TYPE_3U8, NVCV_SWIZZLE_ZYX1}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_BGRA8,
               FMT_IMAGE_PARAMS(RGB, UNDEFINED, NONE, PL, UNSIGNED, ZYXW, ASSOCIATED, X8_Y8_Z8_W8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 8},
               1,
               {{32, 4, NVCV_DATA_TYPE_4U8, NVCV_SWIZZLE_ZYXW}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_2S16,
               FMT_IMAGE_PARAMS(UNDEFINED, UNDEFINED, NONE, PL, SIGNED, XY00, ASSOCIATED, X16_Y16, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {16, 16, 0, 0},
               1,
               {{32, 2, NVCV_DATA_TYPE_2S16, NVCV_SWIZZLE_XY00}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_RGB8_1U_U8,
               FMT_IMAGE_PARAMS(RGB, UNDEFINED, NONE, PL, UNSIGNED, XYZ1, ASSOCIATED, X8_Y8_Z8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 0},
               1,
               {{24, 3, NVCV_DATA_TYPE_3U8, NVCV_SWIZZLE_XYZ1}},
               {1, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_YCCK8,
               FMT_IMAGE_PARAMS(YCCK, UNDEFINED, NONE, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 8},
               1,
               {{32, 4, NVCV_DATA_TYPE_4U8, NVCV_SWIZZLE_XYZW}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}},
        Params{NVCV_IMAGE_FORMAT_CMYK8,
               FMT_IMAGE_PARAMS(CMYK, UNDEFINED, NONE, PL, UNSIGNED, XYZW, ASSOCIATED, X8_Y8_Z8_W8, 0, 0, 0),
               4,
               4,
               NVCV_CHROMA_LOC_EVEN,
               NVCV_CHROMA_LOC_EVEN,
               {8, 8, 8, 8},
               1,
               {{32, 4, NVCV_DATA_TYPE_4U8, NVCV_SWIZZLE_XYZW}},
               {0, 0, NVCV_DATA_KIND_UNSPECIFIED, NVCV_EXTRA_CHANNEL_U}}));

TEST_P(ImageFormatTests, make_image_format)
{
    const Params &p = GetParam();

    NVCVImageFormat             fmt;
    const NVCVExtraChannelInfo *exChannelInfo;
    if (p.exChannelInfo.numChannels == 0)
    {
        exChannelInfo = nullptr;
    }
    else
    {
        exChannelInfo = &p.exChannelInfo;
    }
    if (p.colorModel == NVCV_COLOR_MODEL_YCbCr)
    {
        ASSERT_EQ(NVCV_SUCCESS,
                  nvcvMakeYCbCrImageFormat(&fmt, p.colorSpec, p.chromaSubSamp, p.memLayout, p.dataKind, p.swizzle,
                                           p.packing0, p.packing1, p.packing2, p.packing3, p.alphaType, exChannelInfo));
    }
    else
    {
        ASSERT_EQ(NVCV_SUCCESS,
                  nvcvMakeColorImageFormat(&fmt, p.colorModel, p.colorSpec, p.memLayout, p.dataKind, p.swizzle,
                                           p.packing0, p.packing1, p.packing2, p.packing3, p.alphaType, exChannelInfo));
    }

    EXPECT_EQ(p.imgFormat, fmt);
}

TEST(ImageFormatTests, make_image_format_fourth_plane_128bpp_fails)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NV12;

    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X8, NVCV_PACKING_X128, NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_NV12, fmt) << "Must not have changed output";
}

TEST(ImageFormatTests, get_plane_packing_of_image_format_none)
{
    for (int p = 0; p < 4; ++p)
    {
        std::ostringstream ss;
        ss << "plane #" << p;
        SCOPED_TRACE(ss.str());

        NVCVPacking packing;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(NVCV_IMAGE_FORMAT_NONE, p, &packing));
        EXPECT_EQ(NVCV_PACKING_0, packing);
    }
}

TEST(ImageFormatTests, get_plane_bits_per_pixel_of_image_format_none)
{
    for (int p = 0; p < 4; ++p)
    {
        std::ostringstream ss;
        ss << "plane #" << p;
        SCOPED_TRACE(ss.str());

        int bpp;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(NVCV_IMAGE_FORMAT_NONE, p, &bpp));
        EXPECT_EQ(0, bpp);
    }
}

TEST(ImageFormatTests, get_data_type_of_image_format_none)
{
    NVCVDataKind dataKind;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetDataKind(NVCV_IMAGE_FORMAT_NONE, &dataKind));
    EXPECT_EQ(NVCV_DATA_KIND_UNSIGNED, dataKind);
}

TEST(ImageFormatTests, set_valid_data_type_of_image_format_none)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NONE;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetDataKind(&fmt, NVCV_DATA_KIND_SIGNED));
}

TEST(ImageFormatTests, get_swizzle_of_image_format_none)
{
    NVCVSwizzle swizzle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetSwizzle(NVCV_IMAGE_FORMAT_NONE, &swizzle));
    EXPECT_EQ(NVCV_SWIZZLE_0000, swizzle);
}

TEST(ImageFormatTests, get_color_model_image_format_none)
{
    NVCVColorModel colorModel;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorModel(NVCV_IMAGE_FORMAT_NONE, &colorModel));
    EXPECT_EQ(NVCV_COLOR_MODEL_UNDEFINED, colorModel);
}

TEST(ImageFormatTests, get_alpha_type_image_format_none)
{
    NVCVAlphaType alphaType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetAlphaType(NVCV_IMAGE_FORMAT_NONE, &alphaType));
    EXPECT_EQ(NVCV_ALPHA_ASSOCIATED, alphaType);
}

TEST(ImageFormatTests, get_extra_channel_info_image_format_none)
{
    NVCVExtraChannelInfo exChannelInfo;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetExtraChannelInfo(NVCV_IMAGE_FORMAT_NONE, &exChannelInfo));
    EXPECT_EQ(0, exChannelInfo.numChannels);
    EXPECT_EQ(0, exChannelInfo.bitsPerPixel);
    EXPECT_EQ(NVCV_DATA_KIND_UNSPECIFIED, exChannelInfo.datakind);
    EXPECT_EQ(NVCV_EXTRA_CHANNEL_U, exChannelInfo.channelType);
}

TEST(ImageFormatTests, set_extra_channel_info_image_format_none)
{
    NVCVExtraChannelInfo exChannelInfo = {2, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_POS3D};
    NVCVImageFormat      fmt           = NVCV_IMAGE_FORMAT_NONE;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetExtraChannelInfo(&fmt, &exChannelInfo));
}

TEST(ImageFormatTests, set_extra_channel_info_max_min_bounds)
{
    NVCVExtraChannelInfo exChannelInfo = {8, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_POS3D};
    NVCVImageFormat      fmt           = NVCV_IMAGE_FORMAT_BGRf32;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetExtraChannelInfo(&fmt, &exChannelInfo));

    exChannelInfo.numChannels = -1;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetExtraChannelInfo(&fmt, &exChannelInfo));
}

TEST(ImageFormatTests, set_extra_channel_info_planar_image_format)
{
    NVCVImageFormat      fmt           = NVCV_IMAGE_FORMAT_BGRf32p;
    NVCVExtraChannelInfo exChannelInfo = {2, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_POS3D};
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetExtraChannelInfo(&fmt, &exChannelInfo));
}

TEST(ImageFormatTests, set_extra_channel_info_256bpp_fails)
{
    NVCVImageFormat      fmt           = NVCV_IMAGE_FORMAT_BGRf32;
    NVCVExtraChannelInfo exChannelInfo = {2, 256, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_POS3D};
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetExtraChannelInfo(&fmt, &exChannelInfo));
}

TEST(ImageFormatTests, get_mem_layout_of_image_format_none)
{
    NVCVMemLayout memLayout;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetMemLayout(NVCV_IMAGE_FORMAT_NONE, &memLayout));
    EXPECT_EQ(NVCV_MEM_LAYOUT_PITCH_LINEAR, memLayout);
}

TEST(ImageFormatTests, set_valid_mem_layout_of_image_format_none)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NONE;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetMemLayout(&fmt, NVCV_MEM_LAYOUT_BL));
}

TEST(ImageFormatTests, set_valid_color_spec_of_image_format_none)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NONE;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_BT601));
}

TEST(ImageFormatTests, get_color_spec_of_image_format_none)
{
    NVCVColorSpec cspec;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorSpec(NVCV_IMAGE_FORMAT_NONE, &cspec));
    EXPECT_EQ(NVCV_COLOR_SPEC_UNDEFINED, cspec);
}

TEST(ImageFormatTests, set_valid_raw_pattern_of_image_format_none)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NONE;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetRawPattern(&fmt, NVCV_RAW_BAYER_BGGR));
}

TEST(ImageFormatTests, get_raw_pattern_of_image_format_none)
{
    NVCVRawPattern raw;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetRawPattern(NVCV_IMAGE_FORMAT_NONE, &raw));
}

TEST(ImageFormatTests, get_chroma_subsampling_of_image_format_none)
{
    NVCVChromaSubsampling css;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetChromaSubsampling(NVCV_IMAGE_FORMAT_NONE, &css));
    EXPECT_EQ(NVCV_CSS_NONE, css);
}

TEST(ImageFormatTests, set_valid_chroma_subsampling_of_image_format_none)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_NONE;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetChromaSubsampling(&fmt, NVCV_CSS_420));
}

TEST(ImageFormatTests, get_plane_packing_out_of_bounds)
{
    NVCVPacking packing = NVCV_PACKING_X8_Y8__X8_Z8;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(NVCV_IMAGE_FORMAT_NV12, 342, &packing));
    EXPECT_EQ(NVCV_PACKING_0, packing);
}

TEST(ImageFormatTests, get_plane_bits_per_pixel_out_of_bounds)
{
    int32_t bpp = 123;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(NVCV_IMAGE_FORMAT_NV12, 342, &bpp));
    EXPECT_EQ(0, bpp);
}

TEST_P(ImageFormatTests, check_plane_packing)
{
    const Params &p = GetParam();

    NVCVPacking packing;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(p.imgFormat, 0, &packing));
    EXPECT_EQ(p.packing0, packing);

    EXPECT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(p.imgFormat, 1, &packing));
    EXPECT_EQ(p.packing1, packing);

    EXPECT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(p.imgFormat, 2, &packing));
    EXPECT_EQ(p.packing2, packing);

    EXPECT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(p.imgFormat, 3, &packing));
    EXPECT_EQ(p.packing3, packing);
}

TEST_P(ImageFormatTests, check_plane_bpp)
{
    const Params &p = GetParam();

    int32_t bpp;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(p.imgFormat, 0, &bpp));
    EXPECT_EQ(p.planes[0].bpp, bpp);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(p.imgFormat, 1, &bpp));
    EXPECT_EQ(p.planes[1].bpp, bpp);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(p.imgFormat, 2, &bpp));
    EXPECT_EQ(p.planes[2].bpp, bpp);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(p.imgFormat, 3, &bpp));
    EXPECT_EQ(p.planes[3].bpp, bpp);
}

TEST_P(ImageFormatTests, check_data_type)
{
    const Params &p = GetParam();

    NVCVDataKind dataKind;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetDataKind(p.imgFormat, &dataKind));
    EXPECT_EQ(p.dataKind, dataKind);
}

TEST_P(ImageFormatTests, check_swizzle)
{
    const Params &p = GetParam();

    NVCVSwizzle swizzle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetSwizzle(p.imgFormat, &swizzle));
    EXPECT_EQ(p.swizzle, swizzle);
}

TEST(ImageFormatTests, check_alpha_type)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_RGBA8_UNASSOCIATED_ALPHA;
    NVCVAlphaType   alphaType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetAlphaType(fmt, &alphaType));
    EXPECT_EQ(NVCV_ALPHA_UNASSOCIATED, alphaType);

    fmt = NVCV_IMAGE_FORMAT_RGBA8;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetAlphaType(fmt, &alphaType));
    EXPECT_EQ(NVCV_ALPHA_ASSOCIATED, alphaType);
}

TEST_P(ImageFormatTests, check_extra_channel_info)
{
    const Params        &p = GetParam();
    NVCVExtraChannelInfo exChannelInfo;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetExtraChannelInfo(p.imgFormat, &exChannelInfo));

    EXPECT_EQ(p.exChannelInfo.numChannels, exChannelInfo.numChannels);
    EXPECT_EQ(p.exChannelInfo.bitsPerPixel, exChannelInfo.bitsPerPixel);
    EXPECT_EQ(p.exChannelInfo.datakind, exChannelInfo.datakind);
    EXPECT_EQ(p.exChannelInfo.channelType, exChannelInfo.channelType);
}

TEST_P(ImageFormatTests, check_plane_swizzle)
{
    const Params &p = GetParam();

    NVCVSwizzle swizzle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 0, &swizzle));
    EXPECT_EQ(p.planes[0].swizzle, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 1, &swizzle));
    EXPECT_EQ(p.planes[1].swizzle, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 2, &swizzle));
    EXPECT_EQ(p.planes[2].swizzle, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 3, &swizzle));
    EXPECT_EQ(p.planes[3].swizzle, swizzle);
}

TEST_P(ImageFormatTests, check_memlayout)
{
    const Params &p = GetParam();

    NVCVMemLayout layout;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetMemLayout(p.imgFormat, &layout));
    EXPECT_EQ(p.memLayout, layout);
}

TEST_P(ImageFormatTests, check_colorspec)
{
    const Params &p = GetParam();

    NVCVColorSpec cspec;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorSpec(p.imgFormat, &cspec));
    EXPECT_EQ(p.colorSpec, cspec);
}

TEST_P(ImageFormatTests, check_colormodel)
{
    const Params &p = GetParam();

    NVCVColorModel cmodel;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorModel(p.imgFormat, &cmodel));
    EXPECT_EQ(p.colorModel, cmodel);
}

TEST_P(ImageFormatTests, check_chroma_subsampling)
{
    const Params &p = GetParam();

    NVCVChromaSubsampling css;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetChromaSubsampling(p.imgFormat, &css));
    EXPECT_EQ(p.chromaSubSamp, css);
}

TEST_P(ImageFormatTests, check_plane_channel_count)
{
    const Params &p = GetParam();

    int cnt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(p.imgFormat, 0, &cnt));
    EXPECT_EQ(p.planes[0].channels, cnt);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(p.imgFormat, 1, &cnt));
    EXPECT_EQ(p.planes[1].channels, cnt);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(p.imgFormat, 2, &cnt));
    EXPECT_EQ(p.planes[2].channels, cnt);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(p.imgFormat, 3, &cnt));
    EXPECT_EQ(p.planes[3].channels, cnt);
}

TEST_P(ImageFormatTests, check_plane_count)
{
    const Params &p = GetParam();

    int cnt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetNumPlanes(p.imgFormat, &cnt));
    EXPECT_EQ(p.planeCount, cnt);
}

TEST_P(ImageFormatTests, check_channel_count)
{
    const Params &p = GetParam();

    int cnt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetNumChannels(p.imgFormat, &cnt));

    EXPECT_EQ(p.planes[0].channels + p.planes[1].channels + p.planes[2].channels + p.planes[3].channels, cnt);
}

TEST_P(ImageFormatTests, get_bits_per_channel)
{
    const Params &p = GetParam();

    int bits[4];
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetBitsPerChannel(p.imgFormat, bits));

    EXPECT_EQ(p.bitsPerChannel[0], bits[0]);
    EXPECT_EQ(p.bitsPerChannel[1], bits[1]);
    EXPECT_EQ(p.bitsPerChannel[2], bits[2]);
    EXPECT_EQ(p.bitsPerChannel[3], bits[3]);
}

TEST_P(ImageFormatTests, check_plane_pixel_type)
{
    const Params &p = GetParam();

    NVCVDataType pix;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(p.imgFormat, 0, &pix));
    EXPECT_EQ(p.planes[0].pixFormat, pix);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(p.imgFormat, 1, &pix));
    EXPECT_EQ(p.planes[1].pixFormat, pix);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(p.imgFormat, 2, &pix));
    EXPECT_EQ(p.planes[2].pixFormat, pix);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneDataType(p.imgFormat, 3, &pix));
    EXPECT_EQ(p.planes[3].pixFormat, pix);
}

TEST(ImageFormatTests, invalid_plane_swizzle)
{
    // purposedly wrong fmt (more packing channels than swizzle channels)
    NVCVImageFormat fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(
        NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ1,
        NVCV_ALPHA_ASSOCIATED, 3, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8, NVCV_PACKING_X8);

    NVCVSwizzle sw;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(fmt, 0, &sw));
    EXPECT_EQ(NVCV_SWIZZLE_X001, sw);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(fmt, 1, &sw));
    EXPECT_EQ(NVCV_SWIZZLE_0XY1, sw);

    sw = NVCV_SWIZZLE_X001;
    ASSERT_EQ(NVCV_ERROR_INVALID_IMAGE_FORMAT, nvcvImageFormatGetPlaneSwizzle(fmt, 2, &sw));
    EXPECT_EQ(NVCV_SWIZZLE_X001, sw) << "Must not have changed output";
}

TEST(ImageFormatTests, packing_and_bits_per_pixel)
{
    for (int planes = 1; planes <= 4; ++planes)
    {
        for (int plane = 0; plane < planes; ++plane)
        {
            int begChannels = plane == 0 ? 1 : 0;
            int channelCount;
            switch (plane)
            {
            case 0:
                channelCount = 4;
                break;
            case 1:
                channelCount = 2;
                break;
            case 2:
                channelCount = 2;
                break;
            case 3:
                channelCount = 1;
                break;
            default:
                FAIL() << "Invalid plane";
            }

            for (int nchannels = begChannels; nchannels <= channelCount; ++nchannels)
            {
                int begBPP = plane == 0 ? 1 : 8;
                int maxBPP;
                if (nchannels == 0)
                {
                    maxBPP = 0;
                }
                else if (plane == 0)
                {
                    maxBPP = 256;
                }
                else if (plane <= 2)
                {
                    maxBPP = 128;
                }
                else
                {
                    assert(plane == 3);
                    // 4th plane can have at most 64 bits as it doesn't have channel count nor
                    // pack.
                    maxBPP = 64;
                }

                for (int bpp = begBPP; bpp <= maxBPP;
                     bpp <= 8
                         ? (bpp *= 2)
                         : (bpp < 32 ? (bpp += 8) : (bpp < 64 ? (bpp += 16) : (bpp < 128 ? (bpp += 32) : (bpp += 64)))))
                {
                    int packCount;
                    if (bpp <= 4)
                    {
                        packCount = 1;
                    }
                    else if (bpp <= 8)
                    {
                        if (plane == 3)
                        {
                            // 4th plane doesn't have pack code...
                            packCount = 1;
                        }
                        else
                        {
                            packCount = 3;
                        }
                    }
                    else
                    {
                        switch (plane)
                        {
                        case 0:
                        case 1:
                        case 2:
                            packCount = 8;
                            break;
                        case 3:
                            packCount = 1;
                            break;
                        default:
                            FAIL() << "Invalid plane";
                        }
                    }

                    for (int pack = 0; pack < packCount; (pack == 0 ? ++pack : pack <<= 1))
                    {
                        NVCVPacking packing = (NVCVPacking)(NVCV_DETAIL_BPP_NCH(bpp, nchannels) + pack);

                        uint64_t mask = UINT64_MAX;

                        std::optional<NVCVImageFormat> fmt;
                        switch (plane)
                        {
                        case 0:
                            switch (planes)
                            {
                            case 1:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, packing, 0, 0,
                                                                   0);
                                break;
                            case 2:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, packing, mask,
                                                                   0, 0);
                                break;
                            case 3:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, packing, mask,
                                                                   mask, 0);
                                break;
                            case 4:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, packing, mask,
                                                                   mask, mask);
                                break;
                            }
                            break;

                        case 1:
                            switch (planes)
                            {
                            case 2:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, packing,
                                                                   0, 0);
                                break;
                            case 3:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, packing,
                                                                   mask, 0);
                                break;
                            case 4:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, packing,
                                                                   mask, mask);
                                break;
                            }
                            break;
                        case 2:
                            switch (planes)
                            {
                            case 3:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, mask,
                                                                   packing, 0);
                                break;
                            case 4:
                                fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, mask,
                                                                   packing, mask);
                                break;
                            }
                            break;
                        case 3:
                            fmt = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, mask, mask, 4, mask, mask, mask,
                                                               packing);
                            break;
                        }

                        NVCV_ASSERT(fmt);

                        NVCVPacking testPacking;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(*fmt, plane, &testPacking));
                        EXPECT_EQ(packing, testPacking);

                        int testBPP;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneBitsPerPixel(*fmt, plane, &testBPP));
                        EXPECT_EQ(bpp, testBPP);

                        // these represent 4 channels, but comprise 2 pixels, 3 different channels, not 4.
                        if (packing == NVCV_PACKING_X8_Y8__X8_Z8 || packing == NVCV_PACKING_Y8_X8__Z8_X8)
                        {
                            nchannels -= 1;
                        }

                        int testNChannels;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(*fmt, plane, &testNChannels));
                        EXPECT_EQ(nchannels, testNChannels);

                        int testNumPlanes;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetNumPlanes(*fmt, &testNumPlanes));
                        EXPECT_EQ(planes, testNumPlanes);

                        for (int p = planes; p < 4; ++p)
                        {
                            ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneNumChannels(*fmt, p, &testNChannels));
                            EXPECT_EQ(0, testNChannels);
                        }
                        if (this->HasFailure())
                        {
                            FAIL() << "#planes=" << planes << ", plane=" << plane << ", #channels=" << nchannels
                                   << ", bpp=" << bpp << ", pack=" << pack;
                        }
                    }
                }
            }
        }
    }
}

#if !NVCV_SANITIZED
TEST(ImageFormatTests, get_swizzle)
{
    std::vector<NVCVSwizzle> swizzleList
        = {NVCV_SWIZZLE_0000, NVCV_SWIZZLE_YYYX, NVCV_SWIZZLE_0YX0, NVCV_SWIZZLE_X00Y, NVCV_SWIZZLE_Y00X,
           NVCV_SWIZZLE_X001, NVCV_SWIZZLE_XY01, NVCV_SWIZZLE_0XZ0, NVCV_SWIZZLE_0ZX0, NVCV_SWIZZLE_XZY0,
           NVCV_SWIZZLE_YZX1, NVCV_SWIZZLE_ZYW1, NVCV_SWIZZLE_0YX1, NVCV_SWIZZLE_XYXZ, NVCV_SWIZZLE_YXZX,
           NVCV_SWIZZLE_XZ00, NVCV_SWIZZLE_WYXZ, NVCV_SWIZZLE_YX00, NVCV_SWIZZLE_YX01, NVCV_SWIZZLE_00YX,
           NVCV_SWIZZLE_00XY, NVCV_SWIZZLE_0XY1};
    uint64_t mask = UINT64_MAX;
    for (auto swizzle : swizzleList)
    {
        NVCVImageFormat fmt
            = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, mask, swizzle, mask, 4, mask, mask, mask, mask);

        NVCVSwizzle testSwizzle;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetSwizzle(fmt, &testSwizzle));
        ASSERT_EQ(swizzle, testSwizzle);
    }
}

TEST(ImageFormatTests, get_data_type)
{
    for (int dataKind = 0; dataKind < 8; ++dataKind)
    {
        uint64_t mask = UINT64_MAX;

        NVCVImageFormat fmt
            = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, mask, dataKind, mask, mask, 4, mask, mask, mask, mask);

        NVCVDataKind testDataKind;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetDataKind(fmt, &testDataKind));
        ASSERT_EQ(dataKind, testDataKind);
    }
}

TEST(ImageFormatTests, get_raw_pattern)
{
    for (int raw_pattern = 0; raw_pattern < (1 << 6); raw_pattern ? raw_pattern <<= 1 : ++raw_pattern)
    {
        uint64_t mask = UINT64_MAX;

        NVCVImageFormat fmt
            = NVCV_MAKE_RAW_IMAGE_FORMAT(raw_pattern, mask, mask, mask, mask, 4, mask, mask, mask, mask);

        NVCVRawPattern testRawPattern;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetRawPattern(fmt, &testRawPattern));
        ASSERT_EQ(raw_pattern, testRawPattern);
    }
}

TEST(ImageFormatTests, get_mem_layout)
{
    for (int memLayout = 0; memLayout < 8; ++memLayout)
    {
        uint64_t mask = UINT64_MAX;

        NVCVImageFormat fmt
            = NVCV_MAKE_COLOR_IMAGE_FORMAT(mask, mask, memLayout, mask, mask, mask, 4, mask, mask, mask, mask);

        NVCVMemLayout testMemLayout;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetMemLayout(fmt, &testMemLayout));
        ASSERT_EQ(memLayout, testMemLayout);
    }
}

TEST(ImageFormatTests, get_color_model)
{
    for (int model = 0; model <= 7 + 2; ++model)
    {
        uint64_t mask = UINT64_MAX;

        NVCVImageFormat fmt
            = NVCV_MAKE_COLOR_IMAGE_FORMAT(model, mask, mask, mask, mask, mask, 4, mask, mask, mask, mask);

        NVCVColorModel testColorModel;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorModel(fmt, &testColorModel));
        ASSERT_EQ(model, testColorModel);
    }

    for (int model = 0; model < 1 << 6; model ? model <<= 1 : ++model)
    {
        uint64_t mask = UINT64_MAX;

        NVCVImageFormat fmt
            = NVCV_MAKE_COLOR_IMAGE_FORMAT(model + 7 + 2, mask, mask, mask, mask, mask, 4, mask, mask, mask, mask);

        NVCVColorModel testColorModel;
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorModel(fmt, &testColorModel));
        ASSERT_EQ(model + 7 + 2, testColorModel);
    }
}
#endif

TEST(ImageFormatTests, make_image_format_null_packing_returns_invalid)
{
    NVCVImageFormat imgFormat = NVCV_IMAGE_FORMAT_NV12;

    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_0000, NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_NV12, imgFormat) << "Output must not have changed";

    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_0, NVCV_PACKING_X8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_X8, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, set_datatype)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&gold, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetDataKind(&imgFormat, NVCV_DATA_KIND_FLOAT));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, set_alphatype)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8_Y8_Z8_W8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&gold, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8_Y8_Z8_W8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_UNASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetAlphaType(&imgFormat, NVCV_ALPHA_UNASSOCIATED));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, set_extra_channel_info)
{
    NVCVImageFormat      imgFormat;
    NVCVExtraChannelInfo exChannelInfo{2, 8, NVCV_DATA_KIND_UNSIGNED, NVCV_EXTRA_CHANNEL_U};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED,
                                                     NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW,
                                                     NVCV_PACKING_X8_Y8_Z8_W8, NVCV_PACKING_0, NVCV_PACKING_0,
                                                     NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    NVCVImageFormat      gold;
    NVCVExtraChannelInfo newExChannelInfo{3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_D};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&gold, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED,
                                                     NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW,
                                                     NVCV_PACKING_X8_Y8_Z8_W8, NVCV_PACKING_0, NVCV_PACKING_0,
                                                     NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &newExChannelInfo));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetExtraChannelInfo(&imgFormat, &newExChannelInfo));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, set_alphatype_imageformat_none)
{
    NVCVImageFormat imgFormat = NVCV_IMAGE_FORMAT_NONE;
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetAlphaType(&imgFormat, NVCV_ALPHA_UNASSOCIATED));
}

TEST(ImageFormatTests, set_memlayout)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&gold, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_BL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetMemLayout(&imgFormat, NVCV_MEM_LAYOUT_BL));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, set_raw_pattern)
{
    for (int raw_pattern = 0; raw_pattern < (1 << 7); raw_pattern ? raw_pattern <<= 1 : ++raw_pattern)
    {
        NVCVImageFormat imgFormat;
        ASSERT_EQ(
            NVCV_SUCCESS,
            nvcvMakeRawImageFormat(
                &imgFormat, raw_pattern == NVCV_RAW_BAYER_CRCC ? NVCV_RAW_BAYER_GRBG : (NVCVRawPattern)raw_pattern,
                NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0,
                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

        NVCVImageFormat gold;
        ASSERT_EQ(NVCV_SUCCESS,
                  nvcvMakeRawImageFormat(&gold, (NVCVRawPattern)raw_pattern, NVCV_MEM_LAYOUT_PL,
                                         NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                         NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetRawPattern(&imgFormat, (NVCVRawPattern)raw_pattern));
        EXPECT_EQ(gold, imgFormat) << (NVCVRawPattern)raw_pattern;
    }
}

TEST(ImageFormatTests, set_color_standard_ycbcr)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&imgFormat, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&gold, NVCV_COLOR_SPEC_BT601, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&imgFormat, NVCV_COLOR_SPEC_BT601));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, set_color_standard_rgb_with_ycbcr_colorspec)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVColorSpec cspec = NVCV_COLOR_SPEC_BT601;
    ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetYCbCrEncoding(&cspec, NVCV_YCbCr_ENC_UNDEFINED));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&gold, NVCV_COLOR_MODEL_RGB, cspec, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                       NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&imgFormat, NVCV_COLOR_SPEC_BT601));

    EXPECT_EQ(gold, imgFormat) << "Must have given colorspec, but with YCbCr encoding set to undefined";
}

TEST(ImageFormatTests, set_chroma_subsampling)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&imgFormat, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_422R, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&gold, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetChromaSubsampling(&imgFormat, NVCV_CSS_420));
    EXPECT_EQ(gold, imgFormat);
}

TEST(ImageFormatTests, make_color_image_format_macro)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(imgFormat, NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED,
                                                      NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
                                                      NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_X8_Y8_Z8));
}

TEST(ImageFormatTests, make_color_extra_channel_image_format_macro)
{
    NVCVImageFormat      imgFormat;
    NVCVExtraChannelInfo exChannelInfo{3, 32, NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_D};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED,
                                                     NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
                                                     NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0, NVCV_PACKING_0,
                                                     NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    EXPECT_EQ(imgFormat, NVCV_MAKE_COLOR_IMAGE_EXTRA_CHANNELS_FORMAT(
                             NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                             NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_ALPHA_ASSOCIATED, 3, 32,
                             NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_D, 1, NVCV_PACKING_X8_Y8_Z8));
}

TEST(ImageFormatTests, make_noncolor_image_format_macro)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeNonColorImageFormat(&imgFormat, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                                        NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0,
                                                        NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(imgFormat, NVCV_MAKE_NONCOLOR_IMAGE_FORMAT(NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
                                                         NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_X8_Y8_Z8));
}

TEST(ImageFormatTests, make_noncolor_extra_channel_image_format_macro)
{
    NVCVImageFormat      imgFormat;
    NVCVExtraChannelInfo exChannelInfo{7, 64, NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_U};
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeNonColorImageFormat(&imgFormat, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
                                          NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                          NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    EXPECT_EQ(imgFormat, NVCV_MAKE_NONCOLOR_IMAGE_EXTRA_CHANNELS_FORMAT(
                             NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_ALPHA_ASSOCIATED, 7,
                             64, NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_U, 1, NVCV_PACKING_X8_Y8_Z8));
}

TEST(ImageFormatTests, make_raw_image_format_macro)
{
    NVCVImageFormat imgFormat;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&imgFormat, NVCV_RAW_BAYER_CRBC, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(imgFormat, NVCV_MAKE_RAW_IMAGE_FORMAT(NVCV_RAW_BAYER_CRBC, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                                    NVCV_SWIZZLE_X000, NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_X8));
}

TEST(ImageFormatTests, make_raw_extra_channel_image_format_macro)
{
    NVCVImageFormat      imgFormat;
    NVCVExtraChannelInfo exChannelInfo{7, 64, NVCV_DATA_KIND_FLOAT, NVCV_EXTRA_CHANNEL_U};
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&imgFormat, NVCV_RAW_BAYER_CRBC, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    EXPECT_EQ(imgFormat, NVCV_MAKE_RAW_IMAGE_EXTRA_CHANNELS_FORMAT(NVCV_RAW_BAYER_CRBC, NVCV_MEM_LAYOUT_PL,
                                                                   NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000,
                                                                   NVCV_ALPHA_ASSOCIATED, 7, 64, NVCV_DATA_KIND_FLOAT,
                                                                   NVCV_EXTRA_CHANNEL_U, 1, NVCV_PACKING_X8));
}

namespace {

struct ParamsPlaneSwizzle
{
    ParamsPlaneSwizzle(NVCVSwizzle swizzle_, NVCVPacking packing0_, NVCVPacking packing1_, NVCVPacking packing2_,
                       NVCVPacking packing3_, NVCVSwizzle planeSwizzle0_, NVCVSwizzle planeSwizzle1_,
                       NVCVSwizzle planeSwizzle2_, NVCVSwizzle planeSwizzle3_)
        : planeSwizzle0(planeSwizzle0_)
        , planeSwizzle1(planeSwizzle1_)
        , planeSwizzle2(planeSwizzle2_)
        , planeSwizzle3(planeSwizzle3_)
    {
        EXPECT_EQ(NVCV_SUCCESS,
                  nvcvMakeColorImageFormat(&imgFormat, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED,
                                           NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, swizzle_, packing0_, packing1_,
                                           packing2_, packing3_, NVCV_ALPHA_ASSOCIATED, 0));
    }

    NVCVImageFormat imgFormat;
    NVCVSwizzle     planeSwizzle0, planeSwizzle1, planeSwizzle2, planeSwizzle3;
};

std::ostream &operator<<(std::ostream &out, const ParamsPlaneSwizzle &p)
{
    return out << "imgFormat=" << nvcvImageFormatGetName(p.imgFormat) << ",planeSwizzle=[" << p.planeSwizzle0 << ","
               << p.planeSwizzle1 << "," << p.planeSwizzle2 << "," << p.planeSwizzle3 << "]";
}

#define DEF_PLANE_SWIZZLE(SW, P0, P1, P2, P3, SW0, SW1, SW2, SW3)                                      \
    ParamsPlaneSwizzle                                                                                 \
    {                                                                                                  \
        NVCV_SWIZZLE_##SW, NVCV_PACKING_##P0, NVCV_PACKING_##P1, NVCV_PACKING_##P2, NVCV_PACKING_##P3, \
            NVCV_SWIZZLE_##SW0, NVCV_SWIZZLE_##SW1, NVCV_SWIZZLE_##SW2, NVCV_SWIZZLE_##SW3             \
    }

} // namespace

class ImageFormatPlaneSwizzleTests : public t::TestWithParam<ParamsPlaneSwizzle>
{
};

INSTANTIATE_TEST_SUITE_P(_, ImageFormatPlaneSwizzleTests,
                         t::Values(DEF_PLANE_SWIZZLE(XYZ1, X8, X8, X8, 0, X001, 0X01, 00X1, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, X8_Y8, X8, 0, 0, XY01, 00X1, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, X8, X8_Y8, 0, 0, X001, 0XY1, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, X8, X8, X8, 0, X001, 0X01, 00X1, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, X8, X8_Y8, 0, 0, X001, 0XY1, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX1, X8, X8, X8, 0, 00X1, 0X01, X001, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX1, X8_Y8, X8, 0, 0, 0YX1, X001, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX1, X8, X8_Y8, 0, 0, 00X1, YX01, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX1, X8, X8, X8, 0, 00X1, 0X01, X001, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX1, X8, X8_Y8, 0, 0, 00X1, YX01, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, X8_Y8__X8_Z8, 0, 0, 0, XYZ1, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XYZ1, Y8_X8__Z8_X8, 0, 0, 0, XYZ1, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XZY1, X8_Y8__X8_Z8, 0, 0, 0, XZY1, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XXX1, X8, 0, 0, 0, XXX1, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XXXY, X8_Y8, 0, 0, 0, XXXY, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(ZYX0, X8_Y8_Z8, 0, 0, 0, ZYX0, 0000, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(XY01, X8, X8, 0, 0, X001, 0X01, 0000, 0000),
                                   DEF_PLANE_SWIZZLE(YX01, X8, X8, 0, 0, 0X01, X001, 0000, 0000)));

TEST_P(ImageFormatPlaneSwizzleTests, check_plane_pixel_type)
{
    const ParamsPlaneSwizzle &p = GetParam();

    NVCVSwizzle swizzle;

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 0, &swizzle));
    EXPECT_EQ(p.planeSwizzle0, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 1, &swizzle));
    EXPECT_EQ(p.planeSwizzle1, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 2, &swizzle));
    EXPECT_EQ(p.planeSwizzle2, swizzle);

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneSwizzle(p.imgFormat, 3, &swizzle));
    EXPECT_EQ(p.planeSwizzle3, swizzle);
}

TEST_P(ImageFormatPlaneSwizzleTests, get_plane_format)
{
    const ParamsPlaneSwizzle &params = GetParam();

    NVCVColorModel cmodel;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorModel(params.imgFormat, &cmodel));

    NVCVColorSpec cspec;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorSpec(params.imgFormat, &cspec));

    NVCVDataKind dataKind;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetDataKind(params.imgFormat, &dataKind));

    NVCVMemLayout memLayout;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetMemLayout(params.imgFormat, &memLayout));

    int numPlanes;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetNumPlanes(params.imgFormat, &numPlanes));

    // plane 0
    NVCVPacking planePacking;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(params.imgFormat, 0, &planePacking));

    NVCVImageFormat goldPlaneFormat;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&goldPlaneFormat, cmodel, cspec, memLayout, dataKind,
                                                     params.planeSwizzle0, planePacking, NVCV_PACKING_0, NVCV_PACKING_0,
                                                     NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat testPlaneFormat;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneFormat(params.imgFormat, 0, &testPlaneFormat));
    EXPECT_EQ(goldPlaneFormat, testPlaneFormat);

    if (numPlanes > 1)
    {
        // plane 1
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(params.imgFormat, 1, &planePacking));

        ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&goldPlaneFormat, cmodel, cspec, memLayout, dataKind,
                                                         params.planeSwizzle1, planePacking, NVCV_PACKING_0,
                                                         NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneFormat(params.imgFormat, 1, &testPlaneFormat));
        EXPECT_EQ(goldPlaneFormat, testPlaneFormat);
    }

    if (numPlanes > 2)
    {
        // plane 2
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(params.imgFormat, 2, &planePacking));

        ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&goldPlaneFormat, cmodel, cspec, memLayout, dataKind,
                                                         params.planeSwizzle2, planePacking, NVCV_PACKING_0,
                                                         NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneFormat(params.imgFormat, 2, &testPlaneFormat));
        EXPECT_EQ(goldPlaneFormat, testPlaneFormat);
    }

    if (numPlanes > 3)
    {
        // plane 3
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePacking(params.imgFormat, 3, &planePacking));

        ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&goldPlaneFormat, cmodel, cspec, memLayout, dataKind,
                                                         params.planeSwizzle3, planePacking, NVCV_PACKING_0,
                                                         NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneFormat(params.imgFormat, 3, &testPlaneFormat));
        EXPECT_EQ(goldPlaneFormat, testPlaneFormat);
    }
}

TEST_P(ImageFormatPlaneSwizzleTests, make_imageformat_from_planes)
{
    const ParamsPlaneSwizzle &p = GetParam();

    NVCVImageFormat planes[4];
    for (int i = 0; i < 4; ++i)
    {
        ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlaneFormat(p.imgFormat, i, &planes[i]));
    }

    NVCVImageFormat test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeImageFormatFromPlanes(&test, planes[0], planes[1], planes[2], planes[3]));
    EXPECT_EQ(p.imgFormat, test);
}

TEST(ImageFormatTests, invalid_make_imageformat_from_planes)
{
    NVCVImageFormat fmt;
    // Pointer to output image format cannot be NULL
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(nullptr, NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE,
                                            NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));
    // At least one plane must be specified
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE,
                                            NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));
    // all plane types must have just one plane.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_RGB8p, NVCV_IMAGE_FORMAT_NONE,
                                            NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));
    // total number of channels must be at most 4.
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_2S16, NVCV_IMAGE_FORMAT_2S16, NVCV_IMAGE_FORMAT_U8,
                                            NVCV_IMAGE_FORMAT_U8));
    // color spec, mem layout and data type of all planes must be the same
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_Y8_BL, NVCV_IMAGE_FORMAT_Y8_ER,
                                            NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_U8_BL, NVCV_IMAGE_FORMAT_NONE,
                                            NVCV_IMAGE_FORMAT_NONE));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_S8, NVCV_IMAGE_FORMAT_NONE,
                                            NVCV_IMAGE_FORMAT_NONE));

    NVCVImageFormat fmt_raw1;
    NVCVImageFormat fmt_raw2;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&fmt_raw1, NVCV_RAW_BAYER_RGGB, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, 0));
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&fmt_raw2, NVCV_RAW_BAYER_BGGR, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromPlanes(&fmt, fmt_raw1, fmt_raw2, NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));

    // Only one chroma-subsampling type must be specified
    NVCVImageFormat fmt_ycbcr1;
    NVCVImageFormat fmt_ycbcr2;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt_ycbcr1, NVCV_COLOR_SPEC_BT601, NVCV_CSS_422, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt_ycbcr2, NVCV_COLOR_SPEC_BT601, NVCV_CSS_422R, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcvMakeImageFormatFromPlanes(&fmt, fmt_ycbcr2, fmt_ycbcr1, NVCV_IMAGE_FORMAT_NONE, NVCV_IMAGE_FORMAT_NONE));
}

struct SwizzlePacking
{
    NVCVSwizzle swizzle;
    NVCVPacking packing[4] = {};

    friend std::ostream &operator<<(std::ostream &out, const SwizzlePacking &sp)
    {
        // clang-format off
        return out << sp.swizzle
                   << "," << sp.packing[0]
                   << "," << sp.packing[1]
                   << "," << sp.packing[2]
                   << "," << sp.packing[3];
        // clang-format on
    }
};

class ImageFormatNegativeSwizzlePackingTests : public t::TestWithParam<SwizzlePacking>
{
};

// clang-format off
static std::vector<SwizzlePacking> g_InvalidSwizzlePacking =
{
    { NVCV_SWIZZLE_XY00, {NVCV_PACKING_X8} },
    { NVCV_SWIZZLE_X000, {NVCV_PACKING_X8, NVCV_PACKING_X8} },
    { NVCV_SWIZZLE_X000, {NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_X000, {NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XY00, {NVCV_PACKING_X8, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XY00, {NVCV_PACKING_X8, NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XYZ0, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XYZ0, {NVCV_PACKING_X16_Y16, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XYZ0, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XYZ0, {NVCV_PACKING_X16_Y16, NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XYZW, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XYZW, {NVCV_PACKING_X16_Y16, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XYZW, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X8, NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XYZW, {NVCV_PACKING_X16_Y16, NVCV_PACKING_X8, NVCV_PACKING_X16_Y16} },
    { NVCV_SWIZZLE_XYZ1, {NVCV_PACKING_X8_Y8_Z8_W8} },
    { NVCV_SWIZZLE_XYZ1, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8} },
    { NVCV_SWIZZLE_XYZ1, {NVCV_PACKING_X8_Y8, NVCV_PACKING_X8, NVCV_PACKING_X16_Y16} },
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(_, ImageFormatNegativeSwizzlePackingTests, t::ValuesIn(g_InvalidSwizzlePacking));

TEST_P(ImageFormatNegativeSwizzlePackingTests, wrong_number_of_planes_for_swizzle)
{
    const SwizzlePacking &sp = GetParam();

    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, sp.swizzle, sp.packing[0], sp.packing[1], sp.packing[2],
                                       sp.packing[3], NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, make_yuv422_packed_yuyv)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ1, NVCV_PACKING_X8_Y8__X8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XYZ1, NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_X8_Y8__X8_Z8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, make_yuv422_packed_yuyv_extra_channels)
{
    NVCVImageFormat      fmt;
    NVCVExtraChannelInfo exChannelInfo{3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_U};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeYCbCrImageFormat(
                                &fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ1, NVCV_PACKING_X8_Y8__X8_Z8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_EXTRA_CHANNELS_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XYZ1, NVCV_ALPHA_ASSOCIATED, 3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_U, 1,
        NVCV_PACKING_X8_Y8__X8_Z8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, make_yuv422_packed_yvyu)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XZY1, NVCV_PACKING_X8_Y8__X8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XZY1, NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_X8_Y8__X8_Z8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, make_yuv422_packed_yvyu_extra_channels)
{
    NVCVImageFormat      fmt;
    NVCVExtraChannelInfo exChannelInfo{3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_U};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeYCbCrImageFormat(
                                &fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XZY1, NVCV_PACKING_X8_Y8__X8_Z8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_EXTRA_CHANNELS_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XZY1, NVCV_ALPHA_ASSOCIATED, 3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_U, 1,
        NVCV_PACKING_X8_Y8__X8_Z8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, make_yuv422_packed_uyvy)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ1, NVCV_PACKING_Y8_X8__Z8_X8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XYZ1, NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_Y8_X8__Z8_X8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, make_yuv422_packed_vyuy)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XZY1, NVCV_PACKING_Y8_X8__Z8_X8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat gold = NVCV_MAKE_YCbCr_IMAGE_FORMAT(
        NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_UNSIGNED,
        NVCV_SWIZZLE_XZY1, NVCV_ALPHA_ASSOCIATED, 1, NVCV_PACKING_Y8_X8__Z8_X8);

    EXPECT_EQ(gold, fmt);
}

TEST(ImageFormatTests, get_name_noncolor_predefined)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_U8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_U8));
}

TEST(ImageFormatTests, get_name_noncolor_non_predefined)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeNonColorImageFormat(&fmt, NVCV_MEM_LAYOUT_BLOCK4_LINEAR, NVCV_DATA_KIND_FLOAT,
                                                        NVCV_SWIZZLE_YZWX, NVCV_PACKING_X16, NVCV_PACKING_X32_Y32,
                                                        NVCV_PACKING_X1, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(UNDEFINED,BLOCK4_LINEAR,FLOAT,YZWX,ASSOCIATED,X16,X32_Y32,X1)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_predefined_NONE)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NONE", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NONE));
}

TEST(ImageFormatTests, get_name_predefined_U8_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_U8_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_U8_BL));
}

TEST(ImageFormatTests, get_name_predefined_S16_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_S16_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_S16_BL));
}

TEST(ImageFormatTests, get_name_predefined_S32)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_S32", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_S32));
}

TEST(ImageFormatTests, get_name_predefined_Y8_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y8_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y8_BL));
}

TEST(ImageFormatTests, get_name_predefined_Y8_ER)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y8_ER", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y8_ER));
}

TEST(ImageFormatTests, get_name_predefined_Y8_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y8_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y8_ER_BL));
}

TEST(ImageFormatTests, get_name_predefined_Y16_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y16_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y16_BL));
}

TEST(ImageFormatTests, get_name_predefined_Y16_ER)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y16_ER", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y16_ER));
}

TEST(ImageFormatTests, get_name_predefined_Y16_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_Y16_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_Y16_ER_BL));
}

TEST(ImageFormatTests, get_name_predefined_NV12_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV12_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV12_BL));
}

TEST(ImageFormatTests, get_name_predefined_NV12_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV12_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV12_ER_BL));
}

TEST(ImageFormatTests, get_name_predefined_NV24)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV24", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV24));
}

TEST(ImageFormatTests, get_name_predefined_NV24_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV24_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV24_BL));
}

TEST(ImageFormatTests, get_name_predefined_NV24_ER)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV24_ER", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV24_ER));
}

TEST(ImageFormatTests, get_name_predefined_NV24_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_NV24_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_NV24_ER_BL));
}

TEST(ImageFormatTests, get_name_predefined_2S16_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_2S16_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_2S16_BL));
}

TEST(ImageFormatTests, get_name_predefined_UYVY_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_UYVY_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_UYVY_BL));
}

TEST(ImageFormatTests, get_name_predefined_UYVY_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_UYVY_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_UYVY_ER_BL));
}

TEST(ImageFormatTests, get_name_predefined_YUYV_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_YUYV_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_YUYV_BL));
}

TEST(ImageFormatTests, get_name_predefined_YUYV_ER_BL)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_YUYV_ER_BL", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_YUYV_ER_BL));
}

TEST(ImageFormatTests, get_name_color_predefined)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_RGB8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_RGB8));
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_CMYK8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_CMYK8));
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_YCCK8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_YCCK8));
}

TEST(ImageFormatTests, get_name_color_predefined_extra_channels)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_RGB8_1U_U8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_RGB8_1U_U8));
}

TEST(ImageFormatTests, get_name_color_non_predefined)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT2020,
                                                     NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_FLOAT,
                                                     NVCV_SWIZZLE_YZWX, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8,
                                                     NVCV_PACKING_X32, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(RGB,BT2020,BLOCK16_LINEAR,FLOAT,YZWX,ASSOCIATED,X8,X8_Y8,X32)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_color_non_predefined_extra_channels)
{
    NVCVImageFormat      fmt;
    NVCVExtraChannelInfo exChannelInfo{3, 16, NVCV_DATA_KIND_SIGNED, NVCV_EXTRA_CHANNEL_D};
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(
                                &fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT2020, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8_Y8_Z8_W8, NVCV_PACKING_0,
                                NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, &exChannelInfo));

    EXPECT_STREQ("NVCVImageFormat(RGB,BT2020,BLOCK16_LINEAR,FLOAT,XYZW,ASSOCIATED,3,16,SIGNED,D,X8_Y8_Z8_W8)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_raw_non_predefined)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&fmt, NVCV_RAW_BAYER_CRCC, NVCV_MEM_LAYOUT_BLOCK8_LINEAR, NVCV_DATA_KIND_FLOAT,
                                     NVCV_SWIZZLE_YZWX, NVCV_PACKING_X16, NVCV_PACKING_X8_Y8, NVCV_PACKING_X1,
                                     NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(RAW,BAYER_CRCC,BLOCK8_LINEAR,FLOAT,YZWX,ASSOCIATED,X16,X8_Y8,X1)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_not_predefined_with_predefined_parameters)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_BT2020, NVCV_CSS_422R, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_YZWX, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8,
                                       NVCV_PACKING_X32, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(YCbCr,BT2020,4:2:2R,BLOCK16_LINEAR,FLOAT,YZWX,ASSOCIATED,X8,X8_Y8,X32)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_color_not_predefined)
{
    NVCVColorSpec cspec;

    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorSpec(&cspec, NVCV_COLOR_SPACE_DCIP3, NVCV_YCbCr_ENC_SMPTE240M, NVCV_COLOR_XFER_BT2020,
                                NVCV_COLOR_RANGE_LIMITED, NVCV_CHROMA_LOC_EVEN, NVCV_CHROMA_LOC_ODD));

    NVCVImageFormat fmt;
    NVCVSwizzle     swizzle;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeSwizzle(&swizzle, NVCV_CHANNEL_Z, NVCV_CHANNEL_Y, NVCV_CHANNEL_W, NVCV_CHANNEL_1));
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, cspec, NVCV_CSS_411R, NVCV_MEM_LAYOUT_BLOCK16_LINEAR, NVCV_DATA_KIND_FLOAT,
                                       swizzle, NVCV_PACKING_X8, NVCV_PACKING_X16_Y16, NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_ALPHA_ASSOCIATED, 0));

    // clang-format off
    EXPECT_STREQ(
        "NVCVImageFormat(YCbCr,NVCVColorSpec(SPACE_DCIP3,ENC_SMPTE240M,XFER_BT2020,RANGE_LIMITED,LOC_EVEN,LOC_ODD),4:1:1R,BLOCK16_LINEAR,FLOAT,ZYW1,ASSOCIATED,X8,X16_Y16)",
        nvcvImageFormatGetName(fmt));
    // clang-format on
}

TEST(ImageFormatTests, get_name_no_buffer_corruption)
{
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_RGBA8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_RGBA8));
    EXPECT_STREQ("NVCV_IMAGE_FORMAT_U8", nvcvImageFormatGetName(NVCV_IMAGE_FORMAT_U8));
}

TEST(ImageFormatTests, get_name_yuv422_packed)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_SMPTE240M, NVCV_CSS_422, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_XZY1, NVCV_PACKING_X8_Y8__X8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(YCbCr,SMPTE240M,4:2:2,BLOCK16_LINEAR,FLOAT,XZY1,ASSOCIATED,X8_Y8__X8_Z8)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_ycbcr_444_not_predefined)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_BT2020, NVCV_CSS_444, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_YZWX, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8,
                                       NVCV_PACKING_X32, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(YCbCr,BT2020,4:4:4,BLOCK16_LINEAR,FLOAT,YZWX,ASSOCIATED,X8,X8_Y8,X32)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, get_name_host_endian_packing)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_BT2020, NVCV_CSS_444, NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
                                       NVCV_DATA_KIND_FLOAT, NVCV_SWIZZLE_YZWX, NVCV_PACKING_X2Y10Z10W10,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_STREQ("NVCVImageFormat(YCbCr,BT2020,4:4:4,BLOCK16_LINEAR,FLOAT,YZWX,ASSOCIATED,X2Y10Z10W10)",
                 nvcvImageFormatGetName(fmt));
}

TEST(ImageFormatTests, none_image_format_must_be_0)
{
    EXPECT_EQ(0, (int)NVCV_IMAGE_FORMAT_NONE);
}

TEST(ImageFormatTests, set_colorspec_to_undefined_of_fmt_with_undefined_colorspec)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_U8;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_UNDEFINED));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_U8, fmt);
}

TEST(ImageFormatTests, set_rgb_fmt_to_undefined_colorspec)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT2020, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ1, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_UNDEFINED));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_RGB8, fmt);
}

TEST(ImageFormatTests, set_ycbcr_fmt_to_undefined_colorspec)
{
    NVCVImageFormat fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeYCbCrImageFormat(&fmt, NVCV_COLOR_SPEC_BT2020, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_BT601));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_NV12, fmt);
}

TEST(ImageFormatTests, set_raw_fmt_to_undefined_colorspec)
{
    NVCVImageFormat fmtGold;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeRawImageFormat(&fmtGold, NVCV_RAW_BAYER_BGGR, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, 0));

    NVCVImageFormat fmt = fmtGold;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_UNDEFINED));
    EXPECT_EQ(fmtGold, fmt);
}

TEST(ImageFormatTests, set_non_color_fmt_to_undefined_colorspec)
{
    NVCVImageFormat fmt = NVCV_IMAGE_FORMAT_U8;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&fmt, NVCV_COLOR_SPEC_UNDEFINED));
    EXPECT_EQ(NVCV_IMAGE_FORMAT_U8, fmt);
}

struct SetSwizzlePackingTestParams
{
    NVCVImageFormat input, output;
    NVCVSwizzle     swizzle;
    NVCVPacking     packing[4];
    const char     *msg;

    friend std::ostream &operator<<(std::ostream &out, const SetSwizzlePackingTestParams &p)
    {
        return out << "input=" << nvcvImageFormatGetName(p.input) << ", swizzle=" << p.swizzle << ", packing={"
                   << p.packing[0] << "," << p.packing[1] << "," << p.packing[2] << "," << p.packing[3] << "}"
                   << " -> " << p.msg;
    }
};

#define MAKE_COLOR_IMAGE_FORMAT_ABBREV1(model, cspec, layout, datatype, swizzle, alphatype, pack0)             \
    NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_##model, NVCV_COLOR_SPEC_##cspec, NVCV_MEM_LAYOUT_##layout,  \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 1, \
                                 NVCV_PACKING_##pack0)

#define MAKE_COLOR_IMAGE_FORMAT_ABBREV2(model, cspec, layout, datatype, swizzle, alphatype, pack0, pack1)      \
    NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_##model, NVCV_COLOR_SPEC_##cspec, NVCV_MEM_LAYOUT_##layout,  \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 2, \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1)

#define MAKE_COLOR_IMAGE_FORMAT_ABBREV3(model, cspec, layout, datatype, swizzle, alphatype, pack0, pack1, pack2) \
    NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_##model, NVCV_COLOR_SPEC_##cspec, NVCV_MEM_LAYOUT_##layout,    \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 3,   \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1, NVCV_PACKING_##pack2)

#define MAKE_COLOR_IMAGE_FORMAT_ABBREV4(model, cspec, layout, datatype, swizzle, alphatype, pack0, pack1, pack2, \
                                        pack3)                                                                   \
    NVCV_MAKE_COLOR_IMAGE_FORMAT(NVCV_COLOR_MODEL_##model, NVCV_COLOR_SPEC_##cspec, NVCV_MEM_LAYOUT_##layout,    \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 4,   \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1, NVCV_PACKING_##pack2,               \
                                 NVCV_PACKING_##pack3)

// clang-format off

static std::vector<SetSwizzlePackingTestParams> g_SSPParamsSuccess = {
    {MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, XYZ1, ASSOCIATED, X64, X16, X32),
     MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, ZYX1, ASSOCIATED, X16, X32, X64),
     NVCV_SWIZZLE_ZYX1,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X64},
     "Update all same number of channels"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, XYZ1, ASSOCIATED, X64, X16, X32),
     MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, ZYX0, ASSOCIATED, X16, X32, X64),
     NVCV_SWIZZLE_ZYX0,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X64},
     "Update all same number of channels and remove alpha"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, XYZ1, ASSOCIATED, X64, X16, X32),
     MAKE_COLOR_IMAGE_FORMAT_ABBREV1(RGB, BT601, PL, UNSIGNED, YX00, ASSOCIATED, X8_Y8),
     NVCV_SWIZZLE_YX00,
     {NVCV_PACKING_X8_Y8},
     "Change number of channels"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV4(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8, X16),
     MAKE_COLOR_IMAGE_FORMAT_ABBREV4(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8, X16),
     NVCV_SWIZZLE_XYZW,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8, NVCV_PACKING_X16},
     "Identity, no-op"},
};

static std::vector<SetSwizzlePackingTestParams> g_SSPParamsFailure = {
    {MAKE_COLOR_IMAGE_FORMAT_ABBREV4(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8, X16),
     NVCV_IMAGE_FORMAT_NONE,
     NVCV_SWIZZLE_XYZ1,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8, NVCV_PACKING_X16},
     "swizzle has less channels"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8),
     NVCV_IMAGE_FORMAT_NONE,
     NVCV_SWIZZLE_XYZW,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8},
     "swizzle has more channels"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8_Y8),
     NVCV_IMAGE_FORMAT_NONE,
     NVCV_SWIZZLE_XYZW,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8},
     "packing has less channels, same number of planes"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV4(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8, X16),
     NVCV_IMAGE_FORMAT_NONE,
     NVCV_SWIZZLE_XYZW,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8},
     "packing has less channels, different number of planes"},

    {MAKE_COLOR_IMAGE_FORMAT_ABBREV4(RGB, BT601, PL, UNSIGNED, XYZW, ASSOCIATED, X16, X32, X8, X16),
     NVCV_IMAGE_FORMAT_NONE,
     NVCV_SWIZZLE_XYZW,
     {NVCV_PACKING_X16, NVCV_PACKING_X32, NVCV_PACKING_X8, NVCV_PACKING_X128},
     "fourth plane has more than 64 bpp"},
};

// clang-format on

class ImageFormatSetSwizzlePackingTests : public t::TestWithParam<SetSwizzlePackingTestParams>
{
};

INSTANTIATE_TEST_SUITE_P(Success, ImageFormatSetSwizzlePackingTests, t::ValuesIn(g_SSPParamsSuccess));
INSTANTIATE_TEST_SUITE_P(Failure, ImageFormatSetSwizzlePackingTests, t::ValuesIn(g_SSPParamsFailure));

TEST_P(ImageFormatSetSwizzlePackingTests, run)
{
    NVCVImageFormat fmt   = GetParam().input;
    NVCVStatus goldStatus = GetParam().output == NVCV_IMAGE_FORMAT_NONE ? NVCV_ERROR_INVALID_ARGUMENT : NVCV_SUCCESS;

    ASSERT_EQ(goldStatus,
              nvcvImageFormatSetSwizzleAndPacking(&fmt, GetParam().swizzle, GetParam().packing[0],
                                                  GetParam().packing[1], GetParam().packing[2], GetParam().packing[3]));
    if (goldStatus == NVCV_SUCCESS)
    {
        EXPECT_EQ(GetParam().output, fmt);
    }
}

struct ImageFormatPair
{
    NVCVImageFormat a, b;

    friend std::ostream &operator<<(std::ostream &out, const ImageFormatPair &p)
    {
        return out << nvcvImageFormatGetName(p.a) << ',' << nvcvImageFormatGetName(p.b);
    }
};

class ImageFormatDataLayoutTests : public t::TestWithParam<std::tuple<int, ImageFormatPair>>
{
};

#define MAKE_YCbCr_IMAGE_FORMAT_ABBREV1(cspec, css, layout, datatype, swizzle, alphatype, pack0)               \
    NVCV_MAKE_YCbCr_IMAGE_FORMAT(NVCV_COLOR_SPEC_##cspec, NVCV_CSS_##css, NVCV_MEM_LAYOUT_##layout,            \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 1, \
                                 NVCV_PACKING_##pack0)

#define MAKE_YCbCr_IMAGE_FORMAT_ABBREV2(cspec, css, layout, datatype, swizzle, alphatype, pack0, pack1)        \
    NVCV_MAKE_YCbCr_IMAGE_FORMAT(NVCV_COLOR_SPEC_##cspec, NVCV_CSS_##css, NVCV_MEM_LAYOUT_##layout,            \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 2, \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1)

#define MAKE_YCbCr_IMAGE_FORMAT_ABBREV3(cspec, css, layout, datatype, swizzle, alphatype, pack0, pack1, pack2) \
    NVCV_MAKE_YCbCr_IMAGE_FORMAT(NVCV_COLOR_SPEC_##cspec, NVCV_CSS_##css, NVCV_MEM_LAYOUT_##layout,            \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 3, \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1, NVCV_PACKING_##pack2)

#define MAKE_YCbCr_IMAGE_FORMAT_ABBREV4(cspec, css, layout, datatype, swizzle, alphatype, pack0, pack1, pack2, pack3) \
    NVCV_MAKE_YCbCr_IMAGE_FORMAT(NVCV_COLOR_SPEC_##cspec, NVCV_CSS_##css, NVCV_MEM_LAYOUT_##layout,                   \
                                 NVCV_DATA_KIND_##datatype, NVCV_SWIZZLE_##swizzle, NVCV_ALPHA_##alphatype, 4,        \
                                 NVCV_PACKING_##pack0, NVCV_PACKING_##pack1, NVCV_PACKING_##pack2,                    \
                                 NVCV_PACKING_##pack3)

// clang-format off
static std::vector<ImageFormatPair> g_SameDataLayout = {
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(XYZ, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, BT601, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV3(UNDEFINED, 444, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV3(BT709, 444, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ1, ASSOCIATED, X8, X8, X8),
    },
};

static std::vector<ImageFormatPair> g_DifferentDataLayout = {
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BLOCK4_LINEAR, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, SIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV3(RGB, UNDEFINED, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8, X8),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV2(RGB, UNDEFINED, BL, UNSIGNED, XZY0, ASSOCIATED, X8, X8),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV2(RGB, UNDEFINED, BL, UNSIGNED, XY00, ASSOCIATED, X8, b4X12),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV2(RGB, UNDEFINED, BL, UNSIGNED, XY00, ASSOCIATED, X8, X12b4),
    },
    {
        MAKE_COLOR_IMAGE_FORMAT_ABBREV1(RGB, UNDEFINED, BL, UNSIGNED, XY00, ASSOCIATED, X10b6_Y10b6),
        MAKE_COLOR_IMAGE_FORMAT_ABBREV1(RGB, UNDEFINED, BL, UNSIGNED, XY00, ASSOCIATED, X12b4_Y12b4),
    },
    {
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV2(BT709, 444, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8),
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV2(BT709, 420, BL, UNSIGNED, XYZ0, ASSOCIATED, X8, X8_Y8),
    },
    {
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV1(BT709, 422, BL, UNSIGNED, XYZ0, ASSOCIATED, X8_Y8__X8_Z8),
        MAKE_YCbCr_IMAGE_FORMAT_ABBREV1(BT709, 422, BL, UNSIGNED, XYZ0, ASSOCIATED, Y8_X8__Z8_X8),
    },
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(Equal, ImageFormatDataLayoutTests, t::Combine(t::Values(1), t::ValuesIn(g_SameDataLayout)));

INSTANTIATE_TEST_SUITE_P(Different, ImageFormatDataLayoutTests,
                         t::Combine(t::Values(0), t::ValuesIn(g_DifferentDataLayout)));

TEST_P(ImageFormatDataLayoutTests, data_layout)
{
    int                    res = std::get<0>(GetParam());
    const ImageFormatPair &fmt = std::get<1>(GetParam());

    int8_t has;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatHasSameDataLayout(fmt.a, fmt.b, &has));
    EXPECT_EQ(res != 0, has != 0);
}

TEST(ImageFormatTests, packing1_at_most_128_bpp)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XY00, NVCV_PACKING_X8, NVCV_PACKING_X128,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XY00, NVCV_PACKING_X8, NVCV_PACKING_X192,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing2_at_most_128_bpp)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X128, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X192, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing3_at_most_128_bpp)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X8, NVCV_PACKING_X64, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X8, NVCV_PACKING_X128, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing0_code_at_most_3_bits)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601,
                                                     NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000,
                                                     (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 7), NVCV_PACKING_0,
                                                     NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_X000,
                                       (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 8), NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing1_code_at_most_3_bits)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS, nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601,
                                                     NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XY00,
                                                     NVCV_PACKING_X8, (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 7),
                                                     NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XY00, NVCV_PACKING_X8,
                                       (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 8), NVCV_PACKING_0, NVCV_PACKING_0,
                                       NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing2_code_at_most_3_bits)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 7), NVCV_PACKING_0,
                                       NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 8), NVCV_PACKING_0,
                                       NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing3_code_at_most_0_bits)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X8, (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 0),
                                       NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                       NVCV_PACKING_X8, (NVCVPacking)(NVCV_DETAIL_BPP_NCH(16, 1) + 1),
                                       NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, packing1_code_at_most_2_channels)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));

    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcvMakeColorImageFormat(&fmt, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL,
                                 NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8_Z8,
                                 NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
}

TEST(ImageFormatTests, get_valid_plane_size)
{
    int32_t outPlaneWidth;
    int32_t outPlaneHeight;
    auto    reset_output = [&outPlaneWidth, &outPlaneHeight]() -> void
    {
        outPlaneWidth  = -1;
        outPlaneHeight = -1;
    };
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneSize(NVCV_IMAGE_FORMAT_U8, 0, 224, 224, &outPlaneWidth, &outPlaneHeight));
    EXPECT_EQ(224, outPlaneWidth);
    EXPECT_EQ(224, outPlaneHeight);

    reset_output();
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneSize(NVCV_IMAGE_FORMAT_U8, 0, 224, 112, &outPlaneWidth, &outPlaneHeight));
    EXPECT_EQ(224, outPlaneWidth);
    EXPECT_EQ(112, outPlaneHeight);

    reset_output();
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneSize(NVCV_IMAGE_FORMAT_UYVY, 0, 224, 224, &outPlaneWidth, &outPlaneHeight));
    EXPECT_EQ(224, outPlaneWidth);
    EXPECT_EQ(224, outPlaneHeight);

    reset_output();
    EXPECT_EQ(NVCV_SUCCESS,
              nvcvImageFormatGetPlaneSize(NVCV_IMAGE_FORMAT_UYVY, 1, 224, 224, &outPlaneWidth, &outPlaneHeight));
    EXPECT_EQ(112, outPlaneWidth);
    EXPECT_EQ(224, outPlaneHeight);
}

TEST(ImageFormatTests, null_outputs_or_inputs)
{
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromFourCC(nullptr, 0, NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeColorImageFormat(nullptr, NVCV_COLOR_MODEL_RGB, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeYCbCrImageFormat(nullptr, NVCV_COLOR_SPEC_BT601, NVCV_CSS_420, NVCV_MEM_LAYOUT_PL,
                                       NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0, NVCV_PACKING_X8_Y8_Z8,
                                       NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeNonColorImageFormat(nullptr, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED, NVCV_SWIZZLE_XYZ0,
                                          NVCV_PACKING_X8_Y8_Z8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                          NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeRawImageFormat(nullptr, NVCV_RAW_BAYER_CRBC, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_UNSIGNED,
                                     NVCV_SWIZZLE_X000, NVCV_PACKING_X8, NVCV_PACKING_0, NVCV_PACKING_0, NVCV_PACKING_0,
                                     NVCV_ALPHA_ASSOCIATED, 0));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlanePacking(NVCV_IMAGE_FORMAT_NONE, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlaneBitsPerPixel(NVCV_IMAGE_FORMAT_NONE, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvImageFormatSetSwizzleAndPacking(nullptr, NVCV_SWIZZLE_XYZW, NVCV_PACKING_X8, NVCV_PACKING_X8,
                                                  NVCV_PACKING_X8, NVCV_PACKING_X8));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetDataKind(nullptr, NVCV_DATA_KIND_FLOAT));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetDataKind(NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetSwizzle(NVCV_IMAGE_FORMAT_NONE, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetMemLayout(nullptr, NVCV_MEM_LAYOUT_BL));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetMemLayout(NVCV_IMAGE_FORMAT_NONE, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetColorSpec(nullptr, NVCV_COLOR_SPEC_BT601));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetColorSpec(NVCV_IMAGE_FORMAT_NONE, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetColorModel(NVCV_IMAGE_FORMAT_NONE, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetChromaSubsampling(nullptr, NVCV_CSS_420));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetChromaSubsampling(NVCV_IMAGE_FORMAT_NONE, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlaneNumChannels(NVCV_IMAGE_FORMAT_U8, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlanePixelStrideBytes(NVCV_IMAGE_FORMAT_NV12, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetNumPlanes(NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetNumChannels(NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlaneDataType(NVCV_IMAGE_FORMAT_U8, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlaneSwizzle(NVCV_IMAGE_FORMAT_U8, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetPlaneFormat(NVCV_IMAGE_FORMAT_U8, 0, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatSetRawPattern(nullptr, NVCV_RAW_BAYER_BGGR));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvImageFormatHasSameDataLayout(NVCV_IMAGE_FORMAT_U8, NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatToFourCC(NVCV_IMAGE_FORMAT_RGBA8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetBitsPerChannel(NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatGetRawPattern(NVCV_IMAGE_FORMAT_U8, nullptr));
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvImageFormatGetPlaneSize(NVCV_IMAGE_FORMAT_U8, 0, 224, 224, nullptr, nullptr));
}

class FCC
{
public:
    explicit FCC(uint32_t code)
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        for (int i = 0; i < 4; ++i)
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (int i = 3; i >= 0; --i)
#else
#    error Yeah, that joke.
#endif
        {
            m_code[i] = code & 0xFF;
            code >>= 8;
        }
    }

    FCC(char a, char b, char c, char d)
    {
        m_code[0] = a;
        m_code[1] = b;
        m_code[2] = c;
        m_code[3] = d;
    }

    operator uint32_t() const
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return static_cast<uint32_t>(((int)m_code[3] << 24) | ((int)m_code[2] << 16) | ((int)m_code[1] << 8)
                                     | (int)m_code[0]);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return static_cast<uint32_t>(((int)m_code[0] << 24) | ((int)m_code[1] << 16) | ((int)m_code[2] << 8)
                                     | (int)m_code[3]);
#else
#    error Yeah, that joke.
#endif
    }

    friend std::ostream &operator<<(std::ostream &out, FCC fcc)
    {
        return out << fcc.m_code[0] << fcc.m_code[1] << fcc.m_code[2] << fcc.m_code[3];
    }

private:
    char m_code[4];
};

static const test::ValueList<FCC, nvcv::ImageFormat> g_FromFourCCParams = {
    {FCC{'R', 'G', 'B', '3'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8}},
    {FCC{'B', 'G', 'R', '3'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}},
    {FCC{'R', 'G', 'B', '4'}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8}},
    {FCC{'B', 'G', 'R', '4'}, nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRA8}},
    {FCC{'G', 'R', 'A', 'Y'},    nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8}},
    {FCC{'Y', '8', ' ', ' '},    nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8}},
    {FCC{'Y', '1', '6', ' '},   nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y16}},

    {FCC{'U', 'Y', 'V', 'Y'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY}},
    {FCC{'Y', 'U', 'Y', '2'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_YUYV}},
    {FCC{'Y', 'U', 'Y', 'V'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_YUYV}},
    {FCC{'Y', 'U', 'N', 'V'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_YUYV}},
    {FCC{'N', 'V', '1', '2'},  nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}},
};

static const test::ValueList<nvcv::ImageFormat, FCC> g_ToFourCCParams = {
    { nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGB8}, FCC{'R', 'G', 'B', '3'}},
    { nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGR8}, FCC{'B', 'G', 'R', '3'}},
    {nvcv::ImageFormat{NVCV_IMAGE_FORMAT_RGBA8}, FCC{'R', 'G', 'B', '4'}},
    {nvcv::ImageFormat{NVCV_IMAGE_FORMAT_BGRA8}, FCC{'B', 'G', 'R', '4'}},
    {   nvcv::ImageFormat{NVCV_IMAGE_FORMAT_Y8}, FCC{'G', 'R', 'A', 'Y'}},

    { nvcv::ImageFormat{NVCV_IMAGE_FORMAT_UYVY}, FCC{'U', 'Y', 'V', 'Y'}},
    { nvcv::ImageFormat{NVCV_IMAGE_FORMAT_YUYV}, FCC{'Y', 'U', 'Y', '2'}},
    { nvcv::ImageFormat{NVCV_IMAGE_FORMAT_NV12}, FCC{'N', 'V', '1', '2'}},
};

class ImageFormatFromFourCCTests : public t::TestWithParam<std::tuple<FCC, nvcv::ImageFormat>>
{
public:
    ImageFormatFromFourCCTests()
        : m_fourcc(std::get<0>(GetParam()))
        , m_fmt(std::get<1>(GetParam()))
    {
    }

protected:
    FCC             m_fourcc;
    NVCVImageFormat m_fmt;
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ImageFormatFromFourCCTests, g_FromFourCCParams);

TEST_P(ImageFormatFromFourCCTests, conversion_works)
{
    NVCVColorSpec cspec;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorSpec(m_fmt, &cspec));

    NVCVImageFormat test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeImageFormatFromFourCC(&test, m_fourcc, cspec, NVCV_MEM_LAYOUT_PL));
    EXPECT_EQ(m_fmt, test);
}

TEST_P(ImageFormatFromFourCCTests, conversion_with_undefined_colorspec_works)
{
    NVCVImageFormat gold = m_fmt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&gold, NVCV_COLOR_SPEC_UNDEFINED));

    NVCVImageFormat test;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeImageFormatFromFourCC(&test, m_fourcc, NVCV_COLOR_SPEC_UNDEFINED, NVCV_MEM_LAYOUT_PL));
    EXPECT_EQ(gold, test);
}

TEST_P(ImageFormatFromFourCCTests, conversion_works_while_forcing_mem_layout_works)
{
    NVCVImageFormat gold = m_fmt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetMemLayout(&gold, NVCV_MEM_LAYOUT_BL));

    NVCVColorSpec cspec;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetColorSpec(m_fmt, &cspec));

    NVCVImageFormat test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeImageFormatFromFourCC(&test, m_fourcc, cspec, NVCV_MEM_LAYOUT_BL));
    EXPECT_EQ(gold, test);
}

TEST_P(ImageFormatFromFourCCTests, conversion_works_while_forcing_colorspec_works)
{
    NVCVImageFormat gold = m_fmt;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatSetColorSpec(&gold, NVCV_COLOR_SPEC_BT2020));

    NVCVImageFormat test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeImageFormatFromFourCC(&test, m_fourcc, NVCV_COLOR_SPEC_BT2020, NVCV_MEM_LAYOUT_PL));
    EXPECT_EQ(gold, test);
}

class ImageFormatToFourCCTests : public t::TestWithParam<std::tuple<nvcv::ImageFormat, FCC>>
{
public:
    ImageFormatToFourCCTests()
        : m_fmt(std::get<0>(GetParam()))
        , m_fourcc(std::get<1>(GetParam()))
    {
    }

protected:
    NVCVImageFormat m_fmt;
    FCC             m_fourcc;
};

NVCV_INSTANTIATE_TEST_SUITE_P(_, ImageFormatToFourCCTests, g_ToFourCCParams);

TEST_P(ImageFormatToFourCCTests, conversion_works)
{
    uint32_t fourcc;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatToFourCC(m_fmt, &fourcc));

    EXPECT_EQ(m_fourcc, FCC{fourcc});
}

TEST(ImageFormatFourCCTests, image_doesnt_have_fourcc_return_0)
{
    uint32_t fourcc;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvImageFormatToFourCC(NVCV_IMAGE_FORMAT_2F32, &fourcc));
}

TEST(ImageFormatFourCCTests, invalid_fourcc_returns_invalid_imageformat)
{
    NVCVImageFormat fmt;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcvMakeImageFormatFromFourCC(&fmt, FCC('R', 'O', 'D', 'S'), NVCV_COLOR_SPEC_BT601, NVCV_MEM_LAYOUT_PL));
}

class ImageFormatPlanePixelStrideBytesExecTests
    : public t::TestWithParam<std::tuple<test::Param<"fmt", NVCVImageFormat>, test::Param<"plane", int>,
                                         test::Param<"goldStrideBytes", int>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_,ImageFormatPlanePixelStrideBytesExecTests,
                              test::ValueList<NVCVImageFormat, int, int>
                              {
                                {NVCV_IMAGE_FORMAT_NV12, 0, 1},
                                {NVCV_IMAGE_FORMAT_NV12, 1, 2},
                                {NVCV_IMAGE_FORMAT_RGB8, 0, 3},
                                {NVCV_IMAGE_FORMAT_RGBA8, 0, 4},
                                {NVCV_IMAGE_FORMAT_HSV8, 0, 3},
                                {NVCV_IMAGE_FORMAT_CMYK8, 0, 4},
                                {NVCV_IMAGE_FORMAT_YCCK8, 0, 4},
                                {NVCV_IMAGE_FORMAT_U8, 0, 1},
                                {NVCV_IMAGE_FORMAT_U16, 0, 2},
                              });

// clang-format on

TEST_P(ImageFormatPlanePixelStrideBytesExecTests, works)
{
    const NVCVImageFormat dtype      = std::get<0>(GetParam());
    const int             plane      = std::get<1>(GetParam());
    const int             goldStride = std::get<2>(GetParam());

    int32_t testStride;
    ASSERT_EQ(NVCV_SUCCESS, nvcvImageFormatGetPlanePixelStrideBytes(dtype, plane, &testStride));
    EXPECT_EQ(goldStride, testStride);
}
