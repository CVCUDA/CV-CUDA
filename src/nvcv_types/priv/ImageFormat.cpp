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

#include "ImageFormat.hpp"

#include "Bitfield.hpp"
#include "ColorFormat.hpp"
#include "DataLayout.hpp"
#include "DataType.hpp"
#include "Exception.hpp"

#include <util/Assert.h>

#include <map>
#include <sstream>
#include <vector>

namespace nvcv::priv {

static void ValidateSwizzlePacking(NVCVSwizzle swizzle, NVCVPacking packing0, NVCVPacking packing1,
                                   NVCVPacking packing2, NVCVPacking packing3)
{
    if (packing0 == NVCV_PACKING_0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Packing of first plane must not be 0");
    }

    // packing0's pack code must be at most 3 bits.
    if (packing0 & 0b1000)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Code of 1st plane packing must be limited to 3 bits");
    }

    // packing1's pack code must be at most 3 bits.
    if (packing1 & 0b1000)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Code of 2nd plane packing must be limited to 3 bits");
    }

    // packing2's pack code must be at most 3 bits.
    if (packing2 & 0b1000)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Code of 3rd plane packing must be limited to 3 bits");
    }

    // packing3's pack code must be 0
    if ((packing3 & 0b1111) != 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Code of 4th plane packing must be 0");
    }

    int swchannels = GetNumChannels(swizzle);

    NVCVPacking packing[] = {packing0, packing1, packing2, packing3};

    int packchannels = 0;
    for (int i = 0; i < 4; ++i)
    {
        int nch = GetNumChannels(packing[i]);

        if (i == 3)
        {
            // we use 0b111 to represent 8-bit, so it can't represent 128-bit
            if (GetBitsPerPixel(packing3) > 64)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                "Packing of 3rd plane must have at most 64 bits per pixel");
            }
        }
        else if (i > 0)
        {
            // only packing 0 can have more than 128 bpp
            if (GetBitsPerPixel(packing[i]) > 128)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                "Packing of plane other than 1st must have at most 128 bits per pixel");
            }
        }

        // packing 1 can't have more than 2 channels
        if (i == 1 && nch > 2)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Packing of 2nd plane must not have more than 2 channels");
        }

        packchannels += nch;
    }

    if (swchannels != packchannels)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Number of channels in swizzle, " << swchannels
            << ", doesn't match number of channels defined by packings, " << packchannels;
    }
}

ImageFormat::ImageFormat(NVCVColorModel colorModel, ColorSpec colorSpec, NVCVChromaSubsampling chromaSub,
                         NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                         NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3)
{
    ValidateSwizzlePacking(swizzle, packing0, packing1, packing2, packing3);

    if (colorModel == NVCV_COLOR_MODEL_RAW)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "When specifying colorspec, chroma subsampling and swizzle,"
                                                     << " color model can't be RAW";
    }

    if (colorModel == NVCV_COLOR_MODEL_UNDEFINED
        && (colorSpec != NVCV_COLOR_SPEC_UNDEFINED || chromaSub != NVCV_CSS_NONE))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "If color model is undefined,"
            << " colorspec must be undefined (not " << colorSpec << " and chroma subsampling must be none (not "
            << chromaSub << ")";
    }
    else
    {
        switch (colorModel)
        {
        case NVCV_COLOR_MODEL_RGB:
        case NVCV_COLOR_MODEL_YCbCr:
            break;

        default:
            if (colorSpec != NVCV_COLOR_SPEC_UNDEFINED)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "When color model is not RGB or YCbCr, colorspec must be undefined, not " << colorSpec;
            }
            break;
        }

        if (colorModel == NVCV_COLOR_MODEL_YCbCr)
        {
            m_format = NVCV_MAKE_YCbCr_IMAGE_FORMAT(colorSpec, chromaSub, memLayout, dataKind, swizzle, 4, packing0,
                                                    packing1, packing2, packing3);
        }
        else if (chromaSub != NVCV_CSS_NONE)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "When color model isn't YCbCr, chroma subsampling must be NONE";
        }
        else
        {
            m_format = NVCV_MAKE_COLOR_IMAGE_FORMAT(colorModel, colorSpec, memLayout, dataKind, swizzle, 4, packing0,
                                                    packing1, packing2, packing3);
        }
    }
}

ImageFormat::ImageFormat(NVCVRawPattern rawPattern, NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle,
                         NVCVPacking packing0, NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3)
{
    ValidateSwizzlePacking(swizzle, packing0, packing1, packing2, packing3);

    m_format = NVCV_MAKE_RAW_IMAGE_FORMAT(rawPattern, memLayout, dataKind, swizzle, 4, packing0, packing1, packing2,
                                          packing3);
}

ImageFormat::ImageFormat(NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                         NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3)
{
    ValidateSwizzlePacking(swizzle, packing0, packing1, packing2, packing3);

    m_format = NVCV_MAKE_NONCOLOR_IMAGE_FORMAT(memLayout, dataKind, swizzle, 4, packing0, packing1, packing2, packing3);
}

ImageFormat::ImageFormat(const ColorFormat &colorFormat, NVCVChromaSubsampling chromaSub, NVCVMemLayout memLayout,
                         NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0, NVCVPacking packing1,
                         NVCVPacking packing2, NVCVPacking packing3)
{
    if (colorFormat.model == NVCV_COLOR_MODEL_RAW)
    {
        if (chromaSub != NVCV_CSS_NONE)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "When color model is raw, chroma subsampling must be NONE, not " << chromaSub;
        }

        m_format = ImageFormat{colorFormat.raw, memLayout, dataKind, swizzle, packing0, packing1, packing2, packing3}
                       .value();
    }
    else
    {
        m_format = ImageFormat{colorFormat.model, colorFormat.cspec, chromaSub, memLayout, dataKind,
                               swizzle,           packing0,          packing1,  packing2,  packing3}
                       .value();
        ;
    }
}

ImageFormat ImageFormat::FromPlanes(const util::StaticVector<ImageFormat, 4> &fmtPlanes)
{
    if (fmtPlanes.empty())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "At least one plane must be specified");
    }

    ColorFormat                   colorFormat = fmtPlanes[0].colorFormat();
    NVCVMemLayout                 memLayout   = fmtPlanes[0].memLayout();
    NVCVDataKind                  dataKind    = fmtPlanes[0].dataKind();
    std::optional<NVCVRawPattern> rawPattern  = fmtPlanes[0].rawPattern();

    NVCVChromaSubsampling css = NVCV_CSS_NONE;

    int totChannels = 0;
    for (size_t i = 0; i < fmtPlanes.size(); ++i)
    {
        // all plane types must have just one plane.
        if (fmtPlanes[i].numPlanes() != 1)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Format for plane #" << i << " must have only one plane";
        }

        // first plane must have a valid packing
        if (fmtPlanes[i].planePacking(0) == NVCV_PACKING_0)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Format for plane #" << i << " must have a non-zero packing";
        }

        NVCV_ASSERT(fmtPlanes[i].planePacking(1) == NVCV_PACKING_0);
        NVCV_ASSERT(fmtPlanes[i].planePacking(2) == NVCV_PACKING_0);
        NVCV_ASSERT(fmtPlanes[i].planePacking(3) == NVCV_PACKING_0);

        // total number of channels must be at most 4.
        totChannels += fmtPlanes[i].numChannels();
        if (totChannels > 4)
        {
            // Get the total number of channels for the exception error message.
            for (++i; i < fmtPlanes.size(); ++i)
            {
                totChannels += fmtPlanes[i].numChannels();
            }
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Total number of channels comprised by all valid plane formats"
                << " must be at most 4, not " << totChannels;
        }

        if (i >= 1)
        {
            // color spec, mem layout and data type of all planes must be the same
            if (fmtPlanes[i].colorFormat() != colorFormat)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Color format of all plane formats must be the same";
            }
            if (fmtPlanes[i].memLayout() != memLayout)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Memory layout of all plane formats must be the same, but plane #" << i << "'s is " << memLayout;
            }
            if (fmtPlanes[i].dataKind() != dataKind)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Data type of all plane formats must be the same, but plane #" << i << "'s is " << dataKind;
            }
            if (fmtPlanes[i].rawPattern() != rawPattern)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Raw pattern of all plane formats must be the same";
            }
        }

        NVCVChromaSubsampling plcss = fmtPlanes[i].css();
        if (css == NVCV_CSS_NONE)
        {
            css = plcss;
        }
        else if (css != plcss)
        {
            // only one kind of chroma subsampling is allowed.
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Only one chroma-subsampling type must be specified, but plane #" << i << "'s differ from " << css;
        }
    }

    // at least one channel is allowed
    if (totChannels == 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Total number of channels cannot be 0";
    }

    NVCVSwizzle swPlane[4] = {
        fmtPlanes[0].swizzle(),
        fmtPlanes.size() > 1 ? fmtPlanes[1].swizzle() : NVCV_SWIZZLE_0000,
        fmtPlanes.size() > 2 ? fmtPlanes[2].swizzle() : NVCV_SWIZZLE_0000,
        fmtPlanes.size() > 3 ? fmtPlanes[3].swizzle() : NVCV_SWIZZLE_0000,
    };

    NVCVSwizzle swizzle = MergePlaneSwizzles(swPlane[0], swPlane[1], swPlane[2], swPlane[3]);

    NVCVPacking packPlane[4] = {
        fmtPlanes[0].planePacking(0),
        fmtPlanes.size() > 1 ? fmtPlanes[1].planePacking(0) : NVCV_PACKING_0,
        fmtPlanes.size() > 2 ? fmtPlanes[2].planePacking(0) : NVCV_PACKING_0,
        fmtPlanes.size() > 3 ? fmtPlanes[3].planePacking(0) : NVCV_PACKING_0,
    };

    return ImageFormat{colorFormat,  css,          memLayout,    dataKind,    swizzle,
                       packPlane[0], packPlane[1], packPlane[2], packPlane[3]};
}

namespace {

constexpr uint32_t FCC(char a, char b, char c, char d)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return static_cast<uint32_t>((d << 24) | (c << 16) | (b << 8) | a);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return static_cast<uint32_t>((a << 24) | (b << 16) | (c << 8) | d);
#else
#    error Insert that old PDP-11 joke here.
#endif
}

#define FCC_IF(model, css, type, swizzle, ...)                                                   \
    ImageFormat                                                                                  \
    {                                                                                            \
        NVCV_COLOR_MODEL_##model, NVCV_COLOR_SPEC_UNDEFINED, NVCV_CSS_##css, NVCV_MEM_LAYOUT_PL, \
            NVCV_DATA_KIND_##type, NVCV_SWIZZLE_##swizzle, __VA_ARGS__                           \
    }

#define FCC_BAYER_IF(pattern, type, swizzle, ...)                                                                \
    ImageFormat                                                                                                  \
    {                                                                                                            \
        NVCV_RAW_BAYER_##pattern, NVCV_MEM_LAYOUT_PL, NVCV_DATA_KIND_##type, NVCV_SWIZZLE_##swizzle, __VA_ARGS__ \
    }

// clang-format off
// Using vector b/c we want to preserve ordering and it's probably faster than using std::map
// due to cache locality.
const std::vector<std::pair<uint32_t, ImageFormat>> g_FourCC =
{
    { FCC('R','G','B','1'), FCC_IF(RGB, 444, UNSIGNED, ZYX1, NVCV_PACKING_X3Y3Z2) },         // RGB332
    { FCC('R','4','4','4'), FCC_IF(RGB, 444, UNSIGNED, XYZ1, NVCV_PACKING_b4X4Y4Z4) },       // RGB444
    { FCC('R','G','B','0'), FCC_IF(RGB, 444, UNSIGNED, ZYX1, NVCV_PACKING_b1X5Y5Z5) },       // RGB555
    { FCC('R','G','B','P'), FCC_IF(RGB, 444, UNSIGNED, ZYX1, NVCV_PACKING_X5Y6Z5) },         // RGB565
    { FCC('R','G','B','Q'), FCC_IF(RGB, 444, UNSIGNED, XYZ1, NVCV_PACKING_X5Y5b1Z5) },       // RGB555X
    { FCC('R','G','B','R'), FCC_IF(RGB, 444, UNSIGNED, XYZ1, NVCV_PACKING_X5Y6Z5) },         // RGB565X
    { FCC('B','G','R','3'), FCC_IF(RGB, 444, UNSIGNED, ZYX1, NVCV_PACKING_X8_Y8_Z8) },       // BGR24
    { FCC('R','G','B','3'), FCC_IF(RGB, 444, UNSIGNED, XYZ1, NVCV_PACKING_X8_Y8_Z8) },       // RGB24
    { FCC('B','G','R','4'), FCC_IF(RGB, 444, UNSIGNED, ZYXW, NVCV_PACKING_X8_Y8_Z8_W8) },    // BGR24
    { FCC('R','G','B','4'), FCC_IF(RGB, 444, UNSIGNED, XYZW, NVCV_PACKING_X8_Y8_Z8_W8) },    // RGB24

    { FCC('B','A','8','1'), FCC_BAYER_IF(BGGR, UNSIGNED, X000, NVCV_PACKING_X8) },           // SBGGR8

    { FCC('A','Y','U','V'), FCC_IF(YCbCr, 444, UNSIGNED, YZWX, NVCV_PACKING_X8_Y8_Z8_W8) },
    { FCC('A','Y','U','V'), FCC_IF(YCbCr, 444, UNSIGNED, YZWX, NVCV_PACKING_X8_Y8_Z8_W8) },

    { FCC('G','R','A','Y'), FCC_IF(YCbCr, 444, UNSIGNED, X000, NVCV_PACKING_X8) },
    { FCC('Y','8',' ',' '), FCC_IF(YCbCr, 444, UNSIGNED, X000, NVCV_PACKING_X8) },
    { FCC('Y','8','0','0'), FCC_IF(YCbCr, 444, UNSIGNED, X000, NVCV_PACKING_X8) },
    { FCC('Y','1','6',' '), FCC_IF(YCbCr, 444, UNSIGNED, X000, NVCV_PACKING_X16) },

    { FCC('U','Y','V','Y'), FCC_IF(YCbCr, 422, UNSIGNED, XYZ1, NVCV_PACKING_Y8_X8__Z8_X8) },
    { FCC('Y','U','Y','2'), FCC_IF(YCbCr, 422, UNSIGNED, XYZ1, NVCV_PACKING_X8_Y8__X8_Z8) },
    { FCC('Y','U','Y','V'), FCC_IF(YCbCr, 422, UNSIGNED, XYZ1, NVCV_PACKING_X8_Y8__X8_Z8) },
    { FCC('Y','U','N','V'), FCC_IF(YCbCr, 422, UNSIGNED, XYZ1, NVCV_PACKING_X8_Y8__X8_Z8) },
    { FCC('Y','V','Y','U'), FCC_IF(YCbCr, 422, UNSIGNED, XZY1, NVCV_PACKING_X8_Y8__X8_Z8) },

    { FCC('I','4','2','0'), FCC_IF(YCbCr, 420, UNSIGNED, XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_X8) },
    { FCC('I','Y','U','V'), FCC_IF(YCbCr, 420, UNSIGNED, XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_X8) },

    { FCC('N','V','1','2'), FCC_IF(YCbCr, 420, UNSIGNED, XYZ0, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8) },
    { FCC('N','V','2','1'), FCC_IF(YCbCr, 420, UNSIGNED, XZY0, NVCV_PACKING_X8, NVCV_PACKING_X8_Y8) },

    { FCC('Y','V','1','2'), FCC_IF(YCbCr, 420, UNSIGNED, XZY0, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_X8) },
    { FCC('Y','V','1','6'), FCC_IF(YCbCr, 422, UNSIGNED, XZY0, NVCV_PACKING_X8, NVCV_PACKING_X8, NVCV_PACKING_X8) },
};
// clang-format on

#undef FCC_IF

} // namespace

ImageFormat ImageFormat::FromFourCC(uint32_t fourcc, ColorSpec colorSpec, NVCVMemLayout memLayout)
{
    // First make sure fourcc is uppercase.

    char *tmp = reinterpret_cast<char *>(&fourcc);
    for (int i = 0; i < 4; ++i)
    {
        tmp[i] = toupper(tmp[i]);
    }
    fourcc = *reinterpret_cast<uint32_t *>(tmp);

    for (auto &p : g_FourCC)
    {
        if (p.first == fourcc)
        {
            ImageFormat newFmt = p.second;

            if (NeedsColorspec(p.second.colorModel()))
            {
                newFmt = newFmt.colorSpec(colorSpec);
            }

            return newFmt.memLayout(memLayout);
        }
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
        << tmp[0] << tmp[1] << tmp[2] << tmp[3] << " fourcc code is not supported";
}

bool HasSameDataLayout(ImageFormat a, ImageFormat b) noexcept
{
    if (a.numPlanes() == b.numPlanes() && a.css() == b.css() && a.memLayout() == b.memLayout())
    {
        for (int i = 0; i < a.numPlanes(); ++i)
        {
            // we have to test packing explicitely t as data type's packing isn't the same
            // as image format's plane packing
            if (a.planePacking(i) != b.planePacking(i) || a.planeDataType(i) != b.planeDataType(i))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}

int ImageFormat::planeBPP(int plane) const noexcept
{
    return GetBitsPerPixel(this->planePacking(plane));
}

NVCVSwizzle ImageFormat::swizzle() const noexcept
{
    return static_cast<NVCVSwizzle>(ExtractBitfield(m_format, 0, 3 * 4));
}

NVCVColorModel ImageFormat::colorModel() const noexcept
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        return NVCV_COLOR_MODEL_UNDEFINED;
    }

    if (ExtractBitfield(m_format, 16, 1) == 0)
    {
        return NVCV_COLOR_MODEL_YCbCr;
    }
    else if (ExtractBitfield(m_format, 16, 19) == (1u << 19) - 1)
    {
        return NVCV_COLOR_MODEL_UNDEFINED;
    }
    else
    {
        uint32_t tmp = ExtractBitfield(m_format, 17, 3);
        if (tmp < 7)
        {
            // models (other than YCbCr) with color spec
            return (NVCVColorModel)(tmp + 2);
        }
        else
        {
            if (ExtractBitfield(m_format, 20, 1) == 0)
            {
                return NVCV_COLOR_MODEL_RAW;
            }
            else
            {
                return (NVCVColorModel)(ExtractBitfield(m_format, 21, 6) + (7 + 2 + 1));
            }
        }
    }
}

std::optional<NVCVRawPattern> ImageFormat::rawPattern() const noexcept
{
    if (this->colorModel() == NVCV_COLOR_MODEL_RAW)
    {
        return static_cast<NVCVRawPattern>(ExtractBitfield(m_format, 21, 6));
    }
    else
    {
        return std::nullopt;
    }
}

ColorSpec ImageFormat::colorSpec() const noexcept
{
    NVCVColorModel model = this->colorModel();
    if (model != NVCV_COLOR_MODEL_UNDEFINED && model < 0x7)
    {
        return ColorSpec{static_cast<NVCVColorSpec>(ExtractBitfield(m_format, 20, 15))};
    }
    else
    {
        return NVCV_COLOR_SPEC_UNDEFINED;
    }
}

NVCVMemLayout ImageFormat::memLayout() const noexcept
{
    return static_cast<NVCVMemLayout>(ExtractBitfield(m_format, 12, 3));
}

ColorFormat ImageFormat::colorFormat() const noexcept
{
    ColorFormat colorFormat{this->colorModel()};

    if (colorFormat.model == NVCV_COLOR_MODEL_RAW)
    {
        std::optional<NVCVRawPattern> raw = this->rawPattern();
        NVCV_ASSERT(raw);
        colorFormat.raw = *raw;
    }
    else
    {
        colorFormat.cspec = this->colorSpec();
    }

    return colorFormat;
}

NVCVChromaSubsampling ImageFormat::css() const noexcept
{
    if (this->colorModel() == NVCV_COLOR_MODEL_YCbCr)
    {
        return static_cast<NVCVChromaSubsampling>(ExtractBitfield(m_format, 17, 3));
    }
    else
    {
        return NVCV_CSS_NONE;
    }
}

int ImageFormat::blockHeightLog2() const noexcept
{
    return GetBlockHeightLog2(this->memLayout());
}

int ImageFormat::planeNumChannels(int plane) const noexcept
{
    return GetNumChannels(this->planePacking(plane));
}

int ImageFormat::planeRowAlignment(int plane) const noexcept
{
    return planeDataType(plane).alignment();
}

int ImageFormat::numPlanes() const noexcept
{
    return (this->planeNumChannels(0) != 0 ? 1 : 0) + (this->planeNumChannels(1) != 0 ? 1 : 0)
         + (this->planeNumChannels(2) != 0 ? 1 : 0) + (this->planeNumChannels(3) != 0 ? 1 : 0);
}

int ImageFormat::numChannels() const noexcept
{
    return GetNumChannels(this->swizzle());
}

std::array<int32_t, 4> ImageFormat::bpc() const
{
    std::array<int32_t, 4> bits;

    int nplanes = this->numPlanes();
    int nch     = 0;

    for (int p = 0; p < nplanes; ++p)
    {
        NVCVPacking packing = this->planePacking(p);

        std::array<int32_t, 4> pbits = GetBitsPerComponent(packing);

        switch (packing)
        {
        case NVCV_PACKING_X8_Y8__X8_Z8:
        case NVCV_PACKING_Y8_X8__Z8_X8:
            pbits[3] = 0;
            break;
        default:
            break;
        }

        for (int c = 0; c < 4 && pbits[c] != 0; ++c)
        {
            if (nch >= 4)
            {
                throw Exception(NVCV_ERROR_INVALID_IMAGE_FORMAT)
                    << "Inconsistent image format, sum of planes' channel count " << nch << " > 4";
            }

            bits[nch++] = pbits[c];
        }
    }

    for (int i = nch; i < 4; ++i)
    {
        bits[i] = 0;
    }

    return bits;
}

Size2D ImageFormat::planeSize(Size2D imgSize, int plane) const noexcept
{
    // first plane always has full size
    if (plane >= 1 && this->css() != NVCV_CSS_NONE)
    {
        // different plane size is only meaningful in YUV color specs.
        auto [samplesHoriz, samplesVert] = GetChromaSamples(this->css());

        return {(imgSize.w * samplesHoriz + 3) / 4, (imgSize.h * samplesVert + 3) / 4};
    }

    return imgSize;
}

NVCVSwizzle ImageFormat::planeSwizzle(int plane) const
{
    // shortcut
    int totPlanes = this->numPlanes();
    if (totPlanes == 1)
    {
        if (plane == 0)
        {
            return this->swizzle();
        }
        else
        {
            return NVCV_SWIZZLE_0000;
        }
    }

    NVCVChannel plsw[4] = {};

    int ch = 0;
    for (int i = 0; i < plane; ++i)
    {
        ch += this->planeNumChannels(i);
    }

    NVCVSwizzle sw = this->swizzle();

    std::array<NVCVChannel, 4> tch = GetChannels(sw);
    int                        nch = this->planeNumChannels(plane);

    bool empty = true;
    for (int i = ch; i < ch + nch; ++i)
    {
        assert(i < 4);
        switch (tch[i])
        {
        case NVCV_CHANNEL_X:
        case NVCV_CHANNEL_Y:
        case NVCV_CHANNEL_Z:
        case NVCV_CHANNEL_W:
            plsw[tch[i] - NVCV_CHANNEL_X] = (NVCVChannel)((i - ch) + (int)NVCV_CHANNEL_X);
            empty                         = false;
            break;
        case NVCV_CHANNEL_0:
            ++nch;
            break;
        default:
            plsw[i] = tch[i];
            empty   = false;
            break;
        }
    }

    if (!empty)
    {
        // if swizzle is using channel '1', it must be set in all
        // plane swizzles, as it represents that the pixel has an
        // opaque alpha, although the alpha channel isn't physically there.
        for (int i = 0; i < 4; ++i)
        {
            if (tch[i] == NVCV_CHANNEL_1)
            {
                if (plsw[i] != NVCV_CHANNEL_0)
                {
                    throw Exception(
                        NVCV_ERROR_INVALID_IMAGE_FORMAT,
                        "Plane swizzle is inconsistent, if using '1' channel, it must be set in all planes' swizzles,");
                }
                plsw[i] = tch[i];
            }
        }
    }

    return MakeNVCVSwizzle(plsw[0], plsw[1], plsw[2], plsw[3]);
}

ImageFormat ImageFormat::planeFormat(int plane) const
{
    if (plane < this->numPlanes())
    {
        return ImageFormat{this->colorFormat(),       plane == 0 ? NVCV_CSS_NONE : this->css(),
                           this->memLayout(),         this->dataKind(),
                           this->planeSwizzle(plane), this->planePacking(plane)};
    }
    else
    {
        return ImageFormat{NVCV_IMAGE_FORMAT_NONE};
    }
}

DataType ImageFormat::planeDataType(int plane) const noexcept
{
    NVCVDataKind dataKind = this->dataKind();
    NVCVPacking  packing  = this->planePacking(plane);

    switch (packing)
    {
    case NVCV_PACKING_0:
        return DataType{NVCV_DATA_TYPE_NONE};

    case NVCV_PACKING_X8_Y8__X8_Z8:
    case NVCV_PACKING_Y8_X8__Z8_X8:
        // that's the packing of each pixel
        packing = NVCV_PACKING_X8_Y8;
        break;
    default:
        break;
    }

    return DataType{dataKind, packing};
}

int ImageFormat::planePixelStrideBytes(int plane) const noexcept
{
    return this->planeDataType(plane).strideBytes();
}

uint32_t ImageFormat::fourCC() const
{
    // normalize
    ImageFormat fmtNorm = this->colorSpec(NVCV_COLOR_SPEC_UNDEFINED).memLayout(NVCV_MEM_LAYOUT_PL);

    for (auto &p : g_FourCC)
    {
        if (p.second == fmtNorm)
        {
            return p.first;
        }
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
        << "Format " << *this << " doesn't corresponding to any supported fourcc code";
}

NVCVColorRange ImageFormat::colorRange() const
{
    return this->colorSpec().colorRange();
}

int ImageFormat::bitDepth() const
{
    if (this->hasUniformBitDepth())
    {
        std::array<int32_t, 4> bpc = this->bpc();
        return bpc[0];
    }
    else
    {
        return 0;
    }
}

bool ImageFormat::hasUniformBitDepth() const
{
    std::array<int32_t, 4> bpc = this->bpc();

    int nchannels = this->numChannels();

    for (int i = 1; i < nchannels; ++i)
    {
        if (bpc[i - 1] != bpc[i])
        {
            return false;
        }
    }

    return true;
}

ImageFormat ImageFormat::dataKind(NVCVDataKind newDataKind) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set data type of NONE format");
    }

    // unsigned -> signed is undefined behavior, we have to be careful when handling that.

    // Main problem is that enums can be at most int64_t, but we need actually
    // uint64_t. The 64 bits are still there, we just have to handle the sign
    // bit to be the 64th bit of our bitmask.

    if (ExtractBitfield(newDataKind, 2, 1))
    {
        // here the result won't fit into an int64_t.

        // To avoid unsigned -> signed conversion, we calculate the 63 bits that
        // once negated will be equal to the first 63 bit of 'fmt'. The 64th
        // bit, the sign bit, will be 1, as we want.
        return ImageFormat{static_cast<NVCVImageFormat>(
            -(1
              + (int64_t)(~(((uint64_t)m_format & ~MaskBitfield(61, 3)) | SetBitfield(newDataKind, 61, 2))
                          & ~(1ULL << 63))))};
    }
    else
    {
        // the result fits into an int64_t, we don't need any hackery.
        return ImageFormat{static_cast<NVCVImageFormat>(
            (((uint64_t)m_format & ~MaskBitfield(61, 3)) | SetBitfield(newDataKind, 61, 3)))};
    }
}

ImageFormat ImageFormat::rawPattern(NVCVRawPattern newRawPattern) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set raw pattern of NONE format");
    }

    if (this->colorModel() == NVCV_COLOR_MODEL_RAW)
    {
        return ImageFormat{static_cast<NVCVImageFormat>(
            (((uint64_t)m_format & ~MaskBitfield(21, 6)) | SetBitfield(newRawPattern, 21, 6)))};
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Can't set the raw pattern of a format whose color model is not RAW."
            << " It currently is " << this->colorModel();
    }
}

ImageFormat ImageFormat::colorFormat(const ColorFormat &newColorFormat) const
{
    if (newColorFormat.model == NVCV_COLOR_MODEL_RAW)
    {
        return this->rawPattern(newColorFormat.raw);
    }
    else
    {
        return this->colorSpec(newColorFormat.cspec);
    }
}

ImageFormat ImageFormat::swizzleAndPacking(NVCVSwizzle newSwizzle, NVCVPacking newPacking0, NVCVPacking newPacking1,
                                           NVCVPacking newPacking2, NVCVPacking newPacking3) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set raw pattern of NONE format");
    }

    return ImageFormat{this->colorFormat(), this->css(), this->memLayout(), this->dataKind(), newSwizzle,
                       newPacking0,         newPacking1, newPacking2,       newPacking3};
}

ImageFormat ImageFormat::memLayout(NVCVMemLayout newMemLayout) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set memory layout of NONE format");
    }

    return ImageFormat{
        static_cast<NVCVImageFormat>((((uint64_t)m_format & ~MaskBitfield(12, 3)) | SetBitfield(newMemLayout, 12, 3)))};
}

ImageFormat ImageFormat::colorSpec(ColorSpec newColorSpec) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set colorspec of NONE format");
    }

    switch (this->colorModel())
    {
    case NVCV_COLOR_MODEL_RGB:
        // RGB doesn't use ycbcr encoding, it must be undefined
        newColorSpec = newColorSpec.YCbCrEncoding(NVCV_YCbCr_ENC_UNDEFINED);
        break;

    case NVCV_COLOR_MODEL_YCbCr:
        // YCbCr needs an encoding
        if (newColorSpec.YCbCrEncoding() == NVCV_YCbCr_ENC_UNDEFINED)
        {
            // Let's use the one specified by the undefined color spec.
            newColorSpec.YCbCrEncoding(ColorSpec(NVCV_COLOR_SPEC_UNDEFINED).YCbCrEncoding());
        }
        break;

    default:
        if (this->colorSpec() != NVCV_COLOR_SPEC_UNDEFINED)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "If image format's color model isn't RGB or YCbCr,"
                << " its colorspec must be UNDEFINED,"
                << " but color model is " << this->colorModel() << " and its colorspec is " << this->colorSpec();
        }
        return *this;
    }

    return ImageFormat{static_cast<NVCVImageFormat>(
        (((uint64_t)m_format & ~MaskBitfield(20, 15)) | SetBitfield(newColorSpec, 20, 15)))};
}

ImageFormat ImageFormat::css(NVCVChromaSubsampling newCSS) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set chroma subsampling of NONE format");
    }

    if (this->colorModel() == NVCV_COLOR_MODEL_YCbCr)
    {
        return ImageFormat{
            static_cast<NVCVImageFormat>((((uint64_t)m_format & ~MaskBitfield(17, 3)) | SetBitfield(newCSS, 17, 3)))};
    }
    else if (newCSS == NVCV_CSS_NONE)
    {
        return *this;
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "If color model isn't YCbCr, chroma subsampling must be NONE");
    }
}

ImageFormat ImageFormat::colorRange(NVCVColorRange newColorRange) const
{
    if (m_format == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't set color range of NONE format");
    }

    return this->colorSpec(this->colorSpec().colorRange(newColorRange));
}

ImageFormat UpdateColorSpec(ImageFormat fmt, ColorSpec source)
{
    if (fmt.value() == NVCV_IMAGE_FORMAT_NONE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Can't update colorspec of NONE format");
    }

    ColorSpec cspec = fmt.colorSpec();

    if (cspec == NVCV_COLOR_SPEC_UNDEFINED && source != NVCV_COLOR_SPEC_UNDEFINED)
    {
        cspec = cspec.colorSpace(source.colorSpace());

        NVCVColorModel model = fmt.colorModel();

        if (model == NVCV_COLOR_MODEL_RGB)
        {
            // Always use sRGB for RGB, following what VIC does.
            cspec = cspec.xferFunc(NVCV_COLOR_XFER_sRGB);
        }
        else
        {
            cspec = cspec.xferFunc(source.xferFunc());
        }

        if (model == NVCV_COLOR_MODEL_YCbCr)
        {
            cspec = cspec.YCbCrEncoding(source.YCbCrEncoding())
                        .colorRange(source.colorRange())
                        .chromaLoc(source.chromaLoc());
        }
        else
        {
            ColorSpec und{NVCV_COLOR_SPEC_UNDEFINED};

            cspec = cspec.chromaLoc(und.chromaLoc()).colorRange(und.colorRange());
        }
    }

    return fmt.colorSpec(cspec);
}

std::ostream &operator<<(std::ostream &out, ImageFormat fmt)
{
    switch (fmt.value())
    {
#define NVCV_ENUM(E) \
    case E:          \
        return out << #E;
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NONE);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_U8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_U8_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_S8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_U16);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_S16);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_S16_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_U32);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_S32);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y8_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y8_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y8_ER_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y16);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y16_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y16_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_Y16_ER_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV12);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV12_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV12_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV12_ER_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV24);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV24_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV24_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_NV24_ER_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_RGB8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_RGBA8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_BGR8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_BGRA8);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_F32);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_F64);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_2S16);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_2S16_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_2F32);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_UYVY);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_UYVY_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_UYVY_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_UYVY_ER_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_YUYV);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_YUYV_BL);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_YUYV_ER);
        NVCV_ENUM(NVCV_IMAGE_FORMAT_YUYV_ER_BL);
#undef NVCV_ENUM
    }

    out << "NVCVImageFormat(" << fmt.colorModel() << ",";

    switch (fmt.colorModel())
    {
    case NVCV_COLOR_MODEL_RAW:
    {
        std::optional<NVCVRawPattern> raw = fmt.rawPattern();
        NVCV_ASSERT(raw);
        out << *raw << ",";
    }
    break;

    case NVCV_COLOR_MODEL_YCbCr:
        out << fmt.colorSpec() << "," << fmt.css() << ",";
        break;

    case NVCV_COLOR_MODEL_UNDEFINED:
        break;

    default:
        out << fmt.colorSpec() << ",";
        break;
    }

    out << fmt.memLayout() << "," << fmt.dataKind() << "," << fmt.swizzle();
    for (int i = 0; i < fmt.numPlanes(); ++i)
    {
        out << "," << fmt.planePacking(i);
    }
    out << ")";

    return out;
}

} // namespace nvcv::priv

std::string StrNVCVImageFormat(NVCVImageFormat fmt)
{
    nvcv::priv::ImageFormat pfmt{fmt};

    std::ostringstream ss;
    ss << pfmt;
    return ss.str();
}
