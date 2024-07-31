/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file ColorSpec.hpp
 *
 * @brief Defines C++ types and functions to handle color specs.
 * @defgroup NVCV_CPP_CORE_COLORSPEC Color Models
 * @{
 */

#ifndef NVCV_COLORSPEC_HPP
#define NVCV_COLORSPEC_HPP

#include "ColorSpec.h"

#include <nvcv/detail/CheckError.hpp>

#include <iostream>

namespace nvcv {

enum class ColorModel : int8_t
{
    UNDEFINED = NVCV_COLOR_MODEL_UNDEFINED,
    YCbCr     = NVCV_COLOR_MODEL_YCbCr,
    RGB       = NVCV_COLOR_MODEL_RGB,
    RAW       = NVCV_COLOR_MODEL_RAW,
    XYZ       = NVCV_COLOR_MODEL_XYZ
};

enum class ColorSpace : int8_t
{
    BT601  = NVCV_COLOR_SPACE_BT601,
    BT709  = NVCV_COLOR_SPACE_BT709,
    BT2020 = NVCV_COLOR_SPACE_BT2020,
    DCIP3  = NVCV_COLOR_SPACE_DCIP3,
};

enum class WhitePoint : int8_t
{
    D65 = NVCV_WHITE_POINT_D65
};

enum class YCbCrEncoding : int8_t
{
    UNDEFINED = NVCV_YCbCr_ENC_UNDEFINED,
    BT601     = NVCV_YCbCr_ENC_BT601,
    BT709     = NVCV_YCbCr_ENC_BT709,
    BT2020    = NVCV_YCbCr_ENC_BT2020,
    BT2020c   = NVCV_YCbCr_ENC_BT2020c,
    SMPTE240M = NVCV_YCbCr_ENC_SMPTE240M,
};

enum class ColorTransferFunction : int8_t
{
    LINEAR    = NVCV_COLOR_XFER_LINEAR,
    sRGB      = NVCV_COLOR_XFER_sRGB,
    sYCC      = NVCV_COLOR_XFER_sYCC,
    PQ        = NVCV_COLOR_XFER_PQ,
    BT709     = NVCV_COLOR_XFER_BT709,
    BT2020    = NVCV_COLOR_XFER_BT2020,
    SMPTE240M = NVCV_COLOR_XFER_SMPTE240M,
};

enum class ColorRange : int8_t
{
    FULL    = NVCV_COLOR_RANGE_FULL,
    LIMITED = NVCV_COLOR_RANGE_LIMITED,
};

enum class ChromaLocation : int8_t
{
    EVEN   = NVCV_CHROMA_LOC_EVEN,
    CENTER = NVCV_CHROMA_LOC_CENTER,
    ODD    = NVCV_CHROMA_LOC_ODD,
    BOTH   = NVCV_CHROMA_LOC_BOTH,
};

enum class RawPattern : uint8_t
{
    BAYER_RGGB = NVCV_RAW_BAYER_RGGB,
    BAYER_BGGR = NVCV_RAW_BAYER_BGGR,
    BAYER_GRBG = NVCV_RAW_BAYER_GRBG,
    BAYER_GBRG = NVCV_RAW_BAYER_GBRG,
    BAYER_RCCB = NVCV_RAW_BAYER_RCCB,
    BAYER_BCCR = NVCV_RAW_BAYER_BCCR,
    BAYER_CRBC = NVCV_RAW_BAYER_CRBC,
    BAYER_CBRC = NVCV_RAW_BAYER_CBRC,
    BAYER_RCCC = NVCV_RAW_BAYER_RCCC,
    BAYER_CRCC = NVCV_RAW_BAYER_CRCC,
    BAYER_CCRC = NVCV_RAW_BAYER_CCRC,
    BAYER_CCCR = NVCV_RAW_BAYER_CCCR,
    BAYER_CCCC = NVCV_RAW_BAYER_CCCC,
};

enum class ChromaSubsampling : int8_t
{
    NONE     = NVCV_CSS_NONE,
    CSS_444  = NVCV_CSS_444,
    CSS_422  = NVCV_CSS_422,
    CSS_422R = NVCV_CSS_422R,
    CSS_411  = NVCV_CSS_411,
    CSS_411R = NVCV_CSS_411R,
    CSS_420  = NVCV_CSS_420,
    CSS_440  = NVCV_CSS_440,
    CSS_410  = NVCV_CSS_410,
    CSS_410R = NVCV_CSS_410R
};

/**
 * @class ColorSpec
 * @brief Class for color specification.
 *
 * This class encapsulates various properties related to color space and encoding.
 */
class ColorSpec
{
public:
    /**
     * @brief Construct a new ColorSpec object.
     * @param cspec Existing NVCVColorSpec object.
     */
    constexpr ColorSpec(NVCVColorSpec cspec)
        : m_cspec(cspec)
    {
    }

    /**
     * @brief Create a ColorSpec object.
     *
     * @param cspace Color space.
     * @param encoding YCbCr encoding.
     * @param xferFunc Color transfer function.
     * @param range Color range.
     * @param locHoriz Horizontal chroma location.
     * @param locVert Vertical chroma location.
     * @return A ColorSpec object.
     */
    constexpr static ColorSpec ConstCreate(ColorSpace cspace, YCbCrEncoding encoding, ColorTransferFunction xferFunc,
                                           ColorRange range, ChromaLocation locHoriz, ChromaLocation locVert);

    /**
     * @brief Construct a new ColorSpec object.
     *
     * @param cspace Color space.
     * @param encoding YCbCr encoding.
     * @param xferFunc Color transfer function.
     * @param range Color range.
     * @param locHoriz Horizontal chroma location.
     * @param locVert Vertical chroma location.
     */
    ColorSpec(ColorSpace cspace, YCbCrEncoding encoding, ColorTransferFunction xferFunc, ColorRange range,
              ChromaLocation locHoriz, ChromaLocation locVert);

    /**
     * @brief Get the NVCVColorSpec object.
     *
     * @return NVCVColorSpec object.
     */
    constexpr operator NVCVColorSpec() const;

    /**
     * @brief Set the chroma location and return a new ColorSpec.
     *
     * @param locHoriz Horizontal chroma location.
     * @param locVert Vertical chroma location.
     * @return A new ColorSpec with the specified chroma location.
     */
    ColorSpec chromaLoc(ChromaLocation locHoriz, ChromaLocation locVert) const;

    /**
     * @brief Get the horizontal chroma location.
     *
     * @return Horizontal chroma location.
     */
    ChromaLocation chromaLocHoriz() const;

    /**
     * @brief Get the vertical chroma location.
     *
     * @return Vertical chroma location.
     */
    ChromaLocation chromaLocVert() const;

    /**
     * @brief Set the color space and return a new ColorSpec.
     *
     * @param cspace Color space.
     * @return A new ColorSpec with the specified color space.
     */
    ColorSpec colorSpace(ColorSpace cspace) const;

    /**
     * @brief Get the color space.
     *
     * @return Color space.
     */
    ColorSpace colorSpace() const;

    /**
     * @brief Set the YCbCr encoding and return a new ColorSpec.
     *
     * @param encoding YCbCr encoding.
     * @return A new ColorSpec with the specified YCbCr encoding.
     */
    ColorSpec yCbCrEncoding(YCbCrEncoding encoding) const;

    /**
     * @brief Get the YCbCr encoding.
     *
     * @return YCbCr encoding.
     */
    YCbCrEncoding yCbCrEncoding() const;

    /**
     * @brief Set the color transfer function and return a new ColorSpec.
     *
     * @param xferFunc Color transfer function.
     * @return A new ColorSpec with the specified color transfer function.
     */
    ColorSpec colorTransferFunction(ColorTransferFunction xferFunc) const;

    /**
     * @brief Get the color transfer function.
     *
     * @return Color transfer function.
     */
    ColorTransferFunction colorTransferFunction() const;

    /**
     * @brief Set the color range and return a new ColorSpec.
     *
     * @param range Color range.
     * @return A new ColorSpec with the specified color range.
     */
    ColorSpec colorRange(ColorRange range) const;

    /**
     * @brief Get the color range.
     *
     * @return Color range.
     */
    ColorRange colorRange() const;

private:
    NVCVColorSpec m_cspec;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
constexpr ColorSpec CSPEC_UNDEFINED        = NVCV_COLOR_SPEC_UNDEFINED;
constexpr ColorSpec CSPEC_BT601            = NVCV_COLOR_SPEC_BT601;
constexpr ColorSpec CSPEC_BT601_ER         = NVCV_COLOR_SPEC_BT601_ER;
constexpr ColorSpec CSPEC_BT709            = NVCV_COLOR_SPEC_BT709;
constexpr ColorSpec CSPEC_BT709_ER         = NVCV_COLOR_SPEC_BT709_ER;
constexpr ColorSpec CSPEC_BT709_LINEAR     = NVCV_COLOR_SPEC_BT709_LINEAR;
constexpr ColorSpec CSPEC_BT2020           = NVCV_COLOR_SPEC_BT2020;
constexpr ColorSpec CSPEC_BT2020_ER        = NVCV_COLOR_SPEC_BT2020_ER;
constexpr ColorSpec CSPEC_BT2020_LINEAR    = NVCV_COLOR_SPEC_BT2020_LINEAR;
constexpr ColorSpec CSPEC_BT2020_PQ        = NVCV_COLOR_SPEC_BT2020_PQ;
constexpr ColorSpec CSPEC_BT2020_PQ_ER     = NVCV_COLOR_SPEC_BT2020_PQ_ER;
constexpr ColorSpec CSPEC_BT2020c_ER       = NVCV_COLOR_SPEC_BT2020c_ER;
constexpr ColorSpec CSPEC_MPEG2_BT601      = NVCV_COLOR_SPEC_MPEG2_BT601;
constexpr ColorSpec CSPEC_MPEG2_BT709      = NVCV_COLOR_SPEC_MPEG2_BT709;
constexpr ColorSpec CSPEC_MPEG2_SMPTE240M  = NVCV_COLOR_SPEC_MPEG2_SMPTE240M;
constexpr ColorSpec CSPEC_sRGB             = NVCV_COLOR_SPEC_sRGB;
constexpr ColorSpec CSPEC_sYCC             = NVCV_COLOR_SPEC_sYCC;
constexpr ColorSpec CSPEC_SMPTE240M        = NVCV_COLOR_SPEC_SMPTE240M;
constexpr ColorSpec CSPEC_DISPLAYP3        = NVCV_COLOR_SPEC_DISPLAYP3;
constexpr ColorSpec CSPEC_DISPLAYP3_LINEAR = NVCV_COLOR_SPEC_DISPLAYP3_LINEAR;
#endif

constexpr ColorSpec ColorSpec::ConstCreate(ColorSpace cspace, YCbCrEncoding encoding, ColorTransferFunction xferFunc,
                                           ColorRange range, ChromaLocation locHoriz, ChromaLocation locVert)
{
    return ColorSpec{NVCV_MAKE_COLOR_SPEC(static_cast<NVCVColorSpace>(cspace), static_cast<NVCVYCbCrEncoding>(encoding),
                                          static_cast<NVCVColorTransferFunction>(xferFunc),
                                          static_cast<NVCVColorRange>(range), static_cast<NVCVChromaLocation>(locHoriz),
                                          static_cast<NVCVChromaLocation>(locVert))};
}

inline ColorSpec::ColorSpec(ColorSpace cspace, YCbCrEncoding encoding, ColorTransferFunction xferFunc, ColorRange range,
                            ChromaLocation locHoriz, ChromaLocation locVert)
{
    detail::CheckThrow(
        nvcvMakeColorSpec(&m_cspec, static_cast<NVCVColorSpace>(cspace), static_cast<NVCVYCbCrEncoding>(encoding),
                          static_cast<NVCVColorTransferFunction>(xferFunc), static_cast<NVCVColorRange>(range),
                          static_cast<NVCVChromaLocation>(locHoriz), static_cast<NVCVChromaLocation>(locVert)));
}

constexpr ColorSpec::operator NVCVColorSpec() const
{
    return m_cspec;
}

inline ChromaLocation ColorSpec::chromaLocHoriz() const
{
    NVCVChromaLocation outH, outV;
    detail::CheckThrow(nvcvColorSpecGetChromaLoc(m_cspec, &outH, &outV));
    return static_cast<ChromaLocation>(outH);
}

inline ChromaLocation ColorSpec::chromaLocVert() const
{
    NVCVChromaLocation outH, outV;
    detail::CheckThrow(nvcvColorSpecGetChromaLoc(m_cspec, &outH, &outV));
    return static_cast<ChromaLocation>(outV);
}

inline ColorSpec ColorSpec::chromaLoc(ChromaLocation locHoriz, ChromaLocation locVert) const
{
    NVCVColorSpec out = m_cspec;
    detail::CheckThrow(nvcvColorSpecSetChromaLoc(&out, static_cast<NVCVChromaLocation>(locHoriz),
                                                 static_cast<NVCVChromaLocation>(locVert)));
    return ColorSpec{out};
}

inline ColorSpace ColorSpec::colorSpace() const
{
    NVCVColorSpace out;
    detail::CheckThrow(nvcvColorSpecGetColorSpace(m_cspec, &out));
    return static_cast<ColorSpace>(out);
}

inline ColorSpec ColorSpec::colorSpace(ColorSpace cspace) const
{
    NVCVColorSpec out = m_cspec;
    detail::CheckThrow(nvcvColorSpecSetColorSpace(&out, static_cast<NVCVColorSpace>(cspace)));
    return ColorSpec{out};
}

inline YCbCrEncoding ColorSpec::yCbCrEncoding() const
{
    NVCVYCbCrEncoding out;
    detail::CheckThrow(nvcvColorSpecGetYCbCrEncoding(m_cspec, &out));
    return static_cast<YCbCrEncoding>(out);
}

inline ColorSpec ColorSpec::yCbCrEncoding(YCbCrEncoding encoding) const
{
    NVCVColorSpec out = m_cspec;
    detail::CheckThrow(nvcvColorSpecSetYCbCrEncoding(&out, static_cast<NVCVYCbCrEncoding>(encoding)));
    return ColorSpec{out};
}

inline ColorTransferFunction ColorSpec::colorTransferFunction() const
{
    NVCVColorTransferFunction out;
    detail::CheckThrow(nvcvColorSpecGetColorTransferFunction(m_cspec, &out));
    return static_cast<ColorTransferFunction>(out);
}

inline ColorSpec ColorSpec::colorTransferFunction(ColorTransferFunction xferFunc) const
{
    NVCVColorSpec out = m_cspec;
    detail::CheckThrow(nvcvColorSpecSetColorTransferFunction(&out, static_cast<NVCVColorTransferFunction>(xferFunc)));
    return ColorSpec{out};
}

inline ColorRange ColorSpec::colorRange() const
{
    NVCVColorRange out;
    detail::CheckThrow(nvcvColorSpecGetRange(m_cspec, &out));
    return static_cast<ColorRange>(out);
}

inline ColorSpec ColorSpec::colorRange(ColorRange range) const
{
    NVCVColorSpec out = m_cspec;
    detail::CheckThrow(nvcvColorSpecSetRange(&out, static_cast<NVCVColorRange>(range)));
    return ColorSpec{out};
}

inline ChromaSubsampling MakeChromaSubsampling(int samplesHoriz, int samplesVert)
{
    NVCVChromaSubsampling out;
    detail::CheckThrow(nvcvMakeChromaSubsampling(&out, samplesHoriz, samplesVert));
    return static_cast<ChromaSubsampling>(out);
}

inline int GetSamplesHoriz(ChromaSubsampling css)
{
    int32_t outH;
    detail::CheckThrow(nvcvChromaSubsamplingGetNumSamples(static_cast<NVCVChromaSubsampling>(css), &outH, nullptr));
    return outH;
}

inline int GetSamplesVert(ChromaSubsampling css)
{
    int32_t outV;
    detail::CheckThrow(nvcvChromaSubsamplingGetNumSamples(static_cast<NVCVChromaSubsampling>(css), nullptr, &outV));
    return outV;
}

inline bool NeedsColorspec(ColorModel cmodel)
{
    int8_t out;
    detail::CheckThrow(nvcvColorModelNeedsColorspec(static_cast<NVCVColorModel>(cmodel), &out));
    return out != 0;
}

inline std::ostream &operator<<(std::ostream &out, ColorModel colorModel)
{
    return out << nvcvColorModelGetName(static_cast<NVCVColorModel>(colorModel));
}

inline std::ostream &operator<<(std::ostream &out, ColorSpec cspec)
{
    return out << nvcvColorSpecGetName(static_cast<NVCVColorSpec>(cspec));
}

inline std::ostream &operator<<(std::ostream &out, ChromaSubsampling chromaSub)
{
    return out << nvcvChromaSubsamplingGetName(static_cast<NVCVChromaSubsampling>(chromaSub));
}

inline std::ostream &operator<<(std::ostream &out, ColorTransferFunction xferFunc)
{
    return out << nvcvColorTransferFunctionGetName(static_cast<NVCVColorTransferFunction>(xferFunc));
}

inline std::ostream &operator<<(std::ostream &out, YCbCrEncoding enc)
{
    return out << nvcvYCbCrEncodingGetName(static_cast<NVCVYCbCrEncoding>(enc));
}

inline std::ostream &operator<<(std::ostream &out, ColorRange range)
{
    return out << nvcvColorRangeGetName(static_cast<NVCVColorRange>(range));
}

inline std::ostream &operator<<(std::ostream &out, WhitePoint whitePoint)
{
    return out << nvcvWhitePointGetName(static_cast<NVCVWhitePoint>(whitePoint));
}

inline std::ostream &operator<<(std::ostream &out, ColorSpace color_space)
{
    return out << nvcvColorSpecGetName(static_cast<NVCVColorSpec>(color_space));
}

inline std::ostream &operator<<(std::ostream &out, ChromaLocation loc)
{
    return out << nvcvChromaLocationGetName(static_cast<NVCVChromaLocation>(loc));
}

inline std::ostream &operator<<(std::ostream &out, RawPattern raw)
{
    return out << nvcvRawPatternGetName(static_cast<NVCVRawPattern>(raw));
}

} // namespace nvcv

inline std::ostream &operator<<(std::ostream &out, NVCVColorModel colorModel)
{
    return out << nvcvColorModelGetName(colorModel);
}

inline std::ostream &operator<<(std::ostream &out, NVCVColorSpec colorSpec)
{
    return out << nvcvColorSpecGetName(colorSpec);
}

inline std::ostream &operator<<(std::ostream &out, NVCVChromaSubsampling chromaSub)
{
    return out << nvcvChromaSubsamplingGetName(chromaSub);
}

inline std::ostream &operator<<(std::ostream &out, NVCVColorTransferFunction xferFunc)
{
    return out << nvcvColorTransferFunctionGetName(xferFunc);
}

inline std::ostream &operator<<(std::ostream &out, NVCVYCbCrEncoding enc)
{
    return out << nvcvYCbCrEncodingGetName(enc);
}

inline std::ostream &operator<<(std::ostream &out, NVCVColorRange range)
{
    return out << nvcvColorRangeGetName(range);
}

inline std::ostream &operator<<(std::ostream &out, NVCVWhitePoint whitePoint)
{
    return out << nvcvWhitePointGetName(whitePoint);
}

inline std::ostream &operator<<(std::ostream &out, NVCVColorSpace color_space)
{
    return out << nvcvColorSpaceGetName(color_space);
}

inline std::ostream &operator<<(std::ostream &out, NVCVChromaLocation loc)
{
    return out << nvcvChromaLocationGetName(loc);
}

inline std::ostream &operator<<(std::ostream &out, NVCVRawPattern raw)
{
    return out << nvcvRawPatternGetName(raw);
}

/**@}*/

#endif // NVCV_COLORSPEC_HPP
