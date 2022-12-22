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

#include "ColorSpec.hpp"

#include "Bitfield.hpp"
#include "Exception.hpp"
#include "TLS.hpp"

#include <util/String.hpp>

#include <sstream>

namespace nvcv::priv {

ColorSpec::ColorSpec(NVCVColorSpace cspace, NVCVYCbCrEncoding encoding, NVCVColorTransferFunction xferfunc,
                     NVCVColorRange range, const ChromaLoc &loc) noexcept
    : m_cspec{NVCV_MAKE_COLOR_SPEC(cspace, encoding, xferfunc, range, loc.horiz, loc.vert)}
{
}

ColorSpec::operator NVCVColorSpec() const noexcept
{
    return m_cspec;
}

NVCVColorSpace ColorSpec::colorSpace() const noexcept
{
    int32_t val = ExtractBitfield(m_cspec, 0, 3);
    return (NVCVColorSpace)val;
}

ColorSpec ColorSpec::colorSpace(NVCVColorSpace newColorSpace) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(0, 3)) | SetBitfield(newColorSpace, 0, 3))};
}

ColorSpec ColorSpec::YCbCrEncoding(NVCVYCbCrEncoding newEncoding) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(7, 3)) | SetBitfield(newEncoding, 7, 3))};
}

NVCVYCbCrEncoding ColorSpec::YCbCrEncoding() const noexcept
{
    return (NVCVYCbCrEncoding)ExtractBitfield(m_cspec, 7, 3);
}

ColorSpec ColorSpec::xferFunc(NVCVColorTransferFunction newXferFunc) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(3, 4)) | SetBitfield(newXferFunc, 3, 4))};
}

NVCVColorTransferFunction ColorSpec::xferFunc() const noexcept
{
    return (NVCVColorTransferFunction)ExtractBitfield(m_cspec, 3, 4);
}

ColorSpec ColorSpec::colorRange(NVCVColorRange newRange) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(14, 1)) | SetBitfield(newRange, 14, 1))};
}

NVCVColorRange ColorSpec::colorRange() const noexcept
{
    return (NVCVColorRange)ExtractBitfield(m_cspec, 14, 1);
}

ChromaLoc ColorSpec::chromaLoc() const noexcept
{
    return {
        (NVCVChromaLocation)ExtractBitfield(m_cspec, 10, 2), // horiz
        (NVCVChromaLocation)ExtractBitfield(m_cspec, 12, 2)  // vert
    };
}

ColorSpec ColorSpec::chromaLoc(const ChromaLoc &loc) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(10, 4)) | SetBitfield(loc.horiz, 10, 2)
                                     | SetBitfield(loc.vert, 12, 2))};
}

NVCVWhitePoint ColorSpec::whitePoint() const noexcept
{
    // so far we only support D65...
    return NVCV_WHITE_POINT_D65;
}

NVCVChromaSubsampling MakeNVCVChromaSubsampling(int samplesHoriz, int samplesVert)
{
    switch (samplesHoriz)
    {
    case 4:
        switch (samplesVert)
        {
        case 4:
            return NVCV_CSS_444;
        case 2:
            return NVCV_CSS_422R;
        case 1:
            return NVCV_CSS_411R;
        }
        break;

    case 2:
        switch (samplesVert)
        {
        case 4:
            return NVCV_CSS_422;
        case 2:
            return NVCV_CSS_420;
        }
        break;

    case 1:
        switch (samplesVert)
        {
        case 1:
            return NVCV_CSS_444;
        case 4:
            return NVCV_CSS_411;
        }
        break;
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
        << samplesHoriz << " horizontal and " << samplesVert
        << " vertical samples doesn't correspond to any supported chroma subsampling scheme";
}

std::pair<int, int> GetChromaSamples(NVCVChromaSubsampling css)
{
    switch (css)
    {
    case NVCV_CSS_444:
        return {4, 4};

    case NVCV_CSS_422R:
        return {4, 2};

    case NVCV_CSS_411R:
        return {4, 1};

    case NVCV_CSS_422:
        return {2, 4};

    case NVCV_CSS_420:
        return {2, 2};

    case NVCV_CSS_411:
        return {1, 4};
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid chroma subsampling: " << css;
}

std::ostream &operator<<(std::ostream &out, ColorSpec cspec)
{
    switch (cspec)
    {
#define ENUM_CASE(X) \
    case X:          \
        return out << #X
        ENUM_CASE(NVCV_COLOR_SPEC_UNDEFINED);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_BT601);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_BT709);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_SMPTE240M);
        ENUM_CASE(NVCV_COLOR_SPEC_BT601);
        ENUM_CASE(NVCV_COLOR_SPEC_BT601_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709_LINEAR);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_LINEAR);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_PQ);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_PQ_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020c);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020c_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_SMPTE240M);
        ENUM_CASE(NVCV_COLOR_SPEC_sRGB);
        ENUM_CASE(NVCV_COLOR_SPEC_sYCC);
        ENUM_CASE(NVCV_COLOR_SPEC_DISPLAYP3);
        ENUM_CASE(NVCV_COLOR_SPEC_DISPLAYP3_LINEAR);
#undef ENUM_CASE
    case NVCV_COLOR_SPEC_FORCE32:
        out << "NVCVColorSpec(invalid)";
        break;
    }

    out << "NVCVColorSpec(" << cspec.colorSpace() << "," << cspec.YCbCrEncoding() << "," << cspec.xferFunc() << ","
        << cspec.colorRange() << "," << cspec.chromaLoc().horiz << "," << cspec.chromaLoc().vert << ")";
    return out;
}

bool NeedsColorspec(NVCVColorModel cmodel)
{
    switch (cmodel)
    {
    case NVCV_COLOR_MODEL_YCbCr:
    case NVCV_COLOR_MODEL_RGB:
        return true;
    case NVCV_COLOR_MODEL_UNDEFINED:
    case NVCV_COLOR_MODEL_RAW:
    case NVCV_COLOR_MODEL_XYZ:
    case NVCV_COLOR_MODEL_HSV:
        return false;
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid color model: " << cmodel;
}

const char *GetName(NVCVColorSpec cspec)
{
    priv::CoreTLS &tls = priv::GetCoreTLS();

    char         *buffer  = tls.bufColorSpecName;
    constexpr int bufSize = sizeof(tls.bufColorSpecName);

    try
    {
        util::BufferOStream(buffer, bufSize) << priv::ColorSpec{cspec};

        using namespace std::literals;

        util::ReplaceAllInline(buffer, bufSize, "NVCV_CHROMA_LOC_"sv, "LOC_"sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_YCbCr_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_XFER_"sv, "XFER_"sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_RANGE_"sv, "RANGE_"sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_SPACE_"sv, "SPACE_"sv);
    }
    catch (std::exception &e)
    {
        strncpy(buffer, e.what(), bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }
    catch (...)
    {
        strncpy(buffer, "Unexpected error retrieving NVCVColorSpec string representation", bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }

    return buffer;
}

const char *GetName(NVCVColorModel colorModel)
{
    switch (colorModel)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_COLOR_MODEL_UNDEFINED);
        ENUM_CASE(NVCV_COLOR_MODEL_RGB);
        ENUM_CASE(NVCV_COLOR_MODEL_YCbCr);
        ENUM_CASE(NVCV_COLOR_MODEL_RAW);
        ENUM_CASE(NVCV_COLOR_MODEL_XYZ);
        ENUM_CASE(NVCV_COLOR_MODEL_HSV);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();

    util::BufferOStream(tls.bufColorModelName, sizeof(tls.bufColorModelName))
        << "NVCVColorModel(" << (int)colorModel << ")";

    return tls.bufColorModelName;
}

const char *GetName(NVCVChromaLocation loc)
{
    switch (loc)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_CHROMA_LOC_EVEN);
        ENUM_CASE(NVCV_CHROMA_LOC_CENTER);
        ENUM_CASE(NVCV_CHROMA_LOC_ODD);
        ENUM_CASE(NVCV_CHROMA_LOC_BOTH);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();

    util::BufferOStream(tls.bufChromaLocationName, sizeof(tls.bufChromaLocationName))
        << "NVCVChromaLocation(" << (int)loc << ")";
    return tls.bufChromaLocationName;
}

const char *GetName(NVCVRawPattern raw)
{
    switch (raw)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_RAW_BAYER_RGGB);
        ENUM_CASE(NVCV_RAW_BAYER_BGGR);
        ENUM_CASE(NVCV_RAW_BAYER_GRBG);
        ENUM_CASE(NVCV_RAW_BAYER_GBRG);
        ENUM_CASE(NVCV_RAW_BAYER_RCCB);
        ENUM_CASE(NVCV_RAW_BAYER_BCCR);
        ENUM_CASE(NVCV_RAW_BAYER_CRBC);
        ENUM_CASE(NVCV_RAW_BAYER_CBRC);
        ENUM_CASE(NVCV_RAW_BAYER_RCCC);
        ENUM_CASE(NVCV_RAW_BAYER_CRCC);
        ENUM_CASE(NVCV_RAW_BAYER_CCRC);
        ENUM_CASE(NVCV_RAW_BAYER_CCCR);
        ENUM_CASE(NVCV_RAW_BAYER_CCCC);
#undef ENUM_CASE

    case NVCV_RAW_FORCE8:
        break;
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufRawPatternName, sizeof(tls.bufRawPatternName)) << "NVCVRawPattern(" << (int)raw << ")";

    return tls.bufRawPatternName;
}

const char *GetName(NVCVColorSpace color_space)
{
    switch (color_space)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_COLOR_SPACE_BT601);
        ENUM_CASE(NVCV_COLOR_SPACE_BT709);
        ENUM_CASE(NVCV_COLOR_SPACE_BT2020);
        ENUM_CASE(NVCV_COLOR_SPACE_DCIP3);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufColorSpaceName, sizeof(tls.bufColorSpaceName))
        << "NVCVColorSpace(" << (int)color_space << ")";
    return tls.bufColorSpaceName;
}

const char *GetName(NVCVWhitePoint whitePoint)
{
    switch (whitePoint)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_WHITE_POINT_D65);
#undef ENUM_CASE
    case NVCV_WHITE_POINT_FORCE8:
        break;
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufWhitePointName, sizeof(tls.bufWhitePointName))
        << "NVCVWhitePoint(" << (int)whitePoint << ")";
    return tls.bufWhitePointName;
}

const char *GetName(NVCVColorTransferFunction xferFunc)
{
    switch (xferFunc)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_COLOR_XFER_LINEAR);
        ENUM_CASE(NVCV_COLOR_XFER_sRGB);
        ENUM_CASE(NVCV_COLOR_XFER_sYCC);
        ENUM_CASE(NVCV_COLOR_XFER_PQ);
        ENUM_CASE(NVCV_COLOR_XFER_BT709);
        ENUM_CASE(NVCV_COLOR_XFER_BT2020);
        ENUM_CASE(NVCV_COLOR_XFER_SMPTE240M);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufColorTransferFunctionName, sizeof(tls.bufColorTransferFunctionName))
        << "NVCVColorTransferFunction(" << (int)xferFunc << ")";
    return tls.bufColorTransferFunctionName;
}

const char *GetName(NVCVColorRange range)
{
    switch (range)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_COLOR_RANGE_FULL);
        ENUM_CASE(NVCV_COLOR_RANGE_LIMITED);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufColorRangeName, sizeof(tls.bufColorRangeName)) << "NVCVColorRange(" << (int)range << ")";
    return tls.bufColorRangeName;
}

const char *GetName(NVCVYCbCrEncoding encoding)
{
    switch (encoding)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_YCbCr_ENC_UNDEFINED);
        ENUM_CASE(NVCV_YCbCr_ENC_BT601);
        ENUM_CASE(NVCV_YCbCr_ENC_BT709);
        ENUM_CASE(NVCV_YCbCr_ENC_BT2020);
        ENUM_CASE(NVCV_YCbCr_ENC_BT2020c);
        ENUM_CASE(NVCV_YCbCr_ENC_SMPTE240M);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufYCbCrEncodingName, sizeof(tls.bufYCbCrEncodingName))
        << "NVCVYCbCrEncoding(" << (int)encoding << ")";
    return tls.bufYCbCrEncodingName;
}

const char *GetName(NVCVChromaSubsampling chromaSub)
{
    priv::CoreTLS &tls = priv::GetCoreTLS();

    {
        util::BufferOStream ss(tls.bufChromaSubsamplingName, sizeof(tls.bufChromaSubsamplingName));

        bool ok = false;

        switch (chromaSub)
        {
#define ENUM_CASE_CSS(J, a, b, R)                 \
    case NVCV_CSS_##J##a##b##R:                   \
        ss << J << ':' << a << ':' << b << "" #R; \
        ok = true;                                \
        break
            ENUM_CASE_CSS(4, 4, 4, );
            ENUM_CASE_CSS(4, 2, 2, );
            ENUM_CASE_CSS(4, 2, 2, R);
            ENUM_CASE_CSS(4, 1, 1, );
            ENUM_CASE_CSS(4, 1, 1, R);
            ENUM_CASE_CSS(4, 2, 0, );
#undef ENUM_CASE_CSS
        }

        if (!ok)
        {
            ss << "NVCVChromaSubsampling(" << (int)chromaSub << ")";
        }
    }
    return tls.bufChromaSubsamplingName;
}

} // namespace nvcv::priv

namespace priv = nvcv::priv;

std::ostream &operator<<(std::ostream &out, NVCVColorModel colorModel)
{
    return out << priv::GetName(colorModel);
}

std::ostream &operator<<(std::ostream &out, NVCVChromaLocation loc)
{
    return out << priv::GetName(loc);
}

std::ostream &operator<<(std::ostream &out, NVCVRawPattern raw)
{
    return out << priv::GetName(raw);
}

std::ostream &operator<<(std::ostream &out, NVCVColorSpace color_space)
{
    return out << priv::GetName(color_space);
}

std::ostream &operator<<(std::ostream &out, NVCVWhitePoint whitePoint)
{
    return out << priv::GetName(whitePoint);
}

std::ostream &operator<<(std::ostream &out, NVCVColorTransferFunction xferFunc)
{
    return out << priv::GetName(xferFunc);
}

std::ostream &operator<<(std::ostream &out, NVCVColorRange range)
{
    return out << priv::GetName(range);
}

std::ostream &operator<<(std::ostream &out, NVCVYCbCrEncoding encoding)
{
    return out << priv::GetName(encoding);
}

std::ostream &operator<<(std::ostream &out, NVCVChromaSubsampling chromaSub)
{
    return out << priv::GetName(chromaSub);
}

std::ostream &operator<<(std::ostream &out, NVCVColorSpec cspec)
{
    return out << priv::GetName(cspec);
}

std::string StrNVCVColorSpec(NVCVColorSpec cspec)
{
    std::ostringstream ss;
    ss << nvcv::priv::ColorSpec{cspec};
    return ss.str();
}
