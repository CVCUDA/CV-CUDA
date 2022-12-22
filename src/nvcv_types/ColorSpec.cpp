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

#include "priv/ColorSpec.hpp"

#include "priv/Exception.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"
#include "priv/TLS.hpp"

#include <nvcv/ColorSpec.h>
#include <util/String.hpp>

#include <cstring>

namespace priv = nvcv::priv;
namespace util = nvcv::util;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeColorSpec,
                (NVCVColorSpec * outColorSpec, NVCVColorSpace cspace, NVCVYCbCrEncoding encoding,
                 NVCVColorTransferFunction xferfunc, NVCVColorRange range, NVCVChromaLocation locHoriz,
                 NVCVChromaLocation locVert))
{
    return priv::ProtectCall(
        [&]
        {
            if (outColorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{
                cspace,
                encoding,
                xferfunc,
                range,
                {locHoriz, locVert}
            };

            *outColorSpec = pcspec;
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeChromaSubsampling,
                (NVCVChromaSubsampling * outCSS, int samplesHoriz, int samplesVert))
{
    return priv::ProtectCall(
        [&]
        {
            if (outCSS == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to output chroma subsampling cannot be NULL");
            }

            *outCSS = priv::MakeNVCVChromaSubsampling(samplesHoriz, samplesVert);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvChromaSubsamplingGetNumSamples,
                (NVCVChromaSubsampling css, int32_t *outSamplesHoriz, int32_t *outSamplesVert))
{
    return priv::ProtectCall(
        [&]
        {
            if (outSamplesHoriz == nullptr && outSamplesVert == nullptr)
            {
                throw priv::Exception(
                    NVCV_ERROR_INVALID_ARGUMENT,
                    "Pointer to output number of horizontal and horizontal samples cannot both be NULL");
            }

            std::pair<int, int> nsamp = priv::GetChromaSamples(css);

            if (outSamplesHoriz != nullptr)
            {
                *outSamplesHoriz = nsamp.first;
            }
            if (outSamplesVert != nullptr)
            {
                *outSamplesVert = nsamp.second;
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecSetRange, (NVCVColorSpec * colorSpec, NVCVColorRange range))
{
    return priv::ProtectCall(
        [&]
        {
            if (colorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{*colorSpec};
            *colorSpec = pcspec.colorRange(range);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecGetRange, (NVCVColorSpec colorSpec, NVCVColorRange *outColorRange))
{
    return priv::ProtectCall(
        [&]
        {
            if (outColorRange == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output color range cannot be NULL");
            }

            priv::ColorSpec pcspec{colorSpec};
            *outColorRange = pcspec.colorRange();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecSetColorSpace, (NVCVColorSpec * colorSpec, NVCVColorSpace cspace))
{
    return priv::ProtectCall(
        [&]
        {
            if (colorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{*colorSpec};
            *colorSpec = pcspec.colorSpace(cspace);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecGetColorSpace, (NVCVColorSpec colorSpec, NVCVColorSpace *outColorSpace))
{
    return priv::ProtectCall(
        [&]
        {
            if (outColorSpace == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output color space cannot be NULL");
            }

            priv::ColorSpec pcspec{colorSpec};
            *outColorSpace = pcspec.colorSpace();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecGetYCbCrEncoding,
                (NVCVColorSpec colorSpec, NVCVYCbCrEncoding *outEncoding))
{
    return priv::ProtectCall(
        [&]
        {
            if (outEncoding == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output YCbCr encoding cannot be NULL");
            }

            priv::ColorSpec pcspec{colorSpec};
            *outEncoding = pcspec.YCbCrEncoding();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecSetYCbCrEncoding,
                (NVCVColorSpec * colorSpec, NVCVYCbCrEncoding encoding))
{
    return priv::ProtectCall(
        [&]
        {
            if (colorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{*colorSpec};
            *colorSpec = pcspec.YCbCrEncoding(encoding);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecSetColorTransferFunction,
                (NVCVColorSpec * colorSpec, NVCVColorTransferFunction xferFunc))
{
    return priv::ProtectCall(
        [&]
        {
            if (colorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{*colorSpec};
            *colorSpec = pcspec.xferFunc(xferFunc);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecGetColorTransferFunction,
                (NVCVColorSpec colorSpec, NVCVColorTransferFunction *outXferFunc))
{
    return priv::ProtectCall(
        [&]
        {
            if (outXferFunc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to output color transfer function cannot be NULL");
            }

            priv::ColorSpec pcspec{colorSpec};
            *outXferFunc = pcspec.xferFunc();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecGetChromaLoc,
                (NVCVColorSpec colorSpec, NVCVChromaLocation *outLocHoriz, NVCVChromaLocation *outLocVert))
{
    return priv::ProtectCall(
        [&]
        {
            if (outLocHoriz == nullptr && outLocVert == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to output horizontal and vertical chroma location cannot both be NULL");
            }

            priv::ColorSpec pcspec{colorSpec};

            priv::ChromaLoc loc = pcspec.chromaLoc();

            if (outLocHoriz != nullptr)
            {
                *outLocHoriz = loc.horiz;
            }
            if (outLocVert != nullptr)
            {
                *outLocVert = loc.vert;
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorSpecSetChromaLoc,
                (NVCVColorSpec * colorSpec, NVCVChromaLocation locHoriz, NVCVChromaLocation locVert))
{
    return priv::ProtectCall(
        [&]
        {
            if (colorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input colorspec cannot be NULL");
            }

            priv::ColorSpec pcspec{*colorSpec};
            *colorSpec = pcspec.chromaLoc({locHoriz, locVert});
        });
}

NVCV_DEFINE_API(0, 0, const char *, nvcvColorSpecGetName, (NVCVColorSpec cspec))
{
    priv::CoreTLS &tls = priv::GetCoreTLS();

    char         *buffer  = tls.bufColorSpecName;
    constexpr int bufSize = sizeof(tls.bufColorSpecName);

    try
    {
        priv::ColorSpec pcspec{cspec};

        // Must insert EOS to make 'str' a correctly delimited string
        util::BufferOStream(buffer, bufSize) << pcspec;

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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvColorModelNeedsColorspec, (NVCVColorModel cmodel, int8_t *outBool))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBool == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output boolean value cannot be NULL");
            }

            *outBool = priv::NeedsColorspec(cmodel) ? 1 : 0;
        });
}

NVCV_DEFINE_API(0, 2, const char *, nvcvColorModelGetName, (NVCVColorModel cmodel))
{
    return priv::GetName(cmodel);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvColorSpaceGetName, (NVCVColorSpace cspace))
{
    return priv::GetName(cspace);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvWhitePointGetName, (NVCVWhitePoint wpoint))
{
    return priv::GetName(wpoint);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvYCbCrEncodingGetName, (NVCVYCbCrEncoding enc))
{
    return priv::GetName(enc);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvColorTransferFunctionGetName, (NVCVColorTransferFunction xfer))
{
    return priv::GetName(xfer);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvColorRangeGetName, (NVCVColorRange crange))
{
    return priv::GetName(crange);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvChromaLocationGetName, (NVCVChromaLocation loc))
{
    return priv::GetName(loc);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvRawPatternGetName, (NVCVRawPattern raw))
{
    return priv::GetName(raw);
}

NVCV_DEFINE_API(0, 2, const char *, nvcvChromaSubsamplingGetName, (NVCVChromaSubsampling css))
{
    return priv::GetName(css);
}
