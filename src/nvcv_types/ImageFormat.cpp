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

#include "priv/ImageFormat.hpp"

#include "priv/DataType.hpp"
#include "priv/Exception.hpp"
#include "priv/Status.hpp"
#include "priv/SymbolVersioning.hpp"
#include "priv/TLS.hpp"

#include <util/String.hpp>

#include <cstring>

namespace priv = nvcv::priv;
namespace util = nvcv::util;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeColorImageFormat,
                (NVCVImageFormat * outFormat, NVCVColorModel colorModel, NVCVColorSpec colorSpec,
                 NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                 NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }
            priv::ImageFormat pout{colorModel, colorSpec, NVCV_CSS_NONE, memLayout, dataKind,
                                   swizzle,    packing0,  packing1,      packing2,  packing3};
            *outFormat = pout.value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeYCbCrImageFormat,
                (NVCVImageFormat * outFormat, NVCVColorSpec colorSpec, NVCVChromaSubsampling chromaSub,
                 NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle, NVCVPacking packing0,
                 NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            priv::ImageFormat pout{NVCV_COLOR_MODEL_YCbCr,
                                   colorSpec,
                                   chromaSub,
                                   memLayout,
                                   dataKind,
                                   swizzle,
                                   packing0,
                                   packing1,
                                   packing2,
                                   packing3};
            *outFormat = pout.value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeNonColorImageFormat,
                (NVCVImageFormat * outFormat, NVCVMemLayout memLayout, NVCVDataKind dataKind, NVCVSwizzle swizzle,
                 NVCVPacking packing0, NVCVPacking packing1, NVCVPacking packing2, NVCVPacking packing3))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            priv::ImageFormat pout{memLayout, dataKind, swizzle, packing0, packing1, packing2, packing3};
            *outFormat = pout.value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeRawImageFormat,
                (NVCVImageFormat * outFormat, NVCVRawPattern rawPattern, NVCVMemLayout memLayout, NVCVDataKind dataKind,
                 NVCVSwizzle swizzle, NVCVPacking packing0, NVCVPacking packing1, NVCVPacking packing2,
                 NVCVPacking packing3))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            priv::ImageFormat pout{rawPattern, memLayout, dataKind, swizzle, packing0, packing1, packing2, packing3};
            *outFormat = pout.value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlanePacking,
                (NVCVImageFormat fmt, int plane, NVCVPacking *outPacking))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPacking == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output packing cannot be NULL");
            }

            priv::ImageFormat pfmt{fmt};
            *outPacking = pfmt.planePacking(plane);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneBitsPerPixel,
                (NVCVImageFormat fmt, int32_t plane, int32_t *outBPP))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBPP == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to bits per pixel output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outBPP = pfmt.planeBPP(plane);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetSwizzleAndPacking,
                (NVCVImageFormat * fmt, NVCVSwizzle newSwizzle, NVCVPacking newPacking0, NVCVPacking newPacking1,
                 NVCVPacking newPacking2, NVCVPacking newPacking3))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }

            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.swizzleAndPacking(newSwizzle, newPacking0, newPacking1, newPacking2, newPacking3).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetDataKind, (NVCVImageFormat * fmt, NVCVDataKind newDataKind))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }
            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.dataKind(newDataKind).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetDataKind, (NVCVImageFormat fmt, NVCVDataKind *outDataKind))
{
    return priv::ProtectCall(
        [&]
        {
            if (outDataKind == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outDataKind = pfmt.dataKind();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetSwizzle, (NVCVImageFormat fmt, NVCVSwizzle *outSwizzle))
{
    return priv::ProtectCall(
        [&]
        {
            if (outSwizzle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to swizzle output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outSwizzle = pfmt.swizzle();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetMemLayout, (NVCVImageFormat * fmt, NVCVMemLayout newMemLayout))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }
            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.memLayout(newMemLayout).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetMemLayout, (NVCVImageFormat fmt, NVCVMemLayout *outMemLayout))
{
    return priv::ProtectCall(
        [&]
        {
            if (outMemLayout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to memory layout output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outMemLayout = pfmt.memLayout();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetColorSpec, (NVCVImageFormat * fmt, NVCVColorSpec newColorSpec))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }
            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.colorSpec(newColorSpec).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetColorSpec, (NVCVImageFormat fmt, NVCVColorSpec *outColorSpec))
{
    return priv::ProtectCall(
        [&]
        {
            if (outColorSpec == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to colorspec output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};

            *outColorSpec = pfmt.colorSpec();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetColorModel, (NVCVImageFormat fmt, NVCVColorModel *outColorModel))
{
    return priv::ProtectCall(
        [&]
        {
            if (outColorModel == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to color model output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outColorModel = pfmt.colorModel();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetChromaSubsampling,
                (NVCVImageFormat * fmt, NVCVChromaSubsampling newCSS))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }
            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.css(newCSS).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetChromaSubsampling,
                (NVCVImageFormat fmt, NVCVChromaSubsampling *outCSS))
{
    return priv::ProtectCall(
        [&]
        {
            if (outCSS == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to chroma subsampling output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outCSS = pfmt.css();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneNumChannels,
                (NVCVImageFormat fmt, int32_t plane, int32_t *outNumChannels))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumChannels == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of channels output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outNumChannels = pfmt.planeNumChannels(plane);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlanePixelStrideBytes,
                (NVCVImageFormat fmt, int32_t plane, int32_t *outStrideBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (outStrideBytes == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to pixel stride output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outStrideBytes = pfmt.planePixelStrideBytes(plane);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetBitsPerChannel, (NVCVImageFormat fmt, int32_t *outBits))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBits == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to bits per channel output cannot be NULL");
            }

            priv::ImageFormat pfmt{fmt};

            std::array<int32_t, 4> tmp = pfmt.bpc();
            static_assert(sizeof(tmp) == 4 * sizeof(*outBits));
            memcpy(outBits, &tmp, sizeof(tmp)); // No UB!
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetNumPlanes, (NVCVImageFormat fmt, int32_t *outNumPlanes))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumPlanes == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to number of planes output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outNumPlanes = pfmt.numPlanes();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetNumChannels, (NVCVImageFormat fmt, int32_t *outNumChannels))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumChannels == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of channels output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outNumChannels = pfmt.numChannels();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneDataType,
                (NVCVImageFormat fmt, int plane, NVCVDataType *outPixType))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPixType == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outPixType = pfmt.planeDataType(plane).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneSwizzle,
                (NVCVImageFormat fmt, int plane, NVCVSwizzle *outSwizzle))
{
    return priv::ProtectCall(
        [&]
        {
            if (outSwizzle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to swizzle output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outSwizzle = pfmt.planeSwizzle(plane);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeImageFormatFromPlanes,
                (NVCVImageFormat * outFormat, NVCVImageFormat plane0, NVCVImageFormat plane1, NVCVImageFormat plane2,
                 NVCVImageFormat plane3))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            util::StaticVector<priv::ImageFormat, 4> planes;
            if (plane0 != NVCV_IMAGE_FORMAT_NONE)
            {
                planes.emplace_back(plane0);
            }
            if (plane1 != NVCV_IMAGE_FORMAT_NONE)
            {
                planes.emplace_back(plane1);
            }
            if (plane2 != NVCV_IMAGE_FORMAT_NONE)
            {
                planes.emplace_back(plane2);
            }
            if (plane3 != NVCV_IMAGE_FORMAT_NONE)
            {
                planes.emplace_back(plane3);
            }

            *outFormat = priv::ImageFormat::FromPlanes(planes).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneFormat,
                (NVCVImageFormat fmt, int plane, NVCVImageFormat *outFormat))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to output image plane format cannot be NULL");
            }

            priv::ImageFormat pfmt{fmt};
            *outFormat = pfmt.planeFormat(plane).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetRawPattern, (NVCVImageFormat fmt, NVCVRawPattern *outPattern))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPattern == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to raw pattern outpu cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};

            if (std::optional<NVCVRawPattern> raw = pfmt.rawPattern())
            {
                *outPattern = *raw;
            }
            else
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Color model " << pfmt.colorModel() << " doesn't have a RAW pattern";
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatSetRawPattern, (NVCVImageFormat * fmt, NVCVRawPattern newRawPattern))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input image format cannot be NULL");
            }
            priv::ImageFormat pfmt{*fmt};
            *fmt = pfmt.rawPattern(newRawPattern).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatHasSameDataLayout,
                (NVCVImageFormat a, NVCVImageFormat b, int8_t *outBool))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBool == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to boolean output cannot be NULL");
            }
            priv::ImageFormat pfmtA{a}, pfmtB{b};
            *outBool = HasSameDataLayout(pfmtA, pfmtB) ? 1 : 0;
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeImageFormatFromFourCC,
                (NVCVImageFormat * outFormat, int32_t fourcc, NVCVColorSpec colorSpec, NVCVMemLayout memLayout))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFormat == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to output image plane format cannot be NULL");
            }
            *outFormat = priv::ImageFormat::FromFourCC(fourcc, colorSpec, memLayout).value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatToFourCC, (NVCVImageFormat fmt, uint32_t *outFourCC))
{
    return priv::ProtectCall(
        [&]
        {
            if (outFourCC == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to fourcc output cannot be NULL");
            }
            priv::ImageFormat pfmt{fmt};
            *outFourCC = pfmt.fourCC();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageFormatGetPlaneSize,
                (NVCVImageFormat fmt, int32_t plane, int32_t imgWidth, int32_t imgHeight, int32_t *outPlaneWidth,
                 int32_t *outPlaneHeight))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPlaneWidth == nullptr && outPlaneHeight == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to outPlaneWidth and outPlaneHeight cannot be both NULL");
            }
            priv::ImageFormat pfmt{fmt};
            priv::Size2D      sz = pfmt.planeSize({imgWidth, imgHeight}, plane);
            if (outPlaneWidth != nullptr)
            {
                *outPlaneWidth = sz.w;
            }
            if (outPlaneHeight != nullptr)
            {
                *outPlaneHeight = sz.h;
            }
        });
}

NVCV_DEFINE_API(0, 0, const char *, nvcvImageFormatGetName, (NVCVImageFormat fmt))
{
    priv::CoreTLS &tls = priv::GetCoreTLS(); // noexcept

    char         *buffer  = tls.bufImageFormatName;
    constexpr int bufSize = sizeof(tls.bufImageFormatName);

    try
    {
        util::BufferOStream(buffer, bufSize) << priv::ImageFormat{fmt};

        using namespace std::literals;

        util::ReplaceAllInline(buffer, bufSize, "NVCV_RAW_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_MODEL_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_SPEC_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_CSS_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_MEM_LAYOUT_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_DATA_KIND_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_PACKING_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_CHROMA_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_YCbCr_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_COLOR_"sv, ""sv);
    }
    catch (std::exception &e)
    {
        strncpy(buffer, e.what(), bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }
    catch (...)
    {
        strncpy(buffer, "Unexpected error retrieving NVCVImageFormat string representation", bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }

    return buffer;
}
