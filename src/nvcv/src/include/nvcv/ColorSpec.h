/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file ColorSpec.h
 *
 * @brief Defines types and functions to handle color specs.
 */

#ifndef NVCV_FORMAT_COLORSPEC_H
#define NVCV_FORMAT_COLORSPEC_H

#include "detail/FormatUtils.h"

#include <nvcv/Export.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Defines color models.
 * A color model gives meaning to each channel of an image format. They are specified
 * in a canonical XYZW ordering that can then be swizzled to the desired ordering.
 * @defgroup NVCV_C_CORE_COLORSPEC Color Models
 * @{
 */
typedef enum
{
    NVCV_COLOR_MODEL_UNDEFINED = 0,     /**< Color model is undefined. */
    NVCV_COLOR_MODEL_YCbCr     = 1,     /**< Luma + chroma (blue-luma, red-luma). */
    NVCV_COLOR_MODEL_RGB       = 2,     /**< red, green, blue components. */
    NVCV_COLOR_MODEL_RAW       = 2 + 7, /**< RAW color model, used for Bayer image formats. */
    NVCV_COLOR_MODEL_XYZ,               /**< CIE XYZ tristimulus color spec. */
    NVCV_COLOR_MODEL_HSV,               /**< hue, saturation, value components. */
    NVCV_COLOR_MODEL_CMYK,              /**< cyan, magenta, yellow, black components. */
    NVCV_COLOR_MODEL_YCCK               /**< Luma + chroma (blue-luma, red-luma) and black components. */
} NVCVColorModel;

/** Defines the color primaries and the white point of a \ref NVCVColorSpec. */
typedef enum
{
    NVCV_COLOR_SPACE_BT601,  /**< Color primaries from ITU-R BT.601/625 lines standard, also known as EBU 3213-E. */
    NVCV_COLOR_SPACE_BT709,  /**< Color primaries from ITU-R BT.709 standard, D65 white point. */
    NVCV_COLOR_SPACE_BT2020, /**< Color primaries from ITU-R BT.2020 standard, D65 white point. */
    NVCV_COLOR_SPACE_DCIP3,  /**< Color primaries from DCI-P3 standard, D65 white point. */
} NVCVColorSpace;

/** Defines the white point associated with a \ref NVCVColorSpace. */
typedef enum
{
    NVCV_WHITE_POINT_D65, /**< D65 white point, K = 6504. */

    /** \cond Do not use. */
    NVCV_WHITE_POINT_FORCE8 = UINT8_MAX
    /* \endcond */
} NVCVWhitePoint;

/** Defines the YCbCr encoding used in a particular \ref NVCVColorSpec. */
typedef enum
{
    NVCV_YCbCr_ENC_UNDEFINED = 0, /**< Encoding not defined. Usually used by non-YCbCr color specs. */
    NVCV_YCbCr_ENC_BT601,         /**< Encoding specified by ITU-R BT.601 standard. */
    NVCV_YCbCr_ENC_BT709,         /**< Encoding specified by ITU-R BT.709 standard. */
    NVCV_YCbCr_ENC_BT2020,        /**< Encoding specified by ITU-R BT.2020 standard. */
    NVCV_YCbCr_ENC_BT2020c,       /**< Encoding specified by ITU-R BT.2020 with constant luminance. */
    NVCV_YCbCr_ENC_SMPTE240M,     /**< Encoding specified by SMPTE 240M standard. */
} NVCVYCbCrEncoding;

/** Defines the color transfer function in a particular \ref NVCVColorSpec. */
typedef enum
{
    NVCV_COLOR_XFER_LINEAR,    /**< Linear color transfer function. */
    NVCV_COLOR_XFER_sRGB,      /**< Color transfer function specified by sRGB standard. */
    NVCV_COLOR_XFER_sYCC,      /**< Color transfer function specified by sYCC standard. */
    NVCV_COLOR_XFER_PQ,        /**< Perceptual quantizer color transfer function. */
    NVCV_COLOR_XFER_BT709,     /**< Color transfer function specified by ITU-R BT.709 standard. */
    NVCV_COLOR_XFER_BT2020,    /**< Color transfer function specified by ITU-R BT.2020 standard. */
    NVCV_COLOR_XFER_SMPTE240M, /**< Color transfer function specified by SMPTE 240M standard. */
} NVCVColorTransferFunction;

/** Defines the color range of a particular \ref NVCVColorSpec. */
typedef enum
{
    NVCV_COLOR_RANGE_FULL,   /**< Values cover the full underlying type range. */
    NVCV_COLOR_RANGE_LIMITED /**< Values cover a limited range of the underlying type. */
} NVCVColorRange;

/** Chroma sampling location. */
typedef enum
{
    NVCV_CHROMA_LOC_BOTH = 0, /**< Sample chroma from even and odd coordinates.
                                    This is used when no sub-sampling is taking place. */
    NVCV_CHROMA_LOC_EVEN,     /**< Sample the chroma with even coordinate. */
    NVCV_CHROMA_LOC_CENTER,   /**< Sample the chroma exactly between the even and odd coordinate. */
    NVCV_CHROMA_LOC_ODD,      /**< Sample the chroma with odd coordinate. */
} NVCVChromaLocation;

/** Color spec definitions.
 * These color specs define how color information is to be interpreted.
 * It is defined by several parameters:
 * - \ref NVCVColorModel
 * - \ref NVCVColorSpace
 * - \ref NVCVWhitePoint
 * - \ref NVCVYCbCrEncoding
 * - \ref NVCVColorTransferFunction
 * - \ref NVCVColorRange
 * - \ref NVCVChromaLocation
 *
 * These parameters together defines how the color representation maps to its
 * corresponding absolute color in a chromacity diagram.
 */

/* clang-format off */
typedef enum
{
    /** No color spec defined. Used when color spec isn't relevant or is not defined.
     *  The color spec may be inferred from the context. If this isn't possible, the values for each
     *  color spec component defined below will be used. */
    NVCV_COLOR_SPEC_UNDEFINED        = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_UNDEFINED, XFER_LINEAR,    RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining ITU-R BT.601 standard, limited range, with BT.709 chrominancies and transfer function. */
    NVCV_COLOR_SPEC_BT601            = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.601 standard, full range, with BT.709 chrominancies and transfer function. */
    NVCV_COLOR_SPEC_BT601_ER         = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, limited range. */
    NVCV_COLOR_SPEC_BT709            = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, full range. */
    NVCV_COLOR_SPEC_BT709_ER         = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.709 standard, limited range and linear transfer function. */
    NVCV_COLOR_SPEC_BT709_LINEAR     = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_LINEAR,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range. */
    NVCV_COLOR_SPEC_BT2020           = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_BT2020,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, full range. */
    NVCV_COLOR_SPEC_BT2020_ER        = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_BT2020,    RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range and linear transfer function. */
    NVCV_COLOR_SPEC_BT2020_LINEAR    = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_LINEAR,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, limited range and perceptual quantizer transfer function. */
    NVCV_COLOR_SPEC_BT2020_PQ        = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_PQ,        RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard, full range and perceptual quantizer transfer function. */
    NVCV_COLOR_SPEC_BT2020_PQ_ER     = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020,    XFER_PQ,        RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard for constant luminance, limited range. */
    NVCV_COLOR_SPEC_BT2020c          = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020c,   XFER_BT2020,    RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining ITU-R BT.2020 standard for constant luminance, full range. */
    NVCV_COLOR_SPEC_BT2020c_ER       = NVCV_DETAIL_MAKE_CSPC(SPACE_BT2020, ENC_BT2020c,   XFER_BT2020,    RANGE_FULL,    LOC_EVEN,   LOC_EVEN),

    /** Color spec defining MPEG2 standard using ITU-R BT.601 encoding. */
    NVCV_COLOR_SPEC_MPEG2_BT601      = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining MPEG2 standard using ITU-R BT.709 encoding. */
    NVCV_COLOR_SPEC_MPEG2_BT709      = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT709,     XFER_BT709,     RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining MPEG2 standard using SMPTE 240M encoding. */
    NVCV_COLOR_SPEC_MPEG2_SMPTE240M  = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_SMPTE240M, XFER_SMPTE240M, RANGE_FULL,    LOC_EVEN,   LOC_CENTER),

    /** Color spec defining sRGB standard. */
    NVCV_COLOR_SPEC_sRGB             = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_UNDEFINED, XFER_sRGB,      RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining sYCC standard. */
    NVCV_COLOR_SPEC_sYCC             = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_BT601,     XFER_sYCC,      RANGE_FULL,    LOC_CENTER, LOC_CENTER),

    /** Color spec defining SMPTE 240M standard, limited range. */
    NVCV_COLOR_SPEC_SMPTE240M        = NVCV_DETAIL_MAKE_CSPC(SPACE_BT709,  ENC_SMPTE240M, XFER_SMPTE240M, RANGE_LIMITED, LOC_EVEN,   LOC_EVEN),

    /** Color spec defining Display P3 standard, with sRGB color transfer function. */
    NVCV_COLOR_SPEC_DISPLAYP3        = NVCV_DETAIL_MAKE_CSPC(SPACE_DCIP3,  ENC_UNDEFINED, XFER_sRGB,      RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** Color spec defining Display P3 standard, with linear color transfer function. */
    NVCV_COLOR_SPEC_DISPLAYP3_LINEAR = NVCV_DETAIL_MAKE_CSPC(SPACE_DCIP3,  ENC_UNDEFINED, XFER_LINEAR,    RANGE_FULL,    LOC_BOTH,   LOC_BOTH),

    /** \cond Do not use. */
    NVCV_COLOR_SPEC_FORCE32 = INT32_MAX
    /* \endcond */
} NVCVColorSpec;

/* clang-format on */

/** Creates a user-defined color spec constant.
 *
 * Example:
 * \code{.c}
 *   NVCVColorSpec cspec = NVCV_MAKE_COLOR_SPEC(NVCV_COLOR_SPACE_BT709,
 *                                              NVCV_YCbCr_ENC_sRGB,
 *                                              NVCV_COLOR_XFER_sYCC,
 *                                              NVCV_COLOR_RANGE_FULL,
 *                                              NVCV_CHROMA_LOC_ODD,
 *                                              NVCV_CHROMA_LOC_EVEN);
 * \endcode
 *
 * @param[in] cspace   Color Space.
 * @param[in] encoding R'G'B' <-> Y'CbCr encoding.
 * @param[in] xferFunc Color transfer function.
 * @param[in] range    Color quantization range.
 * @param[in] locHoriz Horizontal chroma location.
 * @param[in] locVert  Vertical chroma location.
 *
 * @returns The user-defined \ref NVCVColorSpec constant.
 */
#ifdef NVCV_DOXYGEN
#    define NVCV_MAKE_COLOR_SPEC(cspace, encoding, xferFunc, range, locHoriz, locVert)
#else
#    define NVCV_MAKE_COLOR_SPEC (NVCVColorSpec) NVCV_DETAIL_MAKE_COLOR_SPEC
#endif

/** Creates a user-defined \ref NVCVColorSpec.
 *
 * @param[out] outColorSpec  Pointer to the output colorspec.
 *                           + Cannot be NULL.
 * @param[in] cspace   Color space.
 * @param[in] encoding R'G'B' -> Y'CbCr encoding.
 * @param[in] xferFunc Color transfer function.
 * @param[in] range    Color quantization range.
 * @param[in] locHoriz Horizontal chroma location.
 * @param[in] locVert  Vertical chroma location.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeColorSpec(NVCVColorSpec *outColorSpec, NVCVColorSpace cspace, NVCVYCbCrEncoding encoding,
                                         NVCVColorTransferFunction xferFunc, NVCVColorRange range,
                                         NVCVChromaLocation locHoriz, NVCVChromaLocation locVert);

/** Defines Bayer patterns used by RAW color model.
 * R,G,B represent the color primaries red, green, blue.
 * C represent a clear channel, it lets all light pass. */
typedef enum
{
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: R G R G R G R G
     * - span 2: G B G B G B G B
     * (Y,Z,W are discarded) */
    NVCV_RAW_BAYER_RGGB,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: B G B G B G B G
     * - span 2: G R G R G R G R
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_BGGR,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: G R G R G R G R
     * - span 2: B G B G B G B G
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_GRBG,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: G B G B G B G B
     * - span 2: R G R G R G R G
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_GBRG,

    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: R C R C R C R C
     * - span 2: C B C B C B C B
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_RCCB,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: B C B C B C B C
     * - span 2: C R C R C R C R
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_BCCR,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C R C R C R C R
     * - span 2: B C B C B C B C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CRBC,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C B C B C B C B
     * - span 2: R C R C R C R C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CBRC,

    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: R C R C R C R C
     * - span 2: C C C C C C C C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_RCCC,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C R C R C R C R
     * - span 2: C C C C C C C C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CRCC,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C C C C C C C C
     * - span 2: R C R C R C R C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CCRC,
    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C C C C C C C C
     * - span 2: C R C R C R C R
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CCCR,

    /** Bayer format with X channel mapped to samples as follows:
     * - span 1: C C C C C C C C
     * - span 2: C C C C C C C C
     * \n(Y,Z,W are discarded)
     */
    NVCV_RAW_BAYER_CCCC,

    /** \cond Do not use. */
    NVCV_RAW_FORCE8 = UINT8_MAX
    /* \endcond */
} NVCVRawPattern;

/** Defines how chroma-subsampling is done.
 * This is only applicable to image formats whose color model is YUV.
 * Other image formats must use \ref NVCV_CSS_NONE.
 * Chroma subsampling is defined by 2 parameters:
 * - Horizontal resolution relative to luma resolution.
 * - Vertical resolution relative to luma resolution.
 */
typedef enum
{
    /** Used when no chroma subsampling takes place, specially for color specs without chroma components. */
    NVCV_CSS_NONE = 0,

    /** 4:4:4 sub-sampling. Chroma has full horizontal and vertical resolution, meaning no chroma subsampling. */
    NVCV_CSS_444 = NVCV_CSS_NONE,

    /** 4:2:2 BT.601 sub-sampling. Chroma has half horizontal and full vertical resolutions.*/
    NVCV_CSS_422,

    /** 4:2:2R BT.601 sub-sampling. Chroma has full horizontal and half vertical resolutions.*/
    NVCV_CSS_422R,

    /** 4:1:1 sub-sampling. Chroma has 1/4 horizontal and full vertical resolutions.*/
    NVCV_CSS_411,

    /** 4:1:1 sub-sampling. Chroma has full horizontal and 1/4 vertical resolutions.*/
    NVCV_CSS_411R,

    /** 4:2:0 sub-sampling. Chroma has half horizontal and vertical resolutions.*/
    NVCV_CSS_420,

    /** 4:1:0 sub-sampling. Chroma has 1/4 horizontal and half vertical resolutions. */
    NVCV_CSS_410,

    /** 4:1:0V sub-sampling. Chroma has half horizontal and 1/4 vertical resolutions. */
    NVCV_CSS_410R,

    /** 4:4:0 sub-sampling. Chroma has full horizontal and half vertical resolutions */
    NVCV_CSS_440 = NVCV_CSS_422R
} NVCVChromaSubsampling;

/** Creates a \ref NVCVChromaSubsampling given the horizontal and vertical sampling.
 *
 * @param[out] outCSS Pointer to output color subsampling.
 *                    + Cannot be NULL.
 * @param[in] samplesHoriz Number of horizontal samples, 1, 2 or 4.
 * @param[in] samplesVert Number of vertical samples, 1, 2 or 4.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeChromaSubsampling(NVCVChromaSubsampling *outCSS, int32_t samplesHoriz,
                                                 int32_t samplesVert);

/** Get the number of chroma samples for each group of 4 horizontal/vertical luma samples.
 *
 * @param[in] css Chroma subsampling to be queried.
 *                + \p css must be valid.
 *
 * @param[out] outSamplesHoriz, outSamplesVert The number of chroma samples for each group
 *                                             of 4 horizontal/vertical luma samples.
 *                                             If NULL, corresponding value won't be output.
 *                                             + Both cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvChromaSubsamplingGetNumSamples(NVCVChromaSubsampling css, int32_t *outSamplesHoriz,
                                                          int32_t *outSamplesVert);

/** Get the chroma sampling location of a given color spec.
 *
 * @param[in] cspec Color spec to be queried.
 *                  + Color spec must be valid.
 *
 * @param[out] outLocVert, outLocHoriz Chroma sample location with respect to
 *                                     luma horizontal/vertical coordinate.
 *                                     If NULL, corresponding value won't be output.
 *                                     + Both cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecGetChromaLoc(NVCVColorSpec cspec, NVCVChromaLocation *outLocHoriz,
                                                 NVCVChromaLocation *outLocVert);

/** Set the chroma sample location of a given color spec
 *
 * @param[inout] cspec Color spec to be updated.
 *                     + Cannot be NULL.
 *                     + \p cspec must be valid.
 *
 * @param[in] locHoriz Horizontal chroma sampling location with respect to luma coordinate.
 *
 * @param[in] locVert Vertical chroma sampling location with respect to luma coordinate.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecSetChromaLoc(NVCVColorSpec *cspec, NVCVChromaLocation locHoriz,
                                                 NVCVChromaLocation locVert);

/** Get the color space of a given color spec.
 *
 * @param[in] cspec Color spec to be queried.
 *                  + \p cspec must be valid.
 * @param[out] outColorSpace Pointer to the output color space associated with the color spec.
 *                           + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecGetColorSpace(NVCVColorSpec cspec, NVCVColorSpace *outColorSpace);

/** Set the color space of a given color spec.
 *
 * @param[inout] cspec Color spec to be updated.
 *                     + Cannot be NULL.
 *                     + \p cspec must be valid.
 *
 * @param[in] cspace The new color_space.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecSetColorSpace(NVCVColorSpec *cspec, NVCVColorSpace cspace);

/** Get the R'G'B' <-> Y'CbCr encoding scheme of a given color spec.
 *
 * @param[in] cspec Color spec to be queried.
 *                  + \p cspec must be valid.
 * @param[out] outEncoding The Y'CbCr encoding scheme associated with the color spec.
 *                         + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecGetYCbCrEncoding(NVCVColorSpec cspec, NVCVYCbCrEncoding *outEncoding);

/** Set the R'G'B' <-> Y'CbCr encoding scheme of a given color spec.
 *
 * @param[inout] cspec Color spec to be updated.
 *                     + Cannot be NULL.
 *
 * @param[in] encoding The new Y'CbCr encoding scheme.
 *                     + \p encoding cannot be #NVCV_YCbCr_ENC_UNDEFINED.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecSetYCbCrEncoding(NVCVColorSpec *cspec, NVCVYCbCrEncoding encoding);

/** Get the color transfer function of a given color spec.
 *
 * @param[in] cspec Color spec to be queried.
 *                  + \p cspec must be valid.
 *
 * @param[out] outXferFunc Pointer to the output color transfer function of the given colorspec.
 *                         + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecGetColorTransferFunction(NVCVColorSpec              cspec,
                                                             NVCVColorTransferFunction *outXferFunc);

/** Set the color transfer function of a given color spec.
 *
 * @param[inout] cspec Color spec to be updated.
 *                     + Cannot be NULL.
 *                     + \p cspec must be valid.
 *
 * @param[in] xferFunc The new color transfer function.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecSetColorTransferFunction(NVCVColorSpec *cspec, NVCVColorTransferFunction xferFunc);

/** Get the color quantization range of a given color spec.
 *
 * @param[in] cspec Color spec to be queried.
 *                  + \p cspec must be valid.
 *
 * @param[out] outColorRange The color quantization range of a given color spec.
 *                           + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecGetRange(NVCVColorSpec cspec, NVCVColorRange *outColorRange);

/** Set the color quantization range of a given color spec.
 *
 * @param[inout] cspec Color spec to be updated.
 *                     + Cannot be NULL.
 *                     + \p cspec must be valid.
 *
 * @param[in] range The new color quantization range.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorSpecSetRange(NVCVColorSpec *cspec, NVCVColorRange range);

/** Returns a string representation of the color spec.
 *
 * @param[in] cspec Color spec whose name is to be returned.
 *
 * @returns The string representation of the color spec.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvColorSpecGetName(NVCVColorSpec cspec);

/** Returns whether a given color model needs a colorspec.
 *
 * @param[in] cmodel Color model to be queried
 *
 * @param[out] outBool Returns the boolean result.
 *                     0 if color model doesn't need a colorspec, != 0 otherwise.
 *                     + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvColorModelNeedsColorspec(NVCVColorModel cmodel, int8_t *outBool);

/** Returns a string representation of the color model.
 *
 * @param[in] cmodel Color model whose name is to be returned.
 *
 * @returns The string representation of the color model.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvColorModelGetName(NVCVColorModel cmodel);

/** Returns a string representation of the color space.
 *
 * @param[in] cspace Color model whose name is to be returned.
 *
 * @returns The string representation of the color model.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvColorSpaceGetName(NVCVColorSpace cspace);

/** Returns a string representation of the white point.
 *
 * @param[in] wpoint White point whose name is to be returned.
 *
 * @returns The string representation of the white point.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvWhitePointGetName(NVCVWhitePoint wpoint);

/** Returns a string representation of the YCbCr encoding.
 *
 * @param[in] enc YCbCr encoding whose name is to be returned.
 *
 * @returns The string representation of the YCbCr encoding.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvYCbCrEncodingGetName(NVCVYCbCrEncoding enc);

/** Returns a string representation of the color transfer function.
 *
 * @param[in] xfer Color transfer function whose name is to be returned.
 *
 * @returns The string representation of the color transfer function.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvColorTransferFunctionGetName(NVCVColorTransferFunction xfer);

/** Returns a string representation of the color range.
 *
 * @param[in] range Color range whose name is to be returned.
 *
 * @returns The string representation of the color range.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvColorRangeGetName(NVCVColorRange range);

/** Returns a string representation of the chroma location.
 *
 * @param[in] loc Chroma location whose name is to be returned.
 *
 * @returns The string representation of the chroma location.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvChromaLocationGetName(NVCVChromaLocation loc);

/** Returns a string representation of the raw pattern.
 *
 * @param[in] raw Raw pattern whose name is to be returned.
 *
 * @returns The string representation of the raw pattern.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvRawPatternGetName(NVCVRawPattern raw);

/** Returns a string representation of the chroma subsampling.
 *
 * @param[in] css Chroma subsampling whose name is to be returned.
 *
 * @returns The string representation of the chroma subsampling.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvChromaSubsamplingGetName(NVCVChromaSubsampling css);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* NVCV_FORMAT_COLORSPEC_H */
