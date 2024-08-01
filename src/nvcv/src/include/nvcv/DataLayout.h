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
 * @file DataLayout.h
 *
 * @brief Defines types and functions to handle data layouts.
 */

#ifndef NVCV_FORMAT_DATALAYOUT_H
#define NVCV_FORMAT_DATALAYOUT_H

#include "detail/FormatUtils.h"

#include <nvcv/Export.h>
#include <nvcv/Status.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @defgroup NVCV_C_CORE_DATALAYOUT Data Layout
 * @{
*/

/**
 * Maximum channel count
 */
#define NVCV_MAX_CHANNEL_COUNT       (4)
#define NVCV_MAX_EXTRA_CHANNEL_COUNT (7)
#define NVCV_MAX_SWIZZLE_COUNT       (57)

/** Defines how channels are packed into an image plane element.
 *
 * Packing encodes how many channels the plane element has, and how they
 * are arranged in memory.
 *
 * Up to 4 channels (denoted by X, Y, Z, W) can be packed into an image
 * plane element, each one occupying a specified number of bits.
 *
 * When two channels are specified one right after the other, they are
 * ordered from most-significant bit to least-significant bit. Words are
 * separated by underscores. For example:
 *
 * X8Y8Z8W8 = a single 32-bit word containing 4 channels, 8 bits each.
 *
 * In little-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            WWWWWWWWZZZZZZZZYYYYYYYYXXXXXXXX
 * </pre>
 *
 * In big-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * X8_Y8_Z8_W8 = four consecutive 8-bit words, corresponding to 4 channels, 8 bits each.
 *
 * In little-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * In big-endian architectures:
 * <pre>
 *      Address  0   ||  1   ||  2   ||  3
 *            XXXXXXXXYYYYYYYYZZZZZZZZWWWWWWWW
 * </pre>
 *
 * In cases where a word is less than 8 bits (e.g., X1 1-bit channel), channels
 * are ordered from LSB to MSB within a word.
 *
 * @note Also note equivalences such as the following:
 * @note In little-endian: X8_Y8_Z8_W8 = W8Z8Y8X8.
 * @note In big-endian: X8_Y8_Z8_W8 = X8Y8Z8W8.
 *
 * Some formats allow different packings when pixels' horizontal coordinate is
 * even or odd. For instance, every pixel of YUV422 packed format contains an Y
 * channel, while only even pixels contain the U channel, and odd pixels contain
 * V channel. Such formats use a double-underscore to separate the even pixels from the odd
 * pixels. The packing just described might be referred to X8_Y8__X8_Z8, where X = luma,
 * Y = U chroma, Z = V chroma.
 */
typedef enum
{
    /** No channels. */
    NVCV_PACKING_0 = 0,

    /** One 1-bit channel. */
    NVCV_PACKING_X1 = NVCV_DETAIL_BPP_NCH(1, 1),
    /** One 2-bit channel. */
    NVCV_PACKING_X2 = NVCV_DETAIL_BPP_NCH(2, 1),
    /** One 4-bit channel. */
    NVCV_PACKING_X4 = NVCV_DETAIL_BPP_NCH(4, 1),

    /** One 8-bit channel. */
    NVCV_PACKING_X8 = NVCV_DETAIL_BPP_NCH(8, 1),
    /** One LSB 4-bit channel in a 8-bit word */
    NVCV_PACKING_b4X4,
    /** One MSB 4-bit channel in a 8-bit word */
    NVCV_PACKING_X4b4,

    /** Two 4-bit channels in one 8-bit word. */
    NVCV_PACKING_X4Y4 = NVCV_DETAIL_BPP_NCH(8, 2),

    /** Three 3-, 3- and 2-bit channels in one 8-bit word. */
    NVCV_PACKING_X3Y3Z2 = NVCV_DETAIL_BPP_NCH(8, 3),

    /** One 16-bit channel. */
    NVCV_PACKING_X16 = NVCV_DETAIL_BPP_NCH(16, 1),
    /** One LSB 10-bit channel in one 16-bit word. */
    NVCV_PACKING_b6X10,
    /** One MSB 10-bit channel in one 16-bit word. */
    NVCV_PACKING_X10b6,
    /** One LSB 12-bit channel in one 16-bit word. */
    NVCV_PACKING_b4X12,
    /** One MSB 12-bit channel in one 16-bit word. */
    NVCV_PACKING_X12b4,
    /** One LSB 14-bit channel in one 16-bit word. */
    NVCV_PACKING_b2X14,
    /** One MSB 14-bit channel in one 16-bit word. */
    NVCV_PACKING_X14b2,

    /** Two 8-bit channels in two 8-bit words. */
    NVCV_PACKING_X8_Y8 = NVCV_DETAIL_BPP_NCH(16, 2),

    /** Three 5-, 5- and 6-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y5Z6 = NVCV_DETAIL_BPP_NCH(16, 3),
    /** Three 5-, 6- and 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y6Z5,
    /** Three 6-, 5- and 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X6Y5Z5,
    /** Three 4-bit channels in one 16-bit word. */
    NVCV_PACKING_b4X4Y4Z4,
    /** Three 5-bit channels in one 16-bit word. */
    NVCV_PACKING_b1X5Y5Z5,
    /** Three 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y5b1Z5,

    /** Four 1-, 5-, 5- and 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X1Y5Z5W5 = NVCV_DETAIL_BPP_NCH(16, 4),
    /** Four 4-bit channels in one 16-bit word. */
    NVCV_PACKING_X4Y4Z4W4,
    /** Four 5-, 1-, 5- and 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y1Z5W5,
    /** Four 5-, 5-, 1- and 5-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y5Z1W5,
    /** Four 5-, 5-, 5- and 1-bit channels in one 16-bit word. */
    NVCV_PACKING_X5Y5Z5W1,

    /** 2 pixels of 2 8-bit channels each, totalling 4 8-bit words. */
    NVCV_PACKING_X8_Y8__X8_Z8,
    /** 2 pixels of 2 swapped 8-bit channels each, totalling 4 8-bit words. */
    NVCV_PACKING_Y8_X8__Z8_X8,

    /** One 24-bit channel. */
    NVCV_PACKING_X24 = NVCV_DETAIL_BPP_NCH(24, 1),

    /** Three 8-bit channels in three 8-bit words. */
    NVCV_PACKING_X8_Y8_Z8 = NVCV_DETAIL_BPP_NCH(24, 3),

    /** One 32-bit channel. */
    NVCV_PACKING_X32 = NVCV_DETAIL_BPP_NCH(32, 1),
    /** One LSB 20-bit channel in one 32-bit word. */
    NVCV_PACKING_b12X20,
    /** One MSB 20-bit channel in one 32-bit word. */
    NVCV_PACKING_X20b12,
    /** One MSB 24-bit channel in one 32-bit word. */
    NVCV_PACKING_X24b8,
    /** One LSB 24-bit channel in one 32-bit word. */
    NVCV_PACKING_b8X24,

    /** Two 16-bit channels in two 16-bit words. */
    NVCV_PACKING_X16_Y16 = NVCV_DETAIL_BPP_NCH(32, 2),
    /** Two MSB 10-bit channels in two 16-bit words. */
    NVCV_PACKING_X10b6_Y10b6,
    /** Two MSB 12-bit channels in two 16-bit words. */
    NVCV_PACKING_X12b4_Y12b4,

    /** Three 10-, 11- and 11-bit channels in one 32-bit word. */
    NVCV_PACKING_X10Y11Z11 = NVCV_DETAIL_BPP_NCH(32, 3),
    /** Three 11-, 11- and 10-bit channels in one 32-bit word. */
    NVCV_PACKING_X11Y11Z10,
    /** Three LSB 10-bit channels in one 32-bit word. */
    NVCV_PACKING_b2X10Y10Z10,
    /** Three MSB 10-bit channels in one 32-bit word. */
    NVCV_PACKING_X10Y10Z10b2,

    /** Four 8-bit channels in one 32-bit word. */
    NVCV_PACKING_X8_Y8_Z8_W8 = NVCV_DETAIL_BPP_NCH(32, 4),
    /** Four 2-, 10-, 10- and 10-bit channels in one 32-bit word. */
    NVCV_PACKING_X2Y10Z10W10,
    /** Four 10-, 10-, 10- and 2-bit channels in one 32-bit word. */
    NVCV_PACKING_X10Y10Z10W2,

    /** One 48-bit channel. */
    NVCV_PACKING_X48 = NVCV_DETAIL_BPP_NCH(48, 1),
    /** Three 16-bit channels in three 16-bit words. */
    NVCV_PACKING_X16_Y16_Z16 = NVCV_DETAIL_BPP_NCH(48, 3),

    /** One 64-bit channel. */
    NVCV_PACKING_X64 = NVCV_DETAIL_BPP_NCH(64, 1),
    /** Two 32-bit channels in two 32-bit words. */
    NVCV_PACKING_X32_Y32 = NVCV_DETAIL_BPP_NCH(64, 2),
    /** Two channels: 32-bit in a 32-bit word, 24-bit MSB in a 32-bit word */
    NVCV_PACKING_X32_Y24b8,

    /** Four 16-bit channels in one 64-bit word. */
    NVCV_PACKING_X16_Y16_Z16_W16 = NVCV_DETAIL_BPP_NCH(64, 4),

    /** One 96-bit channel. */
    NVCV_PACKING_X96 = NVCV_DETAIL_BPP_NCH(96, 1),
    /** Three 32-bit channels in three 32-bit words. */
    NVCV_PACKING_X32_Y32_Z32 = NVCV_DETAIL_BPP_NCH(96, 3),

    /** One 128-bit channel. */
    NVCV_PACKING_X128 = NVCV_DETAIL_BPP_NCH(128, 1),
    /** Two 64-bit channels in two 64-bit words. */
    NVCV_PACKING_X64_Y64 = NVCV_DETAIL_BPP_NCH(128, 2),
    /** Four 32-bit channels in three 32-bit words. */
    NVCV_PACKING_X32_Y32_Z32_W32 = NVCV_DETAIL_BPP_NCH(128, 4),

    /** One 192-bit channel. */
    NVCV_PACKING_X192 = NVCV_DETAIL_BPP_NCH(192, 1),
    /** Three 64-bit channels in three 64-bit words. */
    NVCV_PACKING_X64_Y64_Z64 = NVCV_DETAIL_BPP_NCH(192, 3),

    /** One 256-bit channel. */
    NVCV_PACKING_X256 = NVCV_DETAIL_BPP_NCH(256, 1),
    /** Two 128-bit channels in two 128-bit words. */
    NVCV_PACKING_X128_Y128 = NVCV_DETAIL_BPP_NCH(256, 2),
    /** Four 64-bit channels in four 64-bit words. */
    NVCV_PACKING_X64_Y64_Z64_W64 = NVCV_DETAIL_BPP_NCH(256, 4),

    /** \cond Do not use. */
    NVCV_PACKING_LIMIT32 = INT32_MAX
    /* \endcond */
} NVCVPacking;

/** Defines the channel data type. */
typedef enum
{
    NVCV_DATA_KIND_UNSPECIFIED = -1, /**< Unspecified data kind. */
    NVCV_DATA_KIND_UNSIGNED,         /**< Channels are unsigned integer values. */
    NVCV_DATA_KIND_SIGNED,           /**< Channels are signed integer values. */
    NVCV_DATA_KIND_FLOAT,            /**< Channels are floating point values. */
    NVCV_DATA_KIND_COMPLEX           /**< Channels are complex values. */
} NVCVDataKind;

/** Defines how the 2D plane pixels are laid out in memory.
 * This defines how a pixel are addressed, i.e., given its \f$(x,y)\f$ coordinate,
 * what's its memory address.
 * Block-linear formats have a proprietary memory representation and aren't supposed to
 * be addressed by the user directly.
 */
typedef enum
{
    /** Pixels are laid out in row-major order.
     * \f$(x,y) = y \times \mathit{pitch} + x \times \mathit{pixel stride}\f$. */
    NVCV_MEM_LAYOUT_PITCH_LINEAR,

    /** Pixels are laid out in block-linear format with height = 1. */
    NVCV_MEM_LAYOUT_BLOCK1_LINEAR,

    /** Pixels are laid out in block-linear format with height = 2. */
    NVCV_MEM_LAYOUT_BLOCK2_LINEAR,

    /** Pixels are laid out in block-linear format with height = 4. */
    NVCV_MEM_LAYOUT_BLOCK4_LINEAR,

    /** Pixels are laid out in block-linear format with height = 8. */
    NVCV_MEM_LAYOUT_BLOCK8_LINEAR,

    /** Pixels are laid out in block-linear format with height = 16. */
    NVCV_MEM_LAYOUT_BLOCK16_LINEAR,

    /** Pixels are laid out in block-linear format with height = 32. */
    NVCV_MEM_LAYOUT_BLOCK32_LINEAR,

    /** Default block-linear format.
     * It's guaranteed to be valid in all algorithms that support block-linear format. */
    NVCV_MEM_LAYOUT_BLOCK_LINEAR = NVCV_MEM_LAYOUT_BLOCK2_LINEAR,

    /** @{ Useful aliases. */
    NVCV_MEM_LAYOUT_PL = NVCV_MEM_LAYOUT_PITCH_LINEAR,
    NVCV_MEM_LAYOUT_BL = NVCV_MEM_LAYOUT_BLOCK_LINEAR
    /** @} */
} NVCVMemLayout;

/** Defines the format channel names.
 * The channels are color model-agnostic. */
typedef enum
{
    NVCV_CHANNEL_0 = 0, /**< Don't select a channel. */
    NVCV_CHANNEL_X,     /**< Selects the first channel of the color model. */
    NVCV_CHANNEL_Y,     /**< Selects the second channel of the color model. */
    NVCV_CHANNEL_Z,     /**< Selects the third channel of the color model. */
    NVCV_CHANNEL_W,     /**< Selects the fourth channel of the color model. */
    NVCV_CHANNEL_1,     /**< Sets the corresponding channel to have its maximum value. */

    /** \cond Do not use. */
    NVCV_CHANNEL_FORCE8 = UINT8_MAX,
    /* \endcond */
} NVCVChannel;

/** Defines the kind of alpha channel data present. */
typedef enum
{
    NVCV_ALPHA_ASSOCIATED,   /**< Associated alpha type */
    NVCV_ALPHA_UNASSOCIATED, /**< Unassociated alpha type */
} NVCVAlphaType;

/** Defines the different kinds of extra channels supported. */
typedef enum
{
    NVCV_EXTRA_CHANNEL_U = 0, /**< Unspecified Channel. */
    NVCV_EXTRA_CHANNEL_D,     /**< Depth data. */
    NVCV_EXTRA_CHANNEL_POS3D, /**< 3D Position data. */
} NVCVExtraChannel;

/** Data structure for passing additional channel information. */
typedef struct
{
    int32_t          numChannels;
    int32_t          bitsPerPixel;
    NVCVDataKind     datakind;
    NVCVExtraChannel channelType;
} NVCVExtraChannelInfo;

/** Defines the supported channel swizzle operations.
 *
 * The operations map an input vector \f$(x,y,z,w)\f$ into an output vector
 * \f$(x',y',z',w')\f$. Any output channel can select any of the input
 * channels, or the constants zero or one. For example, the swizzle "X000"
 * selects the first channel, whereas swizzle "ZYXW" swaps the X and Z
 * channels, needed for conversion between RGBA and BGRA image formats.
 */
typedef enum
{
    /** @{ Swizzle operation. */
    NVCV_SWIZZLE_0000,
    NVCV_SWIZZLE_X000,
    NVCV_SWIZZLE_XY00,
    NVCV_SWIZZLE_XYZ0,
    NVCV_SWIZZLE_XYZW,
    NVCV_SWIZZLE_1000,
    NVCV_SWIZZLE_0001,
    NVCV_SWIZZLE_ZYXW,
    NVCV_SWIZZLE_WXYZ,
    NVCV_SWIZZLE_WZYX,
    NVCV_SWIZZLE_YZWX,
    NVCV_SWIZZLE_XYZ1,
    NVCV_SWIZZLE_YZW1,
    NVCV_SWIZZLE_XXX1,
    NVCV_SWIZZLE_XZY1,
    NVCV_SWIZZLE_ZYX1,
    NVCV_SWIZZLE_ZYX0,
    NVCV_SWIZZLE_WZY1,
    NVCV_SWIZZLE_0X00,
    NVCV_SWIZZLE_00X0,
    NVCV_SWIZZLE_000X,
    NVCV_SWIZZLE_Y000,
    NVCV_SWIZZLE_0Y00,
    NVCV_SWIZZLE_00Y0,
    NVCV_SWIZZLE_000Y,
    NVCV_SWIZZLE_0XY0,
    NVCV_SWIZZLE_XXXY,
    NVCV_SWIZZLE_YYYX,
    NVCV_SWIZZLE_0YX0,
    NVCV_SWIZZLE_X00Y,
    NVCV_SWIZZLE_Y00X,
    NVCV_SWIZZLE_X001,
    NVCV_SWIZZLE_XY01,
    NVCV_SWIZZLE_0XZ0,
    NVCV_SWIZZLE_0ZX0,
    NVCV_SWIZZLE_XZY0,
    NVCV_SWIZZLE_YZX1,
    NVCV_SWIZZLE_ZYW1,
    NVCV_SWIZZLE_0YX1,
    NVCV_SWIZZLE_XYXZ,
    NVCV_SWIZZLE_YXZX,
    NVCV_SWIZZLE_XZ00,
    NVCV_SWIZZLE_WYXZ,
    NVCV_SWIZZLE_YX00,
    NVCV_SWIZZLE_YX01,
    NVCV_SWIZZLE_00YX,
    NVCV_SWIZZLE_00XY,
    NVCV_SWIZZLE_0XY1,
    NVCV_SWIZZLE_0X01,
    NVCV_SWIZZLE_YZXW,
    NVCV_SWIZZLE_YW00,
    NVCV_SWIZZLE_XYW0,
    NVCV_SWIZZLE_YZW0,
    NVCV_SWIZZLE_YZ00,
    NVCV_SWIZZLE_00X1,
    NVCV_SWIZZLE_0ZXY,
    NVCV_SWIZZLE_UNSUPPORTED = 0b111111
    /** @} */
} NVCVSwizzle;

/** Creates a user-defined \ref NVCVSwizzle operation.
 * This is similar to \ref NVCV_MAKE_SWIZZLE, but accepts the swizzle channels as runtime variables.
 *
 * @param[out] outSwizzle Swizzle operation as defined by the given channel order.
 *
 * @param[in] x Channel that will correspond to the first component.
 *
 * @param[in] y Channel that will correspond to the second component.
 *
 * @param[in] z Channel that will correspond to the third component.
 *
 * @param[in] w Channel that will correspond to the fourth component.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakeSwizzle(NVCVSwizzle *outSwizzle, NVCVChannel x, NVCVChannel y, NVCVChannel z,
                                       NVCVChannel w);

/** Get the swizzle channels.
 *
 * For example, given swizzle \ref NVCV_SWIZZLE_YZWX, it returns
 * \ref NVCV_CHANNEL_Y, \ref NVCV_CHANNEL_Z, \ref NVCV_CHANNEL_W and
 * \ref NVCV_CHANNEL_X.
 *
 * @param[in] swizzle Swizzle to be queried.
 *
 * @param[out] channels Output channel array with 4 elements.
 *                      + It cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvSwizzleGetChannels(NVCVSwizzle swizzle, NVCVChannel *channels);

/** Get the number of channels specified by the given swizzle.
 *
 * Only the following count as channels:
 * - \ref NVCV_CHANNEL_X
 * - \ref NVCV_CHANNEL_Y
 * - \ref NVCV_CHANNEL_Z
 * - \ref NVCV_CHANNEL_W
 *
 * @param[in] swizzle Swizzle to be queried.
 *
 * @param[out] outNumChannels The channel count specified by swizzle.
 *                            + It cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvSwizzleGetNumChannels(NVCVSwizzle swizzle, int32_t *outNumChannels);

/** Byte/bit order of a \ref NVCVPacking value in a word. */
typedef enum
{
    NVCV_ORDER_LSB, /**< Least significant byte/bit has higher memory address. */
    NVCV_ORDER_MSB  /**< Most significant byte/bit has lower memory address. */
} NVCVByteOrder;

/** Defines the parameters encoded in a \ref NVCVPacking. */
typedef struct
{
    /** Component ordering in a word. */
    NVCVByteOrder byteOrder;

    /** Address alignment requirement, in bytes */
    int32_t alignment;

    /** Channel ordering. */
    NVCVSwizzle swizzle;

    /** Number of bits in each channel.
     *  If channel doesn't exist, corresponding bits==0. */
    int32_t bits[NVCV_MAX_CHANNEL_COUNT];

} NVCVPackingParams;

/** Returns a pre-defined \ref NVCVPacking given its params.
 *
 * This function calculates the \ref NVCVPacking based on the channel characteristics at run time.
 *
 * @param[out] outPacking The packing enum corresponding to \p params.
 *                        + It cannot be NULL.
 *
 * @param[in] params Packing parameters.
 *                   If \ref NVCVPackingParams::swizzle is set to \ref NVCV_SWIZZLE_0000
 *                   the swizzle will be inferred from \ref NVCVPackingParams::bits.
 *                   It'll return the packing with the largest alignment that is smaller
 *                   or equal the requested alignment. If requested alignment is 0, it'll return
 *                   the packing with smallest alignment.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakePacking(NVCVPacking *outPacking, const NVCVPackingParams *params);

/** Returns channels' information from a format packing.
 *
 * @param[in] packing The format packing to be queried.
 *
 * @param[out] outParams The packing parameters.
 *                    + It cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPackingGetParams(NVCVPacking packing, NVCVPackingParams *outParams);

/** Returns the number of components defined by the given packing.
 *
 * @param[in] packing The format packing to be queried.
 *
 * @param[out] outNumComponents Number of components from the given format packing. It's value between 0 and 4.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPackingGetNumComponents(NVCVPacking packing, int32_t *outNumComponents);

/** Returns the number of bits per packing component.
 *
 * @param[in] packing The format packing to be queried.
 *
 * @param[out] bits Pointer to an int32_t array with 4 elements where output will be stored.
 *                  Passing NULL is allowed, to which the function simply does nothing.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPackingGetBitsPerComponent(NVCVPacking packing, int32_t *outBits);

/** Returns the number of bits per pixel of the given packing.
 *
 * @param[in] packing The format packing to be queried.
 *
 * @param[out] outBPP Total number of bits per pixel of the given packing.
 *                    It's the sum of number of bits occupied by all packing channels.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPackingGetBitsPerPixel(NVCVPacking packing, int32_t *outBPP);

/** Get the required address alignment for the packing.
 *
 * The returned alignment is guaranteed to be a power-of-two.
 *
 * @param[in] type Packing to be queried.
 *
 * @param[out] outAlignment Pointer to an int32_t where the required alignment is to be stored.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPackingGetAlignment(NVCVPacking packing, int32_t *outAlignment);

/** Returns a string representation of a packing.
 *
 * @param[in] packing Packing whose name is to be returned.
 *
 * @returns The string representation of the packing.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvPackingGetName(NVCVPacking fmt);

/** Returns a string representation of a data type.
 *
 * @param[in] dtype Data type whose name is to be returned.
 *
 * @returns The string representation of the data type.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvDataKindGetName(NVCVDataKind dtype);

/** Returns a string representation of a memory layout.
 *
 * @param[in] memlayout Memory layout whose name is to be returned.
 *
 * @returns The string representation of the memory layout.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvMemLayoutGetName(NVCVMemLayout memlayout);

/** Returns a string representation of a channel.
 *
 * @param[in] channel Channel whose name is to be returned.
 *
 * @returns The string representation of the channel.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvChannelGetName(NVCVChannel channel);

/** Returns a string representation of a swizzle.
 *
 * @param[in] swizzle Swizzle whose name is to be returned.
 *
 * @returns The string representation of the swizzle.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvSwizzleGetName(NVCVSwizzle swizzle);

/** Returns a string representation of a byte order.
 *
 * @param[in] byteOrder Byte order whose name is to be returned.
 *
 * @returns The string representation of the byte order.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvByteOrderGetName(NVCVByteOrder byteOrder);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* NVCV_FORMAT_DATALAYOUT_H */
