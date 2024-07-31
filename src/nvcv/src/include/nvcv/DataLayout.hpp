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
 * @file DataLayout.hpp
 *
 * @brief Defines C++ types and functions to handle data layouts.
 */

#ifndef NVCV_DATALAYOUT_HPP
#define NVCV_DATALAYOUT_HPP

#include "DataLayout.h"

#include <nvcv/detail/CheckError.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace nvcv {

/**
 * @defgroup NVCV_CPP_CORE_DATALAYOUT Data Layout
 * @{
*/

enum class Packing : int32_t
{
    NONE,
    /** One 1-bit channel. */
    X1 = NVCV_PACKING_X1,
    /** One 2-bit channel. */
    X2 = NVCV_PACKING_X2,
    /** One 4-bit channel. */
    X4 = NVCV_PACKING_X4,
    /** One 8-bit channel. */
    X8 = NVCV_PACKING_X8,
    /** Two 4-bit channels in one word. */
    X4Y4 = NVCV_PACKING_X4Y4,
    /** Three 3-, 3- and 2-bit channels in one 8-bit word. */
    X3Y3Z2 = NVCV_PACKING_X3Y3Z2,

    /** One 16-bit channel. */
    X16 = NVCV_PACKING_X16,
    /** One LSB 10-bit channel in one 16-bit word. */
    b6X10 = NVCV_PACKING_b6X10,
    /** One MSB 10-bit channel in one 16-bit word. */
    X10b6 = NVCV_PACKING_X10b6,
    /** One LSB 12-bit channel in one 16-bit word. */
    b4X12 = NVCV_PACKING_b4X12,
    /** One MSB 12-bit channel in one 16-bit word. */
    X12b4 = NVCV_PACKING_X12b4,
    /** One LSB 14-bit channel in one 16-bit word. */
    b2X14 = NVCV_PACKING_b2X14,

    /** Two 8-bit channels in two 8-bit words. */
    X8_Y8 = NVCV_PACKING_X8_Y8,

    /** Three 5-, 5- and 6-bit channels in one 16-bit word. */
    X5Y5Z6 = NVCV_PACKING_X5Y5Z6,
    /** Three 5-, 6- and 5-bit channels in one 16-bit word. */
    X5Y6Z5 = NVCV_PACKING_X5Y6Z5,
    /** Three 6-, 5- and 5-bit channels in one 16-bit word. */
    X6Y5Z5 = NVCV_PACKING_X6Y5Z5,
    /** Three 4-bit channels in one 16-bit word. */
    b4X4Y4Z4 = NVCV_PACKING_b4X4Y4Z4,
    /** Three 5-bit channels in one 16-bit word. */
    b1X5Y5Z5 = NVCV_PACKING_b1X5Y5Z5,
    /** Three 5-bit channels in one 16-bit word. */
    X5Y5b1Z5 = NVCV_PACKING_X5Y5b1Z5,

    /** Four 1-, 5-, 5- and 5-bit channels in one 16-bit word. */
    X1Y5Z5W5 = NVCV_PACKING_X1Y5Z5W5,
    /** Four 4-bit channels in one 16-bit word. */
    X4Y4Z4W4 = NVCV_PACKING_X4Y4Z4W4,
    /** Four 5-, 1-, 5- and 5-bit channels in one 16-bit word. */
    X5Y1Z5W5 = NVCV_PACKING_X5Y1Z5W5,
    /** Four 5-, 5-, 1- and 5-bit channels in one 16-bit word. */
    X5Y5Z1W5 = NVCV_PACKING_X5Y5Z1W5,
    /** Four 5-, 5-, 5- and 1-bit channels in one 16-bit word. */
    X5Y5Z5W1 = NVCV_PACKING_X5Y5Z5W1,

    /** 2 pixels of 2 8-bit channels each, totalling 4 8-bit words. */
    X8_Y8__X8_Z8 = NVCV_PACKING_X8_Y8__X8_Z8,
    /** 2 pixels of 2 swapped 8-bit channels each, totalling 4 8-bit words. */
    Y8_X8__Z8_X8 = NVCV_PACKING_Y8_X8__Z8_X8,

    /** One 24-bit channel. */
    X24 = NVCV_PACKING_X24,

    /** Three 8-bit channels in three 8-bit words. */
    X8_Y8_Z8 = NVCV_PACKING_X8_Y8_Z8,

    /** One 32-bit channel. */
    X32 = NVCV_PACKING_X32,
    /** One LSB 20-bit channel in one 32-bit word. */
    b12X20 = NVCV_PACKING_b12X20,

    /** Two 16-bit channels in two 16-bit words. */
    X16_Y16 = NVCV_PACKING_X16_Y16,
    /** Two MSB 10-bit channels in two 16-bit words. */
    X10b6_Y10b6 = NVCV_PACKING_X10b6_Y10b6,
    /** Two MSB 12-bit channels in two 16-bit words. */
    X12b4_Y12b4 = NVCV_PACKING_X12b4_Y12b4,

    /** Three 10-, 11- and 11-bit channels in one 32-bit word. */
    X10Y11Z11 = NVCV_PACKING_X10Y11Z11,
    /** Three 11-, 11- and 10-bit channels in one 32-bit word. */
    X11Y11Z10 = NVCV_PACKING_X11Y11Z10,

    /** Four 8-bit channels in one 32-bit word. */
    X8_Y8_Z8_W8 = NVCV_PACKING_X8_Y8_Z8_W8,
    /** Four 2-, 10-, 10- and 10-bit channels in one 32-bit word. */
    X2Y10Z10W10 = NVCV_PACKING_X2Y10Z10W10,
    /** Four 10-, 10-, 10- and 2-bit channels in one 32-bit word. */
    X10Y10Z10W2 = NVCV_PACKING_X10Y10Z10W2,

    /** One 48-bit channel. */
    X48 = NVCV_PACKING_X48,
    /** Three 16-bit channels in three 16-bit words. */
    X16_Y16_Z16 = NVCV_PACKING_X16_Y16_Z16,

    /** One 64-bit channel. */
    X64 = NVCV_PACKING_X64,
    /** Two 32-bit channels in two 32-bit words. */
    X32_Y32 = NVCV_PACKING_X32_Y32,
    /** Four 16-bit channels in one 64-bit word. */
    X16_Y16_Z16_W16 = NVCV_PACKING_X16_Y16_Z16_W16,

    /** One 96-bit channel. */
    X96 = NVCV_PACKING_X96,
    /** Three 32-bit channels in three 32-bit words. */
    X32_Y32_Z32 = NVCV_PACKING_X32_Y32_Z32,

    /** One 128-bit channel. */
    X128 = NVCV_PACKING_X128,
    /** Two 64-bit channels in two 64-bit words. */
    X64_Y64 = NVCV_PACKING_X64_Y64,
    /** Four 32-bit channels in three 32-bit words. */
    X32_Y32_Z32_W32 = NVCV_PACKING_X32_Y32_Z32_W32,

    /** One 192-bit channel. */
    X192 = NVCV_PACKING_X192,
    /** Three 64-bit channels in three 64-bit words. */
    X64_Y64_Z64 = NVCV_PACKING_X64_Y64_Z64,

    /** One 256-bit channel. */
    X256 = NVCV_PACKING_X256,
    /** Two 128-bit channels in two 128-bit words. */
    X128_Y128 = NVCV_PACKING_X128_Y128,
    /** Four 64-bit channels in four 64-bit words. */
    X64_Y64_Z64_W64 = NVCV_PACKING_X64_Y64_Z64_W64,
};

using ExtraChannelInfo = NVCVExtraChannelInfo;

enum class DataKind : int8_t
{
    UNSPECIFIED = NVCV_DATA_KIND_UNSPECIFIED,
    UNSIGNED    = NVCV_DATA_KIND_UNSIGNED,
    SIGNED      = NVCV_DATA_KIND_SIGNED,
    FLOAT       = NVCV_DATA_KIND_FLOAT,
    COMPLEX     = NVCV_DATA_KIND_COMPLEX
};

enum class MemLayout : int8_t
{
    PITCH_LINEAR   = NVCV_MEM_LAYOUT_PITCH_LINEAR,
    BLOCK1_LINEAR  = NVCV_MEM_LAYOUT_BLOCK1_LINEAR,
    BLOCK2_LINEAR  = NVCV_MEM_LAYOUT_BLOCK2_LINEAR,
    BLOCK4_LINEAR  = NVCV_MEM_LAYOUT_BLOCK4_LINEAR,
    BLOCK8_LINEAR  = NVCV_MEM_LAYOUT_BLOCK8_LINEAR,
    BLOCK16_LINEAR = NVCV_MEM_LAYOUT_BLOCK16_LINEAR,
    BLOCK32_LINEAR = NVCV_MEM_LAYOUT_BLOCK32_LINEAR,

    BLOCK_LINEAR = NVCV_MEM_LAYOUT_BLOCK_LINEAR,
    PL           = NVCV_MEM_LAYOUT_PL,
    BL           = NVCV_MEM_LAYOUT_BL
};

enum class Channel : int8_t
{
    NONE = NVCV_CHANNEL_0,
    X    = NVCV_CHANNEL_X,
    Y    = NVCV_CHANNEL_Y,
    Z    = NVCV_CHANNEL_Z,
    W    = NVCV_CHANNEL_W,
    MAX  = NVCV_CHANNEL_1,
};

enum class ExtraChannel : int8_t
{
    U     = NVCV_EXTRA_CHANNEL_U,
    D     = NVCV_EXTRA_CHANNEL_D,
    POS3D = NVCV_EXTRA_CHANNEL_POS3D,
};

enum class AlphaType : int8_t
{
    ASSOCIATED   = NVCV_ALPHA_ASSOCIATED,
    UNASSOCIATED = NVCV_ALPHA_UNASSOCIATED,
};

enum class Swizzle : int32_t
{
    S_0000        = NVCV_SWIZZLE_0000,
    S_1000        = NVCV_SWIZZLE_1000,
    S_0001        = NVCV_SWIZZLE_0001,
    S_XYZW        = NVCV_SWIZZLE_XYZW,
    S_ZYXW        = NVCV_SWIZZLE_ZYXW,
    S_WXYZ        = NVCV_SWIZZLE_WXYZ,
    S_WZYX        = NVCV_SWIZZLE_WZYX,
    S_YZWX        = NVCV_SWIZZLE_YZWX,
    S_XYZ1        = NVCV_SWIZZLE_XYZ1,
    S_XYZ0        = NVCV_SWIZZLE_XYZ0,
    S_YZW1        = NVCV_SWIZZLE_YZW1,
    S_XXX1        = NVCV_SWIZZLE_XXX1,
    S_XZY1        = NVCV_SWIZZLE_XZY1,
    S_ZYX1        = NVCV_SWIZZLE_ZYX1,
    S_ZYX0        = NVCV_SWIZZLE_ZYX0,
    S_WZY1        = NVCV_SWIZZLE_WZY1,
    S_X000        = NVCV_SWIZZLE_X000,
    S_0X00        = NVCV_SWIZZLE_0X00,
    S_00X0        = NVCV_SWIZZLE_00X0,
    S_000X        = NVCV_SWIZZLE_000X,
    S_Y000        = NVCV_SWIZZLE_Y000,
    S_0Y00        = NVCV_SWIZZLE_0Y00,
    S_00Y0        = NVCV_SWIZZLE_00Y0,
    S_000Y        = NVCV_SWIZZLE_000Y,
    S_0XY0        = NVCV_SWIZZLE_0XY0,
    S_XXXY        = NVCV_SWIZZLE_XXXY,
    S_YYYX        = NVCV_SWIZZLE_YYYX,
    S_0YX0        = NVCV_SWIZZLE_0YX0,
    S_X00Y        = NVCV_SWIZZLE_X00Y,
    S_Y00X        = NVCV_SWIZZLE_Y00X,
    S_X001        = NVCV_SWIZZLE_X001,
    S_XY01        = NVCV_SWIZZLE_XY01,
    S_XY00        = NVCV_SWIZZLE_XY00,
    S_0XZ0        = NVCV_SWIZZLE_0XZ0,
    S_0ZX0        = NVCV_SWIZZLE_0ZX0,
    S_XZY0        = NVCV_SWIZZLE_XZY0,
    S_YZX1        = NVCV_SWIZZLE_YZX1,
    S_ZYW1        = NVCV_SWIZZLE_ZYW1,
    S_0YX1        = NVCV_SWIZZLE_0YX1,
    S_XYXZ        = NVCV_SWIZZLE_XYXZ,
    S_YXZX        = NVCV_SWIZZLE_YXZX,
    S_XZ00        = NVCV_SWIZZLE_XZ00,
    S_WYXZ        = NVCV_SWIZZLE_WYXZ,
    S_YX00        = NVCV_SWIZZLE_YX00,
    S_YX01        = NVCV_SWIZZLE_YX01,
    S_00YX        = NVCV_SWIZZLE_00YX,
    S_00XY        = NVCV_SWIZZLE_00XY,
    S_0XY1        = NVCV_SWIZZLE_0XY1,
    S_0X01        = NVCV_SWIZZLE_0X01,
    S_YZXW        = NVCV_SWIZZLE_YZXW,
    S_YW00        = NVCV_SWIZZLE_YW00,
    S_XYW0        = NVCV_SWIZZLE_XYW0,
    S_YZW0        = NVCV_SWIZZLE_YZW0,
    S_YZ00        = NVCV_SWIZZLE_YZ00,
    S_00X1        = NVCV_SWIZZLE_00X1,
    S_0ZXY        = NVCV_SWIZZLE_0ZXY,
    S_UNSUPPORTED = NVCV_SWIZZLE_UNSUPPORTED
};

inline Swizzle MakeSwizzle(Channel x, Channel y, Channel z, Channel w)
{
    NVCVSwizzle out;
    detail::CheckThrow(nvcvMakeSwizzle(&out, static_cast<NVCVChannel>(x), static_cast<NVCVChannel>(y),
                                       static_cast<NVCVChannel>(z), static_cast<NVCVChannel>(w)));
    return static_cast<Swizzle>(out);
}

inline std::array<Channel, 4> GetChannels(Swizzle swizzle)
{
    NVCVChannel channels[4];
    detail::CheckThrow(nvcvSwizzleGetChannels(static_cast<NVCVSwizzle>(swizzle), channels));

    return {static_cast<Channel>(channels[0]), static_cast<Channel>(channels[1]), static_cast<Channel>(channels[2]),
            static_cast<Channel>(channels[3])};
}

inline int32_t GetNumChannels(Swizzle swizzle)
{
    int32_t out;
    detail::CheckThrow(nvcvSwizzleGetNumChannels(static_cast<NVCVSwizzle>(swizzle), &out));
    return out;
}

#ifdef BIG_ENDIAN
#    undef BIG_ENDIAN
#endif

enum class ByteOrder : int8_t
{
    LSB,
    MSB
};

struct PackingParams
{
    ByteOrder byteOrder;
    Swizzle   swizzle;

    std::array<int32_t, NVCV_MAX_CHANNEL_COUNT> bits;
};

inline Packing MakePacking(const PackingParams &params)
{
    NVCVPackingParams p{};
    p.byteOrder = static_cast<NVCVByteOrder>(params.byteOrder);
    p.swizzle   = static_cast<NVCVSwizzle>(params.swizzle);
    for (size_t i = 0; i < params.bits.size(); ++i)
    {
        p.bits[i] = params.bits[i];
    }

    NVCVPacking out{};
    detail::CheckThrow(nvcvMakePacking(&out, &p));
    return static_cast<Packing>(out);
};

inline PackingParams GetParams(Packing packing)
{
    NVCVPackingParams p;
    detail::CheckThrow(nvcvPackingGetParams(static_cast<NVCVPacking>(packing), &p));

    PackingParams out;
    out.byteOrder = static_cast<ByteOrder>(p.byteOrder);
    out.swizzle   = static_cast<Swizzle>(p.swizzle);

    for (size_t i = 0; i < out.bits.size(); ++i)
    {
        out.bits[i] = p.bits[i];
    }
    return out;
}

inline int32_t GetNumComponents(Packing packing)
{
    int32_t out;
    detail::CheckThrow(nvcvPackingGetNumComponents(static_cast<NVCVPacking>(packing), &out));
    return out;
}

inline std::array<int32_t, 4> GetBitsPerComponent(Packing packing)
{
    int32_t bits[4];
    detail::CheckThrow(nvcvPackingGetBitsPerComponent(static_cast<NVCVPacking>(packing), bits));
    return {bits[0], bits[1], bits[2], bits[3]};
}

inline int32_t GetBitsPerPixel(Packing packing)
{
    int32_t out;
    detail::CheckThrow(nvcvPackingGetBitsPerPixel(static_cast<NVCVPacking>(packing), &out));
    return out;
}

inline int32_t GetAlignment(Packing packing)
{
    int32_t out;
    detail::CheckThrow(nvcvPackingGetAlignment(static_cast<NVCVPacking>(packing), &out));
    return out;
}

inline std::ostream &operator<<(std::ostream &out, DataKind dataKind)
{
    return out << nvcvDataKindGetName(static_cast<NVCVDataKind>(dataKind));
}

inline std::ostream &operator<<(std::ostream &out, Packing packing)
{
    return out << nvcvPackingGetName(static_cast<NVCVPacking>(packing));
}

inline std::ostream &operator<<(std::ostream &out, MemLayout memLayout)
{
    return out << nvcvMemLayoutGetName(static_cast<NVCVMemLayout>(memLayout));
}

inline std::ostream &operator<<(std::ostream &out, Channel swizzleChannel)
{
    return out << nvcvChannelGetName(static_cast<NVCVChannel>(swizzleChannel));
}

inline std::ostream &operator<<(std::ostream &out, Swizzle swizzle)
{
    return out << nvcvSwizzleGetName(static_cast<NVCVSwizzle>(swizzle));
}

inline std::ostream &operator<<(std::ostream &out, ByteOrder byteOrder)
{
    return out << nvcvByteOrderGetName(static_cast<NVCVByteOrder>(byteOrder));
}

} // namespace nvcv

inline std::ostream &operator<<(std::ostream &out, NVCVDataKind dataKind)
{
    return out << nvcvDataKindGetName(dataKind);
}

inline std::ostream &operator<<(std::ostream &out, NVCVPacking packing)
{
    return out << nvcvPackingGetName(packing);
}

inline std::ostream &operator<<(std::ostream &out, NVCVMemLayout memLayout)
{
    return out << nvcvMemLayoutGetName(memLayout);
}

inline std::ostream &operator<<(std::ostream &out, NVCVChannel swizzleChannel)
{
    return out << nvcvChannelGetName(swizzleChannel);
}

inline std::ostream &operator<<(std::ostream &out, NVCVSwizzle swizzle)
{
    return out << nvcvSwizzleGetName(swizzle);
}

inline std::ostream &operator<<(std::ostream &out, NVCVByteOrder byteOrder)
{
    return out << nvcvByteOrderGetName(byteOrder);
}

/**@}*/

#endif // NVCV_DATALAYOUT_HPP
