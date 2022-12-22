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

#include "DataLayout.hpp"

#include "Bitfield.hpp"
#include "Exception.hpp"
#include "TLS.hpp"

#include <util/Assert.h>
#include <util/Math.hpp>
#include <util/String.hpp>

#include <map>

//                    |63 62 61|60 59 58|57 56 55|54|53 52 51|50 49 48|47|46 45 44|43 42 41 40|39 38|37 36 35|
//                    |DataKind|  BPP3  |  BPP2  +C2+ Pack2  |  BPP1  +C1+ Pack1  |    BPP0   +Chan0+  Pack0 |
//
//  |34 33 32 31 30 29 28 27 26 25 24 23 22 21 20|19 18 17|16|15|14 13 12|11 10 09|08 07 06|05 04 03|02 01 00|
//  |14|13 12|11 10|09 08 07|06 05 04 03|02 01 00|
//  |RG+CLocV+CLocH+YCbCrEnc+ XferFunc  + CSpace |  CSS   | 0|  | MemLay |SwizzleW|SwizzleZ|SwizzleY|SwizzleX|
//  |RG+CLocV+CLocH+        + XferFunc  + CSpace |RGB,HSV,| 1|
//  |                       |   Raw Pattern   | 0| 1  1  1| 1|
//  |                       | XYZ,LAB,LUV,... | 1| 1  1  1| 1|
//  | 1  1  1  1  1  1  1  UNDEFINED   1  1  1  1  1  1  1  1|

namespace nvcv::priv {

namespace {

struct PackingData
{
    const char       *name;
    NVCVPackingParams params;
};

bool operator<(const PackingData &a, const PackingData &b)
{
    // Not comparing alignment on purpose. User might not know it.
    if (a.params.swizzle == b.params.swizzle)
    {
        for (int i = 0; i < 4; ++i)
        {
            if (a.params.bits[i] != b.params.bits[i])
            {
                return a.params.bits[i] < b.params.bits[i];
            }
        }
        return false;
    }
    else
    {
        return a.params.swizzle < b.params.swizzle;
    }
}

#define STRINGIZE(x) #x

const std::map<NVCVPacking, PackingData> g_packingToData = {
#define DEF_PACK1(x)                                                                     \
    {                                                                                    \
        NVCV_PACKING_X##x,                                                               \
        {                                                                                \
            STRINGIZE(NVCV_PACKING_X##x),                                                \
            {                                                                            \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo(x / 8), NVCV_SWIZZLE_0000, x \
            }                                                                            \
        }                                                                                \
    }

#define DEF_PACK2(x, y)                                                                           \
    {                                                                                             \
        NVCV_PACKING_X##x##Y##y,                                                                  \
        {                                                                                         \
            STRINGIZE(NVCV_PACKING_X##x##Y##y),                                                   \
            {                                                                                     \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((x + y) / 8), NVCV_SWIZZLE_0000, x, y \
            }                                                                                     \
        }                                                                                         \
    }

#define DEF_PACK3(x, y, z)                                                                               \
    {                                                                                                    \
        NVCV_PACKING_X##x##Y##y##Z##z,                                                                   \
        {                                                                                                \
            STRINGIZE(NVCV_PACKING_X##x##Y##y##Z##z),                                                    \
            {                                                                                            \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((x + y + z) / 8), NVCV_SWIZZLE_0000, x, y, z \
            }                                                                                            \
        }                                                                                                \
    }

#define DEF_PACK4(x, y, z, w)                                                                                   \
    {                                                                                                           \
        NVCV_PACKING_X##x##Y##y##Z##z##W##w,                                                                    \
        {                                                                                                       \
            STRINGIZE(NVCV_PACKING_X##x##Y##y##Z##z##W##w),                                                     \
            {                                                                                                   \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((x + y + z + w) / 8), NVCV_SWIZZLE_0000, x, y, z, w \
            }                                                                                                   \
        }                                                                                                       \
    }

#define DEF_FIX_PACK2(x, y)                                                                                            \
    {                                                                                                                  \
        NVCV_PACKING_X##x##_Y##y,                                                                                      \
        {                                                                                                              \
            STRINGIZE(NVCV_PACKING_X##x##_Y##y),                                                                       \
            {                                                                                                          \
                NVCV_ORDER_MSB, util::RoundUpNextPowerOfTwo((x == y) ? (x / 8) : (x + y) / 8), NVCV_SWIZZLE_0000, x, y \
            }                                                                                                          \
        }                                                                                                              \
    }
#define DEF_FIX_PACK3(x, y, z)                                                                               \
    {                                                                                                        \
        NVCV_PACKING_X##x##_Y##y##_Z##z,                                                                     \
        {                                                                                                    \
            STRINGIZE(NVCV_PACKING_X##x##_Y##y##_Z##z),                                                      \
            {                                                                                                \
                NVCV_ORDER_MSB, util::RoundUpNextPowerOfTwo((x == y && y == z) ? (x / 8) : (x + y + z) / 8), \
                    NVCV_SWIZZLE_0000, x, y, z                                                               \
            }                                                                                                \
        }                                                                                                    \
    }

#define DEF_FIX_PACK4(x, y, z, w)                                                                              \
    {                                                                                                          \
        NVCV_PACKING_X##x##_Y##y##_Z##z##_W##w,                                                                \
        {                                                                                                      \
            STRINGIZE(NVCV_PACKING_X##x##_Y##y##_Z##z##_W##w),                                                 \
            {                                                                                                  \
                NVCV_ORDER_MSB,                                                                                \
                    util::RoundUpNextPowerOfTwo((x == y && y == z && z == w) ? (x / 8) : (x + y + z + w) / 8), \
                    NVCV_SWIZZLE_0000, x, y, z, w                                                              \
            }                                                                                                  \
        }                                                                                                      \
    }

#define DEF_MSB_PACK1(x, bx)                                                                        \
    {                                                                                               \
        NVCV_PACKING_X##x##b##bx,                                                                   \
        {                                                                                           \
            STRINGIZE(NVCV_PACKING_X##x##b##bx),                                                    \
            {                                                                                       \
                NVCV_ORDER_MSB, util::RoundUpNextPowerOfTwo((x + bx) / 8), NVCV_SWIZZLE_X000, x, bx \
            }                                                                                       \
        }                                                                                           \
    }

#define DEF_LSB_PACK1(bx, x)                                                                        \
    {                                                                                               \
        NVCV_PACKING_b##bx##X##x,                                                                   \
        {                                                                                           \
            STRINGIZE(NVCV_PACKING_b##bx##X##x),                                                    \
            {                                                                                       \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((x + bx) / 8), NVCV_SWIZZLE_Y000, bx, x \
            }                                                                                       \
        }                                                                                           \
    }

#define DEF_FIX_MSB_PACK2(x, bx, y, by)                                                                       \
    {                                                                                                         \
        NVCV_PACKING_X##x##b##bx##_Y##y##b##by,                                                               \
        {                                                                                                     \
            STRINGIZE(NVCV_PACKING_X##x##b##bx##_Y##y##b##by),                                                \
            {                                                                                                 \
                NVCV_ORDER_MSB,                                                                               \
                    util::RoundUpNextPowerOfTwo((x + bx) == (y + by) ? (x + bx) / 8 : (x + bx + y + by) / 8), \
                    NVCV_SWIZZLE_XZ00, x, bx, y, by                                                           \
            }                                                                                                 \
        }                                                                                                     \
    }

#define DEF_FIX_LSB_PACK2(bx, x, by, y)                                                                       \
    {                                                                                                         \
        NVCV_PACKING_b##bx##X##x##_Y##y##b##by,                                                               \
        {                                                                                                     \
            STRINGIZE(NVCV_PACKING_b##bx##X##x##_Y##y##b##by),                                                \
            {                                                                                                 \
                NVCV_ORDER_LSB,                                                                               \
                    util::RoundUpNextPowerOfTwo((x + bx) == (y + by) ? (x + bx) / 8 : (x + bx + y + by) / 8), \
                    NVCV_SWIZZLE_YW00, bx, x, by, y                                                           \
            }                                                                                                 \
        }                                                                                                     \
    }

#define DEF_LSB_PACK3(bx, x, y, z)                                                                                \
    {                                                                                                             \
        NVCV_PACKING_b##bx##X##x##Y##y##Z##z,                                                                     \
        {                                                                                                         \
            STRINGIZE(NVCV_PACKING_b##bx##X##x##Y##y##Z##z),                                                      \
            {                                                                                                     \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((bx + x + y + z) / 8), NVCV_SWIZZLE_YZW0, bx, x, y, z \
            }                                                                                                     \
        }                                                                                                         \
    }

#define DEF_MSB_PACK3(x, y, z, bz)                                                                                \
    {                                                                                                             \
        NVCV_PACKING_X##x##Y##y##Z##z##b##bz,                                                                     \
        {                                                                                                         \
            STRINGIZE(NVCV_PACKING_b##bx##X##x##Y##y##Z##z),                                                      \
            {                                                                                                     \
                NVCV_ORDER_MSB, util::RoundUpNextPowerOfTwo((x + y + z + bz) / 8), NVCV_SWIZZLE_XYZ0, x, y, z, bz \
            }                                                                                                     \
        }                                                                                                         \
    }

#define DEF_LSB_PACK4(x, y, z, w)                                                                               \
    {                                                                                                           \
        NVCV_PACKING_X##x##Y##y##Z##z##W##w,                                                                    \
        {                                                                                                       \
            STRINGIZE(NVCV_PACKING_X##x##Y##y##Z##z##W##w),                                                     \
            {                                                                                                   \
                NVCV_ORDER_LSB, util::RoundUpNextPowerOfTwo((x + y + z + w) / 8), NVCV_SWIZZLE_0000, x, y, z, w \
            }                                                                                                   \
        }                                                                                                       \
    }

    DEF_PACK1(1),

    DEF_PACK1(2),

    DEF_PACK1(4),

    DEF_PACK1(8),
    DEF_LSB_PACK1(4, 4),
    DEF_MSB_PACK1(4, 4),

    DEF_PACK2(4, 4),
    DEF_PACK3(3, 3, 2),

    DEF_PACK1(16),
    DEF_FIX_PACK2(8, 8),
    DEF_LSB_PACK1(6, 10),
    DEF_MSB_PACK1(10, 6),
    DEF_LSB_PACK1(2, 14),
    DEF_MSB_PACK1(14, 2),
    DEF_MSB_PACK1(12, 4),
    DEF_LSB_PACK1(4, 12),
    DEF_PACK3(5, 5, 6),
    DEF_PACK3(5, 6, 5),
    DEF_PACK3(6, 5, 5),
    DEF_LSB_PACK3(4, 4, 4, 4),
    DEF_LSB_PACK4(4, 4, 4, 4),
    DEF_LSB_PACK3(1, 5, 5, 5),
    DEF_PACK4(1, 5, 5, 5),
    DEF_PACK4(5, 1, 5, 5),
    DEF_PACK4(5, 5, 1, 5),
    DEF_PACK4(5, 5, 5, 1),

    DEF_PACK1(24),
    DEF_FIX_PACK3(8, 8, 8),

    DEF_PACK1(32),
    DEF_FIX_PACK2(16, 16),
    DEF_MSB_PACK1(20, 12),
    DEF_LSB_PACK1(12, 20),
    DEF_MSB_PACK1(24, 8),
    DEF_LSB_PACK1(8, 24),
    DEF_PACK3(10, 11, 11),
    DEF_PACK3(11, 11, 10),
    DEF_PACK4(2, 10, 10, 10),
    DEF_FIX_PACK4(8, 8, 8, 8),
    DEF_FIX_MSB_PACK2(10, 6, 10, 6),
    DEF_PACK4(10, 10, 10, 2),
    DEF_FIX_MSB_PACK2(12, 4, 12, 4),

    DEF_LSB_PACK3(2, 10, 10, 10),
    DEF_MSB_PACK3(10, 10, 10, 2),

    DEF_PACK1(48),
    DEF_FIX_PACK3(16, 16, 16),

    DEF_PACK1(64),
    DEF_FIX_PACK2(32, 32),
    DEF_FIX_PACK4(16, 16, 16, 16),

    DEF_PACK1(96),
    DEF_FIX_PACK3(32, 32, 32),

    DEF_PACK1(128),
    DEF_FIX_PACK2(64, 64),
    DEF_FIX_PACK4(32, 32, 32, 32),

    DEF_PACK1(192),
    DEF_FIX_PACK3(64, 64, 64),

    DEF_PACK1(256),
    DEF_FIX_PACK4(64, 64, 64, 64),

// clang-format-14.0.6 segfaults on the lines below. We have to
// declare them as a macro and use them. It segfaults even if we
// mark them with clang-format off
// clang-format off
#define CLANGFORMAT_WAR                                                             \
    {NVCV_PACKING_0,            {"NVCV_PACKING_0",            {NVCV_ORDER_LSB, 0,NVCV_SWIZZLE_0000}}},          \
    {NVCV_PACKING_X8_Y8__X8_Z8, {"NVCV_PACKING_X8_Y8__X8_Z8", {NVCV_ORDER_MSB, 1, NVCV_SWIZZLE_XYXZ, 8, 8, 8, 8}}}, \
    {NVCV_PACKING_Y8_X8__Z8_X8, {"NVCV_PACKING_Y8_X8__Z8_X8", {NVCV_ORDER_MSB, 1, NVCV_SWIZZLE_YXZX, 8, 8, 8, 8}}}, \
    {NVCV_PACKING_X5Y5b1Z5,     {"NVCV_PACKING_X5Y5b1Z5",     {NVCV_ORDER_LSB, 2,NVCV_SWIZZLE_XYW0, 5, 5, 1, 5}}}, \
    {NVCV_PACKING_X32_Y24b8,    {"NVCV_PACKING_X32_Y24b8",     {NVCV_ORDER_MSB,4, NVCV_SWIZZLE_XY00, 32, 24, 8, 0}}}
    CLANGFORMAT_WAR,
    // clang-format on
};

const std::multimap<PackingData, NVCVPacking> g_dataToPacking = []
{
    std::multimap<PackingData, NVCVPacking> map;

    for (const auto &item : g_packingToData)
    {
        map.emplace(item.second, item.first);
    }

    return map;
}();

} // namespace

std::optional<NVCVPacking> MakeNVCVPacking(int bitsX, int bitsY, int bitsZ, int bitsW) noexcept
{
    NVCVPackingParams params = {};

    if (bitsY == 0 || bitsZ == 0)
    {
        int origX = bitsX, origY = bitsY;

        // We use MSB representation, e.g. X10b6
        switch (origX)
        {
        case 10:
        case 12:
        case 14:
            bitsY          = 16 - origX;
            params.swizzle = NVCV_SWIZZLE_X000;
            break;
        case 20:
            bitsY          = 32 - origX;
            params.swizzle = NVCV_SWIZZLE_X000;
            break;
        }

        switch (origY)
        {
        case 10:
        case 12:
        case 14:
            bitsZ          = 16 - origY;
            params.swizzle = NVCV_SWIZZLE_XZ00;
            break;
        case 20:
            bitsZ          = 32 - origY;
            params.swizzle = NVCV_SWIZZLE_XZ00;
            break;
        }
    }

    params.bits[0] = bitsX;
    params.bits[1] = bitsY;
    params.bits[2] = bitsZ;
    params.bits[3] = bitsW;

    return MakeNVCVPacking(params);
}

static NVCVSwizzle MakeNVCVSwizzleFromBits(const int (&bits)[4])
{
    NVCVChannel swc[4];
    for (int i = 0; i < 4; ++i)
    {
        swc[i] = bits[i] != 0 ? (NVCVChannel)(i + NVCV_CHANNEL_X) : NVCV_CHANNEL_0;
    }
    return MakeNVCVSwizzle(swc[0], swc[1], swc[2], swc[3]);
}

std::optional<NVCVPacking> MakeNVCVPacking(const NVCVPackingParams &params) noexcept
{
    PackingData key;
    key.params = params;

    // Normalize swizzle
    if (key.params.swizzle != NVCV_SWIZZLE_XYXZ && key.params.swizzle != NVCV_SWIZZLE_YXZX && !IsSubWord(key.params))
    {
        key.params.swizzle = NVCV_SWIZZLE_0000;
    }

    auto [itbegin, itend] = g_dataToPacking.equal_range(key);
    if (itbegin != itend)
    {
        auto it = itbegin;
        if (params.alignment == 0)
        {
            for (; itbegin != itend; ++itbegin)
            {
                // choose smallest alignment
                if (itbegin->first.params.alignment < it->first.params.alignment)
                {
                    it = itbegin;
                }
            }
        }
        else
        {
            for (; itbegin != itend; ++itbegin)
            {
                // Smaller alignments are valid.
                if (it->first.params.alignment >= itbegin->first.params.alignment)
                {
                    it = itbegin;
                    break;
                }
            }
            // packing with needed alignment not found
            if (itbegin == itend)
            {
                return std::nullopt;
            }
        }

        // if 0 or one channel, packing is both host and big endian, so don't need to filter out.
        if (GetNumChannels(params.swizzle) >= 2)
        {
            // Endian don't match?
            if (it->first.params.byteOrder != params.byteOrder)
            {
                return std::nullopt;
            }
        }

        // use filters by swizzle?
        if (params.swizzle != NVCV_SWIZZLE_0000)
        {
            // If our swizzle is not specified, let's reconstruct it from bits
            NVCVSwizzle sw = it->first.params.swizzle;
            if (sw == NVCV_SWIZZLE_0000)
            {
                sw = MakeNVCVSwizzleFromBits(params.bits);
            }

            // now we can apply the filter.
            if (sw != params.swizzle)
            {
                return std::nullopt;
            }
        }

        return it->second;
    }
    else
    {
        return std::nullopt;
    }
}

NVCVSwizzle MakeNVCVSwizzle(NVCVChannel x, NVCVChannel y, NVCVChannel z, NVCVChannel w) noexcept
{
    return NVCV_MAKE_SWIZZLE(x, y, z, w);
}

bool IsSubWord(const NVCVPackingParams &p)
{
    switch (p.swizzle)
    {
    case NVCV_SWIZZLE_0000:
    case NVCV_SWIZZLE_XYXZ:
    case NVCV_SWIZZLE_YXZX:
        return false;
    default:
        break;
    }

    int chbits = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (p.bits[i] != 0)
        {
            chbits += 1;
        }
    }

    return GetNumChannels(p.swizzle) != chbits;
}

NVCVPackingParams GetPackingParams(NVCVPacking packing) noexcept
{
    auto it = g_packingToData.find(packing);
    NVCV_ASSERT(it != g_packingToData.end());

    NVCVPackingParams params = it->second.params;
    if (params.swizzle == NVCV_SWIZZLE_0000)
    {
        params.swizzle = MakeNVCVSwizzleFromBits(params.bits);
    }

    return params;
}

int GetBitsPerPixel(NVCVPacking packing) noexcept
{
    int bpp = packing >> 6;

    // If bpp <= 8, we store it in the packing. This limits
    // the number of packings there can be (LSB bits), but this number is
    // naturally limited.
    if (bpp == 0)
    {
        int code = packing & 0b1111;

        if (code == 0)
        {
            NVCV_ASSERT(bpp == 0);
            NVCV_ASSERT(packing == NVCV_PACKING_0);
        }
        else if (code <= 2)
        {
            bpp = code;
        }
        else if (code == 3)
        {
            bpp = 4;
        }
        else if (code <= 0b111)
        {
            bpp = 8;
        }
        else
        {
            // invalid
            return 0;
        }
    }
    else
    {
        // Now we decode like this:
        // 1 -> 16 bpp
        // 2 -> 24 bpp
        // 3 -> 32 bpp
        //
        // 4 -> 48 bpp
        // 5 -> 64 bpp
        //
        // 6 -> 96 bpp
        // 7 -> 128 bpp
        //
        // 8 -> 192 bpp
        // 9 -> 256 bpp

        if (bpp <= 3)
        {
            bpp = (bpp + 1) * 8;
        }
        else if (bpp <= 5)
        {
            bpp = (bpp - 1) * 16;
        }
        else if (bpp <= 7)
        {
            bpp = (bpp - 3) * 32;
        }
        else if (bpp <= 9)
        {
            bpp = (bpp - 5) * 64;
        }
        else
        {
            // invalid;
            return 0;
        }
    }

    return bpp;
}

NVCVChannel GetSwizzleChannel(NVCVSwizzle swizzle, int idx) noexcept
{
    return (NVCVChannel)ExtractBitfield(swizzle, idx * 3, 3);
}

std::array<NVCVChannel, 4> GetChannels(NVCVSwizzle swizzle) noexcept
{
    std::array<NVCVChannel, 4> channels;

    for (int i = 0; i < 4; ++i)
    {
        channels[i] = GetSwizzleChannel(swizzle, i);
    }

    return channels;
}

int GetNumChannels(NVCVSwizzle swizzle) noexcept
{
    std::array<NVCVChannel, 4> channels = GetChannels(swizzle);

    int hist[4] = {};

    int count = 0;
    for (int i = 0; i < 4; ++i)
    {
        // only X,Y,Z,W count as channel
        if (NVCV_CHANNEL_X <= channels[i] && channels[i] <= NVCV_CHANNEL_W)
        {
            int idx = channels[i] - NVCV_CHANNEL_X;

            if (hist[idx] == 0)
            {
                ++count;
            }
            hist[idx] += 1;
        }
    }

    return count;
}

int GetBlockHeightLog2(NVCVMemLayout memLayout) noexcept
{
    NVCV_ASSERT(NVCV_MEM_LAYOUT_BLOCK1_LINEAR <= memLayout && memLayout <= NVCV_MEM_LAYOUT_BLOCK32_LINEAR);
    return memLayout - NVCV_MEM_LAYOUT_BLOCK1_LINEAR;
}

int GetNumComponents(NVCVPacking packing) noexcept
{
    if (packing == NVCV_PACKING_0)
    {
        return 0;
    }
    else
    {
        return ExtractBitfield(packing, 4, 2) + 1;
    }
}

int GetNumChannels(NVCVPacking packing) noexcept
{
    int cnt = GetNumComponents(packing);

    switch (packing)
    {
    case NVCV_PACKING_X8_Y8__X8_Z8:
    case NVCV_PACKING_Y8_X8__Z8_X8:
        NVCV_ASSERT(cnt == 4);
        cnt = 3;
        break;
    default:
        break;
    }
    return cnt;
}

std::array<int32_t, 4> GetBitsPerComponent(NVCVPacking packing) noexcept
{
    std::array<int32_t, 4> bits;

    NVCVPackingParams p = GetPackingParams(packing);

    int ncomp = GetNumComponents(packing);

    std::array<NVCVChannel, 4> channels = GetChannels(p.swizzle);

    int i;
    for (i = 0; i < ncomp; ++i)
    {
        bits[i] = p.bits[channels[i] - NVCV_CHANNEL_X];
    }
    for (; i < 4; ++i)
    {
        bits[i] = 0;
    }

    return bits;
}

int GetAlignment(NVCVPacking packing) noexcept
{
    return GetPackingParams(packing).alignment;
}

NVCVSwizzle MergePlaneSwizzles(NVCVSwizzle sw0, NVCVSwizzle sw1, NVCVSwizzle sw2, NVCVSwizzle sw3)
{
    // just one plane?
    if (sw1 == NVCV_SWIZZLE_0000)
    {
        if (sw2 == NVCV_SWIZZLE_0000 && sw3 == NVCV_SWIZZLE_0000)
        {
            return sw0;
        }
        else
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                            "When specifying only swizzle for plane #0, swizzle for remaining planes must be 0000");
        }
    }

    NVCVSwizzle sw[4] = {sw0, sw1, sw2, sw3};

    NVCVChannel swResult[4] = {};

    int curch = 0;
    for (int i = 0; i < 4 && sw[i] != NVCV_SWIZZLE_0000; ++i)
    {
        int nchannels = GetNumChannels(sw[i]);

        NVCVChannel rev[6] = {};

        NVCVChannel maxSwChannel = NVCV_CHANNEL_0;

        for (int j = 0; j < 4; ++j)
        {
            NVCVChannel swch = GetSwizzleChannel(sw[i], j);
            if (NVCV_CHANNEL_X <= swch && swch <= NVCV_CHANNEL_W)
            {
                maxSwChannel = std::max(maxSwChannel, swch);
            }
        }

        for (int j = 0; j < 4; ++j)
        {
            NVCVChannel ch = GetSwizzleChannel(sw[i], j);
            if (ch == NVCV_CHANNEL_1)
            {
                ch = (NVCVChannel)(maxSwChannel + 1);
                // you can't specify W and also have 1 in the swizzle
                if (ch >= NVCV_CHANNEL_1)
                {
                    throw Exception(
                        NVCV_ERROR_INVALID_ARGUMENT,
                        "When swizzle has W channel, it must not have channel with maximum value (channel '1')");
                }
                rev[ch] = NVCV_CHANNEL_1;
            }
            else
            {
                rev[ch] = (NVCVChannel)(NVCV_CHANNEL_X + j);
            }
        }

        for (int j = 0; j < nchannels; ++j)
        {
            NVCV_ASSERT(curch + j < 4);

            NVCVChannel ch = rev[j + NVCV_CHANNEL_X];

            // assuming that alpha always go to 4th color model channel channel.
            if (ch == NVCV_CHANNEL_1 && j == nchannels - 1)
            {
                swResult[3] = ch;
            }
            else
            {
                swResult[curch + j] = ch;
            }
        }

        curch += nchannels;
    }

    // Fix up for '1' alpha channel. If any plane has a swizzle with '1',
    // the resulting swizzle must have too.
    for (int i = 0; i < 4; ++i)
    {
        if (GetSwizzleChannel(sw[i], 3) == NVCV_CHANNEL_1)
        {
            NVCV_ASSERT(swResult[3] == NVCV_CHANNEL_0);
            swResult[3] = NVCV_CHANNEL_1;
            break;
        }
    }

    return MakeNVCVSwizzle(swResult[0], swResult[1], swResult[2], swResult[3]);
}

NVCVSwizzle FlipByteOrder(NVCVSwizzle swizzle, int off, int len) noexcept
{
    // there's nothing to flip for 0 or 1 channels.
    if (len <= 1)
    {
        return swizzle;
    }

    std::array<NVCVChannel, 4> comp = priv::GetChannels(swizzle);

    NVCV_ASSERT(off >= 0);
    NVCV_ASSERT(off + len <= 4);

    // Flipping byteOrder must occur at memory space, not component space.
    // So first map swizzle to memory space, i.e., sort components in order
    // they will show up in memory, from lowest address to highest.

    NVCVChannel mem[4] = {};
    int         m = INT32_MAX, M = INT32_MIN;

    for (int i = 0; i < 4; ++i)
    {
        if (i >= off && i < off + len && NVCV_CHANNEL_X <= comp[i] && comp[i] <= NVCV_CHANNEL_W)
        {
            int pos  = comp[i] - NVCV_CHANNEL_X;
            m        = std::min(m, pos);
            M        = std::max(M, pos);
            mem[pos] = (NVCVChannel)(i + NVCV_CHANNEL_X);
        }
    }

    // Now flip in memory space
    NVCVChannel flipped[4] = {};
    for (int i = m; i <= M; ++i)
    {
        NVCVChannel ch = mem[m + ((M - m) - (i - m))];
        NVCV_ASSERT(NVCV_CHANNEL_X <= ch && ch <= NVCV_CHANNEL_W);
        flipped[i] = ch;
    }

    // Now map the flipped memory back to component space
    for (int i = 0; i < 4; ++i)
    {
        if (NVCV_CHANNEL_X <= flipped[i] && flipped[i] <= NVCV_CHANNEL_W)
        {
            comp[flipped[i] - NVCV_CHANNEL_X] = (NVCVChannel)(i + NVCV_CHANNEL_X);
        }
    }

    // assemble the resulting swizzle.
    return NVCV_MAKE_SWIZZLE(comp[0], comp[1], comp[2], comp[3]);
}

const char *GetName(NVCVDataKind dataKind)
{
    switch (dataKind)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_DATA_KIND_UNSIGNED);
        ENUM_CASE(NVCV_DATA_KIND_SIGNED);
        ENUM_CASE(NVCV_DATA_KIND_FLOAT);
#undef ENUM_CASE
    }
    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufDataKindName, sizeof(tls.bufDataKindName)) << "NVCVDataKind(" << (int)dataKind << ")";
    return tls.bufDataKindName;
}

const char *GetName(NVCVMemLayout memLayout)
{
    switch (memLayout)
    {
#define ENUM_CASE(X) \
    case X:          \
        return #X
        ENUM_CASE(NVCV_MEM_LAYOUT_PITCH_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK1_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK2_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK4_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK8_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK16_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK32_LINEAR);
#undef ENUM_CASE
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufMemLayoutName, sizeof(tls.bufMemLayoutName))
        << "NVCVMemLayout(" << (int)memLayout << ")";
    return tls.bufMemLayoutName;
}

const char *GetName(NVCVChannel swizzleChannel)
{
    switch (swizzleChannel)
    {
    case NVCV_CHANNEL_0:
        return "0";
    case NVCV_CHANNEL_X:
        return "X";
    case NVCV_CHANNEL_Y:
        return "Y";
    case NVCV_CHANNEL_Z:
        return "Z";
    case NVCV_CHANNEL_W:
        return "W";
    case NVCV_CHANNEL_1:
        return "1";
    case NVCV_CHANNEL_FORCE8:
        break;
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufChannelName, sizeof(tls.bufChannelName)) << "NVCVChannel(" << (int)swizzleChannel << ")";
    return tls.bufChannelName;
}

const char *GetName(NVCVSwizzle swizzle)
{
    std::array<NVCVChannel, 4> channels = priv::GetChannels(swizzle);

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufSwizzleName, sizeof(tls.bufSwizzleName))
        << channels[0] << channels[1] << channels[2] << channels[3];
    return tls.bufSwizzleName;
}

const char *GetName(NVCVByteOrder byteOrder)
{
    switch (byteOrder)
    {
    case NVCV_ORDER_LSB:
        return "LSB";
    case NVCV_ORDER_MSB:
        return "MSB";
    }

    priv::CoreTLS &tls = priv::GetCoreTLS();
    util::BufferOStream(tls.bufByteOrderName, sizeof(tls.bufByteOrderName))
        << "NVCVByteOrder(" << (int)byteOrder << ")";
    return tls.bufByteOrderName;
}

const char *GetName(NVCVPacking packing)
{
    auto it = g_packingToData.find(packing);
    if (it != g_packingToData.end())
    {
        return it->second.name;
    }
    else
    {
        priv::CoreTLS &tls = priv::GetCoreTLS();
        util::BufferOStream(tls.bufPackingName, sizeof(tls.bufPackingName)) << "NVCVPacking(" << (int)packing << ")";
        return tls.bufPackingName;
    }
}

} // namespace nvcv::priv

namespace priv = nvcv::priv;

std::ostream &operator<<(std::ostream &out, NVCVDataKind dataKind)
{
    return out << priv::GetName(dataKind);
}

std::ostream &operator<<(std::ostream &out, NVCVMemLayout memLayout)
{
    return out << priv::GetName(memLayout);
}

std::ostream &operator<<(std::ostream &out, NVCVChannel swizzleChannel)
{
    return out << priv::GetName(swizzleChannel);
}

std::ostream &operator<<(std::ostream &out, NVCVSwizzle swizzle)
{
    return out << priv::GetName(swizzle);
}

std::ostream &operator<<(std::ostream &out, NVCVPacking packing)
{
    return out << priv::GetName(packing);
}

std::ostream &operator<<(std::ostream &out, NVCVByteOrder byteOrder)
{
    return out << priv::GetName(byteOrder);
}
