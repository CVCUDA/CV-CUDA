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

#include "Definitions.hpp"

#include <nvcv/ColorSpec.h>
#include <util/Size.hpp>

#include <unordered_set>

namespace t    = ::testing;
namespace util = nvcv::util;

// Swizzle -----------------------------

class SwizzleTests
    : public t::TestWithParam<std::tuple<NVCVSwizzle, NVCVSwizzle, NVCVChannel, NVCVChannel, NVCVChannel, NVCVChannel>>
{
};

#define MAKE_SWIZZLE(x, y, z, w)                                                                               \
    std::make_tuple(NVCV_SWIZZLE_##x##y##z##w,                                                                 \
                    NVCV_MAKE_SWIZZLE(NVCV_CHANNEL_##x, NVCV_CHANNEL_##y, NVCV_CHANNEL_##z, NVCV_CHANNEL_##w), \
                    NVCV_CHANNEL_##x, NVCV_CHANNEL_##y, NVCV_CHANNEL_##z, NVCV_CHANNEL_##w)

INSTANTIATE_TEST_SUITE_P(
    Predefined, SwizzleTests,
    t::Values(MAKE_SWIZZLE(X, Y, Z, W), MAKE_SWIZZLE(Z, Y, X, W), MAKE_SWIZZLE(W, Z, Y, X), MAKE_SWIZZLE(Y, Z, W, X),
              MAKE_SWIZZLE(X, Y, Z, 1), MAKE_SWIZZLE(X, Y, Z, 0), MAKE_SWIZZLE(Y, Z, W, 1), MAKE_SWIZZLE(X, X, X, 1),
              MAKE_SWIZZLE(X, Z, Y, 1), MAKE_SWIZZLE(Z, Y, X, 1), MAKE_SWIZZLE(Z, Y, X, 0), MAKE_SWIZZLE(W, Z, Y, 1),
              MAKE_SWIZZLE(X, 0, 0, 0), MAKE_SWIZZLE(0, X, 0, 0), MAKE_SWIZZLE(0, 0, X, 0), MAKE_SWIZZLE(0, 0, 0, X),
              MAKE_SWIZZLE(Y, 0, 0, 0), MAKE_SWIZZLE(0, Y, 0, 0), MAKE_SWIZZLE(0, 0, Y, 0), MAKE_SWIZZLE(0, 0, 0, Y),
              MAKE_SWIZZLE(0, X, Y, 0), MAKE_SWIZZLE(X, X, X, Y), MAKE_SWIZZLE(Y, Y, Y, X), MAKE_SWIZZLE(0, Y, X, 0),
              MAKE_SWIZZLE(X, 0, 0, Y), MAKE_SWIZZLE(Y, 0, 0, X), MAKE_SWIZZLE(X, 0, 0, 1), MAKE_SWIZZLE(X, Y, 0, 1),
              MAKE_SWIZZLE(X, Y, 0, 0), MAKE_SWIZZLE(0, X, Z, 0), MAKE_SWIZZLE(0, Z, X, 0), MAKE_SWIZZLE(0, Y, X, 1)));

TEST_P(SwizzleTests, predefined_has_correct_definition)
{
    NVCVSwizzle gold = std::get<0>(GetParam());
    NVCVSwizzle test = NVCV_MAKE_SWIZZLE(std::get<2>(GetParam()), std::get<3>(GetParam()), std::get<4>(GetParam()),
                                         std::get<5>(GetParam()));

    EXPECT_EQ(gold, test);
}

TEST_P(SwizzleTests, make_sizzle_function_works)
{
    NVCVSwizzle gold = std::get<0>(GetParam());
    NVCVSwizzle test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeSwizzle(&test, std::get<2>(GetParam()), std::get<3>(GetParam()),
                                            std::get<4>(GetParam()), std::get<5>(GetParam())));

    EXPECT_EQ(gold, test);
}

TEST_P(SwizzleTests, make_sizzle_macro_works)
{
    NVCVSwizzle gold = std::get<0>(GetParam());
    NVCVSwizzle test = std::get<1>(GetParam());

    EXPECT_EQ(gold, test);
}

TEST(SwizzleTests, make_swizzle_macro)
{
    NVCVSwizzle swzl;
    EXPECT_EQ(NVCV_SUCCESS, nvcvMakeSwizzle(&swzl, NVCV_CHANNEL_X, NVCV_CHANNEL_W, NVCV_CHANNEL_Z, NVCV_CHANNEL_1));
    EXPECT_EQ(swzl, NVCV_MAKE_SWIZZLE(NVCV_CHANNEL_X, NVCV_CHANNEL_W, NVCV_CHANNEL_Z, NVCV_CHANNEL_1));
}

TEST_P(SwizzleTests, get_channel_channels)
{
    NVCVSwizzle swizzle = std::get<0>(GetParam());

    NVCVChannel channels[4];
    nvcvSwizzleGetChannels(swizzle, channels);

    EXPECT_EQ(std::get<2>(GetParam()), channels[0]);
    EXPECT_EQ(std::get<3>(GetParam()), channels[1]);
    EXPECT_EQ(std::get<4>(GetParam()), channels[2]);
    EXPECT_EQ(std::get<5>(GetParam()), channels[3]);
}

TEST_P(SwizzleTests, get_channel_count)
{
    NVCVSwizzle swizzle = std::get<0>(GetParam());
    NVCVChannel channels[]
        = {std::get<2>(GetParam()), std::get<3>(GetParam()), std::get<4>(GetParam()), std::get<5>(GetParam())};

    int hist[4] = {};

    int gold = 0;
    for (int i = 0; i < 4; ++i)
    {
        // only X,Y,Z,W count as channel
        if (NVCV_CHANNEL_X <= channels[i] && channels[i] <= NVCV_CHANNEL_W)
        {
            int idx = channels[i] - NVCV_CHANNEL_X;

            if (hist[idx] == 0)
            {
                ++gold;
            }
            hist[idx] += 1;
        }
    }

    int count;
    ASSERT_EQ(NVCV_SUCCESS, nvcvSwizzleGetNumChannels(swizzle, &count));
    EXPECT_EQ(gold, count);
}

// Packing -----------------------------

TEST(PackingTests, zero_format_packing_must_be_0)
{
    EXPECT_EQ(0, (int)NVCV_PACKING_0);
}

struct PackingTestParams
{
    NVCVPacking packing;

    int bitsPerComponent[4];

    NVCVPackingParams params;

    bool operator==(const PackingTestParams &that) const
    {
        // not comparing alignment on purpose
        if (params.byteOrder == that.params.byteOrder && params.swizzle == that.params.swizzle)
        {
            for (int i = 0; i < 4; ++i)
            {
                if (params.bits[i] != that.params.bits[i])
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

    friend std::ostream &operator<<(std::ostream &out, const PackingTestParams &p)
    {
        out << p.params.byteOrder;
        out << ',' << p.packing;
        out << ',' << p.params.swizzle;
        out << ",align=" << p.params.alignment;
        out << ",X" << p.params.bits[0];
        out << ",Y" << p.params.bits[1];
        out << ",Z" << p.params.bits[2];
        out << ",W" << p.params.bits[3];
        return out;
    }
};

// from boost

inline void hash_combine(std::size_t &seed) {}

template<typename T, typename... Rest>
inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

struct HashPackingTestParams
{
    size_t operator()(const PackingTestParams &p) const
    {
        size_t h = 0x123;
        // not hashing alignment on purpose
        hash_combine(h, (uint32_t)p.params.byteOrder, (uint32_t)p.params.swizzle, (unsigned char)p.params.bits[0],
                     (unsigned char)p.params.bits[1], (unsigned char)p.params.bits[2], (unsigned char)p.params.bits[3]);
        return h;
    }
};

class PackingTests : public t::TestWithParam<PackingTestParams>
{
};

#define DEF_PACK1(x)                                      \
    PackingTestParams                                     \
    {                                                     \
        NVCV_PACKING_X##x, {x},                           \
        {                                                 \
            NVCV_ORDER_LSB, (x / 8), NVCV_SWIZZLE_X000, x \
        }                                                 \
    }

#define DEF_PACK2(x, y)                                            \
    PackingTestParams                                              \
    {                                                              \
        NVCV_PACKING_X##x##Y##y, {x, y},                           \
        {                                                          \
            NVCV_ORDER_LSB, ((x + y) / 8), NVCV_SWIZZLE_XY00, x, y \
        }                                                          \
    }

#define DEF_PACK3(x, y, z)                                                \
    PackingTestParams                                                     \
    {                                                                     \
        NVCV_PACKING_X##x##Y##y##Z##z, {x, y, z},                         \
        {                                                                 \
            NVCV_ORDER_LSB, ((x + y + z) / 8), NVCV_SWIZZLE_XYZ0, x, y, z \
        }                                                                 \
    }

#define DEF_PACK4(x, y, z, w)                                                    \
    PackingTestParams                                                            \
    {                                                                            \
        NVCV_PACKING_X##x##Y##y##Z##z##W##w, {x, y, z, w},                       \
        {                                                                        \
            NVCV_ORDER_LSB, ((x + y + z + w) / 8), NVCV_SWIZZLE_XYZW, x, y, z, w \
        }                                                                        \
    }

#define DEF_FIX_PACK1(x)                                  \
    PackingTestParams                                     \
    {                                                     \
        NVCV_PACKING_X##x, {x},                           \
        {                                                 \
            NVCV_ORDER_MSB, (x / 8), NVCV_SWIZZLE_X000, x \
        }                                                 \
    }

#define DEF_FIX_PACK2(x, y)                                  \
    PackingTestParams                                        \
    {                                                        \
        NVCV_PACKING_X##x##_Y##y, {x, y},                    \
        {                                                    \
            NVCV_ORDER_MSB, (x / 8), NVCV_SWIZZLE_XY00, x, y \
        }                                                    \
    }

#define DEF_FIX_PACK3(x, y, z)                                  \
    PackingTestParams                                           \
    {                                                           \
        NVCV_PACKING_X##x##_Y##y##_Z##z, {x, y, z},             \
        {                                                       \
            NVCV_ORDER_MSB, (x / 8), NVCV_SWIZZLE_XYZ0, x, y, z \
        }                                                       \
    }

#define DEF_FIX_PACK4(x, y, z, w)                                  \
    PackingTestParams                                              \
    {                                                              \
        NVCV_PACKING_X##x##_Y##y##_Z##z##_W##w, {x, y, z, w},      \
        {                                                          \
            NVCV_ORDER_MSB, (x / 8), NVCV_SWIZZLE_XYZW, x, y, z, w \
        }                                                          \
    }

#define DEF_MSB_PACK1(x, bx)                                         \
    {                                                                \
        NVCV_PACKING_X##x##b##bx, {x},                               \
        {                                                            \
            NVCV_ORDER_MSB, ((x + bx) / 8), NVCV_SWIZZLE_X000, x, bx \
        }                                                            \
    }

#define DEF_LSB_PACK1(bx, x)                                         \
    {                                                                \
        NVCV_PACKING_b##bx##X##x, {x},                               \
        {                                                            \
            NVCV_ORDER_LSB, ((bx + x) / 8), NVCV_SWIZZLE_Y000, bx, x \
        }                                                            \
    }

#define DEF_FIX_MSB_PACK2(x, bx, y, by)                                     \
    {                                                                       \
        NVCV_PACKING_X##x##b##bx##_Y##y##b##by, {x, y},                     \
        {                                                                   \
            NVCV_ORDER_MSB, ((bx + x) / 8), NVCV_SWIZZLE_XZ00, x, bx, y, by \
        }                                                                   \
    }

#define DEF_FIX_LSB_PACK2(bx, x, by, y)                                     \
    {                                                                       \
        NVCV_PACKING_b##bx##X##x##_Y##y##b##by, {x, y},                     \
        {                                                                   \
            NVCV_ORDER_MSB, ((bx + x) / 8), NVCV_SWIZZLE_YW00, bx, x, by, y \
        }                                                                   \
    }

#define DEF_LSB_PACK3(bx, x, y, z)                                                 \
    {                                                                              \
        NVCV_PACKING_b##bx##X##x##Y##y##Z##z, {x, y, z},                           \
        {                                                                          \
            NVCV_ORDER_LSB, ((bx + x + y + z) / 8), NVCV_SWIZZLE_YZW0, bx, x, y, z \
        }                                                                          \
    }

#define DEF_MSB_PACK3(x, y, z, bz)                                                 \
    {                                                                              \
        NVCV_PACKING_X##x##Y##y##Z##z##b##bz, {x, y, z},                           \
        {                                                                          \
            NVCV_ORDER_MSB, ((x + y + z + bz) / 8), NVCV_SWIZZLE_XYZ0, x, y, z, bz \
        }                                                                          \
    }

#define DEF_LSB_PACK4(x, y, z, w)                                                \
    {                                                                            \
        NVCV_PACKING_X##x##Y##y##Z##z##W##w, {x, y, z, w},                       \
        {                                                                        \
            NVCV_ORDER_LSB, ((x + y + z + w) / 8), NVCV_SWIZZLE_XYZW, x, y, z, w \
        }                                                                        \
    }

const std::vector<PackingTestParams> g_packingParams = {
    {           NVCV_PACKING_0,          {0},            {NVCV_ORDER_LSB, 0, NVCV_SWIZZLE_0000, 0}},
    {NVCV_PACKING_X8_Y8__X8_Z8, {8, 8, 8, 8},   {NVCV_ORDER_MSB, 1, NVCV_SWIZZLE_XYXZ, 8, 8, 8, 8}},
    {NVCV_PACKING_Y8_X8__Z8_X8, {8, 8, 8, 8},   {NVCV_ORDER_MSB, 1, NVCV_SWIZZLE_YXZX, 8, 8, 8, 8}},
    {    NVCV_PACKING_X5Y5b1Z5, {5, 5, 5, 0},   {NVCV_ORDER_LSB, 2, NVCV_SWIZZLE_XYW0, 5, 5, 1, 5}},
    {   NVCV_PACKING_X32_Y24b8,     {32, 24}, {NVCV_ORDER_MSB, 4, NVCV_SWIZZLE_XY00, 32, 24, 8, 0}},

    DEF_PACK1(1),
    DEF_FIX_PACK1(1),

    DEF_PACK1(2),
    DEF_FIX_PACK1(2),

    DEF_PACK1(4),
    DEF_FIX_PACK1(4),

    DEF_PACK1(8),
    DEF_LSB_PACK1(4, 4),
    DEF_MSB_PACK1(4, 4),

    DEF_FIX_PACK1(8),
    DEF_PACK2(4, 4),
    DEF_PACK3(3, 3, 2),

    DEF_PACK1(16),
    DEF_FIX_PACK1(16),
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
    DEF_FIX_PACK1(24),
    DEF_FIX_PACK3(8, 8, 8),

    DEF_PACK1(32),
    DEF_FIX_PACK1(32),
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
    DEF_FIX_PACK1(48),
    DEF_FIX_PACK3(16, 16, 16),

    DEF_PACK1(64),
    DEF_FIX_PACK1(64),
    DEF_FIX_PACK2(32, 32),
    DEF_FIX_PACK4(16, 16, 16, 16),

    DEF_PACK1(96),
    DEF_FIX_PACK1(96),
    DEF_FIX_PACK3(32, 32, 32),

    DEF_PACK1(128),
    DEF_FIX_PACK1(128),
    DEF_FIX_PACK2(64, 64),
    DEF_FIX_PACK4(32, 32, 32, 32),

    DEF_PACK1(192),
    DEF_FIX_PACK1(192),
    DEF_FIX_PACK3(64, 64, 64),

    DEF_PACK1(256),
    DEF_FIX_PACK1(256),
    DEF_FIX_PACK4(64, 64, 64, 64)
};

INSTANTIATE_TEST_SUITE_P(_, PackingTests, t::ValuesIn(g_packingParams));

TEST_P(PackingTests, make_format_packing)
{
    PackingTestParams p = GetParam();

    NVCVPacking test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakePacking(&test, &p.params));
    EXPECT_EQ(p.packing, test);

    int chCount;
    ASSERT_EQ(NVCV_SUCCESS, nvcvSwizzleGetNumChannels(p.params.swizzle, &chCount));

    if (chCount == 1)
    {
        if (p.params.byteOrder == NVCV_ORDER_LSB)
        {
            p.params.byteOrder = NVCV_ORDER_MSB;
        }
        else
        {
            p.params.byteOrder = NVCV_ORDER_LSB;
        }

        ASSERT_EQ(NVCV_SUCCESS, nvcvMakePacking(&test, &p.params));

        ASSERT_EQ(p.packing, test);
    }
}

TEST_P(PackingTests, get_format_packing_params)
{
    const PackingTestParams &p = GetParam();

    NVCVPackingParams params;

    nvcvPackingGetParams(p.packing, &params);

    if (p.params.bits[1] == 0)
    {
        EXPECT_EQ(NVCV_ORDER_LSB, params.byteOrder);
    }
    else
    {
        EXPECT_EQ(p.params.byteOrder, params.byteOrder);
    }

    EXPECT_EQ(p.params.bits[0], params.bits[0]);
    EXPECT_EQ(p.params.bits[1], params.bits[1]);
    EXPECT_EQ(p.params.bits[2], params.bits[2]);
    EXPECT_EQ(p.params.bits[3], params.bits[3]);
}

TEST_P(PackingTests, get_bits_per_component)
{
    PackingTestParams p = GetParam();

    int bits[4];
    nvcvPackingGetBitsPerComponent(p.packing, bits);

    EXPECT_EQ(p.bitsPerComponent[0], bits[0]) << p.packing;
    EXPECT_EQ(p.bitsPerComponent[1], bits[1]) << p.packing;
    EXPECT_EQ(p.bitsPerComponent[2], bits[2]) << p.packing;
    EXPECT_EQ(p.bitsPerComponent[3], bits[3]) << p.packing;
}

TEST_P(PackingTests, check_component_count)
{
    const PackingTestParams &p = GetParam();

    int gold = 0;

    for (int i = 0; i < 4; ++i)
    {
        if (p.bitsPerComponent[i] != 0)
        {
            ++gold;
        }
    }

    int ncomp;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPackingGetNumComponents(p.packing, &ncomp));
    EXPECT_EQ(gold, ncomp);
}

TEST_P(PackingTests, check_bits_per_pixel)
{
    const PackingTestParams &p = GetParam();

    int gold = 0;

    switch (p.packing)
    {
    case NVCV_PACKING_X8_Y8__X8_Z8:
    case NVCV_PACKING_Y8_X8__Z8_X8:
        gold = 16;
        break;

    default:
        for (int i = 0; i < 4; ++i)
        {
            gold += p.params.bits[i];
        }
        break;
    }

    int bpp;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPackingGetBitsPerPixel(p.packing, &bpp));
    EXPECT_EQ(gold, bpp);
}

TEST(PackingTests, valid_values)
{
    std::unordered_set<PackingTestParams, HashPackingTestParams> packingList(g_packingParams.begin(),
                                                                             g_packingParams.end());

    size_t counter = 0;

    PackingTestParams p;
    p.params.alignment = 0; // don't care

    NVCVChannel swc[4];
    for (int bitsX = 0; bitsX <= 256; bitsX < 32 ? ++bitsX : (bitsX < 128 ? (bitsX += 8) : (bitsX += 32)))
    {
        p.params.bits[0] = bitsX;
        swc[0]           = p.params.bits[0] != 0 ? NVCV_CHANNEL_X : NVCV_CHANNEL_0;
        for (int bitsY = 0; bitsY <= 128; bitsY < 32 ? ++bitsY : (bitsY += 8))
        {
            p.params.bits[1] = bitsY;
            swc[1]           = p.params.bits[1] != 0 ? NVCV_CHANNEL_Y : NVCV_CHANNEL_0;
            for (int bitsZ = 0; bitsZ <= 128; bitsZ < 32 ? ++bitsZ : (bitsZ += 8))
            {
                p.params.bits[2] = bitsZ;
                swc[2]           = p.params.bits[2] != 0 ? NVCV_CHANNEL_Z : NVCV_CHANNEL_0;
                for (int bitsW = 0; bitsW <= 128; bitsW < 32 ? ++bitsW : (bitsW += 8))
                {
                    p.params.bits[3] = bitsW;
                    swc[3]           = p.params.bits[3] != 0 ? NVCV_CHANNEL_W : NVCV_CHANNEL_0;

                    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeSwizzle(&p.params.swizzle, swc[0], swc[1], swc[2], swc[3]));

                    p.params.byteOrder = NVCV_ORDER_MSB;

                    auto it = packingList.find(p);
                    if (it != packingList.end())
                    {
                        NVCVPacking packing;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvMakePacking(&packing, &p.params));

                        EXPECT_EQ(it->packing, packing) << p;
                        packingList.erase(it);
                    }
                    // to save some time, let's do negative tests in only a subset of the parameter space
                    else if (bitsX == 8)
                    {
                        NVCVPacking packing = NVCV_PACKING_X8_Y8__X8_Z8;
                        ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvMakePacking(&packing, &p.params)) << p;
                        EXPECT_EQ(packing, NVCV_PACKING_X8_Y8__X8_Z8) << "should not have modified output";
                    }

                    p.params.byteOrder = NVCV_ORDER_LSB;

                    it = packingList.find(p);
                    if (it != packingList.end())
                    {
                        NVCVPacking packing;
                        ASSERT_EQ(NVCV_SUCCESS, nvcvMakePacking(&packing, &p.params));

                        EXPECT_EQ(it->packing, packing) << p;
                        packingList.erase(it);
                    }
                    // to save some time, let's do negative tests in only a subset of the parameter space
                    else if (bitsX == 8)
                    {
                        NVCVPacking packing = NVCV_PACKING_X8_Y8__X8_Z8;
                        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvMakePacking(&packing, &p.params)) << p;
                        EXPECT_EQ(packing, NVCV_PACKING_X8_Y8__X8_Z8) << "should not have modified output";
                    }
                }

                ++counter;
            }
        }
    }

    // Handle exceptions to the default representation scheme

    auto testPacking = [&packingList](NVCVPacking packing, const std::array<int, 4> &bits, NVCVByteOrder byteOrder,
                                      NVCVSwizzle swizzle)
    {
        PackingTestParams p = {};

        p.packing          = packing;
        p.params.byteOrder = byteOrder;
        p.params.swizzle   = swizzle;
        p.params.bits[0]   = bits[0];
        p.params.bits[1]   = bits[1];
        p.params.bits[2]   = bits[2];
        p.params.bits[3]   = bits[3];

        auto it = packingList.find(p);
        ASSERT_TRUE(it != packingList.end()) << p.packing;
        ASSERT_TRUE(it->packing == packing);

        p.params.alignment = it->params.alignment;

        ASSERT_EQ(NVCV_SUCCESS, nvcvMakePacking(&packing, &p.params));

        EXPECT_EQ(it->packing, packing);
        packingList.erase(it);
    };

    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X8_Y8__X8_Z8, {8, 8, 8, 8}, NVCV_ORDER_MSB, NVCV_SWIZZLE_XYXZ));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_Y8_X8__Z8_X8, {8, 8, 8, 8}, NVCV_ORDER_MSB, NVCV_SWIZZLE_YXZX));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X12b4, {12, 4}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X10b6, {10, 6}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b4X12, {4, 12}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b12X20, {12, 20}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b6X10, {6, 10}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X14b2, {14, 2}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b2X14, {2, 14}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X10b6_Y10b6, {10, 6, 10, 6}, NVCV_ORDER_MSB, NVCV_SWIZZLE_XZ00));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X12b4_Y12b4, {12, 4, 12, 4}, NVCV_ORDER_MSB, NVCV_SWIZZLE_XZ00));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b4X4Y4Z4, {4, 4, 4, 4}, NVCV_ORDER_LSB, NVCV_SWIZZLE_YZW0));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b1X5Y5Z5, {1, 5, 5, 5}, NVCV_ORDER_LSB, NVCV_SWIZZLE_YZW0));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X5Y5b1Z5, {5, 5, 1, 5}, NVCV_ORDER_LSB, NVCV_SWIZZLE_XYW0));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X4b4, {4, 4}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b4X4, {4, 4}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X32_Y24b8, {32, 24, 8}, NVCV_ORDER_MSB, NVCV_SWIZZLE_XY00));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b8X24, {8, 24}, NVCV_ORDER_LSB, NVCV_SWIZZLE_Y000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X20b12, {20, 12}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X24b8, {24, 8}, NVCV_ORDER_MSB, NVCV_SWIZZLE_X000));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_X10Y10Z10b2, {10, 10, 10, 2}, NVCV_ORDER_MSB, NVCV_SWIZZLE_XYZ0));
    ASSERT_NO_FATAL_FAILURE(testPacking(NVCV_PACKING_b2X10Y10Z10, {2, 10, 10, 10}, NVCV_ORDER_LSB, NVCV_SWIZZLE_YZW0));

    EXPECT_TRUE(packingList.empty());

    if (!packingList.empty())
    {
        std::cerr << "Non-matched packings: " << std::endl;
        for (auto &p : packingList)
        {
            std::cerr << "  " << p.packing << std::endl;
        }
    }
}

TEST(PackingTests, get_name)
{
    EXPECT_STREQ("NVCV_PACKING_X16", nvcvPackingGetName(NVCV_PACKING_X16));
    EXPECT_STREQ("NVCVPacking(2147483647)", nvcvPackingGetName(NVCV_PACKING_LIMIT32));
}

TEST(ByteOrderTests, get_name)
{
    EXPECT_STREQ("LSB", nvcvByteOrderGetName(NVCV_ORDER_LSB));
    EXPECT_STREQ("NVCVByteOrder(-1)", nvcvByteOrderGetName(static_cast<NVCVByteOrder>(-1)));
}

TEST(SwizzleTests, get_name)
{
    EXPECT_STREQ("XYZ1", nvcvSwizzleGetName(NVCV_SWIZZLE_XYZ1));
    EXPECT_STREQ("110X",
                 nvcvSwizzleGetName(NVCV_MAKE_SWIZZLE(NVCV_CHANNEL_1, NVCV_CHANNEL_1, NVCV_CHANNEL_0, NVCV_CHANNEL_X)));
}

TEST(SwizzleChannelTests, get_name)
{
    EXPECT_STREQ("Y", nvcvChannelGetName(NVCV_CHANNEL_Y));
    EXPECT_STREQ("NVCVChannel(7)", nvcvChannelGetName(static_cast<NVCVChannel>(7)));
}

TEST(MemLayoutTests, get_name)
{
    EXPECT_STREQ("NVCV_MEM_LAYOUT_BLOCK2_LINEAR", nvcvMemLayoutGetName(NVCV_MEM_LAYOUT_BLOCK2_LINEAR));
    EXPECT_STREQ("NVCVMemLayout(-1)", nvcvMemLayoutGetName(static_cast<NVCVMemLayout>(-1)));
}

TEST(DataKindTests, get_name)
{
    EXPECT_STREQ("NVCV_DATA_KIND_FLOAT", nvcvDataKindGetName(NVCV_DATA_KIND_FLOAT));
    EXPECT_STREQ("NVCVDataKind(-1)", nvcvDataKindGetName(static_cast<NVCVDataKind>(-1)));
}
