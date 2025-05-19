/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nvcv/src/priv/DataLayout.hpp"

TEST(DataLayoutTest, GetBlockHeightLog2)
{
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK1_LINEAR), 0);
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK2_LINEAR), 1);
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK4_LINEAR), 2);
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK8_LINEAR), 3);
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK16_LINEAR), 4);
    EXPECT_EQ(nvcv::priv::GetBlockHeightLog2(NVCV_MEM_LAYOUT_BLOCK32_LINEAR), 5);
}

TEST(DataLayoutTest, FlipByteOrder)
{
    auto makeSwizzle = [](NVCVChannel x, NVCVChannel y, NVCVChannel z, NVCVChannel w) -> NVCVSwizzle
    {
        NVCVSwizzle res;
        EXPECT_EQ(NVCV_SUCCESS, nvcvMakeSwizzle(&res, x, y, z, w));
        return res;
    };

    auto testFlipByteOrder = [&](NVCVChannel x, NVCVChannel y, NVCVChannel z, NVCVChannel w, int off, int len,
                                 NVCVChannel goldX, NVCVChannel goldY, NVCVChannel goldZ, NVCVChannel goldW) -> void
    {
        NVCVSwizzle initialSwizzle = makeSwizzle(x, y, z, w);
        NVCVSwizzle res            = nvcv::priv::FlipByteOrder(initialSwizzle, off, len);
        EXPECT_EQ(res, makeSwizzle(goldX, goldY, goldZ, goldW));
    };

    testFlipByteOrder(NVCV_CHANNEL_X, NVCV_CHANNEL_Y, NVCV_CHANNEL_Z, NVCV_CHANNEL_W, 0, 4, NVCV_CHANNEL_W,
                      NVCV_CHANNEL_Z, NVCV_CHANNEL_Y, NVCV_CHANNEL_X); // XYZW -> WZYX
    testFlipByteOrder(NVCV_CHANNEL_W, NVCV_CHANNEL_X, NVCV_CHANNEL_Y, NVCV_CHANNEL_Z, 1, 2, NVCV_CHANNEL_W,
                      NVCV_CHANNEL_Y, NVCV_CHANNEL_X, NVCV_CHANNEL_Z); // WXYZ -> WYXZ
    testFlipByteOrder(NVCV_CHANNEL_W, NVCV_CHANNEL_X, NVCV_CHANNEL_Y, NVCV_CHANNEL_Z, 1, 3, NVCV_CHANNEL_W,
                      NVCV_CHANNEL_Z, NVCV_CHANNEL_Y, NVCV_CHANNEL_X); // WXYZ -> WZYX
    testFlipByteOrder(NVCV_CHANNEL_X, NVCV_CHANNEL_Y, NVCV_CHANNEL_Z, NVCV_CHANNEL_W, 0, 3, NVCV_CHANNEL_Z,
                      NVCV_CHANNEL_Y, NVCV_CHANNEL_X, NVCV_CHANNEL_W);
    testFlipByteOrder(NVCV_CHANNEL_X, NVCV_CHANNEL_Y, NVCV_CHANNEL_Z, NVCV_CHANNEL_W, 0, 4, NVCV_CHANNEL_W,
                      NVCV_CHANNEL_Z, NVCV_CHANNEL_Y, NVCV_CHANNEL_X);
}

TEST(DataLayoutTest, MakeNVCVPacking)
{
    EXPECT_EQ(NVCV_PACKING_X10Y10Z10W2, nvcv::priv::MakeNVCVPacking(10, 10, 10, 2));

    EXPECT_EQ(NVCV_PACKING_X10b6, nvcv::priv::MakeNVCVPacking(10, 0, 0, 0));
    EXPECT_EQ(NVCV_PACKING_X12b4, nvcv::priv::MakeNVCVPacking(12, 0, 0, 0));
    EXPECT_EQ(NVCV_PACKING_X14b2, nvcv::priv::MakeNVCVPacking(14, 0, 0, 0));
    EXPECT_EQ(NVCV_PACKING_X20b12, nvcv::priv::MakeNVCVPacking(20, 0, 0, 0));

    // Cannot test origY branch because there is not corresponding packing, compare with std::nullopt
    EXPECT_EQ(std::nullopt, nvcv::priv::MakeNVCVPacking(10, 10, 0, 0));
    EXPECT_EQ(std::nullopt, nvcv::priv::MakeNVCVPacking(12, 12, 0, 0));
    EXPECT_EQ(std::nullopt, nvcv::priv::MakeNVCVPacking(14, 14, 0, 0));
    EXPECT_EQ(std::nullopt, nvcv::priv::MakeNVCVPacking(20, 20, 0, 0));
}

TEST(DataLayoutTest, IsSubWord)
{
    NVCVPackingParams params;

    params.swizzle = NVCV_SWIZZLE_0000;
    EXPECT_FALSE(nvcv::priv::IsSubWord(params));

    params.swizzle = NVCV_SWIZZLE_XYXZ;
    EXPECT_FALSE(nvcv::priv::IsSubWord(params));

    params.swizzle = NVCV_SWIZZLE_YXZX;
    EXPECT_FALSE(nvcv::priv::IsSubWord(params));
}

TEST(ByteOrderTests, get_name_operator)
{
    auto testOperatorInsertion = [](std::string expectedStr, NVCVByteOrder order) -> void
    {
        std::ostringstream ss;
        ss << order;
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("LSB", NVCV_ORDER_LSB);
    testOperatorInsertion("MSB", NVCV_ORDER_MSB);
}

TEST(AlphaTypeTests, get_name_operator)
{
    auto testOperatorInsertion = [](std::string expectedStr, NVCVAlphaType alphaType) -> void
    {
        std::ostringstream ss;
        ss << alphaType;
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("ASSOCIATED", NVCV_ALPHA_ASSOCIATED);
    testOperatorInsertion("UNASSOCIATED", NVCV_ALPHA_UNASSOCIATED);
    testOperatorInsertion("NVCVAlphaType(-1)", static_cast<NVCVAlphaType>(-1));
}

TEST(ExtraChannelTests, get_name_operator)
{
    auto testOperatorInsertion = [](std::string expectedStr, NVCVExtraChannel extraChannel) -> void
    {
        std::ostringstream ss;
        ss << extraChannel;
        EXPECT_EQ(expectedStr, ss.str());
        ss.str("");
        ss.clear();
    };

    testOperatorInsertion("U", NVCV_EXTRA_CHANNEL_U);
    testOperatorInsertion("D", NVCV_EXTRA_CHANNEL_D);
    testOperatorInsertion("POS3D", NVCV_EXTRA_CHANNEL_POS3D);
    testOperatorInsertion("NVCVExtraChannel(-1)", static_cast<NVCVExtraChannel>(-1));
}
