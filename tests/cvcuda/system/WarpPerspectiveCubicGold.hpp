// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVCV_TEST_WARP_PERSPECTIVE_CUBIC_GOLD_HPP
#define NVCV_TEST_WARP_PERSPECTIVE_CUBIC_GOLD_HPP

#include <array>
#include <cassert>
#include <cstdint>

namespace nvcv::test::perspective {

inline int ReflectedIndex(int coordinate, int length)
{
    assert(length > 0);
    while (coordinate < 0 || coordinate >= length)
    {
        coordinate = coordinate < 0 ? -coordinate - 1 : 2 * length - coordinate - 1;
    }
    return coordinate;
}

inline std::array<int, 4> QuarterWeights(int fraction)
{
    assert(fraction >= 0 && fraction < 4);
    const int squared = fraction * fraction;
    const int cubed   = squared * fraction;
    return {-16 * fraction + 8 * squared - cubed, 128 - 20 * squared + 3 * cubed,
            16 * fraction + 16 * squared - 3 * cubed, -4 * squared + cubed};
}

inline uint8_t QuantizeCubicNumerator(int64_t numerator)
{
    constexpr int64_t denominator = 128 * 128;
    if (numerator <= 0)
    {
        return 0;
    }
    if (numerator >= 255 * denominator)
    {
        return 255;
    }
    int64_t       rounded   = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    if (remainder > denominator / 2 || (remainder == denominator / 2 && rounded % 2 != 0))
    {
        ++rounded;
    }
    return static_cast<uint8_t>(rounded);
}

template<class ReadPixel>
uint8_t ExactQuarterCubicReflect(int sourceXQuarter, int sourceYQuarter, int width, int height, ReadPixel readPixel)
{
    const int  fractionX  = (sourceXQuarter % 4 + 4) % 4;
    const int  fractionY  = (sourceYQuarter % 4 + 4) % 4;
    const int  left       = (sourceXQuarter - fractionX) / 4;
    const int  top        = (sourceYQuarter - fractionY) / 4;
    const auto horizontal = QuarterWeights(fractionX);
    const auto vertical   = QuarterWeights(fractionY);
    int64_t    numerator  = 0;
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            numerator += static_cast<int64_t>(
                             readPixel(ReflectedIndex(top + row - 1, height), ReflectedIndex(left + column - 1, width)))
                       * horizontal[column] * vertical[row];
        }
    }
    return QuantizeCubicNumerator(numerator);
}

} // namespace nvcv::test::perspective

#endif
