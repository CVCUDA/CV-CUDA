/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_TESTS_COMMON_INTERPUTILS_HPP
#define NVCV_TESTS_COMMON_INTERPUTILS_HPP

#include <common/BorderUtils.hpp>             // for test::IsInside, etc.
#include <cvcuda/Types.h>                     // for NVCVInterpolationType, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>      // for operator +, etc.
#include <cvcuda/cuda_tools/MathWrappers.hpp> // for cuda::round, etc.
#include <cvcuda/cuda_tools/SaturateCast.hpp> // for cuda::SaturateCast, etc.

#include <array>
#include <vector>

#define VEC_EXPECT_NEAR(vec1, vec2, delta)                              \
    ASSERT_EQ(vec1.size(), vec2.size());                                \
    for (std::size_t idx = 0; idx < vec1.size(); ++idx)                 \
    {                                                                   \
        EXPECT_NEAR(vec1[idx], vec2[idx], delta) << "At index " << idx; \
    }

namespace nvcv::test {

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long1 strides, int1 coord)
{
    return *reinterpret_cast<T *>(&vec[coord.x * strides.x]);
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long2 strides, int2 coord)
{
    return *reinterpret_cast<T *>(&vec[coord.y * strides.x + coord.x * strides.y]);
}

template<NVCVBorderType B, typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long2 strides, int2 size, const T &borderValue, int2 coord)
{
    return test::IsInside(coord, size, B)
             ? *reinterpret_cast<const T *>(&vec[coord.y * strides.x + coord.x * strides.y])
             : borderValue;
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long3 strides, int3 coord)
{
    return *reinterpret_cast<T *>(&vec[coord.z * strides.x + coord.y * strides.y + coord.x * strides.z]);
}

template<NVCVBorderType B, typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long3 strides, int2 size, const T &borderValue, int3 coord)
{
    int2 inCoord{coord.x, coord.y};

    return test::IsInside(inCoord, size, B)
             ? *reinterpret_cast<const T *>(&vec[coord.z * strides.x + inCoord.y * strides.y + inCoord.x * strides.z])
             : borderValue;
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, const long4_16a &strides, int4 coord)
{
    return *reinterpret_cast<T *>(
        &vec[coord.w * strides.x + coord.z * strides.y + coord.y * strides.z + coord.x * strides.w]);
}

template<NVCVBorderType B, typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, const long4_16a &strides, int2 size, const T &borderValue,
                        int4 coord)
{
    int2 inCoord{coord.y, coord.z};

    return test::IsInside(inCoord, size, B) ? *reinterpret_cast<const T *>(
               &vec[coord.w * strides.x + inCoord.y * strides.y + inCoord.x * strides.z + coord.x * strides.w])
                                            : borderValue;
}

template<int N, typename RT = cuda::MakeType<int, N>>
inline RT GetCoord(int x, int y, int z = 0, int k = 0)
{
    if constexpr (N == 2)
        return RT{x, y};
    else if constexpr (N == 3)
        return RT{x, y, z};
    else if constexpr (N == 4)
        return RT{k, x, y, z};
}

inline void GetBicubicCoeffs(float delta, float &w0, float &w1, float &w2, float &w3)
{
    w0 = -.5f;
    w0 = w0 * delta + 1.f;
    w0 = w0 * delta - .5f;
    w0 = w0 * delta;

    w1 = 1.5f;
    w1 = w1 * delta - 2.5f;
    w1 = w1 * delta;
    w1 = w1 * delta + 1.f;

    w2 = -1.5f;
    w2 = w2 * delta + 2.f;
    w2 = w2 * delta + .5f;
    w2 = w2 * delta;

    w3 = 1 - w0 - w1 - w2;
}

template<typename StridesType, typename ValueType>
struct GoldInterpContext
{
    const std::vector<uint8_t> &vec;
    const StridesType          &strides;
    int2                        size;
    const ValueType            &bValue;
    int                         z;
    int                         k;
};

struct AreaWindow
{
    float fsx1;
    float fsx2;
    float fsy1;
    float fsy2;
    int   sx1;
    int   sx2;
    int   sy1;
    int   sy2;
};

inline float AreaDelta(int edge, float fractionalEdge)
{
    return static_cast<float>(edge) - fractionalEdge;
}

inline bool AreaEdgeAfter(int edge, float fractionalEdge)
{
    return static_cast<float>(edge) > fractionalEdge;
}

inline bool AreaEdgeBefore(int edge, float fractionalEdge)
{
    return static_cast<float>(edge) < fractionalEdge;
}

template<int N, NVCVBorderType B, typename StridesType, typename ValueType>
inline const ValueType &GoldValueAt(const GoldInterpContext<StridesType, ValueType> &ctx, int x, int y)
{
    return ValueAt<B>(ctx.vec, ctx.strides, ctx.size, ctx.bValue, GetCoord<N>(x, y, ctx.z, ctx.k));
}

template<int N, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterpNearest(const GoldInterpContext<StridesType, ValueType> &ctx, float2 coord)
{
    int2 c = cuda::round<cuda::RoundMode::DOWN, int>(coord + .5f);

    return GoldValueAt<N, B>(ctx, c.x, c.y);
}

template<int N, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterpLinear(const GoldInterpContext<StridesType, ValueType> &ctx, float2 coord)
{
    int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(coord);
    int2 c2 = c1 + 1;

    ValueType v1 = GoldValueAt<N, B>(ctx, c1.x, c1.y);
    ValueType v2 = GoldValueAt<N, B>(ctx, c2.x, c1.y);
    ValueType v3 = GoldValueAt<N, B>(ctx, c1.x, c2.y);
    ValueType v4 = GoldValueAt<N, B>(ctx, c2.x, c2.y);

    auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

    out += v1 * (static_cast<float>(c2.x) - coord.x) * (static_cast<float>(c2.y) - coord.y);
    out += v2 * (coord.x - static_cast<float>(c1.x)) * (static_cast<float>(c2.y) - coord.y);
    out += v3 * (static_cast<float>(c2.x) - coord.x) * (coord.y - static_cast<float>(c1.y));
    out += v4 * (coord.x - static_cast<float>(c1.x)) * (coord.y - static_cast<float>(c1.y));

    return cuda::SaturateCast<ValueType>(out);
}

template<int N, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterpCubic(const GoldInterpContext<StridesType, ValueType> &ctx, float2 coord)
{
    int ix = cuda::round<cuda::RoundMode::DOWN, int>(coord.x);
    int iy = cuda::round<cuda::RoundMode::DOWN, int>(coord.y);

    using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
    auto sum = cuda::SetAll<FT>(0);

    std::array<float, 4> wx;
    test::GetBicubicCoeffs(coord.x - static_cast<float>(ix), wx[0], wx[1], wx[2], wx[3]);
    std::array<float, 4> wy;
    test::GetBicubicCoeffs(coord.y - static_cast<float>(iy), wy[0], wy[1], wy[2], wy[3]);

    for (int cy = -1; cy <= 2; cy++)
    {
        for (int cx = -1; cx <= 2; cx++)
        {
            sum += (wx[cx + 1] * wy[cy + 1]) * GoldValueAt<N, B>(ctx, ix + cx, iy + cy);
        }
    }

    return cuda::SaturateCast<ValueType>(sum);
}

inline AreaWindow GetAreaWindow(float2 scale, float2 coord)
{
    float fsx1 = coord.x * scale.x;
    float fsx2 = fsx1 + scale.x;
    float fsy1 = coord.y * scale.y;
    float fsy2 = fsy1 + scale.y;

    return AreaWindow{fsx1,
                      fsx2,
                      fsy1,
                      fsy2,
                      cuda::round<cuda::RoundMode::UP, int>(fsx1),
                      cuda::round<cuda::RoundMode::DOWN, int>(fsx2),
                      cuda::round<cuda::RoundMode::UP, int>(fsy1),
                      cuda::round<cuda::RoundMode::DOWN, int>(fsy2)};
}

template<int N, NVCVBorderType B, typename AccumType, typename StridesType, typename ValueType>
inline void AddAreaBlock(AccumType &out, const GoldInterpContext<StridesType, ValueType> &ctx, int yBegin, int yEnd,
                         int xBegin, int xEnd, float weight)
{
    for (int dy = yBegin; dy < yEnd; ++dy)
    {
        for (int dx = xBegin; dx < xEnd; ++dx)
        {
            out = out + GoldValueAt<N, B>(ctx, dx, dy) * weight;
        }
    }
}

template<int N, NVCVBorderType B, typename AccumType, typename StridesType, typename ValueType>
inline void AddAreaSideColumns(AccumType &out, const GoldInterpContext<StridesType, ValueType> &ctx,
                               const AreaWindow &window, float invscale)
{
    for (int dy = window.sy1; dy < window.sy2; ++dy)
    {
        if (AreaEdgeAfter(window.sx1, window.fsx1))
        {
            out = out + GoldValueAt<N, B>(ctx, window.sx1 - 1, dy) * (AreaDelta(window.sx1, window.fsx1) * invscale);
        }

        if (AreaEdgeBefore(window.sx2, window.fsx2))
        {
            out = out + GoldValueAt<N, B>(ctx, window.sx2, dy) * (-AreaDelta(window.sx2, window.fsx2) * invscale);
        }
    }
}

template<int N, NVCVBorderType B, typename AccumType, typename StridesType, typename ValueType>
inline void AddAreaSideRows(AccumType &out, const GoldInterpContext<StridesType, ValueType> &ctx,
                            const AreaWindow &window, float invscale)
{
    if (AreaEdgeAfter(window.sy1, window.fsy1))
    {
        for (int dx = window.sx1; dx < window.sx2; ++dx)
        {
            out = out + GoldValueAt<N, B>(ctx, dx, window.sy1 - 1) * (AreaDelta(window.sy1, window.fsy1) * invscale);
        }
    }

    if (AreaEdgeBefore(window.sy2, window.fsy2))
    {
        for (int dx = window.sx1; dx < window.sx2; ++dx)
        {
            out = out + GoldValueAt<N, B>(ctx, dx, window.sy2) * (-AreaDelta(window.sy2, window.fsy2) * invscale);
        }
    }
}

template<int N, NVCVBorderType B, typename AccumType, typename StridesType, typename ValueType>
inline void AddAreaCorners(AccumType &out, const GoldInterpContext<StridesType, ValueType> &ctx,
                           const AreaWindow &window, float invscale)
{
    if (AreaEdgeAfter(window.sy1, window.fsy1) && AreaEdgeAfter(window.sx1, window.fsx1))
    {
        out = out
            + GoldValueAt<N, B>(ctx, window.sx1 - 1, window.sy1 - 1)
                  * (AreaDelta(window.sy1, window.fsy1) * AreaDelta(window.sx1, window.fsx1) * invscale);
    }

    if (AreaEdgeAfter(window.sy1, window.fsy1) && AreaEdgeBefore(window.sx2, window.fsx2))
    {
        out = out
            + GoldValueAt<N, B>(ctx, window.sx2, window.sy1 - 1)
                  * (AreaDelta(window.sy1, window.fsy1) * -AreaDelta(window.sx2, window.fsx2) * invscale);
    }

    if (AreaEdgeBefore(window.sy2, window.fsy2) && AreaEdgeBefore(window.sx2, window.fsx2))
    {
        out = out
            + GoldValueAt<N, B>(ctx, window.sx2, window.sy2)
                  * (-AreaDelta(window.sy2, window.fsy2) * -AreaDelta(window.sx2, window.fsx2) * invscale);
    }

    if (AreaEdgeBefore(window.sy2, window.fsy2) && AreaEdgeAfter(window.sx1, window.fsx1))
    {
        out = out
            + GoldValueAt<N, B>(ctx, window.sx1 - 1, window.sy2)
                  * (-AreaDelta(window.sy2, window.fsy2) * AreaDelta(window.sx1, window.fsx1) * invscale);
    }
}

template<int N, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterpArea(const GoldInterpContext<StridesType, ValueType> &ctx, float2 scale, float2 coord)
{
    AreaWindow window = GetAreaWindow(scale, coord);
    auto       out    = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

    if (std::ceil(scale.x) == scale.x && std::ceil(scale.y) == scale.y)
    {
        AddAreaBlock<N, B>(out, ctx, window.sy1, window.sy2, window.sx1, window.sx2, 1.f / (scale.x * scale.y));
        return cuda::SaturateCast<ValueType>(out);
    }

    float invscale = 1.f
                   / (std::min(scale.x, AreaDelta(ctx.size.x, window.fsx1))
                      * std::min(scale.y, AreaDelta(ctx.size.y, window.fsy1)));

    AddAreaBlock<N, B>(out, ctx, window.sy1, window.sy2, window.sx1, window.sx2, invscale);
    AddAreaSideColumns<N, B>(out, ctx, window, invscale);
    AddAreaSideRows<N, B>(out, ctx, window, invscale);
    AddAreaCorners<N, B>(out, ctx, window, invscale);

    return cuda::SaturateCast<ValueType>(out);
}

template<NVCVInterpolationType I, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterp(const std::vector<uint8_t> &vec, const StridesType &strides, const int2 &size,
                            const ValueType &bValue, float2 scale, float2 coord, int z = 0, int k = 0)
{
    constexpr int N = cuda::NumElements<StridesType>;

    GoldInterpContext<StridesType, ValueType> ctx{vec, strides, size, bValue, z, k};

    if constexpr (I == NVCV_INTERP_NEAREST)
    {
        return GoldInterpNearest<N, B>(ctx, coord);
    }
    else if constexpr (I == NVCV_INTERP_LINEAR)
    {
        return GoldInterpLinear<N, B>(ctx, coord);
    }
    else if constexpr (I == NVCV_INTERP_CUBIC)
    {
        return GoldInterpCubic<N, B>(ctx, coord);
    }
    else if constexpr (I == NVCV_INTERP_AREA)
    {
        return GoldInterpArea<N, B>(ctx, scale, coord);
    }
}

} // namespace nvcv::test

#endif // NVCV_TESTS_COMMON_HASHUTILS_HPP
