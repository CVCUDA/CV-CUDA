/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/BorderUtils.hpp>     // for test::IsInside, etc.
#include <cvcuda/Types.h>             // for NVCVInterpolationType, etc.
#include <nvcv/cuda/MathOps.hpp>      // for operator +, etc.
#include <nvcv/cuda/MathWrappers.hpp> // for cuda::round, etc.
#include <nvcv/cuda/SaturateCast.hpp> // for cuda::SaturateCast, etc.

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
inline T &ValueAt(std::vector<uint8_t> &vec, long4 strides, int4 coord)
{
    return *reinterpret_cast<T *>(
        &vec[coord.w * strides.x + coord.z * strides.y + coord.y * strides.z + coord.x * strides.w]);
}

template<NVCVBorderType B, typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long4 strides, int2 size, const T &borderValue, int4 coord)
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

inline float GetBicubicCoeff(float c)
{
    c = std::fabs(c);
    if (c <= 1.0f)
    {
        return c * c * (1.5f * c - 2.5f) + 1.0f;
    }
    else if (c < 2.0f)
    {
        return c * (c * (-0.5f * c + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }
}

template<NVCVInterpolationType I, NVCVBorderType B, typename StridesType, typename ValueType>
inline ValueType GoldInterp(const std::vector<uint8_t> &vec, const StridesType &strides, const int2 &size,
                            const ValueType &bValue, float2 scale, float2 coord, int z = 0, int k = 0)
{
    constexpr int N = cuda::NumElements<StridesType>;

    if constexpr (I == NVCV_INTERP_NEAREST)
    {
        int2 c = cuda::round<cuda::RoundMode::DOWN, int>(coord + .5f);

        return ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(c.x, c.y, z, k));
    }
    else if constexpr (I == NVCV_INTERP_LINEAR)
    {
        int2 c1 = cuda::round<cuda::RoundMode::DOWN, int>(coord);
        int2 c2 = c1 + 1;

        ValueType v1 = ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(c1.x, c1.y, z, k));
        ValueType v2 = ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(c2.x, c1.y, z, k));
        ValueType v3 = ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(c1.x, c2.y, z, k));
        ValueType v4 = ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(c2.x, c2.y, z, k));

        auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

        out += v1 * (c2.x - coord.x) * (c2.y - coord.y);
        out += v2 * (coord.x - c1.x) * (c2.y - coord.y);
        out += v3 * (c2.x - coord.x) * (coord.y - c1.y);
        out += v4 * (coord.x - c1.x) * (coord.y - c1.y);

        return cuda::SaturateCast<ValueType>(out);
    }
    else if constexpr (I == NVCV_INTERP_CUBIC)
    {
        int xmin = cuda::round<cuda::RoundMode::UP, int>(coord.x - 2.f);
        int ymin = cuda::round<cuda::RoundMode::UP, int>(coord.y - 2.f);
        int xmax = cuda::round<cuda::RoundMode::DOWN, int>(coord.x + 2.f);
        int ymax = cuda::round<cuda::RoundMode::DOWN, int>(coord.y + 2.f);

        using FT = cuda::ConvertBaseTypeTo<float, ValueType>;
        auto sum = cuda::SetAll<FT>(0);

        float w, wsum = 0.f;

        for (int cy = ymin; cy <= ymax; cy++)
        {
            for (int cx = xmin; cx <= xmax; cx++)
            {
                w = GetBicubicCoeff(coord.x - cx) * GetBicubicCoeff(coord.y - cy);
                sum += w * ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(cx, cy, z, k));
                wsum += w;
            }
        }

        sum = (wsum == 0.f) ? cuda::SetAll<FT>(0) : sum / wsum;

        return cuda::SaturateCast<ValueType>(sum);
    }
    else if constexpr (I == NVCV_INTERP_AREA)
    {
        float fsx1 = coord.x * scale.x;
        float fsx2 = fsx1 + scale.x;
        float fsy1 = coord.y * scale.y;
        float fsy2 = fsy1 + scale.y;
        int   sx1  = cuda::round<cuda::RoundMode::UP, int>(fsx1);
        int   sx2  = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);
        int   sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
        int   sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

        auto out = cuda::SetAll<cuda::ConvertBaseTypeTo<float, ValueType>>(0);

        if (std::ceil(scale.x) == scale.x && std::ceil(scale.y) == scale.y)
        {
            float invscale = 1.f / (scale.x * scale.y);

            for (int dy = sy1; dy < sy2; ++dy)
                for (int dx = sx1; dx < sx2; ++dx)
                {
                    out = out + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(dx, dy, z, k)) * invscale;
                }
        }
        else
        {
            float invscale = 1.f / (std::min(scale.x, size.x - fsx1) * std::min(scale.y, size.y - fsy1));

            for (int dy = sy1; dy < sy2; ++dy)
            {
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(dx, dy, z, k)) * invscale;

                if (sx1 > fsx1)
                    out = out
                        + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx1 - 1, dy, z, k))
                              * ((sx1 - fsx1) * invscale);

                if (sx2 < fsx2)
                    out = out
                        + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx2, dy, z, k))
                              * ((fsx2 - sx2) * invscale);
            }

            if (sy1 > fsy1)
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out
                        + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(dx, sy1 - 1, z, k))
                              * ((sy1 - fsy1) * invscale);

            if (sy2 < fsy2)
                for (int dx = sx1; dx < sx2; ++dx)
                    out = out
                        + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(dx, sy2, z, k))
                              * ((fsy2 - sy2) * invscale);

            if ((sy1 > fsy1) && (sx1 > fsx1))
                out = out
                    + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx1 - 1, sy1 - 1, z, k))
                          * ((sy1 - fsy1) * (sx1 - fsx1) * invscale);

            if ((sy1 > fsy1) && (sx2 < fsx2))
                out = out
                    + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx2, sy1 - 1, z, k))
                          * ((sy1 - fsy1) * (fsx2 - sx2) * invscale);

            if ((sy2 < fsy2) && (sx2 < fsx2))
                out = out
                    + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx2, sy2, z, k))
                          * ((fsy2 - sy2) * (fsx2 - sx2) * invscale);

            if ((sy2 < fsy2) && (sx1 > fsx1))
                out = out
                    + ValueAt<B>(vec, strides, size, bValue, GetCoord<N>(sx1 - 1, sy2, z, k))
                          * ((fsy2 - sy2) * (sx1 - fsx1) * invscale);
        }

        return cuda::SaturateCast<ValueType>(out);
    }
}

} // namespace nvcv::test

#endif // NVCV_TESTS_COMMON_HASHUTILS_HPP
