/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "FlipUtils.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>     // for SaturateCast, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>      // for operator *, etc.
#include <cvcuda/cuda_tools/SaturateCast.hpp> // for SaturateCast, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp>   // for BaseType, etc.
#include <nvcv/util/Assert.h>                 // for NVCV_ASSERT, etc.

namespace nvcv::test {

namespace detail {

inline int FlippedX(int x, int width, int flipCode)
{
    return flipCode == 0 ? x : width - 1 - x;
}

inline int FlippedY(int y, int height, int flipCode)
{
    return flipCode > 0 ? y : height - 1 - y;
}

template<typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<const T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

template<typename T>
inline T FlippedValue(const std::vector<uint8_t> &hSrc, const long3 &srcStrides, int b, int y, int x, int2 size,
                      int flipCode)
{
    return ValueAt<T>(hSrc, srcStrides, b, FlippedY(y, size.y, flipCode), FlippedX(x, size.x, flipCode));
}

template<typename T>
inline void flip(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
                 const long3 &srcStrides, const int3 &shape, int flipCode)
{
    using BT  = cuda::BaseType<T>;
    int2 size = cuda::DropCast<2>(shape);

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                ValueAt<T>(hDst, dstStrides, b, y, x)
                    = cuda::SaturateCast<BT>(FlippedValue<T>(hSrc, srcStrides, b, y, x, size, flipCode));
            }
        }
    }
}

#define NVCV_TEST_INST(TYPE)                                                                                         \
    template const TYPE &ValueAt<TYPE>(const std::vector<uint8_t> &, long3, int, int, int);                          \
    template TYPE       &ValueAt<TYPE>(std::vector<uint8_t> &, long3, int, int, int);                                \
    template void flip<TYPE>(std::vector<uint8_t> & hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc, \
                             const long3 &srcStrides, const int3 &shape, int flipCode)

NVCV_TEST_INST(uint8_t);
NVCV_TEST_INST(ushort);
NVCV_TEST_INST(uchar3);
NVCV_TEST_INST(uchar4);
NVCV_TEST_INST(float4);
NVCV_TEST_INST(float3);

#undef NVCV_TEST_INST

} // namespace detail

void FlipCPU(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
             const long3 &srcStrides, const int3 &shape, const ImageFormat &format, int flipCode)
{
    NVCV_ASSERT(format.numPlanes() == 1);

    switch (static_cast<NVCVDataType>(format.planeDataType(0)))
    {
#define NVCV_TEST_CASE(DATATYPE, TYPE)                                           \
    case NVCV_DATA_TYPE_##DATATYPE:                                              \
        detail::flip<TYPE>(hDst, dstStrides, hSrc, srcStrides, shape, flipCode); \
        break

        NVCV_TEST_CASE(U8, uint8_t);
        NVCV_TEST_CASE(U16, ushort);
        NVCV_TEST_CASE(3U8, uchar3);
        NVCV_TEST_CASE(4U8, uchar4);
        NVCV_TEST_CASE(4F32, float4);
        NVCV_TEST_CASE(3F32, float3);

#undef NVCV_TEST_CASE

    default:
        break;
    }
}

} // namespace nvcv::test
