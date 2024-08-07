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

#include "ConvUtils.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>     // for SaturateCast, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>      // for operator *, etc.
#include <cvcuda/cuda_tools/MathWrappers.hpp> // for min/max
#include <cvcuda/cuda_tools/SaturateCast.hpp> // for SaturateCast, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp>   // for BaseType, etc.
#include <nvcv/util/Assert.h>                 // for NVCV_ASSERT, etc.

namespace nvcv::test {

namespace detail {

template<typename T>
inline void Convolve(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
                     const long3 &srcStrides, const int3 &shape, const std::vector<float> &kernel,
                     const Size2D &kernelSize, int2 &kernelAnchor, const NVCVBorderType &borderMode,
                     const float4 &borderValue)
{
    using BT  = cuda::BaseType<T>;
    using WT  = cuda::ConvertBaseTypeTo<float, T>;
    int2 size = cuda::DropCast<2>(shape);

    T borderValueT;
    for (int e = 0; e < cuda::NumElements<T>; ++e)
    {
        cuda::GetElement(borderValueT, e) = static_cast<BT>(cuda::GetElement(borderValue, e));
    }

    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x = kernelSize.w / 2;
    }
    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y = kernelSize.h / 2;
    }

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                WT res = cuda::SetAll<WT>(0);

                int2 coord;

                for (int ky = 0; ky < kernelSize.h; ++ky)
                {
                    coord.y = y + ky - kernelAnchor.y;

                    for (int kx = 0; kx < kernelSize.w; ++kx)
                    {
                        coord.x = x + kx - kernelAnchor.x;

                        T srcValue = IsInside(coord, size, borderMode)
                                       ? ValueAt<T>(hSrc, srcStrides, b, coord.y, coord.x)
                                       : borderValueT;

                        res += srcValue * kernel[ky * kernelSize.w + kx];
                    }
                }

                ValueAt<T>(hDst, dstStrides, b, y, x) = cuda::SaturateCast<BT>(res);
            }
        }
    }
}

template<typename T>
inline void Morph(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
                  const long3 &srcStrides, const int3 &shape, const Size2D &kernelSize, int2 &kernelAnchor,
                  const NVCVBorderType &borderMode, NVCVMorphologyType type)
{
    using BT  = cuda::BaseType<T>;
    int2 size = cuda::DropCast<2>(shape);

    BT val
        = (type == NVCVMorphologyType::NVCV_DILATE) ? std::numeric_limits<BT>::min() : std::numeric_limits<BT>::max();
    T borderValueT;
    for (int e = 0; e < cuda::NumElements<T>; ++e)
    {
        cuda::GetElement(borderValueT, e) = val;
    }

    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x = kernelSize.w / 2;
    }
    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y = kernelSize.h / 2;
    }

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                T res = cuda::SetAll<T>(val);

                int2 coord;

                for (int ky = 0; ky < kernelSize.h; ++ky)
                {
                    coord.y = y + ky - kernelAnchor.y;

                    for (int kx = 0; kx < kernelSize.w; ++kx)
                    {
                        coord.x = x + kx - kernelAnchor.x;

                        T srcValue = IsInside(coord, size, borderMode)
                                       ? ValueAt<T>(hSrc, srcStrides, b, coord.y, coord.x)
                                       : borderValueT;

                        res = (type == NVCVMorphologyType::NVCV_DILATE) ? cuda::max(res, srcValue)
                                                                        : cuda::min(res, srcValue);
                    }
                }
                ValueAt<T>(hDst, dstStrides, b, y, x) = cuda::SaturateCast<BT>(res);
            }
        }
    }
}

#define NVCV_TEST_INST(TYPE)                                                                                            \
    template const TYPE &ValueAt<TYPE>(const std::vector<uint8_t> &, long3, int, int, int);                             \
    template TYPE       &ValueAt<TYPE>(std::vector<uint8_t> &, long3, int, int, int);                                   \
    template void        Convolve<TYPE>(std::vector<uint8_t> & hDst, const long3 &dstStrides,                           \
                                 const std::vector<uint8_t> &hSrc, const long3 &srcStrides, const int3 &shape,   \
                                 const std::vector<float> &kernel, const Size2D &kernelSize, int2 &kernelAnchor, \
                                 const NVCVBorderType &borderMode, const float4 &borderValue);                   \
    template void Morph<TYPE>(std::vector<uint8_t> & hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,   \
                              const long3 &srcStrides, const int3 &shape, const Size2D &kernelSize,                     \
                              int2 &kernelAnchor, const NVCVBorderType &borderMode, NVCVMorphologyType type)

NVCV_TEST_INST(uint8_t);
NVCV_TEST_INST(ushort);
NVCV_TEST_INST(uchar3);
NVCV_TEST_INST(uchar4);
NVCV_TEST_INST(float4);
NVCV_TEST_INST(float3);
NVCV_TEST_INST(float);

#undef NVCV_TEST_INST

} // namespace detail

void Convolve(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
              const long3 &srcStrides, const int3 &shape, const ImageFormat &format, const std::vector<float> &kernel,
              const Size2D &kernelSize, int2 &kernelAnchor, const NVCVBorderType &borderMode, const float4 &borderValue)
{
    NVCV_ASSERT(format.numPlanes() == 1);

    switch (format.planeDataType(0))
    {
#define NVCV_TEST_CASE(DATATYPE, TYPE)                                                                      \
    case NVCV_DATA_TYPE_##DATATYPE:                                                                         \
        detail::Convolve<TYPE>(hDst, dstStrides, hSrc, srcStrides, shape, kernel, kernelSize, kernelAnchor, \
                               borderMode, borderValue);                                                    \
        break

        NVCV_TEST_CASE(U8, uint8_t);
        NVCV_TEST_CASE(U16, ushort);
        NVCV_TEST_CASE(3U8, uchar3);
        NVCV_TEST_CASE(4U8, uchar4);
        NVCV_TEST_CASE(4F32, float4);
        NVCV_TEST_CASE(3F32, float3);
        NVCV_TEST_CASE(F32, float);

#undef NVCV_TEST_CASE

    default:
        break;
    }
}

void Morph(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
           const long3 &srcStrides, const int3 &shape, const ImageFormat &format, const Size2D &kernelSize,
           int2 &kernelAnchor, const NVCVBorderType &borderMode, NVCVMorphologyType type)
{
    NVCV_ASSERT(format.numPlanes() == 1);

    switch (format.planeDataType(0))
    {
#define NVCV_TEST_CASE(DATATYPE, TYPE)                                                                              \
    case NVCV_DATA_TYPE_##DATATYPE:                                                                                 \
        detail::Morph<TYPE>(hDst, dstStrides, hSrc, srcStrides, shape, kernelSize, kernelAnchor, borderMode, type); \
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

std::vector<float> ComputeMeanKernel(nvcv::Size2D kernelSize)
{
    std::size_t ks = kernelSize.w * kernelSize.h;
    float       kv = 1.f / ks;

    std::vector<float> kernel(ks, kv);
    return kernel;
}

std::vector<float> ComputeGaussianKernel(nvcv::Size2D kernelSize, double2 sigma)
{
    std::vector<float> kernel(kernelSize.w * kernelSize.h);

    int2 half{kernelSize.w / 2, kernelSize.h / 2};

    float sx  = 2.f * sigma.x * sigma.x;
    float sy  = 2.f * sigma.y * sigma.y;
    float s   = 2.f * sigma.x * sigma.y * M_PI;
    float sum = 0.f;
    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            float kv = std::exp(-((x * x) / sx + (y * y) / sy)) / s;

            kernel[(y + half.y) * kernelSize.w + (x + half.x)] = kv;

            sum += kv;
        }
    }
    for (int i = 0; i < kernelSize.w * kernelSize.h; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

} // namespace nvcv::test
