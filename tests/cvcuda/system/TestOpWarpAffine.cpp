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

#include "Definitions.hpp"
#include "PlanarParityUtils.hpp"

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>
#include <random>

namespace nvcvcuda = nvcv::cuda;
namespace test     = nvcv::test;
using namespace nvcv::cuda;
using namespace test;

//#define DBG 1

template<typename T>
static float ToFloat(T value)
{
    return static_cast<float>(value);
}

template<typename T>
static T getPixel(const T *srcPtr, const int y, const int x, int k, int width, int height, int srcStride,
                  int elementsPerPixel, NVCVBorderType borderMode, const float4 borderVal)
{
    int2 coord = {x, y};
    int2 size  = {width, height};
    if (borderMode == NVCV_BORDER_CONSTANT)
    {
        return (x >= 0 && x < width && y >= 0 && y < height) ? srcPtr[y * srcStride + x * elementsPerPixel + k]
                                                             : static_cast<T>(GetElement(borderVal, k));
    }
    else if (borderMode == NVCV_BORDER_REPLICATE)
    {
        ReplicateBorderIndex(coord, size);
        return srcPtr[coord.y * srcStride + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT)
    {
        ReflectBorderIndex(coord, size);
        return srcPtr[coord.y * srcStride + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT101)
    {
        Reflect101BorderIndex(coord, size);
        return srcPtr[coord.y * srcStride + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_WRAP)
    {
        WrapBorderIndex(coord, size);
        return srcPtr[coord.y * srcStride + coord.x * elementsPerPixel + k];
    }
    else
    {
        return 0;
    }
}

inline float calcBicubicCoeff(float x_)
{
    float x = std::abs(x_);
    if (x <= 1.0f)
    {
        return x * x * (1.5f * x - 2.5f) + 1.0f;
    }
    else if (x < 2.0f)
    {
        return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }
}

static void invertAffineTransform(const NVCVAffineTransform xform, NVCVAffineTransform inverseXform)
{
    float den       = xform[0] * xform[4] - xform[1] * xform[3];
    den             = std::abs(den) > 1e-5f ? 1.0f / den : 0.0f;
    inverseXform[0] = xform[4] * den;
    inverseXform[1] = -xform[1] * den;
    inverseXform[2] = (xform[1] * xform[5] - xform[4] * xform[2]) * den;
    inverseXform[3] = -xform[3] * den;
    inverseXform[4] = xform[0] * den;
    inverseXform[5] = (xform[3] * xform[2] - xform[0] * xform[5]) * den;
}

template<typename T>
inline T clampU8(float value)
{
    value = std::rint(value);
    if (value < 0.0f)
    {
        return 0;
    }
    if (value > 255.0f)
    {
        return 255;
    }
    return static_cast<T>(value);
}

template<typename T>
static void StoreLinearPixel(T *dstPtr, int dstBase, const T *srcPtr, float src_x, float src_y, int srcWidth,
                             int srcHeight, int srcStride, int elementsPerPixel, NVCVBorderType borderMode,
                             const float4 borderVal)
{
    const auto x1 = static_cast<int>(std::floor(src_x));
    const auto y1 = static_cast<int>(std::floor(src_y));

    const int x2 = x1 + 1;
    const int y2 = y1 + 1;

    for (int k = 0; k < elementsPerPixel; k++)
    {
        float out = 0;

        T src_reg
            = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out += ToFloat(src_reg) * ((ToFloat(x2) - src_x) * (ToFloat(y2) - src_y));

        src_reg
            = getPixel<T>(srcPtr, y1, x2, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = out + ToFloat(src_reg) * ((src_x - ToFloat(x1)) * (ToFloat(y2) - src_y));

        src_reg
            = getPixel<T>(srcPtr, y2, x1, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = out + ToFloat(src_reg) * ((ToFloat(x2) - src_x) * (src_y - ToFloat(y1)));

        src_reg
            = getPixel<T>(srcPtr, y2, x2, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = out + ToFloat(src_reg) * ((src_x - ToFloat(x1)) * (src_y - ToFloat(y1)));

        dstPtr[dstBase + k] = clampU8<T>(out);
    }
}

template<typename T>
static void StoreNearestPixel(T *dstPtr, int dstBase, const T *srcPtr, float src_x, float src_y, int srcWidth,
                              int srcHeight, int srcStride, int elementsPerPixel, NVCVBorderType borderMode,
                              const float4 borderVal)
{
    const auto x1 = static_cast<int>(std::floor(src_x + .5f));
    const auto y1 = static_cast<int>(std::floor(src_y + .5f));

    for (int k = 0; k < elementsPerPixel; k++)
    {
        dstPtr[dstBase + k]
            = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
    }
}

template<typename T>
static void StoreCubicPixel(T *dstPtr, int dstBase, const T *srcPtr, float src_x, float src_y, int srcWidth,
                            int srcHeight, int srcStride, int elementsPerPixel, NVCVBorderType borderMode,
                            const float4 borderVal)
{
    const auto xmin = static_cast<int>(std::ceil(src_x - 2.0f));
    const auto xmax = static_cast<int>(std::floor(src_x + 2.0f));

    const auto ymin = static_cast<int>(std::ceil(src_y - 2.0f));
    const auto ymax = static_cast<int>(std::floor(src_y + 2.0f));

    for (int k = 0; k < elementsPerPixel; k++)
    {
        float sum  = 0;
        float wsum = 0;

        for (int cy = ymin; cy <= ymax; cy += 1)
        {
            for (int cx = xmin; cx <= xmax; cx += 1)
            {
                const float w = calcBicubicCoeff(src_x - ToFloat(cx)) * calcBicubicCoeff(src_y - ToFloat(cy));
                T src_reg = getPixel<T>(srcPtr, cy, cx, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode,
                                        borderVal);
                sum += w * ToFloat(src_reg);
                wsum += w;
            }
        }

        dstPtr[dstBase + k] = clampU8<T>(wsum == 0.0f ? 0.0f : sum / wsum);
    }
}

template<typename T>
static void WarpAffineGold(std::vector<uint8_t> &hDst, int dstStride, nvcv::Size2D dstSize,
                           const std::vector<uint8_t> &hSrc, int srcStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt,
                           const NVCVAffineTransform xform, const int flags, NVCVBorderType borderMode,
                           const float4 borderVal)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    int srcWidth  = srcSize.w;
    int srcHeight = srcSize.h;

    const int interpolation = flags & NVCV_INTERP_MAX;

    NVCVAffineTransform xform1;

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        for (int i = 0; i < 6; i++)
        {
            xform1[i] = xform[i];
        }
    }
    else
    {
        invertAffineTransform(xform, xform1);
    }

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            auto src_x = ToFloat(dst_x) * xform1[0] + ToFloat(dst_y) * xform1[1] + xform1[2];
            auto src_y = ToFloat(dst_x) * xform1[3] + ToFloat(dst_y) * xform1[4] + xform1[5];

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                StoreLinearPixel(dstPtr, dst_y * dstStride + dst_x * elementsPerPixel, srcPtr, src_x, src_y, srcWidth,
                                 srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
            }
            else if (interpolation == NVCV_INTERP_NEAREST)
            {
                StoreNearestPixel(dstPtr, dst_y * dstStride + dst_x * elementsPerPixel, srcPtr, src_x, src_y, srcWidth,
                                  srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
            }
            else if (interpolation == NVCV_INTERP_CUBIC)
            {
                StoreCubicPixel(dstPtr, dst_y * dstStride + dst_x * elementsPerPixel, srcPtr, src_x, src_y, srcWidth,
                                srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
            }
            else
            {
                return;
            }
        }
    }
}

static const std::map<std::vector<int>, std::vector<std::vector<float>>> mapOfTransformationMatrix = {
    {{5, 4, 5, 4},
     {{1.f, 0.f, 0.f, 0.f, 1.f, 0.f},
     {1.f, 0.f, 1.f, 0.f, 1.f, 2.f},
     {1.f, 2.f, 1.f, 2.f, 1.f, 2.f},
     {0.5f, 2.f, 1.f, 0.75f, 1.f, 2.f}}},
    {{5, 4, 6, 8},
     {{1.f, 0.f, 0.f, 0.f, 1.f, 0.f},
     {1.f, 0.f, 1.f, 0.f, 1.f, 2.f},
     {1.f, 2.f, 1.f, 2.f, 1.f, 2.f},
     {0.5f, 2.f, 1.f, 0.75f, 1.f, 2.f}}},
    {{7, 8, 4, 5},
     {{1.f, 0.f, 0.f, 0.f, 1.f, 0.f},
     {1.f, 0.f, 1.f, 0.f, 1.f, 2.f},
     {1.f, 2.f, 1.f, 2.f, 1.f, 2.f},
     {0.5f, 2.f, 1.f, 0.75f, 1.f, 2.f}}}
};

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffine, test::ValueList<int, int, int, int, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int, bool>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,     transformation_matrix,       interpolation,              borderType,  borderValue, batchSize, inverseAffine
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},

    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},

    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},

    // LINEAR INTERP
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0,  NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},

    // number of images in batch
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         1,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         8,         true},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,        16,         true},

    // CUBIC INTERP
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         true},

    // inverse warp affine
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,          false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,          false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,          false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,          false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,          false},

});

// clang-format on

TEST_P(OpWarpAffine, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    const NVCVAffineTransform xform = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
                                       GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>()};

    NVCVInterpolationType interpolation = GetParamValue<10>();

    NVCVBorderType borderMode = GetParamValue<11>();

    const float4 borderValue = {GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()};

    int numberOfImages = GetParamValue<16>();

    bool inverseMap = GetParamValue<17>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecStride = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::uniform_int_distribution rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecStride);
        for (uint8_t &value : srcVec[i])
        {
            value = static_cast<uint8_t>(rand(randEng));
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecStride,
                               srcVecStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    cvcuda::WarpAffine warpAffineOp(0);
    EXPECT_NO_THROW(warpAffineOp(stream, imgSrc, imgDst, xform, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecStride);

        // Generate gold result
        WarpAffineGold<uint8_t>(goldVec, dstVecStride, {dstWidth, dstHeight}, srcVec[i], srcVecStride,
                                {srcWidth, srcHeight}, fmt, xform, flags, borderMode, borderValue);

#if DBG
        std::cout << "\nPrint src vec " << std::endl;
        for (int k = 0; k < srcHeight; k++)
        {
            for (int j = 0; j < srcVecStride; j++)
            {
                std::cout << static_cast<int>(srcVec[i][k * srcVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecStride; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint warped output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecStride; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpWarpAffine, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    std::vector<float> transMatrix;
    transMatrix.resize(6);
    transMatrix[0] = GetParamValue<4>();
    transMatrix[1] = GetParamValue<5>();
    transMatrix[2] = GetParamValue<6>();
    transMatrix[3] = GetParamValue<7>();
    transMatrix[4] = GetParamValue<8>();
    transMatrix[5] = GetParamValue<9>();

    NVCVInterpolationType interpolation = GetParamValue<10>();

    NVCVBorderType borderMode = GetParamValue<11>();

    const float4 borderValue = {GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()};

    int numberOfImages = GetParamValue<16>();

    bool inverseMap = GetParamValue<17>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numberOfImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    auto         transMatrixTensorData = transMatrixTensor.exportData<nvcv::TensorDataStridedCuda>();

    auto transMatrixTensorDataAccess = nvcv::TensorDataAccessStrided::Create(*transMatrixTensorData);
    ASSERT_TRUE(transMatrixTensorDataAccess);

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndInputDimsIndex(0, static_cast<int>(mapOfTransformationMatrix.size() - 1));
    std::uniform_int_distribution rndTransformationMatrixIndex(0, 3);

    std::vector<nvcv::Image>        imgSrc;
    std::vector<nvcv::Image>        imgDst;
    std::vector<std::vector<float>> transMatrixHostVec;
    transMatrixHostVec.resize(numberOfImages);

    // List the keys from the map for easy access
    std::vector<std::vector<int>> keysOfMapOfTransformationMatrix;
    for (const auto &[key, value] : mapOfTransformationMatrix)
    {
        keysOfMapOfTransformationMatrix.push_back(key);
    }

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpSrcWidth  = srcWidth;
        int tmpSrcHeight = srcHeight;

        int tmpDstWidth  = dstWidth;
        int tmpDstHeight = dstHeight;

        std::vector<float> tmpTransMatrix(transMatrix);

        int dictInputIndex          = rndInputDimsIndex(randEng);
        int dictTransformationIndex = rndTransformationMatrixIndex(randEng);

        std::vector<int>   key                        = keysOfMapOfTransformationMatrix[dictInputIndex];
        std::vector<float> chosenTransformationMatrix = mapOfTransformationMatrix.at(key)[dictTransformationIndex];
        // Legacy Reflect & Reflect101 has a bug. So, do special thing for them
        if (i > 0 && !(borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_REFLECT101))
        {
            tmpSrcWidth  = key[0];
            tmpSrcHeight = key[1];

            tmpDstWidth  = key[2];
            tmpDstHeight = key[3];

            tmpTransMatrix = chosenTransformationMatrix;
        }

        imgSrc.emplace_back(nvcv::Size2D{tmpSrcWidth, tmpSrcHeight}, fmt);

        imgDst.emplace_back(nvcv::Size2D{tmpDstWidth, tmpDstHeight}, fmt);

        transMatrixHostVec[i] = tmpTransMatrix;

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(transMatrixTensorDataAccess->sampleData(i),
                                    transMatrixTensorDataAccess->sampleStride(), transMatrixHostVec[i].data(),
                                    sizeof(float) * 6, sizeof(float) * 6, 1, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int sampleSrcWidth  = srcData->plane(0).width;
        int sampleSrcHeight = srcData->plane(0).height;

        int srcStride = sampleSrcWidth * fmt.planePixelStrideBytes(0);

        srcVecStride[i] = srcStride;

        std::uniform_int_distribution rand(0, 255);

        srcVec[i].resize(sampleSrcHeight * srcStride);
        for (uint8_t &value : srcVec[i])
        {
            value = static_cast<uint8_t>(rand(randEng));
        }

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcStride,
                               srcStride, // vec has no padding
                               sampleSrcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::WarpAffine warpAffineOp(numberOfImages);
    EXPECT_NO_THROW(warpAffineOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int sampleSrcWidth  = srcData->plane(0).width;
        int sampleSrcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int sampleDstWidth  = dstData->plane(0).width;
        int sampleDstHeight = dstData->plane(0).height;

        int srcStride = sampleSrcWidth * fmt.planePixelStrideBytes(0);
        int dstStride = sampleDstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(sampleDstHeight * dstStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstStride, // vec has no padding
                               sampleDstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(sampleDstHeight * dstStride);

        NVCVAffineTransform transMatrixForGold;
        transMatrixForGold[0] = transMatrixHostVec[i][0];
        transMatrixForGold[1] = transMatrixHostVec[i][1];
        transMatrixForGold[2] = transMatrixHostVec[i][2];
        transMatrixForGold[3] = transMatrixHostVec[i][3];
        transMatrixForGold[4] = transMatrixHostVec[i][4];
        transMatrixForGold[5] = transMatrixHostVec[i][5];

        // Generate gold result
        WarpAffineGold<uint8_t>(goldVec, dstStride, {sampleDstWidth, sampleDstHeight}, srcVec[i], srcStride,
                                {sampleSrcWidth, sampleSrcHeight}, fmt, transMatrixForGold, flags, borderMode,
                                borderValue);

#if DBG
        std::cout << "\nPrint src vec " << std::endl;
        for (int k = 0; k < srcHeight; k++)
        {
            for (int j = 0; j < srcVecStride; j++)
            {
                std::cout << static_cast<int>(srcVec[i][k * srcVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecStride; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint warped output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecStride; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecStride + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// Warp samples every channel at the same transformed coordinate, so a planar input is warped
// plane-by-plane and must produce exactly the same pixels as the interleaved path. These tests feed
// identical data and transform through cvcuda::WarpAffine in both layouts and require the
// (re-interleaved) planar output to match the interleaved output bit-for-bit, across interpolation
// and border modes. CONSTANT border uses a NON-UNIFORM borderValue to exercise the per-channel
// border path that planar must reproduce. Shared scaffolding lives in PlanarParityUtils.hpp.
// =============================================================================

namespace {

// Non-trivial affine (scale + shear + translate). Any transform works since the parity check only
// requires interleaved and planar to match on identical inputs.
inline std::array<float, 6> PlanarAffine()
{
    return {1.1f, 0.05f, 3.0f, -0.03f, 0.95f, 2.0f};
}

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int dstW, int dstH, NVCVInterpolationType interp, NVCVBorderType borderMode,
                               int numImages)
{
    const std::array<float, 6> xform       = PlanarAffine();
    const float4               borderValue = {13.f, 57.f, 101.f, 211.f};
    const int32_t              flags       = interp;
    test::planar::RunTensorParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [xform, flags, borderMode, borderValue, numImages](cudaStream_t s, const nvcv::Tensor &src,
                                                           const nvcv::Tensor &dst, nvcv::ImageFormat)
        {
            cvcuda::WarpAffine op(numImages);
            EXPECT_NO_THROW(op(s, src, dst, xform.data(), flags, borderMode, borderValue));
        });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int dstW, int dstH, NVCVInterpolationType interp, NVCVBorderType borderMode,
                                 int numImages)
{
    const std::array<float, 6> xform       = PlanarAffine();
    const float4               borderValue = {13.f, 57.f, 101.f, 211.f};
    const int32_t              flags       = interp;

    // Per-image transform tensor (same affine for every image); upload synchronously so it is ready
    // before the operator runs on the parity helper's stream.
    nvcv::Tensor transMatrix(nvcv::TensorShape({numImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    {
        auto data = transMatrix.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(data, nullptr);
        auto acc = nvcv::TensorDataAccessStrided::Create(*data);
        ASSERT_TRUE(acc);
        for (int i = 0; i < numImages; ++i)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(acc->sampleData(i), acc->sampleStride(), xform.data(),
                                                sizeof(float) * 6, sizeof(float) * 6, 1, cudaMemcpyHostToDevice));
        }
    }

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [&transMatrix, flags, borderMode, borderValue, numImages](
            cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::WarpAffine op(numImages);
            EXPECT_NO_THROW(op(s, src, dst, transMatrix, flags, borderMode, borderValue));
        });
}

} // namespace

// Parameters: srcW, srcH, dstW, dstH, interpolation, borderMode, numImages, planarFmt, interleavedFmt
// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffinePlanar,
    test::ValueList<int, int, int, int, NVCVInterpolationType, NVCVBorderType, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    { 64, 48, 64, 48, NVCV_INTERP_NEAREST,  NVCV_BORDER_CONSTANT,  2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64, 48, 96, 72,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 80, 60, 64, 48,   NVCV_INTERP_CUBIC, NVCV_BORDER_REPLICATE,  1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64, 48, 64, 48,  NVCV_INTERP_LINEAR,      NVCV_BORDER_WRAP,  1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 50, 40, 60, 50, NVCV_INTERP_NEAREST,  NVCV_BORDER_CONSTANT,  2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 50, 40, 50, 40,  NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,  1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 64, 48, 96, 72,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 64, 48, 64, 48,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
});

// clang-format on

TEST_P(OpWarpAffinePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                              GetParamValue<6>());
}

TEST_P(OpWarpAffinePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                                GetParamValue<6>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffine_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    // input format, output format,
    {nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},  // interleaved in, planar out: layout mismatch
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},  // planar in, interleaved out: layout mismatch
    {nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16}
});

NVCV_TEST_SUITE_P(OpWarpAffineVarshape_Negative, test::ValueList<int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    // maxBatchSize, numImages, input format, output format
    {5, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
    {5, 5, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {5, 5, nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16},
    {0, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8},
    {2, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8}
});

// clang-format on

TEST_P(OpWarpAffine_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::ImageFormat inFmt  = GetParamValue<0>();
    const nvcv::ImageFormat outFmt = GetParamValue<1>();

    const NVCVAffineTransform xform         = {1, 0, 0, 0, 1, 0};
    NVCVInterpolationType     interpolation = NVCV_INTERP_NEAREST;
    NVCVBorderType            borderMode    = NVCV_BORDER_CONSTANT;
    const float4              borderValue   = {1, 2, 3, 4};
    bool                      inverseMap    = true;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Create input and output
    nvcv::Tensor imgSrc(2, {5, 4}, inFmt);
    nvcv::Tensor imgDst(2, {5, 4}, outFmt);

    cvcuda::WarpAffine warpAffineOp(0);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&warpAffineOp, &stream, &imgSrc, &imgDst, &xform, &flags, &borderMode, &borderValue]
                                { warpAffineOp(stream, imgSrc, imgDst, xform, flags, borderMode, borderValue); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpWarpAffineVarshape_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                     maxBatchSize = GetParamValue<0>();
    int                     numImages    = GetParamValue<1>();
    const nvcv::ImageFormat inFmt        = GetParamValue<2>();
    const nvcv::ImageFormat outFmt       = GetParamValue<3>();

    NVCVInterpolationType interpolation = NVCV_INTERP_NEAREST;
    NVCVBorderType        borderMode    = NVCV_BORDER_CONSTANT;
    const float4          borderValue   = {1, 2, 3, 4};
    bool                  inverseMap    = true;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

    // Create input and output
    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{5, 4}, inFmt);
        imgDst.emplace_back(nvcv::Size2D{5, 4}, outFmt);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::WarpAffine warpAffineOp(maxBatchSize);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&warpAffineOp, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &borderMode, &borderValue]
                  { warpAffineOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffine_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaWarpAffineCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

TEST(OpWarpAffine_Negative, invalid_border_mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVAffineTransform xform       = {1, 0, 0, 0, 1, 0};
    const float4        borderValue = {0, 0, 0, 0};
    const int           flags       = NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP;

    nvcv::Tensor imgSrc(1, {4, 4}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {4, 4}, nvcv::FMT_U8);

    cvcuda::WarpAffine op(0);

    // 5 is one past the last valid NVCVBorderType value (NVCV_BORDER_REFLECT101 = 4)
    auto invalidBorder = static_cast<NVCVBorderType>(5);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &imgSrc, &imgDst, &xform, &flags, &invalidBorder, &borderValue]
                                { op(stream, imgSrc, imgDst, xform, flags, invalidBorder, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffine_Negative, invalid_interpolation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVAffineTransform xform       = {1, 0, 0, 0, 1, 0};
    const float4        borderValue = {0, 0, 0, 0};

    nvcv::Tensor imgSrc(1, {4, 4}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {4, 4}, nvcv::FMT_U8);

    cvcuda::WarpAffine op(0);

    // NVCV_INTERP_AREA (3) is not supported by the warp ops
    const int flags = static_cast<int>(NVCV_INTERP_AREA) | NVCV_WARP_INVERSE_MAP;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &imgSrc, &imgDst, &xform, &flags, &borderValue]
                                { op(stream, imgSrc, imgDst, xform, flags, NVCV_BORDER_CONSTANT, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffineVarshape_Negative, invalid_border_mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    numImages   = 2;
    const float4 borderValue = {0, 0, 0, 0};
    const int    flags       = NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP;

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::WarpAffine op(numImages);

    auto invalidBorder = static_cast<NVCVBorderType>(5);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &invalidBorder, &borderValue]
                          { op(stream, batchSrc, batchDst, transMatrixTensor, flags, invalidBorder, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffineVarshape_Negative, invalid_interpolation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    numImages   = 2;
    const float4 borderValue = {0, 0, 0, 0};

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
        imgDst.emplace_back(nvcv::Size2D{4, 4}, nvcv::FMT_U8);
    }

    nvcv::ImageBatchVarShape batchSrc(numImages);
    nvcv::ImageBatchVarShape batchDst(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::WarpAffine op(numImages);

    // NVCV_INTERP_AREA (3) is not supported by the warp ops
    const int flags = static_cast<int>(NVCV_INTERP_AREA) | NVCV_WARP_INVERSE_MAP;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&op, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &borderValue]
                  { op(stream, batchSrc, batchDst, transMatrixTensor, flags, NVCV_BORDER_CONSTANT, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffineVarshape_Negative, different_format_varshape)
{
    std::vector<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>> extraFmts{
        { nvcv::FMT_RGB8, nvcv::FMT_RGBA8},
        {nvcv::FMT_RGBA8,  nvcv::FMT_RGB8}
    };

    for (const auto &[extraFmtSrc, extraFmtDst] : extraFmts)
    {
        cudaStream_t stream;
        EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        int                     numImages = 3;
        const nvcv::ImageFormat fmt       = nvcv::FMT_RGB8;

        NVCVInterpolationType interpolation = NVCV_INTERP_NEAREST;
        NVCVBorderType        borderMode    = NVCV_BORDER_CONSTANT;
        const float4          borderValue   = {1, 2, 3, 4};
        bool                  inverseMap    = true;

        const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

        nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

        // Create input and output
        std::vector<nvcv::Image> imgSrc;
        std::vector<nvcv::Image> imgDst;

        for (int i = 0; i < numImages - 1; ++i)
        {
            imgSrc.emplace_back(nvcv::Size2D{5, 4}, fmt);
            imgDst.emplace_back(nvcv::Size2D{5, 4}, fmt);
        }

        imgSrc.emplace_back(nvcv::Size2D{5, 4}, extraFmtSrc);
        imgDst.emplace_back(nvcv::Size2D{5, 4}, extraFmtDst);

        nvcv::ImageBatchVarShape batchSrc(numImages);
        batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

        nvcv::ImageBatchVarShape batchDst(numImages);
        batchDst.pushBack(imgDst.begin(), imgDst.end());

        cvcuda::WarpAffine warpAffineOp(numImages);
        EXPECT_EQ(
            NVCV_ERROR_INVALID_ARGUMENT,
            nvcv::ProtectCall(
                [&warpAffineOp, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &borderMode, &borderValue]
                { warpAffineOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue); }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}
