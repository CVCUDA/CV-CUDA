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
#include "HalfTestUtils.hpp"
#include "PlanarParityUtils.hpp"

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpWarpAffine.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <type_traits>

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
    // Floating-point destinations keep the interpolated value: the CUDA kernel only rounds and
    // saturates integer element types (the FP32/F16 gold paths reuse this template with T=float).
    if constexpr (std::is_floating_point_v<T>)
    {
        return value;
    }
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
        // Match InterpolationWrap's left-associated tap product and the device FMA used to
        // accumulate it, so the CPU oracle can remain bit-exact.
        float out = 0;

        T src_reg
            = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = std::fma(ToFloat(src_reg) * (ToFloat(x2) - src_x), ToFloat(y2) - src_y, out);

        src_reg
            = getPixel<T>(srcPtr, y1, x2, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = std::fma(ToFloat(src_reg) * (src_x - ToFloat(x1)), ToFloat(y2) - src_y, out);

        src_reg
            = getPixel<T>(srcPtr, y2, x1, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = std::fma(ToFloat(src_reg) * (ToFloat(x2) - src_x), src_y - ToFloat(y1), out);

        src_reg
            = getPixel<T>(srcPtr, y2, x2, k, srcWidth, srcHeight, srcStride, elementsPerPixel, borderMode, borderVal);
        out = std::fma(ToFloat(src_reg) * (src_x - ToFloat(x1)), src_y - ToFloat(y1), out);

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

// Strides are in T elements (identical to bytes for the uint8 suite); the F16 suite runs this
// reference with T=float on the widened half input.
template<typename T>
static void WarpAffineGold(std::vector<T> &hDst, int dstStride, nvcv::Size2D dstSize, const std::vector<T> &hSrc,
                           int srcStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, const NVCVAffineTransform xform,
                           const int flags, NVCVBorderType borderMode, const float4 borderVal)
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
            // Mirror WarpAffineTransform::calcCoord, fusion included. The kernel pins the
            // established row to __fmaf_rn(fx, c0, fy * c1) + c2.
            auto src_x = std::fma(ToFloat(dst_x), xform1[0], ToFloat(dst_y) * xform1[1]) + xform1[2];
            auto src_y = std::fma(ToFloat(dst_x), xform1[3], ToFloat(dst_y) * xform1[4]) + xform1[5];

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
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
    {         7,         8,        4,         5,        0.5, 2, 1, 0.75, 1, 2, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         true},
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

    // Widths past one block's worth of output. TensorWarpXUnroll<I>() outputs per thread
    // are strided by blockDim.x (32), so for the unrolled modes a thread writes at dstX0,
    // +32, +64 and +96; every other width in this suite is under 32, which guards all but
    // the first iteration off and leaves the unrolled coordinate math uncompared against
    // the gold. 100 exercises all four with a ragged tail and 128 an exact multiple -- but
    // both fit one block (grid.x = ceil(w / 128) = 1), so 300 is here to drive blockIdx.x
    // past zero and exercise the block-origin arithmetic the unroll multiplies. 129 is the
    // narrowest tail: block 1 keeps a single live column at dstX = 128 and guards the other
    // three lanes off entirely, where 100, 128 and 300 leave tails of 4, 32 and 12 columns and
    // never take the guard on fewer than four.
    //
    // The height is 27, deliberately not a multiple of blockDim.y (kBlock / 4 == 8). grid.y then
    // covers 32 rows, so in the last block dstY 24 through 26 do the work and dstY 27 through 31
    // take the unrolled kernel's `dstY >= dstSize.y` early return -- divergently, within one
    // block, after StageWarpTransform's __syncthreads(). Small rows elsewhere in this file
    // already reach that return; what no row reached before is taking it while also driving the
    // unroll lanes and blockIdx.x past zero, because every wide row here was 24 high, an exact
    // multiple of 8. Keep these heights off multiples of 8.
    //
    // The upscaling rows below reach the wide destination but not with in-bounds source
    // taps: {2,2,1,3,1,2} inverse-mapped puts src_y = 3*dst_x + dst_y + 2 past srcHeight=48
    // by dst_x=16, so from there on both gold and kernel return border and agree no matter
    // what the unrolled coordinate math computes. They are kept for the border path; the
    // downscaling rows that follow are the ones that actually compare interpolated taps,
    // and they sample in bounds across the whole destination (every dst pixel, every live
    // unroll lane, and for widths 129 and 300 blockIdx.x past zero).
    {        64,        48,      100,        27,          1, 0, 0, 0, 1, 0,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      100,        27,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      100,        27,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         2,         true},
    {        64,        48,      128,        27,          1, 0, 0, 0, 1, 0,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      128,        27,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      100,        27,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,        false},
    {        64,        48,      300,        27,          5, 0, 0, 0, 5, 0,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,        false},
    {        64,        48,      300,        27,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      300,        27,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         2,         true},

    {        64,        48,      100,        27, 0.5f, 0.1f, 0.5f, 0.1f, 0.5f, 1.f,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      128,        27, 0.4f, 0.1f, 0.5f, 0.1f, 0.5f, 1.f,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         2,         true},
    {        64,        48,      128,        27, 0.4f, 0.1f, 0.5f, 0.1f, 0.5f, 1.f, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      129,        27, 0.4f, 0.1f, 0.5f, 0.1f, 0.5f, 1.f,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      129,        27, 0.4f, 0.1f, 0.5f, 0.1f, 0.5f, 1.f, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      300,        27, 0.2f, 0.1f, 0.5f, 0.05f, 0.5f, 1.f,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         2,         true},
    {        64,        48,      300,        27, 0.2f, 0.1f, 0.5f, 0.05f, 0.5f, 1.f, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         2,         true},

    // Wide rows on a 3-element pixel. Every other wide row is RGBA8, and NumElements<T> == 3 is what
    // makes WarpAffine stage the transform in shared memory (WarpCommon.cuh: useSharedTransform),
    // so warp<true, NIX> is the instantiation these reach. With the unrolled modes it is
    // warp<true, 4> -- the kernel whose
    // __syncthreads() now precedes the dstY early return -- which no other gold comparison in
    // this suite runs. REFLECT101 is the selector because no pre-existing wide row uses it, so
    // adding these does not change the element type of any row already here; keep it that way, or
    // a future wide REFLECT101 row silently loses its 4-element coverage.
    {        64,        48,      300,        27, 0.2f, 0.1f, 0.5f, 0.05f, 0.5f, 1.f,  NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         2,         true},
    {        64,        48,      300,        27, 0.2f, 0.1f, 0.5f, 0.05f, 0.5f, 1.f, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         2,         true},

});

// clang-format on

TEST(OpWarpAffine, rgb_linear_constant_edges)
{
    constexpr int             srcWidth  = 73;
    constexpr int             srcHeight = 41;
    constexpr int             channels  = 3;
    const NVCVAffineTransform xform     = {0.5f, 0.125f, -2.5f, -0.125f, 0.75f, 0.25f};
    const float4              border    = {17, 23, 31, 0};
    const int                 flags     = NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP;
    std::vector<uint8_t>      input(srcWidth * srcHeight * channels);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<uint8_t>((i * 37 + i / 13) & 255);
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    nvcv::Tensor src(1, {srcWidth, srcHeight}, nvcv::FMT_RGB8);
    auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(0), srcAccess->rowStride(), input.data(),
                                        srcWidth * channels, srcWidth * channels, srcHeight, cudaMemcpyHostToDevice));

    cvcuda::WarpAffine op(0);
    for (const auto size : {
             nvcv::Size2D{  1,   7},
             nvcv::Size2D{129,  27},
             nvcv::Size2D{300, 129}
    })
    {
        SCOPED_TRACE(size.w);
        nvcv::Tensor dst(1, size, nvcv::FMT_RGB8);
        op(stream, src, dst, xform, flags, NVCV_BORDER_CONSTANT, border);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(dstData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(dstAccess);
        std::vector<uint8_t> actual(size.w * size.h * channels);
        std::vector<uint8_t> expected(actual.size());
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), size.w * channels, dstAccess->sampleData(0),
                                            dstAccess->rowStride(), size.w * channels, size.h, cudaMemcpyDeviceToHost));
        WarpAffineGold<uint8_t>(expected, size.w * channels, size, input, srcWidth * channels, {srcWidth, srcHeight},
                                nvcv::FMT_RGB8, xform, flags, NVCV_BORDER_CONSTANT, border);
        EXPECT_EQ(expected, actual);
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

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

    // RGB8 selects the 3-element kernel; see the wide REFLECT101 rows in the parameter list.
    const bool threeChannel = (interpolation == NVCV_INTERP_CUBIC && borderMode == NVCV_BORDER_REFLECT && inverseMap)
                           || (dstWidth > 32 && borderMode == NVCV_BORDER_REFLECT101);

    const nvcv::ImageFormat fmt = threeChannel ? nvcv::FMT_RGB8 : nvcv::FMT_RGBA8;

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
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

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
    // 300 is the only width here past one block's worth of unrolled output (32 * 4), so it is the
    // only row that drives the fourth unroll lane and blockIdx.x > 0 through a bit-exact
    // comparison; every other row fits one block in x. The source is 300 wide on purpose: the
    // operator inverts PlanarAffine(), which maps dst_x to about 0.91*dst_x - 2.6, so a 64-wide
    // source runs out at dst_x 74 and every lane past the first two would compare constant border
    // against constant border. At 300x48 the taps stay inside the source across the whole
    // destination (src_x reaches 269, src_y 34) while dst_x 0..2 still lands on the border.
    { 300, 48, 300, 27, NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 80, 60, 64, 48,   NVCV_INTERP_CUBIC, NVCV_BORDER_REPLICATE,  1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 64, 48, 64, 48,  NVCV_INTERP_LINEAR,      NVCV_BORDER_WRAP,  1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 50, 40, 60, 50, NVCV_INTERP_NEAREST,  NVCV_BORDER_CONSTANT,  2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 50, 40, 50, 40,  NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,  1,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 53, 41, 67, 55,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  2,   nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8},
    { 64, 48, 96, 72,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  1,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 64, 48, 64, 48,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    // F32 fusion: C3 exercises folded borders; C4 exercises distinct constant values per plane.
    { 80, 60, 128, 96,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REFLECT,  2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    { 80, 60, 128, 96,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  1, nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32},
    // Odd destination height exercises the final row tile for byte and float planes.
    { 80, 60, 128, 97, NVCV_INTERP_NEAREST, NVCV_BORDER_REPLICATE,  1,    nvcv::FMT_RGB8p,    nvcv::FMT_RGB8},
    { 80, 60, 128, 97, NVCV_INTERP_NEAREST,      NVCV_BORDER_WRAP,  2,  nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32},
    // F16 planes run the same half kernel as interleaved single-channel F16, so parity stays
    // bit-exact; the CUBIC row is the device coverage for the cubic half kernel (the F16 gold
    // suite below sticks to gather/linear cases it can bound).
    { 64, 48, 96, 72,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
    { 50, 40, 50, 40, NVCV_INTERP_NEAREST, NVCV_BORDER_REPLICATE,  1, nvcv::FMT_RGBAf16p, nvcv::FMT_RGBAf16},
    { 64, 48, 64, 48,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  2,  nvcv::FMT_RGBf16p,  nvcv::FMT_RGBf16},
});

// clang-format on

TEST_P(OpWarpAffinePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<0>(), GetParamValue<1>(),
                              GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                              GetParamValue<6>());
}

template<typename T>
static void RunLinearConstantGold(nvcv::ImageFormat format)
{
    struct Case
    {
        nvcv::Size2D         srcSize;
        nvcv::Size2D         dstSize;
        int                  numImages;
        std::array<float, 6> matrix;
        bool                 inverse;
        bool                 nonzeroBorder;
    };

    const std::array<Case, 6> cases = {
        {{{1, 3}, {3, 5}, 1, {1, 0, 0, 0, 1, 0}, true, false},
         {{31, 9}, {33, 11}, 2, {1, 0, -0.5f, 0, 1, -0.25f}, true, true},
         {{129, 9}, {131, 11}, 2, {2, 2, 0, 3, 1, 0}, false, false},
         {{133, 9}, {135, 11}, 1, {2, 0, 1, 0, 4, -1}, false, true},
         {{33, 9}, {35, 11}, 2, {1, 0, 256, 0, 1, -64}, true, true},
         {{131, 17}, {129, 9}, 1, {1, 0, 0.5f, 0, 1, 0.25f}, true, false}}
    };

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::WarpAffine op(0);
    nvcv::MemAlignment alignment;
    alignment.rowAddr(256);

    const auto runCase = [stream, format, &op, &alignment](const Case &testCase)
    {
        const int    srcRowElems = testCase.srcSize.w * format.numChannels();
        const int    dstRowElems = testCase.dstSize.w * format.numChannels();
        const int    srcRowBytes = srcRowElems * sizeof(T);
        const int    dstRowBytes = dstRowElems * sizeof(T);
        const int    flags       = NVCV_INTERP_LINEAR | (testCase.inverse ? NVCV_WARP_INVERSE_MAP : 0);
        const float4 borderValue
            = testCase.nonzeroBorder ? float4{13.f, 57.f, 101.f, 211.f} : float4{0.f, 0.f, 0.f, 0.f};

        nvcv::Tensor src(testCase.numImages, testCase.srcSize, format, alignment);
        nvcv::Tensor dst(testCase.numImages, testCase.dstSize, format, alignment);
        ASSERT_EQ(src.layout(), nvcv::TENSOR_NHWC);
        ASSERT_EQ(dst.layout(), nvcv::TENSOR_NHWC);
        auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData && dstData);
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(srcAccess && dstAccess);
        ASSERT_GT(srcAccess->rowStride(), srcRowBytes);
        ASSERT_GT(dstAccess->rowStride(), dstRowBytes);

        std::vector<std::vector<T>> inputs(testCase.numImages);
        for (int sample = 0; sample < testCase.numImages; ++sample)
        {
            inputs[sample].resize(srcRowElems * testCase.srcSize.h);
            std::ranges::generate(inputs[sample],
                                  [sample, index = size_t{0}]() mutable
                                  {
                                      const auto value = static_cast<int>((index++ * 13 + sample * 17) % 127);
                                      if constexpr (std::is_floating_point_v<T>)
                                          return static_cast<T>(value - 63) / 8;
                                      else if constexpr (std::is_signed_v<T>)
                                          return static_cast<T>((value - 63) * 256);
                                      else if constexpr (sizeof(T) == 2)
                                          return static_cast<T>((value + 16) * 384);
                                      else
                                          return static_cast<T>(value + 16);
                                  });
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(srcAccess->sampleData(sample), srcAccess->rowStride(), inputs[sample].data(),
                                   srcRowBytes, srcRowBytes, testCase.srcSize.h, cudaMemcpyHostToDevice));
            ASSERT_EQ(cudaSuccess, cudaMemset2D(dstAccess->sampleData(sample), dstAccess->rowStride(), 0xff,
                                                dstRowBytes, testCase.dstSize.h));
        }

        EXPECT_NO_THROW(op(stream, src, dst, testCase.matrix.data(), flags, NVCV_BORDER_CONSTANT, borderValue));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        for (int sample = 0; sample < testCase.numImages; ++sample)
        {
            SCOPED_TRACE(sample);
            std::vector<T> actual(dstRowElems * testCase.dstSize.h);
            ASSERT_EQ(cudaSuccess,
                      cudaMemcpy2D(actual.data(), dstRowBytes, dstAccess->sampleData(sample), dstAccess->rowStride(),
                                   dstRowBytes, testCase.dstSize.h, cudaMemcpyDeviceToHost));
            const std::vector<float> goldInput(inputs[sample].begin(), inputs[sample].end());
            std::vector<float>       goldFloat(actual.size());
            WarpAffineGold<float>(goldFloat, dstRowElems, testCase.dstSize, goldInput, srcRowElems, testCase.srcSize,
                                  format, testCase.matrix.data(), flags, NVCV_BORDER_CONSTANT, borderValue);
            std::vector<T> gold(actual.size());
            std::ranges::transform(goldFloat, gold.begin(),
                                   [](float value)
                                   {
                                       if constexpr (std::is_integral_v<T>)
                                           value = std::rint(value);
                                       return static_cast<T>(value);
                                   });
            EXPECT_EQ(gold, actual);
        }
    };
    for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex)
    {
        SCOPED_TRACE(::testing::Message() << "channels=" << format.numChannels() << " case=" << caseIndex);
        ASSERT_NO_FATAL_FAILURE(runCase(cases[caseIndex]));
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffineLinearConstant, u8_matches_gold)
{
    for (nvcv::ImageFormat format : {nvcv::FMT_U8, nvcv::FMT_RGBA8}) RunLinearConstantGold<uint8_t>(format);
}

TEST(OpWarpAffineLinearConstant, unsigned16_matches_gold)
{
    for (nvcv::ImageFormat format : {nvcv::FMT_U16, nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::UNSIGNED),
                                     nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::UNSIGNED)})
        RunLinearConstantGold<uint16_t>(format);
}

TEST(OpWarpAffineLinearConstant, signed16_matches_gold)
{
    for (nvcv::ImageFormat format : {nvcv::FMT_S16, nvcv::FMT_RGBf16.dataKind(nvcv::DataKind::SIGNED),
                                     nvcv::FMT_RGBAf16.dataKind(nvcv::DataKind::SIGNED)})
        RunLinearConstantGold<int16_t>(format);
}

TEST(OpWarpAffineLinearConstant, scalar_f32_matches_gold)
{
    RunLinearConstantGold<float>(nvcv::FMT_F32);
}

TEST(OpWarpAffineLinearConstant, vector_f32_matches_gold)
{
    for (nvcv::ImageFormat format : {nvcv::FMT_RGBf32, nvcv::FMT_RGBAf32}) RunLinearConstantGold<float>(format);
}

static std::vector<uint8_t> MakeIntegerGridInput(int sample, int width, int height, int channels)
{
    std::vector<uint8_t> input(static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<uint8_t>((index * 73 + (index / channels) * 19 + sample * 101) % 256);
    }
    return input;
}

TEST(OpWarpAffineIntegerGrid, tensor_planar_gold)
{
    struct TransformCase
    {
        std::array<float, 6> matrix;
        bool                 inverse;
    };

    const std::array<TransformCase, 8> transforms = {
        {{{1, 0, 0, 0, 1, 0}, true},
         {{2, 2, 0, 3, 1, 0}, true},
         {{-1, 2, -17, 1, -1, 23}, true},
         {{1, 0, 1025, 0, 1, -513}, true},
         {{0, 0, -1, 0, 0, 1}, true},
         {{1, 0, 0.5f, 0, 1, 0.25f}, true},
         {{0.5f, 0, 1, 0, 0.5f, -2}, true},
         {{2, 0, 1, 0, 2, -1}, false}}
    };
    const std::array<nvcv::Size2D, 4> sizes = {
        {{1, 1}, {31, 9}, {129, 17}, {321, 27}}
    };
    const float4 borderValue = {13.f, 57.f, 101.f, 211.f};

    for (size_t transformIndex = 0; transformIndex < transforms.size(); ++transformIndex)
    {
        SCOPED_TRACE(transformIndex);
        const auto &transform = transforms[transformIndex];
        const int   flags     = NVCV_INTERP_CUBIC | (transform.inverse ? NVCV_WARP_INVERSE_MAP : 0);
        for (const auto &srcSize : sizes)
        {
            SCOPED_TRACE(srcSize.w);
            const nvcv::Size2D dstSize{srcSize.w + 2, srcSize.h + 2};
            for (int numImages : {1, 3})
            {
                SCOPED_TRACE(numImages);
                int  comparedSamples = 0;
                auto compare         = [&comparedSamples, &transform, srcSize, dstSize, flags, borderValue](
                                   const std::vector<uint8_t> &interleaved, const std::vector<uint8_t> &planar)
                {
                    const auto           input = MakeIntegerGridInput(comparedSamples, srcSize.w, srcSize.h, 3);
                    std::vector<uint8_t> gold(static_cast<size_t>(dstSize.w) * dstSize.h * 3);
                    WarpAffineGold<uint8_t>(gold, dstSize.w * 3, dstSize, input, srcSize.w * 3, srcSize, nvcv::FMT_RGB8,
                                            transform.matrix.data(), flags, NVCV_BORDER_REFLECT, borderValue);
                    EXPECT_EQ(gold, interleaved);
                    EXPECT_EQ(gold, planar);
                    ++comparedSamples;
                };
                test::planar::RunTensorParity(
                    nvcv::FMT_RGB8p, nvcv::FMT_RGB8, srcSize.w, srcSize.h, dstSize.w, dstSize.h, numImages,
                    [&transform, flags, borderValue](cudaStream_t stream, const nvcv::Tensor &src,
                                                     const nvcv::Tensor &dst, nvcv::ImageFormat)
                    {
                        cvcuda::WarpAffine op(0);
                        EXPECT_NO_THROW(
                            op(stream, src, dst, transform.matrix.data(), flags, NVCV_BORDER_REFLECT, borderValue));
                    },
                    MakeIntegerGridInput, compare);
                EXPECT_EQ(comparedSamples, numImages);
            }
        }
    }
}

TEST(OpWarpAffineIntegerGrid, tensor_planar_coordinate_boundary_gold)
{
    constexpr int      channels = 3;
    const nvcv::Size2D srcSize{17, 9};
    const nvcv::Size2D dstSize{7, 5};
    const float4       borderValue{13.f, 57.f, 101.f, 211.f};
    const int          flags = NVCV_INTERP_CUBIC | NVCV_WARP_INVERSE_MAP;

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    cvcuda::WarpAffine op(0);

    for (int numImages : {1, 2})
    {
        SCOPED_TRACE(numImages);
        nvcv::Tensor src     = nvcv::util::CreateTensor(numImages, srcSize.w, srcSize.h, nvcv::FMT_RGB8p);
        nvcv::Tensor dst     = nvcv::util::CreateTensor(numImages, dstSize.w, dstSize.h, nvcv::FMT_RGB8p);
        auto         srcData = src.exportData<nvcv::TensorDataStridedCuda>();
        auto         dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(srcData && dstData);
        ASSERT_EQ(srcData->layout(), numImages == 1 ? nvcv::TENSOR_CHW : nvcv::TENSOR_NCHW);
        ASSERT_EQ(dstData->layout(), srcData->layout());
        auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
        auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
        ASSERT_TRUE(srcAccess && dstAccess);

        std::vector<std::vector<uint8_t>> inputs(numImages);
        for (int sample = 0; sample < numImages; ++sample)
        {
            inputs[sample] = MakeIntegerGridInput(sample, srcSize.w, srcSize.h, channels);
            test::planar::UploadPlanarSample(
                *srcAccess, sample,
                test::planar::DeinterleaveToPlanes(inputs[sample], srcSize.w, srcSize.h, channels, 1), srcSize.w,
                srcSize.h, channels, 1);
        }

        const auto runTransform = [stream, &src, &dst, &op, &dstAccess, &inputs, srcSize, dstSize, borderValue, flags,
                                   numImages](const NVCVAffineTransform &matrix, float offset)
        {
            const std::vector<uint8_t> sentinel(dstSize.w * dstSize.h * channels, 0xA5);
            for (int sample = 0; sample < numImages; ++sample)
            {
                test::planar::UploadPlanarSample(*dstAccess, sample, sentinel, dstSize.w, dstSize.h, channels, 1);
            }
            EXPECT_NO_THROW(op(stream, src, dst, matrix, flags, NVCV_BORDER_REFLECT, borderValue));
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

            for (int sample = 0; sample < numImages; ++sample)
            {
                SCOPED_TRACE(sample);
                const auto planes
                    = test::planar::DownloadPlanarSample(*dstAccess, sample, dstSize.w, dstSize.h, channels, 1);
                const auto actual = test::planar::InterleaveFromPlanes(planes, dstSize.w, dstSize.h, channels, 1);
                std::vector<uint8_t> gold(actual.size());
                WarpAffineGold<uint8_t>(gold, dstSize.w * channels, dstSize, inputs[sample], srcSize.w * channels,
                                        srcSize, nvcv::FMT_RGB8, matrix, flags, NVCV_BORDER_REFLECT, borderValue);
                EXPECT_EQ(gold, actual);
                if (std::abs(offset) > 8388608.f)
                {
                    std::vector<uint8_t> nearest(actual.size());
                    WarpAffineGold<uint8_t>(nearest, dstSize.w * channels, dstSize, inputs[sample],
                                            srcSize.w * channels, srcSize, nvcv::FMT_RGB8, matrix,
                                            NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP, NVCV_BORDER_REFLECT,
                                            borderValue);
                    EXPECT_NE(gold, nearest);
                }
            }
        };
        for (float offset : {-8388609.f, -8388608.f, -8388607.f, 8388607.f, 8388608.f, 8388609.f})
        {
            SCOPED_TRACE(offset);
            for (int axis : {0, 1})
            {
                SCOPED_TRACE(axis);
                NVCVAffineTransform matrix{1, 0, 0, 0, 1, 0};
                matrix[axis == 0 ? 0 : 4] = 0;
                matrix[axis * 3 + 2]      = offset;
                ASSERT_NO_FATAL_FAILURE(runTransform(matrix, offset));
            }
        }
    }
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpWarpAffinePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<7>(), GetParamValue<8>(), GetParamValue<0>(), GetParamValue<1>(),
                                GetParamValue<2>(), GetParamValue<3>(), GetParamValue<4>(), GetParamValue<5>(),
                                GetParamValue<6>());
}

// =============================================================================
// F16 correctness
//
// The main OpWarpAffine suite has no format axis (it is fixed to RGBA8), so F16 gets a dedicated
// suite. Inputs are half-quantized so kernel and reference consume identical values; the gold is
// the in-file FP32 reference run on the widened input, per the F16 tolerance policy in
// HalfTestUtils.hpp. The transforms use dyadic (power-of-two-fraction) coefficients so the FP32
// source coordinates are exact on host and device and tap selection cannot diverge, and the
// inputs sit in [16, 38] so the relative half-ULP bound stays well above residual FP32 ordering
// noise. CUBIC rows keep integer or dyadic shifts (fractional-cubic device coverage across
// arbitrary transforms is the planar-parity CUBIC row above).
// =============================================================================

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffineF16, test::ValueList<int, int, int, int, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int, bool, nvcv::ImageFormat>
{
    {24, 18, 24, 18,     1, 0, 0.5f,     0, 1, 0.25f,  NVCV_INTERP_LINEAR,   NVCV_BORDER_CONSTANT, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true, nvcv::FMT_RGBAf16}, // C4, fractional translation
    {24, 18, 30, 22, 0.5f, 0.25f, 1.5f, 0.25f, 0.75f, 2,  NVCV_INTERP_LINEAR,  NVCV_BORDER_REPLICATE, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true,  nvcv::FMT_RGBf16}, // C3, general dyadic, upscale
    {25, 17, 20, 15, 0.5f, 0.25f, 1.5f, 0.25f, 0.75f, 2,  NVCV_INTERP_LINEAR,       NVCV_BORDER_WRAP, 32.5f, 100.7f, 96.125f, 200.3f, 3,  true,     nvcv::FMT_F16}, // C1 scalar kernel
    {26, 19, 22, 16, 0.5f, 0.25f, 1.5f, 0.25f, 0.75f, 2,  NVCV_INTERP_LINEAR,    NVCV_BORDER_REFLECT, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true, nvcv::FMT_RGBAf16}, // C4, reflect border
    {24, 18, 24, 18,     2, 2, 1,           3, 1, 2,     NVCV_INTERP_NEAREST,   NVCV_BORDER_CONSTANT, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true,  nvcv::FMT_RGBf16}, // C3, gather + constant border
    {24, 18, 20, 16,     1, 0, 0.5f,     0, 1, 0.25f, NVCV_INTERP_NEAREST, NVCV_BORDER_REFLECT101, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true, nvcv::FMT_RGBAf16}, // C4, fractional gather
    {24, 18, 24, 18,     2, 2, 0,           3, 1, 0,       NVCV_INTERP_CUBIC,  NVCV_BORDER_REPLICATE, 32.5f, 100.7f, 96.125f, 200.3f, 2, false,  nvcv::FMT_RGBf16}, // C3, forward map (exactly invertible)
    {24, 18, 24, 18,     1, 0, 1,           0, 1, 2,       NVCV_INTERP_CUBIC,   NVCV_BORDER_CONSTANT, 32.5f, 100.7f, 96.125f, 200.3f, 2,  true, nvcv::FMT_RGBAf16}, // C4, integer-shift cubic
    {133, 9, 135, 11,   1, 0, -0.5f,    0, 1, -0.25f, NVCV_INTERP_LINEAR,   NVCV_BORDER_CONSTANT, 0.f, 0.f, 0.f, 0.f, 2, true, nvcv::FMT_F16},
    {31, 9, 33, 11,     2, 2, 0,          3, 1, 0,       NVCV_INTERP_LINEAR,   NVCV_BORDER_CONSTANT, 13.f, 57.f, 101.f, 211.f, 1, false, nvcv::FMT_RGBf16},
});

// clang-format on

TEST_P(OpWarpAffineF16, tensor_matches_fp32_gold)
{
    const int srcWidth  = GetParamValue<0>();
    const int srcHeight = GetParamValue<1>();
    const int dstWidth  = GetParamValue<2>();
    const int dstHeight = GetParamValue<3>();

    const NVCVAffineTransform xform = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
                                       GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>()};

    const NVCVInterpolationType interpolation = GetParamValue<10>();
    const NVCVBorderType        borderMode    = GetParamValue<11>();
    const float4                borderValue   = test::QuantizeToHalf(
                         float4{GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()});
    const int               numberOfImages = GetParamValue<16>();
    const bool              inverseMap     = GetParamValue<17>();
    const nvcv::ImageFormat fmt            = GetParamValue<18>();

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    const int srcRowElems = srcWidth * fmt.numChannels();
    const int srcRowBytes = srcWidth * fmt.planePixelStrideBytes(0);
    const int dstRowElems = dstWidth * fmt.numChannels();
    const int dstRowBytes = dstWidth * fmt.planePixelStrideBytes(0);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<float>> srcFloat(numberOfImages);
    for (int i = 0; i < numberOfImages; ++i)
    {
        srcFloat[i] = test::MakeHalfQuantizedPattern(srcHeight * srcRowElems, i, 16.f, 0.25f,
                                                     89); // quarter steps over [16, 38]

        std::vector<uint8_t> srcHalf = test::FloatToHalfBytes(srcFloat[i]);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcHalf.data(),
                                            srcRowBytes, srcRowBytes, srcHeight, cudaMemcpyHostToDevice));
    }

    cvcuda::WarpAffine warpAffineOp(0);
    EXPECT_NO_THROW(warpAffineOp(stream, imgSrc, imgDst, xform, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testHalf(dstHeight * dstRowBytes);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testHalf.data(), dstRowBytes, dstAccess->sampleData(i),
                                            dstAccess->rowStride(), dstRowBytes, dstHeight, cudaMemcpyDeviceToHost));

        std::vector<float> goldFloat(dstHeight * dstRowElems);
        WarpAffineGold<float>(goldFloat, dstRowElems, {dstWidth, dstHeight}, srcFloat[i], srcRowElems,
                              {srcWidth, srcHeight}, fmt, xform, flags, borderMode, borderValue);

        test::ExpectF16InterpOutput(goldFloat, testHalf, interpolation);
    }
}

TEST_P(OpWarpAffineF16, varshape_matches_fp32_gold)
{
    const int srcWidthBase  = GetParamValue<0>();
    const int srcHeightBase = GetParamValue<1>();
    const int dstWidthBase  = GetParamValue<2>();
    const int dstHeightBase = GetParamValue<3>();

    const std::array<float, 6> xform = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
                                        GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>()};

    const NVCVInterpolationType interpolation = GetParamValue<10>();
    const NVCVBorderType        borderMode    = GetParamValue<11>();
    const float4                borderValue   = test::QuantizeToHalf(
                         float4{GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()});
    const int               numberOfImages = GetParamValue<16>();
    const bool              inverseMap     = GetParamValue<17>();
    const nvcv::ImageFormat fmt            = GetParamValue<18>();

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Deterministically vary the per-image size so the var-shape path is exercised for real.
    // REFLECT/REFLECT101 keep uniform sizes, matching the caveat in the main var-shape suite
    // (legacy reflect borders misbehave with mixed sizes).
    const bool varySizes = !(borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_REFLECT101);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numberOfImages, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    {
        auto data = transMatrixTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(nullptr, data);
        auto acc = nvcv::TensorDataAccessStrided::Create(*data);
        ASSERT_TRUE(acc);
        for (int i = 0; i < numberOfImages; ++i)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(acc->sampleData(i), acc->sampleStride(), xform.data(),
                                                sizeof(float) * 6, sizeof(float) * 6, 1, cudaMemcpyHostToDevice));
        }
    }

    std::vector<nvcv::Image>        imgSrc;
    std::vector<nvcv::Image>        imgDst;
    std::vector<std::vector<float>> srcFloat(numberOfImages);

    for (int i = 0; i < numberOfImages; ++i)
    {
        const int srcWidth  = srcWidthBase + (varySizes ? 3 * i : 0);
        const int srcHeight = srcHeightBase + (varySizes ? 2 * i : 0);
        const int dstWidth  = dstWidthBase + (varySizes ? 2 * i : 0);
        const int dstHeight = dstHeightBase + (varySizes ? 3 * i : 0);

        imgSrc.emplace_back(nvcv::Size2D{srcWidth, srcHeight}, fmt);
        imgDst.emplace_back(nvcv::Size2D{dstWidth, dstHeight}, fmt);

        const int srcRowElems = srcWidth * fmt.numChannels();
        const int srcRowBytes = srcWidth * fmt.planePixelStrideBytes(0);

        srcFloat[i] = test::MakeHalfQuantizedPattern(srcHeight * srcRowElems, i, 16.f, 0.25f,
                                                     89); // quarter steps over [16, 38]

        std::vector<uint8_t> srcHalf = test::FloatToHalfBytes(srcFloat[i]);

        const auto data = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(data, nvcv::NullOpt);
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(data->plane(0).basePtr, data->plane(0).rowStride, srcHalf.data(),
                                            srcRowBytes, srcRowBytes, srcHeight, cudaMemcpyHostToDevice));
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    cvcuda::WarpAffine warpAffineOp(numberOfImages);
    EXPECT_NO_THROW(warpAffineOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(srcData, nvcv::NullOpt);
        ASSERT_NE(dstData, nvcv::NullOpt);

        const int srcWidth    = srcData->plane(0).width;
        const int srcHeight   = srcData->plane(0).height;
        const int dstWidth    = dstData->plane(0).width;
        const int dstHeight   = dstData->plane(0).height;
        const int srcRowElems = srcWidth * fmt.numChannels();
        const int dstRowElems = dstWidth * fmt.numChannels();
        const int dstRowBytes = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testHalf(dstHeight * dstRowBytes);
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testHalf.data(), dstRowBytes, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowBytes, dstHeight, cudaMemcpyDeviceToHost));

        NVCVAffineTransform xformForGold;
        std::ranges::copy(xform, xformForGold);

        std::vector<float> goldFloat(dstHeight * dstRowElems);
        WarpAffineGold<float>(goldFloat, dstRowElems, {dstWidth, dstHeight}, srcFloat[i], srcRowElems,
                              {srcWidth, srcHeight}, fmt, xformForGold, flags, borderMode, borderValue);

        test::ExpectF16InterpOutput(goldFloat, testHalf, interpolation);
    }
}

TEST(OpWarpAffinePlanar, single_channel_tensor_matches_interleaved)
{
    const std::array<float, 6> xform       = PlanarAffine();
    const float4               borderValue = {13.f, 57.f, 101.f, 211.f};

    test::planar::RunTensorSingleChannelLayoutParity(
        57, 43, 71, 59, 2, nvcv::TYPE_U8,
        [xform, borderValue](cudaStream_t stream, const nvcv::Tensor &src, const nvcv::Tensor &dst)
        {
            cvcuda::WarpAffine op(2);
            EXPECT_NO_THROW(op(stream, src, dst, xform.data(), NVCV_INTERP_CUBIC, NVCV_BORDER_CONSTANT, borderValue));
        });
}

TEST(OpWarpAffine, varshape_inverse_map_copies_transform)
{
    constexpr int width    = 4;
    constexpr int height   = 3;
    constexpr int channels = 3;
    constexpr int rowBytes = width * channels;

    const std::array<uint8_t, height *rowBytes> source
        = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    const std::array<float, 6> xform = {1.f, 0.f, 1.f, 0.f, 1.f, 0.f};
    const int                  flags = NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP;
    const float4               borderValue{200.f, 201.f, 202.f, 0.f};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);

    const auto srcData = imgSrc.front().exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(srcData, nvcv::NullOpt);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, source.data(), rowBytes,
                                        rowBytes, height, cudaMemcpyHostToDevice));

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor transMatrix(nvcv::TensorShape({1, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    const auto   transMatrixData = transMatrix.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, transMatrixData);
    const auto transMatrixAccess = nvcv::TensorDataAccessStrided::Create(*transMatrixData);
    ASSERT_TRUE(transMatrixAccess);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(transMatrixAccess->sampleData(0), transMatrixAccess->sampleStride(), xform.data(),
                           sizeof(float) * xform.size(), sizeof(float) * xform.size(), 1, cudaMemcpyHostToDevice));

    cvcuda::WarpAffine op(1);
    EXPECT_NO_THROW(op(stream, batchSrc, batchDst, transMatrix, flags, NVCV_BORDER_CONSTANT, borderValue));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto dstData = imgDst.front().exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(dstData, nvcv::NullOpt);

    std::array<uint8_t, height * rowBytes> actual{};
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                                        rowBytes, height, cudaMemcpyDeviceToHost));

    std::vector<uint8_t>       expected(source.size());
    const std::vector<uint8_t> sourceVec(source.begin(), source.end());
    WarpAffineGold<uint8_t>(expected, rowBytes, {width, height}, sourceVec, rowBytes, {width, height}, nvcv::FMT_RGB8,
                            xform.data(), flags, NVCV_BORDER_CONSTANT, borderValue);

    EXPECT_EQ(std::vector<uint8_t>(actual.begin(), actual.end()), expected);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffine, varshape_forward_map_inverts_transform)
{
    constexpr int width    = 4;
    constexpr int height   = 3;
    constexpr int channels = 3;
    constexpr int rowBytes = width * channels;

    const std::array<uint8_t, height *rowBytes> source
        = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    const std::array<float, 6> xform = {1.f, 0.f, 1.f, 0.f, 1.f, 0.f};
    const int                  flags = NVCV_INTERP_NEAREST;
    const float4               borderValue{200.f, 201.f, 202.f, 0.f};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    std::vector<nvcv::Image> imgSrc;
    std::vector<nvcv::Image> imgDst;
    imgSrc.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);
    imgDst.emplace_back(nvcv::Size2D{width, height}, nvcv::FMT_RGB8);

    const auto srcData = imgSrc.front().exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(srcData, nvcv::NullOpt);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, source.data(), rowBytes,
                                        rowBytes, height, cudaMemcpyHostToDevice));

    nvcv::ImageBatchVarShape batchSrc(1);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(1);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    nvcv::Tensor transMatrix(nvcv::TensorShape({1, 6}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    const auto   transMatrixData = transMatrix.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, transMatrixData);
    const auto transMatrixAccess = nvcv::TensorDataAccessStrided::Create(*transMatrixData);
    ASSERT_TRUE(transMatrixAccess);
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy2D(transMatrixAccess->sampleData(0), transMatrixAccess->sampleStride(), xform.data(),
                           sizeof(float) * xform.size(), sizeof(float) * xform.size(), 1, cudaMemcpyHostToDevice));

    cvcuda::WarpAffine op(1);
    EXPECT_NO_THROW(op(stream, batchSrc, batchDst, transMatrix, flags, NVCV_BORDER_CONSTANT, borderValue));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto dstData = imgDst.front().exportData<nvcv::ImageDataStridedCuda>();
    ASSERT_NE(dstData, nvcv::NullOpt);

    std::array<uint8_t, height * rowBytes> actual{};
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                                        rowBytes, height, cudaMemcpyDeviceToHost));

    std::vector<uint8_t>       expected(source.size());
    const std::vector<uint8_t> sourceVec(source.begin(), source.end());
    WarpAffineGold<uint8_t>(expected, rowBytes, {width, height}, sourceVec, rowBytes, {width, height}, nvcv::FMT_RGB8,
                            xform.data(), flags, NVCV_BORDER_CONSTANT, borderValue);

    EXPECT_EQ(std::vector<uint8_t>(actual.begin(), actual.end()), expected);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpAffine, tensor_singular_forward_map_uses_zero_inverse)
{
    constexpr int width    = 3;
    constexpr int height   = 2;
    constexpr int rowBytes = width * 3;

    const std::array<uint8_t, height *rowBytes> source
        = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73};
    const std::array<float, 6> xform = {1.f, 2.f, 3.f, 2.f, 4.f, 6.f};
    const int                  flags = NVCV_INTERP_NEAREST;
    const float4               borderValue{200.f, 201.f, 202.f, 203.f};

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const auto   fmt = nvcv::FMT_RGB8;
    nvcv::Tensor imgSrc(1, {width, height}, fmt);
    nvcv::Tensor imgDst(1, {width, height}, fmt);

    const auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, srcData);
    const auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcAccess->sampleData(0), srcAccess->rowStride(), source.data(), rowBytes,
                                        rowBytes, height, cudaMemcpyHostToDevice));

    cvcuda::WarpAffine op(0);
    EXPECT_NO_THROW(op(stream, imgSrc, imgDst, xform.data(), flags, NVCV_BORDER_CONSTANT, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);
    const auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    std::vector<uint8_t> actual(height * rowBytes);
    ASSERT_EQ(cudaSuccess, cudaMemcpy2D(actual.data(), rowBytes, dstAccess->sampleData(0), dstAccess->rowStride(),
                                        rowBytes, height, cudaMemcpyDeviceToHost));

    std::vector<uint8_t>       expected(actual.size());
    const std::vector<uint8_t> sourceVec(source.begin(), source.end());
    WarpAffineGold<uint8_t>(expected, rowBytes, {width, height}, sourceVec, rowBytes, {width, height}, fmt,
                            xform.data(), flags, NVCV_BORDER_CONSTANT, borderValue);

    EXPECT_EQ(actual, expected);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffine_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    // input format, output format,
    {nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},  // interleaved in, planar out: layout mismatch
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},  // planar in, interleaved out: layout mismatch
    {nvcv::FMT_F64, nvcv::FMT_F64}  // unsupported data type (64-bit float)
});

NVCV_TEST_SUITE_P(OpWarpAffineVarshape_Negative, test::ValueList<int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    // maxBatchSize, numImages, input format, output format
    {5, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
    {5, 5, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {5, 5, nvcv::FMT_F64, nvcv::FMT_F64},  // unsupported data type (64-bit float)
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

// Half-pixel shift makes column 0 exactly half-covered, placing expected values in closed form.
TEST(OpWarpAffine, constant_coverage_blends_by_coverage)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int     kSize  = 4;
    constexpr uint8_t kValue = 100;
    constexpr float   kFill  = 200.f;

    NVCVAffineTransform xform       = {1, 0, -0.5f, 0, 1, 0};
    const float4        borderValue = {kFill, kFill, kFill, kFill};
    const int           flags       = NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP;

    nvcv::Tensor imgSrc(1, {kSize, kSize}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {kSize, kSize}, nvcv::FMT_U8);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    std::vector<uint8_t> srcVec(kSize * kSize, kValue);
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, srcVec));

    // Column 0 samples x = -0.5: the left tap is outside and the right tap is inside, so the
    // coverage is 0.5 and the zero-bordered interpolation is 0.5 * kValue.
    //   coverage : 0.5 * kValue * 0.5 + kFill * 0.5  = 125
    //   constant : 0.5 * kFill  + 0.5 * kValue       = 150
    const uint8_t kCoverageExpected = 125;
    const uint8_t kConstantExpected = 150;

    const std::array cases = {
        std::pair{NVCV_BORDER_CONSTANT_COVERAGE, kCoverageExpected},
        std::pair{         NVCV_BORDER_CONSTANT, kConstantExpected}
    };
    for (auto [border, expectedEdge] : cases)
    {
        cvcuda::WarpAffine op(0);
        op(stream, imgSrc, imgDst, xform, flags, border, borderValue);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

        auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_TRUE(dstData);
        std::vector<uint8_t> dstVec;
        nvcv::util::GetImageVectorFromTensor(*dstData, 0, dstVec);

        for (int y = 0; y < kSize; ++y)
        {
            EXPECT_EQ(expectedEdge, dstVec[y * kSize]) << "border " << border << " row " << y;
            //
            for (int x = 1; x < kSize; ++x)
            {
                EXPECT_EQ(kValue, dstVec[y * kSize + x]) << "border " << border << " at " << x << "," << y;
            }
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Same geometry as the previous test; fill 200.5 verifies fractional fills survive to the blend.
//
//
TEST(OpWarpAffine, constant_coverage_keeps_a_fractional_fill)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int     kSize  = 4;
    constexpr uint8_t kValue = 100;
    constexpr float   kFill  = 200.5f;

    NVCVAffineTransform xform       = {1, 0, -0.5f, 0, 1, 0};
    const float4        borderValue = {kFill, kFill, kFill, kFill};

    nvcv::Tensor imgSrc(1, {kSize, kSize}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {kSize, kSize}, nvcv::FMT_U8);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(srcData);
    std::vector<uint8_t> srcVec(kSize * kSize, kValue);
    ASSERT_NO_THROW(nvcv::util::SetImageTensorFromVector<uint8_t>(*srcData, srcVec));

    // LINEAR, column 0: coverage 0.5, so 0.5 * kValue * 0.5 + kFill * 0.5 = 125.25 -> 125. This
    // pins the blend formula; the fill's own rounding is pinned by the second half of the test,
    // where a wholly uncovered pixel takes the fill alone.
    const uint8_t kLinearExpected = 125;

    cvcuda::WarpAffine op(0);
    ASSERT_NO_THROW(op(stream, imgSrc, imgDst, xform, NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP,
                       NVCV_BORDER_CONSTANT_COVERAGE, borderValue));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_TRUE(dstData);
    std::vector<uint8_t> dstVec;
    ASSERT_NO_THROW(nvcv::util::GetImageVectorFromTensor(*dstData, 0, dstVec));
    for (int y = 0; y < kSize; ++y)
    {
        EXPECT_EQ(kLinearExpected, dstVec[y * kSize]) << "row " << y;
    }

    // A wholly uncovered pixel takes the fill alone, so it shows the fill's own rounding: 200.7
    // rounds to 201, where truncating it to 200 on the way in would give 200. Shift the source
    // fully out of frame so every output pixel is uncovered.
    constexpr float     kOddFill     = 200.7f;
    const float4        oddBorder    = {kOddFill, kOddFill, kOddFill, kOddFill};
    NVCVAffineTransform farXform     = {1, 0, -100.f, 0, 1, 0};
    const uint8_t       kFillRounded = 201;

    const std::array flagsList
        = {NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP, NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP};
    for (int flags : flagsList)
    {
        SCOPED_TRACE(flags);
        op(stream, imgSrc, imgDst, farXform, flags, NVCV_BORDER_CONSTANT_COVERAGE, oddBorder);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        nvcv::util::GetImageVectorFromTensor(*dstData, 0, dstVec);
        for (size_t i = 0; i < dstVec.size(); ++i)
        {
            EXPECT_EQ(kFillRounded, dstVec[i]) << "at " << i;
        }
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

//
//
//
TEST(OpWarpAffine_Negative, varshape_rejects_constant_coverage)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::ImageBatchVarShape batchSrc(1);
    nvcv::ImageBatchVarShape batchDst(1);
    nvcv::Image              src({16, 16}, nvcv::FMT_RGB8);
    nvcv::Image              dst({16, 16}, nvcv::FMT_RGB8);
    batchSrc.pushBack(src);
    batchDst.pushBack(dst);

    nvcv::Tensor xform(
        {
            {1, 6},
            "NW"
    },
        nvcv::TYPE_F32);

    cvcuda::WarpAffine op(1);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                               [&]
                                               {
                                                   op(stream, batchSrc, batchDst, xform,
                                                      NVCV_INTERP_LINEAR | NVCV_WARP_INVERSE_MAP,
                                                      NVCV_BORDER_CONSTANT_COVERAGE, float4{0, 0, 0, 0});
                                               }));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
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

    const int invalidBorder = 255;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              cvcudaWarpAffineSubmit(op.handle(), stream, imgSrc.handle(), imgDst.handle(), xform, flags,
                                     static_cast<NVCVBorderType>(invalidBorder), borderValue));

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

    constexpr NVCVBorderType border = NVCV_BORDER_CONSTANT_COVERAGE;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &border, &borderValue]
                                { op(stream, batchSrc, batchDst, transMatrixTensor, flags, border, borderValue); }));

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
