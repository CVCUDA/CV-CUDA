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
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <cvcuda/cuda_tools/math/LinAlg.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <array>
#include <cmath>
#include <map>
#include <random>
#include <string_view>

namespace cuda = nvcv::cuda;
namespace test = nvcv::test;

// #define DBG_WARP_PERSPECTIVE 1

static void printVec([[maybe_unused]] const std::vector<uint8_t> &vec, [[maybe_unused]] int height,
                     [[maybe_unused]] int rowStride, [[maybe_unused]] int bytesPerPixel,
                     [[maybe_unused]] std::string_view name)
{
#if DBG_WARP_PERSPECTIVE
    for (int i = 0; i < bytesPerPixel; i++)
    {
        std::cout << "\nPrint " << name << " for channel: " << i << std::endl;

        for (int k = 0; k < height; k++)
        {
            for (int j = 0; j < static_cast<int>(rowStride / bytesPerPixel); j++)
            {
                printf("%4d, ", static_cast<int>(vec[k * rowStride + j * bytesPerPixel + i]));
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
#endif
}

struct PerspectiveSource
{
    const uint8_t *ptr;
    nvcv::Size2D   size;
    int            rowStride;
    int            elementsPerPixel;
    NVCVBorderType borderMode;
    float4         borderVal;
};

struct WarpPerspectiveGoldParams
{
    int                             dstRowStride;
    nvcv::Size2D                    dstSize;
    int                             srcRowStride;
    nvcv::Size2D                    srcSize;
    nvcv::ImageFormat               fmt;
    const NVCVPerspectiveTransform &transMatrix;
    int                             flags;
    NVCVBorderType                  borderMode;
    float4                          borderVal;
};

static uint8_t getPixelForPerspectiveTransform(const PerspectiveSource &src, const int y, const int x, int k)
{
    const int width  = src.size.w;
    const int height = src.size.h;
    int2      coord  = {x, y};
    int2      size   = {width, height};
    if (src.borderMode == NVCV_BORDER_CONSTANT)
    {
        return (x >= 0 && x < width && y >= 0 && y < height) ? src.ptr[y * src.rowStride + x * src.elementsPerPixel + k]
                                                             : static_cast<uint8_t>(cuda::GetElement(src.borderVal, k));
    }
    else if (src.borderMode == NVCV_BORDER_REPLICATE)
    {
        test::ReplicateBorderIndex(coord, size);
        return src.ptr[coord.y * src.rowStride + coord.x * src.elementsPerPixel + k];
    }
    else if (src.borderMode == NVCV_BORDER_REFLECT)
    {
        test::ReflectBorderIndex(coord, size);
        return src.ptr[coord.y * src.rowStride + coord.x * src.elementsPerPixel + k];
    }
    else if (src.borderMode == NVCV_BORDER_REFLECT101)
    {
        test::Reflect101BorderIndex(coord, size);
        return src.ptr[coord.y * src.rowStride + coord.x * src.elementsPerPixel + k];
    }
    else if (src.borderMode == NVCV_BORDER_WRAP)
    {
        test::WrapBorderIndex(coord, size);
        return src.ptr[coord.y * src.rowStride + coord.x * src.elementsPerPixel + k];
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

inline uint8_t clampU8(float value)
{
    const float rounded = std::rint(value);
    if (rounded < 0.0f)
    {
        return static_cast<uint8_t>(0);
    }
    if (rounded > 255.0f)
    {
        return static_cast<uint8_t>(255);
    }
    return static_cast<uint8_t>(rounded);
}

static void StoreLinearPixel(uint8_t *dstPtr, int dstBase, const PerspectiveSource &src, float src_x, float src_y)
{
    const auto x1 = static_cast<int>(std::floor(src_x));
    const auto y1 = static_cast<int>(std::floor(src_y));

    const int x2 = x1 + 1;
    const int y2 = y1 + 1;

    for (int k = 0; k < src.elementsPerPixel; k++)
    {
        float out = 0;

        uint8_t srcReg = getPixelForPerspectiveTransform(src, y1, x1, k);
        out += static_cast<float>(srcReg) * ((static_cast<float>(x2) - src_x) * (static_cast<float>(y2) - src_y));

        srcReg = getPixelForPerspectiveTransform(src, y1, x2, k);
        out = out + static_cast<float>(srcReg) * ((src_x - static_cast<float>(x1)) * (static_cast<float>(y2) - src_y));

        srcReg = getPixelForPerspectiveTransform(src, y2, x1, k);
        out = out + static_cast<float>(srcReg) * ((static_cast<float>(x2) - src_x) * (src_y - static_cast<float>(y1)));

        srcReg = getPixelForPerspectiveTransform(src, y2, x2, k);
        out = out + static_cast<float>(srcReg) * ((src_x - static_cast<float>(x1)) * (src_y - static_cast<float>(y1)));

        dstPtr[dstBase + k] = clampU8(out);
    }
}

static void StoreNearestPixel(uint8_t *dstPtr, int dstBase, const PerspectiveSource &src, float src_x, float src_y)
{
    const auto x1 = static_cast<int>(std::floor(src_x + .5f));
    const auto y1 = static_cast<int>(std::floor(src_y + .5f));

    for (int k = 0; k < src.elementsPerPixel; k++)
    {
        dstPtr[dstBase + k] = getPixelForPerspectiveTransform(src, y1, x1, k);
    }
}

static void StoreCubicPixel(uint8_t *dstPtr, int dstBase, const PerspectiveSource &src, float src_x, float src_y)
{
    const auto xmin = static_cast<int>(std::ceil(src_x - 2.0f));
    const auto xmax = static_cast<int>(std::floor(src_x + 2.0f));

    const auto ymin = static_cast<int>(std::ceil(src_y - 2.0f));
    const auto ymax = static_cast<int>(std::floor(src_y + 2.0f));

    for (int k = 0; k < src.elementsPerPixel; k++)
    {
        float sum  = 0;
        float wsum = 0;

        for (int cy = ymin; cy <= ymax; cy += 1)
        {
            for (int cx = xmin; cx <= xmax; cx += 1)
            {
                const float w = calcBicubicCoeff(src_x - static_cast<float>(cx))
                              * calcBicubicCoeff(src_y - static_cast<float>(cy));
                uint8_t srcReg = getPixelForPerspectiveTransform(src, cy, cx, k);
                sum += w * static_cast<float>(srcReg);
                wsum += w;
            }
        }

        dstPtr[dstBase + k] = clampU8(wsum == 0.0f ? 0.0f : sum / wsum);
    }
}

static void WarpPerspectiveGold(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc,
                                const WarpPerspectiveGoldParams &params)
{
    assert(params.fmt.numPlanes() == 1);

    PerspectiveSource src{hSrc.data(),       params.srcSize,  params.srcRowStride, params.fmt.numChannels(),
                          params.borderMode, params.borderVal};

    uint8_t  *dstPtr           = hDst.data();
    const int elementsPerPixel = src.elementsPerPixel;
    const int interpolation    = params.flags & NVCV_INTERP_MAX;

    NVCVPerspectiveTransform finalTransformMatrix;

    if (!(params.flags & NVCV_WARP_INVERSE_MAP))
    {
        cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

        tempMatrixForInverse[0][0] = params.transMatrix[0];
        tempMatrixForInverse[0][1] = params.transMatrix[1];
        tempMatrixForInverse[0][2] = params.transMatrix[2];
        tempMatrixForInverse[1][0] = params.transMatrix[3];
        tempMatrixForInverse[1][1] = params.transMatrix[4];
        tempMatrixForInverse[1][2] = params.transMatrix[5];
        tempMatrixForInverse[2][0] = params.transMatrix[6];
        tempMatrixForInverse[2][1] = params.transMatrix[7];
        tempMatrixForInverse[2][2] = params.transMatrix[8];

        cuda::math::inv_inplace(tempMatrixForInverse);

        finalTransformMatrix[0] = tempMatrixForInverse[0][0];
        finalTransformMatrix[1] = tempMatrixForInverse[0][1];
        finalTransformMatrix[2] = tempMatrixForInverse[0][2];
        finalTransformMatrix[3] = tempMatrixForInverse[1][0];
        finalTransformMatrix[4] = tempMatrixForInverse[1][1];
        finalTransformMatrix[5] = tempMatrixForInverse[1][2];
        finalTransformMatrix[6] = tempMatrixForInverse[2][0];
        finalTransformMatrix[7] = tempMatrixForInverse[2][1];
        finalTransformMatrix[8] = tempMatrixForInverse[2][2];
    }
    else
    {
        for (int i = 0; i < 9; i++)
        {
            finalTransformMatrix[i] = params.transMatrix[i];
        }
    }

    for (int dst_y = 0; dst_y < params.dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < params.dstSize.w; dst_x++)
        {
            const auto dstX = static_cast<float>(dst_x);
            const auto dstY = static_cast<float>(dst_y);
            float      coeff
                = 1.0f / (dstX * finalTransformMatrix[6] + dstY * finalTransformMatrix[7] + finalTransformMatrix[8]);
            float src_x
                = coeff * (dstX * finalTransformMatrix[0] + dstY * finalTransformMatrix[1] + finalTransformMatrix[2]);
            float src_y
                = coeff * (dstX * finalTransformMatrix[3] + dstY * finalTransformMatrix[4] + finalTransformMatrix[5]);

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                StoreLinearPixel(dstPtr, dst_y * params.dstRowStride + dst_x * elementsPerPixel, src, src_x, src_y);
            }
            else if (interpolation == NVCV_INTERP_NEAREST)
            {
                StoreNearestPixel(dstPtr, dst_y * params.dstRowStride + dst_x * elementsPerPixel, src, src_x, src_y);
            }
            else if (interpolation == NVCV_INTERP_CUBIC)
            {
                StoreCubicPixel(dstPtr, dst_y * params.dstRowStride + dst_x * elementsPerPixel, src, src_x, src_y);
            }
            else
            {
                return;
            }
        }
    }
}

// Non-trivial projection matrices use the four input image corners and map them
// to scaled output-image corner positions.

const std::map<std::vector<int>, std::vector<std::vector<float>>> mapOfTransformationMatrix = {
    {{5, 4, 5, 4},
     {{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.5f, 2.0f, 1.0f, 0.75f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.50f, 0.47f, 0.00f, -0.13f, 1.14f, 0.52f, -0.14f, 0.14f, 1.00f}}},
    {{5, 4, 6, 8},
     {{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.5f, 2.0f, 1.0f, 0.75f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.60f, 0.56f, 0.00f, -0.26f, 2.28f, 1.04f, -0.14f, 0.14f, 1.00f}}},
    {{7, 8, 4, 5},
     {{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.5f, 2.0f, 1.0f, 0.75f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f},
     {0.27f, 0.16f, 0.00f, -0.11f, 0.61f, 0.65f, -0.09f, 0.06f, 1.00f}}}
};

// clang-format off
NVCV_TEST_SUITE_P(OpWarpPerspective, test::ValueList<int, int, int, int, float, float, float, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int, bool>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,     transformation_matrix,                                         interpolation,              borderType,  borderValue, batchSize, inverse
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    //------------------ LINEAR INTERP ------------------//
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    //------------------ CUBIC INTERP ------------------//
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, true},
});

// clang-format on

TEST_P(OpWarpPerspective, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVPerspectiveTransform transMatrix;
    transMatrix[0] = GetParamValue<4>();
    transMatrix[1] = GetParamValue<5>();
    transMatrix[2] = GetParamValue<6>();
    transMatrix[3] = GetParamValue<7>();
    transMatrix[4] = GetParamValue<8>();
    transMatrix[5] = GetParamValue<9>();
    transMatrix[6] = GetParamValue<10>();
    transMatrix[7] = GetParamValue<11>();
    transMatrix[8] = GetParamValue<12>();

    NVCVInterpolationType interpolation = GetParamValue<13>();

    NVCVBorderType borderMode = GetParamValue<14>();

    const float4 borderValue = {GetParamValue<15>(), GetParamValue<16>(), GetParamValue<17>(), GetParamValue<18>()};

    int numberOfImages = GetParamValue<19>();

    bool inverseMap = GetParamValue<20>();

    const nvcv::ImageFormat fmt           = nvcv::FMT_RGBA8;
    const int               bytesPerPixel = 4;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    auto srcData = imgSrc.exportData<nvcv::TensorDataStridedCuda>();

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowStride = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecRowStride);
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowStride(), srcVec[i].data(), srcVecRowStride,
                               srcVecRowStride, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    cvcuda::WarpPerspective warpPerspectiveOp(0);
    EXPECT_NO_THROW(warpPerspectiveOp(stream, imgSrc, imgDst, transMatrix, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    auto dstData = imgDst.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowStride = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowStride, dstAccess->sampleData(i), dstAccess->rowStride(),
                               dstVecRowStride, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        // Generate gold result
        const WarpPerspectiveGoldParams goldParams{
            dstVecRowStride,
            {dstWidth, dstHeight},
            srcVecRowStride,
            {srcWidth, srcHeight},
            fmt,
            transMatrix,
            flags,
            borderMode,
            borderValue
        };
        WarpPerspectiveGold(goldVec, srcVec[i], goldParams);

        printVec(srcVec[i], srcHeight, srcVecRowStride, bytesPerPixel, "src vec");
        printVec(goldVec, dstHeight, dstVecRowStride, bytesPerPixel, "golden output");
        printVec(testVec, dstHeight, dstVecRowStride, bytesPerPixel, "warped output");

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpWarpPerspective, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    std::vector<float> transMatrix;
    transMatrix.resize(9);
    transMatrix[0] = GetParamValue<4>();
    transMatrix[1] = GetParamValue<5>();
    transMatrix[2] = GetParamValue<6>();
    transMatrix[3] = GetParamValue<7>();
    transMatrix[4] = GetParamValue<8>();
    transMatrix[5] = GetParamValue<9>();
    transMatrix[6] = GetParamValue<10>();
    transMatrix[7] = GetParamValue<11>();
    transMatrix[8] = GetParamValue<12>();

    NVCVInterpolationType interpolation = GetParamValue<13>();

    NVCVBorderType borderMode = GetParamValue<14>();

    const float4 borderValue = {GetParamValue<15>(), GetParamValue<16>(), GetParamValue<17>(), GetParamValue<18>()};

    int numberOfImages = GetParamValue<19>();

    bool inverseMap = GetParamValue<20>();

    const nvcv::ImageFormat fmt           = nvcv::FMT_RGBA8;
    const int               bytesPerPixel = 4;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numberOfImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    auto         transMatrixTensorData = transMatrixTensor.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, transMatrixTensorData);

    auto transMatrixTensorDataAccess = nvcv::TensorDataAccessStrided::Create(*transMatrixTensorData);
    ASSERT_TRUE(transMatrixTensorDataAccess);

    // Create input and output
    std::default_random_engine    randEng;
    std::uniform_int_distribution rndInputDimsIndex(0, static_cast<int>(mapOfTransformationMatrix.size()) - 1);
    std::uniform_int_distribution rndTransformationMatrixIndex(0, 4);

    std::vector<nvcv::Image>        imgSrc;
    std::vector<nvcv::Image>        imgDst;
    std::vector<std::vector<float>> transMatrixHostVec;
    transMatrixHostVec.resize(numberOfImages);

    // List the keys from the map for easy access
    std::vector<std::vector<int>> keysOfMapOfTransformationMatrix;
    for (const auto &[key, transformationMatrices] : mapOfTransformationMatrix)
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

        const std::vector<int>   &key = keysOfMapOfTransformationMatrix[dictInputIndex];
        const std::vector<float> &chosenTransformationMatrix
            = mapOfTransformationMatrix.at(key)[dictTransformationIndex];
        if (i > 0)
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
                                    sizeof(float) * 9, sizeof(float) * 9, 1, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowStride(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);

        int currentSrcWidth  = srcData->plane(0).width;
        int currentSrcHeight = srcData->plane(0).height;

        int srcRowStride = currentSrcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(currentSrcHeight * srcRowStride);
        std::ranges::generate(srcVec[i], [&rand, &randEng]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).basePtr, srcData->plane(0).rowStride, srcVec[i].data(), srcRowStride,
                               srcRowStride, // vec has no padding
                               currentSrcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    cvcuda::WarpPerspective warpPerspectiveOp(numberOfImages);
    EXPECT_NO_THROW(warpPerspectiveOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(srcData->numPlanes() == 1);
        int currentSrcWidth  = srcData->plane(0).width;
        int currentSrcHeight = srcData->plane(0).height;

        const auto dstData = imgDst[i].exportData<nvcv::ImageDataStridedCuda>();
        assert(dstData->numPlanes() == 1);

        int currentDstWidth  = dstData->plane(0).width;
        int currentDstHeight = dstData->plane(0).height;

        int srcRowStride = currentSrcWidth * fmt.planePixelStrideBytes(0);
        int dstRowStride = currentDstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(currentDstHeight * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr, dstData->plane(0).rowStride,
                               dstRowStride, // vec has no padding
                               currentDstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(currentDstHeight * dstRowStride);
        std::ranges::generate(goldVec, []() { return 0; });

        NVCVPerspectiveTransform transMatrixForGold;
        transMatrixForGold[0] = transMatrixHostVec[i][0];
        transMatrixForGold[1] = transMatrixHostVec[i][1];
        transMatrixForGold[2] = transMatrixHostVec[i][2];
        transMatrixForGold[3] = transMatrixHostVec[i][3];
        transMatrixForGold[4] = transMatrixHostVec[i][4];
        transMatrixForGold[5] = transMatrixHostVec[i][5];
        transMatrixForGold[6] = transMatrixHostVec[i][6];
        transMatrixForGold[7] = transMatrixHostVec[i][7];
        transMatrixForGold[8] = transMatrixHostVec[i][8];

        // Generate gold result
        const WarpPerspectiveGoldParams goldParams{
            dstRowStride, {currentDstWidth, currentDstHeight},
            srcRowStride, {currentSrcWidth, currentSrcHeight},
            fmt,          transMatrixForGold,
            flags,        borderMode,
            borderValue
        };
        WarpPerspectiveGold(goldVec, srcVec[i], goldParams);

        printVec(srcVec[i], currentSrcHeight, srcRowStride, bytesPerPixel, "src vec");
        printVec(goldVec, currentDstHeight, dstRowStride, bytesPerPixel, "golden output");
        printVec(testVec, currentDstHeight, dstRowStride, bytesPerPixel, "warped output");

        EXPECT_EQ(goldVec, testVec);
    }
}

// =============================================================================
// Planar (NCHW/CHW) layout support
//
// WarpPerspective samples every channel at the same transformed coordinate, so
// planar output must match the equivalent interleaved output bit-for-bit after
// re-interleaving. CONSTANT border uses a non-uniform border value to exercise
// per-channel border handling in the planar path.
// =============================================================================

namespace {

inline std::array<float, 9> PlanarPerspective()
{
    return {1.05f, 0.03f, 2.0f, -0.02f, 0.98f, 1.0f, 0.0008f, -0.0004f, 1.0f};
}

void RunPlanarParityTensorCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                               int dstW, int dstH, NVCVInterpolationType interp, NVCVBorderType borderMode,
                               int numImages)
{
    const std::array<float, 9> xform       = PlanarPerspective();
    const float4               borderValue = {13.f, 57.f, 101.f, 211.f};
    const int32_t              flags       = interp;

    test::planar::RunTensorParity(planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
                                  [xform, flags, borderMode, borderValue](cudaStream_t s, const nvcv::Tensor &src,
                                                                          const nvcv::Tensor &dst, nvcv::ImageFormat)
                                  {
                                      cvcuda::WarpPerspective op(0);
                                      EXPECT_NO_THROW(op(s, src, dst, xform.data(), flags, borderMode, borderValue));
                                  });
}

void RunPlanarParityVarShapeCase(nvcv::ImageFormat planarFmt, nvcv::ImageFormat interleavedFmt, int srcW, int srcH,
                                 int dstW, int dstH, NVCVInterpolationType interp, NVCVBorderType borderMode,
                                 int numImages)
{
    const std::array<float, 9> xform       = PlanarPerspective();
    const float4               borderValue = {13.f, 57.f, 101.f, 211.f};
    const int32_t              flags       = interp;

    nvcv::Tensor transMatrix(nvcv::TensorShape({numImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);
    {
        auto data = transMatrix.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(data, nullptr);
        auto acc = nvcv::TensorDataAccessStrided::Create(*data);
        ASSERT_TRUE(acc);
        for (int i = 0; i < numImages; ++i)
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy2D(acc->sampleData(i), acc->sampleStride(), xform.data(),
                                                sizeof(float) * 9, sizeof(float) * 9, 1, cudaMemcpyHostToDevice));
        }
    }

    test::planar::RunVarShapeParity(
        planarFmt, interleavedFmt, srcW, srcH, dstW, dstH, numImages,
        [&transMatrix, flags, borderMode, borderValue, numImages](
            cudaStream_t s, const nvcv::ImageBatchVarShape &src, const nvcv::ImageBatchVarShape &dst, nvcv::ImageFormat)
        {
            cvcuda::WarpPerspective op(numImages);
            EXPECT_NO_THROW(op(s, src, dst, transMatrix, flags, borderMode, borderValue));
        });
}

} // namespace

// Parameters: planarFmt, interleavedFmt, interpolation, borderMode, numImages, srcW, srcH, dstW, dstH
// clang-format off
NVCV_TEST_SUITE_P(OpWarpPerspectivePlanar,
    test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat, NVCVInterpolationType, NVCVBorderType, int, int, int, int, int>{
    {   nvcv::FMT_RGB8p,    nvcv::FMT_RGB8, NVCV_INTERP_NEAREST,  NVCV_BORDER_CONSTANT,  2, 64, 48, 64, 48},
    {   nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2, 64, 48, 96, 72},
    {   nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,   NVCV_INTERP_CUBIC, NVCV_BORDER_REPLICATE,  1, 80, 60, 64, 48},
    {   nvcv::FMT_RGB8p,    nvcv::FMT_RGB8,  NVCV_INTERP_LINEAR,      NVCV_BORDER_WRAP,  1, 64, 48, 64, 48},
    {  nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8, NVCV_INTERP_NEAREST,  NVCV_BORDER_CONSTANT,  2, 50, 40, 60, 50},
    {  nvcv::FMT_RGBA8p,   nvcv::FMT_RGBA8,  NVCV_INTERP_LINEAR, NVCV_BORDER_REPLICATE,  1, 50, 40, 50, 40},
    { nvcv::FMT_RGBf32p,  nvcv::FMT_RGBf32,   NVCV_INTERP_CUBIC,  NVCV_BORDER_CONSTANT,  1, 64, 48, 96, 72},
    {nvcv::FMT_RGBAf32p, nvcv::FMT_RGBAf32,  NVCV_INTERP_LINEAR,  NVCV_BORDER_CONSTANT,  2, 64, 48, 64, 48},
});

// clang-format on

TEST_P(OpWarpPerspectivePlanar, tensor_matches_interleaved)
{
    RunPlanarParityTensorCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<5>(), GetParamValue<6>(),
                              GetParamValue<7>(), GetParamValue<8>(), GetParamValue<2>(), GetParamValue<3>(),
                              GetParamValue<4>());
}

TEST_P(OpWarpPerspectivePlanar, varshape_matches_interleaved)
{
    RunPlanarParityVarShapeCase(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<5>(), GetParamValue<6>(),
                                GetParamValue<7>(), GetParamValue<8>(), GetParamValue<2>(), GetParamValue<3>(),
                                GetParamValue<4>());
}

// clang-format off
NVCV_TEST_SUITE_P(OpWarpPerspective_Negative, test::ValueList<nvcv::ImageFormat, nvcv::ImageFormat>{
    // input format, output format,
    {nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
    {nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16}
});

NVCV_TEST_SUITE_P(OpWarpPerspectiveVarshape_Negative, test::ValueList<int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
    // maxBatchSize, numImages, input format, output format
    {5, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
    {5, 5, nvcv::FMT_RGBA8p, nvcv::FMT_RGBA8},
    {5, 5, nvcv::FMT_RGBAf16, nvcv::FMT_RGBAf16},
    {0, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8},
    {2, 5, nvcv::FMT_RGBA8, nvcv::FMT_RGBA8}
});

// clang-format on

TEST_P(OpWarpPerspective_Negative, op)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const nvcv::ImageFormat inFmt  = GetParamValue<0>();
    const nvcv::ImageFormat outFmt = GetParamValue<1>();

    NVCVPerspectiveTransform transMatrix   = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    NVCVInterpolationType    interpolation = NVCV_INTERP_NEAREST;
    NVCVBorderType           borderMode    = NVCV_BORDER_CONSTANT;
    const float4             borderValue   = {1, 2, 3, 4};
    bool                     inverseMap    = true;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Create input and output
    nvcv::Tensor imgSrc(2, {5, 4}, inFmt);
    nvcv::Tensor imgDst(2, {5, 4}, outFmt);

    cvcuda::WarpPerspective warpPerspectiveOp(0);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&warpPerspectiveOp, &stream, &imgSrc, &imgDst, &transMatrix, &flags, &borderMode, &borderValue]
                  { warpPerspectiveOp(stream, imgSrc, imgDst, transMatrix, flags, borderMode, borderValue); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpWarpPerspectiveVarshape_Negative, op)
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

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

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

    cvcuda::WarpPerspective warpPerspectiveOp(maxBatchSize);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall(
            [&warpPerspectiveOp, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &borderMode, &borderValue]
            { warpPerspectiveOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue); }));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspective_Negative, create_null_handle)
{
    EXPECT_EQ(cvcudaWarpPerspectiveCreate(nullptr, 2), NVCV_ERROR_INVALID_ARGUMENT);
}

// Regression test for CVCUDA issue #249: a projective matrix whose singular line
// falls inside the destination image produces source coordinates near
// +/-INT32_MAX. With BORDER_REPLICATE this used to dereference a wild index,
// triggering cudaErrorIllegalAddress. A plain stream sync is enough to surface
// the kernel crash; the output values are not checked because the test-side
// gold implementation has the same host-side float-to-int saturation hazard as
// the kernel and would need its own hardening to match the fix exactly.
TEST(OpWarpPerspective, extreme_projection_replicate_issue_249)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int               srcWidth  = 1928;
    const int               srcHeight = 1208;
    const int               dstWidth  = 1928;
    const int               dstHeight = 1208;
    const int               batchSize = 1;
    const nvcv::ImageFormat fmt       = nvcv::FMT_RGB8;

    nvcv::Tensor imgSrc(batchSize, {srcWidth, srcHeight}, fmt);
    nvcv::Tensor imgDst(batchSize, {dstWidth, dstHeight}, fmt);

    NVCVPerspectiveTransform transMatrix = {
        8.08776838e-02f,  2.36326631e+00f,  -4.08795000e+02f, -1.28514739e-02f, 2.55201343e-01f,
        -8.45896673e+01f, -2.68404432e-04f, -6.57235630e-04f, 1.00000000e+00f,
    };

    const int    flags       = NVCV_INTERP_LINEAR;
    const float4 borderValue = {0, 0, 0, 0};

    cvcuda::WarpPerspective warpPerspectiveOp(0);
    EXPECT_NO_THROW(warpPerspectiveOp(stream, imgSrc, imgDst, transMatrix, flags, NVCV_BORDER_REPLICATE, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspective_Negative, invalid_border_mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVPerspectiveTransform transMatrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float4             borderValue = {0, 0, 0, 0};
    const int                flags       = NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP;

    nvcv::Tensor imgSrc(1, {4, 4}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {4, 4}, nvcv::FMT_U8);

    cvcuda::WarpPerspective op(0);

    // 5 is one past the last valid NVCVBorderType value (NVCV_BORDER_REFLECT101 = 4)
    auto invalidBorder = static_cast<NVCVBorderType>(5);
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall([&op, &stream, &imgSrc, &imgDst, &transMatrix, &flags, &invalidBorder, &borderValue]
                                { op(stream, imgSrc, imgDst, transMatrix, flags, invalidBorder, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspective_Negative, invalid_interpolation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    NVCVPerspectiveTransform transMatrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float4             borderValue = {0, 0, 0, 0};

    nvcv::Tensor imgSrc(1, {4, 4}, nvcv::FMT_U8);
    nvcv::Tensor imgDst(1, {4, 4}, nvcv::FMT_U8);

    cvcuda::WarpPerspective op(0);

    // NVCV_INTERP_AREA (3) is not supported by the warp ops
    const int flags = static_cast<int>(NVCV_INTERP_AREA) | NVCV_WARP_INVERSE_MAP;
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&op, &stream, &imgSrc, &imgDst, &transMatrix, &flags, &borderValue]
                          { op(stream, imgSrc, imgDst, transMatrix, flags, NVCV_BORDER_CONSTANT, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspectiveVarshape_Negative, invalid_border_mode)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    numImages   = 2;
    const float4 borderValue = {0, 0, 0, 0};
    const int    flags       = NVCV_INTERP_NEAREST | NVCV_WARP_INVERSE_MAP;

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

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

    cvcuda::WarpPerspective op(numImages);

    auto invalidBorder = static_cast<NVCVBorderType>(5);
    EXPECT_EQ(
        NVCV_ERROR_INVALID_ARGUMENT,
        nvcv::ProtectCall([&op, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &invalidBorder, &borderValue]
                          { op(stream, batchSrc, batchDst, transMatrixTensor, flags, invalidBorder, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspectiveVarshape_Negative, invalid_interpolation)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const int    numImages   = 2;
    const float4 borderValue = {0, 0, 0, 0};

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

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

    cvcuda::WarpPerspective op(numImages);

    // NVCV_INTERP_AREA (3) is not supported by the warp ops
    const int flags = static_cast<int>(NVCV_INTERP_AREA) | NVCV_WARP_INVERSE_MAP;
    EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT,
              nvcv::ProtectCall(
                  [&op, &stream, &batchSrc, &batchDst, &transMatrixTensor, &flags, &borderValue]
                  { op(stream, batchSrc, batchDst, transMatrixTensor, flags, NVCV_BORDER_CONSTANT, borderValue); }));

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpWarpPerspectiveVarshape_Negative, different_format_varshape)
{
    std::vector<std::pair<nvcv::ImageFormat, nvcv::ImageFormat>> extraFmts{
        { nvcv::FMT_RGB8, nvcv::FMT_RGBA8},
        {nvcv::FMT_RGBA8,  nvcv::FMT_RGB8}
    };

    for (const auto &[extraFmtSrc, extraFmtDst] : extraFmts)
    {
        cudaStream_t stream;
        EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        const int               numImages = 10;
        const nvcv::ImageFormat fmt       = nvcv::FMT_RGB8;

        NVCVInterpolationType interpolation = NVCV_INTERP_NEAREST;
        NVCVBorderType        borderMode    = NVCV_BORDER_CONSTANT;
        const float4          borderValue   = {1, 2, 3, 4};
        bool                  inverseMap    = true;

        const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

        nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numImages, 9}, nvcv::TENSOR_NW), nvcv::TYPE_F32);

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

        cvcuda::WarpPerspective warpPerspectiveOp(numImages);
        EXPECT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcv::ProtectCall(
                                                   [&warpPerspectiveOp, &stream, &batchSrc, &batchDst,
                                                    &transMatrixTensor, &flags, &borderMode, &borderValue] {
                                                       warpPerspectiveOp(stream, batchSrc, batchDst, transMatrixTensor,
                                                                         flags, borderMode, borderValue);
                                                   }));

        EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }
}
