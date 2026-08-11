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

#include "ResizeUtils.hpp"

#include <cvcuda/cuda_tools/DropCast.hpp>     // for SaturateCast, etc.
#include <cvcuda/cuda_tools/MathOps.hpp>      // for operator *, etc.
#include <cvcuda/cuda_tools/MathWrappers.hpp> // for ROUND, etc
#include <cvcuda/cuda_tools/SaturateCast.hpp> // for SaturateCast, etc.
#include <cvcuda/cuda_tools/TypeTraits.hpp>   // for BaseType, etc.
#include <nvcv/util/Assert.h>                 // for NVCV_ASSERT, etc.

#include <algorithm>
#include <array>
#include <cmath>

namespace nvcv::test {

static bool InBounds(int y, int x, nvcv::Size2D size)
{
    return y >= 0 && y < size.h && x >= 0 && x < size.w;
}

template<typename T>
static void AccumulateArea(double &out, const T *srcPtr, int srcStep, int channels, int c, nvcv::Size2D srcSize, int y,
                           int x, double weight)
{
    if (InBounds(y, x, srcSize))
    {
        out += srcPtr[y * srcStep + x * channels + c] * weight;
    }
}

template<typename T>
static double IntegerAreaPixel(const T *srcPtr, int srcStep, nvcv::Size2D srcSize, int channels, int c, int sy1,
                               int sy2, int sx1, int sx2, double invScale)
{
    double out = 0.0;
    for (int y = sy1; y < sy2; ++y)
    {
        for (int x = sx1; x < sx2; ++x)
        {
            AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, y, x, invScale);
        }
    }
    return out;
}

template<typename T>
static void AccumulateAreaRow(double &out, const T *srcPtr, int srcStep, nvcv::Size2D srcSize, int channels, int c,
                              int y, int sx1, int sx2, double weight)
{
    for (int x = sx1; x < sx2; ++x)
    {
        AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, y, x, weight);
    }
}

template<typename T>
static double FractionalAreaPixel(const T *srcPtr, int srcStep, nvcv::Size2D srcSize, int channels, int c, int sy1,
                                  int sy2, int sx1, int sx2, double fsy1, double fsy2, double fsx1, double fsx2,
                                  double invScale)
{
    double out = 0.0;
    for (int y = sy1; y < sy2; ++y)
    {
        AccumulateAreaRow(out, srcPtr, srcStep, srcSize, channels, c, y, sx1, sx2, invScale);

        if (sx1 > fsx1)
        {
            AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, y, sx1 - 1, (sx1 - fsx1) * invScale);
        }

        if (sx2 < fsx2)
        {
            AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, y, sx2, (fsx2 - sx2) * invScale);
        }
    }

    if (sy1 > fsy1)
    {
        AccumulateAreaRow(out, srcPtr, srcStep, srcSize, channels, c, sy1 - 1, sx1, sx2, (sy1 - fsy1) * invScale);
    }

    if (sy2 < fsy2)
    {
        AccumulateAreaRow(out, srcPtr, srcStep, srcSize, channels, c, sy2, sx1, sx2, (fsy2 - sy2) * invScale);
    }

    if (sy1 > fsy1 && sx1 > fsx1)
    {
        AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, sy1 - 1, sx1 - 1,
                       (sy1 - fsy1) * (sx1 - fsx1) * invScale);
    }

    if (sy1 > fsy1 && sx2 < fsx2)
    {
        AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, sy1 - 1, sx2,
                       (sy1 - fsy1) * (fsx2 - sx2) * invScale);
    }

    if (sy2 < fsy2 && sx2 < fsx2)
    {
        AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, sy2, sx2, (fsy2 - sy2) * (fsx2 - sx2) * invScale);
    }

    if (sy2 < fsy2 && sx1 > fsx1)
    {
        AccumulateArea(out, srcPtr, srcStep, channels, c, srcSize, sy2, sx1 - 1,
                       (fsy2 - sy2) * (sx1 - fsx1) * invScale);
    }

    return out;
}

template<typename T>
static double VarShapeZoomAreaPixel(const T *srcPtr, int srcStep, nvcv::Size2D srcSize, int channels, int c, int dstY,
                                    int dstX, double scaleH, double scaleW, double fsy1, double fsx1)
{
    double scaleHInv = 1.0 / scaleH;
    double scaleWInv = 1.0 / scaleW;

    int sx1 = cuda::round<cuda::RoundMode::DOWN, int>(fsx1);
    int sy1 = cuda::round<cuda::RoundMode::DOWN, int>(fsy1);

    auto fy = static_cast<double>(dstY + 1) - static_cast<double>(sy1 + 1) * scaleHInv;
    fy      = fy <= 0.0 ? 0.0 : fy - static_cast<double>(cuda::round<cuda::RoundMode::DOWN, int>(fy));

    auto fx = static_cast<double>(dstX + 1) - static_cast<double>(sx1 + 1) * scaleWInv;
    fx      = fx <= 0.0 ? 0.0 : fx - static_cast<double>(cuda::round<cuda::RoundMode::DOWN, int>(fx));

    if (sx1 < 0)
    {
        fx  = 0;
        sx1 = 0;
    }
    if (sx1 >= srcSize.w - 1)
    {
        fx  = 0;
        sx1 = srcSize.w - 2;
    }
    if (sy1 >= srcSize.h - 1)
    {
        sy1 = srcSize.h - 2;
    }

    std::array<double, 2> cbufx = {1.0 - fx, fx};
    std::array<double, 2> cbufy = {1.0 - fy, fy};

    return srcPtr[sy1 * srcStep + sx1 * channels + c] * cbufx[0] * cbufy[0]
         + srcPtr[(sy1 + 1) * srcStep + sx1 * channels + c] * cbufx[0] * cbufy[1]
         + srcPtr[sy1 * srcStep + (sx1 + 1) * channels + c] * cbufx[1] * cbufy[0]
         + srcPtr[(sy1 + 1) * srcStep + (sx1 + 1) * channels + c] * cbufx[1] * cbufy[1];
}

template<typename T>
static T ClampResizeValue(double out, T minVal, T maxVal)
{
    if (std::numeric_limits<T>::is_integer)
    {
        out = std::rint(std::numeric_limits<T>::is_signed ? out : std::abs(out));
    }

    if (out < static_cast<double>(minVal))
    {
        return minVal;
    }
    if (out > static_cast<double>(maxVal))
    {
        return maxVal;
    }
    return static_cast<T>(out);
}

template<typename T>
static double AreaResizePixel(const T *srcPtr, int srcStep, nvcv::Size2D srcSize, int channels, int c, int dstY,
                              int dstX, int sy1, int sy2, int sx1, int sx2, double fsy1, double fsy2, double fsx1,
                              double fsx2, double scaleH, double scaleW, bool isVarshape)
{
    if (std::ceil(scaleW) == scaleW && std::ceil(scaleH) == scaleH)
    {
        return IntegerAreaPixel(srcPtr, srcStep, srcSize, channels, c, sy1, sy2, sx1, sx2, 1.f / (scaleW * scaleH));
    }

    if (!isVarshape || (scaleH >= 1.0f && scaleW >= 1.0f))
    {
        double invScale = 1.f / (std::min(scaleW, srcSize.w - fsx1) * std::min(scaleH, srcSize.h - fsy1));
        return FractionalAreaPixel(srcPtr, srcStep, srcSize, channels, c, sy1, sy2, sx1, sx2, fsy1, fsy2, fsx1, fsx2,
                                   invScale);
    }

    return VarShapeZoomAreaPixel(srcPtr, srcStep, srcSize, channels, c, dstY, dstX, scaleH, scaleW, fsy1, fsx1);
}

template<typename T>
void resize(T *dstPtr, int dstStep, nvcv::Size2D dstSize, const T *srcPtr, int srcStep, nvcv::Size2D srcSize,
            nvcv::ImageFormat frmt, bool isVarshape, T minVal, T maxVal)
{
    auto scaleH = static_cast<double>(srcSize.h) / dstSize.h;
    auto scaleW = static_cast<double>(srcSize.w) / dstSize.w;

    assert(frmt.numPlanes() == 1);

    int channels = frmt.numChannels();

    for (int dy = 0; dy < dstSize.h; dy++)
    {
        for (int dx = 0; dx < dstSize.w; dx++)
        {
            double fsx1 = dx * scaleW;
            double fsx2 = fsx1 + scaleW;
            double fsy1 = dy * scaleH;
            double fsy2 = fsy1 + scaleH;
            int    sx1  = cuda::round<cuda::RoundMode::UP, int>(fsx1);
            int    sx2  = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);
            int    sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
            int    sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

            for (int c = 0; c < channels; c++)
            {
                double out = AreaResizePixel(srcPtr, srcStep, srcSize, channels, c, dy, dx, sy1, sy2, sx1, sx2, fsy1,
                                             fsy2, fsx1, fsx2, scaleH, scaleW, isVarshape);

                dstPtr[dy * dstStep + dx * channels + c] = ClampResizeValue(out, minVal, maxVal);
            }
        }
    }
}

template<typename T>
static void StoreNearestCropPixel(T *dstPtr, int dstStep, const T *srcPtr, int srcStep, nvcv::Size2D srcSize,
                                  int channels, int dy, int dx, int top, int left, float scaleH, float scaleW)
{
    float fy = scaleH * (static_cast<float>(dy) + 0.5f) + static_cast<float>(top);
    float fx = scaleW * (static_cast<float>(dx) + 0.5f) + static_cast<float>(left);

    auto sy = static_cast<int>(std::floor(fy));
    auto sx = static_cast<int>(std::floor(fx));

    sy = std::min(sy, srcSize.h - 1);
    sx = std::min(sx, srcSize.w - 1);

    int srcOffset = sy * srcStep + sx * channels;
    int dstOffset = dy * dstStep + dx * channels;

    for (int c = 0; c < channels; c++)
    {
        dstPtr[dstOffset + c] = srcPtr[srcOffset + c];
    }
}

template<typename T>
static void StoreLinearCropPixel(T *dstPtr, int dstStep, const T *srcPtr, int srcStep, nvcv::Size2D srcSize,
                                 int channels, int dy, int dx, int top, int left, float scaleH, float scaleW, T minVal,
                                 T maxVal)
{
    auto fy = static_cast<double>(scaleH) * (static_cast<double>(dy) + 0.5) - 0.5 + static_cast<double>(top);
    auto fx = static_cast<double>(scaleW) * (static_cast<double>(dx) + 0.5) - 0.5 + static_cast<double>(left);

    auto sy = static_cast<int>(std::floor(fy));
    auto sx = static_cast<int>(std::floor(fx));

    if (sy < 0)
    {
        fy = 0.0;
    }
    else if (sy > srcSize.h - 2)
    {
        fy = 1.0;
    }
    else
    {
        fy -= static_cast<double>(sy);
    }

    if (sx < 0)
    {
        fx = 0.0;
    }
    else if (sx > srcSize.w - 2)
    {
        fx = 1.0;
    }
    else
    {
        fx -= static_cast<double>(sx);
    }

    sy = std::clamp(sy, 0, srcSize.h - 2);
    sx = std::clamp(sx, 0, srcSize.w - 2);

    std::array<double, 2> wghtY = {1 - fy, fy};
    std::array<double, 2> wghtX = {1 - fx, fx};

    int dstOffset = dy * dstStep + dx * channels;

    for (int c = 0; c < channels; c++)
    {
        double res = std::rint(std::abs(srcPtr[(sy + 0) * srcStep + (sx + 0) * channels + c] * wghtY[0] * wghtX[0]
                                        + srcPtr[(sy + 1) * srcStep + (sx + 0) * channels + c] * wghtY[1] * wghtX[0]
                                        + srcPtr[(sy + 0) * srcStep + (sx + 1) * channels + c] * wghtY[0] * wghtX[1]
                                        + srcPtr[(sy + 1) * srcStep + (sx + 1) * channels + c] * wghtY[1] * wghtX[1]));

        dstPtr[dstOffset + c] = ClampResizeValue<T>(res, minVal, maxVal);
    }
}

static std::array<double, 4> CubicWeights(double frac)
{
    const double a = -0.75;

    std::array<double, 4> weights;
    weights[0] = ((a * (frac + 1) - 5 * a) * (frac + 1) + 8 * a) * (frac + 1) - 4 * a;
    weights[1] = ((a + 2) * frac - (a + 3)) * frac * frac + 1;
    weights[2] = ((a + 2) * (1 - frac) - (a + 3)) * (1 - frac) * (1 - frac) + 1;
    weights[3] = 1 - weights[0] - weights[1] - weights[2];
    return weights;
}

template<typename T>
static void StoreCubicCropPixel(T *dstPtr, int dstStep, const T *srcPtr, int srcStep, nvcv::Size2D srcSize,
                                int channels, int dy, int dx, int top, int left, float scaleH, float scaleW, T minVal,
                                T maxVal)
{
    auto fy = static_cast<double>(scaleH) * (static_cast<double>(dy) + 0.5) - 0.5 + static_cast<double>(top);
    auto fx = static_cast<double>(scaleW) * (static_cast<double>(dx) + 0.5) - 0.5 + static_cast<double>(left);

    auto sy = static_cast<int>(std::floor(fy));
    auto sx = static_cast<int>(std::floor(fx));

    fy -= static_cast<double>(sy);
    fx -= static_cast<double>(sx);

    std::array<double, 4> wghtY = CubicWeights(fy);
    std::array<double, 4> wghtX = CubicWeights(fx);

    int dstOffset = dy * dstStep + dx * channels;

    for (int c = 0; c < channels; c++)
    {
        double res = 0;
        for (int ky = 0; ky < 4; ky++)
        {
            int csy = std::clamp(sy + ky - 1, 0, srcSize.h - 1);
            for (int kx = 0; kx < 4; kx++)
            {
                int csx = std::clamp(sx + kx - 1, 0, srcSize.w - 1);
                res += srcPtr[csy * srcStep + csx * channels + c] * wghtX[kx] * wghtY[ky];
            }
        }
        res                   = std::rint(std::clamp(res, static_cast<double>(minVal), static_cast<double>(maxVal)));
        dstPtr[dstOffset + c] = static_cast<T>(res);
    }
}

template<typename T>
void resizedCrop(T *dstPtr, int dstStep, nvcv::Size2D dstSize, const T *srcPtr, int srcStep, nvcv::Size2D srcSize,
                 int top, int left, int crop_rows, int crop_cols, nvcv::ImageFormat frmt, NVCVInterpolationType interp,
                 T MinVal, T MaxVal)
{
    auto scaleH = static_cast<float>(crop_rows) / static_cast<float>(dstSize.h);
    auto scaleW = static_cast<float>(crop_cols) / static_cast<float>(dstSize.w);

    assert(frmt.numPlanes() == 1);

    int channels = frmt.numChannels();

    for (int dy = 0; dy < dstSize.h; dy++)
    {
        for (int dx = 0; dx < dstSize.w; dx++)
        {
            switch (interp)
            {
            case NVCV_INTERP_NEAREST:
                StoreNearestCropPixel(dstPtr, dstStep, srcPtr, srcStep, srcSize, channels, dy, dx, top, left, scaleH,
                                      scaleW);
                break;
            case NVCV_INTERP_LINEAR:
                StoreLinearCropPixel(dstPtr, dstStep, srcPtr, srcStep, srcSize, channels, dy, dx, top, left, scaleH,
                                     scaleW, MinVal, MaxVal);
                break;
            case NVCV_INTERP_CUBIC:
                StoreCubicCropPixel(dstPtr, dstStep, srcPtr, srcStep, srcSize, channels, dy, dx, top, left, scaleH,
                                    scaleW, MinVal, MaxVal);
                break;
            default:
                break;
            }
        }
    }
}

template<typename T>
void _Resize(T *dstPtr, int dstStride, nvcv::Size2D dstSize, const T *srcPtr, int srcStride, nvcv::Size2D srcSize,
             nvcv::ImageFormat frmt, NVCVInterpolationType interp, bool isVarShape)
{
    int dstStep = dstStride / sizeof(T);
    int srcStep = srcStride / sizeof(T);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        resizedCrop<T>(dstPtr, dstStep, dstSize, srcPtr, srcStep, srcSize, 0, 0, srcSize.h, srcSize.w, frmt, interp,
                       std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
    else if (interp == NVCV_INTERP_AREA)
    {
        resize<T>(dstPtr, dstStep, dstSize, srcPtr, srcStep, srcSize, frmt, isVarShape, std::numeric_limits<T>::min(),
                  std::numeric_limits<T>::max());
    }
}

void Resize(std::vector<uint8_t> &dst, int dstStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &src,
            int srcStride, nvcv::Size2D srcSize, nvcv::ImageFormat frmt, NVCVInterpolationType interp, bool isVarShape)
{
    if (frmt.planeDataType(0) == nvcv::TYPE_U16 || frmt.planeDataType(0) == nvcv::TYPE_2U16
        || frmt.planeDataType(0) == nvcv::TYPE_3U16 || frmt.planeDataType(0) == nvcv::TYPE_4U16)
    {
        _Resize(reinterpret_cast<uint16_t *>(dst.data()), dstStride, dstSize,
                reinterpret_cast<const uint16_t *>(src.data()), srcStride, srcSize, frmt, interp, isVarShape);
    }
    else if (frmt.planeDataType(0) == nvcv::TYPE_S16 || frmt.planeDataType(0) == nvcv::TYPE_2S16
             || frmt.planeDataType(0) == nvcv::TYPE_3S16 || frmt.planeDataType(0) == nvcv::TYPE_4S16)
    {
        _Resize(reinterpret_cast<int16_t *>(dst.data()), dstStride, dstSize,
                reinterpret_cast<const int16_t *>(src.data()), srcStride, srcSize, frmt, interp, isVarShape);
    }
    else if (frmt.planeDataType(0) == nvcv::TYPE_F32 || frmt.planeDataType(0) == nvcv::TYPE_2F32
             || frmt.planeDataType(0) == nvcv::TYPE_3F32 || frmt.planeDataType(0) == nvcv::TYPE_4F32)
    {
        _Resize(reinterpret_cast<float *>(dst.data()), dstStride, dstSize, reinterpret_cast<const float *>(src.data()),
                srcStride, srcSize, frmt, interp, isVarShape);
    }
    else
    {
        _Resize(dst.data(), dstStride, dstSize, src.data(), srcStride, srcSize, frmt, interp, isVarShape);
    }
}

void Resize(std::vector<float> &dst, int dstStride, nvcv::Size2D dstSize, const std::vector<float> &src, int srcStride,
            nvcv::Size2D srcSize, nvcv::ImageFormat frmt, NVCVInterpolationType interp, bool isVarShape)
{
    _Resize(dst.data(), dstStride, dstSize, src.data(), srcStride, srcSize, frmt, interp, isVarShape);
}

template<typename T>
void _ResizedCrop(T *dstPtr, int dstStride, nvcv::Size2D dstSize, const T *srcPtr, int srcStride, nvcv::Size2D srcSize,
                  int top, int left, int crop_rows, int crop_cols, nvcv::ImageFormat frmt, NVCVInterpolationType interp)
{
    int dstStep = dstStride / sizeof(T);
    int srcStep = srcStride / sizeof(T);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        resizedCrop<T>(dstPtr, dstStep, dstSize, srcPtr, srcStep, srcSize, top, left, crop_rows, crop_cols, frmt,
                       interp, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
}

void ResizedCrop(std::vector<uint8_t> &dst, int dstStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &src,
                 int srcStride, nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols,
                 nvcv::ImageFormat frmt, NVCVInterpolationType interp)
{
    if (frmt.planeDataType(0) == nvcv::TYPE_U16 || frmt.planeDataType(0) == nvcv::TYPE_2U16
        || frmt.planeDataType(0) == nvcv::TYPE_3U16 || frmt.planeDataType(0) == nvcv::TYPE_4U16)
    {
        _ResizedCrop(reinterpret_cast<uint16_t *>(dst.data()), dstStride, dstSize,
                     reinterpret_cast<const uint16_t *>(src.data()), srcStride, srcSize, top, left, crop_rows,
                     crop_cols, frmt, interp);
    }
    else
    {
        _ResizedCrop(dst.data(), dstStride, dstSize, src.data(), srcStride, srcSize, top, left, crop_rows, crop_cols,
                     frmt, interp);
    }
}

void ResizedCrop(std::vector<float> &dst, int dstStride, nvcv::Size2D dstSize, const std::vector<float> &src,
                 int srcStride, nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols,
                 nvcv::ImageFormat frmt, NVCVInterpolationType interp)
{
    _ResizedCrop(dst.data(), dstStride, dstSize, src.data(), srcStride, srcSize, top, left, crop_rows, crop_cols, frmt,
                 interp);
}

} // namespace nvcv::test
