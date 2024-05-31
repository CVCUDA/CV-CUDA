/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/cuda/DropCast.hpp>     // for SaturateCast, etc.
#include <nvcv/cuda/MathOps.hpp>      // for operator *, etc.
#include <nvcv/cuda/MathWrappers.hpp> // for ROUND, etc
#include <nvcv/cuda/SaturateCast.hpp> // for SaturateCast, etc.
#include <nvcv/cuda/TypeTraits.hpp>   // for BaseType, etc.
#include <util/Assert.h>              // for NVCV_ASSERT, etc.

#include <cmath>

namespace nvcv::test {

template<typename T, T MinVal, T MaxVal>
void _Resize(std::vector<T> &hDst, int dstStep, nvcv::Size2D dstSize, const std::vector<T> &hSrc, int srcStep,
             nvcv::Size2D srcSize, nvcv::ImageFormat fmt, NVCVInterpolationType interp, bool isVarshape)
{
    double scaleH = static_cast<double>(srcSize.h) / dstSize.h;
    double scaleW = static_cast<double>(srcSize.w) / dstSize.w;

    assert(fmt.numPlanes() == 1);

    int channels = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    for (int dy = 0; dy < dstSize.h; dy++)
    {
        for (int dx = 0; dx < dstSize.w; dx++)
        {
            if (interp == NVCV_INTERP_AREA)
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
                    double out = 0.0;

                    if (std::ceil(scaleW) == scaleW && std::ceil(scaleH) == scaleH)
                    {
                        double invscale = 1.f / (scaleW * scaleH);

                        for (int dy = sy1; dy < sy2; ++dy)
                        {
                            for (int dx = sx1; dx < sx2; ++dx)
                            {
                                if (dy >= 0 && dy < srcSize.h && dx >= 0 && dx < srcSize.w)
                                {
                                    out = out + srcPtr[dy * srcStep + dx * channels + c] * invscale;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (!isVarshape || (scaleH >= 1.0f && scaleW >= 1.0f))
                        {
                            double invscale
                                = 1.f / (std::min(scaleW, srcSize.w - fsx1) * std::min(scaleH, srcSize.h - fsy1));

                            for (int dy = sy1; dy < sy2; ++dy)
                            {
                                for (int dx = sx1; dx < sx2; ++dx)
                                    if (dy >= 0 && dy < srcSize.h && dx >= 0 && dx < srcSize.w)
                                        out = out + srcPtr[dy * srcStep + dx * channels + c] * invscale;

                                if (sx1 > fsx1)
                                    if (dy >= 0 && dy < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                        out = out
                                            + srcPtr[dy * srcStep + (sx1 - 1) * channels + c]
                                                  * ((sx1 - fsx1) * invscale);

                                if (sx2 < fsx2)
                                    if (dy >= 0 && dy < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                        out = out
                                            + srcPtr[dy * srcStep + sx2 * channels + c] * ((fsx2 - sx2) * invscale);
                            }

                            if (sy1 > fsy1)
                                for (int dx = sx1; dx < sx2; ++dx)
                                    if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && dx >= 0 && dx < srcSize.w)
                                        out = out
                                            + srcPtr[(sy1 - 1) * srcStep + dx * channels + c]
                                                  * ((sy1 - fsy1) * invscale);

                            if (sy2 < fsy2)
                                for (int dx = sx1; dx < sx2; ++dx)
                                    if (sy2 >= 0 && sy2 < srcSize.h && dx >= 0 && dx < srcSize.w)
                                        out = out
                                            + srcPtr[sy2 * srcStep + dx * channels + c] * ((fsy2 - sy2) * invscale);

                            if ((sy1 > fsy1) && (sx1 > fsx1))
                                if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                    out = out
                                        + srcPtr[(sy1 - 1) * srcStep + (sx1 - 1) * channels + c]
                                              * ((sy1 - fsy1) * (sx1 - fsx1) * invscale);

                            if ((sy1 > fsy1) && (sx2 < fsx2))
                                if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                    out = out
                                        + srcPtr[(sy1 - 1) * srcStep + sx2 * channels + c]
                                              * ((sy1 - fsy1) * (fsx2 - sx2) * invscale);

                            if ((sy2 < fsy2) && (sx2 < fsx2))
                                if (sy2 >= 0 && sy2 < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                    out = out
                                        + srcPtr[sy2 * srcStep + sx2 * channels + c]
                                              * ((fsy2 - sy2) * (fsx2 - sx2) * invscale);

                            if ((sy2 < fsy2) && (sx1 > fsx1))
                                if (sy2 >= 0 && sy2 < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                    out = out
                                        + srcPtr[sy2 * srcStep + (sx1 - 1) * channels + c]
                                              * ((fsy2 - sy2) * (sx1 - fsx1) * invscale);
                        }
                        else // zoom in for varshape
                        {
                            double scaleH_inv = 1.0 / scaleH;
                            double scaleW_inv = 1.0 / scaleW;

                            sy1      = cuda::round<cuda::RoundMode::DOWN, int>(fsy1);
                            sx1      = cuda::round<cuda::RoundMode::DOWN, int>(fsx1);
                            float fy = (float)(float(dy + 1) - float(sy1 + 1) * scaleH_inv);
                            fy       = fy <= 0 ? 0.f : fy - cuda::round<cuda::RoundMode::DOWN, int>(fy);

                            float cbufy[2];
                            cbufy[0] = 1.f - fy;
                            cbufy[1] = fy;

                            float fx = (float)(float(dx + 1) - float(sx1 + 1) * scaleW_inv);
                            fx       = fx <= 0 ? 0.f : fx - cuda::round<cuda::RoundMode::DOWN, int>(fx);

                            if (sx1 < 0)
                            {
                                fx = 0, sx1 = 0;
                            }
                            if (sx1 >= srcSize.w - 1)
                            {
                                fx = 0, sx1 = srcSize.w - 2;
                            }
                            if (sy1 >= srcSize.h - 1)
                            {
                                sy1 = srcSize.h - 2;
                            }

                            float cbufx[2];
                            cbufx[0] = 1.f - fx;
                            cbufx[1] = fx;
                            out      = srcPtr[sy1 * srcStep + sx1 * channels + c] * cbufx[0] * cbufy[0]
                                + srcPtr[(sy1 + 1) * srcStep + sx1 * channels + c] * cbufx[0] * cbufy[1]
                                + srcPtr[sy1 * srcStep + (sx1 + 1) * channels + c] * cbufx[1] * cbufy[0]
                                + srcPtr[(sy1 + 1) * srcStep + (sx1 + 1) * channels + c] * cbufx[1] * cbufy[1];
                        }
                    }

                    out = std::rint(std::abs(out));

                    dstPtr[dy * dstStep + dx * channels + c] = out < MinVal ? MinVal : (out > MaxVal ? MaxVal : out);
                }
            }
        }
    }
}

template<typename T, T MinVal, T MaxVal>
void _ResizedCrop(std::vector<T> &hDst, int dstStep, nvcv::Size2D dstSize, const std::vector<T> &hSrc, int srcStep,
                  nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols, nvcv::ImageFormat fmt,
                  NVCVInterpolationType interp)
{
    double scaleH = static_cast<double>(crop_rows) / dstSize.h;
    double scaleW = static_cast<double>(crop_cols) / dstSize.w;

    assert(fmt.numPlanes() == 1);

    int channels = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    for (int dy = 0; dy < dstSize.h; dy++)
    {
        for (int dx = 0; dx < dstSize.w; dx++)
        {
            if (interp == NVCV_INTERP_NEAREST)
            {
                double fy = scaleH * dy + top;
                double fx = scaleW * dx + left;

                int sy = std::floor(fy);
                int sx = std::floor(fx);

                sy = std::min(sy, srcSize.h - 1);
                sx = std::min(sx, srcSize.w - 1);

                for (int c = 0; c < channels; c++)
                {
                    dstPtr[dy * dstStep + dx * channels + c] = srcPtr[sy * srcStep + sx * channels + c];
                }
            }
            else if (interp == NVCV_INTERP_LINEAR)
            {
                double fy = scaleH * (dy + 0.5) - 0.5 + top;
                double fx = scaleW * (dx + 0.5) - 0.5 + left;

                int sy = std::floor(fy);
                int sx = std::floor(fx);

                fy -= sy;
                fx -= sx;

                fx = (sx < 0 || sx >= srcSize.w - 1) ? 0 : fx;

                sy = std::max(0, std::min(sy, srcSize.h - 2));
                sx = std::max(0, std::min(sx, srcSize.w - 2));

                double wghtY[2] = {1 - fy, fy};
                double wghtX[2] = {1 - fx, fx};

                for (int c = 0; c < channels; c++)
                {
                    double res = std::rint(
                        std::abs(srcPtr[(sy + 0) * srcStep + (sx + 0) * channels + c] * wghtY[0] * wghtX[0]
                                 + srcPtr[(sy + 1) * srcStep + (sx + 0) * channels + c] * wghtY[1] * wghtX[0]
                                 + srcPtr[(sy + 0) * srcStep + (sx + 1) * channels + c] * wghtY[0] * wghtX[1]
                                 + srcPtr[(sy + 1) * srcStep + (sx + 1) * channels + c] * wghtY[1] * wghtX[1]));

                    dstPtr[dy * dstStep + dx * channels + c] = res < MinVal ? MinVal : (res > MaxVal ? MaxVal : res);
                }
            }
            else if (interp == NVCV_INTERP_CUBIC)
            {
                double fy = scaleH * (dy + 0.5) - 0.5 + top;
                double fx = scaleW * (dx + 0.5) - 0.5 + left;

                int sy = std::floor(fy);
                int sx = std::floor(fx);

                fy -= sy;
                fx -= sx;

                fx = (sx < 1 || sx >= srcSize.w - 3) ? 0 : fx;

                sy = std::max(1, std::min(sy, srcSize.h - 3));
                sx = std::max(1, std::min(sx, srcSize.w - 3));

                const double A = -0.75;
                double       wghtY[4];
                wghtY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
                wghtY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
                wghtY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
                wghtY[3] = 1 - wghtY[0] - wghtY[1] - wghtY[2];

                double wghtX[4];
                wghtX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
                wghtX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
                wghtX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
                wghtX[3] = 1 - wghtX[0] - wghtX[1] - wghtX[2];

                for (int c = 0; c < channels; c++)
                {
                    double res = std::rint(
                        std::abs(srcPtr[(sy - 1) * srcStep + (sx - 1) * channels + c] * wghtX[0] * wghtY[0]
                                 + srcPtr[(sy + 0) * srcStep + (sx - 1) * channels + c] * wghtX[0] * wghtY[1]
                                 + srcPtr[(sy + 1) * srcStep + (sx - 1) * channels + c] * wghtX[0] * wghtY[2]
                                 + srcPtr[(sy + 2) * srcStep + (sx - 1) * channels + c] * wghtX[0] * wghtY[3]
                                 + srcPtr[(sy - 1) * srcStep + (sx + 0) * channels + c] * wghtX[1] * wghtY[0]
                                 + srcPtr[(sy + 0) * srcStep + (sx + 0) * channels + c] * wghtX[1] * wghtY[1]
                                 + srcPtr[(sy + 1) * srcStep + (sx + 0) * channels + c] * wghtX[1] * wghtY[2]
                                 + srcPtr[(sy + 2) * srcStep + (sx + 0) * channels + c] * wghtX[1] * wghtY[3]
                                 + srcPtr[(sy - 1) * srcStep + (sx + 1) * channels + c] * wghtX[2] * wghtY[0]
                                 + srcPtr[(sy + 0) * srcStep + (sx + 1) * channels + c] * wghtX[2] * wghtY[1]
                                 + srcPtr[(sy + 1) * srcStep + (sx + 1) * channels + c] * wghtX[2] * wghtY[2]
                                 + srcPtr[(sy + 2) * srcStep + (sx + 1) * channels + c] * wghtX[2] * wghtY[3]
                                 + srcPtr[(sy - 1) * srcStep + (sx + 2) * channels + c] * wghtX[3] * wghtY[0]
                                 + srcPtr[(sy + 0) * srcStep + (sx + 2) * channels + c] * wghtX[3] * wghtY[1]
                                 + srcPtr[(sy + 1) * srcStep + (sx + 2) * channels + c] * wghtX[3] * wghtY[2]
                                 + srcPtr[(sy + 2) * srcStep + (sx + 2) * channels + c] * wghtX[3] * wghtY[3]));

                    dstPtr[dy * dstStep + dx * channels + c] = res < MinVal ? MinVal : (res > MaxVal ? MaxVal : res);
                }
            }
        }
    }
}

void Resize(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
            int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, NVCVInterpolationType interp,
            bool isVarShape)
{
    int dstStep = dstRowStride / sizeof(uint8_t);
    int srcStep = srcRowStride / sizeof(uint8_t);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        _ResizedCrop<uint8_t, 0, 255>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, 0, 0, srcSize.h, srcSize.w, fmt,
                                      interp);
    }
    else if (interp == NVCV_INTERP_AREA)
    {
        _Resize<uint8_t, 0, 255>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, fmt, interp, isVarShape);
    }
}

void ResizedCrop(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
                 int srcRowStride, nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols,
                 nvcv::ImageFormat fmt, NVCVInterpolationType interp)
{
    int dstStep = dstRowStride / sizeof(uint8_t);
    int srcStep = srcRowStride / sizeof(uint8_t);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        _ResizedCrop<uint8_t, 0, 255>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, top, left, crop_rows, crop_cols,
                                      fmt, interp);
    }
}

void Resize(std::vector<float> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<float> &hSrc,
            int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, NVCVInterpolationType interp,
            bool isVarShape)
{
    int dstStep = dstRowStride / sizeof(float);
    int srcStep = srcRowStride / sizeof(float);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        _ResizedCrop<float, 0.0f, 255.0f>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, 0, 0, srcSize.h, srcSize.w,
                                          fmt, interp);
    }
    else if (interp == NVCV_INTERP_AREA)
    {
        _Resize<float, 0.0f, 255.0f>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, fmt, interp, isVarShape);
    }
}

void ResizedCrop(std::vector<float> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<float> &hSrc,
                 int srcRowStride, nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols,
                 nvcv::ImageFormat fmt, NVCVInterpolationType interp)
{
    int dstStep = dstRowStride / sizeof(float);
    int srcStep = srcRowStride / sizeof(float);

    if (interp == NVCV_INTERP_NEAREST || interp == NVCV_INTERP_LINEAR || interp == NVCV_INTERP_CUBIC)
    {
        _ResizedCrop<float, 0.0f, 255.0f>(hDst, dstStep, dstSize, hSrc, srcStep, srcSize, top, left, crop_rows,
                                          crop_cols, fmt, interp);
    }
}

} // namespace nvcv::test
