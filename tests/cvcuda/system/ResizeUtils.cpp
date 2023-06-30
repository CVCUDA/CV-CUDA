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

void Resize(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
            int srcRowStride, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, NVCVInterpolationType interpolation)
{
    if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
        || interpolation == NVCV_INTERP_CUBIC)
    {
        ResizedCrop(hDst, dstRowStride, dstSize, hSrc, srcRowStride, srcSize, 0, 0, srcSize.h, srcSize.w, fmt,
                    interpolation);
        return;
    }

    double iScale = static_cast<double>(srcSize.h) / dstSize.h;
    double jScale = static_cast<double>(srcSize.w) / dstSize.w;

    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t       *dstPtr = hDst.data();
    const uint8_t *srcPtr = hSrc.data();

    for (int di = 0; di < dstSize.h; di++)
    {
        for (int dj = 0; dj < dstSize.w; dj++)
        {
            if (interpolation == NVCV_INTERP_AREA)
            {
                double fsx1 = dj * jScale;
                double fsx2 = fsx1 + jScale;
                double fsy1 = di * iScale;
                double fsy2 = fsy1 + iScale;
                int    sx1  = cuda::round<cuda::RoundMode::UP, int>(fsx1);
                int    sx2  = cuda::round<cuda::RoundMode::DOWN, int>(fsx2);
                int    sy1  = cuda::round<cuda::RoundMode::UP, int>(fsy1);
                int    sy2  = cuda::round<cuda::RoundMode::DOWN, int>(fsy2);

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    double out = 0.0;

                    if (std::ceil(jScale) == jScale && std::ceil(iScale) == iScale)
                    {
                        double invscale = 1.f / (jScale * iScale);

                        for (int dy = sy1; dy < sy2; ++dy)
                        {
                            for (int dx = sx1; dx < sx2; ++dx)
                            {
                                if (dy >= 0 && dy < srcSize.h && dx >= 0 && dx < srcSize.w)
                                {
                                    out = out + srcPtr[dy * srcRowStride + dx * elementsPerPixel + k] * invscale;
                                }
                            }
                        }
                    }
                    else
                    {
                        double invscale
                            = 1.f / (std::min(jScale, srcSize.w - fsx1) * std::min(iScale, srcSize.h - fsy1));

                        for (int dy = sy1; dy < sy2; ++dy)
                        {
                            for (int dx = sx1; dx < sx2; ++dx)
                                if (dy >= 0 && dy < srcSize.h && dx >= 0 && dx < srcSize.w)
                                    out = out + srcPtr[dy * srcRowStride + dx * elementsPerPixel + k] * invscale;

                            if (sx1 > fsx1)
                                if (dy >= 0 && dy < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                    out = out
                                        + srcPtr[dy * srcRowStride + (sx1 - 1) * elementsPerPixel + k]
                                              * ((sx1 - fsx1) * invscale);

                            if (sx2 < fsx2)
                                if (dy >= 0 && dy < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                    out = out
                                        + srcPtr[dy * srcRowStride + sx2 * elementsPerPixel + k]
                                              * ((fsx2 - sx2) * invscale);
                        }

                        if (sy1 > fsy1)
                            for (int dx = sx1; dx < sx2; ++dx)
                                if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && dx >= 0 && dx < srcSize.w)
                                    out = out
                                        + srcPtr[(sy1 - 1) * srcRowStride + dx * elementsPerPixel + k]
                                              * ((sy1 - fsy1) * invscale);

                        if (sy2 < fsy2)
                            for (int dx = sx1; dx < sx2; ++dx)
                                if (sy2 >= 0 && sy2 < srcSize.h && dx >= 0 && dx < srcSize.w)
                                    out = out
                                        + srcPtr[sy2 * srcRowStride + dx * elementsPerPixel + k]
                                              * ((fsy2 - sy2) * invscale);

                        if ((sy1 > fsy1) && (sx1 > fsx1))
                            if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                out = out
                                    + srcPtr[(sy1 - 1) * srcRowStride + (sx1 - 1) * elementsPerPixel + k]
                                          * ((sy1 - fsy1) * (sx1 - fsx1) * invscale);

                        if ((sy1 > fsy1) && (sx2 < fsx2))
                            if (sy1 - 1 >= 0 && sy1 - 1 < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                out = out
                                    + srcPtr[(sy1 - 1) * srcRowStride + sx2 * elementsPerPixel + k]
                                          * ((sy1 - fsy1) * (fsx2 - sx2) * invscale);

                        if ((sy2 < fsy2) && (sx2 < fsx2))
                            if (sy2 >= 0 && sy2 < srcSize.h && sx2 >= 0 && sx2 < srcSize.w)
                                out = out
                                    + srcPtr[sy2 * srcRowStride + sx2 * elementsPerPixel + k]
                                          * ((fsy2 - sy2) * (fsx2 - sx2) * invscale);

                        if ((sy2 < fsy2) && (sx1 > fsx1))
                            if (sy2 >= 0 && sy2 < srcSize.h && sx1 - 1 >= 0 && sx1 - 1 < srcSize.w)
                                out = out
                                    + srcPtr[sy2 * srcRowStride + (sx1 - 1) * elementsPerPixel + k]
                                          * ((fsy2 - sy2) * (sx1 - fsx1) * invscale);
                    }

                    out = std::rint(std::abs(out));

                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k] = out < 0 ? 0 : (out > 255 ? 255 : out);
                }
            }
        }
    }
}

void ResizedCrop(std::vector<uint8_t> &hDst, int dstRowStride, nvcv::Size2D dstSize, const std::vector<uint8_t> &hSrc,
                 int srcRowStride, nvcv::Size2D srcSize, int top, int left, int crop_rows, int crop_cols,
                 nvcv::ImageFormat fmt, NVCVInterpolationType interpolation)
{
    double iScale = static_cast<double>(crop_rows) / dstSize.h;
    double jScale = static_cast<double>(crop_cols) / dstSize.w;

    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t       *dstPtr = hDst.data();
    const uint8_t *srcPtr = hSrc.data();

    for (int di = 0; di < dstSize.h; di++)
    {
        for (int dj = 0; dj < dstSize.w; dj++)
        {
            if (interpolation == NVCV_INTERP_NEAREST)
            {
                double fi = iScale * di + top;
                double fj = jScale * dj + left;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                si = std::min(si, srcSize.h - 1);
                sj = std::min(sj, srcSize.w - 1);

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k]
                        = srcPtr[si * srcRowStride + sj * elementsPerPixel + k];
                }
            }
            else if (interpolation == NVCV_INTERP_LINEAR)
            {
                double fi = iScale * (di + 0.5) - 0.5 + top;
                double fj = jScale * (dj + 0.5) - 0.5 + left;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                fi -= si;
                fj -= sj;

                fj = (sj < 0 || sj >= srcSize.w - 1) ? 0 : fj;

                si = std::max(0, std::min(si, srcSize.h - 2));
                sj = std::max(0, std::min(sj, srcSize.w - 2));

                double iWeights[2] = {1 - fi, fi};
                double jWeights[2] = {1 - fj, fj};

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    double res = std::rint(std::abs(
                        srcPtr[(si + 0) * srcRowStride + (sj + 0) * elementsPerPixel + k] * iWeights[0] * jWeights[0]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * iWeights[1] * jWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 1) * elementsPerPixel + k] * iWeights[0] * jWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * iWeights[1]
                              * jWeights[1]));

                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
            else if (interpolation == NVCV_INTERP_CUBIC)
            {
                double fi = iScale * (di + 0.5) - 0.5 + top;
                double fj = jScale * (dj + 0.5) - 0.5 + left;

                int si = std::floor(fi);
                int sj = std::floor(fj);

                fi -= si;
                fj -= sj;

                fj = (sj < 1 || sj >= srcSize.w - 3) ? 0 : fj;

                si = std::max(1, std::min(si, srcSize.h - 3));
                sj = std::max(1, std::min(sj, srcSize.w - 3));

                const double A = -0.75;
                double       iWeights[4];
                iWeights[0] = ((A * (fi + 1) - 5 * A) * (fi + 1) + 8 * A) * (fi + 1) - 4 * A;
                iWeights[1] = ((A + 2) * fi - (A + 3)) * fi * fi + 1;
                iWeights[2] = ((A + 2) * (1 - fi) - (A + 3)) * (1 - fi) * (1 - fi) + 1;
                iWeights[3] = 1 - iWeights[0] - iWeights[1] - iWeights[2];

                double jWeights[4];
                jWeights[0] = ((A * (fj + 1) - 5 * A) * (fj + 1) + 8 * A) * (fj + 1) - 4 * A;
                jWeights[1] = ((A + 2) * fj - (A + 3)) * fj * fj + 1;
                jWeights[2] = ((A + 2) * (1 - fj) - (A + 3)) * (1 - fj) * (1 - fj) + 1;
                jWeights[3] = 1 - jWeights[0] - jWeights[1] - jWeights[2];

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    double res = std::rint(std::abs(
                        srcPtr[(si - 1) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 0) * elementsPerPixel + k] * jWeights[1] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 1) * elementsPerPixel + k] * jWeights[2] * iWeights[3]
                        + srcPtr[(si - 1) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[0]
                        + srcPtr[(si + 0) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[1]
                        + srcPtr[(si + 1) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3] * iWeights[2]
                        + srcPtr[(si + 2) * srcRowStride + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                              * iWeights[3]));

                    dstPtr[di * dstRowStride + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
        }
    }
}

} // namespace nvcv::test
